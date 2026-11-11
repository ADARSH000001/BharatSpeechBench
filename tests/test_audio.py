import shutil
import numpy as np
import pytest
import soundfile as sf
from bharatspeechbench.audio.noise import mix_at_snr
from bharatspeechbench.audio.phone_simulation import simulate_phone


def _snr_db(clean, mixed):
    noise = mixed - clean
    return 10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2))


@pytest.mark.parametrize("snr", [20, 10, 5])
def test_mix_hits_target_snr(snr):
    rng = np.random.default_rng(1)
    speech = 0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
    noise = rng.normal(0, 0.05, 8000)               # shorter than speech: must tile
    mixed = mix_at_snr(speech, noise, snr, seed=3)
    assert _snr_db(speech, mixed) == pytest.approx(snr, abs=0.05)


def test_mix_is_deterministic():
    speech = 0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
    noise = np.random.default_rng(2).normal(0, 0.05, 40000)
    assert np.array_equal(mix_at_snr(speech, noise, 10, seed=5), mix_at_snr(speech, noise, 10, seed=5))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_phone_simulation_band_limits(tmp_path):
    sr = 16000
    t = np.arange(sr * 2) / sr
    low = np.sin(2 * np.pi * 1000 * t)               # inside the telephone band
    high = np.sin(2 * np.pi * 6000 * t)              # outside it
    src, out = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src, 0.3 * (low + high), sr)
    simulate_phone(src, out)
    y, sr_out = sf.read(out)
    assert sr_out == 16000
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr_out)
    p1000 = spec[(freqs > 950) & (freqs < 1050)].max()
    p6000 = spec[(freqs > 5950) & (freqs < 6050)].max()
    assert p6000 < 0.01 * p1000                       # 6 kHz removed, 1 kHz kept


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_noise_converted_once_is_byte_identical(tmp_path):
    import tempfile
    from pathlib import Path
    from bharatspeechbench.audio.conditions import (
        ensure_condition, parse_condition, to_clean, clip_seed, MODEL_SR
    )

    def _old_make_condition(src, out_wav, condition, clip_id, noise_wav=None, seed=0):
        kind, snr = parse_condition(condition)
        out_wav = Path(out_wav)
        with tempfile.TemporaryDirectory() as tmp:
            clean = to_clean(src, Path(tmp) / "clean.wav")
            noise = to_clean(noise_wav, Path(tmp) / "noise.wav")
            speech, _ = sf.read(clean)
            noise_a, _ = sf.read(noise)
            mixed = mix_at_snr(speech, noise_a, snr, seed=clip_seed(seed, clip_id))
            noisy = Path(tmp) / "noisy.wav"
            sf.write(noisy, mixed, MODEL_SR, subtype="PCM_16")
            return simulate_phone(noisy, out_wav, MODEL_SR)

    sr = 16000
    t = np.arange(sr) / sr
    speech_signal = 0.2 * np.sin(2 * np.pi * 300 * t)
    noise_signal = 0.05 * np.sin(2 * np.pi * 800 * t)

    src = tmp_path / "speech.wav"
    noise = tmp_path / "noise.wav"
    sf.write(src, speech_signal, sr, subtype="PCM_16")
    sf.write(noise, noise_signal, sr, subtype="PCM_16")

    old_out = tmp_path / "old.wav"
    _old_make_condition(src, old_out, "phone_noise_10", "clip_1", noise_wav=noise, seed=42)

    new_out = ensure_condition(src, tmp_path / "new_dir", "phone_noise_10", "clip_1", noise_wav=noise, seed=42)

    # Output audio must be byte-identical to before for the same seed
    assert old_out.read_bytes() == new_out.read_bytes()
    # Clean noise was cached and can be reused
    clean_noise_cached = tmp_path / "new_dir" / "_noise_16k.wav"
    assert clean_noise_cached.is_file()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_prepare_noise_from_folder_and_zip(tmp_path, capsys):
    import zipfile
    from bharatspeechbench.utils.hashing import sha256_file
    from scripts.prepare_noise import main as prepare_main

    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out_noise"

    # Environment 1 (folder): two synthetic wav files (2.0s stereo 48kHz + 2.0s mono 16kHz -> request 3.0s)
    orig_hashes = {}
    env1 = src_dir / "street"
    env1.mkdir(parents=True)
    t48 = np.arange(48000 * 2) / 48000
    stereo = np.column_stack([0.1 * np.sin(2 * np.pi * 200 * t48), 0.1 * np.cos(2 * np.pi * 300 * t48)])
    sf.write(env1 / "ch01.wav", stereo, 48000, subtype="PCM_16")
    t16 = np.arange(16000 * 2) / 16000
    sf.write(env1 / "ch02.wav", 0.1 * np.sin(2 * np.pi * 400 * t16), 16000, subtype="PCM_16")
    orig_hashes["street/ch01.wav"] = sha256_file(env1 / "ch01.wav")
    orig_hashes["street/ch02.wav"] = sha256_file(env1 / "ch02.wav")

    # Environment 2 (zip): two synthetic 2.0s mono wav files inside zip -> request 3.0s
    zip_stage = tmp_path / "stage"
    zip_stage.mkdir()
    for ch_name, freq in (("ch01.wav", 250), ("ch02.wav", 350)):
        sf.write(zip_stage / ch_name, 0.1 * np.sin(2 * np.pi * freq * t16), 16000, subtype="PCM_16")
    zpath = src_dir / "CAFETERIA_16k.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(zip_stage / "ch01.wav", arcname="CAFETERIA/ch01.wav")
        zf.write(zip_stage / "ch02.wav", arcname="CAFETERIA/ch02.wav")
    orig_hashes["CAFETERIA_16k.zip"] = sha256_file(zpath)

    rc = prepare_main([
        "--src", str(src_dir),
        "--out", str(out_dir),
        "--env", "street",
        "--env", "CAFETERIA",
        "--seconds", "3",
        "--download-date", "2026-11-11",
    ])
    assert rc == 0
    stdout = capsys.readouterr().out

    # Originals never modified
    assert sha256_file(env1 / "ch01.wav") == orig_hashes["street/ch01.wav"]
    assert sha256_file(env1 / "ch02.wav") == orig_hashes["street/ch02.wav"]
    assert sha256_file(zpath) == orig_hashes["CAFETERIA_16k.zip"]

    sources_md = (out_dir / "SOURCES.md").read_text(encoding="utf-8")
    for name in ("street", "CAFETERIA"):
        wav = out_dir / f"{name}.wav"
        assert wav.is_file()
        info = sf.info(wav)
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.subtype == "PCM_16"
        assert info.duration == pytest.approx(3.0, abs=0.05)

        h = sha256_file(wav)
        assert h in stdout
        assert h in sources_md
        assert name in sources_md
        assert "2026-11-11" in sources_md
        assert "ch01.wav" in sources_md and "ch02.wav" in sources_md
