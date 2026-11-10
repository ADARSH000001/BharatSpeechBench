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
