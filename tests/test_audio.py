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
