"""Mix noise into speech at a target signal-to-noise ratio, deterministically."""
from __future__ import annotations

import numpy as np
import soundfile as sf


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Return speech + scaled noise so that 10*log10(P_speech / P_noise) == snr_db.

    Noise is tiled/cropped to the speech length; the crop offset is seeded for reproducibility.
    """
    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if speech.ndim != 1 or noise.ndim != 1:
        raise ValueError("expected mono 1-D arrays")
    if len(noise) == 0 or len(speech) == 0:
        raise ValueError("empty audio")

    rng = np.random.default_rng(seed)
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(speech) + 1))
    noise = noise[start:start + len(speech)]

    p_speech = np.mean(speech ** 2)
    p_noise = np.mean(noise ** 2)
    if p_noise == 0 or p_speech == 0:
        raise ValueError("silent signal: SNR undefined")
    scale = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    mixed = speech + scale * noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:               # avoid clipping; scales speech and noise together, SNR unchanged
        mixed = mixed / peak
    return mixed


def mix_files(speech_wav: str, noise_wav: str, out_wav: str, snr_db: float, seed: int = 0) -> str:
    speech, sr_s = sf.read(speech_wav, always_2d=False)
    noise, sr_n = sf.read(noise_wav, always_2d=False)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)
    if sr_s != sr_n:
        raise ValueError(f"sample rates differ ({sr_s} vs {sr_n}); resample the noise first")
    sf.write(out_wav, mix_at_snr(speech, noise, snr_db, seed), sr_s, subtype="PCM_16")
    return out_wav
