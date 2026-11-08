"""Simulate a telephone channel with ffmpeg: band-limit, 8 kHz, G.711 mu-law, back to 16 kHz.

This is a simulation of narrow-band telephony, not real network conditions
(no packet loss, no AMR/EVS codecs, no handset or room effects).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class FFmpegNotFound(RuntimeError):
    pass


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-500:]}")


def simulate_phone(in_wav: str | Path, out_wav: str | Path, model_sr: int = 16000) -> Path:
    """Telephone-quality version of `in_wav`, written at `model_sr` so models accept it.

    Chain: mono -> 300-3400 Hz band-pass -> 8 kHz -> G.711 mu-law -> decode -> `model_sr` PCM16.
    """
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFound("ffmpeg is required for phone simulation")
    in_wav, out_wav = Path(in_wav), Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mid = Path(tmp) / "ulaw.wav"
        _run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(in_wav),
            "-ac", "1", "-af", "highpass=f=300,lowpass=f=3400",
            "-ar", "8000", "-acodec", "pcm_mulaw", str(mid),
        ])
        _run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(mid),
            "-ar", str(model_sr), "-acodec", "pcm_s16le", str(out_wav),
        ])
    return out_wav
