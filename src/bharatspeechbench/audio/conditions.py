"""Create the audio conditions a benchmark is run on, from the original clips.

  clean            16 kHz mono PCM16 copy of the original
  phone            clean -> telephone channel (see phone_simulation.py)
  phone_noise_<N>  clean + noise at N dB SNR (seeded) -> telephone channel

Noise is mixed BEFORE the phone chain, so it passes through the channel like real background noise.
Originals are never modified. Output: <out_dir>/<condition>/<clip_id>.wav
"""
from __future__ import annotations

import re
import tempfile
import zlib
from pathlib import Path

import soundfile as sf

from .noise import mix_at_snr
from .phone_simulation import _run, simulate_phone, FFmpegNotFound  # noqa: F401  (re-exported)

DEFAULT_CONDITIONS = ("clean", "phone", "phone_noise_20", "phone_noise_10", "phone_noise_5")
_NOISE_RE = re.compile(r"^phone_noise_(\d+)$")
MODEL_SR = 16000


def parse_condition(name: str) -> tuple[str, float | None]:
    """'clean' -> ('clean', None); 'phone_noise_10' -> ('phone_noise', 10.0)."""
    if name in ("clean", "phone"):
        return name, None
    m = _NOISE_RE.match(name)
    if m:
        return "phone_noise", float(m.group(1))
    raise ValueError(f"unknown condition {name!r}; use clean, phone or phone_noise_<SNR dB>")


def clip_seed(seed: int, clip_id: str) -> int:
    """Stable per-clip seed (same on every machine and Python version)."""
    return (seed * 1_000_003 + zlib.crc32(clip_id.encode("utf-8"))) % (2**32)


def to_clean(in_audio: str | Path, out_wav: str | Path) -> Path:
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(in_audio),
          "-ac", "1", "-ar", str(MODEL_SR), "-acodec", "pcm_s16le", str(out_wav)])
    return out_wav


def make_condition(src: str | Path, out_wav: str | Path, condition: str, clip_id: str,
                   noise_wav: str | Path | None = None, seed: int = 0) -> Path:
    kind, snr = parse_condition(condition)
    out_wav = Path(out_wav)
    if kind == "clean":
        return to_clean(src, out_wav)
    if kind == "phone":
        return simulate_phone(src, out_wav, MODEL_SR)
    if noise_wav is None:
        raise ValueError(f"{condition} needs a noise file (--noise)")
    with tempfile.TemporaryDirectory() as tmp:
        clean = to_clean(src, Path(tmp) / "clean.wav")
        noise = to_clean(noise_wav, Path(tmp) / "noise.wav")
        speech, _ = sf.read(clean)
        noise_a, _ = sf.read(noise)
        mixed = mix_at_snr(speech, noise_a, snr, seed=clip_seed(seed, clip_id))
        noisy = Path(tmp) / "noisy.wav"
        sf.write(noisy, mixed, MODEL_SR, subtype="PCM_16")
        return simulate_phone(noisy, out_wav, MODEL_SR)


def condition_path(out_dir: str | Path, condition: str, clip_id: str) -> Path:
    return Path(out_dir) / condition / f"{clip_id}.wav"


def ensure_condition(src: str | Path, out_dir: str | Path, condition: str, clip_id: str,
                     noise_wav: str | Path | None = None, seed: int = 0, force: bool = False) -> Path:
    """Create the condition file unless it already exists (conditions are re-creatable, so cached)."""
    out = condition_path(out_dir, condition, clip_id)
    if force or not out.is_file():
        make_condition(src, out, condition, clip_id, noise_wav, seed)
    return out
