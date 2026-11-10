"""Record everything needed to reproduce (or explain differences between) benchmark runs.

Store the returned dict next to every set of raw results.
"""
from __future__ import annotations

import ctypes
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .hashing import sha256_file

TRACKED_PACKAGES = (
    "bharatspeechbench", "numpy", "soundfile", "jiwer", "pandas", "pyarrow",
    "faster-whisper", "ctranslate2", "torch", "torchaudio", "transformers", "streamlit",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _total_ram_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _ffmpeg_version() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10).stdout
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty(cwd: str | Path | None = None) -> bool | None:
    """Return True if any tracked file is modified or staged; ignore untracked files."""
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10, cwd=cwd,
        )
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu() -> dict | None:
    try:
        import torch
        if torch.cuda.is_available():
            return {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count(),
                    "cuda": torch.version.cuda}
    except ImportError:
        pass
    return None


def capture_environment(seed: int, manifest_path: str | Path | None = None) -> dict:
    """JSON-serialisable snapshot of the run environment. Pass the manifest to pin the dataset too."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "python": sys.version.split()[0],
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "cpu": {"name": platform.processor() or None, "logical_cores": os.cpu_count()},
        "ram_bytes": _total_ram_bytes(),
        "gpu": _gpu(),
        "ffmpeg": _ffmpeg_version(),
        "packages": _package_versions(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
    }
