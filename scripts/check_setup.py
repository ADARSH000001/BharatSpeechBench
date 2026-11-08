"""Run after installation: python scripts/check_setup.py"""
import importlib
import shutil
import sys

print("python:", sys.version.split()[0])
print("ffmpeg:", shutil.which("ffmpeg") or "NOT FOUND")
for mod in ("numpy", "soundfile", "jiwer", "pandas", "pyarrow", "streamlit", "faster_whisper", "torch"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}: {getattr(m, '__version__', 'ok')}")
    except Exception as e:  # noqa: BLE001
        print(f"{mod}: missing ({type(e).__name__})")
