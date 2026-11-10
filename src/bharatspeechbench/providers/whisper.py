"""faster-whisper baseline. Install with: pip install -e ".[whisper]"
Record the model size: a small Whisper vs a 600M Indic model is NOT a fair comparison.

device="auto" uses the GPU when CTranslate2 can see one (float16), otherwise CPU (int8).
On Windows the GPU also needs the CUDA 12 cuBLAS and cuDNN 9 DLLs; if loading fails, use --device cpu.
"""
from __future__ import annotations

from pathlib import Path

from .base import ASRProvider


def _resolve(device: str, compute_type: str) -> tuple[str, str]:
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


_DEFAULT_MODELS: dict[str, str] = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def _resolve_model_repo(size: str) -> str:
    try:
        from faster_whisper.utils import _MODELS
        return _MODELS.get(size, _DEFAULT_MODELS.get(size, size))
    except Exception:  # noqa: BLE001
        return _DEFAULT_MODELS.get(size, size)


def get_model_revision(repo_id: str, cache_dir: str | Path | None = None) -> str | None:
    """Find the commit hash from the local Hugging Face hub cache without network calls."""
    try:
        if cache_dir is None:
            try:
                import huggingface_hub.constants as hf_const
                cache_dir = Path(hf_const.HF_HUB_CACHE)
            except Exception:  # noqa: BLE001
                cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        else:
            cache_dir = Path(cache_dir)

        if not cache_dir.is_dir():
            return None

        folder_name = f"models--{repo_id.replace('/', '--')}"
        repo_dir = cache_dir / folder_name
        if not repo_dir.is_dir():
            return None

        refs_main = repo_dir / "refs" / "main"
        if refs_main.is_file():
            commit = refs_main.read_text(encoding="utf-8").strip()
            if commit:
                return commit

        snapshots = repo_dir / "snapshots"
        if snapshots.is_dir():
            dirs = [p.name for p in snapshots.iterdir() if p.is_dir()]
            if dirs:
                dirs.sort(key=lambda name: (snapshots / name).stat().st_mtime, reverse=True)
                return dirs[0]
    except Exception:  # noqa: BLE001
        return None
    return None


def _package_version(name: str) -> str | None:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return None


class WhisperProvider(ASRProvider):
    name = "faster-whisper"

    def __init__(self, size: str = "small", device: str = "auto", compute_type: str = "auto",
                 beam_size: int = 5, download_root: str | Path | None = None):
        from faster_whisper import WhisperModel  # lazy import: optional dependency

        self._size = size
        self._device, self._compute = _resolve(device, compute_type)
        self._beam = beam_size
        self._model_repo = _resolve_model_repo(size)
        self._model_revision = get_model_revision(self._model_repo, cache_dir=download_root)
        self._ctranslate2_version = _package_version("ctranslate2")
        self._faster_whisper_version = _package_version("faster-whisper")
        kw = {"download_root": str(download_root)} if download_root else {}
        self._model = WhisperModel(size, device=self._device, compute_type=self._compute, **kw)

    @property
    def model_id(self) -> str:
        return f"whisper-{self._size}"

    @property
    def settings(self) -> dict:
        return {
            "device": self._device,
            "compute_type": self._compute,
            "beam_size": self._beam,
            "decoding": "beam search, temperature 0, no initial prompt",
            "model_repo": self._model_repo,
            "model_revision": self._model_revision,
            "ctranslate2_version": self._ctranslate2_version,
            "faster_whisper_version": self._faster_whisper_version,
        }

    def _transcribe(self, wav_path: str, language: str):
        segments, info = self._model.transcribe(
            wav_path, language=language, beam_size=self._beam, temperature=0.0,
            condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments)
        return text, {"detected_language_prob": info.language_probability}
