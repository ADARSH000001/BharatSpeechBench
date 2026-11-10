"""faster-whisper baseline. Install with: pip install -e ".[whisper]"
Record the model size: a small Whisper vs a 600M Indic model is NOT a fair comparison.

device="auto" uses the GPU when CTranslate2 can see one (float16), otherwise CPU (int8).
On Windows the GPU also needs the CUDA 12 cuBLAS and cuDNN 9 DLLs; if loading fails, use --device cpu.
"""
from __future__ import annotations

from pathlib import Path

from .base import ASRProvider, Transcript


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


def get_model_revision_info(repo_id: str, cache_dir: str | Path | None = None) -> tuple[str | None, str | None]:
    """Find the commit hash from the local Hugging Face hub cache without network calls.

    Returns:
        (commit_hash, None) on success, or (None, failure_reason) on failure.
    """
    # 1. Try huggingface_hub's official scan_cache_dir if installed
    try:
        import huggingface_hub
        scan_kw = {"cache_dir": str(cache_dir)} if cache_dir else {}
        info = huggingface_hub.scan_cache_dir(**scan_kw)
        repo = next((r for r in info.repos if r.repo_id == repo_id), None)
        if repo and repo.revisions:
            for rev in repo.revisions:
                if "main" in rev.refs:
                    return rev.commit_hash, None
            # Fallback to newest revision by last_modified if available
            revs = sorted(repo.revisions, key=lambda r: getattr(r, "last_modified", 0), reverse=True)
            return revs[0].commit_hash, None
    except Exception:  # noqa: BLE001
        pass

    # 2. Fall back to offline filesystem parsing
    try:
        import os

        if cache_dir is None:
            if "HF_HUB_CACHE" in os.environ:
                cache_dir = Path(os.environ["HF_HUB_CACHE"])
            elif "HF_HOME" in os.environ:
                cache_dir = Path(os.environ["HF_HOME"]) / "hub"
            else:
                try:
                    import huggingface_hub.constants as hf_const
                    cache_dir = Path(hf_const.HF_HUB_CACHE)
                except Exception:  # noqa: BLE001
                    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        else:
            cache_dir = Path(cache_dir)

        if not cache_dir.is_dir():
            return None, f"cache directory does not exist: {cache_dir}"

        folder_name = f"models--{repo_id.replace('/', '--')}"
        repo_dir = cache_dir / folder_name
        if not repo_dir.is_dir():
            return None, f"model repo folder not found in cache: {folder_name}"

        # (a) Check refs/main
        refs_main = repo_dir / "refs" / "main"
        if refs_main.is_file():
            commit = refs_main.read_text(encoding="utf-8").strip()
            if commit:
                return commit, None

        # (b) Check any other ref in refs/
        refs_dir = repo_dir / "refs"
        if refs_dir.is_dir():
            for ref_file in sorted(refs_dir.iterdir()):
                if ref_file.is_file():
                    c = ref_file.read_text(encoding="utf-8").strip()
                    if c:
                        return c, None

        # (c) Check snapshots directory
        snapshots = repo_dir / "snapshots"
        if snapshots.is_dir():
            dirs = [p.name for p in snapshots.iterdir() if p.is_dir()]
            if dirs:
                dirs.sort(key=lambda name: (snapshots / name).stat().st_mtime, reverse=True)
                return dirs[0], None

        # (d) Check trees directory (Windows no-symlink layout)
        trees = repo_dir / "trees"
        if trees.is_dir():
            tree_files = [p.stem for p in trees.iterdir() if p.is_file() and p.suffix == ".json"]
            if tree_files:
                tree_files.sort(key=lambda name: (trees / f"{name}.json").stat().st_mtime, reverse=True)
                return tree_files[0], None

        return None, "cache layout not recognised: no valid refs, snapshots, or trees found"
    except Exception as exc:  # noqa: BLE001
        return None, f"error inspecting cache: {exc}"


def get_model_revision(repo_id: str, cache_dir: str | Path | None = None) -> str | None:
    """Find the commit hash from the local Hugging Face hub cache without network calls."""
    rev, _ = get_model_revision_info(repo_id, cache_dir=cache_dir)
    return rev


def _package_version(name: str) -> str | None:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:  # noqa: BLE001
        return None


class WhisperProvider(ASRProvider):
    name = "faster-whisper"

    def __init__(
        self,
        size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        download_root: str | Path | None = None,
        vad_filter: bool = False,
        chunk_length: int | None = None,
        max_new_tokens: int | None = None,
    ):
        from faster_whisper import WhisperModel  # lazy import: optional dependency

        if chunk_length is not None and int(chunk_length) <= 0:
            raise ValueError("chunk_length must be a positive integer")
        if max_new_tokens is not None and int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be a positive integer")

        self._size = size
        self._device, self._compute = _resolve(device, compute_type)
        self._beam = beam_size
        self._vad_filter = bool(vad_filter)
        self._chunk_length = int(chunk_length) if chunk_length is not None else None
        self._max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else None
        self._model_repo = _resolve_model_repo(size)
        self._ctranslate2_version = _package_version("ctranslate2")
        self._faster_whisper_version = _package_version("faster-whisper")
        kw = {"download_root": str(download_root)} if download_root else {}
        self._model = WhisperModel(size, device=self._device, compute_type=self._compute, **kw)
        # Query cache after model load/download to detect freshly-downloaded weights
        self._model_revision, self._model_revision_reason = get_model_revision_info(
            self._model_repo, cache_dir=download_root
        )

    @property
    def model_id(self) -> str:
        return f"whisper-{self._size}"

    @property
    def settings(self) -> dict:
        return {
            "device": self._device,
            "compute_type": self._compute,
            "beam_size": self._beam,
            "vad_filter": self._vad_filter,
            "chunk_length": self._chunk_length,
            "max_new_tokens": self._max_new_tokens,
            "decoding": "beam search, temperature 0, no initial prompt",
            "model_repo": self._model_repo,
            "model_revision": self._model_revision,
            "model_revision_reason": self._model_revision_reason,
            "ctranslate2_version": self._ctranslate2_version,
            "faster_whisper_version": self._faster_whisper_version,
        }

    def _transcribe(self, wav_path: str, language: str):
        transcribe_kw: dict = {
            "language": language,
            "beam_size": self._beam,
            "temperature": 0.0,
            "condition_on_previous_text": False,
        }
        if self._vad_filter:
            transcribe_kw["vad_filter"] = True
        if self._chunk_length is not None:
            transcribe_kw["chunk_length"] = self._chunk_length
        if self._max_new_tokens is not None:
            transcribe_kw["max_new_tokens"] = self._max_new_tokens

        segments, info = self._model.transcribe(wav_path, **transcribe_kw)
        seg_list = list(segments)
        text = " ".join(s.text.strip() for s in seg_list)
        num_segments = len(seg_list)
        last_end = getattr(seg_list[-1], "end", None) if seg_list else None
        last_segment_end_s = (
            float(last_end)
            if isinstance(last_end, (int, float)) and not isinstance(last_end, bool)
            else None
        )
        audio_dur = getattr(info, "duration", None)
        if not (isinstance(audio_dur, (int, float)) and not isinstance(audio_dur, bool) and audio_dur > 0):
            try:
                import soundfile as sf
                audio_dur = float(sf.info(wav_path).duration)
            except Exception:  # noqa: BLE001
                audio_dur = None
        if (
            last_segment_end_s is not None
            and isinstance(audio_dur, (int, float))
            and not isinstance(audio_dur, bool)
            and audio_dur > 0
        ):
            coverage = last_segment_end_s / float(audio_dur)
        else:
            coverage = None

        token_counts = [
            len(s.tokens)
            for s in seg_list
            if hasattr(s, "tokens") and isinstance(getattr(s, "tokens"), (list, tuple))
        ]
        if not seg_list:
            total_tokens: int | None = 0
            max_segment_tokens: int | None = 0
        elif len(token_counts) == len(seg_list):
            total_tokens = int(sum(token_counts))
            max_segment_tokens = int(max(token_counts))
        else:
            total_tokens = None
            max_segment_tokens = None

        meta = {
            "detected_language_prob": getattr(info, "language_probability", None),
            "num_segments": num_segments,
            "last_segment_end_s": last_segment_end_s,
            "coverage": coverage,
            "total_tokens": total_tokens,
            "max_segment_tokens": max_segment_tokens,
        }
        return text, meta

    def transcribe(self, wav_path: str, language: str, audio_seconds: float) -> Transcript:
        t = super().transcribe(wav_path, language, audio_seconds)
        if (
            t.meta.get("coverage") is None
            and t.meta.get("last_segment_end_s") is not None
            and isinstance(audio_seconds, (int, float))
            and not isinstance(audio_seconds, bool)
            and audio_seconds > 0
        ):
            t.meta["coverage"] = float(t.meta["last_segment_end_s"]) / float(audio_seconds)
        return t
