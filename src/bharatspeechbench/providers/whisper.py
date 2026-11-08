"""faster-whisper baseline. Install with: pip install -e ".[whisper]"
Record the model size: a small Whisper vs a 600M Indic model is NOT a fair comparison.

device="auto" uses the GPU when CTranslate2 can see one (float16), otherwise CPU (int8).
On Windows the GPU also needs the CUDA 12 cuBLAS and cuDNN 9 DLLs; if loading fails, use --device cpu.
"""
from __future__ import annotations

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


class WhisperProvider(ASRProvider):
    name = "faster-whisper"

    def __init__(self, size: str = "small", device: str = "auto", compute_type: str = "auto",
                 beam_size: int = 5):
        from faster_whisper import WhisperModel  # lazy import: optional dependency

        self._size = size
        self._device, self._compute = _resolve(device, compute_type)
        self._beam = beam_size
        self._model = WhisperModel(size, device=self._device, compute_type=self._compute)

    @property
    def model_id(self) -> str:
        return f"whisper-{self._size}"

    @property
    def settings(self) -> dict:
        return {"device": self._device, "compute_type": self._compute, "beam_size": self._beam,
                "decoding": "beam search, temperature 0, no initial prompt"}

    def _transcribe(self, wav_path: str, language: str):
        segments, info = self._model.transcribe(
            wav_path, language=language, beam_size=self._beam, temperature=0.0,
            condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments)
        return text, {"detected_language_prob": info.language_probability}
