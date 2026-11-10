"""Common contract every ASR model must follow, so all models are scored identically."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Transcript:
    text: str                      # raw model output, never edited
    provider: str                  # e.g. "faster-whisper"
    model: str                     # exact model name AND revision/version
    language: str                  # language code requested, e.g. "hi", "gu"
    audio_seconds: float
    processing_seconds: float
    meta: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Real-time factor = processing time / audio duration (lower is faster).
        Only compare across local models on the SAME hardware."""
        return self.processing_seconds / self.audio_seconds if self.audio_seconds else float("nan")


class ASRProvider(ABC):
    name: str = "base"

    @abstractmethod
    def _transcribe(self, wav_path: str, language: str) -> tuple[str, dict]:
        """Return (raw_text, extra_metadata). Audio is 16 kHz mono WAV."""

    def transcribe(self, wav_path: str, language: str, audio_seconds: float) -> Transcript:
        start = time.perf_counter()
        text, meta = self._transcribe(wav_path, language)
        elapsed = time.perf_counter() - start
        return Transcript(
            text=text,
            provider=self.name,
            model=self.model_id,
            language=language,
            audio_seconds=audio_seconds,
            processing_seconds=elapsed,
            meta=meta,
        )

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Exact model identifier including size/revision, recorded with every result."""

    @property
    def settings(self) -> dict:
        """Key model settings and environment versions recorded in run config."""
        return {}
