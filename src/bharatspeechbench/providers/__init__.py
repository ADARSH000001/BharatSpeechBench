"""ASR model providers. Add a model by subclassing ASRProvider (see base.py)."""
from .base import ASRProvider, Transcript

__all__ = ["ASRProvider", "Transcript"]
