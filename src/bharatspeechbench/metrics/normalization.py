"""Text normalisation for scoring. Every rule here must be documented in docs/methodology.md.

Always keep and report RAW scores too: normalisation choices change results.
"""
from __future__ import annotations

import re
import unicodedata

# Indic digits -> ASCII (Devanagari U+0966.., Gujarati U+0AE6..)
_DIGITS = {}
for _base in (0x0966, 0x0AE6):
    for _i in range(10):
        _DIGITS[_base + _i] = str(_i)
# punctuation incl. Devanagari danda / double danda
_PUNCT = re.compile(r"[\u0964\u0965.,!?;:\"'()\[\]{}\-\u2013\u2014\u2026]")


def ascii_digits(text: str) -> str:
    """Map Devanagari and Gujarati digits to ASCII 0-9 (nothing else is changed)."""
    return unicodedata.normalize("NFC", text).translate(_DIGITS)


def normalize(text: str, lowercase: bool = True, ascii_digits: bool = True) -> str:
    text = unicodedata.normalize("NFC", text)
    if ascii_digits:
        text = text.translate(_DIGITS)
    text = _PUNCT.sub(" ", text)
    if lowercase:
        text = text.lower()          # only affects Latin letters (e.g. Hinglish in Roman script)
    return re.sub(r"\s+", " ", text).strip()
