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


EXPECTED_SCRIPT: dict[str, str] = {"hi": "Devanagari", "gu": "Gujarati"}
_SCRIPT_TIE_ORDER = ("Gujarati", "Devanagari", "Latin", "Other")
_DEVA_GU_OFFSET = 0x180

# Devanagari codepoints (letters, vowel signs, chandrabindu, anusvara, visarga, nukta, avagraha, virama)
# verified to have matching Gujarati counterparts at codepoint + 0x180.
_DEVANAGARI_FOLD_RANGES = (
    range(0x0901, 0x0904),
    range(0x0905, 0x090E),
    range(0x090F, 0x0912),
    range(0x0913, 0x0929),
    range(0x092A, 0x0931),
    range(0x0932, 0x0934),
    range(0x0935, 0x093A),
    range(0x093C, 0x0946),
    range(0x0947, 0x094A),
    range(0x094B, 0x094E),
    (0x0950,),
    range(0x0960, 0x0964),
    (0x0979,),
)
_DEVANAGARI_TO_GUJARATI = str.maketrans(
    {cp: cp + _DEVA_GU_OFFSET for r in _DEVANAGARI_FOLD_RANGES for cp in r}
)


def _char_script(ch: str) -> str | None:
    cat = unicodedata.category(ch)
    if cat == "Nd" or cat[0] in ("P", "S", "Z", "C"):
        return None
    cp = ord(ch)
    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    if 0x0A80 <= cp <= 0x0AFF:
        return "Gujarati"
    if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
        return "Latin"
    if ch.isalpha():
        return "Other"
    return None


def script_stats(text: str | None, language: str) -> tuple[float | None, str | None]:
    """Return (script_match, dominant_script) for a transcript; script_match is None for hi-en or no letters."""
    counts = {s: 0 for s in _SCRIPT_TIE_ORDER}
    for ch in text or "":
        s = _char_script(ch)
        if s is not None:
            counts[s] += 1
    total = sum(counts.values())
    if total == 0:
        return None, None
    dominant = max(_SCRIPT_TIE_ORDER, key=lambda s: counts[s])
    expected = EXPECTED_SCRIPT.get(language)
    if expected is None:
        return None, dominant
    return counts[expected] / total, dominant


def fold_devanagari_to_gujarati(text: str) -> str:
    """Map Devanagari letters/marks in text to matching Gujarati codepoints (+0x180); leave Latin & unmapped chars untouched."""
    return text.translate(_DEVANAGARI_TO_GUJARATI)


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
