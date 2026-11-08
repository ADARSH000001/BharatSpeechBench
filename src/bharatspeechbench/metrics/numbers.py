"""Number accuracy: how many of the digit sequences in the reference appear in the transcript.

Scope (be honest about it in any write-up):
- Only DIGIT sequences are compared (Devanagari / Gujarati digits are mapped to ASCII first).
  Thousands commas are removed ("1,500" -> "1500"); decimals are kept ("2.5").
- Numbers spelled out as words ("पच्चीस") are NOT understood. If a model writes words where the
  reference has digits, it is scored as wrong. Write references with digits, and say so in the report.
- Each reference number can be matched once (multiset match), so repeats count.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .normalization import ascii_digits

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{2,3}(?!\d))")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class NumberCounts:
    matched: int
    total: int          # numbers in the reference


def extract_numbers(text: str) -> list[str]:
    t = _THOUSANDS.sub("", ascii_digits(text))
    return _NUMBER.findall(t)


def number_counts(ref: str, hyp: str) -> NumberCounts:
    ref_nums = Counter(extract_numbers(ref))
    hyp_nums = Counter(extract_numbers(hyp))
    matched = sum(min(n, hyp_nums[k]) for k, n in ref_nums.items())
    return NumberCounts(matched=matched, total=sum(ref_nums.values()))


def number_accuracy(counts: list[NumberCounts]) -> float | None:
    """Corpus-level: matched / total over all clips. None when no reference contains a number."""
    total = sum(c.total for c in counts)
    return sum(c.matched for c in counts) / total if total else None
