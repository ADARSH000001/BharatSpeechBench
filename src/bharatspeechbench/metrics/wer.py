"""WER / CER per utterance, corpus-level aggregation, and a bootstrap confidence interval."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jiwer

from .normalization import normalize


@dataclass
class Counts:
    errors: int      # S + D + I
    ref_len: int     # N (words or characters)


def _counts(ref: str, hyp: str, level: str) -> Counts:
    if not ref.strip():
        raise ValueError("empty reference transcript")
    if not hyp.strip():             # jiwer rejects empty hypotheses: all reference units deleted
        n = len(ref.split()) if level == "word" else len(ref)
        return Counts(errors=n, ref_len=n)
    fn = jiwer.process_words if level == "word" else jiwer.process_characters
    out = fn(ref, hyp)
    return Counts(
        out.substitutions + out.deletions + out.insertions,
        out.substitutions + out.deletions + out.hits,
    )


def per_utterance(refs: list[str], hyps: list[str], level: str = "word", normalise: bool = True) -> list[Counts]:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps differ in length")
    if normalise:
        refs = [normalize(r) for r in refs]
        hyps = [normalize(h) for h in hyps]
    return [_counts(r, h, level) for r, h in zip(refs, hyps)]


def corpus_rate(counts: list[Counts]) -> float:
    """Corpus-level error rate: total errors / total reference units (not a mean of per-clip rates)."""
    return sum(c.errors for c in counts) / sum(c.ref_len for c in counts)


def bootstrap_ci(counts: list[Counts], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over utterances. With ~30 clips the interval will be wide: report it."""
    rng = np.random.default_rng(seed)
    errs = np.array([c.errors for c in counts], dtype=float)
    lens = np.array([c.ref_len for c in counts], dtype=float)
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    rates = errs[idx].sum(axis=1) / lens[idx].sum(axis=1)
    return float(np.quantile(rates, alpha / 2)), float(np.quantile(rates, 1 - alpha / 2))
