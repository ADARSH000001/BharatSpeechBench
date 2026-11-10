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
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0


def _counts(ref: str, hyp: str, level: str) -> Counts:
    if not ref.strip():
        raise ValueError("empty reference transcript")
    if not hyp.strip():             # jiwer rejects empty hypotheses: all reference units deleted
        n = len(ref.split()) if level == "word" else len(ref)
        return Counts(errors=n, ref_len=n, substitutions=0, deletions=n, insertions=0)
    fn = jiwer.process_words if level == "word" else jiwer.process_characters
    out = fn(ref, hyp)
    sub = out.substitutions
    dele = out.deletions
    ins = out.insertions
    err = sub + dele + ins
    ref_len = sub + dele + out.hits
    return Counts(
        errors=err,
        ref_len=ref_len,
        substitutions=sub,
        deletions=dele,
        insertions=ins,
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


def paired_bootstrap_diff(
    counts_a: list[Counts],
    counts_b: list[Counts],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Jointly resample clip indices to compute (diff, lo, hi) where diff = rate_b - rate_a."""
    if len(counts_a) != len(counts_b):
        raise ValueError(f"counts_a and counts_b must have the same length ({len(counts_a)} != {len(counts_b)})")
    if not counts_a:
        return 0.0, 0.0, 0.0

    sum_len_a = sum(c.ref_len for c in counts_a)
    sum_len_b = sum(c.ref_len for c in counts_b)
    rate_a = sum(c.errors for c in counts_a) / sum_len_a if sum_len_a else 0.0
    rate_b = sum(c.errors for c in counts_b) / sum_len_b if sum_len_b else 0.0
    diff = rate_b - rate_a

    n = len(counts_a)
    rng = np.random.default_rng(seed)
    errs_a = np.array([c.errors for c in counts_a], dtype=float)
    lens_a = np.array([c.ref_len for c in counts_a], dtype=float)
    errs_b = np.array([c.errors for c in counts_b], dtype=float)
    lens_b = np.array([c.ref_len for c in counts_b], dtype=float)

    idx = rng.integers(0, n, size=(n_boot, n))
    denom_a = lens_a[idx].sum(axis=1)
    denom_b = lens_b[idx].sum(axis=1)
    rates_a = errs_a[idx].sum(axis=1) / np.where(denom_a == 0, np.nan, denom_a)
    rates_b = errs_b[idx].sum(axis=1) / np.where(denom_b == 0, np.nan, denom_b)
    diffs = rates_b - rates_a

    lo = float(np.nanquantile(diffs, alpha / 2))
    hi = float(np.nanquantile(diffs, 1 - alpha / 2))
    return float(diff), lo, hi
