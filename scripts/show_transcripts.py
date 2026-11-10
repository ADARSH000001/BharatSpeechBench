"""Inspect raw transcripts with references, alignment counts, and quality warnings.

Usage:
    python scripts/show_transcripts.py results/raw/<run_id> --manifest data/manifests/manifest.jsonl [--condition phone] [--language hi]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bharatspeechbench.benchmark.manifest import load_manifest
from bharatspeechbench.benchmark.reporting import (
    TRUNCATION_LEN_RATIO_THRESHOLD,
    TRUNCATION_MAX_SEGMENT_TOKENS,
    check_truncation,
)
from bharatspeechbench.metrics.normalization import script_stats
from bharatspeechbench.metrics.wer import per_utterance


def is_mostly_latin(text: str) -> bool:
    latin_count = sum(1 for c in text if ("a" <= c <= "z" or "A" <= c <= "Z"))
    alpha_count = sum(1 for c in text if c.isalpha())
    return alpha_count > 0 and (latin_count / alpha_count) > 0.5


def has_indic(text: str) -> bool:
    return any(("\u0900" <= c <= "\u097F") or ("\u0A80" <= c <= "\u0AFF") for c in text)


def get_warnings(
    ref: str,
    hyp: str,
    coverage: float | None = None,
    max_segment_tokens: int | float | None = None,
    script_match: float | None = None,
) -> list[str]:
    warnings = []
    if not hyp.strip():
        warnings.append("hypothesis is empty")
    elif len(hyp.strip()) < 0.5 * len(ref.strip()):
        warnings.append("hypothesis length is less than 50% of reference length")

    if (has_indic(ref) or any("\u0900" <= c <= "\u0AFF" for c in ref)) and is_mostly_latin(hyp):
        warnings.append("hypothesis contains mostly Latin letters for an Indic reference")

    if isinstance(coverage, (int, float)) and not isinstance(coverage, bool) and coverage < 0.85:
        warnings.append(f"low coverage ({coverage:.2f} < 0.85)")

    if isinstance(script_match, (int, float)) and not isinstance(script_match, bool) and script_match < 0.5:
        warnings.append(f"low script match ({script_match:.2f} < 0.50)")

    trunc = check_truncation(ref, hyp, max_segment_tokens)
    if trunc["truncated_by_ratio"]:
        warnings.append(
            f"possible truncation (norm len ratio {trunc['len_ratio_norm']:.2f} < {TRUNCATION_LEN_RATIO_THRESHOLD})"
        )
    if trunc["truncated_by_tokens"]:
        warnings.append(
            f"possible truncation (max_segment_tokens={trunc['max_segment_tokens']} >= {TRUNCATION_MAX_SEGMENT_TOKENS})"
        )

    return warnings


def show_transcripts(
    run_dir: str | Path,
    manifest_path: str | Path,
    condition: str | None = None,
    language: str | None = None,
    out_file=sys.stdout,
) -> int:
    run_path = Path(run_dir)
    transcripts_file = run_path / "transcripts.jsonl"
    if not transcripts_file.is_file():
        print(f"ERROR: {transcripts_file} not found", file=sys.stderr)
        return 1

    clips = load_manifest(manifest_path)
    ref_by_id = {c.id: c for c in clips}

    # Dedup by (clip_id, condition) keeping the latest row (to handle retries correctly)
    rows_by_key = {}
    for line in transcripts_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows_by_key[(row["clip_id"], row["condition"])] = row

    displayed = 0
    for (cid, cond), row in rows_by_key.items():
        lang = row.get("language")
        if condition and cond != condition:
            continue
        if language and lang != language:
            continue

        displayed += 1
        clip = ref_by_id.get(cid)
        ref_raw = clip.reference if clip else ""
        hyp_raw = row.get("text") or ""
        error = row.get("error")

        print(f"[{cid}] condition={cond} language={lang}", file=out_file)
        print(f"  REF: {ref_raw}", file=out_file)

        if error:
            print(f"  ERROR: {error}", file=out_file)
            print("-" * 60, file=out_file)
            continue

        print(f"  HYP: {hyp_raw}", file=out_file)

        meta = row.get("meta") or {}
        coverage = meta.get("coverage")
        is_num_cov = isinstance(coverage, (int, float)) and not isinstance(coverage, bool)
        cov_str = f"{coverage:.2f}" if is_num_cov else "n/a"
        total_tok = meta.get("total_tokens")
        max_seg_tok = meta.get("max_segment_tokens")
        tok_total_str = str(total_tok) if total_tok is not None else "n/a"
        tok_max_str = str(max_seg_tok) if max_seg_tok is not None else "n/a"
        print(f"  COVERAGE: {cov_str}  TOKENS: total={tok_total_str}, max_segment={tok_max_str}", file=out_file)

        sm, dom = script_stats(hyp_raw, lang or "")
        sm_str = f"{sm:.2f}" if sm is not None else "n/a"
        print(f"  SCRIPT: dominant={dom or 'n/a'}, match={sm_str}", file=out_file)

        counts = per_utterance([ref_raw], [hyp_raw], level="word", normalise=False)[0]
        print(
            f"  COUNTS: S={counts.substitutions} D={counts.deletions} I={counts.insertions} "
            f"(errors={counts.errors}, ref_len={counts.ref_len})",
            file=out_file,
        )

        warnings = get_warnings(
            ref_raw,
            hyp_raw,
            coverage=coverage if is_num_cov else None,
            max_segment_tokens=max_seg_tok,
            script_match=sm,
        )
        for w in warnings:
            print(f"  WARNING: {w}", file=out_file)

        print("-" * 60, file=out_file)

    print(f"Displayed {displayed} transcripts.", file=out_file)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect raw transcripts with alignment counts and quality warnings.")
    parser.add_argument("run_dir", help="Path to raw run folder (results/raw/<run_id>)")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL file")
    parser.add_argument("--condition", default=None, help="Optional condition filter")
    parser.add_argument("--language", default=None, help="Optional language filter")

    args = parser.parse_args(argv)
    return show_transcripts(args.run_dir, args.manifest, args.condition, args.language)


if __name__ == "__main__":
    raise SystemExit(main())
