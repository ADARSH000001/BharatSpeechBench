"""Read-only audit of ASR substitution pairs, substitution categories, and top deletion clips.

Usage:
    python scripts/error_audit.py results/raw/<run_id> --manifest data/manifests/pilot.jsonl [--language hi] [--condition clean]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import jiwer

from bharatspeechbench.benchmark.manifest import load_manifest
from bharatspeechbench.benchmark.reporting import _load_transcripts
from bharatspeechbench.metrics.normalization import normalize, normalize_lenient_hi
from bharatspeechbench.metrics.wer import per_utterance


def _has_latin(text: str) -> bool:
    return any("a" <= c <= "z" or "A" <= c <= "Z" for c in text)


def _has_indic(text: str) -> bool:
    return any(("\u0900" <= c <= "\u097F") or ("\u0A80" <= c <= "\u0AFF") for c in text)


def classify_substitution(ref_word: str, hyp_word: str) -> str:
    """Categorize a normalised word substitution pair."""
    if _has_latin(ref_word) and not _has_indic(ref_word) and _has_indic(hyp_word):
        return "latin_to_indic"
    if ref_word.isdigit() and not hyp_word.isdigit() and any(c.isalpha() for c in hyp_word):
        return "digit_to_words"
    if normalize_lenient_hi(ref_word) == normalize_lenient_hi(hyp_word):
        return "orthographic"
    return "other"


def compute_error_audit(
    run_dir: str | Path,
    manifest_path: str | Path,
    language: str | None = None,
    condition: str | None = None,
) -> dict:
    """Compute substitution pairs, substitution types, and worst deletion clips without writing files."""
    run_path = Path(run_dir)
    transcripts_file = run_path / "transcripts.jsonl"
    if not transcripts_file.is_file():
        raise FileNotFoundError(f"{transcripts_file} not found")

    ref_by_id = {c.id: c for c in load_manifest(manifest_path)}
    rows = _load_transcripts(run_path)

    sub_pairs: Counter[tuple[str, str]] = Counter()
    sub_types: Counter[str] = Counter({
        "latin_to_indic": 0,
        "digit_to_words": 0,
        "orthographic": 0,
        "other": 0,
    })
    deletion_rows: list[dict] = []
    n_clips = 0

    for row in rows:
        if row.get("error") is not None:
            continue
        lang = row.get("language")
        cond = row.get("condition")
        if language and lang != language:
            continue
        if condition and cond != condition:
            continue

        cid = row["clip_id"]
        clip = ref_by_id.get(cid)
        if clip is None:
            continue

        n_clips += 1
        ref_raw = clip.reference
        hyp_raw = row.get("text") or ""
        norm_ref = normalize(ref_raw)
        norm_hyp = normalize(hyp_raw)
        ratio = (len(norm_hyp) / len(norm_ref)) if norm_ref else None
        cov = (row.get("meta") or {}).get("coverage")

        if norm_ref and norm_hyp:
            aligned = jiwer.process_words(norm_ref, norm_hyp)
            ref_words = aligned.references[0]
            hyp_words = aligned.hypotheses[0]
            for chunk in aligned.alignments[0]:
                if chunk.type == "substitute":
                    for rw, hw in zip(
                        ref_words[chunk.ref_start_idx : chunk.ref_end_idx],
                        hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx],
                    ):
                        sub_pairs[(rw, hw)] += 1
                        sub_types[classify_substitution(rw, hw)] += 1

        dels = per_utterance([ref_raw], [hyp_raw], level="word", normalise=True)[0].deletions
        deletion_rows.append({
            "clip_id": cid,
            "condition": cond,
            "language": lang,
            "deletions": int(dels),
            "ratio": ratio,
            "coverage": cov,
        })

    top_subs = sorted(sub_pairs.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:25]
    top_deletions = sorted(
        deletion_rows,
        key=lambda r: (-r["deletions"], str(r["clip_id"]), str(r["condition"])),
    )[:10]

    return {
        "n_clips": n_clips,
        "top_substitutions": top_subs,
        "substitution_types": dict(sub_types),
        "top_deletions": top_deletions,
    }


def run_error_audit(
    run_dir: str | Path,
    manifest_path: str | Path,
    language: str | None = None,
    condition: str | None = None,
    out_file=sys.stdout,
) -> int:
    try:
        audit = compute_error_audit(run_dir, manifest_path, language=language, condition=condition)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Error audit for {Path(run_dir).name} ({audit['n_clips']} clips)", file=out_file)
    print("\n(a) Top 25 substitution pairs (ref -> hyp):", file=out_file)
    if not audit["top_substitutions"]:
        print("  (none)", file=out_file)
    else:
        for (rw, hw), cnt in audit["top_substitutions"]:
            print(f"  {cnt:4d}x  {rw} -> {hw}", file=out_file)

    st = audit["substitution_types"]
    total_subs = sum(st.values())

    def _sub_line(label: str, key: str) -> str:
        cnt = st.get(key, 0)
        share = (100.0 * cnt / total_subs) if total_subs > 0 else 0.0
        return f"  {label}: {cnt} ({share:.1f}%)"

    print("\n(b) Substitution types:", file=out_file)
    print(_sub_line("Latin -> Devanagari/Gujarati", "latin_to_indic"), file=out_file)
    print(_sub_line("Digit sequence -> words", "digit_to_words"), file=out_file)
    print(_sub_line("orthographic (equal after lenient folding)", "orthographic"), file=out_file)
    print(_sub_line("Other", "other"), file=out_file)
    print(f"  Total substitutions: {total_subs}", file=out_file)

    print("\n(c) Top 10 clips with the most deletions:", file=out_file)
    if not audit["top_deletions"]:
        print("  (none)", file=out_file)
    else:
        for item in audit["top_deletions"]:
            r_str = f"{item['ratio']:.2f}" if item["ratio"] is not None else "n/a"
            c_val = item["coverage"]
            c_str = f"{float(c_val):.2f}" if isinstance(c_val, (int, float)) and not isinstance(c_val, bool) else "n/a"
            print(
                f"  - {item['clip_id']} ({item['condition']}, {item['language']}): "
                f"deletions={item['deletions']}, ratio={r_str}, coverage={c_str}",
                file=out_file,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only error audit over raw transcripts.")
    parser.add_argument("run_dir", help="Path to raw run folder (results/raw/<run_id>)")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSONL file")
    parser.add_argument("--language", default=None, help="Optional language filter (hi, gu, hi-en)")
    parser.add_argument("--condition", default=None, help="Optional condition filter (e.g. clean, phone)")
    args = parser.parse_args(argv)
    return run_error_audit(
        args.run_dir,
        args.manifest,
        language=args.language,
        condition=args.condition,
    )


if __name__ == "__main__":
    raise SystemExit(main())

