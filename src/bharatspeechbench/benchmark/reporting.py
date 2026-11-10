"""Score a finished (or partial) run: raw transcripts -> per-clip table, summary table, markdown report.

Written to:
  results/processed/<run_id>_per_clip.parquet   one row per (clip, condition), all counts
  results/processed/<run_id>_summary.parquet    one row per (model, language, condition)
  results/reports/<run_id>.md                   human-readable tables + the limits of the numbers
Failed clips (provider errors) are excluded from scores and counted in `n_failed`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics.numbers import number_accuracy, number_counts
from ..metrics.wer import Counts, bootstrap_ci, corpus_rate, paired_bootstrap_diff, per_utterance
from .manifest import Clip

CONDITION_ORDER = ["clean", "phone", "phone_noise_20", "phone_noise_10", "phone_noise_5"]


def _load_transcripts(run_dir: Path) -> list[dict]:
    path = run_dir / "transcripts.jsonl"
    by_key = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_key[(row["clip_id"], row["condition"])] = row
    return list(by_key.values())


def per_clip_table(run_dir: str | Path, clips: list[Clip]) -> pd.DataFrame:
    run_dir = Path(run_dir)
    ref = {c.id: c for c in clips}
    rows = []
    for r in _load_transcripts(run_dir):
        base = {"clip_id": r["clip_id"], "language": r["language"], "condition": r["condition"],
                "provider": r["provider"], "model": r["model"], "error": r["error"],
                "audio_seconds": r["audio_seconds"], "processing_seconds": r["processing_seconds"],
                "rtf": r["rtf"], "hypothesis": r["text"], "source": ref[r["clip_id"]].source}
        if r["error"] is None:
            reference = ref[r["clip_id"]].reference
            hyp = r["text"]
            for tag, norm in (("raw", False), ("norm", True)):
                w = per_utterance([reference], [hyp], "word", norm)[0]
                ch = per_utterance([reference], [hyp], "char", norm)[0]
                base.update({
                    f"word_err_{tag}": w.errors,
                    f"word_sub_{tag}": w.substitutions,
                    f"word_del_{tag}": w.deletions,
                    f"word_ins_{tag}": w.insertions,
                    f"word_len_{tag}": w.ref_len,
                    f"char_err_{tag}": ch.errors,
                    f"char_sub_{tag}": ch.substitutions,
                    f"char_del_{tag}": ch.deletions,
                    f"char_ins_{tag}": ch.insertions,
                    f"char_len_{tag}": ch.ref_len,
                })
            nc = number_counts(reference, hyp)
            base.update(num_matched=nc.matched, num_total=nc.total)
        rows.append(base)
    return pd.DataFrame(rows)


def _ci(df: pd.DataFrame, kind: str, tag: str) -> tuple[float, float]:
    counts = [Counts(int(e), int(n)) for e, n in zip(df[f"{kind}_err_{tag}"], df[f"{kind}_len_{tag}"])]
    return bootstrap_ci(counts)


def summary_table(per_clip: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, lang, cond), g in per_clip.groupby(["model", "language", "condition"], sort=False):
        ok = g[g["error"].isna()]
        n_empty = int((ok["hypothesis"].fillna("").astype(str).str.strip() == "").sum()) if len(ok) else 0
        row = {"model": model, "language": lang, "condition": cond,
               "n_clips": len(ok), "n_failed": int(len(g) - len(ok)), "n_empty_hyp": n_empty}
        if len(ok):
            for kind in ("word", "char"):
                for tag in ("raw", "norm"):
                    row[f"{'wer' if kind == 'word' else 'cer'}_{tag}"] = (
                        ok[f"{kind}_err_{tag}"].sum() / ok[f"{kind}_len_{tag}"].sum())
            row["word_sub_norm"] = int(ok["word_sub_norm"].sum())
            row["word_del_norm"] = int(ok["word_del_norm"].sum())
            row["word_ins_norm"] = int(ok["word_ins_norm"].sum())
            row["wer_norm_lo"], row["wer_norm_hi"] = _ci(ok, "word", "norm")
            row["cer_norm_lo"], row["cer_norm_hi"] = _ci(ok, "char", "norm")
            from ..metrics.numbers import NumberCounts
            row["number_acc"] = number_accuracy(
                [NumberCounts(int(m), int(t)) for m, t in zip(ok["num_matched"], ok["num_total"])])
            row["rtf_median"] = float(np.median(ok["rtf"]))
        else:
            row["word_sub_norm"] = 0
            row["word_del_norm"] = 0
            row["word_ins_norm"] = 0
        out.append(row)
    df = pd.DataFrame(out)
    df["_c"] = df["condition"].map({c: i for i, c in enumerate(CONDITION_ORDER)}).fillna(99)
    return df.sort_values(["model", "language", "_c"]).drop(columns="_c").reset_index(drop=True)


def _pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def markdown_report(run_id: str, summary: pd.DataFrame, config: dict,
                    per_clip: pd.DataFrame | None = None) -> str:
    env = config.get("environment", {})
    lines = [f"# Run `{run_id}`", "",
             f"- Provider: {config['provider']} / model `{config['model']}`  settings: {config.get('provider_settings')}",
             f"- Split: {config['split']}, conditions: {', '.join(config['conditions'])}",
             f"- Git commit: {env.get('git_commit')}, manifest sha256: {env.get('manifest_sha256')}",
             f"- Seed: {env.get('seed')}, GPU: {env.get('gpu')}", "",
             "WER/CER are corpus-level (normalised unless marked raw), with a bootstrap 95% interval over clips.",
             "Small clip counts give wide intervals: do not read small gaps as real differences.", "",
             "| language | condition | clips | failed | empty | S | D | I | WER | WER 95% CI | WER raw | CER | number acc | RTF |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        if r["n_clips"] == 0:
            lines.append(f"| {r['language']} | {r['condition']} | 0 | {r['n_failed']} | 0 | 0 | 0 | 0 | n/a | | | | | |")
            continue
        lines.append(
            f"| {r['language']} | {r['condition']} | {r['n_clips']} | {r['n_failed']} | "
            f"{r.get('n_empty_hyp', 0)} | {r.get('word_sub_norm', 0)} | {r.get('word_del_norm', 0)} | {r.get('word_ins_norm', 0)} | "
            f"{_pct(r['wer_norm'])} | {_pct(r['wer_norm_lo'])} to {_pct(r['wer_norm_hi'])} | "
            f"{_pct(r['wer_raw'])} | {_pct(r['cer_norm'])} | {_pct(r['number_acc'])} | {r['rtf_median']:.2f} |")

    if per_clip is not None and len(per_clip):
        ok = per_clip[per_clip["error"].isna()].copy()
        if len(ok):
            ok["clip_wer"] = ok["word_err_norm"] / ok["word_len_norm"].replace(0, np.nan)
            top5 = ok.sort_values(by=["clip_wer", "word_err_norm"], ascending=[False, False]).head(5)
            lines += ["", "## Highest-error clips"]
            for _, r in top5.iterrows():
                wer_str = _pct(r["clip_wer"])
                lines.append(f"- `{r['clip_id']}` ({r['condition']}): {wer_str}")

        lines += ["", "## Paired comparison vs clean"]
        languages = sorted(per_clip["language"].unique())
        for lang in languages:
            lines += ["", f"### {lang}",
                      "| condition | clean WER | condition WER | absolute difference | relative degradation | 95% paired CI | n clips |",
                      "|---|---|---|---|---|---|---|"]
            lang_df = per_clip[per_clip["language"] == lang]
            clean_df = lang_df[(lang_df["condition"] == "clean") & lang_df["error"].isna()]
            clean_by_id = {r["clip_id"]: r for _, r in clean_df.iterrows()}
            non_clean_conds = [c for c in CONDITION_ORDER if c in lang_df["condition"].unique() and c != "clean"]
            for c in lang_df["condition"].unique():
                if c != "clean" and c not in non_clean_conds:
                    non_clean_conds.append(c)

            for cond in non_clean_conds:
                cond_df = lang_df[(lang_df["condition"] == cond) & lang_df["error"].isna()]
                cond_by_id = {r["clip_id"]: r for _, r in cond_df.iterrows()}
                paired_ids = sorted([cid for cid in clean_by_id if cid in cond_by_id])
                n_paired = len(paired_ids)

                if n_paired == 0:
                    lines.append(f"| {cond} | n/a | n/a | n/a | n/a | too few clips for an interval | 0 |")
                elif n_paired < 5:
                    c_counts = [Counts(int(clean_by_id[cid]["word_err_norm"]), int(clean_by_id[cid]["word_len_norm"])) for cid in paired_ids]
                    b_counts = [Counts(int(cond_by_id[cid]["word_err_norm"]), int(cond_by_id[cid]["word_len_norm"])) for cid in paired_ids]
                    c_sum = sum(c.ref_len for c in c_counts)
                    b_sum = sum(c.ref_len for c in b_counts)
                    c_rate = sum(c.errors for c in c_counts) / c_sum if c_sum else 0.0
                    b_rate = sum(c.errors for c in b_counts) / b_sum if b_sum else 0.0
                    diff = b_rate - c_rate
                    rel = (diff / c_rate) if c_rate > 0 else (0.0 if diff == 0 else None)
                    rel_str = f"{100 * rel:+.1f}%" if rel is not None else "n/a"
                    lines.append(f"| {cond} | {_pct(c_rate)} | {_pct(b_rate)} | {diff:+.1%} | {rel_str} | too few clips for an interval | {n_paired} |")
                else:
                    c_counts = [Counts(int(clean_by_id[cid]["word_err_norm"]), int(clean_by_id[cid]["word_len_norm"])) for cid in paired_ids]
                    b_counts = [Counts(int(cond_by_id[cid]["word_err_norm"]), int(cond_by_id[cid]["word_len_norm"])) for cid in paired_ids]
                    c_sum = sum(c.ref_len for c in c_counts)
                    b_sum = sum(c.ref_len for c in b_counts)
                    c_rate = sum(c.errors for c in c_counts) / c_sum if c_sum else 0.0
                    b_rate = sum(c.errors for c in b_counts) / b_sum if b_sum else 0.0
                    diff, lo, hi = paired_bootstrap_diff(c_counts, b_counts)
                    rel = (diff / c_rate) if c_rate > 0 else (0.0 if diff == 0 else None)
                    rel_str = f"{100 * rel:+.1f}%" if rel is not None else "n/a"
                    lines.append(f"| {cond} | {_pct(c_rate)} | {_pct(b_rate)} | {diff:+.1%} | {rel_str} | {lo:+.1%} to {hi:+.1%} | {n_paired} |")

    lines += ["", "Limits: see docs/methodology.md (phone channel is a simulation; Hinglish script policy; "
              "public sets may overlap with training data; number accuracy compares digits only)."]
    return "\n".join(lines) + "\n"


def report_run(run_dir: str | Path, clips: list[Clip], results_dir: str | Path = "results") -> dict[str, Path]:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    per_clip = per_clip_table(run_dir, clips)
    summary = summary_table(per_clip)
    proc, rep = Path(results_dir) / "processed", Path(results_dir) / "reports"
    proc.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    paths = {"per_clip": proc / f"{run_id}_per_clip.parquet",
             "summary": proc / f"{run_id}_summary.parquet",
             "report": rep / f"{run_id}.md"}
    per_clip.to_parquet(paths["per_clip"], index=False)
    summary.to_parquet(paths["summary"], index=False)
    paths["report"].write_text(markdown_report(run_id, summary, config, per_clip=per_clip), encoding="utf-8")
    return paths
