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

from ..metrics.normalization import fold_devanagari_to_gujarati, normalize, normalize_lenient_hi, script_stats
from ..metrics.numbers import number_accuracy, number_counts
from ..metrics.wer import Counts, bootstrap_ci, paired_bootstrap_diff, per_utterance
from .manifest import Clip

CONDITION_ORDER = ["clean", "phone", "phone_noise_20", "phone_noise_10", "phone_noise_5"]
EARLY_STOP_RATIO_THRESHOLD = 0.8
RUNAWAY_RATIO_THRESHOLD = 1.5
NEAR_TOKEN_CAP_THRESHOLD = 210


def classify_clip_flags(
    reference: str | None,
    hypothesis: str | None,
    max_segment_tokens: int | float | None = None,
    language: str | None = None,
) -> dict:
    """Classify length-ratio anomalies (early_stop, runaway) and informational near_token_cap.

    Uses normalised word-count ratio for Gujarati (`gu`, because ASR outputs in Devanagari or Latin
    differ systematically in character length across scripts) and normalised character-count ratio
    for Hindi (`hi`) and Hinglish (`hi-en`).
    """
    ref_str = "" if reference is None or pd.isna(reference) else str(reference)
    hyp_str = "" if hypothesis is None or pd.isna(hypothesis) else str(hypothesis)
    norm_ref = normalize(ref_str)
    norm_hyp = normalize(hyp_str)
    if language == "gu":
        ref_words = norm_ref.split()
        hyp_words = norm_hyp.split()
        len_ratio_norm = (len(hyp_words) / len(ref_words)) if ref_words else None
    else:
        len_ratio_norm = (len(norm_hyp) / len(norm_ref)) if len(norm_ref) > 0 else None

    if (
        isinstance(max_segment_tokens, (int, float))
        and not isinstance(max_segment_tokens, bool)
        and not pd.isna(max_segment_tokens)
    ):
        max_tok: int | None = int(max_segment_tokens)
    else:
        max_tok = None

    early_stop = bool(len_ratio_norm is not None and len_ratio_norm < EARLY_STOP_RATIO_THRESHOLD)
    runaway = bool(len_ratio_norm is not None and len_ratio_norm > RUNAWAY_RATIO_THRESHOLD)
    near_token_cap = bool(max_tok is not None and max_tok >= NEAR_TOKEN_CAP_THRESHOLD)
    return {
        "len_ratio_norm": len_ratio_norm,
        "max_segment_tokens": max_tok,
        "early_stop": early_stop,
        "runaway": runaway,
        "near_token_cap": near_token_cap,
    }


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
        meta = r.get("meta") or {}
        coverage = meta.get("coverage")
        total_tokens = meta.get("total_tokens")
        max_seg_tokens = meta.get("max_segment_tokens")
        reference = ref[r["clip_id"]].reference
        base = {"clip_id": r["clip_id"], "language": r["language"], "condition": r["condition"],
                "provider": r["provider"], "model": r["model"], "error": r["error"],
                "audio_seconds": r["audio_seconds"], "processing_seconds": r["processing_seconds"],
                "rtf": r["rtf"], "coverage": coverage,
                "total_tokens": total_tokens, "max_segment_tokens": max_seg_tokens,
                "hypothesis": r["text"], "reference": reference, "source": ref[r["clip_id"]].source}
        if r["error"] is None:
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
            flags = classify_clip_flags(reference, hyp, max_seg_tokens, language=r["language"])
            sm, dom = script_stats(hyp, r["language"])
            base.update(
                num_matched=nc.matched,
                num_total=nc.total,
                len_ratio_norm=flags["len_ratio_norm"],
                early_stop=flags["early_stop"],
                runaway=flags["runaway"],
                near_token_cap=flags["near_token_cap"],
                script_match=sm,
                dominant_script=dom,
            )
            if r["language"] == "gu":
                folded_hyp = fold_devanagari_to_gujarati(hyp)
                wf = per_utterance([reference], [folded_hyp], "word", True)[0]
                chf = per_utterance([reference], [folded_hyp], "char", True)[0]
                base.update(
                    word_err_folded=wf.errors,
                    word_len_folded=wf.ref_len,
                    char_err_folded=chf.errors,
                    char_len_folded=chf.ref_len,
                )
            else:
                base.update(word_err_folded=None, word_len_folded=None, char_err_folded=None, char_len_folded=None)
            if r["language"] == "hi":
                len_ref = normalize_lenient_hi(reference)
                len_hyp = normalize_lenient_hi(hyp)
                wl = per_utterance([len_ref], [len_hyp], "word", False)[0]
                chl = per_utterance([len_ref], [len_hyp], "char", False)[0]
                base.update(
                    word_err_lenient=wl.errors,
                    word_len_lenient=wl.ref_len,
                    char_err_lenient=chl.errors,
                    char_len_lenient=chl.ref_len,
                    wer_lenient=(wl.errors / wl.ref_len) if wl.ref_len > 0 else None,
                    cer_lenient=(chl.errors / chl.ref_len) if chl.ref_len > 0 else None,
                )
            else:
                base.update(
                    word_err_lenient=None,
                    word_len_lenient=None,
                    char_err_lenient=None,
                    char_len_lenient=None,
                    wer_lenient=None,
                    cer_lenient=None,
                )
        else:
            base.update(
                len_ratio_norm=None,
                early_stop=False,
                runaway=False,
                near_token_cap=False,
                script_match=None,
                dominant_script=None,
                word_err_folded=None,
                word_len_folded=None,
                char_err_folded=None,
                char_len_folded=None,
                word_err_lenient=None,
                word_len_lenient=None,
                char_err_lenient=None,
                char_len_lenient=None,
                wer_lenient=None,
                cer_lenient=None,
            )
        rows.append(base)
    df = pd.DataFrame(rows)
    for col in ("coverage", "len_ratio_norm", "script_match", "wer_lenient", "cer_lenient"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in (
        "total_tokens",
        "max_segment_tokens",
        "word_err_folded",
        "word_len_folded",
        "char_err_folded",
        "char_len_folded",
        "word_err_lenient",
        "word_len_lenient",
        "char_err_lenient",
        "char_len_lenient",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("early_stop", "runaway", "near_token_cap"):
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    return df


def _ci(df: pd.DataFrame, kind: str, tag: str) -> tuple[float, float]:
    counts = [Counts(int(e), int(n)) for e, n in zip(df[f"{kind}_err_{tag}"], df[f"{kind}_len_{tag}"])]
    return bootstrap_ci(counts)


def _flagged_clip_ids(df: pd.DataFrame) -> set[str]:
    """Return clip_ids flagged early_stop or runaway in ANY row/condition of df."""
    if not len(df):
        return set()
    ok = df[df["error"].isna()] if "error" in df.columns else df
    es = ok["early_stop"].fillna(False).astype(bool) if "early_stop" in ok.columns else pd.Series(False, index=ok.index)
    ra = ok["runaway"].fillna(False).astype(bool) if "runaway" in ok.columns else pd.Series(False, index=ok.index)
    return set(ok.loc[es | ra, "clip_id"].astype(str))


def summary_table(per_clip: pd.DataFrame) -> pd.DataFrame:
    flagged_by_model = {
        model_name: _flagged_clip_ids(mg)
        for model_name, mg in per_clip.groupby("model", sort=False)
    }
    out = []
    for (model, lang, cond), g in per_clip.groupby(["model", "language", "condition"], sort=False):
        ok = g[g["error"].isna()]
        n_empty = int((ok["hypothesis"].fillna("").astype(str).str.strip() == "").sum()) if len(ok) else 0
        if len(ok) and "coverage" in ok.columns:
            cov = pd.to_numeric(ok["coverage"], errors="coerce")
            n_low_coverage = int((cov < 0.85).sum())
        else:
            n_low_coverage = 0
        early_stop_cnt = int(ok["early_stop"].fillna(False).astype(bool).sum()) if len(ok) and "early_stop" in ok.columns else 0
        runaway_cnt = int(ok["runaway"].fillna(False).astype(bool).sum()) if len(ok) and "runaway" in ok.columns else 0
        near_cap_cnt = int(ok["near_token_cap"].fillna(False).astype(bool).sum()) if len(ok) and "near_token_cap" in ok.columns else 0

        flagged_ids = flagged_by_model.get(model, set())
        ok_excl = ok[~ok["clip_id"].astype(str).isin(flagged_ids)] if len(ok) else ok

        row = {"model": model, "language": lang, "condition": cond,
               "n_clips": len(ok), "n_failed": int(len(g) - len(ok)),
               "n_empty_hyp": n_empty, "n_low_coverage": n_low_coverage,
               "early_stop": early_stop_cnt, "runaway": runaway_cnt, "near_cap": near_cap_cnt,
               "n_clips_excl": int(len(ok_excl))}
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

            clip_wers = ok["word_err_norm"] / ok["word_len_norm"].replace(0, np.nan)
            row["wer_median"] = float(np.nanmedian(clip_wers)) if clip_wers.notna().any() else float("nan")
            if len(ok_excl) and ok_excl["word_len_norm"].sum() > 0:
                row["wer_excl_flagged"] = float(ok_excl["word_err_norm"].sum() / ok_excl["word_len_norm"].sum())
            else:
                row["wer_excl_flagged"] = float("nan")

            from ..metrics.numbers import NumberCounts
            row["number_acc"] = number_accuracy(
                [NumberCounts(int(m), int(t)) for m, t in zip(ok["num_matched"], ok["num_total"])])
            row["rtf_median"] = float(np.median(ok["rtf"]))
            if "len_ratio_norm" in ok.columns and ok["len_ratio_norm"].notna().any():
                row["len_ratio_median"] = float(np.median(ok["len_ratio_norm"].dropna()))
            else:
                row["len_ratio_median"] = float("nan")
            if "script_match" in ok.columns and ok["script_match"].notna().any():
                row["script_match_mean"] = float(ok["script_match"].dropna().mean())
            else:
                row["script_match_mean"] = float("nan")
            if (
                lang == "gu"
                and "word_err_folded" in ok.columns
                and ok["word_len_folded"].notna().any()
                and ok["word_len_folded"].sum() > 0
            ):
                row["wer_folded"] = float(ok["word_err_folded"].sum() / ok["word_len_folded"].sum())
                row["cer_folded"] = float(ok["char_err_folded"].sum() / ok["char_len_folded"].sum())
            else:
                row["wer_folded"] = float("nan")
                row["cer_folded"] = float("nan")
            if (
                lang == "hi"
                and "word_err_lenient" in ok.columns
                and ok["word_len_lenient"].notna().any()
                and ok["word_len_lenient"].sum() > 0
            ):
                row["wer_lenient"] = float(ok["word_err_lenient"].sum() / ok["word_len_lenient"].sum())
                row["cer_lenient"] = (
                    float(ok["char_err_lenient"].sum() / ok["char_len_lenient"].sum())
                    if ok["char_len_lenient"].sum() > 0
                    else float("nan")
                )
            else:
                row["wer_lenient"] = float("nan")
                row["cer_lenient"] = float("nan")
        else:
            row["word_sub_norm"] = 0
            row["word_del_norm"] = 0
            row["word_ins_norm"] = 0
            row["wer_median"] = float("nan")
            row["wer_excl_flagged"] = float("nan")
            row["script_match_mean"] = float("nan")
            row["wer_folded"] = float("nan")
            row["cer_folded"] = float("nan")
            row["wer_lenient"] = float("nan")
            row["cer_lenient"] = float("nan")
        out.append(row)
    df = pd.DataFrame(out)
    df["_c"] = df["condition"].map({c: i for i, c in enumerate(CONDITION_ORDER)}).fillna(99)
    return df.sort_values(["model", "language", "_c"]).drop(columns="_c").reset_index(drop=True)


def _pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def _pct_or_blank(x) -> str:
    return "" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def _format_flagged_line(r: pd.Series, fl: dict) -> str:
    ratio_str = f"{fl['len_ratio_norm']:.2f}" if fl["len_ratio_norm"] is not None else "n/a"
    max_tok = fl["max_segment_tokens"]
    tot_tok = r.get("total_tokens")
    if max_tok is not None and tot_tok is not None and pd.notna(tot_tok) and int(tot_tok) != int(max_tok):
        tok_str = f"{max_tok} (total {int(tot_tok)})"
    elif max_tok is not None:
        tok_str = str(max_tok)
    else:
        tok_str = "n/a"
    cov_val = r.get("coverage")
    cov_str = f"{float(cov_val):.2f}" if cov_val is not None and pd.notna(cov_val) else "n/a"
    return f"- `{r['clip_id']}` ({r['condition']}): ratio={ratio_str}, tokens={tok_str}, coverage={cov_str}"


def _paired_table_lines(lang_df: pd.DataFrame, all_conditions: list[str]) -> list[str]:
    lines = [
        "| condition | clean WER | condition WER | absolute difference | relative degradation | 95% paired CI | n clips |",
        "|---|---|---|---|---|---|---|",
    ]
    clean_df = lang_df[(lang_df["condition"] == "clean") & lang_df["error"].isna()]
    clean_by_id = {r["clip_id"]: r for _, r in clean_df.iterrows()}

    for cond in all_conditions:
        cond_df = lang_df[(lang_df["condition"] == cond) & lang_df["error"].isna()]
        cond_by_id = {r["clip_id"]: r for _, r in cond_df.iterrows()}
        paired_ids = sorted([cid for cid in clean_by_id if cid in cond_by_id])
        n_paired = len(paired_ids)

        if n_paired == 0:
            lines.append(f"| {cond} | n/a | n/a | n/a | n/a | too few clips for an interval | 0 |")
            continue

        c_counts = [Counts(int(clean_by_id[cid]["word_err_norm"]), int(clean_by_id[cid]["word_len_norm"])) for cid in paired_ids]
        b_counts = [Counts(int(cond_by_id[cid]["word_err_norm"]), int(cond_by_id[cid]["word_len_norm"])) for cid in paired_ids]
        c_sum = sum(c.ref_len for c in c_counts)
        b_sum = sum(c.ref_len for c in b_counts)
        c_rate = sum(c.errors for c in c_counts) / c_sum if c_sum else 0.0
        b_rate = sum(c.errors for c in b_counts) / b_sum if b_sum else 0.0

        if n_paired < 5:
            diff = b_rate - c_rate
            ci_str = "too few clips for an interval"
        else:
            diff, lo, hi = paired_bootstrap_diff(c_counts, b_counts)
            ci_str = f"{lo:+.1%} to {hi:+.1%}"

        rel = (diff / c_rate) if c_rate > 0 else (0.0 if diff == 0 else None)
        rel_str = f"{100 * rel:+.1f}%" if rel is not None else "n/a"
        lines.append(f"| {cond} | {_pct(c_rate)} | {_pct(b_rate)} | {diff:+.1%} | {rel_str} | {ci_str} | {n_paired} |")

    return lines


def markdown_report(run_id: str, summary: pd.DataFrame, config: dict,
                    per_clip: pd.DataFrame | None = None) -> str:
    env = config.get("environment", {})
    lines = [f"# Run `{run_id}`", "",
             f"- Provider: {config['provider']} / model `{config['model']}`  settings: {config.get('provider_settings')}",
             f"- Split: {config['split']}, conditions: {', '.join(config['conditions'])}",
             f"- Git commit: {env.get('git_commit')}, git dirty: {env.get('git_dirty')}, manifest sha256: {env.get('manifest_sha256')}",
             f"- Seed: {env.get('seed')}, GPU: {env.get('gpu')}", ""]
    if env.get("git_dirty") is True:
        lines += ["WARNING: uncommitted changes when this run was made.", ""]
    lines += [
        "WER/CER are corpus-level (normalised unless marked raw), with a bootstrap 95% interval over clips. Primary WER includes all succeeded clips and is shown first.",
        "Sensitivity columns (`wer_median`, `wer_excl_flagged`, `n_kept`): secondary checks; `wer_excl_flagged` drops any clip flagged `early_stop` or `runaway` in any condition identically across all conditions (`near_cap` is informational only).",
        "`wer_lenient` / `cer_lenient` (Hindi only; secondary: ignores nukta and chandrabindu/anusvara spelling).",
        "`wer_folded` / `cer_folded` (Gujarati only; approximate: ignores script, not spelling conventions).",
        "Small clip counts give wide intervals: do not read small gaps as real differences.", "",
        "| language | condition | clips | failed | empty | S | D | I | WER | WER 95% CI | WER raw | CER | wer_median (sensitivity) | wer_excl_flagged (sensitivity) | n_kept (sensitivity) | wer_lenient | cer_lenient | wer_folded | cer_folded | script match | number acc | RTF | low coverage | early_stop | runaway | near_cap |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        if r["n_clips"] == 0:
            lines.append(
                f"| {r['language']} | {r['condition']} | 0 | {r['n_failed']} | 0 | 0 | 0 | 0 | n/a | | | | n/a | n/a | 0 | | | | | n/a | | | "
                f"{r.get('n_low_coverage', 0)} | {r.get('early_stop', 0)} | {r.get('runaway', 0)} | {r.get('near_cap', 0)} |"
            )
            continue
        lines.append(
            f"| {r['language']} | {r['condition']} | {r['n_clips']} | {r['n_failed']} | "
            f"{r.get('n_empty_hyp', 0)} | {r.get('word_sub_norm', 0)} | {r.get('word_del_norm', 0)} | {r.get('word_ins_norm', 0)} | "
            f"{_pct(r['wer_norm'])} | {_pct(r['wer_norm_lo'])} to {_pct(r['wer_norm_hi'])} | "
            f"{_pct(r['wer_raw'])} | {_pct(r['cer_norm'])} | "
            f"{_pct(r.get('wer_median'))} | {_pct(r.get('wer_excl_flagged'))} | {r.get('n_clips_excl', r['n_clips'])} | "
            f"{_pct_or_blank(r.get('wer_lenient'))} | {_pct_or_blank(r.get('cer_lenient'))} | "
            f"{_pct_or_blank(r.get('wer_folded'))} | {_pct_or_blank(r.get('cer_folded'))} | "
            f"{_pct(r.get('script_match_mean'))} | {_pct(r['number_acc'])} | {r['rtf_median']:.2f} | "
            f"{r.get('n_low_coverage', 0)} | {r.get('early_stop', 0)} | {r.get('runaway', 0)} | {r.get('near_cap', 0)} |")

    if per_clip is not None and len(per_clip):
        ok = per_clip[per_clip["error"].isna()].copy()
        if len(ok):
            ok["clip_wer"] = ok["word_err_norm"] / ok["word_len_norm"].replace(0, np.nan)
            top5 = ok.sort_values(by=["clip_wer", "word_err_norm"], ascending=[False, False]).head(5)
            lines += ["", "## Highest-error clips"]
            for _, r in top5.iterrows():
                wer_str = _pct(r["clip_wer"])
                lines.append(f"- `{r['clip_id']}` ({r['condition']}): {wer_str}")

            early_rows = []
            runaway_rows = []
            for _, r in ok.iterrows():
                max_tok = r.get("max_segment_tokens")
                if "reference" in r and pd.notna(r["reference"]):
                    fl = classify_clip_flags(str(r["reference"]), r.get("hypothesis"), max_tok, language=r.get("language"))
                else:
                    lr = r.get("len_ratio_norm")
                    ratio_val = float(lr) if lr is not None and pd.notna(lr) else None
                    max_tok_int = int(max_tok) if max_tok is not None and pd.notna(max_tok) else None
                    fl = {
                        "len_ratio_norm": ratio_val,
                        "max_segment_tokens": max_tok_int,
                        "early_stop": bool(r["early_stop"]) if "early_stop" in r and pd.notna(r["early_stop"]) else (ratio_val is not None and ratio_val < EARLY_STOP_RATIO_THRESHOLD),
                        "runaway": bool(r["runaway"]) if "runaway" in r and pd.notna(r["runaway"]) else (ratio_val is not None and ratio_val > RUNAWAY_RATIO_THRESHOLD),
                        "near_token_cap": bool(r["near_token_cap"]) if "near_token_cap" in r and pd.notna(r["near_token_cap"]) else (max_tok_int is not None and max_tok_int >= NEAR_TOKEN_CAP_THRESHOLD),
                    }
                if fl["early_stop"]:
                    early_rows.append((r, fl))
                if fl["runaway"]:
                    runaway_rows.append((r, fl))

            if early_rows or runaway_rows:
                lines += ["", "## Flagged clips"]
                lines += ["", f"### Early stops (ratio < {EARLY_STOP_RATIO_THRESHOLD})"]
                if early_rows:
                    early_rows.sort(key=lambda x: (x[1]["len_ratio_norm"] if x[1]["len_ratio_norm"] is not None else 0.0, str(x[0]["clip_id"]), str(x[0]["condition"])))
                    lines.extend(_format_flagged_line(r, fl) for r, fl in early_rows)
                else:
                    lines.append("- None")
                lines += ["", f"### Runaway outputs (ratio > {RUNAWAY_RATIO_THRESHOLD})"]
                if runaway_rows:
                    runaway_rows.sort(key=lambda x: (-(x[1]["len_ratio_norm"] if x[1]["len_ratio_norm"] is not None else 0.0), str(x[0]["clip_id"]), str(x[0]["condition"])))
                    lines.extend(_format_flagged_line(r, fl) for r, fl in runaway_rows)
                else:
                    lines.append("- None")

        lines += ["", "## Paired comparison vs clean"]
        flagged_all = _flagged_clip_ids(per_clip)
        languages = sorted(per_clip["language"].unique())
        for lang in languages:
            lang_df = per_clip[per_clip["language"] == lang]
            non_clean_conds = [c for c in CONDITION_ORDER if c in lang_df["condition"].unique() and c != "clean"]
            for c in lang_df["condition"].unique():
                if c != "clean" and c not in non_clean_conds:
                    non_clean_conds.append(c)

            lines += ["", f"### {lang}"]
            lines.extend(_paired_table_lines(lang_df, non_clean_conds))

            lines += ["", f"#### {lang} (sensitivity: excluding early_stop/runaway clips)"]
            lang_excl = lang_df[~lang_df["clip_id"].astype(str).isin(flagged_all)]
            lines.extend(_paired_table_lines(lang_excl, non_clean_conds))

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
