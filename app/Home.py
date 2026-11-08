"""Read-only dashboard. Shows saved results from results/processed/; never runs models."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

CONDITIONS = ["clean", "phone", "phone_noise_20", "phone_noise_10", "phone_noise_5"]

st.set_page_config(page_title="BharatSpeechBench", layout="wide")
st.title("BharatSpeechBench")
st.caption("Indian-language ASR under clean, telephone and noisy audio. "
           "Run experiments with the `bsb` CLI; this page only reads saved results.")

proc = Path("results/processed")
files = sorted(proc.glob("*_summary.parquet")) if proc.exists() else []
if not files:
    st.info("No results yet. Run `bsb run` then `bsb report` (see README).")
    st.stop()

choice = st.selectbox("Run", files, format_func=lambda p: p.name.removesuffix("_summary.parquet"))
df = pd.read_parquet(choice)
per_clip_file = choice.with_name(choice.name.replace("_summary", "_per_clip"))

langs = st.multiselect("Languages", sorted(df.language.unique()), default=sorted(df.language.unique()))
metric = st.radio("Metric", ["WER", "CER"], horizontal=True)
col = "wer_norm" if metric == "WER" else "cer_norm"
lo, hi = f"{col}_lo", f"{col}_hi"
view = df[df.language.isin(langs) & (df.n_clips > 0)]

fig = go.Figure()
for lang, g in view.groupby("language"):
    g = g.set_index("condition").reindex([c for c in CONDITIONS if c in set(g.condition)])
    fig.add_trace(go.Bar(
        name=f"{lang} (n={int(g.n_clips.max())})", x=g.index, y=100 * g[col],
        error_y=dict(type="data", symmetric=False,
                     array=100 * (g[hi] - g[col]), arrayminus=100 * (g[col] - g[lo]))))
fig.update_layout(barmode="group", yaxis_title=f"{metric} % (lower is better)",
                  xaxis_title="audio condition", legend_title="language")
st.plotly_chart(fig, width="stretch")
st.caption("Bars: corpus-level, normalised text. Whiskers: bootstrap 95% interval over clips. "
           "Few clips means wide whiskers: treat small gaps as noise.")

st.subheader("Summary table")
st.dataframe(df, width="stretch")

if per_clip_file.exists():
    st.subheader("Worst clips (normalised word errors)")
    pc = pd.read_parquet(per_clip_file)
    pc = pc[pc.error.isna() & pc.language.isin(langs)].copy()
    pc["wer_clip"] = pc.word_err_norm / pc.word_len_norm
    st.dataframe(pc.sort_values("wer_clip", ascending=False).head(25)
                 [["clip_id", "language", "condition", "wer_clip", "hypothesis"]],
                 width="stretch")
    failed = int(df.n_failed.sum())
    if failed:
        st.warning(f"{failed} (clip, condition) runs failed and are excluded from the scores.")
