"""End-to-end pipeline with a fake model on synthetic audio (no real ASR model needed)."""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from bharatspeechbench.audio.conditions import clip_seed, parse_condition
from bharatspeechbench.benchmark.manifest import Clip, load_manifest, write_manifest
from bharatspeechbench.benchmark.reporting import report_run
from bharatspeechbench.benchmark.runner import run_benchmark
from bharatspeechbench.cli import main
from bharatspeechbench.providers.base import ASRProvider
from bharatspeechbench.utils.hashing import sha256_file

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

REFS = {"a": "आज 12 workers आए", "b": "कल पाँच बजे मिलना है", "c": "આજે 5 વાગ્યે આવજો"}


class FakeProvider(ASRProvider):
    """Perfect on clean audio; drops the last word on phone audio; crashes on clip 'c' when noisy."""
    name = "fake"
    calls = 0

    @property
    def model_id(self):
        return "fake-1"

    def _transcribe(self, wav_path, language):
        FakeProvider.calls += 1
        p = Path(wav_path)
        clip_id, cond = p.stem, p.parent.name
        text = REFS[clip_id]
        if cond == "phone":
            text = " ".join(text.split()[:-1])
        if clip_id == "c" and cond.startswith("phone_noise"):
            raise RuntimeError("boom")
        return text, {}


@pytest.fixture
def dataset(tmp_path):
    t = np.arange(16000 * 2) / 16000
    clips = []
    for i, (cid, lang) in enumerate([("a", "hi-en"), ("b", "hi"), ("c", "gu")]):
        wav = tmp_path / "audio" / f"{cid}.wav"
        wav.parent.mkdir(exist_ok=True)
        sf.write(wav, 0.2 * np.sin(2 * np.pi * (200 + 50 * i) * t), 16000)
        clips.append(Clip(cid, f"audio/{cid}.wav", lang, "own", "test", REFS[cid], 2.0,
                          sha256_file(wav), "own", True))
    noise = tmp_path / "noise.wav"
    sf.write(noise, np.random.default_rng(0).normal(0, 0.1, 16000 * 3), 16000)
    man = tmp_path / "manifest.jsonl"
    write_manifest(clips, man)
    return tmp_path, man, noise


def test_parse_and_seed():
    assert parse_condition("phone_noise_10") == ("phone_noise", 10.0)
    with pytest.raises(ValueError):
        parse_condition("noisy")
    assert clip_seed(0, "a") == clip_seed(0, "a") != clip_seed(0, "b")


def test_run_report_resume_and_failures(dataset):
    tmp, man, noise = dataset
    clips = load_manifest(man)
    conds = ("clean", "phone", "phone_noise_10")
    run_dir = tmp / "results" / "raw" / "r1"
    kw = dict(root=tmp, conditions=conds, noise_wav=noise, conditions_dir=tmp / "cond",
              manifest_path=man, progress=lambda *_: None)
    run_benchmark(clips, FakeProvider(), run_dir, **kw)

    rows = [json.loads(x) for x in (run_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 9
    assert [r["error"] for r in rows if r["clip_id"] == "c" and r["condition"] == "phone_noise_10"][0].startswith("RuntimeError")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["model"] == "fake-1" and cfg["environment"]["manifest_sha256"] == sha256_file(man)

    # resume: successful clips are not transcribed twice; failed clips are retried
    before = FakeProvider.calls
    run_benchmark(clips, FakeProvider(), run_dir, **kw)
    assert FakeProvider.calls - before == 2          # 1 warm-up call + 1 retry of failed clip 'c'
    assert len((run_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()) == 10

    paths = report_run(run_dir, clips, tmp / "results")
    s = pd.read_parquet(paths["summary"])
    clean = s[s.condition == "clean"]
    assert (clean.wer_norm == 0).all() and (clean.cer_norm == 0).all()
    phone_a = s[(s.condition == "phone") & (s.language == "hi-en")].iloc[0]
    assert phone_a.wer_norm == pytest.approx(1 / 4)            # "आज 12 workers आए": last word dropped
    assert phone_a.number_acc == 1.0                            # the number "12" survived
    failed = s[(s.condition == "phone_noise_10") & (s.language == "gu")].iloc[0]
    assert failed.n_failed == 1 and failed.n_clips == 0         # failure is counted, not scored as empty
    assert "word_sub_norm" in s.columns and "n_empty_hyp" in s.columns
    pc = pd.read_parquet(paths["per_clip"])
    assert "word_sub_norm" in pc.columns and "word_del_norm" in pc.columns

    report_text = paths["report"].read_text(encoding="utf-8")
    assert "Run `r1`" in report_text
    assert "| empty | S | D | I |" in report_text
    assert "## Highest-error clips" in report_text
    assert "## Paired comparison vs clean" in report_text
    assert "too few clips for an interval" in report_text


def test_changed_setup_refuses_to_mix_runs(dataset):
    tmp, man, noise = dataset
    clips = load_manifest(man)
    run_dir = tmp / "r"
    kw = dict(root=tmp, noise_wav=noise, conditions_dir=tmp / "cond", manifest_path=man, progress=lambda *_: None)
    run_benchmark(clips, FakeProvider(), run_dir, conditions=("clean",), **kw)
    with pytest.raises(ValueError, match="different conditions"):
        run_benchmark(clips, FakeProvider(), run_dir, conditions=("clean", "phone"), **kw)


def test_noise_condition_needs_noise_file(dataset):
    tmp, man, _ = dataset
    with pytest.raises(ValueError, match="noise"):
        run_benchmark(load_manifest(man), FakeProvider(), tmp / "r", root=tmp,
                      conditions=("phone_noise_5",), noise_wav=None)


def test_simulate_command(dataset, capsys):
    tmp, man, noise = dataset
    rc = main(["simulate", str(man), "--root", str(tmp), "--noise", str(noise),
               "--conditions", "clean", "phone_noise_5", "--out", str(tmp / "cond"), "--limit", "1"])
    assert rc == 0
    out = capsys.readouterr().out.split()
    assert len(out) == 2 and all(p.endswith(".wav") for p in out)
    y, sr = sf.read(out[1])
    assert sr == 16000 and len(y) > 0


@pytest.mark.parametrize("change", [dict(seed=1), dict(languages=("hi",)), dict(limit=1)])
def test_resume_refuses_changed_seed_languages_limit(dataset, change):
    tmp, man, noise = dataset
    clips = load_manifest(man)
    run_dir = tmp / "r"
    base = dict(root=tmp, noise_wav=noise, conditions_dir=tmp / "cond", manifest_path=man,
                conditions=("clean",), progress=lambda *_: None)
    run_benchmark(clips, FakeProvider(), run_dir, **base)
    with pytest.raises(ValueError, match="different"):
        run_benchmark(clips, FakeProvider(), run_dir, **{**base, **change})


def test_resume_retries_failed_clips_and_reporting_dedup(dataset):
    tmp, man, _ = dataset
    clips = load_manifest(man)
    run_dir = tmp / "retry_run"
    kw = dict(root=tmp, conditions=("clean",), conditions_dir=tmp / "cond",
              manifest_path=man, progress=lambda *_: None)

    class FlakyProvider(ASRProvider):
        name = "flaky"
        model_id = "flaky-1"

        def __init__(self, should_fail: bool = True):
            self.should_fail = should_fail
            self.calls = 0

        def _transcribe(self, wav_path, language):
            self.calls += 1
            p = Path(wav_path)
            if p.stem == "b" and self.should_fail:
                raise RuntimeError("transient failure")
            return REFS[p.stem], {}

    # Run 1: clip 'b' fails
    p1 = FlakyProvider(should_fail=True)
    run_benchmark(clips, p1, run_dir, **kw)
    raw_lines = (run_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 3
    rows = [json.loads(line) for line in raw_lines]
    assert [r["error"] for r in rows if r["clip_id"] == "b"][0] is not None

    # Run 2 (retry fails again): test (a) failed clip retried, (c) counts in n_failed once
    p2 = FlakyProvider(should_fail=True)
    run_benchmark(clips, p2, run_dir, **kw)
    raw_lines2 = (run_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines2) == 4  # retried clip appends a new row
    paths_fail = report_run(run_dir, clips, tmp / "results_fail")
    s_fail = pd.read_parquet(paths_fail["summary"])
    assert s_fail["n_failed"].sum() == 1
    pc_fail = pd.read_parquet(paths_fail["per_clip"])
    assert len(pc_fail) == 3

    # Run 3 (retry succeeds): test (b) after retry succeeds, scored once
    p3 = FlakyProvider(should_fail=False)
    run_benchmark(clips, p3, run_dir, **kw)
    raw_lines3 = (run_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines3) == 5  # new row appended
    paths_ok = report_run(run_dir, clips, tmp / "results_ok")
    s_ok = pd.read_parquet(paths_ok["summary"])
    assert s_ok["n_failed"].sum() == 0
    assert s_ok["n_clips"].sum() == 3
    pc_ok = pd.read_parquet(paths_ok["per_clip"])
    assert len(pc_ok) == 3
    assert (pc_ok["clip_id"] == "b").sum() == 1
    assert pc_ok[pc_ok["clip_id"] == "b"]["error"].isna().all()


def test_limit_per_language(tmp_path):
    t = np.arange(16000) / 16000
    clips = []
    # 3 hi and 3 gu clips
    for i in range(3):
        w_hi = tmp_path / f"hi_{i}.wav"
        sf.write(w_hi, 0.1 * np.sin(2 * np.pi * 300 * t), 16000)
        clips.append(Clip(f"hi_{i}", f"hi_{i}.wav", "hi", "own", "test", f"नमस्ते {i}", 1.0, sha256_file(w_hi), "own", True))

        w_gu = tmp_path / f"gu_{i}.wav"
        sf.write(w_gu, 0.1 * np.sin(2 * np.pi * 350 * t), 16000)
        clips.append(Clip(f"gu_{i}", f"gu_{i}.wav", "gu", "own", "test", f"નમસ્તે {i}", 1.0, sha256_file(w_gu), "own", True))

    man = tmp_path / "manifest.jsonl"
    write_manifest(clips, man)

    run_dir = tmp_path / "run_limit_per_lang"
    kw = dict(
        root=tmp_path,
        conditions=("clean",),
        conditions_dir=tmp_path / "cond",
        manifest_path=man,
        progress=lambda *_: None,
        limit_per_language=2,
    )

    class SimpleProvider(ASRProvider):
        name = "simple"
        model_id = "simple-1"
        def _transcribe(self, p, l): return "ok", {}

    run_benchmark(clips, SimpleProvider(), run_dir, **kw)

    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg["limit_per_language"] == 2
    assert cfg["n_clips"] == 4

    rows = [json.loads(line) for line in (run_dir / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    clip_ids = [r["clip_id"] for r in rows]
    # Selected first 2 per language in manifest order: hi_0, gu_0, hi_1, gu_1
    assert clip_ids == ["hi_0", "gu_0", "hi_1", "gu_1"]

    # Resume check refuses if limit_per_language changes
    with pytest.raises(ValueError, match="different limit_per_language"):
        run_benchmark(clips, SimpleProvider(), run_dir, **dict(kw, limit_per_language=1))


def test_low_coverage_flagged_not_excluded(dataset):
    tmp, man, _ = dataset
    clips = load_manifest(man)

    class TruncatingProvider(ASRProvider):
        name = "trunc"
        model_id = "trunc-1"

        def _transcribe(self, wav_path, language):
            cid = Path(wav_path).stem
            if cid == "a":
                # Half text, coverage = 0.5 -> flagged, still scored
                words = REFS[cid].split()[:2]
                return " ".join(words), {"num_segments": 1, "last_segment_end_s": 1.0, "coverage": 0.5}
            return REFS[cid], {"num_segments": 1, "last_segment_end_s": 2.0, "coverage": 1.0}

    run_dir = tmp / "cov_run"
    run_benchmark(
        clips,
        TruncatingProvider(),
        run_dir,
        root=tmp,
        conditions=("clean",),
        conditions_dir=tmp / "cond",
        manifest_path=man,
        progress=lambda *_: None,
    )
    paths = report_run(run_dir, clips, tmp / "results_cov")

    pc = pd.read_parquet(paths["per_clip"])
    assert "coverage" in pc.columns
    cov_a = pc.loc[pc["clip_id"] == "a", "coverage"].iloc[0]
    cov_b = pc.loc[pc["clip_id"] == "b", "coverage"].iloc[0]
    assert cov_a == pytest.approx(0.5)
    assert cov_b == pytest.approx(1.0)

    s = pd.read_parquet(paths["summary"])
    assert "n_low_coverage" in s.columns
    s_a = s[s["language"] == "hi-en"].iloc[0]
    s_b = s[s["language"] == "hi"].iloc[0]
    assert s_a["n_low_coverage"] == 1
    assert s_a["n_clips"] == 1
    assert s_a["wer_norm"] == pytest.approx(0.5)  # 2 of 4 words deleted -> not excluded from score
    assert s_b["n_low_coverage"] == 0

    report_text = paths["report"].read_text(encoding="utf-8")
    assert "| low coverage | early_stop | runaway | near_cap |" in report_text
    assert "## Flagged clips" in report_text
    assert "possibly_truncated" not in report_text
    assert "possibly_truncated" not in pc.columns
    flagged_section = report_text.split("## Flagged clips", 1)[1].split("\n## ", 1)[0]
    assert "`a` (clean):" in flagged_section
    assert "coverage=0.50" in flagged_section
    assert "ratio=" in flagged_section
    assert "`b`" not in flagged_section

    # Run with all coverage == 1.0 should have n_low_coverage == 0 and no Flagged clips section
    class FullProvider(ASRProvider):
        name = "full"
        model_id = "full-1"

        def _transcribe(self, wav_path, language):
            cid = Path(wav_path).stem
            return REFS[cid], {"num_segments": 1, "last_segment_end_s": 2.0, "coverage": 1.0}

    run_dir2 = tmp / "full_cov_run"
    run_benchmark(
        clips,
        FullProvider(),
        run_dir2,
        root=tmp,
        conditions=("clean",),
        conditions_dir=tmp / "cond",
        manifest_path=man,
        progress=lambda *_: None,
    )
    paths2 = report_run(run_dir2, clips, tmp / "results_full_cov")
    s2 = pd.read_parquet(paths2["summary"])
    assert (s2["n_low_coverage"] == 0).all()
    assert (s2["early_stop"] == 0).all()
    assert (s2["runaway"] == 0).all()
    assert (s2["near_cap"] == 0).all()
    report_text2 = paths2["report"].read_text(encoding="utf-8")
    assert "## Flagged clips" not in report_text2


def test_split_flags_and_settings_resume(dataset):
    tmp, man, _ = dataset
    clips = load_manifest(man)

    class SplitFlagsProvider(ASRProvider):
        name = "flags"
        model_id = "flags-1"

        def __init__(self, vad_filter: bool = False):
            self._vad_filter = vad_filter

        @property
        def settings(self):
            return {"vad_filter": self._vad_filter, "chunk_length": None, "max_new_tokens": None}

        def _transcribe(self, wav_path, language):
            cid = Path(wav_path).stem
            if cid == "a":
                # ratio ~ 0.5 -> early_stop
                half = REFS[cid][: len(REFS[cid]) // 2]
                return half, {
                    "num_segments": 1, "last_segment_end_s": 1.0, "coverage": 0.5,
                    "total_tokens": 10, "max_segment_tokens": 10,
                }
            if cid == "b":
                # ratio ~ 2.4 -> runaway
                ref_b = REFS[cid]
                long_hyp = f"{ref_b} {ref_b} {'अतिरिक्त ' * 2}"[: int(len(ref_b) * 2.45)]
                return long_hyp, {
                    "num_segments": 1, "last_segment_end_s": 2.0, "coverage": 1.0,
                    "total_tokens": 180, "max_segment_tokens": 180,
                }
            # c: ratio 1.0 with 224 tokens -> near_token_cap only (informational, not flagged)
            return REFS[cid], {
                "num_segments": 1, "last_segment_end_s": 2.0, "coverage": 1.0,
                "total_tokens": 224, "max_segment_tokens": 224,
            }

    run_dir = tmp / "flags_run"
    kw = dict(root=tmp, conditions=("clean",), conditions_dir=tmp / "cond",
              manifest_path=man, progress=lambda *_: None)
    run_benchmark(clips, SplitFlagsProvider(vad_filter=False), run_dir, **kw)

    # Resume refuses if decoding settings change
    with pytest.raises(ValueError, match="different provider_settings"):
        run_benchmark(clips, SplitFlagsProvider(vad_filter=True), run_dir, **kw)

    paths = report_run(run_dir, clips, tmp / "results_flags")
    pc = pd.read_parquet(paths["per_clip"]).set_index("clip_id")
    assert "possibly_truncated" not in pc.columns
    for col in ("early_stop", "runaway", "near_token_cap"):
        assert col in pc.columns and pc[col].dtype == bool

    assert pc.loc["a", "len_ratio_norm"] < 0.8
    assert bool(pc.loc["a", "early_stop"]) is True
    assert bool(pc.loc["a", "runaway"]) is False
    assert bool(pc.loc["a", "near_token_cap"]) is False

    assert pc.loc["b", "len_ratio_norm"] > 1.5
    assert bool(pc.loc["b", "early_stop"]) is False
    assert bool(pc.loc["b", "runaway"]) is True
    assert bool(pc.loc["b", "near_token_cap"]) is False

    assert pc.loc["c", "len_ratio_norm"] == pytest.approx(1.0)
    assert bool(pc.loc["c", "early_stop"]) is False
    assert bool(pc.loc["c", "runaway"]) is False
    assert bool(pc.loc["c", "near_token_cap"]) is True

    s = pd.read_parquet(paths["summary"]).set_index("language")
    assert int(s.loc["hi-en", "early_stop"]) == 1 and int(s.loc["hi-en", "runaway"]) == 0
    assert int(s.loc["hi", "runaway"]) == 1 and int(s.loc["hi", "early_stop"]) == 0
    assert int(s.loc["gu", "near_cap"]) == 1 and int(s.loc["gu", "early_stop"]) == 0 and int(s.loc["gu", "runaway"]) == 0
    assert "n_truncated" not in s.columns

    report_text = paths["report"].read_text(encoding="utf-8")
    assert "possibly_truncated" not in report_text
    assert "## Flagged clips" in report_text
    assert "### Early stops" in report_text and "### Runaway outputs" in report_text
    flagged_sec = report_text.split("## Flagged clips", 1)[1].split("\n## ", 1)[0]
    early_part, runaway_part = flagged_sec.split("### Runaway outputs", 1)
    assert "`a` (clean):" in early_part and "tokens=10" in early_part and "coverage=0.50" in early_part
    assert "`b`" not in early_part
    assert "`b` (clean):" in runaway_part and "tokens=180" in runaway_part and "coverage=1.00" in runaway_part
    assert "`a`" not in runaway_part
    # near_cap clip with ratio 1.0 must NOT be listed under Flagged clips
    assert "`c`" not in flagged_sec


def test_script_match_and_folded_wer_gu_and_hi(dataset):
    tmp, man, noise = dataset
    clips = load_manifest(man)

    class ScriptProvider(ASRProvider):
        name = "script"
        model_id = "script-1"

        def _transcribe(self, wav_path, language):
            p = Path(wav_path)
            cid, cond = p.stem, p.parent.name
            if cid == "c":
                if cond == "clean":
                    # Exact Devanagari transliteration of gu reference "આજે 5 વાગ્યે આવજો"
                    return "आजे 5 वाग्ये आवजो", {}
                # Phone: Latin hypothesis -> not folded, remains error
                return "aaje 5 vagye aavjo", {}
            return REFS[cid], {}

    run_dir = tmp / "script_run"
    run_benchmark(
        clips,
        ScriptProvider(),
        run_dir,
        root=tmp,
        conditions=("clean", "phone"),
        noise_wav=noise,
        conditions_dir=tmp / "cond",
        manifest_path=man,
        progress=lambda *_: None,
    )
    paths = report_run(run_dir, clips, tmp / "results_script")

    pc = pd.read_parquet(paths["per_clip"])
    gu_clean = pc[(pc["clip_id"] == "c") & (pc["condition"] == "clean")].iloc[0]
    gu_phone = pc[(pc["clip_id"] == "c") & (pc["condition"] == "phone")].iloc[0]
    hi_clean = pc[(pc["clip_id"] == "b") & (pc["condition"] == "clean")].iloc[0]
    hien_clean = pc[(pc["clip_id"] == "a") & (pc["condition"] == "clean")].iloc[0]

    assert gu_clean["script_match"] == 0.0
    assert gu_clean["dominant_script"] == "Devanagari"
    assert int(gu_clean["word_err_norm"]) == 3   # primary stays script-sensitive
    assert int(gu_clean["word_err_folded"]) == 0
    assert int(gu_clean["char_err_folded"]) == 0

    assert gu_phone["script_match"] == 0.0
    assert gu_phone["dominant_script"] == "Latin"
    assert int(gu_phone["word_err_folded"]) == 3  # Latin untouched, still 3 errors

    assert hi_clean["script_match"] == 1.0
    assert hi_clean["dominant_script"] == "Devanagari"
    assert pd.isna(hi_clean["word_err_folded"])

    assert pd.isna(hien_clean["script_match"])

    s = pd.read_parquet(paths["summary"])
    s_gu_clean = s[(s["language"] == "gu") & (s["condition"] == "clean")].iloc[0]
    s_gu_phone = s[(s["language"] == "gu") & (s["condition"] == "phone")].iloc[0]
    s_hi_clean = s[(s["language"] == "hi") & (s["condition"] == "clean")].iloc[0]
    s_hien_clean = s[(s["language"] == "hi-en") & (s["condition"] == "clean")].iloc[0]

    assert s_gu_clean["wer_norm"] == pytest.approx(0.75)
    assert s_gu_clean["wer_folded"] == pytest.approx(0.0)
    assert s_gu_clean["cer_folded"] == pytest.approx(0.0)
    assert s_gu_clean["script_match_mean"] == pytest.approx(0.0)

    assert s_gu_phone["wer_folded"] == pytest.approx(0.75)  # Latin untouched
    assert s_hi_clean["script_match_mean"] == pytest.approx(1.0)
    assert pd.isna(s_hi_clean["wer_folded"])
    assert pd.isna(s_hien_clean["script_match_mean"])

    report_text = paths["report"].read_text(encoding="utf-8")
    assert "wer_folded" in report_text and "cer_folded" in report_text
    assert "approximate: ignores script, not spelling conventions" in report_text
    header_line = [ln for ln in report_text.splitlines() if ln.startswith("| language | condition |")][0]
    cols = [c.strip() for c in header_line.split("|")[1:-1]]
    idx = {name: i for i, name in enumerate(cols)}
    # Check hi row leaves folded cells empty
    hi_lines = [ln for ln in report_text.splitlines() if ln.startswith("| hi | clean |")]
    assert len(hi_lines) == 1
    cells = [c.strip() for c in hi_lines[0].split("|")[1:-1]]
    assert cells[idx["wer_folded"]] == "" and cells[idx["cer_folded"]] == ""
    hien_lines = [ln for ln in report_text.splitlines() if ln.startswith("| hi-en | clean |")]
    assert len(hien_lines) == 1
    hien_cells = [c.strip() for c in hien_lines[0].split("|")[1:-1]]
    assert hien_cells[idx["wer_folded"]] == "" and hien_cells[idx["cer_folded"]] == "" and hien_cells[idx["script match"]] == "n/a"


def test_sensitivity_columns_and_paired_exclusion(tmp_path):
    sr = 16000
    t = np.arange(sr) / sr
    refs = {
        "k1": "एक दो तीन चार",
        "k2": "पाँच छह सात आठ",
        "k3": "नौ दस ग्यारह बारह",
    }
    clips = []
    for cid, ref in refs.items():
        wav = tmp_path / "audio" / f"{cid}.wav"
        wav.parent.mkdir(exist_ok=True)
        sf.write(wav, 0.1 * np.sin(2 * np.pi * 250 * t), sr)
        clips.append(Clip(cid, f"audio/{cid}.wav", "hi", "own", "test", ref, 1.0,
                          sha256_file(wav), "own", True))
    man = tmp_path / "m.jsonl"
    write_manifest(clips, man)

    class SensProvider(ASRProvider):
        name = "sens"
        model_id = "sens-1"

        def _transcribe(self, wav_path, language):
            p = Path(wav_path)
            cid, cond = p.stem, p.parent.name
            if cid == "k1":
                if cond == "clean":
                    return refs[cid], {"max_segment_tokens": 20}  # 0/4 errors
                # 1 substitution in phone -> 0.25 WER, normal ratio (~1.0)
                return "गलत दो तीन चार", {"max_segment_tokens": 20}
            if cid == "k2":
                if cond == "clean":
                    # clean is completely normal, yet flagged in phone -> still excluded from clean in wer_excl_flagged
                    return refs[cid], {"max_segment_tokens": 20}
                # runaway (>1.5 ratio) in phone with massive insertions (4 extra words -> 4 errors / 4 ref words = 100% WER)
                return f"{refs[cid]} अतिरिक्त अतिरिक्त अतिरिक्त अतिरिक्त", {"max_segment_tokens": 40}
            # k3: perfect 0/4 in both clean & phone, near_token_cap=224 -> must NOT be excluded
            return refs[cid], {"max_segment_tokens": 224}

    run_dir = tmp_path / "sens_run"
    run_benchmark(
        clips, SensProvider(), run_dir, root=tmp_path, conditions=("clean", "phone"),
        conditions_dir=tmp_path / "cond", manifest_path=man, progress=lambda *_: None,
    )
    paths = report_run(run_dir, clips, tmp_path / "results_sens")
    s = pd.read_parquet(paths["summary"]).set_index("condition")

    # Primary WER stays unchanged over ALL 3 clips (clean=0/12=0.0, phone=(1+4+0)/12=5/12)
    assert int(s.loc["clean", "n_clips"]) == 3
    assert s.loc["clean", "wer_norm"] == pytest.approx(0.0)
    assert int(s.loc["phone", "n_clips"]) == 3
    assert s.loc["phone", "wer_norm"] == pytest.approx(5 / 12)

    # Sensitivity: wer_median on phone = median([0.25, 1.00, 0.00]) = 0.25
    assert s.loc["clean", "wer_median"] == pytest.approx(0.0)
    assert s.loc["phone", "wer_median"] == pytest.approx(0.25)

    # Sensitivity: k2 removed identically from BOTH clean & phone; k3 (near_cap only) stays kept -> n_clips_excl == 2
    assert int(s.loc["clean", "n_clips_excl"]) == 2
    assert int(s.loc["phone", "n_clips_excl"]) == 2
    assert s.loc["clean", "wer_excl_flagged"] == pytest.approx(0.0)
    assert s.loc["phone", "wer_excl_flagged"] == pytest.approx(1 / 8)

    report_text = paths["report"].read_text(encoding="utf-8")
    # Primary WER comes before sensitivity columns in table header
    hdr = [ln for ln in report_text.splitlines() if ln.startswith("| language | condition |")][0]
    assert hdr.index("| WER |") < hdr.index("wer_median (sensitivity)") < hdr.index("wer_excl_flagged (sensitivity)")
    # Paired comparison has main hi table (n=3, +41.7%) and sensitivity exclusion table (n=2, +12.5%) below it
    paired = report_text.split("## Paired comparison vs clean", 1)[1]
    assert "### hi" in paired
    assert "#### hi (sensitivity: excluding early_stop/runaway clips)" in paired
    main_sec, sens_sec = paired.split("#### hi (sensitivity: excluding early_stop/runaway clips)", 1)
    assert "| phone | 0.0% | 41.7% | +41.7% |" in main_sec and "| 3 |" in main_sec
    assert "| phone | 0.0% | 12.5% | +12.5% |" in sens_sec and "| 2 |" in sens_sec


def test_hi_lenient_metrics_in_report(tmp_path):
    t = np.arange(16000) / 16000
    clips = []
    # hi clip with nukta, precomposed U+095E, chandrabindu, and non-lenient vowel diff "पुलिस"
    hi_ref = "काफ़ी \u0915\u093e\u095e\u0940 हँ पुलिस"
    gu_ref = "આજે દિવસ છે"
    for cid, lang, ref in [("h1", "hi", hi_ref), ("g1", "gu", gu_ref)]:
        wav = tmp_path / "audio" / f"{cid}.wav"
        wav.parent.mkdir(exist_ok=True)
        sf.write(wav, 0.1 * np.sin(2 * np.pi * 240 * t), 16000)
        clips.append(Clip(cid, f"audio/{cid}.wav", lang, "own", "test", ref, 1.0,
                          sha256_file(wav), "own", True))
    man = tmp_path / "m.jsonl"
    write_manifest(clips, man)

    class LenientProvider(ASRProvider):
        name = "lenient"
        model_id = "lenient-1"

        def _transcribe(self, wav_path, language):
            cid = Path(wav_path).stem
            if cid == "h1":
                # Only पुलिस->पोलिस remains an error under lenient; primary sees all 4 words differ
                return "काफी काफी हं पोलिस", {}
            return gu_ref, {}

    run_dir = tmp_path / "lenient_run"
    run_benchmark(
        clips, LenientProvider(), run_dir, root=tmp_path, conditions=("clean",),
        conditions_dir=tmp_path / "cond", manifest_path=man, progress=lambda *_: None,
    )
    paths = report_run(run_dir, clips, tmp_path / "results_lenient")

    pc = pd.read_parquet(paths["per_clip"]).set_index("clip_id")
    assert int(pc.loc["h1", "word_err_norm"]) == 4
    assert int(pc.loc["h1", "word_err_lenient"]) == 1
    assert pc.loc["h1", "wer_lenient"] == pytest.approx(0.25)
    assert pc.loc["h1", "cer_lenient"] < pc.loc["h1", "char_err_norm"] / pc.loc["h1", "char_len_norm"]
    assert pd.isna(pc.loc["g1", "word_err_lenient"])
    assert pd.isna(pc.loc["g1", "wer_lenient"])
    assert pd.isna(pc.loc["g1", "cer_lenient"])

    s = pd.read_parquet(paths["summary"]).set_index("language")
    assert s.loc["hi", "wer_norm"] == pytest.approx(1.0)
    assert s.loc["hi", "wer_lenient"] == pytest.approx(0.25)
    assert pd.isna(s.loc["gu", "wer_lenient"])
    assert pd.isna(s.loc["gu", "cer_lenient"])

    report_text = paths["report"].read_text(encoding="utf-8")
    assert "secondary: ignores nukta and chandrabindu/anusvara spelling" in report_text
    hdr = [ln for ln in report_text.splitlines() if ln.startswith("| language | condition |")][0]
    cols = [c.strip() for c in hdr.split("|")[1:-1]]
    idx = {name: i for i, name in enumerate(cols)}
    assert idx["WER"] < idx["wer_lenient"] < idx["cer_lenient"]

    hi_cells = [c.strip() for c in [ln for ln in report_text.splitlines() if ln.startswith("| hi | clean |")][0].split("|")[1:-1]]
    assert hi_cells[idx["WER"]] == "100.0%"
    assert hi_cells[idx["wer_lenient"]] == "25.0%"
    gu_cells = [c.strip() for c in [ln for ln in report_text.splitlines() if ln.startswith("| gu | clean |")][0].split("|")[1:-1]]
    assert gu_cells[idx["wer_lenient"]] == "" and gu_cells[idx["cer_lenient"]] == ""
