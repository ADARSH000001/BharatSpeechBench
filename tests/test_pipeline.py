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
    assert "| low coverage |" in report_text
    assert "## Possible truncation" in report_text
    trunc_section = report_text.split("## Possible truncation", 1)[1].split("\n## ", 1)[0]
    assert "`a` (clean): coverage 0.50" in trunc_section
    assert "hyp/ref length ratio" in trunc_section
    assert "`b`" not in trunc_section

    # Run with all coverage == 1.0 should have n_low_coverage == 0 and no Possible truncation section
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
    report_text2 = paths2["report"].read_text(encoding="utf-8")
    assert "## Possible truncation" not in report_text2
