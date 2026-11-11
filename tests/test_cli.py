import json
import pytest
from bharatspeechbench.cli import main
from bharatspeechbench.benchmark.manifest import write_manifest, Clip
from bharatspeechbench.utils.hashing import sha256_file


def test_hash_command(tmp_path, capsys):
    f = tmp_path / "a.wav"
    f.write_bytes(b"xyz")
    assert main(["hash", str(f)]) == 0
    assert capsys.readouterr().out.split()[0] == sha256_file(f)


def test_check_manifest_ok_and_verify(tmp_path, capsys):
    a = tmp_path / "a.wav"
    a.write_bytes(b"xyz")
    clip = Clip(id="1", path="a.wav", language="gu", source="fleurs", split="test",
                reference="x", duration_s=1.0, sha256=sha256_file(a), licence="CC BY 4.0")
    m = tmp_path / "m.jsonl"
    write_manifest([clip], m)
    assert main(["check-manifest", str(m), "--verify-audio", "--root", str(tmp_path)]) == 0
    assert "1 clips" in capsys.readouterr().out
    a.write_bytes(b"changed")
    assert main(["check-manifest", str(m), "--verify-audio", "--root", str(tmp_path)]) == 1


def test_check_manifest_invalid_returns_1(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text('{"id": "x"}\n')
    assert main(["check-manifest", str(m)]) == 1


def test_env_command_outputs_json(capsys):
    assert main(["env", "--seed", "3"]) == 0
    assert json.loads(capsys.readouterr().out)["seed"] == 3


def test_check_manifest_speaker_count_and_overlap_warning(tmp_path, capsys):
    m = tmp_path / "m.jsonl"
    c1 = Clip(id="1", path="a.wav", language="hi", source="fleurs", split="dev",
              reference="x", duration_s=1.0, sha256="0" * 64, licence="CC-BY-4.0", speaker_id="speaker_X")
    c2 = Clip(id="2", path="b.wav", language="hi", source="fleurs", split="test",
              reference="y", duration_s=1.0, sha256="0" * 64, licence="CC-BY-4.0", speaker_id="speaker_X")
    write_manifest([c1, c2], m)
    rc = main(["check-manifest", str(m)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "speakers: 1 distinct" in captured.out
    assert "WARNING: speaker_id in both dev and test: speaker_X" in captured.out


def test_no_clips_match_split_error_message(tmp_path, capsys):
    from bharatspeechbench.benchmark.runner import run_benchmark
    from bharatspeechbench.providers.base import ASRProvider

    class DummyProvider(ASRProvider):
        name = "dummy"
        model_id = "dummy-1"
        def _transcribe(self, p, l): return "ok", {}

    m = tmp_path / "m.jsonl"
    c1 = Clip(id="1", path="a.wav", language="hi", source="fleurs", split="dev",
              reference="x", duration_s=1.0, sha256="0" * 64, licence="CC-BY-4.0")
    write_manifest([c1], m)

    # runner ValueError
    with pytest.raises(ValueError) as exc:
        run_benchmark([c1], DummyProvider(), tmp_path / "out", split="test")
    err_msg = str(exc.value)
    assert "no clips match split 'test'" in err_msg
    assert "dev: 1" in err_msg
    assert "--split dev" in err_msg

    # cli simulate error message
    rc = main(["simulate", str(m), "--split", "test"])
    assert rc == 1
    err_out = capsys.readouterr().err
    assert "no clips match split 'test'" in err_out
    assert "--split dev" in err_out


def test_show_transcripts_script(tmp_path, capsys):
    from io import StringIO
    from scripts.show_transcripts import show_transcripts

    # Create manifest
    clips = [
        Clip(id="c1", path="c1.wav", language="hi", source="fleurs", split="test", reference="नमस्ते दुनिया यहाँ है", duration_s=2.0, sha256="0" * 64, licence="cc-by-4.0"),
        Clip(id="c2", path="c2.wav", language="gu", source="fleurs", split="test", reference="નમસ્તે દુનિયા અહીં છે", duration_s=2.0, sha256="0" * 64, licence="cc-by-4.0"),
        Clip(id="c3", path="c3.wav", language="hi", source="fleurs", split="test", reference="लंबा वाक्य जो बहुत बड़ा है", duration_s=2.0, sha256="0" * 64, licence="cc-by-4.0"),
        Clip(id="c4", path="c4.wav", language="hi", source="fleurs", split="test", reference="सामान्य वाक्य", duration_s=2.0, sha256="0" * 64, licence="cc-by-4.0"),
        Clip(id="c5", path="c5.wav", language="hi", source="fleurs", split="test", reference="सामान्य वाक्य है", duration_s=2.0, sha256="0" * 64, licence="cc-by-4.0"),
        Clip(id="c6", path="c6.wav", language="hi", source="fleurs", split="test", reference="छोटा वाक्य", duration_s=2.0, sha256="0" * 64, licence="cc-by-4.0"),
    ]
    man = tmp_path / "manifest.jsonl"
    write_manifest(clips, man)

    run_dir = tmp_path / "raw_run"
    run_dir.mkdir()
    transcripts = [
        {"clip_id": "c1", "condition": "clean", "language": "hi", "text": "नमस्ते", "meta": {"coverage": 0.5, "total_tokens": 12, "max_segment_tokens": 12}, "error": None},  # early_stop + low coverage
        {"clip_id": "c2", "condition": "phone", "language": "gu", "text": "namaste world here", "meta": {"coverage": 1.0, "total_tokens": 15, "max_segment_tokens": 15}, "error": None},  # latin for indic
        {"clip_id": "c3", "condition": "clean", "language": "hi", "text": "", "error": None},  # empty
        {"clip_id": "c4", "condition": "clean", "language": "hi", "text": None, "error": "RuntimeError: boom"},  # error
        {"clip_id": "c5", "condition": "clean", "language": "hi", "text": "सामान्य वाक्य है", "meta": {"coverage": 1.0, "total_tokens": 224, "max_segment_tokens": 224}, "error": None},  # near_token_cap only (INFO)
        {"clip_id": "c6", "condition": "clean", "language": "hi", "text": "छोटा वाक्य छोटा वाक्य छोटा वाक्य", "meta": {"coverage": 1.0, "total_tokens": 40, "max_segment_tokens": 40}, "error": None},  # runaway
    ]
    with (run_dir / "transcripts.jsonl").open("w", encoding="utf-8") as f:
        for r in transcripts:
            f.write(json.dumps(r) + "\n")

    buf = StringIO()
    rc = show_transcripts(run_dir, man, out_file=buf)
    assert rc == 0
    out = buf.getvalue()

    assert "[c1] condition=clean" in out
    assert "COVERAGE: 0.50" in out
    assert "SCRIPT: dominant=Devanagari, match=1.00" in out
    assert "WARNING: low coverage (0.50 < 0.85)" in out
    assert "WARNING: hypothesis length is less than 50% of reference length" in out
    assert "WARNING: early_stop (norm len ratio" in out
    assert "[c2] condition=phone" in out
    assert "COVERAGE: 1.00" in out
    assert "SCRIPT: dominant=Latin, match=0.00" in out
    assert "WARNING: low script match (0.00 < 0.50)" in out
    assert "WARNING: hypothesis contains mostly Latin letters for an Indic reference" in out
    assert "[c3] condition=clean" in out
    assert "COVERAGE: n/a" in out
    assert "WARNING: hypothesis is empty" in out
    assert "[c4] condition=clean" in out
    assert "ERROR: RuntimeError: boom" in out
    assert "[c5] condition=clean" in out
    assert "TOKENS: total=224, max_segment=224" in out
    assert "INFO: near_token_cap (max_segment_tokens=224 >= 210)" in out
    assert "WARNING: near_token_cap" not in out
    assert "[c6] condition=clean" in out
    assert "WARNING: runaway (norm len ratio" in out
    assert "Displayed 6 transcripts." in out

    # Test filtering by condition
    buf_cond = StringIO()
    show_transcripts(run_dir, man, condition="phone", out_file=buf_cond)
    out_cond = buf_cond.getvalue()
    assert "[c2] condition=phone" in out_cond
    assert "low coverage" not in out_cond
    assert "[c1] condition=clean" not in out_cond
    assert "Displayed 1 transcripts." in out_cond


def test_cli_run_decoding_options_mocked(tmp_path, monkeypatch):
    import shutil
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import MagicMock
    import numpy as np
    import soundfile as sf

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    wav = tmp_path / "hi.wav"
    sf.write(wav, 0.1 * np.sin(2 * np.pi * 300 * np.arange(16000) / 16000), 16000)
    man = tmp_path / "m.jsonl"
    write_manifest([
        Clip(id="h1", path="hi.wav", language="hi", source="own", split="dev",
             reference="नमस्ते", duration_s=1.0, sha256=sha256_file(wav),
             licence="own", speaker_consent=True)
    ], man)

    fake_segs = [SimpleNamespace(start=0.0, end=1.0, text="नमस्ते", tokens=[1, 2, 3])]
    fake_info = SimpleNamespace(language_probability=0.99, duration=1.0)
    fake_model = MagicMock()
    fake_model.transcribe.side_effect = lambda *a, **kw: (iter(fake_segs), fake_info)
    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.chdir(tmp_path)

    rc = main([
        "run", str(man),
        "--root", str(tmp_path),
        "--split", "dev",
        "--conditions", "clean",
        "--device", "cpu",
        "--compute-type", "int8",
        "--vad-filter",
        "--chunk-length", "15",
        "--max-new-tokens", "200",
        "--initial-prompt", "આ એક ગુજરાતી વાક્ય છે.",
        "--run-id", "opt-test",
    ])
    assert rc == 0

    cfg = json.loads((tmp_path / "results" / "raw" / "opt-test" / "config.json").read_text(encoding="utf-8"))
    ps = cfg["provider_settings"]
    assert ps["vad_filter"] is True
    assert ps["chunk_length"] == 15
    assert ps["max_new_tokens"] == 200
    assert ps["initial_prompt"] == "આ એક ગુજરાતી વાક્ય છે."
    _, kw = fake_model.transcribe.call_args
    assert kw["vad_filter"] is True
    assert kw["chunk_length"] == 15
    assert kw["max_new_tokens"] == 200
    assert kw["initial_prompt"] == "આ એક ગુજરાતી વાક્ય છે."


def test_error_audit_script(tmp_path):
    from io import StringIO
    from scripts.error_audit import run_error_audit

    clips = [
        Clip(id="h1", path="h1.wav", language="hi", source="fleurs", split="test",
             reference="TMZ ने 12 फोटो लीं।", duration_s=5.0, sha256="0" * 64, licence="cc-by-4.0"),
        Clip(id="h2", path="h2.wav", language="hi", source="fleurs", split="test",
             reference="एक दो तीन चार पाँच छह।", duration_s=6.0, sha256="0" * 64, licence="cc-by-4.0"),
        Clip(id="h3", path="h3.wav", language="hi", source="fleurs", split="test",
             reference="हँ काफ़ी है", duration_s=4.0, sha256="0" * 64, licence="cc-by-4.0"),
        Clip(id="g1", path="g1.wav", language="gu", source="fleurs", split="test",
             reference="આજે 5 વાગ્યે આવજો", duration_s=4.0, sha256="0" * 64, licence="cc-by-4.0"),
    ]
    man = tmp_path / "manifest.jsonl"
    write_manifest(clips, man)

    run_dir = tmp_path / "audit_run"
    run_dir.mkdir()
    rows = [
        # Old errored attempt (retried below -> deduped out)
        {"clip_id": "h1", "condition": "clean", "language": "hi", "text": None, "error": "RuntimeError: retry"},
        # Retried h1 clean: "tmz" -> "टीमजी" (latin->indic), "12" -> "बारह" (digit->words), "लीं" -> "ली" (other: anusvara not folded)
        {"clip_id": "h1", "condition": "clean", "language": "hi", "text": "टीमजी ने बारह फोटो ली",
         "meta": {"coverage": 0.95}, "error": None},
        # h2 phone: 4 deletions ("एक दो", deleted 4 words)
        {"clip_id": "h2", "condition": "phone", "language": "hi", "text": "एक दो",
         "meta": {"coverage": 0.40}, "error": None},
        # h3 phone: हँ->हं and काफ़ी->काफी (2 orthographic subs under lenient folding)
        {"clip_id": "h3", "condition": "phone", "language": "hi", "text": "हं काफी है",
         "meta": {"coverage": 1.0}, "error": None},
        # g1 clean: "5" -> "पाँच" (digit->words) + Indic subs
        {"clip_id": "g1", "condition": "clean", "language": "gu", "text": "आजे पाँच वाग्ये आवजो",
         "meta": {"coverage": 1.0}, "error": None},
    ]
    tpath = run_dir / "transcripts.jsonl"
    tpath.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    before_bytes = tpath.read_bytes()
    before_files = sorted(p.name for p in run_dir.iterdir())

    # Full run audit: 9 total subs (1 latin, 2 digit, 2 orthographic, 4 other)
    buf = StringIO()
    rc = run_error_audit(run_dir, man, out_file=buf)
    assert rc == 0
    out = buf.getvalue()

    # Read-only guarantee
    assert tpath.read_bytes() == before_bytes
    assert sorted(p.name for p in run_dir.iterdir()) == before_files

    assert "(a) Top 25 substitution pairs" in out
    assert "tmz -> टीमजी" in out
    assert "12 -> बारह" in out
    assert "5 -> पाँच" in out
    assert "(b) Substitution types:" in out
    assert "Latin -> Devanagari/Gujarati: 1 (11.1%)" in out
    assert "Digit sequence -> words: 2 (22.2%)" in out
    assert "orthographic (equal after lenient folding): 2 (22.2%)" in out
    assert "Other: 4 (44.4%)" in out
    assert out.index("orthographic (equal after lenient folding)") < out.index("Other:")
    assert "(c) Top 10 clips with the most deletions:" in out
    assert "h2 (phone, hi): deletions=4, ratio=" in out
    assert "coverage=0.40" in out

    # Filtered by --language hi --condition clean: 3 total subs (1, 1, 0, 1)
    buf_f = StringIO()
    rc_f = run_error_audit(run_dir, man, language="hi", condition="clean", out_file=buf_f)
    assert rc_f == 0
    out_f = buf_f.getvalue()
    assert "(1 clips)" in out_f
    assert "Latin -> Devanagari/Gujarati: 1 (33.3%)" in out_f
    assert "Digit sequence -> words: 1 (33.3%)" in out_f
    assert "orthographic (equal after lenient folding): 0 (0.0%)" in out_f
    assert "Other: 1 (33.3%)" in out_f
    assert "h2" not in out_f and "h3" not in out_f and "g1" not in out_f


def test_gu_flag_ratio_uses_word_count(tmp_path):
    from io import StringIO
    import pandas as pd
    from bharatspeechbench.benchmark.reporting import report_run
    from scripts.show_transcripts import show_transcripts

    clips = [
        # gu normal word ratio (4/4 = 1.0), even though Latin char ratio < 0.8 (23/44 = 0.52)
        Clip(id="g_ok", path="g_ok.wav", language="gu", source="fleurs", split="test",
             reference="મહત્વપૂર્ણ વિશ્વવિદ્યાલય પરિસ્થિતિની જાણકારી", duration_s=4.0,
             sha256="0" * 64, licence="cc-by-4.0"),
        # gu early_stop by word ratio (3/5 = 0.60 < 0.8), even though char ratio >= 0.8
        Clip(id="g_early", path="g_early.wav", language="gu", source="fleurs", split="test",
             reference="આજે અહીં સુંદર હવામાન છે", duration_s=4.0,
             sha256="0" * 64, licence="cc-by-4.0"),
        # gu runaway by word ratio (4/2 = 2.0 > 1.5), even though char ratio <= 1.5
        Clip(id="g_run", path="g_run.wav", language="gu", source="fleurs", split="test",
             reference="ચાલ ભાઈ", duration_s=2.0,
             sha256="0" * 64, licence="cc-by-4.0"),
        # hi control: still uses character ratio (same 4 words, char ratio < 0.8 -> early_stop)
        Clip(id="h_ctrl", path="h_ctrl.wav", language="hi", source="fleurs", split="test",
             reference="महत्वपूर्ण विश्वविद्यालय परिस्थितिकी जानकारी", duration_s=4.0,
             sha256="0" * 64, licence="cc-by-4.0"),
    ]
    man = tmp_path / "m.jsonl"
    write_manifest(clips, man)

    run_dir = tmp_path / "raw" / "gu-ratio-test"
    run_dir.mkdir(parents=True)
    cfg = {
        "provider": "fake", "model": "fake-1", "provider_settings": {},
        "split": "test", "conditions": ["clean"],
        "environment": {"git_commit": "abc", "git_dirty": False, "manifest_sha256": "0" * 64, "seed": 0, "gpu": None},
    }
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    rows = [
        {"clip_id": "g_ok", "language": "gu", "condition": "clean", "provider": "fake", "model": "fake-1",
         "error": None, "audio_seconds": 4.0, "processing_seconds": 0.4, "rtf": 0.1,
         "text": "mahatva uni stithi info", "meta": {"coverage": 1.0, "max_segment_tokens": 20}},
        {"clip_id": "g_early", "language": "gu", "condition": "clean", "provider": "fake", "model": "fake-1",
         "error": None, "audio_seconds": 4.0, "processing_seconds": 0.4, "rtf": 0.1,
         "text": "aaje ahi sundarhavaman", "meta": {"coverage": 1.0, "max_segment_tokens": 20}},
        {"clip_id": "g_run", "language": "gu", "condition": "clean", "provider": "fake", "model": "fake-1",
         "error": None, "audio_seconds": 2.0, "processing_seconds": 0.2, "rtf": 0.1,
         "text": "a b c d", "meta": {"coverage": 1.0, "max_segment_tokens": 20}},
        {"clip_id": "h_ctrl", "language": "hi", "condition": "clean", "provider": "fake", "model": "fake-1",
         "error": None, "audio_seconds": 4.0, "processing_seconds": 0.4, "rtf": 0.1,
         "text": "mahatva uni stithi info", "meta": {"coverage": 1.0, "max_segment_tokens": 20}},
    ]
    (run_dir / "transcripts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    paths = report_run(run_dir, clips, tmp_path / "results")
    pc = pd.read_parquet(paths["per_clip"]).set_index("clip_id")

    assert pc.loc["g_ok", "len_ratio_norm"] == pytest.approx(1.0)
    assert bool(pc.loc["g_ok", "early_stop"]) is False
    assert bool(pc.loc["g_ok", "runaway"]) is False

    assert pc.loc["g_early", "len_ratio_norm"] == pytest.approx(0.6)
    assert bool(pc.loc["g_early", "early_stop"]) is True
    assert bool(pc.loc["g_early", "runaway"]) is False

    assert pc.loc["g_run", "len_ratio_norm"] == pytest.approx(2.0)
    assert bool(pc.loc["g_run", "early_stop"]) is False
    assert bool(pc.loc["g_run", "runaway"]) is True

    # Hindi keeps character ratio (< 0.8 even with 4 words vs 4 words)
    assert pc.loc["h_ctrl", "len_ratio_norm"] < 0.8
    assert bool(pc.loc["h_ctrl", "early_stop"]) is True

    report_text = paths["report"].read_text(encoding="utf-8")
    assert "`g_ok`" not in report_text.split("## Flagged clips", 1)[1].split("\n## ", 1)[0]
    assert "`g_early` (clean): ratio=0.60" in report_text
    assert "`g_run` (clean): ratio=2.00" in report_text

    # show_transcripts agrees
    buf = StringIO()
    assert show_transcripts(run_dir, man, out_file=buf) == 0
    st_out = buf.getvalue()
    g_ok_block = st_out.split("[g_ok]", 1)[1].split("-" * 20, 1)[0]
    assert "early_stop" not in g_ok_block and "runaway" not in g_ok_block
    g_early_block = st_out.split("[g_early]", 1)[1].split("-" * 20, 1)[0]
    assert "early_stop (norm len ratio 0.60 < 0.8)" in g_early_block
    g_run_block = st_out.split("[g_run]", 1)[1].split("-" * 20, 1)[0]
    assert "runaway (norm len ratio 2.00 > 1.5)" in g_run_block
