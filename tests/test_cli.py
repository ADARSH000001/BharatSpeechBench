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
    ]
    man = tmp_path / "manifest.jsonl"
    write_manifest(clips, man)

    run_dir = tmp_path / "raw_run"
    run_dir.mkdir()
    transcripts = [
        {"clip_id": "c1", "condition": "clean", "language": "hi", "text": "नमस्ते", "meta": {"coverage": 0.5, "total_tokens": 12, "max_segment_tokens": 12}, "error": None},  # short (a) + low coverage
        {"clip_id": "c2", "condition": "phone", "language": "gu", "text": "namaste world here", "meta": {"coverage": 1.0, "total_tokens": 15, "max_segment_tokens": 15}, "error": None},  # latin for indic, full cov
        {"clip_id": "c3", "condition": "clean", "language": "hi", "text": "", "error": None},  # empty
        {"clip_id": "c4", "condition": "clean", "language": "hi", "text": None, "error": "RuntimeError: boom"},  # error
        {"clip_id": "c5", "condition": "clean", "language": "hi", "text": "सामान्य वाक्य है", "meta": {"coverage": 1.0, "total_tokens": 224, "max_segment_tokens": 224}, "error": None},  # token cap (b)
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
    assert "WARNING: possible truncation (norm len ratio" in out
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
    assert "WARNING: possible truncation (max_segment_tokens=224 >= 210)" in out
    assert "Displayed 5 transcripts." in out

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
