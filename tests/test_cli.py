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
