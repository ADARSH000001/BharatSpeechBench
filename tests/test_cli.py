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
