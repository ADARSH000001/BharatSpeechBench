import json
import pytest
from bharatspeechbench.benchmark.manifest import load_manifest, write_manifest, Clip


def _clip(**kw):
    base = dict(id="a1", path="x.wav", language="hi", source="own", split="test",
                reference="नमस्ते", duration_s=2.0, sha256="0" * 64, licence="own", speaker_consent=True)
    base.update(kw)
    return Clip(**base)


def test_roundtrip_keeps_unicode(tmp_path):
    p = tmp_path / "m.jsonl"
    write_manifest([_clip(), _clip(id="a2", language="gu", reference="નમસ્તે")], p)
    out = load_manifest(p)
    assert [c.reference for c in out] == ["नमस्ते", "નમસ્તે"]


def test_missing_field_rejected(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"id": "a1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_manifest(p)


def test_bad_split_rejected(tmp_path):
    p = tmp_path / "m.jsonl"
    write_manifest([_clip(split="train")], p)
    with pytest.raises(ValueError, match="split"):
        load_manifest(p)


def test_unknown_language_rejected(tmp_path):
    p = tmp_path / "m.jsonl"
    write_manifest([_clip(language="ta")], p)
    with pytest.raises(ValueError, match="language"):
        load_manifest(p)


def test_duplicate_id_rejected(tmp_path):
    p = tmp_path / "m.jsonl"
    write_manifest([_clip(), _clip()], p)
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(p)


def test_own_recording_needs_consent(tmp_path):
    p = tmp_path / "m.jsonl"
    write_manifest([_clip(speaker_consent=None)], p)
    with pytest.raises(ValueError, match="consent"):
        load_manifest(p)


def test_verify_audio_hashes(tmp_path):
    from bharatspeechbench.utils.hashing import sha256_file
    from bharatspeechbench.benchmark.manifest import verify_audio_hashes
    good = tmp_path / "good.wav"
    good.write_bytes(b"audio")
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"audio")
    clips = [
        _clip(id="ok", path="good.wav", sha256=sha256_file(good)),
        _clip(id="wrong", path="bad.wav", sha256="0" * 64),
        _clip(id="gone", path="nope.wav"),
    ]
    problems = verify_audio_hashes(clips, root=tmp_path)
    assert len(problems) == 2
    assert any("mismatch" in p for p in problems) and any("not found" in p for p in problems)


def test_empty_reference_rejected(tmp_path):
    import json
    import pytest
    from bharatspeechbench.benchmark.manifest import load_manifest

    row = {"id": "x", "path": "a.wav", "language": "hi", "source": "fleurs", "split": "test",
           "reference": "  ", "duration_s": 1.0, "sha256": "0" * 64, "licence": "CC-BY-4.0"}
    f = tmp_path / "m.jsonl"
    f.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="empty reference"):
        load_manifest(f)
