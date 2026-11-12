import json
import pytest
from bharatspeechbench.benchmark.manifest import load_manifest, write_manifest, Clip, verify_audio_hashes


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


def test_optional_speaker_id_and_check_speakers(tmp_path):
    from bharatspeechbench.benchmark.manifest import check_speakers

    # Old manifest without speaker_id still loads
    p = tmp_path / "old.jsonl"
    write_manifest([_clip(id="c1", split="dev"), _clip(id="c2", split="test")], p)
    clips = load_manifest(p)
    assert clips[0].speaker_id is None
    n_spk, overlap = check_speakers(clips)
    assert n_spk == 0 and len(overlap) == 0

    # Manifest with speaker_id
    p2 = tmp_path / "with_spk.jsonl"
    write_manifest([
        _clip(id="c1", split="dev", speaker_id="spk_A"),
        _clip(id="c2", split="dev", speaker_id="spk_B"),
        _clip(id="c3", split="test", speaker_id="spk_A"),  # overlap!
        _clip(id="c4", split="test", speaker_id="spk_C"),
    ], p2)
    clips2 = load_manifest(p2)
    n_spk2, overlap2 = check_speakers(clips2)
    assert n_spk2 == 3
    assert overlap2 == {"spk_A"}


def test_build_fleurs_manifest_offline(tmp_path):
    import numpy as np
    import soundfile as sf
    from scripts.build_fleurs_manifest import build_fleurs_manifest
    from bharatspeechbench.utils.hashing import sha256_file

    sr = 16000
    source_dirs = {}
    for lang in ("hi", "gu"):
        lang_dir = tmp_path / f"source_{lang}"
        lang_dir.mkdir()
        source_dirs[lang] = lang_dir

        tsv_lines = []
        for i in range(5):
            fname = f"clip_{i}.wav"
            # clip_0 is short (1.5s = 24000 samples); clips 1-4 are 3.5s (56000 samples)
            n_samples = 24000 if i == 0 else 56000
            t = np.arange(n_samples) / sr
            wav_data = 0.1 * np.sin(2 * np.pi * (200 + 50 * i) * t)
            sf.write(lang_dir / fname, wav_data, sr, subtype="PCM_16")

            raw_text = f"कच्चा पाठ {i}।" if lang == "hi" else f"કાચો પાઠ {i}."
            norm_text = f"कच्चा पाठ {i}" if lang == "hi" else f"કાચો પાઠ {i}"
            tsv_lines.append(f"{1000 + i}\t{fname}\t{raw_text}\t{norm_text}\t...\t{n_samples}\tMALE")

        (lang_dir / "test.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    man_path = tmp_path / "manifest.jsonl"
    out_audio = tmp_path / "downloads" / "fleurs"

    # Build manifest selecting 2 clips per language
    build_fleurs_manifest(
        languages=("hi", "gu"),
        split="test",
        n=2,
        out_dir=out_audio,
        manifest_path=man_path,
        seed=42,
        root=tmp_path,
        source_dirs=source_dirs,
    )

    # (a) Manifest lines are valid and load with load_manifest
    loaded = load_manifest(man_path)
    assert len(loaded) == 4
    assert {c.language for c in loaded} == {"hi", "gu"}
    assert all(c.source == "fleurs" for c in loaded)
    assert all(c.split == "test" for c in loaded)
    assert all(c.licence == "cc-by-4.0" for c in loaded)
    assert all(c.speaker_id is None for c in loaded)
    assert all(c.speaker_consent is None for c in loaded)
    assert all("।" in c.reference or "." in c.reference for c in loaded)  # raw text with punctuation

    # (b) sha256 and duration are real
    probs = verify_audio_hashes(loaded, root=tmp_path)
    assert probs == []
    for c in loaded:
        fpath = tmp_path / c.path
        assert fpath.is_file()
        assert c.sha256 == sha256_file(fpath)
        assert c.duration_s >= 3.0  # short clip_0 (1.5s) was excluded

    # (c) Re-running adds no duplicates
    build_fleurs_manifest(
        languages=("hi", "gu"),
        split="test",
        n=2,
        out_dir=out_audio,
        manifest_path=man_path,
        seed=42,
        root=tmp_path,
        source_dirs=source_dirs,
    )
    loaded_after = load_manifest(man_path)
    assert len(loaded_after) == 4
    assert [c.id for c in loaded_after] == [c.id for c in loaded]

    # (d) Deterministic selection
    man_path2 = tmp_path / "manifest2.jsonl"
    clips2 = build_fleurs_manifest(
        languages=("hi", "gu"),
        split="test",
        n=2,
        out_dir=out_audio,
        manifest_path=man_path2,
        seed=42,
        root=tmp_path,
        source_dirs=source_dirs,
    )
    assert [c.id for c in clips2] == [c.id for c in loaded]


def test_build_fleurs_manifest_dev_split_offline(tmp_path, monkeypatch):
    import tarfile
    import numpy as np
    import soundfile as sf
    import scripts.build_fleurs_manifest as bfm

    sr = 16000
    n_samples = 56000  # 3.5s >= 3.0s
    raw_dir = tmp_path / "remote"
    raw_dir.mkdir()

    tsv_files = {}
    tar_files = {}
    for lang, code in (("hi", "hi_in"), ("gu", "gu_in")):
        wav_path = raw_dir / f"{lang}_dev_0.wav"
        t = np.arange(n_samples) / sr
        sf.write(wav_path, 0.1 * np.sin(2 * np.pi * 250 * t), sr, subtype="PCM_16")

        raw_text = "देव स्प्लिट वाक्य।" if lang == "hi" else "દેવ સ્પ્લિટ વાક્ય."
        tsv_path = raw_dir / f"{code}_dev.tsv"
        tsv_path.write_text(f"2001\t{wav_path.name}\t{raw_text}\t{raw_text}\t...\t{n_samples}\tFEMALE\n", encoding="utf-8")
        tsv_files[code] = tsv_path

        tar_path = raw_dir / f"{code}_dev.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(wav_path, arcname=f"dev/{wav_path.name}")
        tar_files[code] = tar_path

    def fake_download(url: str):
        for code in ("hi_in", "gu_in"):
            if f"/data/{code}/dev.tsv" in url:
                return tsv_files[code].open("rb")
            if f"/data/{code}/audio/dev.tar.gz" in url:
                return tar_files[code].open("rb")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(bfm, "_download_stream", fake_download)

    rc = bfm.main([
        "--lang", "hi", "gu",
        "--split", "dev",
        "--n", "1",
        "--out", "downloads/fleurs",
        "--manifest", "manifests/dev_manifest.jsonl",
        "--root", str(tmp_path),
    ])
    assert rc == 0

    loaded = load_manifest(tmp_path / "manifests" / "dev_manifest.jsonl")
    assert len(loaded) == 2
    assert {c.language for c in loaded} == {"hi", "gu"}
    assert all(c.split == "dev" for c in loaded)
    assert all(c.source == "fleurs" for c in loaded)
    assert verify_audio_hashes(loaded, root=tmp_path) == []


def test_build_fleurs_manifest_n_exceeds_available_and_exclude_manifest(tmp_path, capsys, monkeypatch):
    import tarfile
    import numpy as np
    import soundfile as sf
    import scripts.build_fleurs_manifest as bfm

    sr = 16000
    raw_dir = tmp_path / "remote"
    raw_dir.mkdir()
    tsv_files = {}
    tar_files = {}
    for lang, code in (("hi", "hi_in"), ("gu", "gu_in")):
        tsv_lines = []
        tar_path = raw_dir / f"{code}_test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            # clip_0 short (1.5s), clips 1..4 eligible (3.5s) -> 4 eligible per language
            for i in range(5):
                n_samples = 24000 if i == 0 else 56000
                wav_path = raw_dir / f"{lang}_clip_{i}.wav"
                t = np.arange(n_samples) / sr
                sf.write(wav_path, 0.1 * np.sin(2 * np.pi * (200 + 30 * i) * t), sr, subtype="PCM_16")
                tar.add(wav_path, arcname=f"test/{wav_path.name}")
                raw_text = f"हिन्दी वाक्य {i}।" if lang == "hi" else f"ગુજરાતી વાક્ય {i}."
                tsv_lines.append(f"{3000 + i}\t{wav_path.name}\t{raw_text}\t{raw_text}\t...\t{n_samples}\tFEMALE")
        tsv_path = raw_dir / f"{code}_test.tsv"
        tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
        tsv_files[code] = tsv_path
        tar_files[code] = tar_path

    def fake_download(url: str):
        for code in ("hi_in", "gu_in"):
            if f"/data/{code}/test.tsv" in url:
                return tsv_files[code].open("rb")
            if f"/data/{code}/audio/test.tar.gz" in url:
                return tar_files[code].open("rb")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(bfm, "_download_stream", fake_download)

    # 1. Request n=20 (> 4 available): fails soft, picks all 4 per lang deterministically, prints counts
    rc1 = bfm.main([
        "--lang", "hi", "gu",
        "--split", "test",
        "--n", "20",
        "--seed", "7",
        "--out", "downloads/all",
        "--manifest", "manifests/all.jsonl",
        "--root", str(tmp_path),
    ])
    assert rc1 == 0
    out1 = capsys.readouterr().out
    assert "requested 20 clips" in out1 and "4 eligible exist" in out1
    all_clips = load_manifest(tmp_path / "manifests" / "all.jsonl")
    assert len(all_clips) == 8  # 4 hi + 4 gu (short clip_0 excluded)

    # Same run with different seed still yields exact same deterministic order when using all 4
    bfm.main([
        "--lang", "hi", "gu",
        "--split", "test",
        "--n", "20",
        "--seed", "99",
        "--out", "downloads/all2",
        "--manifest", "manifests/all2.jsonl",
        "--root", str(tmp_path),
    ])
    capsys.readouterr()
    assert [c.id for c in load_manifest(tmp_path / "manifests" / "all2.jsonl")] == [c.id for c in all_clips]

    # 2. Build a 2-clip-per-lang pilot manifest, then build expanded batch with --exclude-manifest
    assert bfm.main([
        "--lang", "hi", "gu",
        "--split", "test",
        "--n", "2",
        "--seed", "0",
        "--out", "downloads/pilot",
        "--manifest", "manifests/pilot.jsonl",
        "--root", str(tmp_path),
    ]) == 0
    capsys.readouterr()
    pilot_clips = load_manifest(tmp_path / "manifests" / "pilot.jsonl")
    pilot_ids = {c.id for c in pilot_clips}
    assert len(pilot_ids) == 4

    # Exclude pilot, request n=10 -> only 2 remaining eligible per language -> uses all remaining 2
    assert bfm.main([
        "--lang", "hi", "gu",
        "--split", "test",
        "--n", "10",
        "--seed", "0",
        "--out", "downloads/rest",
        "--manifest", "manifests/rest.jsonl",
        "--exclude-manifest", "manifests/pilot.jsonl",
        "--root", str(tmp_path),
    ]) == 0
    out_rest = capsys.readouterr().out
    assert "requested 10 clips, 2 available" in out_rest and "2 excluded" in out_rest
    rest_clips = load_manifest(tmp_path / "manifests" / "rest.jsonl")
    rest_ids = {c.id for c in rest_clips}
    assert len(rest_ids) == 4
    assert pilot_ids.isdisjoint(rest_ids)
    assert pilot_ids | rest_ids == {c.id for c in all_clips}
