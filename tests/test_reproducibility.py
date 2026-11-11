import json
import random
import numpy as np
from bharatspeechbench.utils.reproducibility import capture_environment, set_seed
from bharatspeechbench.utils.hashing import sha256_file


def test_environment_is_json_serialisable_and_complete(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text("{}\n")
    env = capture_environment(seed=7, manifest_path=m)
    json.dumps(env)                                   # must not raise
    assert env["seed"] == 7
    assert env["manifest_sha256"] == sha256_file(m)
    for key in ("python", "os", "cpu", "ram_bytes", "gpu", "ffmpeg", "packages", "git_commit", "git_dirty", "timestamp_utc"):
        assert key in env
    assert env["packages"]["numpy"] is not None


def test_set_seed_reproducible():
    set_seed(1)
    a = (random.random(), np.random.rand())
    set_seed(1)
    b = (random.random(), np.random.rand())
    assert a == b


def test_get_model_revision_from_cache(tmp_path):
    from bharatspeechbench.providers.whisper import (
        get_model_revision,
        get_model_revision_info,
        _resolve_model_repo,
    )

    repo = _resolve_model_repo("small")
    assert repo == "Systran/faster-whisper-small"

    # Missing cache directory returns None and reason string
    rev, reason = get_model_revision_info(repo, cache_dir=tmp_path / "nonexistent")
    assert rev is None
    assert "cache directory does not exist" in reason
    assert get_model_revision(repo, cache_dir=tmp_path / "nonexistent") is None

    # Missing model repo in cache returns None and reason string
    (tmp_path / "cache_empty").mkdir()
    rev, reason = get_model_revision_info(repo, cache_dir=tmp_path / "cache_empty")
    assert rev is None
    assert "model repo folder not found in cache" in reason

    # 1. Symlink / standard layout: refs/main pointing to hash, snapshots with symlinks/dir
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("abcdef1234567890abcdef1234567890abcdef12\n")
    rev, reason = get_model_revision_info(repo, cache_dir=tmp_path)
    assert rev == "abcdef1234567890abcdef1234567890abcdef12"
    assert reason is None

    # 2. Windows no-symlink layout: refs/main exists, snapshots has actual copy, trees has .json
    snaps_dir = repo_dir / "snapshots"
    (snaps_dir / "abcdef1234567890abcdef1234567890abcdef12").mkdir(parents=True)
    (snaps_dir / "abcdef1234567890abcdef1234567890abcdef12" / "model.bin").write_bytes(b"model")
    trees_dir = repo_dir / "trees"
    trees_dir.mkdir(parents=True)
    (trees_dir / "abcdef1234567890abcdef1234567890abcdef12.json").write_text("{}")
    rev, reason = get_model_revision_info(repo, cache_dir=tmp_path)
    assert rev == "abcdef1234567890abcdef1234567890abcdef12"
    assert reason is None

    # 3. Fallback to snapshots folder when refs is deleted
    (refs_dir / "main").unlink()
    rev, reason = get_model_revision_info(repo, cache_dir=tmp_path)
    assert rev == "abcdef1234567890abcdef1234567890abcdef12"
    assert reason is None

    # 4. Fallback to trees folder when snapshots is empty
    import shutil
    shutil.rmtree(snaps_dir)
    rev, reason = get_model_revision_info(repo, cache_dir=tmp_path)
    assert rev == "abcdef1234567890abcdef1234567890abcdef12"
    assert reason is None

    # 5. Unrecognised layout returns None and reason
    shutil.rmtree(trees_dir)
    rev, reason = get_model_revision_info(repo, cache_dir=tmp_path)
    assert rev is None
    assert "cache layout not recognised" in reason


def test_git_dirty_detection(monkeypatch, tmp_path):
    import subprocess
    from bharatspeechbench.utils.reproducibility import _git_dirty, capture_environment
    from bharatspeechbench.benchmark.reporting import markdown_report
    import pandas as pd

    # Clean repo mock (verifies --untracked-files=no is passed)
    seen_cmds = []

    def fake_clean(cmd, *args, **kwargs):
        seen_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_clean)
    assert _git_dirty() is False
    assert "--untracked-files=no" in seen_cmds[0]

    # Dirty repo mock
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=" M file.py\n", stderr=""))
    assert _git_dirty() is True

    # Git unavailable mock
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: not a git repo"))
    assert _git_dirty() is None
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("no git")))
    assert _git_dirty() is None

    # Markdown report header includes warning when dirty
    cfg_dirty = {
        "provider": "whisper",
        "model": "whisper-small",
        "provider_settings": {},
        "conditions": ["clean"],
        "split": "test",
        "environment": {"git_commit": "1234567", "git_dirty": True, "manifest_sha256": "abc"},
    }
    df = pd.DataFrame([{"model": "whisper-small", "language": "hi", "condition": "clean", "n_clips": 1, "n_failed": 0, "wer_norm": 0.0, "wer_raw": 0.0, "cer_norm": 0.0, "number_acc": 1.0, "rtf_median": 0.1, "wer_norm_lo": 0.0, "wer_norm_hi": 0.0, "cer_norm_lo": 0.0, "cer_norm_hi": 0.0}])
    rep = markdown_report("test-run", df, cfg_dirty)
    assert "WARNING: uncommitted changes when this run was made." in rep
    assert "git dirty: True" in rep

    # When clean, no warning
    cfg_clean = dict(cfg_dirty, environment={"git_commit": "1234567", "git_dirty": False, "manifest_sha256": "abc"})
    rep_clean = markdown_report("test-run", df, cfg_clean)
    assert "WARNING: uncommitted changes when this run was made." not in rep_clean
    assert "git dirty: False" in rep_clean


def test_whisper_provider_settings_mocked(monkeypatch, tmp_path):
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock
    from bharatspeechbench.providers.whisper import WhisperProvider

    fake_model = MagicMock()
    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    # Set up mock cache
    cache = tmp_path / "hf_cache"
    repo_dir = cache / "models--Systran--faster-whisper-small" / "refs"
    repo_dir.mkdir(parents=True)
    (repo_dir / "main").write_text("hash12345")

    provider = WhisperProvider(size="small", device="cpu", compute_type="int8", download_root=cache)
    s = provider.settings
    assert s["model_repo"] == "Systran/faster-whisper-small"
    assert s["model_revision"] == "hash12345"
    assert "ctranslate2_version" in s
    assert "faster_whisper_version" in s
    assert s["device"] == "cpu"
    assert s["compute_type"] == "int8"


def test_whisper_provider_coverage_meta_mocked(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import MagicMock
    from bharatspeechbench.providers.whisper import WhisperProvider

    fake_segs = [
        SimpleNamespace(start=0.0, end=2.0, text="पहला"),
        SimpleNamespace(start=2.0, end=5.0, text="दूसरा"),
    ]
    fake_info = SimpleNamespace(language_probability=0.99, duration=10.0)
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segs), fake_info)

    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    provider = WhisperProvider(size="small", device="cpu", compute_type="int8")
    t = provider.transcribe("nonexistent.wav", "hi", audio_seconds=10.0)
    assert t.text == "पहला दूसरा"
    assert t.meta["num_segments"] == 2
    assert t.meta["last_segment_end_s"] == 5.0
    assert t.meta["coverage"] == 0.5

    # When duration is missing from info, WhisperProvider.transcribe falls back to audio_seconds
    fake_info_no_dur = SimpleNamespace(language_probability=0.95)
    fake_model.transcribe.return_value = (iter([SimpleNamespace(start=0.0, end=4.0, text="ok")]), fake_info_no_dur)
    t2 = provider.transcribe("nonexistent.wav", "hi", audio_seconds=8.0)
    assert t2.meta["num_segments"] == 1
    assert t2.meta["last_segment_end_s"] == 4.0
    assert t2.meta["coverage"] == 0.5

    # Unknown duration / no segments -> coverage is None, tokens are 0
    fake_model.transcribe.return_value = (iter([]), SimpleNamespace(language_probability=0.5))
    t3 = provider.transcribe("nonexistent.wav", "hi", audio_seconds=0.0)
    assert t3.meta["num_segments"] == 0
    assert t3.meta["last_segment_end_s"] is None
    assert t3.meta["coverage"] is None
    assert t3.meta["total_tokens"] == 0
    assert t3.meta["max_segment_tokens"] == 0

    # Segments without .tokens attribute record None for token counts
    assert t.meta["total_tokens"] is None
    assert t.meta["max_segment_tokens"] is None


def test_whisper_provider_tokens_and_decoding_options_mocked(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from unittest.mock import MagicMock
    from bharatspeechbench.providers.whisper import WhisperProvider

    fake_segs = [
        SimpleNamespace(start=0.0, end=8.0, text="पहला", tokens=[1, 2, 3, 4]),
        SimpleNamespace(start=8.0, end=20.0, text="दूसरा", tokens=list(range(224))),
    ]
    fake_info = SimpleNamespace(language_probability=0.99, duration=20.0)
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (iter(fake_segs), fake_info)

    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    # Defaults unchanged
    p_def = WhisperProvider(size="small", device="cpu", compute_type="int8")
    assert p_def.settings["vad_filter"] is False
    assert p_def.settings["chunk_length"] is None
    assert p_def.settings["max_new_tokens"] is None
    assert p_def.settings["initial_prompt"] is None
    t_def = p_def.transcribe("nonexistent.wav", "hi", audio_seconds=20.0)
    assert t_def.meta["total_tokens"] == 228
    assert t_def.meta["max_segment_tokens"] == 224
    _, kw_def = fake_model.transcribe.call_args
    assert "vad_filter" not in kw_def
    assert "chunk_length" not in kw_def
    assert "max_new_tokens" not in kw_def
    assert "initial_prompt" not in kw_def

    # Explicit decoding options forwarded and recorded in settings
    fake_model.transcribe.return_value = (iter(fake_segs), fake_info)
    p_opt = WhisperProvider(
        size="small",
        device="cpu",
        compute_type="int8",
        vad_filter=True,
        chunk_length=15,
        max_new_tokens=200,
        initial_prompt="આ એક ગુજરાતી વાક્ય છે.",
    )
    assert p_opt.settings["vad_filter"] is True
    assert p_opt.settings["chunk_length"] == 15
    assert p_opt.settings["max_new_tokens"] == 200
    assert p_opt.settings["initial_prompt"] == "આ એક ગુજરાતી વાક્ય છે."
    p_opt.transcribe("nonexistent.wav", "gu", audio_seconds=20.0)
    _, kw_opt = fake_model.transcribe.call_args
    assert kw_opt["vad_filter"] is True
    assert kw_opt["chunk_length"] == 15
    assert kw_opt["max_new_tokens"] == 200
    assert kw_opt["initial_prompt"] == "આ એક ગુજરાતી વાક્ય છે."


def test_git_dirty_real_repo_ignores_untracked(tmp_path):
    import shutil
    import subprocess
    import pytest
    from bharatspeechbench.utils.reproducibility import _git_dirty

    if shutil.which("git") is None:
        pytest.skip("git not installed")

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True)

    tracked = repo / "tracked.txt"
    tracked.write_text("v1\n", encoding="utf-8")
    (repo / ".gitignore").write_text("ignored_dir/\nresults/raw/\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt", ".gitignore"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    assert _git_dirty(cwd=repo) is False

    # Untracked results/ folder + ignored folder -> still clean (False)
    (repo / "ignored_dir").mkdir()
    (repo / "ignored_dir" / "noise.wav").write_bytes(b"RIFF")
    (repo / "results" / "raw" / "r1").mkdir(parents=True)
    (repo / "results" / "raw" / "r1" / "config.json").write_text("{}")
    (repo / "results" / "reports").mkdir(parents=True)
    (repo / "results" / "reports" / "r1.md").write_text("# report")
    (repo / "untracked.txt").write_text("hello\n", encoding="utf-8")
    assert _git_dirty(cwd=repo) is False

    # Modified tracked file -> dirty (True)
    tracked.write_text("v2\n", encoding="utf-8")
    assert _git_dirty(cwd=repo) is True

    # Non-git directory -> None
    non_git = tmp_path / "not_a_repo"
    non_git.mkdir()
    assert _git_dirty(cwd=non_git) is None
