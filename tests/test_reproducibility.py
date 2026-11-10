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

    # Clean repo mock
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr=""))
    assert _git_dirty() is False

    # Dirty repo mock
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=" M file.py\n", stderr=""))
    assert _git_dirty() is True

    # Git unavailable mock
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: not a git repo"))
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
