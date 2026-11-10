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
    for key in ("python", "os", "cpu", "ram_bytes", "gpu", "ffmpeg", "packages", "git_commit", "timestamp_utc"):
        assert key in env
    assert env["packages"]["numpy"] is not None


def test_set_seed_reproducible():
    set_seed(1)
    a = (random.random(), np.random.rand())
    set_seed(1)
    b = (random.random(), np.random.rand())
    assert a == b


def test_get_model_revision_from_cache(tmp_path):
    from bharatspeechbench.providers.whisper import get_model_revision, _resolve_model_repo

    repo = _resolve_model_repo("small")
    assert repo == "Systran/faster-whisper-small"

    # Missing cache directory
    assert get_model_revision(repo, cache_dir=tmp_path / "nonexistent") is None

    # Cache with refs/main
    repo_dir = tmp_path / f"models--{repo.replace('/', '--')}"
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("abcdef1234567890abcdef1234567890abcdef12\n")
    assert get_model_revision(repo, cache_dir=tmp_path) == "abcdef1234567890abcdef1234567890abcdef12"

    # Cache with snapshots folder fallback
    (refs_dir / "main").unlink()
    snaps_dir = repo_dir / "snapshots"
    (snaps_dir / "commit999").mkdir(parents=True)
    assert get_model_revision(repo, cache_dir=tmp_path) == "commit999"


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
