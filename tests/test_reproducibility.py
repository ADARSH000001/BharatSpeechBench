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
