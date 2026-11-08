import hashlib
from bharatspeechbench.utils.hashing import sha256_file


def test_sha256_matches_hashlib_incl_large_file(tmp_path):
    data = b"abc" * 700_000          # > one 1 MB chunk
    f = tmp_path / "x.bin"
    f.write_bytes(data)
    assert sha256_file(f) == hashlib.sha256(data).hexdigest()
