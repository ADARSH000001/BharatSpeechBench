"""One combined dataset manifest (JSONL, one clip per line). Audio files are NOT committed.

Fields per clip:
  id, path, language (hi | gu | hi-en), source (e.g. fleurs, kathbath, own), split (dev | test),
  reference (verified transcript), duration_s, sha256 (of the audio file), licence,
  speaker_consent (required true for source == "own")
`path` is relative to the root passed to verify_audio_hashes (default: repo root).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from ..utils.hashing import sha256_file

LANGUAGES = ("hi", "gu", "hi-en")
SPLITS = ("dev", "test")
REQUIRED = ("id", "path", "language", "source", "split", "reference", "duration_s", "sha256", "licence")


@dataclass
class Clip:
    id: str
    path: str
    language: str
    source: str
    split: str
    reference: str
    duration_s: float
    sha256: str
    licence: str
    speaker_consent: bool | None = None
    speaker_id: str | None = None


def check_speakers(clips: list[Clip]) -> tuple[int, set[str]]:
    """Return (n_distinct_speakers, overlapping_speakers_between_dev_and_test)."""
    speakers = {c.speaker_id for c in clips if c.speaker_id is not None}
    dev_spk = {c.speaker_id for c in clips if c.split == "dev" and c.speaker_id is not None}
    test_spk = {c.speaker_id for c in clips if c.split == "test" and c.speaker_id is not None}
    return len(speakers), dev_spk & test_spk


def load_manifest(path: str | Path) -> list[Clip]:
    clips: list[Clip] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"line {n}: missing fields {missing}")
        if row["language"] not in LANGUAGES:
            raise ValueError(f"line {n}: language must be one of {LANGUAGES}")
        if row["split"] not in SPLITS:
            raise ValueError(f"line {n}: split must be one of {SPLITS}")
        if not str(row["reference"]).strip():
            raise ValueError(f"line {n}: empty reference transcript")
        if row["id"] in seen:
            raise ValueError(f"line {n}: duplicate id {row['id']!r}")
        if row["source"] == "own" and row.get("speaker_consent") is not True:
            raise ValueError(f"line {n}: own recordings need speaker_consent: true")
        seen.add(row["id"])
        clips.append(Clip(**{k: row.get(k) for k in Clip.__dataclass_fields__}))
    return clips


def write_manifest(clips: list[Clip], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in clips) + "\n", encoding="utf-8"
    )


def verify_audio_hashes(clips: list[Clip], root: str | Path = ".") -> list[str]:
    """Return a list of problems (missing file or checksum mismatch). Empty list = all good."""
    problems: list[str] = []
    for c in clips:
        f = Path(root) / c.path
        if not f.is_file():
            problems.append(f"{c.id}: file not found: {f}")
        elif sha256_file(f) != c.sha256:
            problems.append(f"{c.id}: checksum mismatch: {f}")
    return problems
