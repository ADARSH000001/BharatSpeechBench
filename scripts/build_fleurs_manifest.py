"""Download Google FLEURS Hindi/Gujarati audio and build data/manifests/manifest.jsonl.

Licence: cc-by-4.0 (Creative Commons Attribution 4.0 International)
Source: https://huggingface.co/datasets/google/fleurs
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import soundfile as sf

from bharatspeechbench.benchmark.manifest import (
    Clip,
    check_speakers,
    load_manifest,
    verify_audio_hashes,
    write_manifest,
)
from bharatspeechbench.utils.hashing import sha256_file

FLEURS_LANG_CODES = {"hi": "hi_in", "gu": "gu_in"}
FLEURS_LICENCE = "cc-by-4.0"
HF_BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main"
MIN_SAMPLES = 48000  # 3.0s at 16 kHz


def _download_stream(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (BharatSpeechBench)"})
    return urllib.request.urlopen(req, timeout=30)


def load_tsv_rows(tsv_path: str | Path) -> list[dict]:
    """Parse FLEURS TSV into structured row dicts."""
    rows = []
    lines = Path(tsv_path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 6:
            rows.append({
                "sentence_id": parts[0].strip(),
                "filename": parts[1].strip(),
                "raw_transcription": parts[2].strip(),
                "transcription": parts[3].strip(),
                "num_samples": int(parts[5].strip()) if parts[5].strip().isdigit() else 0,
            })
    return rows


def get_fleurs_tsv(lang: str, split: str, out_dir: Path, source_dir: Path | None = None) -> list[dict]:
    """Fetch FLEURS TSV either from local source_dir or from Hugging Face."""
    out_dir.mkdir(parents=True, exist_ok=True)
    local_tsv = out_dir / f"{split}.tsv"

    if source_dir is not None:
        src = source_dir / f"{split}.tsv"
        return load_tsv_rows(src)

    if not local_tsv.is_file():
        code = FLEURS_LANG_CODES[lang]
        url = f"{HF_BASE_URL}/data/{code}/{split}.tsv"
        with _download_stream(url) as resp, local_tsv.open("wb") as fh:
            shutil.copyfileobj(resp, fh)

    return load_tsv_rows(local_tsv)


def extract_audio_clips(
    lang: str,
    split: str,
    needed_filenames: set[str],
    out_audio_dir: Path,
    source_dir: Path | None = None,
) -> None:
    """Extract only the needed audio files into out_audio_dir."""
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    remaining = {f for f in needed_filenames if not (out_audio_dir / f).is_file()}
    if not remaining:
        return

    if source_dir is not None:
        local_tar = source_dir / f"{split}.tar.gz"
        if local_tar.is_file():
            with tarfile.open(local_tar, "r:*") as tar:
                for member in tar:
                    base = Path(member.name).name
                    if base in remaining:
                        f = tar.extractfile(member)
                        if f:
                            (out_audio_dir / base).write_bytes(f.read())
                            remaining.remove(base)
                    if not remaining:
                        break
        else:
            for fname in list(remaining):
                src_wav = source_dir / fname
                if src_wav.is_file():
                    shutil.copyfile(src_wav, out_audio_dir / fname)
                    remaining.remove(fname)
        return

    # Stream over HTTP and stop as soon as all needed files are extracted
    code = FLEURS_LANG_CODES[lang]
    url = f"{HF_BASE_URL}/data/{code}/audio/{split}.tar.gz"
    resp = _download_stream(url)
    try:
        with tarfile.open(mode="r|gz", fileobj=resp) as tar:
            for member in tar:
                base = Path(member.name).name
                if base in remaining:
                    extracted = tar.extractfile(member)
                    if extracted:
                        (out_audio_dir / base).write_bytes(extracted.read())
                        remaining.remove(base)
                if not remaining:
                    break
    finally:
        resp.close()


def pick_clips_deterministically(candidates: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Filter to >= 3.0s and pick n clips deterministically using a fixed seed."""
    valid = [c for c in candidates if c["num_samples"] >= MIN_SAMPLES and c["raw_transcription"]]
    valid.sort(key=lambda x: x["filename"])
    if len(valid) <= n:
        return valid
    rng = random.Random(seed)
    chosen = rng.sample(valid, n)
    chosen.sort(key=lambda x: x["filename"])
    return chosen


def build_fleurs_manifest(
    languages: tuple[str, ...] = ("hi", "gu"),
    split: str = "test",
    n: int = 3,
    out_dir: str | Path = "data/downloads/fleurs",
    manifest_path: str | Path = "data/manifests/manifest.jsonl",
    seed: int = 0,
    root: str | Path = ".",
    source_dirs: dict[str, Path] | None = None,
    exclude_manifest: str | Path | None = None,
) -> list[Clip]:
    root_path = Path(root).resolve()
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = root_path / manifest_file
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    excluded_ids: set[str] = set()
    if exclude_manifest is not None:
        excl_file = Path(exclude_manifest)
        if not excl_file.is_absolute() and not excl_file.is_file():
            excl_file = root_path / excl_file
        if not excl_file.is_file():
            raise FileNotFoundError(f"exclude-manifest not found: {excl_file}")
        excluded_ids = {c.id for c in load_manifest(excl_file)}

    existing_clips = load_manifest(manifest_file) if manifest_file.is_file() else []
    existing_ids = {c.id for c in existing_clips}

    new_clips: list[Clip] = []

    for lang in languages:
        lang_out = Path(out_dir)
        if not lang_out.is_absolute():
            lang_out = root_path / lang_out
        lang_audio_dir = lang_out / lang

        src_dir = source_dirs.get(lang) if source_dirs else None
        rows = get_fleurs_tsv(lang, split, lang_out / lang, source_dir=src_dir)
        all_valid = [r for r in rows if r["num_samples"] >= MIN_SAMPLES and r["raw_transcription"]]
        if excluded_ids:
            eligible_rows = [
                r for r in all_valid
                if f"fleurs-{lang}-{Path(r['filename']).stem}" not in excluded_ids
            ]
        else:
            eligible_rows = all_valid
        selected_rows = pick_clips_deterministically(eligible_rows, n, seed=seed)
        if len(selected_rows) < n:
            if excluded_ids:
                print(
                    f"Note [{lang}/{split}]: requested {n} clips, {len(selected_rows)} available "
                    f"({len(all_valid)} eligible exist, {len(all_valid) - len(eligible_rows)} excluded); "
                    f"using all {len(selected_rows)}."
                )
            else:
                print(
                    f"Note [{lang}/{split}]: requested {n} clips, only {len(selected_rows)} eligible exist; "
                    f"using all {len(selected_rows)}."
                )
        needed_filenames = {r["filename"] for r in selected_rows}

        extract_audio_clips(lang, split, needed_filenames, lang_audio_dir, source_dir=src_dir)

        for r in selected_rows:
            audio_path = lang_audio_dir / r["filename"]
            stem = Path(r["filename"]).stem
            clip_id = f"fleurs-{lang}-{stem}"
            if clip_id in existing_ids:
                continue

            rel_path = str(audio_path.relative_to(root_path)).replace("\\", "/")
            duration = round(float(sf.info(audio_path).duration), 2)
            sha = sha256_file(audio_path)

            clip = Clip(
                id=clip_id,
                path=rel_path,
                language=lang,
                source="fleurs",
                split=split,
                reference=r["raw_transcription"],
                duration_s=duration,
                sha256=sha,
                licence=FLEURS_LICENCE,
                speaker_consent=None,
                speaker_id=None,
            )
            new_clips.append(clip)
            existing_ids.add(clip_id)

    combined = existing_clips + new_clips
    write_manifest(combined, manifest_file)
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build FLEURS manifest entries and download audio.")
    parser.add_argument("--lang", nargs="+", choices=("hi", "gu"), default=["hi", "gu"],
                        help="Languages to include (default: hi gu)")
    parser.add_argument("--split", choices=("dev", "test", "train"), default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of clips per language (default: 3)")
    parser.add_argument("--out", default="data/downloads/fleurs",
                        help="Directory to save downloaded audio (default: data/downloads/fleurs)")
    parser.add_argument("--manifest", default="data/manifests/manifest.jsonl",
                        help="Manifest file path (default: data/manifests/manifest.jsonl)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Fixed seed for deterministic clip selection (default: 0)")
    parser.add_argument("--root", default=".",
                        help="Root folder relative to which manifest paths are stored (default: .)")
    parser.add_argument("--exclude-manifest", default=None,
                        help="Optional manifest JSONL file whose clip ids will be skipped")

    args = parser.parse_args(argv)

    print(f"Building FLEURS manifest for {args.lang} (split={args.split}, n={args.n})...")
    build_fleurs_manifest(
        languages=tuple(args.lang),
        split=args.split,
        n=args.n,
        out_dir=args.out,
        manifest_path=args.manifest,
        seed=args.seed,
        root=args.root,
        exclude_manifest=args.exclude_manifest,
    )

    manifest_file = Path(args.manifest)
    if not manifest_file.is_absolute():
        manifest_file = Path(args.root).resolve() / manifest_file
    clips = load_manifest(manifest_file)
    print(f"\nManifest verification: {len(clips)} clips")
    for key in ("language", "source", "split"):
        counts = Counter(getattr(c, key) for c in clips)
        print(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    n_spk, overlap = check_speakers(clips)
    print(f"  speakers: {n_spk} distinct")
    if overlap:
        print(f"WARNING: speaker_id in both dev and test: {', '.join(sorted(overlap))}")
    problems = verify_audio_hashes(clips, args.root)
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    if problems:
        print("FAIL: audio verification failed", file=sys.stderr)
        return 1
    print("all audio files present and checksums match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

