"""Prepare 16 kHz mono PCM16 noise clips and provenance (SOURCES.md) from local archives or WAV folders.

Never modifies original files.

Usage:
    python scripts/prepare_noise.py --src data/downloads/noise --out data/noise \\
        --env STRAFFIC --env PCAFETER --env DKITCHEN --seconds 120
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import zipfile

import soundfile as sf

from bharatspeechbench.audio.phone_simulation import _run
from bharatspeechbench.utils.hashing import sha256_file

DEFAULT_SOURCE_URL = "https://zenodo.org/records/1227121"
DEFAULT_LICENCE = (
    "This work, the audio data and the document describing it, is licensed under a "
    "Creative Commons Attribution-ShareAlike 3.0 Unported License. "
    "(DEMAND.pdf / Zenodo description; Zenodo metadata tag: cc-by-4.0)"
)


def _discover_envs(src_dir: Path) -> list[str]:
    """Discover environment names present as subdirectories or zip files in src_dir."""
    names: dict[str, str] = {}
    for p in sorted(src_dir.iterdir(), key=lambda x: x.name.lower()):
        if p.is_dir() and list(p.rglob("*.wav")):
            names.setdefault(p.name.lower(), p.name)
        elif p.is_file() and p.suffix.lower() == ".zip":
            stem = p.stem
            for suffix in ("_16k", "_48k"):
                if stem.lower().endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            names.setdefault(stem.lower(), stem)
    return list(names.values())


def _find_env_sources(src_dir: Path, env: str) -> tuple[str, list[Path | str], Path | None]:
    """Locate sorted WAV inputs for an environment inside a folder or zip archive."""
    if not src_dir.is_dir():
        raise FileNotFoundError(f"source directory not found: {src_dir}")

    env_low = env.lower()

    # 1. Subdirectory matching env or env_16k
    for cand in (env, f"{env}_16k"):
        for p in sorted(src_dir.iterdir(), key=lambda x: x.name.lower()):
            if p.is_dir() and p.name.lower() == cand.lower():
                wavs = sorted(
                    [w for w in p.rglob("*") if w.is_file() and w.suffix.lower() == ".wav"],
                    key=lambda w: w.relative_to(src_dir).as_posix().lower(),
                )
                if wavs:
                    return "files", wavs, None

    # 2. Zip archive matching env_16k.zip, env.zip, env_48k.zip
    zips_by_low = {p.name.lower(): p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip"}
    for zip_name in (f"{env_low}_16k.zip", f"{env_low}.zip", f"{env_low}_48k.zip"):
        if zip_name in zips_by_low:
            zpath = zips_by_low[zip_name]
            with zipfile.ZipFile(zpath, "r") as zf:
                members = sorted(
                    [m for m in zf.namelist() if not m.endswith("/") and m.lower().endswith(".wav")],
                    key=str.lower,
                )
            if not members:
                raise ValueError(f"zip archive {zpath} contains no .wav files")
            return "zip", members, zpath

    # 3. Flat files prefixed with env_ or env-
    flat = sorted(
        [
            p for p in src_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".wav"
            and (p.name.lower().startswith(f"{env_low}_") or p.name.lower().startswith(f"{env_low}-"))
        ],
        key=lambda p: p.name.lower(),
    )
    if flat:
        return "files", flat, None

    raise FileNotFoundError(f"no audio files or zip found for environment {env!r} in {src_dir}")


def _collect_needed_inputs(
    kind: str,
    items: list[Path | str],
    src_dir: Path,
    zpath: Path | None,
    work_tmp: Path,
    seconds: float,
) -> tuple[list[Path], list[str], float]:
    """Select leading channel files in fixed order until covering at least `seconds` of audio."""
    chosen_paths: list[Path] = []
    orig_names: list[str] = []
    cum_dur = 0.0

    if kind == "zip":
        assert zpath is not None
        with zipfile.ZipFile(zpath, "r") as zf:
            for member in items:
                member_str = str(member)
                dest = work_tmp / f"part_{len(chosen_paths):03d}.wav"
                with zf.open(member_str, "r") as src_f, dest.open("wb") as dst_f:
                    dst_f.write(src_f.read())
                dur = float(sf.info(dest).duration)
                chosen_paths.append(dest)
                orig_names.append(f"{zpath.name}:{member_str}")
                cum_dur += dur
                if cum_dur >= seconds:
                    break
    else:
        for p in items:
            wav_p = Path(p)
            dur = float(sf.info(wav_p).duration)
            chosen_paths.append(wav_p)
            orig_names.append(wav_p.relative_to(src_dir).as_posix())
            cum_dur += dur
            if cum_dur >= seconds:
                break

    if cum_dur < seconds:
        raise ValueError(f"insufficient audio duration ({cum_dur:.2f}s < requested {seconds}s)")

    return chosen_paths, orig_names, cum_dur


def _write_16k_mono_pcm16(inputs: list[Path], out_wav: Path, seconds: float) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    dur_str = f"{seconds:g}"
    if len(inputs) == 1:
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(inputs[0]),
            "-t", dur_str,
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
            str(out_wav),
        ])
        return

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in inputs:
        cmd.extend(["-i", str(p)])
    prep = [
        f"[{i}:a]aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono[a{i}]"
        for i in range(len(inputs))
    ]
    concat_in = "".join(f"[a{i}]" for i in range(len(inputs)))
    filt = ";".join(prep) + f";{concat_in}concat=n={len(inputs)}:v=0:a=1[out]"
    cmd.extend([
        "-filter_complex", filt,
        "-map", "[out]",
        "-t", dur_str,
        "-acodec", "pcm_s16le",
        str(out_wav),
    ])
    _run(cmd)


def _format_sources_md(records: list[dict]) -> str:
    lines = [
        "# Noise Sources & Provenance",
        "",
        "Prepared locally via `scripts/prepare_noise.py`. Raw `.wav` noise files are git-ignored; this file is tracked.",
        "",
        "| file | environment | source URL | licence | original file names | sha256 | download date | seconds used |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        orig = ", ".join(f"`{n}`" for n in rec["original_files"])
        lines.append(
            f"| `{rec['file']}` | `{rec['environment']}` | {rec['source_url']} | "
            f"{rec['licence']} | {orig} | `{rec['sha256']}` | {rec['download_date']} | {rec['seconds_used']:g} |"
        )
    return "\n".join(lines) + "\n"


def prepare_noise(
    src_dir: str | Path,
    out_dir: str | Path = "data/noise",
    envs: list[str] | tuple[str, ...] | None = None,
    seconds: float = 120.0,
    source_url: str = DEFAULT_SOURCE_URL,
    licence: str = DEFAULT_LICENCE,
    download_date: str | None = None,
) -> list[dict]:
    """Prepare 16 kHz mono PCM16 noise files and write SOURCES.md. Returns metadata dicts per output."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    src_path = Path(src_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_envs = list(envs) if envs else _discover_envs(src_path)
    if not target_envs:
        raise ValueError(f"no environments specified or discovered in {src_path}")

    records: list[dict] = []
    for env in target_envs:
        kind, items, zpath = _find_env_sources(src_path, env)
        with tempfile.TemporaryDirectory() as tmp:
            inputs, orig_names, _ = _collect_needed_inputs(
                kind, items, src_path, zpath, Path(tmp), float(seconds)
            )
            out_wav = out_path / f"{env}.wav"
            _write_16k_mono_pcm16(inputs, out_wav, float(seconds))

        ref_stat_path = zpath if zpath is not None else Path(items[0])
        dl_date = download_date or datetime.fromtimestamp(
            ref_stat_path.stat().st_mtime, tz=timezone.utc
        ).date().isoformat()
        digest = sha256_file(out_wav)
        info = sf.info(out_wav)
        records.append({
            "file": out_wav.name,
            "path": out_wav,
            "environment": env,
            "source_url": source_url,
            "licence": licence,
            "original_files": orig_names,
            "sha256": digest,
            "download_date": dl_date,
            "seconds_used": round(float(info.duration), 3),
        })

    (out_path / "SOURCES.md").write_text(_format_sources_md(records), encoding="utf-8")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert downloaded noise archives/WAVs into 16 kHz mono PCM16 files with SOURCES.md."
    )
    parser.add_argument("--src", required=True, help="Folder of downloaded WAV files/folders or zips")
    parser.add_argument("--out", default="data/noise", help="Output folder (default: data/noise)")
    parser.add_argument("--env", action="append", default=None,
                        help="Environment name (repeatable; defaults to all environments found in --src)")
    parser.add_argument("--seconds", type=float, default=120.0,
                        help="First N seconds to keep per environment (default: 120)")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL, help="Official dataset URL")
    parser.add_argument("--licence", default=DEFAULT_LICENCE, help="Exact licence string")
    parser.add_argument("--download-date", default=None, help="Download date YYYY-MM-DD (default: source mtime)")

    args = parser.parse_args(argv)
    try:
        records = prepare_noise(
            src_dir=args.src,
            out_dir=args.out,
            envs=args.env,
            seconds=args.seconds,
            source_url=args.source_url,
            licence=args.licence,
            download_date=args.download_date,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for rec in records:
        print(f"{rec['sha256']}  {rec['path']}  ({rec['seconds_used']}s)")
    print(f"wrote {Path(args.out) / 'SOURCES.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

