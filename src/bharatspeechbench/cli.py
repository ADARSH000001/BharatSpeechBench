"""Command line: `bsb` (alias: `bharatspeechbench`).

  bsb hash FILE...                      print SHA-256 for audio files (paste into the manifest)
  bsb check-manifest FILE [--verify-audio --root DIR]
  bsb env [--seed N] [--manifest FILE]  print the reproducibility record as JSON
  bsb simulate MANIFEST --noise FILE    write clean / phone / phone_noise audio (listen to it!)
  bsb run MANIFEST --noise FILE         transcribe every clip x condition, raw results to results/raw/
  bsb report RUN_DIR --manifest FILE    score a run -> results/processed/ and results/reports/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from . import __version__
from .benchmark.manifest import check_speakers, load_manifest, verify_audio_hashes
from .utils.hashing import sha256_file
from .utils.reproducibility import capture_environment


def _cmd_hash(args: argparse.Namespace) -> int:
    for f in args.files:
        print(f"{sha256_file(f)}  {f}")
    return 0


def _cmd_check_manifest(args: argparse.Namespace) -> int:
    try:
        clips = load_manifest(args.manifest)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    print(f"{len(clips)} clips")
    for key in ("language", "source", "split"):
        counts = Counter(getattr(c, key) for c in clips)
        print(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    n_spk, overlap = check_speakers(clips)
    print(f"  speakers: {n_spk} distinct")
    if overlap:
        print(f"WARNING: speaker_id in both dev and test: {', '.join(sorted(overlap))}")
    if args.verify_audio:
        problems = verify_audio_hashes(clips, args.root)
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        if problems:
            return 1
        print("all audio files present and checksums match")
    return 0


def _cmd_env(args: argparse.Namespace) -> int:
    print(json.dumps(capture_environment(args.seed, args.manifest), indent=2, ensure_ascii=False))
    return 0


def _conditions(args) -> tuple[str, ...]:
    return tuple(args.conditions)


def _cmd_simulate(args: argparse.Namespace) -> int:
    from .audio.conditions import ensure_condition
    from .utils.hashing import sha256_file

    all_clips = load_manifest(args.manifest)
    split_counts = Counter(c.split for c in all_clips)
    if args.split is not None and split_counts.get(args.split, 0) == 0:
        counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(split_counts.items()))
        print(f"ERROR: no clips match split {args.split!r} (available splits: {counts_str}); try --split dev", file=sys.stderr)
        return 1
    clips = [c for c in all_clips if args.split in (None, c.split)]
    if args.limit_per_language:
        lang_counts: dict[str, int] = {}
        filtered = []
        for c in clips:
            cnt = lang_counts.get(c.language, 0)
            if cnt < args.limit_per_language:
                filtered.append(c)
                lang_counts[c.language] = cnt + 1
        clips = filtered
    elif args.limit:
        clips = clips[: args.limit]
    noise_sha = sha256_file(args.noise) if args.noise else None
    out = f"{args.out}/seed{args.seed}-{(noise_sha or 'nonoise')[:8]}"
    for c in clips:
        for cond in _conditions(args):
            p = ensure_condition(f"{args.root}/{c.path}", out, cond, c.id, args.noise, args.seed, args.force)
            print(p)
    return 0


def _make_provider(args):
    if args.provider == "whisper":
        from .providers.whisper import WhisperProvider
        return WhisperProvider(
            args.size,
            args.device,
            args.compute_type,
            vad_filter=getattr(args, "vad_filter", False),
            chunk_length=getattr(args, "chunk_length", None),
            max_new_tokens=getattr(args, "max_new_tokens", None),
        )
    raise SystemExit(f"unknown provider {args.provider!r} (available: whisper)")


def _cmd_run(args: argparse.Namespace) -> int:
    from datetime import datetime

    from .benchmark.runner import run_benchmark

    clips = load_manifest(args.manifest)
    problems = verify_audio_hashes(clips, args.root)
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        return 1
    provider = _make_provider(args)
    run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{provider.model_id}"
    try:
        run_dir = run_benchmark(
            clips, provider, f"results/raw/{run_id}", root=args.root, conditions=_conditions(args),
            noise_wav=args.noise, conditions_dir=args.out, split=args.split,
            languages=tuple(args.languages) if args.languages else None, limit=args.limit,
            limit_per_language=args.limit_per_language,
            seed=args.seed, manifest_path=args.manifest)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"raw results: {run_dir}\nnext: bsb report {run_dir} --manifest {args.manifest}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .benchmark.reporting import report_run

    paths = report_run(args.run_dir, load_manifest(args.manifest))
    for k, v in paths.items():
        print(f"{k}: {v}")
    return 0


def _add_audio_args(sp: argparse.ArgumentParser) -> None:
    from .audio.conditions import DEFAULT_CONDITIONS

    sp.add_argument("manifest")
    sp.add_argument("--root", default=".", help="folder that manifest paths are relative to")
    sp.add_argument("--noise", default=None, help="noise WAV (check its licence); needed for phone_noise_*")
    sp.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    sp.add_argument("--split", default="test", choices=("dev", "test"))
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--limit", type=int, default=None, help="only the first N clips (smoke test)")
    sp.add_argument("--limit-per-language", type=int, default=None,
                    help="first N clips per language in manifest order")
    sp.add_argument("--out", default="data/conditions", help="where generated audio is cached")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bsb")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hash", help="SHA-256 of files")
    h.add_argument("files", nargs="+")
    h.set_defaults(fn=_cmd_hash)

    c = sub.add_parser("check-manifest", help="validate a manifest")
    c.add_argument("manifest")
    c.add_argument("--verify-audio", action="store_true", help="also check files exist and checksums match")
    c.add_argument("--root", default=".", help="folder that manifest paths are relative to")
    c.set_defaults(fn=_cmd_check_manifest)

    e = sub.add_parser("env", help="print environment record as JSON")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--manifest", default=None)
    e.set_defaults(fn=_cmd_env)

    sim = sub.add_parser("simulate", help="generate clean/phone/noisy audio for a manifest")
    _add_audio_args(sim)
    sim.add_argument("--force", action="store_true", help="regenerate even if files exist")
    sim.set_defaults(fn=_cmd_simulate)

    r = sub.add_parser("run", help="run a model over the manifest")
    _add_audio_args(r)
    r.add_argument("--provider", default="whisper", choices=("whisper",))
    r.add_argument("--size", default="small", help="Whisper size: tiny, base, small, medium, large-v3")
    r.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    r.add_argument("--compute-type", default="auto", help="auto, float16, int8, ...")
    r.add_argument("--vad-filter", action="store_true", default=False,
                   help="Whisper: enable Silero VAD filter (default: off)")
    r.add_argument("--chunk-length", type=int, default=None,
                   help="Whisper: audio chunk length in seconds (default: library default)")
    r.add_argument("--max-new-tokens", type=int, default=None,
                   help="Whisper: max new tokens per chunk (default: library default)")
    r.add_argument("--languages", nargs="+", choices=("hi", "gu", "hi-en"), default=None)
    r.add_argument("--run-id", default=None, help="reuse an id to RESUME an interrupted run")
    r.set_defaults(fn=_cmd_run)

    rep = sub.add_parser("report", help="score a run")
    rep.add_argument("run_dir")
    rep.add_argument("--manifest", required=True)
    rep.set_defaults(fn=_cmd_report)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
