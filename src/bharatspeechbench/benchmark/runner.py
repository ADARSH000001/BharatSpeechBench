"""Run one provider over every (clip, condition) and store RAW transcripts.

Output (results/raw/<run_id>/):
  config.json        everything needed to reproduce: provider, model, conditions, seed, split, dataset
                     and noise checksums, and the environment record
  transcripts.jsonl  one line per (clip, condition); append-only, so an interrupted run can resume

Raw text is never edited here. Normalisation and scoring happen later (reporting.py).
A failure on one clip is recorded with its error and does NOT stop the run or count as an empty transcript.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import soundfile as sf

from ..audio.conditions import DEFAULT_CONDITIONS, ensure_condition, parse_condition
from ..providers.base import ASRProvider
from ..utils.hashing import sha256_file
from ..utils.reproducibility import capture_environment, set_seed
from .manifest import Clip

# Language code each clip's language is sent to the model as. Hinglish is decoded as Hindi:
# this is a protocol choice and is reported in docs/methodology.md.
ASR_LANGUAGE = {"hi": "hi", "gu": "gu", "hi-en": "hi"}


def _done(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["clip_id"], r["condition"]))
    return done


def run_benchmark(
    clips: list[Clip],
    provider: ASRProvider,
    run_dir: str | Path,
    root: str | Path = ".",
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    noise_wav: str | Path | None = None,
    conditions_dir: str | Path = "data/conditions",
    split: str = "test",
    languages: tuple[str, ...] | None = None,
    limit: int | None = None,
    seed: int = 0,
    manifest_path: str | Path | None = None,
    progress=print,
) -> Path:
    for c in conditions:
        parse_condition(c)
    if any(parse_condition(c)[0] == "phone_noise" for c in conditions) and noise_wav is None:
        raise ValueError("noise conditions need a noise file (--noise)")

    selected = [c for c in clips if c.split == split and (languages is None or c.language in languages)]
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"no clips selected (split={split!r}, languages={languages})")

    set_seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    noise_sha = sha256_file(noise_wav) if noise_wav else None
    # cache key includes seed + noise file, so changing either can never reuse stale audio
    cond_dir = Path(conditions_dir) / f"seed{seed}-{(noise_sha or 'nonoise')[:8]}"

    cfg_path = run_dir / "config.json"
    config = {
        "provider": provider.name,
        "model": provider.model_id,
        "provider_settings": getattr(provider, "settings", {}),
        "conditions": list(conditions),
        "split": split,
        "languages": list(languages) if languages else None,
        "limit": limit,
        "seed": seed,
        "n_clips": len(selected),
        "asr_language": ASR_LANGUAGE,
        "noise_sha256": noise_sha,
        "environment": capture_environment(seed, manifest_path),
    }
    if cfg_path.is_file():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        for k in ("provider", "model", "provider_settings", "conditions", "split", "languages",
                  "limit", "seed", "noise_sha256"):
            if old.get(k) != config[k]:
                raise ValueError(f"run dir {run_dir} was started with a different {k}; use a new --run-id")
        if old["environment"].get("manifest_sha256") != config["environment"]["manifest_sha256"]:
            raise ValueError(f"manifest changed since run dir {run_dir} was started; use a new --run-id")
    else:
        cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    out = run_dir / "transcripts.jsonl"
    done = _done(out)

    # Warm-up: the first call pays model/GPU start-up cost; do it outside the timed results.
    first = selected[0]
    warm = ensure_condition(Path(root) / first.path, cond_dir, "clean", first.id, noise_wav, seed)
    provider.transcribe(str(warm), ASR_LANGUAGE[first.language], sf.info(warm).duration)

    total = len(selected) * len(conditions)
    n = 0
    with out.open("a", encoding="utf-8") as fh:
        for clip in selected:
            for cond in conditions:
                n += 1
                if (clip.id, cond) in done:
                    continue
                row = {"clip_id": clip.id, "language": clip.language, "condition": cond,
                       "provider": provider.name, "model": provider.model_id}
                try:
                    wav = ensure_condition(Path(root) / clip.path, cond_dir, cond, clip.id, noise_wav, seed)
                    secs = sf.info(wav).duration
                    t = provider.transcribe(str(wav), ASR_LANGUAGE[clip.language], secs)
                    row.update(text=t.text, audio_seconds=t.audio_seconds,
                               processing_seconds=t.processing_seconds, rtf=t.rtf, meta=t.meta, error=None)
                except Exception as e:  # noqa: BLE001 - record and continue
                    row.update(text=None, audio_seconds=None, processing_seconds=None, rtf=None,
                               meta={}, error=f"{type(e).__name__}: {e}")
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                progress(f"[{n}/{total}] {clip.id} {cond}" + (f"  ERROR {row['error']}" if row["error"] else ""))
    return run_dir
