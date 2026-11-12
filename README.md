# BharatSpeechBench

Benchmark Indian-language speech recognition (Hindi and Gujarati so far; Hinglish planned (no clips yet)) under clean,
telephone-quality and noisy audio. Local models only: no API keys, no paid services.

**Status: Pilot run done on 60 FLEURS test clips (30 `hi`, 30 `gu`) with `whisper-small`; no published conclusions yet; Gujarati baseline unusable on dev (see `docs/decoding_experiment_dev.md`); Hinglish has no data yet.**

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
bsb --help
```
Needs `ffmpeg` on PATH. Add `".[whisper]"` for faster-whisper and `".[app]"` for Streamlit.

## Commands (`bsb`, also `bharatspeechbench`)
| Command | Purpose |
|---|---|
| `bsb hash FILE...` | SHA-256 of audio files, for the manifest |
| `bsb check-manifest FILE [--verify-audio]` | validate the manifest and audio checksums |
| `bsb env --seed 0 --manifest FILE` | environment record to store with every run |
| `bsb simulate MANIFEST --noise NOISE.wav` | write clean / phone / phone_noise audio so you can listen to it |
| `bsb run MANIFEST --noise NOISE.wav [--size small --device auto]` | transcribe every clip x condition; raw output to `results/raw/<run_id>/` |
| `bsb report results/raw/<run_id> --manifest MANIFEST` | scores to `results/processed/`, markdown to `results/reports/` |

## Run a benchmark
```bash
bsb check-manifest data/manifests/manifest.jsonl --verify-audio

# Smoke test across all languages (a smoke run is not a result):
bsb run data/manifests/manifest.jsonl --conditions clean phone --limit-per-language 2 --run-id smoke-v1

# Inspect raw transcripts and quality warnings:
python scripts/show_transcripts.py results/raw/smoke-v1 --manifest data/manifests/manifest.jsonl

# Full benchmark run and report:
bsb run data/manifests/manifest.jsonl --noise data/downloads/noise.wav --run-id whisper-small-v1
bsb report results/raw/whisper-small-v1 --manifest data/manifests/manifest.jsonl
streamlit run app/Home.py
```
Re-run the same `--run-id` to resume an interrupted run. A run refuses to continue if the model, conditions, split,
noise file, limit, or manifest changed. Note that a smoke run is not a result. `--device cpu` if the GPU fails to load (Windows needs CUDA 12 cuBLAS + cuDNN 9).

## Noise
Noisy conditions use 16 kHz mono clips prepared from the **DEMAND** corpus (Zenodo record `1227121`, CC BY-SA 3.0; environments `STRAFFIC`, `PCAFETER`, `DKITCHEN`).
Original archives live in `data/downloads/noise/` and prepared files in `data/noise/*.wav` (both git-ignored); only `data/noise/SOURCES.md` is tracked.
Use one noise file per run and include the noise type in `--run-id` (synthetic SNR mixing is not a real noisy recording):
```bash
python scripts/prepare_noise.py --src data/downloads/noise --out data/noise --env STRAFFIC --env PCAFETER --env DKITCHEN --seconds 120
bsb run data/manifests/pilot.jsonl --noise data/noise/STRAFFIC.wav --run-id whisper-small-straffic-v1
bsb report results/raw/whisper-small-straffic-v1 --manifest data/manifests/pilot.jsonl
```

## Data attribution
- **Google FLEURS** (`google/fleurs` on Hugging Face; Conneau et al., 2022): Hindi (`hi_in`) and Gujarati (`gu_in`), licensed under CC BY 4.0.
- **DEMAND** (Zenodo record `1227121`, CC BY-SA 3.0; environments `STRAFFIC`, `PCAFETER`, `DKITCHEN`): environmental noise corpus.

## Layout
```
app/                      Streamlit dashboard (read-only)
src/bharatspeechbench/
  providers/              one interface for every ASR model (Whisper written)
  audio/                  phone simulation, noise at fixed SNR, condition generator
  metrics/                normalisation, WER/CER, bootstrap interval, number accuracy
  benchmark/              manifest, runner, reporting
  utils/                  hashing, reproducibility
  cli.py
data/                     manifest + instructions (audio is git-ignored)
results/                  created by runs: raw/ is git-ignored; processed/ (parquet) and reports/ (markdown) are committed
tests/  docs/
```

## Next to build
1. Compare `whisper-medium` and `large-v3` on the dev set.
2. Record ~30 consented Hinglish clips with digits.
3. IndicConformer provider, same interface and conditions.

## Limits
- Small test sets give wide confidence intervals: this is a pilot study, not a language benchmark.
- Phone simulation is an approximation of a telephone channel.
- Public test sets may overlap with model training data.

## Licence
MIT for the code. Datasets and models keep their own licences.
