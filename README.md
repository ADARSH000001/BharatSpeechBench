# BharatSpeechBench

Benchmark Indian-language speech recognition (Hindi, Gujarati, Hinglish) under clean,
telephone-quality and noisy audio. Local models only: no API keys, no paid services.

**Status: pipeline complete for Whisper; no published results yet.** Manifest, audio conditions, runner, scoring,
reports and a read-only dashboard work and are tested with a fake model. The Whisper provider has not yet been
run on real audio in CI. IndicConformer is next. Read `docs/methodology.md` before quoting any number.

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
bsb run data/manifests/manifest.jsonl --noise data/downloads/noise.wav --limit 3 --conditions clean phone  # smoke test
bsb run data/manifests/manifest.jsonl --noise data/downloads/noise.wav --run-id whisper-small-v1
bsb report results/raw/whisper-small-v1 --manifest data/manifests/manifest.jsonl
streamlit run app/Home.py
```
Re-run the same `--run-id` to resume an interrupted run. A run refuses to continue if the model, conditions, split,
noise file or manifest changed. `--device cpu` if the GPU fails to load (Windows needs CUDA 12 cuBLAS + cuDNN 9).

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
1. Collect the data: FLEURS Hindi/Gujarati (check the official licence), your own consented Hinglish clips
   with digits in the script, a licensed noise file; build `data/manifests/manifest.jsonl`.
2. First real Whisper run on your GPU; commit the report in `results/reports/`.
3. IndicConformer provider, same interface, same conditions.

## Limits
- Small test sets give wide confidence intervals: this is a pilot study, not a language benchmark.
- Phone simulation is an approximation of a telephone channel.
- Public test sets may overlap with model training data.

## Licence
MIT for the code. Datasets and models keep their own licences.
