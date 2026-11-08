# Methodology

Decisions marked OPEN are not settled yet. Fill them in before quoting any number.

## Conditions
1. `clean`: original audio converted to 16 kHz mono PCM16.
2. `phone`: `simulate_phone` (300-3400 Hz band-pass, 8 kHz, G.711 mu-law). A codec/band-limit
   approximation only: no packet loss, handset or room effects.
3. `phone_noise_<N>`: noise mixed at N dB SNR (default 20 / 10 / 5) with a fixed seed, THEN the phone
   chain (noise passes through the channel). The per-clip seed is derived from `--seed` and the clip id
   (CRC32), so the same command gives the same audio on any machine. The noise file's SHA-256 is stored in
   `results/raw/<run_id>/config.json`; generated audio is cached under a folder keyed by seed + noise hash.
Originals are never modified. Listen to the generated files (`bsb simulate`) before trusting any number.

## Normalisation (applied to reference and hypothesis)
Unicode NFC; Devanagari and Gujarati digits to ASCII; punctuation incl. danda removed;
Latin text lowercased; whitespace collapsed. Raw (unnormalised) scores are always reported too.
Normalisation is tuned on the `dev` split only.

## Protocol (what is sent to each model)
- Every model gets the same 16 kHz mono PCM16 file per (clip, condition).
- Language is forced, not auto-detected: `hi` -> hi, `gu` -> gu, `hi-en` (Hinglish) -> hi.
  Forcing Hindi for Hinglish is a choice: Whisper may answer in Devanagari or Roman script.
- faster-whisper: beam size 5, temperature 0, no initial prompt, no conditioning on previous text.
  Device and compute type (float16 on GPU, int8 on CPU) are recorded in `config.json`; they can change scores.
- One un-timed warm-up call runs first so model/GPU start-up is not counted in RTF.
- A clip that raises an error is stored with the error, excluded from scores and counted in `n_failed`.
  It is never scored as an empty transcript.

## Metrics
WER and CER are corpus-level (total errors / total reference units), with a bootstrap 95% interval.
RTF (processing time / audio time, median over clips) is compared only across local models on the same hardware.
Number accuracy = matched digit sequences / digit sequences in the reference (see `metrics/numbers.py`).
It only compares digits, so a model that writes a number as words is scored wrong; write references with digits.
It is reported only if the reference text contains numbers, so record scripted clips that do.

## Reproducibility
Every run stores `bsb env --seed N --manifest FILE` output: Python and package versions, OS, CPU,
RAM, GPU, ffmpeg version, git commit, seed, and the manifest checksum. Audio checksums live in the manifest.

## Datasets and licences
For each source record: name, version/commit, licence copied from the OFFICIAL dataset card (not a
mirror), clips used, split. Own recordings: speaker consent, device, room, date, script read.

## Models
Record the exact model size and revision with every result. For IndicConformer, follow the official
AI4Bharat model card and pin the commit hash if loading uses `trust_remote_code=True`.

## OPEN
- Hinglish reference script (Devanagari vs Roman) and how model output is mapped to it.
- Whisper model size; IndicConformer revision.
- Number accuracy: needs clips that contain numbers (record scripted ones).
- Noise corpus and its licence (MUSAN, DEMAND, or your own recording).
- Public test sets may overlap with model training data; report own recordings separately.
