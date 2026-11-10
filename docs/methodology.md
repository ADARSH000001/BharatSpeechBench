# Methodology

Decisions marked OPEN are not settled yet. Fill them in before quoting any number.

## Experiment levels
1. `smoke test`: 3 clips, pipeline check only; not a benchmark result.
2. `pilot`: ~30 clips per language (e.g. 10 dev, 20 test); small sample with wide confidence intervals.
3. `expanded`: many speakers across dialects and acoustic conditions for production comparisons.

## Conditions
1. `clean`: original audio converted to 16 kHz mono PCM16.
2. `phone`: `simulate_phone` (300-3400 Hz band-pass, 8 kHz, G.711 mu-law). A codec/band-limit
   approximation only: no packet loss, handset or room effects.
3. `phone_noise_<N>`: noise mixed at N dB SNR (default 20 / 10 / 5) with a fixed seed, THEN the phone
   chain (noise passes through the channel). The per-clip seed is derived from `--seed` and the clip id
   (CRC32), so the same command gives the same audio on any machine. The noise file's SHA-256 is stored in
   `results/raw/<run_id>/config.json`; generated audio is cached under a folder keyed by seed + noise hash.
Originals are never modified. Listen to the generated files (`bsb simulate`) before trusting any number.

## Synthetic conditions
All degraded conditions are SYNTHETIC:
- `phone` is a software codec simulation (band-pass 300–3400 Hz, 8 kHz downsampling, G.711 mu-law), not an acoustic telephone channel.
- `phone_noise_<N>` mixes noise at calibrated SNR before passing through the simulated codec.
- Real noisy recordings are a separate, future condition. Never refer to synthetic noise as a real recording.

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

## Paired comparisons
Comparisons between conditions are strictly paired on identical clips that succeeded in both conditions.
The paired 95% bootstrap confidence interval resamples clip indices jointly across both conditions,
measuring the true paired degradation (WER_cond - WER_clean) while factoring out inter-clip variance.
If fewer than 5 paired clips exist, intervals are not computed ("too few clips for an interval").

## Reproducibility and provenance
Every run stores `bsb env --seed N --manifest FILE` output: Python and package versions, OS, CPU,
RAM, GPU, ffmpeg version, git commit, seed, and the manifest checksum. Audio checksums live in the manifest.
`config.json` records complete settings, model revision, and parameters.
Speaker-disjoint rule: dev and test splits must remain strictly speaker-disjoint (`speaker_id` must not appear in both splits).

## Datasets and licences
For each source record: name, version/commit, licence copied from the OFFICIAL dataset card (not a
mirror), clips used, split. Own recordings: speaker consent, device, room, date, script read.

- **Google FLEURS**:
  - Source: Hugging Face dataset `google/fleurs` (Conneau et al., 2022).
  - Official Licence: `cc-by-4.0` (Creative Commons Attribution 4.0 International, copied directly from the official dataset card metadata).
  - Languages used: Hindi (`hi_in`) and Gujarati (`gu_in`).
  - Raw-transcript decision: The benchmark uses FLEURS' `raw_transcription` column (containing original punctuation and unnormalized text) as reference rather than pre-normalized text. Normalization is handled uniformly by BharatSpeechBench's `normalize()` pipeline.
  - Speaker ID note: FLEURS metadata does not include speaker identifiers; `speaker_id` remains null.
  - Provenance: Manifest generation computes real audio durations and SHA-256 checksums on downloaded clips.

## Models
Record the exact model size and revision with every result. For IndicConformer, follow the official
AI4Bharat model card and pin the commit hash if loading uses `trust_remote_code=True`.

## OPEN
- Hinglish reference script (Devanagari vs Roman) and how model output is mapped to it.
- Whisper model size; IndicConformer revision.
- Number accuracy: needs clips that contain numbers (record scripted ones).
- Noise corpus and its licence (MUSAN, DEMAND, or your own recording).
- Public test sets may overlap with model training data; report own recordings separately.
