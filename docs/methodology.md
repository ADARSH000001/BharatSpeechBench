# Methodology

Decisions marked OPEN are not settled yet. Fill them in before quoting any number.

## Experiment levels
1. `smoke test`: 3 clips, pipeline check only; not a benchmark result.
2. `pilot`: 30 test clips per language plus a separate 10-clip dev set per language; small sample with wide confidence intervals.
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
- **Synthetic noise at fixed SNR is not a real noisy recording.** Real noisy recordings are a separate, future condition; never refer to synthetic noise as a real recording.
- Rule: **one noise file per run; noise type is part of the run id** (e.g. `pilot-whisper-small-straffic-v1`).

## Normalisation (applied to reference and hypothesis)
Unicode NFC; Devanagari and Gujarati digits to ASCII; punctuation incl. danda removed;
Latin text lowercased; whitespace collapsed. Raw (unnormalised) scores are always reported too.
Normalisation is tuned on the `dev` split only.
- **Lenient Hindi metric (secondary, `hi` only)**: `wer_lenient` / `cer_lenient` apply `normalize()`,
  then NFC, delete nukta (`U+093C`, which also covers precomposed nukta consonants because NFC
  decomposes them), and map chandrabindu (`U+0901`) to anusvara (`U+0902`). Vowel lengths and
  presence/absence of anusvara are left untouched. Primary WER stays first and unchanged.

## Protocol (what is sent to each model)
- Every model gets the same 16 kHz mono PCM16 file per (clip, condition).
- Language is forced, not auto-detected: `hi` -> hi, `gu` -> gu, `hi-en` (Hinglish) -> hi.
  Forcing Hindi for Hinglish is a choice: Whisper may answer in Devanagari or Roman script.
- faster-whisper: beam size 5, temperature 0, no initial prompt, no conditioning on previous text.
  Device and compute type (float16 on GPU, int8 on CPU) are recorded in `config.json`; they can change scores.
- Token-cap behaviour: Whisper's decoder limits generation to 224 tokens per 30 s window (`max_length = 448`). Because Whisper's byte-level BPE splits multi-byte Indic scripts (Hindi/Devanagari, Gujarati) into many tokens per word, long uninterrupted utterances can hit the 224-token cap mid-word without emitting an end-of-text or closing timestamp token. When that happens, `faster-whisper` fills in the full window/clip duration as the segment end time, so `coverage` (`last_segment_end_s / audio_seconds`) stays `1.00` and does **not** detect token-cap stops.
- Decoding experiments (temperature fallback, VAD, chunking) are judged on `dev` clips only; the `test` split is used once per final configuration. Any decoding change gets a new run id and is reported next to the old one. Candidate decoding settings are chosen on `dev` by anomaly/truncation rate (`early_stop` / `runaway`; ties go to the simplest variant), **not** by WER.
- Output quality flags (constants in `src/bharatspeechbench/benchmark/reporting.py`, computed on succeeded clips; clips are **never** excluded from primary WER/CER scores):
  - `early_stop`: normalised hypothesis/reference length ratio `< 0.8` (`EARLY_STOP_RATIO_THRESHOLD = 0.8`).
  - `runaway`: normalised hypothesis/reference length ratio `> 1.5` (`RUNAWAY_RATIO_THRESHOLD = 1.5`).
  - Length-ratio basis by language: `hi` (and `hi-en`) use normalised **character** count, whereas `gu`
    uses normalised **word** count with the same `< 0.8` / `> 1.5` thresholds. Why: Whisper frequently
    outputs Devanagari or Latin script for Gujarati audio while references use Gujarati script;
    character-length ratios across scripts reflect encoding/spelling length differences rather than
    truncation or repetition, whereas word counts stay comparable across scripts.
  - `near_token_cap`: `max_segment_tokens >= 210` (`NEAR_TOKEN_CAP_THRESHOLD = 210`). Informational only (not a failure by itself and not listed in `Flagged clips` unless also flagged by ratio).
  `coverage` and `low coverage` (`coverage < 0.85`) are kept as extra columns to catch early timestamp stops, with the caveat above that `coverage` does not catch token-cap stops.
- Secondary sensitivity checks (never replace primary WER): `wer_median` (median per-clip WER) and `wer_excl_flagged` (corpus WER after removing any clip flagged `early_stop` or `runaway` in **any** condition of that run, applied identically across every condition alongside `n_kept` and an analogous exclusion paired table).
- One un-timed warm-up call runs first so model/GPU start-up is not counted in RTF.
- A clip that raises an error is stored with the error, excluded from scores and counted in `n_failed`.
  It is never scored as an empty transcript.

## Metrics
WER and CER are corpus-level (total errors / total reference units), with a bootstrap 95% interval.
RTF (processing time / audio time, median over clips) is compared only across local models on the same hardware.
Number accuracy = matched digit sequences / digit sequences in the reference (see `metrics/numbers.py`).
It only compares digits, so a model that writes a number as words is scored wrong; write references with digits.
It is reported only if the reference text contains numbers, so record scripted clips that do.

## Script policy
- Primary WER and CER are **script-sensitive on purpose**: writing Gujarati in Devanagari or Latin script counts as errors against the Gujarati reference script.
- `script_match` (`(letters in expected script) / (all letters)`, Devanagari for `hi`, Gujarati for `gu`, skipped/`n/a` for `hi-en`; ignoring digits and punctuation) and `dominant_script` are secondary diagnostic signals.
- For `gu` only, `wer_folded` / `cer_folded` provide a secondary folded metric (`approximate: ignores script, not spelling conventions`) by mapping Devanagari letters, vowel signs, anusvara, chandrabindu, visarga, nukta, avagraha, and virama in the hypothesis to matching Gujarati codepoints (`+0x180`); unmapped Devanagari characters (and all Latin text) remain unchanged, and Latin output remains an error.

## Paired comparisons
Comparisons between conditions are strictly paired on identical clips that succeeded in both conditions.
The paired 95% bootstrap confidence interval resamples clip indices jointly across both conditions,
measuring the true paired degradation (WER_cond - WER_clean) while factoring out inter-clip variance.
If fewer than 5 paired clips exist, intervals are not computed ("too few clips for an interval").

## Reproducibility and provenance
Every run stores `bsb env --seed N --manifest FILE` output: Python and package versions, OS, CPU,
RAM, GPU, ffmpeg version, git commit, git dirty status, seed, and the manifest checksum. Audio checksums live in the manifest.
`config.json` records complete settings, model revision, and parameters.
- `git_dirty`: Uses `git status --porcelain --untracked-files=no` so only modified or staged **tracked** files mark a run dirty (`true`), while untracked files (such as local runs or scratch notes) are ignored (`false`; `null` if git is unavailable). Reports display a warning whenever a run is executed with uncommitted tracked changes. Git dirty status is strictly recorded for provenance and does not block run resumption.
- `model_revision`: Captures the exact commit hash of the model weights from the local cache. If the commit hash cannot be resolved, `model_revision` is set to `null` and a descriptive `model_revision_reason` is stored explaining why (e.g. cache directory missing or cache layout unrecognized).
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

- **DEMAND** (Diverse Environments Multichannel Acoustic Noise Database):
  - Source: Zenodo record `1227121` (`https://zenodo.org/records/1227121`, DOI `10.5281/zenodo.1227121`, Thiemann et al., 2013).
  - Official Licence: `"This work, the audio data and the document describing it, is licensed under a Creative Commons Attribution-ShareAlike 3.0 Unported License."` (`CC BY-SA 3.0`, per `DEMAND.pdf` and Zenodo description; Zenodo metadata tag lists `cc-by-4.0`). Audio files stay git-ignored under `data/downloads/noise/` and `data/noise/*.wav`.
  - Environments prepared: `STRAFFIC` (street traffic intersection), `PCAFETER` (office cafeteria/crowd), `DKITCHEN` (indoor kitchen). Each prepared to 120 s 16 kHz mono PCM16 via `scripts/prepare_noise.py`, with hashes and provenance tracked in `data/noise/SOURCES.md`.
  - Rule: one noise file per run; noise type is part of the run id. Synthetic noise mixed at fixed SNR is not a real noisy recording.

## Models
Record the exact model size and revision with every result. For IndicConformer, follow the official
AI4Bharat model card and pin the commit hash if loading uses `trust_remote_code=True`.

## OPEN
- Hinglish reference script (Devanagari vs Roman) and how model output is mapped to it.
- Hinglish and Latin-script references: transliteration policy undecided.
- English words inside Hindi references are written in Latin script in FLEURS raw transcripts, while Whisper often outputs Devanagari. Decide the policy before quoting numbers.
- Whisper model size; IndicConformer revision.
- Number accuracy: needs clips that contain numbers (record scripted ones).
- Public test sets may overlap with model training data; report own recordings separately.
