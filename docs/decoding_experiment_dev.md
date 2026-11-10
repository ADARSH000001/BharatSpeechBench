# Dev Decoding Experiment (`whisper-small`, CPU `int8`)

Created as requested for the `dev`-split decoding comparison.

## Setup
- Manifest: `data/manifests/dev.jsonl` (`split="dev"`, 20 clips: 10 `hi`, 10 `gu`; checksums verified).
- Conditions: `clean`, `phone` (40 clip-condition evaluations per variant; 20 per language).
- Model: `Systran/faster-whisper-small` (`whisper-small`, revision `536b0662742c02347bc0e980a01041f333bce120`), `device=cpu`, `compute_type=int8`, `beam_size=5`, `temperature=0.0`, `condition_on_previous_text=False`, seed `0`.
- Library versions: `faster-whisper 1.2.1`, `ctranslate2 4.8.5`.
- Variants tested:
  - **(A) `default`**: `dev-whisper-small-default` (`vad_filter=False`, `chunk_length=None`, `max_new_tokens=None`)
  - **(B) `vad-filter`**: `dev-whisper-small-vad` (`--vad-filter`)
  - **(C) `chunk-length 15`**: `dev-whisper-small-chunk15` (`--chunk-length 15`)
  - **(D) other window-length limit**: none exists in `WhisperModel.transcribe()` (`chunk_length` is the only parameter limiting encoder window size; `vad_parameters.max_speech_duration_s` splits VAD chunks that `WhisperModel.transcribe()` concatenates back before chunking, `clip_timestamps` takes explicit per-clip timestamps, and `max_new_tokens` only lowers the per-chunk token cap).

## Selection rule (decided before looking at WER)
1. Prefer the variant with the fewest total `n_truncated` clips across `hi` and `gu`.
2. Break ties in favor of the simplest variant (`default` > `--vad-filter` > `--chunk-length 15`).
3. Do not select on WER or CER.

## Comparison (`variant x language`, across `clean` + `phone`)
Values computed directly from `results/processed/dev-whisper-small-*_per_clip.parquet`:

| variant | language | clips | n_truncated | median length ratio | WER | CER |
|---|---|---|---|---|---|---|
| (A) default | gu | 20 | 1 | 0.984 | 117.3% | 92.4% |
| (A) default | hi | 20 | 1 | 0.978 | 47.2% | 20.2% |
| (B) --vad-filter | gu | 20 | 1 | 0.970 | 118.3% | 91.9% |
| (B) --vad-filter | hi | 20 | 1 | 1.000 | 44.4% | 18.5% |
| (C) --chunk-length 15 | gu | 20 | 3 | 1.009 | 131.9% | 100.2% |
| (C) --chunk-length 15 | hi | 20 | 1 | 0.978 | 48.1% | 20.7% |

### Per-condition breakdown (from `results/processed/dev-whisper-small-*_summary.parquet`)

| variant | language | condition | clips | n_truncated | median length ratio | WER | CER |
|---|---|---|---|---|---|---|---|
| (A) default | gu | clean | 10 | 0 | 0.997 | 118.3% | 92.5% |
| (A) default | gu | phone | 10 | 1 | 0.979 | 116.2% | 92.3% |
| (A) default | hi | clean | 10 | 1 | 0.979 | 47.2% | 21.7% |
| (A) default | hi | phone | 10 | 0 | 0.978 | 47.2% | 18.6% |
| (B) --vad-filter | gu | clean | 10 | 0 | 0.983 | 121.5% | 91.9% |
| (B) --vad-filter | gu | phone | 10 | 1 | 0.965 | 115.2% | 92.0% |
| (B) --vad-filter | hi | clean | 10 | 1 | 1.000 | 44.2% | 20.4% |
| (B) --vad-filter | hi | phone | 10 | 0 | 0.993 | 44.6% | 16.7% |
| (C) --chunk-length 15 | gu | clean | 10 | 1 | 1.014 | 147.1% | 99.8% |
| (C) --chunk-length 15 | gu | phone | 10 | 2 | 0.997 | 116.8% | 100.7% |
| (C) --chunk-length 15 | hi | clean | 10 | 1 | 0.979 | 48.5% | 22.2% |
| (C) --chunk-length 15 | hi | phone | 10 | 0 | 0.978 | 47.6% | 19.3% |

## Outcome
- **Selected variant**: **(A) `default`** (ties with `--vad-filter` at `n_truncated = 2` total, wins on simplicity; `--chunk-length 15` increases `n_truncated` to `4` by causing `224`-token repetition loops on `fleurs-gu-88347442456085261`).
- **No variant eliminates truncation**: `n_truncated = 2` persists under both `default` and `--vad-filter` (`fleurs-hi-9195457276131503091` `clean` via normalised length ratio `< 0.8` with `max_segment_tokens` = 107 / 100, and `fleurs-gu-10938797483603452297` `phone` via both length ratio `0.66 < 0.8` and `max_segment_tokens = 224` on a 14.1 s clip).
- Note on Gujarati WER (`>100%`): in both clean and phone conditions, `whisper-small` frequently outputs Latin/Devanagari/corrupted tokens for Gujarati, causing substitutions + insertions to exceed reference word count; `num_matched`/`number_acc` on `gu` `clean` went from `0.0%` to `100.0%` under VAD on the tiny subset of digit-bearing items, unrelated to truncation.

---

## Gujarati Initial-Prompt Experiment (`--languages gu`, `clean` + `phone`, CPU `int8`)
- Manifest: `data/manifests/dev.jsonl` (`split="dev"`, `--languages gu`, 10 clips x 2 conditions = 20 clip-conditions per variant).
- Prior `dev-whisper-small-default` lacked `"initial_prompt": null` in `provider_settings` (and had `languages=null`), so Variant (A) was cleanly rerun with matching settings as `dev-whisper-small-gu-default`.
- Variants:
  - **(A) `default`**: `dev-whisper-small-gu-default` (`--initial-prompt` omitted / `None`)
  - **(E) `initial-prompt`**: `dev-whisper-small-gu-prompt` (`--initial-prompt "આ એક ગુજરાતી વાક્ય છે."`, neutral Gujarati sentence not in FLEURS)
- Pre-registered selection rule: choose the variant with higher `script_match_mean` on `gu`; if the gap is `< 0.1`, choose `(A) default`.
- Folded columns below (`wer_folded` / `cer_folded`) are **approximate: ignores script, not spelling conventions**.

| variant | condition | clips | script_match_mean | n_truncated | WER | CER | wer_folded | cer_folded |
|---|---|---|---|---|---|---|---|---|
| (A) default (`dev-whisper-small-gu-default`) | clean | 10 | 0.0% | 0 | 118.3% | 92.5% | 102.1% | 46.3% |
| (A) default (`dev-whisper-small-gu-default`) | phone | 10 | 0.0% | 1 | 116.2% | 92.3% | 100.5% | 45.7% |
| **(A) default (overall clean+phone)** | **all** | **20** | **0.0%** | **1** | **117.3%** | **92.4%** | **101.3%** | **46.0%** |
| (E) `--initial-prompt` (`dev-whisper-small-gu-prompt`) | clean | 10 | 100.0% | 9 | 160.2% | 82.6% | 160.2% | 82.6% |
| (E) `--initial-prompt` (`dev-whisper-small-gu-prompt`) | phone | 10 | 100.0% | 9 | 152.4% | 84.6% | 152.4% | 84.6% |
| **(E) `--initial-prompt` (overall clean+phone)** | **all** | **20** | **100.0%** | **18** | **156.3%** | **83.6%** | **156.3%** | **83.6%** |

### Outcome & Caveat
- **Rule selection**: **(E) `--initial-prompt`** (`script_match_mean = 1.000` vs `0.000`, gap = `1.000 >= 0.1`).
- **Severe truncation side effect**: Prompting Gujarati script drives `n_truncated` from **1/20** (`5%`) up to **18/20** (`90%`) and increases `wer_folded` from **101.3%** to **156.3%** (RTF roughly doubles from `0.41-0.43` to `0.80-0.81`). Whisper's GPT-2 byte-level BPE splits Gujarati characters into ~2-3 byte tokens per character, hitting the 224-token per-chunk cap on nearly every clip. Defaults remain unchanged (`initial_prompt=None`); `test` and `pilot` splits were not touched.


