# data/

Audio is never committed. Commit the manifest, checksums and instructions to re-create the data.

- `data/manifests/manifest.jsonl`: ONE combined manifest, one clip per line, with a `language` field. You create it.
- `data/manifests/example.jsonl`: schema example only (fake checksums, no audio). Copy a line and fill in your own.
- `data/own_recordings/`, `data/downloads/`, `data/conditions/`: local audio, git-ignored.
  `conditions/` holds generated clean / phone / phone_noise copies (one subfolder per seed + noise file) and can always be re-created with `bsb simulate`.

## Manifest fields
| Field | Meaning |
|---|---|
| `id` | unique clip id |
| `path` | audio path relative to the repo root |
| `language` | `hi`, `gu` or `hi-en` (Hinglish) |
| `source` | e.g. `fleurs`, `kathbath`, `own` |
| `split` | `dev` or `test` (never tune on `test`) |
| `reference` | human-verified transcript |
| `duration_s` | clip length in seconds |
| `sha256` | checksum of the audio file (`bsb hash FILE`) |
| `licence` | licence from the OFFICIAL dataset card, or `own` |
| `speaker_consent` | must be `true` when `source` is `own` |

Check it: `bsb check-manifest data/manifests/manifest.jsonl --verify-audio`
