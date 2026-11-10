# Run `smoke-whisper-small`

- Provider: faster-whisper / model `whisper-small`  settings: {'device': 'cpu', 'compute_type': 'int8', 'beam_size': 5, 'decoding': 'beam search, temperature 0, no initial prompt', 'model_repo': 'Systran/faster-whisper-small', 'model_revision': None, 'ctranslate2_version': '4.8.5', 'faster_whisper_version': '1.2.1'}
- Split: test, conditions: clean, phone
- Git commit: a6dbc6bac883f81df0adcb24cca5be93137e550b, manifest sha256: 1753f985bb7c6a8f5c929b34f7ebb97b5357327ba657187ddfadffd6ea086125
- Seed: 0, GPU: {'name': 'NVIDIA GeForce RTX 4050 Laptop GPU', 'count': 1, 'cuda': '12.4'}

WER/CER are corpus-level (normalised unless marked raw), with a bootstrap 95% interval over clips.
Small clip counts give wide intervals: do not read small gaps as real differences.

| language | condition | clips | failed | empty | S | D | I | WER | WER 95% CI | WER raw | CER | number acc | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hi | clean | 3 | 0 | 0 | 39 | 18 | 2 | 49.2% | 22.7% to 60.7% | 54.6% | 29.0% | n/a | 0.43 |
| hi | phone | 3 | 0 | 0 | 45 | 5 | 2 | 43.3% | 40.5% to 46.4% | 49.6% | 20.6% | n/a | 0.48 |

## Highest-error clips
- `fleurs-hi-18245634017773510582` (clean): 60.7%
- `fleurs-hi-8647640348946233289` (clean): 47.6%
- `fleurs-hi-18245634017773510582` (phone): 46.4%
- `fleurs-hi-1767137162588763048` (phone): 40.9%
- `fleurs-hi-8647640348946233289` (phone): 40.5%

## Paired comparison vs clean

### hi
| condition | clean WER | condition WER | absolute difference | relative degradation | 95% paired CI | n clips |
|---|---|---|---|---|---|---|
| phone | 49.2% | 43.3% | -5.8% | -11.9% | too few clips for an interval | 3 |

Limits: see docs/methodology.md (phone channel is a simulation; Hinglish script policy; public sets may overlap with training data; number accuracy compares digits only).
