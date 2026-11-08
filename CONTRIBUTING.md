# Contributing

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
```
`ffmpeg` must be on your PATH.

## Rules
- **No audio in git.** Commit the manifest (with SHA-256 from `bsb hash`) and instructions only. Check a dataset's official licence first.
- **New model = new provider.** Subclass `ASRProvider` in `src/bharatspeechbench/providers/`; record the exact model id and revision.
- **Normalisation changes** must be documented in `docs/methodology.md` and must not be tuned on the test split.
- **Every metric or audio change needs a test.**
- Keep raw model transcripts unedited; fixes belong in scoring, not in stored results.
- Own recordings need the speaker's consent, noted in the manifest.

## Pull requests
Small, focused, with tests passing. Describe what changed and why.
