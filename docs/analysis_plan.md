# Analysis Plan

Written before any larger run.

1. **Primary result:** corpus-level normalised WER over all succeeded clips, per language × condition.
2. **Primary question:** does WER rise from `clean` to `phone` and to `phone_noise_20` / `10` / `5`? Reported as paired differences with 95% paired bootstrap intervals.
3. **Secondary metrics:** `wer_lenient` (`hi`), folded metrics (`gu`), and "excluding flagged clips" columns. Flag thresholds (`0.8`, `1.5`, `210` tokens) are fixed now and will not change after the larger run.
4. **Caveat for every report:** excluding `early_stop` / `runaway` clips can remove failures that the noise itself caused, so those numbers can understate degradation. Always show primary next to them.
5. **Run environment:** official runs use one device and compute type, recorded in the report.
6. **Primary and test discipline:** no clip is removed from primary scores, and no decoding change is made after seeing test results.
7. **Pilot scope:** pilot results (30 clips per language) are exploratory.

