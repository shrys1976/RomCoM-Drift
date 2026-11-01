# toxicity data design report

- Timestamp (UTC): 2026-11-01T09:12:43.046535+00:00
- Code version: romcom_data 0.1.0

## Design

The Indi-RomCoM paper (Appendix B.5) keeps the **toxic input prompt unmixed**.
Only classification/refusal **targets** are code-mixed.

## Implications for RomCoM-X

- EXP-01/02/03 for toxicity apply romanization variation to **target-side** reconstructed text.
- Input-side eligible Romanized Hindi coverage is expected to be near zero unless the source prompt already contains it.
- Target-side romanization **must not** be interpreted as evidence of improved Romanized-Hindi **input** understanding.
- Original prompt bytes are preserved. Labels `toxic` / `non_toxic` are preserved.
- Benign rows come from IndicAlign `Indic_ShareLlama` sample for balance, separately identifiable via `is_benign`.
