# indi romcom reconstruction report

- Timestamp (UTC): 2026-11-01T09:12:43.044689+00:00
- Code version: romcom_data 0.1.0

## Status

Records are **reconstructed**, not the official Indi-RomCoM 395,076-instance training release.
No author training artifact was found.

## Paper method reproduced

1. Source Hindi task data from IndicXNLI, IndicXParaphrase, IndicAlign Toxic_Matrix.
2. Table 9 taxonomy: CAT-0 never switched; CAT-C at 25/50/75; CAT-B and CAT-D at 50/75; CAT-A and CAT-E at 75.
3. Table 10 intensities 25/50/75 with those category subsets.
4. Romanization via Dakshina train lexicon canonical forms; unknown native tokens use a documented character-map fallback (`fallback_indic_map`), not Aksharantar/Claude.
5. Toxicity (Appendix B.5): **input prompt is never code-mixed**. Only target/refusal side is mixed.

## Deviations (explicit)

- Official generation used Claude Sonnet 4.6 for taxonomy-guided substitutions. This reconstruction uses a closed bilingual lexicon plus deterministic category rules. No LLM rewrite.
- CAT-E clause rewriting is approximated by an Indic-frame romanization of the parallel Hindi sentence at 75%.
- CAT-D tokens without a lexicon entry remain English (`no_lexicon`).
- Dakshina attestation-weighted EXP-02 is **not** claimed as social-media typing frequency.

## English join

NLI English premises/hypotheses are attached from `facebook/xnli` by split + row index when missing from IndicXNLI.
