# Romanization integrity (audit)

## Method (both sentences treated independently)

1. **Tokenization:** `tokenize_conservative` per sentence (Latin words passthrough; Devanagari tokens romanized).
2. **Dakshina hit:** Lookup native token in `dakshina_hi_variant_inventory.tsv`; canonical variant = highest attestation with tie-break policy from `pick_canonical_variant`.
3. **OOV / no lexicon hit:** Devanagari tokens use `fallback_indic_map` (documented character map in `romanize.py`, not attested Dakshina).
4. **Non-Devanagari:** Punctuation and non-Latin non-Devanagari pass through unchanged.
5. **Labels:** Romanization is applied to input text only; `label` / `normalized_label` are never modified.

## Collapse checks (post-normalization NFC on native; romanized output concatenated)

- **Positive collapse:** `validate_reconstructed_pair` flags positives where romanized s1 == romanized s2 but native s1 != native s2.
- **Negative false-identical:** negatives where romanized forms match while native differ.

Counts are in `integrity_metrics.json` under `romanization`. No rows are silently dropped during audit.
