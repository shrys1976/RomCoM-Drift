# 08 final data handoff report

- **Timestamp (UTC):** 2026-11-01T08:00:02.546619+00:00
- **Code version:** romcom_data 0.1.0
- **Source revision(s):** see manifests/source_manifest.json

## Status

**BLOCKED**

## Primary task dataset

None used for training package construction.

## Dakshina inventory

{'native_words': 24999, 'variants': 44202, 'usable_aliases': 38682, 'ambiguous_aliases': 2563, 'english_collisions': 99, 'canonical_ties': 3337, 'invalid_rows': 2, 'paths': {'inventory': 'data/interim/romanization/dakshina_hi_variant_inventory.tsv', 'alias_index': 'data/interim/romanization/dakshina_hi_alias_index.tsv'}}

## Processed EXP paths

Not created.

## Manifests

- `manifests/source_manifest.json`
- `manifests/checksum_manifest.sha256`
- `manifests/augmentation_manifest.json` (config-only while blocked)

## Acceptance tests

{'exit_code': '0', 'stdout': '............                                                             [100%]\n12 passed in 1.00s\n', 'passed': 'passed'}

## Model scope confirmation

**No model was loaded, validated, evaluated, or trained.**

## Owner decisions required

- Provide official Indi-RomCoM training artifact URL or local files.
- Or approve fallback from IndicXNLI / IndicXParaphrase / IndicAlign with resolved licenses and faithful code-mix procedure.
- Resolve IndicXNLI license metadata vs card conflict.
- Establish IndicXParaphrase license with owner.
