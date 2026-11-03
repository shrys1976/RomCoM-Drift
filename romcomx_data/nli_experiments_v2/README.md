# RomCoM-X NLI experiment datasets (V2)

**Status:** `READY_FOR_OWNER_REVIEW_V2` — corrected EXP-02/EXP-03 generator and rebuilt attested-variant / noise conditions only. EXP-00, EXP-01, and English retention are byte-identical to the V1 snapshot (`romcomx_data/nli_experiments_v1/`). Do not train until owner sign-off.

**Parent:** `dataset_version: v2`, `parent_v1_dir: romcomx_data/nli_experiments_v1`

## 1. Root cause (V1 defect)

In V1, `apply_exp02` and `apply_exp03` called `derive_rng(...)` **inside** the per-token loop with the same `(example_id, field)` salt each time. Every eligible token in a Romanized field therefore reused the **first** Bernoulli draw and the **first** noise draw sequence offset — field-level all-or-nothing EXP-02 selection (~43% of fields with eligible tokens had zero partial mixes) and identical substitution positions across same-length EXP-03 tokens in a field.

V2 initializes **one RNG per field** before iterating eligible tokens (EXP-02) or EXP-02 traces in token-index order (EXP-03), matching `realistic_from_canonical` in `augmentation.py`. `BASE_SEED=42`, `P_VARIANT=0.35`, distance matching, and attested-collision rejection are unchanged.

## 2. V1 vs V2 selection totals

| Metric | V1 (`nli_experiments_v1`) | V2 (`nli_experiments_v2`) |
|--------|---------------------------|---------------------------|
| EXP-02 eligible tokens | 5,857,043 | 5,821,353 |
| EXP-02 selected tokens | 2,047,971 | 2,036,028 |
| Fields with partial selection (train) | 0 | 602,973 |
| Excluded-variant selections in EXP-02 | ~13,869 (audit) | 0 (validated) |

Eligible count drops slightly in V2 because 68 audited `(native, variant)` pairs are removed from pools (native words with empty pools after filtering are ineligible).

## 3. Per-split empirical selection rate

Target: `selected / eligible ∈ [0.33, 0.37]` (nominal `p=0.35`).

| Split | Eligible | Selected | Empirical p |
|-------|----------|----------|-------------|
| train | 5,717,089 | 1,999,519 | 0.349744 |
| dev | 34,512 | 12,123 | 0.351269 |
| test | 69,752 | 24,386 | 0.349610 |

## 4. Field mix (EXP-02, Romanized fields with ≥1 eligible token)

| Split | Untouched | Partial | All selected | Partial fraction |
|-------|-----------|---------|--------------|------------------|
| train | 108,412 | 602,973 | 26,879 | 0.817 |
| dev | 722 | 4,019 | 155 | 0.821 |
| test | 1,428 | 8,096 | 355 | 0.820 |

V1 had **zero** partial fields on all splits (generator bug).

## 5. Exclusion inventory

- Authoritative JSON: `romcomx_data/inventories/dakshina_hi_exp02_variant_exclusions.json`
- Copy under this tree: `reports/dakshina_hi_exp02_variant_exclusions.json`
- **68 excluded** mappings (26 wrong-word + 42 garbled, per owner follow-up audit 2026-11-03)
- **5 quarantine** mappings (still selectable; not treated as genuine attested variants in analysis)
- Dakshina TSV inventory is **not** edited; filtering is at pool build time only
- Manifest: `excluded_variant_inventory_sha256` = `c15edd730f451d71c81d8aa8c1c9de2cf06c359ecb4a04a105b4dc4bb264af9f`

## 6. Alignment and row counts

Same conditionally approved EN–HI alignment as V1: **383,059** examples (train 375,565; dev 2,487; test 5,007). `alignment_manifest.json` is copied from V1 unchanged (`alignment_manifest_sha256` matches V1). IDs, labels, and scopes align across EXP-00–03 and English retention.

## 7. EXP-03 collisions and substitution positions

Validation (`validation_report.json`):

- Attested-collision resamples (expected rare): **454** collision pairs
- Same-field same-length same-distance token pairs: **410,201** total; **146,867** share identical substitution index tuples (~35.8%, not ~100% as in V1)
- Independent per-token RNG simulation on dev+test (and train sample) confirms V1-style ~100% identity is **not** reproduced

## 8. Frozen files (V2 = V1 bytes)

SHA256 of EXP-00, EXP-01 (`both`/`premise_only`/`hypothesis_only`), and `english_retention` JSONL in V2 match `experiment_manifest.json` → `frozen_from_v1_sha256` and on-disk V1 snapshot files. Build path `prepare_v2_tree_from_v1()` hardlinks/copies these; V2 build does not truncate or rewrite them.

## 9. Tests and validation

```bash
uv run python scripts/build_nli_experiments.py    # V2 partial rebuild
uv run python scripts/validate_nli_experiments.py # PASS, 0 issues
uv run pytest tests/test_nli_experiments.py tests/test_xnli_alignment.py -q
```

Unit tests cover token-level partial EXP-02 selection, determinism, non-identical EXP-03 positions, excluded variants absent from pools, and quarantine variants still present.

Post-build validation: **2,036,028 / 2,036,028** EXP-02/03 trace checks OK; **0** excluded mapping selections; frozen-hash checks pass.

## 10. Directory tree and checksums

```
nli_experiments_v2/
  README.md
  alignment_manifest.json          # frozen from V1
  experiment_manifest.json         # v2 metadata + evaluation_matrix
  transformation_statistics.json
  construction_status.json         # READY_FOR_OWNER_REVIEW_V2
  checksums.sha256
  validation_report.md / .json
  exp00_reference/                 # frozen from V1 (dev/test)
  exp01/                           # frozen from V1
  exp02/ exp03/                    # rebuilt
  english_retention/               # frozen from V1
  logs/                            # transformation traces (rebuilt)
  reports/
    dakshina_hi_exp02_variant_exclusions.json
    devanagari_codepoint_audit.json
    oov_inventory.jsonl
```

Large `*.jsonl` splits are gitignored; manifests, reports, and checksums are versioned. Regenerate checksums after manifest/report edits:

```bash
uv run python -c "import sys; sys.path.insert(0,'src'); from romcom_data.nli_exp_build import write_nli_experiment_checksums; write_nli_experiment_checksums()"
```

## Reproduction note

Default paths in `src/romcom_data/nli_exp_paths.py` point at **`nli_experiments_v2`**. V1 remains at `romcomx_data/nli_experiments_v1/` (snapshot of original `nli_experiments/`). Never write to V1.
