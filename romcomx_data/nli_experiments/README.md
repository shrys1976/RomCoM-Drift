# RomCoM-X NLI experiment datasets

Frozen EN–HI XNLI pairs under **conditional approval** (`romcomx_data/nli_alignment/`) are materialized here as EXP-00 through EXP-03 plus an English retention evaluation benchmark. Status: **READY_FOR_OWNER_REVIEW** (not approved for training until owner sign-off).

## Layout

```
nli_experiments/
  alignment_manifest.json      # Approved row counts and exclusion audit trail
  experiment_manifest.json     # Conditions, row_counts, evaluation_matrix (22 dev/test sets)
  transformation_statistics.json
  construction_status.json
  checksums.sha256
  validation_report.md
  exp00_reference/             # Native Hindi reference (dev/test only)
  exp01/ exp02/ exp03/         # Romanization conditions
    both/                      # train + dev + test
    premise_only/              # dev + test only
    hypothesis_only/           # dev + test only
  english_retention/           # Original English text (dev + test only)
  logs/                        # Per-row transformation traces (not for training)
  reports/
    devanagari_codepoint_audit.json
    oov_inventory.jsonl
```

Large `*.jsonl` splits are gitignored; manifests and reports are versioned.

## Approved alignment counts

| Split | Rows |
|-------|-----:|
| train | 375,565 |
| dev   | 2,487 |
| test  | 5,007 |
| **total** | **383,059** |

## Experiment conditions

| ID | Training use | Hindi surface |
|----|--------------|---------------|
| **EXP-00** | None (evaluation reference) | Native Devanagari premise and hypothesis |
| **EXP-01** | Train on `both` scope only | Dakshina canonical romanization (both fields Roman when scope is `both`) |
| **EXP-02** | Same as EXP-01 | Attested noncanonical Dakshina variants at eligible Hindi-origin tokens (seed 42, p=0.35, Levenshtein d ≥ 1) |
| **EXP-03** | Same as EXP-01 | Random noise on EXP-01 text at the same token positions and distances as EXP-02 (control) |

**Input scopes**

- **train:** `both` only (fully Romanized Hindi in premise and hypothesis for EXP-01/02/03).
- **dev / test:** `both`, `premise_only` (Roman premise, native hypothesis), `hypothesis_only` (native premise, Roman hypothesis).

**English retention** (`english_retention/`): unchanged English premise and hypothesis aligned to the same `example_id`s. Dev and test only — an untrained English baseline benchmark, not a code-mixed training condition.

## Record schema (EXP-01/02/03 and EXP-00)

Each JSONL row includes:

- `example_id`, `split`, `condition`, `scope`
- `premise`, `hypothesis`
- `premise_script`, `hypothesis_script` (`roman` | `devanagari`)
- `nli_label`, `label_id` (entailment=0, neutral=1, contradiction=2)
- `source` (English/Hindi row indexes, pair id, sources, matching method)
- `alignment_manifest_sha256`

English retention rows add `benchmark: "english_retention"` and omit `condition` / script fields.

## EXP-00 untrained evaluation matrix

No training is defined for EXP-00. An untrained base model may be evaluated on **22** dev/test files (11 conditions/scopes × 2 splits), listed in `experiment_manifest.json` → `evaluation_matrix.EXP-00_untrained_base.evaluation_sets`. Paths point at the generated JSONL files under this directory (no duplicate copies).

## Reproduction

From the repository root (requires approved `verified_pairs.jsonl` and Dakshina inventory):

```bash
# Pre-build: Devanagari codepoint coverage on approved Hindi text
uv run python scripts/audit_nli_devanagari_codepoints.py

# Full construction (~383k rows; long-running)
uv run python scripts/build_nli_experiments.py

# Refresh manifest row counts / checksums without rebuilding JSONL
uv run python scripts/refresh_nli_experiment_manifest.py

# Post-build validation + validation_report.md
uv run python scripts/validate_nli_experiments.py

uv run pytest tests/test_nli_experiments.py tests/test_xnli_alignment.py -q
```

Configuration: `configs/data_augmentation.yaml` (seed 42, variant probability 0.35). Lexicon: `romcomx_data/inventories/dakshina_hi_variant_inventory.tsv`.

## Notes

- Romanized fields must contain **no residual Devanagari**; any fallback failure blocks construction.
- OOV / fallback tokens are listed in `reports/oov_inventory.jsonl` with aggregate rates in `transformation_statistics.json`.
- Transformation detail per row lives in `logs/transformation_log.{split}.jsonl` (EXP-02/03 native spans and distances).
