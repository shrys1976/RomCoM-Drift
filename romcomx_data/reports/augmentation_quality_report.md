# augmentation quality report

- Timestamp (UTC): 2026-11-01T11:11:55.377252+00:00
- Code version: romcom_data 0.1.0

## Conditions

Same source IDs, splits, labels, and row counts across EXP-01/02/03.

- EXP-01: Dakshina canonical; tied top-frequency variants use lexicographic fallback (reproducibility only).
- EXP-02: p_variant=0.35 attestation-weighted non-canonical Dakshina variants from EXP-01.
- EXP-03: reuses EXP-02 token mask; per-token Levenshtein distance matched to EXP-02 selection.

## Noise stats

{
  "per_token_distance_match_rate": 1.0,
  "tokens_evaluated": 73194,
  "failed_resample_attempts": 941
}

## QA

{
  "train.jsonl": {
    "n_EXP-01": 16800,
    "n_EXP-02": 16800,
    "n_EXP-03": 16800,
    "ids_match": true,
    "labels_match": true,
    "source_ids_and_splits_match": true
  },
  "validation.jsonl": {
    "n_EXP-01": 4200,
    "n_EXP-02": 4200,
    "n_EXP-03": 4200,
    "ids_match": true,
    "labels_match": true,
    "source_ids_and_splits_match": true
  }
}
