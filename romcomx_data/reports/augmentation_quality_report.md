# augmentation quality report

- Timestamp (UTC): 2026-11-01T09:12:43.046676+00:00
- Code version: romcom_data 0.1.0

## Conditions

Same source IDs, splits, labels, and row counts across EXP-01/02/03.

- EXP-01: Dakshina unique max-count canonical (ties broken lexicographically for inventory; tokens with remaining multi-way surface ties left unchanged in text if flagged).
- EXP-02: p_variant=0.35 attestation-weighted non-canonical Dakshina variants from EXP-01.
- EXP-03: random Latin edits at the same EXP-02 token positions, matched edit-op count.

## QA

{
  "train.jsonl": {
    "n_EXP-01": 57606,
    "n_EXP-02": 57606,
    "n_EXP-03": 57606,
    "ids_match": true,
    "labels_match": true
  },
  "validation.jsonl": {
    "n_EXP-01": 12870,
    "n_EXP-02": 12870,
    "n_EXP-03": 12870,
    "ids_match": true,
    "labels_match": true
  }
}
