# split integrity report

- Timestamp (UTC): 2026-11-01T11:11:55.376183+00:00
- Code version: romcom_data 0.1.0

## Policy

- IndicXNLI: official train → project train; official dev → project validation; **test excluded**.
- IndicXParaphrase: no official split files; 80/20 connected-component split, seed 42.
- Toxicity: 80/20 by source_id, seed 42, stratified by toxic/non_toxic.
- Related NLI hypotheses sharing a premise stay together.
- All mixing intensities of one source stay together.

## Integrity

{
  "nli_train_sources": 12000,
  "nli_val_sources": 2490,
  "nli_overlap": 0,
  "para_overlap": 0,
  "tox_overlap": 0,
  "protected_test_leakage": {
    "leaked_source_ids": [],
    "leaked_hashes": [],
    "ok": true
  },
  "quarantined_nli": 0,
  "quarantined_paraphrase": 0,
  "quarantined_toxicity": 0
}
