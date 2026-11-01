# final data handoff report

- Timestamp (UTC): 2026-11-01T11:11:55.377409+00:00
- Code version: romcom_data 0.1.0

## Status

**BLOCKED** (Phase 2.1)

Block reason: `nli_alignment_pending_owner_review`

NLI alignment approved: False

## Primary task sources (reconstructed Indi-RomCoM methodology)

- NLI: Divyanshu/indicxnli (owner-approved CC BY 4.0)
- Paraphrase: omitted unless independent English pivots exist (see `paraphrase_source_semantics.md`)
- Toxicity: ai4bharat/indic-align Toxic_Matrix (owner-approved CC BY 4.0)

These are **not** the official Indi-RomCoM training release.

## Counts

{
  "nli_source_eligible": 14490,
  "nli_included_in_processed": false,
  "paraphrase_source_eligible": 2002,
  "paraphrase_reconstruction_kept": 0,
  "paraphrase_excluded": 2002,
  "toxicity_source": 7000,
  "nli_base": 0,
  "paraphrase_base": 0,
  "toxicity_base": 21000,
  "paraphrase_build_stats": {},
  "stream_counts": {
    "EXP-01:validation:toxicity": 4200,
    "EXP-02:validation:toxicity": 4200,
    "EXP-03:validation:toxicity": 4200,
    "EXP-01:train:toxicity": 16800,
    "EXP-02:train:toxicity": 16800,
    "EXP-03:train:toxicity": 16800
  },
  "qa": {
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
  },
  "noise_stats": {
    "per_token_distance_match_rate": 1.0,
    "tokens_evaluated": 73194,
    "failed_resample_attempts": 941
  }
}

## Paths

- Processed: `romcomx_data/processed/EXP-01_canonical|EXP-02_realistic|EXP-03_matched_noise/`
- Manifests: `romcomx_data/manifests/`
- Reports: `romcomx_data/reports/`
- Traces: `romcomx_data/traces/`

## Owner notes

- Toxicity input is not code-mixed; do not treat target-side EXP variation as input-robustness evidence.
- EXP-02 is Dakshina-informed, not a social-media frequency model.

## Model scope

**No model was loaded, validated, evaluated, or trained.**
