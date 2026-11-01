# final data handoff report

- Timestamp (UTC): 2026-11-01T09:12:43.046811+00:00
- Code version: romcom_data 0.1.0

## Status

**READY_FOR_OWNER_REVIEW**

## Primary task sources (reconstructed Indi-RomCoM methodology)

- NLI: Divyanshu/indicxnli (owner-approved CC BY 4.0)
- Paraphrase: ai4bharat/IndicXParaphrase (owner-approved CC0)
- Toxicity: ai4bharat/indic-align Toxic_Matrix (owner-approved CC BY 4.0)

These are **not** the official Indi-RomCoM training release.

## Counts

{
  "nli_source": 14490,
  "paraphrase_source": 2002,
  "toxicity_source": 7000,
  "nli_base": 43470,
  "paraphrase_base": 6006,
  "toxicity_base": 21000,
  "stream_counts": {
    "EXP-01:validation:nli": 7470,
    "EXP-02:validation:nli": 7470,
    "EXP-03:validation:nli": 7470,
    "EXP-01:train:nli": 36000,
    "EXP-02:train:nli": 36000,
    "EXP-03:train:nli": 36000,
    "EXP-01:validation:paraphrase": 1200,
    "EXP-02:validation:paraphrase": 1200,
    "EXP-03:validation:paraphrase": 1200,
    "EXP-01:train:paraphrase": 4806,
    "EXP-02:train:paraphrase": 4806,
    "EXP-03:train:paraphrase": 4806,
    "EXP-01:validation:toxicity": 4200,
    "EXP-02:validation:toxicity": 4200,
    "EXP-03:validation:toxicity": 4200,
    "EXP-01:train:toxicity": 16800,
    "EXP-02:train:toxicity": 16800,
    "EXP-03:train:toxicity": 16800
  },
  "qa": {
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
