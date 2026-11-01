# source audit report

- Timestamp (UTC): 2026-11-01T11:11:55.375995+00:00
- Code version: romcom_data 0.1.0

## Acquired sources

{
  "nli": {
    "repository": "Divyanshu/indicxnli",
    "revision": "7092c27872e919f31d0496fb8b9c47bd2cba3f6c",
    "license": "CC-BY-4.0",
    "license_status": "cleared_for_internal_use",
    "files": [
      "backward__dev__xnli_hi.json",
      "backward__test__xnli_hi.json",
      "forward__dev__xnli_hi.json",
      "forward__test__xnli_hi.json",
      "forward__train__xnli_hi.json"
    ],
    "all_hi_files": [
      "backward/dev/xnli_hi.json",
      "backward/test/xnli_hi.json",
      "forward/dev/xnli_hi.json",
      "forward/test/xnli_hi.json",
      "forward/train/xnli_hi.json"
    ]
  },
  "paraphrase": {
    "repository": "ai4bharat/IndicXParaphrase",
    "revision": "181031f9e69c465e0c72fc84fd33b66114adb3e8",
    "license": "CC0",
    "license_status": "cleared_for_internal_use",
    "files": [
      "hi.tsv"
    ],
    "split_semantics": "Single language TSV with no official train/dev/test files. Not a test split; project 80/20 is created at source-group level."
  },
  "toxicity": {
    "repository": "ai4bharat/indic-align",
    "revision": "032b6a9070e7f85f1a38e0506419f4590a20455a",
    "license": "CC-BY-4.0",
    "license_status": "cleared_for_internal_use",
    "toxic_config": "Toxic_Matrix",
    "benign_config": "Dolly_T",
    "n_toxic_raw": 5000,
    "n_benign_sampled": 2000,
    "benign_status": "reused",
    "files": [
      "Toxic_Matrix_train.jsonl",
      "Indic_ShareLlama_benign_sample.jsonl"
    ]
  }
}

## Row counts

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

Licenses recorded as owner-approved for intended internal project use. Combined artifacts are not assigned a blanket license.
