# source audit report

- Timestamp (UTC): 2026-11-01T09:12:43.046143+00:00
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
    "benign_status": "sampled_Dolly_T_train_head_2000",
    "files": [
      "Toxic_Matrix_train.jsonl",
      "Indic_ShareLlama_benign_sample.jsonl"
    ]
  }
}

## Row counts

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

Licenses recorded as owner-approved for intended internal project use. Combined artifacts are not assigned a blanket license.
