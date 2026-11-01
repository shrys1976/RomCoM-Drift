# NLI alignment investigation report

{
  "generation_pathway": {
    "repo": "divyanshuaggarwal/IndicXNLI",
    "script": "src/make_dataset.py",
    "datasets_version_pinned_in_repo": "1.16.1",
    "english_input": "load_dataset('xnli','en') unpinned at generation time",
    "indic_fields_saved": [
      "premise",
      "hypothesis",
      "label"
    ],
    "pair_id_persisted_in_indic_release": false
  },
  "backward_train_available": false,
  "generation_snapshot_available": true,
  "train_mapping_method_attempted": "generation_snapshot_rejected_same_as_current_hf",
  "indic_train_vs_xnli_mt_same_index_label_rate": 0.3440165825485992,
  "root_cause": "IndicXNLI train was built by translating whatever row order Hugging Face returned for XNLI en train at generation time. Current facebook/xnli train matches official XNLI-MT TSV at every index (composite keys 100%), but Indic forward train matches that order on labels at only 34.402% of rows. Index joins to current HF/XNLI-MT train are invalid for English alignment.",
  "owner_next_steps": [
    "Provide Divyanshu/indicxnli backward train JSON locally if recovered from git LFS.",
    "Or export generation-era English with scripts/export_nli_generation_snapshot.py (datasets==1.16.1) into romcomx_data/raw/indicxnli/snapshots/xnli_en_train_generation.jsonl."
  ],
  "hf_train_vs_xnli_mt_tsv": {
    "n_compared": 392702,
    "label_agreement": 392702,
    "label_agreement_rate": 1.0,
    "exact_premise_agreement": 392702,
    "composite_key_agreement_rate": 1.0
  },
  "hf_dev_vs_xnli10_dev_label_rate": 1.0,
  "indic_dev_vs_xnli10_same_index_label_rate": 1.0,
  "mt_train_duplicate_composite_keys": 112,
  "train_mapping_stats": {
    "rejected_current_hf_snapshot": 392702,
    "label_agreement_if_used": 0.3440165825485992
  },
  "dev_mapping_stats": {
    "unique_match": 2490
  },
  "train_mapping_reliable": false,
  "normalization": "NFC + outer strip; composite_key=sha256(prem\\0hyp)",
  "owner_review_sample_rows": 200,
  "owner_review_sample_note": "41 dev rows include verified English; 159 train rows are Hindi-only (English withheld until train provenance is established).",
  "datasets_1_16_1_reload_label_agreement_with_indic_train": 0.3440165825485992,
  "datasets_1_16_1_note": "Re-loading xnli en train with datasets==1.16.1 today yields same row order as current HF; does not restore 2021 translation input."
}
