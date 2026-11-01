# NLI alignment investigation report

{
  "investigation_revision": "2026_11_01_rev3",
  "root_cause": "unresolved",
  "known_generation_procedure": {
    "repo": "divyanshuaggarwal/IndicXNLI",
    "script": "src/make_dataset.py",
    "datasets_version_pinned_in_repo": "1.16.1",
    "documented_english_input": "load_dataset('xnli', 'en') (revision unpinned in script)",
    "indic_fields_saved": [
      "premise",
      "hypothesis",
      "label"
    ],
    "pair_id_persisted_in_indic_release": false
  },
  "backward_train_available": false,
  "current_reproduction_snapshot_available": true,
  "current_reproduction_snapshot_verified_2021_generation": false,
  "current_reproduction_snapshot_note": "Local exports including datasets==1.16.1 reconstructed today reproduce current load_dataset behavior; they are not verified 2021-generation English artifacts.",
  "train_mapping_method_attempted": "generation_snapshot_rejected_same_as_current_hf",
  "train_english_mappings_withheld": 392702,
  "owner_approved": false,
  "dev_semantic_review_complete": false,
  "indic_train_vs_xnli_mt_same_index_label_rate": 0.3440165825485992,
  "established_findings": {
    "current_english_sources_agree": "facebook/xnli en train and XNLI-MT-1.0 multinli.train.en.tsv match at every index (composite keys and labels 100%).",
    "indic_train_vs_current_english_order": "Indic forward train same-index label agreement with current English: 34.402% (135096/392702).",
    "production_train_english_join": "Withheld; do not use HF/XNLI-MT index join for train."
  },
  "unverified_historical_explanation": {
    "summary": "Indic forward train order does not match current English sources at same index. We do not know the exact generation-time Hugging Face revision or row ordering.",
    "required_to_resolve": [
      "Original English training artifact used at IndicXNLI translation time",
      "Exact Hugging Face datasets revision or snapshot from generation date"
    ]
  },
  "dev_mapping": {
    "method": "same_index_xnli_10_en",
    "pair_id_independent_join": false,
    "same_index_label_agreement_rate": 1.0,
    "semantic_alignment": "Not proven by labels alone; owner judgment columns required in sample CSV."
  },
  "owner_next_steps": [
    "Provide Divyanshu/indicxnli backward train JSON locally if recovered from git LFS.",
    "Or supply a verified generation-time English artifact (not a current reproduction export).",
    "Complete owner_judgment_* columns for 41 bilingual dev rows in nli_alignment_sample.csv."
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
    "same_index_label_agreement": 2490
  },
  "train_mapping_reliable": false,
  "normalization": "NFC + outer strip; composite_key=sha256(prem\\0hyp)",
  "owner_review_sample_rows": 200,
  "owner_review_sample": {
    "total_rows": 200,
    "bilingual_dev_rows": 41,
    "hindi_only_train_rows": 159,
    "description": "Not 200 bilingual alignment examples. Only 41 dev rows include English; 159 train rows are Hindi-only with English withheld.",
    "owner_judgment_columns": [
      "owner_judgment_premise_aligned",
      "owner_judgment_hypothesis_aligned",
      "owner_judgment_label_consistent"
    ]
  },
  "datasets_1_16_1_reload_label_agreement_with_indic_train": 0.3440165825485992,
  "current_reproduction_datasets_1_16_1_label_agreement_with_indic_train": 0.3440165825485992,
  "current_reproduction_datasets_1_16_1_note": "datasets==1.16.1 export run today matches current HF row order; not a verified 2021 generation snapshot."
}
