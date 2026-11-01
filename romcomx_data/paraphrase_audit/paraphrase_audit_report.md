# IndicXParaphrase audit report

{
  "title": "IndicXParaphrase Phase 2.1 audit (revised)",
  "provenance": {
    "repository": "ai4bharat/IndicXParaphrase",
    "revision": "181031f9e69c465e0c72fc84fd33b66114adb3e8",
    "revision_fetched_utc": "2026-11-01T14:07:04.346876+00:00",
    "hf_file": "data/hi.tsv",
    "local_path": "romcomx_data/raw/indicxparaphrase/hi.tsv",
    "local_sha256": "e35ac3463c2b7cf19160657b527c624f7f697f44326af3f251a8faf96e9f5c64",
    "local_size_bytes": 1214125,
    "card_license": null,
    "card_license_note": "Taken from HF dataset card; not hardcoded in loader.",
    "official_splits": "none_single_tsv",
    "files_on_card": [],
    "readme_excerpt_chars": 277,
    "field_definitions_from_card": {
      "note": "Card/README did not yield explicit per-column definitions; use measured alignment."
    },
    "columns_in_tsv": [
      "en",
      "sentence1",
      "sentence2",
      "label"
    ],
    "license_evidence": {
      "hf_card_data_keys": [],
      "hf_card_license_field": null,
      "hf_dataset_info_license_attr": null,
      "hf_tags": [
        "size_categories:10K<n<100K",
        "modality:text",
        "library:datasets",
        "library:mlcroissant",
        "region:us"
      ],
      "readme_license_mentions": [],
      "loader_hardcoded_cc0_warning": "source_loaders/indicxparaphrase.py hardcodes CC0; not used as audit evidence.",
      "audit_conclusion": "HF card license field remains null. Owner approved the license for the Phase 2.1 paraphrase audit on 2026-11-01; confirmation blocker removed.",
      "owner_approval": {
        "approved": true,
        "approved_on": "2026-11-01",
        "approved_by": "owner",
        "scope": "phase_2_1_paraphrase_audit",
        "decision": "Owner approved the IndicXParaphrase license for this audit and removed the confirmation blocker. The HF card license field is still null."
      }
    },
    "audit_scope": "inspection_only_no_rebuild"
  },
  "known_generation_procedure": "Single TSV from ai4bharat/IndicXParaphrase; columns en, sentence1, sentence2, label.",
  "integrity_summary": {
    "row_count": 2002,
    "label_distribution": {
      "1": 1001,
      "0": 1001
    },
    "missing_fields": {
      "en_blank": 0,
      "sentence1_blank": 0,
      "sentence2_blank": 0,
      "label_missing": 0
    },
    "non_binary_labels": 0,
    "duplicate_pair_rows": 0,
    "duplicate_sentence1_rows": 2002,
    "duplicate_sentence2_rows": 0,
    "identical_sentence1_sentence2_rows": 1,
    "identical_sides_by_label": {
      "0": 0,
      "1": 1
    },
    "sentence1_shared_across_multiple_rows": 1001,
    "sentence1_with_both_positive_and_negative": 1001,
    "en_contains_devanagari_rows": 0,
    "en_latin_only_rows": 2002,
    "official_train_val_test": "not_provided_single_file",
    "cross_split_overlap": "not_applicable_no_official_splits"
  },
  "english_pivot_gate": {
    "unique_counts": {
      "en": 1001,
      "sentence1": 1001,
      "sentence2": 2002,
      "pair": 2002
    },
    "en_constant_for_each_sentence1_groups": 1001,
    "en_varies_within_sentence1_groups": 0,
    "sentence1_groups_sharing_one_en_across_sentence2_and_labels": 1001,
    "en_values_mapping_to_single_sentence1": 1001,
    "en_values_mapping_to_multiple_sentence1": 0,
    "second_english_column_in_source": false,
    "rows_en_equals_sentence1": 0,
    "rows_en_equals_sentence2": 0,
    "rows_with_independent_sentence2_english": 0,
    "measured_en_alignment": "single en column; constant per sentence1 in all groups; not aligned to sentence2",
    "reject_english_pivot_reconstruction_for_code_mixed": true,
    "never_treat_devanagari_as_english": true
  },
  "romanization_summary": {
    "romanization_inventory": "romcomx_data/inventories/dakshina_hi_variant_inventory.tsv",
    "aggregate_token_methods": {
      "method_dakshina_canonical": 27100,
      "dakshina_hits": 27100,
      "non_devanagari_passthrough_tokens": 34788,
      "method_fallback_indic_map": 8998,
      "fallback_tokens": 8998,
      "latin_passthrough_tokens": 10
    },
    "positive_pairs": 1001,
    "negative_pairs": 1001,
    "positive_transformation_induced_collapse": 0,
    "positive_collapse_rate": 0.0,
    "negative_romanized_identical_while_native_differ": 0,
    "negative_false_identical_rate": 0.0,
    "rows_with_zero_dakshina_hits_both_sides": 0,
    "rows_silently_discarded": 0
  },
  "leakage_summary": {
    "sentence1_cluster_count": 1001,
    "sentence2_cluster_count": 2002,
    "max_rows_per_sentence1": 2,
    "max_rows_per_sentence2": 1,
    "sentence1_clusters_with_multiple_rows": 1001,
    "connected_component_count": 1001,
    "largest_component_sentence_count": 3,
    "split_unit_required": "connected_component_on_sentence1_and_sentence2_union",
    "sentence1_only_grouping_insufficient": false,
    "simulated_project_split_seed": 42,
    "simulated_rows": 2001,
    "sentence_leak_train_and_validation_cc_split": 0,
    "leaked_sentences_cc_split_sample": [],
    "sentence_leak_train_and_validation_sentence1_only_split": 0,
    "leaked_sentences_s1_only_sample": [],
    "cross_column_split_mismatch_pairs_observed": 0,
    "required_pipeline_grouping": "Use romcom_data.splitting.paraphrase_groups (union-find on both sentence columns)."
  },
  "quarantine_required": [
    {
      "row_index": 698,
      "source_id": "para-3eae2cd8a914bad0",
      "example_id": "para-3eae2cd8a914bad0:698",
      "label": 1,
      "sentence1": "उन्नाव गैंगरेप और हत्या मामले, जिसमें बीजेपी के विधायक कुलदीप सिंह सेंगर शामिल हैं, के एक मुख्य चश्मदीद की संदिग्ध परिस्थितियों में मौत हो गई है।",
      "sentence2": "उन्नाव गैंगरेप और हत्या मामले, जिसमें बीजेपी के विधायक कुलदीप सिंह सेंगर शामिल हैं, के एक मुख्य चश्मदीद की संदिग्ध परिस्थितियों में मौत हो गई है।",
      "exclusion_reason": "identical_positive_sentence1_equals_sentence2"
    }
  ],
  "recommendation": {
    "verdict": "CONDITIONAL_GO",
    "scope": "hindi_native_only_paraphrase_without_english_pivot",
    "blocking_reason": "NO-GO for English-pivot code-mixed path; CONDITIONAL for Hindi-native-only experiment with documented cluster leakage risk and no RomCoM EXP rebuild.",
    "english_pivot_gate_passed": false,
    "english_pivot_code_mixed_verdict": "NO_GO",
    "train_or_processed_rebuild": false
  },
  "experimental_consistency": {
    "source_of_truth": "src/romcom_data/generate_exps.py::generate_one",
    "exp01": "Canonical Dakshina spellings per eligible token on each input field independently.",
    "exp02": "Same base record; `realistic_from_canonical` with per-field RNG `derive_rng(base_seed, example_id, EXP-02)`.",
    "exp03": "Copies EXP-01 base; applies `noise_from_changes` using EXP-02 `changed_all` token spans so matched-noise uses same positions and target edit distances as realistic-variant.",
    "shared_metadata": "phase2_pipeline.stream_task writes same example_id, source_id, label, project_split for EXP-01/02/03; qa_matching_ids() verifies alignment when processed files exist.",
    "paraphrase_fields": [
      "input.sentence1",
      "input.sentence2"
    ],
    "rebuild_performed_in_this_audit": false,
    "when_paraphrase_enabled": "All three conditions will share identical source IDs, labels, and splits per example_id."
  },
  "handoff_status": {
    "status": "READY_FOR_OWNER_REVIEW",
    "owner_review_scope": "hindi_native_only_paraphrase_path",
    "blockers": [
      "english_pivot_code_mixed_path_rejected"
    ],
    "quarantined_row_count": 1,
    "license_owner_approved": true,
    "dataset_rebuild_allowed": false,
    "training_allowed": false
  },
  "limitations": [
    "No official train/validation/test; split on connected components (both sentence columns).",
    "English-pivot code-mixed reconstruction rejected (NO_GO).",
    "One identical positive row must stay quarantined.",
    "Romanization uses Dakshina inventory; fallback map tokens are not attested variants.",
    "HF card license field remains null. Owner approved the license on 2026-11-01."
  ]
}
