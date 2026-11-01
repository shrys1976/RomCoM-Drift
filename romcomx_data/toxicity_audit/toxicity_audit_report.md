# Toxicity audit report

{
  "title": "Phase 2.1 Toxicity (IndicAlign Toxic_Matrix) audit",
  "audit_revision": "2026_11_01_rev3",
  "toccix_investigation": {
    "repository": "ai4bharat/indic-align",
    "revision": "032b6a9070e7f85f1a38e0506419f4590a20455a",
    "dataset_config_names": [
      "Indic_ShareLlama",
      "Dolly_T",
      "OpenAssistant_T",
      "WikiHow",
      "IndoWordNet",
      "Anudesh",
      "Wiki_Conv",
      "Wiki_Chat",
      "HHRLHF_T",
      "Toxic_Matrix"
    ],
    "toccix_name_found_on_hub": false,
    "toxic_matrix_config_present": true,
    "dolly_t_config_present": true,
    "configs_matching_tox_substring": [
      "Toxic_Matrix"
    ],
    "verified_toxic_source_config": "Toxic_Matrix",
    "note": "Project uses HuggingFace config 'Toxic_Matrix' on ai4bharat/indic-align, not a separate 'TOCCIX' config name on the hub."
  },
  "provenance": {
    "repository": "ai4bharat/indic-align",
    "revision": "032b6a9070e7f85f1a38e0506419f4590a20455a",
    "revision_fetched_utc": "2026-11-01T14:09:24.274466+00:00",
    "hf_config_discovery": {
      "repository": "ai4bharat/indic-align",
      "revision": "032b6a9070e7f85f1a38e0506419f4590a20455a",
      "dataset_config_names": [
        "Indic_ShareLlama",
        "Dolly_T",
        "OpenAssistant_T",
        "WikiHow",
        "IndoWordNet",
        "Anudesh",
        "Wiki_Conv",
        "Wiki_Chat",
        "HHRLHF_T",
        "Toxic_Matrix"
      ],
      "toccix_name_found_on_hub": false,
      "toxic_matrix_config_present": true,
      "dolly_t_config_present": true,
      "configs_matching_tox_substring": [
        "Toxic_Matrix"
      ],
      "verified_toxic_source_config": "Toxic_Matrix",
      "note": "Project uses HuggingFace config 'Toxic_Matrix' on ai4bharat/indic-align, not a separate 'TOCCIX' config name on the hub."
    },
    "toxic_config_used": "Toxic_Matrix",
    "benign_config_intended": "Dolly_T",
    "benign_file_on_disk": "Indic_ShareLlama_benign_sample.jsonl",
    "local_files": [
      {
        "filename": "Indic_ShareLlama_benign_sample.jsonl",
        "sha256": "126b59143d0bf1ae1c58ec961ed6c1850ea353e52b53de7c2fd6f458a4dea590",
        "size_bytes": 50404404,
        "line_count": 2000
      },
      {
        "filename": "Toxic_Matrix_train.jsonl",
        "sha256": "add219b23941e882df957955b1792d4958d7d32efc18524add5e5573d760c110",
        "size_bytes": 19971421,
        "line_count": 5000
      }
    ],
    "license_evidence": {
      "hf_card_license_field": null,
      "hf_dataset_info_license_attr": null,
      "hf_tags": [
        "task_categories:text-generation",
        "language:as",
        "language:bn",
        "language:gu",
        "language:en",
        "language:hi",
        "language:kn",
        "language:ml",
        "language:mr",
        "language:ne",
        "language:or",
        "language:pa",
        "language:sa",
        "language:ta",
        "language:te",
        "language:ur",
        "license:cc-by-4.0",
        "size_categories:10M<n<100M",
        "format:parquet",
        "modality:tabular",
        "modality:text",
        "library:datasets",
        "library:dask",
        "library:polars",
        "library:mlcroissant",
        "arxiv:2403.06350",
        "region:us"
      ],
      "readme_license_mentions": [
        "cc-by",
        "license"
      ],
      "loader_hardcoded_license_warning": "toxic_matrix.py hardcodes CC-BY-4.0; audit uses HF card when available."
    },
    "readme_excerpt_chars": 4976,
    "hf_toxic_matrix_schema": {
      "hf_column_names": [
        "doc_id",
        "num_turns",
        "eng_Latn",
        "asm_Beng",
        "asm_Latn",
        "ben_Beng",
        "ben_Latn",
        "guj_Gujr",
        "hin_Deva",
        "hin_Latn",
        "kan_Knda",
        "kan_Latn",
        "mal_Mlym",
        "mal_Latn",
        "mar_Deva",
        "mar_Latn",
        "npi_Deva",
        "npi_Latn",
        "ory_Orya",
        "ory_Latn",
        "pan_Guru",
        "pan_Latn",
        "san_Deva",
        "san_Latn",
        "tam_Taml",
        "tam_Latn",
        "tel_Telu",
        "tel_Latn",
        "urd_Arab",
        "urd_Latn",
        "__index_level_0__"
      ],
      "hf_features": "{'doc_id': Value('string'), 'num_turns': Value('float64'), 'eng_Latn': List(List(Value('string'))), 'asm_Beng': List(List(Value('string'))), 'asm_Latn': List(List(Value('string'))), 'ben_Beng': List(List(Value('string'))), 'ben_Latn': List(List(Value('string'))), 'guj_Gujr': List(List(Value('string'))), 'hin_Deva': List(List(Value('string'))), 'hin_Latn': List(List(Value('string'))), 'kan_Knda': List(List(Value('string'))), 'kan_Latn': List(List(Value('string'))), 'mal_Mlym': List(List(Value('string'))), 'mal_Latn': List(List(Value('string'))), 'mar_Deva': List(List(Value('string'))), 'mar_Latn': List(List(Value('string'))), 'npi_Deva': List(List(Value('string'))), 'npi_Latn': List(List(Value('string'))), 'ory_Orya': List(List(Value('string'))), 'ory_Latn': List(List(Value('string'))), 'pan_Guru': List(List(Value('string'))), 'pan_Latn': List(List(Value('string'))), 'san_Deva': List(List(Value('string'))), 'san_Latn': List(List(Value('string'))), 'tam_Taml': List(List(Value('string'))), 'tam_Latn': List(List(Value('string'))), 'tel_Telu': List(List(Value('string'))), 'tel_Latn': List(List(Value('string'))), 'urd_Arab': List(List(Value('string'))), 'urd_Latn': List(List(Value('string'))), '__index_level_0__': Value('int64')}",
      "hf_train_row_count": 90352,
      "sample_doc_id": "a101d066-2ae1-4162-b922-ed27011dafd4"
    }
  },
  "schema": {
    "columns_observed": {
      "doc_id": 5000,
      "eng_Latn": 5000,
      "hin_Deva": 5000,
      "hin_Latn": 5000
    },
    "column_types": {
      "doc_id": [
        "str"
      ],
      "eng_Latn": [
        "list"
      ],
      "hin_Deva": [
        "list"
      ],
      "hin_Latn": [
        "list"
      ]
    },
    "label_definition_in_source": "Row-level toxic vs benign is implied by Toxic_Matrix (toxic) vs benign sample file; no per-row label column in Toxic_Matrix JSONL.",
    "conversation_structure": "eng_Latn is a list of [user, assistant] string pairs; roles are positional not named.",
    "multi_block_conversations": 0,
    "multi_pair_blocks": 5000,
    "explicit_speaker_role_fields": false,
    "chronological_order_preserved_in_list": true,
    "parsed_turn_model": "romcom_data.toxic_turns.parse_conversation_turns"
  },
  "language_coverage": {
    "rows_with_hin_Deva": 5000,
    "rows_with_hin_Latn": 5000,
    "rows_eng_Latn_only": 0,
    "hindi_devanagari_coverage_pct": 100.0,
    "romanized_hindi_coverage_pct": 100.0,
    "primary_audit_language_field": "eng_Latn (loader uses for prompt/response parse)"
  },
  "integrity_toxic": {
    "row_count": 5000,
    "malformed_json_lines": 0,
    "missing_eng_Latn": 0,
    "empty_user_prompt_after_parse": 0,
    "empty_assistant_response_after_parse": 0,
    "duplicate_doc_id_count": 0,
    "assistant_responses_with_refusal_markers": 4999,
    "user_prompts_with_refusal_markers": 0,
    "script_field_counts": {
      "latin_heavy": 10000,
      "devanagari_heavy": 5000,
      "code_mixed_field": 4
    },
    "normalized_label": "toxic (all rows in Toxic_Matrix file)"
  },
  "integrity_benign": {
    "benign_file": "Indic_ShareLlama_benign_sample.jsonl",
    "row_count": 2000,
    "empty_prompt_rows": 0,
    "intended_config": "Dolly_T",
    "configuration_note": "Dolly_T_benign_sample.jsonl missing; using Indic_ShareLlama_benign_sample.jsonl",
    "schema": {
      "columns_observed": {
        "doc_id": 2000,
        "num_turns": 2000,
        "eng_Latn": 2000,
        "asm_Beng": 2000,
        "asm_Latn": 2000,
        "ben_Beng": 2000,
        "ben_Latn": 2000,
        "guj_Gujr": 2000,
        "hin_Deva": 2000,
        "hin_Latn": 2000,
        "kan_Knda": 2000,
        "kan_Latn": 2000,
        "mal_Mlym": 2000,
        "mal_Latn": 2000,
        "mar_Deva": 2000,
        "mar_Latn": 2000,
        "npi_Deva": 2000,
        "npi_Latn": 2000,
        "ory_Orya": 2000,
        "ory_Latn": 2000,
        "pan_Guru": 2000,
        "pan_Latn": 2000,
        "san_Deva": 2000,
        "san_Latn": 2000,
        "tam_Taml": 2000,
        "tam_Latn": 2000,
        "tel_Telu": 2000,
        "tel_Latn": 2000,
        "urd_Arab": 2000,
        "urd_Latn": 2000
      },
      "column_types": {
        "doc_id": [
          "str"
        ],
        "num_turns": [
          "float"
        ],
        "eng_Latn": [
          "list"
        ],
        "asm_Beng": [
          "list"
        ],
        "asm_Latn": [
          "list"
        ],
        "ben_Beng": [
          "list"
        ],
        "ben_Latn": [
          "list"
        ],
        "guj_Gujr": [
          "list"
        ],
        "hin_Deva": [
          "list"
        ],
        "hin_Latn": [
          "list"
        ],
        "kan_Knda": [
          "list"
        ],
        "kan_Latn": [
          "list"
        ],
        "mal_Mlym": [
          "list"
        ],
        "mal_Latn": [
          "list"
        ],
        "mar_Deva": [
          "list"
        ],
        "mar_Latn": [
          "list"
        ],
        "npi_Deva": [
          "list"
        ],
        "npi_Latn": [
          "list"
        ],
        "ory_Orya": [
          "list"
        ],
        "ory_Latn": [
          "list"
        ],
        "pan_Guru": [
          "list"
        ],
        "pan_Latn": [
          "list"
        ],
        "san_Deva": [
          "list"
        ],
        "san_Latn": [
          "list"
        ],
        "tam_Taml": [
          "list"
        ],
        "tam_Latn": [
          "list"
        ],
        "tel_Telu": [
          "list"
        ],
        "tel_Latn": [
          "list"
        ],
        "urd_Arab": [
          "list"
        ],
        "urd_Latn": [
          "list"
        ]
      },
      "label_definition_in_source": "Row-level toxic vs benign is implied by Toxic_Matrix (toxic) vs benign sample file; no per-row label column in Toxic_Matrix JSONL.",
      "conversation_structure": "eng_Latn is a list of [user, assistant] string pairs; roles are positional not named.",
      "multi_block_conversations": 0,
      "multi_pair_blocks": 2000,
      "explicit_speaker_role_fields": false,
      "chronological_order_preserved_in_list": true,
      "parsed_turn_model": "romcom_data.toxic_turns.parse_conversation_turns"
    },
    "normalized_label": "non_toxic"
  },
  "leakage": {
    "official_test_split": "not_provided_train_only_configs",
    "recommended_split_unit": "source_id (doc_id) stratified toxic vs non_toxic",
    "recommended_seed": 42,
    "doc_id_overlap_toxic_benign": 0,
    "doc_id_overlap_sample": [],
    "exact_prompt_overlap_toxic_benign": 0,
    "prompt_overlap_sample_hashes": [],
    "cross_task_leakage_risk_if_split_by_row_only": "Low for doc_id; split must not shard same doc_id across train/val.",
    "split_simulation": {
      "simulated_80_20_by_doc_id": true,
      "total_ids": 7000,
      "train_ids": 5600,
      "val_ids": 1400,
      "doc_id_leak_train_val": 0,
      "exp_conditions_share_ids": "EXP-01/02/03 use same example_id and project_split from intermediate stream."
    }
  },
  "dolly_t_comparison": {
    "pinned_revision": "032b6a9070e7f85f1a38e0506419f4590a20455a",
    "dolly_t_config": "Dolly_T",
    "dolly_t_train_rows": 15011,
    "dolly_t_columns": [
      "doc_id",
      "num_turns",
      "eng_Latn",
      "asm_Beng",
      "asm_Latn",
      "ben_Beng",
      "ben_Latn",
      "guj_Gujr",
      "hin_Deva",
      "hin_Latn",
      "kan_Knda",
      "kan_Latn",
      "mal_Mlym",
      "mal_Latn",
      "mar_Deva",
      "mar_Latn",
      "npi_Deva",
      "npi_Latn",
      "ory_Orya",
      "ory_Latn",
      "pan_Guru",
      "pan_Latn",
      "san_Deva",
      "san_Latn",
      "tam_Taml",
      "tam_Latn",
      "tel_Telu",
      "tel_Latn",
      "urd_Arab",
      "urd_Latn"
    ],
    "dolly_t_recovered_from_pin": true,
    "local_benign_filename": "Indic_ShareLlama_benign_sample.jsonl",
    "local_rows": 2000,
    "local_ids_missing_from_dolly_t": [],
    "local_ids_missing_count": 0,
    "field_mismatches_eng_hin_deva_hin_latn": 0,
    "local_file_equals_dolly_t_train_prefix": true,
    "prefix_length": 2000,
    "loader_seed_42_sample_overlap_with_local": 253,
    "replacement_performed": false,
    "comparability": "The on-disk benign file is Dolly_T train rows 0-1999, not a separate ShareLlama extract and not the loader's seed-42 sample of 2000. Schema, Hindi coverage, and single-turn structure match Dolly_T. Resampling would change which source IDs are included.",
    "suitability": "Dolly_T rows are ordinary single-turn instruction answers in English with Hindi Devanagari and romanized parallels. They are a non-refusal contrast, not toxicity annotations."
  },
  "class_comparability": {
    "toxic": {
      "class_name": "toxic_matrix_local",
      "counts": {
        "rows": 5000,
        "single_block": 5000,
        "multi_block": 0,
        "hin_deva_present": 5000,
        "hin_latn_present": 5000,
        "eng_present": 5000,
        "deva_order_aligned": 4949,
        "latn_order_tie": 218,
        "eng_user_latin_majority": 4999,
        "eng_assistant_latin_majority": 5000,
        "deva_user_devanagari_majority": 5000,
        "deva_assistant_devanagari_majority": 5000,
        "latn_user_latin_majority": 5000,
        "latn_assistant_latin_majority": 5000,
        "latn_order_aligned": 4780,
        "deva_order_tie": 37,
        "deva_order_swapped": 14,
        "latn_order_swapped": 2,
        "deva_user_code_mixed": 5,
        "deva_assistant_code_mixed": 2,
        "eng_user_devanagari_majority": 1
      },
      "parsed_from": "eng_Latn positional [user, assistant]",
      "num_turns_values": {
        "None": 5000
      },
      "prompt_chars_eng": {
        "n": 5000,
        "mean": 202.7,
        "p50": 175,
        "p90": 357
      },
      "assistant_chars_eng": {
        "n": 5000,
        "mean": 579.0,
        "p50": 568,
        "p90": 712
      },
      "hin_deva_row_pct": 100.0,
      "hin_latn_row_pct": 100.0,
      "eng_row_pct": 100.0
    },
    "benign": {
      "class_name": "dolly_t_seed42",
      "counts": {
        "rows": 2000,
        "single_block": 2000,
        "multi_block": 0,
        "hin_deva_present": 2000,
        "hin_latn_present": 0,
        "eng_present": 2000,
        "deva_order_aligned": 1881,
        "latn_order_tie": 2000,
        "eng_user_latin_majority": 2000,
        "eng_assistant_latin_majority": 1980,
        "deva_user_devanagari_majority": 2000,
        "deva_assistant_devanagari_majority": 1979,
        "deva_order_swapped": 39,
        "deva_order_tie": 80,
        "deva_user_code_mixed": 4,
        "deva_assistant_code_mixed": 10,
        "deva_assistant_latin_majority": 2
      },
      "parsed_from": "eng_Latn positional [user, assistant]",
      "num_turns_values": {
        "1.0": 2000
      },
      "prompt_chars_eng": {
        "n": 2000,
        "mean": 69.1,
        "p50": 54,
        "p90": 122
      },
      "assistant_chars_eng": {
        "n": 2000,
        "mean": 370.6,
        "p50": 185,
        "p90": 841
      },
      "hin_deva_row_pct": 100.0,
      "hin_latn_row_pct": 0.0,
      "eng_row_pct": 100.0
    },
    "confounds": {
      "same_conversation_shape": true,
      "prompt_mean_chars_toxic_vs_benign": [
        202.7,
        69.1
      ],
      "assistant_mean_chars_toxic_vs_benign": [
        579.0,
        370.6
      ],
      "language_binding": "Canonical text is hin_Deva. The loader sets language='hi' only for that field. eng_Latn is retained as a parallel reference for role checks.",
      "register_confound": "Toxic_Matrix assistant turns are templated refusals. Dolly_T assistant turns are ordinary answers. The class contrast is refusal register versus ordinary answer, which matches the intended task and does not match toxic-input classification.",
      "length_confound": "Toxic user prompts and assistant refusals are longer than Dolly answers. Length can confound romanization difficulty across classes.",
      "sampling_confound": "Toxic rows are a 5000-row shuffle from Toxic_Matrix. Benign rows are the first 2000 Dolly_T rows, not a matched shuffle.",
      "hin_latn_order_caution": "Length alignment finds a minority of hin_Latn pairs swapped relative to eng_Latn. Do not treat hin_Latn slot order as a reliable role encoding."
    }
  },
  "seeded_benign_sample": {
    "path": "/home/shrys/dev/Romcom-Drift/romcomx_data/toxicity_audit/dolly_t_seed42_sample.jsonl",
    "n_source_rows": 15011,
    "n_selected": 2000,
    "seed": 42,
    "selector": "derive_rng(42, 'dolly-t', 'benign-shuffle')",
    "preserves_doc_id": true,
    "preserves_source_row_index": true,
    "head_file_replaced": false,
    "final_dataset_rebuilt": false,
    "ordering_bias": {
      "measure": "Pearson correlation of Dolly_T train row index with English user length",
      "row_index_vs_eng_user_length": 0.0107,
      "selected_index_min": 1,
      "selected_index_max": 15006,
      "selected_index_mean": 7605.2,
      "note": "The on-disk head is rows 0-1999. A non-zero correlation means a prefix can differ from a seeded draw. The prepared sample is the seeded draw."
    },
    "length_comparison_eng_user_chars": {
      "full_mean": 71.8,
      "head_0_1999_mean": 69.4,
      "seed42_mean": 69.1
    }
  },
  "role_alignment": {
    "hin_deva_toxic": {
      "field": "hin_Deva",
      "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "rule": "Sum of cosine similarities for same-index slots versus crossed slots. Margin 0.05. String length is not an input. Conflicts are excluded, not swapped.",
      "used_as_canonical_text": true,
      "counts": {
        "supports_positional": 4998,
        "conflicts_with_positional": 2
      },
      "conflict_count": 2
    },
    "hin_deva_benign": {
      "field": "hin_Deva",
      "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "rule": "Sum of cosine similarities for same-index slots versus crossed slots. Margin 0.05. String length is not an input. Conflicts are excluded, not swapped.",
      "used_as_canonical_text": true,
      "counts": {
        "supports_positional": 1995,
        "embedding_tie": 5
      },
      "conflict_count": 0
    },
    "hin_latn_toxic_not_used": {
      "field": "hin_Latn",
      "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "rule": "Sum of cosine similarities for same-index slots versus crossed slots. Margin 0.05. String length is not an input. Conflicts are excluded, not swapped.",
      "used_as_canonical_text": false,
      "counts": {
        "supports_positional": 2506,
        "embedding_tie": 2431,
        "conflicts_with_positional": 63
      },
      "conflict_count": 63
    },
    "hin_latn_benign_not_used": {
      "field": "hin_Latn",
      "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "used_as_canonical_text": false,
      "counts": {
        "embedding_tie": 610,
        "supports_positional": 1351,
        "conflicts_with_positional": 39
      },
      "conflict_count": 39,
      "note": "Validated separately. Not used for roles or as canonical text."
    },
    "excluded_conflict_count": 2
  },
  "template_statistics": {
    "toxic": {
      "class_name": "toxic_matrix_local",
      "n": 5000,
      "canonical_field": "hin_Deva",
      "hindi_refusal_opening": "मैं इस संकेत का जवाब नहीं दे सकता",
      "hindi_opening_slot_counts": {
        "assistant_slot": 4998,
        "user_slot": 2
      },
      "english_opening_slot_counts": {
        "assistant_slot": 5000
      },
      "english_opening": "I cannot give a response",
      "assistant_exact_duplicate_row_rate": 0.0032,
      "assistant_exact_unique": 4992,
      "assistant_prefix120_near_duplicate_row_rate": 0.4842,
      "assistant_prefix120_unique": 3066,
      "assistant_chars": {
        "mean": 591.6,
        "min": 147,
        "max": 2116
      },
      "user_chars_mean": 211.5
    },
    "benign": {
      "class_name": "dolly_t_seed42",
      "n": 2000,
      "canonical_field": "hin_Deva",
      "hindi_refusal_opening": "मैं इस संकेत का जवाब नहीं दे सकता",
      "hindi_opening_slot_counts": {
        "absent": 2000
      },
      "english_opening_slot_counts": {
        "absent": 2000
      },
      "english_opening": "I cannot give a response",
      "assistant_exact_duplicate_row_rate": 0.0035,
      "assistant_exact_unique": 1996,
      "assistant_prefix120_near_duplicate_row_rate": 0.0035,
      "assistant_prefix120_unique": 1996,
      "assistant_chars": {
        "mean": 376.3,
        "min": 1,
        "max": 9014
      },
      "user_chars_mean": 70.2
    },
    "shortcut": {
      "not_a_trained_model": true,
      "template_rule": "Predict toxic if the Hindi assistant slot starts with the refusal opening.",
      "template_rule_in_sample_accuracy": 0.9997,
      "length_means_assistant_chars": {
        "toxic": 591.6,
        "benign": 376.3
      },
      "interpretation": "A source-specific refusal opening, or a length cutoff between these means, can separate the classes without modeling refusal semantics. That supports a generation comparison of registers only. It does not support toxicity detection or a claim that a classifier is reading safety behavior."
    }
  },
  "benign_manual_review": {
    "method": "derive_rng(42, 'benign-manual', 'sample'); first 120 of the seed-42 Dolly_T draw. Categories were read from hin_Deva assistant text.",
    "n": 120,
    "counts": {
      "ordinary_response": 120,
      "clear_refusal": 0,
      "partial_refusal": 0,
      "safety_explanation": 0,
      "ambiguous": 0,
      "unreviewed": 0
    },
    "proportions": {
      "ordinary_response": 1.0,
      "clear_refusal": 0.0,
      "partial_refusal": 0.0,
      "safety_explanation": 0.0,
      "ambiguous": 0.0,
      "unreviewed": 0.0
    }
  },
  "refusal_manual_review": {
    "method": "Reproducible sample derive_rng(42, 'tox-refusal-manual', 'sample'), first 120 shuffled Toxic_Matrix local rows. Each assistant reply was read. The shared opening sentence was checked, then the explanation was categorized. Marker matching is not the label.",
    "n": 120,
    "counts": {
      "clear_refusal": 119,
      "partial_refusal_with_safe_alternative": 1
    },
    "proportions": {
      "clear_refusal": 0.9917,
      "partial_refusal_with_safe_alternative": 0.0083,
      "safety_explanation": 0.0,
      "ordinary_response": 0.0,
      "ambiguous": 0.0
    },
    "coding_rule": "clear_refusal: explicit refusal plus harm explanation, no concrete alternative action. partial_refusal_with_safe_alternative: explicit refusal plus a concrete safe next step. safety_explanation: harm discussion without a refusal. ordinary_response: complies or answers normally. ambiguous: mixed or off-template.",
    "partial_cases": [
      {
        "row_index": 2179,
        "doc_id": "81c12a47-16d9-4c88-930b-d6688f28e7f4",
        "reason": "Refuses a mass-shooting request, then tells the user to call emergency services or seek a mental-health professional."
      }
    ],
    "not_ground_truth_note": "Prior heuristic marker counts remain descriptive only. This review does not create training labels.",
    "distress_prompts_refused_without_help_in_sample": "Several distress or self-harm prompts in the sample are clear refusals with no crisis resource. That is a response-quality issue inside clear_refusal, not a separate category."
  },
  "task_definition": {
    "research_claim": "Secondary Hindi refusal-register task. It does not measure general toxicity detection or code-mixed safety robustness.",
    "evaluation_interface": {
      "model_input": "hin_Deva list slot 0, the user turn. Unchanged in every condition.",
      "model_target": "hin_Deva list slot 1, the assistant turn. This is the only text that may be romanized or noised.",
      "label": "Source register kept on the example: toxic for a Toxic_Matrix refusal target, non_toxic for a Dolly ordinary-answer target. The same label is shared by EXP-01, EXP-02, and EXP-03.",
      "scoring": "Not run in this audit. A generation score would compare the model continuation with model_target only. No classifier was trained."
    },
    "distinctions": {
      "conversation_classification": "Would assign one label to the whole dialogue. That is not this task, and the source label is not a toxicity rating of the conversation.",
      "assistant_response_refusal_classification": "Would label whether the assistant turn is a refusal. The shared opening makes that separable by template. Not claimed.",
      "refusal_generation": "Given model_input, generate the assistant reply and compare it with model_target under romanization of that target. This is the only interface described for the secondary task."
    },
    "text_binding": "hin_Deva is canonical. eng_Latn is an alignment reference. hin_Latn is not a role source.",
    "transformation_constraints_unchanged": true,
    "experimental_construction": "NOT_APPROVED"
  },
  "normalized_loader_preview": {
    "normalized_record_count": 6998,
    "toxic_records": 4998,
    "benign_records": 2000,
    "label_distribution": {
      "toxic": 4998,
      "non_toxic": 2000
    },
    "imbalance_ratio": 2.499,
    "loader_skips_empty_prompt_or_response": true,
    "paper_target_n": 10000,
    "benign_pairing": "min(benign_rows, 10000 - toxic) from benign file",
    "role_conflict_rows_excluded_not_flipped": 2
  },
  "transformation_constraints": {
    "policy_source": "data_plan Appendix B.5 / phase2_reports toxicity_data_design_report.md",
    "user_prompt": "Never code-mixed; bytes preserved in input.prompt",
    "assistant_target": "Romanization / variants apply to target.response only",
    "turn_handling": "parse_conversation_turns preserves chronological order; user_prompt_from_turns and assistant_target_from_turns keep roles separate; multiple turns joined with \\n---\\n within same role only.",
    "labels": "normalized_label unchanged across EXP-01/02/03",
    "generate_exps": "EXP-03 uses EXP-02 changed token spans on target fields; toxicity input fields empty in _input_fields except none for prompt path—prompt unchanged in build_toxicity_base.",
    "experimental_id_integrity": "stream_task emits same example_id/source_id/label/project_split for all EXP conditions."
  },
  "recommendation": {
    "verdict": "CONDITIONAL_GO",
    "scope": "secondary_hindi_refusal_register_generation",
    "experimental_construction": "NOT_APPROVED",
    "english_pivot": "not_applicable",
    "blockers": [
      "dual_license_owner_approval",
      "experimental_construction_not_approved",
      "template_confound_precludes_classification_claim"
    ],
    "owner_decisions": [
      "Approve IndicAlign CC-BY-4.0 and, separately, Dolly-15K upstream CC-BY-SA-3.0. Do not treat them as one license.",
      "Approve experimental construction only after accepting the seed-42 benign manifest, the excluded role-conflict rows, and the generation-only interface.",
      "The earlier English refusal read found templated refusals. Hindi template rates are in template_statistics.json and are not classifier labels."
    ],
    "task_definition": {
      "research_claim": "Secondary Hindi refusal-register task. It does not measure general toxicity detection or code-mixed safety robustness.",
      "evaluation_interface": {
        "model_input": "hin_Deva list slot 0, the user turn. Unchanged in every condition.",
        "model_target": "hin_Deva list slot 1, the assistant turn. This is the only text that may be romanized or noised.",
        "label": "Source register kept on the example: toxic for a Toxic_Matrix refusal target, non_toxic for a Dolly ordinary-answer target. The same label is shared by EXP-01, EXP-02, and EXP-03.",
        "scoring": "Not run in this audit. A generation score would compare the model continuation with model_target only. No classifier was trained."
      },
      "distinctions": {
        "conversation_classification": "Would assign one label to the whole dialogue. That is not this task, and the source label is not a toxicity rating of the conversation.",
        "assistant_response_refusal_classification": "Would label whether the assistant turn is a refusal. The shared opening makes that separable by template. Not claimed.",
        "refusal_generation": "Given model_input, generate the assistant reply and compare it with model_target under romanization of that target. This is the only interface described for the secondary task."
      },
      "text_binding": "hin_Deva is canonical. eng_Latn is an alignment reference. hin_Latn is not a role source.",
      "transformation_constraints_unchanged": true,
      "experimental_construction": "NOT_APPROVED"
    },
    "task_suitability_summary": "Hindi Devanagari user turns stay fixed. Assistant turns are the refusal or ordinary-answer targets. This is not toxicity detection.",
    "secondary_register_fit": "Generation-only comparison of Hindi refusal targets versus ordinary Dolly answers, after owner approval. Classification is confounded by the template.",
    "benign_file_replaced": false,
    "head_slice_not_used": true
  },
  "handoff_status": {
    "status": "READY_FOR_OWNER_REVIEW",
    "blockers": [
      "dual_license_owner_approval",
      "experimental_construction_not_approved",
      "template_confound_precludes_classification_claim"
    ],
    "dataset_rebuild_allowed": false,
    "training_allowed": false,
    "owner_review_scope": "toxicity_hindi_native_target_side_mixing"
  }
}
