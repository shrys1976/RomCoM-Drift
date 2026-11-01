# Toxicity audit report

{
  "title": "Phase 2.1 Toxicity (IndicAlign Toxic_Matrix) audit",
  "audit_revision": "2026_11_01_rev2",
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
    "assistant_responses_with_refusal_markers": 5000,
    "user_prompts_with_refusal_markers": 76,
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
      "class_name": "benign_local",
      "counts": {
        "rows": 2000,
        "single_block": 2000,
        "multi_block": 0,
        "hin_deva_present": 2000,
        "hin_latn_present": 1994,
        "eng_present": 2000,
        "deva_order_aligned": 1892,
        "latn_order_swapped": 21,
        "eng_user_latin_majority": 2000,
        "eng_assistant_latin_majority": 1994,
        "deva_user_devanagari_majority": 2000,
        "deva_assistant_devanagari_majority": 1994,
        "latn_user_latin_majority": 1994,
        "latn_assistant_latin_majority": 1994,
        "latn_order_aligned": 1824,
        "latn_order_tie": 155,
        "deva_order_tie": 83,
        "deva_order_swapped": 25,
        "deva_assistant_code_mixed": 8,
        "deva_user_code_mixed": 5,
        "eng_assistant_code_mixed": 1
      },
      "parsed_from": "eng_Latn positional [user, assistant]",
      "num_turns_values": {
        "1.0": 2000
      },
      "prompt_chars_eng": {
        "n": 2000,
        "mean": 69.4,
        "p50": 53,
        "p90": 124
      },
      "assistant_chars_eng": {
        "n": 2000,
        "mean": 384.1,
        "p50": 195,
        "p90": 817
      },
      "hin_deva_row_pct": 100.0,
      "hin_latn_row_pct": 99.7,
      "eng_row_pct": 100.0
    },
    "confounds": {
      "same_conversation_shape": true,
      "prompt_mean_chars_toxic_vs_benign": [
        202.7,
        69.4
      ],
      "assistant_mean_chars_toxic_vs_benign": [
        579.0,
        384.1
      ],
      "language_confound": "Both classes store English in eng_Latn and Hindi in hin_Deva/hin_Latn. The current loader parses eng_Latn while setting language='hi'. Romanizing that English assistant text is not Hindi romanization.",
      "register_confound": "Toxic_Matrix assistant turns are templated refusals. Dolly_T assistant turns are ordinary answers. The class contrast is refusal register versus ordinary answer, which matches the intended task and does not match toxic-input classification.",
      "length_confound": "Toxic user prompts and assistant refusals are longer than Dolly answers. Length can confound romanization difficulty across classes.",
      "sampling_confound": "Toxic rows are a 5000-row shuffle from Toxic_Matrix. Benign rows are the first 2000 Dolly_T rows, not a matched shuffle.",
      "hin_latn_order_caution": "Length alignment finds a minority of hin_Latn pairs swapped relative to eng_Latn. Do not treat hin_Latn slot order as a reliable role encoding."
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
    "intended_task": "Assistant safety/refusal-register robustness under romanization. User prompts stay unchanged. Only eligible assistant turns are romanized.",
    "not_this_task": "Not conventional toxic-input classification. The Toxic_Matrix label is not a turn-level toxicity tag on the user text. The assistant side of the toxic file is a refusal, and the benign side is an ordinary answer.",
    "what_a_label_applies_to": "File-level class on the whole example: toxic source versus Dolly benign source. It is not a per-turn toxicity rating.",
    "current_text_binding": "parse_conversation_turns reads eng_Latn. Hindi parallels are present but are not the strings the loader currently treats as the conversation.",
    "transformation_constraints_unchanged": true
  },
  "normalized_loader_preview": {
    "normalized_record_count": 7000,
    "toxic_records": 5000,
    "benign_records": 2000,
    "label_distribution": {
      "toxic": 5000,
      "non_toxic": 2000
    },
    "imbalance_ratio": 2.5,
    "loader_skips_empty_prompt_or_response": true,
    "paper_target_n": 10000,
    "benign_pairing": "min(benign_rows, 10000 - toxic) from benign file"
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
    "scope": "hindi_toxicity_safety_register_secondary_evaluation",
    "english_pivot": "not_applicable",
    "blockers": [
      "license_confirm_on_card",
      "owner_accept_dolly_t_head_slice"
    ],
    "owner_decisions": [
      "Confirm CC-BY-4.0. The dataset card license field is null; the hub tag is license:cc-by-4.0 and the README mentions cc-by.",
      "Accept the on-disk benign file as Dolly_T train rows 0-1999 (filename Indic_ShareLlama_benign_sample.jsonl is wrong), or explicitly request a new seeded sample. Do not treat those as equivalent.",
      "Accept the task as assistant refusal-register robustness under romanization, with user prompts unchanged. Do not approve it as toxic-input classification.",
      "Accept the English eng_Latn binding in the current loader, or require a later switch to hin_Deva before any Hindi romanization experiment. This audit does not make that switch.",
      "Accept the manual refusal review: templated clear refusals dominate; one sampled reply adds a concrete help redirect. Heuristic marker hits are not labels."
    ],
    "task_definition": {
      "intended_task": "Assistant safety/refusal-register robustness under romanization. User prompts stay unchanged. Only eligible assistant turns are romanized.",
      "not_this_task": "Not conventional toxic-input classification. The Toxic_Matrix label is not a turn-level toxicity tag on the user text. The assistant side of the toxic file is a refusal, and the benign side is an ordinary answer.",
      "what_a_label_applies_to": "File-level class on the whole example: toxic source versus Dolly benign source. It is not a per-turn toxicity rating.",
      "current_text_binding": "parse_conversation_turns reads eng_Latn. Hindi parallels are present but are not the strings the loader currently treats as the conversation.",
      "transformation_constraints_unchanged": true
    },
    "task_suitability_summary": "Toxic_Matrix pairs harmful user prompts with templated assistant refusals. Dolly_T supplies ordinary answers. Labels are source-file classes, not turn-level toxicity ratings.",
    "secondary_register_fit": "Fits assistant refusal-register robustness if the owner accepts the Dolly_T head slice, the license, and whether experiments use English eng_Latn or Hindi hin_Deva.",
    "benign_file_replaced": false
  },
  "handoff_status": {
    "status": "READY_FOR_OWNER_REVIEW",
    "blockers": [
      "license_confirm_on_card",
      "owner_accept_dolly_t_head_slice"
    ],
    "dataset_rebuild_allowed": false,
    "training_allowed": false,
    "owner_review_scope": "toxicity_hindi_native_target_side_mixing"
  }
}
