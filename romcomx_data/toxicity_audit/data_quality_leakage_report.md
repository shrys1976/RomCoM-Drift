# Data quality and leakage

The frozen split supersedes the simulation below. Toxic and benign user prompts share one union-find before assignment. Groups are atomic. Frozen rule: 80/10/10 train/validation/test, stratified by label, seed 42. Validation is for tuning only; test is the held-out final evaluation split. See `prompt_group_leakage.json` and `final_source_manifest.csv`.

{


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
}

{
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
}
