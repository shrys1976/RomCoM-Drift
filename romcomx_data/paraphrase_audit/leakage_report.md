# Paraphrase leakage report

{
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
}

Sentence1-only grouping is insufficient when the same sentence text can appear in sentence1 or sentence2 across rows. Use `paraphrase_groups` connected components.
