# RomCoM-X toxicity transformation integrity

The experiment is Hindi refusal-response generation with Romanized assistant targets. The Hindi Devanagari user input remains unchanged. Only the assistant turn may be romanized. This is not Romanized-input understanding and not general code-mixed safety robustness. No transformed rows were built in this freeze.

{
  "policy_source": "data_plan Appendix B.5 / phase2_reports toxicity_data_design_report.md",
  "user_prompt": "Never code-mixed; bytes preserved in input.prompt",
  "assistant_target": "Romanization / variants apply to target.response only",
  "turn_handling": "parse_conversation_turns preserves chronological order; user_prompt_from_turns and assistant_target_from_turns keep roles separate; multiple turns joined with \\n---\\n within same role only.",
  "labels": "normalized_label unchanged across EXP-01/02/03",
  "generate_exps": "EXP-03 uses EXP-02 changed token spans on target fields; toxicity input fields empty in _input_fields except none for prompt path\u2014prompt unchanged in build_toxicity_base.",
  "experimental_id_integrity": "stream_task emits same example_id/source_id/label/project_split for all EXP conditions."
}
