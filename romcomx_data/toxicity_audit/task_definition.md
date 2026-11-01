# Task definition

{
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
}

Transformation constraints are unchanged. Canonical, realistic-variant, and matched-noise conditions still share source IDs, labels, and splits.
