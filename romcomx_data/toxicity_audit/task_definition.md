# Task definition

{
  "intended_task": "Assistant safety/refusal-register robustness under romanization. User prompts stay unchanged. Only eligible assistant turns are romanized.",
  "not_this_task": "Not conventional toxic-input classification. The Toxic_Matrix label is not a turn-level toxicity tag on the user text. The assistant side of the toxic file is a refusal, and the benign side is an ordinary answer.",
  "what_a_label_applies_to": "File-level class on the whole example: toxic source versus Dolly benign source. It is not a per-turn toxicity rating.",
  "current_text_binding": "parse_conversation_turns reads eng_Latn. Hindi parallels are present but are not the strings the loader currently treats as the conversation.",
  "transformation_constraints_unchanged": true
}

Transformation constraints are unchanged. Canonical, realistic-variant, and matched-noise conditions still share source IDs, labels, and splits.
