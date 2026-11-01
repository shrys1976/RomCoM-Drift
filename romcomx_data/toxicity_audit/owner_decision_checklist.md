# Owner decision checklist

Verdict remains CONDITIONAL_GO. Licenses were approved on 2026-11-01 and are not flattened.

## Recorded in this freeze

- [x] IndicAlign CC-BY-4.0 approved, kept separate from Dolly-15K CC-BY-SA-3.0.
- [x] Dolly-15K upstream CC-BY-SA-3.0 approved for the benign control only.
- [x] Experiment name: Hindi refusal-response generation with Romanized assistant targets.
- [x] Hindi Devanagari user input stays unchanged.
- [x] Not Romanized-input understanding. Not general code-mixed safety robustness.
- [x] Five benign embedding ties retained on source-list role order. Not quarantined.
- [x] Two toxic role-conflict rows excluded. Roles were not flipped.
- [x] Toxic user prompts grouped before the split. Groups do not cross the frozen split.
- [x] Source manifest frozen. Augmentation, training, and scoring were not run.
- [x] Future scores use the generation rubric. Classification accuracy is not an outcome.

## Confirm before any augmentation

- [ ] Accept manifest `85bddf9a31a24bfff93eca66e6171c7bbeabd8c37e5d20f8cf10a21bca94b830` as the freeze.
- [ ] Accept retained 6998 rows: toxic 4998, benign 2000.
- [ ] Accept excluded 2 toxic role-conflict rows.
- [ ] Accept frozen splits: train 6299, validation 699, test 0.
- [ ] Accept tie disposition: retain_on_source_list_order.
- [ ] Accept that benign over-refusal is a required generation measure.

## Do not approve

- Classification accuracy as a result.
- A claim about Romanized-input understanding.
- A claim of general code-mixed safety robustness.
