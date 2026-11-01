# Owner decision checklist

Verdict remains CONDITIONAL_GO. Task definition and source decisions approved. Licenses were approved on 2026-11-01 and are not flattened.

## Recorded in this freeze

- [x] IndicAlign CC-BY-4.0 approved, kept separate from Dolly-15K CC-BY-SA-3.0.
- [x] Dolly-15K upstream CC-BY-SA-3.0 approved for the benign control only.
- [x] Experiment name: Hindi refusal-response generation with Romanized assistant targets.
- [x] Hindi Devanagari user input stays unchanged.
- [x] Not Romanized-input understanding. Not general code-mixed safety robustness.
- [x] Five benign embedding ties retained on source-list role order. Not quarantined.
- [x] Two toxic role-conflict rows excluded. Roles were not flipped.
- [x] Toxic and benign user prompts grouped in one union-find before the split.
- [x] Frozen 80/10/10 train/validation/test. Validation is not the final test.
- [x] Source manifest frozen. Augmentation, training, and scoring were not run.
- [x] Future scores use the generation rubric. Benign over-refusal is required. Classification accuracy is not an outcome.

## Confirm before any augmentation

- [ ] Accept manifest `58d85ddaf78bb05600bc802049c8ce12ef8886f4439bfe87b9ca9fd25e1673ae` as the freeze.
- [ ] Accept retained 6998 rows: toxic 4998, benign 2000.
- [ ] Accept excluded 2 toxic role-conflict rows.
- [ ] Accept frozen splits: train 5585, validation 690, test 723.
- [ ] Accept tie disposition: retain_on_source_list_order.
- [ ] Accept that benign over-refusal is a required generation measure.

## Do not approve

- Using validation as the final held-out evaluation split.
- Classification accuracy as a result.
- A claim about Romanized-input understanding.
- A claim of general code-mixed safety robustness.
