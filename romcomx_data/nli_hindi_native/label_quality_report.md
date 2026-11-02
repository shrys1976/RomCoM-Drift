# Label distribution and data quality

```json
{
  "n_examples": 400202,
  "label_distribution": {
    "test:contradiction": 1670,
    "test:entailment": 1670,
    "test:neutral": 1670,
    "train:contradiction": 130903,
    "train:entailment": 130899,
    "train:neutral": 130900,
    "validation:contradiction": 830,
    "validation:entailment": 830,
    "validation:neutral": 830
  },
  "empty_premise": 0,
  "empty_hypothesis": 0,
  "invalid_label": 0,
  "examples_devanagari_ratio_below_0_5": 126,
  "examples_latin_letter_ratio_above_0_2": 1305,
  "label_integrity": "labels taken from Hindi JSON only; not checked against English XNLI",
  "translation_quality": "Not scored against English. Hindi text is the observation. The 41-example bilingual dev review remains supporting evidence only.",
  "bilingual_dev_review": {
    "n": 41,
    "pass": 34,
    "hypothesis_fail_indices": [
      357
    ],
    "uncertain_indices": [
      684,
      777,
      844,
      1027,
      1327,
      1841
    ],
    "role": "supporting_evidence_not_corpus_alignment_proof"
  }
}
```
