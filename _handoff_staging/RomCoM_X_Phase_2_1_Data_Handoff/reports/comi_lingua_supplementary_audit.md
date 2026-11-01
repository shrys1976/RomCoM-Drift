# comi lingua supplementary audit

- Timestamp (UTC): 2026-11-01T11:11:55.375726+00:00
- Code version: romcom_data 0.1.0

## Role

Supplementary Hinglish reference only. **Not** used as NLI/paraphrase/toxicity labels.

## Findings

{
  "repository": "LingoIITGN/COMI-LINGUA",
  "role": "supplementary_reference_only",
  "caveat": "Research preview; not task-label source for NLI/paraphrase/toxicity.",
  "configs": {
    "MT": {
      "splits": {
        "train": {
          "rows": 24692,
          "columns": [
            "Sentences",
            "Predicted_en_translation",
            "Predicted_RH_translation",
            "Predicted_DH_translation",
            "Annotator_1_en_translation",
            "Annotator_1_RH_translation",
            "Annotator_1_DH_translation",
            "Annotator_2_en_translation",
            "annotator2_RH_translation",
            "Annotator_2_DH_translation",
            "Annotator_3_en_translation",
            "annotator3_RH_translation",
            "Annotator_3_DH_translation"
          ]
        },
        "test": {
          "rows": 5000,
          "columns": [
            "Sentences",
            "Predicted_en_translation",
            "Predicted_RH_translation",
            "Predicted_DH_translation",
            "Annotator_1_en_translation",
            "Annotator_1_RH_translation",
            "Annotator_1_DH_translation",
            "Annotator_2_en_translation",
            "annotator2_RH_translation",
            "Annotator_2_DH_translation",
            "Annotator_3_en_translation",
            "annotator3_RH_translation",
            "Annotator_3_DH_translation"
          ]
        }
      }
    },
    "TN": {
      "splits": {
        "train": {
          "rows": 24547,
          "columns": [
            "Sentences",
            "Predicted Tags",
            "Annotated by: Annotator 1",
            "Annotated by: Annotator 2",
            "Annotated by: Annotator 3"
          ]
        },
        "test": {
          "rows": 5000,
          "columns": [
            "Sentences",
            "Predicted Tags",
            "Annotated by: Annotator 1",
            "Annotated by: Annotator 2",
            "Annotated by: Annotator 3"
          ]
        }
      }
    },
    "MLI": {
      "splits": {
        "train": {
          "rows": 26225,
          "columns": [
            "Sentences",
            "Annotated by: Annotator 1",
            "Annotated by: Annotator 2",
            "Annotated by: Annotator 3"
          ]
        },
        "test": {
          "rows": 5000,
          "columns": [
            "Sentences",
            "Annotated by: Annotator 1",
            "Annotated by: Annotator 2",
            "Annotated by: Annotator 3"
          ]
        }
      }
    }
  },
  "mt_train_test_source_overlap": 0,
  "mt_annotator_distinct_strings": {
    "1": 16476,
    "2": 6795,
    "3": 1421
  },
  "mt_pairwise_character_similarity": {
    "Annotator_1_RH_translation_vs_annotator2_RH_translation": {
      "mean": 0.9941865475983142,
      "median": 1.0
    },
    "Annotator_1_RH_translation_vs_annotator3_RH_translation": {
      "mean": 0.9774175890229089,
      "median": 1.0
    },
    "annotator2_RH_translation_vs_annotator3_RH_translation": {
      "mean": 0.9766012100969018,
      "median": 1.0
    }
  },
  "Annotator_1_RH_translation_script": {
    "latin": 24667,
    "devanagari": 21,
    "mixed": 4
  },
  "annotator2_RH_translation_script": {
    "latin": 24672,
    "devanagari": 19,
    "mixed": 1
  },
  "annotator3_RH_translation_script": {
    "latin": 24682,
    "devanagari": 8,
    "mixed": 2
  }
}

MT annotator differences are not treated as word-level romanization variants.
