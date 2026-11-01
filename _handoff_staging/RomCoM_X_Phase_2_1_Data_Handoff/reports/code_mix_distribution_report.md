# code mix distribution report

- Timestamp (UTC): 2026-11-01T11:11:55.376454+00:00
- Code version: romcom_data 0.1.0

## Measured mixing (not benchmark-equivalent)

Intensities are labeled `reconstructed_intensity_25/50/75`.

{
  "by_task_intensity": {
    "toxicity:reconstructed_intensity_25": {
      "hindi_origin_tokens": 119,
      "english_origin_tokens": 757895,
      "language_switches": 238,
      "eligible_for_switch": 119,
      "taxonomy_replacements": 0,
      "missing_dictionary": 0,
      "hindi_origin_proportion": 0.00015698918489632117,
      "english_origin_proportion": 0.9998430108151036,
      "replacement_rate": 0.0
    },
    "toxicity:reconstructed_intensity_50": {
      "hindi_origin_tokens": 13473,
      "english_origin_tokens": 744541,
      "language_switches": 25265,
      "eligible_for_switch": 484338,
      "taxonomy_replacements": 0,
      "missing_dictionary": 470865,
      "hindi_origin_proportion": 0.017774078051328867,
      "english_origin_proportion": 0.9822259219486711,
      "replacement_rate": 0.0
    },
    "toxicity:reconstructed_intensity_75": {
      "hindi_origin_tokens": 70703,
      "english_origin_tokens": 687311,
      "language_switches": 103685,
      "eligible_for_switch": 696547,
      "taxonomy_replacements": 0,
      "missing_dictionary": 625844,
      "hindi_origin_proportion": 0.09327400285482854,
      "english_origin_proportion": 0.9067259971451714,
      "replacement_rate": 0.0
    }
  },
  "zero_switch_examples": {
    "toxicity:reconstructed_intensity_25": {
      "count": 6884,
      "rate": 0.9834285714285714
    },
    "toxicity:reconstructed_intensity_50": {
      "count": 1889,
      "rate": 0.26985714285714285
    },
    "toxicity:reconstructed_intensity_75": {
      "count": 56,
      "rate": 0.008
    }
  }
}
