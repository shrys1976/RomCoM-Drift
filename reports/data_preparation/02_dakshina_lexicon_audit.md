# 02 dakshina lexicon audit

- **Timestamp (UTC):** 2026-11-01T08:00:02.543300+00:00
- **Code version:** romcom_data 0.1.0
- **Source revision(s):** see manifests/source_manifest.json

## Recomputed train lexicon statistics

{'file': 'hi.translit.sampled.train.tsv', 'split': 'train', 'rows': 44204, 'columns': ['native', 'romanized', 'count'], 'unique_native': 25000, 'unique_romanized': 41344, 'missing_romanized_rows': 2, 'duplicate_native_roman_pairs': 0, 'attestation_min': 1, 'attestation_median': 2.0, 'attestation_mean': 1.9215908062618767, 'attestation_max': 10, 'native_with_multiple_variants': 14186, 'variant_count_distribution': {0: 1, 1: 10813, 2: 9927, 3: 3649, 4: 505, 5: 77, 6: 20, 7: 3, 8: 3, 9: 2}, 'missing_rows': [{'native': 'नं', 'romanized': nan, 'count': 3}, {'native': 'नान', 'romanized': nan, 'count': 2}]}

## Native word overlap (train/dev/test)

{'train_dev_overlap': 0, 'train_test_overlap': 0, 'dev_test_overlap': 0}

## Sentence-level dev/test

{'dev': {'native_rows': 5000, 'roman_rows': 5000, 'length_mismatch': False, 'duplicate_native': 28}, 'test': {'native_rows': 5000, 'roman_rows': 5000, 'length_mismatch': False, 'duplicate_native': 24}, 'rejoined_note': 'rejoined.tsv and aligned files are derived held-out sentence views; not independent training data.', 'rejoined_rows': 10000}

## Caveat

Dakshina counts are annotator attestation counts, not social-media typing probabilities.
