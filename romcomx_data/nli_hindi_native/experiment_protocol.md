# Revised NLI experiment protocol

## Claim

Romanized Hindi semantic understanding on official IndicXNLI Hindi partitions.

## Not claimed

Hindi–English code-mixed semantic understanding. English XNLI validation and test stay a separate language-retention benchmark and are not treated as parallel rows.

## Data

- Hindi premise, hypothesis, and label from `forward__*__xnli_hi.json`.
- Splits: train, validation (dev file), test. No regrouping.
- Source IDs: `indicxnli-hi:{split}:{row_index}` from Hindi file order.
- EXP-01/02/03 (not built in this step) share those IDs, labels, and splits. Only the romanization of premise and hypothesis may differ.

## English benchmark

{
  "role": "independent_language_retention_benchmark",
  "claim": "not_parallel_to_indicxnli_hindi_rows",
  "files": [
    "romcomx_data/raw/xnli_mt/XNLI-1.0/xnli.dev.tsv",
    "romcomx_data/raw/xnli_mt/XNLI-1.0/xnli.test.tsv"
  ]
}

## Provenance limit

The original English MultiNLI training snapshot used to generate IndicXNLI could not be recovered. Train English mappings (392,702) remain withheld. No further fuzzy, index-based, or approximate Hindi-English alignment is authorized.

## This step

Audit and protocol only. No dataset rebuild and no training.
