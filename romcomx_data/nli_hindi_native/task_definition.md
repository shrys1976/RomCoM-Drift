# Hindi-native NLI task definition

This experiment measures **Romanized Hindi semantic understanding**.

It does **not** establish Hindi–English code-mixed semantic understanding.

Each IndicXNLI Hindi premise–hypothesis pair is an independent NLI observation.
Official Hindi train, validation (dev file), and test partitions are kept.
No English counterpart is required, and none is joined to these rows.

EXP-01, EXP-02, and EXP-03, when later authorized, transform the Hindi premise and hypothesis only.
Labels and official split assignments stay fixed across conditions.

## Unresolved provenance

The original English MultiNLI training snapshot used to generate IndicXNLI could not be recovered. Train English mappings (392,702) remain withheld. No further fuzzy, index-based, or approximate Hindi-English alignment is authorized.

The existing 41-example bilingual dev review (34 pass, 1 hypothesis fail, 6 uncertain) is supporting evidence only. It is not proof that the corpus is aligned to English.
