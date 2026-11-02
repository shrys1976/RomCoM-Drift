# English–Hindi XNLI alignment investigation

**Verdict: `PARTIALLY_RECOVERED`**

**Independent validation: `CONDITIONALLY_APPROVED`.** The recovery verdict is unchanged. Approval covers only the subset in `validation_report.md`. No training export was built.

A defensible subset of English–Hindi training pairs can be recovered. The full training split cannot. Dev and test pair at the same index. Test has one Hindi premise collapse across six rows, documented in the validation report.

| Split | Hindi rows | English rows | Verified pairs | Ambiguous | Unmatched |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 392,702 | 392,702 | 375,805 | 15,174 | 1,723 |
| Dev | 2,490 | 2,490 | 2,490 | 0 | 0 |
| Test | 5,010 | 5,010 | 5,010 | 0 | 0 |

Same-index train label agreement remains 135,096 / 392,702 (34.402%). That figure is a side effect of 105 local order breaks. It is not evidence that the corpora are unrelated, and it is not a pairing.

No experiment dataset was built. Source files were not modified.

## Phase 1 — Provenance and identifiers

### Schemas

IndicXNLI Hindi forward JSON (`forward/train/xnli_hi.json`, and the dev and test files) is an object with one split key. Each record has exactly three fields:

- `premise` (string)
- `hypothesis` (string)
- `label` (`0` entailment, `1` neutral, `2` contradiction)

There is no pair id, sentence id, translation id, or English text.

XNLI-MT English train (`XNLI-MT-1.0/multinli/multinli.train.en.tsv`) has a header and three fields:

- `premise`
- `hypo`
- `label` (`entailment`, `neutral`, `contradictory`)

There is no pair id. The file has 392,702 data rows when read with `csv.QUOTE_NONE`, which is how Hugging Face `datasets` 1.16.1 reads it.

XNLI 1.0 dev and test (`xnli.dev.tsv`, `xnli.test.tsv`) are multilingual. English rows carry `pairID`, `promptID`, `genre`, `gold_label`, `sentence1`, and `sentence2`. Those ids exist only on dev and test.

IndicXNLI backward dev and test are Hindi-to-English round trips of the forward files. They are not the source English. Backward train is absent from the local snapshot, from the Hugging Face dataset revision `7092c27872e919f31d0496fb8b9c47bd2cba3f6c`, and from the GitHub tree (`data/` contains only `.gitignore`).

### How the Hindi file was produced

`divyanshuaggarwal/IndicXNLI` `src/make_dataset.py` does this:

1. `xnli = load_dataset("xnli", "en")` with no dataset revision pin.
2. `requirements.txt` pins the library at `datasets==1.16.1`.
3. Premises and hypotheses are translated separately, in list order, by IndicTrans.
4. Each output record stores the translated premise, the translated hypothesis, and `dataset["train"][i]["label"]`.
5. The saved fields are only `premise`, `hypothesis`, and `label`.

The library script at tag 1.16.1 (`datasets/xnli/xnli.py`) reads `XNLI-MT-1.0/multinli/multinli.train.{lang}.tsv` in file order for train, and `XNLI-1.0` dev/test filtered to the requested language. It does not keep `pairID` on the train split. Dev and test yield order is file order.

A current reload of `facebook/xnli` `en` matches this TSV at every index (established in the earlier investigation, revision `b8dd5d7af51114dbda02c0e3f6133f332186418e`). Re-running an old library version today still downloads the current Hub snapshot, so it does not recover a 2021 cache.

### Why train same-index agreement is 34.4%

The Hindi train label sequence is the English train label sequence with 105 single-row gaps. The labels of those 105 English rows reappear, in the same order, as the last 105 Hindi labels. Unique 25-label anchors (each occurs once) track a monotonic offset that grows from 0 to +105.

The first gap is at English row 6073. Everything before that matches. After a one-row shift, same-index comparison is chance (about 1/3). Together:

`6073 / 392702 + (1/3) * (392702 - 6073) / 392702 ≈ 34.4%`

which is the observed 135,096 / 392,702.

Text around those gaps shows the label diff skips the wrong row inside a short bubble (observed bubble lengths 0–6). Example: Hindi row 6070 is the translation of the Lawrence premise, while a label-only alignment assigns it to a beach premise a few rows away. Label order is a useful mask for where the file is locally shifted. It is not a safe row pairing.

The 2021 loader cache that would explain why those 105 rows moved is not recoverable from the files on hand. Pairing below does not depend on that explanation.

### Alignment mechanisms

| Mechanism | Result |
| --- | --- |
| Original pair id / sentence id | Train files do not contain any. Exact identifier matches: 0. |
| Exact English–Hindi string match | 0 shared premise strings. The texts are in different scripts. |
| Whitespace-normalized cross-lingual match | Same result. Normalization cannot align Devanagari to English. |
| Same index on train | 34.402% label agreement, and text at the 105 bubbles is the wrong sentence. Rejected as a pairing. |
| Same index on dev and test | Label agreement 2,490/2,490 and 5,010/5,010. Every premise is repeated, and every repetition has one Hindi premise. Accepted, because the premise check passes for every group and each `(premise, label)` slot is unique. |
| Backward-train English as a join key | File is not published. |
| Backward dev English vs XNLI English | Round-trip exact match is partial (premise 477/2,490, hypothesis 1,230/2,490 on dev). Useful as a spot illustration, not as a join key. |
| Monotonic label alignment | Explains the 34.4% figure. Mis-pairs rows inside the 105 bubbles. Used only to choose which copies are safe to learn a translation from. |
| Embedding / fuzzy similarity | Not used. Not treated as evidence. |
| Repeated-premise translation identity, then a unique label slot | The train method. Described below. |

## Phase 2 — Train matching method

IndicTrans translates the premise list on its own. The same English premise must come out as the same Hindi premise wherever it occurs, apart from the short broken regions.

English train premise multiplicity:

| Copies of the premise | Premises |
| --- | ---: |
| 3 | 125,773 |
| 6 | 1,810 |
| 9 | 493 |
| 12 | 6 |
| 2 | 2 |
| 1 | 10 |

Almost every premise is repeated, so the translation can be checked instead of assumed.

Procedure (`src/romcom_data/xnli_alignment.py`, `scripts/align_en_hi_xnli.py`):

1. Build a provisional label alignment and record the 105 skipped English indexes.
2. A row is safe when it is not in the Hindi tail and its provisional English index is more than 20 rows away from a skip.
3. Learn `English premise → Hindi premise` only when at least two safe copies exist and they all share one Hindi string.
4. Drop a Hindi string if two different English premises learned it (198 such strings).
5. Drop an English premise whose safe copies disagree (5 premises).
6. Group the remaining rows by that Hindi premise and by label.
7. If the group contains one English row and one Hindi row, the pair is **verified**.
8. If several rows share the premise and the label, the premise is corroborated and the hypothesis is **ambiguous**. No row is chosen.
9. Otherwise the row stays **unmatched**.

Learned premise dictionary: 127,552 English premises.

This check is independent of the hypothesis text. A wrong global shuffle would make repeated copies of one English premise land on different Hindi premises, and step 3 would refuse them.

Hypothesis pairing inside a verified group is the unique `(premise, label)` slot. MultiNLI often has one entailment, one neutral, and one contradiction hypothesis per premise, so the slot is unique. Where a premise has two hypotheses with the same label (the 6-copy, 9-copy, and 12-copy premises), the slot is not unique and both rows stay ambiguous.

Verified label counts: entailment 125,268, neutral 125,269, contradiction 125,268. Label agreement inside verified pairs is 375,805 / 375,805 because the matcher requires it. That agreement is a filter, not separate semantic proof.

## Phase 3 — Train audit

| Quantity | Count |
| --- | ---: |
| Hindi training examples | 392,702 |
| English training examples | 392,702 |
| Exact identifier matches | 0 |
| Exact cross-lingual premise string matches | 0 |
| Verified pairs | 375,805 (95.697%) |
| Ambiguous matches | 15,174 (3.864%) |
| Unmatched Hindi examples | 1,723 (0.439%) |
| Unmatched English examples | 1,723 |
| English rows that sit only in an ambiguous group | 15,174 |
| Label agreement among verified pairs | 375,805 / 375,805 |
| Premises rejected because safe copies disagreed | 5 |
| Hindi premise strings rejected as collisions | 198 (1,449 rows) |
| Premise/label count conflicts | 6 rows |
| Rows with no corroborated premise | 268 |

Unmatched 1,723 = 1,449 collision rows + 268 uncorroborated rows + 6 count conflicts.

The 6 conflicts are two Hindi copies of one General Accounting Office address string for each of the three labels, against a single English row per label. Both Hindi copies are withheld.

### Semantic checks on verified train pairs

Digit retention, over verified fields whose English side contains a digit (70,302 fields):

- Exact digit sequence preserved: 62,311 (88.633%)
- Every English digit present in the Hindi field: 65,407 (93.037%)

Mismatches that were read are translations, not random sentences: `1st` becomes `पहला`, `10` becomes `दस`, and `4 million` becomes `40 लाख`. One English premise contains a stray `73` (`was it 73 the baser mainspring`) which the Hindi translation omits.

A 40-row verified sample, 15 ambiguous rows, 15 unmatched rows, and 15 label-order false matches are in `review_sample.csv`. Readings are in `validation_report.md`.

### Dev and test control

Both splits are intact:

- Dev: 830 / 830 repeated premises have one Hindi premise at the same indexes. Labels agree 2,490 / 2,490. Every row belongs to a repeated premise. Every `(premise, label)` occurs once.
- Test: 1,670 / 1,670 premises unanimous. Labels agree 5,010 / 5,010. English `(premise, label)` slots are unique. Hindi has 3 slots that each contain 2 rows, because two near-paraphrase premises collapsed to one Hindi premise.

Same-index pairing is accepted on these splits because that premise test passes for every group.

## Phase 4 — Derived data

Written under `romcomx_data/nli_alignment/`. Source JSON and TSV files were not edited.

| File | Role |
| --- | --- |
| `verified_pairs.jsonl` | 383,305 verified pairs (375,805 train, 2,490 dev, 5,010 test) |
| `candidate_alignment.csv` | All 392,702 Hindi train rows, with status, method, evidence, and candidate English indexes |
| `review_sample.csv` | Manual review sample, including label-order false matches |
| `alignment_statistics.json` | Machine-readable counts |
| `validation_report.md` | Independent validation and the approval decision |
| `independent_validation_statistics.json` | Reconciled counts and premise-uniqueness audit |
| `semantic_review_sample.csv` | 255-row semantic review with judgments |
| `conditional_approval_exclusions.json` | Row indexes withheld from the approved subset |
| `checksums.sha256` | SHA-256 of the sources and outputs |

Each verified train record stores English premise and hypothesis, Hindi premise and hypothesis, the NLI label, `english_row_index`, `hindi_row_index`, source paths, `matching_method`, `matching_evidence`, and `validation_status`. Train sources have no original pair id; the row index is the source locator, not a fabricated id. Dev and test records also store `english_pair_id` from XNLI 1.0.

Ambiguous and unmatched train rows are in the candidate table and are absent from `verified_pairs.jsonl`.

`matching_method` for verified train rows is `corroborated_premise_translation_and_unique_label`. For verified dev and test rows it is `same_index_corroborated_premise_and_unique_label`.

### Source revisions

- English train SHA-256: `bf2f6249149240d79b59ba649829dde49f7c9665efc9c9b818e78334c8f9d627`
- Hindi train SHA-256: `ae4d2d59c807a70e699fbc51ad2cc5bf5b1030c0cdfd443c3fa44c1b446a4b3c`
- Generation script: `https://github.com/divyanshuaggarwal/IndicXNLI` `src/make_dataset.py`
- Loader: Hugging Face `datasets` 1.16.1 `datasets/xnli/xnli.py`
- IndicXNLI Hub revision previously recorded: `7092c27872e919f31d0496fb8b9c47bd2cba3f6c`

Reproduce:

```bash
uv run python scripts/align_en_hi_xnli.py
uv run python scripts/validate_xnli_alignment.py
uv run python scripts/apply_semantic_review.py
uv run pytest tests/test_xnli_alignment.py -q
```

## What this verdict does not authorize

Independent validation conditionally approved a subset of these pairs. The subset is defined by the exclusions in `conditional_approval_exclusions.json`. These pairs are still not an experiment split, and no RomCoM-X training export was built. Ambiguous hypotheses were not forced into a one-to-one match. The 1,723 unmatched rows stay in the candidate table.

Hypothesis equivalence for a verified row is the unique label slot inside a premise whose Hindi wording is corroborated by repeated translation. It is not a second, independent string match of the hypothesis, except where digits or the manual sample provide that check. A reviewer who wants a stricter bar can withhold the verified file until the `review_sample.csv` judgments are filled in.
