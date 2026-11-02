# Independent validation of the English–Hindi XNLI alignment

**Decision: `CONDITIONALLY_APPROVED`**

The recovered row alignment is reliable on a documented subset. The full verified train file is not approved as-is. No source file was edited, and no training export was built.

Approved pairs:

| Split | Aligned rows in `verified_pairs.jsonl` | Approved |
| --- | ---: | ---: |
| Train | 375,805 | 375,565 |
| Dev | 2,490 | 2,487 |
| Test | 5,010 | 5,007 |
| Total | 383,305 | 383,059 |

The approved train count removes 186 identical-text rows that carry conflicting labels, 34 degenerate repeated-token translations, and 20 further train rows from the semantic sample whose Hindi text does not preserve the English meaning. One semantic failure (sample 165) is already inside the 34. Dev drops 3 meaning failures. Test drops 3 meaning failures. Indexes are in `conditional_approval_exclusions.json`.

Questionable translations stay in the approved subset. Their row match is the right English sentence, and the damage is partial. They are listed below.

## 1. Count reconciliation

`verified_pairs.jsonl` has 383,305 rows because it stores every split:

`375,805 train + 2,490 dev + 5,010 test = 383,305`

The reported 375,805 figure is the verified train count. Both numbers are exact. Text and indexes in the JSONL match the frozen sources (`jsonl_text_or_index_problems` is empty). Each split’s Hindi and English indexes are unique inside the file.

`candidate_alignment.csv` is train-only and has one row for every Hindi train index (392,702):

| Status | Rows | Method |
| --- | ---: | --- |
| verified | 375,805 | `corroborated_premise_translation_and_unique_label` |
| ambiguous | 15,174 | `corroborated_premise_translation_duplicate_label` |
| unmatched | 1,449 | `premise_translation_collision` |
| unmatched | 268 | `no_corroborated_premise` |
| unmatched | 6 | `premise_label_count_conflict` |

`375,805 + 15,174 + 1,723 = 392,702`. Verified English indexes in the candidate table are exactly the train indexes in the JSONL. Verified and non-verified English indexes do not overlap.

English train also has 392,702 rows. The candidate table names 390,982 of them:

`375,805 verified + 15,174 ambiguous English indexes + 3 conflict indexes = 390,982`

The other 1,720 English rows are never pointed at:

| Reason | Rows |
| --- | ---: |
| Premise learned, then dropped because the Hindi string was shared | 1,449 |
| Fewer than two safe supporting copies | 148 |
| Every copy sits inside the unsafe radius of a label-order skip | 108 |
| Safe copies disagreed | 15 |

`1,449 + 148 + 108 + 15 = 1,720`, and `390,982 + 1,720 = 392,702`.

The 15,174 ambiguous rows are 2,282 repeated premises: 1,791 premises with 6 copies, 488 with 9, and 3 with 12. Each label occurs more than once, so the matcher withholds every hypothesis. `1,791×6 + 488×9 + 3×12 = 15,174`.

The 6 conflict rows are two Hindi copies of the General Accounting Office address for each label, against one English row per label.

Dev and test JSONL rows cover every source index. Pair ids are unique: 2,490 / 2,490 and 5,010 / 5,010.

## 2. Premise uniqueness

On the 375,805 verified train pairs:

- 0 English premises map to more than one Hindi premise.
- 0 Hindi premises map to more than one English premise.
- 0 verified pairs use an English `(premise, label)` slot that is duplicated in the source.
- 0 verified pairs use a Hindi `(premise, label)` slot that is duplicated in the source.
- 0 verified slots are claimed twice.

So, inside the verified train subset, a premise translation is one-to-one, and the label picks one hypothesis on both sides.

That assumption fails outside the verified subset, and those rows were withheld:

- English source: 6,927 non-unique `(premise, label)` slots, 15,369 rows, maximum 4 rows in one slot.
- Hindi source: 7,443 non-unique slots, 16,629 rows, maximum 13.
- The verified file contains 125,269 distinct English premises. 125,267 of them occur exactly three times in English train, once per label. The other two occur twice, each with two distinct labels, and both copies are verified because each slot is still unique. One of those premises is the garbled “kembecis” sentence, whose two rows share the hypothesis “Na” under two labels and are in the 186-row exclusion. The other is the “Non-asthma respiratory…” heading.

Normalization did not create the map. Whitespace plus NFC merges 0 English premises and 0 Hindi premises. Casefold would merge 18 English premise pairs and 0 Hindi premises. The matcher did not casefold, so these stayed separate:

1. Executive Orders / Executive orders
2. Title II / title II (Unfunded Mandates)
3. title II of the Act / title II of the act
4. HOMELESS PEOPLE / homeless people
5. That 's odd / that 's odd
6. Evaluation of Limitations… / Evaluation of limitations…
7. Self-Contradiction / Self-contradiction
8. Incorporated / incorporated an initial regulatory flexibility analysis
9. A BRIEF HISTORY / A Brief History
10. Executive Orders / Executive orders (section iv)
11. some States / some states
12. What was that ? (6 copies) / what was that ? (3 copies)
13. INTERNET / Internet
14. the Rule / the rule
15. HERSHEIMMER / Hersheimmer
16. DEFINITIONS / Definitions
17. SCUBA / Scuba
18. the Act / the act (sections 203 and 204)

Duplicate premise+hypothesis text, before alignment:

| Side | Groups | Conflicting labels | Same label | Rows |
| --- | ---: | ---: | ---: | ---: |
| English | 112 | 84 | 28 | 244 |
| Hindi | 298 | 231 | 67 | 628 |

Among verified train pairs, 83 groups (186 rows) have the same English premise, the same English hypothesis, the same Hindi premise, and the same Hindi hypothesis, with more than one gold label. The label slot is unique, and the wording is not. Those 186 rows are excluded. A separate scan finds 34 verified train rows where one token is at least 40% of a field of 12 or more tokens. None of the 34 are in the 186.

The negation-cue screen flags 62,075 / 375,805 verified train rows. That is a cue mismatch between English `not/no/never/without/cannot/n't` and Hindi `नहीं/बिना/मत`. It is not an error count. The 20-row read of that screen was mostly false alarms. The real failures in it are samples 165 and 180.

## 3. Semantic review

`semantic_review_sample.csv` has 255 judged rows. Premise, hypothesis, and label were judged separately. Label agreement was not treated as proof. “Label” here means whether the Hindi pair still stands in the same relation as the English pair.

| Stratum | Rows |
| --- | ---: |
| One pair from each of the 105 order breaks | 105 |
| Entailment, neutral, contradiction | 8 each |
| Short premises, long premises, numbers, named entities | 12 each |
| Negation-cue screen | 20 |
| Dev, test | 24 each |
| Two-copy premises | 2 |
| Withheld duplicate-label premise | 2 |
| Test premise collapse | 6 |

The original sampler’s `repeated_premise_group` pool was empty: every premise with 6 or more copies is ambiguous and is absent from the verified file. The two-copy rows and the withheld Place des Vosges rows cover that case.

| Reading | Rows |
| --- | ---: |
| Equivalent on premise, hypothesis, and label | 192 |
| Questionable, with no field marked not equivalent | 34 |
| At least one field not equivalent | 27 |
| Withheld, no pair selected | 2 |

All 105 order-break pairs match the verified English sentence. None of them is the label-order neighbor. Meaning failures inside that set are machine-translation errors on the correct row.

Rows with a field marked not equivalent:

| ID | Split | Hindi index | Failure |
| --- | ---: | ---: | --- |
| 2 | train | 11552 | Abdication became expulsion; the proclamation was dropped |
| 4 | train | 21841 | Gore became fair-skinned |
| 11 | train | 36700 | “I've had enough” is not the Hindi premise |
| 32 | train | 141314 | “seat of his pants” was literalized; the contradiction is lost |
| 42 | train | 173430 | Fascination was reversed into a hobby; entailment fails |
| 64 | train | 249421 | “bow out” became “bend”; entailment fails |
| 82 | train | 301297 | “ran out of time” was dropped |
| 84 | train | 303039 | “never been happier” became “were never happy” |
| 90 | train | 322181 | “garrison” and “permanently” were garbled away |
| 93 | train | 328434 | “hold the line” became “support”; entailment flips |
| 98 | train | 364239 | “pat down” became belittle / slap |
| 110 | train | 25433 | “career years” became a career lasting years |
| 111 | train | 103210 | “questionnaires only” reversed its scope |
| 125 | train | 341149 | “pottering” became making toys |
| 133 | train | 205389 | “DILUTION” was dropped |
| 144 | train | 384001 | Band clauses were crossed |
| 156 | train | 166920 | “I always knew it” was dropped, so the entailment fails |
| 165 | train | 346915 | Hypothesis collapsed to a repeated token |
| 180 | train | 299301 | Clinton became Hillary |
| 181 | train | 294975 | “No John Grisham” was parsed as a name |
| 192 | train | 229344 | “financial industry” became “the world” |
| 213 | dev | 69 | Monroe became मोनकॉर्ड |
| 215 | dev | 2284 | “do not stay put” became “do not remain”; the contradiction moves toward entailment |
| 217 | dev | 816 | “did not trouble to rise” became “had no trouble rising” |
| 230 | test | 612 | “A bunch of would” became “a bunch” |
| 231 | test | 1978 | “after a fashion” was literalized; entailment fails |
| 251 | test | 319 | “approve every deal” became “complete every deal” |

Questionable sample ids, kept in the approved subset: 0, 1, 6, 10, 12, 17, 21, 25, 27, 45, 47, 50, 55, 57, 61, 65, 68, 74, 76, 89, 100, 107, 115, 119, 121, 135, 142, 148, 206, 219, 238, 245, 246, 250. Notes are on those CSV rows. Sample 95 is equivalent and keeps the English typo “hour house” as घंटे के घर.

This is a stratified read, not a read of all 375,805 train pairs. The 27 failures are excluded. They are evidence that translation quality varies on the correct source row.

## 4. Dev and test

Every dev row and every test row was checked structurally.

Dev: 830 / 830 premise groups are unanimous, labels agree 2,490 / 2,490, and both sides have 0 non-unique `(premise, label)` slots. The 24-row semantic sample plus the full structural check supports the split as a bilingual benchmark after dropping Hindi indexes 69, 816, and 2284.

Test: 1,670 / 1,670 premise groups are unanimous and labels agree 5,010 / 5,010. English slots are unique. Hindi has 3 non-unique slots (6 rows). Two near-paraphrase premises, “I don't care how you do it” and “I don't care how you get it done,” share the Hindi premise मुझे परवाह नहीं है कि आप इसे कैसे करते हैं। The hypotheses stay different and match the same index. Pair ids 487, 488, 489, 625, 626, and 627. Those six rows were read. Row 319 is not approved, because “approve” became “complete.” Rows 612 and 1978 are the other test meaning failures. The remaining 5,007 test rows are approved as same-index alignments. They were not each read by hand.

## 5. What remains withheld

- 15,174 ambiguous train rows
- 1,723 unmatched train rows
- 1,720 English train rows with no candidate pointer
- 186 verified rows with identical text and conflicting labels
- 34 verified rows with degenerate repetition
- The 27 semantic failures in the table above, one of which is already in the 34

Digit checks on the earlier alignment report still stand: of 70,302 verified fields with an English digit, 62,311 keep the digit sequence and 65,407 keep every English digit. Known rewrites are `1st` → पहला, `10` → दस, and `4 million` → 40 लाख.

## Reproduce

```bash
uv run python scripts/validate_xnli_alignment.py
uv run python scripts/apply_semantic_review.py
uv run pytest tests/test_xnli_alignment.py -q
```

`validate_xnli_alignment.py` reloads the frozen sources, recounts every artifact, audits premise maps, normalization, duplicate pairs, and both evaluation splits, and leaves a judged review sample in place. `apply_semantic_review.py` records the manual judgments and rewrites the exclusion indexes. The unit tests cover the JSONL identity `375,805 + 2,490 + 5,010 = 383,305`.
