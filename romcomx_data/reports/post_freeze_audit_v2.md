# Post-freeze audit: RomCoM Drift NLI V2

**Audit date:** 2026-11-07
**Frozen snapshot:** `romcomx_data/nli_experiments_v2/`
**Read-only:** No files under V1 or V2 were modified.

## Checksum verification (post-audit)

- Manifest: `romcomx_data/nli_experiments_v2/checksums.sha256`
- Entries checked: 40
- Result: **PASS**

---

## Check A: Train / dev / test disjointness

### Methodology

Streamed `example_id` from `exp01/both/`, `exp02/both/`, and `exp03/both/` for each split. Computed pairwise set intersections across splits per condition. Compared `nli_label` and ID sets across EXP-01/02/03 within each split (same IDs across conditions are expected).

### Split sizes (EXP-01 / EXP-02 / EXP-03; identical per split)

**EXP01**
| Split | Distinct example_id count |
| --- | --- |
| train | 375565 |
| dev | 2487 |
| test | 5007 |

**EXP02**
| Split | Distinct example_id count |
| --- | --- |
| train | 375565 |
| dev | 2487 |
| test | 5007 |

**EXP03**
| Split | Distinct example_id count |
| --- | --- |
| train | 375565 |
| dev | 2487 |
| test | 5007 |

### Pairwise intersections (must be 0)

**EXP01**
| Pair | Intersection size |
| --- | --- |
| dev∩test | 0 |
| train∩dev | 0 |
| train∩test | 0 |

**EXP02**
| Pair | Intersection size |
| --- | --- |
| dev∩test | 0 |
| train∩dev | 0 |
| train∩test | 0 |

**EXP03**
| Pair | Intersection size |
| --- | --- |
| dev∩test | 0 |
| train∩dev | 0 |
| train∩test | 0 |

### Cross-condition consistency (within split)

- ID set mismatches (exp01 vs exp02/exp03): **0**
- Label mismatches: **0**

**Check A result:** **PASS**

---

## Check B: Quarantined mapping selections (EXP-02)

### Methodology

Scanned V2 `logs/transformation_log.{split}.jsonl` for `exp02_traces` with `operation: attested_variant`. Matched exact `(native_word, selected variant)` against the five quarantine pairs after NFC + casefold normalization. Native text taken from the matching `*_exp01_tokens` entry.

### Selection counts

| Mapping (native → variant) | Train | Dev | Test | Total | Distinct example_id |
| --- | --- | --- | --- | --- | --- |
| क्षण → shan | 180 | 1 | 2 | 183 | 180 |
| क्षणों → shanon | 42 | 0 | 0 | 42 | 41 |
| क्षमा → shma | 40 | 0 | 1 | 41 | 41 |
| बंगाल → bangla | 15 | 0 | 0 | 15 | 15 |
| पंचर → puncher | 0 | 0 | 0 | 0 | 0 |

All non-zero hits are **EXP-02 attested_variant** selections (not incidental string matches).

### Sample positions (non-zero mappings)

#### क्षण → shan (total 183)
- `train-002893` (train) field=hypothesis token_index=4 canonical=kshan selected=shan
- `train-004708` (train) field=premise token_index=12 canonical=kshan selected=shan
- `train-005086` (train) field=premise token_index=9 canonical=kshan selected=shan
- `train-007197` (train) field=premise token_index=4 canonical=kshan selected=shan
- `train-007812` (train) field=premise token_index=13 canonical=kshan selected=shan

#### क्षणों → shanon (total 42)
- `train-007641` (train) field=premise token_index=1 canonical=kshanon selected=shanon
- `train-012122` (train) field=hypothesis token_index=15 canonical=kshanon selected=shanon
- `train-012473` (train) field=hypothesis token_index=2 canonical=kshanon selected=shanon
- `train-022940` (train) field=premise token_index=24 canonical=kshanon selected=shanon
- `train-033339` (train) field=premise token_index=9 canonical=kshanon selected=shanon

#### क्षमा → shma (total 41)
- `train-004072` (train) field=premise token_index=13 canonical=kshama selected=shma
- `train-012335` (train) field=premise token_index=26 canonical=kshama selected=shma
- `train-012447` (train) field=premise token_index=2 canonical=kshama selected=shma
- `train-016666` (train) field=premise token_index=8 canonical=kshama selected=shma
- `train-017775` (train) field=premise token_index=0 canonical=kshama selected=shma

#### बंगाल → bangla (total 15)
- `train-067341` (train) field=premise token_index=2 canonical=bangal selected=bangla
- `train-086450` (train) field=premise token_index=1 canonical=bangal selected=bangla
- `train-111034` (train) field=premise token_index=13 canonical=bangal selected=bangla
- `train-116737` (train) field=premise token_index=3 canonical=bangal selected=bangla
- `train-159935` (train) field=hypothesis token_index=1 canonical=bangal selected=bangla

#### पंचर → puncher (total 0)
- No selections in V2 logs.

---

## Check C: EXP-03 noise vs attested Dakshina romanizations

### Methodology

Built attested set from `romcomx_data/inventories/dakshina_hi_variant_inventory.tsv`: all distinct `canonical_romanization` and `romanized_variant` values (41,344 normalized forms). Compared each EXP-03 `random_noise` trace `selected` string using NFC + casefold exact equality (no fuzzy match).

### A. Token-level collision rate

| Split | Matching occurrences | Total noise tokens | Rate % |
| --- | --- | --- | --- |
| train | 272862 | 1999519 | 13.646382 |
| dev | 1654 | 12123 | 13.643488 |
| test | 3430 | 24386 | 14.065447 |
| **all** | 277946 | 2036028 | 13.651384 |

### B. Unique-string collision rate

| Split | Matching unique strings | Total unique strings | Rate % |
| --- | --- | --- | --- |
| train | 5508 | 467570 | 1.178005 |
| dev | 569 | 8653 | 6.575754 |
| test | 897 | 15525 | 5.777778 |
| **all** | 5536 | 473990 | 1.167957 |

### C. Collision interpretation (representative examples)

Classes: `same_native` = attested form maps to source token's native word; `different_native` = exact attested string but other lemma; `multiple_natives` = inventory associates string with >1 native word.

Example pool class counts (up to 40 samples): {'different_native': 31, 'multiple_natives': 9}

- `train-000001` (train) hypothesis[2] canonical=`yaad` noise=`laad` class=different_native source_native=`याद` dakshina_natives=['लाद']
- `train-000003` (train) premise[3] canonical=`yah` noise=`yao` class=different_native source_native=`यह` dakshina_natives=['याओ']
- `train-000006` (train) premise[12] canonical=`ke` noise=`kg` class=different_native source_native=`के` dakshina_natives=['केजी']
- `train-000006` (train) premise[13] canonical=`saath` noise=`saakh` class=different_native source_native=`साथ` dakshina_natives=['साख']
- `train-000006` (train) premise[30] canonical=`ko` noise=`eo` class=different_native source_native=`को` dakshina_natives=['ईओ']
- `train-000007` (train) premise[1] canonical=`ke` noise=`kg` class=different_native source_native=`के` dakshina_natives=['केजी']
- `train-000007` (train) premise[7] canonical=`ke` noise=`kh` class=different_native source_native=`के` dakshina_natives=['केएच']
- `train-000011` (train) premise[4] canonical=`ke` noise=`ko` class=different_native source_native=`के` dakshina_natives=['को']
- `train-000012` (train) hypothesis[9] canonical=`ke` noise=`be` class=different_native source_native=`के` dakshina_natives=['बे']
- `train-000013` (train) hypothesis[1] canonical=`ke` noise=`ae` class=different_native source_native=`के` dakshina_natives=['एई']
- `train-000013` (train) hypothesis[16] canonical=`ke` noise=`kh` class=different_native source_native=`के` dakshina_natives=['केएच']
- `train-000013` (train) hypothesis[19] canonical=`diya` noise=`liya` class=different_native source_native=`दिया` dakshina_natives=['लिया']
- `train-000014` (train) premise[3] canonical=`ke` noise=`te` class=multiple_natives source_native=`के` dakshina_natives=['टीई', 'ते']
- `train-000014` (train) premise[18] canonical=`ke` noise=`ye` class=different_native source_native=`के` dakshina_natives=['ये']
- `train-000020` (train) premise[8] canonical=`ki` noise=`ek` class=multiple_natives source_native=`की` dakshina_natives=['इक', 'एक']

### Limitations

- Collisions are exact inventory membership; homographs across lemmas count.
- Canonical-only forms in the inventory are included in the attested set.
- Random noise that matches an attested form for a **different** native word still counts as a token-level collision.

---

## Summary

| Check | Result | Issues found |
| --- | --- | --- |
| Train/dev/test disjointness | PASS | 0 overlapping IDs across splits |
| Quarantined mapping selections | Counts recorded | See Check B table; total selections 281 |
| EXP-03 attested-string collisions | Counts recorded | 277946/2036028 token occurrences (13.6514%) |

Prior owner review: `romcomx_data/nli_experiments_v2/reports/owner_review_v2_final.md`

