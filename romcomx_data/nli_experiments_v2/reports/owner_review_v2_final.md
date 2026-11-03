# Final V2 owner review (read-only verification)

**Review date:** 2026-11-03  
**Reviewer role:** Automated + manual sample pass (no JSONL regeneration)

## 1. Executive summary

V2 corrects the V1 per-token RNG defect, excludes 68 audited Dakshina `(native, variant)` mappings from EXP-02 selection, and rebuilds EXP-02/EXP-03 only. Automated validation passes with **0 issues**; frozen splits remain **byte-identical** to V1. **11 hardlinks** from V2 to V1 were replaced with independent copies (same SHA256, different inodes). EXP-02/EXP-03 **matched alignment is 100%** on all 2,036,028 selected-token pairs. The earlier **~35.8%** figure is **not** alignment failure; it is the rate at which unrelated EXP-03 noise tokens in the same field share the same random edit-index pattern (see §5–6).

**Decision:** `APPROVED_WITH_DOCUMENTED_CAVEATS` — safe to **freeze V2** and proceed to Kaggle QLoRA compatibility testing with Qwen3.5-2B, with paper caveats in §10.

## 2. V1/V2 hardlink and inode audit

| Relative path | Pre-fix | Post-fix |
|---------------|---------|----------|
| `english_retention/dev.jsonl`, `test.jsonl` | Shared inode with V1 | Independent inode |
| `exp00_reference/dev.jsonl`, `test.jsonl` | Shared | Independent |
| `exp01/both/{train,dev,test}.jsonl` | Shared | Independent |
| `exp01/premise_only/{dev,test}.jsonl` | Shared | Independent |
| `exp01/hypothesis_only/{dev,test}.jsonl` | Shared | Independent |
| `alignment_manifest.json` | Already separate inode | Unchanged |

**Total hardlinked JSONL files:** 11. **Remaining shared inodes (V1∩V2):** 0.

**Safety:** Before remediation, editing a hardlinked file via either path would mutate both trees. After remediation, V1 is an independently preserved baseline.

## 3. SHA256 preservation

- All 11 frozen JSONL paths: V1 hash == V2 hash (verified after copy break).
- `experiment_manifest.json` → `frozen_from_v1_sha256`: **0 mismatches**.
- `checksums.sha256`: **38 entries, 0 failures** (content unchanged on frozen files; README label fix only).

## 4. EXP-02/EXP-03 alignment (concept A)

For each selected transformation, paired traces must match on field, `token_index`, `native_offsets`, and Levenshtein distance ≥ 1.

| Check | Result |
|-------|--------|
| Paired trace checks | **2,036,028 / 2,036,028 (100%)** |
| `validate_experiments()` | PASS |
| Manual sample (100 rows, seed 20260303) | **0 alignment issues** |

## 5. Clarification of the ~35.8% statistic

**Misleading label (corrected in V2 README):** “EXP-03 identical position pairs ~35.8%” sounds like EXP-02/03 misalignment. It is not.

**Actual metric:** Among unordered pairs of **EXP-03 noise tokens within the same Romanized field** that share the same **canonical length** and the same **target distance**, the fraction where the **index tuples** of character edits (`canonical[i] != selected[i]`) are **identical**.

| Split | Identical pattern pairs | Eligible pairs | Rate |
|-------|-------------------------|----------------|------|
| train | 144,928 | 404,763 | 35.81% |
| dev | 641 | 1,817 | 35.28% |
| test | 1,298 | 3,621 | 35.85% |
| **all** | **146,867** | **410,201** | **35.80%** |

This is expected under sequential RNG noise: many short tokens at distance 1 share a single edit index. V1’s bug produced ~100% identity on this diagnostic; V2 ~36% confirms the RNG fix.

## 6. Random-noise collision analysis (concept B)

| Metric | train | dev | test | total |
|--------|------:|----:|-----:|------:|
| Different canonicals, same noise string in one field (`exp03_noise_collision_pairs`) | 443 | 2 | 9 | **454** |
| Repeated identical noise string within one field (pair count) | 1,792 | 12 | 23 | 1,827 |

Collisions are rare at dataset scale (~454 canonical pairs vs millions of noise tokens). They are documented for the control condition; no data change required.

## 7. Manual sample inspection (seed **20260303**)

Full IDs and traces: `reports/owner_review_v2_sample_ids.json` (50 train + 25 dev + 25 test).

**Automated checks on sample:** 0 excluded mappings; 0 pool/surface mismatches; EXP-02 selections are attested pool variants; EXP-03 shares positions with EXP-02 on all inspected rows.

**Representative examples:**

| example_id | Notes |
|------------|--------|
| `train-225592` | Partial field; dup native **यह**; informal attested **aur→or**, **ke→k**, **yah→yeh** |
| `train-134914` | Partial; **ke→k** (informal, not excluded) |
| `train-362556` | Partial; **sweden→swedan**, **kabja→kabza** |
| `dev-002162`, `dev-001996`, `test-*` (see JSON) | Dev/test partial/multi-eligible cases |

Qualitative: selected variants read as Dakshina-attested spelling/noise alternatives; unselected eligible tokens stay canonical in JSONL; English/latin tokens and punctuation unchanged in reviewed traces.

## 8. Exclusion and quarantine audit

**File:** `romcomx_data/inventories/dakshina_hi_exp02_variant_exclusions.json`

| Category | Count |
|----------|------:|
| Excluded | **68** (26 `wrong_word_substitution`, 42 `garbled_spelling`) |
| Quarantine (still in pool) | **5** |

Quarantine: क्षण/shan, क्षणों/shanon, क्षमा/shma, बंगाल/bangla, पंचर/puncher — all **present in pools**.

Exclusions are **mapping-level** `(native_word, variant)` triples; other variants for the same native remain eligible.

**EXP-02 V2 log scan:** **0** excluded mapping selections.

Informal lookalikes (e.g. **ke→k**, **aur→or**) remain by design.

## 9. Final validation and tests

| Gate | Result |
|------|--------|
| `scripts/validate_nli_experiments.py` | PASS (0 issues) |
| `pytest tests/test_nli_experiments.py tests/test_xnli_alignment.py` | 21 passed |
| Row counts vs manifest | Match |
| Approved alignment total | **383,059** (375,565 / 2,487 / 5,007) |
| Empirical p (train/dev/test) | 0.350 / 0.351 / 0.350 |
| Partial fields (train) | 602,973 |
| Stale train artifacts (EXP-00 / English) | None |

## 10. Remaining caveats (paper / training)

1. EN–HI alignment remains **conditionally approved** (see `nli_alignment/validation_report.md`); not re-litigated here.
2. **Quarantine** mappings remain selectable; flag in analyses if needed.
3. Dakshina **OOV ~14.6%** and fallback romanization on uncovered tokens.
4. Residual Dakshina tail quality beyond the 68 exclusions (not re-audited exhaustively).
5. EXP-03 noise string collisions (~454 cross-canonical pairs) are expected at low rate.

## Remediation performed during this review

- Replaced 11 V2→V1 hardlinks with independent copies (content unchanged; V1 untouched).
- Clarified README §7 (35.8% label).

**No experiment JSONL bytes were edited except inode-breaking copies of frozen files (hash-identical to prior content).**

---

`APPROVED_WITH_DOCUMENTED_CAVEATS`
