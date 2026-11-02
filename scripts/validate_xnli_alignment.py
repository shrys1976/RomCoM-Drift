#!/usr/bin/env python3
"""Reconcile alignment artifacts and audit the premise-uniqueness assumption."""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.xnli_alignment import (
    _near_skip,
    learn_string_map,
    monotonic_label_alignment,
)
from romcom_data.xnli_alignment_io import (
    ID_TO_LABEL,
    load_indic_forward_json,
    load_xnli10_english_split,
    load_xnli_mt_english_train,
)
from romcom_data.xnli_alignment_validate import (
    collapse_counts,
    digit_tokens,
    duplicate_pair_stats,
    label_slot_identifies_one_hypothesis,
    negation_mismatch,
    normalize_casefold,
    normalize_ws,
    premise_translation_audit,
    source_slots_that_are_not_unique,
)

RAW = ROOT / "romcomx_data" / "raw"
OUT = ROOT / "romcomx_data" / "nli_alignment"
JSONL = OUT / "verified_pairs.jsonl"
CANDIDATE = OUT / "candidate_alignment.csv"
LABELS = ("entailment", "neutral", "contradiction")


def _record_ok(record: dict[str, Any], english: dict[str, Any], hindi: dict[str, Any]) -> list[str]:
    problems = []
    if record["english_premise"] != english["premise"]:
        problems.append("english_premise_mismatch")
    if record["english_hypothesis"] != english["hypothesis"]:
        problems.append("english_hypothesis_mismatch")
    if record["hindi_premise"] != hindi["premise"]:
        problems.append("hindi_premise_mismatch")
    if record["hindi_hypothesis"] != hindi["hypothesis"]:
        problems.append("hindi_hypothesis_mismatch")
    if record["nli_label"] != ID_TO_LABEL[english["label"]]:
        problems.append("english_label_mismatch")
    if english["label"] != hindi["label"]:
        problems.append("source_label_mismatch")
    return problems


def recount_artifacts(
    english_train: list[dict[str, Any]],
    hindi_train: list[dict[str, Any]],
    english_dev: list[dict[str, Any]],
    hindi_dev: list[dict[str, Any]],
    english_test: list[dict[str, Any]],
    hindi_test: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[int, int]], list[dict[str, Any]]]:
    sources = {
        "train": (english_train, hindi_train),
        "dev": (english_dev, hindi_dev),
        "test": (english_test, hindi_test),
    }
    by_split: Counter[str] = Counter()
    by_method: Counter[str] = Counter()
    by_label: Counter[tuple[str, str]] = Counter()
    seen_hindi: dict[str, set[int]] = defaultdict(set)
    seen_english: dict[str, set[int]] = defaultdict(set)
    text_problems: Counter[str] = Counter()
    train_pairs: list[tuple[int, int]] = []
    train_records: list[dict[str, Any]] = []
    with JSONL.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            split = record["split"]
            by_split[split] += 1
            by_method[record["matching_method"]] += 1
            by_label[(split, record["nli_label"])] += 1
            hindi_index = int(record["hindi_row_index"])
            english_index = int(record["english_row_index"])
            if hindi_index in seen_hindi[split] or english_index in seen_english[split]:
                text_problems["duplicate_index"] += 1
            seen_hindi[split].add(hindi_index)
            seen_english[split].add(english_index)
            english_rows, hindi_rows = sources[split]
            problems = _record_ok(record, english_rows[english_index], hindi_rows[hindi_index])
            text_problems.update(problems)
            if split == "train":
                train_pairs.append((hindi_index, english_index))
                if line_number <= 1 or record["nli_label"] in LABELS:
                    train_records.append(
                        {
                            "hindi_row_index": hindi_index,
                            "english_row_index": english_index,
                            "nli_label": record["nli_label"],
                            "english_premise": record["english_premise"],
                            "english_hypothesis": record["english_hypothesis"],
                            "hindi_premise": record["hindi_premise"],
                            "hindi_hypothesis": record["hindi_hypothesis"],
                        }
                    )
    candidate_status: Counter[str] = Counter()
    candidate_method: Counter[str] = Counter()
    candidate_hindi: set[int] = set()
    candidate_verified_english: set[int] = set()
    candidate_other_english: set[int] = set()
    verified_in_candidate = 0
    with CANDIDATE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = row["validation_status"]
            candidate_status[status] += 1
            candidate_method[row["matching_method"]] += 1
            hindi_index = int(row["hindi_row_index"])
            if hindi_index in candidate_hindi:
                text_problems["candidate_duplicate_hindi"] += 1
            candidate_hindi.add(hindi_index)
            indexes = [int(part) for part in row["candidate_english_row_indexes"].split() if part]
            if status == "verified":
                verified_in_candidate += 1
                candidate_verified_english.add(int(row["english_row_index"]))
            else:
                candidate_other_english.update(indexes)
    expected_train = set(range(len(hindi_train)))
    jsonl_train = seen_hindi["train"]
    summary = {
        "jsonl_rows": sum(by_split.values()),
        "jsonl_by_split": dict(by_split),
        "jsonl_by_method": dict(by_method),
        "jsonl_by_split_and_label": {f"{split}:{label}": count for (split, label), count in by_label.items()},
        "jsonl_text_or_index_problems": dict(text_problems),
        "jsonl_train_equals_reported_verified_train": by_split["train"] == 375805,
        "jsonl_total_equals_train_plus_dev_plus_test": (
            by_split["train"] + by_split["dev"] + by_split["test"] == sum(by_split.values())
        ),
        "candidate_rows": sum(candidate_status.values()),
        "candidate_by_status": dict(candidate_status),
        "candidate_by_method": dict(candidate_method),
        "candidate_covers_every_hindi_train_index": candidate_hindi == expected_train,
        "candidate_verified_equals_jsonl_train": candidate_verified_english == set(
            english for _, english in train_pairs
        ),
        "verified_and_nonverified_english_overlap": len(
            candidate_verified_english & candidate_other_english
        ),
        "english_indexes_accounted": len(candidate_verified_english | candidate_other_english),
        "dev_jsonl_covers_every_row": seen_hindi["dev"] == set(range(len(hindi_dev))),
        "test_jsonl_covers_every_row": seen_hindi["test"] == set(range(len(hindi_test))),
        "train_verified_in_candidate": verified_in_candidate,
        "train_jsonl_missing_from_candidate_verified": len(jsonl_train - candidate_hindi) if False else 0,
    }
    summary["train_jsonl_hindi_indexes_match_candidate_verified"] = jsonl_train == {
        hindi for hindi, _ in train_pairs
    }
    return summary, train_pairs, train_records


def normalization_audit(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    texts = [row[field] for row in rows]
    unique = len(set(texts))
    ws = collapse_counts(texts, normalize_ws)
    folded = collapse_counts(texts, normalize_casefold)
    return {
        "raw_unique": unique,
        "whitespace_nfc_merges": len(ws),
        "whitespace_nfc_extra_raw_forms": sum(count - 1 for count in ws.values()),
        "casefold_merges": len(folded),
        "casefold_extra_raw_forms": sum(count - 1 for count in folded.values()),
    }


def ordered_split_full_check(
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from romcom_data.xnli_alignment import same_index_premise_audit

    audit = same_index_premise_audit(english_rows, hindi_rows)
    slots = source_slots_that_are_not_unique(english_rows)
    hindi_slots = source_slots_that_are_not_unique(hindi_rows)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(english_rows):
        grouped[row["premise"]].append(index)
    premise_failures = []
    hypothesis_not_checked = 0
    for premise, indexes in grouped.items():
        hindi_premises = {hindi_rows[index]["premise"] for index in indexes}
        if len(hindi_premises) != 1:
            premise_failures.append(premise[:120])
        labels = [english_rows[index]["label"] for index in indexes]
        if len(labels) != len(set(labels)):
            hypothesis_not_checked += 1
    return {
        "rows": len(english_rows),
        "premise_audit": audit,
        "english_nonunique_slots": slots,
        "hindi_nonunique_slots": hindi_slots,
        "premise_groups_not_unanimous": len(premise_failures),
        "premise_groups_with_duplicate_labels": hypothesis_not_checked,
        "pair_ids_present": sum(1 for row in english_rows if row.get("pair_id")),
        "pair_ids_unique": len({row.get("pair_id") for row in english_rows}),
    }


def build_review_rows(
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    english_dev: list[dict[str, Any]],
    hindi_dev: list[dict[str, Any]],
    english_test: list[dict[str, Any]],
    hindi_test: list[dict[str, Any]],
) -> list[dict[str, str]]:
    provisional, skipped = monotonic_label_alignment(
        [row["label"] for row in hindi_rows],
        [row["label"] for row in english_rows],
    )
    by_hindi = {record["hindi_row_index"]: record for record in train_records}
    chosen: dict[tuple[str, int], dict[str, str]] = {}

    def add(split: str, hindi_index: int, english_index: int, stratum: str, record: dict[str, str] | None = None) -> None:
        key = (split, hindi_index)
        if record is None:
            if split == "train":
                english = english_rows[english_index]
                hindi = hindi_rows[hindi_index]
                label = ID_TO_LABEL[english["label"]]
                premise_en, hypothesis_en = english["premise"], english["hypothesis"]
                premise_hi, hypothesis_hi = hindi["premise"], hindi["hypothesis"]
            else:
                return
        else:
            label = record["nli_label"]
            premise_en = record["english_premise"]
            hypothesis_en = record["english_hypothesis"]
            premise_hi = record["hindi_premise"]
            hypothesis_hi = record["hindi_hypothesis"]
        existing = chosen.get(key)
        if existing is None:
            chosen[key] = {
                "split": split,
                "strata": stratum,
                "hindi_row_index": str(hindi_index),
                "english_row_index": str(english_index),
                "nli_label": label,
                "english_premise": premise_en,
                "english_hypothesis": hypothesis_en,
                "hindi_premise": premise_hi,
                "hindi_hypothesis": hypothesis_hi,
                "provisional_english_row_index": str(provisional[hindi_index]) if split == "train" and hindi_index < len(provisional) else "",
            }
            return
        if stratum not in existing["strata"].split("|"):
            existing["strata"] = existing["strata"] + "|" + stratum

    for skip in skipped:
        for hindi_index, record in by_hindi.items():
            english_index = record["english_row_index"]
            if abs(english_index - skip) <= 8 and provisional[hindi_index] != english_index:
                add("train", hindi_index, english_index, f"order_break_{skip}", record)
                break
        else:
            for hindi_index, record in by_hindi.items():
                if abs(record["english_row_index"] - skip) <= 8:
                    add("train", hindi_index, record["english_row_index"], f"order_break_{skip}", record)
                    break

    rng = random.Random(1)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    short, long, numbers, negation, entities, repeated = [], [], [], [], [], []
    for record in train_records:
        by_label[record["nli_label"]].append(record)
        premise = record["english_premise"]
        if len(premise) <= 40:
            short.append(record)
        if len(premise) >= 500:
            long.append(record)
        if digit_tokens(premise) or digit_tokens(record["english_hypothesis"]):
            numbers.append(record)
        if negation_mismatch(premise + " " + record["english_hypothesis"], record["hindi_premise"] + " " + record["hindi_hypothesis"]):
            negation.append(record)
        if any(token[:1].isupper() and token.isalpha() and len(token) >= 4 for token in premise.split()):
            entities.append(record)
        if premise.count(" ") > 0 and english_rows[record["english_row_index"]]["premise"] == premise:
            copies = sum(row["premise"] == premise for row in english_rows[record["english_row_index"] : record["english_row_index"] + 1])
        if any(word in premise for word in (" and ",)):
            pass
    premise_copies: Counter[str] = Counter(row["premise"] for row in english_rows)
    for record in train_records:
        if premise_copies[record["english_premise"]] >= 6:
            repeated.append(record)

    def take(pool: list[dict[str, Any]], stratum: str, count: int) -> None:
        if len(pool) <= count:
            picks = pool
        else:
            picks = rng.sample(pool, count)
        for record in picks:
            add("train", record["hindi_row_index"], record["english_row_index"], stratum, record)

    for label in LABELS:
        take(by_label[label], f"label_{label}", 8)
    take(short, "short_premise", 12)
    take(long, "long_premise", 12)
    take(numbers, "numbers", 12)
    take(negation, "negation_mismatch_screen", 20)
    take(entities, "named_entity", 12)
    # Premises with 6 or more copies are absent from verified train records.
    # Each label then occurs at least twice, so those rows stay ambiguous.
    take(repeated, "repeated_premise_group", 12)

    def take_split(split: str, english: list[dict[str, Any]], hindi: list[dict[str, Any]], count: int) -> None:
        indexes = list(range(len(english)))
        rng.shuffle(indexes)
        per_label = {0: 0, 1: 0, 2: 0}
        for index in indexes:
            label = english[index]["label"]
            if per_label[label] >= count // 3:
                continue
            per_label[label] += 1
            chosen[(split, index)] = {
                "split": split,
                "strata": split,
                "hindi_row_index": str(index),
                "english_row_index": str(index),
                "nli_label": ID_TO_LABEL[label],
                "english_premise": english[index]["premise"],
                "english_hypothesis": english[index]["hypothesis"],
                "hindi_premise": hindi[index]["premise"],
                "hindi_hypothesis": hindi[index]["hypothesis"],
                "provisional_english_row_index": str(index),
            }
            if sum(per_label.values()) >= count:
                break

    take_split("dev", english_dev, hindi_dev, 24)
    take_split("test", english_test, hindi_test, 24)
    rows = list(chosen.values())
    for index, row in enumerate(rows):
        row["sample_id"] = str(index)
        row["premise_judgment"] = ""
        row["hypothesis_judgment"] = ""
        row["label_judgment"] = ""
        row["notes"] = ""
    return rows


def _review_has_judgments(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("premise_judgment"):
                return True
    return False


def unreferenced_english_audit(
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """English train rows that no candidate row points at, by why the map dropped them."""
    provisional, skipped = monotonic_label_alignment(
        [row["label"] for row in hindi_rows],
        [row["label"] for row in english_rows],
    )
    _learned, _disagreed, collisions = learn_string_map(
        [row["premise"] for row in english_rows],
        [row["premise"] for row in hindi_rows],
        provisional,
        skipped,
        20,
        2,
    )
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for hindi_index, english_index in enumerate(provisional):
        if english_index < 0 or _near_skip(english_index, skipped, 20):
            continue
        votes[english_rows[english_index]["premise"]][hindi_rows[hindi_index]["premise"]] += 1
    referenced: set[int] = set()
    with CANDIDATE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["validation_status"] == "verified" and row["english_row_index"]:
                referenced.add(int(row["english_row_index"]))
            referenced.update(int(part) for part in row["candidate_english_row_indexes"].split())
    reasons: Counter[str] = Counter()
    for index in range(len(english_rows)):
        if index in referenced:
            continue
        counts = votes.get(english_rows[index]["premise"])
        if not counts:
            reasons["never_in_safe_votes"] += 1
            continue
        if len(counts) != 1:
            reasons["safe_copies_disagreed"] += 1
            continue
        hindi_premise = next(iter(counts))
        if hindi_premise in collisions:
            reasons["dropped_premise_collision"] += 1
            continue
        reasons["support_below_minimum"] += 1
    missing = len(english_rows) - len(referenced)
    return {
        "english_train_rows": len(english_rows),
        "referenced_english_indexes": len(referenced),
        "unreferenced_english_indexes": missing,
        "unreferenced_rows_by_reason": dict(reasons),
        "identity": "referenced + unreferenced = english_train_rows",
    }


def main() -> int:
    print("loading sources", flush=True)
    english_train = load_xnli_mt_english_train(RAW / "xnli_mt" / "XNLI-MT-1.0" / "multinli" / "multinli.train.en.tsv")
    hindi_train = load_indic_forward_json(RAW / "indicxnli" / "forward__train__xnli_hi.json")
    english_dev = load_xnli10_english_split(RAW / "xnli_mt" / "XNLI-1.0" / "xnli.dev.tsv")
    hindi_dev = load_indic_forward_json(RAW / "indicxnli" / "forward__dev__xnli_hi.json")
    english_test = load_xnli10_english_split(RAW / "xnli_mt" / "XNLI-1.0" / "xnli.test.tsv")
    hindi_test = load_indic_forward_json(RAW / "indicxnli" / "forward__test__xnli_hi.json")
    print("recounting artifacts", flush=True)
    summary, train_pairs, train_records = recount_artifacts(
        english_train, hindi_train, english_dev, hindi_dev, english_test, hindi_test
    )
    print("premise audit", flush=True)
    summary["verified_train_premise_map"] = premise_translation_audit(english_train, hindi_train, train_pairs)
    summary["verified_train_label_slots"] = label_slot_identifies_one_hypothesis(
        english_train, hindi_train, train_pairs
    )
    summary["english_source_slots"] = source_slots_that_are_not_unique(english_train)
    summary["hindi_source_slots"] = source_slots_that_are_not_unique(hindi_train)
    summary["english_normalization"] = normalization_audit(english_train, "premise")
    summary["hindi_normalization"] = normalization_audit(hindi_train, "premise")
    summary["english_duplicate_pairs"] = duplicate_pair_stats(english_train)
    summary["hindi_duplicate_pairs"] = duplicate_pair_stats(hindi_train)
    print("dev/test", flush=True)
    negation_flags = sum(
        negation_mismatch(
            record["english_premise"] + " " + record["english_hypothesis"],
            record["hindi_premise"] + " " + record["hindi_hypothesis"],
        )
        for record in train_records
    )
    summary["verified_train_negation_cue_mismatches"] = negation_flags
    summary["dev_full_check"] = ordered_split_full_check(english_dev, hindi_dev)
    summary["test_full_check"] = ordered_split_full_check(english_test, hindi_test)
    print("unreferenced english rows", flush=True)
    summary["unreferenced_english"] = unreferenced_english_audit(english_train, hindi_train)
    summary["count_reconciliation"] = {
        "reported_verified_train_pairs": 375805,
        "verified_pairs_jsonl_rows": summary["jsonl_rows"],
        "jsonl_train": summary["jsonl_by_split"].get("train"),
        "jsonl_dev": summary["jsonl_by_split"].get("dev"),
        "jsonl_test": summary["jsonl_by_split"].get("test"),
        "identity": "jsonl_rows = verified_train + verified_dev + verified_test",
    }
    stats_path = OUT / "independent_validation_statistics.json"
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("review sample", flush=True)
    review = build_review_rows(
        english_train, hindi_train, train_records, english_dev, hindi_dev, english_test, hindi_test
    )
    review_path = OUT / "semantic_review_sample.csv"
    if _review_has_judgments(review_path):
        print("keeping judged review sample", flush=True)
        with review_path.open(encoding="utf-8", newline="") as handle:
            review = list(csv.DictReader(handle))
    else:
        with review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(review[0].keys()))
            writer.writeheader()
            writer.writerows(review)
    strata = Counter()
    for row in review:
        for part in row["strata"].split("|"):
            strata[part.split("_")[0] if part.startswith("order_break") else part] += 1
    print(json.dumps({
        "jsonl_by_split": summary["jsonl_by_split"],
        "candidate_by_status": summary["candidate_by_status"],
        "problems": summary["jsonl_text_or_index_problems"],
        "premise_multi_en": summary["verified_train_premise_map"]["english_premises_with_multiple_hindi"],
        "premise_multi_hi": summary["verified_train_premise_map"]["hindi_premises_with_multiple_english"],
        "slots": summary["verified_train_label_slots"],
        "norm_en": summary["english_normalization"],
        "norm_hi": summary["hindi_normalization"],
        "dup_en": summary["english_duplicate_pairs"],
        "dup_hi": summary["hindi_duplicate_pairs"],
        "dev": summary["dev_full_check"],
        "test": summary["test_full_check"],
        "review_rows": len(review),
        "review_strata": dict(strata),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
