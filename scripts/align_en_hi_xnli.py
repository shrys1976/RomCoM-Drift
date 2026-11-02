#!/usr/bin/env python3
"""Recover corroborated English–Hindi XNLI pairs. Does not edit source files."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.xnli_alignment import (
    monotonic_label_alignment,
    recover_train_pairs,
    same_index_premise_audit,
)
from romcom_data.xnli_alignment_io import (
    ID_TO_LABEL,
    load_indic_forward_json,
    load_xnli10_english_split,
    load_xnli_mt_english_train,
)

RAW = ROOT / "romcomx_data" / "raw"
OUT = ROOT / "romcomx_data" / "nli_alignment"
EN_TRAIN = RAW / "xnli_mt" / "XNLI-MT-1.0" / "multinli" / "multinli.train.en.tsv"
HI_TRAIN = RAW / "indicxnli" / "forward__train__xnli_hi.json"
REVIEW_PER_STATUS = {"verified": 40, "ambiguous": 15, "unmatched": 15}
DIGIT_RE = re.compile(r"\d+")
SOURCE_EN = "XNLI-MT-1.0/multinli/multinli.train.en.tsv"
SOURCE_HI = "Divyanshu/indicxnli forward/train/xnli_hi.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digit_audit(
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    checked = exact = subset = 0
    mismatches: list[dict[str, str]] = []
    for item in assignments:
        if item["validation_status"] != "verified":
            continue
        english = english_rows[item["english_row_index"]]
        hindi = hindi_rows[item["hindi_row_index"]]
        for field in ("premise", "hypothesis"):
            english_digits = tuple(DIGIT_RE.findall(english[field]))
            hindi_digits = tuple(DIGIT_RE.findall(hindi[field]))
            if not english_digits:
                continue
            checked += 1
            if english_digits == hindi_digits:
                exact += 1
            if all(token in hindi_digits for token in english_digits):
                subset += 1
            elif len(mismatches) < 8:
                mismatches.append(
                    {
                        "field": field,
                        "english_digits": list(english_digits),
                        "hindi_digits": list(hindi_digits),
                        "english": english[field][:180],
                        "hindi": hindi[field][:180],
                    }
                )
    return {
        "verified_fields_with_english_digits": checked,
        "exact_digit_sequence": exact,
        "english_digits_present_in_hindi": subset,
        "mismatch_examples": mismatches,
    }


def ordered_split_control(name: str, english_path: Path, hindi_path: Path) -> dict[str, Any]:
    english_rows = load_xnli10_english_split(english_path)
    hindi_rows = load_indic_forward_json(hindi_path)
    audit = same_index_premise_audit(english_rows, hindi_rows)
    grouped: Counter[str] = Counter(row["premise"] for row in english_rows)
    audit["split"] = name
    audit["english_path"] = str(english_path.relative_to(ROOT))
    audit["hindi_path"] = str(hindi_path.relative_to(ROOT))
    audit["rows_in_repeated_premises"] = sum(count for count in grouped.values() if count >= 2)
    audit["singleton_premise_rows"] = sum(count for count in grouped.values() if count == 1)
    return audit


def write_verified_jsonl(
    path: Path,
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in assignments:
            if item["validation_status"] != "verified":
                continue
            english = english_rows[item["english_row_index"]]
            hindi = hindi_rows[item["hindi_row_index"]]
            record = {
                "english_premise": english["premise"],
                "english_hypothesis": english["hypothesis"],
                "hindi_premise": hindi["premise"],
                "hindi_hypothesis": hindi["hypothesis"],
                "nli_label": ID_TO_LABEL[english["label"]],
                "english_row_index": english["row_index"],
                "hindi_row_index": hindi["row_index"],
                "english_source": SOURCE_EN,
                "hindi_source": SOURCE_HI,
                "matching_method": item["matching_method"],
                "matching_evidence": item["matching_evidence"],
                "validation_status": "verified",
                "split": "train",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_candidate_csv(
    path: Path,
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> None:
    fields = [
        "hindi_row_index",
        "english_row_index",
        "validation_status",
        "matching_method",
        "nli_label",
        "candidate_english_row_indexes",
        "matching_evidence",
        "english_premise",
        "english_hypothesis",
        "hindi_premise",
        "hindi_hypothesis",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in assignments:
            english_index = item["english_row_index"]
            hindi = hindi_rows[item["hindi_row_index"]]
            english = english_rows[english_index] if english_index is not None else None
            label_id = hindi["label"] if english is None else english["label"]
            writer.writerow(
                {
                    "hindi_row_index": item["hindi_row_index"],
                    "english_row_index": "" if english_index is None else english_index,
                    "validation_status": item["validation_status"],
                    "matching_method": item["matching_method"],
                    "nli_label": ID_TO_LABEL[label_id],
                    "candidate_english_row_indexes": " ".join(
                        str(index) for index in item["candidate_english_row_indexes"]
                    ),
                    "matching_evidence": json.dumps(
                        item["matching_evidence"], ensure_ascii=False, sort_keys=True
                    ),
                    "english_premise": "" if english is None else english["premise"],
                    "english_hypothesis": "" if english is None else english["hypothesis"],
                    "hindi_premise": hindi["premise"],
                    "hindi_hypothesis": hindi["hypothesis"],
                }
            )


def write_review_sample(
    path: Path,
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    provisional_disagreements: list[dict[str, Any]],
) -> None:
    rng = random.Random(0)
    chosen: list[dict[str, Any]] = []
    by_status: dict[str, list[dict[str, Any]]] = {}
    for item in assignments:
        by_status.setdefault(item["validation_status"], []).append(item)
    for status, limit in REVIEW_PER_STATUS.items():
        pool = by_status.get(status, [])
        picks = pool if len(pool) <= limit else rng.sample(pool, limit)
        for item in picks:
            chosen.append(_review_row(english_rows, hindi_rows, item, "status_sample"))
    for item in provisional_disagreements[:15]:
        chosen.append(item)
    fields = list(chosen[0].keys()) if chosen else ["note"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)


def _review_row(
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
    item: dict[str, Any],
    sample_kind: str,
) -> dict[str, Any]:
    hindi = hindi_rows[item["hindi_row_index"]]
    english_index = item["english_row_index"]
    english = english_rows[english_index] if english_index is not None else None
    return {
        "sample_kind": sample_kind,
        "validation_status": item["validation_status"],
        "matching_method": item["matching_method"],
        "hindi_row_index": item["hindi_row_index"],
        "english_row_index": "" if english_index is None else english_index,
        "candidate_english_row_indexes": " ".join(
            str(index) for index in item["candidate_english_row_indexes"]
        ),
        "nli_label": ID_TO_LABEL[hindi["label"]],
        "english_premise": "" if english is None else english["premise"],
        "english_hypothesis": "" if english is None else english["hypothesis"],
        "hindi_premise": hindi["premise"],
        "hindi_hypothesis": hindi["hypothesis"],
    }


def provisional_false_matches(
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    provisional: list[int],
) -> list[dict[str, Any]]:
    """Rows where label-order alignment and the verified pair disagree."""
    verified = {
        item["hindi_row_index"]: item
        for item in assignments
        if item["validation_status"] == "verified"
    }
    examples: list[dict[str, Any]] = []
    for hindi_index, english_index in enumerate(provisional):
        item = verified.get(hindi_index)
        if item is None or english_index < 0 or item["english_row_index"] == english_index:
            continue
        wrong = english_rows[english_index]
        right = english_rows[item["english_row_index"]]
        hindi = hindi_rows[hindi_index]
        examples.append(
            {
                "sample_kind": "label_order_false_match",
                "validation_status": "rejected_label_order",
                "matching_method": "monotonic_label_alignment_rejected",
                "hindi_row_index": hindi_index,
                "english_row_index": item["english_row_index"],
                "candidate_english_row_indexes": str(english_index),
                "nli_label": ID_TO_LABEL[hindi["label"]],
                "english_premise": (
                    f"VERIFIED: {right['premise'][:240]} || LABEL_ORDER: {wrong['premise'][:240]}"
                ),
                "english_hypothesis": right["hypothesis"],
                "hindi_premise": hindi["premise"],
                "hindi_hypothesis": hindi["hypothesis"],
            }
        )
        if len(examples) >= 15:
            break
    return examples


def summarize(
    english_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    recovery: dict[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(item["validation_status"] for item in assignments)
    method_counts = Counter(item["matching_method"] for item in assignments)
    verified_labels = Counter(
        ID_TO_LABEL[english_rows[item["english_row_index"]]["label"]]
        for item in assignments
        if item["validation_status"] == "verified"
    )
    ambiguous_english = {
        index
        for item in assignments
        if item["validation_status"] == "ambiguous"
        for index in item["candidate_english_row_indexes"]
    }
    verified_english = {
        item["english_row_index"]
        for item in assignments
        if item["validation_status"] == "verified"
    }
    return {
        "hindi_train_rows": len(assignments),
        "english_train_rows": len(english_rows),
        "exact_identifier_matches": 0,
        "status_counts": dict(status_counts),
        "method_counts": dict(method_counts),
        "verified_label_counts": dict(verified_labels),
        "verified_label_agreement": status_counts["verified"],
        "english_in_verified_pair": len(verified_english),
        "english_only_in_ambiguous_groups": len(ambiguous_english - verified_english),
        "english_unmatched": len(english_rows) - len(verified_english | ambiguous_english),
        "premise_map_size": recovery["premise_map_size"],
        "premise_disagreements": recovery["premise_disagreements"],
        "premise_collisions": recovery["premise_collisions"],
        "label_order_skipped_english_rows": len(recovery["provisional_skipped_english_indexes"]),
        "label_order_skipped_english_indexes_head": recovery["provisional_skipped_english_indexes"][:20],
        "anchor": recovery["anchor"],
        "safe_radius": recovery["safe_radius"],
        "min_support": recovery["min_support"],
    }


ORDERED_METHOD = "same_index_corroborated_premise_and_unique_label"


def append_ordered_split(
    path: Path,
    split: str,
    english_path: Path,
    hindi_path: Path,
    audit: dict[str, Any],
) -> int:
    """Append dev or test rows when same-index pairing is fully corroborated."""
    fully_corroborated = (
        audit["unanimous_hindi_premises"] == audit["repeated_english_premises"]
        and audit["singleton_premise_rows"] == 0
        and audit["same_index_label_hits"] == audit["rows"]
    )
    if not fully_corroborated:
        return 0
    english_rows = load_xnli10_english_split(english_path)
    hindi_rows = load_indic_forward_json(hindi_path)
    label_counts: Counter[tuple[str, int]] = Counter(
        (row["premise"], row["label"]) for row in english_rows
    )
    if any(count != 1 for count in label_counts.values()):
        return 0
    evidence = {
        "repeated_english_premises": audit["repeated_english_premises"],
        "unanimous_hindi_premises": audit["unanimous_hindi_premises"],
        "same_index_label_hits": audit["same_index_label_hits"],
        "premise_label_slots_unique": True,
    }
    with path.open("a", encoding="utf-8") as handle:
        for index, (english, hindi) in enumerate(zip(english_rows, hindi_rows, strict=True)):
            record = {
                "english_premise": english["premise"],
                "english_hypothesis": english["hypothesis"],
                "hindi_premise": hindi["premise"],
                "hindi_hypothesis": hindi["hypothesis"],
                "nli_label": ID_TO_LABEL[english["label"]],
                "english_row_index": index,
                "english_pair_id": english.get("pair_id") or "",
                "hindi_row_index": index,
                "english_source": str(english_path.relative_to(ROOT)),
                "hindi_source": str(hindi_path.relative_to(ROOT)),
                "matching_method": ORDERED_METHOD,
                "matching_evidence": evidence,
                "validation_status": "verified",
                "split": split,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(english_rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading", flush=True)
    english_rows = load_xnli_mt_english_train(EN_TRAIN)
    hindi_rows = load_indic_forward_json(HI_TRAIN)
    print(f"loaded en={len(english_rows)} hi={len(hindi_rows)}", flush=True)
    recovery = recover_train_pairs(english_rows, hindi_rows)
    print("paired", flush=True)
    provisional, _skipped = monotonic_label_alignment(
        [row["label"] for row in hindi_rows],
        [row["label"] for row in english_rows],
    )
    assignments = recovery["assignments"]
    stats = summarize(english_rows, assignments, recovery)
    same_index_hits = sum(
        english_rows[index]["label"] == hindi_rows[index]["label"]
        for index in range(len(english_rows))
    )
    stats["same_index_label_hits"] = same_index_hits
    stats["exact_cross_lingual_premise_overlaps"] = len(
        {row["premise"] for row in english_rows} & {row["premise"] for row in hindi_rows}
    )
    stats["digits"] = digit_audit(english_rows, hindi_rows, assignments)
    stats["dev"] = ordered_split_control(
        "dev",
        RAW / "xnli_mt" / "XNLI-1.0" / "xnli.dev.tsv",
        RAW / "indicxnli" / "forward__dev__xnli_hi.json",
    )
    stats["test"] = ordered_split_control(
        "test",
        RAW / "xnli_mt" / "XNLI-1.0" / "xnli.test.tsv",
        RAW / "indicxnli" / "forward__test__xnli_hi.json",
    )
    stats["source_sha256"] = {
        str(EN_TRAIN.relative_to(ROOT)): sha256_file(EN_TRAIN),
        str(HI_TRAIN.relative_to(ROOT)): sha256_file(HI_TRAIN),
    }
    stats["verdict"] = "PARTIALLY_RECOVERED"
    verified_path = OUT / "verified_pairs.jsonl"
    candidate_path = OUT / "candidate_alignment.csv"
    sample_path = OUT / "review_sample.csv"
    stats_path = OUT / "alignment_statistics.json"
    verified_count = write_verified_jsonl(verified_path, english_rows, hindi_rows, assignments)
    stats["verified_dev_rows"] = append_ordered_split(
        verified_path,
        "dev",
        RAW / "xnli_mt" / "XNLI-1.0" / "xnli.dev.tsv",
        RAW / "indicxnli" / "forward__dev__xnli_hi.json",
        stats["dev"],
    )
    stats["verified_test_rows"] = append_ordered_split(
        verified_path,
        "test",
        RAW / "xnli_mt" / "XNLI-1.0" / "xnli.test.tsv",
        RAW / "indicxnli" / "forward__test__xnli_hi.json",
        stats["test"],
    )
    write_candidate_csv(candidate_path, english_rows, hindi_rows, assignments)
    false_matches = provisional_false_matches(
        english_rows, hindi_rows, assignments, provisional
    )
    write_review_sample(sample_path, english_rows, hindi_rows, assignments, false_matches)
    stats["verified_train_rows"] = verified_count
    stats["verified_jsonl_rows"] = (
        verified_count + stats["verified_dev_rows"] + stats["verified_test_rows"]
    )
    stats["output_sha256"] = {
        verified_path.name: sha256_file(verified_path),
        candidate_path.name: sha256_file(candidate_path),
        sample_path.name: sha256_file(sample_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checksum_path = OUT / "checksums.sha256"
    lines = []
    for path in (EN_TRAIN, HI_TRAIN, verified_path, candidate_path, sample_path, stats_path):
        lines.append(f"{sha256_file(path)}  {path.relative_to(ROOT)}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: stats[key] for key in (
        "hindi_train_rows",
        "english_train_rows",
        "status_counts",
        "verified_label_counts",
        "english_unmatched",
        "premise_map_size",
        "premise_disagreements",
        "premise_collisions",
        "same_index_label_hits",
        "verdict",
        "dev",
        "test",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
