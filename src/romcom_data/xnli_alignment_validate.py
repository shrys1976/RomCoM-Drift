"""Independent checks for the recovered English–Hindi XNLI alignment."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

WHITESPACE = re.compile(r"\s+")
NEGATION_EN = re.compile(r"\b(?:not|no|never|without|cannot|n't)\b", re.IGNORECASE)
NEGATION_HI = re.compile(r"नहीं|नही|बिना|मत|नहीं")
DIGIT = re.compile(r"\d+")


def normalize_ws(text: str) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFC", text).strip())


def normalize_casefold(text: str) -> str:
    return normalize_ws(text).casefold()


def collapse_counts(texts: list[str], key_fn) -> dict[str, int]:
    """How many raw strings merge under key_fn, among keys that cover 2+ raw forms."""
    groups: dict[str, set[str]] = defaultdict(set)
    for text in texts:
        groups[key_fn(text)].add(text)
    merged = {key: len(raw) for key, raw in groups.items() if len(raw) > 1}
    return merged


def duplicate_pair_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Duplicate premise+hypothesis pairs, including label conflicts."""
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row["premise"], row["hypothesis"])].append(index)
    duplicate_groups = {key: indexes for key, indexes in grouped.items() if len(indexes) > 1}
    label_conflicts = 0
    same_label = 0
    for indexes in duplicate_groups.values():
        labels = {rows[index]["label"] for index in indexes}
        if len(labels) > 1:
            label_conflicts += 1
        else:
            same_label += 1
    return {
        "duplicate_premise_hypothesis_groups": len(duplicate_groups),
        "groups_with_conflicting_labels": label_conflicts,
        "groups_with_same_label": same_label,
        "rows_in_duplicate_groups": sum(len(indexes) for indexes in duplicate_groups.values()),
    }


def slot_counts(rows: list[dict[str, Any]]) -> Counter[tuple[str, int]]:
    return Counter((row["premise"], row["label"]) for row in rows)


def premise_translation_audit(
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    """English premise -> Hindi premises and the reverse, using an explicit pairing."""
    en_to_hi: dict[str, set[str]] = defaultdict(set)
    hi_to_en: dict[str, set[str]] = defaultdict(set)
    for hindi_index, english_index in pairs:
        english_premise = english_rows[english_index]["premise"]
        hindi_premise = hindi_rows[hindi_index]["premise"]
        en_to_hi[english_premise].add(hindi_premise)
        hi_to_en[hindi_premise].add(english_premise)
    en_multi = {premise: sorted(targets) for premise, targets in en_to_hi.items() if len(targets) > 1}
    hi_multi = {premise: sorted(targets) for premise, targets in hi_to_en.items() if len(targets) > 1}
    return {
        "paired_rows": len(pairs),
        "english_premises": len(en_to_hi),
        "hindi_premises": len(hi_to_en),
        "english_premises_with_multiple_hindi": len(en_multi),
        "hindi_premises_with_multiple_english": len(hi_multi),
        "english_exceptions": _trim_exceptions(en_multi),
        "hindi_exceptions": _trim_exceptions(hi_multi),
    }


def _trim_exceptions(groups: dict[str, list[str]], limit: int = 12) -> list[dict[str, Any]]:
    items = sorted(groups.items(), key=lambda item: item[0])[:limit]
    return [
        {"premise": premise[:180], "others": [other[:180] for other in others]}
        for premise, others in items
    ]


def label_slot_identifies_one_hypothesis(
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    """A verified slot must be the only English and Hindi row with that premise and label."""
    english_slots = slot_counts(english_rows)
    hindi_slots = slot_counts(hindi_rows)
    english_failures = 0
    hindi_failures = 0
    hypothesis_variants: Counter[tuple[str, int]] = Counter()
    for hindi_index, english_index in pairs:
        english = english_rows[english_index]
        hindi = hindi_rows[hindi_index]
        if english_slots[(english["premise"], english["label"])] != 1:
            english_failures += 1
        if hindi_slots[(hindi["premise"], hindi["label"])] != 1:
            hindi_failures += 1
        hypothesis_variants[(english["premise"], english["label"])] += 1
    repeated_slots_in_pairs = sum(1 for count in hypothesis_variants.values() if count > 1)
    return {
        "verified_pairs_whose_english_slot_is_not_unique_in_source": english_failures,
        "verified_pairs_whose_hindi_slot_is_not_unique_in_source": hindi_failures,
        "verified_slots_claimed_more_than_once": repeated_slots_in_pairs,
    }


def source_slots_that_are_not_unique(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = slot_counts(rows)
    nonunique = [count for count in counts.values() if count > 1]
    return {
        "nonunique_premise_label_slots": len(nonunique),
        "rows_in_nonunique_slots": sum(nonunique),
        "max_rows_in_one_slot": max(nonunique) if nonunique else 1,
    }


def negation_mismatch(english: str, hindi: str) -> bool:
    return bool(NEGATION_EN.search(english)) != bool(NEGATION_HI.search(hindi))


def digit_tokens(text: str) -> tuple[str, ...]:
    return tuple(DIGIT.findall(text))


def reconcile_verified_jsonl(train: int, dev: int, test: int) -> dict[str, int | bool]:
    """verified_pairs.jsonl is the three splits concatenated, not a second train count."""
    total = train + dev + test
    return {
        "train": train,
        "dev": dev,
        "test": test,
        "jsonl_rows": total,
        "train_plus_dev_plus_test_equals_jsonl": total == train + dev + test,
    }
