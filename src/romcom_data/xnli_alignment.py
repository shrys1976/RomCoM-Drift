"""Recover English–Hindi XNLI pairs without assuming same-index rows match."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

METHOD_VERIFIED = "corroborated_premise_translation_and_unique_label"
METHOD_AMBIGUOUS = "corroborated_premise_translation_duplicate_label"
METHOD_UNMATCHED = "no_corroborated_premise"
METHOD_COLLISION = "premise_translation_collision"
ANCHOR = 25
SAFE_RADIUS = 20


def monotonic_label_alignment(
    hindi_labels: list[int],
    english_labels: list[int],
    anchor: int = ANCHOR,
) -> tuple[list[int], list[int]]:
    """Align label sequences, allowing English rows to be skipped.

    This is only a mask for which rows are safe to learn from. It is not a
    pairing. Returned Hindi-to-English indexes are -1 for the unaligned tail.
    """
    mapping = [-1] * len(hindi_labels)
    skipped: list[int] = []
    hindi_index = 0
    english_index = 0
    hindi_len = len(hindi_labels)
    english_len = len(english_labels)
    while hindi_index < hindi_len and english_index < english_len:
        hindi_end = hindi_index + anchor
        english_end = english_index + anchor
        if (
            hindi_end <= hindi_len
            and english_end <= english_len
            and hindi_labels[hindi_index:hindi_end] == english_labels[english_index:english_end]
        ):
            for offset in range(anchor):
                mapping[hindi_index + offset] = english_index + offset
            hindi_index = hindi_end
            english_index = english_end
            continue
        if hindi_labels[hindi_index] == english_labels[english_index]:
            mapping[hindi_index] = english_index
            hindi_index += 1
            english_index += 1
            continue
        next_english = english_index + 1
        if (
            hindi_end <= hindi_len
            and next_english + anchor <= english_len
            and hindi_labels[hindi_index:hindi_end]
            == english_labels[next_english : next_english + anchor]
        ):
            skipped.append(english_index)
            english_index = next_english
            continue
        raise RuntimeError(
            "Label sequences diverged at "
            f"hindi={hindi_index} english={english_index}"
        )
    return mapping, skipped


def _near_skip(english_index: int, skipped: list[int], radius: int) -> bool:
    if not skipped or radius < 0:
        return False
    low = 0
    high = len(skipped)
    while low < high:
        mid = (low + high) // 2
        if skipped[mid] < english_index:
            low = mid + 1
        else:
            high = mid
    for position in (low - 1, low):
        if 0 <= position < len(skipped) and abs(skipped[position] - english_index) <= radius:
            return True
    return False


def learn_string_map(
    english_texts: list[str],
    hindi_texts: list[str],
    provisional: list[int],
    skipped: list[int],
    radius: int,
    min_support: int,
) -> tuple[dict[str, str], int, set[str]]:
    """Learn English text to Hindi text from safe, repeated, unanimous copies.

    Hindi strings learned from more than one English string are removed from
    the map and returned so those rows stay unresolved.
    """
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for hindi_index, english_index in enumerate(provisional):
        if english_index < 0 or _near_skip(english_index, skipped, radius):
            continue
        votes[english_texts[english_index]][hindi_texts[hindi_index]] += 1

    learned: dict[str, str] = {}
    disagreed = 0
    for english_text, counts in votes.items():
        support = sum(counts.values())
        if support < min_support or len(counts) != 1:
            if len(counts) != 1 and support >= min_support:
                disagreed += 1
            continue
        learned[english_text] = next(iter(counts))

    hindi_owners: dict[str, list[str]] = defaultdict(list)
    for english_text, hindi_text in learned.items():
        hindi_owners[hindi_text].append(english_text)
    collisions = {hindi for hindi, owners in hindi_owners.items() if len(owners) > 1}
    if collisions:
        learned = {
            english: hindi for english, hindi in learned.items() if hindi not in collisions
        }
    return learned, disagreed, collisions


def recover_train_pairs(
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
    radius: int = SAFE_RADIUS,
    anchor: int = ANCHOR,
    min_support: int = 2,
) -> dict[str, Any]:
    """Pair train rows. Verified pairs are a subset; nothing is forced 1:1."""
    if not english_rows or not hindi_rows:
        raise ValueError("Both train splits must be non-empty")
    provisional, skipped = monotonic_label_alignment(
        [row["label"] for row in hindi_rows],
        [row["label"] for row in english_rows],
        anchor=anchor,
    )
    premise_map, premise_disagreements, premise_collision_hindi = learn_string_map(
        [row["premise"] for row in english_rows],
        [row["premise"] for row in hindi_rows],
        provisional,
        skipped,
        radius,
        min_support,
    )
    english_by_premise: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(english_rows):
        english_by_premise[row["premise"]].append(index)
    hindi_by_premise: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(hindi_rows):
        hindi_by_premise[row["premise"]].append(index)

    hindi_to_english_premise = {hindi: english for english, hindi in premise_map.items()}
    english_to_hindi = {
        english_index: hindi_index
        for hindi_index, english_index in enumerate(provisional)
        if english_index >= 0
    }
    assignments: list[dict[str, Any]] = []
    used_english: set[int] = set()

    for hindi_premise, hindi_indexes in hindi_by_premise.items():
        english_premise = hindi_to_english_premise.get(hindi_premise)
        if english_premise is None:
            collided = hindi_premise in premise_collision_hindi
            method = METHOD_COLLISION if collided else METHOD_UNMATCHED
            reason = "premise_translation_collision" if collided else "hindi_premise_not_corroborated"
            for hindi_index in hindi_indexes:
                assignments.append(
                    _assignment(
                        hindi_index,
                        None,
                        "unmatched",
                        method,
                        {"reason": reason},
                        [],
                    )
                )
            continue
        english_indexes = english_by_premise[english_premise]
        support = _safe_support(
            english_indexes, english_to_hindi, skipped, radius, hindi_rows, hindi_premise
        )
        for label in sorted({hindi_rows[index]["label"] for index in hindi_indexes}):
            hindi_group = [index for index in hindi_indexes if hindi_rows[index]["label"] == label]
            english_group = [
                index for index in english_indexes if english_rows[index]["label"] == label
            ]
            evidence = {
                "english_premise_copies": len(english_indexes),
                "safe_supporting_copies": support,
                "english_rows_with_premise_and_label": len(english_group),
                "hindi_rows_with_premise_and_label": len(hindi_group),
                "candidate_english_row_indexes": english_group,
            }
            unique = len(hindi_group) == 1 and len(english_group) == 1
            if unique:
                english_index = english_group[0]
                used_english.add(english_index)
                assignments.append(
                    _assignment(
                        hindi_group[0],
                        english_index,
                        "verified",
                        METHOD_VERIFIED,
                        evidence,
                        english_group,
                    )
                )
                continue
            conflict = len(hindi_group) != len(english_group)
            status = "unmatched" if conflict else "ambiguous"
            method = "premise_label_count_conflict" if conflict else METHOD_AMBIGUOUS
            for hindi_index in hindi_group:
                assignments.append(
                    _assignment(hindi_index, None, status, method, evidence, english_group)
                )

    assignments.sort(key=lambda item: item["hindi_row_index"])
    if len(assignments) != len(hindi_rows):
        raise RuntimeError("Every Hindi row must be classified exactly once")
    return {
        "assignments": assignments,
        "provisional_skipped_english_indexes": skipped,
        "premise_map_size": len(premise_map),
        "premise_disagreements": premise_disagreements,
        "premise_collisions": len(premise_collision_hindi),
        "used_english_indexes": used_english,
        "anchor": anchor,
        "safe_radius": radius,
        "min_support": min_support,
    }


def _safe_support(
    english_indexes: list[int],
    english_to_hindi: dict[int, int],
    skipped: list[int],
    radius: int,
    hindi_rows: list[dict[str, Any]],
    hindi_premise: str,
) -> int:
    support = 0
    for english_index in english_indexes:
        hindi_index = english_to_hindi.get(english_index)
        if hindi_index is None or _near_skip(english_index, skipped, radius):
            continue
        if hindi_rows[hindi_index]["premise"] == hindi_premise:
            support += 1
    return support


def _assignment(
    hindi_index: int,
    english_index: int | None,
    status: str,
    method: str,
    evidence: dict[str, Any],
    candidates: list[int],
) -> dict[str, Any]:
    return {
        "hindi_row_index": hindi_index,
        "english_row_index": english_index,
        "validation_status": status,
        "matching_method": method,
        "matching_evidence": evidence,
        "candidate_english_row_indexes": candidates,
    }


def same_index_premise_audit(
    english_rows: list[dict[str, Any]],
    hindi_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Control for splits whose generation order is still intact."""
    if len(english_rows) != len(hindi_rows):
        raise ValueError("Same-index audit requires equal lengths")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(english_rows):
        grouped[row["premise"]].append(index)
    repeated = {premise: indexes for premise, indexes in grouped.items() if len(indexes) >= 2}
    unanimous = 0
    for indexes in repeated.values():
        hindi_premises = {hindi_rows[index]["premise"] for index in indexes}
        if len(hindi_premises) == 1:
            unanimous += 1
    label_hits = sum(
        english_rows[index]["label"] == hindi_rows[index]["label"]
        for index in range(len(english_rows))
    )
    return {
        "rows": len(english_rows),
        "repeated_english_premises": len(repeated),
        "unanimous_hindi_premises": unanimous,
        "same_index_label_hits": label_hits,
    }
