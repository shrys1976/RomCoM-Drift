from romcom_data.xnli_alignment import recover_train_pairs
from romcom_data.xnli_alignment_validate import (
    duplicate_pair_stats,
    reconcile_verified_jsonl,
)


def _row(premise: str, hypothesis: str, label: int) -> dict:
    return {"premise": premise, "hypothesis": hypothesis, "label": label}


def test_moved_row_is_recovered_by_repeated_premise() -> None:
    english = [
        _row("premise one", "hyp a", 0),
        _row("premise one", "hyp b", 1),
        _row("premise one", "hyp c", 2),
        _row("premise two", "hyp d", 0),
        _row("premise two", "hyp e", 1),
        _row("premise two", "hyp f", 2),
    ]
    order = [0, 1, 2, 4, 5, 3]
    hindi = [
        _row(f"hi {english[index]['premise']}", f"hi {english[index]['hypothesis']}", english[index]["label"])
        for index in order
    ]
    result = recover_train_pairs(english, hindi, radius=0, anchor=2, min_support=2)
    verified = {
        item["hindi_row_index"]: item["english_row_index"]
        for item in result["assignments"]
        if item["validation_status"] == "verified"
    }
    assert verified == {hindi_index: english_index for hindi_index, english_index in enumerate(order)}
    assert result["premise_disagreements"] == 0


def test_duplicate_label_stays_ambiguous() -> None:
    english = [
        _row("same premise", "hyp a", 0),
        _row("same premise", "hyp b", 0),
        _row("same premise", "hyp c", 1),
        _row("other premise", "hyp d", 0),
        _row("other premise", "hyp e", 1),
        _row("other premise", "hyp f", 2),
    ]
    hindi = [
        _row(f"hi {row['premise']}", f"hi {row['hypothesis']}", row["label"]) for row in english
    ]
    result = recover_train_pairs(english, hindi, radius=0, anchor=2, min_support=2)
    statuses = [item["validation_status"] for item in result["assignments"]]
    assert statuses[:2] == ["ambiguous", "ambiguous"]
    assert statuses[2:] == ["verified", "verified", "verified", "verified"]


def test_disagreeing_translations_are_not_verified() -> None:
    english = [
        _row("unstable", "hyp a", 0),
        _row("unstable", "hyp b", 1),
        _row("stable", "hyp c", 0),
        _row("stable", "hyp d", 1),
        _row("stable", "hyp e", 2),
        _row("stable two", "hyp f", 0),
        _row("stable two", "hyp g", 1),
        _row("stable two", "hyp h", 2),
    ]
    hindi_premises = [
        "hindi A",
        "hindi B",
        "hindi stable",
        "hindi stable",
        "hindi stable",
        "hindi two",
        "hindi two",
        "hindi two",
    ]
    hindi = [
        _row(hindi_premises[index], f"h{index}", english[index]["label"])
        for index in range(len(english))
    ]
    result = recover_train_pairs(english, hindi, radius=0, anchor=2, min_support=2)
    unstable = [item for item in result["assignments"] if item["hindi_row_index"] < 2]
    assert {item["validation_status"] for item in unstable} == {"unmatched"}
    assert result["premise_disagreements"] == 1


def test_verified_jsonl_count_is_train_plus_dev_plus_test() -> None:
    identity = reconcile_verified_jsonl(375805, 2490, 5010)
    assert identity["jsonl_rows"] == 383305
    assert identity["train_plus_dev_plus_test_equals_jsonl"] is True


def test_duplicate_pair_stats_counts_label_conflicts() -> None:
    rows = [
        _row("p", "h", 0),
        _row("p", "h", 2),
        _row("q", "same", 1),
        _row("q", "same", 1),
    ]
    stats = duplicate_pair_stats(rows)
    assert stats["duplicate_premise_hypothesis_groups"] == 2
    assert stats["groups_with_conflicting_labels"] == 1
    assert stats["groups_with_same_label"] == 1
    assert stats["rows_in_duplicate_groups"] == 4
