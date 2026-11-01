from __future__ import annotations

import json
from pathlib import Path

import pytest

from romcom_data.augmentation import apply_random_ops_reverse, levenshtein_distance, levenshtein_ops
from romcom_data.canonical_tie import pick_canonical_variant
from romcom_data.paraphrase_gate import validate_reconstructed_pair
from romcom_data.split_freeze import nli_alignment_approved
from romcom_data.toxic_turns import parse_conversation_turns, user_prompt_from_turns, assistant_target_from_turns


def test_pick_canonical_lexicographic_tie():
    chosen, count, tied = pick_canonical_variant([("zebra", 5), ("alpha", 5), ("beta", 3)])
    assert tied
    assert chosen == "alpha"
    assert count == 5


def test_noise_reverse_ops_distance():
    token = "hello"
    selected = "hallo"
    ops = levenshtein_ops(token, selected)
    rng = __import__("random").Random(42)
    noisy = apply_random_ops_reverse(token, ops, rng)
    assert levenshtein_distance(token, noisy) == levenshtein_distance(token, selected)


def test_paraphrase_positive_collapse_excluded():
    ok, reason = validate_reconstructed_pair("same", "same", "paraphrase", "a", "b")
    assert not ok
    assert reason == "transformation_induced_positive_collapse"
    ok2, reason2 = validate_reconstructed_pair("same", "same", "paraphrase", "a", "a")
    assert ok2
    assert reason2 == "source_identical_positive"


def test_toxic_turns_separate_roles():
    row = {
        "eng_Latn": [
            [
                "toxic user question",
                "I cannot answer because it is harmful.",
            ]
        ]
    }
    turns = parse_conversation_turns(row)
    prompt = user_prompt_from_turns(turns)
    target = assistant_target_from_turns(turns)
    assert "cannot answer" not in prompt
    assert "harmful" in target
    assert "toxic user" in prompt


def test_map_dev_rows_same_index_not_pair_id_join():
    from romcom_data.nli_provenance import map_dev_rows

    hi = [{"label": 0, "premise_hi": "a", "hypothesis_hi": "b"}]
    en = [
        {
            "label": 0,
            "pair_id": "99",
            "row_index": 0,
            "composite_key": "k",
            "premise_en": "e1",
            "hypothesis_en": "e2",
            "source_dataset": "XNLI-1.0",
        }
    ]
    manifest, stats = map_dev_rows(hi, en)
    assert manifest[0]["mapping_method"] == "same_index_xnli_10_en"
    assert manifest[0]["pair_id_used_for_join"] is False
    assert stats["same_index_label_agreement"] == 1


def test_nli_alignment_not_approved_by_default(tmp_path: Path):
    p = tmp_path / "nli_alignment_status.json"
    p.write_text(json.dumps({"owner_approved": False}), encoding="utf-8")
    assert not nli_alignment_approved(p)
