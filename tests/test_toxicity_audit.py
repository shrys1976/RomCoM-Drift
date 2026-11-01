from __future__ import annotations

from romcom_data.toxic_turns import (
    assistant_target_from_turns,
    parse_conversation_turns,
    user_prompt_from_turns,
)
from romcom_data.toxicity_audit import (
    audit_integrity_toxic,
    audit_leakage,
    build_recommendation,
    simulate_stratified_split,
)


def test_parse_toxic_pair_roles_separate():
    row = {
        "doc_id": "x",
        "eng_Latn": [["toxic user text", "I cannot give a response"]],
    }
    turns = parse_conversation_turns(row)
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"
    assert user_prompt_from_turns(turns) == "toxic user text"
    assert "cannot" in assistant_target_from_turns(turns).lower()


def test_leakage_no_doc_overlap_disjoint():
    tox = [{"doc_id": "a", "eng_Latn": [["u", "a"]]}]
    ben = [{"doc_id": "b", "eng_Latn": [["u", "a"]]}]
    leak = audit_leakage(tox, ben)
    assert leak["doc_id_overlap_toxic_benign"] == 0
    assert leak["split_simulation"]["doc_id_leak_train_val"] == 0


def test_recommendation_conditional_on_legacy_benign():
    rec = build_recommendation(
        {"row_count": 10, "empty_user_prompt_after_parse": 0},
        {"benign_file": "Indic_ShareLlama_benign_sample.jsonl"},
        {"doc_id_overlap_toxic_benign": 0},
        {},
        {"toxic_matrix_config_present": True},
    )
    assert rec["verdict"] == "CONDITIONAL_GO"
    assert rec["experimental_construction"] == "NOT_APPROVED"
    assert "dual_license_owner_approval" in rec["blockers"]


def test_construction_stays_unapproved():
    rec = build_recommendation(
        {"row_count": 10, "empty_user_prompt_after_parse": 0},
        {"benign_file": "Indic_ShareLlama_benign_sample.jsonl"},
        {"doc_id_overlap_toxic_benign": 0},
        {},
        {"toxic_matrix_config_present": True},
        {"local_file_equals_dolly_t_train_prefix": True},
        {"n": 120},
    )
    assert rec["verdict"] == "CONDITIONAL_GO"
    assert rec["experimental_construction"] == "NOT_APPROVED"
    assert "dual_license_owner_approval" in rec["blockers"]
    assert "owner_accept_dolly_t_head_slice" not in rec["blockers"]


def test_hindi_field_is_canonical_and_latn_is_rejected():
    row = {
        "hin_Deva": [["उपयोगकर्ता", "सहायक"]],
        "eng_Latn": [["user", "assistant"]],
        "hin_Latn": [["swapped roman", "other"]],
    }
    turns = parse_conversation_turns(row)
    assert [turn["text"] for turn in turns] == ["उपयोगकर्ता", "सहायक"]
    try:
        parse_conversation_turns(row, text_field="hin_Latn")
    except ValueError:
        return
    raise AssertionError("hin_Latn should not be a role source")


def test_manual_refusal_exception_is_not_a_keyword_rule():
    from romcom_data.toxicity_audit import assign_manual_refusal_category, STOCK_REFUSAL

    text = STOCK_REFUSAL + "The prompt is harmful. Instead, seek help."
    assert assign_manual_refusal_category(2179, text) == "partial_refusal_with_safe_alternative"
    assert assign_manual_refusal_category(1, text) == "clear_refusal"
    assert assign_manual_refusal_category(1, "Sure, here is how.") == "ambiguous"
