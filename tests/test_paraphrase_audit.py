from __future__ import annotations

import pandas as pd

from romcom_data.paraphrase_audit import (
    audit_english_pivot,
    audit_integrity,
    audit_leakage_connected_components,
    build_recommendation,
    find_identical_positive_quarantine,
)


def test_english_pivot_rejects_single_en_column():
    df = pd.DataFrame(
        {
            "en": ["english one", "english one"],
            "sentence1": ["h1", "h1"],
            "sentence2": ["h2a", "h2b"],
            "label": [1, 0],
        }
    )
    pivot = audit_english_pivot(df)
    assert pivot["reject_english_pivot_reconstruction_for_code_mixed"]
    assert pivot["rows_with_independent_sentence2_english"] == 0
    rec = build_recommendation(pivot, {"row_count": 2}, {"positive_transformation_induced_collapse": 0})
    assert rec["english_pivot_gate_passed"] is False


def test_connected_component_split_no_sentence_leak():
    import pandas as pd

    df = pd.DataFrame(
        {
            "en": ["e1", "e1", "e2", "e2"],
            "sentence1": ["a", "a", "b", "b"],
            "sentence2": ["x", "y", "p", "q"],
            "label": [1, 0, 1, 0],
        }
    )
    q = find_identical_positive_quarantine(df, "rev")
    leak = audit_leakage_connected_components(df, "rev", set())
    assert leak["sentence_leak_train_and_validation_cc_split"] == 0


def test_owner_license_approval_removes_confirmation_blocker():
    from romcom_data.paraphrase_audit import (
        OWNER_LICENSE_APPROVAL,
        determine_handoff_status,
    )

    status = determine_handoff_status(
        {"hf_card_license_field": None, "owner_approval": OWNER_LICENSE_APPROVAL},
        {"english_pivot_code_mixed_verdict": "NO_GO"},
        {"sentence_leak_train_and_validation_cc_split": 0},
        [],
    )
    assert status["license_owner_approved"] is True
    assert "license_not_declared_on_hf_card_owner_must_confirm" not in status["blockers"]
    assert "english_pivot_code_mixed_path_rejected" in status["blockers"]


def test_integrity_counts_shared_sentence1():
    df = pd.DataFrame(
        {
            "en": ["e", "e"],
            "sentence1": ["s1", "s1"],
            "sentence2": ["a", "b"],
            "label": [1, 0],
        }
    )
    integrity = audit_integrity(df)
    assert integrity["sentence1_with_both_positive_and_negative"] == 1
