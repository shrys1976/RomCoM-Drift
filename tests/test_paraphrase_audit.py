from __future__ import annotations

import pandas as pd

from romcom_data.paraphrase_audit import audit_english_pivot, audit_integrity, build_recommendation


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
