"""Paraphrase source semantics and eligibility for English-matrix reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from romcom_data.text_utils import nfc


def document_en_field_semantics(tsv_path: Path) -> str:
    return (
        "IndicXParaphrase `hi.tsv` provides columns `en`, `sentence1`, `sentence2`, `label`. "
        "The `en` field is a single English string aligned with `sentence1` (translation of the "
        "first Hindi sentence), not an independent English pivot for `sentence2`. "
        "There is no `sentence2_en` column in the pinned release."
    )


def filter_paraphrase_for_reconstruction(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep rows only when both sides have distinct, non-empty English pivots."""
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    reason = "no_independent_sentence2_en_in_source"
    for row in rows:
        s1_en = (row.get("sentence1_en") or "").strip()
        s2_en = (row.get("sentence2_en") or "").strip()
        if s1_en and s2_en and s1_en != s2_en:
            kept.append(row)
        else:
            ex = dict(row)
            ex["exclusion_reason"] = reason
            excluded.append(ex)
    stats = {
        "n_source": len(rows),
        "n_kept": len(kept),
        "n_excluded": len(excluded),
        "exclusion_reason": reason if excluded else "",
        "semantics": document_en_field_semantics(Path(row.get("source_file", "hi.tsv")) if rows else Path("hi.tsv")),
    }
    return kept, excluded, stats


def validate_reconstructed_pair(
    s1: str,
    s2: str,
    label: str,
    s1_native: str,
    s2_native: str,
) -> tuple[bool, str]:
    if s1 != s2:
        return True, "ok"
    if label == "non_paraphrase":
        return False, "identical_non_paraphrase"
    n1 = nfc(s1_native).strip()
    n2 = nfc(s2_native).strip()
    if n1 == n2:
        return True, "source_identical_positive"
    return False, "transformation_induced_positive_collapse"
