"""Romanization coverage on full bases with Hindi-derived vs English denominators."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from romcom_data.romanize import DakshinaRomanizer
from romcom_data.text_utils import tokenize_conservative


def _hindi_derived_tokens(row: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for rec in row.get("mix_trace") or []:
        if rec.get("operation") in {"taxonomy_switch", "indic_frame_romanize", "hindi_only_romanize"}:
            rep = rec.get("replacement") or rec.get("selected")
            if rep:
                tokens.append(str(rep))
    return tokens


def coverage_by_task_intensity(
    bases: list[dict[str, Any]], romanizer: DakshinaRomanizer
) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "hindi_derived_occurrences": 0,
            "mapped_occurrences": 0,
            "hindi_types": set(),
            "mapped_types": set(),
            "unknown_latin": 0,
            "ambiguous_alias": 0,
            "english_tokens": 0,
        }
    )
    for row in bases:
        key = f"{row.get('task')}:{row.get('mixing_intensity')}"
        b = buckets[key]
        for text in _hindi_derived_tokens(row):
            for tok in tokenize_conservative(text):
                if not tok.is_latin_word:
                    continue
                b["hindi_derived_occurrences"] += 1
                b["hindi_types"].add(tok.text.casefold())
                hit = romanizer.lookup_alias(tok.text)
                if hit is None:
                    if tok.text.casefold() in romanizer.ambiguous_aliases:
                        b["ambiguous_alias"] += 1
                    else:
                        b["unknown_latin"] += 1
                else:
                    b["mapped_occurrences"] += 1
                    b["mapped_types"].add(tok.text.casefold())
        inp = row.get("input") or {}
        for val in inp.values():
            for tok in tokenize_conservative(str(val)):
                if tok.is_latin_word:
                    b["english_tokens"] += 1

    out: dict[str, Any] = {}
    for key, b in buckets.items():
        occ = b["hindi_derived_occurrences"]
        out[key] = {
            "occurrence_coverage": b["mapped_occurrences"] / occ if occ else 0,
            "type_coverage": len(b["mapped_types"]) / len(b["hindi_types"]) if b["hindi_types"] else 0,
            "hindi_derived_occurrences": occ,
            "english_token_occurrences": b["english_tokens"],
            "unknown_rate": b["unknown_latin"] / occ if occ else 0,
            "ambiguous_rate": b["ambiguous_alias"] / occ if occ else 0,
        }
    return out
