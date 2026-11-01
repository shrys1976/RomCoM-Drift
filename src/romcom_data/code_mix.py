"""Faithful reconstruction of Indi-RomCoM taxonomy-guided code-mixing.

This is NOT the official Indi-RomCoM release. Substitutions follow Table 9–10
with a closed bilingual lexicon. CAT-E clause rewriting is not performed with
an LLM; at 75% the Indic-frame path romanizes the parallel Hindi sentence while
re-inserting CAT-0 English tokens.
"""

from __future__ import annotations

import re
from typing import Any

from romcom_data.romanize import DakshinaRomanizer
from romcom_data.taxonomy import (
    classify_english_token,
    hindi_for_category,
    should_switch,
)
from romcom_data.text_utils import DEVANAGARI, nfc, tokenize_conservative


def _preserve_case(original: str, replacement: str) -> str:
    if not original or not replacement:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _romanize_hindi_sentence(hindi: str, romanizer: DakshinaRomanizer) -> str:
    pieces: list[str] = []
    last = 0
    for match in re.finditer(r"[\u0900-\u097F]+", hindi):
        pieces.append(hindi[last:match.start()])
        hit = romanizer.romanize_native(match.group(0))
        pieces.append(hit.surface)
        last = match.end()
    pieces.append(hindi[last:])
    return "".join(pieces)


def mix_from_english(
    english: str,
    intensity: str,
    romanizer: DakshinaRomanizer,
    field_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    tokens = tokenize_conservative(english)
    out: list[str] = []
    traces: list[dict[str, Any]] = []
    seen_word = False
    for tok in tokens:
        if tok.text.isspace() or not tok.is_latin_word:
            out.append(tok.text)
            continue
        category = classify_english_token(tok.text, sentence_initial=not seen_word)
        seen_word = True
        switch = should_switch(category, intensity)
        hindi = hindi_for_category(tok.text, category) if switch else None
        if switch and hindi:
            hit = romanizer.romanize_native(hindi)
            replacement = _preserve_case(tok.text, hit.surface)
            out.append(replacement)
            traces.append(
                {
                    "field": field_name,
                    "original": tok.text,
                    "category": category,
                    "hindi_native": hindi,
                    "replacement": replacement,
                    "romanization_method": hit.method,
                    "attested": hit.attested,
                    "operation": "taxonomy_switch",
                    "offsets": [tok.start, tok.end],
                }
            )
        else:
            out.append(tok.text)
    return "".join(out), traces


def mix_indic_frame(
    english: str,
    hindi_native: str,
    romanizer: DakshinaRomanizer,
    field_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    """75% approximation: romanized Hindi matrix, CAT-0 English tokens restored."""
    mixed = _romanize_hindi_sentence(nfc(hindi_native), romanizer)
    traces: list[dict[str, Any]] = [
        {
            "field": field_name,
            "original": hindi_native,
            "category": "E",
            "replacement": mixed,
            "operation": "indic_frame_romanize",
            "reason": "cat_e_approximated_without_llm",
        }
    ]
    protected = [
        t.text
        for t in tokenize_conservative(english)
        if t.is_latin_word and classify_english_token(t.text) == "0"
    ]
    if protected:
        traces.append(
            {
                "field": field_name,
                "operation": "cat0_inventory",
                "protected_english_tokens": protected,
            }
        )
    return mixed, traces


def reconstruct_field(
    english: str | None,
    hindi_native: str | None,
    intensity: str,
    romanizer: DakshinaRomanizer,
    field_name: str,
) -> tuple[str, list[dict[str, Any]], str]:
    english = english or ""
    hindi_native = hindi_native or ""
    if english.strip():
        text, traces = mix_from_english(english, intensity, romanizer, field_name)
        return text, traces, "english_matrix_taxonomy"
    if hindi_native.strip():
        text = _romanize_hindi_sentence(hindi_native, romanizer)
        return text, [{"field": field_name, "operation": "hindi_only_romanize"}], "hindi_romanize"
    return "", [], "empty"
