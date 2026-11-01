"""Origin-based code-mix composition (Latin script does not imply language)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from romcom_data.taxonomy import classify_english_token, hindi_for_category, should_switch
from romcom_data.text_utils import tokenize_conservative


def _field_texts(row: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    inp = row.get("input") or {}
    texts.extend(str(v) for v in inp.values())
    if row.get("task") == "toxicity":
        texts.append(str((row.get("target") or {}).get("response", "")))
    return texts


def measure_row(row: dict[str, Any], *, english_source: str | None = None) -> dict[str, int]:
    intensity = str(row.get("mixing_intensity", "")).replace("reconstructed_intensity_", "")
    hindi_origin = 0
    english_origin = 0
    switches = 0
    eligible = 0
    replaced = 0
    missing_dict = 0
    last_origin: str | None = None
    en_src = english_source or ""
    en_tokens = {t.text.casefold(): t.text for t in tokenize_conservative(en_src) if t.is_latin_word}

    for text in _field_texts(row):
        for tok in tokenize_conservative(text):
            if not tok.is_latin_word:
                continue
            cat = classify_english_token(tok.text)
            if cat == "0":
                origin = "english"
            else:
                low = tok.text.casefold()
                en_tok = en_tokens.get(low, tok.text)
                cat_en = classify_english_token(en_tok)
                if intensity and should_switch(cat_en, intensity):
                    eligible += 1
                    if hindi_for_category(en_tok, cat_en):
                        if tok.text.casefold() != low:
                            replaced += 1
                        origin = "hindi"
                    else:
                        missing_dict += 1
                        origin = "english"
                else:
                    origin = "english"
            if origin == "hindi":
                hindi_origin += 1
            else:
                english_origin += 1
            if last_origin and last_origin != origin:
                switches += 1
            last_origin = origin
    return {
        "hindi_origin_tokens": hindi_origin,
        "english_origin_tokens": english_origin,
        "language_switches": switches,
        "eligible_for_switch": eligible,
        "taxonomy_replacements": replaced,
        "missing_dictionary": missing_dict,
    }


def aggregate_mix_report(bases: list[dict[str, Any]]) -> dict[str, Any]:
    by_task_int: dict[str, Counter] = defaultdict(Counter)
    zero_switch = Counter()
    n_rows = Counter()
    for row in bases:
        key = f"{row.get('task')}:{row.get('mixing_intensity')}"
        n_rows[key] += 1
        m = measure_row(row)
        for k, v in m.items():
            by_task_int[key][k] += v
        if m["language_switches"] == 0:
            zero_switch[key] += 1
    report: dict[str, Any] = {"by_task_intensity": {}, "zero_switch_examples": {}}
    for key, counts in by_task_int.items():
        total = counts["hindi_origin_tokens"] + counts["english_origin_tokens"]
        report["by_task_intensity"][key] = {
            **dict(counts),
            "hindi_origin_proportion": counts["hindi_origin_tokens"] / total if total else 0,
            "english_origin_proportion": counts["english_origin_tokens"] / total if total else 0,
            "replacement_rate": (
                counts["taxonomy_replacements"] / counts["eligible_for_switch"]
                if counts["eligible_for_switch"]
                else 0
            ),
        }
        report["zero_switch_examples"][key] = {
            "count": zero_switch[key],
            "rate": zero_switch[key] / n_rows[key] if n_rows[key] else 0,
        }
    return report
