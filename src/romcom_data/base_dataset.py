from __future__ import annotations

from collections import Counter
from typing import Any

from romcom_data.code_mix import reconstruct_field
from romcom_data.paraphrase_gate import validate_reconstructed_pair
from romcom_data.romanize import DakshinaRomanizer
from romcom_data.stable_hash import stable_hash_hex

INTENSITY_LEVELS = ("25", "50", "75")


def intensity_label(level: str) -> str:
    return f"reconstructed_intensity_{level}"


def _hash_row(*parts: str) -> str:
    return stable_hash_hex(*parts)


def build_nli_base(rows: list[dict[str, Any]], romanizer: DakshinaRomanizer) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for intensity in INTENSITY_LEVELS:
            p_text, p_tr, p_mode = reconstruct_field(
                row.get("premise_en"),
                None,
                intensity,
                romanizer,
                "premise",
            )
            h_text, h_tr, h_mode = reconstruct_field(
                row.get("hypothesis_en"),
                None,
                intensity,
                romanizer,
                "hypothesis",
            )
            example_id = f"{row['source_id']}:cm{intensity}"
            out.append(
                {
                    **{k: row[k] for k in row},
                    "example_id": example_id,
                    "task": "nli",
                    "mixing_intensity": intensity_label(intensity),
                    "mixing_intensity_level": intensity,
                    "reconstruction_status": "reconstructed_not_official_indi_romcom",
                    "code_mix_mode_premise": p_mode,
                    "code_mix_mode_hypothesis": h_mode,
                    "input": {"premise": p_text, "hypothesis": h_text},
                    "target": {"label": row["normalized_label"]},
                    "label": row["normalized_label"],
                    "mix_trace": p_tr + h_tr,
                    "source_row_hash": _hash_row(
                        row["source_id"], row["premise_native"], row["hypothesis_native"]
                    ),
                }
            )
    return out


def build_paraphrase_base(
    rows: list[dict[str, Any]], romanizer: DakshinaRomanizer
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for row in rows:
        s1_en = (row.get("sentence1_en") or "").strip()
        s2_en = (row.get("sentence2_en") or "").strip()
        if not s1_en or not s2_en or s1_en == s2_en:
            stats["excluded_no_independent_english"] += 1
            continue
        for intensity in INTENSITY_LEVELS:
            s1, t1, m1 = reconstruct_field(
                s1_en,
                None,
                intensity,
                romanizer,
                "sentence1",
            )
            s2, t2, m2 = reconstruct_field(
                s2_en,
                None,
                intensity,
                romanizer,
                "sentence2",
            )
            ok, reason = validate_reconstructed_pair(
                s1,
                s2,
                str(row["normalized_label"]),
                row["sentence1_native"],
                row["sentence2_native"],
            )
            if not ok:
                stats[f"excluded_{reason}"] += 1
                continue
            if reason == "source_identical_positive":
                stats["kept_source_identical_positive"] += 1
            example_id = f"{row['source_id']}:cm{intensity}"
            out.append(
                {
                    **{k: row[k] for k in row if k != "conversation_turns"},
                    "example_id": example_id,
                    "task": "paraphrase",
                    "mixing_intensity": intensity_label(intensity),
                    "mixing_intensity_level": intensity,
                    "reconstruction_status": "reconstructed_not_official_indi_romcom",
                    "code_mix_mode_sentence1": m1,
                    "code_mix_mode_sentence2": m2,
                    "input": {"sentence1": s1, "sentence2": s2},
                    "target": {"label": row["normalized_label"]},
                    "label": row["normalized_label"],
                    "mix_trace": t1 + t2,
                    "source_row_hash": _hash_row(
                        row["source_id"], row["sentence1_native"], row["sentence2_native"]
                    ),
                }
            )
            stats["kept_examples"] += 1
            label = str(row["normalized_label"])
            stats[f"kept_{label}"] += 1
    return out, dict(stats)


def build_toxicity_base(
    rows: list[dict[str, Any]], romanizer: DakshinaRomanizer
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        prompt = row["original_prompt"]
        for intensity in INTENSITY_LEVELS:
            resp, tr, mode = reconstruct_field(
                row.get("original_response"),
                row.get("hin_deva") or None,
                intensity,
                romanizer,
                "target_response",
            )
            example_id = f"{row['source_id']}:cm{intensity}"
            out.append(
                {
                    **{k: row[k] for k in row},
                    "example_id": example_id,
                    "task": "toxicity",
                    "mixing_intensity": intensity_label(intensity),
                    "mixing_intensity_level": intensity,
                    "reconstruction_status": "reconstructed_not_official_indi_romcom",
                    "code_mix_mode_target": mode,
                    "input": {"prompt": prompt},
                    "target": {
                        "label": row["normalized_label"],
                        "response": resp,
                    },
                    "label": row["normalized_label"],
                    "mix_trace": [
                        {
                            "field": "prompt",
                            "operation": "unchanged",
                            "reason": "toxicity_input_never_code_mixed_per_paper_B5",
                        }
                    ]
                    + tr,
                    "augmentation_side": "target_only",
                    "source_row_hash": _hash_row(row["source_id"], prompt),
                }
            )
    return out
