from __future__ import annotations

from typing import Any

from romcom_data.code_mix import reconstruct_field
from romcom_data.romanize import DakshinaRomanizer
from romcom_data.stable_hash import stable_hash_hex

INTENSITIES = ("25", "50", "75")


def _hash_row(*parts: str) -> str:
    return stable_hash_hex(*parts)


def build_nli_base(rows: list[dict[str, Any]], romanizer: DakshinaRomanizer) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for intensity in INTENSITIES:
            p_text, p_tr, p_mode = reconstruct_field(
                row.get("premise_en"),
                row["premise_native"],
                intensity,
                romanizer,
                "premise",
            )
            h_text, h_tr, h_mode = reconstruct_field(
                row.get("hypothesis_en"),
                row["hypothesis_native"],
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
                    "mixing_intensity": intensity,
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
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for intensity in INTENSITIES:
            s1, t1, m1 = reconstruct_field(
                row.get("sentence_en"),
                row["sentence1_native"],
                intensity,
                romanizer,
                "sentence1",
            )
            s2, t2, m2 = reconstruct_field(
                row.get("sentence_en"),
                row["sentence2_native"],
                intensity,
                romanizer,
                "sentence2",
            )
            example_id = f"{row['source_id']}:cm{intensity}"
            out.append(
                {
                    **{k: row[k] for k in row},
                    "example_id": example_id,
                    "task": "paraphrase",
                    "mixing_intensity": intensity,
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
    return out


def build_toxicity_base(
    rows: list[dict[str, Any]], romanizer: DakshinaRomanizer
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        prompt = row["original_prompt"]
        for intensity in INTENSITIES:
            resp, tr, mode = reconstruct_field(
                row.get("original_response"),
                row.get("hin_deva") or row.get("original_response"),
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
                    "mixing_intensity": intensity,
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
