from __future__ import annotations

import json
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi

from romcom_data.handoff_paths import RAW_TOX
from romcom_data.stable_hash import stable_hash_hex, derive_rng
from romcom_data.text_utils import nfc

REPO = "ai4bharat/indic-align"
TOXIC_CONFIG = "Toxic_Matrix"
INSTRUCT_CONFIG = "Dolly_T"
PAPER_TOXIC_N = 10000
PAPER_EVAL_SAMPLE = 100


def _turn_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, list):
                parts.extend(str(x) for x in item if x is not None)
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def acquire_toxic_matrix(revision: str | None = None) -> dict[str, Any]:
    api = HfApi()
    info = api.dataset_info(REPO, revision=revision)
    sha = info.sha
    RAW_TOX.mkdir(parents=True, exist_ok=True)

    toxic_path = RAW_TOX / "Toxic_Matrix_train.jsonl"
    instruct_path = RAW_TOX / "Indic_ShareLlama_benign_sample.jsonl"
    if toxic_path.is_file() and toxic_path.stat().st_size > 1000 and instruct_path.is_file() and instruct_path.stat().st_size > 100:
        n_toxic = sum(1 for _ in toxic_path.open())
        n_benign = sum(1 for _ in instruct_path.open())
        return {
            "repository": REPO,
            "revision": sha,
            "license": "CC-BY-4.0",
            "license_status": "cleared_for_internal_use",
            "toxic_config": TOXIC_CONFIG,
            "benign_config": INSTRUCT_CONFIG,
            "n_toxic_raw": n_toxic,
            "n_benign_sampled": n_benign,
            "benign_status": "reused",
            "files": [toxic_path.name, instruct_path.name],
        }

    ds = load_dataset(REPO, name=TOXIC_CONFIG, split="train", revision=sha)
    keep_cols = [c for c in ("doc_id", "eng_Latn", "hin_Deva", "hin_Latn") if c in ds.column_names]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols and not str(c).startswith("__")])
    sample_n = min(PAPER_TOXIC_N // 2, len(ds))
    ds = ds.shuffle(seed=42).select(range(sample_n))
    n_toxic = 0
    with toxic_path.open("w", encoding="utf-8") as fh:
        for row in ds:
            clean = {k: v for k, v in dict(row).items() if not str(k).startswith("__")}
            fh.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")
            n_toxic += 1

    instruct_path = RAW_TOX / "Indic_ShareLlama_benign_sample.jsonl"
    n_benign = 0
    try:
        ins = load_dataset(
            REPO, name=INSTRUCT_CONFIG, split="train[:2000]", revision=sha
        )
        with instruct_path.open("w", encoding="utf-8") as fh:
            for row in ins:
                clean = {k: v for k, v in dict(row).items() if not str(k).startswith("__")}
                fh.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")
                n_benign += 1
        benign_status = "sampled_Dolly_T_train_head_2000"
    except Exception as exc:
        instruct_path.write_text("", encoding="utf-8")
        benign_status = f"unavailable: {exc}"

    return {
        "repository": REPO,
        "revision": sha,
        "license": "CC-BY-4.0",
        "license_status": "cleared_for_internal_use",
        "toxic_config": TOXIC_CONFIG,
        "benign_config": INSTRUCT_CONFIG,
        "n_toxic_raw": n_toxic,
        "n_benign_sampled": n_benign,
        "benign_status": benign_status,
        "files": [toxic_path.name, instruct_path.name],
    }


def load_normalized_toxicity(meta: dict[str, Any], seed: int = 42) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    toxic_path = RAW_TOX / "Toxic_Matrix_train.jsonl"
    with toxic_path.open(encoding="utf-8") as fh:
        toxic_rows = [json.loads(line) for line in fh if line.strip()]

    rng = derive_rng(seed, "toxicity-sample", "source")
    if len(toxic_rows) > PAPER_TOXIC_N // 2:
        idxs = list(range(len(toxic_rows)))
        rng.shuffle(idxs)
        keep = idxs[: PAPER_TOXIC_N // 2]
        toxic_rows = [toxic_rows[i] for i in sorted(keep)]

    for idx, row in enumerate(toxic_rows):
        prompt = _turn_text(row.get("eng_Latn") or row.get("hin_Deva") or "")
        response = _turn_text(row.get("hin_Latn") or "")
        hin_deva = _turn_text(row.get("hin_Deva") or "")
        hin_latn = _turn_text(row.get("hin_Latn") or "")
        doc_id = str(row.get("doc_id") or f"tox-{idx}")
        records.append(
            {
                "source_id": f"toxic-{doc_id}",
                "example_id": f"toxic-{doc_id}:{idx}",
                "original_prompt": nfc(prompt),
                "original_response": nfc(response),
                "hin_deva": nfc(hin_deva),
                "hin_latn": nfc(hin_latn),
                "original_label": "toxic",
                "normalized_label": "toxic",
                "is_benign": False,
                "language": "hi",
                "original_split": "train",
                "source_dataset": f"{REPO}/{TOXIC_CONFIG}",
                "source_revision": meta["revision"],
                "source_file": "Toxic_Matrix_train.jsonl",
                "row_index": idx,
                "license_ref": "CC-BY-4.0",
                "input_is_code_mixed": False,
                "target_may_be_code_mixed": True,
            }
        )

    benign_path = RAW_TOX / "Indic_ShareLlama_benign_sample.jsonl"
    if benign_path.is_file() and benign_path.stat().st_size > 0:
        with benign_path.open(encoding="utf-8") as fh:
            benign_rows = [json.loads(line) for line in fh if line.strip()]
        n_needed = min(len(benign_rows), PAPER_TOXIC_N - len(records))
        for idx, row in enumerate(benign_rows[:n_needed]):
            prompt = _turn_text(
                row.get("eng_Latn") or row.get("hin_Deva") or row.get("prompt") or ""
            )
            response = _turn_text(
                row.get("hin_Latn") or row.get("response") or row.get("output") or ""
            )
            doc_id = str(row.get("doc_id") or f"benign-{idx}")
            records.append(
                {
                    "source_id": f"benign-{doc_id}",
                    "example_id": f"benign-{doc_id}:{idx}",
                    "original_prompt": nfc(prompt),
                    "original_response": nfc(response),
                    "hin_deva": nfc(_turn_text(row.get("hin_Deva") or "")),
                    "hin_latn": nfc(_turn_text(row.get("hin_Latn") or "")),
                    "original_label": "non_toxic",
                    "normalized_label": "non_toxic",
                    "is_benign": True,
                    "language": "hi",
                    "original_split": "train",
                    "source_dataset": f"{REPO}/{INSTRUCT_CONFIG}",
                    "source_revision": meta["revision"],
                    "source_file": "Indic_ShareLlama_benign_sample.jsonl",
                    "row_index": idx,
                    "license_ref": "CC-BY-4.0",
                    "input_is_code_mixed": False,
                    "target_may_be_code_mixed": True,
                }
            )
    return records
