from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from romcom_data.handoff_paths import RAW_PARA
from romcom_data.stable_hash import stable_hash_hex
from romcom_data.text_utils import DEVANAGARI, nfc

REPO = "ai4bharat/IndicXParaphrase"


def acquire_indicxparaphrase(revision: str | None = None) -> dict[str, Any]:
    api = HfApi()
    info = api.dataset_info(REPO, revision=revision)
    sha = info.sha
    dest = RAW_PARA / "hi.tsv"
    local = hf_hub_download(
        repo_id=REPO, filename="data/hi.tsv", repo_type="dataset", revision=sha
    )
    dest.write_bytes(Path(local).read_bytes())
    return {
        "repository": REPO,
        "revision": sha,
        "license": "CC0",
        "license_status": "cleared_for_internal_use",
        "files": ["hi.tsv"],
        "split_semantics": (
            "Single language TSV with no official train/dev/test files. "
            "Not a test split; project 80/20 is created at source-group level."
        ),
    }


def load_normalized_paraphrase(meta: dict[str, Any]) -> list[dict[str, Any]]:
    path = RAW_PARA / "hi.tsv"
    df = pd.read_csv(path, sep="\t")
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        s1 = nfc(str(row.get("sentence1", "")))
        s2 = nfc(str(row.get("sentence2", "")))
        en = nfc(str(row.get("en", "")))
        label = row.get("label")
        if pd.isna(s1) or pd.isna(s2) or not s1.strip() or not s2.strip():
            continue
        original_label = int(label) if not pd.isna(label) else None
        normalized = None
        if original_label == 1:
            normalized = "paraphrase"
        elif original_label == 0:
            normalized = "non_paraphrase"
        source_id = "para-" + stable_hash_hex(meta["revision"], str(idx), s1, s2)[:16]
        records.append(
            {
                "source_id": source_id,
                "example_id": f"{source_id}:{idx}",
                "sentence1_native": s1,
                "sentence2_native": s2,
                "sentence_en": en,
                "sentence1_is_devanagari": bool(DEVANAGARI.search(s1)),
                "sentence2_is_devanagari": bool(DEVANAGARI.search(s2)),
                "original_label": original_label,
                "normalized_label": normalized,
                "language": "hi",
                "original_split": "unspecified",
                "source_dataset": REPO,
                "source_revision": meta["revision"],
                "source_file": "data/hi.tsv",
                "row_index": int(idx),
                "license_ref": "CC0",
            }
        )
    return records
