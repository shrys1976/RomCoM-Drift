from __future__ import annotations

from typing import Any

from datasets import load_dataset


def attach_xnli_english(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    need = [r for r in records if not r.get("premise_en")]
    if not need:
        return records
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in need:
        split = row["original_split"]
        hf_split = "validation" if split == "dev" else split
        if hf_split not in {"train", "validation", "test"}:
            continue
        by_split.setdefault(hf_split, []).append(row)
    for hf_split, rows in by_split.items():
        ds = load_dataset("facebook/xnli", "en", split=hf_split)
        for row in rows:
            idx = int(row["row_index"])
            if idx >= len(ds):
                continue
            ex = ds[idx]
            row["premise_en"] = ex["premise"]
            row["hypothesis_en"] = ex["hypothesis"]
            row["english_joined_from"] = f"facebook/xnli:{hf_split}:{idx}"
    return records
