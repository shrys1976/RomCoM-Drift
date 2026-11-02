"""Loaders for the XNLI English and IndicXNLI Hindi files used in alignment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

LABEL_TO_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "contradictory": 2,
}
ID_TO_LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}


def label_to_int(label: str | int) -> int:
    if isinstance(label, int):
        return label
    text = str(label).strip().lower()
    if text.isdigit():
        return int(text)
    if text not in LABEL_TO_ID:
        raise ValueError(f"Unknown NLI label: {label!r}")
    return LABEL_TO_ID[text]


def load_xnli_mt_english_train(path: Path) -> list[dict[str, Any]]:
    """Parse XNLI-MT English train the way datasets 1.16.1 xnli.py does."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for index, row in enumerate(reader):
            raw_label = row["label"].replace("contradictory", "contradiction")
            rows.append(
                {
                    "row_index": index,
                    "premise": row["premise"],
                    "hypothesis": row["hypo"],
                    "label": label_to_int(raw_label),
                }
            )
    return rows


def load_indic_forward_json(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload[next(iter(payload))] if isinstance(payload, dict) else payload
    return [
        {
            "row_index": index,
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "label": label_to_int(row["label"]),
        }
        for index, row in enumerate(records)
    ]


def load_xnli10_english_split(path: Path, language: str = "en") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            if row.get("language") != language:
                continue
            rows.append(
                {
                    "row_index": index,
                    "pair_id": row.get("pairID") or "",
                    "premise": row["sentence1"],
                    "hypothesis": row["sentence2"],
                    "label": label_to_int(row["gold_label"]),
                }
            )
    return rows
