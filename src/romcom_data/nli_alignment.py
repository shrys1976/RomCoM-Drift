"""NLI English–Hindi alignment sample for owner review."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from romcom_data.stable_hash import derive_rng


def write_alignment_sample(
    rows: list[dict[str, Any]],
    path: Path,
    n: int = 200,
    seed: int = 42,
) -> int:
    pool = [r for r in rows if r.get("premise_en") and r.get("premise_native")]
    rng = derive_rng(seed, "nli-alignment", "sample")
    rng.shuffle(pool)
    sample = pool[: min(n, len(pool))]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_id",
        "original_split",
        "label",
        "premise_native",
        "hypothesis_native",
        "premise_en",
        "hypothesis_en",
        "english_joined_from",
        "label_agrees_with_xnli",
        "same_premise",
        "same_hypothesis",
        "reviewer",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in sample:
            w.writerow(
                {
                    "source_id": row.get("source_id"),
                    "original_split": row.get("original_split"),
                    "label": row.get("normalized_label"),
                    "premise_native": row.get("premise_native"),
                    "hypothesis_native": row.get("hypothesis_native"),
                    "premise_en": row.get("premise_en"),
                    "hypothesis_en": row.get("hypothesis_en"),
                    "english_joined_from": row.get("english_joined_from", ""),
                    "label_agrees_with_xnli": "",
                    "same_premise": "",
                    "same_hypothesis": "",
                    "reviewer": "",
                    "notes": "",
                }
            )
    return len(sample)
