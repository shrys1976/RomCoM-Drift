from __future__ import annotations

from typing import Any

def attach_xnli_english(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach English only from verified nli_alignment_manifest.csv; never HF index join."""
    import csv
    import json

    from romcom_data.handoff_paths import MANIFESTS

    status_path = MANIFESTS / "nli_alignment_status.json"
    manifest_path = MANIFESTS / "nli_alignment_manifest.csv"
    if not status_path.is_file() or not manifest_path.is_file():
        return records
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("train_mapping_reliable") or not status.get("owner_approved"):
        for row in records:
            row.pop("premise_en", None)
            row.pop("hypothesis_en", None)
            row.pop("english_joined_from", None)
        return records

    by_key: dict[tuple[str, int], dict[str, str]] = {}
    with manifest_path.open(encoding="utf-8") as fh:
        for mrow in csv.DictReader(fh):
            split = mrow.get("original_split") or "train"
            idx = int(mrow["indic_row_index"])
            by_key[(split, idx)] = {
                "premise_en": mrow["premise_en"],
                "hypothesis_en": mrow["hypothesis_en"],
                "english_joined_from": mrow.get("mapping_method", "manifest"),
            }
    for row in records:
        split = row.get("original_split")
        idx = int(row.get("row_index", -1))
        hit = by_key.get((split, idx))
        if hit:
            row.update(hit)
    return records
