"""Source-level grouping and split freeze before any augmentation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from romcom_data.splitting import assign_splits, nli_group_key, paraphrase_groups, _norm
from romcom_data.stable_hash import stable_hash_hex


def group_key_for_row(row: dict[str, Any], task: str, para_root: dict[str, str] | None = None) -> str:
    if task == "nli":
        return nli_group_key(row)
    if task == "paraphrase":
        para_root = para_root or {}
        return para_root.get(_norm(row["sentence1_native"]), _norm(row["sentence1_native"]))
    return str(row["source_id"])


def assign_group_ids(rows: list[dict[str, Any]], task: str) -> None:
    para_root = paraphrase_groups(rows) if task == "paraphrase" else None
    for row in rows:
        row["group_id"] = group_key_for_row(row, task, para_root)


def partition_test_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_rows: list[dict[str, Any]] = []
    nontest: list[dict[str, Any]] = []
    for row in rows:
        if row.get("original_split") == "test":
            test_rows.append(row)
        else:
            nontest.append(row)
    return nontest, test_rows


def quarantine_cross_partition_groups(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        gid = row.get("group_id") or row["source_id"]
        by_group[gid].add(row.get("original_split") or "unspecified")

    bad_groups: set[str] = set()
    for gid, splits in by_group.items():
        has_train = "train" in splits
        has_dev = bool(splits & {"dev", "validation"})
        if has_train and has_dev:
            bad_groups.add(gid)

    kept: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for row in rows:
        gid = row.get("group_id") or row["source_id"]
        if gid in bad_groups:
            row["exclusion_status"] = "quarantined"
            row["exclusion_reason"] = "group_crosses_official_train_and_validation"
            quarantined.append(row)
        else:
            row["exclusion_status"] = "kept"
            row["exclusion_reason"] = ""
            kept.append(row)
    return kept, quarantined


def write_protected_test_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_id",
        "task",
        "label",
        "original_split",
        "source_row_hash",
        "source_dataset",
        "source_revision",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            sh = row.get("source_row_hash")
            if not sh:
                sh = stable_hash_hex(
                    row.get("source_id", ""),
                    str(row.get("premise_native", "")),
                    str(row.get("hypothesis_native", "")),
                    str(row.get("sentence1_native", "")),
                    str(row.get("sentence2_native", "")),
                )
            w.writerow(
                {
                    "source_id": row.get("source_id"),
                    "task": row.get("task"),
                    "label": row.get("normalized_label") or row.get("label"),
                    "original_split": "test",
                    "source_row_hash": sh,
                    "source_dataset": row.get("source_dataset", ""),
                    "source_revision": row.get("source_revision", ""),
                }
            )


def freeze_task_splits(
    rows: list[dict[str, Any]],
    task: str,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Assign groups, drop test, quarantine, then project splits. Mutates rows in place."""
    for row in rows:
        row["task"] = task
    assign_group_ids(rows, task)
    nontest, test_rows = partition_test_rows(rows)
    eligible, quarantined = quarantine_cross_partition_groups(nontest)
    assign_splits(eligible, task, seed=seed)
    for row in eligible:
        if not row.get("group_id"):
            assign_group_ids([row], task)
    return {
        "task": task,
        "n_total": len(rows),
        "n_test": len(test_rows),
        "n_quarantined": len(quarantined),
        "n_eligible": len(eligible),
        "test_rows": test_rows,
        "quarantined_rows": quarantined,
        "eligible_rows": eligible,
    }


def leakage_check_protected_test(
    protected_path: Path,
    generated_source_ids: set[str],
    generated_hashes: set[str],
) -> dict[str, Any]:
    leaked_ids: list[str] = []
    leaked_hashes: list[str] = []
    if protected_path.is_file():
        with protected_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                sid = row.get("source_id") or ""
                sh = row.get("source_row_hash") or ""
                if sid and sid in generated_source_ids:
                    leaked_ids.append(sid)
                if sh and sh in generated_hashes:
                    leaked_hashes.append(sh)
    return {
        "leaked_source_ids": leaked_ids,
        "leaked_hashes": leaked_hashes,
        "ok": not leaked_ids and not leaked_hashes,
    }


def nli_alignment_approved(manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return bool(
        data.get("owner_approved")
        and data.get("train_mapping_reliable")
        and data.get("investigation_complete")
    )
