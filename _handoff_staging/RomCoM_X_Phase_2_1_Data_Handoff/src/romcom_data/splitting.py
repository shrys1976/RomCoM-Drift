from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from romcom_data.stable_hash import derive_rng


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def nli_group_key(row: dict[str, Any]) -> str:
    return _norm(row["premise_native"])


def paraphrase_groups(rows: list[dict[str, Any]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in rows:
        s1, s2 = _norm(row["sentence1_native"]), _norm(row["sentence2_native"])
        union(s1, s2)
    return {k: find(k) for k in parent}


def assign_splits(
    rows: list[dict[str, Any]],
    task: str,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> list[dict[str, Any]]:
    eligible = [r for r in rows if r.get("original_split") != "test"]
    official_train = [r for r in eligible if r.get("original_split") == "train"]
    official_dev = [r for r in eligible if r.get("original_split") in {"dev", "validation"}]

    if official_train and official_dev:
        para_root = paraphrase_groups(eligible) if task == "paraphrase" else {}
        train_ids = {r["source_id"] for r in official_train}
        for r in eligible:
            if task == "nli":
                r["group_id"] = nli_group_key(r)
            elif task == "paraphrase":
                r["group_id"] = para_root[_norm(r["sentence1_native"])]
            else:
                r["group_id"] = r["source_id"]
            r["project_split"] = (
                "train"
                if r["source_id"] in train_ids or r["original_split"] == "train"
                else "validation"
            )
            if r["original_split"] in {"dev", "validation"}:
                r["project_split"] = "validation"
        return eligible

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    para_root = paraphrase_groups(eligible) if task == "paraphrase" else {}
    for row in eligible:
        if task == "nli":
            key = nli_group_key(row)
        elif task == "paraphrase":
            key = para_root[_norm(row["sentence1_native"])]
        else:
            key = row["source_id"]
        row["group_id"] = key
        groups[key].append(row)

    items = list(groups.items())
    rng = derive_rng(seed, task, "split")
    rng.shuffle(items)

    by_label: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for key, members in items:
        label = str(members[0].get("normalized_label"))
        by_label[label].append((key, members))

    train_keys: set[str] = set()
    val_keys: set[str] = set()
    for label, bucket in by_label.items():
        n_train = max(1, int(round(len(bucket) * train_ratio))) if len(bucket) > 1 else len(bucket)
        if len(bucket) == 1:
            n_train = 1
        for i, (key, _) in enumerate(bucket):
            if i < n_train:
                train_keys.add(key)
            else:
                val_keys.add(key)

    for key, members in groups.items():
        split = "train" if key in train_keys else "validation"
        for row in members:
            row["project_split"] = split
            row["group_id"] = key
    return eligible
