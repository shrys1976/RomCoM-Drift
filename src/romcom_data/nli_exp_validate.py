"""Validate NLI experiment artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from romcom_data.augmentation import levenshtein_distance
from romcom_data.nli_exp_freeze import load_exclusion_keys, load_approved_pairs
from romcom_data.nli_exp_build import collect_row_counts_from_disk
from romcom_data.nli_exp_paths import (
    CONDITIONS,
    ENGLISH_RETENTION,
    ENGLISH_RETENTION_SPLITS,
    EXPECTED_COUNTS,
    EXP00,
    EXP00_SPLITS,
    NLI_EXPERIMENTS,
    exp_condition_root,
)
from romcom_data.nli_exp_romanize import NliLexicon, tokenize_and_romanize_field
from romcom_data.nli_exp_variants import VariantPools, apply_exp02, apply_exp03
from romcom_data.text_utils import DEVANAGARI


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_experiments(root: Path | None = None) -> dict[str, Any]:
    root = root or NLI_EXPERIMENTS
    issues: list[str] = []
    approved = {r["example_id"]: r for r in load_approved_pairs()}
    excluded = load_exclusion_keys()

    def check_file(path: Path, split: str, cond: str, scope: str) -> list[dict[str, Any]]:
        if not path.is_file():
            issues.append(f"missing {path}")
            return []
        rows = _load_jsonl(path)
        if len(rows) != EXPECTED_COUNTS[split]:
            issues.append(f"{path} rows {len(rows)} != {EXPECTED_COUNTS[split]}")
        ids = [r["example_id"] for r in rows]
        if len(set(ids)) != len(ids):
            issues.append(f"duplicate ids in {path}")
        for eid in ids:
            if eid not in approved:
                issues.append(f"unknown id {eid} in {path}")
            base = approved.get(eid)
            if base and base["split"] != split:
                issues.append(f"split mismatch {eid} in {path}")
            if base:
                hi = base["hindi_row_index"]
                if hi in excluded.get(split, set()):
                    issues.append(f"excluded row {split}:{hi} in {path}")
        return rows

    by_split_ids: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "dev", "test"):
        for cond in CONDITIONS:
            for scope in ("both", "premise_only", "hypothesis_only"):
                if split == "train" and scope != "both":
                    continue
                path = exp_condition_root(cond) / scope / f"{split}.jsonl"
                rows = check_file(path, split, cond, scope)
                by_split_ids[split].update(r["example_id"] for r in rows)
        if split in ("dev", "test"):
            check_file(EXP00 / f"{split}.jsonl", split, "EXP-00", "both")
            eng = _load_jsonl(ENGLISH_RETENTION / f"{split}.jsonl")
            if len(eng) != EXPECTED_COUNTS[split]:
                issues.append(
                    f"english_retention/{split}.jsonl rows {len(eng)} != {EXPECTED_COUNTS[split]}"
                )
            for row in eng:
                base = approved[row["example_id"]]
                if row["premise"] != base["english_premise"]:
                    issues.append(f"english premise changed {row['example_id']}")
                if row["hypothesis"] != base["english_hypothesis"]:
                    issues.append(f"english hypothesis changed {row['example_id']}")

    for split, ids in by_split_ids.items():
        expected = {eid for eid, r in approved.items() if r["split"] == split}
        if ids != expected:
            issues.append(f"{split} id set mismatch: missing {len(expected - ids)} extra {len(ids - expected)}")

    # Scope consistency on dev/test
    for split in ("dev", "test"):
        for cond in CONDITIONS:
            both = {r["example_id"]: r for r in _load_jsonl(exp_condition_root(cond) / "both" / f"{split}.jsonl")}
            po = {r["example_id"]: r for r in _load_jsonl(exp_condition_root(cond) / "premise_only" / f"{split}.jsonl")}
            ho = {r["example_id"]: r for r in _load_jsonl(exp_condition_root(cond) / "hypothesis_only" / f"{split}.jsonl")}
            for eid, row in both.items():
                base = approved[eid]
                if po[eid]["premise"] != row["premise"]:
                    issues.append(f"premise_only premise mismatch {eid} {cond}")
                if po[eid]["hypothesis"] != base["hindi_hypothesis"]:
                    issues.append(f"premise_only hypothesis not native {eid} {cond}")
                if ho[eid]["hypothesis"] != row["hypothesis"]:
                    issues.append(f"hypothesis_only hypothesis mismatch {eid} {cond}")
                if ho[eid]["premise"] != base["hindi_premise"]:
                    issues.append(f"hypothesis_only premise not native {eid} {cond}")

    # EXP-02/03 trace match from logs
    exp02_exp03_checks = 0
    exp02_exp03_ok = 0
    for split in ("train", "dev", "test"):
        log_path = root / "logs" / f"transformation_log.{split}.jsonl"
        if not log_path.is_file():
            issues.append(f"missing log {log_path}")
            continue
        with log_path.open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                t2 = [t for t in rec.get("exp02_traces", []) if t.get("operation") == "attested_variant"]
                t3 = [t for t in rec.get("exp03_traces", []) if t.get("operation") == "random_noise"]
                if len(t2) != len(t3):
                    issues.append(f"EXP02/03 count mismatch {rec['example_id']}")
                    continue
                for a, b in zip(t2, t3, strict=True):
                    exp02_exp03_checks += 1
                    if a["field"] != b["field"] or a["token_index"] != b["token_index"]:
                        issues.append(f"token index mismatch {rec['example_id']}")
                        continue
                    if a["native_offsets"] != b["native_offsets"]:
                        issues.append(f"native span mismatch {rec['example_id']}")
                        continue
                    da = int(a["levenshtein_distance"])
                    db = int(b["levenshtein_distance"])
                    if da != db or da < 1:
                        issues.append(f"distance mismatch {rec['example_id']}")
                        continue
                    exp02_exp03_ok += 1
                for row in t2:
                    if row["levenshtein_distance"] < 1:
                        issues.append(f"EXP-02 d=0 {rec['example_id']}")

    # Residual Devanagari in romanized fields
    for split in ("train", "dev", "test"):
        for cond in CONDITIONS:
            path = exp_condition_root(cond) / "both" / f"{split}.jsonl"
            if not path.is_file():
                continue
            for row in _load_jsonl(path):
                if DEVANAGARI.search(row["premise"]) or DEVANAGARI.search(row["hypothesis"]):
                    issues.append(f"residual Devanagari {row['example_id']} {cond} both")

    for stale in (ENGLISH_RETENTION / "train.jsonl", EXP00 / "train.jsonl"):
        if stale.is_file():
            issues.append(f"unexpected non-training artifact (remove): {stale.relative_to(root)}")

    disk_counts = collect_row_counts_from_disk()
    manifest_path = root / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_counts = manifest.get("row_counts") or {}
        for key, on_disk in disk_counts.items():
            recorded = manifest_counts.get(key)
            if recorded != on_disk:
                issues.append(f"manifest row_counts[{key}]={recorded} != on_disk {on_disk}")
        for key in manifest_counts:
            if key not in disk_counts:
                issues.append(f"manifest row_counts lists missing file: {key}")
        for entry in manifest.get("evaluation_matrix", {}).get("EXP-00_untrained_base", {}).get(
            "evaluation_sets", []
        ):
            rel = entry["path"]
            if "/train.jsonl" in rel and (
                "english_retention" in rel or "exp00_reference" in rel
            ):
                issues.append(f"train file in evaluation matrix: {rel}")
            from romcom_data.paths import PROJECT_ROOT

            path = PROJECT_ROOT / rel
            if not path.is_file():
                issues.append(f"eval matrix missing {rel}")
                continue
            rows_on_disk = len(_load_jsonl(path))
            if entry.get("rows") != rows_on_disk:
                issues.append(
                    f"eval matrix rows {entry.get('rows')} != {rows_on_disk} for {rel}"
                )

    return {
        "ok": not issues,
        "issues": issues[:200],
        "issue_count": len(issues),
        "exp02_exp03_checks": exp02_exp03_checks,
        "exp02_exp03_ok": exp02_exp03_ok,
    }


def reproduce_sample_hash(sample_size: int = 2000, seed: int = 0) -> dict[str, Any]:
    import random

    lexicon = NliLexicon()
    pools = VariantPools(lexicon)
    rows = load_approved_pairs()
    rng = random.Random(seed)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    mismatches = 0
    checked = 0
    for split, split_rows in by_split.items():
        picks = split_rows if len(split_rows) <= sample_size else rng.sample(split_rows, sample_size)
        for row in picks:
            prem = tokenize_and_romanize_field(row["hindi_premise"], "premise", lexicon)
            hyp = tokenize_and_romanize_field(row["hindi_hypothesis"], "hypothesis", lexicon)
            if prem.fallback_failures or hyp.fallback_failures:
                mismatches += 1
                continue
            eid = row["example_id"]
            _, t2p = apply_exp02(prem, pools, eid)
            _, t2h = apply_exp02(hyp, pools, eid)
            apply_exp03(prem, t2p, pools, eid)
            apply_exp03(hyp, t2h, pools, eid)
            checked += 1
    return {"reproducibility_checked": checked, "fallback_failures_in_sample": mismatches}
