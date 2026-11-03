"""Freeze conditionally approved EN–HI XNLI pairs for experiment construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from romcom_data.checksums import sha256_file
from romcom_data.nli_exp_paths import (
    ALIGNMENT_CHECKSUMS,
    EXCLUSIONS_JSON,
    EXPECTED_COUNTS,
    EXPECTED_TOTAL,
    LABEL_TO_ID,
    NLI_EXPERIMENTS,
    VERIFIED_JSONL,
)

INDICXNLI_REVISION = "7092c27872e919f31d0496fb8b9c47bd2cba3f6c"


def _parse_checksums(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        mapping[rel.strip()] = digest.strip()
    return mapping


def load_exclusion_keys(path: Path | None = None) -> dict[str, set[int]]:
    data = json.loads((path or EXCLUSIONS_JSON).read_text(encoding="utf-8"))
    by_split: dict[str, set[int]] = {"train": set(), "dev": set(), "test": set()}
    for idx in data.get("identical_text_conflicting_label_rows", []):
        by_split["train"].add(int(idx))
    for idx in data.get("degenerate_repetition_rows", []):
        by_split["train"].add(int(idx))
    for row in data.get("semantic_not_equivalent_rows", []):
        split = row["split"]
        hi = row.get("hindi_row_index")
        if hi is not None:
            by_split[split].add(int(hi))
    return by_split


def _should_drop(record: dict[str, Any], excluded: dict[str, set[int]]) -> bool:
    split = record["split"]
    hindi_index = int(record["hindi_row_index"])
    return hindi_index in excluded.get(split, set())


def approved_record(raw: dict[str, Any]) -> dict[str, Any]:
    split = raw["split"]
    hindi_index = int(raw["hindi_row_index"])
    label = raw["nli_label"]
    pair_id = raw.get("english_pair_id")
    return {
        "example_id": f"{split}-{hindi_index:06d}",
        "split": split,
        "nli_label": label,
        "label_id": LABEL_TO_ID[label],
        "english_premise": raw["english_premise"],
        "english_hypothesis": raw["english_hypothesis"],
        "hindi_premise": raw["hindi_premise"],
        "hindi_hypothesis": raw["hindi_hypothesis"],
        "english_row_index": int(raw["english_row_index"]),
        "hindi_row_index": hindi_index,
        "english_pair_id": raw.get("english_pair_id") or pair_id or "",
        "english_source": raw.get("english_source", ""),
        "hindi_source": raw.get("hindi_source", ""),
        "matching_method": raw.get("matching_method", ""),
        "matching_evidence": raw.get("matching_evidence"),
        "validation_status": raw.get("validation_status", "verified"),
    }


def iter_approved_pairs(
    *,
    verify_jsonl_checksum: bool = True,
) -> Iterator[dict[str, Any]]:
    if verify_jsonl_checksum:
        expected = _parse_checksums(ALIGNMENT_CHECKSUMS).get(
            "romcomx_data/nli_alignment/verified_pairs.jsonl"
        )
        if expected and sha256_file(VERIFIED_JSONL) != expected:
            raise ValueError("verified_pairs.jsonl checksum mismatch")
    excluded = load_exclusion_keys()
    with VERIFIED_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if _should_drop(raw, excluded):
                continue
            yield approved_record(raw)


def load_approved_pairs(*, verify_jsonl_checksum: bool = True) -> list[dict[str, Any]]:
    return list(iter_approved_pairs(verify_jsonl_checksum=verify_jsonl_checksum))


def assert_expected_counts(rows: list[dict[str, Any]]) -> None:
    counts = Counter(r["split"] for r in rows)
    for split, expected in EXPECTED_COUNTS.items():
        if counts[split] != expected:
            raise ValueError(f"{split} count {counts[split]} != expected {expected}")
    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"total {len(rows)} != expected {EXPECTED_TOTAL}")


def write_alignment_manifest(
    rows: list[dict[str, Any]],
    out_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or NLI_EXPERIMENTS
    checksums = _parse_checksums(ALIGNMENT_CHECKSUMS)
    excluded = load_exclusion_keys()
    ids = sorted(r["example_id"] for r in rows)
    id_digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    before = Counter()
    with VERIFIED_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            split = json.loads(line)["split"]
            before[split] += 1
    after = Counter(r["split"] for r in rows)
    manifest = {
        "approved_total": len(rows),
        "expected_total": EXPECTED_TOTAL,
        "counts_before_exclusion": dict(before),
        "counts_after_exclusion": dict(after),
        "excluded_hindi_indexes": {
            split: sorted(idxs) for split, idxs in excluded.items() if idxs
        },
        "approved_example_ids_sha256": id_digest,
        "artifact_sha256": {
            "verified_pairs.jsonl": checksums.get(
                "romcomx_data/nli_alignment/verified_pairs.jsonl"
            ),
            "conditional_approval_exclusions.json": sha256_file(EXCLUSIONS_JSON),
            "validation_report.md": checksums.get(
                "romcomx_data/nli_alignment/validation_report.md"
            ),
            "independent_validation_statistics.json": checksums.get(
                "romcomx_data/nli_alignment/independent_validation_statistics.json"
            ),
            "english_train_tsv": checksums.get(
                "romcomx_data/raw/xnli_mt/XNLI-MT-1.0/multinli/multinli.train.en.tsv"
            ),
            "hindi_train_json": checksums.get(
                "romcomx_data/raw/indicxnli/forward__train__xnli_hi.json"
            ),
        },
        "source_revisions": {
            "indicxnli_hub_revision": INDICXNLI_REVISION,
            "indicxnli_make_dataset": "https://github.com/divyanshuaggarwal/IndicXNLI/src/make_dataset.py",
        },
    }
    path = out_dir / "alignment_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["_path"] = str(path)
    manifest["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest
