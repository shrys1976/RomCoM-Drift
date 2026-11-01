from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from romcom_data.handoff_paths import RAW_NLI
from romcom_data.stable_hash import stable_hash_hex
from romcom_data.text_utils import nfc

REPO = "Divyanshu/indicxnli"
LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


def _load_json_records(path: Path) -> tuple[str, list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return "unknown", data
    if isinstance(data, dict):
        if "premise" in data or (data and isinstance(next(iter(data.values())), list)):
            split_name, rows = next(iter(data.items()))
            if isinstance(rows, list):
                return str(split_name), rows
        rows = data.get("data") or data.get("rows")
        if isinstance(rows, list):
            return "unknown", rows
    raise ValueError(f"Unrecognized IndicXNLI JSON layout: {path}")


def acquire_indicxnli(revision: str | None = None) -> dict[str, Any]:
    api = HfApi()
    info = api.dataset_info(REPO, revision=revision)
    sha = info.sha
    hi_files = sorted(
        f.rfilename
        for f in (info.siblings or [])
        if "xnli_hi" in f.rfilename.lower() or (
            "/hi" in f.rfilename.lower() and f.rfilename.endswith(".json")
        )
    )
    if not hi_files:
        hi_files = sorted(
            f.rfilename for f in (info.siblings or []) if "hi" in f.rfilename.lower()
        )
    stored: list[str] = []
    RAW_NLI.mkdir(parents=True, exist_ok=True)
    for rel in hi_files:
        local = hf_hub_download(
            repo_id=REPO, filename=rel, repo_type="dataset", revision=sha
        )
        dest = RAW_NLI / rel.replace("/", "__")
        dest.write_bytes(Path(local).read_bytes())
        stored.append(dest.name)
    return {
        "repository": REPO,
        "revision": sha,
        "license": "CC-BY-4.0",
        "license_status": "cleared_for_internal_use",
        "files": stored,
        "all_hi_files": hi_files,
    }


def _record_fields(row: dict[str, Any]) -> tuple[str, str, Any]:
    premise = row.get("premise") or row.get("sentence1") or row.get("Premises") or ""
    hypothesis = row.get("hypothesis") or row.get("sentence2") or row.get("Hypothesis") or ""
    label = row.get("label")
    if label is None:
        label = row.get("gold_label")
    return str(premise), str(hypothesis), label


def load_normalized_nli(meta: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fname in meta["files"]:
        path = RAW_NLI / fname
        lower = fname.lower()
        if "backward" in lower:
            continue
        orig_split, rows = _load_json_records(path)
        if "test" in lower:
            continue
        if "dev" in lower or "valid" in lower:
            partition = "dev"
        elif "train" in lower:
            partition = "train"
        else:
            partition = orig_split if orig_split in {"train", "dev", "test"} else "unknown"
        direction = "backward" if "backward" in lower else "forward"
        for idx, row in enumerate(rows):
            premise, hypothesis, label = _record_fields(row)
            if not premise.strip() or not hypothesis.strip():
                continue
            if isinstance(label, str) and label.isdigit():
                label = int(label)
            original_label = label
            if isinstance(label, int) and label in LABEL_MAP:
                normalized = LABEL_MAP[label]
            else:
                normalized = str(label).lower()
            source_id = str(row.get("pairID") or row.get("id") or "")
            if not source_id:
                source_id = "nli-" + stable_hash_hex(
                    meta["revision"], direction, partition, str(idx), premise, hypothesis
                )[:16]
            if direction == "backward":
                continue
            records.append(
                {
                    "source_id": source_id,
                    "example_id": f"{source_id}:{direction}:{partition}:{idx}",
                    "premise_native": nfc(premise),
                    "hypothesis_native": nfc(hypothesis),
                    "premise_en": nfc(str(row.get("premise_en") or row.get("english_premise") or "")),
                    "hypothesis_en": nfc(
                        str(row.get("hypothesis_en") or row.get("english_hypothesis") or "")
                    ),
                    "original_label": original_label,
                    "normalized_label": normalized,
                    "language": "hi",
                    "original_split": partition,
                    "source_dataset": REPO,
                    "source_revision": meta["revision"],
                    "source_file": fname,
                    "direction": direction,
                    "row_index": idx,
                    "license_ref": "CC-BY-4.0",
                }
            )
    return records
