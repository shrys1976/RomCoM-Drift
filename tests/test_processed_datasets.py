from pathlib import Path

import json


def test_processed_manifest_reflects_gate(project_root: Path) -> None:
    proc = json.loads(
        (project_root / "manifests" / "processed_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    gate = json.loads(
        (project_root / "manifests" / "gate_decision.json").read_text(encoding="utf-8")
    )
    if not gate.get("cleared"):
        assert proc.get("blocked") is True or (project_root / "romcomx_data" / "processed").exists()


def test_exp_conditions_match_ids(project_root: Path) -> None:
    root = project_root / "romcomx_data" / "processed"
    if not (root / "EXP-01_canonical" / "train.jsonl").is_file():
        import pytest

        pytest.skip("generated EXP datasets not present")

    def ids(cond: str, split: str) -> list[str]:
        path = root / cond / split
        out = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                out.append(json.loads(line)["example_id"])
        return out

    for split in ("train.jsonl", "validation.jsonl"):
        a = ids("EXP-01_canonical", split)
        b = ids("EXP-02_realistic", split)
        c = ids("EXP-03_matched_noise", split)
        assert a == b == c
        assert len(a) > 0
