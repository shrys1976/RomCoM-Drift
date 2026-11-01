from pathlib import Path

import json


def test_no_processed_conditions_when_blocked(project_root: Path) -> None:
    gate = json.loads(
        (project_root / "manifests" / "gate_decision.json").read_text(encoding="utf-8")
    )
    if not gate.get("cleared"):
        for cond in ("exp01_canonical", "exp02_realistic", "exp03_matched_noise"):
            train = project_root / "data" / "processed" / cond / "train.jsonl"
            assert not train.is_file()


def test_config_hash_in_augmentation_manifest(project_root: Path) -> None:
    manifest = json.loads(
        (project_root / "manifests" / "augmentation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.get("config_sha256")
    assert len(manifest["config_sha256"]) == 64
