"""Paths for RomCoM-X NLI experiment artifacts."""

from __future__ import annotations

from pathlib import Path

from romcom_data.paths import PROJECT_ROOT

NLI_ALIGNMENT = PROJECT_ROOT / "romcomx_data" / "nli_alignment"
NLI_EXPERIMENTS = PROJECT_ROOT / "romcomx_data" / "nli_experiments"
NLI_INVENTORIES = PROJECT_ROOT / "romcomx_data" / "inventories"

VERIFIED_JSONL = NLI_ALIGNMENT / "verified_pairs.jsonl"
EXCLUSIONS_JSON = NLI_ALIGNMENT / "conditional_approval_exclusions.json"
ALIGNMENT_CHECKSUMS = NLI_ALIGNMENT / "checksums.sha256"

EXP00 = NLI_EXPERIMENTS / "exp00_reference"
EXP01 = NLI_EXPERIMENTS / "exp01"
EXP02 = NLI_EXPERIMENTS / "exp02"
EXP03 = NLI_EXPERIMENTS / "exp03"
ENGLISH_RETENTION = NLI_EXPERIMENTS / "english_retention"
NLI_EXP_LOGS = NLI_EXPERIMENTS / "logs"
NLI_EXP_REPORTS = NLI_EXPERIMENTS / "reports"

LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}

EXPECTED_COUNTS = {"train": 375_565, "dev": 2_487, "test": 5_007}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())

SCOPES_EVAL = ("both", "premise_only", "hypothesis_only")
SCOPES_TRAIN = ("both",)
CONDITIONS = ("EXP-01", "EXP-02", "EXP-03")
ENGLISH_RETENTION_SPLITS = ("dev", "test")
EXP00_SPLITS = ("dev", "test")


def exp_condition_root(condition: str) -> Path:
    if condition == "EXP-01":
        return EXP01
    if condition == "EXP-02":
        return EXP02
    if condition == "EXP-03":
        return EXP03
    raise ValueError(f"unknown condition {condition}")


def ensure_nli_experiment_layout() -> None:
    for root in (EXP00, ENGLISH_RETENTION, NLI_EXP_LOGS, NLI_EXP_REPORTS):
        root.mkdir(parents=True, exist_ok=True)
    for exp in (EXP01, EXP02, EXP03):
        for scope in ("both", "premise_only", "hypothesis_only"):
            (exp / scope).mkdir(parents=True, exist_ok=True)
