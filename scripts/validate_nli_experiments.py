#!/usr/bin/env python3
"""Validate built NLI experiment artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.nli_exp_paths import NLI_EXPERIMENTS
from romcom_data.nli_exp_validate import reproduce_sample_hash, validate_experiments


def main() -> int:
    validation = validate_experiments()
    repro = reproduce_sample_hash()
    report = {**validation, "reproducibility": repro}
    path = NLI_EXPERIMENTS / "validation_report.md"
    lines = [
        "# NLI experiment validation",
        "",
        f"**Status:** {'PASS' if validation['ok'] else 'FAIL'}",
        "",
        f"- EXP-02/EXP-03 checks: {validation['exp02_exp03_ok']}/{validation['exp02_exp03_checks']}",
        f"- Issues: {validation['issue_count']}",
        "",
    ]
    if validation["issues"]:
        lines.append("## Issues")
        for issue in validation["issues"]:
            lines.append(f"- {issue}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (NLI_EXPERIMENTS / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": validation["ok"], "issues": validation["issue_count"]}, indent=2))
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
