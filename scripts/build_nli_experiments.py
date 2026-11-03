#!/usr/bin/env python3
"""Build RomCoM-X NLI experiment datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.nli_exp_build import run_build
from romcom_data.nli_exp_paths import NLI_EXPERIMENTS, NLI_EXP_REPORTS


def main() -> int:
    result = run_build(v2_partial=True)
    status_path = NLI_EXPERIMENTS / "construction_status.json"
    status_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "block_reason": result.get("block_reason", "")}, indent=2))
    ok = result["status"] in ("READY_FOR_OWNER_REVIEW", "READY_FOR_OWNER_REVIEW_V2")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
