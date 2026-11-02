#!/usr/bin/env python3
"""Build Phase 2.1 toxicity EXP-01/02/03 datasets from the frozen source manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from load_env import bootstrap_project
from romcom_data.toxicity_construct import run_toxicity_construction


def main() -> int:
    bootstrap_project()
    result = run_toxicity_construction()
    print(json.dumps({"status": result["status"], "validation_ok": result["validation"]["ok"]}, indent=2))
    return 0 if result["status"] == "READY_FOR_OWNER_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
