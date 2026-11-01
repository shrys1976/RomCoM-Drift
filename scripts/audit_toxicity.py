#!/usr/bin/env python3
"""Run Toxic_Matrix / IndicAlign toxicity source audit (inspection only)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.toxicity_audit import build_audit_zip, run_toxicity_audit


def main() -> int:
    refresh = os.environ.get("TOXICITY_AUDIT_REFRESH_HF", "").lower() in ("1", "true", "yes")
    result = run_toxicity_audit(refresh_hf=refresh)
    zip_path = build_audit_zip(ROOT / "RomCoM_X_Phase_2_1_Toxicity_Audit.zip")
    print(json.dumps(result["handoff_status"], indent=2))
    print(json.dumps(result["recommendation"], indent=2))
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
