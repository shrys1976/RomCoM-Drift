#!/usr/bin/env python3
"""Apply owner review, close NLI investigation, and write final handoff ZIP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.nli_investigation_close import build_final_handoff_zip, close_nli_investigation

ZIP_NAME = "RomCoM_X_Phase_2_1_NLI_Investigation_Final.zip"


def main() -> int:
    result = close_nli_investigation()
    zip_path = build_final_handoff_zip(ROOT / ZIP_NAME)
    print(json.dumps(result["status"], indent=2))
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1024**2:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
