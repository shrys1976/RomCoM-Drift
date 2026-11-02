#!/usr/bin/env python3
"""Audit Hindi-native IndicXNLI. Does not align English or build EXP datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.nli_hindi_native import (
    iter_official_rows,
    quality_report,
    split_integrity,
    transformation_feasibility,
    write_documents,
)


def main() -> int:
    rows, file_meta = iter_official_rows()
    integrity = split_integrity(rows, file_meta)
    quality = quality_report(rows)
    feasibility = transformation_feasibility(rows)
    status = write_documents(rows, file_meta, integrity, quality, feasibility)
    print(json.dumps({"status": status, "counts": integrity["counts"]}, indent=2))
    return 0 if status == "READY_FOR_OWNER_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
