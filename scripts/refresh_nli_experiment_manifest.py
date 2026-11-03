#!/usr/bin/env python3
"""Refresh experiment_manifest row counts and checksums from on-disk JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.nli_exp_build import refresh_experiment_manifest


def main() -> int:
    manifest = refresh_experiment_manifest()
    print(json.dumps({"row_counts": manifest["row_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
