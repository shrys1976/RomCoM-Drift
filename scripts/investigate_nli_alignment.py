#!/usr/bin/env python3
"""Run NLI provenance investigation; does not rebuild processed NLI datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.nli_provenance import run_investigation


def main() -> int:
    print(json.dumps(run_investigation(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
