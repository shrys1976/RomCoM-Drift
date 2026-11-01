#!/usr/bin/env python3
"""Export XNLI en train for generation-index verification.

Run inside an isolated env with ``datasets==1.16.1`` (IndicXNLI pin). Exports from
current ``facebook/xnli`` in this project's venv are rejected by ``run_investigation``
when they match today's HF ordering (not the 2021 translation input).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.nli_provenance import GENERATION_SNAPSHOT_PATH


def main() -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install datasets==1.16.1 in an isolated env, then re-run this script."
        ) from exc

    ds = load_dataset("facebook/xnli", "en", split="train")
    GENERATION_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GENERATION_SNAPSHOT_PATH.open("w", encoding="utf-8") as fh:
        for row in ds:
            fh.write(
                json.dumps(
                    {
                        "premise": row["premise"],
                        "hypothesis": row["hypothesis"],
                        "label": int(row["label"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Wrote {len(ds)} rows to {GENERATION_SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
