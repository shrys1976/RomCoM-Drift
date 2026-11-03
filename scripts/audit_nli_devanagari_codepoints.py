#!/usr/bin/env python3
"""Read-only audit of Devanagari codepoints in approved NLI Hindi text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.nli_exp_freeze import iter_approved_pairs
from romcom_data.nli_exp_paths import NLI_EXP_REPORTS, ensure_nli_experiment_layout
from romcom_data.nli_exp_romanize import audit_devanagari_codepoints


def main() -> int:
    ensure_nli_experiment_layout()
    texts: list[str] = []
    for row in iter_approved_pairs():
        texts.append(row["hindi_premise"])
        texts.append(row["hindi_hypothesis"])
    report = audit_devanagari_codepoints(texts)
    out = NLI_EXP_REPORTS / "devanagari_codepoint_audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "unhandled": report["unhandled_codepoints"][:20]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
