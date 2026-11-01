#!/usr/bin/env python3
"""Build RomCoM_X_Phase_2_1_NLI_Investigation.zip for owner NLI alignment review."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.handoff_paths import NLI_INVESTIGATION
from romcom_data.nli_provenance import run_investigation

ZIP_NAME = "RomCoM_X_Phase_2_1_NLI_Investigation.zip"
BUNDLE_FILES = (
    "nli_alignment_investigation_report.md",
    "nli_alignment_manifest.csv",
    "nli_alignment_sample.csv",
    "nli_alignment_metrics.json",
    "nli_source_revisions.json",
    "alignment_method.py",
    "make_dataset.py",
    "alignment_audit.log",
    "nli_alignment_status.json",
    "README.txt",
)


def main() -> int:
    log_buf = io.StringIO()
    log_buf.write(f"started_utc={datetime.now(UTC).isoformat()}\n")
    try:
        report = run_investigation()
        log_buf.write("run_investigation=ok\n")
        log_buf.write(json.dumps(report, indent=2))
        log_buf.write("\n")
    except Exception as exc:
        log_buf.write(f"run_investigation=failed\n{type(exc).__name__}: {exc}\n")
        (NLI_INVESTIGATION / "alignment_audit.log").write_text(log_buf.getvalue(), encoding="utf-8")
        raise

    log_buf.write("\nvalidation:\n")
    log_buf.write(f"  train_mapping_reliable={report.get('train_mapping_reliable')}\n")
    log_buf.write(f"  owner_review_sample_rows={report.get('owner_review_sample_rows')}\n")
    (NLI_INVESTIGATION / "alignment_audit.log").write_text(log_buf.getvalue(), encoding="utf-8")

    zip_path = ROOT / ZIP_NAME
    lines: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in BUNDLE_FILES:
            src = NLI_INVESTIGATION / name
            if not src.is_file():
                raise FileNotFoundError(f"Missing bundle file: {src}")
            data = src.read_bytes()
            zf.writestr(name, data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
        zf.writestr("checksums.sha256", "\n".join(lines) + "\n")

    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1024**2:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
