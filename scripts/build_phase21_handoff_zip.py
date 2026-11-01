#!/usr/bin/env python3
"""Package RomCoM_X_Phase_2_1_Data_Handoff.zip from romcomx_data/."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "romcomx_data"
PKG = ROOT / "_handoff_staging" / "RomCoM_X_Phase_2_1_Data_Handoff"
ZIP = ROOT / "RomCoM_X_Phase_2_1_Data_Handoff.zip"

SKIP_PARTS = {"__pycache__", ".pyc"}


def should_skip(p: Path) -> bool:
    return any(part in SKIP_PARTS for part in p.parts) or p.suffix == ".pyc"


def main() -> None:
    if PKG.parent.exists():
        shutil.rmtree(PKG.parent)
    PKG.mkdir(parents=True)
    for name in (
        "reports",
        "inventories",
        "manifests",
        "processed",
        "intermediate",
        "traces",
    ):
        src = HANDOFF / name
        if src.is_dir():
            shutil.copytree(src, PKG / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "src" / "romcom_data", PKG / "src" / "romcom_data", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "tests", PKG / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "configs" / "data_augmentation.yaml", PKG / "manifests" / "augmentation_config.yaml")
    (PKG / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "download_raw_datasets.py", PKG / "scripts" / "download_raw_datasets.py")
    shutil.copy2(ROOT / "scripts" / "build_phase21_handoff_zip.py", PKG / "scripts" / "build_phase21_handoff_zip.py")
    for rep in ("final_data_handoff_report.md", "source_audit_report.md"):
        src = HANDOFF / "reports" / rep
        if src.is_file():
            shutil.copy2(src, PKG / rep)
    lines = []
    for p in sorted(PKG.rglob("*")):
        if p.is_file() and not should_skip(p):
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.relative_to(PKG).as_posix()}")
    (PKG / "manifests" / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(PKG.rglob("*")):
            if p.is_file() and not should_skip(p):
                zf.write(p, arcname=f"RomCoM_X_Phase_2_1_Data_Handoff/{p.relative_to(PKG).as_posix()}")
    print(f"Wrote {ZIP} ({ZIP.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
