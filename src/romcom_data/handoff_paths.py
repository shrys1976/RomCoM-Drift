from __future__ import annotations

from pathlib import Path

from romcom_data.paths import PROJECT_ROOT

HANDOFF = PROJECT_ROOT / "romcomx_data"

RAW = HANDOFF / "raw"
RAW_NLI = RAW / "indicxnli"
RAW_PARA = RAW / "indicxparaphrase"
RAW_TOX = RAW / "toxic_matrix"
RAW_DAK = RAW / "dakshina"
RAW_COMI = RAW / "comi_lingua"

INVENTORIES = HANDOFF / "inventories"
INTERMEDIATE = HANDOFF / "intermediate"
PROCESSED = HANDOFF / "processed"
EXP01 = PROCESSED / "EXP-01_canonical"
EXP02 = PROCESSED / "EXP-02_realistic"
EXP03 = PROCESSED / "EXP-03_matched_noise"
MANIFESTS = HANDOFF / "manifests"
TRACES = HANDOFF / "traces"
REPORTS = HANDOFF / "reports"
SRC_MIRROR = HANDOFF / "src"


def ensure_handoff_layout() -> None:
    for path in (
        RAW_NLI,
        RAW_PARA,
        RAW_TOX,
        RAW_DAK,
        RAW_COMI,
        INVENTORIES,
        INTERMEDIATE,
        EXP01,
        EXP02,
        EXP03,
        MANIFESTS,
        TRACES / "EXP-01",
        TRACES / "EXP-02",
        TRACES / "EXP-03",
        REPORTS,
        SRC_MIRROR / "source_loaders",
        SRC_MIRROR / "preprocessing",
        SRC_MIRROR / "augmentation",
        SRC_MIRROR / "splitting",
        SRC_MIRROR / "validation",
    ):
        path.mkdir(parents=True, exist_ok=True)
