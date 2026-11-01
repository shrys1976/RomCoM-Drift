from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_INDI = PROJECT_ROOT / "data" / "raw" / "indi_romcom"
RAW_DAKSHINA_HI = PROJECT_ROOT / "data" / "raw" / "dakshina" / "hi"
RAW_COMI = PROJECT_ROOT / "data" / "raw" / "comi_lingua"
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MANIFESTS = PROJECT_ROOT / "manifests"
REPORTS = PROJECT_ROOT / "reports" / "data_preparation"
CONFIG = PROJECT_ROOT / "configs" / "data_augmentation.yaml"


def ensure_layout() -> None:
    for path in (
        RAW_INDI,
        RAW_DAKSHINA_HI,
        RAW_COMI,
        INTERIM / "source_records",
        INTERIM / "split_manifests",
        INTERIM / "romanization",
        INTERIM / "qa_samples",
        PROCESSED / "source_original",
        PROCESSED / "exp01_canonical",
        PROCESSED / "exp02_realistic",
        PROCESSED / "exp03_matched_noise",
        MANIFESTS,
        REPORTS,
    ):
        path.mkdir(parents=True, exist_ok=True)
