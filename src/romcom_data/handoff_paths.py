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
NLI_INVESTIGATION = HANDOFF / "nli_investigation"
PARAPHRASE_AUDIT = HANDOFF / "paraphrase_audit"
TOXICITY_AUDIT = HANDOFF / "toxicity_audit"
TOXICITY_PROCESSED = HANDOFF / "toxicity_processed"
TOXICITY_EXP01 = TOXICITY_PROCESSED / "EXP-01_canonical"
TOXICITY_EXP02 = TOXICITY_PROCESSED / "EXP-02_realistic"
TOXICITY_EXP03 = TOXICITY_PROCESSED / "EXP-03_matched_noise"
TOXICITY_TRACES = TOXICITY_PROCESSED / "traces"
TOXICITY_MANIFESTS = TOXICITY_PROCESSED / "manifests"
TOXICITY_REPORTS = TOXICITY_PROCESSED / "reports"
TOXICITY_EVAL_REF = TOXICITY_PROCESSED / "evaluation_reference"
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
        NLI_INVESTIGATION,
        PARAPHRASE_AUDIT,
        TOXICITY_AUDIT,
        TOXICITY_PROCESSED,
        TOXICITY_EXP01,
        TOXICITY_EXP02,
        TOXICITY_EXP03,
        TOXICITY_TRACES / "EXP-01",
        TOXICITY_TRACES / "EXP-02",
        TOXICITY_TRACES / "EXP-03",
        TOXICITY_MANIFESTS,
        TOXICITY_REPORTS,
        TOXICITY_EVAL_REF,
        SRC_MIRROR / "source_loaders",
        SRC_MIRROR / "preprocessing",
        SRC_MIRROR / "augmentation",
        SRC_MIRROR / "splitting",
        SRC_MIRROR / "validation",
    ):
        path.mkdir(parents=True, exist_ok=True)
