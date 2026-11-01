from pathlib import Path

import pandas as pd


def test_variant_inventory_columns(project_root: Path) -> None:
    import pytest

    path = project_root / "data" / "interim" / "romanization" / "dakshina_hi_variant_inventory.tsv"
    if not path.is_file():
        pytest.skip("generated romanization inventory not present")
    df = pd.read_csv(path, sep="\t")
    required = {
        "native",
        "variant",
        "normalized_variant",
        "attestation_count",
        "canonical_tie",
        "license_ref",
    }
    assert required.issubset(df.columns)


def test_gate_decision_recorded(project_root: Path) -> None:
    path = project_root / "manifests" / "gate_decision.json"
    assert path.is_file()
