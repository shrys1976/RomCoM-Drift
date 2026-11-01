from pathlib import Path

import pandas as pd


def test_train_lexicon_only_in_alias_index(project_root: Path) -> None:
    import pytest

    path = project_root / "data" / "interim" / "romanization" / "dakshina_hi_variant_inventory.tsv"
    if not path.is_file():
        pytest.skip("generated romanization inventory not present")
    inv = pd.read_csv(
        path,
        sep="\t",
    )
    assert (inv["source_split"] == "train").all()


def test_ambiguous_aliases_excluded_from_index(project_root: Path) -> None:
    amb = project_root / "data" / "interim" / "romanization" / "ambiguous_aliases.tsv"
    alias = project_root / "data" / "interim" / "romanization" / "dakshina_hi_alias_index.tsv"
    if amb.is_file() and amb.stat().st_size > 0:
        amb_df = pd.read_csv(amb, sep="\t")
        alias_df = pd.read_csv(alias, sep="\t")
        overlap = set(amb_df["normalized_variant"]) & set(alias_df["normalized_variant"])
        assert not overlap


def test_invalid_rows_logged(project_root: Path) -> None:
    import pytest

    path = project_root / "data" / "interim" / "romanization" / "invalid_lexicon_rows.tsv"
    if not path.is_file():
        pytest.skip("generated romanization inventory not present")
    df = pd.read_csv(path, sep="\t")
    assert len(df) >= 2
