from pathlib import Path

import pandas as pd


def test_dakshina_train_dev_native_disjoint(project_root: Path) -> None:
    import pytest

    hi = project_root / "data" / "raw" / "dakshina" / "hi"
    if not (hi / "hi.translit.sampled.train.tsv").is_file():
        pytest.skip("raw Dakshina files not downloaded")
    def natives(name: str) -> set[str]:
        df = pd.read_csv(hi / name, sep="\t", header=None, usecols=[0], names=["native"])
        return set(df["native"].dropna())

    train = natives("hi.translit.sampled.train.tsv")
    dev = natives("hi.translit.sampled.dev.tsv")
    test = natives("hi.translit.sampled.test.tsv")
    assert train.isdisjoint(dev)
    assert train.isdisjoint(test)


def test_inventory_sourced_from_dakshina_train_only(project_root: Path) -> None:
    import pytest

    inv = project_root / "data" / "interim" / "romanization" / "dakshina_hi_variant_inventory.tsv"
    if not inv.is_file():
        pytest.skip("generated romanization inventory not present")
    df = pd.read_csv(inv, sep="\t")
    assert (df["source_dataset"] == "google-research-datasets/dakshina").all()
    assert (df["source_split"] == "train").all()
