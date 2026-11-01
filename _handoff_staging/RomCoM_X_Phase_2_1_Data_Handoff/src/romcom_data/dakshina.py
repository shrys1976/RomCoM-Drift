from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from romcom_data.paths import RAW_DAKSHINA_HI
from romcom_data.text_utils import DEVANAGARI, LATIN, nfc


LEXICON_COLS = ["native", "romanized", "count"]


def _read_lexicon(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=LEXICON_COLS,
        dtype={"native": str, "romanized": str, "count": "Int64"},
        encoding="utf-8",
    )
    return df


def audit_lexicon(path: Path, split_name: str) -> dict[str, Any]:
    df = _read_lexicon(path)
    missing_rom = df[df["romanized"].isna() | (df["romanized"].str.strip() == "")]
    dup_pairs = df.duplicated(subset=["native", "romanized"], keep=False)
    variant_counts = df.groupby("native")["romanized"].nunique(dropna=True)
    attestation = df["count"].dropna()
    return {
        "file": path.name,
        "split": split_name,
        "rows": int(len(df)),
        "columns": LEXICON_COLS,
        "unique_native": int(df["native"].nunique()),
        "unique_romanized": int(df["romanized"].nunique(dropna=True)),
        "missing_romanized_rows": int(len(missing_rom)),
        "duplicate_native_roman_pairs": int(dup_pairs.sum()),
        "attestation_min": int(attestation.min()) if len(attestation) else None,
        "attestation_median": float(attestation.median()) if len(attestation) else None,
        "attestation_mean": float(attestation.mean()) if len(attestation) else None,
        "attestation_max": int(attestation.max()) if len(attestation) else None,
        "native_with_multiple_variants": int((variant_counts > 1).sum()),
        "variant_count_distribution": variant_counts.value_counts().sort_index().to_dict(),
        "missing_rows": missing_rom.to_dict(orient="records"),
    }


def audit_train_dev_test_overlap() -> dict[str, Any]:
    paths = {
        "train": RAW_DAKSHINA_HI / "hi.translit.sampled.train.tsv",
        "dev": RAW_DAKSHINA_HI / "hi.translit.sampled.dev.tsv",
        "test": RAW_DAKSHINA_HI / "hi.translit.sampled.test.tsv",
    }
    natives = {k: set(_read_lexicon(p)["native"].dropna()) for k, p in paths.items()}
    return {
        "train_dev_overlap": len(natives["train"] & natives["dev"]),
        "train_test_overlap": len(natives["train"] & natives["test"]),
        "dev_test_overlap": len(natives["dev"] & natives["test"]),
    }


def audit_sentence_pairs() -> dict[str, Any]:
    def load_lines(name: str) -> list[str]:
        return (RAW_DAKSHINA_HI / name).read_text(encoding="utf-8").splitlines()

    out: dict[str, Any] = {}
    for split in ("dev", "test"):
        nat = load_lines(f"hi.romanized.rejoined.{split}.native.txt")
        rom = load_lines(f"hi.romanized.rejoined.{split}.roman.txt")
        out[split] = {
            "native_rows": len(nat),
            "roman_rows": len(rom),
            "length_mismatch": len(nat) != len(rom),
            "duplicate_native": len(nat) - len(set(nat)),
        }
    rejoined = RAW_DAKSHINA_HI / "hi.romanized.rejoined.tsv"
    out["rejoined_note"] = (
        "rejoined.tsv and aligned files are derived held-out sentence views; "
        "not independent training data."
    )
    out["rejoined_rows"] = sum(1 for _ in rejoined.open(encoding="utf-8"))
    return out


def full_dakshina_audit() -> dict[str, Any]:
    audits = [
        audit_lexicon(RAW_DAKSHINA_HI / "hi.translit.sampled.train.tsv", "train"),
        audit_lexicon(RAW_DAKSHINA_HI / "hi.translit.sampled.dev.tsv", "dev"),
        audit_lexicon(RAW_DAKSHINA_HI / "hi.translit.sampled.test.tsv", "test"),
    ]
    return {
        "lexicons": audits,
        "native_word_overlap": audit_train_dev_test_overlap(),
        "sentence_level": audit_sentence_pairs(),
        "attestation_caveat": (
            "Dakshina counts are annotator attestation counts, not social-media "
            "typing probabilities."
        ),
    }


def save_audit_json(audit: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
