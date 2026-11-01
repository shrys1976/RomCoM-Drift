from __future__ import annotations

from pathlib import Path

import pandas as pd

from romcom_data.handoff_paths import INVENTORIES
from romcom_data.paths import INTERIM


def export_phase2_inventories(inv: pd.DataFrame, out_dir: Path | None = None) -> None:
    out = out_dir or INVENTORIES
    out.mkdir(parents=True, exist_ok=True)
    cols = [
        "native_word",
        "canonical_romanization",
        "romanized_variant",
        "attestation_count",
        "normalized_probability",
        "is_canonical",
        "number_of_variants",
        "source_split",
        "source_dataset",
    ]
    inv[cols].to_csv(out / "dakshina_hi_variant_inventory.tsv", sep="\t", index=False)

    interim = INTERIM / "romanization"
    alias = pd.read_csv(interim / "dakshina_hi_alias_index.tsv", sep="\t")

    def _maybe(name: str) -> pd.DataFrame:
        p = interim / name
        if not p.is_file() or p.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(p, sep="\t")

    amb = _maybe("ambiguous_aliases.tsv")
    eng = _maybe("english_collision_aliases.tsv")

    rows = []
    for r in alias.itertuples():
        native = getattr(r, "native")
        grp = inv[inv["native"] == native]
        counts = "|".join(str(int(c)) for c in grp["attestation_count"].tolist())
        rows.append(
            {
                "romanized_alias": r.normalized_variant,
                "candidate_native_words": native,
                "candidate_count": 1,
                "ambiguity_status": "unique",
                "associated_variant_counts": counts,
            }
        )
    if len(amb):
        for r in amb.itertuples():
            rows.append(
                {
                    "romanized_alias": r.normalized_variant,
                    "candidate_native_words": r.native_words,
                    "candidate_count": len(str(r.native_words).split("|")),
                    "ambiguity_status": "ambiguous",
                    "associated_variant_counts": "",
                }
            )
    if len(eng):
        for r in eng.itertuples():
            rows.append(
                {
                    "romanized_alias": r.normalized_variant,
                    "candidate_native_words": r.native_word,
                    "candidate_count": 1,
                    "ambiguity_status": "english_collision",
                    "associated_variant_counts": "",
                }
            )
    pd.DataFrame(rows).to_csv(out / "dakshina_hi_alias_index.tsv", sep="\t", index=False)

    freq = (
        inv.groupby("number_of_variants")
        .size()
        .reset_index(name="native_word_count")
    )
    freq.to_csv(out / "variant_frequency_report.csv", index=False)

    if len(amb):
        amb.to_csv(out / "ambiguity_report.csv", index=False)
    else:
        pd.DataFrame(columns=["normalized_variant", "native_words"]).to_csv(
            out / "ambiguity_report.csv", index=False
        )

    ties = inv[inv["canonical_tie"]].drop_duplicates("native")
    ties[["native", "canonical_candidate", "variant_count_for_native"]].to_csv(
        out / "canonical_tie_report.csv", index=False
    )
