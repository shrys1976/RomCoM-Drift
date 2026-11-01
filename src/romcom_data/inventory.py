from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from romcom_data.paths import INTERIM, PROJECT_ROOT
from romcom_data.text_utils import DEVANAGARI, LATIN, case_fold_key, nfc


def load_english_collision_set(config: dict[str, Any]) -> set[str]:
    rel = config.get("english_collision_wordlist", "")
    path = PROJECT_ROOT / rel
    words = {
        line.strip().casefold()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    return words


def build_variant_inventory(
    lexicon_path: Path,
    config: dict[str, Any],
    out_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or (INTERIM / "romanization")
    out_dir.mkdir(parents=True, exist_ok=True)

    inv_cfg = config.get("inventory", {})
    license_ref = inv_cfg.get("license_ref", "CC-BY-SA-4.0")
    source_dataset = inv_cfg.get("source_dataset", "google-research-datasets/dakshina")

    df = pd.read_csv(
        lexicon_path,
        sep="\t",
        header=None,
        names=["native", "variant", "attestation_count"],
        dtype={"native": str, "variant": str, "attestation_count": "Int64"},
    )

    english = load_english_collision_set(config)
    invalid_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        native = nfc(str(row["native"]).strip())
        variant_raw = row["variant"]
        count = row["attestation_count"]
        reasons = []
        if pd.isna(variant_raw) or str(variant_raw).strip() == "":
            reasons.append("missing_romanized")
        if pd.isna(count) or int(count) <= 0:
            reasons.append("invalid_count")
        if not DEVANAGARI.search(native):
            reasons.append("native_not_devanagari")
        if variant_raw is not None and str(variant_raw).strip():
            if not LATIN.search(str(variant_raw)):
                reasons.append("romanized_not_latin")
        if reasons:
            invalid_rows.append(
                {
                    "row_index": int(idx),
                    "native": native,
                    "variant": variant_raw,
                    "attestation_count": count,
                    "reasons": "|".join(reasons),
                }
            )
            continue

        variant = nfc(str(variant_raw).strip())
        norm = case_fold_key(variant)
        rows.append(
            {
                "native": native,
                "variant": variant,
                "normalized_variant": norm,
                "attestation_count": int(count),
            }
        )

    inv = pd.DataFrame(rows)
    if inv.empty:
        raise ValueError("No valid lexicon rows for inventory.")

    grouped = inv.groupby("native", sort=False)
    summary = grouped.agg(
        total_native_attestations=("attestation_count", "sum"),
        variant_count_for_native=("variant", "nunique"),
    )
    canonical_info: dict[str, dict[str, Any]] = {}
    for native, grp in inv.groupby("native"):
        mc = int(grp["attestation_count"].max())
        top = grp[grp["attestation_count"] == mc]["variant"].tolist()
        tie = len(top) > 1
        canonical_info[native] = {
            "canonical_candidate": top[0] if len(top) == 1 else None,
            "canonical_tie": tie,
            "is_canonical": False,
        }

    inv = inv.merge(summary, on="native")
    inv["canonical_candidate"] = inv["native"].map(
        lambda n: canonical_info[n]["canonical_candidate"]
    )
    inv["canonical_tie"] = inv["native"].map(lambda n: canonical_info[n]["canonical_tie"])
    inv["is_canonical"] = inv.apply(
        lambda r: r["variant"] == r["canonical_candidate"]
        if r["canonical_candidate"]
        else False,
        axis=1,
    )
    inv["source_dataset"] = source_dataset
    inv["source_file"] = lexicon_path.name
    inv["source_split"] = "train"
    inv["license_ref"] = license_ref

    ties = inv[inv["canonical_tie"]].drop_duplicates(subset=["native"])
    ties_path = out_dir / "canonical_ties.tsv"
    ties[["native", "canonical_tie", "variant_count_for_native"]].to_csv(
        ties_path, sep="\t", index=False
    )

    invalid_path = out_dir / "invalid_lexicon_rows.tsv"
    pd.DataFrame(invalid_rows).to_csv(invalid_path, sep="\t", index=False)

    inv_path = out_dir / "dakshina_hi_variant_inventory.tsv"
    inv.to_csv(inv_path, sep="\t", index=False)

    alias_to_natives: dict[str, set[str]] = {}
    for _, r in inv.iterrows():
        alias_to_natives.setdefault(r["normalized_variant"], set()).add(r["native"])

    alias_rows = []
    ambiguous = []
    english_collisions = []
    for alias, natives in sorted(alias_to_natives.items()):
        if len(natives) > 1:
            ambiguous.append(
                {"normalized_variant": alias, "native_words": "|".join(sorted(natives))}
            )
            continue
        native = next(iter(natives))
        if alias in english:
            english_collisions.append(
                {"normalized_variant": alias, "native_word": native}
            )
            continue
        surface = inv[
            (inv["native"] == native) & (inv["normalized_variant"] == alias)
        ]["variant"].iloc[0]
        alias_rows.append(
            {
                "normalized_variant": alias,
                "surface_variant": surface,
                "native": native,
            }
        )

    alias_df = pd.DataFrame(alias_rows)
    alias_path = out_dir / "dakshina_hi_alias_index.tsv"
    alias_df.to_csv(alias_path, sep="\t", index=False)
    pd.DataFrame(ambiguous).to_csv(out_dir / "ambiguous_aliases.tsv", sep="\t", index=False)
    pd.DataFrame(english_collisions).to_csv(
        out_dir / "english_collision_aliases.tsv", sep="\t", index=False
    )

    return {
        "native_words": int(inv["native"].nunique()),
        "variants": int(len(inv)),
        "usable_aliases": int(len(alias_df)),
        "ambiguous_aliases": int(len(ambiguous)),
        "english_collisions": int(len(english_collisions)),
        "canonical_ties": int(ties["native"].nunique()) if len(ties) else 0,
        "invalid_rows": int(len(invalid_rows)),
        "paths": {
            "inventory": str(inv_path.relative_to(PROJECT_ROOT)),
            "alias_index": str(alias_path.relative_to(PROJECT_ROOT)),
        },
        "alias_df": alias_df,
        "inv_df": inv,
    }
