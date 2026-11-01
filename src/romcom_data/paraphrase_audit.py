"""Phase 2.1 IndicXParaphrase source audit (inspection only, no rebuild)."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from romcom_data.handoff_paths import HANDOFF, INVENTORIES, PARAPHRASE_AUDIT, RAW_PARA
from romcom_data.paraphrase_gate import validate_reconstructed_pair
from romcom_data.romanize import DakshinaRomanizer
from romcom_data.splitting import _norm, assign_splits, paraphrase_groups
from romcom_data.stable_hash import derive_rng, stable_hash_hex
from romcom_data.text_utils import DEVANAGARI, LATIN, nfc, tokenize_conservative

REPO = "ai4bharat/IndicXParaphrase"
HF_FILE = "data/hi.tsv"
ZIP_NAME = "RomCoM_X_Phase_2_1_Paraphrase_Audit.zip"
OWNER_LICENSE_APPROVAL = {
    "approved": True,
    "approved_on": "2026-11-01",
    "approved_by": "owner",
    "scope": "phase_2_1_paraphrase_audit",
    "decision": (
        "Owner approved the IndicXParaphrase license for this audit and "
        "removed the confirmation blocker. The HF card license field is still null."
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_or_pin_provenance(refresh_hf: bool = False) -> dict[str, Any]:
    cached = PARAPHRASE_AUDIT / "dataset_manifest.json"
    if cached.is_file() and not refresh_hf:
        return json.loads(cached.read_text(encoding="utf-8"))
    return pin_provenance()


def pin_provenance() -> dict[str, Any]:
    api = HfApi()
    info = api.dataset_info(REPO, revision="main")
    card_license = info.card_data.get("license") if info.card_data else None
    card_license = card_license or getattr(info, "license", None)
    readme_excerpt = ""
    try:
        readme_path = hf_hub_download(
            repo_id=REPO, filename="README.md", repo_type="dataset", revision=info.sha
        )
        readme_excerpt = Path(readme_path).read_text(encoding="utf-8")[:8000]
    except Exception as exc:
        readme_excerpt = f"(README fetch failed: {exc})"

    dest = RAW_PARA / "hi.tsv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=REPO, filename=HF_FILE, repo_type="dataset", revision=info.sha
    )
    dest.write_bytes(Path(local).read_bytes())

    field_defs_from_card = _extract_field_definitions(readme_excerpt)
    license_evidence = gather_license_evidence(info, readme_excerpt)
    return {
        "repository": REPO,
        "revision": info.sha,
        "revision_fetched_utc": datetime.now(UTC).isoformat(),
        "hf_file": HF_FILE,
        "local_path": str(dest.relative_to(HANDOFF.parent)),
        "local_sha256": sha256_file(dest),
        "local_size_bytes": dest.stat().st_size,
        "card_license": card_license,
        "card_license_note": "Taken from HF dataset card; not hardcoded in loader.",
        "official_splits": "none_single_tsv",
        "files_on_card": [],
        "readme_excerpt_chars": len(readme_excerpt),
        "field_definitions_from_card": field_defs_from_card,
        "columns_in_tsv": ["en", "sentence1", "sentence2", "label"],
        "license_evidence": license_evidence,
    }


def gather_license_evidence(info: Any, readme: str) -> dict[str, Any]:
    card = info.card_data or {}
    tags = list(getattr(info, "tags", []) or [])
    license_mentions = re.findall(
        r"(?i)(cc0|cc-by|creative commons|mit license|apache|license)",
        readme,
    )
    return {
        "hf_card_data_keys": sorted(card.keys()) if isinstance(card, dict) else [],
        "hf_card_license_field": card.get("license") if isinstance(card, dict) else None,
        "hf_dataset_info_license_attr": getattr(info, "license", None),
        "hf_tags": tags,
        "readme_license_mentions": sorted(set(license_mentions)),
        "loader_hardcoded_cc0_warning": (
            "source_loaders/indicxparaphrase.py hardcodes CC0; not used as audit evidence."
        ),
        "audit_conclusion": (
            "License not declared on HF card metadata; owner must confirm terms before release."
            if not card.get("license") and not getattr(info, "license", None)
            else "See hf_card_license_field."
        ),
    }


def _extract_field_definitions(readme: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for col in ("en", "sentence1", "sentence2", "label"):
        pat = re.compile(rf"`?{col}`?[:\s]+([^\n]+)", re.I)
        m = pat.search(readme)
        if m:
            out[col] = m.group(1).strip()[:500]
    if not out:
        out["note"] = "Card/README did not yield explicit per-column definitions; use measured alignment."
    return out


def load_tsv() -> pd.DataFrame:
    path = RAW_PARA / "hi.tsv"
    df = pd.read_csv(path, sep="\t")
    for col in ("en", "sentence1", "sentence2"):
        df[col] = df[col].map(lambda x: nfc(str(x)) if pd.notna(x) else "")
    return df


def audit_integrity(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    labels = df["label"].value_counts(dropna=False).to_dict()
    missing = {
        "en_blank": int((df["en"].str.strip() == "").sum()),
        "sentence1_blank": int((df["sentence1"].str.strip() == "").sum()),
        "sentence2_blank": int((df["sentence2"].str.strip() == "").sum()),
        "label_missing": int(df["label"].isna().sum()),
    }
    non_binary = int((~df["label"].isin([0, 1])).sum())
    df["_pair_key"] = df["sentence1"] + "\0" + df["sentence2"]
    dup_pairs = int(df.duplicated(subset=["sentence1", "sentence2"], keep=False).sum())
    dup_s1 = int(df.duplicated(subset=["sentence1"], keep=False).sum())
    dup_s2 = int(df.duplicated(subset=["sentence2"], keep=False).sum())
    identical_sides = int((df["sentence1"] == df["sentence2"]).sum())
    identical_by_label = {
        str(lab): int((grp["sentence1"] == grp["sentence2"]).sum())
        for lab, grp in df.groupby("label")
    }

    s1_both_labels = 0
    s1_pos_neg_pattern = 0
    for _, grp in df.groupby("sentence1"):
        labs = set(grp["label"].tolist())
        if len(grp) > 1:
            s1_both_labels += 1
        if 0 in labs and 1 in labs:
            s1_pos_neg_pattern += 1

    en_devanagari_rows = int(df["en"].apply(lambda t: bool(DEVANAGARI.search(t))).sum())
    en_latin_rows = int(df["en"].apply(lambda t: bool(LATIN.search(t)) and not DEVANAGARI.search(t)).sum())

    return {
        "row_count": n,
        "label_distribution": {str(k): int(v) for k, v in labels.items()},
        "missing_fields": missing,
        "non_binary_labels": non_binary,
        "duplicate_pair_rows": dup_pairs,
        "duplicate_sentence1_rows": dup_s1,
        "duplicate_sentence2_rows": dup_s2,
        "identical_sentence1_sentence2_rows": identical_sides,
        "identical_sides_by_label": {str(k): v for k, v in identical_by_label.items()},
        "sentence1_shared_across_multiple_rows": s1_both_labels,
        "sentence1_with_both_positive_and_negative": s1_pos_neg_pattern,
        "en_contains_devanagari_rows": en_devanagari_rows,
        "en_latin_only_rows": en_latin_rows,
        "official_train_val_test": "not_provided_single_file",
        "cross_split_overlap": "not_applicable_no_official_splits",
    }


def audit_english_pivot(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    nunique = {
        "en": int(df["en"].nunique()),
        "sentence1": int(df["sentence1"].nunique()),
        "sentence2": int(df["sentence2"].nunique()),
        "pair": int(df.drop_duplicates(subset=["sentence1", "sentence2"]).shape[0]),
    }
    en_per_s1 = df.groupby("sentence1")["en"].nunique()
    en_constant_given_s1 = int((en_per_s1 == 1).sum())
    en_multi_given_s1 = int((en_per_s1 > 1).sum())
    s1_groups = df.groupby("sentence1")
    same_en_across_s2_and_labels = 0
    for s1, grp in s1_groups:
        if grp["en"].nunique() == 1 and len(grp) > 1:
            same_en_across_s2_and_labels += 1

    en_to_s1 = df.groupby("en")["sentence1"].nunique()
    en_maps_one_s1 = int((en_to_s1 == 1).sum())
    en_maps_many_s1 = int((en_to_s1 > 1).sum())

    second_en_column_present = False
    en_equals_s1 = int((df["en"] == df["sentence1"]).sum())
    en_equals_s2 = int((df["en"] == df["sentence2"]).sum())

    independent_s2_en_rows = 0
    for _, row in df.iterrows():
        s1e = str(row.get("sentence1_en", row["en"])).strip()
        s2e = str(row.get("sentence2_en", "")).strip()
        if s1e and s2e and s1e != s2e:
            independent_s2_en_rows += 1

    reject_english_pivot = independent_s2_en_rows == 0
    en_tracks_sentence1_only = en_constant_given_s1 == len(en_per_s1) and en_equals_s2 == 0

    return {
        "unique_counts": nunique,
        "en_constant_for_each_sentence1_groups": en_constant_given_s1,
        "en_varies_within_sentence1_groups": en_multi_given_s1,
        "sentence1_groups_sharing_one_en_across_sentence2_and_labels": same_en_across_s2_and_labels,
        "en_values_mapping_to_single_sentence1": en_maps_one_s1,
        "en_values_mapping_to_multiple_sentence1": en_maps_many_s1,
        "second_english_column_in_source": second_en_column_present,
        "rows_en_equals_sentence1": en_equals_s1,
        "rows_en_equals_sentence2": en_equals_s2,
        "rows_with_independent_sentence2_english": independent_s2_en_rows,
        "measured_en_alignment": (
            "single en column; constant per sentence1 in all groups; not aligned to sentence2"
            if en_tracks_sentence1_only
            else "see counts; do not assume from column name"
        ),
        "reject_english_pivot_reconstruction_for_code_mixed": reject_english_pivot,
        "never_treat_devanagari_as_english": True,
    }


def _romanizer() -> DakshinaRomanizer:
    inv = INVENTORIES / "dakshina_hi_variant_inventory.tsv"
    if not inv.is_file():
        inv = HANDOFF / "inventories" / "dakshina_hi_variant_inventory.tsv"
    return DakshinaRomanizer(inventory_path=inv if inv.is_file() else None)


def romanize_sentence(text: str, romanizer: DakshinaRomanizer) -> tuple[str, dict[str, int]]:
    stats: Counter[str] = Counter()
    parts: list[str] = []
    for tok in tokenize_conservative(text):
        if tok.is_latin_word or not DEVANAGARI.search(tok.text):
            parts.append(tok.text)
            if tok.is_latin_word:
                stats["latin_passthrough_tokens"] += 1
            else:
                stats["non_devanagari_passthrough_tokens"] += 1
            continue
        hit = romanizer.romanize_native(tok.text)
        parts.append(hit.surface)
        stats[f"method_{hit.method}"] += 1
        if hit.method == "dakshina_canonical":
            stats["dakshina_hits"] += 1
        elif hit.method == "fallback_indic_map":
            stats["fallback_tokens"] += 1
    return "".join(parts), dict(stats)


def audit_romanization(df: pd.DataFrame) -> dict[str, Any]:
    romanizer = _romanizer()
    pos_collapse = 0
    neg_become_identical = 0
    pos_total = int((df["label"] == 1).sum())
    neg_total = int((df["label"] == 0).sum())
    token_stats: Counter[str] = Counter()
    side_coverage: list[dict[str, int]] = []

    for _, row in df.iterrows():
        s1, s2 = row["sentence1"], row["sentence2"]
        r1, st1 = romanize_sentence(s1, romanizer)
        r2, st2 = romanize_sentence(s2, romanizer)
        for k, v in {**st1, **st2}.items():
            token_stats[k] += v
        side_coverage.append(
            {
                "s1_dakshina": st1.get("dakshina_hits", 0),
                "s2_dakshina": st2.get("dakshina_hits", 0),
                "s1_fallback": st1.get("fallback_tokens", 0),
                "s2_fallback": st2.get("fallback_tokens", 0),
            }
        )
        label_str = "paraphrase" if row["label"] == 1 else "non_paraphrase"
        ok, reason = validate_reconstructed_pair(r1, r2, label_str, s1, s2)
        if row["label"] == 1 and not ok and reason == "transformation_induced_positive_collapse":
            pos_collapse += 1
        if row["label"] == 0 and r1 == r2 and s1 != s2:
            neg_become_identical += 1

    rows_no_dakshina_either_side = sum(
        1 for sc in side_coverage if sc["s1_dakshina"] == 0 and sc["s2_dakshina"] == 0
    )
    return {
        "romanization_inventory": str(
            (INVENTORIES / "dakshina_hi_variant_inventory.tsv").relative_to(HANDOFF.parent)
            if (INVENTORIES / "dakshina_hi_variant_inventory.tsv").is_file()
            else "fallback_sample_lexicon"
        ),
        "aggregate_token_methods": dict(token_stats),
        "positive_pairs": pos_total,
        "negative_pairs": neg_total,
        "positive_transformation_induced_collapse": pos_collapse,
        "positive_collapse_rate": pos_collapse / pos_total if pos_total else 0,
        "negative_romanized_identical_while_native_differ": neg_become_identical,
        "negative_false_identical_rate": neg_become_identical / neg_total if neg_total else 0,
        "rows_with_zero_dakshina_hits_both_sides": rows_no_dakshina_either_side,
        "rows_silently_discarded": 0,
    }


def df_to_records(df: pd.DataFrame, revision: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        s1 = row["sentence1"]
        s2 = row["sentence2"]
        lab = int(row["label"])
        sid = "para-" + stable_hash_hex(revision, str(idx), s1, s2)[:16]
        records.append(
            {
                "source_id": sid,
                "example_id": f"{sid}:{idx}",
                "sentence1_native": s1,
                "sentence2_native": s2,
                "normalized_label": "paraphrase" if lab == 1 else "non_paraphrase",
                "original_label": lab,
                "original_split": "unspecified",
                "source_revision": revision,
            }
        )
    return records


def find_identical_positive_quarantine(df: pd.DataFrame, revision: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        if int(row["label"]) != 1:
            continue
        if row["sentence1"] != row["sentence2"]:
            continue
        sid = "para-" + stable_hash_hex(revision, str(idx), row["sentence1"], row["sentence2"])[:16]
        out.append(
            {
                "row_index": int(idx),
                "source_id": sid,
                "example_id": f"{sid}:{idx}",
                "label": 1,
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "exclusion_reason": "identical_positive_sentence1_equals_sentence2",
            }
        )
    return out


def _sentence_split_map(rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = row.get("project_split", "unassigned")
        for key in ("sentence1_native", "sentence2_native"):
            mapping[_norm(row[key])].add(split)
    return mapping


def audit_leakage_connected_components(
    df: pd.DataFrame, revision: str, quarantine_ids: set[str]
) -> dict[str, Any]:
    records = df_to_records(df, revision)
    eligible = [r for r in records if r["source_id"] not in quarantine_ids]
    roots = paraphrase_groups(eligible)
    components: dict[str, set[str]] = defaultdict(set)
    for sent, root in roots.items():
        components[root].add(sent)

    sim_rows = [dict(r) for r in eligible]
    assign_splits(sim_rows, "paraphrase", seed=42)
    cc_map = _sentence_split_map(sim_rows)
    cc_leaks = [
        sent for sent, splits in cc_map.items() if "train" in splits and "validation" in splits
    ]

    s1_only_rows: list[dict[str, Any]] = []
    for r in eligible:
        copy_r = dict(r)
        copy_r["group_id"] = _norm(r["sentence1_native"])
        s1_only_rows.append(copy_r)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in s1_only_rows:
        groups[r["group_id"]].append(r)
    items = list(groups.items())
    rng = derive_rng(42, "paraphrase", "split-s1only-audit")
    rng.shuffle(items)
    train_keys: set[str] = set()
    for i, (key, _) in enumerate(items):
        if i < max(1, int(round(len(items) * 0.8))):
            train_keys.add(key)
    for key, members in groups.items():
        split = "train" if key in train_keys else "validation"
        for m in members:
            m["project_split"] = split
    s1_map = _sentence_split_map([m for ms in groups.values() for m in ms])
    s1_leaks = [
        sent for sent, splits in s1_map.items() if "train" in splits and "validation" in splits
    ]

    cross_column_risk = 0
    for r in sim_rows:
        s2n = _norm(r["sentence2_native"])
        for other in sim_rows:
            if other["source_id"] == r["source_id"]:
                continue
            if s2n == _norm(other["sentence1_native"]) and r["project_split"] != other["project_split"]:
                cross_column_risk += 1

    return {
        "connected_component_count": len(components),
        "largest_component_sentence_count": max((len(v) for v in components.values()), default=0),
        "split_unit_required": "connected_component_on_sentence1_and_sentence2_union",
        "sentence1_only_grouping_insufficient": len(s1_leaks) > 0 or cross_column_risk > 0,
        "simulated_project_split_seed": 42,
        "simulated_rows": len(sim_rows),
        "sentence_leak_train_and_validation_cc_split": len(cc_leaks),
        "leaked_sentences_cc_split_sample": cc_leaks[:10],
        "sentence_leak_train_and_validation_sentence1_only_split": len(s1_leaks),
        "leaked_sentences_s1_only_sample": s1_leaks[:10],
        "cross_column_split_mismatch_pairs_observed": cross_column_risk,
        "required_pipeline_grouping": (
            "Use romcom_data.splitting.paraphrase_groups (union-find on both sentence columns)."
        ),
    }


def audit_leakage_clusters(df: pd.DataFrame) -> dict[str, Any]:
    s1_clusters = df.groupby("sentence1").size()
    s2_clusters = df.groupby("sentence2").size()
    return {
        "sentence1_cluster_count": int(s1_clusters.shape[0]),
        "sentence2_cluster_count": int(s2_clusters.shape[0]),
        "max_rows_per_sentence1": int(s1_clusters.max()),
        "max_rows_per_sentence2": int(s2_clusters.max()),
        "sentence1_clusters_with_multiple_rows": int((s1_clusters > 1).sum()),
    }


def romanization_integrity_documentation() -> str:
    return """# Romanization integrity (audit)

## Method (both sentences treated independently)

1. **Tokenization:** `tokenize_conservative` per sentence (Latin words passthrough; Devanagari tokens romanized).
2. **Dakshina hit:** Lookup native token in `dakshina_hi_variant_inventory.tsv`; canonical variant = highest attestation with tie-break policy from `pick_canonical_variant`.
3. **OOV / no lexicon hit:** Devanagari tokens use `fallback_indic_map` (documented character map in `romanize.py`, not attested Dakshina).
4. **Non-Devanagari:** Punctuation and non-Latin non-Devanagari pass through unchanged.
5. **Labels:** Romanization is applied to input text only; `label` / `normalized_label` are never modified.

## Collapse checks (post-normalization NFC on native; romanized output concatenated)

- **Positive collapse:** `validate_reconstructed_pair` flags positives where romanized s1 == romanized s2 but native s1 != native s2.
- **Negative false-identical:** negatives where romanized forms match while native differ.

Counts are in `integrity_metrics.json` under `romanization`. No rows are silently dropped during audit.
"""


def audit_experimental_consistency_code() -> dict[str, Any]:
    return {
        "source_of_truth": "src/romcom_data/generate_exps.py::generate_one",
        "exp01": "Canonical Dakshina spellings per eligible token on each input field independently.",
        "exp02": "Same base record; `realistic_from_canonical` with per-field RNG `derive_rng(base_seed, example_id, EXP-02)`.",
        "exp03": (
            "Copies EXP-01 base; applies `noise_from_changes` using EXP-02 `changed_all` token spans "
            "so matched-noise uses same positions and target edit distances as realistic-variant."
        ),
        "shared_metadata": (
            "phase2_pipeline.stream_task writes same example_id, source_id, label, project_split "
            "for EXP-01/02/03; qa_matching_ids() verifies alignment when processed files exist."
        ),
        "paraphrase_fields": ["input.sentence1", "input.sentence2"],
        "rebuild_performed_in_this_audit": False,
        "when_paraphrase_enabled": (
            "All three conditions will share identical source IDs, labels, and splits per example_id."
        ),
    }


def determine_handoff_status(
    license_evidence: dict[str, Any],
    recommendation: dict[str, Any],
    leakage: dict[str, Any],
    quarantine: list[dict[str, Any]],
) -> dict[str, Any]:
    owner_approval = license_evidence.get("owner_approval") or {}
    license_clear = bool(
        license_evidence.get("hf_card_license_field")
        or license_evidence.get("hf_dataset_info_license_attr")
        or owner_approval.get("approved")
    )
    blockers: list[str] = []
    if not license_clear:
        blockers.append("license_not_declared_on_hf_card_owner_must_confirm")
    if leakage.get("sentence_leak_train_and_validation_cc_split", 0) > 0:
        blockers.append("simulated_cc_split_leaked_sentences")
    if recommendation.get("english_pivot_code_mixed_verdict") == "NO_GO":
        blockers.append("english_pivot_code_mixed_path_rejected")

    status = (
        "BLOCKED"
        if leakage.get("sentence_leak_train_and_validation_cc_split", 0) > 0
        else "READY_FOR_OWNER_REVIEW"
    )

    return {
        "status": status,
        "owner_review_scope": "hindi_native_only_paraphrase_path",
        "blockers": blockers,
        "quarantined_row_count": len(quarantine),
        "license_owner_approved": bool(owner_approval.get("approved")),
        "dataset_rebuild_allowed": False,
        "training_allowed": False,
    }


def build_recommendation(
    pivot: dict[str, Any], integrity: dict[str, Any], roman: dict[str, Any]
) -> dict[str, Any]:
    if pivot["rows_with_independent_sentence2_english"] > 0:
        verdict = "GO"
        scope = "english_pivot_code_mixed_experiment"
        reason = "Independent English for both sides present in source."
    elif pivot["reject_english_pivot_reconstruction_for_code_mixed"]:
        verdict = "NO_GO"
        scope = "english_pivot_independent_code_mixed_reconstruction"
        reason = (
            "Source has a single en column aligned with sentence1 only; "
            "cannot reconstruct independent English for sentence2."
        )
        if integrity["row_count"] > 0 and roman["positive_transformation_induced_collapse"] == 0:
            verdict = "CONDITIONAL_GO"
            scope = "hindi_native_only_paraphrase_without_english_pivot"
            reason = (
                "NO-GO for English-pivot code-mixed path; CONDITIONAL for Hindi-native-only "
                "experiment with documented cluster leakage risk and no RomCoM EXP rebuild."
            )
    else:
        verdict = "NO_GO"
        scope = "unknown"
        reason = "English pivot semantics could not be established safely."

    return {
        "verdict": verdict,
        "scope": scope,
        "blocking_reason": reason,
        "english_pivot_gate_passed": pivot["rows_with_independent_sentence2_english"] > 0,
        "english_pivot_code_mixed_verdict": "NO_GO"
        if pivot["reject_english_pivot_reconstruction_for_code_mixed"]
        else "GO",
        "train_or_processed_rebuild": False,
    }


def write_owner_sample(df: pd.DataFrame, path: Path, n_each: int = 25) -> int:
    rng = derive_rng(42, "paraphrase-audit", "owner-sample")
    pos_idx = df.index[df["label"] == 1].tolist()
    neg_idx = df.index[df["label"] == 0].tolist()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pick_idx = pos_idx[:n_each] + neg_idx[:n_each]
    fields = [
        "row_index",
        "label",
        "en",
        "sentence1",
        "sentence2",
        "shared_sentence1_with_opposite_label_in_dataset",
        "owner_judgment_pair_valid",
        "owner_judgment_en_aligns_to",
        "owner_notes",
    ]
    s1_label_sets: dict[str, set[int]] = defaultdict(set)
    for _, row in df.iterrows():
        s1_label_sets[row["sentence1"]].add(int(row["label"]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for idx in pick_idx:
            row = df.loc[idx]
            s1 = row["sentence1"]
            w.writerow(
                {
                    "row_index": int(idx),
                    "label": int(row["label"]),
                    "en": row["en"],
                    "sentence1": s1,
                    "sentence2": row["sentence2"],
                    "shared_sentence1_with_opposite_label_in_dataset": (
                        "yes"
                        if len(s1_label_sets[s1]) > 1
                        else "no"
                    ),
                    "owner_judgment_pair_valid": "",
                    "owner_judgment_en_aligns_to": "",
                    "owner_notes": "",
                }
            )
    return len(pick_idx)


def run_paraphrase_audit(*, refresh_hf: bool = False) -> dict[str, Any]:
    PARAPHRASE_AUDIT.mkdir(parents=True, exist_ok=True)
    provenance = load_or_pin_provenance(refresh_hf=refresh_hf)
    revision = provenance["revision"]
    if not provenance.get("license_evidence"):
        try:
            info = HfApi().dataset_info(REPO, revision=revision)
            readme_path = hf_hub_download(
                repo_id=REPO, filename="README.md", repo_type="dataset", revision=revision
            )
            readme = Path(readme_path).read_text(encoding="utf-8")[:8000]
            provenance["license_evidence"] = gather_license_evidence(info, readme)
        except Exception as exc:
            provenance["license_evidence"] = {"fetch_error": str(exc)}
    df = load_tsv()
    df = df.reset_index(drop=True)
    integrity = audit_integrity(df)
    pivot = audit_english_pivot(df)
    roman = audit_romanization(df)
    leakage_basic = audit_leakage_clusters(df)
    quarantine = find_identical_positive_quarantine(df, revision)
    quarantine_ids = {q["source_id"] for q in quarantine}
    leakage_cc = audit_leakage_connected_components(df, revision, quarantine_ids)
    exp_consistency = audit_experimental_consistency_code()
    recommendation = build_recommendation(pivot, integrity, roman)
    license_evidence = dict(provenance.get("license_evidence") or {})
    license_evidence["owner_approval"] = OWNER_LICENSE_APPROVAL
    if OWNER_LICENSE_APPROVAL["approved"]:
        license_evidence["audit_conclusion"] = (
            "HF card license field remains null. Owner approved the license "
            "for the Phase 2.1 paraphrase audit on 2026-11-01; confirmation blocker removed."
        )
    provenance["license_evidence"] = license_evidence
    handoff_status = determine_handoff_status(
        license_evidence, recommendation, leakage_cc, quarantine
    )

    metrics = {
        "integrity": integrity,
        "english_pivot": pivot,
        "romanization": roman,
        "leakage_clusters_basic": leakage_basic,
        "leakage_connected_components": leakage_cc,
        "quarantine": quarantine,
        "experimental_consistency": exp_consistency,
        "handoff_status": handoff_status,
        "audit_utc": datetime.now(UTC).isoformat(),
        "audit_revision": "2026_11_01_owner_review",
    }

    manifest = {**provenance, "audit_scope": "inspection_only_no_rebuild"}

    report = {
        "title": "IndicXParaphrase Phase 2.1 audit (revised)",
        "provenance": provenance,
        "known_generation_procedure": (
            "Single TSV from ai4bharat/IndicXParaphrase; columns en, sentence1, sentence2, label."
        ),
        "integrity_summary": integrity,
        "english_pivot_gate": pivot,
        "romanization_summary": roman,
        "leakage_summary": {**leakage_basic, **leakage_cc},
        "quarantine_required": quarantine,
        "recommendation": recommendation,
        "experimental_consistency": exp_consistency,
        "handoff_status": handoff_status,
        "limitations": [
            "No official train/validation/test; split on connected components (both sentence columns).",
            "English-pivot code-mixed reconstruction rejected (NO_GO).",
            "One identical positive row must stay quarantined.",
            "Romanization uses Dakshina inventory; fallback map tokens are not attested variants.",
            "HF card license field remains null. Owner approved the license on 2026-11-01.",
        ],
    }

    (PARAPHRASE_AUDIT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (PARAPHRASE_AUDIT / "integrity_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (PARAPHRASE_AUDIT / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
    )
    md = "# IndicXParaphrase audit report\n\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (PARAPHRASE_AUDIT / "paraphrase_audit_report.md").write_text(md, encoding="utf-8")

    sample_n = write_owner_sample(df, PARAPHRASE_AUDIT / "owner_review_sample.csv")

    (PARAPHRASE_AUDIT / "quarantined_rows.csv").write_text(
        _dicts_to_csv(quarantine,
        ["row_index", "source_id", "example_id", "label", "sentence1", "sentence2", "exclusion_reason"]),
        encoding="utf-8",
    )
    (PARAPHRASE_AUDIT / "license_evidence.json").write_text(
        json.dumps(license_evidence, indent=2) + "\n", encoding="utf-8"
    )
    (PARAPHRASE_AUDIT / "leakage_report.md").write_text(
        "# Paraphrase leakage report\n\n"
        + json.dumps(leakage_cc, indent=2, ensure_ascii=False)
        + "\n\n"
        + "Sentence1-only grouping is insufficient when the same sentence text can appear in "
        "sentence1 or sentence2 across rows. Use `paraphrase_groups` connected components.\n",
        encoding="utf-8",
    )
    (PARAPHRASE_AUDIT / "romanization_integrity.md").write_text(
        romanization_integrity_documentation(), encoding="utf-8"
    )
    (PARAPHRASE_AUDIT / "paraphrase_audit_status.json").write_text(
        json.dumps(handoff_status, indent=2) + "\n", encoding="utf-8"
    )

    (PARAPHRASE_AUDIT / "audit_log.txt").write_text(
        f"completed_utc={datetime.now(UTC).isoformat()}\n"
        f"rows={integrity['row_count']}\n"
        f"recommendation={recommendation['verdict']}\n"
        f"handoff_status={handoff_status['status']}\n"
        f"quarantined={len(quarantine)}\n"
        f"owner_sample_rows={sample_n}\n",
        encoding="utf-8",
    )

    return {
        "report": report,
        "manifest": manifest,
        "metrics": metrics,
        "recommendation": recommendation,
        "handoff_status": handoff_status,
    }


def _dicts_to_csv(rows: list[dict[str, Any]], fields: list[str]) -> str:
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def build_audit_zip(zip_path: Path | None = None) -> Path:
    out = zip_path or (HANDOFF.parent / ZIP_NAME)
    files = (
        "paraphrase_audit_report.md",
        "dataset_manifest.json",
        "integrity_metrics.json",
        "owner_review_sample.csv",
        "recommendation.json",
        "leakage_report.md",
        "romanization_integrity.md",
        "license_evidence.json",
        "quarantined_rows.csv",
        "paraphrase_audit_status.json",
        "audit_log.txt",
    )
    lines: list[str] = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in files:
            src = PARAPHRASE_AUDIT / name
            data = src.read_bytes()
            zf.writestr(name, data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
        zf.writestr("checksums.sha256", "\n".join(lines) + "\n")
    return out
