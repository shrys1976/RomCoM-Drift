"""Validate NLI experiment artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from romcom_data.augmentation import levenshtein_distance
from romcom_data.checksums import sha256_file
from romcom_data.nli_exp_freeze import load_exclusion_keys, load_approved_pairs
from romcom_data.nli_exp_build import collect_row_counts_from_disk
from romcom_data.nli_exp_paths import (
    CONDITIONS,
    ENGLISH_RETENTION,
    EXPECTED_COUNTS,
    EXP00,
    NLI_EXPERIMENTS,
    NLI_EXPERIMENTS_V1,
    exp_condition_root,
)
from romcom_data.nli_exp_romanize import NliLexicon, tokenize_and_romanize_field
from romcom_data.nli_exp_variant_exclusions import excluded_variant_keys
from romcom_data.nli_exp_variants import (
    VariantPools,
    _noise_at_distance,
    apply_exp02,
    apply_exp03,
)
from romcom_data.paths import PROJECT_ROOT
from romcom_data.stable_hash import derive_rng
from romcom_data.text_utils import DEVANAGARI, nfc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_experiments(root: Path | None = None) -> dict[str, Any]:
    root = root or NLI_EXPERIMENTS
    issues: list[str] = []
    approved = {r["example_id"]: r for r in load_approved_pairs()}
    excluded = load_exclusion_keys()

    def check_file(path: Path, split: str, cond: str, scope: str) -> list[dict[str, Any]]:
        if not path.is_file():
            issues.append(f"missing {path}")
            return []
        rows = _load_jsonl(path)
        if len(rows) != EXPECTED_COUNTS[split]:
            issues.append(f"{path} rows {len(rows)} != {EXPECTED_COUNTS[split]}")
        ids = [r["example_id"] for r in rows]
        if len(set(ids)) != len(ids):
            issues.append(f"duplicate ids in {path}")
        for eid in ids:
            if eid not in approved:
                issues.append(f"unknown id {eid} in {path}")
            base = approved.get(eid)
            if base and base["split"] != split:
                issues.append(f"split mismatch {eid} in {path}")
            if base:
                hi = base["hindi_row_index"]
                if hi in excluded.get(split, set()):
                    issues.append(f"excluded row {split}:{hi} in {path}")
        return rows

    by_split_ids: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "dev", "test"):
        for cond in CONDITIONS:
            for scope in ("both", "premise_only", "hypothesis_only"):
                if split == "train" and scope != "both":
                    continue
                path = exp_condition_root(cond) / scope / f"{split}.jsonl"
                rows = check_file(path, split, cond, scope)
                by_split_ids[split].update(r["example_id"] for r in rows)
        if split in ("dev", "test"):
            check_file(EXP00 / f"{split}.jsonl", split, "EXP-00", "both")
            eng = _load_jsonl(ENGLISH_RETENTION / f"{split}.jsonl")
            if len(eng) != EXPECTED_COUNTS[split]:
                issues.append(
                    f"english_retention/{split}.jsonl rows {len(eng)} != {EXPECTED_COUNTS[split]}"
                )
            for row in eng:
                base = approved[row["example_id"]]
                if row["premise"] != base["english_premise"]:
                    issues.append(f"english premise changed {row['example_id']}")
                if row["hypothesis"] != base["english_hypothesis"]:
                    issues.append(f"english hypothesis changed {row['example_id']}")

    for split, ids in by_split_ids.items():
        expected = {eid for eid, r in approved.items() if r["split"] == split}
        if ids != expected:
            issues.append(f"{split} id set mismatch: missing {len(expected - ids)} extra {len(ids - expected)}")

    # Scope consistency on dev/test
    for split in ("dev", "test"):
        for cond in CONDITIONS:
            both = {r["example_id"]: r for r in _load_jsonl(exp_condition_root(cond) / "both" / f"{split}.jsonl")}
            po = {r["example_id"]: r for r in _load_jsonl(exp_condition_root(cond) / "premise_only" / f"{split}.jsonl")}
            ho = {r["example_id"]: r for r in _load_jsonl(exp_condition_root(cond) / "hypothesis_only" / f"{split}.jsonl")}
            for eid, row in both.items():
                base = approved[eid]
                if po[eid]["premise"] != row["premise"]:
                    issues.append(f"premise_only premise mismatch {eid} {cond}")
                if po[eid]["hypothesis"] != base["hindi_hypothesis"]:
                    issues.append(f"premise_only hypothesis not native {eid} {cond}")
                if ho[eid]["hypothesis"] != row["hypothesis"]:
                    issues.append(f"hypothesis_only hypothesis mismatch {eid} {cond}")
                if ho[eid]["premise"] != base["hindi_premise"]:
                    issues.append(f"hypothesis_only premise not native {eid} {cond}")

    lexicon = NliLexicon()
    pools = VariantPools(lexicon)
    banned = excluded_variant_keys()
    exp02_rows: dict[str, dict[str, dict[str, Any]]] = {}
    exp03_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        exp02_rows[split] = {
            r["example_id"]: r
            for r in _load_jsonl(exp_condition_root("EXP-02") / "both" / f"{split}.jsonl")
        }
        exp03_rows[split] = {
            r["example_id"]: r
            for r in _load_jsonl(exp_condition_root("EXP-03") / "both" / f"{split}.jsonl")
        }

    selection_stats: dict[str, dict[str, int | float]] = {}
    field_mix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    exp02_exp03_checks = 0
    exp02_exp03_ok = 0
    exp03_collision_pairs = 0
    exp03_identical_pos_pairs = 0
    exp03_pos_pair_total = 0
    excluded_hits = 0

    def _eligible_count(toks: list[dict[str, Any]]) -> int:
        n = 0
        for t in toks:
            if t.get("method") != "dakshina_canonical" or not t.get("canonical"):
                continue
            key = nfc(str(t.get("native_text", "")).rstrip("।॥").strip())
            if pools.pool(key):
                n += 1
        return n

    for split in ("train", "dev", "test"):
        log_path = root / "logs" / f"transformation_log.{split}.jsonl"
        elig_tok = sel_tok = 0
        if not log_path.is_file():
            issues.append(f"missing log {log_path}")
            continue
        with log_path.open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                eid = rec["example_id"]
                t2 = [t for t in rec.get("exp02_traces", []) if t.get("operation") == "attested_variant"]
                t3 = [t for t in rec.get("exp03_traces", []) if t.get("operation") == "random_noise"]
                if len(t2) != len(t3):
                    issues.append(f"EXP02/03 count mismatch {eid}")
                    continue
                row2 = exp02_rows[split].get(eid)
                row3 = exp03_rows[split].get(eid)
                for field in ("premise", "hypothesis"):
                    toks = rec.get(f"{field}_exp01_tokens") or []
                    n_e = _eligible_count(toks)
                    n_s = len([t for t in t2 if t["field"] == field])
                    elig_tok += n_e
                    sel_tok += n_s
                    if n_e == 0:
                        continue
                    if n_s == 0:
                        field_mix[split]["untouched"] += 1
                    elif n_s == n_e:
                        field_mix[split]["all"] += 1
                    else:
                        field_mix[split]["partial"] += 1
                for tr in t2:
                    native_key = None
                    for tok in (rec.get("premise_exp01_tokens") or []) + (
                        rec.get("hypothesis_exp01_tokens") or []
                    ):
                        if tok.get("token_index") == tr["token_index"] and tok.get("field") == tr["field"]:
                            native_key = nfc(str(tok.get("native_text", "")).rstrip("।॥").strip())
                            break
                    sel = tr["selected"].rstrip(".")
                    if native_key and (native_key, sel.casefold()) in banned:
                        excluded_hits += 1
                        issues.append(f"excluded variant selected {eid} {native_key}->{sel}")
                    if row2 and sel not in row2["premise"] + row2["hypothesis"]:
                        issues.append(f"EXP-02 selected not in sentence {eid} {sel}")
                for tr in t3:
                    if row3 and tr["selected"] not in row3["premise"] + row3["hypothesis"]:
                        issues.append(f"EXP-03 selected not in sentence {eid} {tr['selected']}")
                for a, b in zip(t2, t3, strict=True):
                    exp02_exp03_checks += 1
                    if a["field"] != b["field"] or a["token_index"] != b["token_index"]:
                        issues.append(f"token index mismatch {eid}")
                        continue
                    if a["native_offsets"] != b["native_offsets"]:
                        issues.append(f"native span mismatch {eid}")
                        continue
                    da = int(a["levenshtein_distance"])
                    db = int(b["levenshtein_distance"])
                    if da != db or da < 1:
                        issues.append(f"distance mismatch {eid}")
                        continue
                    exp02_exp03_ok += 1
                for row in t2:
                    if row["levenshtein_distance"] < 1:
                        issues.append(f"EXP-02 d=0 {eid}")
                by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for t in t3:
                    by_field[t["field"]].append(t)
                for field_name, ts in by_field.items():
                    by_noise: dict[str, set[str]] = defaultdict(set)
                    for t in ts:
                        by_noise[t["selected"]].add(t["canonical"])
                    for noise, canons in by_noise.items():
                        if len(canons) > 1:
                            exp03_collision_pairs += len(canons) * (len(canons) - 1) // 2
                    for i in range(len(ts)):
                        for j in range(i + 1, len(ts)):
                            a, b = ts[i], ts[j]
                            if len(a["canonical"]) != len(b["canonical"]):
                                continue
                            if a["levenshtein_distance"] != b["levenshtein_distance"]:
                                continue
                            exp03_pos_pair_total += 1
                            pa = tuple(
                                k
                                for k, (x, y) in enumerate(zip(a["canonical"], a["selected"]))
                                if x != y
                            )
                            pb = tuple(
                                k
                                for k, (x, y) in enumerate(zip(b["canonical"], b["selected"]))
                                if x != y
                            )
                            if pa == pb:
                                exp03_identical_pos_pairs += 1
        empirical_p = sel_tok / elig_tok if elig_tok else 0.0
        selection_stats[split] = {
            "eligible_tokens": elig_tok,
            "selected_tokens": sel_tok,
            "empirical_p_variant": round(empirical_p, 6),
        }
        if field_mix[split]["partial"] == 0:
            issues.append(f"no partial fields in {split} (token-level selection check)")
        if not (0.33 <= empirical_p <= 0.37):
            issues.append(f"empirical p out of band for {split}: {empirical_p:.4f}")

    if exp03_pos_pair_total and exp03_identical_pos_pairs / exp03_pos_pair_total > 0.95:
        issues.append(
            f"EXP-03 identical substitution positions {exp03_identical_pos_pairs}/{exp03_pos_pair_total}"
        )

    if NLI_EXPERIMENTS_V1.is_dir():
        for rel in ("exp00_reference", "exp01", "english_retention"):
            v1_root = NLI_EXPERIMENTS_V1 / rel
            v2_root = root / rel
            for path in sorted(v1_root.rglob("*.jsonl")):
                rel_path = path.relative_to(NLI_EXPERIMENTS_V1)
                v2_path = root / rel_path
                if not v2_path.is_file():
                    issues.append(f"missing frozen v2 file {rel_path}")
                    continue
                if sha256_file(path) != sha256_file(v2_path):
                    issues.append(f"frozen file hash mismatch {rel_path}")

    # Residual Devanagari in romanized fields
    for split in ("train", "dev", "test"):
        for cond in CONDITIONS:
            path = exp_condition_root(cond) / "both" / f"{split}.jsonl"
            if not path.is_file():
                continue
            for row in _load_jsonl(path):
                if DEVANAGARI.search(row["premise"]) or DEVANAGARI.search(row["hypothesis"]):
                    issues.append(f"residual Devanagari {row['example_id']} {cond} both")

    for stale in (ENGLISH_RETENTION / "train.jsonl", EXP00 / "train.jsonl"):
        if stale.is_file():
            issues.append(f"unexpected non-training artifact (remove): {stale.relative_to(root)}")

    disk_counts = collect_row_counts_from_disk()
    manifest_path = root / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_counts = manifest.get("row_counts") or {}
        for key, on_disk in disk_counts.items():
            recorded = manifest_counts.get(key)
            if recorded != on_disk:
                issues.append(f"manifest row_counts[{key}]={recorded} != on_disk {on_disk}")
        for key in manifest_counts:
            if key not in disk_counts:
                issues.append(f"manifest row_counts lists missing file: {key}")
        for entry in manifest.get("evaluation_matrix", {}).get("EXP-00_untrained_base", {}).get(
            "evaluation_sets", []
        ):
            rel = entry["path"]
            if "/train.jsonl" in rel and (
                "english_retention" in rel or "exp00_reference" in rel
            ):
                issues.append(f"train file in evaluation matrix: {rel}")
            from romcom_data.paths import PROJECT_ROOT

            path = PROJECT_ROOT / rel
            if not path.is_file():
                issues.append(f"eval matrix missing {rel}")
                continue
            rows_on_disk = len(_load_jsonl(path))
            if entry.get("rows") != rows_on_disk:
                issues.append(
                    f"eval matrix rows {entry.get('rows')} != {rows_on_disk} for {rel}"
                )

    return {
        "ok": not issues,
        "issues": issues[:200],
        "issue_count": len(issues),
        "exp02_exp03_checks": exp02_exp03_checks,
        "exp02_exp03_ok": exp02_exp03_ok,
        "exp02_selection_by_split": selection_stats,
        "field_mix_by_split": {k: dict(v) for k, v in field_mix.items()},
        "excluded_variant_selections": excluded_hits,
        "exp03_noise_collision_pairs": exp03_collision_pairs,
        "exp03_identical_substitution_position_pairs": exp03_identical_pos_pairs,
        "exp03_substitution_position_pair_total": exp03_pos_pair_total,
    }


def reproduce_sample_hash(sample_size: int = 2000, seed: int = 0) -> dict[str, Any]:
    import random

    lexicon = NliLexicon()
    pools = VariantPools(lexicon)
    rows = load_approved_pairs()
    rng = random.Random(seed)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    mismatches = 0
    checked = 0
    for split, split_rows in by_split.items():
        picks = split_rows if len(split_rows) <= sample_size else rng.sample(split_rows, sample_size)
        for row in picks:
            prem = tokenize_and_romanize_field(row["hindi_premise"], "premise", lexicon)
            hyp = tokenize_and_romanize_field(row["hindi_hypothesis"], "hypothesis", lexicon)
            if prem.fallback_failures or hyp.fallback_failures:
                mismatches += 1
                continue
            eid = row["example_id"]
            _, t2p = apply_exp02(prem, pools, eid)
            _, t2h = apply_exp02(hyp, pools, eid)
            apply_exp03(prem, t2p, pools, eid)
            apply_exp03(hyp, t2h, pools, eid)
            checked += 1
    return {"reproducibility_checked": checked, "fallback_failures_in_sample": mismatches}
