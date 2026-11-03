"""Build RomCoM-X NLI experiment JSONL artifacts."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from romcom_data.checksums import sha256_file, write_checksum_manifest
from romcom_data.config_loader import load_augmentation_config
from romcom_data.nli_exp_freeze import (
    assert_expected_counts,
    load_approved_pairs,
    write_alignment_manifest,
)
from romcom_data.nli_exp_paths import (
    CONDITIONS,
    ENGLISH_RETENTION,
    ENGLISH_RETENTION_SPLITS,
    EXPECTED_COUNTS,
    EXP00,
    EXP00_SPLITS,
    NLI_EXPERIMENTS,
    NLI_EXPERIMENTS_V1,
    NLI_EXP_LOGS,
    NLI_EXP_REPORTS,
    SCOPES_EVAL,
    SCOPES_TRAIN,
    exp_condition_root,
    ensure_nli_experiment_layout,
)
from romcom_data.nli_exp_romanize import FieldRomanization, NliLexicon, tokenize_and_romanize_field
from romcom_data.nli_exp_variant_exclusions import (
    EXCLUSIONS_JSON,
    exclusions_sha256,
    write_exclusion_file,
)
from romcom_data.nli_exp_variants import (
    BASE_SEED,
    P_VARIANT,
    VariantPools,
    apply_exp02,
    apply_exp03,
    build_noncanonical_pools,
)
from romcom_data.paths import PROJECT_ROOT


@dataclass
class BuildStats:
    token_methods: Counter[str] = field(default_factory=Counter)
    exp02_selected: int = 0
    exp02_eligible: int = 0
    exp02_dist: Counter[int] = field(default_factory=Counter)
    exp03_resamples: Counter[int] = field(default_factory=Counter)
    fallback_failures: int = 0
    oov_words: Counter[str] = field(default_factory=Counter)
    rows_by_split: Counter[str] = field(default_factory=Counter)
    exp02_eligible_by_split: Counter[str] = field(default_factory=Counter)
    exp02_selected_by_split: Counter[str] = field(default_factory=Counter)
    field_untouched: Counter[str] = field(default_factory=Counter)
    field_partial: Counter[str] = field(default_factory=Counter)
    field_all: Counter[str] = field(default_factory=Counter)


def _compose_row(
    base: dict[str, Any],
    *,
    condition: str,
    scope: str,
    premise: str,
    hypothesis: str,
    alignment_manifest_sha256: str,
) -> dict[str, Any]:
    prem_roman = scope in ("both", "premise_only")
    hyp_roman = scope in ("both", "hypothesis_only")
    return {
        "example_id": base["example_id"],
        "split": base["split"],
        "condition": condition,
        "scope": scope,
        "premise": premise,
        "hypothesis": hypothesis,
        "premise_script": "roman" if prem_roman else "devanagari",
        "hypothesis_script": "roman" if hyp_roman else "devanagari",
        "nli_label": base["nli_label"],
        "label_id": base["label_id"],
        "source": {
            "english_row_index": base["english_row_index"],
            "hindi_row_index": base["hindi_row_index"],
            "english_pair_id": base.get("english_pair_id") or "",
            "english_source": base.get("english_source") or "",
            "hindi_source": base.get("hindi_source") or "",
            "matching_method": base.get("matching_method") or "",
        },
        "alignment_manifest_sha256": alignment_manifest_sha256,
    }


def _write_line(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _record_field_mix(
    field_rom: FieldRomanization,
    traces: list[dict[str, Any]],
    split: str,
    pools: VariantPools,
    stats: BuildStats,
) -> None:
    eligible = sum(1 for t in field_rom.tokens if pools.eligible(t))
    selected = len(traces)
    if eligible == 0:
        return
    if selected == 0:
        stats.field_untouched[split] += 1
    elif selected == eligible:
        stats.field_all[split] += 1
    else:
        stats.field_partial[split] += 1


def _transform_record(
    base: dict[str, Any],
    lexicon: NliLexicon,
    pools: VariantPools,
    stats: BuildStats,
) -> dict[str, Any]:
    example_id = base["example_id"]
    split = base["split"]
    prem = tokenize_and_romanize_field(base["hindi_premise"], "premise", lexicon)
    hyp = tokenize_and_romanize_field(base["hindi_hypothesis"], "hypothesis", lexicon)
    failures = prem.fallback_failures + hyp.fallback_failures
    if failures:
        stats.fallback_failures += len(failures)
        raise ValueError(f"fallback_failure on {example_id}: {failures[:1]}")
    for field_rom in (prem, hyp):
        for tok in field_rom.tokens:
            stats.token_methods[tok.method] += 1
            if tok.method == "fallback_charmap":
                stats.oov_words[tok.native_text.rstrip("।॥")] += 1
        for tok in field_rom.tokens:
            if pools.eligible(tok):
                stats.exp02_eligible += 1
                stats.exp02_eligible_by_split[split] += 1
    exp02_p, t2p = apply_exp02(prem, pools, example_id)
    exp02_h, t2h = apply_exp02(hyp, pools, example_id)
    stats.exp02_selected += len(t2p) + len(t2h)
    stats.exp02_selected_by_split[split] += len(t2p) + len(t2h)
    _record_field_mix(prem, t2p, split, pools, stats)
    _record_field_mix(hyp, t2h, split, pools, stats)
    for tr in t2p + t2h:
        stats.exp02_dist[tr["levenshtein_distance"]] += 1
    exp03_p, t3p = apply_exp03(prem, t2p, pools, example_id)
    exp03_h, t3h = apply_exp03(hyp, t2h, pools, example_id)
    for tr in t3p + t3h:
        stats.exp03_resamples[tr.get("resample_attempts", 0)] += 1
    return {
        "native_premise": base["hindi_premise"],
        "native_hypothesis": base["hindi_hypothesis"],
        "EXP-01": (prem.roman, hyp.roman),
        "EXP-02": (exp02_p, exp02_h),
        "EXP-03": (exp03_p, exp03_h),
        "log": {
            "example_id": example_id,
            "split": base["split"],
            "premise_exp01_tokens": [t.__dict__ for t in prem.tokens],
            "hypothesis_exp01_tokens": [t.__dict__ for t in hyp.tokens],
            "exp02_traces": t2p + t2h,
            "exp03_traces": t3p + t3h,
            "seeds": {"base_seed": BASE_SEED, "p_variant": P_VARIANT},
        },
    }


def _jsonl_line_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def collect_row_counts_from_disk() -> Counter[str]:
    """Row counts from generated JSONL files (source of truth for manifests)."""
    counts: Counter[str] = Counter()
    for split in ("train", "dev", "test"):
        scopes = SCOPES_TRAIN if split == "train" else SCOPES_EVAL
        for cond in CONDITIONS:
            for scope in scopes:
                path = exp_condition_root(cond) / scope / f"{split}.jsonl"
                n = _jsonl_line_count(path)
                if n:
                    counts[f"{cond}:{scope}:{split}"] = n
        if split in EXP00_SPLITS:
            counts[f"EXP-00:both:{split}"] = _jsonl_line_count(EXP00 / f"{split}.jsonl")
        if split in ENGLISH_RETENTION_SPLITS:
            counts[f"english_retention:both:{split}"] = _jsonl_line_count(
                ENGLISH_RETENTION / f"{split}.jsonl"
            )
    return counts


def _copy_tree_from_v1(rel: str) -> None:
    src = NLI_EXPERIMENTS_V1 / rel
    dst = NLI_EXPERIMENTS / rel
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(src, dst, copy_function=os.link, dirs_exist_ok=False)
    except (OSError, AttributeError):
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def prepare_v2_tree_from_v1() -> None:
    """Link/copy frozen V1 artifacts into V2; do not modify V1."""
    ensure_nli_experiment_layout()
    if not NLI_EXPERIMENTS_V1.is_dir():
        raise FileNotFoundError(f"missing V1 snapshot {NLI_EXPERIMENTS_V1}")
    for rel in ("exp00_reference", "exp01", "english_retention"):
        _copy_tree_from_v1(rel)
    shutil.copy2(
        NLI_EXPERIMENTS_V1 / "alignment_manifest.json",
        NLI_EXPERIMENTS / "alignment_manifest.json",
    )
    audit = NLI_EXPERIMENTS_V1 / "reports/devanagari_codepoint_audit.json"
    if audit.is_file():
        NLI_EXP_REPORTS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audit, NLI_EXP_REPORTS / "devanagari_codepoint_audit.json")
    if not EXCLUSIONS_JSON.is_file():
        write_exclusion_file()
    NLI_EXP_REPORTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXCLUSIONS_JSON, NLI_EXP_REPORTS / EXCLUSIONS_JSON.name)


def _open_outputs(split: str, *, v2_partial: bool = True) -> dict[str, TextIO]:
    handles: dict[str, TextIO] = {}
    split_file = f"{split}.jsonl"
    if not v2_partial:
        if split in EXP00_SPLITS:
            handles["exp00"] = (EXP00 / split_file).open("w", encoding="utf-8")
        if split in ENGLISH_RETENTION_SPLITS:
            handles["english"] = (ENGLISH_RETENTION / split_file).open("w", encoding="utf-8")
        scopes = SCOPES_TRAIN if split == "train" else SCOPES_EVAL
        for cond in CONDITIONS:
            root = exp_condition_root(cond)
            for scope in scopes:
                path = root / scope / split_file
                handles[f"{cond}:{scope}"] = path.open("w", encoding="utf-8")
    else:
        scopes = SCOPES_TRAIN if split == "train" else SCOPES_EVAL
        for cond in ("EXP-02", "EXP-03"):
            root = exp_condition_root(cond)
            for scope in scopes:
                path = root / scope / split_file
                handles[f"{cond}:{scope}"] = path.open("w", encoding="utf-8")
    handles[f"log:{split}"] = (NLI_EXP_LOGS / f"transformation_log.{split}.jsonl").open(
        "w", encoding="utf-8"
    )
    return handles


def _close_handles(handles: dict[str, TextIO]) -> None:
    for handle in handles.values():
        handle.close()


def _frozen_file_hashes_v1() -> dict[str, str]:
    out: dict[str, str] = {}
    roots = (
        NLI_EXPERIMENTS_V1 / "exp00_reference",
        NLI_EXPERIMENTS_V1 / "exp01",
        NLI_EXPERIMENTS_V1 / "english_retention",
    )
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            rel = path.relative_to(NLI_EXPERIMENTS_V1)
            out[str(rel.as_posix())] = sha256_file(path)
    return out


def run_build(*, v2_partial: bool = True) -> dict[str, Any]:
    if v2_partial:
        prepare_v2_tree_from_v1()
        if not EXCLUSIONS_JSON.is_file():
            write_exclusion_file()
    else:
        ensure_nli_experiment_layout()
    config, config_hash = load_augmentation_config()
    lexicon = NliLexicon()
    pools = VariantPools(lexicon)
    pool_sha = __import__("hashlib").sha256(
        json.dumps(build_noncanonical_pools(lexicon), sort_keys=True).encode("utf-8")
    ).hexdigest()
    excl_sha = exclusions_sha256()

    rows = load_approved_pairs()
    assert_expected_counts(rows)
    if v2_partial:
        am_path = NLI_EXPERIMENTS / "alignment_manifest.json"
        alignment_manifest = json.loads(am_path.read_text(encoding="utf-8"))
        alignment_manifest["_sha256"] = sha256_file(am_path)
    else:
        alignment_manifest = write_alignment_manifest(rows)
    am_sha = alignment_manifest["_sha256"]

    stats = BuildStats()
    split_handles: dict[str, dict[str, TextIO]] = {}
    row_counts: Counter[str] = Counter()
    status = "READY_FOR_OWNER_REVIEW_V2" if v2_partial else "READY_FOR_OWNER_REVIEW"
    block_reason = ""

    try:
        for row in rows:
            split = row["split"]
            if split not in split_handles:
                split_handles[split] = _open_outputs(split, v2_partial=v2_partial)
            handles = split_handles[split]
            transformed = _transform_record(row, lexicon, pools, stats)
            stats.rows_by_split[split] += 1

            if not v2_partial and split in ("dev", "test"):
                _write_line(
                    handles["exp00"],
                    _compose_row(
                        row,
                        condition="EXP-00",
                        scope="both",
                        premise=transformed["native_premise"],
                        hypothesis=transformed["native_hypothesis"],
                        alignment_manifest_sha256=am_sha,
                    ),
                )
                _write_line(
                    handles["english"],
                    {
                        "example_id": row["example_id"],
                        "split": split,
                        "premise": row["english_premise"],
                        "hypothesis": row["english_hypothesis"],
                        "nli_label": row["nli_label"],
                        "label_id": row["label_id"],
                        "source": {
                            "english_row_index": row["english_row_index"],
                            "hindi_row_index": row["hindi_row_index"],
                            "english_pair_id": row.get("english_pair_id") or "",
                        },
                        "alignment_manifest_sha256": am_sha,
                        "benchmark": "english_retention",
                    },
                )

            scopes = SCOPES_TRAIN if split == "train" else SCOPES_EVAL
            texts = {
                "EXP-01": transformed["EXP-01"],
                "EXP-02": transformed["EXP-02"],
                "EXP-03": transformed["EXP-03"],
            }
            write_conds = ("EXP-02", "EXP-03") if v2_partial else CONDITIONS
            for cond in write_conds:
                prem_text, hyp_text = texts[cond]
                for scope in scopes:
                    if scope == "both":
                        p_out, h_out = prem_text, hyp_text
                    elif scope == "premise_only":
                        p_out, h_out = prem_text, transformed["native_hypothesis"]
                    else:
                        p_out, h_out = transformed["native_premise"], hyp_text
                    key = f"{cond}:{scope}"
                    _write_line(
                        handles[key],
                        _compose_row(
                            row,
                            condition=cond,
                            scope=scope,
                            premise=p_out,
                            hypothesis=h_out,
                            alignment_manifest_sha256=am_sha,
                        ),
                    )
                    row_counts[f"{cond}:{scope}:{split}"] += 1
            _write_line(handles[f"log:{split}"], transformed["log"])
    except ValueError as exc:
        status = "BLOCKED"
        block_reason = str(exc)
    finally:
        for handles in split_handles.values():
            _close_handles(handles)

    if status == "BLOCKED":
        return {"status": status, "block_reason": block_reason, "stats": stats}

    am_sha = alignment_manifest["_sha256"]
    row_counts = collect_row_counts_from_disk()
    eval_matrix = _build_evaluation_matrix(row_counts)
    experiment_manifest = {
        "constructed_utc": datetime.now(UTC).isoformat(),
        "dataset_version": "v2" if v2_partial else "v1",
        "parent_v1_dir": str(NLI_EXPERIMENTS_V1.relative_to(PROJECT_ROOT)) if v2_partial else None,
        "config_hash": config_hash,
        "base_seed": BASE_SEED,
        "p_variant": P_VARIANT,
        "lexicon_sha256": lexicon.inventory_sha256,
        "noncanonical_pool_table_sha256": pool_sha,
        "excluded_variant_inventory_sha256": excl_sha,
        "alignment_manifest_sha256": am_sha,
        "row_counts": dict(row_counts),
        "frozen_from_v1_sha256": _frozen_file_hashes_v1() if v2_partial else None,
        "conditions": {
            "EXP-00": "native_hindi_reference_no_training",
            "EXP-01": "dakshina_canonical_romanization",
            "EXP-02": "attested_noncanonical_dakshina_variants",
            "EXP-03": "matched_random_noise_control",
        },
        "evaluation_matrix": eval_matrix,
    }
    (NLI_EXPERIMENTS / "experiment_manifest.json").write_text(
        json.dumps(experiment_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    transform_stats = _stats_payload(stats, lexicon, v2_partial=v2_partial)
    (NLI_EXPERIMENTS / "transformation_statistics.json").write_text(
        json.dumps(transform_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    oov_path = NLI_EXP_REPORTS / "oov_inventory.jsonl"
    with oov_path.open("w", encoding="utf-8") as oov_handle:
        for word, count in sorted(stats.oov_words.items(), key=lambda x: (-x[1], x[0])):
            _write_line(oov_handle, {"native_word": word, "token_count": count})
    write_nli_experiment_checksums()
    return {
        "status": status,
        "alignment_manifest": alignment_manifest,
        "experiment_manifest": experiment_manifest,
        "stats": transform_stats,
        "row_counts": dict(row_counts),
    }


def write_nli_experiment_checksums() -> None:
    rel_files = [
        p.relative_to(PROJECT_ROOT)
        for p in NLI_EXPERIMENTS.rglob("*")
        if p.is_file()
        and not str(p).endswith(".jsonl")
        and p.name != "checksums.sha256"
    ]
    rel_files.extend(
        p.relative_to(PROJECT_ROOT) for p in NLI_EXPERIMENTS.rglob("*.jsonl")
    )
    write_checksum_manifest(PROJECT_ROOT, rel_files, NLI_EXPERIMENTS / "checksums.sha256")


def refresh_experiment_manifest() -> dict[str, Any]:
    """Recompute row counts and evaluation matrix from on-disk JSONL (no rebuild)."""
    manifest_path = NLI_EXPERIMENTS / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row_counts = collect_row_counts_from_disk()
    manifest["row_counts"] = dict(row_counts)
    manifest["evaluation_matrix"] = _build_evaluation_matrix(row_counts)
    manifest["manifest_refreshed_utc"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_nli_experiment_checksums()
    status_path = NLI_EXPERIMENTS / "construction_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["row_counts"] = dict(row_counts)
        status["experiment_manifest"] = manifest
        status["status"] = "READY_FOR_OWNER_REVIEW"
        status_path.write_text(json.dumps(status, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


def _stats_payload(
    stats: BuildStats,
    lexicon: NliLexicon,
    *,
    v2_partial: bool = True,
) -> dict[str, Any]:
    lex_hits = stats.token_methods.get("dakshina_canonical", 0)
    fallback = stats.token_methods.get("fallback_charmap", 0)
    hindi_tokens = lex_hits + fallback
    payload: dict[str, Any] = {
        "rows_by_split": dict(stats.rows_by_split),
        "token_methods": dict(stats.token_methods),
        "dictionary_coverage_pct": round(100 * lex_hits / hindi_tokens, 4) if hindi_tokens else 0,
        "oov_rate_pct": round(100 * fallback / hindi_tokens, 4) if hindi_tokens else 0,
        "fallback_failure_count": stats.fallback_failures,
        "exp02_eligible_tokens": stats.exp02_eligible,
        "exp02_selected_tokens": stats.exp02_selected,
        "exp02_distance_distribution": dict(stats.exp02_dist),
        "exp03_resample_histogram": dict(stats.exp03_resamples),
        "unique_oov_native_words": len(stats.oov_words),
    }
    if v2_partial:
        by_split: dict[str, dict[str, float | int]] = {}
        for split in ("train", "dev", "test"):
            elig = stats.exp02_eligible_by_split[split]
            sel = stats.exp02_selected_by_split[split]
            fields = (
                stats.field_untouched[split]
                + stats.field_partial[split]
                + stats.field_all[split]
            )
            by_split[split] = {
                "eligible_tokens": elig,
                "selected_tokens": sel,
                "empirical_p_variant": round(sel / elig, 6) if elig else 0.0,
                "fields_with_eligible_tokens": fields,
                "fields_untouched": stats.field_untouched[split],
                "fields_partial": stats.field_partial[split],
                "fields_all_selected": stats.field_all[split],
                "partial_field_fraction": round(stats.field_partial[split] / fields, 6)
                if fields
                else 0.0,
            }
        payload["exp02_selection_by_split"] = by_split
    return payload


def _build_evaluation_matrix(row_counts: Counter[str]) -> dict[str, Any]:
    sets: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        paths = [
            ("EXP-00", "both", EXP00 / f"{split}.jsonl"),
            ("EXP-01", "both", exp_condition_root("EXP-01") / "both" / f"{split}.jsonl"),
            ("EXP-01", "premise_only", exp_condition_root("EXP-01") / "premise_only" / f"{split}.jsonl"),
            ("EXP-01", "hypothesis_only", exp_condition_root("EXP-01") / "hypothesis_only" / f"{split}.jsonl"),
            ("EXP-02", "both", exp_condition_root("EXP-02") / "both" / f"{split}.jsonl"),
            ("EXP-02", "premise_only", exp_condition_root("EXP-02") / "premise_only" / f"{split}.jsonl"),
            ("EXP-02", "hypothesis_only", exp_condition_root("EXP-02") / "hypothesis_only" / f"{split}.jsonl"),
            ("EXP-03", "both", exp_condition_root("EXP-03") / "both" / f"{split}.jsonl"),
            ("EXP-03", "premise_only", exp_condition_root("EXP-03") / "premise_only" / f"{split}.jsonl"),
            ("EXP-03", "hypothesis_only", exp_condition_root("EXP-03") / "hypothesis_only" / f"{split}.jsonl"),
            ("english_retention", "both", ENGLISH_RETENTION / f"{split}.jsonl"),
        ]
        for cond, scope, path in paths:
            rel = str(path.relative_to(PROJECT_ROOT))
            count = row_counts.get(f"{cond}:{scope}:{split}")
            if count is None:
                count = _jsonl_line_count(path) or EXPECTED_COUNTS[split]
            sha = sha256_file(path) if path.is_file() else ""
            sets.append(
                {
                    "split": split,
                    "condition": cond,
                    "scope": scope,
                    "path": rel,
                    "rows": count,
                    "sha256": sha,
                    "training": "none",
                }
            )
    return {
        "EXP-00_untrained_base": {
            "training": "none",
            "evaluation_sets": sets,
        },
        "note": "English retention is not code-mixed robustness; no model evaluation is run by this pipeline.",
    }
