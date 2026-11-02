"""Phase 2.1 Hindi refusal-response dataset construction (EXP-01/02/03)."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from romcom_data.augmentation import levenshtein_distance
from romcom_data.checksums import write_checksum_manifest
from romcom_data.config_loader import load_augmentation_config
from romcom_data.generate_exps import generate_one
from romcom_data.handoff_paths import (
    INVENTORIES,
    RAW_TOX,
    TOXICITY_AUDIT,
    TOXICITY_EVAL_REF,
    TOXICITY_EXP01,
    TOXICITY_EXP02,
    TOXICITY_EXP03,
    TOXICITY_MANIFESTS,
    TOXICITY_PROCESSED,
    TOXICITY_REPORTS,
    TOXICITY_TRACES,
    ensure_handoff_layout,
)
from romcom_data.paths import PROJECT_ROOT
from romcom_data.romanize import DakshinaRomanizer, residual_devanagari_chars, romanize_mixed_text
from romcom_data.source_discovery import save_json
from romcom_data.text_utils import nfc

_DANDA = ("\u0964", "\u0965")
_IGNORED_DEVANAGARI = frozenset({"\u0964", "\u0965", "\u093c", "\u094d"})
from romcom_data.toxicity_alignment import _pair
from romcom_data.toxicity_freeze import checksum_file

FROZEN_MANIFEST_SHA256 = (
    "58d85ddaf78bb05600bc802049c8ce12ef8886f4439bfe87b9ca9fd25e1673ae"
)
TASK_NAME = "hindi_refusal_response_generation"
SPLIT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_frozen_manifest(path: Path | None = None) -> tuple[list[dict[str, str]], str]:
    path = path or (TOXICITY_AUDIT / "final_source_manifest.csv")
    digest = checksum_file(path)
    if digest != FROZEN_MANIFEST_SHA256:
        raise ValueError(
            f"manifest sha256 mismatch: expected {FROZEN_MANIFEST_SHA256}, got {digest}"
        )
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    retained = [r for r in rows if r.get("exclusion_status") == "retained"]
    return retained, digest


def _index_sources() -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    toxic_by_doc: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(RAW_TOX / "Toxic_Matrix_train.jsonl"):
        toxic_by_doc[str(row["doc_id"])] = row
    benign_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    benign_path = TOXICITY_AUDIT / "dolly_t_seed42_sample.jsonl"
    for row in _load_jsonl(benign_path):
        key = (str(row["doc_id"]), str(row["source_row_index"]))
        benign_by_key[key] = row
    return toxic_by_doc, benign_by_key


def _resolve_source_row(
    manifest_row: dict[str, str],
    toxic_by_doc: dict[str, dict[str, Any]],
    benign_by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    doc_id = manifest_row["doc_id"]
    label = manifest_row["label"]
    if label == "toxic":
        row = toxic_by_doc.get(doc_id)
        if row is None:
            raise KeyError(f"missing toxic source row for doc_id={doc_id}")
        return row
    key = (doc_id, str(manifest_row.get("source_row_index") or ""))
    row = benign_by_key.get(key)
    if row is None:
        raise KeyError(f"missing benign source row for {key}")
    return row


def build_base_record(
    manifest_row: dict[str, str],
    source_row: dict[str, Any],
    romanizer: DakshinaRomanizer,
) -> dict[str, Any]:
    user, assistant = _pair(source_row, "hin_Deva")
    user = nfc(user)
    assistant = nfc(assistant)
    expected_prompt_hash = hashlib.sha256(user.encode("utf-8")).hexdigest()
    if manifest_row.get("user_prompt_sha256") and manifest_row["user_prompt_sha256"] != expected_prompt_hash:
        raise ValueError(
            f"user prompt hash mismatch for {manifest_row['example_id']}"
        )
    romanized, pre_traces = romanize_mixed_text(assistant, romanizer)
    return {
        "example_id": manifest_row["example_id"],
        "source_id": manifest_row["source_id"],
        "doc_id": manifest_row["doc_id"],
        "source_row_index": manifest_row.get("source_row_index") or "",
        "task": TASK_NAME,
        "label": manifest_row["label"],
        "language": manifest_row.get("language", "hi"),
        "project_split": manifest_row["project_split"],
        "group_id": manifest_row.get("group_id", ""),
        "original_split": manifest_row.get("original_split", "train"),
        "input": {"prompt": user},
        "target": {
            "response": romanized,
            "source_devanagari": assistant,
        },
        "augmentation_side": "target_only",
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "pre_romanization_trace": pre_traces,
        "user_prompt_sha256": expected_prompt_hash,
    }


def _processed_view(row: dict[str, Any], condition: str) -> dict[str, Any]:
    target = dict(row.get("target") or {})
    target.pop("source_devanagari", None)
    return {
        "example_id": row["example_id"],
        "source_id": row["source_id"],
        "task": row["task"],
        "label": row["label"],
        "language": row.get("language", "hi"),
        "project_split": row["project_split"],
        "group_id": row.get("group_id", ""),
        "condition": condition,
        "input": row["input"],
        "target": target,
        "augmentation_side": row.get("augmentation_side", "target_only"),
        "source_manifest_sha256": row.get("source_manifest_sha256"),
        "user_prompt_sha256": row.get("user_prompt_sha256"),
        "p_variant": row.get("p_variant"),
    }


def _eval_reference_row(base: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": base["example_id"],
        "source_id": base["source_id"],
        "label": base["label"],
        "project_split": base["project_split"],
        "input": base["input"],
        "target_devanagari": base["target"]["source_devanagari"],
        "user_prompt_sha256": base["user_prompt_sha256"],
        "source_manifest_sha256": FROZEN_MANIFEST_SHA256,
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _aggregate_transformation_stats(
    bases: list[dict[str, Any]],
    trace_paths: dict[str, Path],
) -> dict[str, Any]:
    pre = Counter()
    oov_tokens = 0
    lexicon_hits = 0
    fallback_tokens = 0
    latin_passthrough = 0
    for base in bases:
        for tok in base.get("pre_romanization_trace") or []:
            op = str(tok.get("operation", ""))
            method = str(tok.get("method", op))
            pre[method] += 1
            if op == "latin_passthrough":
                latin_passthrough += 1
            elif op == "devanagari_romanize":
                if method == "dakshina_canonical":
                    lexicon_hits += 1
                elif method.startswith("fallback"):
                    fallback_tokens += 1
                    oov_tokens += 1
                elif not tok.get("attested"):
                    oov_tokens += 1
    total_dev_tokens = lexicon_hits + fallback_tokens
    dictionary_coverage_pct = (
        round(100 * lexicon_hits / total_dev_tokens, 4) if total_dev_tokens else 0.0
    )
    cond_stats: dict[str, Any] = {}
    for cond, path in trace_paths.items():
        token_ops = Counter()
        ties = 0
        changed = 0
        unchanged = 0
        noise_total = 0
        noise_matched = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                for tok in rec.get("tokens") or []:
                    op = str(tok.get("operation", ""))
                    token_ops[op] += 1
                    if op == "canonical_tie_fallback":
                        ties += 1
                    if op in {"canonicalize", "attested_variant", "random_noise"}:
                        changed += 1
                    if op == "unchanged":
                        unchanged += 1
                    if op == "random_noise":
                        noise_total += 1
                        if tok.get("distance_match"):
                            noise_matched += 1
        cond_stats[cond] = {
            "token_operation_counts": dict(token_ops),
            "canonical_tie_markers": ties,
            "tokens_changed": changed,
            "tokens_unchanged": unchanged,
            "exp03_noise_distance_match_rate": (
                noise_matched / noise_total if noise_total else 1.0
            ),
            "exp03_noise_tokens_evaluated": noise_total,
        }
    sentence_level = {"examples": len(bases), "pre_romanization_methods": dict(pre)}
    residual = sum(
        residual_devanagari_chars(str(b["target"].get("response", ""))) for b in bases
    )
    sentence_level["residual_devanagari_in_precanonical_base"] = residual
    return {
        "token_level": cond_stats,
        "sentence_level": sentence_level,
        "dictionary_coverage": {
            "devanagari_tokens_lexicon_hit": lexicon_hits,
            "devanagari_tokens_fallback_or_oov": fallback_tokens,
            "latin_passthrough_tokens": latin_passthrough,
            "dictionary_coverage_pct": dictionary_coverage_pct,
            "canonical_tie_markers_exp01": cond_stats.get("EXP-01", {}).get(
                "canonical_tie_markers", 0
            ),
            "note_ties": (
                "canonical_tie_fallback marks frequency ties; lexicographic fallback is "
                "reproducibility only, not an empirical preference claim."
            ),
        },
    }


def _residual_devanagari_letters(text: str) -> int:
    return sum(
        1
        for ch in text
        if ord(ch) >= 0x0900 and ord(ch) <= 0x097F and ch not in _IGNORED_DEVANAGARI
    )


def validate_outputs(
    *,
    bases: list[dict[str, Any]],
    manifest_rows: list[dict[str, str]],
    trace_paths: dict[str, Path],
) -> dict[str, Any]:
    issues: list[str] = []
    exp_dirs = {
        "EXP-01": TOXICITY_EXP01,
        "EXP-02": TOXICITY_EXP02,
        "EXP-03": TOXICITY_EXP03,
    }
    by_split: dict[str, dict[str, list[dict[str, Any]]]] = {
        c: {s: [] for s in SPLIT_FILES} for c in exp_dirs
    }
    for cond, folder in exp_dirs.items():
        for split_file in SPLIT_FILES:
            path = folder / split_file
            if not path.is_file():
                issues.append(f"missing {path}")
                continue
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    by_split[cond][split_file].append(json.loads(line))

    manifest_ids = {r["example_id"] for r in manifest_rows}
    for cond in exp_dirs:
        seen: set[str] = set()
        for split_file in SPLIT_FILES:
            for row in by_split[cond][split_file]:
                eid = row["example_id"]
                if eid in seen:
                    issues.append(f"duplicate example_id {eid} in {cond}")
                seen.add(eid)
                if eid not in manifest_ids:
                    issues.append(f"unknown example_id {eid} in {cond}")
        if len(seen) != len(manifest_ids):
            issues.append(f"{cond} row count {len(seen)} != manifest {len(manifest_ids)}")

    for split_file in SPLIT_FILES:
        ids = [r["example_id"] for r in by_split["EXP-01"][split_file]]
        for cond in ("EXP-02", "EXP-03"):
            other = [r["example_id"] for r in by_split[cond][split_file]]
            if ids != other:
                issues.append(f"example_id order mismatch {cond} {split_file}")

    train_ids = {r["example_id"] for r in by_split["EXP-01"]["train.jsonl"]}
    for split_file in ("validation.jsonl", "test.jsonl"):
        overlap = train_ids & {r["example_id"] for r in by_split["EXP-01"][split_file]}
        if overlap:
            issues.append(f"train ids leaked into {split_file}: {len(overlap)}")

    base_by_id = {b["example_id"]: b for b in bases}
    for split_file in SPLIT_FILES:
        rows1 = by_split["EXP-01"][split_file]
        if not rows1:
            continue
        for cond in ("EXP-02", "EXP-03"):
            rows_other = {r["example_id"]: r for r in by_split[cond][split_file]}
            for r1 in rows1:
                eid = r1["example_id"]
                r2 = rows_other[eid]
                for field in ("source_id", "label", "project_split"):
                    if r1.get(field) != r2.get(field):
                        issues.append(f"{field} mismatch {eid} {cond}")
                if json.dumps(r1["input"], ensure_ascii=False) != json.dumps(
                    r2["input"], ensure_ascii=False
                ):
                    issues.append(f"input prompt differs across conditions for {eid}")
                for row in (r1, r2):
                    if row["target"].get("source_devanagari"):
                        issues.append(f"source_devanagari leaked into processed {eid}")

    exp02_changes: dict[str, list[dict[str, Any]]] = {}
    with trace_paths["EXP-02"].open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            exp02_changes[rec["example_id"]] = [
                t for t in rec.get("tokens") or [] if t.get("operation") == "attested_variant"
            ]
    exp03_by_id: dict[str, list[dict[str, Any]]] = {}
    with trace_paths["EXP-03"].open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            exp03_by_id[rec["example_id"]] = rec.get("tokens") or []

    pos_checks = 0
    pos_ok = 0
    for eid, changes in exp02_changes.items():
        noise_tokens = [
            t for t in exp03_by_id.get(eid, []) if t.get("operation") == "random_noise"
        ]
        if len(changes) != len(noise_tokens):
            issues.append(f"EXP-02/03 position count mismatch for {eid}")
            continue
        for ch, nz in zip(changes, noise_tokens, strict=True):
            pos_checks += 1
            if ch.get("offsets") != nz.get("offsets"):
                issues.append(f"offset mismatch EXP-02/03 for {eid}")
                continue
            orig = ch.get("original", "")
            sel2 = ch.get("selected", "")
            target_dist = levenshtein_distance(orig.casefold(), sel2.casefold())
            if int(nz.get("target_levenshtein_distance", -1)) != target_dist:
                issues.append(f"edit distance metadata mismatch for {eid}")
                continue
            if nz.get("distance_match"):
                pos_ok += 1

    residual_examples = 0
    for split_file in SPLIT_FILES:
        for row in by_split["EXP-01"][split_file]:
            resp = str(row.get("target", {}).get("response", ""))
            if _residual_devanagari_letters(resp) > 0:
                residual_examples += 1
                if residual_examples <= 20:
                    issues.append(
                        f"residual Devanagari letters in model_target for {row['example_id']}"
                    )
    if residual_examples > 20:
        issues.append(
            f"residual Devanagari letters in model_target for {residual_examples} examples total"
        )

    split_counts = Counter(r["project_split"] for r in manifest_rows)
    return {
        "ok": not issues,
        "issues": issues,
        "split_counts_manifest": dict(split_counts),
        "exp02_exp03_position_checks": pos_checks,
        "exp02_exp03_position_checks_distance_ok": pos_ok,
        "train_only_in_train_file": len(train_ids),
        "examples_with_residual_devanagari_letters": residual_examples,
    }


def run_toxicity_construction() -> dict[str, Any]:
    ensure_handoff_layout()
    manifest_rows, manifest_digest = load_frozen_manifest()
    config, config_hash = load_augmentation_config()
    p_variant = float(config.get("exp02", {}).get("p_variant", 0.35))
    base_seed = int(config.get("base_seed", 42))

    romanizer = DakshinaRomanizer(INVENTORIES / "dakshina_hi_variant_inventory.tsv")
    toxic_by_doc, benign_by_key = _index_sources()

    out_paths: dict[str, dict[str, Path]] = {}
    for cond, root in (
        ("EXP-01", TOXICITY_EXP01),
        ("EXP-02", TOXICITY_EXP02),
        ("EXP-03", TOXICITY_EXP03),
    ):
        out_paths[cond] = {}
        for split_file in SPLIT_FILES:
            path = root / split_file
            path.write_text("", encoding="utf-8")
            out_paths[cond][split_file] = path

    trace_paths = {
        "EXP-01": TOXICITY_TRACES / "EXP-01" / "traces.jsonl",
        "EXP-02": TOXICITY_TRACES / "EXP-02" / "traces.jsonl",
        "EXP-03": TOXICITY_TRACES / "EXP-03" / "traces.jsonl",
    }
    transform_log = TOXICITY_REPORTS / "transformation_log.jsonl"
    transform_log.write_text("", encoding="utf-8")
    for p in trace_paths.values():
        p.write_text("", encoding="utf-8")

    eval_paths = {
        "validation.jsonl": TOXICITY_EVAL_REF / "validation.jsonl",
        "test.jsonl": TOXICITY_EVAL_REF / "test.jsonl",
    }
    for p in eval_paths.values():
        p.write_text("", encoding="utf-8")

    bases: list[dict[str, Any]] = []
    split_to_file = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
    stream_counts: Counter[str] = Counter()

    for i, mrow in enumerate(manifest_rows):
        if i and i % 1000 == 0:
            print(f"  construct {i}/{len(manifest_rows)}", flush=True)
        source_row = _resolve_source_row(mrow, toxic_by_doc, benign_by_key)
        base = build_base_record(mrow, source_row, romanizer)
        bases.append(base)
        split = mrow["project_split"]
        fname = split_to_file[split]
        conds = generate_one(base, romanizer, p_variant=p_variant, base_seed=base_seed)
        if split in ("validation", "test"):
            _append_jsonl(eval_paths[fname], _eval_reference_row(base))
        for cond_name in ("EXP-01", "EXP-02", "EXP-03"):
            view = _processed_view(conds[cond_name], cond_name)
            _append_jsonl(out_paths[cond_name][fname], view)
            _append_jsonl(
                trace_paths[cond_name],
                {
                    "example_id": base["example_id"],
                    "source_id": base["source_id"],
                    "project_split": split,
                    "condition": cond_name,
                    "pre_romanization": base.get("pre_romanization_trace"),
                    "tokens": conds["traces"][cond_name],
                },
            )
            stream_counts[f"{cond_name}:{split}"] += 1
        _append_jsonl(
            transform_log,
            {
                "example_id": base["example_id"],
                "source_id": base["source_id"],
                "project_split": split,
                "label": base["label"],
                "pre_romanization_trace": base.get("pre_romanization_trace"),
                "exp01_tokens": conds["traces"]["EXP-01"],
                "exp02_tokens": conds["traces"]["EXP-02"],
                "exp03_tokens": conds["traces"]["EXP-03"],
            },
        )

    stats = _aggregate_transformation_stats(bases, trace_paths)
    validation = validate_outputs(bases=bases, manifest_rows=manifest_rows, trace_paths=trace_paths)

    rel_files: list[Path] = []
    for folder in (TOXICITY_EXP01, TOXICITY_EXP02, TOXICITY_EXP03, TOXICITY_TRACES, TOXICITY_EVAL_REF, TOXICITY_REPORTS):
        rel_files.extend(p.relative_to(PROJECT_ROOT) for p in folder.rglob("*") if p.is_file())
    checksum_path = TOXICITY_MANIFESTS / "checksums.sha256"
    write_checksum_manifest(PROJECT_ROOT, rel_files, checksum_path)

    dataset_manifest = {
        "phase": "2.1",
        "task": TASK_NAME,
        "constructed_utc": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": manifest_digest,
        "config_hash": config_hash,
        "exp_conditions": {
            "EXP-00": "untrained_base_model_no_dataset",
            "EXP-01": "canonical_hindi_romanization",
            "EXP-02": "attested_dakshina_variants_p_variant",
            "EXP-03": "matched_random_noise_from_exp02_positions",
        },
        "p_variant": p_variant,
        "base_seed": base_seed,
        "row_counts": dict(stream_counts),
        "retained_examples": len(manifest_rows),
        "splits": validation.get("split_counts_manifest"),
        "training_files": {
            "EXP-01": str((TOXICITY_EXP01 / "train.jsonl").relative_to(PROJECT_ROOT)),
            "EXP-02": str((TOXICITY_EXP02 / "train.jsonl").relative_to(PROJECT_ROOT)),
            "EXP-03": str((TOXICITY_EXP03 / "train.jsonl").relative_to(PROJECT_ROOT)),
        },
        "evaluation_reference": {
            "validation": str(eval_paths["validation.jsonl"].relative_to(PROJECT_ROOT)),
            "test": str(eval_paths["test.jsonl"].relative_to(PROJECT_ROOT)),
            "note": "Devanagari assistant references only; not for training or hyperparameter selection.",
        },
    }
    save_json(TOXICITY_MANIFESTS / "dataset_manifest.json", dataset_manifest)
    save_json(TOXICITY_REPORTS / "transformation_statistics.json", stats)
    save_json(TOXICITY_REPORTS / "validation_report.json", validation)

    status = "READY_FOR_OWNER_REVIEW" if validation["ok"] else "BLOCKED"
    save_json(
        TOXICITY_REPORTS / "construction_status.json",
        {
            "status": status,
            "block_reason": validation["issues"][:5] if not validation["ok"] else "",
            "source_manifest_sha256": manifest_digest,
        },
    )
    (TOXICITY_AUDIT / "toxicity_audit_status.json").write_text(
        json.dumps(
            {
                **json.loads((TOXICITY_AUDIT / "toxicity_audit_status.json").read_text(encoding="utf-8")),
                "augmentation": "completed",
                "construction_status": status,
                "toxicity_processed_manifest": str(
                    (TOXICITY_MANIFESTS / "dataset_manifest.json").relative_to(PROJECT_ROOT)
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": status,
        "validation": validation,
        "stats": stats,
        "dataset_manifest": dataset_manifest,
    }
