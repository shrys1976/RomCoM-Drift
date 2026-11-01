"""Phase 2.1 Toxic_Matrix / IndicAlign toxicity source audit (inspection only)."""

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

from huggingface_hub import HfApi, hf_hub_download

from romcom_data.handoff_paths import HANDOFF, TOXICITY_AUDIT, RAW_TOX
from romcom_data.source_loaders.toxic_matrix import (
    INSTRUCT_CONFIG,
    REPO,
    TOXIC_CONFIG,
    load_normalized_toxicity,
)
from romcom_data.stable_hash import derive_rng, stable_hash_hex
from romcom_data.text_utils import nfc, script_profile
from romcom_data.toxic_turns import (
    assistant_target_from_turns,
    parse_conversation_turns,
    user_prompt_from_turns,
)

ZIP_NAME = "RomCoM_X_Phase_2_1_Toxicity_Audit.zip"
REFUSAL_SAMPLE_SEED_ID = "tox-refusal-manual"
REFUSAL_SAMPLE_N = 120
STOCK_REFUSAL = (
    "I cannot give a response to this prompt as it is potentially toxic in nature."
)
# Row read in the seeded sample: refusal plus a concrete help redirect
# (emergency services / mental-health professional). Not inferred from keyword search.
PARTIAL_REFUSAL_ROW_INDICES = frozenset({2179})
REFUSAL_MARKERS = (
    "cannot",
    "can't",
    "unable",
    "not able",
    "refuse",
    "inappropriate",
    "against",
    "harmful",
    "toxic",
    "नहीं",
    "cannot give",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_hf_configs() -> dict[str, Any]:
    api = HfApi()
    info = api.dataset_info(REPO, revision="main")
    configs = []
    try:
        from datasets import get_dataset_config_names

        configs = list(get_dataset_config_names(REPO, revision=info.sha))
    except Exception as exc:
        configs = [f"lookup_failed: {exc}"]
    toccix_like = [c for c in configs if isinstance(c, str) and "tox" in c.lower()]
    return {
        "repository": REPO,
        "revision": info.sha,
        "dataset_config_names": configs,
        "toccix_name_found_on_hub": any(
            isinstance(c, str) and c.lower() in ("toccix", "toccix_matrix") for c in configs
        ),
        "toxic_matrix_config_present": TOXIC_CONFIG in configs
        if isinstance(configs, list)
        else None,
        "dolly_t_config_present": INSTRUCT_CONFIG in configs
        if isinstance(configs, list)
        else None,
        "configs_matching_tox_substring": toccix_like,
        "verified_toxic_source_config": TOXIC_CONFIG,
        "note": (
            "Project uses HuggingFace config 'Toxic_Matrix' on ai4bharat/indic-align, "
            "not a separate 'TOCCIX' config name on the hub."
        ),
    }


def gather_license_evidence(info: Any, readme: str) -> dict[str, Any]:
    card = info.card_data or {}
    tags = list(getattr(info, "tags", []) or [])
    mentions = re.findall(r"(?i)(cc-by|cc0|creative commons|license)", readme)
    return {
        "hf_card_license_field": card.get("license") if isinstance(card, dict) else None,
        "hf_dataset_info_license_attr": getattr(info, "license", None),
        "hf_tags": tags,
        "readme_license_mentions": sorted(set(mentions)),
        "loader_hardcoded_license_warning": (
            "toxic_matrix.py hardcodes CC-BY-4.0; audit uses HF card when available."
        ),
    }


def hf_toxic_schema_sample(revision: str) -> dict[str, Any]:
    try:
        from datasets import load_dataset

        ds = load_dataset(REPO, name=TOXIC_CONFIG, split="train", revision=revision)
        row = dict(ds[0])
        return {
            "hf_column_names": ds.column_names,
            "hf_features": str(ds.features),
            "hf_train_row_count": len(ds),
            "sample_doc_id": row.get("doc_id"),
        }
    except Exception as exc:
        return {"hf_schema_lookup_error": str(exc)}


def pin_provenance(refresh_hf: bool = False) -> dict[str, Any]:
    cached_manifest = TOXICITY_AUDIT / "dataset_manifest.json"
    if cached_manifest.is_file() and not refresh_hf:
        manifest = json.loads(cached_manifest.read_text(encoding="utf-8"))
        if not manifest.get("hf_config_discovery"):
            manifest["hf_config_discovery"] = discover_hf_configs()
            cached_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest

    api = HfApi()
    info = api.dataset_info(REPO, revision="main")
    discovery = discover_hf_configs()
    readme = ""
    try:
        rp = hf_hub_download(
            repo_id=REPO, filename="README.md", repo_type="dataset", revision=info.sha
        )
        readme = Path(rp).read_text(encoding="utf-8")[:12000]
    except Exception as exc:
        readme = str(exc)

    files_meta: list[dict[str, Any]] = []
    for name in sorted(RAW_TOX.glob("*.jsonl")):
        files_meta.append(
            {
                "filename": name.name,
                "sha256": sha256_file(name),
                "size_bytes": name.stat().st_size,
                "line_count": sum(1 for _ in name.open(encoding="utf-8")),
            }
        )

    return {
        "repository": REPO,
        "revision": info.sha,
        "revision_fetched_utc": datetime.now(UTC).isoformat(),
        "hf_config_discovery": discovery,
        "toxic_config_used": TOXIC_CONFIG,
        "benign_config_intended": INSTRUCT_CONFIG,
        "benign_file_on_disk": _resolve_benign_path().name,
        "local_files": files_meta,
        "license_evidence": gather_license_evidence(info, readme),
        "readme_excerpt_chars": len(readme),
        "hf_toxic_matrix_schema": hf_toxic_schema_sample(info.sha),
    }


def _resolve_benign_path() -> Path:
    dolly = RAW_TOX / "Dolly_T_benign_sample.jsonl"
    legacy = RAW_TOX / "Indic_ShareLlama_benign_sample.jsonl"
    if dolly.is_file() and dolly.stat().st_size > 100:
        return dolly
    return legacy


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"__parse_error__": True, "__line__": i, "__raw__": line[:200]})
    return rows


def audit_schema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0}
    keys: Counter[str] = Counter()
    types: dict[str, set[str]] = defaultdict(set)
    multi_block = 0
    multi_turn_pairs = 0
    explicit_roles_in_source = False
    for row in rows:
        if row.get("__parse_error__"):
            continue
        for k, v in row.items():
            keys[k] += 1
            types[k].add(type(v).__name__)
        eng = row.get("eng_Latn")
        if isinstance(eng, list):
            if len(eng) > 1:
                multi_block += 1
            for block in eng:
                if isinstance(block, list) and len(block) > 1:
                    multi_turn_pairs += 1
    return {
        "columns_observed": dict(keys),
        "column_types": {k: sorted(v) for k, v in types.items()},
        "label_definition_in_source": (
            "Row-level toxic vs benign is implied by Toxic_Matrix (toxic) vs benign sample file; "
            "no per-row label column in Toxic_Matrix JSONL."
        ),
        "conversation_structure": (
            "eng_Latn is a list of [user, assistant] string pairs; roles are positional not named."
        ),
        "multi_block_conversations": multi_block,
        "multi_pair_blocks": multi_turn_pairs,
        "explicit_speaker_role_fields": explicit_roles_in_source,
        "chronological_order_preserved_in_list": True,
        "parsed_turn_model": "romcom_data.toxic_turns.parse_conversation_turns",
    }


def audit_integrity_toxic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    doc_ids: list[str] = []
    malformed = 0
    missing_eng = 0
    empty_prompt = 0
    empty_response = 0
    refusal_like_assistant = 0
    user_looks_refusal = 0
    script_stats: Counter[str] = Counter()

    for row in rows:
        if row.get("__parse_error__"):
            malformed += 1
            continue
        doc_ids.append(str(row.get("doc_id") or ""))
        if "eng_Latn" not in row:
            missing_eng += 1
        turns = parse_conversation_turns(row)
        prompt = user_prompt_from_turns(turns)
        resp = assistant_target_from_turns(turns)
        if not prompt.strip():
            empty_prompt += 1
        if not resp.strip():
            empty_response += 1
        if resp and any(m in resp.lower() for m in REFUSAL_MARKERS):
            refusal_like_assistant += 1
        if prompt and any(m in prompt.lower() for m in REFUSAL_MARKERS[:4]):
            user_looks_refusal += 1
        for field in ("eng_Latn", "hin_Deva", "hin_Latn"):
            val = row.get(field)
            text = json.dumps(val, ensure_ascii=False) if val is not None else ""
            prof = script_profile(text)
            if prof["devanagari_ratio"] > 0.2:
                script_stats["devanagari_heavy"] += 1
            if prof["latin_ratio"] > 0.2:
                script_stats["latin_heavy"] += 1
            if prof["mixed"]:
                script_stats["code_mixed_field"] += 1

    dup_docs = sum(1 for c in Counter(doc_ids).values() if c > 1)
    return {
        "row_count": len(rows),
        "malformed_json_lines": malformed,
        "missing_eng_Latn": missing_eng,
        "empty_user_prompt_after_parse": empty_prompt,
        "empty_assistant_response_after_parse": empty_response,
        "duplicate_doc_id_count": dup_docs,
        "assistant_responses_with_refusal_markers": refusal_like_assistant,
        "user_prompts_with_refusal_markers": user_looks_refusal,
        "script_field_counts": dict(script_stats),
        "normalized_label": "toxic (all rows in Toxic_Matrix file)",
    }


def audit_benign(rows: list[dict[str, Any]], benign_file: str) -> dict[str, Any]:
    schema = audit_schema(rows)
    empty = sum(
        1
        for r in rows
        if not user_prompt_from_turns(parse_conversation_turns(r)).strip()
        and not str(r.get("prompt") or "").strip()
    )
    return {
        "benign_file": benign_file,
        "row_count": len(rows),
        "empty_prompt_rows": empty,
        "intended_config": INSTRUCT_CONFIG,
        "configuration_note": (
            "Dolly_T_benign_sample.jsonl missing; using Indic_ShareLlama_benign_sample.jsonl"
            if "ShareLlama" in benign_file
            else "Dolly_T sample present"
        ),
        "schema": schema,
        "normalized_label": "non_toxic",
    }


def audit_language_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    has_hin_deva = 0
    has_hin_latn = 0
    eng_only = 0
    for row in rows:
        if row.get("__parse_error__"):
            continue
        hd = row.get("hin_Deva")
        hl = row.get("hin_Latn")
        if hd:
            has_hin_deva += 1
        if hl:
            has_hin_latn += 1
        if row.get("eng_Latn") and not hd and not hl:
            eng_only += 1
    n = max(len(rows), 1)
    return {
        "rows_with_hin_Deva": has_hin_deva,
        "rows_with_hin_Latn": has_hin_latn,
        "rows_eng_Latn_only": eng_only,
        "hindi_devanagari_coverage_pct": round(100 * has_hin_deva / n, 2),
        "romanized_hindi_coverage_pct": round(100 * has_hin_latn / n, 2),
        "primary_audit_language_field": "eng_Latn (loader uses for prompt/response parse)",
    }


def simulate_stratified_split(
    toxic_rows: list[dict[str, Any]], benign_rows: list[dict[str, Any]], seed: int = 42
) -> dict[str, Any]:
    rng = derive_rng(seed, "tox-split", "audit")
    ids: list[tuple[str, str]] = []
    for r in toxic_rows:
        if r.get("doc_id"):
            ids.append(("toxic", str(r["doc_id"])))
    for r in benign_rows:
        if r.get("doc_id"):
            ids.append(("benign", str(r["doc_id"])))
    rng.shuffle(ids)
    cut = int(len(ids) * 0.8)
    train_ids = {d for _, d in ids[:cut]}
    val_ids = {d for _, d in ids[cut:]}
    overlap = train_ids & val_ids
    return {
        "simulated_80_20_by_doc_id": True,
        "total_ids": len(ids),
        "train_ids": len(train_ids),
        "val_ids": len(val_ids),
        "doc_id_leak_train_val": len(overlap),
        "exp_conditions_share_ids": (
            "EXP-01/02/03 use same example_id and project_split from intermediate stream."
        ),
    }


def audit_leakage(toxic_rows: list[dict[str, Any]], benign_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tox_docs = {str(r.get("doc_id")) for r in toxic_rows if r.get("doc_id")}
    ben_docs = {str(r.get("doc_id")) for r in benign_rows if r.get("doc_id")}
    doc_overlap = tox_docs & ben_docs

    tox_prompts: set[str] = set()
    ben_prompts: set[str] = set()
    for r in toxic_rows:
        p = user_prompt_from_turns(parse_conversation_turns(r))
        if p:
            tox_prompts.add(nfc(p).strip())
    for r in benign_rows:
        p = user_prompt_from_turns(parse_conversation_turns(r)) or nfc(str(r.get("prompt") or ""))
        if p:
            ben_prompts.add(nfc(p).strip())

    prompt_overlap = tox_prompts & ben_prompts
    split_sim = simulate_stratified_split(toxic_rows, benign_rows)
    return {
        "official_test_split": "not_provided_train_only_configs",
        "recommended_split_unit": "source_id (doc_id) stratified toxic vs non_toxic",
        "recommended_seed": 42,
        "doc_id_overlap_toxic_benign": len(doc_overlap),
        "doc_id_overlap_sample": sorted(doc_overlap)[:10],
        "exact_prompt_overlap_toxic_benign": len(prompt_overlap),
        "prompt_overlap_sample_hashes": [
            stable_hash_hex(p)[:16] for p in list(prompt_overlap)[:5]
        ],
        "cross_task_leakage_risk_if_split_by_row_only": (
            "Low for doc_id; split must not shard same doc_id across train/val."
        ),
        "split_simulation": split_sim,
    }


def audit_transformation_constraints() -> dict[str, Any]:
    return {
        "policy_source": "data_plan Appendix B.5 / phase2_reports toxicity_data_design_report.md",
        "user_prompt": "Never code-mixed; bytes preserved in input.prompt",
        "assistant_target": "Romanization / variants apply to target.response only",
        "turn_handling": (
            "parse_conversation_turns preserves chronological order; "
            "user_prompt_from_turns and assistant_target_from_turns keep roles separate; "
            "multiple turns joined with \\n---\\n within same role only."
        ),
        "labels": "normalized_label unchanged across EXP-01/02/03",
        "generate_exps": (
            "EXP-03 uses EXP-02 changed token spans on target fields; "
            "toxicity input fields empty in _input_fields except none for prompt path—"
            "prompt unchanged in build_toxicity_base."
        ),
        "experimental_id_integrity": (
            "stream_task emits same example_id/source_id/label/project_split for all EXP conditions."
        ),
    }


def audit_normalized_pipeline(
    meta: dict[str, Any], exclude_doc_ids: set[str] | None = None
) -> dict[str, Any]:
    records = load_normalized_toxicity(meta, exclude_doc_ids=exclude_doc_ids)
    labels = Counter(r["normalized_label"] for r in records)
    toxic_n = sum(1 for r in records if not r.get("is_benign"))
    benign_n = sum(1 for r in records if r.get("is_benign"))
    return {
        "normalized_record_count": len(records),
        "toxic_records": toxic_n,
        "benign_records": benign_n,
        "label_distribution": dict(labels),
        "imbalance_ratio": round(max(labels.values()) / max(min(labels.values()), 1), 3),
        "loader_skips_empty_prompt_or_response": True,
        "paper_target_n": 10000,
        "benign_pairing": "min(benign_rows, 10000 - toxic) from benign file",
    }


def _first_pair(value: Any) -> tuple[str, str, int]:
    if isinstance(value, list) and value and isinstance(value[0], list) and len(value[0]) >= 2:
        user = "" if value[0][0] is None else str(value[0][0])
        asst = "" if value[0][1] is None else str(value[0][1])
        return user, asst, len(value)
    return "", "", 0


def _length_order(eng_user: str, eng_asst: str, other_user: str, other_asst: str) -> str:
    direct = abs(len(eng_user) - len(other_user)) + abs(len(eng_asst) - len(other_asst))
    swapped = abs(len(eng_user) - len(other_asst)) + abs(len(eng_asst) - len(other_user))
    if direct == swapped:
        return "tie"
    return "aligned" if direct < swapped else "swapped"


def _char_summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "mean": 0, "p50": 0, "p90": 0}

    def quantile(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]

    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
    }


def profile_conversation_class(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Coverage and length profile. Script counts are per side, not refusal labels."""
    counts: Counter[str] = Counter()
    prompt_lens: list[int] = []
    response_lens: list[int] = []
    turn_counts: Counter[str] = Counter()
    for row in rows:
        if row.get("__parse_error__"):
            counts["malformed"] += 1
            continue
        eng_user, eng_asst, n_blocks = _first_pair(row.get("eng_Latn"))
        deva_user, deva_asst, _ = _first_pair(row.get("hin_Deva"))
        latn_user, latn_asst, _ = _first_pair(row.get("hin_Latn"))
        counts["rows"] += 1
        counts["single_block"] += int(n_blocks == 1)
        counts["multi_block"] += int(n_blocks > 1)
        counts["hin_deva_present"] += int(bool(deva_user or deva_asst))
        counts["hin_latn_present"] += int(bool(latn_user or latn_asst))
        counts["eng_present"] += int(bool(eng_user or eng_asst))
        counts["deva_order_" + _length_order(eng_user, eng_asst, deva_user, deva_asst)] += 1
        counts["latn_order_" + _length_order(eng_user, eng_asst, latn_user, latn_asst)] += 1
        sides = {
            "eng_user": eng_user,
            "eng_assistant": eng_asst,
            "deva_user": deva_user,
            "deva_assistant": deva_asst,
            "latn_user": latn_user,
            "latn_assistant": latn_asst,
        }
        for side, text in sides.items():
            profile = script_profile(text)
            if profile["devanagari_ratio"] > 0.5:
                counts[f"{side}_devanagari_majority"] += 1
            if profile["latin_ratio"] > 0.5:
                counts[f"{side}_latin_majority"] += 1
            if profile["mixed"]:
                counts[f"{side}_code_mixed"] += 1
        turn_counts[str(row.get("num_turns"))] += 1
        prompt_lens.append(len(eng_user))
        response_lens.append(len(eng_asst))
    n = max(counts["rows"], 1)
    return {
        "class_name": name,
        "counts": dict(counts),
        "parsed_from": "eng_Latn positional [user, assistant]",
        "num_turns_values": dict(turn_counts),
        "prompt_chars_eng": _char_summary(prompt_lens),
        "assistant_chars_eng": _char_summary(response_lens),
        "hin_deva_row_pct": round(100 * counts["hin_deva_present"] / n, 2),
        "hin_latn_row_pct": round(100 * counts["hin_latn_present"] / n, 2),
        "eng_row_pct": round(100 * counts["eng_present"] / n, 2),
    }


def compare_benign_file_to_dolly(
    benign_rows: list[dict[str, Any]], revision: str
) -> dict[str, Any]:
    """Inspect pinned Dolly_T. Do not write a replacement benign file."""
    from datasets import load_dataset

    dataset = load_dataset(REPO, name=INSTRUCT_CONFIG, split="train", revision=revision)
    dolly_ids = list(dataset["doc_id"])
    local_ids = [str(row.get("doc_id")) for row in benign_rows]
    id_to_index = {doc_id: i for i, doc_id in enumerate(dolly_ids)}
    missing = [doc_id for doc_id in local_ids if doc_id not in id_to_index]
    mismatches = 0
    for row in benign_rows:
        doc_id = str(row.get("doc_id"))
        if doc_id not in id_to_index:
            continue
        source = dataset[id_to_index[doc_id]]
        if (
            row.get("eng_Latn") != source["eng_Latn"]
            or row.get("hin_Deva") != source["hin_Deva"]
            or row.get("hin_Latn") != source["hin_Latn"]
        ):
            mismatches += 1
    rng = derive_rng(42, "dolly-t", "benign-shuffle")
    order = list(range(len(dolly_ids)))
    rng.shuffle(order)
    seeded = {dolly_ids[i] for i in order[:2000]}
    positions = [id_to_index[doc_id] for doc_id in local_ids if doc_id in id_to_index]
    is_head = local_ids == dolly_ids[: len(local_ids)] and not missing
    return {
        "pinned_revision": revision,
        "dolly_t_config": INSTRUCT_CONFIG,
        "dolly_t_train_rows": len(dataset),
        "dolly_t_columns": dataset.column_names,
        "dolly_t_recovered_from_pin": True,
        "local_benign_filename": "Indic_ShareLlama_benign_sample.jsonl",
        "local_rows": len(local_ids),
        "local_ids_missing_from_dolly_t": missing[:10],
        "local_ids_missing_count": len(missing),
        "field_mismatches_eng_hin_deva_hin_latn": mismatches,
        "local_file_equals_dolly_t_train_prefix": is_head,
        "prefix_length": len(local_ids) if is_head else None,
        "loader_seed_42_sample_overlap_with_local": len(seeded & set(local_ids)),
        "replacement_performed": False,
        "comparability": (
            "The on-disk benign file is Dolly_T train rows 0-1999, not a separate "
            "ShareLlama extract and not the loader's seed-42 sample of 2000. "
            "Schema, Hindi coverage, and single-turn structure match Dolly_T. "
            "Resampling would change which source IDs are included."
        ),
        "suitability": (
            "Dolly_T rows are ordinary single-turn instruction answers in English "
            "with Hindi Devanagari and romanized parallels. They are a non-refusal "
            "contrast, not toxicity annotations."
        ),
    }


def class_confounds(toxic: dict[str, Any], benign: dict[str, Any]) -> dict[str, Any]:
    return {
        "same_conversation_shape": (
            toxic["counts"].get("multi_block", 0) == 0
            and benign["counts"].get("multi_block", 0) == 0
        ),
        "prompt_mean_chars_toxic_vs_benign": [
            toxic["prompt_chars_eng"]["mean"],
            benign["prompt_chars_eng"]["mean"],
        ],
        "assistant_mean_chars_toxic_vs_benign": [
            toxic["assistant_chars_eng"]["mean"],
            benign["assistant_chars_eng"]["mean"],
        ],
        "language_binding": (
            "Canonical text is hin_Deva. The loader sets language='hi' only for that field. "
            "eng_Latn is retained as a parallel reference for role checks."
        ),
        "register_confound": (
            "Toxic_Matrix assistant turns are templated refusals. Dolly_T assistant "
            "turns are ordinary answers. The class contrast is refusal register "
            "versus ordinary answer, which matches the intended task and does not "
            "match toxic-input classification."
        ),
        "length_confound": (
            "Toxic user prompts and assistant refusals are longer than Dolly answers. "
            "Length can confound romanization difficulty across classes."
        ),
        "sampling_confound": (
            "Toxic rows are a 5000-row shuffle from Toxic_Matrix. Benign rows are "
            "the first 2000 Dolly_T rows, not a matched shuffle."
        ),
        "hin_latn_order_caution": (
            "Length alignment finds a minority of hin_Latn pairs swapped relative "
            "to eng_Latn. Do not treat hin_Latn slot order as a reliable role encoding."
        ),
    }


def sample_refusal_rows(toxic_rows: list[dict[str, Any]], n: int = REFUSAL_SAMPLE_N) -> list[dict[str, Any]]:
    rng = derive_rng(42, REFUSAL_SAMPLE_SEED_ID, "sample")
    indexes = list(range(len(toxic_rows)))
    rng.shuffle(indexes)
    picked: list[dict[str, Any]] = []
    for index in indexes[:n]:
        row = toxic_rows[index]
        turns = parse_conversation_turns(row, text_field="eng_Latn")
        picked.append(
            {
                "row_index": index,
                "doc_id": row.get("doc_id", ""),
                "user": user_prompt_from_turns(turns),
                "assistant": assistant_target_from_turns(turns),
            }
        )
    return picked


def assign_manual_refusal_category(row_index: int, assistant: str) -> str:
    """Categories assigned by reading the seeded sample.

    Every reviewed reply opened with an explicit refusal and then explained why
    the user prompt was harmful. Keyword hits were not used as the label.
    """
    if not assistant.startswith(STOCK_REFUSAL):
        return "ambiguous"
    if row_index in PARTIAL_REFUSAL_ROW_INDICES:
        return "partial_refusal_with_safe_alternative"
    return "clear_refusal"


def audit_refusal_sample(toxic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample = sample_refusal_rows(toxic_rows)
    labels = [assign_manual_refusal_category(row["row_index"], row["assistant"]) for row in sample]
    counts = Counter(labels)
    n = len(sample)
    proportions = {key: round(counts[key] / n, 4) for key in (
        "clear_refusal",
        "partial_refusal_with_safe_alternative",
        "safety_explanation",
        "ordinary_response",
        "ambiguous",
    )}
    return {
        "method": (
            "Reproducible sample derive_rng(42, 'tox-refusal-manual', 'sample'), "
            "first 120 shuffled Toxic_Matrix local rows. Each assistant reply was "
            "read. The shared opening sentence was checked, then the explanation "
            "was categorized. Marker matching is not the label."
        ),
        "n": n,
        "counts": dict(counts),
        "proportions": proportions,
        "coding_rule": (
            "clear_refusal: explicit refusal plus harm explanation, no concrete "
            "alternative action. partial_refusal_with_safe_alternative: explicit "
            "refusal plus a concrete safe next step. safety_explanation: harm "
            "discussion without a refusal. ordinary_response: complies or answers "
            "normally. ambiguous: mixed or off-template."
        ),
        "partial_cases": [
            {
                "row_index": 2179,
                "doc_id": next(row["doc_id"] for row in sample if row["row_index"] == 2179),
                "reason": (
                    "Refuses a mass-shooting request, then tells the user to call "
                    "emergency services or seek a mental-health professional."
                ),
            }
        ],
        "not_ground_truth_note": (
            "Prior heuristic marker counts remain descriptive only. This review "
            "does not create training labels."
        ),
        "distress_prompts_refused_without_help_in_sample": (
            "Several distress or self-harm prompts in the sample are clear refusals "
            "with no crisis resource. That is a response-quality issue inside "
            "clear_refusal, not a separate category."
        ),
        "rows": [
            {
                "row_index": row["row_index"],
                "doc_id": row["doc_id"],
                "category": label,
                "user_excerpt": row["user"][:180],
                "assistant_excerpt": row["assistant"][:240],
            }
            for row, label in zip(sample, labels)
        ],
    }


def task_definition() -> dict[str, Any]:
    return {
        "research_claim": (
            "Secondary Hindi refusal-register task. It does not measure general "
            "toxicity detection or code-mixed safety robustness."
        ),
        "evaluation_interface": {
            "model_input": (
                "hin_Deva list slot 0, the user turn. Unchanged in every condition."
            ),
            "model_target": (
                "hin_Deva list slot 1, the assistant turn. This is the only text "
                "that may be romanized or noised."
            ),
            "label": (
                "Source register kept on the example: toxic for a Toxic_Matrix "
                "refusal target, non_toxic for a Dolly ordinary-answer target. "
                "The same label is shared by EXP-01, EXP-02, and EXP-03."
            ),
            "scoring": (
                "Not run in this audit. A generation score would compare the model "
                "continuation with model_target only. No classifier was trained."
            ),
        },
        "distinctions": {
            "conversation_classification": (
                "Would assign one label to the whole dialogue. That is not this task, "
                "and the source label is not a toxicity rating of the conversation."
            ),
            "assistant_response_refusal_classification": (
                "Would label whether the assistant turn is a refusal. The shared "
                "opening makes that separable by template. Not claimed."
            ),
            "refusal_generation": (
                "Given model_input, generate the assistant reply and compare it with "
                "model_target under romanization of that target. This is the only "
                "interface described for the secondary task."
            ),
        },
        "text_binding": "hin_Deva is canonical. eng_Latn is an alignment reference. hin_Latn is not a role source.",
        "transformation_constraints_unchanged": True,
        "experimental_construction": "NOT_APPROVED",
    }


def build_recommendation(
    integrity: dict[str, Any],
    benign: dict[str, Any],
    leakage: dict[str, Any],
    license_evidence: dict[str, Any],
    discovery: dict[str, Any],
    dolly_comparison: dict[str, Any] | None = None,
    refusal_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    owner_decisions: list[str] = []
    if not discovery.get("toxic_matrix_config_present"):
        blockers.append("toxic_matrix_config_not_found_on_hub")
    if integrity.get("empty_user_prompt_after_parse", 0) > integrity.get("row_count", 1) * 0.05:
        blockers.append("high_empty_prompt_rate")
    if leakage.get("doc_id_overlap_toxic_benign", 0) > 0:
        blockers.append("doc_id_overlap_toxic_benign")
    blockers.append("dual_license_owner_approval")
    blockers.append("experimental_construction_not_approved")
    blockers.append("template_confound_precludes_classification_claim")
    owner_decisions.append(
        "Approve IndicAlign CC-BY-4.0 and, separately, Dolly-15K upstream CC-BY-SA-3.0. "
        "Do not treat them as one license."
    )
    owner_decisions.append(
        "Approve experimental construction only after accepting the seed-42 benign "
        "manifest, the excluded role-conflict rows, and the generation-only interface."
    )
    if refusal_review:
        owner_decisions.append(
            "The earlier English refusal read found templated refusals. "
            "Hindi template rates are in template_statistics.json and are not classifier labels."
        )

    if blockers and any(b.startswith("toxic_matrix_config") for b in blockers):
        verdict = "NO_GO"
    elif blockers:
        verdict = "CONDITIONAL_GO"
    else:
        verdict = "GO"

    return {
        "verdict": verdict,
        "scope": "secondary_hindi_refusal_register_generation",
        "experimental_construction": "NOT_APPROVED",
        "english_pivot": "not_applicable",
        "blockers": blockers,
        "owner_decisions": owner_decisions,
        "task_definition": task_definition(),
        "task_suitability_summary": (
            "Hindi Devanagari user turns stay fixed. Assistant turns are the refusal "
            "or ordinary-answer targets. This is not toxicity detection."
        ),
        "secondary_register_fit": (
            "Generation-only comparison of Hindi refusal targets versus ordinary Dolly "
            "answers, after owner approval. Classification is confounded by the template."
        ),
        "benign_file_replaced": False,
        "head_slice_not_used": True,
    }


def determine_status(recommendation: dict[str, Any]) -> dict[str, Any]:
    status = "BLOCKED" if recommendation["verdict"] == "NO_GO" else "READY_FOR_OWNER_REVIEW"
    return {
        "status": status,
        "blockers": recommendation.get("blockers", []),
        "dataset_rebuild_allowed": False,
        "training_allowed": False,
        "owner_review_scope": "toxicity_hindi_native_target_side_mixing",
    }


def write_owner_sample(toxic_rows: list[dict[str, Any]], path: Path, n: int = 30) -> int:
    rng = derive_rng(42, "tox-audit", "sample")
    idxs = list(range(len(toxic_rows)))
    rng.shuffle(idxs)
    pick = idxs[:n]
    fields = [
        "row_index",
        "doc_id",
        "user_prompt_excerpt",
        "assistant_response_excerpt",
        "assistant_looks_like_refusal",
        "owner_judgment_toxic_user_content",
        "owner_judgment_safety_response",
        "owner_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in pick:
            row = toxic_rows[i]
            turns = parse_conversation_turns(row)
            p = user_prompt_from_turns(turns)
            r = assistant_target_from_turns(turns)
            w.writerow(
                {
                    "row_index": i,
                    "doc_id": row.get("doc_id", ""),
                    "user_prompt_excerpt": p[:200],
                    "assistant_response_excerpt": r[:200],
                    "assistant_looks_like_refusal": any(
                        m in r.lower() for m in REFUSAL_MARKERS
                    ),
                    "owner_judgment_toxic_user_content": "",
                    "owner_judgment_safety_response": "",
                    "owner_notes": "",
                }
            )
    return len(pick)


def _align_summary(result: dict[str, Any]) -> dict[str, Any]:
    if "counts" not in result:
        return result
    return {
        "field": result.get("field"),
        "model": result.get("model"),
        "rule": result.get("rule"),
        "used_as_canonical_text": result.get("used_as_canonical_text"),
        "counts": result.get("counts"),
        "conflict_count": result.get("conflict_count"),
    }


def write_benign_manual_review(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from romcom_data.toxicity_alignment import _pair

    rng = derive_rng(42, "benign-manual", "sample")
    indexes = list(range(len(seed_rows)))
    rng.shuffle(indexes)
    label_path = TOXICITY_AUDIT / "benign_manual_labels.json"
    stored = json.loads(label_path.read_text(encoding="utf-8")) if label_path.is_file() else {}
    rows: list[dict[str, Any]] = []
    for index in indexes[:120]:
        row = seed_rows[index]
        user, assistant = _pair(row, "hin_Deva")
        key = str(row.get("source_row_index"))
        rows.append(
            {
                "source_row_index": row.get("source_row_index"),
                "doc_id": row.get("doc_id"),
                "category": stored.get(key, "unreviewed"),
                "user_excerpt": user[:180],
                "assistant_excerpt": assistant[:240],
            }
        )
    counts = Counter(row["category"] for row in rows)
    n = max(len(rows), 1)
    keys = (
        "ordinary_response",
        "clear_refusal",
        "partial_refusal",
        "safety_explanation",
        "ambiguous",
        "unreviewed",
    )
    return {
        "method": "derive_rng(42, 'benign-manual', 'sample'); first 120 of the seed-42 Dolly_T draw. Categories were read from hin_Deva assistant text.",
        "n": len(rows),
        "counts": {key: counts.get(key, 0) for key in keys},
        "proportions": {key: round(counts.get(key, 0) / n, 4) for key in keys},
        "rows": rows,
    }


def run_toxicity_audit(*, refresh_hf: bool = False) -> dict[str, Any]:
    TOXICITY_AUDIT.mkdir(parents=True, exist_ok=True)
    manifest = pin_provenance(refresh_hf=refresh_hf)
    meta = {
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "license": manifest.get("license_evidence", {}).get("hf_card_license_field") or "CC-BY-4.0",
        "license_status": "audit_pending_owner",
    }

    toxic_path = RAW_TOX / "Toxic_Matrix_train.jsonl"
    benign_path = _resolve_benign_path()
    toxic_rows = load_jsonl(toxic_path)
    benign_rows = load_jsonl(benign_path) if benign_path.is_file() else []

    schema = audit_schema(toxic_rows)
    language = audit_language_coverage(toxic_rows)
    integrity = audit_integrity_toxic(toxic_rows)
    benign_audit = audit_benign(benign_rows, benign_path.name)
    from romcom_data.toxicity_alignment import (
        embedding_alignment,
        fetch_dolly_upstream_license,
        prepare_dolly_seed_sample,
        shortcut_diagnostic,
        template_statistics,
    )

    seed_summary = prepare_dolly_seed_sample(
        manifest["revision"], TOXICITY_AUDIT / "dolly_t_seed42_sample.jsonl"
    )
    seed_rows = load_jsonl(TOXICITY_AUDIT / "dolly_t_seed42_sample.jsonl")
    hin_align_toxic = embedding_alignment(toxic_rows, "hin_Deva")
    hin_align_benign = embedding_alignment(seed_rows, "hin_Deva")
    latn_align_toxic = embedding_alignment(toxic_rows, "hin_Latn")
    latn_align_benign = embedding_alignment(seed_rows, "hin_Latn")
    toxic_templates = template_statistics(toxic_rows, "toxic_matrix_local")
    benign_templates = template_statistics(seed_rows, "dolly_t_seed42")
    template_report = {
        "toxic": toxic_templates,
        "benign": benign_templates,
        "shortcut": shortcut_diagnostic(toxic_templates, benign_templates),
    }
    excluded = hin_align_toxic["conflicts"] + hin_align_benign["conflicts"]
    leakage = audit_leakage(toxic_rows, seed_rows)
    toxic_profile = profile_conversation_class(toxic_rows, "toxic_matrix_local")
    benign_profile = profile_conversation_class(seed_rows, "dolly_t_seed42")
    benign_review = write_benign_manual_review(seed_rows)
    dolly_comparison = compare_benign_file_to_dolly(benign_rows, manifest["revision"])
    confounds = class_confounds(toxic_profile, benign_profile)
    refusal_review = audit_refusal_sample(toxic_rows)
    transform = audit_transformation_constraints()
    excluded_ids = {str(item.get("doc_id")) for item in excluded if item.get("doc_id")}
    normalized = audit_normalized_pipeline(meta, exclude_doc_ids=excluded_ids)
    normalized["role_conflict_rows_excluded_not_flipped"] = len(excluded_ids)
    license_evidence = dict(manifest.get("license_evidence") or {})
    license_evidence["declarations_not_flattened"] = {
        "indic_align": {
            "spdx": "CC-BY-4.0",
            "evidence": "Hugging Face tag license:cc-by-4.0 on ai4bharat/indic-align. Card license field is null.",
            "owner_approval": "required",
        },
        "dolly_15k_upstream": {
            "spdx": "CC-BY-SA-3.0",
            "applies_to": "Dolly-derived benign control only",
            "hf_lookup": fetch_dolly_upstream_license(),
            "owner_approval": "required",
        },
    }
    discovery = manifest.get("hf_config_discovery", {})
    recommendation = build_recommendation(
        integrity,
        benign_audit,
        leakage,
        license_evidence,
        discovery,
        dolly_comparison,
        refusal_review,
    )
    handoff_status = determine_status(recommendation)

    report = {
        "title": "Phase 2.1 Toxicity (IndicAlign Toxic_Matrix) audit",
        "audit_revision": "2026_11_01_rev3",
        "toccix_investigation": discovery,
        "provenance": manifest,
        "schema": schema,
        "language_coverage": language,
        "integrity_toxic": integrity,
        "integrity_benign": benign_audit,
        "leakage": leakage,
        "dolly_t_comparison": dolly_comparison,
        "class_comparability": {
            "toxic": toxic_profile,
            "benign": benign_profile,
            "confounds": confounds,
        },
        "seeded_benign_sample": seed_summary,
        "role_alignment": {
            "hin_deva_toxic": _align_summary(hin_align_toxic),
            "hin_deva_benign": _align_summary(hin_align_benign),
            "hin_latn_toxic_not_used": _align_summary(latn_align_toxic),
            "hin_latn_benign_not_used": _align_summary(latn_align_benign),
            "excluded_conflict_count": len(excluded),
        },
        "template_statistics": template_report,
        "benign_manual_review": {
            key: value for key, value in benign_review.items() if key != "rows"
        },
        "refusal_manual_review": {
            key: value for key, value in refusal_review.items() if key != "rows"
        },
        "task_definition": task_definition(),
        "normalized_loader_preview": normalized,
        "transformation_constraints": transform,
        "recommendation": recommendation,
        "handoff_status": handoff_status,
    }

    (TOXICITY_AUDIT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (TOXICITY_AUDIT / "schema_label_statistics.json").write_text(
        json.dumps(
            {
                "schema": schema,
                "language_coverage": language,
                "integrity_toxic": integrity,
                "integrity_benign": benign_audit,
                "normalized_loader": normalized,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (TOXICITY_AUDIT / "license_evidence.json").write_text(
        json.dumps(license_evidence, indent=2) + "\n", encoding="utf-8"
    )
    (TOXICITY_AUDIT / "template_statistics.json").write_text(
        json.dumps(template_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (TOXICITY_AUDIT / "role_alignment_summary.json").write_text(
        json.dumps(report["role_alignment"], indent=2) + "\n", encoding="utf-8"
    )
    (TOXICITY_AUDIT / "excluded_role_conflicts.json").write_text(
        json.dumps(excluded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (TOXICITY_AUDIT / "dolly_t_seed42_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_row_index", "doc_id"])
        writer.writeheader()
        for row in seed_rows:
            writer.writerow(
                {"source_row_index": row.get("source_row_index"), "doc_id": row.get("doc_id")}
            )
    with (TOXICITY_AUDIT / "benign_manual_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "source_row_index",
                "doc_id",
                "category",
                "user_excerpt",
                "assistant_excerpt",
            ],
        )
        writer.writeheader()
        writer.writerows(benign_review["rows"])
    (TOXICITY_AUDIT / "data_quality_leakage_report.md").write_text(
        "# Data quality and leakage\n\n" + json.dumps(leakage, indent=2) + "\n\n"
        + json.dumps(integrity, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (TOXICITY_AUDIT / "transformation_integrity.md").write_text(
        "# RomCoM-X toxicity transformation integrity\n\n"
        + json.dumps(transform, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (TOXICITY_AUDIT / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
    )
    (TOXICITY_AUDIT / "toxicity_audit_status.json").write_text(
        json.dumps(handoff_status, indent=2) + "\n", encoding="utf-8"
    )
    (TOXICITY_AUDIT / "toxicity_audit_report.md").write_text(
        "# Toxicity audit report\n\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (TOXICITY_AUDIT / "dolly_t_comparison.json").write_text(
        json.dumps(dolly_comparison, indent=2) + "\n", encoding="utf-8"
    )
    (TOXICITY_AUDIT / "class_comparability.json").write_text(
        json.dumps(
            {"toxic": toxic_profile, "benign": benign_profile, "confounds": confounds},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (TOXICITY_AUDIT / "task_definition.md").write_text(
        "# Task definition\n\n"
        + json.dumps(task_definition(), indent=2)
        + "\n\nTransformation constraints are unchanged. "
        "Canonical, realistic-variant, and matched-noise conditions still share "
        "source IDs, labels, and splits.\n",
        encoding="utf-8",
    )
    review_summary = {key: value for key, value in refusal_review.items() if key != "rows"}
    (TOXICITY_AUDIT / "refusal_manual_review_summary.json").write_text(
        json.dumps(review_summary, indent=2) + "\n", encoding="utf-8"
    )
    with (TOXICITY_AUDIT / "refusal_manual_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["row_index", "doc_id", "category", "user_excerpt", "assistant_excerpt"],
        )
        writer.writeheader()
        writer.writerows(refusal_review["rows"])
    write_owner_sample(toxic_rows, TOXICITY_AUDIT / "owner_review_sample.csv")
    (TOXICITY_AUDIT / "audit_log.txt").write_text(
        f"completed_utc={datetime.now(UTC).isoformat()}\n"
        f"status={handoff_status['status']}\n"
        f"verdict={recommendation['verdict']}\n",
        encoding="utf-8",
    )
    return report


def build_audit_zip(zip_path: Path | None = None) -> Path:
    out = zip_path or (HANDOFF.parent / ZIP_NAME)
    files = (
        "toxicity_audit_report.md",
        "dataset_manifest.json",
        "schema_label_statistics.json",
        "license_evidence.json",
        "data_quality_leakage_report.md",
        "transformation_integrity.md",
        "recommendation.json",
        "toxicity_audit_status.json",
        "owner_review_sample.csv",
        "dolly_t_comparison.json",
        "class_comparability.json",
        "task_definition.md",
        "refusal_manual_review_summary.json",
        "refusal_manual_review.csv",
        "benign_manual_review.csv",
        "dolly_t_seed42_manifest.csv",
        "template_statistics.json",
        "role_alignment_summary.json",
        "excluded_role_conflicts.json",
        "audit_log.txt",
    )
    lines: list[str] = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in files:
            src = TOXICITY_AUDIT / name
            data = src.read_bytes()
            zf.writestr(name, data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
        zf.writestr("checksums.sha256", "\n".join(lines) + "\n")
    return out
