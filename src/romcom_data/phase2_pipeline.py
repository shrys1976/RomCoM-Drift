from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from romcom_data.base_dataset import (
    build_nli_base,
    build_paraphrase_base,
    build_toxicity_base,
)
from romcom_data.checksums import sha256_file, write_checksum_manifest
from romcom_data.comi_lingua import audit_comi
from romcom_data.config_loader import load_augmentation_config
from romcom_data.dakshina import full_dakshina_audit, save_audit_json
from romcom_data.generate_exps import generate_one
from romcom_data.handoff_paths import (
    EXP01,
    EXP02,
    EXP03,
    HANDOFF,
    INTERMEDIATE,
    INVENTORIES,
    MANIFESTS,
    RAW_COMI,
    RAW_DAK,
    REPORTS,
    TRACES,
    ensure_handoff_layout,
)
from romcom_data.inventory import build_variant_inventory
from romcom_data.inventory_export import export_phase2_inventories
from romcom_data.io_jsonl import write_jsonl
from romcom_data.paths import INTERIM, PROJECT_ROOT, RAW_DAKSHINA_HI, RAW_COMI as SRC_COMI
from romcom_data.phase2_reports import write_phase2_reports
from romcom_data.romanize import DakshinaRomanizer
from romcom_data.source_discovery import save_json
from romcom_data.source_loaders.indicxnli import acquire_indicxnli, load_normalized_nli
from romcom_data.source_loaders.indicxparaphrase import (
    acquire_indicxparaphrase,
    load_normalized_paraphrase,
)
from romcom_data.source_loaders.toxic_matrix import acquire_toxic_matrix, load_normalized_toxicity
from romcom_data.source_loaders.xnli_english import attach_xnli_english
from romcom_data.splitting import assign_splits
from romcom_data.stable_hash import derive_rng
from romcom_data.text_utils import tokenize_conservative


def _copy_raw() -> None:
    for src in RAW_DAKSHINA_HI.glob("*"):
        shutil.copy2(src, RAW_DAK / src.name)
    if SRC_COMI.is_dir():
        for src in SRC_COMI.glob("*"):
            shutil.copy2(src, RAW_COMI / src.name)


def _write_split_manifest(rows: Iterable[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "source_id",
        "example_id",
        "task",
        "label",
        "language",
        "original_split",
        "project_split",
        "mixing_intensity",
        "source_hash",
        "group_hash",
        "exclusion_status",
        "exclusion_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "source_id": row.get("source_id"),
                    "example_id": row.get("example_id"),
                    "task": row.get("task"),
                    "label": row.get("label") or row.get("normalized_label"),
                    "language": row.get("language", "hi"),
                    "original_split": row.get("original_split"),
                    "project_split": row.get("project_split"),
                    "mixing_intensity": row.get("mixing_intensity", ""),
                    "source_hash": row.get("source_row_hash", ""),
                    "group_hash": row.get("group_id", ""),
                    "exclusion_status": "kept",
                    "exclusion_reason": "",
                }
            )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _processed_view(row: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "example_id",
        "source_id",
        "task",
        "label",
        "language",
        "project_split",
        "original_split",
        "mixing_intensity",
        "condition",
        "input",
        "target",
        "source_dataset",
        "source_revision",
        "license_ref",
        "reconstruction_status",
        "source_row_hash",
        "augmentation_side",
        "p_variant",
    }
    return {k: row[k] for k in keep if k in row}


def stream_task(
    bases: list[dict[str, Any]],
    romanizer: DakshinaRomanizer,
    p_variant: float,
    writers: dict[str, dict[str, Path]],
    trace_paths: dict[str, Path],
    split_rows: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = Counter()
    for i, base in enumerate(bases):
        if i and i % 1000 == 0:
            print(f"  {base.get('task')} {i}/{len(bases)}", flush=True)
        conds = generate_one(base, romanizer, p_variant=p_variant)
        split = base["project_split"]
        fname = "train.jsonl" if split == "train" else "validation.jsonl"
        base_out = dict(base)
        base_out.pop("mix_trace", None)
        _append_jsonl(INTERMEDIATE / f"{base['task']}_base.jsonl", base_out)
        split_rows.append(base)
        for name in ("EXP-01", "EXP-02", "EXP-03"):
            view = _processed_view(conds[name])
            _append_jsonl(writers[name][fname], view)
            _append_jsonl(
                trace_paths[name],
                {
                    "example_id": base["example_id"],
                    "source_id": base["source_id"],
                    "task": base["task"],
                    "split": split,
                    "condition": name,
                    "tokens": conds["traces"][name],
                },
            )
            counts[f"{name}:{split}:{base['task']}"] += 1
    return dict(counts)


def coverage_summary(bases: list[dict[str, Any]], romanizer: DakshinaRomanizer) -> dict[str, Any]:
    total_tokens = 0
    latin_tokens = 0
    mapped = 0
    types: set[str] = set()
    mapped_types: set[str] = set()
    for row in bases:
        texts = []
        inp = row.get("input") or {}
        texts.extend(str(v) for v in inp.values())
        for text in texts:
            for tok in tokenize_conservative(text):
                if not tok.is_latin_word:
                    continue
                total_tokens += 1
                latin_tokens += 1
                types.add(tok.text.casefold())
                if romanizer.lookup_alias(tok.text):
                    mapped += 1
                    mapped_types.add(tok.text.casefold())
    return {
        "latin_token_occurrences": latin_tokens,
        "mapped_occurrences": mapped,
        "occurrence_coverage": mapped / latin_tokens if latin_tokens else 0,
        "latin_types": len(types),
        "mapped_types": len(mapped_types),
        "type_coverage": len(mapped_types) / len(types) if types else 0,
    }


def qa_matching_ids() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train.jsonl", "validation.jsonl"):
        ids = {}
        labels = {}
        for cond, folder in (
            ("EXP-01", EXP01),
            ("EXP-02", EXP02),
            ("EXP-03", EXP03),
        ):
            path = folder / split
            sid = []
            lab = Counter()
            if path.is_file():
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        row = json.loads(line)
                        sid.append(row["example_id"])
                        lab[str(row.get("label"))] += 1
            ids[cond] = sid
            labels[cond] = dict(lab)
        result[split] = {
            "n_EXP-01": len(ids["EXP-01"]),
            "n_EXP-02": len(ids["EXP-02"]),
            "n_EXP-03": len(ids["EXP-03"]),
            "ids_match": ids["EXP-01"] == ids["EXP-02"] == ids["EXP-03"],
            "labels_match": labels["EXP-01"] == labels["EXP-02"] == labels["EXP-03"],
        }
    return result


def human_sample(n: int = 100) -> None:
    rows: list[dict[str, Any]] = []
    for split in ("train.jsonl", "validation.jsonl"):
        path = EXP01 / split
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
    rng = derive_rng(42, "human-review", "sample")
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["task"], row["project_split"])].append(row)
    sample: list[dict[str, Any]] = []
    keys = list(by_key)
    i = 0
    while len(sample) < n and keys:
        key = keys[i % len(keys)]
        bucket = by_key[key]
        if bucket:
            sample.append(bucket.pop(rng.randrange(len(bucket))))
        else:
            keys.remove(key)
            continue
        i += 1
    out = REPORTS / "human_review_sample.csv"
    fields = [
        "example_id",
        "source_id",
        "task",
        "project_split",
        "mixing_intensity",
        "label",
        "exp01_input",
        "meaning_preserved",
        "English_preserved",
        "entities_preserved",
        "code_switch_preserved",
        "spelling_plausible",
        "issue_type",
        "reviewer_comment",
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in sample:
            w.writerow(
                {
                    "example_id": row["example_id"],
                    "source_id": row["source_id"],
                    "task": row["task"],
                    "project_split": row["project_split"],
                    "mixing_intensity": row.get("mixing_intensity"),
                    "label": row.get("label"),
                    "exp01_input": json.dumps(row.get("input"), ensure_ascii=False)[:500],
                    "meaning_preserved": "",
                    "English_preserved": "",
                    "entities_preserved": "",
                    "code_switch_preserved": "",
                    "spelling_plausible": "",
                    "issue_type": "",
                    "reviewer_comment": "",
                }
            )


def dakshina_samples(inv_stats: dict[str, Any]) -> None:
    from pandas import read_csv

    path = INVENTORIES / "dakshina_hi_variant_inventory.tsv"
    df = read_csv(path, sep="\t")
    parts = []
    parts.append(df[df["number_of_variants"] == 1].head(15))
    parts.append(df[df["number_of_variants"] >= 3].head(15))
    ties = df[df["canonical_romanization"] != df["romanized_variant"]].head(10)
    parts.append(ties)
    import pandas as pd

    sample = pd.concat(parts, ignore_index=True).head(50)
    out = INTERIM / "qa_samples" / "dakshina_inventory_samples.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, sep="\t", index=False)


def main() -> int:
    from load_env import bootstrap_project

    bootstrap_project()
    ensure_handoff_layout()
    config, config_hash = load_augmentation_config()
    p_variant = float(config.get("exp02", {}).get("p_variant", 0.35))

    print("copying reused raw dakshina/comi", flush=True)
    _copy_raw()
    print("dakshina audit", flush=True)
    dakshina_audit = full_dakshina_audit()
    save_audit_json(dakshina_audit, INTERIM / "dakshina_audit.json")
    print("building dakshina inventory", flush=True)
    inv_stats = build_variant_inventory(
        PROJECT_ROOT / config["inventory"]["source_file"], config
    )
    print("inventory done", inv_stats.get("native_words"), flush=True)
    export_phase2_inventories(inv_stats["inv_df"])
    dakshina_samples(inv_stats)
    romanizer = DakshinaRomanizer(INVENTORIES / "dakshina_hi_variant_inventory.tsv")

    print("acquiring task sources", flush=True)
    nli_meta = acquire_indicxnli()
    print("indicxnli ready", nli_meta.get("revision"), flush=True)
    print("acquiring paraphrase", flush=True)
    para_meta = acquire_indicxparaphrase()
    print("acquiring toxicity", flush=True)
    tox_meta = acquire_toxic_matrix()
    print("toxicity ready", tox_meta.get("benign_status"), tox_meta.get("n_toxic_raw"), flush=True)
    comi_audit = audit_comi() if (SRC_COMI / "MT_train.jsonl").is_file() else {}

    print("loading normalized nli", flush=True)
    nli = load_normalized_nli(nli_meta)
    print("nli loaded", len(nli), flush=True)
    para = load_normalized_paraphrase(para_meta)
    tox = load_normalized_toxicity(tox_meta)

    max_train = int(config.get("nli", {}).get("max_train_rows") or 0)
    if max_train:
        train_rows = [r for r in nli if r["original_split"] == "train"]
        other = [r for r in nli if r["original_split"] != "train"]
        rng = derive_rng(int(config.get("nli", {}).get("subsample_seed", 42)), "nli", "subsample")
        by_lab: dict[str, list] = defaultdict(list)
        for r in train_rows:
            by_lab[str(r["normalized_label"])].append(r)
        sampled: list[dict[str, Any]] = []
        labels = list(by_lab)
        while labels and len(sampled) < max_train:
            for lab in list(labels):
                bucket = by_lab[lab]
                if not bucket:
                    labels.remove(lab)
                    continue
                idx = rng.randrange(len(bucket))
                sampled.append(bucket.pop(idx))
                if len(sampled) >= max_train:
                    break
        nli = other + sampled

    nli = attach_xnli_english(nli)
    nli = assign_splits(nli, "nli")
    para = assign_splits(para, "paraphrase")
    tox = assign_splits(tox, "toxicity")

    for path in (
        EXP01 / "train.jsonl",
        EXP01 / "validation.jsonl",
        EXP02 / "train.jsonl",
        EXP02 / "validation.jsonl",
        EXP03 / "train.jsonl",
        EXP03 / "validation.jsonl",
        INTERMEDIATE / "nli_base.jsonl",
        INTERMEDIATE / "paraphrase_base.jsonl",
        INTERMEDIATE / "toxicity_base.jsonl",
        TRACES / "EXP-01" / "traces.jsonl",
        TRACES / "EXP-02" / "traces.jsonl",
        TRACES / "EXP-03" / "traces.jsonl",
    ):
        path.write_text("", encoding="utf-8")

    writers = {
        "EXP-01": {"train.jsonl": EXP01 / "train.jsonl", "validation.jsonl": EXP01 / "validation.jsonl"},
        "EXP-02": {"train.jsonl": EXP02 / "train.jsonl", "validation.jsonl": EXP02 / "validation.jsonl"},
        "EXP-03": {"train.jsonl": EXP03 / "train.jsonl", "validation.jsonl": EXP03 / "validation.jsonl"},
    }
    traces = {
        "EXP-01": TRACES / "EXP-01" / "traces.jsonl",
        "EXP-02": TRACES / "EXP-02" / "traces.jsonl",
        "EXP-03": TRACES / "EXP-03" / "traces.jsonl",
    }

    romanizer = DakshinaRomanizer()
    split_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    nli_base = build_nli_base(nli, romanizer)
    para_base = build_paraphrase_base(para, romanizer)
    tox_base = build_toxicity_base(tox, romanizer)
    counts.update(stream_task(nli_base, romanizer, p_variant, writers, traces, split_rows))
    counts.update(stream_task(para_base, romanizer, p_variant, writers, traces, split_rows))
    counts.update(stream_task(tox_base, romanizer, p_variant, writers, traces, split_rows))

    _write_split_manifest(split_rows, MANIFESTS / "source_split_manifest.csv")
    coverage = coverage_summary(nli_base[:2000] + para_base + tox_base[:2000], romanizer)
    qa = qa_matching_ids()
    human_sample(100)

    rels = [p.relative_to(PROJECT_ROOT) for p in HANDOFF.rglob("*") if p.is_file()]
    write_checksum_manifest(PROJECT_ROOT, rels, MANIFESTS / "checksums.sha256")
    shutil.copy2(PROJECT_ROOT / "configs" / "data_augmentation.yaml", MANIFESTS / "augmentation_config.yaml")

    provenance = []
    for task, meta in (
        ("nli", nli_meta),
        ("paraphrase", para_meta),
        ("toxicity", tox_meta),
    ):
        provenance.append({"task": task, **{k: v for k, v in meta.items() if k != "files" or True}})
    save_json(MANIFESTS / "data_provenance.json", provenance)

    row_counts = {
        "nli_source": len(nli),
        "paraphrase_source": len(para),
        "toxicity_source": len(tox),
        "nli_base": len(nli_base),
        "paraphrase_base": len(para_base),
        "toxicity_base": len(tox_base),
        "stream_counts": dict(counts),
        "qa": qa,
    }
    split_stats = {
        "nli_train_sources": len({r["source_id"] for r in nli if r.get("project_split") == "train"}),
        "nli_val_sources": len({r["source_id"] for r in nli if r.get("project_split") == "validation"}),
        "nli_overlap": len(
            {r["source_id"] for r in nli if r.get("project_split") == "train"}
            & {r["source_id"] for r in nli if r.get("project_split") == "validation"}
        ),
        "para_overlap": len(
            {r["source_id"] for r in para if r.get("project_split") == "train"}
            & {r["source_id"] for r in para if r.get("project_split") == "validation"}
        ),
        "tox_overlap": len(
            {r["source_id"] for r in tox if r.get("project_split") == "train"}
            & {r["source_id"] for r in tox if r.get("project_split") == "validation"}
        ),
        "test_in_training": any(r.get("original_split") == "test" for r in nli + para + tox),
    }

    ctx = {
        "status": "READY_FOR_OWNER_REVIEW" if qa.get("train.jsonl", {}).get("ids_match") else "NOT_READY",
        "inventory_stats": {k: v for k, v in inv_stats.items() if k not in {"alias_df", "inv_df"}},
        "source_meta": {"nli": nli_meta, "paraphrase": para_meta, "toxicity": tox_meta},
        "row_counts": row_counts,
        "split_stats": split_stats,
        "coverage": coverage,
        "qa": qa,
        "comi_audit": comi_audit,
        "config_hash": config_hash,
    }
    write_phase2_reports(ctx)
    save_json(MANIFESTS / "phase2_context.json", {k: v for k, v in ctx.items() if k != "comi_audit"})
    print(json.dumps({"status": ctx["status"], "row_counts": row_counts, "qa": qa, "split_stats": split_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
