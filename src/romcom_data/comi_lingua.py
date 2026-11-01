from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from romcom_data.paths import PROJECT_ROOT, RAW_COMI
from romcom_data.source_discovery import COMI_CONFIGS, COMI_REPO

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")


def classify_script(text: str) -> str:
    text = str(text)
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    total = dev + lat
    if total == 0:
        return "other"
    if lat / total >= 0.8:
        return "latin"
    if dev / total >= 0.8:
        return "devanagari"
    return "mixed"


def download_comi_snapshots(max_rows: int | None = None) -> dict[str, Any]:
    RAW_COMI.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(COMI_REPO)
    revision = info.sha
    stored: list[str] = []

    for config_name in COMI_CONFIGS:
        for split in ("train", "test"):
            ds = load_dataset(COMI_REPO, name=config_name, split=split)
            df = ds.to_pandas()
            if max_rows is not None and len(df) > max_rows:
                df = df.head(max_rows)
            out = RAW_COMI / f"{config_name}_{split}.jsonl"
            df.to_json(out, orient="records", lines=True, force_ascii=False)
            stored.append(out.name)

    return {"revision": revision, "configs_stored": stored, "license": "CC-BY-4.0"}


def audit_comi() -> dict[str, Any]:
    audit: dict[str, Any] = {
        "repository": COMI_REPO,
        "role": "supplementary_reference_only",
        "caveat": "Research preview; not task-label source for NLI/paraphrase/toxicity.",
        "configs": {},
    }
    for config_name in COMI_CONFIGS:
        cfg_audit: dict[str, Any] = {"splits": {}}
        for split in ("train", "test"):
            path = RAW_COMI / f"{config_name}_{split}.jsonl"
            if not path.is_file():
                cfg_audit["splits"][split] = {"missing": True}
                continue
            df = pd.read_json(path, lines=True)
            cfg_audit["splits"][split] = {
                "rows": int(len(df)),
                "columns": df.columns.tolist(),
            }
        audit["configs"][config_name] = cfg_audit

    mt_train = RAW_COMI / "MT_train.jsonl"
    mt_test = RAW_COMI / "MT_test.jsonl"
    if mt_train.is_file() and mt_test.is_file():
        tr = pd.read_json(mt_train, lines=True)
        te = pd.read_json(mt_test, lines=True)
        src_col = "Sentences" if "Sentences" in tr.columns else tr.columns[0]
        overlap = len(set(tr[src_col].astype(str)) & set(te[src_col].astype(str)))
        audit["mt_train_test_source_overlap"] = overlap

        rh_cols = [
            c
            for c in tr.columns
            if "RH" in c or "rh" in c.lower() or "Roman" in c
        ]
        if rh_cols:
            annotator_cols = [
                c
                for c in rh_cols
                if "annotator" in c.lower() or "Annotator" in c
            ][:3]
            if len(annotator_cols) >= 3:
                distinct = tr[annotator_cols].nunique(axis=1)
                audit["mt_annotator_distinct_strings"] = (
                    distinct.value_counts().sort_index().to_dict()
                )
                pairs = [
                    (annotator_cols[0], annotator_cols[1]),
                    (annotator_cols[0], annotator_cols[2]),
                    (annotator_cols[1], annotator_cols[2]),
                ]
                sim_stats = {}
                for a, b in pairs:
                    ratios = [
                        SequenceMatcher(None, str(x), str(y)).ratio()
                        for x, y in zip(tr[a], tr[b], strict=True)
                    ]
                    sim_stats[f"{a}_vs_{b}"] = {
                        "mean": float(sum(ratios) / len(ratios)),
                        "median": float(sorted(ratios)[len(ratios) // 2]),
                    }
                audit["mt_pairwise_character_similarity"] = sim_stats

                for col in annotator_cols:
                    audit[f"{col}_script"] = (
                        tr[col].astype(str).apply(classify_script).value_counts().to_dict()
                    )

    audit_path = PROJECT_ROOT / "data" / "interim" / "comi_lingua_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit
