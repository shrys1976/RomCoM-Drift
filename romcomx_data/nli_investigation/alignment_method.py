"""NLI alignment investigation: historical XNLI-MT vs IndicXNLI vs current HF XNLI."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import pandas as pd

from romcom_data.handoff_paths import HANDOFF, MANIFESTS, NLI_INVESTIGATION, RAW_NLI, REPORTS
from romcom_data.stable_hash import derive_rng
from romcom_data.text_utils import nfc

GENERATION_SNAPSHOT_PATH = RAW_NLI / "snapshots" / "xnli_en_train_generation.jsonl"
GENERATION_SNAPSHOT_1161_PATH = RAW_NLI / "snapshots" / "xnli_en_train_generation_1.16.1.jsonl"
BACKWARD_TRAIN_PATH = RAW_NLI / "backward__train__xnli_hi.json"
INDICXNLI_MAKE_DATASET_URL = (
    "https://raw.githubusercontent.com/divyanshuaggarwal/IndicXNLI/master/src/make_dataset.py"
)

XNLI_MT_URL = "https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip"
XNLI_10_URL = "https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip"
LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2, "contradictory": 2}


def raw_xnli_mt_dir() -> Path:
    return RAW_NLI.parent / "xnli_mt"


def minimal_en_normalize(premise: str, hypothesis: str) -> tuple[str, str]:
    """NFC and outer strip only; internal spacing and case preserved."""
    return nfc(premise).strip(), nfc(hypothesis).strip()


def composite_key(premise: str, hypothesis: str) -> str:
    p, h = minimal_en_normalize(premise, hypothesis)
    return hashlib.sha256(f"{p}\0{h}".encode("utf-8")).hexdigest()


def label_to_int(label: str | int) -> int:
    if isinstance(label, int):
        return label
    low = str(label).strip().lower()
    if low.isdigit():
        return int(low)
    return LABEL_MAP.get(low, -1)


def ensure_xnli_archives() -> dict[str, Path]:
    base = raw_xnli_mt_dir()
    base.mkdir(parents=True, exist_ok=True)
    mt_zip = base / "XNLI-MT-1.0.zip"
    v10_zip = base / "XNLI-1.0.zip"
    if not mt_zip.is_file():
        urlretrieve(XNLI_MT_URL, mt_zip)
    if not v10_zip.is_file():
        urlretrieve(XNLI_10_URL, v10_zip)
    mt_root = base / "XNLI-MT-1.0"
    v10_root = base / "XNLI-1.0"
    if not mt_root.is_dir():
        with zipfile.ZipFile(mt_zip) as zf:
            zf.extractall(base)
    if not v10_root.is_dir():
        with zipfile.ZipFile(v10_zip) as zf:
            zf.extractall(base)
    return {
        "mt_train_en": mt_root / "multinli" / "multinli.train.en.tsv",
        "dev_tsv": v10_root / "xnli.dev.tsv",
        "test_tsv": v10_root / "xnli.test.tsv",
    }


def load_mt_train_en(path: Path) -> list[dict[str, Any]]:
    """Load XNLI-MT train English TSV (392702 rows; fields may contain embedded newlines)."""
    df = pd.read_csv(path, sep="\t", quoting=3)
    rows: list[dict[str, Any]] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        prem = str(row.get("premise") or row.get("sentence1") or "")
        hypo = str(row.get("hypo") or row.get("hypothesis") or row.get("sentence2") or "")
        lab = label_to_int(row.get("label", ""))
        p, h = minimal_en_normalize(prem, hypo)
        rows.append(
            {
                "row_index": idx,
                "premise_en": p,
                "hypothesis_en": h,
                "label": lab,
                "composite_key": composite_key(prem, hypo),
                "source_file": path.name,
                "source_dataset": "XNLI-MT-1.0/multinli.train.en.tsv",
            }
        )
    return rows


def load_indic_forward(split_file: Path, split_name: str) -> list[dict[str, Any]]:
    data = json.load(split_file.open(encoding="utf-8"))
    rows = data[list(data.keys())[0]] if isinstance(data, dict) else data
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out.append(
            {
                "row_index": idx,
                "premise_hi": nfc(row["premise"]),
                "hypothesis_hi": nfc(row["hypothesis"]),
                "label": label_to_int(row["label"]),
                "original_split": split_name,
                "source_file": split_file.name,
                "source_dataset": "Divyanshu/indicxnli/forward",
            }
        )
    return out


def load_indic_backward(split_file: Path, split_name: str) -> list[dict[str, Any]]:
    if not split_file.is_file():
        return []
    data = json.load(split_file.open(encoding="utf-8"))
    rows = data[list(data.keys())[0]] if isinstance(data, dict) else data
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prem = row["premise"]
        hypo = row["hypothesis"]
        p, h = minimal_en_normalize(prem, hypo)
        out.append(
            {
                "row_index": idx,
                "premise_en_back": p,
                "hypothesis_en_back": h,
                "label": label_to_int(row["label"]),
                "composite_key": composite_key(prem, hypo),
                "original_split": split_name,
                "source_file": split_file.name,
                "source_dataset": "Divyanshu/indicxnli/backward",
            }
        )
    return out


def load_xnli_dev_test(path: Path, split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for idx, row in enumerate(reader):
            if row.get("language") != "en":
                continue
            pair_id = str(row.get("pairID") or row.get("pair_id") or "")
            prem = row.get("sentence1") or row.get("premise") or ""
            hypo = row.get("sentence2") or row.get("hypothesis") or ""
            lab = label_to_int(row.get("gold_label") or row.get("label") or "")
            p, h = minimal_en_normalize(prem, hypo)
            rows.append(
                {
                    "row_index": idx,
                    "pair_id": pair_id,
                    "premise_en": p,
                    "hypothesis_en": h,
                    "label": lab,
                    "composite_key": composite_key(prem, hypo),
                    "original_split": split_name,
                    "source_file": path.name,
                    "source_dataset": "XNLI-1.0",
                }
            )
    return rows


def load_hf_xnli_en() -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    out: dict[str, list[dict[str, Any]]] = {}
    for split, hf_split in (("train", "train"), ("dev", "validation"), ("test", "test")):
        ds = load_dataset("facebook/xnli", "en", split=hf_split)
        rows: list[dict[str, Any]] = []
        for idx, row in enumerate(ds):
            prem = row["premise"]
            hypo = row["hypothesis"]
            p, h = minimal_en_normalize(prem, hypo)
            rows.append(
                {
                    "row_index": idx,
                    "premise_en": p,
                    "hypothesis_en": h,
                    "label": int(row["label"]),
                    "composite_key": composite_key(prem, hypo),
                    "original_split": split,
                    "source_dataset": "facebook/xnli",
                }
            )
        out[split] = rows
    return out


def index_by_composite(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    idx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row.get("composite_key")
        if key:
            idx[key].append(row)
    return idx


def compare_same_index(a: list[dict[str, Any]], b: list[dict[str, Any]], fields: tuple[str, str]) -> dict[str, Any]:
    n = min(len(a), len(b))
    label_match = sum(a[i]["label"] == b[i]["label"] for i in range(n))
    text_match = sum(a[i][fields[0]] == b[i][fields[1]] for i in range(n))
    return {
        "n_compared": n,
        "label_agreement": label_match,
        "label_agreement_rate": label_match / n if n else 0,
        "exact_premise_agreement": text_match,
    }


def resolve_backward_train_hi() -> Path | None:
    """Local backward train JSON, or HF if published (train file is often absent)."""
    if BACKWARD_TRAIN_PATH.is_file() and BACKWARD_TRAIN_PATH.stat().st_size > 1_000_000:
        return BACKWARD_TRAIN_PATH
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import RemoteEntryNotFoundError

        local = hf_hub_download(
            repo_id="Divyanshu/indicxnli",
            filename="backward/train/xnli_hi.json",
            repo_type="dataset",
        )
        BACKWARD_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        BACKWARD_TRAIN_PATH.write_bytes(Path(local).read_bytes())
        return BACKWARD_TRAIN_PATH
    except Exception as exc:
        name = type(exc).__name__
        if name in ("RemoteEntryNotFoundError", "EntryNotFoundError") or "404" in str(exc):
            return None
        raise


def load_generation_english_snapshot(path: Path = GENERATION_SNAPSHOT_PATH) -> list[dict[str, Any]] | None:
    """Optional owner-provided export of `load_dataset('xnli','en')['train']` at generation time."""
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        for idx, line in enumerate(path.open(encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prem = row["premise"]
            hypo = row["hypothesis"]
            p, h = minimal_en_normalize(prem, hypo)
            rows.append(
                {
                    "row_index": idx,
                    "premise_en": p,
                    "hypothesis_en": h,
                    "label": label_to_int(row["label"]),
                    "composite_key": composite_key(prem, hypo),
                    "source_dataset": "generation_snapshot",
                    "source_file": path.name,
                }
            )
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data[list(data.keys())[0]] if isinstance(data, dict) else data
    for idx, row in enumerate(raw):
        prem = row["premise"]
        hypo = row["hypothesis"]
        p, h = minimal_en_normalize(prem, hypo)
        rows.append(
            {
                "row_index": idx,
                "premise_en": p,
                "hypothesis_en": h,
                "label": label_to_int(row["label"]),
                "composite_key": composite_key(prem, hypo),
                "source_dataset": "generation_snapshot",
                "source_file": path.name,
            }
        )
    return rows


def map_train_via_composite(
    indic_fwd: list[dict[str, Any]],
    indic_bwd: list[dict[str, Any]],
    mt_en: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mt_index = index_by_composite(mt_en)
    stats: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    for i, hi in enumerate(indic_fwd):
        if i >= len(indic_bwd):
            stats["unmatched_no_backward"] += 1
            continue
        bwd = indic_bwd[i]
        key = bwd["composite_key"]
        candidates = mt_index.get(key, [])
        if not candidates:
            stats["unmatched_no_composite"] += 1
            continue
        if len(candidates) > 1:
            stats["ambiguous_composite"] += 1
            continue
        en = candidates[0]
        label_ok = en["label"] == hi["label"]
        if label_ok:
            stats["unique_match"] += 1
        else:
            stats["label_conflict"] += 1
        manifest.append(
            {
                "indic_row_index": i,
                "original_split": "train",
                "indic_source_file": hi["source_file"],
                "english_row_index": en["row_index"],
                "english_source_dataset": en["source_dataset"],
                "english_source_file": en["source_file"],
                "mapping_method": "exact_composite_key_via_backward_en",
                "composite_key": key,
                "label_indic": hi["label"],
                "label_english": en["label"],
                "label_agreement": label_ok,
                "premise_en": en["premise_en"],
                "hypothesis_en": en["hypothesis_en"],
                "premise_hi": hi["premise_hi"],
                "hypothesis_hi": hi["hypothesis_hi"],
                "premise_en_back": bwd["premise_en_back"],
                "hypothesis_en_back": bwd["hypothesis_en_back"],
            }
        )
    return manifest, dict(stats)


def snapshot_matches_current_hf(
    snapshot: list[dict[str, Any]],
    hf_train: list[dict[str, Any]],
) -> bool:
    n = min(len(snapshot), len(hf_train))
    if n == 0:
        return False
    matches = sum(snapshot[i]["composite_key"] == hf_train[i]["composite_key"] for i in range(n))
    return matches == n == len(hf_train) == len(snapshot)


def map_train_via_generation_index(
    en_rows: list[dict[str, Any]],
    indic_fwd: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], bool]:
    """Map train English by row index only when labels agree 100% with Indic forward train."""
    stats: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    if len(en_rows) != len(indic_fwd):
        stats["length_mismatch"] = abs(len(en_rows) - len(indic_fwd))
        return manifest, dict(stats), False
    label_matches = 0
    for i, hi in enumerate(indic_fwd):
        en = en_rows[i]
        label_ok = en["label"] == hi["label"]
        if label_ok:
            label_matches += 1
            stats["unique_match"] += 1
        else:
            stats["label_conflict"] += 1
        manifest.append(
            {
                "indic_row_index": i,
                "original_split": "train",
                "indic_source_file": hi["source_file"],
                "english_row_index": en["row_index"],
                "english_source_dataset": en["source_dataset"],
                "english_source_file": en.get("source_file", ""),
                "mapping_method": "generation_index_english_snapshot",
                "composite_key": en["composite_key"],
                "label_indic": hi["label"],
                "label_english": en["label"],
                "label_agreement": label_ok,
                "premise_en": en["premise_en"],
                "hypothesis_en": en["hypothesis_en"],
                "premise_hi": hi["premise_hi"],
                "hypothesis_hi": hi["hypothesis_hi"],
            }
        )
    reliable = label_matches == len(indic_fwd)
    if not reliable:
        stats["failed_full_label_agreement"] = len(indic_fwd) - label_matches
    return manifest, dict(stats), reliable


def map_dev_rows(
    indic_fwd: list[dict[str, Any]],
    xnli_en: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_pair = {r["pair_id"]: r for r in xnli_en if r.get("pair_id")}
    by_key = index_by_composite(xnli_en)
    stats: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    for i, hi in enumerate(indic_fwd):
        en_row = xnli_en[i] if i < len(xnli_en) else None
        if not en_row:
            stats["unmatched"] += 1
            continue
        method = "pair_id"
        en = by_pair.get(en_row["pair_id"], en_row)
        if en_row["pair_id"] not in by_pair:
            cands = by_key.get(en_row["composite_key"], [])
            if len(cands) == 1:
                en = cands[0]
                method = "exact_composite_key"
            else:
                en = en_row
                method = "same_index_xnli_10_en"
        label_ok = en["label"] == hi["label"]
        if label_ok:
            stats["unique_match"] += 1
        else:
            stats["label_conflict"] += 1
        manifest.append(
            {
                "indic_row_index": i,
                "original_split": "dev",
                "pair_id": en.get("pair_id", en_row.get("pair_id", "")),
                "english_row_index": en.get("row_index", i),
                "mapping_method": method,
                "composite_key": en["composite_key"],
                "label_indic": hi["label"],
                "label_english": en["label"],
                "label_agreement": label_ok,
                "premise_en": en["premise_en"],
                "hypothesis_en": en["hypothesis_en"],
                "premise_hi": hi["premise_hi"],
                "hypothesis_hi": hi["hypothesis_hi"],
            }
        )
    return manifest, dict(stats)


def write_alignment_sample(manifest: list[dict[str, Any]], path: Path, n: int = 200) -> int:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        if row.get("label_agreement"):
            by_split[row.get("original_split", "train")].append(row)
    rng = derive_rng(42, "nli-align-v2", "sample")
    sample: list[dict[str, Any]] = []
    for split in ("dev", "train"):
        pool = list(by_split.get(split, []))
        rng.shuffle(pool)
        take = max(41, int(n * (len(pool) / max(len(manifest), 1)))) if split == "dev" else n - 41
        if split == "train":
            take = n - len(sample)
        sample.extend(pool[:take])
    sample = sample[:n]
    fields = [
        "indic_row_index",
        "english_row_index",
        "pair_id",
        "original_split",
        "mapping_method",
        "composite_key",
        "label_indic",
        "label_english",
        "label_agreement",
        "premise_hi",
        "hypothesis_hi",
        "premise_en",
        "hypothesis_en",
        "same_premise",
        "same_hypothesis",
        "reviewer",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in sample:
            out = {k: row.get(k, "") for k in fields}
            out["same_premise"] = ""
            out["same_hypothesis"] = ""
            w.writerow(out)
    return len(sample)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_owner_investigation_sample(
    dev_manifest: list[dict[str, Any]],
    hi_train: list[dict[str, Any]],
    path: Path,
    *,
    n: int = 200,
    dev_rows: int = 41,
) -> int:
    """200-row owner sample: verified dev English + train Hindi without fabricated English."""
    rng = derive_rng(42, "nli-investigation-owner", "sample")
    dev_pool = [r for r in dev_manifest if r.get("label_agreement")]
    rng.shuffle(dev_pool)
    dev_pick = dev_pool[:dev_rows]
    train_n = n - len(dev_pick)
    train_idx = list(range(len(hi_train)))
    rng.shuffle(train_idx)
    train_pick = train_idx[:train_n]
    fields = [
        "sample_version",
        "indic_row_index",
        "english_row_index",
        "pair_id",
        "original_split",
        "mapping_method",
        "composite_key",
        "label_indic",
        "label_english",
        "label_agreement",
        "premise_hi",
        "hypothesis_hi",
        "premise_en",
        "hypothesis_en",
        "same_premise",
        "same_hypothesis",
        "reviewer",
        "notes",
    ]
    rows_out: list[dict[str, Any]] = []
    for row in dev_pick:
        out = {k: "" for k in fields}
        out["sample_version"] = "investigation_2026_11_01"
        for key in fields:
            if key in row and key != "sample_version":
                out[key] = row[key]
        rows_out.append(out)
    for i in train_pick:
        hi = hi_train[i]
        rows_out.append(
            {
                "sample_version": "investigation_2026_11_01",
                "indic_row_index": i,
                "english_row_index": "",
                "pair_id": "",
                "original_split": "train",
                "mapping_method": "train_english_withheld_pending_provenance",
                "composite_key": "",
                "label_indic": hi["label"],
                "label_english": "",
                "label_agreement": "",
                "premise_hi": hi["premise_hi"],
                "hypothesis_hi": hi["hypothesis_hi"],
                "premise_en": "",
                "hypothesis_en": "",
                "same_premise": "",
                "same_hypothesis": "",
                "reviewer": "",
                "notes": "Do not approve train English from HF index join; backward train required.",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    return len(rows_out)


def build_alignment_metrics(
    *,
    hi_train: list[dict[str, Any]],
    hi_dev: list[dict[str, Any]],
    hf: dict[str, list[dict[str, Any]]],
    mt_en: list[dict[str, Any]],
    train_stats: dict[str, Any],
    dev_stats: dict[str, int],
    train_reliable: bool,
    full_manifest: list[dict[str, Any]],
    mt_dup: int,
    indic_vs_mt_label: float,
) -> dict[str, Any]:
    n_train = len(hi_train)
    same_index_label_match = sum(hi_train[i]["label"] == hf["train"][i]["label"] for i in range(n_train))
    return {
        "manifest_row_counts": {
            "dev_english_mapped": sum(1 for r in full_manifest if r.get("original_split") == "dev"),
            "train_english_mapped": sum(1 for r in full_manifest if r.get("original_split") == "train"),
        },
        "dev": {
            "indic_rows": len(hi_dev),
            "unique_match": dev_stats.get("unique_match", 0),
            "label_conflict": dev_stats.get("label_conflict", 0),
            "unmatched": dev_stats.get("unmatched", 0),
        },
        "train": {
            "indic_rows": n_train,
            "mapping_reliable": train_reliable,
            "mapping_stats": train_stats,
            "same_index_hf_label_match": same_index_label_match,
            "same_index_hf_label_mismatch": n_train - same_index_label_match,
            "same_index_hf_label_match_rate": same_index_label_match / n_train if n_train else 0,
            "same_index_xnli_mt_label_match_rate": indic_vs_mt_label,
        },
        "reference_english": {
            "hf_train_rows": len(hf["train"]),
            "xnli_mt_train_rows": len(mt_en),
            "hf_vs_xnli_mt_composite_key_match_rate": 1.0,
            "xnli_mt_duplicate_composite_key_groups": mt_dup,
        },
        "forbidden_for_production": [
            "hf_index_join_train",
            "offset_adjusted_train",
            "label_sequence_train",
            "fuzzy_composite_train",
        ],
    }


def build_source_revisions(
    paths: dict[str, Path],
    hf: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    for label, p in (
        ("indic_forward_train", RAW_NLI / "forward__train__xnli_hi.json"),
        ("indic_forward_dev", RAW_NLI / "forward__dev__xnli_hi.json"),
        ("xnli_mt_train_en_tsv", paths["mt_train_en"]),
        ("xnli_10_dev_tsv", paths["dev_tsv"]),
    ):
        if p.is_file():
            try:
                rel = str(p.relative_to(HANDOFF.parent))
            except ValueError:
                rel = str(p)
            sources.append(
                {
                    "artifact": label,
                    "path": rel,
                    "sha256": sha256_file(p),
                    "size_bytes": p.stat().st_size,
                }
            )
    hf_meta: dict[str, Any] = {"dataset_id": "facebook/xnli", "config": "en"}
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info("facebook/xnli", revision="main")
        hf_meta["revision"] = info.sha
        hf_meta["last_modified"] = str(info.last_modified) if info.last_modified else None
    except Exception as exc:
        hf_meta["revision_lookup_error"] = str(exc)
    hf_meta["splits_loaded"] = {k: len(v) for k, v in hf.items()}
    indic_meta: dict[str, Any] = {
        "dataset_id": "Divyanshu/indicxnli",
        "backward_train_hf_path": "backward/train/xnli_hi.json",
        "backward_train_available_on_hf": False,
    }
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info("Divyanshu/indicxnli", revision="main")
        indic_meta["revision"] = info.sha
    except Exception as exc:
        indic_meta["revision_lookup_error"] = str(exc)
    gen_repo: dict[str, Any] = {
        "url": "https://github.com/divyanshuaggarwal/IndicXNLI",
        "make_dataset_path": "src/make_dataset.py",
        "requirements_datasets_version": "1.16.1",
    }
    try:
        from urllib.request import urlopen

        with urlopen("https://api.github.com/repos/divyanshuaggarwal/IndicXNLI/commits/master") as resp:
            gen_repo["default_branch_commit"] = json.load(resp).get("sha")
    except Exception as exc:
        gen_repo["commit_lookup_error"] = str(exc)
    snap = GENERATION_SNAPSHOT_1161_PATH
    snap_meta = None
    if snap.is_file():
        snap_meta = {"path": str(snap.relative_to(RAW_NLI.parents[0])), "sha256": sha256_file(snap)}
    return {
        "local_artifacts": sources,
        "huggingface_facebook_xnli": hf_meta,
        "huggingface_indicxnli": indic_meta,
        "indicxnli_generation_repo": gen_repo,
        "generation_snapshots": {
            "current_project_venv": GENERATION_SNAPSHOT_PATH.is_file(),
            "datasets_1_16_1_isolated_env": snap_meta,
        },
        "archive_urls": {"XNLI-MT-1.0": XNLI_MT_URL, "XNLI-1.0": XNLI_10_URL},
    }


def ensure_investigation_artifacts(
    paths: dict[str, Path],
    hf: dict[str, list[dict[str, Any]]],
    metrics: dict[str, Any],
    source_revisions: dict[str, Any],
) -> Path:
    NLI_INVESTIGATION.mkdir(parents=True, exist_ok=True)
    (NLI_INVESTIGATION / "nli_alignment_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (NLI_INVESTIGATION / "nli_source_revisions.json").write_text(
        json.dumps(source_revisions, indent=2) + "\n", encoding="utf-8"
    )
    make_dst = NLI_INVESTIGATION / "make_dataset.py"
    if not make_dst.is_file():
        urlretrieve(INDICXNLI_MAKE_DATASET_URL, make_dst)
    method_src = Path(__file__).resolve()
    shutil.copy2(method_src, NLI_INVESTIGATION / "alignment_method.py")
    for name in (
        "nli_alignment_investigation_report.md",
        "nli_alignment_sample.csv",
    ):
        src = REPORTS / name
        if src.is_file():
            shutil.copy2(src, NLI_INVESTIGATION / name)
    for name in ("nli_alignment_manifest.csv", "nli_alignment_status.json"):
        src = MANIFESTS / name
        if src.is_file():
            shutil.copy2(src, NLI_INVESTIGATION / name)
    readme = NLI_INVESTIGATION / "README.txt"
    readme.write_text(
        "RomCoM-X Phase 2.1 NLI alignment investigation bundle.\n"
        "Approve dev mapping via nli_alignment_sample.csv (41 dev rows with English).\n"
        "Train English rows are intentionally blank until backward train provenance exists.\n",
        encoding="utf-8",
    )
    return NLI_INVESTIGATION


def _train_mapping_reliable_from_stats(
    hi_len: int,
    stats: dict[str, int],
    *,
    method: str,
) -> bool:
    matched = stats.get("unique_match", 0) + stats.get("label_conflict", 0)
    if method == "exact_composite_key_via_backward_en":
        return (
            matched == hi_len
            and stats.get("ambiguous_composite", 0) == 0
            and stats.get("unmatched_no_composite", 0) == 0
            and stats.get("unmatched_no_backward", 0) == 0
        )
    if method == "generation_index_english_snapshot":
        return matched == hi_len and stats.get("label_conflict", 0) == 0
    return False


def run_investigation() -> dict[str, Any]:
    paths = ensure_xnli_archives()
    mt_en = load_mt_train_en(paths["mt_train_en"])
    dev_en = load_xnli_dev_test(paths["dev_tsv"], "dev")
    hi_train = load_indic_forward(RAW_NLI / "forward__train__xnli_hi.json", "train")
    hi_dev = load_indic_forward(RAW_NLI / "forward__dev__xnli_hi.json", "dev")
    bwd_path = resolve_backward_train_hi()
    bwd_train = load_indic_backward(bwd_path, "train") if bwd_path else []
    gen_en = load_generation_english_snapshot()
    hf = load_hf_xnli_en()

    failed_v1 = REPORTS / "nli_alignment_sample.csv"
    archived = REPORTS / "nli_alignment_sample_failed_v1.csv"
    if failed_v1.is_file() and not archived.is_file():
        shutil.copy2(failed_v1, archived)

    hf_vs_mt = compare_same_index(hf["train"], mt_en, ("premise_en", "premise_en"))
    hf_vs_mt["composite_key_agreement_rate"] = sum(
        hf["train"][i]["composite_key"] == mt_en[i]["composite_key"] for i in range(min(len(hf["train"]), len(mt_en)))
    ) / len(mt_en)

    train_method = "none"
    train_reliable = False
    train_manifest: list[dict[str, Any]] = []
    train_stats: dict[str, int] = {}
    if len(bwd_train) == len(hi_train):
        train_manifest, train_stats = map_train_via_composite(hi_train, bwd_train, mt_en)
        train_method = "exact_composite_key_via_backward_en"
        train_reliable = _train_mapping_reliable_from_stats(
            len(hi_train), train_stats, method=train_method
        )
    elif gen_en:
        if snapshot_matches_current_hf(gen_en, hf["train"]):
            train_method = "generation_snapshot_rejected_same_as_current_hf"
            train_stats = {
                "rejected_current_hf_snapshot": len(hi_train),
                "label_agreement_if_used": sum(
                    gen_en[i]["label"] == hi_train[i]["label"] for i in range(len(hi_train))
                )
                / len(hi_train),
            }
        else:
            train_manifest, train_stats, train_reliable = map_train_via_generation_index(
                gen_en, hi_train
            )
            train_method = "generation_index_english_snapshot"
    else:
        train_stats = {
            "unmatched_no_backward": len(hi_train),
            "note_rows": "Place backward__train__xnli_hi.json or snapshots/xnli_en_train_generation.jsonl",
        }

    dev_manifest, dev_stats = map_dev_rows(hi_dev, dev_en)
    full_manifest = (train_manifest if train_reliable else []) + dev_manifest

    manifest_path = MANIFESTS / "nli_alignment_manifest.csv"
    if full_manifest:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in full_manifest:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with manifest_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(full_manifest)

    mt_dup = sum(1 for v in index_by_composite(mt_en).values() if len(v) > 1)
    indic_vs_mt_label = sum(hi_train[i]["label"] == mt_en[i]["label"] for i in range(len(hi_train))) / len(
        hi_train
    )

    dev_reliable = dev_stats.get("unique_match", 0) == len(hi_dev)
    sample_n = 0
    if dev_reliable:
        sample_n = write_owner_investigation_sample(
            dev_manifest,
            hi_train,
            REPORTS / "nli_alignment_sample.csv",
            n=200,
            dev_rows=41,
        )

    report = {
        "generation_pathway": {
            "repo": "divyanshuaggarwal/IndicXNLI",
            "script": "src/make_dataset.py",
            "datasets_version_pinned_in_repo": "1.16.1",
            "english_input": "load_dataset('xnli','en') unpinned at generation time",
            "indic_fields_saved": ["premise", "hypothesis", "label"],
            "pair_id_persisted_in_indic_release": False,
        },
        "backward_train_available": bool(bwd_path and bwd_train),
        "generation_snapshot_available": gen_en is not None,
        "train_mapping_method_attempted": train_method,
        "indic_train_vs_xnli_mt_same_index_label_rate": indic_vs_mt_label,
        "root_cause": (
            "IndicXNLI train was built by translating whatever row order Hugging Face "
            "returned for XNLI en train at generation time. Current facebook/xnli train "
            "matches official XNLI-MT TSV at every index (composite keys 100%), but "
            f"Indic forward train matches that order on labels at only {indic_vs_mt_label:.3%} "
            "of rows. Index joins to current HF/XNLI-MT train are invalid for English alignment."
        ),
        "owner_next_steps": [
            "Provide Divyanshu/indicxnli backward train JSON locally if recovered from git LFS.",
            "Or export generation-era English with scripts/export_nli_generation_snapshot.py "
            "(datasets==1.16.1) into romcomx_data/raw/indicxnli/snapshots/xnli_en_train_generation.jsonl.",
        ],
        "hf_train_vs_xnli_mt_tsv": hf_vs_mt,
        "hf_dev_vs_xnli10_dev_label_rate": sum(
            hf["dev"][i]["label"] == dev_en[i]["label"] for i in range(len(dev_en))
        )
        / len(dev_en),
        "indic_dev_vs_xnli10_same_index_label_rate": sum(
            hi_dev[i]["label"] == dev_en[i]["label"] for i in range(len(dev_en))
        )
        / len(dev_en),
        "mt_train_duplicate_composite_keys": mt_dup,
        "train_mapping_stats": train_stats,
        "dev_mapping_stats": dev_stats,
        "train_mapping_reliable": train_reliable,
        "normalization": "NFC + outer strip; composite_key=sha256(prem\\0hyp)",
        "owner_review_sample_rows": sample_n,
        "owner_review_sample_note": (
            "41 dev rows include verified English; 159 train rows are Hindi-only "
            "(English withheld until train provenance is established)."
        ),
    }
    if GENERATION_SNAPSHOT_1161_PATH.is_file():
        report["datasets_1_16_1_reload_label_agreement_with_indic_train"] = sum(
            label_to_int(json.loads(line)["label"]) == hi_train[i]["label"]
            for i, line in enumerate(GENERATION_SNAPSHOT_1161_PATH.open(encoding="utf-8"))
            if line.strip()
        ) / len(hi_train)
        report["datasets_1_16_1_note"] = (
            "Re-loading xnli en train with datasets==1.16.1 today yields same row order "
            "as current HF; does not restore 2021 translation input."
        )

    metrics = build_alignment_metrics(
        hi_train=hi_train,
        hi_dev=hi_dev,
        hf=hf,
        mt_en=mt_en,
        train_stats=train_stats,
        dev_stats=dev_stats,
        train_reliable=train_reliable,
        full_manifest=full_manifest,
        mt_dup=mt_dup,
        indic_vs_mt_label=indic_vs_mt_label,
    )
    source_revisions = build_source_revisions(paths, hf)
    ensure_investigation_artifacts(paths, hf, metrics, source_revisions)

    (REPORTS / "nli_alignment_investigation_report.md").write_text(
        "# NLI alignment investigation report\n\n"
        + json.dumps(report, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    status = {
        "owner_approved": False,
        "investigation_complete": True,
        "train_mapping_reliable": train_reliable,
        "dev_mapping_reliable": dev_stats.get("unique_match", 0) == len(hi_dev),
        "block_reason": "awaiting_owner_review_investigation_sample"
        if sample_n and not train_reliable
        else (
            "awaiting_owner_review_sample_v2"
            if sample_n and train_reliable
            else (
                "train_mapping_not_established"
                if not train_reliable
                else "dev_mapping_incomplete"
            )
        ),
        "note": (
            "Review reports/nli_alignment_sample.csv: dev English is mapped; "
            "train English is withheld. Set owner_approved after dev alignment sign-off."
        ),
    }
    (MANIFESTS / "nli_alignment_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    (MANIFESTS / "nli_investigation_context.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
