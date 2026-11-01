"""Streamed schema inspection for Week 1 data gate (A–D)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from load_env import huggingface_token_configured, load_project_env  # noqa: E402

load_project_env(PROJECT_DIR)

import pandas as pd  # noqa: E402
from datasets import (  # noqa: E402
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    print("HF_TOKEN configured:", huggingface_token_configured())
    api = HfApi()

    section("A. IndicXNLI")
    repo = "Divyanshu/indicxnli"
    info = api.dataset_info(repo)
    card = info.card_data or {}
    print("HF card license field:", card.get("license"))
    all_names = sorted(f.rfilename for f in info.siblings)
    hi_train = [n for n in all_names if "hi" in n.lower() and "train" in n.lower()]
    hi_all = [n for n in all_names if "hi" in n.lower()]
    print("HINDI TRAIN FILES:", hi_train)
    print("ALL HINDI FILES:", hi_all)
    sample_file = "forward/dev/xnli_hi.json"
    path = hf_hub_download(repo_id=repo, filename=sample_file, repo_type="dataset")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    split_name, rows = next(iter(data.items()))
    print(f"Sample file: {sample_file} | split={split_name} | n={len(rows)}")
    rec = rows[0]
    for key, value in rec.items():
        print(f"  {key}: {str(value)[:350]}")
    try:
        print("get_dataset_config_names:", get_dataset_config_names(repo))
    except Exception as exc:
        print("get_dataset_config_names error:", exc)

    section("B. IndicXParaphrase")
    repo = "ai4bharat/IndicXParaphrase"
    info = api.dataset_info(repo)
    print("License metadata:", (info.card_data or {}).get("license"))
    print(
        "TSV files:",
        sorted(f.rfilename for f in info.siblings if f.rfilename.endswith(".tsv")),
    )
    file_path = hf_hub_download(repo_id=repo, filename="data/hi.tsv", repo_type="dataset")
    frame = pd.read_csv(file_path, sep="\t")
    print("Shape:", frame.shape)
    print("Columns:", frame.columns.tolist())
    pd.set_option("display.max_colwidth", 85)
    print(frame.head(5))
    print("Label distribution:\n", frame["label"].value_counts(dropna=False))

    section("C. IndicAlign Toxic_Matrix")
    repo = "ai4bharat/indic-align"
    configs = get_dataset_config_names(repo)
    print("CONFIGURATIONS:", configs)
    toxic_config = next(
        c for c in configs if "toxic" in c.lower() and "matrix" in c.lower()
    )
    print("Selected:", toxic_config)
    splits = get_dataset_split_names(repo, config_name=toxic_config)
    print("Splits:", splits)
    ds = load_dataset(repo, name=toxic_config, split=splits[0], streaming=True)
    print("Features:", ds.features)
    for idx, row in enumerate(ds.take(3)):
        print(f"\nEXAMPLE {idx + 1}")
        for key, value in row.items():
            if key.startswith("__"):
                continue
            print(f"{key}: {str(value)[:500]}")

    section("D. COMI-LINGUA")
    repo = "LingoIITGN/COMI-LINGUA"
    print("License:", (api.dataset_info(repo).card_data or {}).get("license"))
    configs = get_dataset_config_names(repo)
    print("CONFIGURATIONS:", configs)
    for config_name in configs:
        print(f"\n{'=' * 50}\nCONFIG: {config_name}")
        splits = get_dataset_split_names(repo, config_name=config_name)
        print("Splits:", splits)
        split_name = "train" if "train" in splits else splits[0]
        ds = load_dataset(repo, name=config_name, split=split_name, streaming=True)
        print("Using split:", split_name, "| Features:", ds.features)
        for idx, row in enumerate(ds.take(2)):
            print(f"\nSample {idx + 1}")
            for key, value in row.items():
                print(f"{key}: {str(value)[:400]}")


if __name__ == "__main__":
    main()
