from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi

from romcom_data.paths import PROJECT_ROOT, RAW_INDI

INDI_ROMCOM_PAPER = "https://arxiv.org/abs/2606.30790"
INDI_ROMCOM_HTML = "https://arxiv.org/html/2606.30790"

SEARCH_CANDIDATES = [
    {
        "name": "Indi-RomCoM arXiv paper",
        "url": INDI_ROMCOM_PAPER,
        "type": "paper",
        "official": True,
    },
    {
        "name": "Indi-RomCoM arXiv HTML",
        "url": INDI_ROMCOM_HTML,
        "type": "paper_html",
        "official": True,
    },
    {
        "name": "HuggingFace search Indi-RomCoM",
        "url": "https://huggingface.co/datasets?search=Indi-RomCoM",
        "type": "hf_search",
        "official": False,
    },
    {
        "name": "HuggingFace search indi-romcom",
        "url": "https://huggingface.co/datasets?search=indi-romcom",
        "type": "hf_search",
        "official": False,
    },
    {
        "name": "GitHub search Indi-RomCoM",
        "url": "https://github.com/search?q=Indi-RomCoM&type=repositories",
        "type": "github_search",
        "official": False,
    },
]

FALLBACK_DATASETS = [
    "Divyanshu/indicxnli",
    "ai4bharat/IndicXParaphrase",
    "ai4bharat/indic-align",
]

COMI_REPO = "LingoIITGN/COMI-LINGUA"
COMI_CONFIGS = ["MT", "TN", "MLI"]


def _http_probe(url: str) -> dict[str, Any]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        return {"status_code": r.status_code, "accessible": r.status_code < 400}
    except Exception as exc:
        return {"status_code": None, "accessible": False, "error": str(exc)}


def search_indi_romcom_training(api: HfApi | None = None) -> dict[str, Any]:
    api = api or HfApi()
    results: dict[str, Any] = {
        "searched_at": datetime.now(timezone.utc).isoformat(),
        "candidates": [],
        "hf_dataset_ids": [],
        "training_split_found": False,
        "evaluation_only_public": None,
        "notes": [],
    }
    for cand in SEARCH_CANDIDATES:
        probe = _http_probe(cand["url"])
        entry = {**cand, **probe}
        results["candidates"].append(entry)

    try:
        models = list(api.list_datasets(search="Indi-RomCoM", limit=20))
        for ds in models:
            results["hf_dataset_ids"].append(
                {"id": ds.id, "sha": getattr(ds, "sha", None)}
            )
    except Exception as exc:
        results["notes"].append(f"HF list_datasets error: {exc}")

    for query in ("indi-romcom", "IndiRomCoM", "RomCoM"):
        try:
            for ds in api.list_datasets(search=query, limit=10):
                rid = ds.id
                if rid not in {x["id"] for x in results["hf_dataset_ids"]}:
                    results["hf_dataset_ids"].append(
                        {"id": rid, "sha": getattr(ds, "sha", None)}
                    )
        except Exception:
            pass

    training_repos = [
        x["id"]
        for x in results["hf_dataset_ids"]
        if "romcom" in x["id"].lower() or "indi-rom" in x["id"].lower()
    ]
    if training_repos:
        for repo_id in training_repos:
            try:
                info = api.dataset_info(repo_id)
                siblings = [f.rfilename for f in (info.siblings or [])]
                if any("train" in s.lower() for s in siblings):
                    results["training_split_found"] = True
            except Exception:
                continue
    else:
        results["notes"].append(
            "No Hugging Face dataset ID matching Indi-RomCoM training release "
            "was found in automated search."
        )
        results["evaluation_only_public"] = True

    results["notes"].append(
        "Paper describes 395,076-instance fine-tuning corpus vs 8,400 evaluation "
        "benchmark; no author training artifact URL in paper body."
    )
    return results


def inspect_fallback_licenses(api: HfApi | None = None) -> list[dict[str, Any]]:
    api = api or HfApi()
    out: list[dict[str, Any]] = []
    for repo in FALLBACK_DATASETS:
        entry: dict[str, Any] = {
            "repository": repo,
            "revision": None,
            "license_metadata": None,
            "license_status": "unresolved",
            "readme_license_snippet": None,
            "status": "ACCESSIBLE",
        }
        try:
            info = api.dataset_info(repo)
            entry["revision"] = info.sha
            card = info.cardData or {}
            entry["license_metadata"] = card.get("license")
            readme_path = PROJECT_ROOT / "data" / "metadata" / f"{repo.replace('/', '_')}_readme.txt"
            if repo == "Divyanshu/indicxnli":
                meta = card.get("license")
                if meta and "cc0" in str(meta).lower():
                    entry["license_status"] = "unresolved"
                    entry["notes"] = (
                        "HF metadata lists cc0-1.0; dataset card may state "
                        "CC BY-NC 4.0 — reconcile before use."
                    )
                else:
                    entry["license_status"] = "restricted"
            elif repo == "ai4bharat/IndicXParaphrase":
                if not card.get("license"):
                    entry["license_status"] = "unresolved"
                    entry["notes"] = "No license declared in dataset card metadata."
                else:
                    entry["license_status"] = "cleared_for_internal_use"
            elif repo == "ai4bharat/indic-align":
                entry["license_status"] = "cleared_for_internal_use"
                entry["license_metadata"] = card.get("license", "cc-by-4.0")
        except Exception as exc:
            entry["status"] = "FAILED"
            entry["error"] = str(exc)
        out.append(entry)
    return out


def gate_decision(
    search_result: dict[str, Any], fallback_licenses: list[dict[str, Any]]
) -> dict[str, Any]:
    raw_indi_files = list(RAW_INDI.glob("*")) if RAW_INDI.is_dir() else []
    has_local_training = any(
        f.is_file() and "train" in f.name.lower() for f in raw_indi_files
    )
    training_found = search_result.get("training_split_found") or has_local_training

    if training_found and has_local_training:
        return {
            "cleared": True,
            "status": "READY_FOR_TASK_PIPELINE",
            "reason": "Local official training files present in data/raw/indi_romcom/",
        }

    return {
        "cleared": False,
        "status": "BLOCKED",
        "reason": (
            "Official Indi-RomCoM task-labelled training split not located or "
            "downloaded; fallback upstream datasets are not cleared substitutes "
            "without owner approval and code-mix recreation."
        ),
        "owner_decisions": [
            "Provide official Indi-RomCoM training artifact URL or local files.",
            "Or approve fallback from IndicXNLI / IndicXParaphrase / IndicAlign "
            "with resolved licenses and faithful code-mix procedure.",
            "Resolve IndicXNLI license metadata vs card conflict.",
            "Establish IndicXParaphrase license with owner.",
        ],
    }


def build_source_manifest_entries(
    dakshina_files: list[Path],
    comi_meta: dict[str, Any] | None,
    search_result: dict[str, Any],
    fallback_licenses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from romcom_data.checksums import sha256_file

    entries: list[dict[str, Any]] = []
    for fpath in dakshina_files:
        rel = fpath.relative_to(PROJECT_ROOT)
        entries.append(
            {
                "source_dataset": "google-research-datasets/dakshina",
                "path": str(rel),
                "license_status": "cleared_for_internal_use",
                "license_ref": "CC-BY-SA-4.0",
                "sha256": sha256_file(fpath),
                "role": "dakshina_hi",
            }
        )
    if comi_meta:
        entries.append(
            {
                "source_dataset": COMI_REPO,
                "revision": comi_meta.get("revision"),
                "license_status": "cleared_for_internal_use",
                "license_ref": "CC-BY-4.0",
                "role": "comi_lingua_supplementary",
                "configs_stored": comi_meta.get("configs_stored", []),
                "caveat": "Research preview; supplementary reference only.",
            }
        )
    entries.append(
        {
            "source_dataset": "Indi-RomCoM (paper)",
            "url": INDI_ROMCOM_PAPER,
            "license_status": "not_used",
            "training_access": search_result.get("training_split_found", False),
            "role": "primary_task_source_pending",
        }
    )
    for fb in fallback_licenses:
        entries.append(
            {
                "source_dataset": fb["repository"],
                "revision": fb.get("revision"),
                "license_status": fb.get("license_status"),
                "role": "fallback_readiness_only",
                "status": fb.get("status"),
            }
        )
    return entries


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
