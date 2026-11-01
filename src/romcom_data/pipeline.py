from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from load_env import bootstrap_project  # noqa: E402

from romcom_data.checksums import write_checksum_manifest
from romcom_data.comi_lingua import audit_comi, download_comi_snapshots
from romcom_data.config_loader import load_augmentation_config
from romcom_data.dakshina import full_dakshina_audit, save_audit_json
from romcom_data.inventory import build_variant_inventory
from romcom_data.paths import (
    CONFIG,
    INTERIM,
    MANIFESTS,
    PROJECT_ROOT,
    RAW_DAKSHINA_HI,
    ensure_layout,
)
from romcom_data.reports import write_all_reports
from romcom_data.source_discovery import (
    build_source_manifest_entries,
    gate_decision,
    inspect_fallback_licenses,
    save_json,
    search_indi_romcom_training,
)
from romcom_data.tasks import run_task_pipeline_if_cleared


def list_raw_dakshina_files() -> list[Path]:
    return sorted(RAW_DAKSHINA_HI.glob("*"))


def run_pytest() -> dict[str, str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "exit_code": str(result.returncode),
        "stdout": result.stdout[-4000:] if result.stdout else "",
        "passed": "passed" if result.returncode == 0 else "failed",
    }


def main() -> int:
    bootstrap_project()
    ensure_layout()
    config, config_hash = load_augmentation_config()

    api = HfApi()
    search_result = search_indi_romcom_training(api)
    fallback_licenses = inspect_fallback_licenses(api)
    gate = gate_decision(search_result, fallback_licenses)

    save_json(MANIFESTS / "indi_romcom_search.json", search_result)
    save_json(MANIFESTS / "fallback_license_inspection.json", fallback_licenses)
    save_json(MANIFESTS / "gate_decision.json", gate)

    comi_meta = download_comi_snapshots()
    comi_audit = audit_comi()

    dakshina_audit = full_dakshina_audit()
    save_audit_json(dakshina_audit, INTERIM / "dakshina_audit.json")

    lex_path = PROJECT_ROOT / config["inventory"]["source_file"]
    inventory_stats = build_variant_inventory(lex_path, config)

    invalid_path = INTERIM / "romanization" / "invalid_lexicon_rows.tsv"
    if invalid_path.is_file():
        import pandas as pd

        inv_invalid = pd.read_csv(invalid_path, sep="\t")
        inv_invalid["exclusion_status"] = "excluded"
        inv_invalid["exclusion_reason"] = inv_invalid.get("reasons", "invalid_lexicon")
        inv_invalid.to_csv(MANIFESTS / "exclusion_log.csv", index=False)

    task_result = run_task_pipeline_if_cleared(gate)

    raw_files = list_raw_dakshina_files()
    rel_paths = [p.relative_to(PROJECT_ROOT) for p in raw_files]
    write_checksum_manifest(PROJECT_ROOT, rel_paths, MANIFESTS / "checksum_manifest.sha256")

    source_entries = build_source_manifest_entries(
        raw_files, comi_meta, search_result, fallback_licenses
    )
    save_json(MANIFESTS / "source_manifest.json", {"sources": source_entries})

    aug_manifest = {
        "config_path": str(CONFIG.relative_to(PROJECT_ROOT)),
        "config_sha256": config_hash,
        "english_wordlist_sha256": config.get("_english_wordlist_sha256"),
        "conditions_generated": [] if not gate.get("cleared") else ["exp01", "exp02", "exp03"],
        "gate_status": gate.get("status"),
    }
    save_json(MANIFESTS / "augmentation_manifest.json", aug_manifest)

    save_json(
        MANIFESTS / "processed_dataset_manifest.json",
        {"processed": [], "blocked": not gate.get("cleared")},
    )

    test_results = run_pytest()

    ctx = {
        "gate": gate,
        "search_result": search_result,
        "fallback_licenses": fallback_licenses,
        "dakshina_audit": dakshina_audit,
        "inventory_stats": {k: v for k, v in inventory_stats.items() if k not in ("alias_df", "inv_df")},
        "comi_audit": comi_audit,
        "config_hash": config_hash,
        "test_results": test_results,
        "task_result": task_result,
        "final_status": "BLOCKED" if not gate.get("cleared") else "NOT_READY",
    }
    write_all_reports(ctx)

    print(json.dumps({"gate": gate, "inventory": ctx["inventory_stats"], "tests": test_results}, indent=2))
    return 0 if gate.get("cleared") or gate.get("status") == "BLOCKED" else 1


if __name__ == "__main__":
    sys.exit(main())
