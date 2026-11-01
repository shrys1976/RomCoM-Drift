"""Close NLI investigation with owner review outcomes (no alignment rerun)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from romcom_data.handoff_paths import MANIFESTS, NLI_INVESTIGATION, REPORTS

CLOSURE_REVISION = "2026_11_01_closed"
SAMPLE_VERSION = "investigation_2026_11_01_closed"
REVIEWER = "owner"

HYPOTHESIS_FAIL_INDICES = {357}
UNCERTAIN_INDICES = {684, 777, 844, 1027, 1327, 1841}

BUNDLE_FILES = (
    "nli_alignment_investigation_report.md",
    "nli_investigation_closure.md",
    "nli_alignment_manifest.csv",
    "nli_alignment_sample.csv",
    "nli_alignment_metrics.json",
    "nli_source_revisions.json",
    "nli_owner_review_summary.json",
    "alignment_method.py",
    "make_dataset.py",
    "alignment_audit.log",
    "nli_alignment_status.json",
    "README.txt",
)


def _apply_owner_judgments(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "bilingual_dev_rows_reviewed": 0,
        "pass_all_fields": 0,
        "hypothesis_fail": [],
        "uncertain_rows": [],
        "train_hindi_only_rows": 0,
    }
    for row in rows:
        kind = row.get("sample_row_kind", "")
        if kind == "hindi_only_train":
            summary["train_hindi_only_rows"] += 1
            continue
        if kind != "bilingual_dev":
            continue
        idx = int(row["indic_row_index"])
        summary["bilingual_dev_rows_reviewed"] += 1
        row["sample_version"] = SAMPLE_VERSION
        row["reviewer"] = REVIEWER
        if idx in HYPOTHESIS_FAIL_INDICES:
            row["owner_judgment_premise_aligned"] = "PASS"
            row["owner_judgment_hypothesis_aligned"] = "FAIL"
            row["owner_judgment_label_consistent"] = "UNCERTAIN"
            row["notes"] = "Independent review: hypothesis alignment FAIL."
            summary["hypothesis_fail"].append(idx)
        elif idx in UNCERTAIN_INDICES:
            row["owner_judgment_premise_aligned"] = "UNCERTAIN"
            row["owner_judgment_hypothesis_aligned"] = "UNCERTAIN"
            row["owner_judgment_label_consistent"] = "UNCERTAIN"
            row["notes"] = "Independent review: alignment UNCERTAIN."
            summary["uncertain_rows"].append(idx)
        else:
            row["owner_judgment_premise_aligned"] = "PASS"
            row["owner_judgment_hypothesis_aligned"] = "PASS"
            row["owner_judgment_label_consistent"] = "PASS"
            row["notes"] = "Independent review: PASS."
            summary["pass_all_fields"] += 1
    summary["uncertain_rows"].sort()
    summary["hypothesis_fail"].sort()
    return summary


def _write_sample(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _load_report_context() -> dict[str, Any]:
    ctx_path = MANIFESTS / "nli_investigation_context.json"
    if ctx_path.is_file():
        return json.loads(ctx_path.read_text(encoding="utf-8"))
    md = REPORTS / "nli_alignment_investigation_report.md"
    if md.is_file():
        text = md.read_text(encoding="utf-8")
        if text.startswith("#"):
            text = text.split("\n", 2)[-1]
        return json.loads(text)
    return {}


def _closure_report(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "investigation_revision": CLOSURE_REVISION,
        "investigation_closed": True,
        "root_cause": "unresolved",
        "further_alignment_attempts": (
            "None unless original historical source data (generation-time English "
            "artifact or verified revision) becomes available."
        ),
        "train": {
            "english_mappings_withheld": 392702,
            "mapping_reliable": False,
            "alternative_joins_attempted": False,
        },
        "dev": {
            "manifest_rows_provisional_same_index": 2490,
            "mapping_method": "same_index_xnli_10_en",
            "full_corpus_semantically_verified": False,
            "provisional_retention": True,
            "known_sample_level_exceptions": {
                "independent_review_sample_size": summary["bilingual_dev_rows_reviewed"],
                "hypothesis_fail_indic_row_indices": summary["hypothesis_fail"],
                "uncertain_indic_row_indices": summary["uncertain_rows"],
                "pass_all_judgment_fields": summary["pass_all_fields"],
            },
        },
        "owner_independent_review": summary,
        "limitations": [
            "Train English remains unavailable for RomCoM-X NLI training exports.",
            "Dev English is provisional same-index XNLI-1.0; not all 2490 rows were semantically reviewed.",
            "Historical train ordering mismatch cause remains unresolved.",
        ],
    }


def close_nli_investigation() -> dict[str, Any]:
    sample_src = NLI_INVESTIGATION / "nli_alignment_sample.csv"
    if not sample_src.is_file():
        sample_src = REPORTS / "nli_alignment_sample.csv"
    with sample_src.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    summary = _apply_owner_judgments(rows)
    for dest in (REPORTS / "nli_alignment_sample.csv", NLI_INVESTIGATION / "nli_alignment_sample.csv"):
        _write_sample(dest, rows, fieldnames)

    closure = _closure_report(summary)
    base_report = _load_report_context()
    base_report.update(closure)
    base_report["owner_review_sample_completed"] = True
    base_report["dev_semantic_review_complete"] = False
    base_report["dev_full_corpus_semantically_verified"] = False

    (REPORTS / "nli_alignment_investigation_report.md").write_text(
        "# NLI alignment investigation report (closed)\n\n"
        + json.dumps(base_report, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    closure_md = (
        "# NLI investigation closure\n\n"
        "The NLI alignment investigation is **closed**. No further alignment attempts unless "
        "original historical source data becomes available.\n\n"
        "## Decisions\n\n"
        "- **Train:** all 392,702 English mappings **rejected/withheld**.\n"
        "- **Dev:** 2,490 **provisional** same-index mappings retained.\n"
        "- **Owner sample (41 bilingual dev):** "
        f"{summary['pass_all_fields']} PASS, "
        f"{len(summary['hypothesis_fail'])} hypothesis FAIL "
        f"(indic_row_index {summary['hypothesis_fail']}), "
        f"{len(summary['uncertain_rows'])} UNCERTAIN "
        f"(indic_row_index {summary['uncertain_rows']}).\n\n"
        "**Not** all 2,490 dev examples are semantically verified.\n"
    )
    for path in (
        REPORTS / "nli_investigation_closure.md",
        NLI_INVESTIGATION / "nli_investigation_closure.md",
    ):
        path.write_text(closure_md, encoding="utf-8")

    review_summary_path = NLI_INVESTIGATION / "nli_owner_review_summary.json"
    review_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(review_summary_path, REPORTS / "nli_owner_review_summary.json")

    metrics_path = NLI_INVESTIGATION / "nli_alignment_metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update(
        {
            "investigation_closed": True,
            "investigation_revision": CLOSURE_REVISION,
            "train_english_mappings_withheld": 392702,
            "dev_provisional_same_index_rows": 2490,
            "dev_full_corpus_semantically_verified": False,
            "owner_independent_review": summary,
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(metrics_path, REPORTS / "nli_alignment_metrics.json")

    status = {
        "investigation_closed": True,
        "investigation_complete": True,
        "investigation_revision": CLOSURE_REVISION,
        "root_cause": "unresolved",
        "further_alignment_attempts": "none_unless_historical_source_recovered",
        "train_english_mappings_withheld": 392702,
        "train_mapping_reliable": False,
        "dev_provisional_same_index_retained": True,
        "dev_manifest_rows": 2490,
        "dev_mapping_method": "same_index_xnli_10_en",
        "dev_full_corpus_semantically_verified": False,
        "owner_independent_review_sample_size": summary["bilingual_dev_rows_reviewed"],
        "owner_review_hypothesis_fail_indices": summary["hypothesis_fail"],
        "owner_review_uncertain_indices": summary["uncertain_rows"],
        "owner_review_pass_count": summary["pass_all_fields"],
        "dev_semantic_review_complete": False,
        "dev_mapping_reliable": False,
        "dev_provisional_retention_approved": True,
        "owner_approved": False,
        "block_reason": "nli_train_english_unavailable",
        "note": (
            "Investigation closed. Dev mappings provisional with known sample exceptions. "
            "NLI training exports remain blocked (train English withheld)."
        ),
    }
    (MANIFESTS / "nli_alignment_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    (MANIFESTS / "nli_investigation_context.json").write_text(
        json.dumps(base_report, indent=2) + "\n", encoding="utf-8"
    )

    NLI_INVESTIGATION.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPORTS / "nli_alignment_investigation_report.md", NLI_INVESTIGATION / "nli_alignment_investigation_report.md")
    shutil.copy2(MANIFESTS / "nli_alignment_status.json", NLI_INVESTIGATION / "nli_alignment_status.json")
    if (MANIFESTS / "nli_alignment_manifest.csv").is_file():
        shutil.copy2(MANIFESTS / "nli_alignment_manifest.csv", NLI_INVESTIGATION / "nli_alignment_manifest.csv")

    readme = (
        "RomCoM-X Phase 2.1 NLI investigation — CLOSED.\n\n"
        "Train: 392702 English mappings withheld.\n"
        "Dev: 2490 provisional same-index mappings (not fully semantically verified).\n"
        "Owner sample: 41 bilingual dev rows with recorded judgments in nli_alignment_sample.csv.\n"
        "See nli_investigation_closure.md and nli_owner_review_summary.json.\n"
    )
    (NLI_INVESTIGATION / "README.txt").write_text(readme, encoding="utf-8")

    audit = NLI_INVESTIGATION / "alignment_audit.log"
    prior = audit.read_text(encoding="utf-8") if audit.is_file() else ""
    audit.write_text(
        prior
        + f"\nclosure_utc={datetime.now(UTC).isoformat()}\n"
        + f"closure_revision={CLOSURE_REVISION}\n"
        + json.dumps(closure, indent=2)
        + "\n",
        encoding="utf-8",
    )

    return {"closure": closure, "status": status, "owner_review_summary": summary}


def build_final_handoff_zip(zip_path: Path) -> Path:
    lines: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in BUNDLE_FILES:
            src = NLI_INVESTIGATION / name
            if not src.is_file():
                raise FileNotFoundError(f"Missing handoff file: {src}")
            data = src.read_bytes()
            zf.writestr(name, data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
        zf.writestr("checksums.sha256", "\n".join(lines) + "\n")
    return zip_path
