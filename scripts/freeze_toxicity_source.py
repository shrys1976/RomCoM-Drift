#!/usr/bin/env python3
"""Freeze the toxicity source manifest. No rebuild, augmentation, training, or scoring."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.handoff_paths import RAW_TOX, TOXICITY_AUDIT
from romcom_data.toxicity_audit import (
    build_audit_zip,
    build_recommendation,
    determine_status,
    load_jsonl,
    task_definition,
)
from romcom_data.toxicity_freeze import (
    EXCLUDED_TOXIC_DOC_IDS,
    _pair,
    assign_group_splits,
    build_freeze_records,
    checksum_file,
    cross_split_groups,
    generation_rubric,
    group_retained_user_prompts,
    naive_row_leak_count,
    write_manifest,
    _embed_users,
)

TIE_DISPOSITION = "retain_on_source_list_order"
FROZEN_RATIOS = (0.8, 0.1, 0.1)


def main() -> int:
    TOXICITY_AUDIT.mkdir(parents=True, exist_ok=True)
    toxic_rows = load_jsonl(RAW_TOX / "Toxic_Matrix_train.jsonl")
    benign_rows = load_jsonl(TOXICITY_AUDIT / "dolly_t_seed42_sample.jsonl")
    ties = json.loads((TOXICITY_AUDIT / "benign_embedding_ties.json").read_text(encoding="utf-8"))
    tie_ids = {item["doc_id"] for item in ties}

    kept_toxic = [row for row in toxic_rows if row["doc_id"] not in EXCLUDED_TOXIC_DOC_IDS]
    all_rows = kept_toxic + benign_rows
    users = [_pair(row, "hin_Deva")[0] for row in all_rows]
    vectors = _embed_users(users)
    grouped = group_retained_user_prompts(kept_toxic, benign_rows, vectors)
    group_of = grouped["group_of"]
    retained, excluded = build_freeze_records(
        toxic_rows,
        benign_rows,
        tie_doc_ids=tie_ids,
        tie_disposition=TIE_DISPOSITION,
        toxic_group_of=group_of,
        benign_group_of=group_of,
    )
    frozen_assignment = assign_group_splits(retained, ratios=FROZEN_RATIOS)
    for record in retained:
        record["project_split"] = frozen_assignment[record["group_id"]]
    manifest_rows = retained + excluded
    manifest_path = TOXICITY_AUDIT / "final_source_manifest.csv"
    manifest_digest = write_manifest(manifest_path, manifest_rows)

    split_counts = Counter(row["project_split"] or "excluded" for row in manifest_rows)
    label_split = Counter((row["label"], row["project_split"] or "excluded") for row in manifest_rows)
    summary = {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "task": "hindi_refusal_response_generation",
        "claim": (
            "Hindi refusal-response generation with Romanized assistant targets. "
            "The Hindi Devanagari user input remains unchanged."
        ),
        "not_claimed": [
            "Romanized-input understanding",
            "general code-mixed safety robustness",
            "general toxicity detection",
            "classification accuracy",
        ],
        "licenses_approved": {
            "indic_align": "CC-BY-4.0",
            "dolly_15k_upstream": "CC-BY-SA-3.0",
            "flattened": False,
            "approved_on": "2026-11-01",
        },
        "tie_disposition": TIE_DISPOSITION,
        "tie_count": len(ties),
        "excluded_toxic_role_conflicts": list(EXCLUDED_TOXIC_DOC_IDS),
        "counts": {
            "toxic_source_rows": len(toxic_rows),
            "toxic_excluded_role_conflict": len(EXCLUDED_TOXIC_DOC_IDS),
            "toxic_retained": sum(1 for row in retained if row["label"] == "toxic"),
            "benign_seeded_rows": len(benign_rows),
            "benign_ties_retained": len(tie_ids),
            "benign_quarantined": 0,
            "retained_total": len(retained),
            "excluded_total": len(excluded),
            "manifest_rows": len(manifest_rows),
        },
        "frozen_split": {
            "rule": (
                "80/10/10 train/validation/test at prompt-group level, stratified by label "
                "(mixed groups assigned using the toxic pool if any member is toxic), seed 42. "
                "Validation is for tuning only. Test is the held-out final evaluation split."
            ),
            "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "external_holdout": None,
            "counts": dict(split_counts),
            "by_label": {
                f"{label}:{split}": count for (label, split), count in sorted(label_split.items())
            },
            "groups_crossing_frozen_splits": cross_split_groups(retained, frozen_assignment),
        },
        "manifest_sha256": checksum_file(manifest_path),
        "manifest_id": manifest_digest,
        "augmentation": "not_run",
        "training": "not_run",
        "scoring": "not_run",
    }
    leakage = {
        "population": grouped["population"],
        "excluded_before_grouping": list(EXCLUDED_TOXIC_DOC_IDS),
        "method": (
            "Single union-find over retained toxic and benign Hindi user prompts. "
            "Links: exact normalized text, identical 80-character prefixes, "
            "first-6-token families with full token Jaccard at least 0.5, "
            "and embedding cosine at least 0.85. Source IDs are not rewritten."
        ),
        "n_toxic": len(kept_toxic),
        "n_benign": len(benign_rows),
        "n_total": grouped["n"],
        "group_count": grouped["group_count"],
        "multi_member_groups": grouped["multi_member_groups"],
        "rows_in_multi_member_groups": grouped["rows_in_multi_member_groups"],
        "cross_class_multi_member_groups": grouped["cross_class_multi_member_groups"],
        "largest_group": grouped["largest_group"],
        "links": grouped["links"],
        "first6_token_family_size_histogram": grouped["first6_token_family_size_histogram"],
        "near_duplicate_cosine": grouped["near_duplicate_cosine"],
        "embedding_model": grouped["embedding_model"],
        "frozen_group_cross_split_count": summary["frozen_split"]["groups_crossing_frozen_splits"],
        "near_duplicate_groups_that_would_cross_if_split_by_row": naive_row_leak_count(
            retained, FROZEN_RATIOS
        ),
        "multi_member_groups_detail": [
            {
                "group_id": key,
                "member_ids": members,
                "size": len(members),
                "cross_class": len(
                    {
                        "toxic" if member.startswith("toxic-") else "benign"
                        for member in members
                    }
                )
                > 1,
            }
            for key, members in sorted(
                grouped["multi_member_group_ids"].items(), key=lambda item: -len(item[1])
            )
        ],
    }
    disposition = {
        "decision": TIE_DISPOSITION,
        "reason": (
            "All five benign ties have a higher same-slot cosine sum than the crossed-slot sum. "
            "They missed the 0.05 conflict margin. Reading both languages shows slot 0 is the "
            "question and slot 1 is the answer. They are retained on source-list role order and "
            "are not quarantined."
        ),
        "rows": [
            {
                "doc_id": item["doc_id"],
                "source_row_index": item["source_row_index"],
                "direct_cosine_sum": item["direct_cosine_sum"],
                "crossed_cosine_sum": item["crossed_cosine_sum"],
                "margin": item["margin"],
                "disposition": TIE_DISPOSITION,
            }
            for item in ties
        ],
    }
    rubric = generation_rubric()
    (TOXICITY_AUDIT / "final_source_manifest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (TOXICITY_AUDIT / "prompt_group_leakage.json").write_text(
        json.dumps(leakage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (TOXICITY_AUDIT / "benign_tie_disposition.json").write_text(
        json.dumps(disposition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (TOXICITY_AUDIT / "generation_evaluation_rubric.json").write_text(
        json.dumps(rubric, indent=2) + "\n", encoding="utf-8"
    )
    checklist = _checklist(summary, disposition)
    (TOXICITY_AUDIT / "owner_decision_checklist.md").write_text(checklist, encoding="utf-8")
    (TOXICITY_AUDIT / "task_definition.md").write_text(
        "# Hindi refusal-response generation\n\n"
        "The experiment is Hindi refusal-response generation with Romanized assistant targets. "
        "The Hindi Devanagari user input remains unchanged. It is not Romanized-input "
        "understanding and it is not general code-mixed safety robustness.\n\n"
        + json.dumps(task_definition(), indent=2)
        + "\n",
        encoding="utf-8",
    )
    recommendation = build_recommendation(
        {"row_count": 5000, "empty_user_prompt_after_parse": 0},
        {"benign_file": "dolly_t_seed42_sample.jsonl"},
        {"doc_id_overlap_toxic_benign": 0},
        {},
        {"toxic_matrix_config_present": True},
    )
    recommendation["source_manifest"] = summary["manifest_sha256"]
    recommendation["tie_disposition"] = TIE_DISPOSITION
    (TOXICITY_AUDIT / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
    )
    status = determine_status(recommendation)
    status["source_manifest_frozen"] = True
    status["augmentation_allowed"] = False
    status["training_allowed"] = False
    status["scoring_allowed"] = False
    (TOXICITY_AUDIT / "toxicity_audit_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    license_path = TOXICITY_AUDIT / "license_evidence.json"
    license_evidence = json.loads(license_path.read_text(encoding="utf-8"))
    declarations = license_evidence.setdefault("declarations_not_flattened", {})
    for key in ("indic_align", "dolly_15k_upstream"):
        if key in declarations:
            declarations[key]["owner_approval"] = "approved_2026-11-01"
    license_evidence["owner_approved_on"] = "2026-11-01"
    license_evidence["flattened"] = False
    license_path.write_text(json.dumps(license_evidence, indent=2) + "\n", encoding="utf-8")
    zip_path = build_audit_zip(ROOT / "RomCoM_X_Phase_2_1_Toxicity_Audit.zip")
    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["frozen_split"], indent=2))
    print(json.dumps(
        {
            "groups_crossing_splits": leakage["frozen_group_cross_split_count"],
            "cross_class_groups": leakage["cross_class_multi_member_groups"],
            "row_wise_leak_groups": leakage["near_duplicate_groups_that_would_cross_if_split_by_row"],
        },
        indent=2,
    ))
    print(f"Wrote {zip_path}")
    return 0


def _checklist(summary: dict, disposition: dict) -> str:
    counts = summary["counts"]
    split = summary["frozen_split"]["counts"]
    return f"""# Owner decision checklist

Verdict remains CONDITIONAL_GO. Task definition and source decisions approved. Licenses were approved on 2026-11-01 and are not flattened.

## Recorded in this freeze

- [x] IndicAlign CC-BY-4.0 approved, kept separate from Dolly-15K CC-BY-SA-3.0.
- [x] Dolly-15K upstream CC-BY-SA-3.0 approved for the benign control only.
- [x] Experiment name: Hindi refusal-response generation with Romanized assistant targets.
- [x] Hindi Devanagari user input stays unchanged.
- [x] Not Romanized-input understanding. Not general code-mixed safety robustness.
- [x] Five benign embedding ties retained on source-list role order. Not quarantined.
- [x] Two toxic role-conflict rows excluded. Roles were not flipped.
- [x] Toxic and benign user prompts grouped in one union-find before the split.
- [x] Frozen 80/10/10 train/validation/test. Validation is not the final test.
- [x] Source manifest frozen. Augmentation, training, and scoring were not run.
- [x] Future scores use the generation rubric. Benign over-refusal is required. Classification accuracy is not an outcome.

## Confirm before any augmentation

- [ ] Accept manifest `{summary["manifest_sha256"]}` as the freeze.
- [ ] Accept retained {counts["retained_total"]} rows: toxic {counts["toxic_retained"]}, benign {counts["benign_seeded_rows"]}.
- [ ] Accept excluded {counts["excluded_total"]} toxic role-conflict rows.
- [ ] Accept frozen splits: train {split.get("train", 0)}, validation {split.get("validation", 0)}, test {split.get("test", 0)}.
- [ ] Accept tie disposition: {disposition["decision"]}.
- [ ] Accept that benign over-refusal is a required generation measure.

## Do not approve

- Using validation as the final held-out evaluation split.
- Classification accuracy as a result.
- A claim about Romanized-input understanding.
- A claim of general code-mixed safety robustness.
"""


if __name__ == "__main__":
    raise SystemExit(main())
