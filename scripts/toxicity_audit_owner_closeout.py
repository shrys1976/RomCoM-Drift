#!/usr/bin/env python3
"""Refresh comparability metadata and inspect the cross-class prompt group.

Does not rebuild processed data, change the frozen manifest, augment, train, or score.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.handoff_paths import RAW_TOX, TOXICITY_AUDIT
from romcom_data.toxicity_audit import (
    build_audit_zip,
    build_class_comparability_document,
    load_jsonl,
)
from romcom_data.toxicity_freeze import (
    EXCLUDED_TOXIC_DOC_IDS,
    _embed_users,
    _member_id_for_row,
    _pair,
    group_retained_user_prompts,
    inspect_cross_class_mixed_group,
)

CROSS_CLASS_GROUP_ANCHOR = "benign-2eedabfa-c7f0-40e6-a175-270c923c8639"


def _load_manifest_splits() -> dict[str, str]:
    path = TOXICITY_AUDIT / "final_source_manifest.csv"
    split_by_member: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            split_by_member[row["source_id"]] = row.get("project_split") or ""
    return split_by_member


def _cross_class_member_ids(grouped: dict) -> list[str]:
    for members in grouped["multi_member_group_ids"].values():
        classes = {
            "toxic" if member.startswith("toxic-") else "benign" for member in members
        }
        if len(classes) > 1 and CROSS_CLASS_GROUP_ANCHOR in members:
            return members
    raise RuntimeError("cross-class group not found")


def main() -> int:
    TOXICITY_AUDIT.mkdir(parents=True, exist_ok=True)
    comparability_path = TOXICITY_AUDIT / "class_comparability.json"
    prior = None
    if comparability_path.is_file():
        prior = json.loads(comparability_path.read_text(encoding="utf-8"))

    toxic_rows = load_jsonl(RAW_TOX / "Toxic_Matrix_train.jsonl")
    benign_rows = load_jsonl(TOXICITY_AUDIT / "dolly_t_seed42_sample.jsonl")
    kept_toxic = [row for row in toxic_rows if row["doc_id"] not in EXCLUDED_TOXIC_DOC_IDS]

    comparability = build_class_comparability_document(
        kept_toxic, benign_rows, prior=prior
    )
    comparability_path.write_text(
        json.dumps(comparability, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    all_rows = kept_toxic + benign_rows
    users = [_pair(row, "hin_Deva")[0] for row in all_rows]
    vectors = _embed_users(users)
    grouped = group_retained_user_prompts(
        kept_toxic, benign_rows, vectors, record_edges=True
    )
    member_ids = _cross_class_member_ids(grouped)
    split_by_member = _load_manifest_splits()
    assistant_by_member: dict[str, str] = {}
    for row in kept_toxic:
        member = _member_id_for_row(row, source="toxic")
        assistant_by_member[member] = _pair(row, "hin_Deva")[1]
    for row in benign_rows:
        member = _member_id_for_row(row, source="benign")
        assistant_by_member[member] = _pair(row, "hin_Deva")[1]

    inspection = inspect_cross_class_mixed_group(
        grouped,
        member_ids=member_ids,
        split_by_member=split_by_member,
        assistant_by_member=assistant_by_member,
    )
    inspection["interpretation"] = {
        "cluster_shape": (
            "118 toxic rows form a dense subgraph (mostly embedding_cosine and "
            "shared \"क्या आप…\" token-family links). Four benign rows are Dolly "
            "paragraph-reading questions; they are not exact or prefix duplicates."
        ),
        "cross_class_bridge": (
            "Cross-class union is almost entirely embedding_cosine >= 0.85, chiefly "
            "through benign-3a65dba9 (hub). Many toxic rows have one-hop embedding "
            "links to that benign prompt despite unrelated semantics."
        ),
        "semantic_near_duplicate_verdict": (
            "Not a genuine cross-class semantic near-duplicate cluster; likely "
            "spurious embedding hub plus toxic template transitivity. Train-only "
            "supervision conflict (refusal vs ordinary answer) remains real."
        ),
        "automatic_removal": False,
    }
    inspection["manifest_unchanged"] = True
    inspection["manifest_sha256"] = json.loads(
        (TOXICITY_AUDIT / "final_source_manifest_summary.json").read_text(encoding="utf-8")
    )["manifest_sha256"]
    inspection_path = TOXICITY_AUDIT / "cross_class_group_inspection.json"
    inspection_path.write_text(
        json.dumps(inspection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    checklist = _checklist(inspection)
    (TOXICITY_AUDIT / "owner_decision_checklist.md").write_text(checklist, encoding="utf-8")

    recommendation = json.loads(
        (TOXICITY_AUDIT / "recommendation.json").read_text(encoding="utf-8")
    )
    recommendation["blockers"] = [
        "owner_accept_cross_class_supervision_findings",
    ]
    recommendation["owner_decisions"] = [
        "Frozen manifest accepted provisionally. Augmentation stays blocked until "
        "the owner reviews cross_class_group_inspection.json and accepts or rejects "
        "training with the 122-member mixed prompt group on train.",
        "class_comparability.json was refreshed to the seed-42 benign sample and "
        "hin_Deva interface lengths. Prior eng_Latn / prefix-sampling text is "
        "historical_superseded only.",
    ]
    (TOXICITY_AUDIT / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8"
    )
    status = {
        "status": "READY_FOR_OWNER_REVIEW",
        "blockers": recommendation["blockers"],
        "dataset_rebuild_allowed": False,
        "training_allowed": False,
        "augmentation_allowed": False,
        "scoring_allowed": False,
        "owner_review_scope": "toxicity_hindi_native_target_side_mixing",
        "source_manifest_frozen": True,
        "source_manifest_provisional_acceptance": True,
        "class_comparability_refreshed": True,
    }
    (TOXICITY_AUDIT / "toxicity_audit_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    zip_path = build_audit_zip(ROOT / "RomCoM_X_Phase_2_1_Toxicity_Audit.zip")
    print(json.dumps({"comparability": comparability_path.name, "inspection": inspection_path.name}, indent=2))
    print(json.dumps(
        {
            "benign_prompts": len(inspection["benign_members"]),
            "intra_group_edges": inspection["intra_group_edges"],
            "split_counts": inspection["project_split_counts"],
        },
        indent=2,
    ))
    print(f"Wrote {zip_path}")
    return 0


def _checklist(inspection: dict) -> str:
    manifest_sha = inspection["manifest_sha256"]
    splits = inspection["project_split_counts"]
    return f"""# Owner decision checklist

Verdict remains CONDITIONAL_GO. Frozen manifest accepted **provisionally**. Augmentation, training, and scoring stay blocked.

## Closed by this closeout

- [x] `class_comparability.json` refreshed: seed-42 Dolly benign sample, hin_Deva interface lengths, corrected sampling confound.
- [x] Prior eng_Latn / Dolly prefix comparability retained under `historical_superseded` only.
- [x] Cross-class 122-member group inspected: `cross_class_group_inspection.json` (prompts, paths, link evidence). No manifest change.

## Already verified (owner)

- [x] Manifest SHA-256 `{manifest_sha}`
- [x] 6,998 retained / 2 excluded; 80/10/10 split; zero cross-split groups
- [x] Seed-42 benign sample matches final manifest

## Confirm before augmentation

- [ ] Read `cross_class_group_inspection.json` for the four benign prompts, toxic counterparts, and link chains.
- [ ] Accept or reject **conflicting supervision** from the mixed train-only group (118 toxic refusals + 4 benign ordinary answers; splits in group: {splits}).
- [ ] If rejected, request a manifest regrouping or exclusion policy **before** augmentation (new manifest hash required).

## Do not approve

- Using validation as the final test split.
- Classification accuracy as an outcome.
- Romanized-input understanding or general code-mixed safety robustness claims.
"""


if __name__ == "__main__":
    raise SystemExit(main())
