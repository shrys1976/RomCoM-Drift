"""Hindi-native IndicXNLI source audit. No English alignment and no EXP rebuild."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from romcom_data.handoff_paths import HANDOFF, INVENTORIES, RAW_NLI
from romcom_data.romanize import DakshinaRomanizer, residual_devanagari_chars, romanize_mixed_text
from romcom_data.stable_hash import derive_rng
from romcom_data.text_utils import DEVANAGARI, LATIN, nfc

NLI_NATIVE = HANDOFF / "nli_hindi_native"
LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
OFFICIAL_FILES = (
    ("train", "forward__train__xnli_hi.json", "train"),
    ("validation", "forward__dev__xnli_hi.json", "validation"),
    ("test", "forward__test__xnli_hi.json", "test"),
)
ENGLISH_BENCHMARK = {
    "role": "independent_language_retention_benchmark",
    "claim": "not_parallel_to_indicxnli_hindi_rows",
    "files": [
        "romcomx_data/raw/xnli_mt/XNLI-1.0/xnli.dev.tsv",
        "romcomx_data/raw/xnli_mt/XNLI-1.0/xnli.test.tsv",
    ],
}
PROVENANCE_ISSUE = (
    "The original English MultiNLI training snapshot used to generate IndicXNLI "
    "could not be recovered. Train English mappings (392,702) remain withheld. "
    "No further fuzzy, index-based, or approximate Hindi-English alignment is authorized."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(nfc(text).encode("utf-8")).hexdigest()


def _normalize_label(raw: Any) -> tuple[int | None, str]:
    if isinstance(raw, str) and raw.isdigit():
        raw = int(raw)
    if isinstance(raw, int) and raw in LABEL_MAP:
        return raw, LABEL_MAP[raw]
    return None, ""


def iter_official_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    file_meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for project_split, filename, _ in OFFICIAL_FILES:
        path = RAW_NLI / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        split_key, records = next(iter(payload.items()))
        file_meta[project_split] = {
            "filename": filename,
            "json_split_key": split_key,
            "sha256": _sha256_file(path),
            "n_rows": len(records),
        }
        for index, record in enumerate(records):
            premise = nfc(str(record.get("premise") or ""))
            hypothesis = nfc(str(record.get("hypothesis") or ""))
            label_id, label = _normalize_label(record.get("label"))
            source_id = f"indicxnli-hi:{project_split}:{index:06d}"
            rows.append(
                {
                    "source_id": source_id,
                    "example_id": source_id,
                    "project_split": project_split,
                    "official_partition": project_split,
                    "row_index": index,
                    "label_id": label_id,
                    "label": label,
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "premise_sha256": _text_sha(premise),
                    "hypothesis_sha256": _text_sha(hypothesis),
                    "source_file": filename,
                    "json_split_key": split_key,
                }
            )
    return rows, file_meta


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "source_id",
        "example_id",
        "project_split",
        "row_index",
        "label_id",
        "label",
        "premise_sha256",
        "hypothesis_sha256",
        "premise_chars",
        "hypothesis_chars",
        "source_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_id": row["source_id"],
                    "example_id": row["example_id"],
                    "project_split": row["project_split"],
                    "row_index": row["row_index"],
                    "label_id": "" if row["label_id"] is None else row["label_id"],
                    "label": row["label"],
                    "premise_sha256": row["premise_sha256"],
                    "hypothesis_sha256": row["hypothesis_sha256"],
                    "premise_chars": len(row["premise"]),
                    "hypothesis_chars": len(row["hypothesis"]),
                    "source_file": row["source_file"],
                }
            )


def split_integrity(rows: list[dict[str, Any]], file_meta: dict[str, Any]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["project_split"]].append(row)
    pair_owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    within_dups = Counter()
    for split, group in by_split.items():
        seen: set[tuple[str, str]] = set()
        for row in group:
            key = (row["premise_sha256"], row["hypothesis_sha256"])
            if key in seen:
                within_dups[split] += 1
            seen.add(key)
            pair_owners[key].add(split)
    cross = {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    for owners in pair_owners.values():
        if owners >= {"train", "validation"}:
            cross["train_validation"] += 1
        if owners >= {"train", "test"}:
            cross["train_test"] += 1
        if owners >= {"validation", "test"}:
            cross["validation_test"] += 1
    id_dups = len(rows) - len({row["source_id"] for row in rows})
    return {
        "official_files": file_meta,
        "counts": {split: len(group) for split, group in by_split.items()},
        "source_id_duplicates": id_dups,
        "within_split_duplicate_pairs": dict(within_dups),
        "cross_split_identical_premise_hypothesis": cross,
        "english_alignment": "not_used",
        "id_policy": "indicxnli-hi:{official_split}:{row_index} from Hindi JSON order only",
    }


def quality_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter((row["project_split"], row["label"] or "invalid") for row in rows)
    empty_premise = sum(1 for row in rows if not row["premise"].strip())
    empty_hypothesis = sum(1 for row in rows if not row["hypothesis"].strip())
    invalid_label = sum(1 for row in rows if row["label_id"] is None)
    low_dev = 0
    latin_heavy = 0
    for row in rows:
        text = row["premise"] + row["hypothesis"]
        dev = len(DEVANAGARI.findall(text))
        lat = len(LATIN.findall(text))
        total = dev + lat
        if total and dev / total < 0.5:
            low_dev += 1
        if total and lat / total > 0.2:
            latin_heavy += 1
    return {
        "n_examples": len(rows),
        "label_distribution": {f"{split}:{label}": count for (split, label), count in sorted(labels.items())},
        "empty_premise": empty_premise,
        "empty_hypothesis": empty_hypothesis,
        "invalid_label": invalid_label,
        "examples_devanagari_ratio_below_0_5": low_dev,
        "examples_latin_letter_ratio_above_0_2": latin_heavy,
        "label_integrity": "labels taken from Hindi JSON only; not checked against English XNLI",
        "translation_quality": (
            "Not scored against English. Hindi text is the observation. "
            "The 41-example bilingual dev review remains supporting evidence only."
        ),
        "bilingual_dev_review": {
            "n": 41,
            "pass": 34,
            "hypothesis_fail_indices": [357],
            "uncertain_indices": [684, 777, 844, 1027, 1327, 1841],
            "role": "supporting_evidence_not_corpus_alignment_proof",
        },
    }


def transformation_feasibility(rows: list[dict[str, Any]], sample_n: int = 300) -> dict[str, Any]:
    rng = derive_rng(42, "nli-hindi-native", "feasibility")
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["project_split"] == "train" and row["label"]:
            by_label[row["label"]].append(row)
    sample: list[dict[str, Any]] = []
    per_label = sample_n // max(len(by_label), 1)
    for label, bucket in sorted(by_label.items()):
        chosen = list(bucket)
        rng.shuffle(chosen)
        sample.extend(chosen[:per_label])
    romanizer = DakshinaRomanizer(INVENTORIES / "dakshina_hi_variant_inventory.tsv")
    methods = Counter()
    residual = 0
    fields = 0
    for row in sample:
        for field in ("premise", "hypothesis"):
            romanized, traces = romanize_mixed_text(row[field], romanizer)
            fields += 1
            residual += residual_devanagari_chars(romanized.replace("\u0964", "").replace("\u0965", ""))
            for trace in traces:
                methods[str(trace.get("method") or trace.get("operation"))] += 1
    lexicon = methods.get("dakshina_canonical", 0)
    fallback = sum(count for name, count in methods.items() if name.startswith("fallback"))
    dev_tokens = lexicon + fallback
    return {
        "sample_examples": len(sample),
        "sample_fields": fields,
        "sample_seed": 42,
        "applies_to": "premise and hypothesis (both Hindi observations)",
        "conditions": {
            "EXP-01": "canonical Dakshina romanization of Hindi tokens",
            "EXP-02": "attested non-canonical Dakshina variants",
            "EXP-03": "matched-distance noise on EXP-02 selected tokens",
        },
        "token_methods": dict(methods),
        "dictionary_coverage_pct": round(100 * lexicon / dev_tokens, 2) if dev_tokens else 0,
        "residual_devanagari_chars_excluding_danda": residual,
        "feasible": dev_tokens > 0,
        "pre_build_note": (
            "A few Devanagari characters can remain after the current fallback map. "
            "That is a construction fix before EXP generation, not a reason to reject "
            "the Hindi-native source strategy."
            if residual
            else "Sample left no residual Devanagari letters."
        ),
        "not_run": "full EXP dataset generation and training",
    }


def write_documents(
    rows: list[dict[str, Any]],
    file_meta: dict[str, Any],
    integrity: dict[str, Any],
    quality: dict[str, Any],
    feasibility: dict[str, Any],
) -> str:
    NLI_NATIVE.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, NLI_NATIVE / "hindi_source_manifest.csv")
    blocked: list[str] = []
    if integrity["source_id_duplicates"]:
        blocked.append("duplicate source ids")
    if quality["empty_premise"] or quality["empty_hypothesis"] or quality["invalid_label"]:
        blocked.append("empty text or invalid labels")
    if any(integrity["cross_split_identical_premise_hypothesis"].values()):
        blocked.append("identical premise-hypothesis pairs cross official splits")
    if not feasibility["feasible"]:
        blocked.append("romanization sample produced no Devanagari lexicon or fallback tokens")
    status = "BLOCKED" if blocked else "READY_FOR_OWNER_REVIEW"
    (NLI_NATIVE / "task_definition.md").write_text(
        "\n".join(
            [
                "# Hindi-native NLI task definition",
                "",
                "This experiment measures **Romanized Hindi semantic understanding**.",
                "",
                "It does **not** establish Hindi–English code-mixed semantic understanding.",
                "",
                "Each IndicXNLI Hindi premise–hypothesis pair is an independent NLI observation.",
                "Official Hindi train, validation (dev file), and test partitions are kept.",
                "No English counterpart is required, and none is joined to these rows.",
                "",
                "EXP-01, EXP-02, and EXP-03, when later authorized, transform the Hindi premise and hypothesis only.",
                "Labels and official split assignments stay fixed across conditions.",
                "",
                "## Unresolved provenance",
                "",
                PROVENANCE_ISSUE,
                "",
                "The existing 41-example bilingual dev review (34 pass, 1 hypothesis fail, 6 uncertain) is supporting evidence only. It is not proof that the corpus is aligned to English.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (NLI_NATIVE / "split_integrity_report.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )
    (NLI_NATIVE / "label_quality_report.json").write_text(
        json.dumps(quality, indent=2) + "\n", encoding="utf-8"
    )
    (NLI_NATIVE / "transformation_feasibility.json").write_text(
        json.dumps(feasibility, indent=2) + "\n", encoding="utf-8"
    )
    (NLI_NATIVE / "experiment_protocol.md").write_text(
        "\n".join(
            [
                "# Revised NLI experiment protocol",
                "",
                "## Claim",
                "",
                "Romanized Hindi semantic understanding on official IndicXNLI Hindi partitions.",
                "",
                "## Not claimed",
                "",
                "Hindi–English code-mixed semantic understanding. English XNLI validation and test stay a separate language-retention benchmark and are not treated as parallel rows.",
                "",
                "## Data",
                "",
                "- Hindi premise, hypothesis, and label from `forward__*__xnli_hi.json`.",
                "- Splits: train, validation (dev file), test. No regrouping.",
                "- Source IDs: `indicxnli-hi:{split}:{row_index}` from Hindi file order.",
                "- EXP-01/02/03 (not built in this step) share those IDs, labels, and splits. Only the romanization of premise and hypothesis may differ.",
                "",
                "## English benchmark",
                "",
                json.dumps(ENGLISH_BENCHMARK, indent=2),
                "",
                "## Provenance limit",
                "",
                PROVENANCE_ISSUE,
                "",
                "## This step",
                "",
                "Audit and protocol only. No dataset rebuild and no training.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (NLI_NATIVE / "unresolved_english_provenance.md").write_text(
        "# Unresolved English training alignment\n\n" + PROVENANCE_ISSUE + "\n\n"
        "Dev same-index mappings previously marked provisional are not used by this Hindi-native strategy.\n"
        "See `romcomx_data/nli_investigation/nli_alignment_status.json`.\n",
        encoding="utf-8",
    )
    summary = {
        "status": status,
        "blockers": blocked,
        "audited_utc": datetime.now(UTC).isoformat(),
        "n_examples": len(rows),
        "counts": integrity["counts"],
        "research_claim": "romanized_hindi_semantic_understanding",
        "not_claimed": "hindi_english_code_mixed_semantic_understanding",
        "english_training_alignment": "unresolved_withheld",
        "rebuild": "not_run",
        "training": "not_run",
    }
    (NLI_NATIVE / "status.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (NLI_NATIVE / "split_integrity_report.md").write_text(
        "# Official Hindi split integrity\n\n```json\n"
        + json.dumps(integrity, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (NLI_NATIVE / "label_quality_report.md").write_text(
        "# Label distribution and data quality\n\n```json\n"
        + json.dumps(quality, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    (NLI_NATIVE / "transformation_feasibility.md").write_text(
        "# Transformation feasibility\n\n"
        "Sample romanization of Hindi premise and hypothesis. Full EXP files were not built.\n\n```json\n"
        + json.dumps(feasibility, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return status
