#!/usr/bin/env python3
"""Write recorded semantic judgments and the conditional-approval exclusions.

Judgments are a manual reading of semantic_review_sample.csv. This script does
not score pairs with a model. It only records that reading and the row sets
the validation report excludes.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "romcomx_data" / "nli_alignment"
SAMPLE = OUT / "semantic_review_sample.csv"
JSONL = OUT / "verified_pairs.jsonl"
WORD = re.compile(r"\S+")

# sample_id -> premise, hypothesis, label, note
# Values: equivalent, questionable, not_equivalent
OVERRIDES: dict[str, tuple[str, str, str, str]] = {
    "0": ("questionable", "equivalent", "equivalent", "Opening clause dropped; the Lawrence sentence matches."),
    "1": ("questionable", "equivalent", "questionable", "vain rendered as useless, and squashing rendered as finding."),
    "2": ("not_equivalent", "equivalent", "equivalent", "Abdication became expulsion; proclamation of the Republic was dropped."),
    "4": ("not_equivalent", "equivalent", "equivalent", "Gore was rendered as fair-skinned (गोर)."),
    "6": ("questionable", "equivalent", "equivalent", "An intended return was rendered as already completed."),
    "10": ("questionable", "equivalent", "equivalent", "owns up to was rendered as ownership."),
    "11": ("not_equivalent", "equivalent", "questionable", "I've had enough is not what the Hindi premise says."),
    "12": ("questionable", "equivalent", "equivalent", "Greek family beaches became among Greek families."),
    "17": ("questionable", "equivalent", "questionable", "conjure was rendered as imagine."),
    "21": ("questionable", "equivalent", "equivalent", "The unit minutes was dropped from 11 or 12."),
    "25": ("questionable", "equivalent", "equivalent", "same here became right here the same way."),
    "27": ("questionable", "equivalent", "equivalent", "talking heads became talking chiefs; the first-instance claim remains."),
    "32": ("equivalent", "not_equivalent", "not_equivalent", "seat of his pants was literalized, so the contradiction is lost."),
    "42": ("equivalent", "not_equivalent", "not_equivalent", "Fascination was reversed into the visitors having a hobby."),
    "45": ("questionable", "equivalent", "equivalent", "talk back was weakened to talking."),
    "47": ("questionable", "equivalent", "equivalent", "all roads led was lost; the Tokyo rename still contradicts the hypothesis."),
    "50": ("questionable", "equivalent", "equivalent", "Bedside table became table."),
    "55": ("questionable", "equivalent", "equivalent", "stubbled face became a stubborn face."),
    "57": ("questionable", "questionable", "questionable", "cynicism was rendered as whim (सनक)."),
    "61": ("questionable", "equivalent", "equivalent", "wheel and deal was split into wheel and deal."),
    "64": ("not_equivalent", "equivalent", "not_equivalent", "bow out became bend, so it no longer entails retreat."),
    "65": ("equivalent", "questionable", "equivalent", "a year alter was rendered as changed."),
    "68": ("questionable", "equivalent", "equivalent", "A long turn was compressed and pretty good job is muddled."),
    "74": ("questionable", "equivalent", "equivalent", "disposes of that is only weakly translated."),
    "76": ("questionable", "equivalent", "equivalent", "understand and memorize are both present but hard to parse."),
    "82": ("not_equivalent", "equivalent", "equivalent", "ran out of time was dropped."),
    "84": ("equivalent", "not_equivalent", "not_equivalent", "never been happier became were never happy."),
    "89": ("questionable", "questionable", "questionable", "Construct validity lost the word construct."),
    "90": ("not_equivalent", "equivalent", "equivalent", "garrison and permanently were garbled away."),
    "93": ("not_equivalent", "equivalent", "not_equivalent", "hold the line became support (समर्थन); entailment flips."),
    "95": ("equivalent", "equivalent", "equivalent", "hour house was translated as a house of hours; the typo was kept."),
    "98": ("not_equivalent", "not_equivalent", "questionable", "pat down became belittle on the premise and slap on the hypothesis."),
    "100": ("questionable", "equivalent", "equivalent", "irreverent became unpleasant."),
    "107": ("questionable", "equivalent", "equivalent", "The film title was translated word for word."),
    "110": ("questionable", "not_equivalent", "not_equivalent", "career years became a career that lasts for years."),
    "111": ("equivalent", "not_equivalent", "not_equivalent", "Questionnaires are the only method became questions are used only in activity analysis."),
    "115": ("questionable", "equivalent", "equivalent", "The painful grip was dropped from the premise."),
    "119": ("questionable", "equivalent", "equivalent", "The double negative was weakened."),
    "121": ("questionable", "equivalent", "equivalent", "The Piet name was garbled."),
    "125": ("not_equivalent", "equivalent", "equivalent", "pottering became making toys, and swell became a swollen head."),
    "133": ("not_equivalent", "equivalent", "equivalent", "DILUTION was dropped from the premise."),
    "135": ("equivalent", "questionable", "equivalent", "funny became strange (अजीब)."),
    "142": ("questionable", "equivalent", "equivalent", "The method list was merged across clause boundaries."),
    "144": ("not_equivalent", "equivalent", "equivalent", "Band clauses were crossed."),
    "148": ("questionable", "equivalent", "equivalent", "A short stay was dropped; the hypothesis matches the English hypothesis."),
    "156": ("not_equivalent", "equivalent", "not_equivalent", "I always knew it and Rot were dropped, so the entailment no longer follows."),
    "165": ("equivalent", "not_equivalent", "not_equivalent", "The hypothesis collapsed into a repeated token."),
    "180": ("not_equivalent", "equivalent", "equivalent", "Clinton was rendered as Hillary."),
    "181": ("not_equivalent", "equivalent", "equivalent", "No John Grisham was parsed as a name."),
    "192": ("equivalent", "not_equivalent", "equivalent", "the financial industry became the world; the relation stays neutral."),
    "206": ("questionable", "equivalent", "equivalent", "ended up in jail became put himself in jail."),
    "213": ("not_equivalent", "equivalent", "equivalent", "Monroe was rendered as मोनकॉर्ड."),
    "215": ("not_equivalent", "equivalent", "not_equivalent", "do not stay put became do not remain; the contradiction moves toward entailment."),
    "217": ("not_equivalent", "equivalent", "not_equivalent", "did not trouble to rise became had no trouble rising."),
    "219": ("questionable", "equivalent", "equivalent", "P.S. was rendered as initials; the theatre claim remains."),
    "230": ("equivalent", "not_equivalent", "equivalent", "A bunch of would became a bunch; the relation stays neutral."),
    "231": ("not_equivalent", "equivalent", "not_equivalent", "after a fashion was literalized, so the entailment fails."),
    "238": ("questionable", "equivalent", "equivalent", "muscle hijackers is literal; the counts and Mihdhar remain."),
}


def _blank(judgment: tuple[str, str, str, str]) -> dict[str, str]:
    premise, hypothesis, label, note = judgment
    return {
        "premise_judgment": premise,
        "hypothesis_judgment": hypothesis,
        "label_judgment": label,
        "notes": note,
    }


def apply_judgments(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        judgment = OVERRIDES.get(row["sample_id"], ("equivalent", "equivalent", "equivalent", ""))
        row.update(_blank(judgment))
    return rows


def _degenerate(text: str) -> bool:
    tokens = WORD.findall(text)
    if len(tokens) < 12:
        return False
    _token, count = Counter(tokens).most_common(1)[0]
    return count >= 8 and count / len(tokens) >= 0.4


def exclusion_indexes() -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    degenerate: list[int] = []
    with JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] != "train":
                continue
            key = (
                record["english_premise"],
                record["english_hypothesis"],
                record["hindi_premise"],
                record["hindi_hypothesis"],
            )
            groups[key].append(record)
            if _degenerate(record["hindi_premise"]) or _degenerate(record["hindi_hypothesis"]):
                degenerate.append(int(record["hindi_row_index"]))
    conflicting: list[int] = []
    conflicting_groups = 0
    for records in groups.values():
        labels = {record["nli_label"] for record in records}
        if len(records) > 1 and len(labels) > 1:
            conflicting_groups += 1
            conflicting.extend(int(record["hindi_row_index"]) for record in records)
    degenerate_set = set(degenerate)
    conflicting_set = set(conflicting)
    return {
        "identical_text_conflicting_label_groups": conflicting_groups,
        "identical_text_conflicting_label_rows": sorted(conflicting_set),
        "degenerate_repetition_rows": sorted(degenerate_set),
        "overlap": sorted(conflicting_set & degenerate_set),
    }


def main() -> int:
    with SAMPLE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())
    base_rows = [row for row in rows if int(row["sample_id"]) < 245]
    apply_judgments(base_rows)
    kept = [row for row in rows if int(row["sample_id"]) >= 245]
    # Re-applying replaces appended rows so the script stays idempotent.
    _ = kept
    appended = _appended_rows()
    all_rows = base_rows + appended
    with SAMPLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    recorded = {
        sample_id: {
            "premise_judgment": premise,
            "hypothesis_judgment": hypothesis,
            "label_judgment": label,
            "notes": note,
        }
        for sample_id, (premise, hypothesis, label, note) in OVERRIDES.items()
    }
    for row in appended:
        recorded[row["sample_id"]] = {
            "premise_judgment": row["premise_judgment"],
            "hypothesis_judgment": row["hypothesis_judgment"],
            "label_judgment": row["label_judgment"],
            "notes": row["notes"],
        }
    judgments = {"sample_ids": len(all_rows), "overrides": recorded}
    (OUT / "semantic_review_judgments.json").write_text(
        json.dumps(judgments, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    exclusions = exclusion_indexes()
    exclusions["semantic_not_equivalent_rows"] = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "hindi_row_index": int(row["hindi_row_index"]) if row["hindi_row_index"] else None,
            "premise_judgment": row["premise_judgment"],
            "hypothesis_judgment": row["hypothesis_judgment"],
            "label_judgment": row["label_judgment"],
        }
        for row in all_rows
        if "not_equivalent" in (
            row["premise_judgment"],
            row["hypothesis_judgment"],
            row["label_judgment"],
        )
    ]
    (OUT / "conditional_approval_exclusions.json").write_text(
        json.dumps(exclusions, indent=2) + "\n",
        encoding="utf-8",
    )
    counts = Counter()
    for row in all_rows:
        for field in ("premise_judgment", "hypothesis_judgment", "label_judgment"):
            counts[f"{field}:{row[field]}"] += 1
    print(json.dumps({"rows": len(all_rows), "judgments": dict(counts), "exclusions": {
        "conflicting_rows": len(exclusions["identical_text_conflicting_label_rows"]),
        "degenerate_rows": len(exclusions["degenerate_repetition_rows"]),
        "overlap": len(exclusions["overlap"]),
        "groups": exclusions["identical_text_conflicting_label_groups"],
    }}, indent=2))
    return 0


def _appended_rows() -> list[dict[str, str]]:
    """Repeated-premise and test-collapse rows that the verified-only sampler missed."""
    specs = [
        ("245", "train", "repeated_premise_two_copies", "74008", "74025", "neutral",
         "questionable", "equivalent", "equivalent",
         "Two-copy premise. The same Hindi hypothesis Na is also the contradiction row, so both rows are in the conflicting-label exclusion."),
        ("246", "train", "repeated_premise_two_copies", "99955", "99979", "entailment",
         "questionable", "equivalent", "equivalent",
         "Two-copy premise. The heading is reordered in Hindi; the hypothesis matches."),
        ("247", "train", "repeated_premise_duplicate_label_withheld", "9", "", "contradiction",
         "withheld", "withheld", "withheld",
         "Place des Vosges has two contradiction rows. English candidates 9 and 57272. No pair was selected."),
        ("248", "train", "repeated_premise_duplicate_label_withheld", "57257", "", "contradiction",
         "withheld", "withheld", "withheld",
         "Second Hindi hypothesis for the same Place des Vosges contradiction slot. Withheld."),
        ("249", "test", "test_premise_collapse", "318", "318", "neutral",
         "equivalent", "equivalent", "equivalent",
         "Near-paraphrase English premises share one Hindi premise. This hypothesis stays distinct."),
        ("250", "test", "test_premise_collapse", "417", "417", "neutral",
         "equivalent", "questionable", "equivalent",
         "hire it out was rendered as rent it. Same Hindi premise as row 318."),
        ("251", "test", "test_premise_collapse", "319", "319", "contradiction",
         "equivalent", "not_equivalent", "questionable",
         "approve every deal became complete every deal."),
        ("252", "test", "test_premise_collapse", "419", "419", "contradiction",
         "equivalent", "equivalent", "equivalent",
         "Hypothesis stays distinct under the shared Hindi premise."),
        ("253", "test", "test_premise_collapse", "320", "320", "entailment",
         "equivalent", "equivalent", "equivalent",
         "Hypothesis stays distinct under the shared Hindi premise."),
        ("254", "test", "test_premise_collapse", "418", "418", "entailment",
         "equivalent", "equivalent", "equivalent",
         "Hypothesis stays distinct under the shared Hindi premise."),
    ]
    texts = _texts_for_appended()
    rows = []
    for spec in specs:
        sample_id, split, strata, hindi_index, english_index, label, premise_j, hyp_j, label_j, note = spec
        text = texts[(split, int(hindi_index))]
        rows.append({
            "split": split,
            "strata": strata,
            "hindi_row_index": hindi_index,
            "english_row_index": english_index,
            "nli_label": label,
            "english_premise": text["english_premise"],
            "english_hypothesis": text["english_hypothesis"],
            "hindi_premise": text["hindi_premise"],
            "hindi_hypothesis": text["hindi_hypothesis"],
            "provisional_english_row_index": english_index,
            "sample_id": sample_id,
            "premise_judgment": premise_j,
            "hypothesis_judgment": hyp_j,
            "label_judgment": label_j,
            "notes": note,
        })
    return rows


def _texts_for_appended() -> dict[tuple[str, int], dict[str, str]]:
    wanted = {74008, 99955, 9, 57257}
    found: dict[tuple[str, int], dict[str, str]] = {}
    with JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] == "train" and int(record["hindi_row_index"]) in wanted:
                found[("train", int(record["hindi_row_index"]))] = {
                    "english_premise": record["english_premise"],
                    "english_hypothesis": record["english_hypothesis"],
                    "hindi_premise": record["hindi_premise"],
                    "hindi_hypothesis": record["hindi_hypothesis"],
                }
            if record["split"] == "test" and int(record["hindi_row_index"]) in {318, 319, 320, 417, 418, 419}:
                found[("test", int(record["hindi_row_index"]))] = {
                    "english_premise": record["english_premise"],
                    "english_hypothesis": record["english_hypothesis"],
                    "hindi_premise": record["hindi_premise"],
                    "hindi_hypothesis": record["hindi_hypothesis"],
                }
    with (OUT / "candidate_alignment.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            hindi_index = int(row["hindi_row_index"])
            if hindi_index not in {9, 57257}:
                continue
            found[("train", hindi_index)] = {
                "english_premise": "",
                "english_hypothesis": "",
                "hindi_premise": row["hindi_premise"],
                "hindi_hypothesis": row["hindi_hypothesis"],
            }
    missing = [key for key in (
        ("train", 74008), ("train", 99955), ("train", 9), ("train", 57257),
        ("test", 318), ("test", 417), ("test", 319), ("test", 419), ("test", 320), ("test", 418),
    ) if key not in found]
    if missing:
        raise RuntimeError(f"Missing appended texts: {missing}")
    return found


if __name__ == "__main__":
    raise SystemExit(main())
