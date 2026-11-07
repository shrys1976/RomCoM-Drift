#!/usr/bin/env python3
"""Read-only post-freeze audit for frozen NLI V2. Does not modify V1/V2."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.checksums import sha256_file
from romcom_data.text_utils import nfc

V1 = ROOT / "romcomx_data/nli_experiments_v1"
V2 = ROOT / "romcomx_data/nli_experiments_v2"
INVENTORY = ROOT / "romcomx_data/inventories/dakshina_hi_variant_inventory.tsv"
EXCLUSIONS = ROOT / "romcomx_data/inventories/dakshina_hi_exp02_variant_exclusions.json"
REPORT = ROOT / "romcomx_data/reports/post_freeze_audit_v2.md"
CHECKSUMS = V2 / "checksums.sha256"

CONDITIONS = ("exp01", "exp02", "exp03")
SPLITS = ("train", "dev", "test")

QUARANTINE = [
    ("क्षण", "shan"),
    ("क्षणों", "shanon"),
    ("क्षमा", "shma"),
    ("बंगाल", "bangla"),
    ("पंचर", "puncher"),
]


def norm_roman(s: str) -> str:
    return nfc(s.strip()).casefold()


def norm_native(s: str) -> str:
    return nfc(s.rstrip("।॥").strip())


def load_ids_labels(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["example_id"]] = row["nli_label"]
    return out


def verify_checksums() -> tuple[bool, int, int]:
    bad = missing = 0
    for line in CHECKSUMS.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        p = ROOT / rel
        if not p.is_file():
            missing += 1
            continue
        if sha256_file(p) != expected:
            bad += 1
    n = sum(1 for ln in CHECKSUMS.read_text().splitlines() if ln.strip())
    return bad == 0 and missing == 0, n, bad + missing


@dataclass
class CheckA:
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    intersections: dict[str, dict[str, int]] = field(default_factory=dict)
    overlap_ids: dict[str, list[str]] = field(default_factory=dict)
    label_mismatches: list[str] = field(default_factory=list)
    cross_condition_issues: list[str] = field(default_factory=list)


def run_check_a() -> CheckA:
    result = CheckA()
    ids_by_cond_split: dict[str, dict[str, dict[str, str]]] = {
        c: {s: {} for s in SPLITS} for c in CONDITIONS
    }
    for cond in CONDITIONS:
        for split in SPLITS:
            path = V2 / cond / "both" / f"{split}.jsonl"
            ids_by_cond_split[cond][split] = load_ids_labels(path)
            result.counts.setdefault(cond, {})[split] = len(ids_by_cond_split[cond][split])

    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    for cond in CONDITIONS:
        result.intersections[cond] = {}
        result.overlap_ids[cond] = []
        for a, b in pairs:
            sa = set(ids_by_cond_split[cond][a])
            sb = set(ids_by_cond_split[cond][b])
            inter = sa & sb
            key = f"{a}∩{b}"
            result.intersections[cond][key] = len(inter)
            if inter:
                result.overlap_ids[cond].extend(sorted(inter))

    for split in SPLITS:
        base = ids_by_cond_split["exp01"][split]
        for cond in ("exp02", "exp03"):
            other = ids_by_cond_split[cond][split]
            if set(base) != set(other):
                only_b = set(other) - set(base)
                only_a = set(base) - set(other)
                result.cross_condition_issues.append(
                    f"{split} {cond}: id set mismatch "
                    f"(only_exp01={len(only_a)}, only_{cond}={len(only_b)})"
                )
            for eid, lab in base.items():
                lab2 = other.get(eid)
                if lab2 is not None and lab2 != lab:
                    result.label_mismatches.append(f"{split} {eid} exp01={lab} {cond}={lab2}")

    return result


@dataclass
class QuarantineStats:
    by_mapping: dict[tuple[str, str], Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    examples: dict[tuple[str, str], list[dict]] = field(default_factory=lambda: defaultdict(list))
    distinct_eids: dict[tuple[str, str], set[str]] = field(default_factory=lambda: defaultdict(set))


def native_for_trace(rec: dict, tr: dict) -> str | None:
    key = f"{tr['field']}_exp01_tokens"
    for tok in rec.get(key) or []:
        if tok.get("token_index") == tr["token_index"] and tok.get("field") == tr["field"]:
            return norm_native(str(tok.get("native_text", "")))
    return None


def run_check_b() -> QuarantineStats:
    stats = QuarantineStats()
    qkeys = {(norm_native(n), norm_roman(v)) for n, v in QUARANTINE}
    for split in SPLITS:
        path = V2 / "logs" / f"transformation_log.{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                eid = rec["example_id"]
                for tr in rec.get("exp02_traces") or []:
                    if tr.get("operation") != "attested_variant":
                        continue
                    native = native_for_trace(rec, tr)
                    if not native:
                        continue
                    sel = norm_roman(tr["selected"].rstrip("."))
                    pair = (native, sel)
                    if pair not in qkeys:
                        continue
                    stats.by_mapping[pair][split] += 1
                    stats.distinct_eids[pair].add(eid)
                    if len(stats.examples[pair]) < 5:
                        stats.examples[pair].append(
                            {
                                "example_id": eid,
                                "split": split,
                                "field": tr["field"],
                                "token_index": tr["token_index"],
                                "canonical": tr.get("canonical"),
                                "selected": tr.get("selected"),
                                "operation": tr.get("operation"),
                            }
                        )
    return stats


@dataclass
class DakshinaIndex:
    attested_romans: set[str]
    roman_to_natives: dict[str, set[str]]


def load_dakshina_index() -> DakshinaIndex:
    attested: set[str] = set()
    roman_to_natives: dict[str, set[str]] = defaultdict(set)
    with INVENTORY.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            native = norm_native(row["native_word"])
            for col in ("canonical_romanization", "romanized_variant"):
                rom = norm_roman(row[col])
                if not rom:
                    continue
                attested.add(rom)
                roman_to_natives[rom].add(native)
    return DakshinaIndex(attested_romans=attested, roman_to_natives=dict(roman_to_natives))


@dataclass
class CheckC:
    token_occurrences: Counter[str] = field(default_factory=Counter)
    token_matches: Counter[str] = field(default_factory=Counter)
    unique_strings: dict[str, set[str]] = field(default_factory=lambda: {s: set() for s in SPLITS})
    unique_hits: dict[str, set[str]] = field(default_factory=lambda: {s: set() for s in SPLITS})
    examples: list[dict] = field(default_factory=list)


def classify_collision(
    source_native: str | None,
    canon: str,
    noise: str,
    idx: DakshinaIndex,
) -> str:
    natives = idx.roman_to_natives.get(noise, set())
    if not natives:
        return "unknown"
    if source_native and source_native in natives:
        return "same_native"
    if len(natives) > 1:
        return "multiple_natives"
    return "different_native"


def run_check_c(idx: DakshinaIndex) -> CheckC:
    result = CheckC()
    seen_example_keys: set[tuple[str, str, str, int]] = set()
    for split in SPLITS:
        path = V2 / "logs" / f"transformation_log.{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                rec = json.loads(line)
                eid = rec["example_id"]
                for tr in rec.get("exp03_traces") or []:
                    if tr.get("operation") != "random_noise":
                        continue
                    noise = norm_roman(tr["selected"].rstrip("."))
                    result.token_occurrences[split] += 1
                    result.unique_strings[split].add(noise)
                    if noise in idx.attested_romans:
                        result.token_matches[split] += 1
                        result.unique_hits[split].add(noise)
                        key = (eid, tr["field"], noise, tr["token_index"])
                        if key not in seen_example_keys and len(result.examples) < 40:
                            seen_example_keys.add(key)
                            source_native = native_for_trace(rec, tr)
                            result.examples.append(
                                {
                                    "example_id": eid,
                                    "split": split,
                                    "field": tr["field"],
                                    "token_index": tr["token_index"],
                                    "canonical": tr.get("canonical"),
                                    "noise_token": tr.get("selected"),
                                    "matching_roman_norm": noise,
                                    "dakshina_native_words": sorted(
                                        idx.roman_to_natives.get(noise, set())
                                    ),
                                    "source_native": source_native,
                                    "collision_class": classify_collision(
                                        source_native,
                                        tr.get("canonical", ""),
                                        noise,
                                        idx,
                                    ),
                                }
                            )
    return result


def md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = len(rows[0])
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    checksum_ok, checksum_n, checksum_bad = verify_checksums()
    check_a = run_check_a()
    check_b = run_check_b()
    dakshina = load_dakshina_index()
    check_c = run_check_c(dakshina)

    _disjoint_pass = all(
        v == 0
        for cond in CONDITIONS
        for v in check_a.intersections[cond].values()
    )

    lines: list[str] = [
        "# Post-freeze audit: RomCoM Drift NLI V2",
        "",
        f"**Audit date:** {date.today().isoformat()}",
        f"**Frozen snapshot:** `romcomx_data/nli_experiments_v2/`",
        f"**Read-only:** No files under V1 or V2 were modified.",
        "",
        "## Checksum verification (post-audit)",
        "",
        f"- Manifest: `romcomx_data/nli_experiments_v2/checksums.sha256`",
        f"- Entries checked: {checksum_n}",
        f"- Result: **{'PASS' if checksum_ok else 'FAIL'}**"
        + (f" ({checksum_bad} missing/mismatch)" if not checksum_ok else ""),
        "",
        "---",
        "",
        "## Check A: Train / dev / test disjointness",
        "",
        "### Methodology",
        "",
        "Streamed `example_id` from `exp01/both/`, `exp02/both/`, and `exp03/both/` "
        "for each split. Computed pairwise set intersections across splits per condition. "
        "Compared `nli_label` and ID sets across EXP-01/02/03 within each split "
        "(same IDs across conditions are expected).",
        "",
        "### Split sizes (EXP-01 / EXP-02 / EXP-03; identical per split)",
        "",
    ]

    for cond in CONDITIONS:
        lines.append(f"**{cond.upper()}**")
        lines.append(
            md_table(
                [
                    ["Split", "Distinct example_id count"],
                    *[[s, str(check_a.counts[cond][s])] for s in SPLITS],
                ]
            )
        )
        lines.append("")

    lines.extend(["### Pairwise intersections (must be 0)", ""])
    for cond in CONDITIONS:
        lines.append(f"**{cond.upper()}**")
        lines.append(
            md_table(
                [
                    ["Pair", "Intersection size"],
                    *[
                        [k, str(v)]
                        for k, v in sorted(check_a.intersections[cond].items())
                    ],
                ]
            )
        )
        if check_a.overlap_ids[cond]:
            lines.append("")
            lines.append(f"Overlapping IDs ({cond}): " + ", ".join(check_a.overlap_ids[cond][:50]))
        lines.append("")

    lines.extend(
        [
            "### Cross-condition consistency (within split)",
            "",
            f"- ID set mismatches (exp01 vs exp02/exp03): **{len(check_a.cross_condition_issues)}**",
            f"- Label mismatches: **{len(check_a.label_mismatches)}**",
        ]
    )
    if check_a.cross_condition_issues:
        for item in check_a.cross_condition_issues[:20]:
            lines.append(f"  - {item}")
    if check_a.label_mismatches:
        for item in check_a.label_mismatches[:20]:
            lines.append(f"  - {item}")

    lines.extend(
        [
            "",
            f"**Check A result:** **{'PASS' if _disjoint_pass else 'FAIL'}**",
            "",
            "---",
            "",
            "## Check B: Quarantined mapping selections (EXP-02)",
            "",
            "### Methodology",
            "",
            "Scanned V2 `logs/transformation_log.{split}.jsonl` for `exp02_traces` with "
            "`operation: attested_variant`. Matched exact `(native_word, selected variant)` "
            "against the five quarantine pairs after NFC + casefold normalization. "
            "Native text taken from the matching `*_exp01_tokens` entry.",
            "",
            "### Selection counts",
            "",
            md_table(
                [
                    ["Mapping (native → variant)", "Train", "Dev", "Test", "Total", "Distinct example_id"],
                    *[
                        [
                            f"{n} → {v}",
                            str(check_b.by_mapping[(norm_native(n), norm_roman(v))]["train"]),
                            str(check_b.by_mapping[(norm_native(n), norm_roman(v))]["dev"]),
                            str(check_b.by_mapping[(norm_native(n), norm_roman(v))]["test"]),
                            str(sum(check_b.by_mapping[(norm_native(n), norm_roman(v))].values())),
                            str(len(check_b.distinct_eids[(norm_native(n), norm_roman(v))])),
                        ]
                        for n, v in QUARANTINE
                    ],
                ]
            ),
            "",
            "All non-zero hits are **EXP-02 attested_variant** selections (not incidental string matches).",
            "",
            "### Sample positions (non-zero mappings)",
            "",
        ]
    )

    for n, v in QUARANTINE:
        pair = (norm_native(n), norm_roman(v))
        total = sum(check_b.by_mapping[pair].values())
        lines.append(f"#### {n} → {v} (total {total})")
        if not check_b.examples[pair]:
            lines.append("- No selections in V2 logs.")
        else:
            for ex in check_b.examples[pair]:
                lines.append(
                    f"- `{ex['example_id']}` ({ex['split']}) field={ex['field']} "
                    f"token_index={ex['token_index']} canonical={ex['canonical']} "
                    f"selected={ex['selected']}"
                )
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Check C: EXP-03 noise vs attested Dakshina romanizations",
            "",
            "### Methodology",
            "",
            f"Built attested set from `{INVENTORY.relative_to(ROOT)}`: "
            "all distinct `canonical_romanization` and `romanized_variant` values "
            f"({len(dakshina.attested_romans):,} normalized forms). "
            "Compared each EXP-03 `random_noise` trace `selected` string using NFC + casefold "
            "exact equality (no fuzzy match).",
            "",
            "### A. Token-level collision rate",
            "",
            md_table(
                [
                    ["Split", "Matching occurrences", "Total noise tokens", "Rate %"],
                    *[
                        [
                            s,
                            str(check_c.token_matches[s]),
                            str(check_c.token_occurrences[s]),
                            f"{100 * check_c.token_matches[s] / check_c.token_occurrences[s]:.6f}"
                            if check_c.token_occurrences[s]
                            else "n/a",
                        ]
                        for s in SPLITS
                    ],
                    [
                        "**all**",
                        str(sum(check_c.token_matches.values())),
                        str(sum(check_c.token_occurrences.values())),
                        f"{100 * sum(check_c.token_matches.values()) / sum(check_c.token_occurrences.values()):.6f}",
                    ],
                ]
            ),
            "",
            "### B. Unique-string collision rate",
            "",
            md_table(
                [
                    ["Split", "Matching unique strings", "Total unique strings", "Rate %"],
                    *[
                        [
                            s,
                            str(len(check_c.unique_hits[s])),
                            str(len(check_c.unique_strings[s])),
                            f"{100 * len(check_c.unique_hits[s]) / len(check_c.unique_strings[s]):.6f}"
                            if check_c.unique_strings[s]
                            else "n/a",
                        ]
                        for s in SPLITS
                    ],
                    [
                        "**all**",
                        str(len(set().union(*check_c.unique_hits.values()))),
                        str(len(set().union(*check_c.unique_strings.values()))),
                        f"{100 * len(set().union(*check_c.unique_hits.values())) / len(set().union(*check_c.unique_strings.values())):.6f}",
                    ],
                ]
            ),
            "",
            "### C. Collision interpretation (representative examples)",
            "",
            "Classes: `same_native` = attested form maps to source token's native word; "
            "`different_native` = exact attested string but other lemma; "
            "`multiple_natives` = inventory associates string with >1 native word.",
            "",
        ]
    )

    class_counts = Counter(ex["collision_class"] for ex in check_c.examples)
    lines.append(f"Example pool class counts (up to 40 samples): {dict(class_counts)}")
    lines.append("")
    for ex in check_c.examples[:15]:
        lines.append(
            f"- `{ex['example_id']}` ({ex['split']}) {ex['field']}[{ex['token_index']}] "
            f"canonical=`{ex['canonical']}` noise=`{ex['noise_token']}` "
            f"class={ex['collision_class']} "
            f"source_native=`{ex['source_native']}` "
            f"dakshina_natives={ex['dakshina_native_words'][:5]}"
            + ("…" if len(ex["dakshina_native_words"]) > 5 else "")
        )

    lines.extend(
        [
            "",
            "### Limitations",
            "",
            "- Collisions are exact inventory membership; homographs across lemmas count.",
            "- Canonical-only forms in the inventory are included in the attested set.",
            "- Random noise that matches an attested form for a **different** native word "
            "still counts as a token-level collision.",
            "",
            "---",
            "",
            "## Summary",
            "",
            md_table(
                [
                    ["Check", "Result", "Issues found"],
                    [
                        "Train/dev/test disjointness",
                        "PASS" if _disjoint_pass else "FAIL",
                        "0 overlapping IDs across splits"
                        if _disjoint_pass
                        else "Non-empty train/dev/test intersections",
                    ],
                    [
                        "Quarantined mapping selections",
                        "Counts recorded",
                        f"See Check B table; total selections "
                        f"{sum(sum(c.values()) for c in check_b.by_mapping.values())}",
                    ],
                    [
                        "EXP-03 attested-string collisions",
                        "Counts recorded",
                        f"{sum(check_c.token_matches.values())}/{sum(check_c.token_occurrences.values())} "
                        f"token occurrences ({100 * sum(check_c.token_matches.values()) / max(1, sum(check_c.token_occurrences.values())):.4f}%)",
                    ],
                ]
            ),
            "",
            f"Prior owner review: `romcomx_data/nli_experiments_v2/reports/owner_review_v2_final.md`",
            "",
        ]
    )

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {REPORT}")
    print(f"CHECK_A {'PASS' if _disjoint_pass else 'FAIL'}")
    print(f"CHECKSUM {'PASS' if checksum_ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
