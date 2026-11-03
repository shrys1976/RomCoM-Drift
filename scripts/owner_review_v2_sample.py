#!/usr/bin/env python3
"""Read-only owner sample review (seeded). Does not modify JSONL."""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from romcom_data.nli_exp_paths import NLI_EXPERIMENTS
from romcom_data.nli_exp_romanize import NliLexicon
from romcom_data.nli_exp_variant_exclusions import excluded_variant_keys
from romcom_data.nli_exp_variants import VariantPools
from romcom_data.text_utils import nfc

SEED = 20260303


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    root = NLI_EXPERIMENTS
    pools = VariantPools(NliLexicon())
    banned = excluded_variant_keys()
    rng = random.Random(SEED)

    noise_collisions: dict[str, int] = defaultdict(int)
    ident_pos: dict[str, int] = defaultdict(int)
    pos_pairs: dict[str, int] = defaultdict(int)
    noise_repeat: dict[str, int] = defaultdict(int)

    for split in ("train", "dev", "test"):
        for rec in load_jsonl(root / "logs" / f"transformation_log.{split}.jsonl"):
            t3 = [t for t in rec.get("exp03_traces", []) if t.get("operation") == "random_noise"]
            by_field: dict[str, list[dict]] = defaultdict(list)
            for t in t3:
                by_field[t["field"]].append(t)
            for ts in by_field.values():
                by_noise: dict[str, set[str]] = defaultdict(set)
                for t in ts:
                    by_noise[t["selected"]].add(t["canonical"])
                for canons in by_noise.values():
                    if len(canons) > 1:
                        noise_collisions[split] += len(canons) * (len(canons) - 1) // 2
                seen: dict[str, int] = defaultdict(int)
                for t in ts:
                    seen[t["selected"]] += 1
                for c in seen.values():
                    if c > 1:
                        noise_repeat[split] += c * (c - 1) // 2
                for i in range(len(ts)):
                    for j in range(i + 1, len(ts)):
                        a, b = ts[i], ts[j]
                        if len(a["canonical"]) != len(b["canonical"]):
                            continue
                        if a["levenshtein_distance"] != b["levenshtein_distance"]:
                            continue
                        pos_pairs[split] += 1
                        pa = tuple(
                            k
                            for k, (x, y) in enumerate(zip(a["canonical"], a["selected"]))
                            if x != y
                        )
                        pb = tuple(
                            k
                            for k, (x, y) in enumerate(zip(b["canonical"], b["selected"]))
                            if x != y
                        )
                        if pa == pb:
                            ident_pos[split] += 1

    print("NOISE_STRING_COLLISIONS", dict(noise_collisions))
    print("REPEATED_NOISE_FIELD_PAIRS", dict(noise_repeat))
    for split in ("train", "dev", "test"):
        t, i = pos_pairs[split], ident_pos[split]
        if t:
            print(f"POS_PATTERN_COLLISION {split}: {i}/{t} = {100 * i / t:.4f}%")

    def tags_for(rec: dict) -> set[str]:
        t2 = [t for t in rec.get("exp02_traces", []) if t.get("operation") == "attested_variant"]
        tags: set[str] = set()
        for field in ("premise", "hypothesis"):
            toks = rec.get(f"{field}_exp01_tokens") or []
            n_e = sum(
                1
                for t in toks
                if t.get("method") == "dakshina_canonical"
                and pools.pool(nfc(str(t.get("native_text", "")).rstrip("।॥").strip()))
            )
            n_s = len([t for t in t2 if t["field"] == field])
            if n_e and 0 < n_s < n_e:
                tags.add("partial")
            if n_e >= 2:
                tags.add("multi_elig")
        all_t = (rec.get("premise_exp01_tokens") or []) + (rec.get("hypothesis_exp01_tokens") or [])
        natives = [nfc(str(t.get("native_text", ""))) for t in all_t]
        if len(natives) != len(set(natives)):
            tags.add("dup_native")
        text = " ".join(str(t.get("native_text", "")) for t in all_t)
        if re.search(r"[A-Za-z]", text):
            tags.add("latin")
        if re.search(r"[0-9]|[^\w\s]", text):
            tags.add("punct_num")
        for t in all_t:
            key = nfc(str(t.get("native_text", "")).rstrip("।॥").strip())
            if len(pools.pool(key)) >= 2:
                tags.add("multi_var")
                break
        return tags

    def pick_samples(split: str, n: int) -> list[tuple[dict, set[str]]]:
        path = root / "logs" / f"transformation_log.{split}.jsonl"
        reservoir: list[tuple[dict, set[str]]] = []
        seen = 0
        for rec in load_jsonl(path):
            t2 = [t for t in rec.get("exp02_traces", []) if t.get("operation") == "attested_variant"]
            if split == "train" and not t2 and seen % 7:
                seen += 1
                continue
            tags = tags_for(rec)
            weight = len(tags) + (3 if "partial" in tags else 0) + min(len(t2), 5)
            if weight < 2 and split == "train":
                seen += 1
                continue
            seen += 1
            if len(reservoir) < n:
                reservoir.append((rec, tags))
            else:
                j = rng.randrange(seen)
                if j < n:
                    reservoir[j] = (rec, tags)
        if len(reservoir) < n:
            for rec in load_jsonl(path):
                if len(reservoir) >= n:
                    break
                if rec not in [r for r, _ in reservoir]:
                    reservoir.append((rec, tags_for(rec)))
        return reservoir[:n]

    samples: dict[str, list[tuple[dict, set[str]]]] = {
        "train": pick_samples("train", 50),
        "dev": pick_samples("dev", 25),
        "test": pick_samples("test", 25),
    }

    issues: list[str] = []
    detailed: list[dict] = []
    for split, items in samples.items():
        e2 = {r["example_id"]: r for r in load_jsonl(root / "exp02" / "both" / f"{split}.jsonl")}
        for rec, tags in items:
            eid = rec["example_id"]
            t2 = [t for t in rec.get("exp02_traces", []) if t.get("operation") == "attested_variant"]
            t3 = [t for t in rec.get("exp03_traces", []) if t.get("operation") == "random_noise"]
            row2 = e2[eid]
            tr_ex = []
            for tr in t2:
                native_key = None
                for tok in (rec.get("premise_exp01_tokens") or []) + (
                    rec.get("hypothesis_exp01_tokens") or []
                ):
                    if tok.get("token_index") == tr["token_index"] and tok.get("field") == tr["field"]:
                        native_key = nfc(str(tok.get("native_text", "")).rstrip("।॥").strip())
                        break
                sel = tr["selected"].rstrip(".")
                pool_set = {v.casefold() for v, _ in pools.pool(native_key or "")}
                if (native_key, sel.casefold()) in banned:
                    issues.append(f"excluded {eid}")
                if pool_set and sel.casefold() not in pool_set:
                    issues.append(f"pool {eid} {native_key}->{sel}")
                if sel not in row2["premise"] and sel not in row2["hypothesis"]:
                    issues.append(f"surface {eid}")
                tr_ex.append(
                    {
                        "field": tr["field"],
                        "native": native_key,
                        "canon": tr["canonical"],
                        "sel": sel,
                        "d": tr["levenshtein_distance"],
                    }
                )
            for a, b in zip(t2, t3, strict=True):
                if a["token_index"] != b["token_index"] or a["field"] != b["field"]:
                    issues.append(f"align {eid}")
            detailed.append(
                {
                    "example_id": eid,
                    "split": split,
                    "tags": sorted(tags),
                    "n_sel": len(t2),
                    "traces": tr_ex[:5],
                }
            )

    out = root / "reports" / "owner_review_v2_sample_ids.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"seed": SEED, "manual_issues": issues, "samples": detailed},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("MANUAL_ISSUES", len(issues))
    print("WROTE", out, "n=", len(detailed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
