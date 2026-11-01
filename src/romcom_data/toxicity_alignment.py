"""Hindi role checks, Dolly_T seed selection, and refusal-template statistics.

Role assignment uses source list position. Semantic checks never use string length
and never relabel a slot. hin_Latn is validated separately and is not a text source.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from romcom_data.source_loaders.toxic_matrix import (
    BENIGN_SAMPLE_N,
    INSTRUCT_CONFIG,
    REPO,
    select_dolly_row_indices,
)
from romcom_data.text_utils import nfc
from romcom_data.toxic_turns import _blocks

HINDI_REFUSAL_OPEN = "मैं इस संकेत का जवाब नहीं दे सकता"
ENGLISH_REFUSAL_OPEN = "I cannot give a response"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_DIGIT = re.compile(r"\d+")
_LATIN = re.compile(r"[A-Za-z]{3,}")


def _pair(row: dict[str, Any], field: str) -> tuple[str, str]:
    if field == "hin_Latn":
        field = "hin_Latn_alignment_only" if "hin_Latn_alignment_only" in row else "hin_Latn"
    blocks = _blocks(row.get(field))
    if not blocks:
        return "", ""
    return blocks[0]


def _anchors(text: str) -> set[str]:
    return set(_DIGIT.findall(text)) | {token.casefold() for token in _LATIN.findall(text)}


def anchor_verdict(eng_user: str, eng_asst: str, other_user: str, other_asst: str) -> str:
    direct = len(_anchors(eng_user) & _anchors(other_user)) + len(
        _anchors(eng_asst) & _anchors(other_asst)
    )
    crossed = len(_anchors(eng_user) & _anchors(other_asst)) + len(
        _anchors(eng_asst) & _anchors(other_user)
    )
    if direct == 0 and crossed == 0:
        return "no_shared_anchor"
    if direct > crossed:
        return "supports_positional"
    if crossed > direct:
        return "conflicts_with_positional"
    return "anchor_tie"


def template_slot(user: str, assistant: str, opening: str) -> str:
    user_has = user.startswith(opening)
    asst_has = assistant.startswith(opening)
    if asst_has and not user_has:
        return "assistant_slot"
    if user_has and not asst_has:
        return "user_slot"
    if user_has and asst_has:
        return "both_slots"
    return "absent"


def prepare_dolly_seed_sample(revision: str, dest: Path) -> dict[str, Any]:
    """Write the seed-42 Dolly_T sample. Does not replace the head file or rebuild tasks."""
    from datasets import load_dataset

    dataset = load_dataset(REPO, name=INSTRUCT_CONFIG, split="train", revision=revision)
    n_rows = len(dataset)
    chosen = select_dolly_row_indices(n_rows, BENIGN_SAMPLE_N, 42)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prompt_lengths: list[int] = []
    for block in dataset["eng_Latn"]:
        if isinstance(block, list) and block and isinstance(block[0], list) and block[0]:
            prompt_lengths.append(len(str(block[0][0] or "")))
        else:
            prompt_lengths.append(0)
    head = prompt_lengths[:BENIGN_SAMPLE_N]
    seeded = [prompt_lengths[i] for i in chosen]
    dest_tmp = dest.with_suffix(".jsonl.tmp")
    with dest_tmp.open("w", encoding="utf-8") as fh:
        for source_row_index in chosen:
            row = dataset[int(source_row_index)]
            record = {
                "doc_id": row["doc_id"],
                "source_row_index": int(source_row_index),
                "num_turns": row.get("num_turns"),
                "hin_Deva": row["hin_Deva"],
                "eng_Latn": row["eng_Latn"],
                "hin_Latn_alignment_only": row.get("hin_Latn"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    dest_tmp.replace(dest)
    return {
        "path": str(dest),
        "n_source_rows": n_rows,
        "n_selected": len(chosen),
        "seed": 42,
        "selector": "derive_rng(42, 'dolly-t', 'benign-shuffle')",
        "preserves_doc_id": True,
        "preserves_source_row_index": True,
        "head_file_replaced": False,
        "final_dataset_rebuilt": False,
        "ordering_bias": _ordering_bias(prompt_lengths, chosen),
        "length_comparison_eng_user_chars": {
            "full_mean": round(sum(prompt_lengths) / len(prompt_lengths), 1),
            "head_0_1999_mean": round(sum(head) / len(head), 1),
            "seed42_mean": round(sum(seeded) / len(seeded), 1),
        },
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return round(num / (dx * dy), 4)


def _ordering_bias(prompt_lengths: list[int], chosen: list[int]) -> dict[str, Any]:
    index = list(range(len(prompt_lengths)))
    return {
        "measure": "Pearson correlation of Dolly_T train row index with English user length",
        "row_index_vs_eng_user_length": _pearson(
            [float(i) for i in index], [float(v) for v in prompt_lengths]
        ),
        "selected_index_min": min(chosen),
        "selected_index_max": max(chosen),
        "selected_index_mean": round(sum(chosen) / len(chosen), 1),
        "note": (
            "The on-disk head is rows 0-1999. A non-zero correlation means a prefix "
            "can differ from a seeded draw. The prepared sample is the seeded draw."
        ),
    }


def _mean_pool(hidden: Any, mask: Any) -> Any:
    import torch

    expanded = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)


def _embed_texts(texts: list[str]) -> list[list[float]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL)
    model.eval()
    vectors: list[list[float]] = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [text[:2000] for text in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=96, return_tensors="pt"
            )
            hidden = model(**encoded).last_hidden_state
            pooled = _mean_pool(hidden, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.tolist())
    return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def embedding_alignment(rows: list[dict[str, Any]], other_field: str) -> dict[str, Any]:
    """Compare other_field slots with eng_Latn. Does not change roles."""
    texts: list[str] = []
    for row in rows:
        eng_user, eng_asst = _pair(row, "eng_Latn")
        other_user, other_asst = _pair(row, other_field)
        texts.extend((eng_user, eng_asst, other_user, other_asst))
    vectors = _embed_texts(texts)
    counts: Counter[str] = Counter()
    conflicts: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        base = i * 4
        eng_user, eng_asst, other_user, other_asst = vectors[base : base + 4]
        direct = _cosine(other_user, eng_user) + _cosine(other_asst, eng_asst)
        crossed = _cosine(other_user, eng_asst) + _cosine(other_asst, eng_user)
        if direct > crossed + 0.05:
            status = "supports_positional"
        elif crossed > direct + 0.05:
            status = "conflicts_with_positional"
        else:
            status = "embedding_tie"
        counts[status] += 1
        if status == "conflicts_with_positional":
            conflicts.append(
                {
                    "doc_id": row.get("doc_id"),
                    "source_row_index": row.get("source_row_index"),
                    "field": other_field,
                    "direct_cosine_sum": round(direct, 4),
                    "crossed_cosine_sum": round(crossed, 4),
                    "action": "excluded_from_experiment_ready_set_roles_not_flipped",
                }
            )
    return {
        "field": other_field,
        "model": EMBED_MODEL,
        "rule": (
            "Sum of cosine similarities for same-index slots versus crossed slots. "
            "Margin 0.05. String length is not an input. Conflicts are excluded, not swapped."
        ),
        "used_as_canonical_text": other_field == "hin_Deva",
        "counts": dict(counts),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }


def template_statistics(rows: list[dict[str, Any]], class_name: str) -> dict[str, Any]:
    assistant_texts: list[str] = []
    user_texts: list[str] = []
    hindi_slots: Counter[str] = Counter()
    english_slots: Counter[str] = Counter()
    for row in rows:
        hin_user, hin_asst = _pair(row, "hin_Deva")
        eng_user, eng_asst = _pair(row, "eng_Latn")
        assistant_texts.append(nfc(hin_asst))
        user_texts.append(nfc(hin_user))
        hindi_slots[template_slot(hin_user, hin_asst, HINDI_REFUSAL_OPEN)] += 1
        english_slots[template_slot(eng_user, eng_asst, ENGLISH_REFUSAL_OPEN)] += 1
    exact = Counter(_normalize(text) for text in assistant_texts)
    prefix = Counter(_normalize(text)[:120] for text in assistant_texts)
    n = max(len(assistant_texts), 1)
    exact_dup_rows = sum(count for count in exact.values() if count > 1)
    near_dup_rows = sum(count for count in prefix.values() if count > 1)
    lengths = [len(text) for text in assistant_texts]
    return {
        "class_name": class_name,
        "n": len(assistant_texts),
        "canonical_field": "hin_Deva",
        "hindi_refusal_opening": HINDI_REFUSAL_OPEN,
        "hindi_opening_slot_counts": dict(hindi_slots),
        "english_opening_slot_counts": dict(english_slots),
        "english_opening": ENGLISH_REFUSAL_OPEN,
        "assistant_exact_duplicate_row_rate": round(exact_dup_rows / n, 4),
        "assistant_exact_unique": len(exact),
        "assistant_prefix120_near_duplicate_row_rate": round(near_dup_rows / n, 4),
        "assistant_prefix120_unique": len(prefix),
        "assistant_chars": {
            "mean": round(sum(lengths) / n, 1) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "user_chars_mean": round(sum(len(text) for text in user_texts) / n, 1),
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", nfc(text).casefold()).strip()


def shortcut_diagnostic(toxic: dict[str, Any], benign: dict[str, Any]) -> dict[str, Any]:
    toxic_template = toxic["hindi_opening_slot_counts"].get("assistant_slot", 0)
    benign_template = benign["hindi_opening_slot_counts"].get("assistant_slot", 0)
    n_toxic = toxic["n"]
    n_benign = benign["n"]
    n = n_toxic + n_benign
    template_correct = toxic_template + (n_benign - benign_template)
    return {
        "not_a_trained_model": True,
        "template_rule": (
            "Predict toxic if the Hindi assistant slot starts with the refusal opening."
        ),
        "template_rule_in_sample_accuracy": round(template_correct / n, 4) if n else None,
        "length_means_assistant_chars": {
            "toxic": toxic["assistant_chars"]["mean"],
            "benign": benign["assistant_chars"]["mean"],
        },
        "interpretation": (
            "A source-specific refusal opening, or a length cutoff between these means, "
            "can separate the classes without modeling refusal semantics. "
            "That supports a generation comparison of registers only. "
            "It does not support toxicity detection or a claim that a classifier "
            "is reading safety behavior."
        ),
    }


def fetch_dolly_upstream_license() -> dict[str, Any]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info("databricks/databricks-dolly-15k")
    card = info.card_data or {}
    return {
        "repository": "databricks/databricks-dolly-15k",
        "hf_card_license_field": card.get("license") if isinstance(card, dict) else None,
        "hf_license_attr": getattr(info, "license", None),
        "tags": [tag for tag in (info.tags or []) if "license" in tag],
        "documented_upstream": "CC-BY-SA-3.0",
    }
