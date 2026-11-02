from __future__ import annotations

import copy
from typing import Any

from romcom_data.augmentation import (
    canonicalize_text,
    noise_from_changes,
    realistic_from_canonical,
)
from romcom_data.romanize import DakshinaRomanizer
from romcom_data.stable_hash import derive_rng

TARGET_ONLY_TASKS = frozenset({"toxicity", "hindi_refusal_response_generation"})


def _target_only(task: str) -> bool:
    return task in TARGET_ONLY_TASKS


def _input_fields(task: str) -> list[tuple[str, str]]:
    if task == "nli":
        return [("input", "premise"), ("input", "hypothesis")]
    if task == "paraphrase":
        return [("input", "sentence1"), ("input", "sentence2")]
    return []


def _get(row: dict[str, Any], container: str, key: str) -> str:
    return str(row.get(container, {}).get(key, ""))


def _set(row: dict[str, Any], container: str, key: str, value: str) -> None:
    row.setdefault(container, {})[key] = value


def generate_one(
    base: dict[str, Any],
    romanizer: DakshinaRomanizer,
    p_variant: float = 0.35,
    base_seed: int = 42,
) -> dict[str, dict[str, Any]]:
    task = base["task"]
    e1 = copy.deepcopy(base)
    t1: list[dict[str, Any]] = []
    for container, key in _input_fields(task):
        text = _get(e1, container, key)
        _set(e1, container, key, canonicalize_text(text, romanizer, t1, key))
    if _target_only(task):
        text = str(e1.get("target", {}).get("response", ""))
        e1.setdefault("target", {})["response"] = canonicalize_text(
            text, romanizer, t1, "target_response"
        )
    e1["condition"] = "EXP-01"
    e1.pop("mix_trace", None)

    e2 = copy.deepcopy(e1)
    t2: list[dict[str, Any]] = []
    changed_all: list[dict[str, Any]] = []
    rng = derive_rng(base_seed, e2["example_id"], "EXP-02")
    for container, key in _input_fields(task):
        text = _get(e2, container, key)
        new, changed = realistic_from_canonical(
            text, romanizer, rng, p_variant, t2, key
        )
        _set(e2, container, key, new)
        changed_all.extend(changed)
    if _target_only(task):
        text = str(e2.get("target", {}).get("response", ""))
        new, changed = realistic_from_canonical(
            text, romanizer, rng, p_variant, t2, "target_response"
        )
        e2.setdefault("target", {})["response"] = new
        changed_all.extend(changed)
    e2["condition"] = "EXP-02"
    e2["p_variant"] = p_variant

    e3 = copy.deepcopy(e1)
    t3: list[dict[str, Any]] = []
    rng3 = derive_rng(base_seed, e3["example_id"], "EXP-03")
    for container, key in _input_fields(task):
        text = _get(e3, container, key)
        new = noise_from_changes(text, changed_all, romanizer, rng3, t3, key)
        _set(e3, container, key, new)
    if _target_only(task):
        text = str(e3.get("target", {}).get("response", ""))
        new = noise_from_changes(
            text, changed_all, romanizer, rng3, t3, "target_response"
        )
        e3.setdefault("target", {})["response"] = new
    e3["condition"] = "EXP-03"

    return {
        "EXP-01": e1,
        "EXP-02": e2,
        "EXP-03": e3,
        "traces": {"EXP-01": t1, "EXP-02": t2, "EXP-03": t3},
    }


def generate_conditions(
    bases: list[dict[str, Any]],
    romanizer: DakshinaRomanizer,
    p_variant: float = 0.35,
    base_seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    exp01: list[dict[str, Any]] = []
    exp02: list[dict[str, Any]] = []
    exp03: list[dict[str, Any]] = []
    traces01: list[dict[str, Any]] = []
    traces02: list[dict[str, Any]] = []
    traces03: list[dict[str, Any]] = []

    for base in bases:
        task = base["task"]
        e1 = copy.deepcopy(base)
        t1: list[dict[str, Any]] = []
        for container, key in _input_fields(task):
            text = _get(e1, container, key)
            _set(e1, container, key, canonicalize_text(text, romanizer, t1, key))
        if _target_only(task):
            text = str(e1.get("target", {}).get("response", ""))
            e1.setdefault("target", {})["response"] = canonicalize_text(
                text, romanizer, t1, "target_response"
            )
        e1["condition"] = "EXP-01"
        e1["augmentation_trace"] = t1
        exp01.append(e1)
        traces01.append(
            {"example_id": e1["example_id"], "condition": "EXP-01", "tokens": t1}
        )

        e2 = copy.deepcopy(e1)
        t2: list[dict[str, Any]] = []
        changed_all: list[dict[str, Any]] = []
        rng = derive_rng(base_seed, e2["example_id"], "EXP-02")
        for container, key in _input_fields(task):
            text = _get(e2, container, key)
            new, changed = realistic_from_canonical(
                text, romanizer, rng, p_variant, t2, key
            )
            _set(e2, container, key, new)
            changed_all.extend(changed)
        if _target_only(task):
            text = str(e2.get("target", {}).get("response", ""))
            new, changed = realistic_from_canonical(
                text, romanizer, rng, p_variant, t2, "target_response"
            )
            e2.setdefault("target", {})["response"] = new
            changed_all.extend(changed)
        e2["condition"] = "EXP-02"
        e2["p_variant"] = p_variant
        e2["augmentation_trace"] = t2
        exp02.append(e2)
        traces02.append(
            {"example_id": e2["example_id"], "condition": "EXP-02", "tokens": t2}
        )

        e3 = copy.deepcopy(e1)
        t3: list[dict[str, Any]] = []
        rng3 = derive_rng(base_seed, e3["example_id"], "EXP-03")
        for container, key in _input_fields(task):
            text = _get(e3, container, key)
            new = noise_from_changes(text, changed_all, romanizer, rng3, t3, key)
            _set(e3, container, key, new)
        if _target_only(task):
            text = str(e3.get("target", {}).get("response", ""))
            new = noise_from_changes(
                text, changed_all, romanizer, rng3, t3, "target_response"
            )
            e3.setdefault("target", {})["response"] = new
        e3["condition"] = "EXP-03"
        e3["augmentation_trace"] = t3
        exp03.append(e3)
        traces03.append(
            {"example_id": e3["example_id"], "condition": "EXP-03", "tokens": t3}
        )

    return {
        "EXP-01": exp01,
        "EXP-02": exp02,
        "EXP-03": exp03,
        "traces": {"EXP-01": traces01, "EXP-02": traces02, "EXP-03": traces03},
    }
