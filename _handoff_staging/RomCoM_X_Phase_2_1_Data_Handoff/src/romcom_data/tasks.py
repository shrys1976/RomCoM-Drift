from __future__ import annotations

from typing import Any


def run_task_pipeline_if_cleared(gate: dict[str, Any]) -> dict[str, Any]:
    if not gate.get("cleared"):
        return {"skipped": True, "reason": gate.get("reason")}
    raise NotImplementedError(
        "Task normalization requires official Indi-RomCoM training files."
    )
