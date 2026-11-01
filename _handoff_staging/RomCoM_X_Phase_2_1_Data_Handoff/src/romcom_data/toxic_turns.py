"""Parse Toxic_Matrix / Dolly_T nested conversation fields."""

from __future__ import annotations

from typing import Any

from romcom_data.text_utils import nfc


def _first_pair(value: Any) -> tuple[str, str]:
    """Extract (user_text, assistant_text) from eng_Latn / hin_* nested lists."""
    if value is None:
        return "", ""
    if isinstance(value, str):
        return nfc(value), ""
    if isinstance(value, list):
        if not value:
            return "", ""
        inner = value[0]
        if isinstance(inner, list):
            user = nfc(str(inner[0])) if len(inner) > 0 and inner[0] is not None else ""
            asst = nfc(str(inner[1])) if len(inner) > 1 and inner[1] is not None else ""
            return user, asst
        if isinstance(inner, str):
            return nfc(inner), ""
    return nfc(str(value)), ""


def parse_conversation_turns(row: dict[str, Any]) -> list[dict[str, str]]:
    """Chronological turns with roles; no concatenation across roles."""
    turns: list[dict[str, str]] = []
    eng = row.get("eng_Latn")
    if isinstance(eng, list):
        for block in eng:
            if isinstance(block, list):
                if len(block) > 0 and block[0]:
                    turns.append({"role": "user", "text": nfc(str(block[0]))})
                if len(block) > 1 and block[1]:
                    turns.append({"role": "assistant", "text": nfc(str(block[1]))})
            elif block:
                turns.append({"role": "user", "text": nfc(str(block))})
    elif eng:
        turns.append({"role": "user", "text": nfc(str(eng))})

    if not any(t["role"] == "assistant" for t in turns):
        _, asst = _first_pair(row.get("hin_Latn") or row.get("hin_Deva"))
        if asst:
            turns.append({"role": "assistant", "text": asst})
    return turns


def user_prompt_from_turns(turns: list[dict[str, str]]) -> str:
    user_parts = [t["text"] for t in turns if t["role"] == "user" and t["text"]]
    if len(user_parts) == 1:
        return user_parts[0]
    if len(user_parts) > 1:
        return "\n---\n".join(user_parts)
    return ""


def assistant_target_from_turns(turns: list[dict[str, str]]) -> str:
    asst_parts = [t["text"] for t in turns if t["role"] == "assistant" and t["text"]]
    if len(asst_parts) == 1:
        return asst_parts[0]
    if len(asst_parts) > 1:
        return "\n---\n".join(asst_parts)
    return ""
