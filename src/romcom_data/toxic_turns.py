"""Parse Toxic_Matrix / Dolly_T conversation fields.

The Hindi experiment uses ``hin_Deva``. Slot 0 is the user turn and slot 1 is
the assistant turn because that is the source list structure. ``eng_Latn`` is a
parallel reference for alignment checks only. ``hin_Latn`` is never a role source.
"""

from __future__ import annotations

from typing import Any

from romcom_data.text_utils import nfc

CANONICAL_TEXT_FIELD = "hin_Deva"
REFERENCE_TEXT_FIELD = "eng_Latn"


def _blocks(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        return []
    blocks: list[tuple[str, str]] = []
    for block in value:
        if isinstance(block, list):
            user = nfc(str(block[0])) if len(block) > 0 and block[0] is not None else ""
            assistant = nfc(str(block[1])) if len(block) > 1 and block[1] is not None else ""
            blocks.append((user, assistant))
        elif block:
            blocks.append((nfc(str(block)), ""))
    return blocks


def canonical_text_field(row: dict[str, Any]) -> str:
    """Hindi Devanagari when the row has that field; English only for fixtures without it."""
    if "hin_Deva" in row and row.get("hin_Deva") not in (None, "", []):
        return CANONICAL_TEXT_FIELD
    if "hin_Deva" not in row and row.get("eng_Latn"):
        return REFERENCE_TEXT_FIELD
    return CANONICAL_TEXT_FIELD


def parse_conversation_turns(
    row: dict[str, Any],
    *,
    text_field: str | None = None,
) -> list[dict[str, str]]:
    """Chronological turns. Roles come from list position, not from string length."""
    field_name = text_field or canonical_text_field(row)
    if field_name == "hin_Latn":
        raise ValueError("hin_Latn cannot be used as a role or alignment source")
    turns: list[dict[str, str]] = []
    for user_text, assistant_text in _blocks(row.get(field_name)):
        if user_text:
            turns.append(
                {
                    "role": "user",
                    "text": user_text,
                    "text_field": field_name,
                    "source_slot": "0",
                }
            )
        if assistant_text:
            turns.append(
                {
                    "role": "assistant",
                    "text": assistant_text,
                    "text_field": field_name,
                    "source_slot": "1",
                }
            )
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
