from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")
REPLACEMENT = "\ufffd"
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def case_fold_key(text: str) -> str:
    return nfc(text.strip()).casefold()


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int
    is_latin_word: bool


def tokenize_conservative(text: str) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    for match in re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?|\S+|\s+", text):
        piece = match.group(0)
        start, end = match.span()
        is_latin = bool(re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", piece))
        spans.append(TokenSpan(piece, start, end, is_latin))
    return spans


def script_profile(text: str) -> dict[str, float]:
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    total = dev + lat
    if total == 0:
        return {"devanagari_ratio": 0.0, "latin_ratio": 0.0, "mixed": False}
    dr = dev / total
    lr = lat / total
    mixed = dr > 0.05 and lr > 0.05
    return {"devanagari_ratio": dr, "latin_ratio": lr, "mixed": mixed}


def has_malformed_unicode(text: str) -> bool:
    if REPLACEMENT in text:
        return True
    if CONTROL.search(text):
        return True
    return False
