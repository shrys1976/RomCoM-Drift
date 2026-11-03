"""Hindi field tokenization and EXP-01 romanization for NLI experiments."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from romcom_data.canonical_tie import pick_canonical_variant
from romcom_data.nli_exp_paths import NLI_INVENTORIES
from romcom_data.romanize import fallback_transliterate
from romcom_data.text_utils import DEVANAGARI, nfc

DEVANAGARI_BLOCK = re.compile(r"[\u0900-\u097F]")
_LATIN_RUN = re.compile(r"[A-Za-z0-9]+")
_DANDA = {"\u0964": ".", "\u0965": "."}
_DEVANAGARI_DIGIT = str.maketrans("०१२३४५६७८९", "0123456789")
_VOWEL_SIGN = set("ािीुूेैोौॉॊॅॆृ")
_CONSONANTS_END = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
_PRE_FALLBACK_MAP = {
    "\u0970": "",  # abbreviation sign
    "\u0929": "lla",
    "\u090e": "e",
    "\u090d": "o",
}


@dataclass
class HindiToken:
    token_index: int
    field: str
    native_start: int
    native_end: int
    native_text: str
    output: str
    method: str
    canonical: str | None = None
    attestation_count: int | None = None
    tied_canonical: bool = False
    roman_start: int = 0
    roman_end: int = 0


@dataclass
class FieldRomanization:
    native: str
    roman: str
    tokens: list[HindiToken] = field(default_factory=list)
    fallback_failures: list[dict[str, Any]] = field(default_factory=list)


class NliLexicon:
    def __init__(self, inventory_path: Path | None = None) -> None:
        path = inventory_path or (NLI_INVENTORIES / "dakshina_hi_variant_inventory.tsv")
        inv = pd.read_csv(path, sep="\t")
        self.inventory_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.native_to_canonical: dict[str, str] = {}
        self.native_to_variants: dict[str, list[tuple[str, int]]] = {}
        native_col = "native_word" if "native_word" in inv.columns else "native"
        var_col = "romanized_variant" if "romanized_variant" in inv.columns else "variant"
        count_col = "attestation_count" if "attestation_count" in inv.columns else "count"
        for native, grp in inv.groupby(native_col):
            pairs: list[tuple[str, int]] = []
            for row in grp.itertuples():
                variant = str(getattr(row, var_col))
                count = int(getattr(row, count_col))
                if variant.strip():
                    pairs.append((variant, count))
            if not pairs:
                continue
            if "is_canonical" in grp.columns:
                canon_rows = grp[grp["is_canonical"].astype(str).str.lower().isin(["true", "1"])]
                if len(canon_rows):
                    canon = str(canon_rows.iloc[0][var_col])
                else:
                    canon, _, _ = pick_canonical_variant(pairs)
            else:
                canon, _, _ = pick_canonical_variant(pairs)
            self.native_to_canonical[str(native)] = canon
            self.native_to_variants[str(native)] = pairs

    def canonical_lower(self, native: str) -> tuple[str, int, bool]:
        word = nfc(native.strip())
        variants = self.native_to_variants.get(word, [])
        if not variants:
            return "", 0, False
        chosen, count, tied = pick_canonical_variant(variants)
        return chosen.casefold(), count, tied


def handled_devanagari_codepoints() -> set[str]:
    from romcom_data.romanize import (
        _CONSONANTS,
        _INDEPENDENT,
        _MATRA,
        _VIRAMA,
    )

    chars = set(_CONSONANTS) | set(_INDEPENDENT) | set(_MATRA) | {_VIRAMA}
    chars |= set("ऽ")  # avagraha
    chars |= set(_DANDA)
    chars |= set("०१२३४५६७८९")
    chars |= {"\u093c", "\u094d", "\u0901", "\u0902", "\u0903"}
    chars |= set(_PRE_FALLBACK_MAP)
    return chars


def _normalize_for_fallback(word: str) -> str:
    out: list[str] = []
    for ch in word:
        out.append(_PRE_FALLBACK_MAP.get(ch, ch))
    return "".join(out)


def _prepare_fallback_input(word: str) -> str:
    return _normalize_for_fallback(word).translate(_DEVANAGARI_DIGIT)


def audit_devanagari_codepoints(texts: list[str]) -> dict[str, Any]:
    handled = handled_devanagari_codepoints()
    counts: dict[str, int] = {}
    unhandled: dict[str, int] = {}
    for text in texts:
        for ch in nfc(text):
            if not DEVANAGARI_BLOCK.match(ch):
                continue
            counts[ch] = counts.get(ch, 0) + 1
            if ch not in handled:
                unhandled[ch] = unhandled.get(ch, 0) + 1
    return {
        "total_devanagari_codepoints": sum(counts.values()),
        "unique_codepoints": len(counts),
        "unhandled_codepoints": [
            {"char": ch, "codepoint": f"U+{ord(ch):04X}", "count": c}
            for ch, c in sorted(unhandled.items(), key=lambda x: -x[1])
        ],
        "ok": len(unhandled) == 0,
    }


def _strip_word_danda(word: str) -> tuple[str, str]:
    suffix = ""
    while word and word[-1] in _DANDA:
        suffix = _DANDA[word[-1]] + suffix
        word = word[:-1]
    return word, suffix


def _schwa_trim(roman: str, native: str) -> str:
    if len(roman) <= 2 or not roman.endswith("a"):
        return roman
    core = native.rstrip("।॥")
    if len(core) < 2:
        return roman
    last = core[-1]
    if last in _VOWEL_SIGN or last == "्":
        return roman
    if last in _CONSONANTS_END or last in "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह":
        return roman[:-1]
    return roman


def _romanize_word(word: str, lexicon: NliLexicon) -> tuple[str, str, str | None, int | None, bool]:
    core, danda_suffix = _strip_word_danda(word)
    if not core:
        return danda_suffix, "punctuation", None, None, False
    if re.fullmatch(r"[A-Za-z0-9]+", core):
        return core, "source_latin", None, None, False
    digit_translated = core.translate(_DEVANAGARI_DIGIT)
    if digit_translated != core and digit_translated.isascii() and digit_translated.isdigit():
        return digit_translated + danda_suffix, "digit_map", None, None, False
    lookup = nfc(core)
    if lookup in lexicon.native_to_canonical:
        canon = lexicon.native_to_canonical[lookup].casefold()
        if DEVANAGARI.search(canon):
            out = _schwa_trim(
                fallback_transliterate(_prepare_fallback_input(lookup)).casefold(), lookup
            )
            return out + danda_suffix, "fallback_charmap", None, None, False
        _, count, tied = pick_canonical_variant(lexicon.native_to_variants[lookup])
        return canon + danda_suffix, "dakshina_canonical", canon, count, tied
    out = _schwa_trim(
        fallback_transliterate(_prepare_fallback_input(lookup)).casefold(), lookup
    )
    return out + danda_suffix, "fallback_charmap", None, None, False


def _is_devanagari_char(ch: str) -> bool:
    return "\u0900" <= ch <= "\u097F"


def tokenize_and_romanize_field(text: str, field: str, lexicon: NliLexicon) -> FieldRomanization:
    text = nfc(text)
    tokens: list[HindiToken] = []
    failures: list[dict[str, Any]] = []
    parts: list[str] = []
    roman_cursor = 0
    token_index = 0
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch.isspace():
            j = i + 1
            while j < length and text[j].isspace():
                j += 1
            parts.append(text[i:j])
            roman_cursor += j - i
            i = j
            continue
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            j = i + 1
            while j < length and (
                ("A" <= text[j] <= "Z")
                or ("a" <= text[j] <= "z")
                or ("0" <= text[j] <= "9")
            ):
                j += 1
            segment = text[i:j]
            parts.append(segment)
            roman_cursor += len(segment)
            i = j
            continue
        if _is_devanagari_char(ch):
            j = i + 1
            while j < length and _is_devanagari_char(text[j]):
                j += 1
            segment = text[i:j]
            start, end = i, j
            output, method, canon, att_count, tied = _romanize_word(segment, lexicon)
            if method not in {"source_latin", "punctuation"} and DEVANAGARI.search(output):
                failures.append(
                    {
                        "field": field,
                        "native_text": segment,
                        "output": output,
                        "native_start": start,
                        "native_end": end,
                    }
                )
            roman_start = roman_cursor
            parts.append(output)
            roman_cursor += len(output)
            if method in {"dakshina_canonical", "fallback_charmap", "digit_map"}:
                tokens.append(
                    HindiToken(
                        token_index=token_index,
                        field=field,
                        native_start=start,
                        native_end=end,
                        native_text=segment,
                        output=output,
                        method=method,
                        canonical=canon,
                        attestation_count=att_count,
                        tied_canonical=tied,
                        roman_start=roman_start,
                        roman_end=roman_cursor,
                    )
                )
                token_index += 1
            i = j
            continue
        parts.append(ch)
        roman_cursor += 1
        i += 1
    roman = "".join(parts)
    if DEVANAGARI.search(roman):
        failures.append({"field": field, "native_text": text, "output": roman, "reason": "field_residual"})
    return FieldRomanization(native=text, roman=roman, tokens=tokens, fallback_failures=failures)
