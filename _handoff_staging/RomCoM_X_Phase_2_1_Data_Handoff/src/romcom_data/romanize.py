"""Dakshina-first Hindi romanization with a documented character-map fallback."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from romcom_data.canonical_tie import pick_canonical_variant
from romcom_data.paths import INTERIM, PROJECT_ROOT, RAW_DAKSHINA_HI
from romcom_data.text_utils import DEVANAGARI, case_fold_key, nfc

# Compact NFC Devanagari -> Latin map used ONLY when Dakshina has no entry.
# This is a reconstruction fallback, not an attested Dakshina spelling.
_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gy",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "dh", "फ़": "f",
}
_INDEPENDENT = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऋ": "ri",
}
_MATRA = {
    "ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri",
    "ं": "n", "ः": "h", "ँ": "n",
}
_VIRAMA = "्"


@dataclass(frozen=True)
class RomanizationHit:
    surface: str
    method: str
    native: str | None
    canonical: str | None
    attested: bool


def fallback_transliterate(native: str) -> str:
    text = nfc(native)
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _INDEPENDENT:
            out.append(_INDEPENDENT[ch])
            i += 1
            continue
        if ch in _CONSONANTS:
            cons = _CONSONANTS[ch]
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt == _VIRAMA:
                out.append(cons)
                i += 2
                continue
            if nxt in _MATRA:
                vowel = _MATRA[nxt]
                out.append(cons + vowel)
                i += 2
                continue
            out.append(cons + "a")
            i += 1
            continue
        if ch in _MATRA:
            out.append(_MATRA[ch])
            i += 1
            continue
        if ch == _VIRAMA:
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


class DakshinaRomanizer:
    def __init__(self, inventory_path: Path | None = None) -> None:
        path = inventory_path or (
            INTERIM / "romanization" / "dakshina_hi_variant_inventory.tsv"
        )
        if not path.is_file():
            path = RAW_DAKSHINA_HI / "hi.translit.sampled.train.tsv"
            df = pd.read_csv(
                path, sep="\t", header=None, names=["native", "variant", "count"]
            )
            self.native_to_canonical: dict[str, str] = {}
            self.native_to_variants: dict[str, list[tuple[str, int]]] = {}
            for native, grp in df.groupby("native"):
                pairs = [
                    (str(r.variant), int(r.count))
                    for r in grp.itertuples()
                    if isinstance(r.variant, str) and str(r.variant).strip()
                ]
                if not pairs:
                    continue
                chosen, _, _ = pick_canonical_variant(pairs)
                self.native_to_canonical[str(native)] = chosen
                self.native_to_variants[str(native)] = pairs
            self.alias_to_native: dict[str, str] = {}
            self.ambiguous_aliases: set[str] = set()
            return

        inv = pd.read_csv(path, sep="\t")
        native_col = "native_word" if "native_word" in inv.columns else "native"
        var_col = "romanized_variant" if "romanized_variant" in inv.columns else "variant"
        self.native_to_canonical = {}
        self.native_to_variants = {}
        for native, grp in inv.groupby(native_col):
            pairs = [
                (str(getattr(r, var_col)), int(r.attestation_count))
                for r in grp.itertuples()
            ]
            if "is_canonical" in grp.columns:
                flag = grp["is_canonical"].astype(str).str.lower().isin(["true", "1"])
                canon_rows = grp[flag]
            else:
                canon_rows = grp.iloc[0:0]
            if len(canon_rows):
                canon = str(canon_rows.iloc[0][var_col])
            else:
                canon, _, _ = pick_canonical_variant(pairs)
            self.native_to_canonical[str(native)] = canon
            self.native_to_variants[str(native)] = pairs

        alias_path = path.parent / "dakshina_hi_alias_index.tsv"
        self.alias_to_native = {}
        self.ambiguous_aliases = set()
        if alias_path.is_file():
            alias = pd.read_csv(alias_path, sep="\t")
            status_col = "ambiguity_status" if "ambiguity_status" in alias.columns else None
            for r in alias.itertuples():
                if status_col:
                    status = str(getattr(r, status_col))
                    key = str(getattr(r, "romanized_alias", getattr(r, "normalized_variant", "")))
                    if status != "unique":
                        self.ambiguous_aliases.add(case_fold_key(key))
                        continue
                    native = str(getattr(r, "candidate_native_words", getattr(r, "native", "")))
                    self.alias_to_native[case_fold_key(key)] = native.split("|")[0]
                else:
                    key = str(r.normalized_variant)
                    self.alias_to_native[key] = str(r.native)

    def romanize_native(self, native: str) -> RomanizationHit:
        word = nfc(native.strip())
        if word in self.native_to_canonical:
            canon = self.native_to_canonical[word]
            return RomanizationHit(canon, "dakshina_canonical", word, canon, True)
        if DEVANAGARI.search(word):
            fb = fallback_transliterate(word)
            return RomanizationHit(fb, "fallback_indic_map", word, None, False)
        return RomanizationHit(word, "passthrough", None, None, False)

    def lookup_alias(self, latin_token: str) -> RomanizationHit | None:
        key = case_fold_key(latin_token)
        if key in self.ambiguous_aliases:
            return None
        native = self.alias_to_native.get(key)
        if not native:
            return None
        return self.romanize_native(native)


@lru_cache(maxsize=1)
def default_romanizer() -> DakshinaRomanizer:
    return DakshinaRomanizer()
