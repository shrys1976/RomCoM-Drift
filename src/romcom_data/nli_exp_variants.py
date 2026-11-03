"""EXP-02 attested variants and EXP-03 matched noise for NLI."""

from __future__ import annotations

import random
from typing import Any

from romcom_data.augmentation import levenshtein_distance
from romcom_data.nli_exp_romanize import FieldRomanization, HindiToken, NliLexicon
from romcom_data.stable_hash import derive_rng
from romcom_data.text_utils import DEVANAGARI

LATIN = "abcdefghijklmnopqrstuvwxyz"
P_VARIANT = 0.35
BASE_SEED = 42
MAX_NOISE_ATTEMPTS = 64


def build_noncanonical_pools(lexicon: NliLexicon) -> dict[str, list[list[Any]]]:
    """native -> [[variant, count], ...] excluding canonical with d>=1."""
    pools: dict[str, list[list[Any]]] = {}
    for native, pairs in lexicon.native_to_variants.items():
        canon = lexicon.native_to_canonical.get(native, "").casefold()
        alts: list[list[Any]] = []
        for variant, count in pairs:
            v = variant.casefold()
            if DEVANAGARI.search(v):
                continue
            if levenshtein_distance(canon, v) < 1:
                continue
            alts.append([v, count])
        if alts:
            pools[native] = alts
    return pools


class VariantPools:
    def __init__(self, lexicon: NliLexicon) -> None:
        self._pools = build_noncanonical_pools(lexicon)
        self._full_variants: dict[str, set[str]] = {}
        for native, pairs in lexicon.native_to_variants.items():
            self._full_variants[native] = {v.casefold() for v, _ in pairs}

    def pool(self, native_word: str) -> list[tuple[str, int]]:
        core = native_word.rstrip("।॥").strip()
        from romcom_data.text_utils import nfc

        key = nfc(core)
        raw = self._pools.get(key, [])
        return [(str(v), int(c)) for v, c in raw]

    def attested_all(self, native_word: str) -> set[str]:
        core = native_word.rstrip("।॥").strip()
        from romcom_data.text_utils import nfc

        return self._full_variants.get(nfc(core), set())

    def eligible(self, token: HindiToken) -> bool:
        if token.method != "dakshina_canonical" or not token.canonical:
            return False
        lookup = token.native_text.rstrip("।॥").strip()
        from romcom_data.text_utils import nfc

        return bool(self.pool(nfc(lookup)))


def _apply_replacements(text: str, repl: dict[tuple[int, int], str]) -> str:
    if not repl:
        return text
    out = text
    for start, end in sorted(repl.keys(), reverse=True):
        out = out[:start] + repl[(start, end)] + out[end:]
    return out


def _noise_at_distance(
    canonical: str,
    target_dist: int,
    rng: random.Random,
    forbidden: set[str],
    must_differ_from: str,
) -> tuple[str, bool, int]:
    if target_dist < 1:
        return canonical, False, 0
    for attempt in range(1, MAX_NOISE_ATTEMPTS + 1):
        chars = list(canonical)
        length = len(chars)
        subs = min(target_dist, length)
        indices = rng.sample(range(length), subs) if subs else []
        for idx in indices:
            orig = chars[idx]
            cand = rng.choice(LATIN)
            while cand == orig and len(LATIN) > 1:
                cand = rng.choice(LATIN)
            chars[idx] = cand
        extra_ins = target_dist - subs
        for _ in range(extra_ins):
            pos = rng.randrange(len(chars) + 1)
            chars.insert(pos, rng.choice(LATIN))
        candidate = "".join(chars)
        dist = levenshtein_distance(canonical.casefold(), candidate.casefold())
        if (
            dist == target_dist
            and candidate.casefold() not in forbidden
            and candidate.casefold() != must_differ_from.casefold()
        ):
            return candidate, True, attempt
    return canonical, False, MAX_NOISE_ATTEMPTS


def apply_exp02(
    field: FieldRomanization,
    pools: VariantPools,
    example_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    repl: dict[tuple[int, int], str] = {}
    traces: list[dict[str, Any]] = []
    for token in field.tokens:
        if not pools.eligible(token):
            continue
        rng = derive_rng(BASE_SEED, example_id, "EXP-02", salt=token.field)
        if rng.random() > P_VARIANT:
            continue
        from romcom_data.text_utils import nfc

        native_key = nfc(token.native_text.rstrip("।॥").strip())
        pool = pools.pool(native_key)
        if not pool:
            continue
        weights = [c for _, c in pool]
        chosen, count = rng.choices(pool, weights=weights, k=1)[0]
        canon = token.canonical or ""
        d = levenshtein_distance(canon, chosen)
        if d < 1:
            raise ValueError(f"EXP-02 d=0 for {example_id} {token.field} {native_key}")
        danda = ""
        if token.output.endswith(".") and token.native_text.rstrip().endswith(("।", "॥")):
            danda = "."
        new_tok = chosen + danda if token.output.endswith(".") and not chosen.endswith(".") else chosen
        repl[(token.roman_start, token.roman_end)] = new_tok
        traces.append(
            {
                "field": token.field,
                "token_index": token.token_index,
                "native_offsets": [token.native_start, token.native_end],
                "canonical": canon,
                "selected": new_tok,
                "operation": "attested_variant",
                "attestation_count": count,
                "pool_size": len(pool),
                "levenshtein_distance": d,
            }
        )
    return _apply_replacements(field.roman, repl), traces


def apply_exp03(
    field: FieldRomanization,
    exp02_traces: list[dict[str, Any]],
    pools: VariantPools,
    example_id: str,
) -> tuple[str, list[dict[str, Any]]]:
    repl: dict[tuple[int, int], str] = {}
    traces: list[dict[str, Any]] = []
    for change in exp02_traces:
        if change.get("operation") != "attested_variant":
            continue
        field_name = change["field"]
        rng = derive_rng(BASE_SEED, example_id, "EXP-03", salt=field_name)
        token_index = change["token_index"]
        token = next(t for t in field.tokens if t.token_index == token_index)
        canon = change["canonical"]
        exp02_sel = change["selected"]
        d = int(change["levenshtein_distance"])
        native_key = token.native_text.rstrip("।॥").strip()
        from romcom_data.text_utils import nfc

        forbidden = pools.attested_all(nfc(native_key))
        noisy, ok, attempts = _noise_at_distance(canon, d, rng, forbidden, exp02_sel)
        if not ok:
            raise ValueError(f"EXP-03 failed for {example_id} field={field_name} token={token_index}")
        repl[(token.roman_start, token.roman_end)] = noisy
        traces.append(
            {
                "field": field_name,
                "token_index": token_index,
                "native_offsets": change["native_offsets"],
                "canonical": canon,
                "exp02_selected": exp02_sel,
                "selected": noisy,
                "operation": "random_noise",
                "levenshtein_distance": d,
                "resample_attempts": attempts,
            }
        )
    return _apply_replacements(field.roman, repl), traces
