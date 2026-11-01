from __future__ import annotations

import random
from typing import Any

from romcom_data.canonical_tie import pick_canonical_variant
from romcom_data.romanize import DakshinaRomanizer
from romcom_data.stable_hash import derive_example_seed, derive_rng
from romcom_data.text_utils import case_fold_key, tokenize_conservative

LATIN_CHARS = "abcdefghijklmnopqrstuvwxyz"


def _preserve_case(original: str, replacement: str) -> str:
    if not original or not replacement:
        return replacement
    if original.isupper() and original.isalpha():
        return replacement.upper()
    if original[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def levenshtein_ops(a: str, b: str) -> list[tuple[str, int, str]]:
    """Return approximate edit ops from a -> b as (op, index, char)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    ops: list[tuple[str, int, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("sub", i - 1, b[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", i, b[j - 1]))
            j -= 1
        else:
            ops.append(("del", i - 1, a[i - 1] if i else ""))
            i -= 1
    ops.reverse()
    return ops


def levenshtein_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def _one_random_edit(token: str, rng: random.Random) -> str:
    if not token:
        return rng.choice(LATIN_CHARS)
    op = rng.choice(("sub", "ins", "del"))
    chars = list(token)
    if op == "sub":
        i = rng.randrange(len(chars))
        orig = chars[i]
        cand = rng.choice(LATIN_CHARS)
        while cand == orig.casefold() and len(LATIN_CHARS) > 1:
            cand = rng.choice(LATIN_CHARS)
        chars[i] = cand.upper() if orig.isupper() else cand
    elif op == "ins":
        i = rng.randrange(len(chars) + 1)
        chars.insert(i, rng.choice(LATIN_CHARS))
    elif len(chars) > 1:
        i = rng.randrange(len(chars))
        chars.pop(i)
    return "".join(chars)


def noise_token_at_distance(
    token: str,
    target_dist: int,
    rng: random.Random,
    forbidden: set[str],
    max_attempts: int = 48,
) -> tuple[str, bool, int]:
    if target_dist == 0:
        return token, False, 0
    for attempt in range(1, max_attempts + 1):
        current = token
        steps = max(target_dist * 3, 4)
        for _ in range(steps):
            current = _one_random_edit(current, rng)
            dist = levenshtein_distance(token.casefold(), current.casefold())
            if (
                dist == target_dist
                and current.casefold() not in forbidden
                and current.casefold() != token.casefold()
            ):
                return current, True, attempt
    return token, False, max_attempts


def apply_random_ops_reverse(token: str, ops: list[tuple[str, int, str]], rng: random.Random) -> str:
    """Apply edit ops from end to start so indices stay valid."""
    chars = list(token)
    for op, idx, ch in reversed(ops):
        if not chars:
            chars = [rng.choice(LATIN_CHARS)]
            continue
        if op == "sub":
            i = min(max(idx, 0), len(chars) - 1)
            orig = chars[i]
            cand = ch if ch and ch.isalpha() else rng.choice(LATIN_CHARS)
            while cand.casefold() == orig.casefold() and len(LATIN_CHARS) > 1:
                cand = rng.choice(LATIN_CHARS)
            chars[i] = cand.upper() if orig.isupper() else cand
        elif op == "ins":
            i = min(max(idx, 0), len(chars))
            chars.insert(i, ch if ch and ch.isalpha() else rng.choice(LATIN_CHARS))
        elif op == "del" and len(chars) > 1:
            i = min(max(idx, 0), len(chars) - 1)
            chars.pop(i)
    result = "".join(chars) if chars else rng.choice(LATIN_CHARS)
    return result


def replace_token_spans(text: str, replacements: dict[tuple[int, int], str]) -> str:
    if not replacements:
        return text
    out: list[str] = []
    last = 0
    for start, end in sorted(replacements):
        out.append(text[last:start])
        out.append(replacements[(start, end)])
        last = end
    out.append(text[last:])
    return "".join(out)


def eligible_alias_spans(
    text: str,
    romanizer: DakshinaRomanizer,
    protected: set[str] | None = None,
) -> list[dict[str, Any]]:
    protected = protected or set()
    hits: list[dict[str, Any]] = []
    for tok in tokenize_conservative(text):
        if not tok.is_latin_word:
            continue
        if tok.text.isupper() and len(tok.text) >= 2:
            continue
        key = case_fold_key(tok.text)
        if key in protected:
            continue
        hit = romanizer.lookup_alias(tok.text)
        if hit is None:
            continue
        variants = romanizer.native_to_variants.get(hit.native or "", [])
        hits.append(
            {
                "start": tok.start,
                "end": tok.end,
                "token": tok.text,
                "native": hit.native,
                "canonical": hit.canonical or hit.surface,
                "variants": variants,
            }
        )
    return hits


def canonicalize_text(
    text: str, romanizer: DakshinaRomanizer, traces: list[dict[str, Any]], field: str
) -> str:
    spans = eligible_alias_spans(text, romanizer)
    repl: dict[tuple[int, int], str] = {}
    for span in spans:
        variants = span["variants"]
        if not variants:
            continue
        chosen, max_c, tied = pick_canonical_variant(variants)
        new = _preserve_case(span["token"], chosen)
        if tied:
            traces.append(
                {
                    "field": field,
                    "operation": "canonical_tie_fallback",
                    "reason": "lexicographic_reproducibility_not_empirical_preference",
                    "native": span["native"],
                    "tied_highest_frequency": True,
                }
            )
        if new != span["token"]:
            repl[(span["start"], span["end"])] = new
            traces.append(
                {
                    "field": field,
                    "original": span["token"],
                    "selected": new,
                    "operation": "canonicalize",
                    "native": span["native"],
                    "attestation_count": max_c,
                    "offsets": [span["start"], span["end"]],
                }
            )
        else:
            traces.append(
                {
                    "field": field,
                    "original": span["token"],
                    "operation": "unchanged",
                    "reason": "already_canonical",
                    "native": span["native"],
                    "offsets": [span["start"], span["end"]],
                }
            )
    return replace_token_spans(text, repl)


def realistic_from_canonical(
    canonical_text: str,
    romanizer: DakshinaRomanizer,
    rng: random.Random,
    p_variant: float,
    traces: list[dict[str, Any]],
    field: str,
) -> tuple[str, list[dict[str, Any]]]:
    spans = eligible_alias_spans(canonical_text, romanizer)
    repl: dict[tuple[int, int], str] = {}
    changed: list[dict[str, Any]] = []
    for span in spans:
        variants = [(v, c) for v, c in span["variants"] if v.casefold() != span["token"].casefold()]
        if not variants:
            continue
        if rng.random() > p_variant:
            continue
        weights = [c for _, c in variants]
        chosen_v, chosen_c = rng.choices(variants, weights=weights, k=1)[0]
        new = _preserve_case(span["token"], chosen_v)
        repl[(span["start"], span["end"])] = new
        rec = {
            "field": field,
            "original": span["token"],
            "selected": new,
            "operation": "attested_variant",
            "native": span["native"],
            "attestation_count": chosen_c,
            "p_variant": p_variant,
            "offsets": [span["start"], span["end"]],
        }
        traces.append(rec)
        changed.append(rec)
    return replace_token_spans(canonical_text, repl), changed


def noise_from_changes(
    canonical_text: str,
    changes: list[dict[str, Any]],
    romanizer: DakshinaRomanizer,
    rng: random.Random,
    traces: list[dict[str, Any]],
    field: str,
    max_resample: int = 32,
) -> str:
    repl: dict[tuple[int, int], str] = {}
    for ch in changes:
        if ch.get("field") != field:
            continue
        start, end = ch["offsets"]
        token = canonical_text[start:end]
        selected = ch["selected"]
        native = ch.get("native")
        target_dist = levenshtein_distance(token.casefold(), selected.casefold())
        known = {v.casefold() for v, _ in romanizer.native_to_variants.get(native, [])} if native else set()
        noisy, matched, attempts = noise_token_at_distance(
            token, target_dist, rng, known, max_attempts=max_resample
        )
        noisy = _preserve_case(token, noisy)
        repl[(start, end)] = noisy
        traces.append(
            {
                "field": field,
                "original": token,
                "canonical_token": token,
                "exp02_selected": selected,
                "selected": noisy,
                "operation": "random_noise",
                "edit_magnitude": target_dist,
                "target_levenshtein_distance": target_dist,
                "distance_match": matched,
                "resample_attempts": attempts,
                "native": native,
                "offsets": [start, end],
                "mask_from_exp02": True,
            }
        )
    return replace_token_spans(canonical_text, repl)
