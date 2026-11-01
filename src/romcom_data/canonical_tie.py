"""Deterministic canonical choice among tied highest-attestation variants."""

from __future__ import annotations

from typing import Sequence


def pick_canonical_variant(variants: Sequence[tuple[str, int]]) -> tuple[str, int, bool]:
    """Return (surface, attestation_count, tied_highest_frequency)."""
    if not variants:
        return "", 0, False
    max_c = max(c for _, c in variants)
    tied = sorted({v for v, c in variants if c == max_c})
    chosen = tied[0]
    return chosen, max_c, len(tied) > 1
