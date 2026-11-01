from __future__ import annotations

import hashlib
import struct


def stable_hash_bytes(*parts: str) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.digest()


def stable_hash_hex(*parts: str) -> str:
    return stable_hash_bytes(*parts).hex()


def derive_example_seed(base_seed: int, example_id: str, condition: str) -> int:
    digest = stable_hash_bytes(str(base_seed), example_id, condition)
    return struct.unpack(">Q", digest[:8])[0]


def derive_rng(base_seed: int, example_id: str, condition: str, salt: str = ""):
    import random

    seed = derive_example_seed(base_seed, example_id, condition)
    if salt:
        seed ^= derive_example_seed(base_seed, example_id, salt)
    return random.Random(seed)
