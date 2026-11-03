"""Unit tests for NLI experiment romanization and variants."""

from __future__ import annotations

import pytest

from romcom_data.augmentation import levenshtein_distance
from romcom_data.nli_exp_freeze import load_exclusion_keys
from romcom_data.nli_exp_romanize import (
    NliLexicon,
    audit_devanagari_codepoints,
    tokenize_and_romanize_field,
)
from romcom_data.nli_exp_variants import (
    VariantPools,
    _apply_replacements,
    _noise_at_distance,
    apply_exp02,
    apply_exp03,
    build_noncanonical_pools,
)
from romcom_data.text_utils import DEVANAGARI, nfc


@pytest.fixture(scope="module")
def lexicon() -> NliLexicon:
    return NliLexicon()


def test_danda_and_digits_romanize(lexicon: NliLexicon) -> None:
    field = tokenize_and_romanize_field("वह १२ बजे आया।", "premise", lexicon)
    assert "12" in field.roman
    assert field.roman.endswith(".")
    assert not DEVANAGARI.search(field.roman)


def test_latin_passthrough_unchanged(lexicon: NliLexicon) -> None:
    field = tokenize_and_romanize_field("NASA और FBI", "premise", lexicon)
    assert "NASA" in field.roman
    assert "FBI" in field.roman


def test_noncanonical_pool_excludes_canonical(lexicon: NliLexicon) -> None:
    pools = build_noncanonical_pools(lexicon)
    for native, alts in list(pools.items())[:50]:
        canon = lexicon.native_to_canonical[native].casefold()
        for variant, _ in alts:
            assert variant != canon
            assert levenshtein_distance(canon, variant) >= 1


def test_exp03_exact_distance() -> None:
    import random

    rng = random.Random(0)
    noisy, ok, _ = _noise_at_distance("hello", 2, rng, set(), "world")
    assert ok
    assert levenshtein_distance("hello", noisy) == 2


def test_apply_replacements_reverse_order() -> None:
    text = "0123456789"
    repl = {(0, 2): "aa", (5, 7): "bbb"}
    out = _apply_replacements(text, repl)
    assert out == "aa234bbb789"


def test_exclusion_keys_load() -> None:
    keys = load_exclusion_keys()
    assert "train" in keys
    assert len(keys["dev"]) >= 3


def test_handled_codepoints_cover_audit_sample(lexicon: NliLexicon) -> None:
    sample = ["नमस्ते दुनिया।", "मैं ५ साल का हूँ।"]
    report = audit_devanagari_codepoints(sample)
    assert report["ok"]


def test_exp02_exp03_roundtrip_small(lexicon: NliLexicon) -> None:
    pools = VariantPools(lexicon)
    text = nfc("यह एक परीक्षण वाक्य है।")
    field = tokenize_and_romanize_field(text, "premise", lexicon)
    assert not field.fallback_failures
    exp02, t2 = apply_exp02(field, pools, "train-000001")
    exp03, t3 = apply_exp03(field, t2, pools, "train-000001")
    assert isinstance(exp02, str)
    assert isinstance(exp03, str)
    if t2:
        assert len(t2) == len(t3)
        for a, b in zip(t2, t3, strict=True):
            assert a["token_index"] == b["token_index"]
            assert a["native_offsets"] == b["native_offsets"]
            assert a["levenshtein_distance"] == b["levenshtein_distance"]


def test_fallback_strips_nukta(lexicon: NliLexicon) -> None:
    field = tokenize_and_romanize_field("\u092a\u0921\u093c", "premise", lexicon)
    assert not field.fallback_failures
    assert not DEVANAGARI.search(field.roman)


def test_mixed_devanagari_digits_in_token(lexicon: NliLexicon) -> None:
    field = tokenize_and_romanize_field("१४वीं सदी", "premise", lexicon)
    assert not field.fallback_failures
    assert "14" in field.roman
    assert not DEVANAGARI.search(field.roman)


def test_experiment_manifest_row_counts_match_disk() -> None:
    from pathlib import Path

    from romcom_data.nli_exp_build import collect_row_counts_from_disk
    from romcom_data.nli_exp_paths import NLI_EXPERIMENTS

    manifest_path = NLI_EXPERIMENTS / "experiment_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("experiment artifacts not built")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    disk = dict(collect_row_counts_from_disk())
    assert manifest.get("row_counts") == disk
    assert manifest["row_counts"]["EXP-00:both:dev"] == 2487
    assert manifest["row_counts"]["english_retention:both:test"] == 5007
    assert not Path(NLI_EXPERIMENTS / "english_retention/train.jsonl").is_file()
