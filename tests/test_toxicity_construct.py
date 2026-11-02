from __future__ import annotations

from romcom_data.generate_exps import TARGET_ONLY_TASKS
from romcom_data.toxicity_construct import FROZEN_MANIFEST_SHA256, load_frozen_manifest


def test_frozen_manifest_sha_matches():
    rows, digest = load_frozen_manifest()
    assert digest == FROZEN_MANIFEST_SHA256
    assert len(rows) == 6998


def test_hindi_refusal_in_target_only_tasks():
    assert "hindi_refusal_response_generation" in TARGET_ONLY_TASKS
