from pathlib import Path

import json

from romcom_data.checksums import sha256_file


def test_source_manifest_exists(project_root: Path) -> None:
    path = project_root / "manifests" / "source_manifest.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "sources" in data
    dakshina = [s for s in data["sources"] if s.get("role") == "dakshina_hi"]
    assert len(dakshina) >= 10


def test_checksum_manifest_matches_dakshina(project_root: Path) -> None:
    import pytest

    manifest = project_root / "manifests" / "checksum_manifest.sha256"
    assert manifest.is_file()
    present = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        full = project_root / rel
        if full.is_file():
            present.append((digest, full))
    if not present:
        pytest.skip("raw Dakshina files not downloaded")
    for digest, full in present:
        assert sha256_file(full) == digest
