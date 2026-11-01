"""Download pinned raw datasets into data/raw/ (local only, not Git).

Reuses checksums in manifests/checksum_manifest.sha256 when present.
Does not overwrite files whose SHA-256 already matches the manifest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from load_env import bootstrap_project  # noqa: E402

bootstrap_project()

import requests  # noqa: E402
from datasets import load_dataset  # noqa: E402

from romcom_data.checksums import sha256_file  # noqa: E402
from romcom_data.paths import RAW_COMI, RAW_DAKSHINA_HI  # noqa: E402
from romcom_data.source_loaders.indicxnli import acquire_indicxnli  # noqa: E402
from romcom_data.source_loaders.indicxparaphrase import acquire_indicxparaphrase  # noqa: E402
from romcom_data.source_loaders.toxic_matrix import acquire_toxic_matrix  # noqa: E402

PINS = json.loads((PROJECT_DIR / "data" / "metadata" / "source_pins.json").read_text())
CHECKSUMS = PROJECT_DIR / "manifests" / "checksum_manifest.sha256"
DAKSHINA_URL = PINS["dakshina"]["archive_url"]
TAR_BLOCK = 512


def load_checksums() -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not CHECKSUMS.is_file():
        return mapping
    for line in CHECKSUMS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        mapping[rel] = digest
    return mapping


def already_valid(path: Path, checksums: dict[str, str]) -> bool:
    rel = str(path.relative_to(PROJECT_DIR))
    if not path.is_file():
        return False
    expected = checksums.get(rel)
    if expected is None:
        return True
    return sha256_file(path) == expected


def scan_dakshina_members(wanted: set[str]) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    offset = 0
    empty = 0
    while empty < 2:
        r = requests.get(
            DAKSHINA_URL,
            headers={"Range": f"bytes={offset}-{offset + TAR_BLOCK - 1}"},
            timeout=60,
        )
        r.raise_for_status()
        header = r.content
        if len(header) < TAR_BLOCK or header.strip(b"\0") == b"":
            empty += 1
            offset += TAR_BLOCK
            continue
        empty = 0
        name = header[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
        size = int(header[124:136].split(b"\0", 1)[0].decode("ascii").strip() or "0", 8)
        padded = (size + TAR_BLOCK - 1) // TAR_BLOCK * TAR_BLOCK
        basename = name.split("/")[-1]
        data_start = offset + TAR_BLOCK
        if basename in wanted and basename not in found:
            found[basename] = (size, data_start)
        offset = data_start + padded
        if set(found) >= wanted:
            return found
    return found


def download_dakshina_range(filename: str, size: int, data_offset: int, dest: Path) -> None:
    start = data_offset
    end = data_offset + size - 1
    r = requests.get(
        DAKSHINA_URL,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=120,
    )
    r.raise_for_status()
    if len(r.content) != size:
        raise RuntimeError(f"Size mismatch for {filename}: {len(r.content)} != {size}")
    dest.write_bytes(r.content)


def acquire_dakshina(checksums: dict[str, str]) -> None:
    RAW_DAKSHINA_HI.mkdir(parents=True, exist_ok=True)
    wanted = set(PINS["dakshina"]["members"])
    missing = [
        name
        for name in wanted
        if not already_valid(RAW_DAKSHINA_HI / name, checksums)
    ]
    if not missing:
        print("Dakshina Hindi files already present and checksummed.")
        return
    print("Scanning Dakshina tar headers for Hindi members (not downloading the full archive)...")
    lookup = scan_dakshina_members(wanted)
    for name in missing:
        if name not in lookup:
            raise FileNotFoundError(f"{name} not found in Dakshina tar")
        size, data_offset = lookup[name]
        print(f"Downloading {name} ({size} bytes)")
        download_dakshina_range(name, size, data_offset, RAW_DAKSHINA_HI / name)


def acquire_comi(checksums: dict[str, str]) -> None:
    RAW_COMI.mkdir(parents=True, exist_ok=True)
    pin = PINS["comi_lingua"]
    for config_name in pin["configs"]:
        for split in ("train", "test"):
            out = RAW_COMI / f"{config_name}_{split}.jsonl"
            rel = str(out.relative_to(PROJECT_DIR))
            if already_valid(out, checksums) and out.is_file():
                print(f"reuse {rel}")
                continue
            print(f"download {pin['repository']} {config_name}/{split}")
            ds = load_dataset(
                pin["repository"],
                name=config_name,
                split=split,
                revision=pin["revision"],
            )
            ds.to_pandas().to_json(out, orient="records", lines=True, force_ascii=False)


def main() -> int:
    checksums = load_checksums()
    acquire_dakshina(checksums)
    acquire_comi(checksums)
    print("IndicXNLI", acquire_indicxnli(PINS["indicxnli"]["revision"])["revision"])
    print("IndicXParaphrase", acquire_indicxparaphrase(PINS["indicxparaphrase"]["revision"])["revision"])
    print("Toxic_Matrix", acquire_toxic_matrix(PINS["indic_align"]["revision"])["revision"])
    print("Done. Raw files stay under data/raw/ and romcomx_data/raw/ (gitignored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
