from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def write_checksum_manifest(root: Path, relative_paths: list[Path], out_path: Path) -> None:
    lines: list[str] = []
    for rel in sorted(relative_paths, key=lambda p: str(p)):
        full = root / rel
        if not full.is_file():
            continue
        digest = sha256_file(full)
        lines.append(f"{digest}  {rel.as_posix()}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
