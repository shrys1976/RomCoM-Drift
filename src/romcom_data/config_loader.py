from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from romcom_data.paths import CONFIG, PROJECT_ROOT


def load_augmentation_config(path: Path | None = None) -> tuple[dict[str, Any], str]:
    cfg_path = path or CONFIG
    raw = cfg_path.read_text(encoding="utf-8")
    config = yaml.safe_load(raw)
    config_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    wordlist_rel = config.get("english_collision_wordlist", "")
    wordlist_path = PROJECT_ROOT / wordlist_rel
    if wordlist_path.is_file():
        config["_english_wordlist_sha256"] = hashlib.sha256(
            wordlist_path.read_bytes()
        ).hexdigest()
    return config, config_hash
