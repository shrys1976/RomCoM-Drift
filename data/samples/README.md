# Data samples

These are **small excerpts** for documentation and code review. They are not training data.

Full raw files stay on the local machine under `data/raw/` and `romcomx_data/raw/` and are gitignored.

Restore local datasets with:

```bash
PYTHONPATH=src uv run python scripts/download_raw_datasets.py
```

Pinned revisions and licenses: [`data/metadata/source_pins.json`](../metadata/source_pins.json).  
File checksums: [`manifests/checksum_manifest.sha256`](../../manifests/checksum_manifest.sha256).
