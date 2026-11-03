# RomCoM Drift NLI V2 — dataset freeze record

| Field | Value |
|-------|--------|
| **Dataset name and version** | RomCoM Drift NLI V2 |
| **Snapshot path** | `romcomx_data/nli_experiments_v2/` |
| **Freeze date** | 2026-11-03 |
| **Owner-review verdict** | `APPROVED_WITH_DOCUMENTED_CAVEATS` |
| **Owner review report** | `romcomx_data/nli_experiments_v2/reports/owner_review_v2_final.md` |

## Final checksum verification

Verified read-only against `romcomx_data/nli_experiments_v2/checksums.sha256` on **2026-11-03**:

| Result | Detail |
|--------|--------|
| **Outcome** | **PASS** |
| Manifest entries checked | 40 |
| Missing files | 0 |
| SHA256 mismatches | 0 |

The manifest file itself was not re-hashed into the manifest (unchanged policy).

## Freeze policy

**`romcomx_data/nli_experiments_v2/` is frozen as the approved V2 dataset snapshot.** Its files must not be regenerated, rebuilt, transformed, cleaned, normalized, or otherwise modified.

**`romcomx_data/nli_experiments_v1/`** remains the preserved V1 baseline and must not be overwritten or modified.

Any future dataset changes must produce a **new version** (e.g. V3) under a new path or version label. **Do not overwrite V2.**

This freeze record lives **outside** the V2 tree so the snapshot directory contents stay unchanged.
