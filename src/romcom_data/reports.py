from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from romcom_data import __version__
from romcom_data.paths import REPORTS


def _header(title: str, source_revisions: list[str] | None = None) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    revs = ", ".join(source_revisions or ["see manifests/source_manifest.json"])
    return (
        f"# {title}\n\n"
        f"- **Timestamp (UTC):** {ts}\n"
        f"- **Code version:** romcom_data {__version__}\n"
        f"- **Source revision(s):** {revs}\n\n"
    )


def write_report(name: str, body: str, source_revisions: list[str] | None = None) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    path.write_text(_header(name.replace(".md", "").replace("_", " "), source_revisions) + body, encoding="utf-8")
    return path


def write_all_reports(ctx: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    gate = ctx["gate"]
    search = ctx["search_result"]
    fallback = ctx["fallback_licenses"]
    dakshina = ctx["dakshina_audit"]
    inventory = ctx.get("inventory_stats", {})
    comi = ctx.get("comi_audit", {})
    test_results = ctx.get("test_results", {})

    paths.append(
        write_report(
            "00_source_access_and_license_report.md",
            f"""## Indi-RomCoM training access

Official training split found in search: **{search.get('training_split_found')}**

Automated search notes:
{chr(10).join('- ' + n for n in search.get('notes', []))}

## Candidate URLs searched

{chr(10).join(f"- {c['name']}: {c['url']} (HTTP {c.get('status_code')})" for c in search.get('candidates', []))}

## Fallback dataset license summary

{chr(10).join(f"- **{f['repository']}**: status `{f.get('license_status')}` — {f.get('notes', '')}" for f in fallback)}

## Go / no-go

| Source | Decision |
|--------|----------|
| Indi-RomCoM training | **{'GO' if gate.get('cleared') else 'NO-GO / BLOCKED'}** |
| Dakshina hi lexicon train | GO (CC BY-SA 4.0) |
| COMI-LINGUA | GO supplementary audit only (CC BY 4.0) |

## Blocker

{gate.get('reason', 'N/A')}

Owner decisions:
{chr(10).join('- ' + d for d in gate.get('owner_decisions', []))}
""",
        )
    )

    train_lex = dakshina["lexicons"][0]
    paths.append(
        write_report(
            "01_source_schema_audit.md",
            f"""## Dakshina Hindi lexicon train

- Rows: {train_lex['rows']}
- Unique native: {train_lex['unique_native']}
- Missing romanized: {train_lex['missing_romanized_rows']}

## COMI-LINGUA (if stored)

{comi}

## File roles

- `data/raw/dakshina/hi/*` — immutable Dakshina Hindi extracts
- `data/raw/comi_lingua/*` — supplementary COMI snapshots
- `data/raw/indi_romcom/` — reserved for official Indi-RomCoM training files
""",
        )
    )

    paths.append(
        write_report(
            "02_dakshina_lexicon_audit.md",
            f"""## Recomputed train lexicon statistics

{train_lex}

## Native word overlap (train/dev/test)

{dakshina['native_word_overlap']}

## Sentence-level dev/test

{dakshina['sentence_level']}

## Caveat

{dakshina['attestation_caveat']}
""",
        )
    )

    paths.append(
        write_report(
            "03_comi_lingua_audit.md",
            f"""## Role

COMI-LINGUA is **supplementary only**. It does not provide NLI, paraphrase, or toxicity task labels for this project.

## Audit summary

```json
{comi}
```

MT train/test overlap (Sentences): {comi.get('mt_train_test_source_overlap', 'N/A')}
""",
        )
    )

    paths.append(
        write_report(
            "04_task_data_and_split_audit.md",
            f"""## Task source outcome

**Gate status:** {gate.get('status')}

{gate.get('reason')}

## Hindi task rows

No production Hindi NLI/paraphrase/toxicity training package was built (blocked at source gate).

## Toxicity augmentation caveat

Not evaluated on task rows until official source is available. IndicAlign Toxic_Matrix schema includes `hin_Latn` columns for inspection only.

## Split manifest

Not created (no cleared task source).
""",
        )
    )

    paths.append(
        write_report(
            "05_romanization_inventory_report.md",
            f"""## Production inventory (Dakshina train lexicon only)

{inventory}

## Coverage on task inputs

Not computed — no cleared task input records.

Paths:
- `data/interim/romanization/dakshina_hi_variant_inventory.tsv`
- `data/interim/romanization/dakshina_hi_alias_index.tsv`
""",
        )
    )

    paths.append(
        write_report(
            "06_augmentation_design_and_config.md",
            f"""## Configuration

Path: `configs/data_augmentation.yaml`

Config hash: `{ctx.get('config_hash', 'N/A')}`

## EXP definitions

- **EXP-01:** canonical max-attestation Dakshina variant (ties unchanged)
- **EXP-02:** attestation-weighted alternative from Uniform(0.10, 0.35) aggressiveness
- **EXP-03:** matched random Latin character noise at EXP-02 positions

## Seed derivation

Documented in config: stable SHA-256 based per `(example_id, condition)`.

## Status

Augmentation **not run** — blocked pending official task-labelled training source.
""",
        )
    )

    paths.append(
        write_report(
            "07_augmentation_qa_report.md",
            f"""## Processed datasets

None generated (source gate BLOCKED).

## Tests executed

{test_results}

## Human review sample

Not created (no processed conditions).
""",
        )
    )

    status = "BLOCKED" if not gate.get("cleared") else ctx.get("final_status", "NOT_READY")
    paths.append(
        write_report(
            "08_final_data_handoff_report.md",
            f"""## Status

**{status}**

## Primary task dataset

None used for training package construction.

## Dakshina inventory

{inventory}

## Processed EXP paths

Not created.

## Manifests

- `manifests/source_manifest.json`
- `manifests/checksum_manifest.sha256`
- `manifests/augmentation_manifest.json` (config-only while blocked)

## Acceptance tests

{test_results}

## Model scope confirmation

**No model was loaded, validated, evaluated, or trained.**

## Owner decisions required

{chr(10).join('- ' + d for d in gate.get('owner_decisions', []))}
""",
        )
    )
    return paths
