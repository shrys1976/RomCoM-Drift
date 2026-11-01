from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from romcom_data import __version__
from romcom_data.handoff_paths import REPORTS


def _write(name: str, body: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    header = (
        f"# {name.replace('_', ' ').replace('.md', '')}\n\n"
        f"- Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"- Code version: romcom_data {__version__}\n\n"
    )
    path.write_text(header + body, encoding="utf-8")
    return path


def write_phase2_reports(ctx: dict[str, Any]) -> None:
    rec = ctx.get("reconstruction", {})
    _write(
        "indi_romcom_reconstruction_report.md",
        f"""## Status

Records are **reconstructed**, not the official Indi-RomCoM 395,076-instance training release.
No author training artifact was found.

## Paper method reproduced

1. Source Hindi task data from IndicXNLI, IndicXParaphrase, IndicAlign Toxic_Matrix.
2. Table 9 taxonomy: CAT-0 never switched; CAT-C at 25/50/75; CAT-B and CAT-D at 50/75; CAT-A and CAT-E at 75.
3. Table 10 intensities 25/50/75 with those category subsets.
4. Romanization via Dakshina train lexicon canonical forms; unknown native tokens use a documented character-map fallback (`fallback_indic_map`), not Aksharantar/Claude.
5. Toxicity (Appendix B.5): **input prompt is never code-mixed**. Only target/refusal side is mixed.

## Deviations (explicit)

- Official generation used Claude Sonnet 4.6 for taxonomy-guided substitutions. This reconstruction uses a closed bilingual lexicon plus deterministic category rules. No LLM rewrite.
- CAT-E clause rewriting is approximated by an Indic-frame romanization of the parallel Hindi sentence at 75%.
- CAT-D tokens without a lexicon entry remain English (`no_lexicon`).
- Dakshina attestation-weighted EXP-02 is **not** claimed as social-media typing frequency.

## English join

NLI English premises/hypotheses are attached from `facebook/xnli` by split + row index when missing from IndicXNLI.
""",
    )

    dak = ctx.get("inventory_stats", {})
    _write(
        "dakshina_inventory_quality_report.md",
        f"""## Recomputed inventory

{json.dumps(dak, indent=2, default=str)}

Attestation counts are annotator counts, not WhatsApp frequencies.

Samples: `romcomx_data/inventories/` and `data/interim/qa_samples/dakshina_inventory_samples.tsv`
""",
    )

    _write(
        "comi_lingua_supplementary_audit.md",
        f"""## Role

Supplementary Hinglish reference only. **Not** used as NLI/paraphrase/toxicity labels.

## Findings

{json.dumps(ctx.get("comi_audit", {}), indent=2, default=str)[:8000]}

MT annotator differences are not treated as word-level romanization variants.
""",
    )

    _write(
        "source_audit_report.md",
        f"""## Acquired sources

{json.dumps(ctx.get("source_meta", {}), indent=2, default=str)}

## Row counts

{json.dumps(ctx.get("row_counts", {}), indent=2, default=str)}

Licenses recorded as owner-approved for intended internal project use. Combined artifacts are not assigned a blanket license.
""",
    )

    _write(
        "split_integrity_report.md",
        f"""## Policy

- IndicXNLI: official train → project train; official dev → project validation; **test excluded**.
- IndicXParaphrase: no official split files; 80/20 connected-component split, seed 42.
- Toxicity: 80/20 by source_id, seed 42, stratified by toxic/non_toxic.
- Related NLI hypotheses sharing a premise stay together.
- All mixing intensities of one source stay together.

## Integrity

{json.dumps(ctx.get("split_stats", {}), indent=2, default=str)}
""",
    )

    _write(
        "romanization_coverage_report.md",
        f"""## Coverage

{json.dumps(ctx.get("coverage", {}), indent=2, default=str)}

Type coverage and occurrence coverage are reported separately. Matching rules were not relaxed.
""",
    )

    _write(
        "toxicity_data_design_report.md",
        """## Design

The Indi-RomCoM paper (Appendix B.5) keeps the **toxic input prompt unmixed**.
Only classification/refusal **targets** are code-mixed.

## Implications for RomCoM-X

- EXP-01/02/03 for toxicity apply romanization variation to **target-side** reconstructed text.
- Input-side eligible Romanized Hindi coverage is expected to be near zero unless the source prompt already contains it.
- Target-side romanization **must not** be interpreted as evidence of improved Romanized-Hindi **input** understanding.
- Original prompt bytes are preserved. Labels `toxic` / `non_toxic` are preserved.
- Benign rows come from IndicAlign `Indic_ShareLlama` sample for balance, separately identifiable via `is_benign`.
""",
    )

    _write(
        "augmentation_quality_report.md",
        f"""## Conditions

Same source IDs, splits, labels, and row counts across EXP-01/02/03.

- EXP-01: Dakshina unique max-count canonical (ties broken lexicographically for inventory; tokens with remaining multi-way surface ties left unchanged in text if flagged).
- EXP-02: p_variant=0.35 attestation-weighted non-canonical Dakshina variants from EXP-01.
- EXP-03: random Latin edits at the same EXP-02 token positions, matched edit-op count.

## QA

{json.dumps(ctx.get("qa", {}), indent=2, default=str)}
""",
    )

    _write(
        "final_data_handoff_report.md",
        f"""## Status

**{ctx.get("status", "READY_FOR_OWNER_REVIEW")}**

## Primary task sources (reconstructed Indi-RomCoM methodology)

- NLI: Divyanshu/indicxnli (owner-approved CC BY 4.0)
- Paraphrase: ai4bharat/IndicXParaphrase (owner-approved CC0)
- Toxicity: ai4bharat/indic-align Toxic_Matrix (owner-approved CC BY 4.0)

These are **not** the official Indi-RomCoM training release.

## Counts

{json.dumps(ctx.get("row_counts", {}), indent=2, default=str)}

## Paths

- Processed: `romcomx_data/processed/EXP-01_canonical|EXP-02_realistic|EXP-03_matched_noise/`
- Manifests: `romcomx_data/manifests/`
- Reports: `romcomx_data/reports/`
- Traces: `romcomx_data/traces/`

## Owner notes

- Toxicity input is not code-mixed; do not treat target-side EXP variation as input-robustness evidence.
- EXP-02 is Dakshina-informed, not a social-media frequency model.

## Model scope

**No model was loaded, validated, evaluated, or trained.**
""",
    )
