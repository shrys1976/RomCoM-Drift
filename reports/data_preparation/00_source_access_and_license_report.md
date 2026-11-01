# 00 source access and license report

- **Timestamp (UTC):** 2026-11-01T08:00:02.539712+00:00
- **Code version:** romcom_data 0.1.0
- **Source revision(s):** see manifests/source_manifest.json

## Indi-RomCoM training access

Official training split found in search: **False**

Automated search notes:
- No Hugging Face dataset ID matching Indi-RomCoM training release was found in automated search.
- Paper describes 395,076-instance fine-tuning corpus vs 8,400 evaluation benchmark; no author training artifact URL in paper body.

## Candidate URLs searched

- Indi-RomCoM arXiv paper: https://arxiv.org/abs/2606.30790 (HTTP 200)
- Indi-RomCoM arXiv HTML: https://arxiv.org/html/2606.30790 (HTTP 200)
- HuggingFace search Indi-RomCoM: https://huggingface.co/datasets?search=Indi-RomCoM (HTTP 200)
- HuggingFace search indi-romcom: https://huggingface.co/datasets?search=indi-romcom (HTTP 200)
- GitHub search Indi-RomCoM: https://github.com/search?q=Indi-RomCoM&type=repositories (HTTP 200)

## Fallback dataset license summary

- **Divyanshu/indicxnli**: status `unresolved` — HF metadata lists cc0-1.0; dataset card may state CC BY-NC 4.0 — reconcile before use.
- **ai4bharat/IndicXParaphrase**: status `unresolved` — No license declared in dataset card metadata.
- **ai4bharat/indic-align**: status `cleared_for_internal_use` — 

## Go / no-go

| Source | Decision |
|--------|----------|
| Indi-RomCoM training | **NO-GO / BLOCKED** |
| Dakshina hi lexicon train | GO (CC BY-SA 4.0) |
| COMI-LINGUA | GO supplementary audit only (CC BY 4.0) |

## Blocker

Official Indi-RomCoM task-labelled training split not located or downloaded; fallback upstream datasets are not cleared substitutes without owner approval and code-mix recreation.

Owner decisions:
- Provide official Indi-RomCoM training artifact URL or local files.
- Or approve fallback from IndicXNLI / IndicXParaphrase / IndicAlign with resolved licenses and faithful code-mix procedure.
- Resolve IndicXNLI license metadata vs card conflict.
- Establish IndicXParaphrase license with owner.
