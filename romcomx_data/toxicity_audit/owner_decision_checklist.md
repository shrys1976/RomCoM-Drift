# Owner decision checklist

Verdict: **CONDITIONAL_GO** (augmentation authorized; training and scoring not authorized).

## Closed

- [x] Manifest SHA-256 `58d85ddaf78bb05600bc802049c8ce12ef8886f4439bfe87b9ca9fd25e1673ae` accepted **final** (2026-11-02).
- [x] 6,998 retained / 2 excluded; 80/10/10 split; zero cross-split groups.
- [x] Seed-42 benign sample matches final manifest.
- [x] `class_comparability.json` refreshed (historical superseded only).
- [x] Cross-class 122-member group: **ACCEPT** — keep manifest unchanged; no regrouping.
- [x] **Augmentation authorized** for EXP-01/02/03 under frozen manifest and split rules.

## Active constraints for augmentation

- [ ] Apply existing EXP-01/02/03 consistency checks (shared `example_id`, label, `project_split`).
- [ ] User prompt (hin_Deva slot 0) unchanged; romanize/noise assistant target only.
- [ ] Do not use validation as the final test split; test is held out for final evaluation.
- [ ] Generation rubric applies when scoring is run later; classification accuracy is not an outcome.

## Not authorized

- Training.
- Scoring / model evaluation runs.
- Manifest regrouping without a new owner-approved hash.

Record: `owner_acceptance_record.json`.
