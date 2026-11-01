# 06 augmentation design and config

- **Timestamp (UTC):** 2026-11-01T08:00:02.545766+00:00
- **Code version:** romcom_data 0.1.0
- **Source revision(s):** see manifests/source_manifest.json

## Configuration

Path: `configs/data_augmentation.yaml`

Config hash: `54421038f7bd64630bf46cc7e6fc8a0c012ef4929b9ec8b14b7d847ae39a4686`

## EXP definitions

- **EXP-01:** canonical max-attestation Dakshina variant (ties unchanged)
- **EXP-02:** attestation-weighted alternative from Uniform(0.10, 0.35) aggressiveness
- **EXP-03:** matched random Latin character noise at EXP-02 positions

## Seed derivation

Documented in config: stable SHA-256 based per `(example_id, condition)`.

## Status

Augmentation **not run** — blocked pending official task-labelled training source.
