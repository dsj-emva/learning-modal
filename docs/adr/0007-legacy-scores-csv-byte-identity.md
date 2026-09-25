# 0007. label_source omitted from legacy scores.csv to keep byte identity (R4)

- Status: Accepted
- Date: 2026-09-25
- Source: `reports/phase1.md`, "Orchestrator decisions (review of Phase 1)" R4

## Context

Plan 1.1 adds `label_source` to every row, and plan 3 expects `scores.csv` to carry `label_source` and
`matured_at`. But `--label-mode legacy` must reproduce `baseline/` byte for byte (ADR 0001), and adding
columns to legacy `scores.csv` would break that. Computing `label_source` in legacy mode also made an
unrecognised final stage raise, where the baseline leaves it unlabelled.

## Decision

Legacy mode writes exactly the baseline's `scores.csv` columns and does not compute `label_source`.
Horizon mode appends `won_within_h`, `label_source` and `matured_at`
(`LabelConfig.extra_score_columns`, `HORIZON_SCORE_COLUMNS`).

## Consequences

- `make baseline` and the integration tests assert byte-equal stdout, `scores.csv` and `weights.csv`
  against `baseline/emva_score.py` in legacy mode.
- Consumers that need `label_source` must use horizon mode (the default).
- In horizon mode `y` is the censored training label and `won_within_h` the raw outcome.
