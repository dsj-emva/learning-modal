# 0006. The 458-lead mature test set is frozen; Phase 4 rolling-origin becomes the headline (R3)

- Status: Accepted
- Date: 2026-09-25
- Source: `reports/phase1.md`, "Orchestrator decisions (review of Phase 1)" R3 and "Test set sizes"

## Context

With `TEST_FROM` = 2026-05-01, H = 120 and AS_OF = 2026-09-24, only leads created before
2026-05-27T00:00Z are mature: 458 labelled leads (80 won) covering 26 days. Its AUC CI is about ±0.05,
twice the legacy test set's width (baseline 0.778 [0.726, 0.827]). Options were to start the horizon test
window earlier (costing training rows) or to keep it and rely on another evaluation for the headline.

## Decision

The mature test set is frozen as defined (ground rule 4): `emva.eval.report.FROZEN_HORIZON` (H = 120,
ghosted excluded, stalled censored), independent of the candidate's flags. Phase 4's rolling-origin
evaluation (six split dates, 2026-02-01 to 2026-07-01) becomes the headline number. No code change.

## Consequences

- Every report shows both frozen test sets: (a) legacy 2,453 leads, (b) mature 458 leads.
- Comparisons on (b) use the paired bootstrap (about ±0.012 on v1), not overlapping CIs.
- A horizon of about 147 days or more leaves no mature test lead; the pipeline then trains and scores but
  skips test metrics with a message.
- Headline claims wait for Phase 4.
