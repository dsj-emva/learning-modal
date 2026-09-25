# 0001. Adjust in place behind a frozen baseline; do not rewrite

- Status: Accepted
- Date: 2026-09-25
- Source: `REBUILD_PLAN.md` (preamble, ground rules 1-5, Phase 0); implemented in `reports/phase0.md`

## Context

The 2026-09-25 review of the POC found right-censored labels, missing telemetry in reference buckets,
two perfectly collinear features, a constant copied from the ground truth, and a generator that the
model's regex and bot rule matched by construction. The question was whether to rebuild from scratch.

The pipeline shape (load → build → design → fit → report) was sound, the baseline reproduced exactly
(test AUC 0.814, top-20% wins 0.571, revenue 0.795, identical `weights.csv`), and every defect was
confined to one layer. A rewrite would lose the regression reference that each fix has to be measured
against.

## Decision

Adjust in place. Freeze the POC as `baseline/` (moved with `git mv`, never edited). Split it into
`emva/` modules with no behaviour change, proven by `make baseline` (same metrics, canonically equal
weights, byte-identical `scores.csv`/`weights.csv` between the emva and baseline runs). Every later
change is a numbered plan task behind a switch whose legacy value reproduces the baseline, and every task
reports baseline vs candidate on the same frozen test set. Only the context agent's output contract and
the second data generator are rewritten.

## Consequences

- Behaviour changes arrive as switches (`--label-mode`, `--feature-set`) with a `legacy` value;
  `--label-mode legacy --feature-set legacy` must stay byte-identical to `baseline/`.
- `make baseline` is a permanent regression gate; the standard report always includes the baseline run.
- Some baseline defects are preserved on purpose in legacy paths (e.g. the unstable tie order, the silent
  missing-reference-level behaviour in `design()`), documented rather than fixed.
