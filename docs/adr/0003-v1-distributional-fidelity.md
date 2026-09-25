# 0003. Exact reproduction of v1 is infeasible; distributional fidelity is the 5.1 pass criterion

- Status: Accepted
- Date: 2026-09-25
- Source: `reports/phase5-1.md`, "Exact reproduction: not achieved (expected)" and "Orchestrator decision"

## Context

Plan 5.1 asked to recover `scripts/generate_data.py` (seed 20260924), referenced in
`data/v1/ground_truth.md` but absent from the repo, or rewrite it and "confirm it reproduces v1 before
changing anything". It could not be recovered. The rewrite (`scripts/generate_data_v1.py`) matches 0 of
v1's rows: the original's RNG library, call order and vocabulary list orders are unknown, and each lead
involves about 50 conditional draws. One weak hint (Python `random.Random(20260924)` on sorted lists gives
v1's first company) diverged by the third draw.

## Decision

Exact row-level reproduction is accepted as infeasible. Distributional fidelity is the pass criterion:
every `ground_truth.md` table, scalar statistics and the baseline model compared between v1 and the
generated data, with seed-to-seed spread as the yardstick (`scripts/compare_to_v1.py`).

## Consequences

- Measured at seed 20260924, n = 10,000: mean win-rate gap 1.14 pp over 90 cells, mean share gap 0.66 pp;
  across 8 other seeds 5 of 90 cells sit beyond 2 sd, none beyond 3 (sampling noise). Baseline AUC 0.799
  on the regenerated seed; 0.805 ± 0.013 across 8 seeds, against v1's 0.814.
- Seed choice alone moves AUC by about ±0.015 and top-20% shares by about ±0.03. Later phases compare
  baseline and candidate on the same data, or across seeds, never against v1's single numbers.
- A regression of logit(`p_close_true`) on the planted features recovers every planted coefficient in v1
  (residual sd 0.505), so the documented model is what produced v1.
- No regenerated data is committed for v1; it is reproducible from the command. Generator v2 builds on
  this file behind `Config` switches that leave v1 behaviour byte-identical when off.
