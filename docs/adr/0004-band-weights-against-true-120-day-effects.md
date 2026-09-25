# 0004. Band-weight criterion restated against true 120-day effects (R1)

- Status: Accepted
- Date: 2026-09-25
- Source: `reports/phase1.md`, "Orchestrator decisions (review of Phase 1)" R1 and "R1: true 120-day size effects vs learned weights"

## Context

Phase 1's acceptance asked the learned `band=201-1000` and `band=1000+` weights to move toward the
planted 1.3 and 1.0 (baseline 0.873 and 0.703). Under the horizon label 201-1000 moved to 0.971 (pass)
but 1000+ moved *away*, to 0.419. The planted values are effects on *eventual* close; the generator makes
1000+ deals close 1.8× slower, so 19.8% of eventual 1000+ training wins land after day 120 and become 0s.
The criterion as written conflicts with a fixed 120-day horizon. H and the flags were not tuned to hit it
(`--horizon-days 150` gives 0.616 but leaves no mature test leads).

## Decision

Restate the criterion: the learned horizon weights must be consistent with the **true 120-day** effects.
Per band, pass = the horizon weight's 95% CI (1,000 refits) contains the true effect, or the horizon point
estimate is closer to it than the legacy one. The true 120-day probability is derived from the planted
truth, `p120 = p_close_true × P(time to close ≤ 120 d)` (lognormal, median 40 days, log-sd 1.09, ×1.8 for
1000+, parsed from `ground_truth.md`), and the conditional effect is the same model fitted to `p120` as
soft labels (`emva/eval/ground_truth_reference.py`).

## Consequences

- Result: pass for all four bands on the conditional truth (true effects 0.378 / 1.099 / 1.043 / 0.532;
  horizon weights 0.235 / 1.146 / 0.971 / 0.419, each CI containing the truth). Caveat: each legacy CI
  contains it too; on point distance horizon is closer for 11-50, 201-1000 and 1000+, legacy for 51-200.
- Acceptance criteria phrased against planted eventual-close effects must be checked for consistency with
  the label definition before they are applied.
- Ground truth is read only in `emva/eval/ground_truth_reference.py`, run as a script.
