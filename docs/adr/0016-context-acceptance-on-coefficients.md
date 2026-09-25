# 0016. Context acceptance on coefficients; persona pooled at the feature level

- Status: Accepted
- Date: 2026-09-25
- Source: orchestrator rulings R12 to R15 on Phase 6 (`phase6-context-agent`, `reports/phase6.md`,
  "Orchestrator decisions")

## Context

Plan Phase 6 asked that the combined model recover more than half of the planted context-only AUC gap,
with the `persona` weights individually significant. On data/v2 the planted persona (-1.0 on about 8% of
leads) is worth very little AUC: 0.0037 even with the generator's true close probability over all labelled
leads, and the cross-fitted oracle gap on the 1,000-lead sample is +0.0037 [-0.005, +0.011]. A fraction of a
gap that is not distinguishable from zero has no usable CI (1.87 [-10.6, 12.2] after R13). Haiku's persona
judgments were accurate (87% of planted persona leads judged a non-buyer persona), but spread over six
non-buyer levels of 29 to 76 rows each, so no per-level weight was significant.

## Decision

- **R12.** The AUC-gap criterion is recorded as *unmeasurable at the planted effect size*; it is not
  restated as passed. The acceptance for context features from now on is coefficient-level: the context
  feature's weight must have a 95% bootstrap CI excluding zero **and** a point estimate inside the oracle
  weight's 95% CI, both estimated on the same rows (`emva.eval.context_harness.coefficient_criterion`). In
  Phase 6 the judged column is `ctx_persona_group=non_buyer` in formula + `ctx_persona_group`, against the
  oracle persona indicator in formula + oracle: -0.606 [-1.067, -0.179] vs -0.612 [-1.021, -0.209]: pass.
- **R13.** Persona is pooled at the feature level (`emva/context/features.py`, `PERSONA_GROUPS`):
  `ctx_persona_group` in {buyer, non_buyer (vendor, student, job_seeker, competitor, nonprofit,
  agency_pitching), unclear}, reference buyer. The contract enum stays fine-grained: no prompt or contract
  version bump and no new API calls. **The pooling rule was chosen after seeing the per-level results of the
  first run.** It was applied to the unchanged cached judgments and is now fixed in code, so the next
  customer run is pre-registered. Per-level weights stay in reports as secondary and underpowered.
- **R12 scope (ruling on the Phase 6 caveat).** R12 is judged one context feature at a time, next to the
  formula (formula + that feature's columns). The feature's weight in the full combined model is reported as
  secondary, with the shared-signal explanation: the other judgments carry part of the same signal, so the
  full-model weight can lose significance without the feature being uninformative.
- **R14.** `agency_pitching` stays in the contract; Haiku's vendor/agency confusion is documented; both
  pool to non_buyer.
- **R15.** The Phase 6 sample design (300 persona / 700 other labelled leads, seed 0) and cross-fitting as
  the primary comparison (time split secondary) are accepted.

## Consequences

- The context design has 13 fixed columns instead of 18. Any contract persona not in `PERSONA_GROUPS`
  fails at import; any judged value outside the contract raises.
- In the full 13-column combined model the pooled persona column is -0.383 [-0.887, 0.156], not significant:
  the other judgments share part of its signal. Under the R12 scope ruling this is secondary evidence, not
  a failure.
- A customer pilot applies R12 per context feature against whatever oracle or holdout evidence exists; the
  AUC gap is still reported, but not used as a pass criterion when the planted or expected effect is small.
