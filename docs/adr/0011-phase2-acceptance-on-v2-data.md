# 0011. Phase 2 tie and collinearity criteria are judged on data/v2; boilerplate recall gap goes to Phase 6

- Status: Accepted
- Date: 2026-09-25
- Source: orchestrator rulings R5-R7 on Phase 2, branch `phase2-features`; `reports/phase2.md`,
  "Orchestrator decisions" and "Acceptance"

## Context

With the indicator encoding of ADR 0009, every session-less lead in v1 is a Meta lead-ads lead (932 of
932), so `session_missing=yes` and `channel=meta_leadads` are the same column. On v1 the strict
collinearity check fails on that one pair (|corr| = 1.000) and the L2 fit gives the two columns the same
weight (−0.345), which also fails "no two coefficients identical to three decimals". This is a property of
how v1 was generated, not of the model. data/v2 (Phase 5, regenerated at da72dad) adds 1,278
consent-declined website leads without telemetry, which separates the two columns.

Two further Phase 2 findings on v2: the similarity-based boilerplate detector (plan 2.5, token-set
Jaccard ≥ 0.6) finds 50 of 812 paraphrased copy-paste texts (recall 0.062, precision 1.000; v1: 1.000 /
1.000); and the deal-value residual sd estimated from training residuals is 0.5505 on v2 against the
0.45 the baseline hard-coded from the v1 generator's documentation (0.4546 estimated on v1).

## Decision

- **R5.** The "no identical coefficients" and collinearity criteria of Phase 2 are evaluated on
  `data/v2`. The AUC criterion is evaluated on v1 (the baseline CI is a v1 number). Both results are
  kept in the report. On v2 the default model (v2 features, horizon labels) has no equal pair and the
  design's max |corr| is 0.817: Phase 2 passes.
- **R6.** The boilerplate detector's recall on paraphrased v2 text is a known gap handed to Phase 6:
  semantic detection belongs to the context agent. Phase 2 does not tune the snippet list or threshold
  to v2.
- **R7.** The v2 residual sd (0.5505 vs 0.45) is recorded as evidence for plan 2.4: a constant copied
  from one generator's documentation understates another dataset's deal values (mean predicted / mean
  recorded deal value on won v2 test leads 0.910 with 0.45 vs 0.957 estimated, horizon labels).

## Consequences

- On v1, `python -m emva.eval.collinearity` exits 1 for the v2 feature set and the report's collinearity
  section says FAIL; this is expected and is not a regression. `make report` does not exit non-zero for
  it (ADR 0009, `--strict`).
- Tests assert both sides: v1 + v2 features fails the strict check on exactly the one pair; data/v2
  passes, and the default v2-data model has no equal coefficients.
- Coincidental 3-dp ties between unrelated columns (v2 features + legacy labels on v2:
  `sessions_3plus=yes` = `hiring=hiring`, corr 0.000) are reported, not engineered away.
- Phase 6 inherits the copy-paste detection problem on paraphrased text.
