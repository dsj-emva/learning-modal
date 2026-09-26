# 0009. Fixed design schema: missingness as explicit indicators, no data-dependent column pruning

- Status: Accepted, implemented (merged with Phase 2; unknown levels of declared extras amended by ADR 0024)
- Date: 2026-09-25
- Source: orchestrator rulings on Phase 2, branch `phase2-features`: indicator encoding (commit 4e1b763),
  fixed level lists (09199d0), report `--strict` (4c1b0fd); `reports/phase2.md`, "Orchestrator decisions"

## Context

Plan 2.1 and 2.2 give absent inputs a `missing` level instead of the reference level. On v1, every
behavioural `missing` level is present for exactly the Meta lead-ads leads (they have no on-site session),
so each is the same column as `channel=meta_leadads`; and `spend/crm/hiring=missing` are the same column
as `enrichment_missing=yes` by construction. Kept as separate columns they are perfectly collinear. The
first implementation dropped any design column identical on the training rows to an earlier one
(`drop_aliased_columns`), which made the design's column set depend on the data: a column would appear
or vanish depending on which leads were in the training window, and a future consent-declined website
lead would silently get weight 0 for those levels. Separately, `pd.get_dummies` makes a column only for
levels present in the data, so a one-row frame (single-lead scoring, a hosted app) would get a different
schema from the training design.

## Decision

The v2 design's column set is fixed by the feature spec, never by the data.

- `session_missing` = yes when the on-site session is absent: blank `landing_url`, or any of the raw
  behavioural inputs (`emva.features.SESSION_INPUTS`) blank. The seven behavioural features
  (`time_on_page`, `hesitation_90s`, `sessions_3plus`, `viewed_pricing`, `ip_country`, `ip_type`,
  `business_hours`) then take their reference level and the indicator carries the effect. No channel
  special case.
- `enrichment_missing` = yes when there is no companies.csv row by domain or by normalised name; `spend`,
  `crm`, `hiring` take their reference level. `band` keeps its own `missing` level (distinct because of the
  typed company-size fallback).
- Every v2 feature has a declared, complete level list, reference first (`emva.constants.V2_LEVELS`).
  `emva.design.fixed_design` emits exactly one column per non-reference level (39 columns for the v2
  feature set), whatever occurs in the data: an absent level is an all-zero column; a value outside the
  list raises `ValueError` naming the feature and the values. The legacy feature set keeps the
  baseline's data-driven `design()` (byte identity).
- No automatic column dropping anywhere.
- The collinearity check (plan 2.7, fails on any pair with |corr| > 0.95) stays strict as a standalone
  command (`python -m emva.eval.collinearity`, exit 1). The standard report prints it as a prominent
  PASS/FAIL section and exits 1 on a failure only with `--strict`, so a data property cannot fail
  `make report` (see ADR 0011 for where the criterion is judged).

## Consequences

- On v1 the check fails on `session_missing=yes` vs `channel=meta_leadads` (identical columns in that
  data) and the L2 fit gives the two equal weights; tests and the report state this. On data/v2,
  consent-declined website leads separate the two columns and the check passes (ADR 0011).
- Meta lead-ads and consent-declined leads share one "no session" effect.
- A new level (for example a new form variant) must be added to `V2_LEVELS` before it can be scored;
  until then scoring refuses the lead loudly instead of mis-encoding it.
- Plan 2.6 re-run under the final encoding drops the same four features (`c_budget`, `c_timeline`,
  `ip_type`, `edits_1_4`).
- Not covered: the deal-value design (`emva.value.deal_value_design`) is still data-driven; Phase 3
  owns the value layer.
