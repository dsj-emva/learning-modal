# 0009. Fixed design schema: missingness as explicit indicators, no data-dependent column pruning

- Status: Accepted, implementation in progress
- Date: 2026-09-25
- Source: orchestrator ruling on Phase 2, branch `phase2-features` (in progress), commit 4e1b763;
  `reports/phase2.md` on that branch, "Deviations, not done, open questions" item 1 (which describes the
  superseded pruning approach and is being updated)

## Context

Plan 2.1 and 2.2 give absent inputs a `missing` level instead of the reference level. On v1, every
behavioural `missing` level is present for exactly the Meta lead-ads leads (they have no on-site session),
so each is the same column as `channel=meta_leadads`; and `spend/crm/hiring=missing` are the same column
as `enrichment_missing=yes` by construction. Kept as separate columns they are perfectly collinear. The
first implementation dropped any design column identical on the training rows to an earlier one
(`drop_aliased_columns`), which made the design's column set depend on the data: a column would appear
or vanish depending on which leads were in the training window, and a future consent-declined website
lead would silently get weight 0 for those levels.

## Decision

The v2 design's column set is fixed by the feature spec, never by the data.

- `session_missing` = yes when the on-site session is absent (blank `landing_url` or any behavioural input
  blank). The seven behavioural features (`time_on_page`, `hesitation_90s`, `sessions_3plus`,
  `viewed_pricing`, `ip_country`, `ip_type`, `business_hours`) then take their reference level and the
  indicator carries the effect. No channel special case.
- `enrichment_missing` = yes when there is no companies.csv row by domain or by normalised name; `spend`,
  `crm`, `hiring` take their reference level. `band` keeps its own `missing` level (distinct because of the
  typed company-size fallback).
- No automatic column dropping.
- The collinearity check (plan 2.7, |corr| > 0.95 fails) stays strict.

## Consequences

- On v1 the check fails on `session_missing=yes` vs `channel=meta_leadads` (identical columns in that
  data). Tests and the report state this rather than hide it; on v2 data, consent-declined website leads
  separate the two columns.
- Meta lead-ads and consent-declined leads share one "no session" effect, estimable once both exist.
- Plan 2.6 re-run under the new encoding drops the same four features (`c_budget`, `c_timeline`,
  `ip_type`, `edits_1_4`).
- Figures in `reports/phase2.md` computed under the pruning approach (e.g. 38 design columns, max |corr|
  0.772, the alias list) must be re-measured before merge. Verify on merge.
