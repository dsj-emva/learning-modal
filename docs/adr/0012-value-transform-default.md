# 0012. Default value transform: cap at p97, log compression rescaled to the training total, floor £25

- Status: Proposed
- Date: 2026-09-25
- Source: `reports/phase3.md` (plan 3.1 / 3.2), branch `phase3-value`; `emva/value_transform.py`,
  `emva.constants.VALUE_CAP_PERCENTILE` / `VALUE_COMPRESSION` / `VALUE_FLOOR_GBP`

## Context

The value sent to the ad platforms, `p × E[deal value] × margin`, is heavy-tailed: over all scored v1 leads the
baseline's max/median is 121.7× and the top 1% of leads hold 11.5% of the total value (candidate: 100.0× /
10.6%). Plan 3.2 asks for a transform that keeps ≥ 95% of the baseline's top-20% revenue capture while cutting
max/median below 20×.

Measured on v1 (legacy and mature test sets, all scored leads) and v2 (the same three scopes), every
transform fitted on the candidate's training leads (`reports/phase3.md`, "Acceptance"):

- The plan's own default, cap p97 + floor £25, leaves max/median at 23.7× to 31.9×: **fails** on every scope.
- cap p97 + sqrt and cap p97 + log: max/median 4.7× to 5.7×, capture unchanged (the transforms are strictly
  monotone above the floor, so the top 20% is the same set): **pass** everywhere.
- 5 quantile tiers: 13.4× to 16.5×, capture 98.7% to 109.4% of the baseline's: **pass**, but only five distinct
  values (every lead in the top 20% gets the same value). 10 tiers: 17.7× to 32.2×, **fails** (the top tier's
  mean).
- Compression anchored at the median alone put the total value at 0.36 to 0.45 × recorded revenue on v1
  test sets, which would misstate ROAS in the platform. `emva.value_transform` therefore rescales the
  compressed values so the training leads' (capped) total is unchanged; total value / recorded revenue is then
  0.99 to 1.25 on v1 and 1.13 to 1.43 on v2 under log (the identity gives 1.11 to 1.30 and 1.19 to 1.40).
- Near zero, sqrt inflates low values far more than log: after the rescale, the p1 of the v1 test values is
  £152-156 under sqrt and £23-25 under log (identity £6). Sending £150 for a lead the model values at £6
  would pay the platform to buy junk.

## Decision

The default value transform (`ValueTransform()`, used for `value_at_submit` in horizon mode) is:

1. cap at the 97th percentile of the training leads' expected values (£13,221 on v1, £15,259 on v2);
2. log compression `m × log2(1 + v / m)`, `m` = median of the capped training values, then rescaled so the
   training leads' capped total is unchanged;
3. floor at £25.

The cap, anchor and scale are learned once from the training leads and fixed for scoring. The CLI can
switch any step off or choose sqrt or tiers (`--no-value-cap`, `--value-compression {none,log,sqrt}`,
`--no-value-floor`, `--value-tiers N`).

## Consequences

- Passes plan 3.2 on v1 (legacy and mature test sets) and on v2: max/median 4.7× to 5.5×, top-20% capture
  equal to the untransformed candidate's (102.5% / 103.0% of the baseline on v1, 108.3% / 111.7% on v2).
- Top-20% capture is rank-based, so it cannot distinguish monotone transforms; the choice between log, sqrt
  and tiers rests on the value's scale (total vs revenue, low end), not on the acceptance metric. A lead's
  value is no longer proportional to its expected revenue: high values are compressed (a lead at the cap gets
  about 0.5× its expected value, the top v1 lead 0.15×), low values inflated (about 2.5 to 4× below the
  median). Platforms bidding on it
  will spread spend more evenly than a pure expected-revenue value would.
- The rescale ties the level of `value_at_submit` to the training leads; a retrain moves it.
- Legacy mode is unaffected (it writes no `value_at_submit`).
- If not accepted, the fallback is 5 tiers (passes, keeps the revenue scale, loses resolution in the top 20%)
  or the plan's literal default (fails the max/median criterion).
