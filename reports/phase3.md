# Phase 3: value layer and platform contract

Branch `phase3-value` from `origin/main` c349833. Plan items 3.1 to 3.5. All numbers are **on simulated
data** (v1 and v2); nothing here is for external quotation. `make baseline` passes (legacy mode byte-identical);
the value columns exist only in horizon mode.

## Acceptance

| criterion | result | evidence |
|---|---|---|
| A transform keeps ≥ 95% of the baseline's top-20% revenue capture and cuts max/median below 20× (v1 legacy test set, v1 mature test set, v2) | **pass**: cap p97 + log + floor £25 (proposed default, ADR 0012); also cap p97 + sqrt, cap p97 + log, 5 tiers | table below; full tables in the pasted reports |
| `scores.csv` (horizon) carries `value_at_submit`, `value_at_close`, `label_source`, `matured_at` and the two timestamps | **pass** | columns `value_at_submit`, `value_at_submit_ts`, `value_at_close`, `value_at_close_ts` after `won_within_h, label_source, matured_at`; `tests/test_integration.py::test_cli_default_is_horizon_and_writes_label_columns`, `tests/test_value_two_stage.py` |
| `make baseline` passes | **pass** | legacy `scores.csv` / `weights.csv` byte-identical to the baseline run |
| `grep -rn "0.45\|ground_truth" emva/` only under `emva/eval/` | **pass** | no other lines (test-enforced) |

### Transform comparison

Candidate = default pipeline (horizon labels, v2 features). Each transform is fitted on the candidate's
training leads (v1: 4,049; v2: 3,789). "capture" = tie-averaged share of recorded won deal value in the top
20% by value, "vs base" = over the baseline's own capture by p×value (v1 legacy 0.796, v1 mature 0.752, v2
legacy 0.719, v2 mature 0.663). max/median on the test set / on all scored leads. Criterion: vs base ≥ 95% and
both max/median < 20×.

| transform | v1 legacy: vs base, max/med | v1 mature: vs base, max/med | v1 all leads max/med | v2 legacy: vs base, max/med | v2 mature: vs base, max/med | v2 all leads max/med | result |
|---|---|---|---|---|---|---|---|
| baseline value (reference) | 100%, 131.8× | 100%, 80.5× | 121.7× | 100%, 73.9× | 100%, 58.8× | 72.8× | |
| identity | 102.5%, 107.8× | 103.0%, 62.3× | 100.0× | 108.3%, 65.1× | 111.7%, 55.4× | 67.6× | fail |
| cap p97 | 102.5%, 31.9× | 103.0%, 25.5× | 29.6× | 108.3%, 27.9× | 111.7%, 23.7× | 25.5× | fail |
| cap p97 + floor £25 (plan default) | 102.5%, 31.9× | 103.0%, 25.5× | 29.6× | 108.3%, 27.9× | 111.7%, 23.7× | 25.5× | fail |
| cap p97 + sqrt | 102.5%, 5.7× | 103.0%, 5.1× | 5.4× | 108.3%, 5.3× | 111.7%, 4.9× | 5.0× | PASS |
| cap p97 + log | 102.5%, 5.5× | 103.0%, 4.7× | 5.2× | 108.3%, 5.3× | 111.7%, 4.7× | 5.0× | PASS |
| tiers 5 | 99.5%, 16.5× | 98.7%, 16.5× | 16.5× | 106.6%, 13.4× | 109.4%, 13.4× | 13.4× | PASS |
| tiers 10 | 100.8%, 32.2× | 99.3%, 17.7× | 32.2× | 107.6%, 25.8× | 109.5%, 25.8× | 25.8× | fail |
| **cap p97 + log + floor £25 (proposed default)** | 102.5%, 5.5× | 103.0%, 4.7× | 5.2× | 108.3%, 5.3× | 111.7%, 4.7× | 5.0× | **PASS** |

Top-1% share of value (v1 all leads): baseline 11.5%, candidate identity 10.6%, proposed default 3.1%.
The baseline figures reproduce Phase 0 (121.7× / 11.5% over all scored v1 leads; the plan's 116× / 11% do not
reproduce, as Phase 0 noted).

**Recommended default: cap p97 + log + floor £25 (ADR 0012, status Proposed).** Why:

- Top-20% capture is rank-based, so every strictly monotone transform keeps it exactly; it cannot choose
  between them. The choice rests on the value's scale.
- The plan's cap + floor alone fails max/median (23.7× to 31.9×). 10 tiers fail. 5 tiers pass but send only
  five distinct values (no resolution inside the top 20%).
- Compression is rescaled so the training leads' capped total is unchanged (otherwise total value fell to
  0.36 to 0.45 × recorded revenue on the v1 test sets and ROAS in the platform would be misstated). With the
  rescale, value / recorded revenue is 1.25 / 0.99 (v1 legacy / mature) and 1.43 / 1.13 (v2) under log,
  against 1.30 / 1.11 and 1.40 / 1.19 for the identity.
- log beats sqrt at the low end: after the rescale the p1 of v1 test values is £23-25 under log vs £152-156
  under sqrt (identity £6), so junk leads are not sold to the platform as £150 leads.
- Cost: values are no longer proportional to expected revenue (a lead at the cap gets about 0.5× its
  expected value, low values about 2.5 to 4×). Fitted parameters: v1 cap £13,221, anchor £505, scale 2.771;
  v2 cap £15,259, anchor £680, scale 2.453.

## tROAS eligibility (plan 3.4)

`python -m emva.troas`. Positive rate = `won_within_h` over mature leads: v1 0.133, v2 0.131. Volumes = scored
leads of the last 12 months per `utm_campaign`; Meta pools website and lead-form leads of a campaign.
Thresholds (constants in `emva/troas.py`, marked verify): Google tROAS 30 conversions per campaign per 30
days; Meta ~50 optimisation events per ad set per week. "submit stage" counts every lead (value at
submit), "close stage" only wins.

| volume | platform | per campaign: leads / month | conversions per window: submit / close | submit | close |
|---|---|---|---|---|---|
| v1, 4 Google campaigns | Google (30 d) | 54.4-60.5 | 53.6-59.6 / 7.1-7.9 | ok | BELOW (3.8-4.2× short) |
| v1, Google pooled | Google (30 d) | 230.2 | 226.9 / 30.1 | ok | ok (at threshold) |
| v1, 4 Meta campaigns | Meta (7 d) | 75.6-77.2 | 17.4-17.7 / 2.3-2.4 | BELOW | BELOW (~21× short) |
| v1, Meta pooled | Meta (7 d) | 305.4 | 70.2 / 9.3 | ok | BELOW (5.4×) |
| v2, 4 Google campaigns | Google (30 d) | 54.1-57.6 | 53.3-56.8 / 7.0-7.4 | ok | BELOW (4.0-4.3×) |
| v2, Google pooled | Google (30 d) | 225.8 | 222.5 / 29.1 | ok | BELOW (1.03×) |
| v2, 4 Meta campaigns | Meta (7 d) | 71.9-76.9 | 16.5-17.7 / 2.2-2.3 | BELOW | BELOW (~22×) |
| v2, Meta pooled | Meta (7 d) | 299.8 | 68.9 / 9.0 | ok | BELOW (5.5×) |
| 2,000 leads / year, 1 campaign, rate 0.132 | Google | 166.7 | 164.3 / 21.7 | ok | BELOW (1.4×) |
| same | Meta | 166.7 | 38.3 / 5.1 | BELOW | BELOW (9.9×) |
| 2,000 leads / year, 3 campaigns | Google | 55.6 | 54.8 / 7.2 | ok | BELOW (4.2×) |
| same | Meta | 55.6 | 12.8 / 1.7 | BELOW | BELOW (29.6×) |

Reading: bidding on closed-won alone is not eligible anywhere except pooled Google volume at v1 scale; the
submit-stage value (every lead) is eligible on Google per campaign and on Meta only when pooled. A 2,000-lead
customer can use Google tROAS on submit-stage values in one campaign and cannot exit Meta's learning phase per
ad set even on leads. This is why the platform contract bids on `value_at_submit`.

## Platform contract (plan 3.5)

`docs/platform_contract.md` (design only). Click-ID coverage, recomputed (report section "Click-ID coverage"):
landing-page paid scored leads without any click id **28.8% on v1** (7,454 leads) and **29.2% on v2** (7,435);
over all paid leads including Meta lead forms (which carry a lead id) 25.6% / 26.2% have no id at all. 22.9% /
20.1% of landing-page paid leads have no click id but an `fbp` cookie; every lead has an email, about 25% a
phone. The plan's 29.7% does not reproduce exactly (same definition on the raw 10,000 v1 rows: 29.0%).
Also found: 15.2% of v1 wins (14.4% v2) close more than 90 days after `created_at` (median 35 days, p90 114),
which matters for how long a Google conversion adjustment is accepted (contract section 2).

## What was done

- **3.1** `emva/value_transform.py`: `ValueTransform` (cap percentile, compression none/log/sqrt, floor, tiers)
  → `.fit(training values)` → `FittedValueTransform.apply` (one lead or many, fixed parameters). Defaults in
  `emva/constants.py` (`VALUE_CAP_PERCENTILE`, `VALUE_COMPRESSION`, `VALUE_FLOOR_GBP`). CLI (horizon only;
  with `--label-mode legacy` a usage error): `--value-cap-percentile`, `--no-value-cap`,
  `--value-compression`, `--value-floor`, `--no-value-floor`, `--value-tiers` (`emva/cli.py`).
  **Fixed-schema deal-value design** (the open point of `reports/phase2.md`): `emva.value.fixed_deal_value_design`
  emits one column per level of `constants.DEAL_VALUE_LEVELS` (band, sector, channel, spend; 26 columns), an
  undeclared value raises; selected through a new `FeatureSpec.value_design` field (legacy keeps the
  data-driven `deal_value_design` for byte identity). The candidate's numbers are unchanged by it (v1 legacy
  test p×value capture 0.815, as in Phase 2).
- **3.2** `emva/eval/value_report.py`, called by one line in `emva/eval/report.py`: the transform tables per
  test set and over all scored leads, the PASS/fail table, click-ID coverage.
- **3.3** `emva.value.value_at_close` and the pipeline (horizon mode): `value_at_submit` = `value_formula`
  through the transform fitted on the training leads, `value_at_submit_ts` = `created_at`; `value_at_close` =
  recorded deal value if Won by AS_OF (blank = 0) at `won_at`, 0 at `matured_at` for a mature non-win, NaN /
  empty while immature. `value_formula` is unchanged (the report still ranks by it). v1: 6,545 of 9,311 leads
  (70.3%) have a known `value_at_close`, 1,087 of them positive.
- **3.4** `emva/troas.py`, `python -m emva.troas` (by numbers, `--campaigns` or `--split`, or `--data DIR`).
- **3.5** `docs/platform_contract.md`.
- Tests: `tests/test_value_transform.py`, `tests/test_value_two_stage.py`, `tests/test_troas.py`,
  `tests/test_value_report.py`, new cases in `tests/test_integration.py`. **389 passed.**
- Docs: status rows in `CLAUDE.md` and `docs/CONTEXT.md` ("ready for review"; `CLAUDE.md`'s combined
  "3, 4" row is split in two), glossary, `docs/ARCHITECTURE.md`, ADR 0012 (Proposed).

## Deviations, not done, open questions

1. **Won with a blank deal value** (112 of 1,199 v1 wins, 96 of 1,128 v2): `value_at_close` = 0, following
   ADR 0002's revenue convention. Uploaded as is this would retract real wins; the contract tells the upload job
   to hold such rows. Alternative for the orchestrator: NaN (close value unknown). Needs a ruling.
2. **The default is not the plan's.** The plan names cap p97 + floor £25 as defaults; that fails the plan's own
   max/median criterion, so the code default adds log compression (ADR 0012, Proposed). Rejecting the ADR is a
   one-constant change (`VALUE_COMPRESSION = "none"`).
3. **Rescaled compression** is my addition (not in the plan): without it log/sqrt pass the acceptance equally
   but put values at 0.36-0.45 × revenue. It changes no acceptance number (max/median and capture are
   scale-invariant).
4. `value_at_close` for immature `crm_lost` leads is NaN, as specified ("NaN while immature"); the Phase 1 open
   question on young `crm_lost` leads applies here too.
5. In `--context` mode `value_at_submit` is still based on `value_formula`, not `value_combined` (the context
   model scores only leads with a context score). Phase 6 should decide.
6. The plan's 29.7% click-ID figure is not reproduced exactly (28.8-29.0% on the closest definition).
7. `feature_spec.py` and `constants.py` were edited (a `value_design` field; `DEAL_VALUE_LEVELS` and the value
   defaults) in addition to the files listed for this phase; both edits are additive.
8. Platform limits in `emva/troas.py` and `docs/platform_contract.md` are from memory and marked verify; no
   platform call or doc fetch was made.

## Standard report, data/v1 (`make report`)

".venv/bin/python" -m emva.eval.report --data "data/v1"
### EMVA standard report

Data: `data/v1`. baseline = frozen `baseline/emva_score.py`; candidate = `emva` pipeline trained with `horizon H=120` labels and `v2` features; status quo = `status_quo_rules.json` reconstructed. Every model is scored on the same rows under two test definitions. AUC CI: percentile bootstrap, 1000 resamples, seed 0. Top-20% capture ranks by p (status quo: by its value) or by p×value.

#### Label definitions

Counts over all 9311 scored leads, as wins / losses / unlabelled. `label_source` is the CRM state at 2026-09-24. legacy y = baseline rules. won_within_h = Won within 120 days of created_at (NaN if younger and not Won). horizon y = won_within_h with ghosted-at-H leads excluded and stalled leads censored (the default training and test label).

| label_source | leads | legacy y | won_within_h | horizon y |
|---|---|---|---|---|
| won | 1199 | 1199 / 0 / 0 | 1099 / 100 / 0 | 1099 / 100 / 0 |
| crm_lost | 4896 | 0 / 4896 / 0 | 0 / 3505 / 1391 | 0 / 3505 / 1391 |
| stalled | 618 | 0 / 618 / 0 | 0 / 564 / 54 | 0 / 0 / 618 |
| ghosted | 1207 | 0 / 1207 / 0 | 0 / 1207 / 0 | 0 / 0 / 1207 |
| open | 1391 | 0 / 121 / 1270 | 0 / 70 / 1321 | 0 / 70 / 1321 |
| all | 9311 | 1199 / 6842 / 1270 | 1099 / 5446 / 2766 | 1099 / 3675 / 4537 |

Mature = created_at + 120 days <= 2026-09-24T00:00:00+00:00: the latest mature lead was created 2026-05-26T20:06:58+00:00. Train = created before 2026-05-01 (all mature); test = created on or after 2026-05-01.

| definition | train (wins) | test (wins) |
|---|---|---|
| legacy | 5588 (847) | 2453 (352) |
| horizon H=120 | 4049 (752) | 458 (80) |

Candidate trained with: `horizon H=120`.

#### (a) Legacy labels, legacy test set

2453 leads labelled by the baseline rules, created on or after 2026-05-01 (352 won). Label = legacy y.

##### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.814 [0.789, 0.836] | 0.1006 | 0.571 | 0.742 | 0.796 |
| candidate | 0.814 [0.790, 0.837] | 0.1020 | 0.571 | 0.733 | 0.815 |
| status quo | 0.639 [0.605, 0.673] | n/a | 0.392 | 0.454 | 0.454 |

##### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.000 | [-0.006, +0.006] | 0.972 |
| status quo | -0.174 | [-0.209, -0.141] | 0.000 |

##### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 246 | 0.009 | 0.016 | 0.009 | 0.012 |
| 2 | 245 | 0.020 | 0.012 | 0.022 | 0.012 |
| 3 | 245 | 0.033 | 0.016 | 0.040 | 0.029 |
| 4 | 245 | 0.050 | 0.053 | 0.061 | 0.049 |
| 5 | 246 | 0.075 | 0.077 | 0.092 | 0.065 |
| 6 | 245 | 0.109 | 0.135 | 0.132 | 0.135 |
| 7 | 245 | 0.150 | 0.131 | 0.184 | 0.147 |
| 8 | 245 | 0.213 | 0.176 | 0.254 | 0.167 |
| 9 | 245 | 0.316 | 0.327 | 0.363 | 0.347 |
| 10 | 246 | 0.511 | 0.492 | 0.559 | 0.472 |

Status quo has no probability, so it has no Brier score or calibration.

##### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 754 | 103 | 0.800 | 0.801 | 0.641 |
| 2026-06 | 665 | 93 | 0.797 | 0.806 | 0.631 |
| 2026-07 | 541 | 88 | 0.833 | 0.826 | 0.602 |
| 2026-08 | 371 | 55 | 0.813 | 0.812 | 0.664 |
| 2026-09 | 122 | 13 | 0.920 | 0.914 | 0.772 |

##### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 2453 | 346 | 17167 | 45562 | 131.8× | 11.8% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 2453 | 414 | 17889 | 44625 | 107.8× | 10.9% |
| candidate | all scored leads | 9311 | 446 | 18646 | 44625 | 100.0× | 10.6% |
| status quo | test set | 2453 | 96 | 1440 | 1872 | 19.5× | 6.7% |
| status quo | all scored leads | 9311 | 96 | 1440 | 1872 | 19.5× | 6.7% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

##### Notes

- status quo: 203 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.397.
- status quo: 203 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454.
- status quo: 203 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454.

#### (b) Horizon labels, mature test set

458 mature leads created on or after 2026-05-01 and up to 2026-05-26T20:06:58+00:00 that the default horizon definition labels (80 won). Label = won within 120 days; ghosted-at-H leads excluded, stalled leads censored.

##### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.778 [0.726, 0.827] | 0.1258 | 0.487 | 0.670 | 0.752 |
| candidate | 0.778 [0.730, 0.827] | 0.1265 | 0.450 | 0.608 | 0.774 |
| status quo | 0.644 [0.571, 0.717] | n/a | 0.400 | 0.501 | 0.501 |

##### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.000 | [-0.015, +0.015] | 0.974 |
| status quo | -0.134 | [-0.199, -0.069] | 0.000 |

##### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 46 | 0.008 | 0.022 | 0.009 | 0.000 |
| 2 | 46 | 0.020 | 0.000 | 0.023 | 0.043 |
| 3 | 46 | 0.038 | 0.022 | 0.045 | 0.000 |
| 4 | 45 | 0.060 | 0.111 | 0.072 | 0.089 |
| 5 | 46 | 0.090 | 0.174 | 0.106 | 0.109 |
| 6 | 46 | 0.124 | 0.152 | 0.151 | 0.261 |
| 7 | 45 | 0.171 | 0.178 | 0.208 | 0.222 |
| 8 | 46 | 0.237 | 0.239 | 0.280 | 0.239 |
| 9 | 46 | 0.357 | 0.391 | 0.391 | 0.348 |
| 10 | 46 | 0.555 | 0.457 | 0.592 | 0.435 |

Status quo has no probability, so it has no Brier score or calibration.

##### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 458 | 80 | 0.778 | 0.778 | 0.644 |

##### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 458 | 432 | 18577 | 34763 | 80.5× | 10.9% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 458 | 518 | 20121 | 32268 | 62.3× | 9.7% |
| candidate | all scored leads | 9311 | 446 | 18646 | 44625 | 100.0× | 10.6% |
| status quo | test set | 458 | 96 | 1440 | 1872 | 19.5× | 6.4% |
| status quo | all scored leads | 9311 | 96 | 1440 | 1872 | 19.5× | 6.7% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

##### Notes

- status quo: 37 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.398.
- status quo: 37 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.506.
- status quo: 37 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.506.

#### Bottom-decile ghosted share

Share of leads in the 10% of each population with the lowest p that are *ghosted* (`label_source`: mature and not contacted within 120 days) or *still New* at 2026-09-24 whatever their age (the definition behind the plan's baseline figure of 27%). The horizon test set (b) excludes ghosted leads, so the comparison uses populations that keep them.

| population | n | ghosted: overall | ghosted: baseline bottom decile | ghosted: candidate bottom decile | still New: overall | still New: baseline bottom decile | still New: candidate bottom decile |
|---|---|---|---|---|---|---|---|
| all leads labelled by legacy rules (train + test) | 8041 | 15.0% | 24.5% | 24.4% | 16.5% | 27.5% | 27.5% |
| (a) legacy test set | 2453 | 5.7% | 9.4% | 9.0% | 10.7% | 19.2% | 19.2% |
| all mature leads, every label_source | 6278 | 19.2% | 30.6% | 31.3% | 19.2% | 30.6% | 31.3% |
| mature leads created on or after 2026-05-01, every label_source | 650 | 21.7% | 29.2% | 27.7% | 21.7% | 29.2% | 27.7% |

#### Value transforms (plan 3.2)

Candidate value = `value_formula` (p × E[deal value] × margin) through each transform, fitted on the candidate's 4049 training leads. Top-20% revenue = tie-averaged share of recorded won deal value in the top 20% by value; vs baseline = that share over the baseline's own (p×value, identity). value / revenue = total value over total recorded revenue. Every transform is monotone, so capture moves only through ties at the cut.

##### (a) Legacy labels, legacy test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | vs baseline | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (identity) | 11 | 346 | 5670 | 17167 | 45562 | 131.8× | 11.8% | 0.796 | 1 | 100.0% | 1.14 |
| candidate: identity | 6 | 414 | 6875 | 17889 | 44625 | 107.8× | 10.9% | 0.815 | 1 | 102.5% | 1.30 |
| candidate: cap p97 | 6 | 414 | 6875 | 13221 | 13221 | 31.9× | 6.6% | 0.815 | 1 | 102.5% | 1.21 |
| candidate: cap p97 + floor £25 | 25 | 414 | 6875 | 13221 | 13221 | 31.9× | 6.6% | 0.815 | 1 | 102.5% | 1.22 |
| candidate: cap p97 + sqrt | 156 | 1274 | 5193 | 7201 | 7201 | 5.7× | 3.5% | 0.815 | 1 | 102.5% | 1.26 |
| candidate: cap p97 + log | 25 | 1208 | 5412 | 6664 | 6664 | 5.5× | 3.2% | 0.815 | 1 | 102.5% | 1.25 |
| candidate: tiers 5 | 44 | 541 | 8910 | 8910 | 8910 | 16.5× | 4.2% | 0.792 | 479 | 99.5% | 1.29 |
| candidate: tiers 10 | 23 | 384 | 5471 | 12350 | 12350 | 32.2× | 5.8% | 0.802 | 227 | 100.8% | 1.29 |
| candidate: cap p97 + log + floor £25 | 25 | 1208 | 5412 | 6664 | 6664 | 5.5× | 3.2% | 0.815 | 1 | 102.5% | 1.25 |

##### (b) Horizon labels, mature test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | vs baseline | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (identity) | 11 | 432 | 7048 | 18577 | 34763 | 80.5× | 10.9% | 0.752 | 1 | 100.0% | 0.99 |
| candidate: identity | 6 | 518 | 7873 | 20121 | 32268 | 62.3× | 9.7% | 0.774 | 1 | 103.0% | 1.11 |
| candidate: cap p97 | 6 | 518 | 7873 | 13221 | 13221 | 25.5× | 5.0% | 0.774 | 1 | 103.0% | 1.02 |
| candidate: cap p97 + floor £25 | 25 | 518 | 7873 | 13221 | 13221 | 25.5× | 5.0% | 0.774 | 1 | 103.0% | 1.02 |
| candidate: cap p97 + sqrt | 152 | 1425 | 5557 | 7201 | 7201 | 5.1× | 2.8% | 0.774 | 1 | 103.0% | 0.99 |
| candidate: cap p97 + log | 23 | 1425 | 5668 | 6664 | 6664 | 4.7× | 2.6% | 0.774 | 1 | 103.0% | 0.99 |
| candidate: tiers 5 | 44 | 541 | 8910 | 8910 | 8910 | 16.5× | 3.3% | 0.742 | 105 | 98.7% | 1.03 |
| candidate: tiers 10 | 23 | 698 | 12350 | 12350 | 12350 | 17.7× | 4.5% | 0.747 | 47 | 99.3% | 1.07 |
| candidate: cap p97 + log + floor £25 | 25 | 1425 | 5668 | 6664 | 6664 | 4.7× | 2.6% | 0.774 | 1 | 103.0% | 0.99 |

##### All scored leads

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline (identity) | 11 | 374 | 6317 | 17841 | 45562 | 121.7× | 11.5% |
| candidate: identity | 6 | 446 | 7332 | 18646 | 44625 | 100.0× | 10.6% |
| candidate: cap p97 | 6 | 446 | 7332 | 13221 | 13221 | 29.6× | 6.3% |
| candidate: cap p97 + floor £25 | 25 | 446 | 7332 | 13221 | 13221 | 29.6× | 6.3% |
| candidate: cap p97 + sqrt | 152 | 1323 | 5363 | 7201 | 7201 | 5.4× | 3.4% |
| candidate: cap p97 + log | 23 | 1278 | 5534 | 6664 | 6664 | 5.2× | 3.1% |
| candidate: tiers 5 | 44 | 541 | 8910 | 8910 | 8910 | 16.5× | 4.0% |
| candidate: tiers 10 | 23 | 384 | 5471 | 12350 | 12350 | 32.2× | 5.5% |
| candidate: cap p97 + log + floor £25 | 25 | 1278 | 5534 | 6664 | 6664 | 5.2× | 3.1% |

##### Acceptance (plan 3.2): ≥ 95% of the baseline's capture on every test set and max/median < 20× on every test set and on all scored leads

| transform | result |
|---|---|
| identity | fail |
| cap p97 | fail |
| cap p97 + floor £25 | fail |
| cap p97 + sqrt | PASS |
| cap p97 + log | PASS |
| tiers 5 | PASS |
| tiers 10 | fail |
| cap p97 + log + floor £25 | PASS |

#### Click-ID coverage (plan 3.5)

Scored leads per paid channel by the identifier an upload could match on (see `docs/platform_contract.md`): a click id (gclid, gbraid, wbraid, fbclid, li_fat_id, oppref), else the lead-form id, else nothing but hashed email / phone (Meta: plus the `fbp` browser cookie).

| channel | leads | click id | lead-form id | no id: fbp present | no id at all | email present | phone present |
|---|---|---|---|---|---|---|---|
| google | 2763 | 72.3% | 0.0% | 22.0% | 27.7% | 100.0% | 25.2% |
| meta | 2733 | 70.7% | 0.0% | 22.6% | 29.3% | 100.0% | 24.4% |
| linkedin | 1485 | 70.8% | 0.0% | 24.5% | 29.2% | 100.0% | 23.7% |
| chatgpt | 473 | 69.6% | 0.0% | 24.9% | 30.4% | 100.0% | 24.5% |
| meta_leadads | 932 | 0.0% | 100.0% | 0.0% | 0.0% | 100.0% | 80.8% |
| landing-page paid (all but meta_leadads) | 7454 | 71.2% | 0.0% | 22.9% | 28.8% | 100.0% | 24.6% |
| all paid | 8386 | 63.3% | 11.1% | 20.4% | 25.6% | 100.0% | 30.8% |

#### Design collinearity (plan 2.7): FAIL

Candidate design (`v2` features, 39 columns) on its 4049 training rows; fails when two columns have |corr| > 0.95. The report exits 1 on a failure only with `--strict`; `python -m emva.eval.collinearity` always does.

- FAIL (1 pairs with |corr| > 0.95): max |corr| = 1.000 between channel=meta_leadads and session_missing=yes
  - channel=meta_leadads ~ session_missing=yes: +1.000

#### Baseline script output

- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, which fills blank deal values with its predicted value, printed:

```
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795
```

## Standard report, data/v2 (`make report DATA=data/v2`): value sections

Only the Phase 3 sections of the v2 report are pasted here (full output: `make report DATA=data/v2`):

#### Value transforms (plan 3.2)

Candidate value = `value_formula` (p × E[deal value] × margin) through each transform, fitted on the candidate's 3789 training leads. Top-20% revenue = tie-averaged share of recorded won deal value in the top 20% by value; vs baseline = that share over the baseline's own (p×value, identity). value / revenue = total value over total recorded revenue. Every transform is monotone, so capture moves only through ties at the cut.

##### (a) Legacy labels, legacy test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | vs baseline | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (identity) | 29 | 447 | 5351 | 14236 | 33007 | 73.9× | 11.1% | 0.719 | 1 | 100.0% | 1.08 |
| candidate: identity | 10 | 548 | 6719 | 17096 | 35656 | 65.1× | 10.3% | 0.778 | 1 | 108.3% | 1.40 |
| candidate: cap p97 | 10 | 548 | 6719 | 15259 | 15259 | 27.9× | 7.2% | 0.778 | 1 | 108.3% | 1.35 |
| candidate: cap p97 + floor £25 | 25 | 548 | 6719 | 15259 | 15259 | 27.9× | 7.2% | 0.778 | 1 | 108.3% | 1.35 |
| candidate: cap p97 + sqrt | 204 | 1523 | 5335 | 8039 | 8039 | 5.3× | 3.5% | 0.778 | 1 | 108.3% | 1.44 |
| candidate: cap p97 + log | 35 | 1422 | 5743 | 7589 | 7589 | 5.3× | 3.4% | 0.778 | 1 | 108.3% | 1.43 |
| candidate: tiers 5 | 64 | 729 | 9784 | 9784 | 9784 | 13.4× | 4.3% | 0.766 | 422 | 106.6% | 1.46 |
| candidate: tiers 10 | 30 | 532 | 5826 | 13742 | 13742 | 25.8× | 6.1% | 0.774 | 225 | 107.6% | 1.44 |
| candidate: cap p97 + log + floor £25 | 35 | 1422 | 5743 | 7589 | 7589 | 5.3× | 3.4% | 0.778 | 1 | 108.3% | 1.43 |

##### (b) Horizon labels, mature test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | vs baseline | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (identity) | 27 | 561 | 6278 | 16858 | 33007 | 58.8× | 10.7% | 0.663 | 1 | 100.0% | 0.96 |
| candidate: identity | 9 | 644 | 7963 | 20446 | 35656 | 55.4× | 10.1% | 0.741 | 1 | 111.7% | 1.19 |
| candidate: cap p97 | 9 | 644 | 7963 | 15259 | 15259 | 23.7× | 5.8% | 0.741 | 1 | 111.7% | 1.12 |
| candidate: cap p97 + floor £25 | 25 | 644 | 7963 | 15259 | 15259 | 23.7× | 5.8% | 0.741 | 1 | 111.7% | 1.12 |
| candidate: cap p97 + sqrt | 193 | 1651 | 5808 | 8039 | 8039 | 4.9× | 3.0% | 0.741 | 1 | 111.7% | 1.13 |
| candidate: cap p97 + log | 31 | 1603 | 6117 | 7589 | 7589 | 4.7× | 2.8% | 0.741 | 1 | 111.7% | 1.13 |
| candidate: tiers 5 | 64 | 729 | 9784 | 9784 | 9784 | 13.4× | 3.5% | 0.725 | 84 | 109.4% | 1.17 |
| candidate: tiers 10 | 30 | 532 | 5826 | 13742 | 13742 | 25.8× | 4.8% | 0.726 | 54 | 109.5% | 1.21 |
| candidate: cap p97 + log + floor £25 | 31 | 1603 | 6117 | 7589 | 7589 | 4.7× | 2.8% | 0.741 | 1 | 111.7% | 1.13 |

##### All scored leads

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline (identity) | 27 | 498 | 5862 | 16919 | 36283 | 72.8× | 11.3% |
| candidate: identity | 10 | 599 | 7410 | 20432 | 40539 | 67.6× | 10.4% |
| candidate: cap p97 | 10 | 599 | 7410 | 15259 | 15259 | 25.5× | 6.7% |
| candidate: cap p97 + floor £25 | 25 | 599 | 7410 | 15259 | 15259 | 25.5× | 6.7% |
| candidate: cap p97 + sqrt | 204 | 1593 | 5602 | 8039 | 8039 | 5.0× | 3.4% |
| candidate: cap p97 + log | 35 | 1521 | 5958 | 7589 | 7589 | 5.0× | 3.2% |
| candidate: tiers 5 | 64 | 729 | 9784 | 9784 | 9784 | 13.4× | 4.0% |
| candidate: tiers 10 | 30 | 532 | 5826 | 13742 | 13742 | 25.8× | 5.7% |
| candidate: cap p97 + log + floor £25 | 35 | 1521 | 5958 | 7589 | 7589 | 5.0× | 3.2% |

##### Acceptance (plan 3.2): ≥ 95% of the baseline's capture on every test set and max/median < 20× on every test set and on all scored leads

| transform | result |
|---|---|
| identity | fail |
| cap p97 | fail |
| cap p97 + floor £25 | fail |
| cap p97 + sqrt | PASS |
| cap p97 + log | PASS |
| tiers 5 | PASS |
| tiers 10 | fail |
| cap p97 + log + floor £25 | PASS |

#### Click-ID coverage (plan 3.5)

Scored leads per paid channel by the identifier an upload could match on (see `docs/platform_contract.md`): a click id (gclid, gbraid, wbraid, fbclid, li_fat_id, oppref), else the lead-form id, else nothing but hashed email / phone (Meta: plus the `fbp` browser cookie).

| channel | leads | click id | lead-form id | no id: fbp present | no id at all | email present | phone present |
|---|---|---|---|---|---|---|---|
| google | 2709 | 71.9% | 0.0% | 19.2% | 28.1% | 100.0% | 24.8% |
| meta | 2725 | 69.4% | 0.0% | 21.2% | 30.6% | 100.0% | 24.7% |
| linkedin | 1521 | 70.5% | 0.0% | 19.8% | 29.5% | 100.0% | 25.6% |
| chatgpt | 480 | 72.7% | 0.0% | 20.0% | 27.3% | 100.0% | 26.0% |
| meta_leadads | 872 | 0.0% | 100.0% | 0.0% | 0.0% | 100.0% | 80.5% |
| landing-page paid (all but meta_leadads) | 7435 | 70.8% | 0.0% | 20.1% | 29.2% | 100.0% | 25.0% |
| all paid | 8307 | 63.3% | 10.5% | 18.0% | 26.2% | 100.0% | 30.8% |

## tROAS calculator output

### v1: `python -m emva.troas --data data/v1`

data/v1: positive rate 0.133 (won within H over mature leads)

Google Ads tROAS: threshold 30 conversions per campaign per 30 days (verify)

| campaign | leads / month | lead conversions per 30 days | win conversions per 30 days | submit stage | close stage | close volume needed × |
|---|---|---|---|---|---|---|
| attribution_software | 60.5 | 59.6 | 7.9 | ok | BELOW | 3.8 |
| brand_search | 59.4 | 58.6 | 7.8 | ok | BELOW | 3.9 |
| competitor_terms | 54.4 | 53.6 | 7.1 | ok | BELOW | 4.2 |
| generic_lead_scoring | 55.9 | 55.1 | 7.3 | ok | BELOW | 4.1 |
| all campaigns pooled (portfolio) | 230.2 | 226.9 | 30.1 | ok | ok | 1.0 |

Meta value optimisation: threshold 50 conversions per ad set per 7 days (verify)

| campaign | leads / month | lead conversions per 7 days | win conversions per 7 days | submit stage | close stage | close volume needed × |
|---|---|---|---|---|---|---|
| guide_download | 76.5 | 17.6 | 2.3 | BELOW | BELOW | 21.4 |
| prospecting_lookalike | 77.2 | 17.7 | 2.4 | BELOW | BELOW | 21.3 |
| retargeting_30d | 76.2 | 17.5 | 2.3 | BELOW | BELOW | 21.5 |
| uk_leadgen_q | 75.6 | 17.4 | 2.3 | BELOW | BELOW | 21.7 |
| all campaigns pooled (portfolio) | 305.4 | 70.2 | 9.3 | ok | BELOW | 5.4 |

### v2: `python -m emva.troas --data data/v2`

data/v2: positive rate 0.131 (won within H over mature leads)

Google Ads tROAS: threshold 30 conversions per campaign per 30 days (verify)

| campaign | leads / month | lead conversions per 30 days | win conversions per 30 days | submit stage | close stage | close volume needed × |
|---|---|---|---|---|---|---|
| attribution_software | 57.4 | 56.6 | 7.4 | ok | BELOW | 4.0 |
| brand_search | 56.7 | 55.9 | 7.3 | ok | BELOW | 4.1 |
| competitor_terms | 57.6 | 56.8 | 7.4 | ok | BELOW | 4.0 |
| generic_lead_scoring | 54.1 | 53.3 | 7.0 | ok | BELOW | 4.3 |
| all campaigns pooled (portfolio) | 225.8 | 222.5 | 29.1 | ok | BELOW | 1.0 |

Meta value optimisation: threshold 50 conversions per ad set per 7 days (verify)

| campaign | leads / month | lead conversions per 7 days | win conversions per 7 days | submit stage | close stage | close volume needed × |
|---|---|---|---|---|---|---|
| guide_download | 75.7 | 17.4 | 2.3 | BELOW | BELOW | 21.9 |
| prospecting_lookalike | 75.2 | 17.3 | 2.3 | BELOW | BELOW | 22.1 |
| retargeting_30d | 71.9 | 16.5 | 2.2 | BELOW | BELOW | 23.1 |
| uk_leadgen_q | 76.9 | 17.7 | 2.3 | BELOW | BELOW | 21.6 |
| all campaigns pooled (portfolio) | 299.8 | 68.9 | 9.0 | ok | BELOW | 5.5 |

### 2,000 leads a year, one campaign: `--leads-per-month 166.67 --positive-rate 0.132 --campaigns 1`

Google Ads tROAS: threshold 30 conversions per campaign per 30 days (verify)

| campaign | leads / month | lead conversions per 30 days | win conversions per 30 days | submit stage | close stage | close volume needed × |
|---|---|---|---|---|---|---|
| campaign 1 | 166.7 | 164.3 | 21.7 | ok | BELOW | 1.4 |

Meta value optimisation: threshold 50 conversions per ad set per 7 days (verify)

| campaign | leads / month | lead conversions per 7 days | win conversions per 7 days | submit stage | close stage | close volume needed × |
|---|---|---|---|---|---|---|
| campaign 1 | 166.7 | 38.3 | 5.1 | BELOW | BELOW | 9.9 |

### 2,000 leads a year, three campaigns: `--campaigns 3`

Google Ads tROAS: threshold 30 conversions per campaign per 30 days (verify)

| campaign | leads / month | lead conversions per 30 days | win conversions per 30 days | submit stage | close stage | close volume needed × |
|---|---|---|---|---|---|---|
| campaign 1 | 55.6 | 54.8 | 7.2 | ok | BELOW | 4.2 |
| campaign 2 | 55.6 | 54.8 | 7.2 | ok | BELOW | 4.2 |
| campaign 3 | 55.6 | 54.8 | 7.2 | ok | BELOW | 4.2 |

Meta value optimisation: threshold 50 conversions per ad set per 7 days (verify)

| campaign | leads / month | lead conversions per 7 days | win conversions per 7 days | submit stage | close stage | close volume needed × |
|---|---|---|---|---|---|---|
| campaign 1 | 55.6 | 12.8 | 1.7 | BELOW | BELOW | 29.6 |
| campaign 2 | 55.6 | 12.8 | 1.7 | BELOW | BELOW | 29.6 |
| campaign 3 | 55.6 | 12.8 | 1.7 | BELOW | BELOW | 29.6 |

