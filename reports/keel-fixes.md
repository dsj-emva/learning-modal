# Keel QA fix round (M1-M4, M6)

Fixes for the issues found by the smoke test and hands-on QA of the live Keel deployment (commit 31dd8b8,
2026-09-27). Rulings are in ADR 0026 (Proposed). Three tasks ran in parallel, each on its own branch; the
orchestrator reviewed and merged each one. Every Olist number here is **on public data** (CC BY-NC-SA 4.0,
non-commercial; not for quoting).

| issue | fix | branch, commits |
|---|---|---|
| M1 the Score page's example lead refused on a legacy-features run of a converted dataset, one unseen level at a time | `UnknownLevelError.problems` names every offending feature; Keel shows them by form field; a legacy run whose built-in example would be refused starts from a training lead (ADR 0026 decision 1) | `keel-fix-scoring` 1df71c4 |
| M2 source-format scoring accepted a blank `created_at` | `convert_leads` requires `created_at` (missing column or blank refused, naming the rows); the source form marks it `*`; the CLI `score` does the same (decision 2) | `keel-fix-scoring` 11f40cf |
| M4 Olist amounts (BRL) shown as £ | optional mapping `currency` goes to `dataset.json`; one formatter `app.components.money`: £ for the sample, `BRL 41,106` for a declared code, a plain number plus a note when unknown; `olist_funnel` = BRL (decision 3) | `keel-fix-claims` f73b74b, 1585428 |
| M3 signals constant on every training lead shown as movers | constant design columns (no variance on the training rows) are left out of the scorecard, with a one-line note; the Score breakdown folds them into the starting score and drops rows that round to 0, and the parts add up (decision 4) | `keel-fix-claims` 50cdd46 |
| M6 a won-only column mapped to a field raised no warning | `convert` records `outcome_fill_gaps` (Won vs Lost filled share differs by ≥ 0.5) for fields and extras; the CLI prints them, Keel warns after Convert and in Check results; never a block (decision 5) | `keel-fix-leak-warning` 49ba259, 81fecb5 |

## Checks (on public data where Olist)

- **M1:** a legacy/legacy run on the real Olist export refused the built-in example on six features in one
  message (band, text, seniority, time_on_page, business_hours, edits_1_4). The form then started from training
  lead `dac32acd…`, which scores.
- **M2:** `python -m emva.ingest score` with a blank `first_contact_date` is refused, naming the rows.
- **M4:** Olist's `dataset.json` has `"currency": "BRL"`, and the transform reads "capped at BRL 189,647 … floored at
  BRL 25". The currencies of `crm_opportunities` and `hotel_bookings` are left unset:
  - the Maven data dictionary gives USD only for `accounts.revenue`, not for the deal's `close_value`;
  - the hotel paper does not state a currency for `adr`.
- **M3:** Olist generic run: 37 signals are the same on every training lead. Five of them are on for every lead
  (Company size · missing, Company not in database · yes, Seniority · not_asked/blank, No on-site session · yes,
  What they want to solve · vague); their −45 points now sit in the starting score. On data/v1 no column is
  constant for any label mode or feature set, so the sample's scorecard is unchanged.
- **M6:** the unchanged `mappings/olist_funnel.toml` gives `outcome_fill_gaps: []`, and the five training files plus
  `extra_features.csv` are byte-identical to the conversion before the change. The QA repro (`business_segment` →
  `answers.what_to_solve`) is flagged: won 100%, lost 0%.
- Merged main: `make test` 842 passed; `make baseline` PASS (byte-identical weights and scores). The v1 standard
  report below is byte-identical to 44969b7's.

## Orchestrator decisions

- The three open choices of the plan were taken as recommended: a `currency` field rather than a page note only,
  constant signals hidden with a note rather than tagged, and won-only fields as a warning rather than a block.
- `created_at` is required rather than filled with the current time, although no model feature reads it: a row the
  conversion would refuse should not be scored silently.
- Currencies are set only where a source states them (Olist: BRL, a Brazilian marketplace).

## Open

- The API key in Railway's `ANTHROPIC_API_KEY` must be rotated and the variable split in two. The draft path also
  still logs the rejected header, which is the key.
- Not addressed from the QA list: M5 (Map & convert state lost when leaving the page), M7 (the source form offers
  "other" for `origin`, but a CSV with an unlisted origin is refused), and the minor and cosmetic items.
- `VALUE_FLOOR_GBP` is misnamed for non-GBP data. The review table keeps a row's stale reason after its target is
  edited.

<details>
<summary>Standard report: data/v1 (on simulated data; identical to 44969b7)</summary>

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
WARNING: candidate design collinearity check: FAIL (1 pairs with |corr| > 0.95): max |corr| = 1.000 between channel=meta_leadads and session_missing=yes


</details>
