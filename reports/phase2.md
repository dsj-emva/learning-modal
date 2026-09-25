# Phase 2: features and leakage

Branch `phase2-features` from `origin/main` 92168cf. Plan items 2.1 to 2.7. `--feature-set {legacy,v2}`
(default `v2`) sits next to `--label-mode`; `--label-mode legacy --feature-set legacy` is still
byte-identical to `baseline/` (`make baseline` passes, weights.csv and scores.csv byte-identical).

## Acceptance

| criterion | result | evidence |
|---|---|---|
| No two coefficients identical to 3 dp | **pass** for the default model (v2 features, horizon labels): no equal pair. Caveat below for v2 + legacy labels | baseline: `email=free` = `no_company=yes` = −0.391 (corr +1.000). v2 horizon: none. v2 legacy labels: two coincidental ties, `search_term=brand` = `business_hours=wkday_9-18` (0.166, corr +0.059) and `spend=£25k-£100k` = `spend=£100k+` (0.521, corr −0.081) |
| Meta lead-ads leads no longer in the reference bucket for every behavioural feature | **pass** (at feature level; see the alias caveat) | all 932 cleaned lead-ads leads: legacy = reference for all 7 features; v2 = `missing` for all 7 (table in section 2.1) |
| AUC within baseline CI [0.789, 0.836], legacy test set | **pass** | v2 + legacy labels 0.816 [0.792, 0.838], paired vs baseline +0.002 [−0.001, +0.006]; v2 + horizon labels 0.814 [0.790, 0.837], paired +0.000 [−0.006, +0.006] |
| AUC on the frozen mature test set (R3) | **reported** (baseline CI there is [0.726, 0.827]) | v2 + legacy labels 0.780 [0.729, 0.828], paired +0.002 [−0.005, +0.011]; v2 + horizon 0.778 [0.730, 0.827], paired +0.000 [−0.015, +0.015] |
| `grep -rn "0.45\|ground_truth" emva/` only `emva/eval/` | **pass** | enforced by `tests/test_label_study.py::test_ground_rule_2_grep` (expects nothing outside `emva/eval/`) |

The ties in the v2 + legacy-labels model are two unrelated columns (|corr| < 0.1) whose fitted
log-odds happen to round alike; the criterion targets duplicated columns such as the baseline's
`email=free`/`no_company=yes`. I report it rather than call it a clean pass. The baseline also had
one such coincidental tie (`business_hours=wkday_9-18` = `channel=organic_direct`, 0.160).

## Summary of the other required numbers

- **2.3 match count:** of the **2,904** cleaned leads whose email/company domain misses companies.csv,
  **948 (32.6%)** match a companies.csv name exactly after normalisation (all through the
  `historical_leads.company_name` column; the `company` answer adds none because on v1 it is the same
  string). This is far more than the "few" expected: on v1 every domain miss is a free-email lead, and
  1,317 of them typed a company name. Ground truth (evaluation only) confirms every match: matched
  band and sector equal the generator's for 948/948. Unmatched: 1,721 blank names, "Freelance" (118),
  "Self-employed" (117). No fuzzy matching yet. On all 10,000 raw leads there are 3,267 domain misses
  (the 2,904 is after bot/duplicate removal).
- **2.5 boilerplate:** threshold Jaccard ≥ **0.6** against 16 snippets. On v1: precision **1.000**, recall
  **1.000** (807/807 against the generator's `text_category`); the legacy prefix rule scores the same.
  Margin: true copy-paste texts all score 1.000, the highest other text 0.179.
- **2.6:** all four dropped; no level's 95% CI (1,000 refits, horizon labels) excludes zero. Same
  decision with legacy labels. Table in section 2.6. `ip_type=missing` has no coefficient of its own
  on v1 (alias of `channel=meta_leadads`).
- **2.7 max |corr|:** legacy **1.000** (`email=free` ~ `no_company=yes`, FAIL, both label modes);
  v2 **0.772** horizon / 0.782 legacy labels (`enrichment_missing=yes` ~ `band=missing`, pass).
- **2.4 value model:** fixed sd 0.45 → correction 1.1066; estimated sd **0.4546** (770 training deals)
  → correction **1.1089** (+0.2%). Top-20% revenue by p×value is unchanged by construction (one
  constant factor): (a) 0.8163, (b) 0.7742 for v2 + horizon under both. Mean predicted / mean recorded
  deal value on won test leads 0.968 → 0.970 (v2 horizon), 1.003 → 1.005 (v2 legacy labels).
- **Feature change alone** (same labels, paired): legacy labels +0.002 [−0.001, +0.006] (a),
  +0.002 [−0.005, +0.011] (b); horizon labels +0.001 [−0.002, +0.005] (a), +0.001 [−0.007, +0.009] (b).
  The feature fixes do not change ranking quality on v1; they change what the weights mean.

## Before/after weights for changed features

Log-odds. "= X" = column dropped as identical to X on the training rows. Full table in the study below.

| column | baseline | v2, legacy labels | v2, horizon (default) |
|---|---|---|---|
| no_company=yes | −0.391 | (removed) | (removed) |
| enrichment_missing=yes | | 0.088 | 0.172 |
| email=free | −0.391 | −1.097 | −1.102 |
| band=11-50 / 51-200 / 201-1000 / 1000+ | 0.193 / 1.078 / 0.873 / 0.703 | 0.202 / 1.092 / 0.809 / 0.675 | 0.268 / 1.170 / 0.928 / 0.393 |
| band=missing | | 0.058 | −0.106 |
| spend=none / £5k-£25k / £25k-£100k / £100k+ | −0.571 / 0.246 / 0.484 / 0.429 | −0.588 / 0.260 / 0.521 / 0.521 | −0.601 / 0.195 / 0.475 / 0.675 |
| crm=hubspot_sf | 0.131 | 0.121 | 0.098 |
| hiring=hiring | 0.360 | 0.364 | 0.379 |
| spend/crm/hiring=missing | | = enrichment_missing=yes | = enrichment_missing=yes |
| time_on_page=60-300s / 300-600s / >600s | 0.508 / −0.090 / −0.214 | 0.515 / −0.096 / −0.213 | 0.587 / 0.007 / −0.144 |
| hesitation_90s=yes | −0.312 | −0.290 | −0.261 |
| sessions_3plus=yes | 0.406 | 0.401 | 0.394 |
| viewed_pricing=yes | 0.229 | 0.238 | 0.236 |
| ip_country=mismatch | −0.331 | −0.305 | −0.230 |
| business_hours=wkday_9-18 | 0.160 | 0.166 | 0.079 |
| 7 behavioural `=missing` levels (ip_type's included) | | = channel=meta_leadads | = channel=meta_leadads |
| channel=meta_leadads | −0.489 | −0.479 | −0.622 |
| text=copy_paste | −0.950 | −0.939 | −1.000 |
| c_budget (4 levels) | −0.082 to 0.152 | (dropped, 2.6) | (dropped) |
| c_timeline (4 levels) | 0.002 to 0.195 | (dropped) | (dropped) |
| ip_type=dc | 0.022 | (dropped) | (dropped) |
| edits_1_4=yes | 0.050 | (dropped) | (dropped) |

The free-email effect was split equally between two identical columns in the baseline (−0.391
each, −0.782 together); v2 gives it one column (−1.10), and name enrichment moves 948 free-email
leads out of "no company", so `enrichment_missing` is now separately estimable (small, positive).

## What was done

- **Switch.** `emva.features.FeatureSet` (`legacy`, `v2`), `--feature-set` in `emva/cli.py` for
  `python -m emva`, `emva.eval.report` and `emva.eval.collinearity`. The legacy path (`add_features`,
  `design(X)`, fixed residual sd) is intact; `email_type` and `search_term` were factored out as
  helpers shared by both sets (byte identity verified). `make baseline` runs
  `--label-mode legacy --feature-set legacy`. `run_experiments.py` gains `legacy-labels` and
  `horizon-legacy-features`. Phase 1 study scripts (`label_study`, `ground_truth_reference`) and
  `scripts/compare_to_v1.py` are pinned to legacy features so their numbers stay those of phase1.md.
- **2.1** `missing` level for time_on_page, hesitation_90s, sessions_3plus, viewed_pricing,
  ip_country, ip_type, business_hours when the raw input is NaN. `business_hours` has no channel
  test: it is `missing` when weekday/hour is blank **or the lead has no on-site session (blank
  `landing_url`)**. Lead-ads leads have a local submit hour in v1, so a purely "hour missing" rule
  would have put them in `wkday_9-18`/`outside`; the no-session rule is how they become `missing`
  without naming the channel.
- **2.2** `enrichment_missing` (no companies.csv row by domain or by name) replaces `no_company`;
  `band` = enrichment → typed `company_size` (blank = none) → `missing`; `spend`, `crm`, `hiring`
  (and `sector` in the value model) `missing` without enrichment.
- **2.3** `emva.io.normalise_company_name`, `match_company_names`: lower-case, drop dots and
  apostrophes ("B.V." → "bv"), other punctuation → space, collapse whitespace, strip trailing
  suffixes ltd/limited/inc/llc/gmbh/bv/sa/plc/co **plus sas** (companies.csv uses SAS). Names that
  normalise to the same key for two companies are skipped. Exact match only.
- **2.4** `emva.value.DealValueModel`; `fit_deal_value` estimates sd = sample sd (ddof 1) of the
  in-sample training residuals of the log-value ridge and applies exp(sd²/2).
- **2.5** `emva/boilerplate.py`: 16 snippets (the three v1 filler texts observed in leads plus 13
  generic ones written for this), token-set Jaccard, `BOILERPLATE_SIMILARITY_THRESHOLD = 0.6`.
- **2.6** `python -m emva.eval.feature_selection` (≥ 200 refits enforced; run with 1,000);
  outcome hard-coded in `emva.features.V2_DROPPED` with a comment citing this report.
- **2.7** `emva/eval/collinearity.py` (`check_collinearity`, `assert_no_collinearity`, CLI exits 1);
  the standard report has a "Design collinearity" section and exits 1 when the candidate fails.

## Deviations, not done, open questions

1. **Exact-alias pruning in the v2 design (the important one).** On v1 every behavioural `missing`
   level is present for exactly the lead-ads leads, so each is the same column as
   `channel=meta_leadads`; and `spend/crm/hiring=missing` are the same column as
   `enrichment_missing=yes` by construction. Kept as separate columns they would be perfectly
   collinear (2.7 would fail v2) and L2 would give them identical weights (acceptance 1 would fail).
   v2 therefore drops any design column identical on the training rows to an earlier one
   (`emva.design.drop_aliased_columns`), keeps the first (channel, enrichment_missing), and reports
   the alias list in `PipelineResult.aliases`, the report and the study. Consequence: on v1 the
   "no session" effect and the lead-ads effect cannot be separated, the lead-ads weight carries both,
   and a future non-lead-ads lead with missing telemetry (v2 data, consent declined) would get 0 for
   those levels until the data contains such leads, at which point the columns differ and get their
   own weights automatically. The collinearity check runs after pruning; legacy is not pruned.
   Please confirm this is acceptable, or say which alternative you prefer (single `session_missing`
   feature; fail loudly instead).
2. **The legacy residual sd constant lives in `emva/eval/regression.py`** and is imported by
   `emva/pipeline.py` for `--feature-set legacy` only: byte identity with `baseline/` needs it and
   the grep rule allows it only under `emva/eval/`. The v2 path never reads it.
3. `sas` added to the suffix list (not in the brief's list).
4. 2.3 match count is 948, not "few"; ground truth says all are correct.
5. `edits_1_4` got no missing level (not in the 2.1 list; dropped by 2.6 anyway).
6. Coincidental 3-dp ties in the v2 + legacy-labels model (see acceptance).
7. Intermediate commits on the branch are grouped by plan item and are not each green on their own
   (value/io API changes land before the pipeline commit); the branch head is.
8. `tests/test_generate_data_v1.py` and `scripts/compare_to_v1.py` (shared with the Phase 5 agent)
   got a one-line change each to pin legacy features; expect a trivial merge.

## Commands

```
make PY=.venv/bin/python baseline
python -m emva.eval.report                       # (1) v2 features, horizon labels (default)
python -m emva.eval.report --label-mode legacy   # (2) v2 features, legacy labels
python -m emva.eval.phase2_study
python -m emva.eval.feature_selection --n-refits 1000
python -m emva.eval.collinearity [--feature-set legacy]
```

The outputs follow, verbatim.


---

## Output: `python -m emva.eval.report` (v2 features, horizon labels)

## EMVA standard report

Data: `data/v1`. baseline = frozen `baseline/emva_score.py`; candidate = `emva` pipeline trained with `horizon H=120` labels and `v2` features; status quo = `status_quo_rules.json` reconstructed. Every model is scored on the same rows under two test definitions. AUC CI: percentile bootstrap, 1000 resamples, seed 0. Top-20% capture ranks by p (status quo: by its value) or by p×value.

### Label definitions

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

### (a) Legacy labels, legacy test set

2453 leads labelled by the baseline rules, created on or after 2026-05-01 (352 won). Label = legacy y.

#### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.814 [0.789, 0.836] | 0.1006 | 0.571 | 0.742 | 0.796 |
| candidate | 0.814 [0.790, 0.837] | 0.1020 | 0.571 | 0.733 | 0.816 |
| status quo | 0.639 [0.605, 0.673] | n/a | 0.392 | 0.454 | 0.454 |

#### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.000 | [-0.006, +0.006] | 0.992 |
| status quo | -0.174 | [-0.209, -0.141] | 0.000 |

#### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 246 | 0.009 | 0.016 | 0.009 | 0.012 |
| 2 | 245 | 0.020 | 0.012 | 0.022 | 0.012 |
| 3 | 245 | 0.033 | 0.016 | 0.039 | 0.029 |
| 4 | 245 | 0.050 | 0.053 | 0.061 | 0.049 |
| 5 | 246 | 0.075 | 0.077 | 0.092 | 0.065 |
| 6 | 245 | 0.109 | 0.135 | 0.132 | 0.135 |
| 7 | 245 | 0.150 | 0.131 | 0.184 | 0.151 |
| 8 | 245 | 0.213 | 0.176 | 0.254 | 0.163 |
| 9 | 245 | 0.316 | 0.327 | 0.363 | 0.347 |
| 10 | 246 | 0.511 | 0.492 | 0.560 | 0.472 |

Status quo has no probability, so it has no Brier score or calibration.

#### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 754 | 103 | 0.800 | 0.801 | 0.641 |
| 2026-06 | 665 | 93 | 0.797 | 0.807 | 0.631 |
| 2026-07 | 541 | 88 | 0.833 | 0.826 | 0.602 |
| 2026-08 | 371 | 55 | 0.813 | 0.811 | 0.664 |
| 2026-09 | 122 | 13 | 0.920 | 0.914 | 0.772 |

#### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 2453 | 346 | 17167 | 45562 | 131.8× | 11.8% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 2453 | 412 | 17943 | 44669 | 108.5× | 10.9% |
| candidate | all scored leads | 9311 | 446 | 18808 | 44669 | 100.1× | 10.6% |
| status quo | test set | 2453 | 96 | 1440 | 1872 | 19.5× | 6.7% |
| status quo | all scored leads | 9311 | 96 | 1440 | 1872 | 19.5× | 6.7% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

#### Notes

- status quo: 203 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.397.
- status quo: 203 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454.
- status quo: 203 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454.

### (b) Horizon labels, mature test set

458 mature leads created on or after 2026-05-01 and up to 2026-05-26T20:06:58+00:00 that the default horizon definition labels (80 won). Label = won within 120 days; ghosted-at-H leads excluded, stalled leads censored.

#### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.778 [0.726, 0.827] | 0.1258 | 0.487 | 0.670 | 0.752 |
| candidate | 0.778 [0.730, 0.827] | 0.1265 | 0.450 | 0.608 | 0.774 |
| status quo | 0.644 [0.571, 0.717] | n/a | 0.400 | 0.501 | 0.501 |

#### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.000 | [-0.015, +0.015] | 0.970 |
| status quo | -0.134 | [-0.199, -0.069] | 0.000 |

#### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 46 | 0.008 | 0.022 | 0.009 | 0.000 |
| 2 | 46 | 0.020 | 0.000 | 0.022 | 0.043 |
| 3 | 46 | 0.038 | 0.022 | 0.045 | 0.000 |
| 4 | 45 | 0.060 | 0.111 | 0.072 | 0.089 |
| 5 | 46 | 0.090 | 0.174 | 0.105 | 0.109 |
| 6 | 46 | 0.124 | 0.152 | 0.151 | 0.261 |
| 7 | 45 | 0.171 | 0.178 | 0.208 | 0.222 |
| 8 | 46 | 0.237 | 0.239 | 0.280 | 0.239 |
| 9 | 46 | 0.357 | 0.391 | 0.392 | 0.348 |
| 10 | 46 | 0.555 | 0.457 | 0.593 | 0.435 |

Status quo has no probability, so it has no Brier score or calibration.

#### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 458 | 80 | 0.778 | 0.778 | 0.644 |

#### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 458 | 432 | 18577 | 34763 | 80.5× | 10.9% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 458 | 522 | 20161 | 32268 | 61.9× | 9.7% |
| candidate | all scored leads | 9311 | 446 | 18808 | 44669 | 100.1× | 10.6% |
| status quo | test set | 458 | 96 | 1440 | 1872 | 19.5× | 6.4% |
| status quo | all scored leads | 9311 | 96 | 1440 | 1872 | 19.5× | 6.7% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

#### Notes

- status quo: 37 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.398.
- status quo: 37 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.506.
- status quo: 37 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.506.

### Bottom-decile ghosted share

Share of leads in the 10% of each population with the lowest p that are *ghosted* (`label_source`: mature and not contacted within 120 days) or *still New* at 2026-09-24 whatever their age (the definition behind the plan's baseline figure of 27%). The horizon test set (b) excludes ghosted leads, so the comparison uses populations that keep them.

| population | n | ghosted: overall | ghosted: baseline bottom decile | ghosted: candidate bottom decile | still New: overall | still New: baseline bottom decile | still New: candidate bottom decile |
|---|---|---|---|---|---|---|---|
| all leads labelled by legacy rules (train + test) | 8041 | 15.0% | 24.5% | 24.4% | 16.5% | 27.5% | 27.5% |
| (a) legacy test set | 2453 | 5.7% | 9.4% | 9.4% | 10.7% | 19.2% | 19.6% |
| all mature leads, every label_source | 6278 | 19.2% | 30.6% | 31.3% | 19.2% | 30.6% | 31.3% |
| mature leads created on or after 2026-05-01, every label_source | 650 | 21.7% | 29.2% | 27.7% | 21.7% | 29.2% | 27.7% |

### Design collinearity (plan 2.7)

Candidate design (`v2` features, 38 columns) on its 4049 training rows; fails when two columns have |corr| > 0.95.

- PASS: max |corr| = 0.772 between enrichment_missing=yes and band=missing
- Dropped before the check as identical on the training rows to an earlier column (the earlier column carries the shared weight):
  - spend=missing = enrichment_missing=yes
  - crm=missing = enrichment_missing=yes
  - hiring=missing = enrichment_missing=yes
  - time_on_page=missing = channel=meta_leadads
  - hesitation_90s=missing = channel=meta_leadads
  - sessions_3plus=missing = channel=meta_leadads
  - viewed_pricing=missing = channel=meta_leadads
  - business_hours=missing = channel=meta_leadads
  - ip_country=missing = channel=meta_leadads

### Baseline script output

- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, which fills blank deal values with its predicted value, printed:

```
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795
```

---

## Output: `python -m emva.eval.report --label-mode legacy` (v2 features, legacy labels)

## EMVA standard report

Data: `data/v1`. baseline = frozen `baseline/emva_score.py`; candidate = `emva` pipeline trained with `legacy` labels and `v2` features; status quo = `status_quo_rules.json` reconstructed. Every model is scored on the same rows under two test definitions. AUC CI: percentile bootstrap, 1000 resamples, seed 0. Top-20% capture ranks by p (status quo: by its value) or by p×value.

### Label definitions

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

Candidate trained with: `legacy`.

### (a) Legacy labels, legacy test set

2453 leads labelled by the baseline rules, created on or after 2026-05-01 (352 won). Label = legacy y.

#### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.814 [0.789, 0.836] | 0.1006 | 0.571 | 0.742 | 0.796 |
| candidate | 0.816 [0.792, 0.838] | 0.1004 | 0.582 | 0.750 | 0.801 |
| status quo | 0.639 [0.605, 0.673] | n/a | 0.392 | 0.454 | 0.454 |

#### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.002 | [-0.001, +0.006] | 0.292 |
| status quo | -0.174 | [-0.209, -0.141] | 0.000 |

#### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 246 | 0.009 | 0.016 | 0.008 | 0.016 |
| 2 | 245 | 0.020 | 0.012 | 0.018 | 0.008 |
| 3 | 245 | 0.033 | 0.016 | 0.032 | 0.024 |
| 4 | 245 | 0.050 | 0.053 | 0.049 | 0.049 |
| 5 | 246 | 0.075 | 0.077 | 0.074 | 0.085 |
| 6 | 245 | 0.109 | 0.135 | 0.108 | 0.114 |
| 7 | 245 | 0.150 | 0.131 | 0.150 | 0.135 |
| 8 | 245 | 0.213 | 0.176 | 0.217 | 0.167 |
| 9 | 245 | 0.316 | 0.327 | 0.319 | 0.343 |
| 10 | 246 | 0.511 | 0.492 | 0.512 | 0.492 |

Status quo has no probability, so it has no Brier score or calibration.

#### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 754 | 103 | 0.800 | 0.802 | 0.641 |
| 2026-06 | 665 | 93 | 0.797 | 0.805 | 0.631 |
| 2026-07 | 541 | 88 | 0.833 | 0.831 | 0.602 |
| 2026-08 | 371 | 55 | 0.813 | 0.810 | 0.664 |
| 2026-09 | 122 | 13 | 0.920 | 0.915 | 0.772 |

#### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 2453 | 346 | 17167 | 45562 | 131.8× | 11.8% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 2453 | 335 | 17177 | 45040 | 134.3× | 11.6% |
| candidate | all scored leads | 9311 | 363 | 17790 | 45040 | 124.1× | 11.4% |
| status quo | test set | 2453 | 96 | 1440 | 1872 | 19.5× | 6.7% |
| status quo | all scored leads | 9311 | 96 | 1440 | 1872 | 19.5× | 6.7% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

#### Notes

- status quo: 203 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.397.
- status quo: 203 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454.
- status quo: 203 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454.

### (b) Horizon labels, mature test set

458 mature leads created on or after 2026-05-01 and up to 2026-05-26T20:06:58+00:00 that the default horizon definition labels (80 won). Label = won within 120 days; ghosted-at-H leads excluded, stalled leads censored.

#### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.778 [0.726, 0.827] | 0.1258 | 0.487 | 0.670 | 0.752 |
| candidate | 0.780 [0.729, 0.828] | 0.1257 | 0.500 | 0.677 | 0.752 |
| status quo | 0.644 [0.571, 0.717] | n/a | 0.400 | 0.501 | 0.501 |

#### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.002 | [-0.005, +0.011] | 0.608 |
| status quo | -0.134 | [-0.199, -0.069] | 0.000 |

#### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 46 | 0.008 | 0.022 | 0.007 | 0.022 |
| 2 | 46 | 0.020 | 0.000 | 0.018 | 0.000 |
| 3 | 46 | 0.038 | 0.022 | 0.038 | 0.022 |
| 4 | 45 | 0.060 | 0.111 | 0.059 | 0.089 |
| 5 | 46 | 0.090 | 0.174 | 0.088 | 0.174 |
| 6 | 46 | 0.124 | 0.152 | 0.123 | 0.174 |
| 7 | 45 | 0.171 | 0.178 | 0.171 | 0.200 |
| 8 | 46 | 0.237 | 0.239 | 0.242 | 0.196 |
| 9 | 46 | 0.357 | 0.391 | 0.354 | 0.413 |
| 10 | 46 | 0.555 | 0.457 | 0.556 | 0.457 |

Status quo has no probability, so it has no Brier score or calibration.

#### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 458 | 80 | 0.778 | 0.780 | 0.644 |

#### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 458 | 432 | 18577 | 34763 | 80.5× | 10.9% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 458 | 417 | 19205 | 35179 | 84.3× | 10.6% |
| candidate | all scored leads | 9311 | 363 | 17790 | 45040 | 124.1× | 11.4% |
| status quo | test set | 458 | 96 | 1440 | 1872 | 19.5× | 6.4% |
| status quo | all scored leads | 9311 | 96 | 1440 | 1872 | 19.5× | 6.7% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

#### Notes

- status quo: 37 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.398.
- status quo: 37 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.506.
- status quo: 37 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.506.

### Bottom-decile ghosted share

Share of leads in the 10% of each population with the lowest p that are *ghosted* (`label_source`: mature and not contacted within 120 days) or *still New* at 2026-09-24 whatever their age (the definition behind the plan's baseline figure of 27%). The horizon test set (b) excludes ghosted leads, so the comparison uses populations that keep them.

| population | n | ghosted: overall | ghosted: baseline bottom decile | ghosted: candidate bottom decile | still New: overall | still New: baseline bottom decile | still New: candidate bottom decile |
|---|---|---|---|---|---|---|---|
| all leads labelled by legacy rules (train + test) | 8041 | 15.0% | 24.5% | 24.0% | 16.5% | 27.5% | 27.1% |
| (a) legacy test set | 2453 | 5.7% | 9.4% | 9.4% | 10.7% | 19.2% | 19.2% |
| all mature leads, every label_source | 6278 | 19.2% | 30.6% | 30.6% | 19.2% | 30.6% | 30.6% |
| mature leads created on or after 2026-05-01, every label_source | 650 | 21.7% | 29.2% | 29.2% | 21.7% | 29.2% | 29.2% |

### Design collinearity (plan 2.7)

Candidate design (`v2` features, 38 columns) on its 5588 training rows; fails when two columns have |corr| > 0.95.

- PASS: max |corr| = 0.782 between enrichment_missing=yes and band=missing
- Dropped before the check as identical on the training rows to an earlier column (the earlier column carries the shared weight):
  - spend=missing = enrichment_missing=yes
  - crm=missing = enrichment_missing=yes
  - hiring=missing = enrichment_missing=yes
  - time_on_page=missing = channel=meta_leadads
  - hesitation_90s=missing = channel=meta_leadads
  - sessions_3plus=missing = channel=meta_leadads
  - viewed_pricing=missing = channel=meta_leadads
  - business_hours=missing = channel=meta_leadads
  - ip_country=missing = channel=meta_leadads

### Baseline script output

- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, which fills blank deal values with its predicted value, printed:

```
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795
```

---

## Output: `python -m emva.eval.phase2_study`

## Phase 2 study

Data: `data/v1`. Missing level name: `missing`.

### 2.3 Company-name enrichment

Exact match after normalisation (lower-case, dots/apostrophes dropped, other punctuation to spaces, whitespace collapsed, trailing legal suffixes stripped); no fuzzy matching. The typed `historical_leads.company_name` is tried first, then the `company` form answer. Names shared by two companies after normalisation would be skipped (none on this data).

| quantity | value |
|---|---|
| cleaned leads | 9311 |
| email-domain misses (no companies.csv row by domain) | 2904 |
| ... with a typed company name (company_name or the company answer) | 1317 (45.4%) |
| ... matched by exact normalised name | 948 (32.6%) |
| ... of which the historical_leads.company_name column matched | 948 |
| still without enrichment (enrichment_missing=yes) | 1956 (21.0%) |
| truth: matched company's employee band = generator's band (ground truth) | 948 (100.0%) |
| truth: matched company's sector = generator's sector (ground truth) | 948 (100.0%) |
| truth: generator has_company among the matched | 948 (100.0%) |
| truth: generator has_company among unmatched domain misses | 788 (40.3%) |

Most common unmatched typed names among domain misses: (blank) (1721), Freelance (118), Self-employed (117).

### 2.5 Boilerplate detector

Cleaned leads (9311); truth = the generator's `text_category == copy_paste` (ground truth, evaluation only). Similarity = token-set Jaccard with the closest of the snippets in `emva/boilerplate.py`.

| detector | flagged | true copy_paste | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|---|
| v2 similarity (Jaccard >= 0.6) | 807 | 807 | 807 | 0 | 0 | 1.000 | 1.000 |
| legacy prefix rule | 807 | 807 | 807 | 0 | 0 | 1.000 | 1.000 |

Similarity margin: lowest similarity among true copy_paste texts 1.000; highest among all other texts 0.179. Any threshold between the two gives the same result on v1.

### 2.1 Meta lead-ads leads: behavioural buckets

932 cleaned Meta lead-ads leads. They have no on-site session, so every behavioural input is blank. Legacy puts them in each feature's reference level; v2 puts them in `missing`.

| feature | feature set | buckets (lead-ads leads) |
|---|---|---|
| time_on_page | legacy | 15-60s: 932 |
| time_on_page | v2 | missing: 932 |
| hesitation_90s | legacy | no: 932 |
| hesitation_90s | v2 | missing: 932 |
| sessions_3plus | legacy | no: 932 |
| sessions_3plus | v2 | missing: 932 |
| viewed_pricing | legacy | no: 932 |
| viewed_pricing | v2 | missing: 932 |
| ip_country | legacy | match: 932 |
| ip_country | v2 | missing: 932 |
| ip_type | legacy | residential: 932 |
| ip_type | v2 | missing: 932 |
| business_hours | legacy | outside: 932 |
| business_hours | v2 | missing: 932 |

### Weights before and after

Log-odds per design column. Empty = the column does not exist in that design; `= X` = dropped as identical to column X on the training rows (X carries the shared weight). The baseline has `no_company`; v2 has `enrichment_missing` and `missing` levels, and drops c_budget, c_timeline, ip_type and edits_1_4 (2.6).

| column | baseline (weights.csv) | legacy features, horizon labels | v2, legacy labels | v2, horizon labels (default) |
|---|---|---|---|---|
| band=1000+ | 0.703 | 0.419 | 0.675 | 0.393 |
| band=11-50 | 0.193 | 0.235 | 0.202 | 0.268 |
| band=201-1000 | 0.873 | 0.971 | 0.809 | 0.928 |
| band=51-200 | 1.078 | 1.146 | 1.092 | 1.170 |
| band=missing |  |  | 0.058 | -0.106 |
| business_hours=missing |  |  | = channel=meta_leadads | = channel=meta_leadads |
| business_hours=wkday_9-18 | 0.160 | 0.071 | 0.166 | 0.079 |
| c_budget=Under £5k | 0.122 | 0.038 |  |  |
| c_budget=£20k-£50k | 0.152 | 0.110 |  |  |
| c_budget=£50k+ | 0.149 | 0.057 |  |  |
| c_budget=£5k-£20k | -0.082 | 0.037 |  |  |
| c_timeline=Just researching | 0.126 | 0.291 |  |  |
| c_timeline=Next 6 months | 0.019 | -0.095 |  |  |
| c_timeline=This month | 0.002 | -0.074 |  |  |
| c_timeline=This quarter | 0.195 | 0.120 |  |  |
| channel=chatgpt | 0.053 | 0.060 | 0.049 | 0.052 |
| channel=linkedin | 0.226 | 0.229 | 0.226 | 0.226 |
| channel=meta | -0.259 | -0.315 | -0.262 | -0.323 |
| channel=meta_leadads | -0.489 | -0.632 | -0.479 | -0.622 |
| channel=organic_direct | 0.160 | 0.208 | 0.157 | 0.215 |
| crm=hubspot_sf | 0.131 | 0.135 | 0.121 | 0.098 |
| crm=missing |  |  | = enrichment_missing=yes | = enrichment_missing=yes |
| edits_1_4=yes | 0.050 | -0.006 |  |  |
| email=free | -0.391 | -0.394 | -1.097 | -1.102 |
| enrichment_missing=yes |  |  | 0.088 | 0.172 |
| form_variant=B | 0.213 | 0.106 | 0.195 | 0.048 |
| form_variant=C | 0.342 | 0.243 | 0.507 | 0.341 |
| form_variant=D | -0.381 | -0.324 | -0.441 | -0.424 |
| hesitation_90s=missing |  |  | = channel=meta_leadads | = channel=meta_leadads |
| hesitation_90s=yes | -0.312 | -0.282 | -0.290 | -0.261 |
| hiring=hiring | 0.360 | 0.372 | 0.364 | 0.379 |
| hiring=missing |  |  | = enrichment_missing=yes | = enrichment_missing=yes |
| ip_country=mismatch | -0.331 | -0.309 | -0.305 | -0.230 |
| ip_country=missing |  |  | = channel=meta_leadads | = channel=meta_leadads |
| ip_type=dc | 0.022 | 0.161 |  |  |
| no_company=yes | -0.391 | -0.394 |  |  |
| search_term=brand | 0.155 | 0.101 | 0.166 | 0.100 |
| search_term=not_google | -0.310 | -0.452 | -0.309 | -0.452 |
| seniority=mid | 0.077 | 0.112 | 0.082 | 0.116 |
| seniority=not_asked/blank | -0.277 | -0.145 | -0.278 | -0.177 |
| seniority=senior | 0.481 | 0.307 | 0.489 | 0.304 |
| seniority=student | -1.471 | -1.462 | -1.458 | -1.493 |
| sessions_3plus=missing |  |  | = channel=meta_leadads | = channel=meta_leadads |
| sessions_3plus=yes | 0.406 | 0.400 | 0.401 | 0.394 |
| spend=missing |  |  | = enrichment_missing=yes | = enrichment_missing=yes |
| spend=none | -0.571 | -0.576 | -0.588 | -0.601 |
| spend=£100k+ | 0.429 | 0.585 | 0.521 | 0.675 |
| spend=£25k-£100k | 0.484 | 0.461 | 0.521 | 0.475 |
| spend=£5k-£25k | 0.246 | 0.190 | 0.260 | 0.195 |
| text=copy_paste | -0.950 | -1.004 | -0.939 | -1.000 |
| text=specific | 0.507 | 0.606 | 0.514 | 0.596 |
| text=vague | -0.899 | -0.823 | -0.892 | -0.819 |
| time_on_page=300-600s | -0.090 | 0.007 | -0.096 | 0.007 |
| time_on_page=60-300s | 0.508 | 0.587 | 0.515 | 0.587 |
| time_on_page=>600s | -0.214 | -0.171 | -0.213 | -0.144 |
| time_on_page=missing |  |  | = channel=meta_leadads | = channel=meta_leadads |
| viewed_pricing=missing |  |  | = channel=meta_leadads | = channel=meta_leadads |
| viewed_pricing=yes | 0.229 | 0.229 | 0.238 | 0.236 |

#### Coefficients equal to three decimals

- baseline (weights.csv): email=free = no_company=yes (-0.391; corr +1.000); business_hours=wkday_9-18 = channel=organic_direct (0.160; corr +0.068)
- legacy features, horizon labels: email=free = no_company=yes (-0.394; corr +1.000); viewed_pricing=yes = channel=linkedin (0.229; corr +0.027)
- v2, legacy labels: search_term=brand = business_hours=wkday_9-18 (0.166; corr +0.059); spend=£25k-£100k = spend=£100k+ (0.521; corr -0.081)
- v2, horizon labels (default): none

### 2.7 Collinearity

Threshold |corr| > 0.95 on the training rows of each model's design.

| model | design columns | training rows | max |corr| | pair | check | aliases dropped |
|---|---|---|---|---|---|---|
| legacy features, legacy labels (baseline) | 47 | 5588 | 1.000 | email=free ~ no_company=yes | FAIL | 0 |
| legacy features, horizon labels | 47 | 4049 | 1.000 | email=free ~ no_company=yes | FAIL | 0 |
| v2, legacy labels | 38 | 5588 | 0.782 | enrichment_missing=yes ~ band=missing | pass | 9 |
| v2, horizon labels (default) | 38 | 4049 | 0.772 | enrichment_missing=yes ~ band=missing | pass | 9 |

### Feature change alone: v2 vs legacy features, same labels

Paired bootstrap (1000 resamples, seed 0) of AUC(v2) − AUC(legacy features), both models trained on the same label definition and scored on the same frozen test rows.

| labels | test set | AUC legacy features | AUC v2 features | v2 − legacy [95% CI] | bootstrap p |
|---|---|---|---|---|---|
| legacy | (a) legacy test set (n=2453) | 0.814 | 0.816 | +0.002 [-0.001, +0.006] | 0.292 |
| legacy | (b) mature test set (n=458) | 0.778 | 0.780 | +0.002 [-0.005, +0.011] | 0.608 |
| horizon H=120 | (a) legacy test set (n=2453) | 0.812 | 0.814 | +0.001 [-0.002, +0.005] | 0.584 |
| horizon H=120 | (b) mature test set (n=458) | 0.778 | 0.778 | +0.001 [-0.007, +0.009] | 0.912 |

### 2.4 Deal-value model: residual sd

The correction is exp(sd²/2). It multiplies every lead's expected deal value by the same factor, so it cannot change any ranking: top-20% capture by p×value is identical by construction. It changes the level of the value sent, checked here against recorded deal values of won test leads. Revenue (a)/(b) = recorded deal value on the frozen legacy / mature test sets; the baseline-style figure fills blank deal values with the model's own prediction (so it can move slightly).

| model | residual sd | sd | correction | training deals | top-20% revenue by p×value (a) | top-20% revenue by p×value (b) | mean predicted / mean recorded deal value, won test leads | baseline-style top20_revenue (imputes blanks) |
|---|---|---|---|---|---|---|---|---|
| v2, legacy labels | fixed (baseline constant) | 0.4500 | 1.1066 | 770 | 0.8014 | 0.7519 | 1.003 | 0.8000 |
| v2, legacy labels | estimated from training residuals | 0.4546 | 1.1089 | 770 | 0.8014 | 0.7519 | 1.005 | 0.8000 |
| v2, horizon labels (default) | fixed (baseline constant) | 0.4500 | 1.1066 | 770 | 0.8163 | 0.7742 | 0.968 | 0.7652 |
| v2, horizon labels (default) | estimated from training residuals | 0.4546 | 1.1089 | 770 | 0.8163 | 0.7742 | 0.970 | 0.7652 |


---

## Output: `python -m emva.eval.feature_selection --n-refits 1000` (2.6 Feature selection)

### Plan 2.6: feature selection by coefficient bootstrap

Data `data/v1`, labels `horizon H=120`, v2 design with every candidate feature, training rows (created before 2026-05-01). 95% percentile CI from 1000 refits on resampled training rows (seed 0).

| feature | level | log-odds [95% CI] | excludes 0 | n train |
|---|---|---|---|---|
| c_budget | c_budget=Under £5k | 0.067 [-0.306, 0.411] | no | 169 |
| c_budget | c_budget=£20k-£50k | 0.098 [-0.274, 0.480] | no | 82 |
| c_budget | c_budget=£50k+ | 0.013 [-0.470, 0.466] | no | 36 |
| c_budget | c_budget=£5k-£20k | 0.052 [-0.312, 0.427] | no | 120 |
| c_timeline | c_timeline=Just researching | 0.285 [-0.097, 0.677] | no | 121 |
| c_timeline | c_timeline=Next 6 months | -0.088 [-0.490, 0.343] | no | 101 |
| c_timeline | c_timeline=This month | -0.100 [-0.648, 0.431] | no | 41 |
| c_timeline | c_timeline=This quarter | 0.133 [-0.214, 0.478] | no | 144 |
| ip_type | ip_type=dc | 0.188 [-0.489, 0.739] | no | 104 |
| ip_type | ip_type=missing | alias of channel=meta_leadads | n/a | 388 |
| edits_1_4 | edits_1_4=yes | -0.004 [-0.219, 0.246] | no | 2933 |

| feature | decision | reason |
|---|---|---|
| c_budget | drop | every level's CI contains 0 |
| c_timeline | drop | every level's CI contains 0 |
| ip_type | drop | every level's CI contains 0 |
| edits_1_4 | drop | every level's CI contains 0 |

