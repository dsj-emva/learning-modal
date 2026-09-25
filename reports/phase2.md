# Phase 2: features and leakage

Branch `phase2-features` from `origin/main` 92168cf. Plan items 2.1 to 2.7, then the orchestrator's
indicator-encoding ruling. `--feature-set {legacy,v2}` (default `v2`) sits next to `--label-mode`;
`--label-mode legacy --feature-set legacy` is still byte-identical to `baseline/` (`make baseline`
passes; weights.csv and scores.csv byte-identical).

v2 numbers use `data/v2` from `origin/phase5-generator-v2` at ae5b7c6 (the data files come from
5c32dd8, "data/v2: generate_data_v2.py, seed 20260924, n 10000, with paraphrase"), extracted with
`git archive`; it is not on this branch.

## Orchestrator decisions (review of Phase 2)

- **Deviation 1 (pruning identical design columns per dataset): not accepted.** The model's
  column set must be fixed by the feature spec (single-row scoring and a hosted app need a stable
  schema). Replaced by explicit indicators:
  - `session_missing` (yes/no): yes when the on-site session is absent (blank `landing_url`, or any
    of the behavioural inputs blank). The seven behavioural features (`time_on_page`,
    `hesitation_90s`, `sessions_3plus`, `viewed_pricing`, `ip_country`, `ip_type`, `business_hours`)
    then take their reference level and the indicator carries the effect. Their individual `missing`
    levels are gone.
  - `enrichment_missing` (yes/no) as before; `spend`, `crm`, `hiring` take their reference level
    without enrichment. `band` keeps its own `missing` level (the typed company-size fallback makes
    it distinct from the indicator).
  - No automatic column dropping anywhere. The 2.7 check stays strict and runs in the report.
- **Deviation 2** (the legacy residual-sd constant in `emva/eval/regression.py`, imported by the
  pipeline for `--feature-set legacy` only): accepted; the import site now carries a one-line comment.
- **Deviations 3 to 8:** accepted.

What the ruling means on v1: every session-less v1 lead is a Meta lead-ads lead (932 of 932, and no
other lead lacks telemetry), so `session_missing=yes` and `channel=meta_leadads` are the same column.
**The collinearity check therefore fails on v1 for the v2 feature set, on exactly that one pair
(|corr| = 1.000), and `make report` / `python -m emva.eval.report` exit 1 on v1.** This is a property
of the data, not of the feature design. On data/v2, 1,278 consent-declined website leads also have
`session_missing=yes`, the two columns separate, and the check passes (max |corr| 0.820). The legacy
feature set's failure (`email=free` = `no_company=yes`) is a design flaw: v2 fixes it on both datasets.

## Acceptance

| criterion | v1 | v2 data (ae5b7c6) |
|---|---|---|
| No two coefficients identical to 3 dp | **fail**, on the one structural pair above: `channel=meta_leadads` = `session_missing=yes` = −0.345 (horizon) / −0.273 (legacy labels). The columns are identical, so the L2 penalty splits one effect equally between them (sum −0.690 / −0.546). Also one coincidental tie in the default model: `hiring=hiring` = `band=1000+` = 0.382 (corr +0.217). The baseline's `email=free` = `no_company=yes` tie is gone. | **pass**: no equal pair in v2 + horizon or v2 + legacy labels |
| Lead-ads leads no longer in the reference bucket for every behavioural feature | **changed by the ruling, pass as re-specified**: the 932 lead-ads leads sit in the reference level of all seven behavioural features *and* carry `session_missing=yes` (intended encoding). `session_missing=yes` weight −0.345 (horizon) / −0.273 (legacy labels), but on v1 it is inseparable from the lead-ads channel weight (same column). | 872 lead-ads leads plus 1,278 consent-declined website leads have `session_missing=yes`; weight **+0.632** (horizon) / +0.559 (legacy labels), now separately identified from `channel=meta_leadads` (−0.933 / −0.762) |
| AUC within the baseline CI [0.789, 0.836], legacy test set | **pass**: v2 + legacy labels 0.816 [0.792, 0.838], paired vs baseline +0.002 [−0.002, +0.006]; v2 + horizon 0.814 [0.790, 0.837], paired +0.000 [−0.006, +0.006] | v2 features 0.802 (legacy labels) / 0.803 (horizon) vs legacy features 0.785, paired +0.018 [+0.010, +0.026] / +0.019 [+0.011, +0.027] |
| AUC on the frozen mature test set (R3) | reported: v2 + legacy labels 0.781 [0.730, 0.828], paired +0.002 [−0.005, +0.011]; v2 + horizon 0.778 [0.730, 0.827], paired +0.000 [−0.015, +0.015] | v2 + horizon 0.805 vs legacy features 0.791, paired +0.014 [−0.002, +0.031]; v2 + legacy labels +0.017 [+0.001, +0.036] |
| 2.7 collinearity (strict) | v2 **FAIL** on `channel=meta_leadads` ~ `session_missing=yes` only (data property); legacy FAIL on `email=free` ~ `no_company=yes` | v2 **PASS**, max 0.820 (`enrichment_missing=yes` ~ `band=missing`); legacy passes too (0.690) because v2 data breaks the free-email / no-company identity |
| `grep -rn "0.45\|ground_truth" emva/` only `emva/eval/` | **pass** (test-enforced) | n/a |

On v2 data the v2 features beat legacy features by +0.014 to +0.019 AUC; on v1 they are level.

## Indicator weights

| weight (log-odds) | v1, legacy labels | v1, horizon (default) | v2 data, legacy labels | v2 data, horizon |
|---|---|---|---|---|
| session_missing=yes | −0.273 | −0.345 | +0.559 | +0.632 |
| channel=meta_leadads | −0.273 | −0.345 | −0.762 | −0.933 |
| enrichment_missing=yes | +0.091 | +0.168 | +0.018 | +0.039 |
| band=missing | +0.041 | −0.084 | +1.176 | +1.128 |
| email=free | −1.090 | −1.115 | −1.390 | −1.500 |

On v1 the first two rows are one effect split in half (see above). On v2 data the positive
`session_missing` and the large positive `band=missing` weights are what the fitted model says; I
have not checked them against the v2 generator's planted effects (Phase 5's ground truth) and flag
them for that review.

## Other required numbers (v1 unless stated)

- **2.3 match count:** of the **2,904** cleaned domain-miss leads, **948 (32.6%)** match a
  companies.csv name exactly after normalisation; ground truth confirms 948/948. On v2 data: 2,079 of
  4,772 (43.6%), again all correct. Exact match only, no fuzzy matching.
- **2.5 boilerplate:** Jaccard ≥ **0.6**, 16 snippets. v1: precision 1.000, recall 1.000 (807/807).
  **v2 data: precision 1.000, recall 0.065 (53 of 817)**; the legacy prefix rule gets 0.016. The
  paraphrased v2 boilerplate is not close to any snippet by token overlap. The detector is no longer a
  template match, but it is not good enough for v2 text; that is Phase 6's context layer (or a
  better snippet list / embedding similarity).
- **2.6:** re-run after the ruling (1,000 refits, horizon labels): c_budget, c_timeline, ip_type,
  edits_1_4 all still dropped; no level's CI excludes zero. `ip_type` now has only its `dc` level.
- **2.7 max |corr|:** see acceptance; full CLI output in "Collinearity and pipeline runs" below.
- **2.4 value model:** fixed sd 0.45 (correction 1.1066) vs estimated **0.4546** (1.1089) on v1;
  on v2 data the estimated sd is **0.5516** (1.1643), so the fixed constant would under-state v2
  deal values by 5%. Top-20% capture by p×value is unchanged by construction (a constant factor).
  Mean predicted / mean recorded deal value on won test leads: v1 0.968 → 0.970, v2 data 0.899 → 0.945.

## Before/after weights for changed features (v1)

| column | baseline | v2, legacy labels | v2, horizon (default) |
|---|---|---|---|
| no_company=yes | −0.391 | (removed) | (removed) |
| enrichment_missing=yes | | 0.091 | 0.168 |
| session_missing=yes | | −0.273 | −0.345 |
| channel=meta_leadads | −0.489 | −0.273 | −0.345 |
| email=free | −0.391 | −1.090 | −1.115 |
| band=11-50 / 51-200 / 201-1000 / 1000+ | .193 / 1.078 / .873 / .703 | .199 / 1.089 / .805 / .672 | .257 / 1.160 / .919 / .382 |
| band=missing | | 0.041 | −0.084 |
| spend=none / £5k-£25k / £25k-£100k / £100k+ | −.571 / .246 / .484 / .429 | −.590 / .261 / .521 / .523 | −.604 / .194 / .471 / .667 |
| crm=hubspot_sf | 0.131 | 0.120 | 0.099 |
| hiring=hiring | 0.360 | 0.363 | 0.382 |
| time_on_page=60-300s / 300-600s / >600s | .508 / −.090 / −.214 | .512 / −.098 / −.218 | .583 / .000 / −.149 |
| hesitation_90s=yes | −0.312 | −0.292 | −0.251 |
| sessions_3plus=yes | 0.406 | 0.399 | 0.393 |
| viewed_pricing=yes | 0.229 | 0.238 | 0.236 |
| ip_country=mismatch | −0.331 | −0.304 | −0.226 |
| business_hours=wkday_9-18 | 0.160 | 0.164 | 0.075 |
| text=copy_paste | −0.950 | −0.936 | −0.983 |
| c_budget, c_timeline, ip_type=dc, edits_1_4=yes | fitted | dropped (2.6) | dropped |

## What was done

- **Switch.** `emva.features.FeatureSet`, `--feature-set` for `python -m emva`, `emva.eval.report`,
  `emva.eval.collinearity`. The legacy path is intact; `make baseline` runs
  `--label-mode legacy --feature-set legacy`. Phase 1 study scripts and `scripts/compare_to_v1.py`
  are pinned to legacy features.
- **2.1** (as ruled) `session_missing` from `emva.features.session_absent`; behavioural features at
  reference when it is yes; no channel rule in `business_hours`.
- **2.2** (as ruled) `enrichment_missing` replaces `no_company`; `spend`/`crm`/`hiring` at reference
  without enrichment; `band` = enrichment → typed size → `missing`.
- **2.3** `emva.io.normalise_company_name` / `match_company_names`: exact match after lower-casing,
  punctuation handling and trailing legal-suffix stripping (the plan's list plus `sas`); ambiguous
  names skipped.
- **2.4** `emva.value.DealValueModel`: sd = sample sd of the in-sample training residuals of the
  log-value ridge; correction exp(sd²/2).
- **2.5** `emva/boilerplate.py`, token-set Jaccard, threshold 0.6 (configurable).
- **2.6** `python -m emva.eval.feature_selection`, outcome hard-coded in `emva.features.V2_DROPPED`.
- **2.7** `emva/eval/collinearity.py`; the report prints every pair above 0.95 and exits 1.

## Deviations, not done, open questions

1. `make report` exits 1 on v1 with the default feature set (the ruling's intended behaviour; the
   report text is still printed in full). Anything scripting `make report` on v1 will see the failure.
2. `design()` still builds dummies from the levels present in the data (baseline behaviour): a level
   that never occurs produces no column. The ruling removed the data-dependent *pruning*; making the
   level list itself fixed (declared levels per feature) is not done and would be the next step for a
   stable scoring schema.
3. The coincidental 3-dp tie `hiring=hiring` = `band=1000+` in the default v1 model is reported, not
   engineered away.
4. Boilerplate recall on v2 data is 6.5% (above).
5. The v2-data weights for `session_missing` and `band=missing` are unexplained here (above).
6. Accepted earlier: legacy constant location, `sas` suffix, 948 matches, no missing level for the
   dropped `edits_1_4`, intermediate commits not individually green, one-line changes in
   `scripts/compare_to_v1.py` and `tests/test_generate_data_v1.py`.

## Commands

```
make PY=.venv/bin/python baseline
python -m emva.eval.report                       # (1) v2 features, horizon labels (default); exits 1 on v1
python -m emva.eval.report --label-mode legacy   # (2) v2 features, legacy labels; exits 1 on v1
python -m emva.eval.phase2_study [--data DIR]
python -m emva.eval.feature_selection --n-refits 1000
python -m emva.eval.collinearity [--data DIR] [--feature-set legacy] [--label-mode legacy]
```

The outputs follow, verbatim (v2 data path shortened to `data/v2`).


---

## Output: Collinearity and pipeline runs, v1 and v2 data

### data/v1: `--label-mode horizon --feature-set v2`

```
$ python -m emva --label-mode horizon --feature-set v2
  model   auc  brier  top20_wins  top20_revenue
formula 0.778 0.1265        0.45          0.765
$ python -m emva.eval.collinearity --label-mode horizon --feature-set v2
feature set v2, labels horizon H=120, 4049 training rows, 39 design columns
- FAIL (1 pairs with |corr| > 0.95): max |corr| = 1.000 between channel=meta_leadads and session_missing=yes
  - channel=meta_leadads ~ session_missing=yes: +1.000
(exit 1)
```

### data/v1: `--label-mode horizon --feature-set legacy`

```
$ python -m emva --label-mode horizon --feature-set legacy
  model   auc  brier  top20_wins  top20_revenue
formula 0.778 0.1263       0.462          0.762
$ python -m emva.eval.collinearity --label-mode horizon --feature-set legacy
feature set legacy, labels horizon H=120, 4049 training rows, 47 design columns
- FAIL (1 pairs with |corr| > 0.95): max |corr| = 1.000 between email=free and no_company=yes
  - email=free ~ no_company=yes: +1.000
(exit 1)
```

### data/v1: `--label-mode legacy --feature-set v2`

```
$ python -m emva --label-mode legacy --feature-set v2
  model   auc  brier  top20_wins  top20_revenue
formula 0.816 0.1004       0.585            0.8
$ python -m emva.eval.collinearity --label-mode legacy --feature-set v2
feature set v2, labels legacy, 5588 training rows, 39 design columns
- FAIL (1 pairs with |corr| > 0.95): max |corr| = 1.000 between channel=meta_leadads and session_missing=yes
  - channel=meta_leadads ~ session_missing=yes: +1.000
(exit 1)
```

### data/v1: `--label-mode legacy --feature-set legacy`

```
$ python -m emva --label-mode legacy --feature-set legacy
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795
$ python -m emva.eval.collinearity --label-mode legacy --feature-set legacy
feature set legacy, labels legacy, 5588 training rows, 47 design columns
- FAIL (1 pairs with |corr| > 0.95): max |corr| = 1.000 between email=free and no_company=yes
  - email=free ~ no_company=yes: +1.000
(exit 1)
```

### data/v2 (origin/phase5-generator-v2 @ ae5b7c6): `--label-mode horizon --feature-set v2`

```
$ python -m emva --label-mode horizon --feature-set v2
  model   auc  brier  top20_wins  top20_revenue
formula 0.805 0.1138       0.493          0.712
$ python -m emva.eval.collinearity --label-mode horizon --feature-set v2
feature set v2, labels horizon H=120, 3787 training rows, 39 design columns
- PASS: max |corr| = 0.820 between enrichment_missing=yes and band=missing
(exit 0)
```

### data/v2 (origin/phase5-generator-v2 @ ae5b7c6): `--label-mode horizon --feature-set legacy`

```
$ python -m emva --label-mode horizon --feature-set legacy
  model   auc  brier  top20_wins  top20_revenue
formula 0.791 0.1164       0.493          0.657
$ python -m emva.eval.collinearity --label-mode horizon --feature-set legacy
feature set legacy, labels horizon H=120, 3787 training rows, 47 design columns
- PASS: max |corr| = 0.690 between ip_country=mismatch and ip_type=dc
(exit 0)
```

### data/v2 (origin/phase5-generator-v2 @ ae5b7c6): `--label-mode legacy --feature-set v2`

```
$ python -m emva --label-mode legacy --feature-set v2
  model   auc  brier  top20_wins  top20_revenue
formula 0.802 0.0999       0.541           0.79
$ python -m emva.eval.collinearity --label-mode legacy --feature-set v2
feature set v2, labels legacy, 5507 training rows, 39 design columns
- PASS: max |corr| = 0.814 between enrichment_missing=yes and band=missing
(exit 0)
```

### data/v2 (origin/phase5-generator-v2 @ ae5b7c6): `--label-mode legacy --feature-set legacy`

```
$ python -m emva --label-mode legacy --feature-set legacy
  model   auc  brier  top20_wins  top20_revenue
formula 0.785 0.1025       0.529          0.728
$ python -m emva.eval.collinearity --label-mode legacy --feature-set legacy
feature set legacy, labels legacy, 5507 training rows, 47 design columns
- PASS: max |corr| = 0.693 between ip_country=mismatch and ip_type=dc
(exit 0)
```


---

## Output: `python -m emva.eval.report` on v1 (v2 features, horizon labels; exit 1, collinearity)

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
| candidate | 0.814 [0.790, 0.837] | 0.1020 | 0.571 | 0.733 | 0.815 |
| status quo | 0.639 [0.605, 0.673] | n/a | 0.392 | 0.454 | 0.454 |

#### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.000 | [-0.006, +0.006] | 0.972 |
| status quo | -0.174 | [-0.209, -0.141] | 0.000 |

#### Calibration by decile of p

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

#### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 754 | 103 | 0.800 | 0.801 | 0.641 |
| 2026-06 | 665 | 93 | 0.797 | 0.806 | 0.631 |
| 2026-07 | 541 | 88 | 0.833 | 0.826 | 0.602 |
| 2026-08 | 371 | 55 | 0.813 | 0.812 | 0.664 |
| 2026-09 | 122 | 13 | 0.920 | 0.914 | 0.772 |

#### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 2453 | 346 | 17167 | 45562 | 131.8× | 11.8% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 2453 | 414 | 17889 | 44625 | 107.8× | 10.9% |
| candidate | all scored leads | 9311 | 446 | 18646 | 44625 | 100.0× | 10.6% |
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
| candidate | +0.000 | [-0.015, +0.015] | 0.974 |
| status quo | -0.134 | [-0.199, -0.069] | 0.000 |

#### Calibration by decile of p

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

#### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 458 | 80 | 0.778 | 0.778 | 0.644 |

#### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 458 | 432 | 18577 | 34763 | 80.5× | 10.9% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 458 | 518 | 20121 | 32268 | 62.3× | 9.7% |
| candidate | all scored leads | 9311 | 446 | 18646 | 44625 | 100.0× | 10.6% |
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
| (a) legacy test set | 2453 | 5.7% | 9.4% | 9.0% | 10.7% | 19.2% | 19.2% |
| all mature leads, every label_source | 6278 | 19.2% | 30.6% | 31.3% | 19.2% | 30.6% | 31.3% |
| mature leads created on or after 2026-05-01, every label_source | 650 | 21.7% | 29.2% | 27.7% | 21.7% | 29.2% | 27.7% |

### Design collinearity (plan 2.7)

Candidate design (`v2` features, 39 columns) on its 4049 training rows; fails when two columns have |corr| > 0.95.

- FAIL (1 pairs with |corr| > 0.95): max |corr| = 1.000 between channel=meta_leadads and session_missing=yes
  - channel=meta_leadads ~ session_missing=yes: +1.000

### Baseline script output

- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, which fills blank deal values with its predicted value, printed:

```
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795
```

---

## Output: `python -m emva.eval.report --label-mode legacy` on v1 (v2 features, legacy labels; exit 1)

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
| candidate | 0.816 [0.792, 0.838] | 0.1004 | 0.585 | 0.751 | 0.801 |
| status quo | 0.639 [0.605, 0.673] | n/a | 0.392 | 0.454 | 0.454 |

#### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.002 | [-0.002, +0.006] | 0.282 |
| status quo | -0.174 | [-0.209, -0.141] | 0.000 |

#### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 246 | 0.009 | 0.016 | 0.008 | 0.016 |
| 2 | 245 | 0.020 | 0.012 | 0.018 | 0.008 |
| 3 | 245 | 0.033 | 0.016 | 0.032 | 0.024 |
| 4 | 245 | 0.050 | 0.053 | 0.049 | 0.049 |
| 5 | 246 | 0.075 | 0.077 | 0.075 | 0.085 |
| 6 | 245 | 0.109 | 0.135 | 0.108 | 0.114 |
| 7 | 245 | 0.150 | 0.131 | 0.150 | 0.131 |
| 8 | 245 | 0.213 | 0.176 | 0.217 | 0.167 |
| 9 | 245 | 0.316 | 0.327 | 0.319 | 0.347 |
| 10 | 246 | 0.511 | 0.492 | 0.512 | 0.492 |

Status quo has no probability, so it has no Brier score or calibration.

#### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 754 | 103 | 0.800 | 0.802 | 0.641 |
| 2026-06 | 665 | 93 | 0.797 | 0.805 | 0.631 |
| 2026-07 | 541 | 88 | 0.833 | 0.831 | 0.602 |
| 2026-08 | 371 | 55 | 0.813 | 0.809 | 0.664 |
| 2026-09 | 122 | 13 | 0.920 | 0.915 | 0.772 |

#### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 2453 | 346 | 17167 | 45562 | 131.8× | 11.8% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 2453 | 335 | 17202 | 45108 | 134.8× | 11.6% |
| candidate | all scored leads | 9311 | 363 | 17820 | 45108 | 124.4× | 11.4% |
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
| candidate | 0.781 [0.730, 0.828] | 0.1257 | 0.500 | 0.677 | 0.752 |
| status quo | 0.644 [0.571, 0.717] | n/a | 0.400 | 0.501 | 0.501 |

#### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.002 | [-0.005, +0.011] | 0.540 |
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
| 8 | 46 | 0.237 | 0.239 | 0.243 | 0.196 |
| 9 | 46 | 0.357 | 0.391 | 0.354 | 0.413 |
| 10 | 46 | 0.555 | 0.457 | 0.555 | 0.457 |

Status quo has no probability, so it has no Brier score or calibration.

#### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 458 | 80 | 0.778 | 0.781 | 0.644 |

#### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 458 | 432 | 18577 | 34763 | 80.5× | 10.9% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 458 | 418 | 19228 | 35169 | 84.0× | 10.6% |
| candidate | all scored leads | 9311 | 363 | 17820 | 45108 | 124.4× | 11.4% |
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

Candidate design (`v2` features, 39 columns) on its 5588 training rows; fails when two columns have |corr| > 0.95.

- FAIL (1 pairs with |corr| > 0.95): max |corr| = 1.000 between channel=meta_leadads and session_missing=yes
  - channel=meta_leadads ~ session_missing=yes: +1.000

### Baseline script output

- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, which fills blank deal values with its predicted value, printed:

```
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795
```

---

## Output: `python -m emva.eval.phase2_study` (v1)

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

Similarity margin: lowest similarity among true copy_paste texts 1.000; highest among all other texts 0.179. Any threshold between the two gives the same result on this data.

### 2.1 Meta lead-ads leads: behavioural buckets

932 cleaned Meta lead-ads leads. They have no on-site session, so every behavioural input is blank. Legacy silently puts them in each feature's reference level. v2 also leaves the seven behavioural features at their reference level but sets `session_missing=yes`, so the indicator carries the effect of the absent session (the intended encoding: the column set is fixed by the spec). Other cleaned leads with `session_missing=yes` in this data: 0.

| feature | feature set | buckets (lead-ads leads) |
|---|---|---|
| time_on_page | legacy | 15-60s: 932 |
| time_on_page | v2 | 15-60s: 932 |
| hesitation_90s | legacy | no: 932 |
| hesitation_90s | v2 | no: 932 |
| sessions_3plus | legacy | no: 932 |
| sessions_3plus | v2 | no: 932 |
| viewed_pricing | legacy | no: 932 |
| viewed_pricing | v2 | no: 932 |
| ip_country | legacy | match: 932 |
| ip_country | v2 | match: 932 |
| ip_type | legacy | residential: 932 |
| ip_type | v2 | residential: 932 |
| business_hours | legacy | outside: 932 |
| business_hours | v2 | outside: 932 |
| session_missing | legacy | (no feature) |
| session_missing | v2 | yes: 932 |

### Weights before and after

Log-odds per design column. Empty = the column does not exist in that design. The baseline column is the frozen `baseline/weights.csv` (fitted on data/v1, whatever `--data` is). The baseline has `no_company`; v2 has the `session_missing` and `enrichment_missing` indicators and `band=missing`, and drops c_budget, c_timeline, ip_type and edits_1_4 (2.6).

| column | baseline (weights.csv) | legacy features, horizon labels | v2, legacy labels | v2, horizon labels (default) |
|---|---|---|---|---|
| band=1000+ | 0.703 | 0.419 | 0.672 | 0.382 |
| band=11-50 | 0.193 | 0.235 | 0.199 | 0.257 |
| band=201-1000 | 0.873 | 0.971 | 0.805 | 0.919 |
| band=51-200 | 1.078 | 1.146 | 1.089 | 1.160 |
| band=missing |  |  | 0.041 | -0.084 |
| business_hours=wkday_9-18 | 0.160 | 0.071 | 0.164 | 0.075 |
| c_budget=Under £5k | 0.122 | 0.038 |  |  |
| c_budget=£20k-£50k | 0.152 | 0.110 |  |  |
| c_budget=£50k+ | 0.149 | 0.057 |  |  |
| c_budget=£5k-£20k | -0.082 | 0.037 |  |  |
| c_timeline=Just researching | 0.126 | 0.291 |  |  |
| c_timeline=Next 6 months | 0.019 | -0.095 |  |  |
| c_timeline=This month | 0.002 | -0.074 |  |  |
| c_timeline=This quarter | 0.195 | 0.120 |  |  |
| channel=chatgpt | 0.053 | 0.060 | 0.009 | -0.003 |
| channel=linkedin | 0.226 | 0.229 | 0.184 | 0.170 |
| channel=meta | -0.259 | -0.315 | -0.303 | -0.377 |
| channel=meta_leadads | -0.489 | -0.632 | -0.273 | -0.345 |
| channel=organic_direct | 0.160 | 0.208 | 0.116 | 0.160 |
| crm=hubspot_sf | 0.131 | 0.135 | 0.120 | 0.099 |
| edits_1_4=yes | 0.050 | -0.006 |  |  |
| email=free | -0.391 | -0.394 | -1.090 | -1.115 |
| enrichment_missing=yes |  |  | 0.091 | 0.168 |
| form_variant=B | 0.213 | 0.106 | 0.192 | 0.084 |
| form_variant=C | 0.342 | 0.243 | 0.503 | 0.378 |
| form_variant=D | -0.381 | -0.324 | -0.443 | -0.388 |
| hesitation_90s=yes | -0.312 | -0.282 | -0.292 | -0.251 |
| hiring=hiring | 0.360 | 0.372 | 0.363 | 0.382 |
| ip_country=mismatch | -0.331 | -0.309 | -0.304 | -0.226 |
| ip_type=dc | 0.022 | 0.161 |  |  |
| no_company=yes | -0.391 | -0.394 |  |  |
| search_term=brand | 0.155 | 0.101 | 0.165 | 0.105 |
| search_term=not_google | -0.310 | -0.452 | -0.267 | -0.395 |
| seniority=mid | 0.077 | 0.112 | 0.082 | 0.119 |
| seniority=not_asked/blank | -0.277 | -0.145 | -0.279 | -0.139 |
| seniority=senior | 0.481 | 0.307 | 0.488 | 0.306 |
| seniority=student | -1.471 | -1.462 | -1.468 | -1.441 |
| session_missing=yes |  |  | -0.273 | -0.345 |
| sessions_3plus=yes | 0.406 | 0.400 | 0.399 | 0.393 |
| spend=none | -0.571 | -0.576 | -0.590 | -0.604 |
| spend=£100k+ | 0.429 | 0.585 | 0.523 | 0.667 |
| spend=£25k-£100k | 0.484 | 0.461 | 0.521 | 0.471 |
| spend=£5k-£25k | 0.246 | 0.190 | 0.261 | 0.194 |
| text=copy_paste | -0.950 | -1.004 | -0.936 | -0.983 |
| text=specific | 0.507 | 0.606 | 0.515 | 0.596 |
| text=vague | -0.899 | -0.823 | -0.892 | -0.819 |
| time_on_page=300-600s | -0.090 | 0.007 | -0.098 | 0.000 |
| time_on_page=60-300s | 0.508 | 0.587 | 0.512 | 0.583 |
| time_on_page=>600s | -0.214 | -0.171 | -0.218 | -0.149 |
| viewed_pricing=yes | 0.229 | 0.229 | 0.238 | 0.236 |

#### Coefficients equal to three decimals

- baseline (weights.csv): email=free = no_company=yes (-0.391; corr +1.000); business_hours=wkday_9-18 = channel=organic_direct (0.160; corr +0.068)
- legacy features, horizon labels: email=free = no_company=yes (-0.394; corr +1.000); viewed_pricing=yes = channel=linkedin (0.229; corr +0.027)
- v2, legacy labels: channel=meta_leadads = session_missing=yes (-0.273; corr +1.000)
- v2, horizon labels (default): channel=meta_leadads = session_missing=yes (-0.345; corr +1.000); hiring=hiring = band=1000+ (0.382; corr +0.217)

### 2.7 Collinearity

Threshold |corr| > 0.95 on the training rows of each model's design.

| model | design columns | training rows | max |corr| | check | pairs above threshold |
|---|---|---|---|---|---|
| legacy features, legacy labels (baseline) | 47 | 5588 | 1.000 | FAIL | email=free ~ no_company=yes (+1.000) |
| legacy features, horizon labels | 47 | 4049 | 1.000 | FAIL | email=free ~ no_company=yes (+1.000) |
| v2, legacy labels | 39 | 5588 | 1.000 | FAIL | channel=meta_leadads ~ session_missing=yes (+1.000) |
| v2, horizon labels (default) | 39 | 4049 | 1.000 | FAIL | channel=meta_leadads ~ session_missing=yes (+1.000) |

### Feature change alone: v2 vs legacy features, same labels

Paired bootstrap (1000 resamples, seed 0) of AUC(v2) − AUC(legacy features), both models trained on the same label definition and scored on the same frozen test rows.

| labels | test set | AUC legacy features | AUC v2 features | v2 − legacy [95% CI] | bootstrap p |
|---|---|---|---|---|---|
| legacy | (a) legacy test set (n=2453) | 0.814 | 0.816 | +0.002 [-0.002, +0.006] | 0.282 |
| legacy | (b) mature test set (n=458) | 0.778 | 0.781 | +0.002 [-0.005, +0.011] | 0.540 |
| horizon H=120 | (a) legacy test set (n=2453) | 0.812 | 0.814 | +0.001 [-0.002, +0.005] | 0.566 |
| horizon H=120 | (b) mature test set (n=458) | 0.778 | 0.778 | +0.001 [-0.006, +0.009] | 0.908 |

### 2.4 Deal-value model: residual sd

The correction is exp(sd²/2). It multiplies every lead's expected deal value by the same factor, so it cannot change any ranking: top-20% capture by p×value is identical by construction. It changes the level of the value sent, checked here against recorded deal values of won test leads. Revenue (a)/(b) = recorded deal value on the frozen legacy / mature test sets; the baseline-style figure fills blank deal values with the model's own prediction (so it can move slightly).

| model | residual sd | sd | correction | training deals | top-20% revenue by p×value (a) | top-20% revenue by p×value (b) | mean predicted / mean recorded deal value, won test leads | baseline-style top20_revenue (imputes blanks) |
|---|---|---|---|---|---|---|---|---|
| v2, legacy labels | fixed (baseline constant) | 0.4500 | 1.1066 | 770 | 0.8014 | 0.7519 | 1.003 | 0.8000 |
| v2, legacy labels | estimated from training residuals | 0.4546 | 1.1089 | 770 | 0.8014 | 0.7519 | 1.005 | 0.8000 |
| v2, horizon labels (default) | fixed (baseline constant) | 0.4500 | 1.1066 | 770 | 0.8153 | 0.7742 | 0.968 | 0.7652 |
| v2, horizon labels (default) | estimated from training residuals | 0.4546 | 1.1089 | 770 | 0.8153 | 0.7742 | 0.970 | 0.7652 |


---

## Output: `python -m emva.eval.phase2_study --data data/v2` (v2 data)

## Phase 2 study

Data: `data/v2`. Missing level name: `missing`.

### 2.3 Company-name enrichment

Exact match after normalisation (lower-case, dots/apostrophes dropped, other punctuation to spaces, whitespace collapsed, trailing legal suffixes stripped); no fuzzy matching. The typed `historical_leads.company_name` is tried first, then the `company` form answer. Names shared by two companies after normalisation would be skipped (none on this data).

| quantity | value |
|---|---|
| cleaned leads | 9212 |
| email-domain misses (no companies.csv row by domain) | 4772 |
| ... with a typed company name (company_name or the company answer) | 2496 (52.3%) |
| ... matched by exact normalised name | 2079 (43.6%) |
| ... of which the historical_leads.company_name column matched | 2079 |
| still without enrichment (enrichment_missing=yes) | 2693 (29.2%) |
| truth: matched company's employee band = generator's band (ground truth) | 2079 (100.0%) |
| truth: matched company's sector = generator's sector (ground truth) | 2079 (100.0%) |
| truth: generator has_company among the matched | 2079 (100.0%) |
| truth: generator has_company among unmatched domain misses | 1612 (59.9%) |

Most common unmatched typed names among domain misses: (blank) (2401), Self-employed (117), Freelance (109), Lindenwood Digital Incorporated (2), Brightwater Industries, Incorporated (1).

### 2.5 Boilerplate detector

Cleaned leads (9212); truth = the generator's `text_category == copy_paste` (ground truth, evaluation only). Similarity = token-set Jaccard with the closest of the snippets in `emva/boilerplate.py`.

| detector | flagged | true copy_paste | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|---|
| v2 similarity (Jaccard >= 0.6) | 53 | 817 | 53 | 0 | 764 | 1.000 | 0.065 |
| legacy prefix rule | 13 | 817 | 13 | 0 | 804 | 1.000 | 0.016 |

Similarity margin: lowest similarity among true copy_paste texts 0.000; highest among all other texts 0.214. The two overlap: no threshold separates boilerplate from other text on this data.

### 2.1 Meta lead-ads leads: behavioural buckets

872 cleaned Meta lead-ads leads. They have no on-site session, so every behavioural input is blank. Legacy silently puts them in each feature's reference level. v2 also leaves the seven behavioural features at their reference level but sets `session_missing=yes`, so the indicator carries the effect of the absent session (the intended encoding: the column set is fixed by the spec). Other cleaned leads with `session_missing=yes` in this data: 1278.

| feature | feature set | buckets (lead-ads leads) |
|---|---|---|
| time_on_page | legacy | 15-60s: 872 |
| time_on_page | v2 | 15-60s: 872 |
| hesitation_90s | legacy | no: 872 |
| hesitation_90s | v2 | no: 872 |
| sessions_3plus | legacy | no: 872 |
| sessions_3plus | v2 | no: 872 |
| viewed_pricing | legacy | no: 872 |
| viewed_pricing | v2 | no: 872 |
| ip_country | legacy | match: 872 |
| ip_country | v2 | match: 872 |
| ip_type | legacy | residential: 872 |
| ip_type | v2 | residential: 872 |
| business_hours | legacy | outside: 872 |
| business_hours | v2 | outside: 872 |
| session_missing | legacy | (no feature) |
| session_missing | v2 | yes: 872 |

### Weights before and after

Log-odds per design column. Empty = the column does not exist in that design. The baseline column is the frozen `baseline/weights.csv` (fitted on data/v1, whatever `--data` is). The baseline has `no_company`; v2 has the `session_missing` and `enrichment_missing` indicators and `band=missing`, and drops c_budget, c_timeline, ip_type and edits_1_4 (2.6).

| column | baseline (weights.csv) | legacy features, horizon labels | v2, legacy labels | v2, horizon labels (default) |
|---|---|---|---|---|
| band=1000+ | 0.703 | 0.591 | 1.254 | 0.798 |
| band=11-50 | 0.193 | 0.287 | 0.433 | 0.389 |
| band=201-1000 | 0.873 | 1.045 | 1.427 | 1.305 |
| band=51-200 | 1.078 | 1.101 | 1.412 | 1.281 |
| band=missing |  |  | 1.176 | 1.128 |
| business_hours=wkday_9-18 | 0.160 | 0.476 | 0.321 | 0.449 |
| c_budget=Under £5k | 0.122 | 0.144 |  |  |
| c_budget=£20k-£50k | 0.152 | 0.103 |  |  |
| c_budget=£50k+ | 0.149 | -0.142 |  |  |
| c_budget=£5k-£20k | -0.082 | 0.193 |  |  |
| c_timeline=Just researching | 0.126 | -0.194 |  |  |
| c_timeline=Next 6 months | 0.019 | 0.274 |  |  |
| c_timeline=This month | 0.002 | 0.217 |  |  |
| c_timeline=This quarter | 0.195 | 0.001 |  |  |
| channel=chatgpt | 0.053 | 0.079 | 0.022 | 0.098 |
| channel=linkedin | 0.226 | 0.748 | 0.604 | 0.778 |
| channel=meta | -0.259 | -0.279 | -0.287 | -0.261 |
| channel=meta_leadads | -0.489 | -0.851 | -0.762 | -0.933 |
| channel=organic_direct | 0.160 | 0.197 | 0.244 | 0.202 |
| crm=hubspot_sf | 0.131 | 0.273 | 0.260 | 0.285 |
| edits_1_4=yes | 0.050 | -0.080 |  |  |
| email=free | -0.391 | -1.555 | -1.390 | -1.500 |
| enrichment_missing=yes |  |  | 0.018 | 0.039 |
| form_variant=B | 0.213 | 0.023 | 0.195 | 0.094 |
| form_variant=C | 0.342 | 0.297 | 0.661 | 0.584 |
| form_variant=D | -0.381 | -0.286 | -0.460 | -0.452 |
| hesitation_90s=yes | -0.312 | -0.402 | -0.388 | -0.409 |
| hiring=hiring | 0.360 | 0.081 | 0.100 | 0.096 |
| ip_country=mismatch | -0.331 | -0.261 | -0.057 | -0.087 |
| ip_type=dc | 0.022 | 0.381 |  |  |
| no_company=yes | -0.391 | 0.781 |  |  |
| search_term=brand | 0.155 | 0.593 | 0.462 | 0.641 |
| search_term=not_google | -0.310 | -0.106 | -0.180 | -0.117 |
| seniority=mid | 0.077 | 0.119 | 0.137 | 0.126 |
| seniority=not_asked/blank | -0.277 | -0.104 | -0.537 | -0.273 |
| seniority=senior | 0.481 | 0.155 | 0.418 | 0.196 |
| seniority=student | -1.471 | -0.776 | -1.062 | -0.836 |
| session_missing=yes |  |  | 0.559 | 0.632 |
| sessions_3plus=yes | 0.406 | 0.200 | 0.211 | 0.231 |
| spend=none | -0.571 | -0.469 | -0.347 | -0.368 |
| spend=£100k+ | 0.429 | 0.606 | 0.336 | 0.292 |
| spend=£25k-£100k | 0.484 | 0.476 | 0.443 | 0.498 |
| spend=£5k-£25k | 0.246 | 0.271 | 0.141 | 0.224 |
| text=copy_paste | -0.950 | 0.385 | 0.124 | 0.246 |
| text=specific | 0.507 | 0.872 | 0.804 | 0.913 |
| text=vague | -0.899 | -0.600 | -0.401 | -0.609 |
| time_on_page=300-600s | -0.090 | -0.504 | -0.252 | -0.403 |
| time_on_page=60-300s | 0.508 | -0.048 | 0.123 | 0.068 |
| time_on_page=>600s | -0.214 | -0.343 | 0.005 | -0.146 |
| viewed_pricing=yes | 0.229 | 0.406 | 0.463 | 0.443 |

#### Coefficients equal to three decimals

- baseline (weights.csv): email=free = no_company=yes (-0.391; corr +0.626); business_hours=wkday_9-18 = channel=organic_direct (0.160; corr +0.048)
- legacy features, horizon labels: spend=£25k-£100k = business_hours=wkday_9-18 (0.476; corr +0.020)
- v2, legacy labels: none
- v2, horizon labels (default): none

### 2.7 Collinearity

Threshold |corr| > 0.95 on the training rows of each model's design.

| model | design columns | training rows | max |corr| | check | pairs above threshold |
|---|---|---|---|---|---|
| legacy features, legacy labels (baseline) | 47 | 5507 | 0.693 | pass | none |
| legacy features, horizon labels | 47 | 3787 | 0.690 | pass | none |
| v2, legacy labels | 39 | 5507 | 0.814 | pass | none |
| v2, horizon labels (default) | 39 | 3787 | 0.820 | pass | none |

### Feature change alone: v2 vs legacy features, same labels

Paired bootstrap (1000 resamples, seed 0) of AUC(v2) − AUC(legacy features), both models trained on the same label definition and scored on the same frozen test rows.

| labels | test set | AUC legacy features | AUC v2 features | v2 − legacy [95% CI] | bootstrap p |
|---|---|---|---|---|---|
| legacy | (a) legacy test set (n=2336) | 0.785 | 0.802 | +0.018 [+0.010, +0.026] | 0.000 |
| legacy | (b) mature test set (n=448) | 0.787 | 0.804 | +0.017 [+0.001, +0.036] | 0.038 |
| horizon H=120 | (a) legacy test set (n=2336) | 0.785 | 0.803 | +0.019 [+0.011, +0.027] | 0.000 |
| horizon H=120 | (b) mature test set (n=448) | 0.791 | 0.805 | +0.014 [-0.002, +0.031] | 0.088 |

### 2.4 Deal-value model: residual sd

The correction is exp(sd²/2). It multiplies every lead's expected deal value by the same factor, so it cannot change any ranking: top-20% capture by p×value is identical by construction. It changes the level of the value sent, checked here against recorded deal values of won test leads. Revenue (a)/(b) = recorded deal value on the frozen legacy / mature test sets; the baseline-style figure fills blank deal values with the model's own prediction (so it can move slightly).

| model | residual sd | sd | correction | training deals | top-20% revenue by p×value (a) | top-20% revenue by p×value (b) | mean predicted / mean recorded deal value, won test leads | baseline-style top20_revenue (imputes blanks) |
|---|---|---|---|---|---|---|---|---|
| v2, legacy labels | fixed (baseline constant) | 0.4500 | 1.1066 | 753 | 0.7810 | 0.6875 | 0.975 | 0.7900 |
| v2, legacy labels | estimated from training residuals | 0.5516 | 1.1643 | 753 | 0.7810 | 0.6875 | 1.026 | 0.7904 |
| v2, horizon labels (default) | fixed (baseline constant) | 0.4500 | 1.1066 | 753 | 0.7503 | 0.6866 | 0.899 | 0.7112 |
| v2, horizon labels (default) | estimated from training residuals | 0.5516 | 1.1643 | 753 | 0.7503 | 0.6866 | 0.945 | 0.7124 |


---

## Output: `python -m emva.eval.feature_selection --n-refits 1000` (2.6 Feature selection, v1)

### Plan 2.6: feature selection by coefficient bootstrap

Data `data/v1`, labels `horizon H=120`, v2 design with every candidate feature, training rows (created before 2026-05-01). 95% percentile CI from 1000 refits on resampled training rows (seed 0).

| feature | level | log-odds [95% CI] | excludes 0 | n train |
|---|---|---|---|---|
| c_budget | c_budget=Under £5k | 0.069 [-0.305, 0.413] | no | 169 |
| c_budget | c_budget=£20k-£50k | 0.105 [-0.272, 0.486] | no | 82 |
| c_budget | c_budget=£50k+ | 0.027 [-0.464, 0.460] | no | 36 |
| c_budget | c_budget=£5k-£20k | 0.058 [-0.313, 0.426] | no | 120 |
| c_timeline | c_timeline=Just researching | 0.288 [-0.099, 0.672] | no | 121 |
| c_timeline | c_timeline=Next 6 months | -0.079 [-0.496, 0.341] | no | 101 |
| c_timeline | c_timeline=This month | -0.079 [-0.644, 0.432] | no | 41 |
| c_timeline | c_timeline=This quarter | 0.129 [-0.225, 0.474] | no | 144 |
| ip_type | ip_type=dc | 0.171 [-0.482, 0.741] | no | 104 |
| edits_1_4 | edits_1_4=yes | -0.008 [-0.229, 0.240] | no | 2933 |

| feature | decision | reason |
|---|---|---|
| c_budget | drop | every level's CI contains 0 |
| c_timeline | drop | every level's CI contains 0 |
| ip_type | drop | every level's CI contains 0 |
| edits_1_4 | drop | every level's CI contains 0 |

