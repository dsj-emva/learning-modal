# Phase 2: features and leakage

Branch `phase2-features` from `origin/main` 92168cf, with `origin/main` merged in twice (Phase 5 v2 at
504f11c / 01c56f4, then the docs context layer at e98a586). Plan items 2.1 to 2.7 plus the orchestrator's
rulings below. `--feature-set {legacy,v2}` (default `v2`) sits next to `--label-mode`;
`--label-mode legacy --feature-set legacy` is byte-identical to `baseline/` (`make baseline` passes).

**v2 data:** `data/v2` on this branch, as merged from `main` (regenerated at da72dad, "Regenerate data/v2
after the review fixes (paraphrase on)"; merged via 504f11c / 01c56f4). All v2-data numbers below are
from that data.

## Orchestrator decisions (review of Phase 2)

- **Deviation 1 (data-dependent pruning of identical design columns): not accepted (ADR 0009).** The
  model's column set must be fixed by the feature spec. Implemented as:
  - `session_missing` (yes/no): yes when the on-site session is absent (blank `landing_url`, or any
    behavioural input blank). The seven behavioural features (`time_on_page`, `hesitation_90s`,
    `sessions_3plus`, `viewed_pricing`, `ip_country`, `ip_type`, `business_hours`) then take their
    reference level and the indicator carries the effect.
  - `enrichment_missing` (yes/no); `spend`, `crm`, `hiring` take their reference level without
    enrichment; `band` keeps its own `missing` level (the typed company-size fallback makes it distinct).
  - **Fixed level lists:** `emva.constants.V2_LEVELS` declares every v2 feature's complete level list
    (reference first); `emva.design.fixed_design` emits exactly those columns (**39 for the v2 feature
    set**) whatever occurs in the data. A level absent from the data gives an all-zero column (a one-row
    frame gets the full schema); an undeclared value raises `ValueError` naming the feature. The legacy
    design stays data-driven (byte identity).
  - No automatic column dropping. `python -m emva.eval.collinearity` is strict (exit 1). The standard
    report prints the check as a PASS/FAIL section and exits 1 only with `--strict`, so `make report`
    on v1 is not failed by a data property.
- **Deviation 2** (legacy residual-sd constant in `emva/eval/regression.py`, imported by the pipeline for
  `--feature-set legacy` only): accepted; commented at the import site. **Deviations 3 to 8:** accepted.
- **R5 (ADR 0011).** The "no identical coefficients" and collinearity criteria are evaluated on
  `data/v2`: on v1, `session_missing` and `channel=meta_leadads` are the same column by construction of
  the data (every session-less v1 lead is a lead-ads lead), not of the model. Phase 2 passes on v2; the
  AUC criterion passes on v1. Both results are kept below.
- **R6 (ADR 0011).** The boilerplate detector's low recall on paraphrased v2 text (0.062 now) is a
  known gap handed to Phase 6: semantic detection belongs to the context agent. Not fixed here.
- **R7 (ADR 0011).** The residual sd estimated on v2 (0.5505 vs the old fixed 0.45) is evidence for plan
  2.4: a hard-coded generator constant would understate v2 deal values.

## Acceptance

| criterion | evaluated on | result | numbers |
|---|---|---|---|
| No two coefficients identical to 3 dp | **data/v2 (R5)** | **pass** for the default model (v2 features, horizon labels): no equal pair | v2 features + legacy labels on v2 has one coincidental tie, `sessions_3plus=yes` = `hiring=hiring` = 0.062 (corr 0.000), not a duplicated column. v1 for reference: `channel=meta_leadads` = `session_missing=yes` = −0.345 (identical columns; the L2 penalty splits one effect equally), plus a coincidental `hiring=hiring` = `band=1000+` = 0.382 |
| Collinearity, no pair above 0.95 | **data/v2 (R5)** | **pass** | v2 features: max |corr| 0.817 horizon / 0.813 legacy labels (`enrichment_missing=yes` ~ `band=missing`). v1 for reference: FAIL on exactly `channel=meta_leadads` ~ `session_missing=yes` (1.000); legacy features FAIL on `email=free` ~ `no_company=yes` (1.000) on v1 and pass on v2 (0.690) |
| Lead-ads leads not silently in the reference bucket | v1 and v2 | **pass as re-specified** | v1: all 932 lead-ads leads have the seven behavioural features at reference and `session_missing=yes` (weight −0.345, not separable from the channel on v1). v2: 872 lead-ads + 1,278 consent-declined website leads have `session_missing=yes`; weight +0.536 horizon / +0.530 legacy labels, separately identified from `channel=meta_leadads` (−0.775 / −0.679) |
| AUC within the baseline CI [0.789, 0.836], legacy test set | **v1** | **pass** | v2 features + legacy labels 0.816 [0.792, 0.838], paired vs baseline +0.002 [−0.002, +0.006]; + horizon labels 0.814 [0.790, 0.837], paired +0.000 [−0.006, +0.006] |
| AUC on the frozen mature test set (R3) | v1 | reported | 0.781 [0.730, 0.828] (legacy labels), 0.778 [0.730, 0.827] (horizon); paired +0.002 / +0.000 |
| grep `0.45\|ground_truth` only in `emva/eval/` | code | **pass** (test-enforced) | |

v2 data, v2 vs legacy features on the same labels (paired, 1,000 resamples): legacy test set (n = 2,309)
+0.015 [+0.007, +0.024] with either label definition; mature test set (n = 441) +0.014 [−0.003, +0.030]
(legacy labels) and +0.014 [−0.003, +0.029] (horizon). AUC 0.806 vs 0.791 (horizon labels, legacy test
set). On v1 the two feature sets are level (+0.001 to +0.002, CIs include 0).

## Indicator weights

| log-odds | v1, legacy labels | v1, horizon (default) | v2, legacy labels | v2, horizon |
|---|---|---|---|---|
| session_missing=yes | −0.273 | −0.345 | +0.530 | +0.536 |
| channel=meta_leadads | −0.273 | −0.345 | −0.679 | −0.775 |
| enrichment_missing=yes | +0.091 | +0.168 | −0.159 | −0.066 |
| band=missing | +0.041 | −0.084 | +1.173 | +0.981 |
| email=free | −1.090 | −1.115 | −1.308 | −1.282 |

On v1 the first two rows are one effect split in half. The positive `session_missing` and large
positive `band=missing` weights on v2 are what the fitted model says; I have not checked them against
the v2 generator's planted effects and flag them for the Phase 5 / Phase 4 review.

## Other required numbers

- **2.3 match count:** v1: of **2,904** cleaned domain-miss leads, **948 (32.6%)** match a companies.csv
  name exactly after normalisation; ground truth confirms 948/948. v2: 2,081 of 4,763 (43.7%), all
  correct. Exact match only.
- **2.5 boilerplate:** Jaccard ≥ **0.6**, 16 snippets. v1: precision 1.000, recall 1.000 (807/807).
  v2: precision 1.000, **recall 0.062** (50/812); legacy prefix rule 0.012. Handed to Phase 6 (R6).
- **2.6:** re-run on the final design (1,000 refits, horizon labels, v1): c_budget, c_timeline, ip_type,
  edits_1_4 all dropped (no level's CI excludes zero).
- **2.7:** see acceptance; full CLI output in "Collinearity and pipeline runs".
- **2.4 value model (R7):** v1: fixed 0.45 (correction 1.1066) vs estimated **0.4546** (1.1089). v2:
  estimated **0.5505** (1.1636); mean predicted / mean recorded deal value on won test leads 0.910 with
  the fixed constant vs 0.957 estimated (horizon), 0.970 vs 1.020 (legacy labels). Top-20% capture by
  p×value is unchanged by construction (a constant factor).

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

- **Switch.** `emva.features.FeatureSet`, `--feature-set` on `python -m emva`, `emva.eval.report`,
  `emva.eval.collinearity`. Legacy path intact; `make baseline` runs `--label-mode legacy --feature-set
  legacy`. Phase 1 study scripts and `scripts/compare_to_v1.py` pinned to legacy features.
- **2.1 / 2.2** indicator encoding and fixed level lists as ruled (above).
- **2.3** `emva.io.normalise_company_name` / `match_company_names`: lower-case, dots and apostrophes
  dropped, other punctuation to spaces, whitespace collapsed, trailing legal suffixes stripped (the plan's
  list plus `sas`); ambiguous names skipped; exact match only.
- **2.4** `emva.value.DealValueModel`: sd = sample sd of the in-sample training residuals of the log-value
  ridge; correction exp(sd²/2).
- **2.5** `emva/boilerplate.py`, token-set Jaccard, threshold 0.6 (configurable).
- **2.6** `python -m emva.eval.feature_selection`; outcome hard-coded in `emva.features.V2_DROPPED`.
- **2.7** `emva/eval/collinearity.py` (strict CLI); report section with `--strict`.

## Deviations, not done, open questions

1. The deal-value design (`emva.value.deal_value_design`: band, sector, channel, spend) is still built
   from the levels present in the data; the fixed-schema ruling was applied to the formula design only.
   A hosted scorer needs the same treatment there (Phase 3 owns the value layer).
2. Coincidental 3-dp ties between unrelated columns are reported (v1 default model; v2 with legacy
   labels), not engineered away.
3. The v2-data weights for `session_missing` and `band=missing` are unexplained here (above).
4. Accepted earlier: legacy constant location, `sas` suffix, 948 matches, no missing level for the
   dropped `edits_1_4`, intermediate commits not all green, one-line changes in `scripts/compare_to_v1.py`
   and `tests/test_generate_data_v1.py`.

## Commands

```
make PY=.venv/bin/python baseline
python -m emva.eval.report [--strict]                     # v2 features, horizon labels (default)
python -m emva.eval.report --label-mode legacy [--strict] # v2 features, legacy labels
python -m emva.eval.phase2_study [--data data/v2]
python -m emva.eval.feature_selection --n-refits 1000
python -m emva.eval.collinearity [--data DIR] [--feature-set legacy] [--label-mode legacy]   # strict
```

The outputs follow, verbatim.


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

### data/v2 (main, regenerated at da72dad): `--label-mode horizon --feature-set v2`

```
$ python -m emva --label-mode horizon --feature-set v2
  model   auc  brier  top20_wins  top20_revenue
formula 0.795 0.1215       0.513          0.728
$ python -m emva.eval.collinearity --label-mode horizon --feature-set v2
feature set v2, labels horizon H=120, 3789 training rows, 39 design columns
- PASS: max |corr| = 0.817 between enrichment_missing=yes and band=missing
(exit 0)
```

### data/v2 (main, regenerated at da72dad): `--label-mode horizon --feature-set legacy`

```
$ python -m emva --label-mode horizon --feature-set legacy
  model   auc  brier  top20_wins  top20_revenue
formula 0.781 0.1249       0.513          0.685
$ python -m emva.eval.collinearity --label-mode horizon --feature-set legacy
feature set legacy, labels horizon H=120, 3789 training rows, 47 design columns
- PASS: max |corr| = 0.690 between ip_country=mismatch and ip_type=dc
(exit 0)
```

### data/v2 (main, regenerated at da72dad): `--label-mode legacy --feature-set v2`

```
$ python -m emva --label-mode legacy --feature-set v2
  model   auc  brier  top20_wins  top20_revenue
formula 0.804 0.0965       0.541          0.766
$ python -m emva.eval.collinearity --label-mode legacy --feature-set v2
feature set v2, labels legacy, 5509 training rows, 39 design columns
- PASS: max |corr| = 0.813 between enrichment_missing=yes and band=missing
(exit 0)
```

### data/v2 (main, regenerated at da72dad): `--label-mode legacy --feature-set legacy`

```
$ python -m emva --label-mode legacy --feature-set legacy
  model   auc  brier  top20_wins  top20_revenue
formula 0.789 0.0983       0.525          0.722
$ python -m emva.eval.collinearity --label-mode legacy --feature-set legacy
feature set legacy, labels legacy, 5509 training rows, 47 design columns
- PASS: max |corr| = 0.686 between ip_country=mismatch and ip_type=dc
(exit 0)
```


---

## Output: `python -m emva.eval.report` on v1 (v2 features, horizon labels)

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

### Design collinearity (plan 2.7): FAIL

Candidate design (`v2` features, 39 columns) on its 4049 training rows; fails when two columns have |corr| > 0.95. The report exits 1 on a failure only with `--strict`; `python -m emva.eval.collinearity` always does.

- FAIL (1 pairs with |corr| > 0.95): max |corr| = 1.000 between channel=meta_leadads and session_missing=yes
  - channel=meta_leadads ~ session_missing=yes: +1.000

### Baseline script output

- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, which fills blank deal values with its predicted value, printed:

```
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795
```

---

## Output: `python -m emva.eval.report --label-mode legacy` on v1 (v2 features, legacy labels)

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

### Design collinearity (plan 2.7): FAIL

Candidate design (`v2` features, 39 columns) on its 5588 training rows; fails when two columns have |corr| > 0.95. The report exits 1 on a failure only with `--strict`; `python -m emva.eval.collinearity` always does.

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
- v2, legacy labels: session_missing=yes = channel=meta_leadads (-0.273; corr +1.000)
- v2, horizon labels (default): session_missing=yes = channel=meta_leadads (-0.345; corr +1.000); hiring=hiring = band=1000+ (0.382; corr +0.217)

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
| cleaned leads | 9209 |
| email-domain misses (no companies.csv row by domain) | 4763 |
| ... with a typed company name (company_name or the company answer) | 2494 (52.4%) |
| ... matched by exact normalised name | 2081 (43.7%) |
| ... of which the historical_leads.company_name column matched | 2081 |
| still without enrichment (enrichment_missing=yes) | 2682 (29.1%) |
| truth: matched company's employee band = generator's band (ground truth) | 2081 (100.0%) |
| truth: matched company's sector = generator's sector (ground truth) | 2081 (100.0%) |
| truth: generator has_company among the matched | 2081 (100.0%) |
| truth: generator has_company among unmatched domain misses | 1612 (60.1%) |

Most common unmatched typed names among domain misses: (blank) (2390), Self-employed (116), Freelance (109), Lindenwood Digital Incorporated (2), Brightwater Industries, Incorporated (1).

### 2.5 Boilerplate detector

Cleaned leads (9209); truth = the generator's `text_category == copy_paste` (ground truth, evaluation only). Similarity = token-set Jaccard with the closest of the snippets in `emva/boilerplate.py`.

| detector | flagged | true copy_paste | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|---|
| v2 similarity (Jaccard >= 0.6) | 50 | 812 | 50 | 0 | 762 | 1.000 | 0.062 |
| legacy prefix rule | 10 | 812 | 10 | 0 | 802 | 1.000 | 0.012 |

Similarity margin: lowest similarity among true copy_paste texts 0.000; highest among all other texts 0.243. The two overlap: no threshold separates boilerplate from other text on this data.

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
| band=1000+ | 0.703 | 0.817 | 1.144 | 0.798 |
| band=11-50 | 0.193 | 0.310 | 0.480 | 0.431 |
| band=201-1000 | 0.873 | 1.152 | 1.233 | 1.121 |
| band=51-200 | 1.078 | 1.070 | 1.282 | 1.254 |
| band=missing |  |  | 1.173 | 0.981 |
| business_hours=wkday_9-18 | 0.160 | 0.312 | 0.296 | 0.281 |
| c_budget=Under £5k | 0.122 | 0.113 |  |  |
| c_budget=£20k-£50k | 0.152 | 0.011 |  |  |
| c_budget=£50k+ | 0.149 | -0.141 |  |  |
| c_budget=£5k-£20k | -0.082 | 0.287 |  |  |
| c_timeline=Just researching | 0.126 | 0.004 |  |  |
| c_timeline=Next 6 months | 0.019 | 0.184 |  |  |
| c_timeline=This month | 0.002 | 0.297 |  |  |
| c_timeline=This quarter | 0.195 | -0.215 |  |  |
| channel=chatgpt | 0.053 | -0.096 | -0.144 | -0.078 |
| channel=linkedin | 0.226 | 0.707 | 0.623 | 0.717 |
| channel=meta | -0.259 | -0.344 | -0.339 | -0.355 |
| channel=meta_leadads | -0.489 | -0.768 | -0.679 | -0.775 |
| channel=organic_direct | 0.160 | 0.137 | 0.164 | 0.125 |
| crm=hubspot_sf | 0.131 | 0.102 | 0.156 | 0.190 |
| edits_1_4=yes | 0.050 | -0.095 |  |  |
| email=free | -0.391 | -1.315 | -1.308 | -1.282 |
| enrichment_missing=yes |  |  | -0.159 | -0.066 |
| form_variant=B | 0.213 | -0.024 | 0.187 | 0.035 |
| form_variant=C | 0.342 | 0.270 | 0.641 | 0.500 |
| form_variant=D | -0.381 | -0.414 | -0.575 | -0.589 |
| hesitation_90s=yes | -0.312 | -0.376 | -0.402 | -0.375 |
| hiring=hiring | 0.360 | 0.077 | 0.062 | 0.082 |
| ip_country=mismatch | -0.331 | -0.422 | -0.314 | -0.238 |
| ip_type=dc | 0.022 | 0.380 |  |  |
| no_company=yes | -0.391 | 0.495 |  |  |
| search_term=brand | 0.155 | 0.305 | 0.162 | 0.312 |
| search_term=not_google | -0.310 | -0.364 | -0.376 | -0.367 |
| seniority=mid | 0.077 | 0.280 | 0.263 | 0.285 |
| seniority=not_asked/blank | -0.277 | 0.107 | -0.264 | -0.040 |
| seniority=senior | 0.481 | 0.275 | 0.506 | 0.300 |
| seniority=student | -1.471 | -0.629 | -0.691 | -0.670 |
| session_missing=yes |  |  | 0.530 | 0.536 |
| sessions_3plus=yes | 0.406 | 0.041 | 0.062 | 0.070 |
| spend=none | -0.571 | -0.531 | -0.479 | -0.515 |
| spend=£100k+ | 0.429 | 0.200 | 0.428 | 0.368 |
| spend=£25k-£100k | 0.484 | 0.329 | 0.426 | 0.421 |
| spend=£5k-£25k | 0.246 | 0.193 | 0.236 | 0.292 |
| text=copy_paste | -0.950 | 0.334 | -0.228 | -0.258 |
| text=specific | 0.507 | 0.780 | 0.674 | 0.805 |
| text=vague | -0.899 | -0.404 | -0.495 | -0.404 |
| time_on_page=300-600s | -0.090 | -0.021 | 0.109 | 0.110 |
| time_on_page=60-300s | 0.508 | 0.192 | 0.244 | 0.311 |
| time_on_page=>600s | -0.214 | -0.069 | 0.046 | 0.123 |
| viewed_pricing=yes | 0.229 | 0.499 | 0.568 | 0.534 |

#### Coefficients equal to three decimals

- baseline (weights.csv): email=free = no_company=yes (-0.391; corr +0.626); business_hours=wkday_9-18 = channel=organic_direct (0.160; corr +0.049)
- legacy features, horizon labels: none
- v2, legacy labels: sessions_3plus=yes = hiring=hiring (0.062; corr +0.000)
- v2, horizon labels (default): none

### 2.7 Collinearity

Threshold |corr| > 0.95 on the training rows of each model's design.

| model | design columns | training rows | max |corr| | check | pairs above threshold |
|---|---|---|---|---|---|
| legacy features, legacy labels (baseline) | 47 | 5509 | 0.686 | pass | none |
| legacy features, horizon labels | 47 | 3789 | 0.690 | pass | none |
| v2, legacy labels | 39 | 5509 | 0.813 | pass | none |
| v2, horizon labels (default) | 39 | 3789 | 0.817 | pass | none |

### Feature change alone: v2 vs legacy features, same labels

Paired bootstrap (1000 resamples, seed 0) of AUC(v2) − AUC(legacy features), both models trained on the same label definition and scored on the same frozen test rows.

| labels | test set | AUC legacy features | AUC v2 features | v2 − legacy [95% CI] | bootstrap p |
|---|---|---|---|---|---|
| legacy | (a) legacy test set (n=2309) | 0.789 | 0.804 | +0.015 [+0.007, +0.024] | 0.000 |
| legacy | (b) mature test set (n=441) | 0.776 | 0.790 | +0.014 [-0.003, +0.030] | 0.100 |
| horizon H=120 | (a) legacy test set (n=2309) | 0.791 | 0.806 | +0.015 [+0.007, +0.024] | 0.000 |
| horizon H=120 | (b) mature test set (n=441) | 0.781 | 0.795 | +0.014 [-0.003, +0.029] | 0.088 |

### 2.4 Deal-value model: residual sd

The correction is exp(sd²/2). It multiplies every lead's expected deal value by the same factor, so it cannot change any ranking: top-20% capture by p×value is identical by construction. It changes the level of the value sent, checked here against recorded deal values of won test leads. Revenue (a)/(b) = recorded deal value on the frozen legacy / mature test sets; the baseline-style figure fills blank deal values with the model's own prediction (so it can move slightly).

| model | residual sd | sd | correction | training deals | top-20% revenue by p×value (a) | top-20% revenue by p×value (b) | mean predicted / mean recorded deal value, won test leads | baseline-style top20_revenue (imputes blanks) |
|---|---|---|---|---|---|---|---|---|
| v2, legacy labels | fixed (baseline constant) | 0.4500 | 1.1066 | 760 | 0.7672 | 0.7231 | 0.970 | 0.7662 |
| v2, legacy labels | estimated from training residuals | 0.5505 | 1.1636 | 760 | 0.7672 | 0.7231 | 1.020 | 0.7661 |
| v2, horizon labels (default) | fixed (baseline constant) | 0.4500 | 1.1066 | 760 | 0.7785 | 0.7408 | 0.910 | 0.7287 |
| v2, horizon labels (default) | estimated from training residuals | 0.5505 | 1.1636 | 760 | 0.7785 | 0.7408 | 0.957 | 0.7282 |


---

## Output: `python -m emva.eval.feature_selection --n-refits 1000` (2.6 Feature selection, v1)

### Plan 2.6: feature selection by coefficient bootstrap

Data `data/v1`, labels `horizon H=120`, v2 design with every candidate feature, training rows (created before 2026-05-01). 95% percentile CI from 1000 refits on resampled training rows (seed 0).

| feature | level | log-odds [95% CI] | excludes 0 | n train |
|---|---|---|---|---|
| c_budget | c_budget=Under £5k | 0.069 [-0.305, 0.413] | no | 169 |
| c_budget | c_budget=£5k-£20k | 0.058 [-0.313, 0.426] | no | 120 |
| c_budget | c_budget=£20k-£50k | 0.105 [-0.272, 0.486] | no | 82 |
| c_budget | c_budget=£50k+ | 0.027 [-0.464, 0.460] | no | 36 |
| c_timeline | c_timeline=This month | -0.079 [-0.644, 0.432] | no | 41 |
| c_timeline | c_timeline=This quarter | 0.129 [-0.225, 0.474] | no | 144 |
| c_timeline | c_timeline=Next 6 months | -0.079 [-0.496, 0.341] | no | 101 |
| c_timeline | c_timeline=Just researching | 0.288 [-0.099, 0.672] | no | 121 |
| ip_type | ip_type=dc | 0.171 [-0.482, 0.741] | no | 104 |
| edits_1_4 | edits_1_4=yes | -0.008 [-0.229, 0.240] | no | 2933 |

| feature | decision | reason |
|---|---|---|
| c_budget | drop | every level's CI contains 0 |
| c_timeline | drop | every level's CI contains 0 |
| ip_type | drop | every level's CI contains 0 |
| edits_1_4 | drop | every level's CI contains 0 |

