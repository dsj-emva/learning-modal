# Phase 1: labels

Branch `phase1-labels` (from `main` 15f90fc, `origin/main` 869805a merged in after review). Plan items 1.1
to 1.6, then the review fixes and rulings below.

## Acceptance

| criterion | result | legacy (baseline) | horizon H=120 (candidate) | reference |
|---|---|---|---|---|
| `band=201-1000` weight moves toward 1.3 | **pass** (direction; CIs overlap) | 0.873 [0.591, 1.162] | 0.971 [0.680, 1.275] | planted 1.3 |
| `band=1000+` weight moves toward 1.0 | **fail** (moves away) | 0.703 [0.331, 1.060] | 0.419 [0.046, 0.806] | planted 1.0 |
| labelled win rate, band 201-1000, training | **pass** | 27.7% (n=692) | 33.5% (n=514) | 39.4% true (mean `p_close_true`, same 692 rows); 39.5% on the 514 horizon rows |
| AUC on mature leads within baseline CI or higher | **pass** | 0.778 [0.726, 0.827] | 0.778 [0.726, 0.827]; paired diff −0.001 [−0.013, +0.010], p = 0.98 | same 458 mature test leads |
| bottom-decile ghosted share reported for both | **reported** | 27.5% still New, 24.5% ghosted | 26.5% still New, 23.9% ghosted | population: all leads labelled by the legacy rules (8,041) |
| **R1 (restated):** horizon size weights consistent with the true 120-day effects | **pass, all four bands** (see below) | | | true conditional 120-day effects |

Weight CIs: 95% percentile, 1,000 refits of the formula model on bootstrap resamples of each
definition's training rows (`python -m emva.eval.label_study`, seed 0, ~50 s for the whole
study). AUC CIs: 1,000 resamples (`make report`). On the legacy test set the candidate also sits
inside the baseline CI: 0.812 vs 0.814 [0.789, 0.836], paired diff −0.001 [−0.005, +0.003].

## Orchestrator decisions (review of Phase 1)

- **R1.** The size criteria as written target eventual-close effects (1.3, 1.0); a 120-day label
  estimates the 120-day effect. Restated: the learned horizon weights must be consistent with the
  *true 120-day* effects. Pass per band = the horizon weight's 95% CI contains the true effect, or
  the horizon point is closer to it than the legacy point. Result below.
- **R2.** Mature non-wins still open at AS_OF are 0 under the fixed horizon (they demonstrably did
  not win within H). Stalled leads are in the same position but stay censored by default (plan
  1.4), because stalling is read as sales neglect rather than a buyer decision; that is a
  deliberate asymmetry, and the `--stalled-as-lost` row in the flag table shows its effect
  (+513 training zeros, 201-1000 rate 33.5% to 29.2%, AUC unchanged). No code change; the
  paragraph is also in the `emva/labels.py` docstring.
- **R3.** The mature test set (458 leads, 26 days) is frozen per ground rule 4. Phase 4's
  rolling-origin evaluation becomes the headline number. No code change.
- **R4.** Leaving `label_source` (and the other label columns) out of legacy `scores.csv` to keep
  byte-identity is accepted.
- Review fixes applied: legacy mode no longer computes `label_source`, so an unrecognised final
  stage is unlabelled as in the baseline instead of raising; `label_source == ghosted` now needs a
  mature lead (not contacted by `matured_at`), so the 528 still-New leads younger than 120 days
  are `open`; `LabelMode` enum and `LabelConfig.eligible()` / `extra_score_columns()` replace the
  string switches; one `_idle_days` / `_stalled` helper serves both label rules (legacy output
  still byte-identical); `FROZEN_HORIZON` in `report.py`; the `make baseline` test passes `PY`
  so it runs in a worktree; `ground_truth_reference` builds the leads once.

### R1: true 120-day size effects vs learned weights

From `python -m emva.eval.ground_truth_reference` (1,000 refits, seed 0). **Which truth:** the
hidden truth gives only the eventual close probability `p_close_true`, so each lead's planted
120-day probability is derived from the planted timing, not from the observed labels:
`p120 = p_close_true × P(time to close ≤ 120 d)`. Time to close is the generator's lognormal
(median 40 days, log-sd 1.09, ×1.8 for true band 1000+), parsed from `ground_truth.md`, giving
0.843 (0.680 for 1000+). The derived rates match the observed won-within-120 labels by band:

| band | rows | mean p_close_true (eventual) | derived p120 | observed won within 120 d (horizon label) |
|---|---|---|---|---|
| 1-10 | 1615 | 8.5% | 7.2% | 7.8% |
| 11-50 | 922 | 18.9% | 16.0% | 15.7% |
| 51-200 | 723 | 36.9% | 31.1% | 33.2% |
| 201-1000 | 514 | 39.5% | 33.2% | 33.5% |
| 1000+ | 275 | 37.0% | 25.2% | 25.1% |

The true effect is computed on the horizon training rows in two ways. *Conditional* (the
like-for-like one): the same formula model, design, rows and regularisation fitted to `p120` as
soft labels, i.e. the weight a noise-free 120-day label would give. Fitting `p_close_true` the
same way returns 0.38 / 1.17 / 1.11 / 0.88 against the planted 0.5 / 1.3 / 1.3 / 1.0, so the method
recovers the planted effects up to feature mismatch and regularisation. *Marginal*, the ruling's
literal wording: logit(mean `p120` in band) − logit(mean `p120` in 1-10). It absorbs every feature
correlated with size (free email, missing company, ad spend), so it is 0.5 to 0.9 higher than any
multivariable weight and is shown for completeness only.

| band | true 120-day effect, conditional | true 120-day effect, marginal | legacy weight [95% CI] | horizon weight [95% CI] | R1 (conditional) | R1 (marginal) | eventual-close effect, conditional (check) |
|---|---|---|---|---|---|---|---|
| 11-50 | 0.378 | 0.898 | 0.193 [-0.064, 0.417] | 0.235 [-0.008, 0.500] | pass (in CI) | pass (closer) | 0.379 |
| 51-200 | 1.099 | 1.766 | 1.078 [0.851, 1.328] | 1.146 [0.894, 1.412] | pass (in CI) | pass (closer) | 1.172 |
| 201-1000 | 1.043 | 1.862 | 0.873 [0.591, 1.162] | 0.971 [0.680, 1.275] | pass (in CI) | pass (closer) | 1.105 |
| 1000+ | 0.532 | 1.471 | 0.703 [0.331, 1.060] | 0.419 [0.046, 0.806] | pass (in CI) | fail | 0.883 |

Per band, on the conditional truth: **pass for all four**; each horizon CI contains the true
120-day effect. Caveat: each legacy CI contains it as well, so the CI test alone does not
separate the two definitions. On point distance, horizon is closer than legacy for 11-50
(0.143 vs 0.185), 201-1000 (0.072 vs 0.170) and 1000+ (0.113 vs 0.171), and legacy is closer
for 51-200 (0.021 vs 0.047). On the marginal truth, 11-50 to 201-1000 pass (closer) and 1000+
fails. The 1000+ result reverses the original criterion: the true 120-day 1000+ effect (0.53) is
well below the eventual one (1.0), and the horizon weight (0.42) sits nearer to it than the legacy
weight (0.70) does.

### Why `band=1000+` moves the wrong way (original criterion)

`won_within_H` measures "won within 120 days", not "eventually won". The generator makes 1000+
deals close 1.8× slower (median 48 days to Won vs 31 to 35 for the other bands), so 19.8% of
the eventual 1000+ wins in training land after day 120 and become 0s, against 8% to 14% for the
other bands (label study section 3). Excluding ghosted leads also lifts the 1-10 reference rate
(5.8% with them, 7.8% without), which pulls every size weight down a little. Both effects are the new label
behaving as defined, not a bug:

- `--include-ghosted` alone gives 0.570; `--horizon-days 150` gives 0.616 (still below the
  baseline's 0.703, and it leaves no mature test leads); longer horizons shrink the training set
  (H = 240: 2,279 rows, 0.482).
- The planted 1.0 is an effect on eventual close. Under a 120-day horizon the "true" 1000+
  effect is smaller, so this criterion, as written, conflicts with the fixed-horizon label. I
  did not tune H or the flags to hit it. See open questions.

The 201-1000 rate moves most of the way (27.7% to 33.5%) but cannot reach 39.4%: that figure is
the planted probability of *eventually* closing, and 10.4% of 201-1000 wins arrive after day 120.

## What was done

- **1.1** `emva/labels.py::label_source`, from the CRM state at AS_OF (2026-09-24T00:00Z):
  `won` (last stage Won), `crm_lost` (last stage Lost), `stalled` (last stage Contacted /
  Qualified / Demo booked / Proposal, unchanged for ≥ 90 whole days), `ghosted` (mature and not
  contacted by `matured_at`; takes precedence over stalled/open), `open` (everything else
  undecided: open stage changed < 90 days ago, or still New but younger than H). Horizon mode
  only; there an unrecognised final stage raises instead of guessing, while legacy mode leaves it
  unlabelled like the baseline. `emva/io.py::load` now also returns `won_at` and
  `first_contact_at` (first change to a stage other than New) and rejects a lead with two Won
  rows.
- **1.2** `won_within_h`: 1 if the Won change is at or before `created_at + H`; 0 if mature and
  not Won by then (late wins are 0); NaN if young and not Won. `matured_at = created_at + H`.
  `HORIZON_DAYS = 120` in `constants.py`; CLI `--horizon-days`.
- **1.3** Ghosted = still at New at `matured_at` (no `first_contact_at` by then). Excluded by
  default; `--include-ghosted` counts them as 0. `label_source == ghosted` is the same group on
  mature leads by construction.
- **1.4** Stalled (`label_source == stalled`) censored by default; `--stalled-as-lost` counts
  them as 0. Neither flag ever labels a young lead ("never Lost").
- **1.5** Horizon mode trains and evaluates on mature leads only (`split_masks(eligible=...)`).
  `--label-mode {legacy,horizon}` (default horizon). `--label-mode legacy` reproduces the
  baseline byte for byte (`make baseline` now runs it that way; tests assert stdout,
  `scores.csv` and `weights.csv` byte-equal to `baseline/emva_score.py` and that `make baseline`
  passes). Horizon-only flags with `--label-mode legacy` are a usage error.
  Horizon-mode `scores.csv` = baseline columns + `won_within_h`, `label_source`, `matured_at`
  (`y` is the censored training label).
- **1.6** `make report` shows label counts by `label_source` under both definitions, then the
  full standard table for (a) legacy labels on the legacy test set and (b) horizon labels on the
  mature test set, with baseline, candidate and status quo scored on the same rows in each, then
  the bottom-decile ghosted share. The test definitions are fixed (legacy rules; default horizon
  H = 120 with default flags) whatever options the candidate is trained with
  (`python -m emva.eval.report --include-ghosted` etc. changes only the candidate).
- `emva/eval/bootstrap.py::coef_bootstrap` (percentile CI per coefficient, refits on resampled
  rows). `emva/eval/label_study.py` (weights with CIs, flag combinations, horizon sensitivity).
  `emva/eval/ground_truth_reference.py` is the only module that reads `ground_truth_labels.csv`;
  it is run as a script and nothing imports it (a test checks this).
- `scripts/run_experiments.py`: experiments `baseline` (legacy), `horizon`,
  `horizon-include-ghosted`, `horizon-stalled-as-lost`, `horizon-both-flags`.
- A horizon of about 147 days or more leaves no mature test lead; the pipeline now still trains
  and scores, prints why, and skips the metrics instead of crashing.

## Test set sizes

- (a) legacy: 2,453 leads created on or after 2026-05-01, 352 won (unchanged).
- (b) horizon: **458 leads, 80 won**. Mature means `created_at + 120 d <= 2026-09-24T00:00Z`, so
  the window is leads created from 2026-05-01 up to 2026-05-27T00:00Z (the latest is
  2026-05-26T20:06Z): 650 cleaned leads. 141 ghosted and 51 stalled are excluded, leaving 343
  crm_lost, 85 won (5 of them after day 120, so 0) and 30 still open (0): 80 wins, 378 losses.
- Training: legacy 5,588 rows (847 wins); horizon 4,049 rows (752 wins).

The mature test set covers 26 days of leads, so its AUC CI is about ±0.05, twice the legacy
width. Both models are compared on it with the paired bootstrap, which is much tighter
(±0.012).

## Label counts, training sets and the effect of each flag

From `make report` (all 9,311 scored leads; wins / losses / unlabelled):

| label_source | leads | legacy y | won_within_h | horizon y |
|---|---|---|---|---|
| won | 1199 | 1199 / 0 / 0 | 1099 / 100 / 0 | 1099 / 100 / 0 |
| crm_lost | 4896 | 0 / 4896 / 0 | 0 / 3505 / 1391 | 0 / 3505 / 1391 |
| stalled | 618 | 0 / 618 / 0 | 0 / 564 / 54 | 0 / 0 / 618 |
| ghosted | 1207 | 0 / 1207 / 0 | 0 / 1207 / 0 | 0 / 0 / 1207 |
| open | 1391 | 0 / 121 / 1270 | 0 / 70 / 1321 | 0 / 70 / 1321 |
| all | 9311 | 1199 / 6842 / 1270 | 1099 / 5446 / 2766 | 1099 / 3675 / 4537 |

From `python -m emva.eval.label_study` (training = created before 2026-05-01; AUC of each
definition's model on the two frozen test sets):

| labels | train n | train wins | train win rate | band=201-1000 | band=1000+ | AUC (a) legacy test | AUC (b) mature test |
|---|---|---|---|---|---|---|---|
| legacy | 5588 | 847 | 15.2% | 0.873 | 0.703 | 0.814 [0.789, 0.836] | 0.778 [0.726, 0.827] |
| horizon H=120 | 4049 | 752 | 18.6% | 0.971 | 0.419 | 0.812 [0.788, 0.836] | 0.778 [0.726, 0.827] |
| horizon H=120 +ghosted | 5115 | 752 | 14.7% | 0.942 | 0.570 | 0.813 [0.788, 0.836] | 0.779 [0.727, 0.827] |
| horizon H=120 +stalled-as-lost | 4562 | 752 | 16.5% | 0.886 | 0.387 | 0.812 [0.788, 0.836] | 0.778 [0.727, 0.828] |
| horizon H=120 +ghosted +stalled-as-lost | 5628 | 752 | 13.4% | 0.877 | 0.513 | 0.813 [0.788, 0.836] | 0.779 [0.728, 0.827] |

Effect of each flag alone, against the horizon default:

- `--include-ghosted`: +1,066 training rows (all 0s); 201-1000 labelled rate 33.5% → 27.6%;
  `band=201-1000` 0.971 → 0.942; `band=1000+` 0.419 → 0.570. Ghosted exclusion is the larger
  of the two label fixes.
- `--stalled-as-lost`: +513 training rows (all 0s); 201-1000 rate 33.5% → 29.2%;
  `band=201-1000` 0.971 → 0.886; `band=1000+` 0.419 → 0.387.
- Ranking quality barely moves under any definition (AUC within ±0.002 on both test sets).
  The label fix changes calibration and weights, not the ordering of these leads.

## Deviations, things not done, open questions

Deviations and choices made:

- **Bottom-decile ghosted share population.** The plan's "baseline 27%" reproduces (27.5%) only
  on all leads labelled by the legacy rules, train and test together, ranked by the baseline's p.
  On the legacy test set alone it is 19.2%. That figure counts leads *still New at AS_OF, any
  age*; with `ghosted` now meaning mature and not contacted by H, the same cell is 24.5%. The
  report shows both flags. The horizon test set excludes ghosted leads by definition, so "for
  both definitions" is reported as both models (legacy-trained baseline, horizon-trained
  candidate) on four populations that keep ghosted leads.
- **39.4% reference.** It is the mean planted `p_close_true` of the 692 training rows with
  `band == 201-1000` labelled by the legacy rules (39.5% on all training-period leads of that
  band and on the 514 horizon-labelled rows). It is an eventual-close probability, so it is an
  upper bound for a 120-day label.
- **Young `crm_lost` leads are unlabelled** (1,391 rows), as specified ("never Lost"), even
  though a CRM loss is terminal in v1. Young leads that are already Won are 1 (the Won change is
  before H). Neither reaches training or evaluation, which use mature leads only, so the
  one-sided young labels cannot bias the model; they appear only in `scores.csv`.
- **Mature cut is a timestamp**: `created_at + 120 d <= 2026-09-24T00:00Z`, so leads created on
  2026-05-27 itself (after midnight) are not mature. The brief's "on or before 2026-05-27" is
  read as "before 2026-05-27T00:00Z"; this matches the legacy rule's whole-day arithmetic.
- **`scores.csv` in legacy mode keeps the baseline's columns** so it stays byte-identical;
  `won_within_h`, `label_source` and `matured_at` are written in horizon mode only. `y` in
  horizon mode is the censored training label, `won_within_h` the raw outcome.
- **`tests/test_generate_data_v1.py`** (merged from main): its baseline-pipeline smoke test now
  asks for legacy labels explicitly, since `build()` defaults to horizon labels. One line; it
  failed after the merge otherwise.
- **Report test updated**: the Phase 0 check `"| candidate | 0.814 ["` is dropped because the
  candidate now trains on horizon labels by design (it scores 0.812 on the legacy test set); the
  legacy-mode 0.814 is still asserted by `test_pipeline_reproduces_baseline_metrics` and
  `make baseline`. All the status-quo checks are kept, now scoped to section (a).
- **Extra modules**: `emva/cli.py` (label options shared by `python -m emva` and the report),
  `emva/eval/label_study.py`, `emva/eval/ground_truth_reference.py`; `emva/model.py::make_lr`
  (the unfitted formula model, so the coefficient bootstrap refits the same estimator; `fit_lr`
  is unchanged in behaviour). `emva.eval.report._md_table` is now public `md_table`.
- `make experiment NAME=baseline` now means legacy labels explicitly (it was "default settings",
  which is now horizon).

Not done (belongs to later phases): no feature or value-model changes (Phase 2/3), so the
`emva/value.py:19` constant stays; `value_at_submit`/`value_at_close` (Phase 3).

Open questions (1 and 2 resolved by rulings R1 and R3; 3 still open):

1. **The `band=1000+` criterion conflicts with a 120-day horizon** on this generator (slow 1000+
   deals). Options: accept the fail as expected behaviour of the fixed-horizon label; restate
   the criterion as "towards the effect on 120-day closing"; or choose H from the time-to-close
   distribution (H = 150 captures 93% of 1000+ wins, but then no lead in the current test window
   is mature). I made no change.
2. **The mature test set is small (458 leads, 26 days).** With TEST_FROM = 2026-05-01 and H = 120
   it can only grow as time passes. Freezing it again (ground rule 4) freezes a ±0.05 AUC CI.
   Worth deciding before Phase 2 whether the horizon test window should start earlier (e.g.
   2026-03-01, costing training rows) or whether Phase 4's rolling-origin evaluation should be the
   headline instead.
3. Should young `crm_lost` leads be 0 (the CRM loss is terminal) rather than NaN? It changes
   only `scores.csv` today.

## Verification

`make baseline`:

```
".venv/bin/python" scripts/check_baseline.py --data "data/v1"
baseline/emva_score.py:
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795

baseline weights.csv vs committed: values equal, tie order differs for rows ['channel=organic_direct', 'business_hours=wkday_9-18']

python -m emva --label-mode legacy:
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795

emva weights.csv vs baseline run: byte-identical
emva scores.csv vs baseline run: byte-identical

PASS: baseline and emva/ both reproduce AUC 0.814, Brier 0.1006, top-20% wins 0.571, revenue 0.795 and the frozen weights.
```

`make test`: **193 passed** (about 60 s, including the 37 generator tests merged from main), then `python -m compileall -q emva scripts baseline tests`
is clean. Every commit on the branch passes its own tests.

New and changed tests:

| file | covers |
|---|---|
| `test_labels.py` | 20 hand-built CRM histories run through `load`: Won exactly at H (1), at H+1 day and H+1 s (0), early win, CRM loss before and after H, ghosted mature, contacted after H vs exactly at H, stalled at 90 idle days, open at 89 and 89.99 days, young open/won/lost/New/stalled, contacted after H and still open (ghosted wins over open), maturity at exactly AS_OF − H and 1 s later; for each: `label_source`, `won_within_h`, `y` default / `--include-ghosted` / `--stalled-as-lost` / both, and legacy `y`; `matured_at`, H as a parameter, `LabelConfig` validation and enum coercion, `eligible()` / `extra_score_columns()`, legacy mode with an unrecognised stage (unlabelled, no error) vs horizon (error), `split_masks(eligible=...)` |
| `test_io.py` | `won_at`, `first_contact_at`, two Won rows rejected |
| `test_integration.py` | `--label-mode legacy` byte-equal stdout, `scores.csv`, `weights.csv` vs the baseline script; `make baseline` passes; default CLI writes the label columns; horizon-only flags rejected with legacy; horizon pipeline uses mature leads only (4,049 / 458); report has both definitions, the new ghosted/open counts, the 27.5% still-New and 24.5% ghosted shares; `make baseline` run with `PY=` so it works in a worktree; no-mature-test-lead horizons |
| `test_bootstrap.py` | `coef_bootstrap` covers planted coefficients, is seeded, rejects bad input |
| `test_metrics_regression.py` | `bottom_share` |
| `test_label_study.py` | size weights reproduce 0.873 / 0.703 (legacy) and 0.971 (horizon); flag table sizes and rates; sensitivity with an empty test set; ground-truth reference prints 39.4%, the 0.843 / 0.680 timing, the derivation check and the true 120-day effects; nothing imports it; the ground rule 2 grep |

Ground rule 2 grep (`grep -rn "0.45\|ground_truth" emva/`):

```
emva/value.py:19:DEAL_LOG_RESIDUAL_SD_FROM_GENERATOR = 0.45
emva/eval/label_study.py:14:``python -m emva.eval.ground_truth_reference``.
emva/eval/ground_truth_reference.py:3:``python -m emva.eval.ground_truth_reference [--data data/v1] [--n-refits 1000]`` prints:
emva/eval/ground_truth_reference.py:14:``ground_truth.md`` ("Time to close") states; those three numbers are parsed from that file.
emva/eval/ground_truth_reference.py:65:    """Parse the time-to-close parameters from ``ground_truth.md``; raises if the sentence is not found."""
emva/eval/ground_truth_reference.py:66:    text = (Path(data) / "ground_truth.md").read_text()
emva/eval/ground_truth_reference.py:69:        raise ValueError(f"time-to-close sentence not found in {data}/ground_truth.md")
emva/eval/ground_truth_reference.py:74:    """``ground_truth_labels.csv`` indexed by ``lead_id``."""
emva/eval/ground_truth_reference.py:75:    return pd.read_csv(Path(data) / "ground_truth_labels.csv", index_col="lead_id")
emva/eval/ground_truth_reference.py:154:        f"Time to close parsed from ground_truth.md: median {ttc.median_days:g} days, log-sd {ttc.log_sd:g}, "
emva/eval/ground_truth_reference.py:177:    ap = argparse.ArgumentParser(prog="python -m emva.eval.ground_truth_reference")
```

Only `emva/eval/` lines plus the known `emva/value.py:19` constant (Phase 2, plan 2.4).
`label_study.py:14` is a docstring pointer; the test asserts it is not an import.

## Appendix A: `make report`

Data: `data/v1`. baseline = frozen `baseline/emva_score.py`; candidate = `emva` pipeline trained with `horizon H=120` labels; status quo = `status_quo_rules.json` reconstructed. Every model is scored on the same rows under two test definitions. AUC CI: percentile bootstrap, 1000 resamples, seed 0. Top-20% capture ranks by p (status quo: by its value) or by p×value.

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
| candidate | 0.812 [0.788, 0.836] | 0.1022 | 0.568 | 0.723 | 0.812 |
| status quo | 0.639 [0.605, 0.673] | n/a | 0.392 | 0.454 | 0.454 |

##### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | -0.001 | [-0.005, +0.003] | 0.616 |
| status quo | -0.174 | [-0.209, -0.141] | 0.000 |

##### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 246 | 0.009 | 0.016 | 0.011 | 0.012 |
| 2 | 245 | 0.020 | 0.012 | 0.024 | 0.016 |
| 3 | 245 | 0.033 | 0.016 | 0.041 | 0.020 |
| 4 | 245 | 0.050 | 0.053 | 0.063 | 0.053 |
| 5 | 246 | 0.075 | 0.077 | 0.092 | 0.077 |
| 6 | 245 | 0.109 | 0.135 | 0.131 | 0.118 |
| 7 | 245 | 0.150 | 0.131 | 0.182 | 0.151 |
| 8 | 245 | 0.213 | 0.176 | 0.252 | 0.171 |
| 9 | 245 | 0.316 | 0.327 | 0.361 | 0.339 |
| 10 | 246 | 0.511 | 0.492 | 0.560 | 0.476 |

Status quo has no probability, so it has no Brier score or calibration.

##### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 754 | 103 | 0.800 | 0.799 | 0.641 |
| 2026-06 | 665 | 93 | 0.797 | 0.798 | 0.631 |
| 2026-07 | 541 | 88 | 0.833 | 0.827 | 0.602 |
| 2026-08 | 371 | 55 | 0.813 | 0.817 | 0.664 |
| 2026-09 | 122 | 13 | 0.920 | 0.914 | 0.772 |

##### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 2453 | 346 | 17167 | 45562 | 131.8× | 11.8% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 2453 | 426 | 17692 | 44480 | 104.3× | 10.9% |
| candidate | all scored leads | 9311 | 454 | 18638 | 44480 | 98.1× | 10.6% |
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
| candidate | 0.778 [0.726, 0.827] | 0.1263 | 0.463 | 0.614 | 0.762 |
| status quo | 0.644 [0.571, 0.717] | n/a | 0.400 | 0.501 | 0.501 |

##### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | -0.001 | [-0.013, +0.010] | 0.976 |
| status quo | -0.134 | [-0.199, -0.069] | 0.000 |

##### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 46 | 0.008 | 0.022 | 0.010 | 0.000 |
| 2 | 46 | 0.020 | 0.000 | 0.025 | 0.022 |
| 3 | 46 | 0.038 | 0.022 | 0.045 | 0.022 |
| 4 | 45 | 0.060 | 0.111 | 0.076 | 0.133 |
| 5 | 46 | 0.090 | 0.174 | 0.108 | 0.065 |
| 6 | 46 | 0.124 | 0.152 | 0.149 | 0.283 |
| 7 | 45 | 0.171 | 0.178 | 0.204 | 0.156 |
| 8 | 46 | 0.237 | 0.239 | 0.280 | 0.261 |
| 9 | 46 | 0.357 | 0.391 | 0.390 | 0.391 |
| 10 | 46 | 0.555 | 0.457 | 0.596 | 0.413 |

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
| candidate | test set | 458 | 522 | 19945 | 32904 | 63.0× | 9.8% |
| candidate | all scored leads | 9311 | 454 | 18638 | 44480 | 98.1× | 10.6% |
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
| all leads labelled by legacy rules (train + test) | 8041 | 15.0% | 24.5% | 23.9% | 16.5% | 27.5% | 26.5% |
| (a) legacy test set | 2453 | 5.7% | 9.4% | 9.0% | 10.7% | 19.2% | 17.6% |
| all mature leads, every label_source | 6278 | 19.2% | 30.6% | 30.0% | 19.2% | 30.6% | 30.0% |
| mature leads created on or after 2026-05-01, every label_source | 650 | 21.7% | 29.2% | 27.7% | 21.7% | 29.2% | 27.7% |

#### Baseline script output

- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, which fills blank deal values with its predicted value, printed:

```
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795
```

## Appendix B: `python -m emva.eval.label_study` (1,000 refits, 1,000 resamples)

Data: `data/v1`.

#### 1. Company-size weights with bootstrap CIs

Log-odds vs band 1-10. 95% percentile CI from 1000 refits of the formula model on training rows resampled with replacement (seed 0).

| labels | train n | band=11-50 | band=51-200 | band=201-1000 | band=1000+ |
|---|---|---|---|---|---|
| legacy | 5588 | 0.193 [-0.064, 0.417] | 1.078 [0.851, 1.328] | 0.873 [0.591, 1.162] | 0.703 [0.331, 1.060] |
| horizon H=120 | 4049 | 0.235 [-0.008, 0.500] | 1.146 [0.894, 1.412] | 0.971 [0.680, 1.275] | 0.419 [0.046, 0.806] |

#### 2. Label definitions and flags

Training set = labelled rows created before 2026-05-01 (mature rows only for horizon labels). Weights are the fitted log-odds (weights.csv). AUC is for the model trained with each definition, on the two frozen test sets (2453 and 458 leads), with a 1000-resample percentile CI.

| labels | train n | train wins | train win rate | band=201-1000 | band=1000+ | AUC (a) legacy test | AUC (b) mature test |
|---|---|---|---|---|---|---|---|
| legacy | 5588 | 847 | 15.2% | 0.873 | 0.703 | 0.814 [0.789, 0.836] | 0.778 [0.726, 0.827] |
| horizon H=120 | 4049 | 752 | 18.6% | 0.971 | 0.419 | 0.812 [0.788, 0.836] | 0.778 [0.726, 0.827] |
| horizon H=120 +ghosted | 5115 | 752 | 14.7% | 0.942 | 0.570 | 0.813 [0.788, 0.836] | 0.779 [0.727, 0.827] |
| horizon H=120 +stalled-as-lost | 4562 | 752 | 16.5% | 0.886 | 0.387 | 0.812 [0.788, 0.836] | 0.778 [0.727, 0.828] |
| horizon H=120 +ghosted +stalled-as-lost | 5628 | 752 | 13.4% | 0.877 | 0.513 | 0.813 [0.788, 0.836] | 0.779 [0.728, 0.827] |

Labelled win rate in training by company size (`band` feature), computed from each definition's labels:

| labels | 1-10 | 11-50 | 51-200 | 201-1000 | 1000+ |
|---|---|---|---|---|---|
| legacy | 6.2% (n=2357) | 12.7% (n=1260) | 27.8% (n=944) | 27.7% (n=692) | 25.7% (n=335) |
| horizon H=120 | 7.8% (n=1615) | 15.7% (n=922) | 33.2% (n=723) | 33.5% (n=514) | 25.1% (n=275) |
| horizon H=120 +ghosted | 5.8% (n=2155) | 12.5% (n=1156) | 27.5% (n=872) | 27.6% (n=623) | 22.3% (n=309) |
| horizon H=120 +stalled-as-lost | 6.9% (n=1823) | 14.0% (n=1035) | 29.8% (n=806) | 29.2% (n=589) | 22.3% (n=309) |
| horizon H=120 +ghosted +stalled-as-lost | 5.3% (n=2363) | 11.4% (n=1269) | 25.1% (n=955) | 24.6% (n=698) | 20.1% (n=343) |

#### 3. Horizon sensitivity

Default flags (ghosted excluded, stalled censored). Training rows must be mature, so a longer H trains on fewer, older leads; `mature test n` is the size of the frozen-style test set that H would allow.

| H (days) | train n | train wins | rate 201-1000 | rate 1000+ | band=201-1000 | band=1000+ | mature test n |
|---|---|---|---|---|---|---|---|
| 90 | 4049 | 678 | 30.5% (n=514) | 20.4% (n=275) | 0.946 | 0.249 | 1048 |
| 120 | 4049 | 752 | 33.5% (n=514) | 25.1% (n=275) | 0.971 | 0.419 | 458 |
| 150 | 3973 | 781 | 34.6% (n=508) | 29.4% (n=269) | 0.958 | 0.616 | 0 |
| 180 | 3461 | 697 | 34.2% (n=441) | 30.1% (n=239) | 0.890 | 0.575 | 0 |
| 240 | 2279 | 461 | 35.6% (n=281) | 31.1% (n=161) | 0.908 | 0.482 | 0 |

Share of mature training leads that were eventually Won whose Won date is after H (these count as 0 under won_within_H), by company size:

| H (days) | 1-10 | 11-50 | 51-200 | 201-1000 | 1000+ |
|---|---|---|---|---|---|
| 90 | 25.2% | 20.0% | 13.4% | 18.2% | 34.9% |
| 120 | 14.3% | 9.4% | 8.4% | 10.4% | 19.8% |
| 150 | 9.8% | 4.4% | 5.4% | 7.4% | 7.1% |
| 180 | 6.2% | 3.5% | 3.1% | 6.2% | 2.7% |
| 240 | 2.5% | 0.0% | 2.7% | 1.0% | 2.0% |

## Appendix C: `python -m emva.eval.ground_truth_reference` (1,000 refits)

#### Mean planted close probability (p_close_true) by company size (band feature)

| population | 1-10 | 11-50 | 51-200 | 201-1000 | 1000+ |
|---|---|---|---|---|---|
| training rows labelled by legacy | 8.0% (n=2357) | 18.2% (n=1260) | 35.9% (n=944) | 39.4% (n=692) | 35.5% (n=335) |
| training rows labelled by horizon H=120 | 8.5% (n=1615) | 18.9% (n=922) | 36.9% (n=723) | 39.5% (n=514) | 37.0% (n=275) |
| all cleaned leads created before 2026-05-01 (every label_source) | 8.0% (n=2363) | 18.3% (n=1269) | 36.1% (n=955) | 39.5% (n=698) | 36.3% (n=343) |

#### R1: true 120-day size effects vs learned weights

Time to close parsed from ground_truth.md: median 40 days, log-sd 1.09, ×1.8 for 1000+. P(close within 120 d | Won) = 0.843 (other bands), 0.680 (1000+). p120 = p_close_true × that, per lead, using the true band. Rows: the horizon training set (4049 leads; the legacy weights come from its own 5588 training rows).

Derivation check (horizon training rows, `band` feature):

| band | rows | mean p_close_true (eventual) | derived p120 | observed won within 120 d (horizon label) |
|---|---|---|---|---|
| 1-10 | 1615 | 8.5% | 7.2% | 7.8% |
| 11-50 | 922 | 18.9% | 16.0% | 15.7% |
| 51-200 | 723 | 36.9% | 31.1% | 33.2% |
| 201-1000 | 514 | 39.5% | 33.2% | 33.5% |
| 1000+ | 275 | 37.0% | 25.2% | 25.1% |

True effects are log-odds relative to band 1-10. *Conditional* = the formula model (same design, same rows, same regularisation) fitted to p120 as soft labels: the weight a noise-free 120-day label would give, holding the other features fixed, which is what the learned weights estimate. *Marginal* = logit(mean p120 in band) − logit(mean p120 in 1-10) on the same rows, as the ruling literally states; it includes every feature correlated with size (free email, missing company, spend), so it is not comparable with a multivariable weight. The last column fits p_close_true the same way, as a check that the conditional method recovers the planted eventual effects (11-50 +0.5, 51-200 and 201-1000 +1.3, 1000+ +1.0) up to feature mismatch and regularisation.

Learned CIs: 1000 bootstrap refits, seed 0. R1 pass = the horizon CI contains the true effect, or the horizon point is closer to it than the legacy point.

| band | true 120-day effect, conditional | true 120-day effect, marginal | legacy weight [95% CI] | horizon weight [95% CI] | R1 (conditional) | R1 (marginal) | eventual-close effect, conditional (check) |
|---|---|---|---|---|---|---|---|
| 11-50 | 0.378 | 0.898 | 0.193 [-0.064, 0.417] | 0.235 [-0.008, 0.500] | pass (in CI) | pass (closer) | 0.379 |
| 51-200 | 1.099 | 1.766 | 1.078 [0.851, 1.328] | 1.146 [0.894, 1.412] | pass (in CI) | pass (closer) | 1.172 |
| 201-1000 | 1.043 | 1.862 | 0.873 [0.591, 1.162] | 0.971 [0.680, 1.275] | pass (in CI) | pass (closer) | 1.105 |
| 1000+ | 0.532 | 1.471 | 0.703 [0.331, 1.060] | 0.419 [0.046, 0.806] | pass (in CI) | fail | 0.883 |
