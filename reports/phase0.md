# Phase 0: freeze and instrument

Branch `phase0-freeze-instrument`. Plan items 0.2 to 0.5 (0.1 was already done).

## What was done

- **0.2** `git mv LearningPYApp baseline` (history kept; the first commit already holds the
  untouched original). `make baseline` (`scripts/check_baseline.py`) runs the frozen
  `baseline/emva_score.py` and `python -m emva` on `data/v1` into temp dirs and fails (exit 1,
  every mismatch listed) unless both print AUC 0.814, Brier 0.1006, top-20% wins 0.571 and
  revenue 0.795 and both `weights.csv` match `baseline/weights.csv`. It also requires the emva
  run's `weights.csv` and `scores.csv` to be byte-identical to the baseline run's. Verified to
  fail on a perturbed dataset (first 8,999 leads only: AUC 0.806, exit 2 from make).
- **0.3** `emva/` split with no behaviour change: `constants.py` (every constant in one place),
  `io.py` (load + bot/duplicate cleaning), `labels.py`, `features.py`, `design.py`,
  `model.py`, `value.py`, `pipeline.py` (orchestration) and `__main__.py`
  (`python -m emva --data data/v1 --out DIR`, same arguments and output as the baseline).
  Byte-identical outputs to the baseline, in plain and `--context` mode (tested).
  `context/agent.py` is `context_agent.py` moved; `context/contract.py` names the current
  implicit JSON/CSV contract; `context/features.py` holds the existing clip + logit transform.
- **0.4** `emva/eval/`: `bootstrap.py` (percentile CI, 1000 resamples, paired AUC difference with
  bootstrap p), `status_quo.py`, `metrics.py` (top-k capture, decile calibration, AUC by month,
  value scale), `regression.py` (baseline checks) and `report.py` (`python -m emva.eval.report`).
  `rolling.py` and `ceiling.py` are left to Phase 4.
- **0.5** `requirements.txt` pinned to the verified versions; `Makefile` with `baseline`,
  `report`, `test` (pytest, then `compileall`) and `experiment NAME=...`.
- `README.md` with setup and run instructions.

## `make baseline`

```
".venv/bin/python" scripts/check_baseline.py --data "data/v1"
baseline/emva_score.py:
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795

baseline weights.csv vs committed: values equal, tie order differs for rows ['channel=organic_direct', 'business_hours=wkday_9-18']

python -m emva:
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795

emva weights.csv vs baseline run: byte-identical
emva scores.csv vs baseline run: byte-identical

PASS: baseline and emva/ both reproduce AUC 0.814, Brier 0.1006, top-20% wins 0.571, revenue 0.795 and the frozen weights.
```

## Standard report (`make report`)

The report compares three score sets on the frozen test set: **baseline** = the frozen script,
run fresh; **candidate** = the current `emva` pipeline (identical in Phase 0); **status quo** =
`status_quo_rules.json` reconstructed.


Data: `data/v1`. Frozen test set: 2453 labelled leads created on or after 2026-05-01 (352 won). AUC CI: percentile bootstrap, 1000 resamples, seed 0. Top-20% capture ranks by p (status quo: by its value) or by p×value.

### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.814 [0.789, 0.836] | 0.1006 | 0.571 | 0.742 | 0.796 |
| candidate | 0.814 [0.789, 0.836] | 0.1006 | 0.571 | 0.742 | 0.796 |
| status quo | 0.639 [0.605, 0.673] | n/a | 0.392 | 0.454 | 0.454 |

### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.000 | [+0.000, +0.000] | 1.000 |
| status quo | -0.174 | [-0.209, -0.141] | 0.000 |

### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 246 | 0.009 | 0.016 | 0.009 | 0.016 |
| 2 | 245 | 0.020 | 0.012 | 0.020 | 0.012 |
| 3 | 245 | 0.033 | 0.016 | 0.033 | 0.016 |
| 4 | 245 | 0.050 | 0.053 | 0.050 | 0.053 |
| 5 | 246 | 0.075 | 0.077 | 0.075 | 0.077 |
| 6 | 245 | 0.109 | 0.135 | 0.109 | 0.135 |
| 7 | 245 | 0.150 | 0.131 | 0.150 | 0.131 |
| 8 | 245 | 0.213 | 0.176 | 0.213 | 0.176 |
| 9 | 245 | 0.316 | 0.327 | 0.316 | 0.327 |
| 10 | 246 | 0.511 | 0.492 | 0.511 | 0.492 |

Status quo has no probability, so it has no Brier score or calibration.

### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 754 | 103 | 0.800 | 0.800 | 0.641 |
| 2026-06 | 665 | 93 | 0.797 | 0.797 | 0.631 |
| 2026-07 | 541 | 88 | 0.833 | 0.833 | 0.602 |
| 2026-08 | 371 | 55 | 0.813 | 0.813 | 0.664 |
| 2026-09 | 122 | 13 | 0.920 | 0.920 | 0.772 |

### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 2453 | 346 | 17167 | 45562 | 131.8× | 11.8% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 2453 | 346 | 17167 | 45562 | 131.8× | 11.8% |
| candidate | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| status quo | test set | 2453 | 96 | 1440 | 1872 | 19.5× | 6.7% |
| status quo | all scored leads | 9311 | 96 | 1440 | 1872 | 19.5× | 6.7% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

### Notes

- status quo: 203 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.397.
- status quo: 203 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454.
- status quo: 203 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454.
- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, which fills blank deal values with its predicted value, printed:

```
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795
```

The status quo hits the targets exactly: AUC 0.639, top-20% wins 0.392, top-20% revenue 0.454.
The baseline AUC CI [0.789, 0.836] matches the figure quoted in plan Phase 2.

### How the status quo is reconstructed

Value at submit = `base_value_by_form[variant]` × 1.2 if UK/US × 1.3 if channel is LinkedIn ×
tier multiplier (A 2.0 at 55+ points, B 1.0 at 30+, C 0.4). Points = job title (substring
match in the listed keyword order, first match wins; 10 if there's a title but no keyword; 0 if
blank) + typed company-size dropdown + pages visited (first threshold met; Lead Ads have none,
0). `stage_values` are post-submit CRM uploads and are not used for ranking. First-match and
max-match keyword scoring give the same numbers on v1.

Two things were needed to hit 0.454, and both are now report conventions:

1. **Revenue = recorded deal value** (blank on a Won row = 0). With the baseline script's
   definition (blank filled with its own `deal_value_hat`) the status quo gets 0.449. The report
   uses recorded value for every model because the imputed definition moves with each
   candidate's value model. Under it the baseline gets 0.796 by p×value (0.795 under its own
   definition, printed in the report notes, and still asserted by `make baseline`).
2. **Ties.** The status quo has 32 distinct values and 203 test leads tie at the top-20% cut.
   Like the baseline, the report ranks with numpy's default (unstable) argsort, which gives
   0.392 / 0.454 here. A stable sort would give 0.409 / 0.460; the tie-averaged expectation,
   which doesn't depend on sort order, is 0.397 / 0.454. The report prints this footnote
   automatically for any score with ties at the cut.

## Tests

`make test`: **111 passed** (about 14 s), then `python -m compileall -q emva scripts baseline tests`
is clean. Every commit on the branch passes its own tests.

| file | covers |
|---|---|
| `test_labels.py` | Won/Lost, stalled and ghosted at 89 / 89.99 / 90 days, age vs idle, split masks |
| `test_features.py` | `text_cat`, `seniority`, channel mapping, reference-level fallbacks, bucket thresholds |
| `test_io.py` | bot and duplicate rules, stage normalisation, enrichment join |
| `test_design.py` | reference levels dropped, the 47-column set, `extra` join |
| `test_model_value.py` | scorecard points, lognormal correction, realised revenue |
| `test_bootstrap.py` | CI contains the point, width shrinks with n, seeds, paired identical/better scores |
| `test_status_quo.py` | keyword order, page points, six hand-built leads across all tiers and rules |
| `test_metrics_regression.py` | top-k and tie averaging, calibration, month AUC, value scale, weights compare |
| `test_context.py` | lead card, cache hit without network, logged retries, contract, context logit |
| `test_integration.py` | emva on v1 = baseline metrics and weights, `--context` byte-equal, CLI, status-quo targets, report |

## Pandas pin decision and the `weights.csv` check

Option (a) (find a pin that makes the file byte-identical) didn't work. The regenerated file has
the same values, but the two rows with log-odds 0.16 (`business_hours=wkday_9-18`,
`channel=organic_direct`) come out swapped under every combination tried on this arm64 Mac:
pandas 3.0.6 + numpy 2.4.6, pandas 2.2.3 + numpy 2.4.6, and pandas 2.2.3 + numpy 1.26.4 +
scikit-learn 1.5.2. The order comes from `sort_values` using numpy's unstable quicksort on
tied keys, so it depends on numpy's CPU-specific sort kernels (the committed file was probably
made on x86), not on the pandas version. So pandas stays at the installed 3.0.6.

Option (b): `make baseline` compares `weights.csv` canonically: same row names and columns,
values equal to 3 dp, row order ignored (`emva/eval/regression.py`), and prints which rows
moved. The stricter byte-identity check is kept where it makes sense, between the emva run and
the baseline run on the same machine.

## Ground rule 2 check (expected to fail until Phase 2)

```
$ grep -rn "0.45\|ground_truth" emva/
emva/value.py:14:# Lognormal mean correction exp(sd**2 / 2) uses a residual sd of 0.45 on log deal value.
emva/value.py:15:# This number is the generator's planted noise sd (data/v1/ground_truth.md), not something
emva/value.py:19:DEAL_LOG_RESIDUAL_SD_FROM_GENERATOR = 0.45
```

The constant is kept in Phase 0 for exact reproduction, named for where it comes from, with
`TODO(plan 2.4)`. Nothing under `emva/` reads a ground-truth file.

## Baseline issues noticed (left as they are)

- `top20_revenue` in the baseline fills blank deal values with the model's own prediction, so
  it isn't comparable across value models (the report uses recorded value, see above).
- The value-scale figures in plan 3.2 (max/median 116×, top-1% share 11%) don't reproduce. I
  get 121.7× / 11.5% on all 9,311 scored leads and 131.8× / 11.8% on the test set.
- `context_agent.py` joins enrichment without `.str.strip()` on the domain (`emva_score.py`
  strips), and it shows the agent enrichment bands (plan 6.2 removes them).
- `design()` silently keeps a reference level's siblings when the reference level is absent
  from the data (no column dropped). This never happens on v1.
- Scorecard row order for tied log-odds depends on the platform (see above).
- Plan ground rule 6 still applies: `scripts/generate_data.py` is not in the repo.

## Deviations from the plan

- `git mv` instead of copy for 0.2 (agreed with the orchestrator).
- Extra modules beyond the target layout: `emva/constants.py` (one place for constants),
  `emva/pipeline.py` + `emva/__main__.py` (entry point), `emva/eval/metrics.py` (shared metrics,
  kept apart from `report.py` to avoid a pipeline ↔ report import cycle), `emva/eval/regression.py`
  (baseline checks shared by `make baseline` and tests), `scripts/check_baseline.py`.
- `make baseline` also asserts Brier 0.1006 on top of the plan's three metrics.
- `requirements.txt` also pins scipy (it drives the LR solver) and requests (the unchanged
  agent's HTTP client). `anthropic` is pinned but unused until Phase 6.
- The agent now logs each failed attempt instead of swallowing it silently (outputs unchanged),
  and its `--data`/`--brief` defaults point at `data/v1` and `baseline/business_brief.md`
  instead of `/mnt/project`.
