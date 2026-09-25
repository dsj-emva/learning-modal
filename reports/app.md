# Phase 8: Keel, the hosted internal tool (branch `app-ui`)

Keel is the Streamlit app an EMVA colleague uses to read model results, upload CRM exports and train, and
score a single lead. Entry point `streamlit run app/main.py` (`make app` locally on :8501, `scripts/serve.sh`
in the image). Module ownership is in `docs/ARCHITECTURE.md` section 8. All numbers in the screenshots come
from the bundled v1 sample: **on simulated data**, not for external use.

## Sign in

![Sign in](app/login.png)

A shared password from `APP_PASSWORD`, compared with `hmac.compare_digest`; a wrong password waits 1 s. If the
variable is unset or blank the app shows "APP_PASSWORD is not set" and never opens. Login lives in the
Streamlit session: "Sign out" clears it, and reloading the browser tab asks again (sessions are per
websocket connection).

## Model results

![Model results](app/results.png)

- **Run picker** (registry, newest first: date, dataset, options, AUC, status) and a **test-set switch**: the
  frozen mature test set (horizon labels, the headline since Phase 1) or the legacy one (the POC's labels), the
  two definitions of `emva.eval.report.frozen_test_labels`.
- **KPI strip**: AUC with its 95% bootstrap CI (1,000 resamples, seed 0), Brier, share of wins and of recorded
  revenue in the top 20%, each with its delta against the status-quo rules.
- **Standard table** (model vs status quo), **calibration by decile** (predicted vs observed, diagonal),
  **AUC by month** with bootstrap error bars for months of 30+ leads, the **scorecard** as a signed bar chart
  (accent = raises the score, warm red = lowers it) and as a sortable table with strength bars and odds
  multipliers, and the **value per lead** stats (p1 / median / p90 / p99 / max, max/median, top-1% share of
  `value_at_submit` and `value_formula`) with a log-scale histogram.
- Downloads of `scores.csv`, `weights.csv` and the run's standard report, which is also rendered in an expander.

Every frame is computed by `app/results.py` from the run's `scores.csv` / `weights.csv` with `emva.eval` code
(`metrics.calibration_by_decile`, `metrics.auc_by_month`, `metrics.top_share`, `bootstrap.auc_ci`,
`value_report.scale_stats`, `status_quo.status_quo_value`); nothing is parsed out of `report.txt`, which is
kept only for display. The baseline column lives in the report (it needs the frozen baseline script run).

## Upload & train

![Upload & train: validation](app/train.png)

1. Five uploaders (leads, CRM history, companies, people, optional status-quo rules). A file dropped in the
   wrong slot is flagged; a file named `ground_truth*` is refused by name and never parsed.
2. **Validate** (`app/validation.py`) checks what the pipeline reads: required columns per file, ISO datetimes,
   numeric deal values, one JSON object of `answers` per row with form-option values, known CRM stages, at most
   one Won row per lead, unique company domains, referential integrity both ways, at least 200 leads. Errors
   (red) block saving; notes (amber) do not. Preview: rows, date range, won leads, leads by current CRM stage.
3. **Save** under a slug name (`DATA_DIR/datasets/<name>/`, written aside then renamed).
4. **Train**: dataset picker (stored datasets plus the read-only bundled sample), labels (horizon default,
   legacy = reproduces the frozen POC) and features (v2 default, legacy), and the pre-training summary computed
   by `emva.pipeline.build` + `emva.labels.split_masks` (leads after cleaning, training and test sizes with wins,
   the frozen mature test set, labels by `label_source`). "Train model" registers a run and returns at once;
   the log streams live (a fragment polls the log file every 1.5 s) behind a status pill.

![Training in progress](app/train-running.png)
![Training finished](app/train-done.png)

**How training is invoked.** `app.training.start_training` spawns `python -m app.job`, which runs exactly the
local commands with the run's options, output appended to `runs/<run_id>/train.log`:

```sh
python -m emva --data <dataset> --out <run dir> --label-mode horizon --feature-set v2   # scores, weights, model.joblib
python -m emva.eval.report --data <dataset> --label-mode horizon --feature-set v2 > <run dir>/report.txt
```

The report runs only when the dataset has `status_quo_rules.json`; a failed report leaves the run
`succeeded` with the failure in the log. The job records the outcome in `registry.json` itself, so a run
finishes with no browser open; a run whose job process disappears (server restart) is marked `failed` when
next shown. On the full v1 sample: training ~4 s, report ~40 s.

## Score a lead

![Score a lead](app/score.png)

A model picker (completed runs with `model.joblib`) and a form generated from `emva.scoring.submit_time_fields()`,
grouped as Form answers / Contact / Attribution / Session, prefilled with a plausible lead whose company is the
first row of the dataset's `companies.csv`. Every session field is filled with a typical visit, because a blank
one makes v2 set `session_missing=yes`; the form says so. The result card shows P(close) with the training win
rate for context, the expected deal value if won, the value sent at submit with the fitted transform spelled out
(cap, compression, floor), and bot / duplicate flags (a lead scored alone is never a duplicate; bots are flagged
and still scored). Below it, the points breakdown: starting score, signals and total, and a signed bar per
active signal. `UnknownLevelError` and bad input become a plain message. Scoring goes through `app/scoring.py`
-> `emva.scoring.lead_from_form` / `score_leads` / `points_breakdown`; no feature is re-implemented.

**ulp caveat (ADR 0018).** A lead scored alone can differ from the batch run's score in the last bits (BLAS
kernels depend on the batch shape): up to 64 ulp. `tests/test_app_scoring.py` compares with that tolerance.

## Design

Name and identity: **Keel** ("lead scoring by EMVA"), a small glyph plus the name set in Instrument Serif.
Warm paper background (#F6F4EF), white surfaces with hairline borders instead of shadows, one accent (deep teal
#0F5C5A) used for meaning only, muted warm red (#B5543C) for negatives, amber for warnings. Instrument Sans for
the interface, Instrument Serif for page titles, IBM Plex Mono for every number. 8 px spacing. Streamlit's menu,
footer, deploy button and default page nav are hidden; the sidebar is a custom wordmark + page links. CSS lives
in `app/theme.py::inject_css`, HTML components (KPI cards, pills, validation cards, result card, tables, empty
states) in `app/components.py`, the Plotly template in `app/charts.py`, the widget theme in
`.streamlit/config.toml`. Material icons only; no emojis.

## Limits

- One service, one process: training runs as a subprocess on the same machine (a 10k-lead run is seconds, the
  report under a minute). Concurrent trainings are allowed but not queued.
- One shared password; no per-user accounts, no audit of who trained or scored what. The registry records runs,
  not people.
- Labels use the pipeline's fixed snapshot date (`AS_OF` 2026-09-24) and test boundary (2026-05-01): data from
  another period trains, but its test sets can be empty (the results page says so).
- Uploaded datasets and runs are never deleted from the UI (delete on the volume if needed).
- Login does not survive a browser reload.

## Tests

`tests/test_app_storage.py` (17), `test_app_validation.py` (18), `test_app_training.py` (9, one end-to-end job on
a 1,000-lead v1 subsample, shared at session scope), `test_app_scoring.py` (9, including the password gate),
`test_app_smoke.py` (6, Streamlit `AppTest`: unset password refuses, wrong password blocked, every page renders,
a lead is scored, sign out). About 35 s together.

## Orchestrator decisions

_To be filled at review._
