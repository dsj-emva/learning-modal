# 0022. The results page shows model and status quo without a baseline row when the frozen script cannot apply

- Status: Proposed
- Date: 2026-09-26
- Source: Phase 9 orchestrator decision 6 (amends R18 / ADR 0019 decision 1); branch `phase9-app-base`:
  `app/results.py` (`baseline_scores`, `kpi_reference`), `app/views/results.py`, `app/storage.py`

## Context

R18 (ADR 0019) made the results page show the full standard table: the frozen baseline, this run's model and the
status quo, plus the paired AUC comparison against the baseline. Phase 9 lets Keel train on datasets converted
from foreign CSVs (`emva/ingest/`), which carry a `dataset.json` with their own `as_of` / `test_from` and a
coverage table. The frozen baseline script (`baseline/emva_score.py`, never edited, ground rule 1) hard-codes
AS_OF 2026-09-24, TEST_FROM 2026-05-01 and the v1 file format. On a converted dataset it either fails or, worse,
runs and scores rows against the wrong period, so its row would look like a comparison and mean nothing. Before
this ruling the page already dropped the baseline row when the script exited non-zero, but only that case, and the
KPI strip fell back to the baseline whenever there were no rules.

## Decision

1. **No baseline row when the script cannot apply.** `app.results.baseline_scores` returns no scores, with a
   reason the page shows, when:
   - the dataset carries `dataset.json` (a converted dataset, `app.storage.is_converted`): the script is not run
     at all (`BASELINE_CONVERTED`);
   - the script fails, its `scores.csv` is missing or unreadable, or it leaves a test lead unscored
     (`BASELINE_FAILED`, logged as a warning).
2. **The rest of the page still renders.** The standard table shows this run's model and, when the dataset has
   `status_quo_rules.json`, the status quo (converted datasets usually have none; that note stays as in R18). The
   paired comparison against the baseline is hidden and the reason is shown in its place.
3. **KPI strip.** It compares the model with the status quo, else with the baseline, else with nothing (plain
   figures, no delta chips). All four combinations of baseline yes/no x rules yes/no render without errors.
4. **Storage.** A dataset directory may hold `mapping.toml` (the confirmed mapping it was converted with) and
   `dataset.json` next to the training files; `save_dataset` accepts exactly these two extra names. The store's
   own record moves from `dataset.json` to `keel_meta.json`, and `init_root` renames legacy records (only a file
   that parses as a store record), so `dataset.json` means one thing only.

## Consequences

- A converted dataset is never compared with the frozen POC in the app; its judgement is model vs status quo
  (when rules exist) and the model's own CIs, calibration and stability.
- The model's row is unchanged by the baseline's absence (the same `headline_frame` row; tested).
- Per-dataset test dates are not yet threaded through the page: `training_base_rate` takes `test_from`
  (default `TEST_FROM`) so the wiring after `emva/ingest/` lands is a parameter, not a restructure.
- Pre-Phase 9 data roots are migrated in place on first start; no manual step on the Railway volume.
