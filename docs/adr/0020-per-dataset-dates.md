# 0020. Each dataset carries its own snapshot date and test boundary in `dataset.json`; the constants stay the default

- Status: Proposed
- Date: 2026-09-26
- Source: Phase 9 (a), generic dataset converter (`emva/ingest/`, `emva/dataset_meta.py`); branch
  `claude/epic-edison-onl83z`

## Context

`emva.constants.AS_OF` (2026-09-24) and `TEST_FROM` (2026-05-01) are the snapshot date and frozen train/test
boundary of the synthetic data (ground rule 4). Every label rule reads the CRM state at `AS_OF` (stalled = idle 90+
days at `AS_OF`, mature = `created_at + H <= AS_OF`, `value_at_close` known at `AS_OF`), and the split is
`created_at < TEST_FROM`. A dataset converted from another source (Phase 9: Olist 2017-2018, a CRM export 2016-2017,
hotel bookings 2015-2017) has no lead created after 2018, so with the constants every lead is mature, every open
deal is stalled and the test set is empty. The dates are a property of the dataset, not of the code.

## Decision

1. A dataset directory may hold `dataset.json` with `as_of` (ISO date or datetime; naive = UTC) and `test_from`
   (`YYYY-MM-DD`); the converter also writes `"emva_dataset_format": 1`, counts and a coverage table.
   `emva.dataset_meta.dataset_dates(data)` returns `(as_of, test_from)` from it, else `(AS_OF, TEST_FROM)`;
   `read_dataset_meta(data)` returns the whole file (or None) for the app. A `dataset.json` that is not an object
   with both dates, has a malformed date, another format version, or `test_from` after `as_of` is refused with a
   `ValueError` naming the file, never read as "no dates". The name is reserved for the converter: the app's own
   dataset record (called `dataset.json` before Phase 9) moves to `keel_meta.json` on the app side.
2. `emva.ingest.convert` writes `dataset.json` from the mapping's `as_of` / `test_from`; the mapping is where the
   split is chosen and frozen for that dataset (ground rule 4 applied per dataset: never moved to improve a number).
3. `emva.pipeline.run` (and so `python -m emva`), and `emva.eval.report` read the dates through `dataset_dates`:
   labels, `LabelConfig.eligible`, `value_at_close` and the report's frozen test definitions use `as_of`; the split
   uses `test_from`. `run(test_from=..., as_of=...)` still override explicitly; `PipelineResult` records both.
   Every function that took the constants as defaults keeps them as defaults (`frozen_test_labels(L)`,
   `label_sections`, `build(L, ...)`), so existing callers are unchanged.
4. data/v1, data/v2 and the bundled sample have no `dataset.json`: `make baseline` gives AUC 0.814, Brier 0.1006,
   top-20% wins 0.571, revenue 0.795 and the canonical weights, and `make report` on data/v1 and data/v2 is byte
   identical to before this change.

## Consequences

- The Phase 4 evaluation modules (`emva.eval.rolling`, `hardening`, `calibration_decay`, `feature_selection`,
  `label_study`, `phase2_study`, `ground_truth_reference`) and `emva.troas` still use the constants; they run on
  data/v1 and data/v2 only. Running them on a converted dataset needs the same threading first.
- `app.validation` still warns about leads dated after `AS_OF`; for a converted dataset that warning uses the wrong
  date and never fires. It is the app's concern.
- The report on a converted dataset omits the frozen baseline (it hard-codes the v1 dates and format; ADR 0022).
