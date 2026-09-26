# Phase 10 (a): generic feature set (branch `phase10-generic-features`)

Phase 9 showed that the fixed design schema (ADR 0009: 39 columns from form answers, UTM tags, email, on-site session
telemetry and company enrichment) barely covers foreign sources. On public data, Olist fills 3 of 39 signals
(horizon test AUC 0.553 [0.530, 0.574]); CRM opportunities and hotel bookings fill 0 of 39 (AUC 0.500 exactly).
Phase 10 (a) builds the deferred **generic feature set**: the v2 design plus extra source columns that a mapping
declares and a person confirms as known at submit time. The encoding is fitted on the training leads only and frozen
in the model bundle. Part (b), the Keel UI, comes later.

ADR 0024 (Proposed) records the decisions; it amends ADR 0009 (unknown levels raise: not for extras) and ADR 0017
(decision 2: no customer-specific columns: allowed as declared extras, outside the pooled part).

## Pre-registered acceptance criterion (ground rule 5)

Written and committed **before** any Kaggle dataset was converted with extras or trained with the generic feature
set. Nothing below (bins, minimum level count, the feature lists in `mappings/*.toml`, H, flags, dates) is changed
after seeing a test number. A dataset that fails, fails, and this report says so.

**Per Kaggle dataset (olist_funnel, crm_opportunities, hotel_bookings): PASS iff, on the horizon test set (b) of the
standard report, the paired bootstrap AUC difference generic − v2 (1000 resamples, seed 0, both models trained by
`emva.pipeline.run` on the same data, labels and split) has a 95% CI entirely above 0, AND the generic design passes
the collinearity check (no pair of design columns with |corr| > 0.95 on the training rows, plan 2.7).**

Fixed before the run:

- Encoding constants: `GENERIC_MIN_LEVEL_COUNT = 30`, `GENERIC_NUMERIC_BINS = 5` (`emva/constants.py`); reference
  levels and bin edges as in `emva/generic.py` (module docstring).
- Feature lists: `[[features]]` of `mappings/olist_funnel.toml` (1 extra), `mappings/crm_opportunities.toml`
  (9 extras over 3 sources), `mappings/hotel_bookings.toml` (23 extras), chosen from the orchestrator's leakage
  guard (decision D7 below) and the dataset documentation, not from model output.
- Labels and dates: the Phase 9 mappings' `as_of` / `test_from`, H = 120, default flags. Value transform default.
- The leakage screen (single-feature training AUC > 0.90) flags only; a flagged extra is not dropped and the verdict
  is computed with it.
