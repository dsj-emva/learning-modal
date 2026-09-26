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

## What was built

| module | role |
|---|---|
| `emva/generic.py` | The extras: `ExtraFeature` (name, kind, source), `read_extra_features` (`extra_features.csv` + `dataset.json` `features`, refused when absent), `fit_encoder` / `GenericEncoder` (categorical: levels with >= 30 training leads + `other` + `missing`, reference the most frequent kept level; numeric: 5 quantile bins with `inverted_cdf` edges, duplicate edges collapsed, an edge at the maximum moved down, + `missing`, reference the first bin) |
| `emva/feature_spec.py`, `emva/features.py`, `emva/cli.py` | `FeatureSet.GENERIC` / `GENERIC_SPEC` (v2 spec with `extras = True`); `--feature-set generic` on every entry point that offers the flag (`python -m emva`, `emva.eval.report`, `emva.eval.collinearity`) |
| `emva/pipeline.py` | With generic: joins the raw extras (`x_<name>`) on `lead_id`, fits the encoder on the pipeline's train mask, appends the `x_` design columns to the v2 design; `PipelineResult.extras` |
| `emva/persist.py` | `ModelBundle.extras` (the frozen encoder); `FORMAT_VERSION` 1 -> 2 |
| `emva/scoring.py` | A generic bundle reads `x_<name>` columns (absent = `missing`), unseen category -> `other`, out-of-range number -> edge bin; `x_` columns a bundle does not use are accepted and not read |
| `emva/ingest/mapping.py` | `[[features]]` rows (`FeatureMap`: expression source, kind, name, reason, confidence) and `features_confirmed`; `check_confirmed` refuses unconfirmed features; features are submit-time columns in `source_columns` |
| `emva/ingest/convert.py` | `extra_values`; `convert` writes `extra_features.csv` and `dataset.json` `features` only when the mapping declares features; `convert_leads` adds the `x_` columns |
| `emva/ingest/draft.py` | Reply schema: target `feature` plus `feature_kind` (`numeric` / `categorical` / `none`); `PROMPT_VERSION` `mapping-draft-v2`; drafted features always unconfirmed |
| `emva/eval/generic_report.py`, `emva/eval/report.py` | "Generic vs v2" section: a v2 model on the same rows and split, paired AUC difference on both test sets, the criterion's verdict, the leakage screen |
| `mappings/*.toml` | Declared extras: Olist 1, CRM 9 (third source `sales_teams.csv`), hotel 23; `features_confirmed = true` after a review by hand |
| `app/storage.py`, `app/ingest.py` | Minimal pass-through so the app tests keep passing (see decision D11) |

## Orchestrator decisions

D1-D10 were given with the task; they are recorded here and in ADR 0024 (Proposed).

- **D1** Extras are declared in the mapping (`[[features]]`: `source`, `kind`, `name`, `reason`, `confidence`; design
  prefix `x_<name>`) with a top-level `features_confirmed`; convert refuses non-empty features without it. A mapping
  without `[[features]]` behaves exactly as in Phase 9 (the TOML round trip and the Phase 9 dump are tested). The LLM
  draft may propose extras (`feature` + kind; `PROMPT_VERSION` bumped) but is never confirmed. Derived expressions
  are allowed as sources.
- **D2** `convert` writes `extra_features.csv` (`lead_id` + one raw column per extra) and a `features` section in
  `dataset.json`; `historical_leads.csv` is unchanged (verified: all four training files byte-identical to the Phase 9
  conversion on the three real sources). The pipeline refuses the generic set without the file.
- **D3** Encoding fitted on the training leads only and frozen in the bundle. Categorical: `GENERIC_MIN_LEVEL_COUNT
  = 30`, `other`, `missing`, reference = most frequent kept level. Numeric: `GENERIC_NUMERIC_BINS = 5`, `missing`;
  **reference = the first bin** (the choice left open in the brief: deterministic and independent of the data's
  centre). Unseen category -> `other` (amends ADR 0009 for extras only); out-of-range number -> edge bin.
- **D4** `FeatureSet.GENERIC` = v2's 39 columns + the `x_` columns; default stays v2; deal-value model unchanged.
- **D5** Report section "generic vs v2" with paired AUC (1000 resamples, seed 0) on (b) and (a), the collinearity
  check on the generic design and a leakage screen (folded single-feature training AUC, > 0.90 flagged, never
  dropped).
- **D6** Pre-registered criterion (above), committed in 4df4e4a before any Kaggle number was computed.
- **D7** Leakage guard for the committed mappings (the lists in each mapping header). Hotel: the caveats on
  `required_car_parking_spaces` and `total_of_special_requests` (can change after booking) are kept as the dataset
  documentation describes them as booking attributes; `adr` also carries a caveat (the paper defines it from lodging
  transactions). CRM: pipeline + accounts + sales_teams (3 sources); `sales_agent` and `manager` stay redacted in
  profiles. Olist: `landing_page_id` only; `origin` is not double-encoded; no date-part operation was added.
- **D8** Tests: `tests/test_generic_features.py` (25 test functions, 29 cases) plus extensions of `test_ingest.py`, `test_scoring.py`,
  `test_persist.py`, `test_app_ingest.py`, `test_app_scoring.py`.
- **D9** Real-data runs (below), all on public data, nothing committed from `data/external/` or `runs/`.
- **D10** Docs: ADR 0024, this report, `docs/ARCHITECTURE.md` section 10, status rows in `CLAUDE.md` and
  `docs/CONTEXT.md`, the `--feature-set generic` flag in the CLAUDE.md CLI table.

Decisions taken during the work (for the orchestrator's review):

- **D11 (app, minimal):** once the committed Olist mapping declares a feature, the app's `app_converted` fixture and
  Map & convert produce `extra_features.csv`, which `app.storage` refused as an unknown file, and the mapping form
  dropped `[[features]]` on its round trip. Minimal compatible change: `app/storage.py` lists `extra_features.csv`
  among `METADATA_FILES` (stored as given, not validated) and `app/ingest.MappingForm` carries `features` /
  `features_confirmed` through unchanged. A committed mapping's `features_confirmed = true` survives the form; the
  app has no control to confirm or reset it yet (part b). Two app test expectations (hand-built draft reply,
  CRM source fields) were updated for the new contract.
- **D12 (scoring):** `x_` columns a bundle does not use are accepted and not read (like the `historical_leads.csv`
  columns the models do not read), so a v2 run still scores the `convert_leads` output of a mapping with features.
  Any other unknown column is still refused.
- **D13 (bins):** numeric edges use numpy's `inverted_cdf` quantiles (every edge an observed value; with the linear
  default a 0/5 column got a spurious edge at 1.0000000000000142), and an edge equal to the training maximum is moved
  to the largest value below it, so a skewed 0/1 column keeps two bins. Decided on unit fixtures before any Kaggle
  run.
- **D14 (draft names):** a drafted feature's name is its column name as a slug, prefixed with the source name when
  two files share the column.

## Results (on public data)

Converted with the committed mappings, trained with `python -m emva --feature-set generic` (pipeline defaults: H =
120, the mapping's dates, default flags and value transform) and reported with `python -m emva.eval.report
--feature-set generic` (1000 resamples, seed 0). **Every number in this section is on public data** (real Kaggle
exports, not simulated, and not B2B enquiries in the hotel case). **Nothing was tuned** after the criterion was
committed: bins, minimum counts, feature lists, H and dates are as registered.

| dataset | extras | `x_` columns | (b) horizon: generic − v2 [95% CI] | (a) legacy: generic − v2 [95% CI] | collinearity (generic) | leakage flags | criterion |
|---|---|---|---|---|---|---|---|
| olist_funnel | 1 | 24 | +0.094 [+0.064, +0.123] | +0.086 [+0.057, +0.114] | PASS (max 0.716) | none | **PASS** |
| crm_opportunities | 9 | 89 | +0.016 [−0.015, +0.044] | −0.062 [−0.080, −0.044] | PASS (max 0.744) | none | **FAIL** |
| hotel_bookings | 23 | 302 | +0.430 [+0.427, +0.433] | +0.301 [+0.296, +0.306] | **FAIL** (0.969, `x_distribution_channel=GDS` ~ `x_agent=195`) | none (highest: lead_time 0.888) | **FAIL** |

| dataset | test set | v2 AUC | generic AUC [95% CI] | generic Brier | generic top-20% wins |
|---|---|---|---|---|---|
| olist_funnel | (b) horizon, 3,829 (444 won) | 0.553 | 0.646 [0.618, 0.671] | 0.1020 | 0.338 |
| | (a) legacy, 3,829 (480 won) | 0.557 | 0.643 [0.616, 0.666] | 0.1100 | 0.335 |
| crm_opportunities | (b) horizon, 1,393 (788 won) | 0.500 | 0.516 [0.485, 0.544] | 0.2505 | 0.207 |
| | (a) legacy, 3,825 (1,844 won) | 0.500 | 0.438 [0.420, 0.456] | 0.2806 | 0.157 |
| hotel_bookings (bookings) | (b) horizon, 25,491 (12,873 won) | 0.500 | 0.930 [0.927, 0.933] | 0.1052 | 0.383 |
| | (a) legacy, 31,578 (20,972 won) | 0.500 | 0.801 [0.796, 0.806] | 0.2161 | 0.292 |

### Reading the results

- **olist_funnel: PASS.** One extra, `landing_page_id`, adds 24 columns (the 23 landing pages with at least 30
  training MQLs plus `other`; `missing` is the only other level and never occurs). Test AUC rises from 0.553 to
  0.646 on the mature test set, CI of the difference [+0.064, +0.123]. The single-feature training AUC of the
  landing page is 0.714, below the leakage flag. Licence: **CC BY-NC-SA 4.0, non-commercial**: these numbers are for
  this evaluation only, not for commercial use or marketing.
- **crm_opportunities: FAIL.** Nine extras (89 columns) carry almost no signal: every single-feature training AUC is
  0.507 to 0.538. On the horizon test set the gain is +0.016 with a CI that includes 0; on the legacy test set the
  generic model is worse than a constant score (AUC 0.438): the extras separate wins from losses slightly in
  training and slightly the other way on the legacy test set's labels, which count stalled Engaging deals as lost.
  The CRM data (a fictitious company, Apache 2.0) look close to random with respect to these attributes.
- **hotel_bookings: FAIL** (on the collinearity half of the criterion). The AUC gain is large (+0.430, CI [+0.427,
  +0.433]), but the generic design has one pair above the 0.95 limit: `x_distribution_channel=GDS` and `x_agent=195`
  (|corr| 0.969: nearly every GDS booking comes through agent 195). The criterion is not relaxed. **Warning on the
  AUC itself:** the 0.930 is mostly the label definition, not foresight about cancellations. A kept booking is Won at
  its arrival (R28), and arrival = booking date + `lead_time`; with H = 120 a booking made more than 120 days before
  arrival can never be "won within H", whatever the guest does. `lead_time` therefore nearly fixes the horizon label
  (single-feature training AUC 0.888, just under the 0.90 flag; weights −280 points for `>216` days and −183 for
  `(115, 216]`). On the legacy test set, whose label is "not cancelled" without a horizon, the generic AUC is 0.801,
  and `deposit_type=Non Refund` (−120 points) is the next strongest signal: a known quirk of this dataset (almost
  every non-refundable booking in it was cancelled). These are bookings, not enquiries.
- **Leakage screen:** no extra on any dataset exceeds 0.90. The screen is a coarse guard: hotel `lead_time`
  (0.888) is mechanically tied to the horizon label as explained above without being post-outcome data.

### Source-format scoring of new leads (on public data)

Five test-period rows per dataset, outcome columns removed, scored with `python -m emva.ingest score --mapping
mappings/<name>.toml --raw <rows> --run runs/<name>-generic --data data/external/<name>`. `convert_leads` carries
the extras as `x_<name>` columns; every p equals the batch run's `scores.csv` value exactly (relative difference 0).

| dataset | lead | inputs (abridged) | p_formula | value_at_submit | batch y |
|---|---|---|---|---|---|
| olist_funnel | a4341fce… | social | 0.0184 | 18,881 | 0 |
| | 5da9cf96… | unknown | 0.0893 | 50,309 | 0 |
| | c89fb04b… | organic_search | 0.0393 | 27,434 | 0 |
| | 10ccbc80… | social | 0.0172 | 17,811 | 0 |
| | 742ac6b5… | display | 0.0393 | 27,434 | 0 |
| crm_opportunities | TJ7PNIXL | GTXPro, no account | 0.6665 | 2,479 | (unlabelled) |
| | 44Q1W18Y | MG Special, Zoomit | 0.7018 | 2,571 | (unlabelled) |
| | AIU1UWJZ | GTXPro, Zotware | 0.6626 | 2,469 | 1 |
| | G62CPKZO | MG Advanced, Dontechi | 0.5977 | 2,292 | 0 |
| | J25NMK6Q | MG Special, Bluth Company | 0.6703 | 2,489 | 0 |
| hotel_bookings | row 68858 | City Hotel, lead_time 29 | 0.5395 | 213 | 0 |
| | row 67061 | City Hotel, lead_time 28 | 0.6663 | 245 | 0 |
| | row 33562 | Resort Hotel, lead_time 27 | 0.5572 | 217 | 1 |
| | row 68941 | City Hotel, lead_time 141 | 0.0115 | 25 (floor) | 0 |
| | row 68682 | City Hotel, lead_time 165 | 0.00002 | 25 (floor) | 0 |

The last two hotel rows show the label effect above: booked more than 120 days ahead, they cannot be won within H.
Amounts are in each source's own currency (the report's "GBP" formatting is fixed); Olist's values are inflated by
the 50,000,000 outlier noted in Phase 9. Every row is `is_bot = False`, `is_duplicate = False`.

<details>
<summary>Standard report: olist_funnel, generic feature set (on public data)</summary>

#### EMVA standard report

Data: `data/external/olist_funnel`. candidate = `emva` pipeline trained with `horizon H=120` labels and `generic` features. Every model is scored on the same rows under two test definitions. AUC CI: percentile bootstrap, 1000 resamples, seed 0. Top-20% capture ranks by p or by p×value.

Dates: as_of 2018-11-15T00:00:00+00:00, test_from 2018-03-01 (from dataset.json).

- Baseline omitted: this dataset carries its own dates in `dataset.json` (a converted dataset, ADR 0020). The frozen `baseline/emva_score.py` hard-codes the v1 snapshot date, test boundary and file format, so its scores would not be comparable (ADR 0022). The paired comparison against it and the check that the legacy labels equal the baseline's are omitted too.
- Status quo omitted: the dataset has no `status_quo_rules.json` (the rules the status quo is reconstructed from; converted datasets have none).

##### Label definitions

Counts over all 8000 scored leads, as wins / losses / unlabelled. `label_source` is the CRM state at 2018-11-15. legacy y = baseline rules. won_within_h = Won within 120 days of created_at (NaN if younger and not Won). horizon y = won_within_h with ghosted-at-H leads excluded and stalled leads censored (the default training and test label).

| label_source | leads | legacy y | won_within_h | horizon y |
|---|---|---|---|---|
| won | 842 | 842 / 0 / 0 | 726 / 116 / 0 | 726 / 116 / 0 |
| crm_lost | 7158 | 0 / 7158 / 0 | 0 / 7158 / 0 | 0 / 7158 / 0 |
| stalled | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| ghosted | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| open | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| all | 8000 | 842 / 7158 / 0 | 726 / 7274 / 0 | 726 / 7274 / 0 |

Mature = created_at + 120 days <= 2018-11-15T00:00:00+00:00: the latest mature lead was created 2018-05-31T00:00:00+00:00. Train = created before 2018-03-01 (all mature); test = created on or after 2018-03-01.

| definition | train (wins) | test (wins) |
|---|---|---|
| legacy | 4171 (362) | 3829 (480) |
| horizon H=120 | 4171 (282) | 3829 (444) |

Candidate trained with: `horizon H=120`.

##### (a) Legacy labels, legacy test set

3829 leads labelled by the baseline rules, created on or after 2018-03-01 (480 won). Label = legacy y.

###### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| candidate | 0.643 [0.616, 0.666] | 0.1100 | 0.335 | 0.850 | 0.854 |

###### Paired AUC comparison vs baseline

Omitted: there is no baseline row (see the note at the top).

###### Calibration by decile of p

| decile | n | candidate mean p | candidate observed |
|---|---|---|---|
| 1 | 383 | 0.018 | 0.055 |
| 2 | 383 | 0.024 | 0.076 |
| 3 | 383 | 0.038 | 0.117 |
| 4 | 383 | 0.039 | 0.055 |
| 5 | 383 | 0.039 | 0.065 |
| 6 | 382 | 0.041 | 0.113 |
| 7 | 383 | 0.076 | 0.141 |
| 8 | 383 | 0.095 | 0.211 |
| 9 | 383 | 0.134 | 0.185 |
| 10 | 383 | 0.147 | 0.235 |

###### AUC by test month

| month | n | wins | candidate |
|---|---|---|---|
| 2018-03 | 1174 | 167 | 0.584 |
| 2018-04 | 1352 | 183 | 0.672 |
| 2018-05 | 1303 | 130 | 0.661 |

###### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| candidate | test set | 3829 | 36779 | 189647 | 189647 | 5.2× | 4.1% |
| candidate | all scored leads | 8000 | 33563 | 189647 | 189647 | 5.7× | 4.1% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

###### Notes

- candidate: 10 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.335.
- candidate: 10 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.850.
- candidate: 173 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.854.

##### (b) Horizon labels, mature test set

3829 mature leads created on or after 2018-03-01 and up to 2018-05-31T00:00:00+00:00 that the default horizon definition labels (444 won). Label = won within 120 days; ghosted-at-H leads excluded, stalled leads censored.

###### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| candidate | 0.646 [0.618, 0.671] | 0.1020 | 0.338 | 0.182 | 0.182 |

###### Paired AUC comparison vs baseline

Omitted: there is no baseline row (see the note at the top).

###### Calibration by decile of p

| decile | n | candidate mean p | candidate observed |
|---|---|---|---|
| 1 | 383 | 0.018 | 0.052 |
| 2 | 383 | 0.024 | 0.070 |
| 3 | 383 | 0.038 | 0.102 |
| 4 | 383 | 0.039 | 0.050 |
| 5 | 383 | 0.039 | 0.047 |
| 6 | 382 | 0.041 | 0.113 |
| 7 | 383 | 0.076 | 0.128 |
| 8 | 383 | 0.095 | 0.206 |
| 9 | 383 | 0.134 | 0.172 |
| 10 | 383 | 0.147 | 0.219 |

###### AUC by test month

| month | n | wins | candidate |
|---|---|---|---|
| 2018-03 | 1174 | 149 | 0.598 |
| 2018-04 | 1352 | 174 | 0.667 |
| 2018-05 | 1303 | 121 | 0.665 |

###### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| candidate | test set | 3829 | 36779 | 189647 | 189647 | 5.2× | 4.1% |
| candidate | all scored leads | 8000 | 33563 | 189647 | 189647 | 5.7× | 4.1% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

###### Notes

- candidate: 10 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.338.
- candidate: 10 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.182.
- candidate: 173 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.182.

##### Bottom-decile ghosted share

Share of leads in the 10% of each population with the lowest p that are *ghosted* (`label_source`: mature and not contacted within 120 days) or *still New* at 2018-11-15 whatever their age (the definition behind the plan's baseline figure of 27%). The horizon test set (b) excludes ghosted leads, so the comparison uses populations that keep them.

| population | n | ghosted: overall | ghosted: candidate bottom decile | still New: overall | still New: candidate bottom decile |
|---|---|---|---|---|---|
| all leads labelled by legacy rules (train + test) | 8000 | 0.0% | 0.0% | 0.0% | 0.0% |
| (a) legacy test set | 3829 | 0.0% | 0.0% | 0.0% | 0.0% |
| all mature leads, every label_source | 8000 | 0.0% | 0.0% | 0.0% | 0.0% |
| mature leads created on or after 2018-03-01, every label_source | 3829 | 0.0% | 0.0% | 0.0% | 0.0% |

##### Value transforms (plan 3.2)

Candidate value = `value_formula` (p × E[deal value] × margin) through each transform, fitted on the candidate's 4171 training leads. Top-20% revenue = tie-averaged share of recorded won deal value in the top 20% by value; vs baseline = that share over the baseline's own (p×value, identity). value / revenue = total value over total recorded revenue. Every transform is monotone, so capture moves only through ties at the cut.

###### (a) Legacy labels, legacy test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|
| candidate: identity | 12583 | 36779 | 73515 | 189647 | 189647 | 5.2× | 4.1% | 0.854 | 173 | 2.93 |
| candidate: cap p97 | 12583 | 36779 | 73515 | 189647 | 189647 | 5.2× | 4.1% | 0.854 | 173 | 2.93 |
| candidate: cap p97 + floor £25 | 12583 | 36779 | 73515 | 189647 | 189647 | 5.2× | 4.1% | 0.854 | 173 | 2.93 |
| candidate: cap p97 + sqrt | 25842 | 44181 | 62462 | 100324 | 100324 | 2.3× | 2.2% | 0.854 | 173 | 2.92 |
| candidate: cap p97 + log | 18881 | 43881 | 68799 | 112360 | 112360 | 2.6× | 2.5% | 0.854 | 173 | 2.90 |
| candidate: tiers 5 | 16111 | 33104 | 101485 | 101485 | 101485 | 3.1× | 2.2% | 0.717 | 912 | 2.96 |
| candidate: tiers 10 | not fitted: 10 tiers: the reference values leave 1 tier(s) empty (too few distinct values) | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| candidate: cap p97 + log + floor £25 | 18881 | 43881 | 68799 | 112360 | 112360 | 2.6× | 2.5% | 0.854 | 173 | 2.90 |

###### (b) Horizon labels, mature test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|
| candidate: identity | 12583 | 36779 | 73515 | 189647 | 189647 | 5.2× | 4.1% | 0.182 | 173 | 1058.70 |
| candidate: cap p97 | 12583 | 36779 | 73515 | 189647 | 189647 | 5.2× | 4.1% | 0.182 | 173 | 1058.70 |
| candidate: cap p97 + floor £25 | 12583 | 36779 | 73515 | 189647 | 189647 | 5.2× | 4.1% | 0.182 | 173 | 1058.70 |
| candidate: cap p97 + sqrt | 25842 | 44181 | 62462 | 100324 | 100324 | 2.3× | 2.2% | 0.182 | 173 | 1055.70 |
| candidate: cap p97 + log | 18881 | 43881 | 68799 | 112360 | 112360 | 2.6× | 2.5% | 0.182 | 173 | 1047.61 |
| candidate: tiers 5 | 16111 | 33104 | 101485 | 101485 | 101485 | 3.1× | 2.2% | 0.153 | 912 | 1070.61 |
| candidate: tiers 10 | not fitted: 10 tiers: the reference values leave 1 tier(s) empty (too few distinct values) | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| candidate: cap p97 + log + floor £25 | 18881 | 43881 | 68799 | 112360 | 112360 | 2.6× | 2.5% | 0.182 | 173 | 1047.61 |

###### All scored leads

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| candidate: identity | 12303 | 33563 | 73515 | 189647 | 189647 | 5.7× | 4.1% |
| candidate: cap p97 | 12303 | 33563 | 73515 | 189647 | 189647 | 5.7× | 4.1% |
| candidate: cap p97 + floor £25 | 12303 | 33563 | 73515 | 189647 | 189647 | 5.7× | 4.1% |
| candidate: cap p97 + sqrt | 25553 | 42205 | 62462 | 100324 | 100324 | 2.4× | 2.2% |
| candidate: cap p97 + log | 18521 | 41106 | 68799 | 112360 | 112360 | 2.7× | 2.5% |
| candidate: tiers 5 | 16111 | 33104 | 101485 | 101485 | 101485 | 3.1× | 2.2% |
| candidate: tiers 10 | not fitted: 10 tiers: the reference values leave 1 tier(s) empty (too few distinct values) | nan | nan | nan | nan | nan | nan |
| candidate: cap p97 + log + floor £25 | 18521 | 41106 | 68799 | 112360 | 112360 | 2.7× | 2.5% |

###### Acceptance (plan 3.2): ≥ 95% of the baseline's capture on every test set and max/median < 20× on every test set and on all scored leads

Not computed: the criterion is defined against the baseline, which this report omits.

##### Click-ID coverage (plan 3.5)

Scored leads per paid channel by the identifier an upload could match on (see `docs/platform_contract.md`): a click id (gclid, gbraid, wbraid, fbclid, li_fat_id, oppref), else the lead-form id, else nothing but hashed email / phone (Meta: plus the `fbp` browser cookie).

| channel | leads | click id | lead-form id | no id: fbp present | no id at all | email present | phone present |
|---|---|---|---|---|---|---|---|
| google | 1586 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% |
| meta | 1350 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% |
| linkedin | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| chatgpt | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| meta_leadads | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| landing-page paid (all but meta_leadads) | 2936 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% |
| all paid | 2936 | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.0% |

##### Design collinearity (plan 2.7): PASS

Candidate design (`generic` features, 63 columns) on its 4171 training rows; fails when two columns have |corr| > 0.95. The report exits 1 on a failure only with `--strict`; `python -m emva.eval.collinearity` always does.

- PASS: max |corr| = 0.716 between channel=meta and x_landing_page=88740e65d5d6b056e0cda098e1ea6313
- Constant on the training rows (skipped): channel=meta_leadads, channel=linkedin, channel=chatgpt, form_variant=B, form_variant=C, form_variant=D, session_missing=yes, enrichment_missing=yes, band=11-50, band=51-200, band=201-1000, band=1000+, band=missing, email=free, text=copy_paste, text=vague, text=specific, seniority=student, seniority=senior, seniority=mid, seniority=not_asked/blank, spend=none, spend=£5k-£25k, spend=£25k-£100k, spend=£100k+, crm=hubspot_sf, hiring=hiring, time_on_page=60-300s, time_on_page=300-600s, time_on_page=>600s, hesitation_90s=yes, sessions_3plus=yes, viewed_pricing=yes, search_term=brand, business_hours=wkday_9-18, ip_country=mismatch, x_landing_page=missing

##### Generic vs v2 (Phase 10, ADR 0024)

Both models trained on the same rows and split with the same labels: `v2` (the fixed 39-column design) and `generic` (the same plus 24 `x_` columns from 1 declared extras, encoded on the 4171 training leads only). Paired bootstrap of AUC(generic) − AUC(v2) on shared resamples (1000 resamples, seed 0).

| test set | n | v2 AUC | generic AUC | generic − v2 | 95% CI | bootstrap p |
|---|---|---|---|---|---|---|
| (a) Legacy labels, legacy test set | 3829 | 0.557 | 0.643 | +0.086 | [+0.057, +0.114] | 0.000 |
| (b) Horizon labels, mature test set | 3829 | 0.553 | 0.646 | +0.094 | [+0.064, +0.123] | 0.000 |

Generic design collinearity: PASS: max |corr| = 0.716 between channel=meta and x_landing_page=88740e65d5d6b056e0cda098e1ea6313 (details in the collinearity section above).

**Phase 10 criterion (pre-registered): PASS.** PASS iff on the horizon test set (b) the 95% CI of generic − v2 lies entirely above 0 and the generic design passes the collinearity check.

###### Leakage screen

Single-feature training AUC of each extra's encoded levels (each level scored by its training win rate), folded as |AUC − 0.5| + 0.5. Above 0.90: flagged "check for leakage" (a flag only; nothing is dropped).

| extra | kind | levels | design columns | training AUC | flag |
|---|---|---|---|---|---|
| landing_page | categorical | 25 | 24 | 0.714 |  |

Flagged: none.

##### Baseline script output

- Baseline omitted: this dataset carries its own dates in `dataset.json` (a converted dataset, ADR 0020). The frozen `baseline/emva_score.py` hard-codes the v1 snapshot date, test boundary and file format, so its scores would not be comparable (ADR 0022). The paired comparison against it and the check that the legacy labels equal the baseline's are omitted too.

</details>

<details>
<summary>Standard report: crm_opportunities, generic feature set (on public data)</summary>

#### EMVA standard report

Data: `data/external/crm_opportunities`. candidate = `emva` pipeline trained with `horizon H=120` labels and `generic` features. Every model is scored on the same rows under two test definitions. AUC CI: percentile bootstrap, 1000 resamples, seed 0. Top-20% capture ranks by p or by p×value.

Dates: as_of 2018-01-01T00:00:00+00:00, test_from 2017-07-01 (from dataset.json).

- Baseline omitted: this dataset carries its own dates in `dataset.json` (a converted dataset, ADR 0020). The frozen `baseline/emva_score.py` hard-codes the v1 snapshot date, test boundary and file format, so its scores would not be comparable (ADR 0022). The paired comparison against it and the check that the legacy labels equal the baseline's are omitted too.
- Status quo omitted: the dataset has no `status_quo_rules.json` (the rules the status quo is reconstructed from; converted datasets have none).

##### Label definitions

Counts over all 8300 scored leads, as wins / losses / unlabelled. `label_source` is the CRM state at 2018-01-01. legacy y = baseline rules. won_within_h = Won within 120 days of created_at (NaN if younger and not Won). horizon y = won_within_h with ghosted-at-H leads excluded and stalled leads censored (the default training and test label).

| label_source | leads | legacy y | won_within_h | horizon y |
|---|---|---|---|---|
| won | 4238 | 4238 / 0 / 0 | 4092 / 146 / 0 | 4092 / 146 / 0 |
| crm_lost | 2473 | 0 / 2473 / 0 | 0 / 1822 / 651 | 0 / 1822 / 651 |
| stalled | 1479 | 0 / 1479 / 0 | 0 / 1385 / 94 | 0 / 0 / 1479 |
| ghosted | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| open | 110 | 0 / 0 / 110 | 0 / 0 / 110 | 0 / 0 / 110 |
| all | 8300 | 4238 / 3952 / 110 | 4092 / 3353 / 855 | 4092 / 1968 / 2240 |

Mature = created_at + 120 days <= 2018-01-01T00:00:00+00:00: the latest mature lead was created 2017-09-03T00:00:00+00:00. Train = created before 2017-07-01 (all mature); test = created on or after 2017-07-01.

| definition | train (wins) | test (wins) |
|---|---|---|
| legacy | 4365 (2394) | 3825 (1844) |
| horizon H=120 | 3649 (2286) | 1393 (788) |

Candidate trained with: `horizon H=120`.

##### (a) Legacy labels, legacy test set

3825 leads labelled by the baseline rules, created on or after 2017-07-01 (1844 won). Label = legacy y.

###### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| candidate | 0.438 [0.420, 0.456] | 0.2806 | 0.157 | 0.160 | 0.160 |

###### Paired AUC comparison vs baseline

Omitted: there is no baseline row (see the note at the top).

###### Calibration by decile of p

| decile | n | candidate mean p | candidate observed |
|---|---|---|---|
| 1 | 383 | 0.542 | 0.527 |
| 2 | 382 | 0.580 | 0.510 |
| 3 | 383 | 0.600 | 0.512 |
| 4 | 382 | 0.614 | 0.547 |
| 5 | 383 | 0.627 | 0.580 |
| 6 | 382 | 0.642 | 0.516 |
| 7 | 382 | 0.657 | 0.432 |
| 8 | 383 | 0.671 | 0.439 |
| 9 | 382 | 0.689 | 0.416 |
| 10 | 383 | 0.722 | 0.342 |

###### AUC by test month

| month | n | wins | candidate |
|---|---|---|---|
| 2017-07 | 1198 | 441 | 0.402 |
| 2017-08 | 793 | 341 | 0.421 |
| 2017-09 | 779 | 425 | 0.415 |
| 2017-10 | 708 | 400 | 0.548 |
| 2017-11 | 244 | 157 | 0.531 |
| 2017-12 | 103 | 80 | 0.538 |

###### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| candidate | test set | 3825 | 2400 | 2814 | 3001 | 1.3× | 1.2% |
| candidate | all scored leads | 8300 | 2395 | 2810 | 3001 | 1.3× | 1.2% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

###### Notes

- No ties at the top-20% cut.

##### (b) Horizon labels, mature test set

1393 mature leads created on or after 2017-07-01 and up to 2017-09-03T00:00:00+00:00 that the default horizon definition labels (788 won). Label = won within 120 days; ghosted-at-H leads excluded, stalled leads censored.

###### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| candidate | 0.516 [0.485, 0.544] | 0.2505 | 0.207 | 0.211 | 0.211 |

###### Paired AUC comparison vs baseline

Omitted: there is no baseline row (see the note at the top).

###### Calibration by decile of p

| decile | n | candidate mean p | candidate observed |
|---|---|---|---|
| 1 | 140 | 0.539 | 0.529 |
| 2 | 139 | 0.576 | 0.612 |
| 3 | 139 | 0.596 | 0.532 |
| 4 | 139 | 0.608 | 0.518 |
| 5 | 140 | 0.620 | 0.550 |
| 6 | 139 | 0.634 | 0.619 |
| 7 | 139 | 0.649 | 0.583 |
| 8 | 139 | 0.664 | 0.547 |
| 9 | 139 | 0.681 | 0.576 |
| 10 | 140 | 0.716 | 0.593 |

###### AUC by test month

| month | n | wins | candidate |
|---|---|---|---|
| 2017-07 | 796 | 432 | 0.502 |
| 2017-08 | 539 | 312 | 0.527 |
| 2017-09 | 58 | 44 | 0.625 |

###### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| candidate | test set | 1393 | 2372 | 2806 | 2939 | 1.2× | 1.1% |
| candidate | all scored leads | 8300 | 2395 | 2810 | 3001 | 1.3× | 1.2% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

###### Notes

- No ties at the top-20% cut.

##### Bottom-decile ghosted share

Share of leads in the 10% of each population with the lowest p that are *ghosted* (`label_source`: mature and not contacted within 120 days) or *still New* at 2018-01-01 whatever their age (the definition behind the plan's baseline figure of 27%). The horizon test set (b) excludes ghosted leads, so the comparison uses populations that keep them.

| population | n | ghosted: overall | ghosted: candidate bottom decile | still New: overall | still New: candidate bottom decile |
|---|---|---|---|---|---|
| all leads labelled by legacy rules (train + test) | 8190 | 0.0% | 0.0% | 0.0% | 0.0% |
| (a) legacy test set | 3825 | 0.0% | 0.0% | 0.0% | 0.0% |
| all mature leads, every label_source | 6427 | 0.0% | 0.0% | 0.0% | 0.0% |
| mature leads created on or after 2017-07-01, every label_source | 2062 | 0.0% | 0.0% | 0.0% | 0.0% |

##### Value transforms (plan 3.2)

Candidate value = `value_formula` (p × E[deal value] × margin) through each transform, fitted on the candidate's 3649 training leads. Top-20% revenue = tie-averaged share of recorded won deal value in the top 20% by value; vs baseline = that share over the baseline's own (p×value, identity). value / revenue = total value over total recorded revenue. Every transform is monotone, so capture moves only through ties at the cut.

###### (a) Legacy labels, legacy test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|
| candidate: identity | 1938 | 2400 | 2646 | 2814 | 3001 | 1.3× | 1.2% | 0.160 | 1 | 2.11 |
| candidate: cap p97 | 1938 | 2400 | 2646 | 2720 | 2720 | 1.1× | 1.1% | 0.160 | 1 | 2.11 |
| candidate: cap p97 + floor £25 | 1938 | 2400 | 2646 | 2720 | 2720 | 1.1× | 1.1% | 0.160 | 1 | 2.11 |
| candidate: cap p97 + sqrt | 2144 | 2385 | 2505 | 2540 | 2540 | 1.1× | 1.1% | 0.160 | 1 | 2.10 |
| candidate: cap p97 + log | 2045 | 2393 | 2566 | 2615 | 2615 | 1.1× | 1.1% | 0.160 | 1 | 2.10 |
| candidate: tiers 5 | 2110 | 2364 | 2629 | 2629 | 2629 | 1.1× | 1.1% | 0.162 | 1021 | 2.11 |
| candidate: tiers 10 | 2042 | 2386 | 2692 | 2692 | 2692 | 1.1× | 1.1% | 0.157 | 451 | 2.11 |
| candidate: cap p97 + log + floor £25 | 2045 | 2393 | 2566 | 2615 | 2615 | 1.1× | 1.1% | 0.160 | 1 | 2.10 |

###### (b) Horizon labels, mature test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|
| candidate: identity | 1919 | 2372 | 2616 | 2806 | 2939 | 1.2× | 1.1% | 0.211 | 1 | 1.84 |
| candidate: cap p97 | 1919 | 2372 | 2616 | 2720 | 2720 | 1.1× | 1.1% | 0.211 | 1 | 1.84 |
| candidate: cap p97 + floor £25 | 1919 | 2372 | 2616 | 2720 | 2720 | 1.1× | 1.1% | 0.211 | 1 | 1.84 |
| candidate: cap p97 + sqrt | 2133 | 2372 | 2490 | 2540 | 2540 | 1.1× | 1.0% | 0.211 | 1 | 1.84 |
| candidate: cap p97 + log | 2030 | 2373 | 2545 | 2615 | 2615 | 1.1× | 1.0% | 0.211 | 1 | 1.84 |
| candidate: tiers 5 | 2110 | 2364 | 2629 | 2629 | 2629 | 1.1× | 1.0% | 0.206 | 315 | 1.84 |
| candidate: tiers 10 | 2042 | 2386 | 2692 | 2692 | 2692 | 1.1× | 1.1% | 0.204 | 158 | 1.84 |
| candidate: cap p97 + log + floor £25 | 2030 | 2373 | 2545 | 2615 | 2615 | 1.1× | 1.0% | 0.211 | 1 | 1.84 |

###### All scored leads

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| candidate: identity | 1947 | 2395 | 2645 | 2810 | 3001 | 1.3× | 1.2% |
| candidate: cap p97 | 1947 | 2395 | 2645 | 2720 | 2720 | 1.1× | 1.1% |
| candidate: cap p97 + floor £25 | 1947 | 2395 | 2645 | 2720 | 2720 | 1.1× | 1.1% |
| candidate: cap p97 + sqrt | 2149 | 2383 | 2504 | 2540 | 2540 | 1.1× | 1.1% |
| candidate: cap p97 + log | 2052 | 2390 | 2565 | 2615 | 2615 | 1.1× | 1.1% |
| candidate: tiers 5 | 2110 | 2364 | 2629 | 2629 | 2629 | 1.1× | 1.1% |
| candidate: tiers 10 | 2042 | 2386 | 2692 | 2692 | 2692 | 1.1× | 1.1% |
| candidate: cap p97 + log + floor £25 | 2052 | 2390 | 2565 | 2615 | 2615 | 1.1× | 1.1% |

###### Acceptance (plan 3.2): ≥ 95% of the baseline's capture on every test set and max/median < 20× on every test set and on all scored leads

Not computed: the criterion is defined against the baseline, which this report omits.

##### Click-ID coverage (plan 3.5)

Scored leads per paid channel by the identifier an upload could match on (see `docs/platform_contract.md`): a click id (gclid, gbraid, wbraid, fbclid, li_fat_id, oppref), else the lead-form id, else nothing but hashed email / phone (Meta: plus the `fbp` browser cookie).

| channel | leads | click id | lead-form id | no id: fbp present | no id at all | email present | phone present |
|---|---|---|---|---|---|---|---|
| google | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| meta | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| linkedin | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| chatgpt | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| meta_leadads | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| landing-page paid (all but meta_leadads) | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| all paid | 0 | nan% | nan% | nan% | nan% | nan% | nan% |

##### Design collinearity (plan 2.7): PASS

Candidate design (`generic` features, 128 columns) on its 3649 training rows; fails when two columns have |corr| > 0.95. The report exits 1 on a failure only with `--strict`; `python -m emva.eval.collinearity` always does.

- PASS: max |corr| = 0.744 between x_revenue=>3922.42 and x_employees=>8801
- Constant on the training rows (skipped): channel=meta, channel=meta_leadads, channel=linkedin, channel=chatgpt, channel=organic_direct, form_variant=B, form_variant=C, form_variant=D, session_missing=yes, enrichment_missing=yes, band=11-50, band=51-200, band=201-1000, band=1000+, band=missing, email=free, text=copy_paste, text=vague, text=specific, seniority=student, seniority=senior, seniority=mid, seniority=not_asked/blank, spend=none, spend=£5k-£25k, spend=£25k-£100k, spend=£100k+, crm=hubspot_sf, hiring=hiring, time_on_page=60-300s, time_on_page=300-600s, time_on_page=>600s, hesitation_90s=yes, sessions_3plus=yes, viewed_pricing=yes, search_term=brand, search_term=not_google, business_hours=wkday_9-18, ip_country=mismatch, x_product=missing, x_sales_agent=other, x_sales_agent=missing, x_sector=other, x_sector=missing, x_revenue=missing, x_employees=missing, x_office_location=missing, x_year_established=missing, x_regional_office=other, x_regional_office=missing, x_manager=other, x_manager=missing

##### Generic vs v2 (Phase 10, ADR 0024)

Both models trained on the same rows and split with the same labels: `v2` (the fixed 39-column design) and `generic` (the same plus 89 `x_` columns from 9 declared extras, encoded on the 3649 training leads only). Paired bootstrap of AUC(generic) − AUC(v2) on shared resamples (1000 resamples, seed 0).

| test set | n | v2 AUC | generic AUC | generic − v2 | 95% CI | bootstrap p |
|---|---|---|---|---|---|---|
| (a) Legacy labels, legacy test set | 3825 | 0.500 | 0.438 | -0.062 | [-0.080, -0.044] | 0.000 |
| (b) Horizon labels, mature test set | 1393 | 0.500 | 0.516 | +0.016 | [-0.015, +0.044] | 0.318 |

Generic design collinearity: PASS: max |corr| = 0.744 between x_revenue=>3922.42 and x_employees=>8801 (details in the collinearity section above).

**Phase 10 criterion (pre-registered): FAIL.** PASS iff on the horizon test set (b) the 95% CI of generic − v2 lies entirely above 0 and the generic design passes the collinearity check.

###### Leakage screen

Single-feature training AUC of each extra's encoded levels (each level scored by its training win rate), folded as |AUC − 0.5| + 0.5. Above 0.90: flagged "check for leakage" (a flag only; nothing is dropped).

| extra | kind | levels | design columns | training AUC | flag |
|---|---|---|---|---|---|
| product | categorical | 8 | 7 | 0.518 |  |
| sales_agent | categorical | 32 | 31 | 0.538 |  |
| sector | categorical | 12 | 11 | 0.516 |  |
| revenue | numeric | 6 | 5 | 0.519 |  |
| employees | numeric | 6 | 5 | 0.520 |  |
| office_location | categorical | 15 | 14 | 0.513 |  |
| year_established | numeric | 6 | 5 | 0.519 |  |
| regional_office | categorical | 5 | 4 | 0.507 |  |
| manager | categorical | 8 | 7 | 0.510 |  |

Flagged: none.

##### Baseline script output

- Baseline omitted: this dataset carries its own dates in `dataset.json` (a converted dataset, ADR 0020). The frozen `baseline/emva_score.py` hard-codes the v1 snapshot date, test boundary and file format, so its scores would not be comparable (ADR 0022). The paired comparison against it and the check that the legacy labels equal the baseline's are omitted too.

</details>

<details>
<summary>Standard report: hotel_bookings, generic feature set (on public data)</summary>

#### EMVA standard report

Data: `data/external/hotel_bookings`. candidate = `emva` pipeline trained with `horizon H=120` labels and `generic` features. Every model is scored on the same rows under two test definitions. AUC CI: percentile bootstrap, 1000 resamples, seed 0. Top-20% capture ranks by p or by p×value.

Dates: as_of 2017-09-01T00:00:00+00:00, test_from 2016-12-01 (from dataset.json).

- Baseline omitted: this dataset carries its own dates in `dataset.json` (a converted dataset, ADR 0020). The frozen `baseline/emva_score.py` hard-codes the v1 snapshot date, test boundary and file format, so its scores would not be comparable (ADR 0022). The paired comparison against it and the check that the legacy labels equal the baseline's are omitted too.
- Status quo omitted: the dataset has no `status_quo_rules.json` (the rules the status quo is reconstructed from; converted datasets have none).

##### Label definitions

Counts over all 119390 scored leads, as wins / losses / unlabelled. `label_source` is the CRM state at 2017-09-01. legacy y = baseline rules. won_within_h = Won within 120 days of created_at (NaN if younger and not Won). horizon y = won_within_h with ghosted-at-H leads excluded and stalled leads censored (the default training and test label).

| label_source | leads | legacy y | won_within_h | horizon y |
|---|---|---|---|---|
| won | 75166 | 75166 / 0 / 0 | 55722 / 19444 / 0 | 55722 / 19444 / 0 |
| crm_lost | 44224 | 0 / 44224 / 0 | 0 / 42616 / 1608 | 0 / 42616 / 1608 |
| stalled | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| ghosted | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| open | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| all | 119390 | 75166 / 44224 / 0 | 55722 / 62060 / 1608 | 55722 / 62060 / 1608 |

Mature = created_at + 120 days <= 2017-09-01T00:00:00+00:00: the latest mature lead was created 2017-05-04T00:00:00+00:00. Train = created before 2016-12-01 (all mature); test = created on or after 2016-12-01.

| definition | train (wins) | test (wins) |
|---|---|---|
| legacy | 87812 (54194) | 31578 (20972) |
| horizon H=120 | 87812 (38370) | 25491 (12873) |

Candidate trained with: `horizon H=120`.

##### (a) Legacy labels, legacy test set

31578 leads labelled by the baseline rules, created on or after 2016-12-01 (20972 won). Label = legacy y.

###### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| candidate | 0.801 [0.796, 0.806] | 0.2161 | 0.292 | 0.198 | 0.198 |

###### Paired AUC comparison vs baseline

Omitted: there is no baseline row (see the note at the top).

###### Calibration by decile of p

| decile | n | candidate mean p | candidate observed |
|---|---|---|---|
| 1 | 3158 | 0.003 | 0.191 |
| 2 | 3158 | 0.013 | 0.442 |
| 3 | 3158 | 0.133 | 0.655 |
| 4 | 3157 | 0.454 | 0.411 |
| 5 | 3158 | 0.652 | 0.586 |
| 6 | 3158 | 0.762 | 0.710 |
| 7 | 3157 | 0.835 | 0.796 |
| 8 | 3158 | 0.905 | 0.912 |
| 9 | 3158 | 0.951 | 0.953 |
| 10 | 3158 | 0.988 | 0.985 |

###### AUC by test month

| month | n | wins | candidate |
|---|---|---|---|
| 2016-12 | 5013 | 2809 | 0.834 |
| 2017-01 | 7869 | 5187 | 0.803 |
| 2017-02 | 5772 | 3745 | 0.802 |
| 2017-03 | 3609 | 2539 | 0.769 |
| 2017-04 | 2736 | 1884 | 0.794 |
| 2017-05 | 2883 | 2048 | 0.802 |
| 2017-06 | 1626 | 1168 | 0.804 |
| 2017-07 | 1340 | 978 | 0.809 |
| 2017-08 | 730 | 614 | 0.761 |

###### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| candidate | test set | 31578 | 250 | 346 | 347 | 1.4× | 1.7% |
| candidate | all scored leads | 119390 | 190 | 346 | 347 | 1.8× | 2.1% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

###### Notes

- No ties at the top-20% cut.

##### (b) Horizon labels, mature test set

25491 mature leads created on or after 2016-12-01 and up to 2017-05-04T00:00:00+00:00 that the default horizon definition labels (12873 won). Label = won within 120 days; ghosted-at-H leads excluded, stalled leads censored.

###### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| candidate | 0.930 [0.927, 0.933] | 0.1052 | 0.383 | 0.282 | 0.282 |

###### Paired AUC comparison vs baseline

Omitted: there is no baseline row (see the note at the top).

###### Calibration by decile of p

| decile | n | candidate mean p | candidate observed |
|---|---|---|---|
| 1 | 2550 | 0.002 | 0.007 |
| 2 | 2549 | 0.009 | 0.020 |
| 3 | 2549 | 0.032 | 0.058 |
| 4 | 2549 | 0.268 | 0.229 |
| 5 | 2549 | 0.536 | 0.455 |
| 6 | 2549 | 0.717 | 0.678 |
| 7 | 2549 | 0.808 | 0.774 |
| 8 | 2549 | 0.886 | 0.898 |
| 9 | 2549 | 0.944 | 0.944 |
| 10 | 2549 | 0.986 | 0.988 |

###### AUC by test month

| month | n | wins | candidate |
|---|---|---|---|
| 2016-12 | 5013 | 2055 | 0.943 |
| 2017-01 | 7869 | 3797 | 0.946 |
| 2017-02 | 5772 | 2857 | 0.937 |
| 2017-03 | 3609 | 2094 | 0.910 |
| 2017-04 | 2736 | 1741 | 0.867 |
| 2017-05 | 492 | 329 | 0.792 |

###### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| candidate | test set | 25491 | 223 | 346 | 347 | 1.6× | 1.9% |
| candidate | all scored leads | 119390 | 190 | 346 | 347 | 1.8× | 2.1% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

###### Notes

- candidate: 6 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.383.
- candidate: 6 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.282.
- candidate: 6 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.282.

##### Bottom-decile ghosted share

Share of leads in the 10% of each population with the lowest p that are *ghosted* (`label_source`: mature and not contacted within 120 days) or *still New* at 2017-09-01 whatever their age (the definition behind the plan's baseline figure of 27%). The horizon test set (b) excludes ghosted leads, so the comparison uses populations that keep them.

| population | n | ghosted: overall | ghosted: candidate bottom decile | still New: overall | still New: candidate bottom decile |
|---|---|---|---|---|---|
| all leads labelled by legacy rules (train + test) | 119390 | 0.0% | 0.0% | 0.0% | 0.0% |
| (a) legacy test set | 31578 | 0.0% | 0.0% | 0.0% | 0.0% |
| all mature leads, every label_source | 113303 | 0.0% | 0.0% | 0.0% | 0.0% |
| mature leads created on or after 2016-12-01, every label_source | 25491 | 0.0% | 0.0% | 0.0% | 0.0% |

##### Value transforms (plan 3.2)

Candidate value = `value_formula` (p × E[deal value] × margin) through each transform, fitted on the candidate's 87812 training leads. Top-20% revenue = tie-averaged share of recorded won deal value in the top 20% by value; vs baseline = that share over the baseline's own (p×value, identity). value / revenue = total value over total recorded revenue. Every transform is monotone, so capture moves only through ties at the cut.

###### (a) Legacy labels, legacy test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|
| candidate: identity | 0 | 250 | 337 | 346 | 347 | 1.4× | 1.7% | 0.198 | 1 | 0.78 |
| candidate: cap p97 | 0 | 250 | 337 | 343 | 343 | 1.4× | 1.7% | 0.198 | 1 | 0.78 |
| candidate: cap p97 + floor £25 | 25 | 250 | 337 | 343 | 343 | 1.4× | 1.7% | 0.198 | 1 | 0.80 |
| candidate: cap p97 + sqrt | 2 | 247 | 287 | 289 | 289 | 1.2× | 1.5% | 0.198 | 1 | 0.76 |
| candidate: cap p97 + log | 0 | 258 | 310 | 313 | 313 | 1.2× | 1.6% | 0.198 | 1 | 0.77 |
| candidate: tiers 5 | 0 | 280 | 332 | 332 | 332 | 1.2× | 1.7% | 0.208 | 8314 | 0.77 |
| candidate: tiers 10 | 0 | 262 | 340 | 340 | 340 | 1.3× | 1.7% | 0.203 | 4105 | 0.77 |
| candidate: cap p97 + log + floor £25 | 25 | 258 | 310 | 313 | 313 | 1.2× | 1.6% | 0.198 | 1 | 0.79 |

###### (b) Horizon labels, mature test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|
| candidate: identity | 0 | 223 | 336 | 346 | 347 | 1.6× | 1.9% | 0.282 | 6 | 1.17 |
| candidate: cap p97 | 0 | 223 | 336 | 343 | 343 | 1.5× | 1.9% | 0.282 | 6 | 1.17 |
| candidate: cap p97 + floor £25 | 25 | 223 | 336 | 343 | 343 | 1.5× | 1.8% | 0.282 | 6 | 1.21 |
| candidate: cap p97 + sqrt | 1 | 233 | 286 | 289 | 289 | 1.2× | 1.6% | 0.282 | 6 | 1.17 |
| candidate: cap p97 + log | 0 | 240 | 309 | 313 | 313 | 1.3× | 1.7% | 0.282 | 6 | 1.17 |
| candidate: tiers 5 | 0 | 141 | 332 | 332 | 332 | 2.3× | 1.8% | 0.297 | 6014 | 1.17 |
| candidate: tiers 10 | 0 | 199 | 340 | 340 | 340 | 1.7× | 1.9% | 0.290 | 3008 | 1.17 |
| candidate: cap p97 + log + floor £25 | 25 | 240 | 309 | 313 | 313 | 1.3× | 1.7% | 0.282 | 6 | 1.20 |

###### All scored leads

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| candidate: identity | 0 | 190 | 334 | 346 | 347 | 1.8× | 2.1% |
| candidate: cap p97 | 0 | 190 | 334 | 343 | 343 | 1.8× | 2.1% |
| candidate: cap p97 + floor £25 | 25 | 190 | 334 | 343 | 343 | 1.8× | 2.0% |
| candidate: cap p97 + sqrt | 0 | 215 | 285 | 289 | 289 | 1.3× | 1.8% |
| candidate: cap p97 + log | 0 | 214 | 308 | 313 | 313 | 1.5× | 1.9% |
| candidate: tiers 5 | 0 | 141 | 332 | 332 | 332 | 2.3× | 2.0% |
| candidate: tiers 10 | 0 | 199 | 340 | 340 | 340 | 1.7× | 2.1% |
| candidate: cap p97 + log + floor £25 | 25 | 214 | 308 | 313 | 313 | 1.5× | 1.8% |

###### Acceptance (plan 3.2): ≥ 95% of the baseline's capture on every test set and max/median < 20× on every test set and on all scored leads

Not computed: the criterion is defined against the baseline, which this report omits.

##### Click-ID coverage (plan 3.5)

Scored leads per paid channel by the identifier an upload could match on (see `docs/platform_contract.md`): a click id (gclid, gbraid, wbraid, fbclid, li_fat_id, oppref), else the lead-form id, else nothing but hashed email / phone (Meta: plus the `fbp` browser cookie).

| channel | leads | click id | lead-form id | no id: fbp present | no id at all | email present | phone present |
|---|---|---|---|---|---|---|---|
| google | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| meta | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| linkedin | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| chatgpt | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| meta_leadads | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| landing-page paid (all but meta_leadads) | 0 | nan% | nan% | nan% | nan% | nan% | nan% |
| all paid | 0 | nan% | nan% | nan% | nan% | nan% | nan% |

##### Design collinearity (plan 2.7): FAIL

Candidate design (`generic` features, 341 columns) on its 87812 training rows; fails when two columns have |corr| > 0.95. The report exits 1 on a failure only with `--strict`; `python -m emva.eval.collinearity` always does.

- FAIL (1 pairs with |corr| > 0.95): max |corr| = 0.969 between x_distribution_channel=GDS and x_agent=195
  - x_distribution_channel=GDS ~ x_agent=195: +0.969
- Constant on the training rows (skipped): channel=meta, channel=meta_leadads, channel=linkedin, channel=chatgpt, channel=organic_direct, form_variant=B, form_variant=C, form_variant=D, session_missing=yes, enrichment_missing=yes, band=11-50, band=51-200, band=201-1000, band=1000+, band=missing, email=free, text=copy_paste, text=vague, text=specific, seniority=student, seniority=senior, seniority=mid, seniority=not_asked/blank, spend=none, spend=£5k-£25k, spend=£25k-£100k, spend=£100k+, crm=hubspot_sf, hiring=hiring, time_on_page=60-300s, time_on_page=300-600s, time_on_page=>600s, hesitation_90s=yes, sessions_3plus=yes, viewed_pricing=yes, search_term=brand, search_term=not_google, business_hours=wkday_9-18, ip_country=mismatch, x_hotel=other, x_hotel=missing, x_lead_time=missing, x_arrival_month=other, x_arrival_month=missing, x_weekend_nights=missing, x_week_nights=missing, x_adults=missing, x_babies=missing, x_meal=other, x_meal=missing, x_market_segment=missing, x_distribution_channel=missing, x_repeated_guest=other, x_repeated_guest=missing, x_previous_cancellations=missing, x_previous_bookings_kept=missing, x_reserved_room_type=missing, x_deposit_type=other, x_deposit_type=missing, x_customer_type=other, x_customer_type=missing, x_adr=missing, x_parking_spaces=missing, x_special_requests=missing

##### Generic vs v2 (Phase 10, ADR 0024)

Both models trained on the same rows and split with the same labels: `v2` (the fixed 39-column design) and `generic` (the same plus 302 `x_` columns from 23 declared extras, encoded on the 87812 training leads only). Paired bootstrap of AUC(generic) − AUC(v2) on shared resamples (1000 resamples, seed 0).

| test set | n | v2 AUC | generic AUC | generic − v2 | 95% CI | bootstrap p |
|---|---|---|---|---|---|---|
| (a) Legacy labels, legacy test set | 31578 | 0.500 | 0.801 | +0.301 | [+0.296, +0.306] | 0.000 |
| (b) Horizon labels, mature test set | 25491 | 0.500 | 0.930 | +0.430 | [+0.427, +0.433] | 0.000 |

Generic design collinearity: FAIL (1 pairs with |corr| > 0.95): max |corr| = 0.969 between x_distribution_channel=GDS and x_agent=195 (details in the collinearity section above).

**Phase 10 criterion (pre-registered): FAIL.** PASS iff on the horizon test set (b) the 95% CI of generic − v2 lies entirely above 0 and the generic design passes the collinearity check.

###### Leakage screen

Single-feature training AUC of each extra's encoded levels (each level scored by its training win rate), folded as |AUC − 0.5| + 0.5. Above 0.90: flagged "check for leakage" (a flag only; nothing is dropped).

| extra | kind | levels | design columns | training AUC | flag |
|---|---|---|---|---|---|
| hotel | categorical | 4 | 3 | 0.538 |  |
| lead_time | numeric | 6 | 5 | 0.888 |  |
| arrival_month | categorical | 14 | 13 | 0.613 |  |
| weekend_nights | numeric | 5 | 4 | 0.541 |  |
| week_nights | numeric | 5 | 4 | 0.597 |  |
| adults | numeric | 3 | 2 | 0.504 |  |
| children | numeric | 3 | 2 | 0.504 |  |
| babies | numeric | 3 | 2 | 0.503 |  |
| meal | categorical | 7 | 6 | 0.539 |  |
| country | categorical | 61 | 60 | 0.628 |  |
| market_segment | categorical | 9 | 8 | 0.683 |  |
| distribution_channel | categorical | 6 | 5 | 0.580 |  |
| repeated_guest | categorical | 4 | 3 | 0.517 |  |
| previous_cancellations | numeric | 3 | 2 | 0.556 |  |
| previous_bookings_kept | numeric | 3 | 2 | 0.521 |  |
| reserved_room_type | categorical | 10 | 9 | 0.535 |  |
| deposit_type | categorical | 5 | 4 | 0.624 |  |
| agent | categorical | 129 | 128 | 0.716 |  |
| company | categorical | 26 | 25 | 0.541 |  |
| customer_type | categorical | 6 | 5 | 0.530 |  |
| adr | numeric | 6 | 5 | 0.556 |  |
| parking_spaces | numeric | 3 | 2 | 0.546 |  |
| special_requests | numeric | 4 | 3 | 0.585 |  |

Flagged: none.

##### Baseline script output

- Baseline omitted: this dataset carries its own dates in `dataset.json` (a converted dataset, ADR 0020). The frozen `baseline/emva_score.py` hard-codes the v1 snapshot date, test boundary and file format, so its scores would not be comparable (ADR 0022). The paired comparison against it and the check that the legacy labels equal the baseline's are omitted too.

</details>

## Verification

- Tests: **780 passed, 1 skipped** (`make test`, then compileall) (751 passed + 1 skipped at the start of the phase).
- `make baseline`: PASS. AUC 0.814, Brier 0.1006, top-20% wins 0.571, revenue 0.795, weights and scores
  byte-identical between the baseline and emva runs (on simulated data).
- `python -m emva.eval.report --data data/v1` and `--data data/v2`: byte-identical to the reports saved before the
  phase (`cmp`), so v1 / v2 / legacy outputs are unchanged.
- Conversion of the three real sources: `historical_leads.csv`, `crm_history.csv`, `companies.csv` and
  `people.csv` byte-identical to the Phase 9 conversion (the CRM's third source changes nothing but the extras).
- Grep rule: `grep -rnI "0.45\|ground_truth" emva/` returns only `emva/eval/` lines.

## Open items

- **Keel (part b):** validate `extra_features.csv` (today stored unchecked), offer `generic` on Upload & train, a
  features-confirmed control in Map & convert (the form only carries the flag through), show the "Generic vs v2"
  section and the leakage screen, and ask for the extras on the Score page's EMVA form (a form lead scores with every
  extra `missing`).
- The hotel collinearity failure (`distribution_channel=GDS` ~ `agent=195`) and the horizon label's mechanical tie
  to `lead_time` are properties of that dataset under R28; neither is fixed here (no tuning after the numbers).
- A date-part operation (weekday / month of a date) would allow Olist sign-up timing as an extra; not added.
- Live API draft run with the v2 prompt: still pending (no key in this environment).
- `sales_agent` / `manager` are person names used as categories; fine as model features, but a real customer's
  export would need a data-protection review before training on staff names.
