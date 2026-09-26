# 0024. Generic feature set: declared, person-confirmed extra source columns, encoded on the training leads only

- Status: Proposed
- Date: 2026-09-26
- Source: Phase 10 (a), `reports/phase10.md` "Orchestrator decisions" D1-D10 (decisions 1-8), Phase 10 (b), D15-D22
  (decisions 9-14, proposed at part b), and the Phase 10 fix round, D23-D33 (decisions 15-18); `emva/generic.py`,
  `emva/feature_spec.py`, `emva/pipeline.py`, `emva/persist.py`, `emva/scoring.py`, `emva/ingest/`,
  `emva/eval/generic_report.py`, `emva/eval/report.py`, `mappings/*.toml`, `app/` (part b); branch
  `phase10-generic-features`.
  Amends ADR 0009 (fixed design schema) and ADR 0017 (decision 2: no customer-specific columns)

## Context

Phase 9 converted three public Kaggle datasets with hand-written mappings. The fixed v2 schema (ADR 0009: 39
declared columns from form answers, UTM tags, email, on-site session telemetry and company enrichment) barely covers
them: Olist fills 3 of 39 signals (horizon test AUC 0.553 [0.530, 0.574]), CRM opportunities and hotel bookings 0
of 39 (AUC 0.500 exactly), all on public data. The columns these sources do have (product, sector, agent, lead time,
deposit type, landing page, ...) had nowhere to go: ADR 0017 decision 2 kept customer-specific columns out of the
formula design so that coefficients stay comparable across customers. The user decided to build the deferred
"generic feature set" that also learns from a source's extra columns.

## Decision

1. **Extras are declared in the mapping and confirmed by a person.** A mapping may list `[[features]]`, each
   `{source, kind = "numeric" | "categorical", name, reason, confidence}`; `source` is an expression (a column or
   one of the mapping's ops, e.g. hotel nights = weekend + week nights); `name` is a lower-case slug (raw column
   `x_<name>`, design columns `x_<name>=<level>`). A top-level `features_confirmed` says a person checked that
   every declared extra is known at submit time (leakage guard). `convert` and `convert_leads` refuse a mapping
   with `[[features]]` and `features_confirmed = false`. A mapping without `[[features]]` loads, converts and dumps
   exactly as in Phase 9. The LLM draft may propose extras (target `feature` with `feature_kind`;
   `PROMPT_VERSION` `mapping-draft-v2`) but always sets `features_confirmed = false`.
2. **Extras live beside the fixed schema.** `convert` writes `extra_features.csv` (`lead_id` plus one raw column per
   declared extra, no encoding) and a `features` list (`name`, `kind`, `source`) in `dataset.json`.
   `historical_leads.csv` stays exactly the fixed schema. The pipeline reads the file only with the generic feature
   set and refuses loudly when it is absent.
3. **Encoding is fitted on the training leads only and frozen** (`emva.generic.fit_encoder`, on the pipeline's
   train mask). Categorical: levels with at least `GENERIC_MIN_LEVEL_COUNT = 30` training leads are kept, plus
   `other` (rare and unseen values) and `missing`; reference = the most frequent kept level (ties by value).
   Numeric: `GENERIC_NUMERIC_BINS = 5` quantile bins (`inverted_cdf`, so edges are observed values; duplicate edges
   collapsed; an edge at the training maximum moved to the largest value below it), plus `missing`; reference = the
   **first bin** (deterministic; `missing` when no training lead has a value). The encoder is stored in the model
   bundle (`ModelBundle.extras`, `FORMAT_VERSION` 2) so `emva.scoring` reproduces the same columns.
4. **Amendment to ADR 0009, extras only:** at scoring an unseen categorical value is `other`, a defined level, so it
   is not refused; a numeric value outside the training range falls into the edge bin. The fixed-schema columns
   still raise on an unknown level. The extras' design column set is fixed by the training data of one run
   (not by a declaration), and is frozen with the model.
5. **`FeatureSet.GENERIC`** (`--feature-set generic` on `python -m emva`, `python -m emva.eval.report` and
   `python -m emva.eval.collinearity`): the v2 fixed 39 columns plus the `x_` columns. v2 stays the default; v1, v2
   and legacy outputs are byte-identical. The deal-value model is unchanged (extras are not used there).
6. **Report.** With a generic candidate the standard report trains a v2 model on the same data, labels and split and
   adds "Generic vs v2": the paired bootstrap AUC difference generic − v2 (1000 resamples, seed 0) on both frozen
   test sets, the generic design's collinearity check, the pre-registered criterion's verdict, and a leakage screen
   (per extra, the folded single-feature training AUC of its encoded levels; above 0.90 flagged "check for
   leakage", never dropped).
7. **Pre-registered acceptance (ground rule 5):** per dataset, PASS iff on the horizon test set (b) the 95% CI of
   generic − v2 lies entirely above 0 and the generic design passes the collinearity check. Bins, minimum counts,
   feature lists and H are not tuned after seeing test numbers.
8. **Amendment to ADR 0017 decision 2:** customer-specific columns may enter a per-customer model as declared
   extras. They carry no cross-customer meaning; the fixed 39 columns stay the shared, comparable part and the
   precondition for pooling. A hierarchical model (ADR 0017 decision 3) pools only the fixed columns unless a later
   ADR maps extras across customers.

9. **(b) Old bundles still load.** `load_bundle` reads format 1 (Phase 8-9 runs, e.g. on Railway) as a bundle with
   `extras = None`, equivalent to its v2 / legacy run; format 2 is written; any other version is refused.
10. **(b) `extra_features.csv` is a training file** of the dataset store (`app.storage.TRAINING_FILES`), carried only by
    a converted dataset whose `dataset.json` declares its `features`; `app.validation` checks it (columns = `lead_id`
    plus the declared names in order, unique `lead_id`, the same leads as `historical_leads.csv` both ways, numeric
    extras numbers or blank) and refuses the file and the declaration apart.
11. **(b) Declaring and confirming extras in Keel.** Extras are a `feature` / `kind` / `name` part of the field review
    table (a column may be a field and a feature at once; derived features are TOML only). A second required
    confirmation, "Extra features are known when the lead is submitted", appears only with declared features, is
    keyed by them (any change clears it) and is never carried over from a loaded mapping.
12. **(b) Training.** Keel offers `generic` only for a dataset with `extra_features.csv`; v2 stays the default.
13. **(b) Results.** A generic run shows "Generic vs v2" computed with `emva.eval.generic_report.compare` on the
    selected test set (both models retrained in the server on the same leads, labels and split; refused when the
    retrained model does not reproduce the run's scores), the generic design's collinearity check and the leakage
    screen, but not the Phase 10 acceptance verdict (a pre-registered criterion for three public datasets, not a rule
    for a customer run). Long scorecards show the 25 strongest signals plus an expander.
14. **(b) Scoring.** The source format carries the extras (a form lists them, with a generic bundle's training levels
    plus `other` for a categorical extra and a number input for a numeric one; a CSV upload has them as columns). The
    EMVA form has no inputs for extras and says that a lead scored there has every extra blank: `missing`, or the
    reference level wherever the model never saw a missing value (no training lead had it missing, so the L2 weight
    of `missing` is 0); Keel warns per such extra, on the form and for a blank extra in the source format (fix round).

15. **(fix round) Missingness screen.** The leakage screen (report and Keel) also shows, per extra, the share of
    cleaned leads with the extra missing among closed leads (`label_source` won / crm_lost) and among open ones
    (open / stalled / ghosted), at the run's `as_of` whatever the label mode, and flags "missingness tracks outcome
    status — check for leakage" at an absolute difference >= `GENERIC_MISSINGNESS_FLAG = 0.5`. A flag only, never a
    drop; the threshold was fixed before the Kaggle results were recomputed. It exists because censored horizon
    labels hide from the AUC screen a column that is blank only while a deal is open (the CRM `account`).
16. **(fix round) Bundle consistency.** `load_bundle` refuses a bundle unless (feature set is generic) == (extras is
    not None). A generic bundle refuses `x_` columns it does not declare (a misspelt extra); a bundle without extras
    still accepts and ignores them (Phase 10 D12).
17. **(fix round) Refuse malformed extras loudly.** `extra_features.csv` must hold exactly the leads of
    `historical_leads.csv` in the pipeline as in `app.validation`; numeric extras refuse non-finite values (`inf`,
    `-inf`, `1e400`) when read, converted, validated and scored.
18. **(fix round) Draft names never clash.** A drafted feature's name is its column's slug, prefixed with the source
    name when that slug is drafted twice, then given the first free suffix `_2`, `_3`, ... (reply order) while it is
    still taken; the same column drafted twice as a feature is a `ContractError`.

## Consequences

- A converted source with no form, session or enrichment data can still be scored on its own columns, measured
  against v2 on the same rows.
- Leakage moves from "cannot happen" (no extra columns) to "a person's confirmation plus a screen": a post-outcome
  column that a reviewer wrongly confirms will inflate the numbers; the screen flags only the blatant cases.
- Bundles written before Phase 10 (format version 1) load as bundles without extras (decision 9), so deployed runs
  keep scoring; only format 2 carries an encoder.
- The extras' columns differ per dataset and per run, so their coefficients are not comparable across customers.
- Keel runs the comparison in the server process: two pipeline fits and a bootstrap per run and test set (cached),
  seconds on Olist, longer on a dataset the size of the hotel bookings.
- The missingness screen flags, on the three public datasets, exactly the five CRM account extras (0.000 missing
  among closed leads vs 0.685 among open ones); Olist and hotel have no open leads.
- Open: date-part operations (weekday, month of a date) are not in the mapping's op set; the EMVA form has no inputs
  for extras; the CRM account missingness is disclosed, not fixed (post hoc).
