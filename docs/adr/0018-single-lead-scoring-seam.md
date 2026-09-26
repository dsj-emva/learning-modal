# 0018. Single-lead scoring reuses the training path; unknown levels raise; equality with the batch is to the last bits

- Status: Proposed
- Date: 2026-09-25
- Source: Phase 8 (hosted internal tool), branch `app-ui`; `emva/persist.py`, `emva/scoring.py`, `tests/test_scoring.py`

## Context

The Streamlit app scores one lead (or a few) at a time with the models `python -m emva` trained. Three questions
had to be settled: how to guarantee the app scores a lead the way training did, what to do with a categorical value
the model never saw, and what "the same score" can mean numerically.

## Decision

1. **One code path.** `python -m emva --out DIR` also writes `model.joblib` (`emva.persist.ModelBundle`:
   format_version 1). `emva.scoring.score_leads` calls the training functions (`io.enrich`, the bot/duplicate
   flags of `io.clean`, `FeatureSpec.featurise`/`design`/`value_design`, `model.predict`,
   `value.predict_deal_value`, `value.expected_value`, `FittedValueTransform.apply`); nothing is re-implemented.
   `io.load` was split so its submit-time half (`enrich`) needs no CRM history.
2. **Unknown categorical levels raise** `UnknownLevelError` naming the feature, the values, the leads and the allowed
   levels (`ModelBundle.levels`). Mapping them to a missing level would price the lead with a coefficient that
   means something else.
3. **Bots and duplicates are flagged, not dropped**, when scoring; duplicates are detected within the scored batch
   only (a lead scored alone is never a duplicate of the history).
4. **Numerical equality.** Scoring every cleaned lead in one call is byte-equal to the batch run (`p_formula`,
   `deal_value_hat`, `value_formula`, `value_at_submit`; v1 and v2, both feature sets). A smaller batch builds a
   byte-equal design matrix, but sklearn's `predict_proba` and `Ridge.predict` are BLAS matrix products whose
   kernel and summation order depend on the number of rows, so scores differ from the batch run by at most
   14 ulp (`p_formula`), 16 ulp (`deal_value_hat`, `value_formula`) and 62 ulp (`value_at_submit`), measured
   over 3,000 leads in batches of 10 and 600 single leads per dataset and feature set (Accelerate BLAS, x86_64).
   The tests assert byte-equal design matrices and full batches, and at most 64 ulp for small batches. Making
   the small-batch scores byte-equal would need a row-order-independent dot product in `model.predict`, which
   would move the batch pipeline's own scores by an ulp and break `baseline/` byte identity (ADR 0007); not done.

## Consequences

- A bundle from another `format_version` is refused; other numpy / pandas / scikit-learn versions warn.
- The app needs a dataset directory with `companies.csv` next to the bundle (the enrichment join).
- Context-mode models (`--context`) are not saved; the bundle scores the formula model only.
