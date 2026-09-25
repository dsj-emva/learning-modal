# 0017. Per-customer LR on the fixed schema first; a hierarchical model is the multi-customer path; the LLM stays a feature

- Status: Proposed
- Date: 2026-09-25
- Source: Phase 7, `reports/production.md` sections 7 and 8; branch `phase7-production-doc`

## Context

An external data scientist argued that EMVA should not build one model per customer but either (a) a single pooled,
industry-agnostic model with industry as a feature, trained across customers, or (b) LLM-only profiling of each lead
against the customer's brief, with no fitted outcome model.

What the POC shows, all on simulated data:

- Per-customer L2 logistic regression beats a monotone GBDT at every training size on both datasets (v2, N = 2,000:
  0.789 ± 0.005 vs 0.756 ± 0.012; full pool 0.794 vs 0.783; `reports/phase4.md` §4.2).
  The generators are additive logistic, so this comparison favours LR by construction.
- Customer-sized pools are small: 1,000 to 4,000 labelled rows a year for the customers `reports/production.md`
  section 6 would accept, and calibration is over-confident at small N (slope 0.70 to 0.91, Phase 4 §4.5).
- The design schema is fixed: 39 declared columns whatever the data (ADR 0009), so coefficients from different
  customers mean the same thing.
- The context agent returns enum judgments, never a score (ADR 0014). On v2 the planted context-only signal is worth
  about +0.005 AUC even when read perfectly (Phase 4 §4.4, ADR 0013); acceptance for context features is
  coefficient-level (ADR 0016).
- The ad platforms receive `p × E[deal value]` as a conversion value (ADR 0012, `docs/platform_contract.md`), so the
  score must be a calibrated probability.
- There is exactly one (synthetic) advertiser. No second customer's outcomes exist to pool, and no data-sharing terms
  exist that would allow it.

## Decision

1. **Day one: one regularised LR per customer on the fixed feature schema**, trained on that customer's horizon labels
   and validated on that customer's rolling-origin backtest (ADR 0013).
2. **The fixed schema is kept as the precondition for pooling.** No customer-specific columns are added to the formula
   design; new levels go into the shared `V2_LEVELS` declaration. This keeps coefficients comparable across customers.
3. **The multi-customer evolution is a hierarchical model**: a shared prior over the schema's coefficients with a
   per-customer adjustment (partial pooling), which shrinks small customers toward the population and lets large
   customers keep their own weights. Industry enters as a level of the hierarchy (or a feature) once several customers
   share an industry. Preconditions: several customers' labelled outcomes, each at least 120 days mature, and contract
   terms that allow training on pooled outcomes. It must beat per-customer LR on each customer's own rolling-origin
   backtest (paired CI) before it replaces it.
4. **The LLM stays a measured feature, not the scorer.** Its judgments enter the per-customer (later hierarchical)
   model as bucketed features and are admitted one feature at a time by the coefficient criterion (ADR 0016). An
   LLM-only score is not used because it is not a calibrated probability and cannot pass the pilot protocol.
5. **Ingestion adapters are per CRM platform, not per customer**: one adapter per CRM system maps exports and stage
   names to the inputs `emva.io.load` expects, so a new customer on a known CRM is configuration.

## Consequences

- Small customers are declined or piloted with conditions until pooling exists (`reports/production.md` section 6);
  the hierarchical model is what would lower that threshold.
- Industry-agnostic pooling without a hierarchy is not pursued: a single set of weights would average away
  per-customer differences in funnel, sales process and deal size that the customer's own backtest would expose.
- Changing the design schema becomes a cross-customer decision (it breaks comparability), not a per-customer tweak.
- The pooled model needs its own evaluation design (leave-one-customer-out and per-customer rolling origin) and a
  data-governance review before any customer's outcomes are shared.
- If a pilot shows the LLM judgments carry large, stable, calibrated signal on real data, revisit point 4 with a new
  ADR; nothing on simulated data supports it today.
