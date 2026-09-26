# Architecture decision records

One file per decision or orchestrator ruling that changes what the code or the reports must do.
Rulings are made at the two-axis review of a phase (standards + spec) and are also recorded in that
phase's report under "Orchestrator decisions"; the ADR is the durable, findable copy.

## Format

File name `NNNN-kebab-case-title.md`, numbered sequentially from 0001. Numbers are never reused or
renumbered. Each file has:

```
# NNNN. Title (the decision, stated as a sentence)

- Status: Proposed | Accepted | Accepted, implementation in progress | Superseded by NNNN
- Date: YYYY-MM-DD
- Source: the report section, plan item or commit where the ruling is recorded

## Context      what forced a decision, with the numbers that mattered
## Decision     what was decided, precisely enough to check code against it
## Consequences what this makes true, costs, or leaves open
```

To change a decision, write a new ADR that supersedes the old one and set the old one's status to
"Superseded by NNNN"; do not rewrite history. Every merged phase adds an ADR for each ruling it received.

## Index

| # | decision | status | source |
|---|---|---|---|
| [0001](0001-adjust-in-place-behind-frozen-baseline.md) | Adjust in place behind a frozen baseline, not rewrite | Accepted | `REBUILD_PLAN.md` |
| [0002](0002-report-conventions.md) | Report conventions: recorded deal value, tie-averaged top-20%, canonical weights comparison | Accepted | `reports/phase0.md` |
| [0003](0003-v1-distributional-fidelity.md) | Exact reproduction of v1 is infeasible; distributional fidelity is the 5.1 pass criterion | Accepted | `reports/phase5-1.md` |
| [0004](0004-band-weights-against-true-120-day-effects.md) | Band-weight criterion restated against true 120-day effects (R1) | Accepted | `reports/phase1.md` |
| [0005](0005-open-mature-non-wins-are-zero-stalled-censored.md) | Mature still-open non-wins are 0; stalled stays censored (R2) | Accepted | `reports/phase1.md` |
| [0006](0006-frozen-mature-test-set-rolling-origin-headline.md) | The 458-lead mature test set is frozen; Phase 4 rolling-origin becomes the headline (R3) | Accepted | `reports/phase1.md` |
| [0007](0007-legacy-scores-csv-byte-identity.md) | `label_source` omitted from legacy `scores.csv` to keep byte identity (R4) | Accepted | `reports/phase1.md` |
| [0008](0008-persona-always-readable-from-text.md) | Persona always readable from text; only on non-vague, non-copy-paste texts | Accepted, implementation in progress | `phase5-generator-v2` |
| [0009](0009-fixed-design-schema-missingness-indicators.md) | Fixed design schema: `session_missing` / `enrichment_missing` indicators, declared level lists, strict collinearity check (report `--strict`) | Accepted, implementation in progress | `phase2-features` |
| [0010](0010-llm-api-access.md) | API access: `.env`, workspace header, Haiku 4.5, committed caches, no fake LLM output | Accepted | Phase 5 v2, `.gitignore` |
| [0011](0011-phase2-acceptance-on-v2-data.md) | Phase 2 tie/collinearity criteria judged on data/v2; boilerplate recall gap to Phase 6; v2 residual sd as 2.4 evidence (R5-R7) | Accepted | `phase2-features` |
| [0012](0012-value-transform-default.md) | Default value transform: cap p97, log compression rescaled to the training total, floor £25; wins without an amount get a blank `value_at_close` (R9-R11) | Accepted | `phase3-value` |
| [0013](0013-rolling-origin-headline-definition.md) | Rolling-origin headline: training mature at the split, one-month mature test windows, mean AUC over sufficient splits on v2; `make report-full` inherits Phase 4 (R16) | Accepted | `reports/phase4.md` |
| [0014](0014-context-contract.md) | Context contract: five enum judgments + reason, four-field lead card, parse vs transport errors, stamped rows, cache keyed by card/brief/prompt version/prompt fingerprint/model | Accepted | `phase6-context-agent` |
| [0016](0016-context-acceptance-on-coefficients.md) | Context acceptance on coefficients (R12, judged one feature at a time next to the formula): weight CI excludes 0 and lies in the oracle CI; persona pooled to buyer / non_buyer / unclear at the feature level (R13-R15) | Accepted | `phase6-context-agent` |
| [0017](0017-multi-customer-roadmap.md) | Per-customer LR on the fixed schema first; a hierarchical model (shared prior, per-customer adjustment) is the multi-customer path; the LLM stays a measured feature; ingestion adapters per CRM platform | Proposed | `phase7-production-doc` |
| [0018](0018-single-lead-scoring-seam.md) | Single-lead scoring (`emva.scoring`) reuses the training path via `model.joblib`; unknown categorical levels raise; small-batch scores equal the batch run to ≤ 64 ulp (BLAS summation order), full batches byte-equal | Proposed | `app-ui` |
| [0019](0019-hosted-app.md) | Hosted app (Keel): results page shows the full standard table from `emva.eval.report` (baseline, model, status quo); "Upload & train" naming; Railway settings on the service, `RAILWAY_RUN_UID=0`; floating base image, exact pins; shared-password trust model, no ground truth in the image (R18-R21) | Proposed | `app-ui` |
