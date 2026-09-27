# 0026. Keel QA fix round: unseen levels named per field, created_at required for source-format scoring, amounts in the dataset's currency, constant signals folded into the starting score, won-only fields flagged at conversion

- Status: Proposed
- Date: 2026-09-27
- Source: hands-on QA of the live deployment (31dd8b8), issues M1-M4 and M6; `reports/keel-fixes.md`; branches
  `keel-fix-scoring`, `keel-fix-claims`, `keel-fix-leak-warning`

## Context

A smoke test and a hands-on QA pass of the deployed Keel (Phases 9 and 10) found five problems. Each one either
broke a working path or made the page claim more than the data supports. All numbers were on public data (Olist).

- **M1.** On a legacy-features run of a converted dataset, the Score page's example lead was refused, one unseen
  level at a time, with internal feature names.
- **M2.** Source-format scoring accepted a lead with a blank `created_at`.
- **M3.** The scorecard and the breakdown showed design columns that are constant on every training lead as
  signals that move a lead.
- **M4.** Olist amounts (BRL) were formatted as £ with no disclosure.
- **M6.** A column that only won leads have could be mapped to a field without any warning. The leakage and
  missingness screens (ADR 0024) cover extras only, and only after training.

## Decision

1. **Unseen levels, all at once, in form terms** (M1; ADR 0018 unchanged: unknown levels still raise).
   - `UnknownLevelError` carries every offending feature (`.problems`, one `LevelProblem` per feature, in
     `ModelBundle.levels` order). Its message keeps the old wording, one line per feature.
   - Keel shows them by form field ("Company size (typed): 51-200 was not seen in training; allowed: 1-10").
   - When the built-in example would be refused (legacy feature set, whose design keeps only the training levels),
     the form starts from the first lead of `historical_leads.csv` that was created before `test_from`, is not a
     bot or repeat, and has only known levels. The page says so. v2 and generic runs keep the built-in example.
2. **`created_at` is required on every new source-format lead** (M2; amends the Phase 9 source-format rule,
   ADR 0023). `emva.ingest.convert_leads`, and with it Keel's source format and `python -m emva.ingest score`,
   refuses a missing `created_at` column or a blank value, naming the rows. It no longer fills in the current time.
   No model feature reads `created_at`; the within-batch duplicate check orders submissions by it.
3. **Amounts in the dataset's currency** (M4). A mapping may declare `currency` (ISO 4217, three upper-case
   letters; absent = unknown). `convert` writes it to `dataset.json` (null when unknown).
   - Keel formats every amount with one rule, `app.components.money`:

     | Dataset | Shown as |
     |---|---|
     | No `dataset.json` (bundled sample, EMVA-format upload) | £ |
     | Declared code | `CODE 1,234` (GBP shows as £) |
     | Unknown | the plain number, plus a page note that amounts are in the source's currency |

   - The value transform's cap, median and floor are applied in the source's units, so they are labelled in them
     too ("floored at BRL 25").
   - `olist_funnel` declares BRL. `crm_opportunities` and `hotel_bookings` stay unset: no source consulted states
     the currency of their deal amount.
   - The standard report's "GBP" wording is unchanged (the report output must not change).
4. **Constant signals are part of the starting score** (M3).
   - A design column with no variance on the run's training rows (the collinearity check's `constant`) is not a
     signal. Keel finds these columns by rebuilding the training design, without fitting.
   - The results scorecard leaves them out and adds one line: the ones on for every lead by name, with their summed
     points, which belong to the starting score, and the ones never on by count.
   - The Score breakdown folds active constant columns into "Starting score (incl. N …)" and lists no row that
     rounds to 0. The total shown is the sum of the rounded parts.
   - `weights.csv`, `model.joblib` and the downloads are unchanged.
5. **Won-only fields are flagged at conversion, never blocked** (M6; extends the ADR 0024 missingness screen to
   fields and to conversion time). For every field row that sets a target, and every declared extra, `convert`
   compares the share of Won leads whose source value is filled with the share of Lost leads. A gap of at least
   `GENERIC_MISSINGNESS_FLAG` (0.5) is recorded in `dataset.json` as `outcome_fill_gaps`. The CLI prints the gaps,
   Keel shows them as an amber warning after Convert, and validation keeps a note on every saved dataset. Nothing is
   dropped.

## Consequences

- Leads that scored before score identically. `make baseline` passes, and the v1 standard report is byte-identical.
- Older stored datasets keep working. A `dataset.json` without `currency` or `outcome_fill_gaps` loads as unknown
  currency and no flags. New converted datasets carry both keys.
- A source-format CSV without a filled creation date is now refused rather than scored.
- Open items:
  - `VALUE_FLOOR_GBP` is misnamed for non-GBP datasets.
  - The review table still shows a row's stale reason after its target is edited.
  - The fill-gap warning appears only after Convert, not while the table is edited.
