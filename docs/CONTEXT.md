# Context: ubiquitous language, headline numbers, status

Every term below is defined **as implemented** (file cited). If code and this file disagree, the code
wins and this file is wrong: fix it in the same PR. State: `main` 92168cf, 2026-09-25.

## Glossary

### Leads and cleaning

- **Lead**: one form submission, a row of `historical_leads.csv` keyed by `lead_id`. After
  `emva.io.load` it carries `final_stage`, `last_change`, `deal_value`, `won_at`, `first_contact_at`,
  `a_<answer>` columns and `co_<enrichment>` columns.
- **Bot**: `user_agent` contains "Headless" (case-insensitive) **or** `time_on_page_s < 15`
  (`BOT_MIN_TIME_ON_PAGE_S`; `emva/io.py::flag_bots_and_duplicates`). Dropped before labelling. On v1 the
  rule is 100% precise because the generator made it so; on v2 precision is 72.7% (fast humans, 5.8).
- **Duplicate**: any submission that is not the first (by `created_at`) for its normalised email
  (`email.strip().lower()`). Dropped; the first submission is kept.
- **Scored leads**: leads left after bot and duplicate removal (9,311 on v1).
- **Enrichment**: the `companies.csv` row joined by normalised company domain, as `co_sector`,
  `co_employee_band`, `co_monthly_ad_spend_band`, `co_crm_platform`, `co_is_hiring` (NaN when the domain
  misses). Phase 2 (in progress) adds a fallback exact match on the normalised typed company name
  (`en_<column>`, `enrichment_source`).
- **Band**: the company-size feature. `co_employee_band`, else the typed `a_company_size` answer, else
  `"1-10"` (legacy, the reference level) or `"missing"` (Phase 2 v2 features). Levels 1-10, 11-50,
  51-200, 201-1000, 1000+.

### CRM state and labels (`emva/labels.py`)

- **AS_OF**: `2026-09-24T00:00Z`, the data snapshot (`emva/constants.py`). All ages are measured to it.
- **Stalled** (condition): final stage in `OPEN_STAGES` (Contacted, Qualified, Demo booked, Proposal)
  and no CRM change for ≥ 90 whole days at AS_OF.
- **label_source** (horizon mode only; first match wins):
  - `won`: last CRM stage is Won.
  - `crm_lost`: last CRM stage is Lost.
  - `ghosted`: mature and not contacted by `matured_at` (no change from New to another stage by then).
    Takes precedence over stalled and open.
  - `stalled`: the stalled condition above.
  - `open`: everything else undecided: open stage changed < 90 days ago, or still New but younger than H.
  An unrecognised final stage raises in horizon mode; legacy mode leaves it unlabelled.
- **Legacy label** (`label`, `--label-mode legacy`): Won = 1; Lost = 0; stalled = 0; final stage New and
  ≥ 90 days old = 0 (`GHOSTED_DAYS`); otherwise NaN. The baseline's rules, kept for comparison. Their
  defect: slow, big or neglected leads are labelled lost (right censoring).
- **Horizon label** (`--label-mode horizon`, default): H = `HORIZON_DAYS` = 120, the same for every lead.
  - `matured_at` = `created_at + H`. **Mature lead**: `matured_at <= AS_OF`.
  - `won_within_h` (outcome): 1 if Won at or before `matured_at` (even if not yet mature); 0 if mature and
    not Won by then (late wins are 0); NaN if not mature and not Won ("never Lost").
  - `y` = `horizon_label`: `won_within_h` with ghosted-at-H leads censored (`--include-ghosted` → 0) and
    stalled leads censored (`--stalled-as-lost` → 0). Mature leads still in an open stage and not Won
    are 0 (ADR 0005).
  - Only mature leads train or evaluate (`LabelConfig.eligible`).

### Test sets, models, metrics

- **Frozen test sets** (ground rule 4; ADR 0006):
  - **Legacy**: leads labelled by the legacy rules, created on or after `TEST_FROM` = 2026-05-01:
    **2,453 leads, 352 won** on v1. Training: created before 2026-05-01 (5,588 rows, 847 wins).
  - **Mature**: leads created on or after 2026-05-01 and mature under H = 120 (created before
    2026-05-27T00:00Z; the latest is 2026-05-26T20:06Z), labelled by the default horizon definition:
    **458 leads, 80 won** on v1 (650 cleaned leads minus 141 ghosted and 51 stalled). Fixed to the default
    flags whatever the candidate uses (`emva.eval.report.FROZEN_HORIZON`). Horizon training: 4,049 rows,
    752 wins. On v2: 448 leads, 75 won (in progress).
- **Baseline**: the frozen `baseline/emva_score.py`, run fresh by the report. Never edited.
- **Candidate**: the current `emva` pipeline with the options under test (default: horizon labels).
- **Status quo**: the advertiser's existing rule-based value, reconstructed from `status_quo_rules.json`
  (`emva/eval/status_quo.py`): `base_value_by_form[variant]` × 1.2 if UK/US × 1.3 if LinkedIn × tier
  multiplier. Tier from lead-score points (job title keywords + typed company size + pages visited):
  A ≥ 55 points (×2.0), B ≥ 30 (×1.0), C otherwise (×0.4). Has no probability, so no Brier or calibration.
- **Top-20% wins / revenue**: share of all test wins (or of recorded won deal value) captured by the
  20% of test leads ranked highest, by p or by p×value. Ranking uses numpy's default argsort; with ties at
  the cut the report also prints the **tie-averaged** value (expected share over all tie orders).
  Revenue = recorded deal value, blank = 0 (ADR 0002).
- **Value** (today): `p × deal_value_hat × margin` (`emva/value.py::expected_value`), with
  `deal_value_hat` = exp(ridge prediction of log value) × exp(sd²/2).
- **value_at_submit / value_at_close**: planned, Phase 3 (plan 3.3), not implemented: the predicted value
  sent at submit, and the actual deal value (or 0) sent as an adjustment when the deal closes.
- **Reference level**: the level of each feature that gets no design column; every weight is relative to
  it (`CATS` in `emva/constants.py`, e.g. band `1-10`, channel `google`, email `business`).
- **Scorecard points**: `log_odds × 20 / ln 2`, rounded, in `weights.csv`; **+20 points = odds of closing
  double** (`POINTS_TO_DOUBLE_ODDS`).

### Context layer and generator

- **Context agent**: `emva/context/agent.py`, one Haiku call per lead card (free text, title, typed
  company, email domain, country, some answers and, today, enrichment bands), returning a 0-1
  `context_score`; the pipeline appends `context_logit` as a feature with `--context`. Rewritten in
  Phase 6 (named enum judgments, no enrichment in the card). It has not been run on real LLM output yet.
- **Persona**: in generator v2 (5.7), a hidden `context_persona` (job seeker, agency, nonprofit,
  competitor) on about 8% of genuine-human leads, planted effect −1.0 on close log-odds, readable only
  from the meaning of the free text (invisible to the regex; ADR 0008). In Phase 6, `persona` is also the
  name of one of the context agent's enum judgments.
- **Paraphrase cache**: `scripts/paraphrase_cache.json` (v2 branch, committed): Haiku paraphrases of
  each free-text template, 5 per template, keyed by sha256 of (template, model id, prompt version). The
  generator only reads it. The context agent keeps its own cache (`<out>.cache.json`, keyed by sha256 of
  system prompt + lead card).
- **Generator v1**: `scripts/generate_data_v1.py`, the seeded rewrite of the lost original; reproduces v1
  distributions, not rows (ADR 0003). Its close model is additive logistic, which is why a correctly
  specified LR and the exact regex match it perfectly.
- **Generator v2** (in progress): the same generator with plan 5.2-5.8 switched on: Haiku paraphrase,
  typos, DE/FR/NL language mixing, wider boilerplate (5.2); 30% enrichment domain dropout and noisy typed
  company names (5.3); 15% consent-declined website sessions with blank telemetry (5.4); two interactions,
  LinkedIn × 51+ employees +0.8 and senior × form D −0.8 (5.5); ghosting driven by status-quo tier (5.6);
  the persona signal (5.7); 2% of genuine humans under 15 s on the page (5.8). Output in `data/v2/`.

## Headline numbers (on simulated data)

Claims policy: nothing from v1 is quoted externally. Internally, always with the CI and "on simulated data".

| number | value | source |
|---|---|---|
| Baseline AUC, legacy test set (v1) | **0.814 [0.789, 0.836]**; Brier 0.1006; top-20% wins 0.571; revenue 0.795 (baseline's own imputed definition) / 0.796 by p×value (recorded) | `reports/phase0.md` |
| Status quo AUC, legacy test set (v1) | **0.639** [0.605, 0.673]; top-20% wins 0.392 (tie-averaged 0.397); revenue 0.454 | `reports/phase0.md` |
| Baseline and horizon candidate AUC, mature test set (v1) | **0.778 [0.726, 0.827]**; paired diff −0.001 [−0.013, +0.010] | `reports/phase1.md` |
| Horizon candidate AUC, legacy test set (v1) | 0.812 [0.788, 0.836] | `reports/phase1.md` |
| Regenerated v1, 8 seeds | AUC 0.805 ± 0.013 | `reports/phase5-1.md` |
| Baseline AUC on v2, legacy test set | 0.785 [0.758, 0.810] (in progress, may change) | `phase5-generator-v2:reports/phase5.md` |

The mature test set spans 26 days, so its CI is about ±0.05, twice the legacy width; compare models on
it with the paired bootstrap.

## Status (2026-09-25)

| phase | state | branch | report |
|---|---|---|---|
| 0 freeze and instrument | merged 15f90fc | `phase0-freeze-instrument` | `reports/phase0.md` |
| 5.1 generator v1 rewrite | merged ed3f54b | `phase5-generator-v1` | `reports/phase5-1.md` |
| 1 labels | merged 92168cf | `phase1-labels` | `reports/phase1.md` |
| 2 features and leakage | in progress: indicator encoding after ruling (ADR 0009), report being updated | `phase2-features` | `reports/phase2.md` |
| 5.2-5.8 generator v2 | in progress: persona assignment fix (ADR 0008), report being updated | `phase5-generator-v2` | `reports/phase5.md` |
| 3 value layer and platform contract | pending (needs 2) | | |
| 4 evaluation hardening | pending (needs 2) | | |
| 6 context agent v2 | pending (needs 2 and 5) | | |
| 7 production readiness document | pending (needs all) | | |

Open question carried from Phase 1 (not yet ruled): should young `crm_lost` leads be 0 rather than NaN?
It changes only `scores.csv` today.

**Rule:** every merged phase updates this table, the glossary for any new term, and adds an ADR for each
orchestrator ruling.
