# EMVA scoring POC: adjustment plan

Written 2026-09-25 after the review of `LearningPYApp/` against `testLearningData/`.
Decision: **adjust in place, behind a frozen baseline. Do not rebuild from scratch.**

Why adjust: the pipeline shape (load → build → design → fit → report) is correct, the
baseline reproduces exactly (test AUC 0.814, top-20% wins 0.571, revenue 0.795, identical
`weights.csv`), and every defect found is confined to one layer. A rewrite would lose the
regression reference that each fix must be measured against.

What does get rewritten: the context agent's output contract (small file) and a second data
generator (not in the repo today).

---

## Ground rules for every agent task

1. **Never edit `baseline/`.** It is the frozen copy of today's `emva_score.py`. Every task
   reports its metrics next to the baseline's on the same frozen test set.
2. **Ground truth is evaluation-only.** No code under `emva/` may import, read, or hard-code
   anything from `ground_truth_labels.csv` or `ground_truth.md`. Only `emva/eval/` may read
   them. A grep for `ground_truth` and for the constant `0.45` in `emva/` must be empty.
3. **Standard report table.** Every task ends by running `python -m emva.eval.report` and
   pasting the output into `reports/<task>.md`. The table has, for baseline and candidate:
   AUC with bootstrap 95% CI, Brier, top-20% wins, top-20% revenue (ranked by p and by
   p×value), calibration by decile, AUC by test month, and value-scale stats
   (max/median, top-1% share).
4. **Frozen test set.** Test = labelled leads created on or after 2026-05-01, exactly as the
   baseline defines it, until Phase 1 replaces the labelling rule. After Phase 1 the new
   test definition is frozen again and both are reported for one release.
5. **One task, one PR, one report.** Acceptance criteria below are pass/fail. A task that
   fails its criterion still ships its report; it does not ship its code.
6. **Missing input to flag on day one:** `scripts/generate_data.py` (seed 20260924) is
   referenced in `ground_truth.md:3` but is not in the repo. Phase 5 needs it or rewrites
   it from `ground_truth.md`.

---

## Target layout

```
EMVA/
  baseline/
    emva_score.py            frozen copy, never edited
    context_agent.py         frozen copy
    weights.csv              frozen copy
  emva/
    __init__.py
    io.py                    load CSVs, normalise stages, join enrichment (from load())
    labels.py                label rules, label_source column, maturity mask
    features.py              bucketing with explicit missing levels (from build())
    design.py                dummy matrix (from design())
    model.py                 fit, predict, scorecard export
    value.py                 deal-value model, capping/compression, value to send
    context/
      agent.py               LLM call, cache, versioning
      contract.py            the JSON schema the LLM must return
      features.py            turns agent output into bucketed features
    eval/
      report.py              standard table (rule 3)
      bootstrap.py           CIs, paired comparisons
      rolling.py             rolling-origin splits
      ceiling.py             oracle-feature ceiling (reads ground truth: allowed here only)
      status_quo.py          reconstruction of status_quo_rules.json
  scripts/
    generate_data_v1.py      original generator, if it can be recovered
    generate_data_v2.py      Phase 5
    run_experiments.py       runs a named experiment end to end
  data/
    v1/                      today's testLearningData (moved, not copied)
    v2/                      Phase 5 output
  reports/                   one markdown file per task
  requirements.txt
  Makefile                   make baseline / make report / make experiment NAME=...
```

---

## Phase 0: freeze and instrument (prerequisite for everything)

**Goal:** a repo where any change can be measured.

Tasks:
- 0.1 `git init`, commit the current state untouched as the first commit.
- 0.2 Copy `LearningPYApp/*` to `baseline/`. Add `make baseline` that runs it against
  `data/v1` and asserts AUC 0.814, top-20 wins 0.571, revenue 0.795, and a byte-identical
  `weights.csv`.
- 0.3 Split `emva_score.py` into the `emva/` modules above with **no behaviour change**.
  `make baseline` must still pass when run through `emva/`.
- 0.4 Build `emva/eval/`: bootstrap CIs (1000 resamples), paired comparison, per-month
  AUC, decile calibration, status-quo reconstruction, and the standard report.
- 0.5 `requirements.txt` pinned (numpy, pandas, scikit-learn, anthropic). Python 3.11.

Acceptance: `make baseline` passes; `make report` prints the standard table for the
baseline with the status quo at AUC 0.639, top-20 wins 0.392, revenue 0.454.

---

## Phase 1: labels (critical)

**Goal:** stop teaching the model that slow, big, or neglected leads are losses.

Tasks:
- 1.1 Add `label_source` to every row: `won`, `crm_lost`, `stalled`, `ghosted`, `open`.
- 1.2 Fixed-horizon label: `won_within_H` with H = 120 days from `created_at`, same H for
  every row. Leads younger than H with no Won are **unlabelled**, not Lost.
- 1.3 Ghosted leads (stage still New at H) are excluded from training by default; a flag
  `--include-ghosted` keeps the old behaviour for comparison.
- 1.4 Stalled open deals are treated as censored (unlabelled), not Lost. Flag
  `--stalled-as-lost` keeps the old behaviour.
- 1.5 Evaluation uses only leads with `created_at + H <= AS_OF` (mature leads).
- 1.6 Report both label definitions side by side for one release.

Acceptance (evaluation-only, in `reports/phase1.md`):
- Learned `band=201-1000` and `band=1000+` weights move toward the planted 1.3 and 1.0
  (baseline: 0.873 and 0.703).
- Labelled win rate for band 201-1000 in training moves from 27.7% toward the 39.4% true
  rate on mature leads.
- AUC on mature leads is within the baseline CI or higher.
- The bottom-decile ghosted share (baseline 27%) is reported for both definitions.

---

## Phase 2: features and leakage (major)

**Goal:** missing means missing; no ground-truth constants; no perfectly collinear pairs.

Tasks:
- 2.1 Explicit `missing` level for `time_on_page`, `hesitation_90s`, `sessions_3plus`,
  `viewed_pricing`, `ip_country`, `ip_type`, `business_hours`. Remove the lead-ads special
  case in `business_hours`.
- 2.2 Replace `no_company` with `enrichment_missing`. Add `missing` levels to `band`,
  `spend`, `crm`, `hiring` instead of defaulting to the reference level.
- 2.3 Company-name enrichment: match typed `company_name` to `companies.csv` (normalised,
  suffix-stripped) when the email domain misses. Report how many of the 2,904 domain-miss
  leads get a match.
- 2.4 Remove the `exp(0.45**2/2)` constant from the deal model; estimate the residual
  variance from training residuals.
- 2.5 Replace the three-prefix copy-paste detector with a similarity check against a
  boilerplate list, and report its precision on v1 (it will be 100% on v1; the point is
  that it is no longer a template match).
- 2.6 Drop `c_budget`, `c_timeline`, `ip_type`, `edits_1_4` unless a bootstrap CI on the
  coefficient excludes zero. Report each.
- 2.7 Collinearity check in `eval/`: fail if any two design columns have |corr| > 0.95.

Acceptance:
- No two coefficients identical to three decimals (baseline: `email=free` and
  `no_company=yes` both −0.391).
- Meta lead-ads leads no longer sit in the reference bucket for every behavioural feature.
- AUC within the baseline CI [0.789, 0.836] on the Phase 1 test set.
- `grep -rn "0.45\|ground_truth" emva/` returns only `emva/eval/`.

---

## Phase 3: value layer and platform contract (major)

**Goal:** a value the ad platforms can learn from.

Tasks:
- 3.1 `value.py`: expected value = p × E[deal value] × margin, then a configurable
  transform: cap at a percentile (default p97), floor (default £25), optional log or sqrt
  compression, optional bucketing into N tiers.
- 3.2 Value-scale report: p1/p50/p90/p99/max, max/median ratio, top-1% share, and revenue
  captured in the top 20% under each transform. Baseline: max/median 116×, top-1% share 11%.
- 3.3 Two-stage design in code, not just a doc: `value_at_submit` (predicted) and
  `value_at_close` (actual deal value or 0). Output both columns with timestamps so an
  upload job can send the first and adjust with the second.
- 3.4 tROAS eligibility calculator: given leads per month and positive rate, output
  conversions per month per campaign and flag below-threshold cases.
- 3.5 Design document only (no platform calls): click-ID coverage (29.7% of paid leads
  have none), Google Enhanced Conversions for Leads via hashed email, Meta CAPI with
  `fbp` + hashed email, and what to do with the remaining unattributable share.

Acceptance:
- A transform that keeps ≥95% of the baseline's top-20% revenue capture while cutting
  max/median below 20×.
- `scores.csv` carries `value_at_submit`, `value_at_close`, `label_source`, `matured_at`.

---

## Phase 4: evaluation hardening (major, can run alongside Phase 3)

Tasks:
- 4.1 Rolling-origin evaluation: six split dates (2026-02-01 to 2026-07-01), report the
  spread. Baseline spread: 0.814 to 0.836.
- 4.2 Subsampling curve: N = 300, 500, 1000, 2000, 3000, full; 20 seeds; LR and
  `HistGradientBoostingClassifier` with monotone constraints on ordered features.
  Baseline: LR 0.807 ± 0.004 at N=2000.
- 4.3 Customer-sized test set simulation: for N=2000 total leads, hold out 20% and report
  the bootstrap CI width a customer would see.
- 4.4 Oracle-feature ceiling: fit LR on the ground-truth categorical columns (no noise
  term, no reply time) to get the reachable ceiling. Replace the 0.836 figure with this.
- 4.5 Calibration decay: fit on data up to month M, evaluate calibration on M+1..M+3.
- 4.6 Regularisation sweep as a standing check (baseline: flat, 0.8107 to 0.8138).

Acceptance: `reports/phase4.md` with all six sections; the report becomes part of
`make report` so later phases inherit it.

---

## Phase 5: generator v2 (critical, independent of Phases 1-4)

**Goal:** data the current regex and bot rule cannot match perfectly, so the model is no
longer the generator.

Tasks:
- 5.1 Recover `scripts/generate_data.py` or rewrite it from `ground_truth.md`. Confirm
  it reproduces v1 (seed 20260924) before changing anything.
- 5.2 Free text: paraphrase every template through Claude Haiku (one call per template,
  cached), inject typos, mix languages for DE/FR/NL leads, and add boilerplate that is
  not one of three fixed prefixes.
- 5.3 Enrichment dropout: 30% of business-email domains missing from `companies.csv`;
  company name typed with suffix/case variation so 2.3 has something to match.
- 5.4 Partial behavioural missingness for website leads (consent declined, 15%).
- 5.5 Two interaction effects (for example channel × band, seniority × form variant).
- 5.6 Ghosting correlated with the status-quo tier more strongly than with true p, to
  simulate sales neglect that follows the old score.
- 5.7 A planted context-only signal readable only from text semantics (nonprofit, job
  seeker, competitor, agency pitching), effect −1.0, invisible to the regex.
- 5.8 Time-on-page under 15s for 2% of genuine humans (autofill, returning visitors).

Acceptance:
- Regex agreement with generator category < 100% and reported.
- Bot rule precision < 100% and reported.
- Full Phase 0-4 pipeline runs on v2 with the standard report; AUC reported with CI.

---

## Phase 6: context agent v2 (depends on Phases 2 and 5)

**Goal:** a context feature that is orthogonal to the formula and provably adds value.

Tasks:
- 6.1 New contract in `context/contract.py`: the LLM returns named judgments, each a
  small enum, for example `is_real_business`, `persona` (buyer / vendor / student /
  job_seeker / competitor / unclear), `problem_specificity`, `urgency`,
  `brief_fit`. One free-text `reason`. No single 0-1 score.
- 6.2 Lead card contains only what the formula cannot use: free text, typed company
  name, job title string, email domain string, brief. No enrichment bands, no spend, no
  CRM, no hiring.
- 6.3 `context/features.py` turns each judgment into a bucketed feature with its own
  weight; the harness reports each weight with a CI.
- 6.4 Agent runtime: Anthropic Python SDK, structured output for the contract, bots and
  duplicates filtered before calling, parse errors separated from transport errors,
  `(brief_hash, prompt_version, model_id)` stamped into every output row.
- 6.5 Harness power curve: sweep oracle noise (sd 0.25, 0.5, 1.0, 2.0) and plot fraction
  of the oracle-feature gap recovered. This replaces the unreproducible "0.814 → 0.818".
- 6.6 Run real Haiku on 1,000 v2 leads containing the planted context-only signal.

Acceptance:
- On v2, the combined model recovers more than half of the planted context-only gap,
  with the `persona` weights individually significant.
- Correlation between the context features and the formula score is reported and
  below 0.5.
- Re-running with the same brief and model reproduces the scores exactly from cache;
  a brief edit produces a new `brief_hash` and a documented retrain.

---

## Phase 7: production readiness document (no code)

`reports/production.md` covering: retrain cadence and the 120-day label lag; drift
monitors (feature distribution, score distribution, calibration by month); what the
platform feedback loop does when the model is wrong, with mitigations (capped values,
holdout campaigns, an exploration share); click-ID coverage and the unattributable share;
the minimum lead volume at which EMVA should decline a customer.

---

## Dependency order and parallelism

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ─┐
                                   └► Phase 4 ─┼─► Phase 6 ──► Phase 7
Phase 0 ──► Phase 5 ─────────────────────────┘
```

Phases 3, 4 and 5 can run as three parallel agents once Phase 2 lands (Phase 5 can start
right after Phase 0). Phase 6 waits for 2 and 5. Phase 7 waits for everything.

## Claims policy until Phase 5 is done

Nothing from v1 is quoted externally. After Phase 5, numbers are quoted with CIs and the
phrase "on simulated data". Real-data numbers require a customer pilot (rolling-origin
backtest, LR vs monotone GBDT, CIs), which is outside this plan.
