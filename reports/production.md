# Phase 7: production readiness

Branch `phase7-production-doc` from `origin/main` cd77159. Plan Phase 7 (no code). Audience: a CTO deciding what
to build and sell, and a data scientist who has to run it.

**Read this first.** Every number below comes from a report in this repo and is **on simulated data** (generator
v1 or v2; no real customer data exists here). v2 is the headline dataset. v1 numbers are marked *(v1, internal)*:
the claims policy forbids quoting them externally. Phase 4 is not yet merged: its numbers are cited from
`origin/phase4-eval-hardening` (head bd60559, `reports/phase4.md`, ADR 0013 status Proposed); they have been
independently reproduced and are treated as final. Anything proposed here that is not measured is labelled
**proposal** or **assumption**; platform limits that come from memory are labelled **verify**.

Source keys: P0 to P6 = `reports/phase0.md` ... `reports/phase6.md` (P5.1 = `reports/phase5-1.md`), P4 =
`origin/phase4-eval-hardening:reports/phase4.md`, PC = `docs/platform_contract.md`, ADR NNNN = `docs/adr/NNNN-*.md`.

---

## 1. What is production-ready and what is not

**Ready to build a production service on (engineering, not evidence):**

| component | state | why it is ready |
|---|---|---|
| Pipeline (`python -m emva`) | load, clean, horizon labels, fixed 39-column design, L2 LR, ridge deal value, value transform, `scores.csv` + `weights.csv` | Frozen-baseline regression gate (`make baseline`, ADR 0001); ground truth unreachable outside `emva/eval/` (grep rule, test-enforced); fixed design schema so a one-row frame scores identically to a batch and an undeclared level is refused loudly (ADR 0009); transform parameters fitted once and fixed (ADR 0012, PC §1) |
| Labels | `won_within_h`, H = 120, ghosted excluded, stalled censored, mature leads only (ADR 0005) | The right-censoring defect found in the review is fixed (P1). Ranking barely moved (AUC within ±0.002), calibration and weights did (P1) |
| Evaluation harness | standard report, rolling-origin headline, subsampling, customer-sized holdout, calibration decay, C sweep (P4, ADR 0013) | Every later claim can be measured the same way on a customer's data; this is what a pilot runs (section 10) |
| Value contract | `value_at_submit` (cap p97 + log + floor £25), `value_at_close` with `value_at_close_status` (known / unknown_amount / pending), two-stage upload, idempotency ledger (ADR 0012, PC) | max/median cut from 67.6× to 5.0× over all scored v2 leads with top-20% revenue capture unchanged (P3/P4, on simulated data) |

**POC-only (do not sell on it):**

1. **The data is synthetic.** Both generators are additive logistic models with planted effects (CLAUDE.md, ADR 0003); a
   correctly specified LR matches them by construction. Headline, on simulated data: rolling-origin AUC on v2
   **0.765 [0.739, 0.790]**, four sufficient splits, range 0.743 to 0.776 (P4 §4.1, ADR 0013). Nothing here predicts
   what a real customer will see.
2. **The context agent's measurable value is tiny.** The planted persona on v2 is worth +0.006 [+0.001, +0.009] AUC on
   the legacy test set and +0.002 [−0.001, +0.004] rolling even when read perfectly (P4 §4.4). Haiku reads it well (87%
   of persona leads judged non-buyer; 8.3% false positives) and passes the coefficient criterion (−0.606 [−1.067,
   −0.179] vs oracle −0.612 [−1.021, −0.209]) only one feature at a time, with a pooling rule chosen after seeing the
   results (P6, ADR 0016). `value_at_submit` does not use the context model at all today (P3 deviation 5, P6 merge note).
3. **No platform call has ever been made.** The upload design (PC) is untested against Google Ads or Meta; its limits
   are from memory and marked verify.
4. **Test windows are thin.** The frozen mature test set spans 26 days (441 leads on v2, CI ±0.05); only four
   rolling splits are usable under H = 120 (ADR 0006, ADR 0013).

---

## 2. Retrain cadence and the 120-day label lag

### When a lead becomes labelled

| outcome | known at | source |
|---|---|---|
| Won within 120 days | `won_at` (can be before maturity; `won_within_h` = 1 at once) | CONTEXT glossary, `emva/labels.py` |
| Not won within 120 days (incl. a later CRM loss, a late win, still open) | `matured_at` = `created_at` + 120 d | ADR 0005 |
| Ghosted at H, stalled | never used for training (censored by default) | P1, ADR 0005 |
| Young CRM loss | unlabelled until mature (open question, section 11) | P1 open question 3 |

Positives arrive early and negatives only at day 120, which is why only **mature** leads train or evaluate: training on
"labels so far" would over-represent wins. Time to win *(v1, internal)*: median 35 days, p90 114, and 15.2% of wins
(v2 14.4%) close after day 90 (PC §2). About **68% of mature v2 leads get a horizon label** (4,230 of 6,201; the rest
are ghosted or stalled; derived from P6 sample section and the P4 standard report's bottom-decile table). That yield is
a property of the simulated sales team; a real customer's yield must be measured at onboarding.

Consequence: a model trained today learns only from leads created at least 120 days ago. Every scored lead is at
least four months newer than the newest training lead. Phase 4's calibration-decay experiment uses exactly this rule
(training mature at the split), so its "M+1" is already four-plus months past the newest training lead.

### Evidence for the cadence (P4 §4.5, v2, headline rule, on simulated data)

| fit through (train rows) | slope M+1 / M+2 / M+3 | intercept M+1 / M+2 / M+3 |
|---|---|---|
| 2026-01 (182) | 0.89 / 0.82 / 0.70 | 0.00 / +0.13 / −0.05 |
| 2026-02 (647) | 0.89 / 0.81 / 0.78 | +0.04 / −0.08 / −0.22 |
| 2026-03 (1,160) | 0.87 / 0.82 / n/a | +0.01 / −0.12 / n/a |
| 2026-04 (1,689) | 0.91 / n/a / n/a | −0.11 / n/a / n/a |

Slope < 1 means over-confident. On v2 the slope falls with distance from the fit month in every fit that has three
evaluation months (0.89 → 0.70 for the 2026-01 fit), and the overall range is 0.70 to 0.91 with intercepts −0.22 to
+0.13. Caveats: the evaluation months hold only 441 to 550 leads each; the same pattern is not monotone on v1 (slopes
0.73 to 1.05, *v1, internal*); and the generator plants no time trend that we have identified, so this is small-sample
over-confidence plus noise, not real drift. Real data will add real drift on top.

**Recommendation (proposal):**

- **Retrain monthly**, when the next monthly cohort matures. An LR fit is cheap (the whole Phase 4 run with GBDTs took
  about 10 minutes). Monthly keeps every scored lead in the "M+1" regime, the best-calibrated row above.
- **Retrain out of cycle** when a calibration monitor fires (section 3).
- **Change control on the value level.** The transform's rescale ties the level of `value_at_submit` to the training
  leads, so a retrain moves every value sent (ADR 0012). Before switching models, compare the new and old models'
  `value_at_submit` on the same last 30 days of leads; hold the switch and review if the median moves by more than
  25% (**proposal**). Smart bidding reacts to value changes (a learning period after significant changes, **verify**),
  so avoid retraining more often than monthly.

### Minimum labelled volume per retrain (P4 §4.2, v2, 20 seeds, test = frozen mature set (b), on simulated data)

| N labelled training rows | LR AUC mean ± sd | loss vs full pool (0.794) |
|---|---|---|
| 300 | 0.740 ± 0.023 | −0.054 |
| 500 | 0.768 ± 0.015 | −0.026 |
| 1,000 | 0.776 ± 0.011 | −0.018 |
| 2,000 | 0.789 ± 0.005 | −0.005 |
| 3,000 | 0.792 ± 0.003 | −0.002 |
| 3,789 (full) | 0.794 | 0 |

The pool's labelled win rate is 19.4% (734 of 3,789), so N = 1,000 holds about 190 wins and N = 2,000 about 390.
Calibration is the binding constraint at small N: the fits on 182 and 647 rows are the over-confident ones above, and
only the 1,689-row fit reaches slope 0.91. The regularisation sweep is not flat on the rolling headline (v2 0.750 to
0.765 for C in [0.1, 10]) because small training sets want stronger regularisation (P4 §4.6).

**Minimums (proposal, from the table):** do not ship a retrain on fewer than **1,000 labelled rows and ~200 wins**;
treat **2,000 rows / ~400 wins** as the level where training variance stops mattering (sd 0.005). Below 2,000 rows,
rerun the C sweep at every retrain (`emva/eval/regularisation.py`) instead of assuming C = 0.5. The "~200 wins"
translation to other win rates assumes wins, not rows, carry the information; Phase 4 did not vary the win rate.

### Legacy vs horizon labels in a customer's first 120 days

- **Training uses horizon labels only**, built from the customer's CRM history: every lead created more than 120 days
  before onboarding is labelled from its stage history. A customer who cannot supply at least the minimum above in
  mature history has no model on day one (section 6).
- **Legacy labels (read at the snapshot) are never a training label.** They are the right-censored definition Phase 1
  removed. They look well calibrated on old months (v2 legacy slopes 0.90 to 1.08, apart from 1.91 on the 85-lead 2026-09 month) but their youngest months are
  biased toward fast decisions: intercepts −0.38 / −0.39 for 2026-08 and −0.54 for 2026-09 (85 leads) on v2 (P4 §4.5).
- **They are a leading indicator for monitoring.** Ranking is nearly label-independent (P1: AUC within ±0.002 across
  definitions; v2 legacy one-month test AUC 0.794 to 0.820 across all six splits, P4 §4.1), so for the leads scored in
  the first 120 days, AUC on early outcomes (wins so far, CRM losses so far) is a usable early check of discrimination.
  Calibration read from them is not.
- **In the first 120 days after go-live** no scored lead has a mature label, so the calibration-by-month monitor has
  nothing to read. Run it on the backtest cohorts, watch AUC on early outcomes, and keep the holdout campaign
  (section 4) running from day one.

---

## 3. Drift monitors

All thresholds are **proposals**. None is calibrated on real data; each is placed outside the range Phase 4 observed
on simulated data where nothing drifts, so it should not fire on noise alone. Reset them from the customer's own
backtest in the pilot.

| monitor | metric and cadence | baseline to compare with | threshold | action |
|---|---|---|---|---|
| **Unknown level** | leads refused by `fixed_design` (undeclared value raises, ADR 0009), per run | 0 | any | page; the lead is not scored. Add the level to `V2_LEVELS` (a contract change) and retrain |
| **Feature distribution** | per feature, share of each fixed level (39 design columns + reference levels, ADR 0009), weekly; population stability index (PSI) against the training window | training window | PSI > 0.10 warn; > 0.25 alert (conventional cut-offs, not derived here) | warn: note in the monthly review. Alert: check the source (form change, tag, enrichment vendor) before the next retrain; if it is a population change, retrain early; if it is a pipeline fault, see missingness row |
| **Missingness: `session_missing`** | rate per channel, daily | v2: 23.3% of scored leads (872 lead-ads + 1,278 consent-declined of 9,209, P2); consent-declined 14.8% of website rows (P5) | website-channel rate > 2× the training rate, or +10 pp absolute, for 2 days | a broken site tag makes every website lead look session-less, which carries a learned weight (+0.535 on v2, P2). **Hold submit-stage uploads for website leads** until fixed; do not retrain on the outage window |
| **Missingness: `enrichment_missing`** | rate, daily | v2 28.4% (2,615 of 9,209), v1 21.0% *(internal)* (P2) | +10 pp absolute for 2 days | enrichment vendor or domain-join fault. `band=missing` carries +0.816 on v2 (P2): hold uploads for affected leads; do not retrain on the window |
| **Bot filter** | share flagged (headless UA or < 15 s), weekly | v1 bot share 3.89% (P5.1); v2 rule precision 72.4%, recall 90.2% (P5) | ±50% relative change | a jump usually means a traffic source changed; review before the next retrain (flagged leads never reach the platform as conversions) |
| **Score distribution: p** | mean p and p deciles of newly scored leads, weekly | training leads' mean p | mean p moves > 20% relative without a retrain | compare with feature PSI to find the driver; if none, investigate the scoring build |
| **Score distribution: `value_at_submit`** (ADR 0012 transform) | median, p99, max/median, share at the cap, share at the £25 floor, weekly | max/median 4.7× to 5.5× on test sets (P3); cap at p97 of training values, so ~3% of leads at the cap by construction | max/median > 10×; share at cap > 6%; median moves > 25% week on week without a retrain | the platform is being sent a different value scale than it learned on: freeze the model version, find the driver, retrain or roll back |
| **Calibration by month** | for each monthly cohort as it matures: slope, intercept (calibration-in-the-large), mean p vs observed, AUC | P4 v2 headline rule: slopes 0.70 to 0.91, intercepts −0.22 to +0.13; per-split AUC 0.743 to 0.776 | slope < 0.80 or \|intercept\| > 0.20: **retrain now**. Slope < 0.70 or \|intercept\| > 0.30, or AUC below the pilot backtest's lower CI bound for two consecutive cohorts: **stop sending new values** (fall back to the previous model or to the status-quo value) and investigate | these read the one number the platform acts on. A 441-lead cohort gives a noisy slope; require the breach on two consecutive cohorts before the stop action unless both slope and intercept breach together |
| **Sales-neglect feedback** | share of ghosted leads in EMVA's bottom decile of p, per matured cohort | v2: 24.8% candidate vs 20.3% overall on all mature leads (P4 standard report) | bottom-decile ghosted share rising > 5 pp over two cohorts | sales is following the score and the censoring rule hides EMVA's own errors (section 4). Raise with the customer; enlarge the random "work regardless of score" share |
| **Context agent: errors** (only if the context model is live) | parse-error and transport-error rates per run (`runs.jsonl`) | P6 real run: 0 parse, 0 transport errors in 1,000 requests | parse > 1% of calls or transport (after retries) > 2% in a day | parse errors mean the model or contract changed: stop and check the stamp. Leads without an ok judgment have no context (ADR 0014); in production they must be scored by the formula alone (proposal) |
| **Context agent: judgment drift** | share of each enum level per judgment, weekly | the customer's first production run (the P6 sample is persona-enriched to 30% and is not a baseline) | any level's share moves > 10 pp absolute | brief, traffic or model behaviour changed; re-check the R12 coefficient criterion on the next retrain |
| **Context agent: stamp** | `(brief_hash, prompt_version, model_id)` on the judgments file vs the stamp the context weights were fitted on; `prompt_fingerprint` in `runs.jsonl` and in the cache key | one stamp per file (enforced, ADR 0014) | any mismatch | block scoring with context; refit the context weights on new judgments (weights never carry over between stamps, ADR 0014). Note: `prompt_fingerprint` is logged per run and keyed in the cache, not stamped on each row |

---

## 4. Platform feedback loop: failure modes and mitigations

**How it fails.** The platform bids more for whatever EMVA values highly and buys more of it. If EMVA over-values a
segment, the platform shifts spend toward it, and nothing corrects this for at least 120 days, because the losses
only become labels at maturity. Three things make it worse:

1. **Spurious effects look real.** On v1, where nothing is planted, the LinkedIn × 51+ interaction comes out
   +0.56 [+0.18, +0.98] *(v1, internal)*, mostly sampling chance amplified by the 120-day truncation (P4, open
   finding). A model that shipped such a term would tell the platform to buy LinkedIn enterprise traffic.
2. **The training population follows the model.** Next year's training leads are the ones the platform bought on
   EMVA's values, so segments EMVA undervalued become rare and their weights stop being re-estimated.
3. **Sales neglect hides the errors.** On v2 ghosting tracks the old tier more than true quality (correlation 0.240
   with tier vs −0.065 with true p, P5). If sales work EMVA's high scores first, EMVA's low-scored leads get ghosted,
   ghosted leads are excluded from training (ADR 0005), and the model never sees that it was wrong about them.

**Mitigations:**

| mitigation | what it does | evidence or status |
|---|---|---|
| **Capped, compressed values** (ADR 0012) | cap at p97 + log: max/median about 5×, a lead at the cap is sent about 0.5× its expected value (the top v1 lead 0.15×), below-median leads 2.5 to 4×. Bounds how hard the platform can chase any single segment | max/median 4.7× to 5.5× on every test set, capture unchanged (P3, on simulated data). Cost: values are not proportional to expected revenue |
| **Holdout campaigns** | keep part of the budget on the customer's status-quo value (or plain lead optimisation), split by campaign or geography, for the life of the contract. Compare **recorded** revenue per £ between arms | proposal: 10 to 20% of spend. The size needed to detect a revenue difference depends on the customer's deal-value variance, which is not known here |
| **Exploration share** | (a) a random 5% of leads sent a flat value (the training median of `value_at_submit`), so the platform keeps buying some leads EMVA undervalues and EMVA keeps getting labels outside its own preference; (b) a random 5% of leads worked by sales regardless of score, so ghosting can be measured against score | proposal, not measured. (b) is also what makes the sales-neglect monitor in section 3 interpretable |
| **Two-stage submit / close** (PC §2, ADR 0012 R10) | the close stage restates the submit value to the recorded deal value, or retracts it at maturity for a non-win (Google); on Meta a second event at close | `value_at_close_status`: send `known`, **hold** `unknown_amount` (112 of 1,199 v1 wins, 96 of 1,128 v2; never send 0 for these, which would retract a real win), wait on `pending`. Google adjustments are accepted only within a window (at most 90 days historically, **verify**), so the 14.4% (v2) of wins closing after day 90 fall back to a separate "closed-won" reporting action. Meta rejects events older than 7 days (**verify**) |
| **ROAS on recorded revenue only** | `value_at_submit` is a bidding signal: a median lead is sent 2.45× (v2) / 2.77× *(v1)* its expected value, and total sent value / recorded revenue is 1.13 to 1.43 on v2 under log (ADR 0012, P3). Platform-reported ROAS on submit values is therefore not revenue | report ROAS from the close stage and the CRM, never from the platform's conversion value column |
| **Kill switch** | the calibration and value-scale monitors (section 3) can stop new values and revert to the previous model or the status-quo value | proposal; the status-quo value is already reconstructed in `emva/eval/status_quo.py` |
| **Pre-registered model changes** | new features or interaction terms enter only through a pilot backtest with paired CIs, never because one coefficient is "significant" | the LinkedIn × 51+ finding above; ADR 0016 R13 (pooling chosen after results) is the counter-example to avoid |

---

## 5. Click-ID coverage and the unattributable share

From PC §3 (report section "Click-ID coverage"), on simulated data:

| scored paid leads | v1 *(internal)* | v2 |
|---|---|---|
| landing-page paid leads (Google, Meta website, LinkedIn, ChatGPT) | 7,454 | 7,435 |
| ... with no click id (`gclid`, `gbraid`, `wbraid`, `fbclid`, `li_fat_id`, `oppref`) | **28.8%** | **29.2%** |
| ... no click id but a Meta `fbp` cookie | 22.9% | 20.1% |
| Meta lead-form leads (matched by lead id) | 932 | 872 |
| all paid leads with no id at all | 25.6% | 26.2% |
| email present / phone present (landing-page paid) | 100% / 24.6% | 100% / 25.0% |

The plan's 29.7% was not reproduced (29.0% on raw v1 rows, 28.8% on scored leads; PC §3).

**What to do** (PC §3, §4):

- **Google: Enhanced Conversions for Leads.** Upload with the click id when present and always with SHA-256 of the
  normalised email (and E.164 phone when given), so a lead without a click id can still match the signed-in user.
  Needs the account setting, customer-data terms and the site tag capturing the hashed email (**verify**).
- **Meta: Conversions API** with `event_id` = `lead_id`, `fbc` from `fbclid`, `fbp`, hashed email and phone, IP and
  user agent; lead-form leads through the CRM integration.
- **Consent.** Only leads with consent may be uploaded with hashed identifiers; EEA/UK uploads need the platforms'
  consent signals (**verify**).
- **The rest.** Do not reassign unmatched value to matched leads (it biases the bidder toward whoever matches best);
  track the platforms' match rate (Google's Enhanced Conversions match rate, Meta's Event Match Quality) as a KPI;
  report EMVA's predicted value for unmatched leads by channel and campaign (`utm_*`) so the business still sees it;
  reduce the share at source (hidden click-id fields on every landing page, `fbp` on the form post, phone where the
  form allows).
- What the match rate will be after hashed email is **unknown**: it depends on sign-in rates and email reuse, which
  synthetic data cannot tell us. The pilot must measure it.

---

## 6. Minimum lead volume at which EMVA should decline a customer

Two constraints: enough labelled history to fit and validate a per-customer model (P4), and enough conversions for
the platforms to bid on (P3, `emva/troas.py`).

**Model side (P4, on simulated data):**
- N = 1,000 labelled rows (~190 wins) gives LR 0.776 ± 0.011; N = 2,000 (~390 wins) 0.789 ± 0.005 (section 2).
- A customer with 2,000 mature labelled leads validating on the latest 20% (400 leads, ~70 wins) sees a 95% CI about
  **0.11 wide (±0.055)**, and the AUC itself varies ±0.02 between draws (P4 §4.3). That is enough to separate EMVA from
  a status-quo-like rule (paired gap −0.122 [−0.190, −0.051] on v2's mature test set, P4 standard report) but not to
  separate two models less than ~0.05 apart without a paired comparison.
- Labelled rows per year = monthly leads × 12 × label yield (0.68 on v2, **assumption** for a real customer).

**Platform side (`emva/troas.py`, thresholds verify):**
- Google tROAS: 30 conversions per campaign per 30 days, so the submit stage (every lead a conversion) needs about
  **30.4 leads per campaign per month**; bidding on closed-won alone needs 30.4 / win rate.
- Meta: ~50 optimisation events per ad set per week, so the submit stage needs about **217 leads per month** in one
  ad set (or pooled). On v2 each Meta campaign has about 17 leads a week; pooled, 69 (P3).
- The platform contract therefore bids on `value_at_submit` (PC §5). A 2,000-lead-a-year customer can use Google
  tROAS in one campaign on submit values and cannot leave Meta's learning phase per ad set even on leads (P3).

**Decision rule (proposal).** With W = wins per year = monthly leads × 12 × win rate (win rate = won within 120 days
over all mature leads, as `emva.troas` defines it; v2 0.131) and R = labelled rows per year:
**accept** if R ≥ 2,000 and W ≥ 400; **pilot with conditions** if R ≥ 1,000 and W ≥ 200; **decline** otherwise.
Conditions for the middle band: one consolidated Google campaign (or a portfolio strategy), no per-ad-set Meta value
bidding, C sweep at every retrain, and a pilot that states its CI width up front.

| leads / month | win rate | labelled rows / yr (R) | wins / yr (W) | max Google campaigns on submit value | Google close-stage per 30 d (1 campaign) | Meta submit, pooled, per week | verdict |
|---|---|---|---|---|---|---|---|
| 100 | any 5-20% | 819 | 60-240 | 3 | 4.9-19.7 (below 30) | 23 (below 50) | **decline** |
| 150 | 5% / 10% | 1,228 | 90 / 180 | 4 | 7.4 / 14.8 | 34.5 | **decline** |
| 150 | 13% / 20% | 1,228 | 234 / 360 | 4 | 19.2 / 29.6 | 34.5 | pilot with conditions |
| 250 | 5% | 2,046 | 150 | 8 | 12.3 | 57.5 | **decline** (too few wins) |
| 250 | 10% / 13% | 2,046 | 300 / 390 | 8 | 24.6 / 32.0 | 57.5 | pilot with conditions |
| 250 | 20% | 2,046 | 600 | 8 | 49.3 | 57.5 | accept |
| 500 | 5% | 4,093 | 300 | 16 | 24.6 | 115 | pilot with conditions |
| 500 | ≥ 10% | 4,093 | ≥ 600 | 16 | ≥ 49.3 | 115 | accept |
| 1,000+ | ≥ 5% | ≥ 8,186 | ≥ 600 | ≥ 32 | ≥ 49.3 | ≥ 230 | accept (Meta per-ad-set bidding possible at ~217+ leads per ad set per month) |

Computed with the formulas of `emva/troas.py` (30.4375-day months) and the v2 label yield; the wins and rows
thresholds come from the P4 subsampling table. The thresholds are judgement calls on simulated evidence and should be
revisited after the first pilot. The table also bounds how long a new customer waits: a customer with no usable CRM
history needs at least four months (maturity) plus the months of leads that reach the minimum.

---

## 7. Model choice, defended

**Choice: L2-regularised logistic regression, one per customer, on the fixed 39-column schema (ADR 0009), with a
ridge deal-value model.**

1. **It wins on the evidence we have.** LR beats the monotone GBDT at every N on both datasets (P4 §4.2, on simulated
   data). v2: N = 300 0.740 vs 0.735; N = 2,000 **0.789 ± 0.005 vs 0.756 ± 0.012**; full pool 0.794 vs 0.783. v1
   *(internal)*: full pool 0.778 vs 0.738. The unconstrained GBDT is no better, so the (partly wrong) monotone
   constraints are not what holds it back.
2. **It is a scorecard.** `weights.csv` is log-odds and points (+20 points = odds of closing doubled). A customer's
   sales lead can read why a lead is valued, and a reviewer can spot a spurious weight (the LinkedIn × 51+ lesson).
3. **Calibration is the product.** The platform receives `p × E[deal value]`, so p must be calibrated, not just
   ranked. LR's probabilities are directly interpretable and the calibration monitors in section 3 read them.
4. **It fits small N.** The customer sizes in section 6 are 1,000 to 4,000 labelled rows a year, where the GBDT's
   gap to LR is widest (v2: 0.033 at N = 2,000, 0.011 at the full 3,789).

**What would change the decision:** a customer pilot where a monotone GBDT (tuned, then Platt- or isotonic-calibrated)
beats LR on the rolling-origin headline by more than the paired CI, with calibration at least as good; or pooled data
across customers reaching sizes where the GBDT's gap closes (it narrows with N on v2); or real interactions large enough
that LR with explicit terms is not enough (on v2 the planted interactions add +0.002 [−0.000, +0.005] rolling, P4).

**Honest caveats:**
- The generators are additive logistic models (CLAUDE.md, ADR 0003). LR is correctly specified for them by
  construction; the comparison is tilted toward LR, and the GBDT was run at sklearn defaults, untuned (P4, ground rule 5).
- On v2 the pipeline sits 0.021 [0.011, 0.031] below the formula-plus-interactions oracle on the legacy test set and
  0.012 [−0.004, +0.027] rolling (P4 §4.4): some reachable signal is missed, mostly from enrichment dropout and
  paraphrased text.
- **Nothing is proven until a customer pilot** (section 10).

---

## 8. Multi-customer roadmap (ADR 0017, Proposed)

An external data scientist argued for a single pooled, industry-agnostic model with industry as a feature, or for
LLM-only profiling, instead of per-customer models. Position (full text in `docs/adr/0017-multi-customer-roadmap.md`):

- **Day one: per-customer LR on the fixed feature schema.** Each customer's funnel, sales team and deal sizes differ,
  there is no second customer's data to pool, and a per-customer model can be validated on that customer's own
  rolling-origin backtest (section 10).
- **The fixed schema is the precondition for pooling.** Because every customer's design has the same declared columns
  (ADR 0009), their coefficients are comparable and can later share a prior. Keeping it fixed now is what makes the
  pooled future possible.
- **The multi-customer evolution is a hierarchical model**: a shared prior over coefficients with a per-customer
  adjustment (partial pooling), which helps exactly the small customers section 6 would otherwise decline. It needs
  several customers' labelled outcomes (each at least 120 days old) and data-sharing terms that allow training on
  pooled outcomes. Industry enters as a level of the hierarchy or a feature once there are several customers per
  industry; with one customer it is a constant.
- **The LLM stays a measured feature, not the scorer.** Platforms need a calibrated expected value per lead;
  the context agent returns enum judgments, never a score (ADR 0014), and is admitted per feature by the coefficient
  criterion (ADR 0016). On v2 its planted signal is worth about +0.005 AUC at most (P4 §4.4). An LLM-only profile has
  no calibration and no way to pass the pilot protocol.
- **Ingestion adapters are per CRM platform, not per customer**: one adapter per CRM the customer runs (HubSpot,
  Salesforce, ...) mapping its export to the lead / stage-history / enrichment inputs `emva.io.load` expects and its
  stages to `OPEN_STAGES` / Won / Lost, so a new customer on a known CRM is configuration, not code.

---

## 9. Hosting shape (to inform a later deployment)

- **One service.** A small web app for scoring and internal review; the scorer loads the current `weights.csv`, the
  deal-value model and the fitted transform parameters. The fixed schema makes single-lead scoring identical to the
  batch run (ADR 0009, PC §1).
- **Training as a CLI subprocess**: `python -m emva --data DIR --out DIR` (plus the monthly report), triggered by a
  scheduler, never inside a request. A new model is promoted only after the change-control check in section 2.
- **One persistent volume** for customer datasets (CRM exports), `runs/`, model artefacts per version, the context
  agent's cache, and the upload ledger keyed by (`lead_id`, stage) (PC §2). Paths contain spaces locally; quote them.
- **Secrets as environment variables**: `ANTHROPIC_API_KEY` and `ANTHROPIC_WORKSPACE_ID` (the key is not
  workspace-scoped; every request sends the workspace header, ADR 0010), plus platform credentials when an upload job
  exists. Never in the image, the volume or logs.
- **A shared-password gate** in front of any internal UI (scores, weights and CRM data are customer-confidential).
- **No ground-truth file is ever deployed.** `ground_truth_labels.csv`, `ground_truth.md` and
  `ground_truth_companies.csv` are evaluation-only (ground rule 2); the deploy artefact must exclude `data/*/ground_truth*`
  and `emva/eval/` readers of them. Synthetic `data/` should not ship at all.

---

## 10. Customer pilot protocol (before any external claim)

A pilot runs on the customer's historical CRM and lead data, offline, before any value reaches a platform:

1. **Data audit**: lead count, label yield (share of mature leads with a horizon label), win rate, time-to-win
   distribution, click-id coverage, `session_missing` / `enrichment_missing` rates, unknown levels. Apply section 6.
2. **Status quo reconstructed** from the customer's own rules, as `emva/eval/status_quo.py` does for the synthetic
   advertiser.
3. **Rolling-origin backtest** under ADR 0013's rule (training mature at each split, one-month mature test windows,
   splits with < 100 leads or < 10 of a class listed but left out), with the mean AUC and its bootstrap CI (1,000
   resamples) and the per-split range.
4. **LR vs monotone GBDT**, same rows, paired differences on shared resamples (P4 §4.2 machinery), plus the subsampling
   curve on the customer's pool.
5. **Calibration by month** (slope, intercept, deciles) and the C sweep.
6. **Value layer**: value scale before and after the transform, top-20% recorded-revenue capture vs the status quo,
   total value / recorded revenue.
7. **Context agent** (optional): pre-register the pooling and the brief before the run; judge each context feature by
   the coefficient criterion (ADR 0016); report the AUC gain with its paired CI as secondary.
8. **Live phase**: holdout campaign and exploration share from day one (section 4); report ROAS on recorded revenue.

**Claims wording.** Until a pilot is complete: "on simulated data, rolling-origin AUC 0.765 [0.739, 0.790]" (v2 only;
nothing from v1). After a pilot: "on <customer>'s historical data, rolling-origin AUC X [lo, hi] over k monthly
splits, against Y [lo, hi] for their existing rules (paired difference D [lo, hi])", never a point estimate without
its CI, never extrapolated to other customers.

---

## 11. Open risks and unknowns

| # | risk or unknown | source | consequence |
|---|---|---|---|
| 1 | Synthetic data only; both generators additive logistic; seed alone moves AUC ±0.015 | CLAUDE.md, ADR 0003, P5.1 | every number here is an upper bound on how well LR fits, not a forecast |
| 2 | Spurious significant interaction: LinkedIn × 51+ +0.56 [+0.18, +0.98] on v1 where nothing is planted; on v2 the planted +0.8 is not detected (+0.27 [−0.11, +0.67]) | P4 open finding, §interactions | one significant coefficient is not evidence; model changes need paired held-out gains |
| 3 | Boilerplate detection: Jaccard recall 0.062 on v2 (0.026 on the P6 sample); the LLM reaches 0.88 recall at 0.583 precision | P2, ADR 0011, P6 | copy-paste leads are mostly unflagged without the context agent |
| 4 | Young `crm_lost` leads are NaN rather than 0 (1,391 on v1, 1,244 on v2); not ruled | P1 open question 3, CONTEXT, P3 deviation 4 | affects `scores.csv` and `value_at_close` (pending instead of a retraction) |
| 5 | Platform limits from memory: Google 30 conversions / 30 days, Meta ~50 / week, Google adjustment window (≤ 90 days), Meta 7-day `event_time`, email normalisation, consent signals | `emva/troas.py`, PC, P3 deviation 8 | section 6's table and the upload design must be re-checked against current docs before building |
| 6 | Mature test window of 26 days (458 / 441 leads, ±0.05); only four rolling splits, the last one partial | ADR 0006, ADR 0013 | small evaluation sets on the POC; a real customer's history will be longer or shorter |
| 7 | Calibration over-confident at small N (slope 0.70 to 0.91) | P4 §4.5 | small customers get over-confident values; minimums in section 2 |
| 8 | Context agent: planted gap +0.002 to +0.006 AUC; pooling rule chosen post hoc; persona column not significant in the full model (−0.383 [−0.887, 0.156]); 8.3% non-buyer false positives; vendor/agency confusion (38 of 67) | P4 §4.4, P6, ADR 0016 | not a selling point until a pilot passes R12 pre-registered |
| 9 | `value_at_submit` ignores the context model in `--context` mode | P3 deviation 5, P6 merge note | if context goes live, the value hook must be built and tested |
| 10 | Unexplained v2 weights: `session_missing=yes` +0.535 and `band=missing` +0.816 | P2 deviation 3 | missingness indicators carry large learned effects; outages move values (section 3) |
| 11 | Bot rule precision 72.4% on v2 (137 fast humans dropped); 39 consent-declined bots missed | P5 | real humans filtered out; some bots scored |
| 12 | Ghosting follows the old score (0.240 with tier vs −0.065 with true p on v2); censoring hides the model's errors on neglected leads | P5, ADR 0005 | feedback loop in section 4; needs the "work regardless of score" share |
| 13 | Value level ties to training (rescale); a retrain moves every value sent | ADR 0012 | change control in section 2 |
| 14 | Deal-value residual sd is dataset-specific (0.5431 v2 vs the old fixed 0.45) | P2, ADR 0011 | estimated per customer; do not reuse constants |
| 15 | Click-id match rate after hashed email and `fbp` is unknown | PC §4 | the unattributable share after enhancement must be measured in the pilot |
| 16 | GBDT compared at defaults, untuned | P4 deviations | the LR choice must be re-tested in each pilot |
| 17 | Rolling splits read stalled censoring from CRM state at AS_OF (no stage history) | ADR 0013 | a small look-ahead that can drop, never relabel, a training row; a production backtest should use stage history |
| 18 | Context agent cost: about $1 per 1,000 leads, estimated (token usage not logged) | P6 | unverified; log usage in production |
| 19 | Phase 4 and ADR 0013 not yet merged (Proposed) | P4 | this document's headline and section 2/6 evidence depend on it |
| 20 | Plan figures that did not reproduce: 29.7% no-click-id (28.8-29.0%), 116× / 11% value scale (121.7× / 11.5%) | PC §3, P0 | minor; quote the reproduced figures |

---

## Numbers not taken directly from a report

- **Label yield 0.68 (v2)** = 4,230 horizon-labelled of 6,201 mature scored leads; both counts are in P6 ("4,230
  cleaned v2 leads with a horizon label") and the P4 v2 standard report ("all mature leads, every label_source", 6,201).
  The ratio is mine.
- **`session_missing` 23.3% (v2)** = (872 + 1,278) / 9,209 from P2; the ratio is mine.
- **Wins at N = 1,000 / 2,000 (~190 / ~390)** = N × the pool's 734 / 3,789 labelled win rate (P4); arithmetic, not
  measured.
- **Section 6 table** = `emva/troas.py` formulas and the v2 label yield applied to hypothetical volumes; not a
  `python -m emva.troas` run.
- **All thresholds in sections 2, 3, 4 and 6** are proposals; none is measured.
- Whether generator v2 plants any time trend: not verified here (it would mean reading the generator's ground truth,
  which is evaluation-only); section 2 says "that we have identified".
