# Phase 5.2-5.8: generator v2

Branch `phase5-generator-v2`, rebased onto `origin/main` at 92168cf (Phase 1 labels). Plan items 5.2 to 5.8.
All numbers are on simulated data.

## What was built

| File | Purpose |
|---|---|
| `scripts/generate_data_v1.py` | The v2 options are implemented on its `Config` (10 fields tagged 5.2-5.8, all off by default). Each option draws from its own RNG stream (`rng_for(cfg, "v2_...")`), so switching one on never shifts a v1 stage. With every option off, the output for seed 20260924 is byte-identical to before: checked at n = 10,000 (all seven files, `diff -r`) and pinned by sha256 at n = 600 in the tests. |
| `scripts/generate_data_v2.py` | Entry point: the v1 defaults plus `V2_SETTINGS`, writes `data/v2/`. `--no-paraphrase` drops only 5.2 (i). |
| `scripts/paraphrase_templates.py` | Claude Haiku (`claude-haiku-4-5-20251001`) paraphrases: one call per template, temperature 0 (sent through `extra_body`, since anthropic 1.x removed the typed parameter), 5 paraphrases per call as structured JSON output, validated (count, distinct, same `{placeholder}` set). The cache `scripts/paraphrase_cache.json` is keyed by sha256 of (template, model id, prompt version). `--populate`, `--check`, `--probe`. Reads `ANTHROPIC_API_KEY` and `ANTHROPIC_WORKSPACE_ID` (sent as `anthropic-workspace-id`) from the environment, else from the first `.env` found walking up from `scripts/`. Neither value is printed or committed. |
| `scripts/v2_text.py` | Hand-written phrase banks: persona sentences (6-7 per persona), 15 new boilerplate snippets plus openers, lead-ins and tails, DE/FR/NL banks (greetings, closings, one native answer set per text category), keyboard-typo helper. None of it is machine translated. |
| `scripts/v2_checks.py` | Measures every mechanism on a data directory (regex confusion, bot rule, dropout, consent, persona, fast humans, ghosting, interactions, planted-effect recovery). Writes the "v2 mechanisms" section of `ground_truth.md`; CLI `python scripts/v2_checks.py --data data/v2 --ref data/v1`. Evaluation side only (reads ground truth). |
| `scripts/ground_truth_report.py` | Reads `ground_truth_companies.csv` when present; consent-declined rows get their own level in the telemetry tables. v1 output unchanged. |
| `scripts/compare_to_v1.py` | `--gen-label v2` names the compared directory; new table levels (consent) no longer crash the "largest deviations" step. `python scripts/compare_to_v1.py --gen data/v2 --gen-label v2 --md out.md` gives the v1-vs-v2 fidelity tables. |
| `tests/test_generate_data_v2.py` | 35 tests, about 57 s. The per-option tests run twice, with paraphrase off and on (with a fixture cache). |
| `data/v2/` | Committed output: `python scripts/generate_data_v2.py` (seed 20260924, n = 10,000, with paraphrase). Adds `ground_truth_companies.csv` (hidden full firmographics) and five label columns: `context_persona`, `consent_declined`, `fast_human`, `domain_in_companies`, `language_mix`. |

## Paraphrase (5.2 i): done with the real API

The key became available during the task. `--probe` returned 200, then `--populate` filled the cache: **62 templates x 5
paraphrases** (76 calls in total: 14 were spent before the two fixes below). `data/v2/` is generated **with** paraphrase.

- Prompt v1 gave `{team}` as an example of a placeholder to keep. Haiku then put `{team}` into a paraphrase of a template
  that has no placeholder, and validation stopped the run. Prompt v2 names only the template's own placeholders, or says
  "no curly braces". The v1-prompt entries were never committed.
- Lorem ipsum is not sent. Haiku returned it unchanged five times, which fails the distinctness check, and pasted filler is
  not paraphrased in real life either.
- Haiku turned "money is tight" into "our budget is limited" in 2 of the 125 persona paraphrases (25 sentences x 5). That would give the regex a
  keyword for the persona, so `persona_safe` drops persona paraphrases that `text_cat` would not call neutral. Paraphrases
  of every other template are used as returned; for example "Trying to get more from our LinkedIn spend" → "maximize our
  LinkedIn advertising budget" now reads as specific to the regex while its true category stays neutral. That is the
  kind of disagreement 5.2 is meant to create.
- To regenerate: `python scripts/paraphrase_templates.py --populate` (a no-op while the cache is complete), then
  `python scripts/generate_data_v2.py`. Without the API: `python scripts/generate_data_v2.py --no-paraphrase`. A run with
  paraphrase on and any template missing stops with `MissingParaphraseError` and prints the populate command.

## Orchestrator decisions

- **D1. The persona must always be readable from the text.** Personas are assigned only to genuine-human leads whose
  answer is neutral or specific (`PERSONA_TEXT_CATEGORIES`). The per-lead probability is scaled up by the eligible share,
  so the overall share stays near 8%. Every persona lead gets its persona sentence, so the −1.0 is never applied without a
  text trace. On `data/v2`, **0 persona leads have no text trace**. Checked two ways:
  - by construction: no persona lead has a vague or copy-paste category or an empty answer;
  - against a twin: regenerating `data/v2` with only the persona off changes the text of all 834 persona rows and of no
    other row.
- **D2.** One report for 5.2-5.8 (one PR) is accepted.
- **D3. Native DE/FR/NL answers keep the English persona sentence** (code-switching). Accepted as realistic, and documented
  in the 5.7 row of `data/v2/ground_truth.md`.
- **Persona paraphrase filter at generation time.** `paraphrase_cache.json` stays raw Haiku output; `persona_safe` in the
  generator drops the 2 keyworded persona paraphrases.
- **LinkedIn × 51+ stays at +0.8.** Phase 4 should check that its models can detect this term. The outcome regression on
  decided leads gives +0.87 [+0.50, +1.23] on `data/v2`.
- **Consent missingness includes bots, as intended.** Bots hiding behind declined consent is realistic, and it is where
  the bot rule's missed bots come from.

## Review fixes

- **Independent text streams.** Before the fix, paraphrase draws for the base answer, the boilerplate and the persona
  sentence shared one stream, and typo draws depended on text length. Switching on one text option therefore changed
  other leads' texts: at n = 1,200, turning the persona on changed 854 non-persona texts. Now:
  - each slot has its own stream (`v2_paraphrase_base`, `v2_paraphrase_boilerplate`, `v2_paraphrase_persona`,
    `v2_boilerplate`, `v2_language`, `v2_typos`);
  - each lead takes exactly one draw per stream, which seeds a per-lead generator (`_child`);
  - persona assignment always takes three draws per lead;
  - the fast-human draw is made for every human session whether or not consent was declined, so the consent and
    fast-human streams no longer interact.

  The per-option "changes only what it should" tests now run twice: once plain and once with paraphrase on, using a
  fixture cache.
- **Smaller fixes.**
  - `CONSENT_COLUMNS` is imported from the generator instead of being duplicated in `v2_checks.py`.
  - The `company_name_variation_share` → `enrichment_dropout` coupling is documented in the `Config` docstring.
  - `ghost_tier_eff` and `stall_by_tier` are typed `dict[str, float]`, and the reply-time map is a named constant
    (`REPLY_TIER_LOG_SHIFT`).
  - `compare_to_v1.py` now passes `labels=LEGACY` explicitly: since Phase 1, `emva.pipeline.run` defaults to horizon
    labels, and this section is meant to show the frozen baseline.
- **v1 output still byte-identical** with every option off. The sha256 pins in `tests/test_generate_data_v2.py` did not
  need updating; the v1 pin test passes unchanged.

## Acceptance

All numbers below are for `data/v2` as regenerated after the review fixes.

| Criterion | Result | |
|---|---|---|
| Regex agreement with the generator's text category < 100% | **66.5%** of leads (68.5% of genuine humans). v1: 100%. Same data without paraphrase: 79.9% | Pass |
| Bot-rule precision < 100% (`emva.io`: headless UA or under 15 s) | **Precision 72.4%, recall 90.2%.** 496 leads flagged; of the 398 true bots, 359 are caught. All 137 false positives are 5.8 fast humans. All 39 missed bots are consent-declined, non-headless bots with a blank time on page. v1: 100% / 100% | Pass |
| Current pipeline runs on v2 with the standard report, AUC with CI | Legacy test set: baseline **0.789 [0.763, 0.815]**, status quo 0.637 [0.600, 0.673]. Horizon mature test set: baseline **0.776 [0.721, 0.829]**, candidate 0.781 [0.729, 0.832] | Pass |
| `--label-mode legacy` and default (horizon) | `python -m emva --data data/v2 --label-mode legacy`: AUC 0.789, Brier 0.0983, top-20% wins 0.525, revenue 0.722. Default horizon: AUC 0.781, Brier 0.1249, wins 0.513, revenue 0.685. The v2 **mature test set has 441 leads (78 won)**; the legacy test set has 2,309 (303 won) | Pass |
| All planted v1 effects still present | OLS of logit(p_close_true) on every planted feature recovers all 44 coefficients, residual sd 0.496 (planted 0.5). Largest gaps: under 15 s −1.33 vs −1.5 (truncation at p > 0.001, since bots sit near 0) and over 600 s −0.28 vs −0.4 (small cell). The v1 decided-win-rate test also passes on v2 | Pass |

Regex category (columns) vs generator category (rows), `data/v2`:

| | specific | neutral | vague | copy_paste |
|---|---|---|---|---|
| specific | 1904 | 278 | 0 | 0 |
| neutral | 91 | 3862 | 0 | 0 |
| vague | 0 | 1980 | 873 | 0 |
| copy_paste | 7 | 990 | 0 | 15 |

Vague answers lose the most. The regex needs an exact match, and paraphrases ("hi" → "hello there"), typos and native
fragments ("Infos bitte") never match. The copy-paste detector is nearly blind: 15 of 1,012 are caught, because v2
boilerplate rarely starts with one of the three fixed prefixes. Plan item 2.5 replaces that detector.

### Standard report on data/v2 (`python -m emva.eval.report --data data/v2`)

The report scores every model under both label definitions. Candidate = the `emva` pipeline trained with horizon labels
(H = 120). Phases 2 to 4 will re-run this on v2.

Label counts over the 9,209 scored leads: 1,128 won, 4,498 CRM-lost, 786 stalled, 1,256 ghosted, 1,541 open. The latest
mature lead was created 2026-05-26.

| definition | train (wins) | test (wins) |
|---|---|---|
| legacy | 5,509 (825) | 2,309 (303) |
| horizon H=120 | 3,789 (734) | **441 (78)** |

**(a) Legacy labels, legacy test set** (2,309 leads created on or after 2026-05-01, 303 won)

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.789 [0.763, 0.815] | 0.0983 | 0.525 | 0.673 | 0.719 |
| candidate | 0.791 [0.764, 0.816] | 0.1004 | 0.535 | 0.677 | 0.727 |
| status quo | 0.637 [0.600, 0.673] | n/a | 0.393 | 0.427 | 0.427 |

- **Paired AUC vs baseline:** candidate +0.002 [−0.004, +0.008], status quo −0.152 [−0.185, −0.120].
- **AUC by month** (baseline / status quo): May 0.788 / 0.660, Jun 0.781 / 0.627, Jul 0.803 / 0.623, Aug 0.750 / 0.608,
  Sep 0.937 / 0.689 (85 leads).
- **Value scale, test set** (max/median, top-1% share): baseline 73.9x, 11.1%; candidate 57.8x, 10.0%; status quo
  19.5x, 6.5%.

**(b) Horizon labels, mature test set** (441 mature leads created 2026-05-01 to 2026-05-26, 78 won; ghosted-at-H
excluded, stalled censored)

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.776 [0.721, 0.829] | 0.1250 | 0.526 | 0.632 | 0.663 |
| candidate | 0.781 [0.729, 0.832] | 0.1249 | 0.513 | 0.646 | 0.674 |
| status quo | 0.654 [0.575, 0.728] | n/a | 0.449 | 0.512 | 0.512 |

- **Paired AUC vs baseline:** candidate +0.005 [−0.009, +0.018], status quo −0.122 [−0.190, −0.051].
- The mature test set covers one month (May), so its CI is about twice as wide as the legacy one.
- **Value scale, test set:** baseline 58.8x, 10.7%; candidate 49.3x, 9.6%; status quo 15.0x, 5.8%.

**For reference:**
- The same data without paraphrase gives a legacy AUC of 0.795.
- The frozen v1 baseline is 0.814 [0.789, 0.836].
- The regenerated v1 across 8 seeds is 0.805 ± 0.013 (reports/phase5-1.md).

## Per-option numbers (data/v2, n = 10,000)

| Item | Planted | Measured |
|---|---|---|
| 5.2 typos | 30% of texts, 1.5% of letters (at least 1), letters only | part of the 66.5% agreement |
| 5.2 languages | 35% of DE/FR/NL answers: half wrapped in a native greeting/closing, half replaced by a native answer of the same class (a persona sentence stays in English, D3) | 1,216 original leads (614 native, 602 wrapped) |
| 5.2 boilerplate | 3 v1 texts + 15 new snippets, 9 openings, 35% mid-text, 15% two snippets | regex copy-paste recall 1.5% (15 of 1,012) |
| 5.3 dropout | 30% of business-email domains replaced in companies.csv by a legacy domain; row kept | **Denominator:** the 1,919 distinct company domains of business-email original leads, the set the dropout samples from. **576 of 1,919 (30.0%) are missing** from companies.csv. (1,966 is the number of distinct company domains over all leads with a company, free-email leads included; the same 576 are missing, and 2,000 companies exist in the hidden list.) **2,031 domain-miss leads** (business email, no domain join). 1,219 of them typed a company name (forms B/C/D), and **all 1,219 normalise to the right company** (lower-case, drop `.,&-`, strip Ltd/Limited/Inc/GmbH/SAS/BV). The other 812 are on form A or Lead Ads, which have no company field. All leads without a domain join: 5,315, of which 2,260 typed a name that matches some companies.csv row |
| 5.3 name noise | 70% of typed names: suffix spelling, comma before the suffix, missing suffix added back, `&` or `-` between words, case, stray spaces | 838 of 5,296 typed names are still the exact companies.csv string |
| 5.4 consent | 15% of website sessions (bots included, as intended), 16 telemetry columns blank | **1,347 rows = 14.8% of the 9,107 website rows**; exactly those columns are blank; no Lead Ads rows |
| 5.5 interactions | LinkedIn × company with 51+ employees +0.8; senior × form D −0.8 | Outcome LR on decided original leads (main effects plus both terms): **+0.87 [+0.50, +1.23]** (n = 412) and **−0.82 [−1.27, −0.38]** (n = 345). The same regression on v1: +0.15 [−0.20, +0.50] and −0.05 [−0.44, +0.35]. logit(p) regression: +0.78 and −0.78 |
| 5.6 ghosting | never-contacted log-odds −5.3 + {A 0, B 2.8, C 4.4}, nothing else; stalls A 5%, B 10%, C 17% | See below |
| 5.7 persona | 8% of original leads, among genuine humans with a neutral or specific answer (D1), −1.0 | **809 of 9,700 original leads (8.3%)**; 834 of 10,000 rows (8.3%, duplicates copy the original's persona). Nonprofit 213, competitor 205, job seeker 197, agency 194. **Persona leads without a text trace: 0.** Decided win rate among eligible leads (genuine human, neutral/specific, original): 15.2% with a persona (495 decided) vs 26.3% without. Outcome LR −0.99 [−1.27, −0.70]. Regex category shares, persona vs other neutral/specific human texts: specific 33.0 / 32.6%, neutral 67.0 / 67.4%; largest gap 0.4 pp |
| 5.8 fast humans | 2% of genuine-human website sessions with telemetry, 5-14.9 s, close odds unchanged | **137 sessions = 1.8%** of 7,425 eligible |

### Ghosting: correlation of never-contacted with tier and with true p (original leads)

| data | never contacted | by tier A / B / C | corr with tier (A=0, B=1, C=2) | corr with true p |
|---|---|---|---|---|
| v1 (`data/v1`) | 19.1% | 2.0% / 10.6% / 24.8% | 0.194 | −0.120 |
| v1 regenerated (seed 20260924) | 18.9% | 1.7% / 10.7% / 24.2% | 0.188 | −0.117 |
| v2 (`data/v2`) | 20.6% | 0.8% / 8.1% / 27.7% | **0.240** | **−0.065** |

In v1, ghosting already followed tier more than p. v2 removes the free-email and vague-text terms, which carry real
signal, and sharpens the tier gradient. So ghosting now tracks the old score more (0.194 → 0.240) and true quality less
(|−0.120| → |−0.065|).

## Fidelity v2 vs v1 (`compare_to_v1.py --gen data/v2 --gen-label v2`)

- **Outcomes, v2 vs v1:**

  | statistic | v2 | v1 |
  |---|---|---|
  | won share | 11.4% | 12.0% |
  | decided win rate | 19.3% | 19.0% |
  | never contacted | 1,999 | 1,850 |
  | open | 1,772 | 1,538 |

  Open leads rise because stalls now follow tier.
- **Win-rate gaps:** mean absolute gap over cells with ≥ 100 decided v1 leads is 1.59 pp. The largest are LinkedIn
  +7.5 pp (the planted interaction), income under £25k 6.0 pp, under 15 s +5.8 pp (fast humans are real leads) and
  1000+ 5.4 pp.
- **Share gaps:** the largest come from the consent level. The "under 90s", "match", "no" and 60-300 s levels each lose
  8-12 pp to "blank (consent declined)".
- **Baseline model, legacy labels:** AUC 0.789 vs 0.814.
- **Baseline weights, v2 vs v1:**
  - `email=free` −1.34 (v1 −0.39) and `no_company=yes` +0.48 (v1 −0.39). Dropout breaks the v1 identity between the two
    columns, so the fit no longer splits the free-email weight in half.
  - `text=copy_paste` flips to +0.37, since only 22 leads are still regex copy-paste.
  - `time_on_page=60-300s` drops from 0.51 to 0.11.
  - Across all 47 weights: correlation 0.710, mean absolute difference 0.241.

## Deviations and judgement calls

- **Two v1 tests changed.** `test_v2_options_refused_until_implemented` asserted `NotImplementedError` for every option, and
  `test_v2_options_cover_all_seven_phase5_tasks` asserted exactly seven option names. Neither can pass once the options are
  implemented, and 5.2 needed four switches (paraphrase, typos, languages, boilerplate). I replaced both with one test:
  every task 5.2-5.8 has an option, and all are off by default. Every other v1 test is unchanged and passes.
- **Hidden truth for enrichment dropout.** A dropped company keeps its companies.csv row under a legacy domain
  (`<word><sect>-group.example`), rather than losing the row. Without the row, plan item 2.3 (typed-name matching) would
  have nothing to match against. A blank domain was avoided because pandas joins NaN to NaN in `emva.io.load`. The full
  table is written as `ground_truth_companies.csv`.
- **What consent-declined sessions keep.** User agent, device, referrer, landing URL and click IDs (request and URL data)
  are kept. The `consent` marketing-opt-in column is independent of `consent_declined`.
- **What is true and what is recorded.** Consent-declined and fast-human sessions keep their real behaviour in the close
  log-odds; only the recorded columns change.
  - The status-quo tier, and so ghosting, uses the recorded pages. A declined session therefore loses its page points,
    which shifts the v1 `hidden` stream for later rows (a within-stage coupling, documented in the generator).
  - 5.6 consumes the v1 contact and stall draws and decides from its own streams. The noise and deal-value draws stay
    aligned, and `p_close_true` changes only where the reply time changed (tested).
- **Interaction choice.** LinkedIn × 201+ employees was too small a cell (about 45 decided at n = 3,000). I widened it to
  51+ employees.
- **Generator imports `emva.features.text_cat`**, only for `persona_safe`. `emva/` does not import the generator.

## Open questions

None outstanding.

## Runtime

- **Generation:** about 21 s for v2 at n = 10,000; paraphrase adds nothing measurable.
- **`test_generate_data_v2.py`:** about 57 s.
- **Full `pytest`** after the Phase 1 rebase and the review fixes: 221 passed in about 119 s.
