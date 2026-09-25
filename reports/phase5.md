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
| `tests/test_generate_data_v2.py` | 26 tests, about 45 s. |
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

## Acceptance

| Criterion | Result | |
|---|---|---|
| Regex agreement with the generator's text category < 100% | **66.5%** of leads (68.5% of genuine humans). v1: 100%. Without paraphrase: 80.4% | Pass |
| Bot-rule precision < 100% (`emva.io`: headless UA or under 15 s) | **Precision 72.7%, recall 90.2%.** 494 flagged, 359 true bots. All 135 false positives are 5.8 fast humans. All 39 missed bots are consent-declined, non-headless bots with a blank time on page. v1: 100% / 100% | Pass |
| Current pipeline runs on v2 with the standard report, AUC with CI | Baseline AUC **0.785 [0.758, 0.810]**, status quo 0.649 [0.618, 0.682]. Table below | Pass |
| `--label-mode legacy` and default (horizon) | After rebasing onto Phase 1 (92168cf), both run. `python -m emva --data data/v2 --label-mode legacy`: AUC 0.785. Default horizon: AUC 0.791. Both standard tables are below. The v2 mature test set has **448 leads (75 won)**, against 2,336 (327 won) under legacy labels | Pass |
| All planted v1 effects still present | OLS of logit(p_close_true) on every planted feature recovers all 44 coefficients, residual sd 0.497 (planted 0.5). Largest gaps: under 15 s −1.33 vs −1.5 (truncation at p > 0.001, since bots sit near 0) and over 600 s −0.26 vs −0.4 (small cell). The v1 decided-win-rate test also passes on v2 | Pass |

Regex category (columns) vs generator category (rows), `data/v2`:

| | specific | neutral | vague | copy_paste |
|---|---|---|---|---|
| specific | 1917 | 265 | 0 | 0 |
| neutral | 95 | 3858 | 0 | 0 |
| vague | 0 | 1996 | 857 | 0 |
| copy_paste | 11 | 982 | 0 | 19 |

Vague answers lose the most. The regex needs an exact match, and paraphrases ("hi" → "hello there"), typos and native
fragments ("Infos bitte") never match. The copy-paste detector is nearly blind: 19 of 1,012 are caught, because v2
boilerplate rarely starts with one of the three fixed prefixes. Plan item 2.5 replaces that detector.

### Standard report on data/v2 (`python -m emva.eval.report --data data/v2`)

The report scores every model under both label definitions. Candidate = the `emva` pipeline trained with horizon labels
(H = 120). Phases 2 to 4 will re-run this on v2.

Label counts over the 9,212 scored leads: 1,139 won, 4,516 CRM-lost, 785 stalled, 1,251 ghosted, 1,521 open. The latest
mature lead was created 2026-05-26.

| definition | train (wins) | test (wins) |
|---|---|---|
| legacy | 5,507 (812) | 2,336 (327) |
| horizon H=120 | 3,787 (718) | **448 (75)** |

**(a) Legacy labels, legacy test set** (2,336 leads created on or after 2026-05-01, 327 won)

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.785 [0.758, 0.810] | 0.1025 | 0.529 | 0.635 | 0.714 |
| candidate | 0.785 [0.758, 0.809] | 0.1057 | 0.511 | 0.645 | 0.718 |
| status quo | 0.649 [0.618, 0.682] | n/a | 0.385 | 0.441 | 0.441 |

Paired AUC vs baseline: candidate −0.000 [−0.006, +0.006], status quo −0.136 [−0.164, −0.107]. AUC by month (baseline /
status quo): May 0.793 / 0.701, Jun 0.784 / 0.623, Jul 0.798 / 0.658, Aug 0.731 / 0.562, Sep 0.910 / 0.675 (95 leads).
Value scale, test set: baseline max/median 95.0x, top-1% share 11.5%; candidate 73.4x, 10.6%; status quo 19.5x, 6.4%.

**(b) Horizon labels, mature test set** (448 mature leads created 2026-05-01 to 2026-05-26, 75 won; ghosted-at-H
excluded, stalled censored)

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.787 [0.730, 0.835] | 0.1154 | 0.493 | 0.578 | 0.620 |
| candidate | 0.791 [0.731, 0.839] | 0.1164 | 0.493 | 0.589 | 0.623 |
| status quo | 0.663 [0.588, 0.731] | n/a | 0.453 | 0.446 | 0.446 |

Paired AUC vs baseline: candidate +0.004 [−0.010, +0.017], status quo −0.125 [−0.185, −0.065]. The mature test set is
one month (May), so its CI is about twice as wide as the legacy one. Value scale, test set: baseline 80.5x, 11.1%;
candidate 70.2x, 10.4%; status quo 15.0x, 5.7%.

`python -m emva` summaries on data/v2 (blank deal values filled): legacy AUC 0.785, Brier 0.1025, top-20% wins 0.529,
revenue 0.728; horizon AUC 0.791, Brier 0.1164, wins 0.493, revenue 0.657. Before Phase 1, the same data without paraphrase
gave AUC 0.790 (legacy). For reference, the frozen v1 baseline is 0.814 [0.789, 0.836], and the regenerated v1 across
8 seeds is 0.805 ± 0.013 (reports/phase5-1.md).

## Per-option numbers (data/v2, n = 10,000)

| Item | Planted | Measured |
|---|---|---|
| 5.2 typos | 30% of texts, 1.5% of letters (at least 1), letters only | part of the 66.5% agreement |
| 5.2 languages | 35% of DE/FR/NL answers: half wrapped in a native greeting/closing, half replaced by a native answer of the same class | 1,183 originals (602 native, 581 wrapped) |
| 5.2 boilerplate | 3 v1 texts + 15 new snippets, 9 openings, 35% mid-text, 15% two snippets | regex copy-paste recall 1.9% |
| 5.3 dropout | 30% of business-email domains replaced in companies.csv by a legacy domain; row kept | **576 of 1,919** business domains missing. **2,031 domain-miss leads** (business email, no domain join), 1,219 of them typed a company name (forms B/C/D), and **all 1,219 normalise to the right company** (lower-case, drop `.,&-`, strip Ltd/Limited/Inc/GmbH/SAS/BV). 812 are variant A or Lead Ads with no company field. All leads without a domain join: 5,315, of which 2,260 typed a name that matches some companies.csv row |
| 5.3 name noise | 70% of typed names: suffix spelling, comma before the suffix, missing suffix added back, `&` or `-` between words, case, stray spaces | 838 of 5,296 typed names are still the exact companies.csv string |
| 5.4 consent | 15% of website sessions (bots included), 16 telemetry columns blank | **1,347 rows = 14.8% of website rows**; exactly those columns are blank; no Lead Ads rows |
| 5.5 interactions | LinkedIn × company with 51+ employees +0.8; senior × form D −0.8 | Outcome LR on decided originals (main effects plus both terms): **+0.47 [+0.10, +0.84]** (n = 417), **−0.99 [−1.44, −0.54]** (n = 350). The same regression on v1 gives +0.15 [−0.20, +0.50] and −0.05 [−0.44, +0.35]. logit(p) regression: +0.79 and −0.78 |
| 5.6 ghosting | never-contacted log-odds −5.3 + {A 0, B 2.8, C 4.4}, nothing else; stalls A 5%, B 10%, C 17% | See below |
| 5.7 persona | 8% of genuine-human leads, −1.0 | **7.9% of all leads** (8.2% of human originals): job seeker 193, agency 191, nonprofit 190, competitor 188. Decided win rate 10.6% vs 20.2%. Outcome LR −0.94 [−1.27, −0.61]. Regex category shares, persona vs other human texts: specific 23.4 / 20.9%, neutral 68.5 / 70.7%, vague 8.1 / 8.2%, copy-paste 0.0 / 0.2%. Largest gap 2.4 pp, the same size as the gap in the true categories (2.6 pp), so it is sampling noise |
| 5.8 fast humans | 2% of genuine-human website sessions with telemetry, 5-14.9 s, close odds unchanged | **135 sessions = 1.8%** of 7,425 eligible |

### Ghosting: correlation of never-contacted with tier and with true p (original leads)

| data | never contacted | by tier A / B / C | corr with tier (A=0, B=1, C=2) | corr with true p |
|---|---|---|---|---|
| v1 (`data/v1`) | 19.1% | 2.0% / 10.6% / 24.8% | 0.194 | −0.120 |
| v1 regenerated (seed 20260924) | 18.9% | 1.7% / 10.7% / 24.2% | 0.188 | −0.117 |
| v2 (`data/v2`) | 20.6% | 0.8% / 8.1% / 27.7% | **0.240** | **−0.066** |

In v1, ghosting already followed tier more than p. v2 removes the free-email and vague-text terms, which carry real
signal, and sharpens the tier gradient. So ghosting now tracks the old score more (0.194 → 0.240) and true quality less
(|−0.120| → |−0.066|).

## Fidelity v2 vs v1 (`compare_to_v1.py --gen data/v2 --gen-label v2`)

- Won share 11.5% vs 12.0%, decided win rate 19.4% vs 19.0%, never contacted 1,999 vs 1,850, open 1,747 vs 1,538 (stalls now follow tier).
- Mean absolute win-rate gap over cells with ≥ 100 decided v1 leads: 1.49 pp. Largest: LinkedIn +8.0 pp and 1000+ +6.9 pp (the planted interaction), under 15 s +4.8 pp (fast humans are real leads).
- The largest share gaps are the consent level: the "under 90s", "match", "no" and 60-300 s levels each lose about 10 pp to "blank (consent declined)".
- Baseline weights, v2 vs v1: `email=free` −1.40 (v1 −0.39) and `no_company=yes` +0.65 (v1 −0.39). Dropout breaks the v1 identity between the two columns, so the fit no longer splits the free-email weight in half. `text=copy_paste` flips to +0.43, since only 30 leads are still regex copy-paste. `time_on_page=60-300s` drops from 0.51 to −0.01.

## Deviations and judgement calls

- **Two v1 tests changed.** `test_v2_options_refused_until_implemented` asserted `NotImplementedError` for every option and
  `test_v2_options_cover_all_seven_phase5_tasks` asserted exactly seven option names. Neither can pass once the options are
  implemented, and 5.2 needed four switches (paraphrase, typos, languages, boilerplate). I replaced both with one test:
  every task 5.2-5.8 has an option, and all are off by default. Every other v1 test is unchanged and passes.
- **Hidden truth for enrichment dropout.** A dropped company keeps its companies.csv row under a legacy domain
  (`<word><sect>-group.example`), rather than losing the row. Without the row, plan item 2.3 (typed-name matching) would
  have nothing to match against. A blank domain was avoided because pandas joins NaN to NaN in `emva.io.load`. The full
  table is written as `ground_truth_companies.csv`.
- **Consent covers bots too, as intended** (the plan says "website leads"; orchestrator decision: accepted). Bots hiding
  behind declined consent is realistic, and it is where the bot rule's missed bots come from.
  Consent-declined sessions keep user agent, device, referrer, landing URL and click IDs (request and URL data). The
  `consent` marketing-opt-in column is independent of `consent_declined`.
- **What is true and what is recorded.** Consent-declined and fast-human sessions keep their real behaviour in the close
  log-odds; only the recorded columns change. The status-quo tier, and so ghosting, uses the recorded pages. A declined
  session therefore loses its page points, which shifts the v1 `hidden` stream for later rows (a within-stage coupling,
  documented in the generator). 5.6 consumes the v1 contact and stall draws and decides from its own streams. The
  noise and deal-value draws stay aligned, and `p_close_true` changes only where the reply time changed (tested).
- **Personas are written only onto non-vague answers.** A one-word answer cannot carry a persona sentence, and appending
  one would move the text out of the regex's vague list. About 28% of persona leads therefore have no textual trace. This
  caps what Phase 6 can recover.
- **Interaction choice.** LinkedIn × 201+ employees was too small a cell (about 45 decided at n = 3,000). I widened it to
  51+ employees. Orchestrator decision: the planted +0.8 stays. The outcome regression recovers only +0.47 [+0.10, +0.84]
  (the logit(p) regression recovers +0.79), so **Phase 4 should check that its models can detect this term**.
- **Persona paraphrases are filtered at generation time** (orchestrator decision), so `paraphrase_cache.json` stays raw
  Haiku output. The filter is `persona_safe` in the generator, which imports `emva.features.text_cat` for it. `emva/`
  does not import the generator.

## Open questions

None outstanding. The persona filter, the interaction size and consent-for-bots were decided by the orchestrator (see above).

## Runtime

n = 10,000: v2 generation about 21 s (paraphrase adds nothing measurable). `test_generate_data_v2.py`: about 45 s. Full
`pytest` after the Phase 1 rebase: 212 passed in about 108 s.
