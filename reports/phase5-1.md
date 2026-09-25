# Phase 5.1: recover / rewrite the v1 data generator

Branch `phase5-generator-v1`. Plan item 5.1 only; no v2 behaviour.

## What was built

| File | Purpose |
|---|---|
| `scripts/generate_data_v1.py` | Deterministic, seeded generator: `python scripts/generate_data_v1.py --seed 20260924 --as-of 2026-09-24 --out DIR --n 10000`. Writes the six v1 files plus `ground_truth.md` measured on its own output. |
| `scripts/ground_truth_report.py` | Computes every `ground_truth.md` table from a data directory and renders the markdown. Run on `data/v1` it reproduces the published v1 tables (band "none" 917 leads at 2.5%, tier A 402 at 45.5%, 2,139 messy stage rows, 3,837 unmatched leads), which validates the measurement code. |
| `scripts/compare_to_v1.py` | Every table side by side with absolute differences, scalar statistics, the baseline model (`emva.pipeline.run`) on both directories, optional seed-to-seed spread (`--seeds 1,2,...`). |
| `tests/test_generate_data_v1.py` | 22 pytest tests at n = 2000: determinism, schema equality with v1, sanity ranges, pipeline smoke test. |

Pipeline: one function per stage, each with its own RNG stream (`default_rng([seed, crc32(stage)])`) so a later
option in one stage does not shift the draws of the others: companies → personas → leads → behaviour → text →
hidden log-odds (tier, contact decision, reply time, deal value) → outcome → duplicates → CRM → vendor people →
labels. The Phase 5.2-5.8 options exist as named `Config` fields; setting any of them raises `NotImplementedError`
until it is implemented.

Every planted effect is taken verbatim from `ground_truth.md`. A regression of logit(`p_close_true`) on the planted
features in v1 recovers every planted coefficient with residual sd 0.505, so the documented model is exactly what
produced v1. Everything the document does not state was estimated from a profile of the v1 files and hard-coded; the
generator never reads `data/v1` at runtime.

## Exact reproduction: not achieved (expected)

Across all five CSVs, 0 of v1's rows appear anywhere in the output. The original's RNG library, call order and
vocabulary list orders are unknown, and each lead involves about 50 conditional draws.

One weak hint: Python's `random.Random(20260924)`, drawing from alphabetically sorted lists, gives `Ashford` then
`Healthcare`, which matches v1's first company and would happen by chance about 1 time in 560. The third draw
already diverges and the second company could not be aligned, so the attempt stopped there. Distributional fidelity
is reported instead.

## Orchestrator decision

Exact row-level reproduction of v1 is accepted as infeasible. Distributional fidelity, as measured below, is the
accepted pass criterion for 5.1.

## Fidelity (seed 20260924, n = 10,000)

| Statistic | v1 | Generated |
|---|---|---|
| Won share of all leads | 11.99% | 11.99% |
| Win rate among decided leads | 19.0% | 19.1% |
| Never contacted | 1,850 | 1,833 |
| Open | 1,538 | 1,588 |
| Bot share (baseline rule) | 3.89% | 3.98% |
| Duplicate share | 3.00% | 3.00% |
| Label positive rate | 14.91% | 14.98% |
| Missing click IDs | 2,340 | 2,397 |
| Messy stage rows | 2,139 | 2,119 |
| Won deal median | £8,850 | £9,100 |
| Won deal mean | £12,963 | £13,325 |
| Median days to Won | 35.3 | 34.7 |

- Over the 90 table cells with at least 100 decided v1 leads, the mean win-rate gap is 1.14 pp and the mean share gap 0.66 pp.
- Largest win-rate gaps: income £100k-£150k 3.8 pp, time on page 300-600s 3.3, income under £25k 3.2, pages visited = 2 at 3.1, IP mismatch 3.0.
- Largest share gaps: time on page 60-300s +3.0 pp, band 201-1000 +2.3, one session +2.1.
- Across 8 other seeds, 5 of 90 cells sit more than 2 sd from the seed mean and none more than 3: about what sampling noise alone gives.

## Baseline model on generated vs v1 data

| Metric | v1 | Seed 20260924 | 8 other seeds, mean ± sd [min, max] |
|---|---|---|---|
| AUC | 0.814 | 0.799 | 0.805 ± 0.013 [0.781, 0.824] |
| Top-20% wins | 0.571 | 0.538 | 0.542 ± 0.021 [0.501, 0.565] |
| Top-20% revenue | 0.795 | 0.745 | 0.777 ± 0.029 [0.751, 0.834] |

- Seed choice alone moves these metrics by about ±0.015 AUC and ±0.03 on the top-20% shares. Later phases should compare baseline and candidate on the same data, or across seeds, not against v1's single numbers.
- 31 of the 35 planted-effect weights put v1 within 2 seed-sd of the generated mean. Outside: student (−1.47 v1 vs −0.86 generated; looks like v1 luck, since true p is the same 0.043 vs 0.045 but v1 had 9 student wins where about 16 were expected), band 201-1000, time on page 60-300s and hiring (each about 2 sd).
- Seed 20260924 vs v1 across all 47 weights: correlation 0.863, mean absolute difference 0.166.
- The second-largest single-seed weight gap is a sign flip on `ip_country=mismatch`: +0.303 generated vs −0.331 v1
  (the largest is student, 0.89). The planted −0.3 is applied in code (`Config.ip_mismatch_eff`, used in
  `close_log_odds`), and across the 8 other seeds the learned weight is −0.375 ± 0.339, so the flip is sampling noise
  on a small group (about 7% of decided leads), not a missing effect.
- `no_company=yes` always equals `email=free` in the table. That is an artefact of the baseline encoding: both columns
  are identical for every lead, so the L2 fit splits the free-email weight evenly between them. It is not a planted
  signal. Phase 2 (2.2, 2.7) removes the collinearity.

Full tables: `python scripts/compare_to_v1.py --gen DIR --seeds 1,2,3,4,5,6,7,8 --md out.md`.

## Runtime

Generate n = 10,000: 17 s. Comparison: about 10 s, or 3 min with 8 seeds. Tests: 14 s.

## Deviations and judgement calls

- `status_quo_rules.json` is a fixed config, emitted with the same content as v1's.
- The rewritten `ground_truth.md` adds the ad-spend multiplier to the deal-value formula line (v1 applied it but omitted it from the line) and drops v1's unreproducible "AUC 0.778 vs 0.779" sentence.
- v1 quirks kept on purpose: 30% of bots are labelled business-email but have a free-domain email; duplicate rows copy the original's hidden labels while channel and click-ID fields describe the new visit; `created_at` is capped at as-of minus 1 hour.
- Small structural gaps remain: time on page 60-300s share about 3 pp high; form C budget/timeline correlation differs (not a planted signal); v1's stronger learned hiring and sessions weights not fully reproduced.
- No regenerated data is committed; it is reproducible from the command above.

## What Phase 5.2 to 5.8 will need to change

| Task | Where | Change |
|---|---|---|
| 5.2 free text | `add_text` | Paraphrase each template once via Haiku into a cache; add typos, DE/FR/NL language mixing and a wider boilerplate pool. The test asserting 100% regex agreement becomes the "< 100%, reported" check. |
| 5.3 enrichment dropout | companies output step, `typed_company` | Drop 30% of business domains from the emitted `companies.csv` but keep them in the hidden truth; widen typed company-name variation. |
| 5.4 consent missingness | `session_behaviour` | Blank the website telemetry for 15% of website leads, flagged in the labels. |
| 5.5 interactions | `hidden_log_odds` | Add channel × band and seniority × form terms. |
| 5.6 ghosting | never-contacted rule, `draw_outcome` | v1 already ghosts by tier, free email and vague text, not by true p. Strengthen the tier effect and make stalls follow tier too. |
| 5.7 context-only signal | `add_text`, `hidden_log_odds` | Hidden persona (nonprofit, job seeker, competitor, agency) with a −1.0 effect, visible only in the text's meaning. |
| 5.8 fast humans | `session_behaviour` (time-on-page line marked as the hook) | 2% of genuine humans under 15s on the page. |
