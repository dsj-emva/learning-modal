# CLAUDE.md: EMVA lead scoring

Read this first. Then `REBUILD_PLAN.md` (the roadmap), `docs/CONTEXT.md` (vocabulary, headline
numbers), `docs/ARCHITECTURE.md` (data flow, module ownership) and `docs/adr/` (every ruling so far).

## What this is and why

**Product.** EMVA scores inbound B2B leads and sends `P(close) × expected deal value × margin` back to
Google Ads and Meta as a conversion value, so the platforms bid for leads that become revenue rather
than form fills (`baseline/business_brief.md`). The score is an L2 logistic regression over bucketed
form, enrichment and on-site features, plus a ridge deal-value model; an optional Claude Haiku
"context agent" reads each lead against a business brief.

**Status: proof of concept on synthetic data.** v1 data (`data/v1/`) came from a generator with seed
20260924, as-of 2026-09-24; the original generator was lost and rewritten (`scripts/generate_data_v1.py`,
distributional fidelity only, ADR 0003). Generator v2 (`data/v2/`, merged 01c56f4) adds paraphrased text, missing enrichment, consent gaps, interactions and a hidden persona (ADR 0008). No real customer data
exists in this repo. The frozen POC lives in `baseline/`; the package being fixed is `emva/`.

**Why the plan exists.** The 2026-09-25 review found: labels were right-censored (slow or neglected
leads, stalled 90+ days or never contacted, were labelled lost); missing telemetry fell into the reference
bucket; `email=free` and `no_company=yes` were perfectly collinear (both −0.391); the deal model copied
its residual sd 0.45 from the ground-truth document; and the generator was itself an additive logistic
model, so the regex text categories and the bot rule matched it perfectly. The plan adjusts in place
behind a frozen baseline instead of rewriting, so every change is measured against the same reference
(ADR 0001).

## Ground rules (from `REBUILD_PLAN.md`; not negotiable)

1. **Never edit `baseline/`.** Every task reports its metrics next to the baseline's on the same frozen test set.
2. **Ground truth is evaluation-only.** `ground_truth_labels.csv`, `ground_truth.md` (and on v2
   `ground_truth_companies.csv`) may be read only under `emva/eval/` and by the generator side
   (`scripts/generate_data_v*.py` and its measurement scripts). Nothing else under `emva/` may import,
   read or hard-code anything from them.
   **The grep rule:** `grep -rn "0.45\|ground_truth" emva/` must return only `emva/eval/` lines
   (`tests/test_label_study.py::test_ground_rule_2_grep`). There is no exception outside `emva/eval/`
   since Phase 2 (plan 2.4): the baseline's 0.45 residual sd lives only in `emva/eval/regression.py`
   (`BASELINE_DEAL_LOG_RESIDUAL_SD`), used by `--feature-set legacy` to reproduce the baseline.
3. **Standard report.** Every task ends with `make report` (`python -m emva.eval.report`) pasted into
   `reports/<task>.md`: baseline vs candidate vs status quo, AUC with bootstrap CI, Brier, top-20% wins and
   revenue (by p and by p×value), decile calibration, AUC by month, value scale.
4. **Frozen test sets.** Legacy: labelled leads created on or after 2026-05-01. Mature (after Phase 1):
   the 458 leads of `emva.eval.report.FROZEN_HORIZON`. Both are reported. Never redefine either to
   make a number look better (ADR 0006).
5. **One task, one PR, one report.** Acceptance criteria are pass/fail. A task that fails its criterion
   still ships its report; it does not ship its code. Do not tune to hit an acceptance number (Phase 1
   deliberately did not tune H to rescue `band=1000+`).
6. **Claims policy.** Nothing from v1 is quoted externally. After Phase 5, numbers are quoted with CIs
   and the phrase "on simulated data". Real-data claims need a customer pilot (outside this plan).

## Orchestration conventions (used in every phase so far)

- One branch per task: `phaseN-<slug>` (`phase0-freeze-instrument`, `phase1-labels`,
  `phase5-generator-v1`, `phase2-features`, `phase5-generator-v2`). Report file `reports/phaseN.md`
  (sub-items: `reports/phase5-1.md`).
- Parallel agents work in separate git worktrees (`.claude/worktrees/`, gitignored). Agents push their
  branch; they never push to `main` and never open PRs. The orchestrator merges.
- Before merge: two-axis review, **standards** (does it follow this file and the repo conventions?) and
  **spec** (does it do what the plan item and rulings say?), then one fix round. Rulings made at review
  are recorded in the phase report under "Orchestrator decisions" and as an ADR.
- Every commit message ends with a `Co-Authored-By:` trailer for the agent. Commits are small and
  focused; intermediate commits should be green (Phase 2 noted where they are not).
- **Every merged phase must update the status table in `docs/CONTEXT.md` and this file, and add an ADR
  in `docs/adr/` for any orchestrator ruling.**

## Setup and verification

Python 3.11. The repo path contains spaces (`.../Mobile Documents/com~apple~CloudDocs/...`): quote every path.

```sh
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # pinned: numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.17.1, anthropic 1.8.0, requests, pytest
make baseline      # frozen baseline + `python -m emva --label-mode legacy` must give AUC 0.814, Brier 0.1006, top-20% wins 0.571, revenue 0.795, canonical weights
make test          # pytest (193 passed at the Phase 1 merge, ~60 s) then compileall
make report        # standard report on data/v1, both test definitions
make report-full   # Phase 4 hardening + standard report, data/v2 + data/v1 -> reports/phase4.md; how later phases inherit Phase 4 (R16, ADR 0013); ~10 min uncached, cached in runs/phase4/
make experiment NAME=horizon   # runs/NAME/ outputs + report; names in scripts/run_experiments.py
APP_PASSWORD=... make app      # Keel (the hosted tool) on :8501, data under DATA_DIR (default runs/app)
```

In a worktree there is no `.venv`: pass the main checkout's interpreter, quoted, e.g.
`make PY="<main checkout>/.venv/bin/python" baseline`. `DATA=data/v2` points make targets at v2.

**Pipeline CLI** (`python -m emva --data data/v1 --out DIR [--context FILE] [--margin 1.0]`, writes
`weights.csv` and `scores.csv`, and now also `model.joblib`, the `emva.persist.ModelBundle` that `emva.scoring` loads):

| flag | meaning |
|---|---|
| `--label-mode {legacy,horizon}` | `horizon` (default): won within H days, mature leads only. `legacy`: baseline labels, byte-identical outputs |
| `--horizon-days N` | H, default 120 (horizon mode only; with legacy it is a usage error) |
| `--include-ghosted` | horizon: count leads still New at H as 0 instead of excluding them |
| `--stalled-as-lost` | horizon: count stalled open deals as 0 instead of censoring them |
| `--value-cap-percentile P`, `--no-value-cap`, `--value-compression {none,log,sqrt}`, `--value-floor GBP`, `--no-value-floor`, `--value-tiers N` | horizon only (Phase 3): the transform from `value_formula` to `value_at_submit`; default cap p97 + log + floor £25 (ADR 0012). Horizon `scores.csv` also carries `value_at_submit(_ts)`, `value_at_close(_ts)`, `value_at_close_status` |
| `--feature-set {legacy,v2,generic}` | `v2` (default, Phase 2): `session_missing` / `enrichment_missing` indicators, name enrichment, boilerplate similarity, fixed 39-column design from `V2_LEVELS` (ADR 0009), residual sd estimated. `legacy`: baseline features; with `--label-mode legacy` byte-identical to `baseline/`. `generic` (Phase 10, ADR 0024): v2 plus the `x_` columns of a converted dataset's declared extras (`extra_features.csv`, encoding fitted on training leads; refused without it); the report adds "Generic vs v2" |

The report also takes `--strict` (exit 1 when the candidate design fails the collinearity check;
`python -m emva.eval.collinearity` is always strict).

The same label flags work on `python -m emva.eval.report`, where they change only the candidate.
Other entry points: `python -m emva.eval.label_study`, `python -m emva.eval.ground_truth_reference`
(reads ground truth), `python -m emva.eval.hardening [--data data/v2 data/v1] [--out FILE] [--no-cache]`
(Phase 4; the headline number is its rolling-origin mean on v2, ADR 0013),
`python -m emva.context.agent --data data/v2 [--ids FILE] [--limit N] [--dry-run]`
(judgments CSV + committed cache in `data/v2/context/`), `python scripts/run_context_agent.py [--probe N]`,
`python -m emva.eval.context_harness {sample,evaluate}` (reads ground truth).

**Generator.**
```sh
python scripts/generate_data_v1.py --seed 20260924 --as-of 2026-09-24 --out DIR --n 10000   # ~17 s
python scripts/compare_to_v1.py --gen DIR [--seeds 1,2,3] [--md out.md]                    # fidelity vs data/v1
# generator v2 (merged):
python scripts/generate_data_v2.py [--no-paraphrase]        # writes data/v2/
python scripts/paraphrase_templates.py --check|--populate|--probe
python scripts/v2_checks.py --data data/v2 --ref data/v1
```

**Dataset converter (Phase 9).** A confirmed TOML mapping (`mappings/*.toml`) turns 1-3 foreign CSVs into the five
training files plus `dataset.json`; drafts are refused (ADR 0021).
```sh
python -m emva.ingest draft --raw DIR --out FILE.toml [--name NAME] [--cache FILE] [--as-of DATE] [--test-from DATE]  # Haiku draft, needs .env
python -m emva.ingest convert --mapping FILE.toml --raw DIR --out DATASET_DIR    # then python -m emva --data DATASET_DIR ...
python -m emva.ingest score --mapping FILE.toml --raw FILE_OR_DIR --run RUN_DIR [--data DATASET_DIR] [--out FILE]
```
A dataset directory with `dataset.json` uses its own `as_of` / `test_from` in the pipeline and the report
(`emva.dataset_meta.dataset_dates`; malformed = refused, ADR 0020) and gets no frozen baseline row (ADR 0022);
without it the constants apply, so data/v1 and data/v2 are unchanged.

**LLM access (ADR 0010).** `.env` at the repo root (gitignored, never committed, never printed) holds
`ANTHROPIC_API_KEY` (or, where a host strips that name, `EMVA_ANTHROPIC_API_KEY`; `emva.env.ENV_ALIASES`) and `ANTHROPIC_WORKSPACE_ID`. The key is not workspace-scoped, so every client must
send the `anthropic-workspace-id` header. Model for all LLM calls: `claude-haiku-4-5-20251001`.
Responses are cached on disk and the caches are committed; never fabricate LLM output.

## Report conventions (standard since Phase 0, ADR 0002)

- Revenue = **recorded** deal value of won test leads (blank = 0), never the model's imputed value.
  The baseline script's own `top20_revenue` (0.795, imputes blanks) is printed in the notes only.
- Top-20% capture ranks with numpy's default argsort, like the baseline; when scores tie at the cut the
  report adds a footnote with the **tie-averaged** value. Tests assert tie-averaged values (status quo
  wins 0.397, revenue 0.454) so they pass on any CPU.
- `weights.csv` is compared **canonically** (same rows, values equal to 3 dp, order ignored); byte
  identity is required only between the emva run and the baseline run on the same machine.
- CIs: percentile bootstrap, **1000 resamples, seed 0**; paired AUC differences on shared resamples;
  coefficient CIs from 1,000 refits.

## Repo map (detail in `docs/ARCHITECTURE.md`)

```
baseline/            frozen POC: emva_score.py, context_agent.py, weights.csv, business_brief.md. Never edit.
data/v1/             synthetic data + ground truth (eval/generator only). data/v2/ arrives with Phase 5.
data/external/       gitignored: raw public downloads (raw/) and datasets converted from them (Phase 9)
emva/constants.py    every constant (AS_OF, TEST_FROM, HORIZON_DAYS, CATS reference levels, patterns)
emva/dataset_meta.py per-dataset as_of / test_from from <data>/dataset.json, else the constants (ADR 0020)
emva/io.py           load CSVs, normalise CRM stages, join enrichment, won_at/first_contact_at; bot + duplicate cleaning
emva/labels.py       LabelMode/LabelConfig, legacy label, label_source, won_within_h, horizon_label, maturity, split
emva/features.py     FeatureSet; legacy and v2 bucketed features (indicators, name enrichment, boilerplate text)
emva/feature_spec.py FeatureSpec per feature set (levels, featuriser, design, fixed residual sd), like LabelConfig
emva/generic.py      generic feature set (Phase 10): declared extras (extra_features.csv), train-only encoder, ADR 0024
emva/boilerplate.py  boilerplate snippets + token-set Jaccard detector (v2 text=copy_paste)
emva/design.py       legacy data-driven dummies; v2 fixed_design from V2_LEVELS (unknown level raises)
emva/model.py        L2 logistic regression, predict, scorecard (weights.csv)
emva/value.py        ridge deal-value model (fixed-schema design for v2), expected value, value_at_close, realised revenue
emva/value_transform.py  ValueTransform: cap / log-sqrt compression / floor / tiers, fitted on training leads (plan 3.1)
emva/troas.py        tROAS eligibility calculator, python -m emva.troas (plan 3.4)
emva/pipeline.py     run(): load -> clean -> label -> features -> design -> fit -> value -> summary
emva/cli.py          label flags shared by python -m emva and the report;  emva/__main__.py: the CLI
emva/context/        card.py (lead card), contract.py (enum judgments), agent.py (SDK runtime), cache.py, features.py (ctx_ dummies)
emva/ingest/         Phase 9 converter: mapping.py (TOML schema), profile.py (redacted column profiles), draft.py (Haiku
                     draft, ADR 0021), convert.py (deterministic conversion, coverage, convert_leads); python -m emva.ingest
emva/eval/           bootstrap, metrics, regression, status_quo, report, label_study, ground_truth_reference,
                     collinearity, feature_selection, phase2_study, value_report (value transforms, click-ID coverage),
                     context_harness; Phase 4: hardening (entry point), rolling, subsampling, ceiling (reads
                     ground truth), calibration_decay, regularisation, interactions; Phase 10: generic_report
app/                 Keel, the hosted internal tool (Phase 8; docs/ARCHITECTURE.md section 8, reports/app.md):
                     storage, validation, training + job, results, scoring, ingest (Map & convert, Phase 9), auth (pure)
                     and main/ui/views/theme (Streamlit)
mappings/            committed dataset mappings (olist_funnel, crm_opportunities, hotel_bookings; hand-written)
scripts/             check_baseline (make baseline), run_experiments, generate_data_v1, ground_truth_report, compare_to_v1
tests/               pytest, one file per module + integration; conftest runs v1 in legacy and horizon mode
reports/             one write-up per task; docs/ context layer (this set of files); docs/platform_contract.md (upload design)
```

**Hosted app.** `streamlit run app/main.py` (Keel) behind a shared `APP_PASSWORD`, data under `DATA_DIR`.
It trains with the same CLI (`python -m emva` then `python -m emva.eval.report`, via `python -m app.job`), scores
single leads through `emva.scoring` and computes its charts with `emva.eval`; it never reads ground truth
(uploads named `ground_truth*` are refused by name). Tests: `tests/test_app_*.py` (~35 s, one shared trained run).

## Engineering standards

- Type hints on every function; a docstring on every module and public function saying what it
  needs and returns. Constants live in `emva/constants.py` (or next to their one user, named for
  where they come from).
- No dead code, no commented-out code, no speculative options (the generator's v2 `Config` fields
  raised `NotImplementedError` until implemented).
- Never swallow exceptions silently: log and re-raise, or log each retry (see `emva/context/agent.py`).
  Refuse ambiguous input loudly (`io.load` rejects two Won rows; `label_source` rejects unknown stages).
- Tests alongside the code in the same commit; hand-built fixtures at the boundaries (e.g. 89 / 89.99 /
  90 idle days, won exactly at H vs H + 1 s).
- Pandas 3 gotchas already hit: missing values in object/str columns may be `None`, `NaN` or `pd.NA`
  (test with a helper like the generator's `_present`, not `is None`); boolean columns with gaps are
  object dtype, so compare `== True` (NaN-safe, `# noqa: E712`) rather than truthiness; `sort_values`
  and `argsort` use an unstable sort, so tied rows order differently on arm64 vs x86 (why weights are
  compared canonically and top-k has a tie-averaged figure).
- The path has spaces: quote it in shell, Makefile (`"$(PY)"`) and subprocess calls.

## Status (2026-09-26)

| phase | state | branch | report |
|---|---|---|---|
| 0 freeze and instrument | merged (15f90fc) | `phase0-freeze-instrument` | `reports/phase0.md` |
| 5.1 generator v1 rewrite | merged (ed3f54b) | `phase5-generator-v1` | `reports/phase5-1.md` |
| 1 labels | merged (92168cf) | `phase1-labels` | `reports/phase1.md` |
| 2 features and leakage | merged (c1b465f) | `phase2-features` | `reports/phase2.md` |
| 5.2-5.8 generator v2 | merged (01c56f4) | `phase5-generator-v2` | `reports/phase5.md` |
| 3 value layer | merged (a13491d) | `phase3-value` | `reports/phase3.md` |
| 4 evaluation hardening | merged (3e374e8) | `phase4-eval-hardening` | `reports/phase4.md` |
| 6 context agent v2 | merged (9e3dfae) | `phase6-context-agent` | `reports/phase6.md` |
| 7 production readiness doc | merged (fa9bc4d) | `phase7-production-doc` | `reports/production.md`, ADR 0017 (Proposed) |
| 8 hosted app (Keel) | merged (c3da2b0) | `app-ui` | `reports/app.md`, ADR 0018/0019 (Proposed) |
| 9 dataset converter + per-dataset dates + Keel Map & convert | merged (be2f674) | `claude/epic-edison-onl83z` | `reports/phase9.md`, ADRs 0020-0023 (Proposed) |
| 10 generic feature set + Keel support + live mapping-draft check | merged (febc0a3) | `phase10-generic-features` | `reports/phase10.md`, ADRs 0024-0025 (Proposed) |
| 10 (a) generic feature set, (b) Keel support | on branch `phase10-generic-features`, not merged | `phase10-generic-features` | `reports/phase10.md`, ADR 0024 (Proposed) |

## Do not

- Read, import or hard-code ground truth outside `emva/eval/` and the generator side.
- Edit `baseline/`, or change the frozen test sets.
- Tune H, flags, thresholds or features to hit an acceptance number; report the fail instead.
- Quote v1 numbers externally (claims policy). Label any number "on simulated data".
- Commit `.env`, API keys, `runs/`, or regenerated data outside `data/v1/` and `data/v2/`.
- Run bare `git stash` / `git stash pop` in a shared checkout or worktree (the stash stack is shared
  across worktrees); use a WIP commit instead.
- Switch branches in the main checkout while another agent is working there; use a worktree.
- Push to `main`, open PRs, or merge your own branch (the orchestrator does).
- Fake LLM responses or use a model other than `claude-haiku-4-5-20251001` without an ADR.
