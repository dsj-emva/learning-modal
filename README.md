# EMVA lead scoring

Scores inbound B2B leads (logistic-regression formula × expected deal value) so the value can be
sent to ad platforms. `REBUILD_PLAN.md` is the roadmap; `reports/` has one write-up per task.

```
baseline/     frozen POC (emva_score.py, context_agent.py, weights.csv). Never edit.
emva/         the package: io, labels, features, design, model, value, pipeline, context/, eval/
scripts/      check_baseline.py (make baseline), run_experiments.py (make experiment)
tests/        pytest unit + integration tests
data/v1/      synthetic data (ground_truth* files are for emva/eval/ only)
data/v2/      generator v2 output (Phase 5): same schema, messier text, dropout, missingness, hidden persona
```

## Setup

Python 3.11.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```sh
make baseline               # frozen baseline and emva/ (--label-mode legacy --feature-set legacy) must reproduce AUC 0.814, wins 0.571, revenue 0.795
make report                 # standard table: baseline vs candidate vs status quo, legacy and horizon test sets
make test                   # pytest + compile check
make experiment NAME=horizon    # pipeline outputs + report into runs/horizon/ (see scripts/run_experiments.py)

.venv/bin/python -m emva --data data/v1 --out OUT_DIR    # writes weights.csv, scores.csv
.venv/bin/python -m emva.eval.label_study                # Phase 1: label flags, size weights with bootstrap CIs
.venv/bin/python -m emva.eval.ground_truth_reference     # true close rates by size (reads ground truth; eval only)
.venv/bin/python scripts/generate_data_v1.py --seed 20260924 --out DIR && .venv/bin/python scripts/compare_to_v1.py --gen DIR   # regenerate v1-like data, compare with data/v1
```

### Labels

The default label is fixed-horizon: `won_within_h` = Won within `--horizon-days` (120) of
`created_at`; leads younger than that and not Won are unlabelled, and only mature leads
(`created_at + H <= 2026-09-24`) are used to train and evaluate. Leads still New at H are
excluded (`--include-ghosted` counts them as 0) and stalled open deals are censored
(`--stalled-as-lost` counts them as 0). `--label-mode legacy --feature-set legacy` restores the
baseline's labels and features and reproduces `baseline/` byte for byte. In horizon mode `scores.csv` also carries
`won_within_h`, `label_source` (won / crm_lost / stalled / ghosted / open) and `matured_at`.
Definitions: `emva/labels.py`.

### Features

The default `--feature-set v2` (plan Phase 2, `emva/features.py`): `session_missing` and
`enrichment_missing` indicators for absent session telemetry and absent enrichment (the affected
features stay at their reference level), `band=missing`, company-name matching when the email
domain misses companies.csv, similarity-based boilerplate detection, c_budget / c_timeline /
ip_type / edits_1_4 dropped (2.6), and the deal-value residual sd estimated from training. The
design's columns are fixed by the spec. `--feature-set legacy` is the baseline's. Evidence and
checks: `python -m emva.eval.phase2_study`, `python -m emva.eval.feature_selection`,
`python -m emva.eval.collinearity` (exits 1 on |corr| > 0.95; `make report` runs it too and exits 1
on data/v1, where `session_missing` equals `channel=meta_leadads`; see reports/phase2.md).

## Synthetic data v2

`scripts/generate_data_v2.py` is the v1 generator with the Phase 5.2-5.8 options on: Claude Haiku paraphrases
of every free-text template (cached in `scripts/paraphrase_cache.json`), typos, DE/FR/NL language mixing, a wide
boilerplate pool, 30% enrichment-domain dropout with noisy typed company names, 15% consent-declined telemetry,
two interaction effects, ghosting that follows the status-quo tier, a hidden context-only persona and 2% fast
humans. `data/v2/ground_truth.md` lists every mechanism with its planted and measured size; `reports/phase5.md`
has the acceptance numbers.

```sh
.venv/bin/python scripts/generate_data_v2.py                   # rewrites data/v2 (needs the complete paraphrase cache)
.venv/bin/python scripts/generate_data_v2.py --no-paraphrase --out DIR
.venv/bin/python scripts/paraphrase_templates.py --check       # which templates are missing from the cache
.venv/bin/python scripts/paraphrase_templates.py --populate    # fill them (ANTHROPIC_API_KEY, optional
                                                               # ANTHROPIC_WORKSPACE_ID, from env or repo-root .env)
.venv/bin/python scripts/v2_checks.py --data data/v2 --ref data/v1                   # regex, bot rule, shares, ghosting
.venv/bin/python scripts/compare_to_v1.py --gen data/v2 --gen-label v2 --md OUT.md   # fidelity tables v2 vs v1
```

With every option off the generator's v1 output is byte-identical (pinned in `tests/test_generate_data_v2.py`).

The context agent (`python -m emva.context.agent`, needs `ANTHROPIC_API_KEY`) writes a CSV that
`python -m emva --context FILE` uses for the context-layer ablation.
