# EMVA lead scoring

Scores inbound B2B leads (logistic-regression formula × expected deal value) so the value can be
sent to ad platforms. `REBUILD_PLAN.md` is the roadmap; `reports/` has one write-up per task.

```
baseline/     frozen POC (emva_score.py, context_agent.py, weights.csv). Never edit.
emva/         the package: io, labels, features, design, model, value, pipeline, context/, eval/
scripts/      check_baseline.py (make baseline), run_experiments.py (make experiment)
tests/        pytest unit + integration tests
data/v1/      synthetic data (ground_truth* files are for emva/eval/ only)
```

## Setup

Python 3.11.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```sh
make baseline               # frozen baseline and emva/ (--label-mode legacy) must reproduce AUC 0.814, wins 0.571, revenue 0.795
make report                 # standard table: baseline vs candidate vs status quo, legacy and horizon test sets
make test                   # pytest + compile check
make experiment NAME=horizon    # pipeline outputs + report into runs/horizon/ (see scripts/run_experiments.py)

.venv/bin/python -m emva --data data/v1 --out OUT_DIR    # writes weights.csv, scores.csv
.venv/bin/python -m emva.eval.label_study                # Phase 1: label flags, size weights with bootstrap CIs
.venv/bin/python -m emva.eval.ground_truth_reference     # true close rates by size (reads ground truth; eval only)
```

### Labels

The default label is fixed-horizon: `won_within_h` = Won within `--horizon-days` (120) of
`created_at`; leads younger than that and not Won are unlabelled, and only mature leads
(`created_at + H <= 2026-09-24`) are used to train and evaluate. Leads still New at H are
excluded (`--include-ghosted` counts them as 0) and stalled open deals are censored
(`--stalled-as-lost` counts them as 0). `--label-mode legacy` restores the baseline's labels
and reproduces `baseline/` byte for byte. In horizon mode `scores.csv` also carries
`won_within_h`, `label_source` (won / crm_lost / stalled / ghosted / open) and `matured_at`.
Definitions: `emva/labels.py`.

The context agent (`python -m emva.context.agent`, needs `ANTHROPIC_API_KEY`) writes a CSV that
`python -m emva --context FILE` uses for the context-layer ablation.
