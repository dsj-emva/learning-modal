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
make baseline               # frozen baseline and emva/ must reproduce AUC 0.814, wins 0.571, revenue 0.795
make report                 # standard table: baseline vs candidate vs status quo on the frozen test set
make test                   # pytest + compile check
make experiment NAME=baseline   # pipeline outputs + report into runs/baseline/

.venv/bin/python -m emva --data data/v1 --out OUT_DIR    # writes weights.csv, scores.csv
```

The context agent (`python -m emva.context.agent`, needs `ANTHROPIC_API_KEY`) writes a CSV that
`python -m emva --context FILE` uses for the context-layer ablation.
