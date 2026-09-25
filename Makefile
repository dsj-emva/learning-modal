# EMVA scoring. Paths contain spaces, so every path is quoted.
PY ?= .venv/bin/python
DATA ?= data/v1

.PHONY: baseline report test compile experiment

## baseline: frozen baseline and emva/ must reproduce the published metrics and weights
baseline:
	"$(PY)" scripts/check_baseline.py --data "$(DATA)"

## report: standard table (baseline vs candidate vs status quo) on the frozen test set
report:
	"$(PY)" -m emva.eval.report --data "$(DATA)"

## test: unit + integration tests, then a compile check
test:
	"$(PY)" -m pytest -q
	$(MAKE) compile

compile:
	"$(PY)" -m compileall -q emva scripts baseline tests

## experiment NAME=...: run a named experiment into runs/NAME/
experiment:
	@test -n "$(NAME)" || (echo "usage: make experiment NAME=<name>" && exit 1)
	"$(PY)" scripts/run_experiments.py "$(NAME)" --data "$(DATA)"
