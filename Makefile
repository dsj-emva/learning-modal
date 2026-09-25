# EMVA scoring. Paths contain spaces, so every path is quoted.
PY ?= .venv/bin/python
DATA ?= data/v1

.PHONY: baseline report report-full test compile experiment

## baseline: frozen baseline and emva/ must reproduce the published metrics and weights
baseline:
	"$(PY)" scripts/check_baseline.py --data "$(DATA)"

## report: standard table (baseline vs candidate vs status quo) on the frozen test set
report:
	"$(PY)" -m emva.eval.report --data "$(DATA)"

## report-full: Phase 4 hardening (rolling origin, subsampling, ceiling, calibration decay, C sweep, interactions)
## plus the standard report, on data/v2 (headline) and data/v1, into reports/phase4.md (cached under runs/phase4/)
report-full:
	"$(PY)" -m emva.eval.hardening --data data/v2 data/v1 --out reports/phase4.md

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
