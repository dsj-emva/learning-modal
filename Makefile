# EMVA scoring. Paths contain spaces, so every path is quoted.
PY ?= .venv/bin/python
DATA ?= data/v1

.PHONY: baseline report report-full test compile experiment app docker

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
	"$(PY)" -m compileall -q emva scripts baseline tests app

## experiment NAME=...: run a named experiment into runs/NAME/
experiment:
	@test -n "$(NAME)" || (echo "usage: make experiment NAME=<name>" && exit 1)
	"$(PY)" scripts/run_experiments.py "$(NAME)" --data "$(DATA)"

## app: run the Streamlit app locally on :8501, data under DATA_DIR (default runs/app)
DATA_DIR ?= runs/app
app: export DATA_DIR := $(DATA_DIR)
app:
	mkdir -p "$(DATA_DIR)"
	"$(PY)" -m streamlit run app/main.py --server.port 8501

## docker APP_PASSWORD=...: build learning-modal:local and run it on :8080, data in runs/docker-data
docker:
	@test -n "$(APP_PASSWORD)" || (echo "usage: make docker APP_PASSWORD=<password>" && exit 1)
	docker build -t learning-modal:local .
	mkdir -p runs/docker-data
	docker run --rm -p 8080:8080 -e PORT=8080 -e APP_PASSWORD="$(APP_PASSWORD)" -v "$(CURDIR)/runs/docker-data:/data" learning-modal:local
