# 0019. Hosted app (Keel): full standard table, UI naming, Railway deployment, trust model

- Status: Proposed
- Date: 2026-09-26
- Source: Phase 8 review of branch `app-ui` (rulings R18-R21, `reports/app.md`); `app/`, `Dockerfile`,
  `scripts/serve.sh`, `railway.toml`, `docs/deploy.md`

## Context

Phase 8 put a Streamlit app (Keel) in front of the pipeline so an EMVA colleague can read results, upload CRM
exports, train and score one lead. The two-axis review asked for rulings on what the results page must show, how
the UI and the docs name things, how the Railway service is configured, and what the app does and does not protect.

## Decision

1. **R18, full standard table.** The results page shows the standard table from `emva.eval.report`'s own
   functions (`headline_frame`, `paired_frame`): the frozen baseline, this run's model and the status quo, plus
   the paired AUC comparison against the baseline. Without `status_quo_rules.json` the status-quo row is omitted
   and the page says so. The app never re-derives a metric the report already computes.
2. **R19, naming.** The page is "Upload & train" (it does both) and the README follows the UI. Channel is
   derived from UTM source/medium by the training feature code, so the form asks for UTM fields and the points
   breakdown shows the derived channel.
3. **R20, Railway.** Railway config-as-code is deprecated: service settings are applied on the service and
   `railway.toml` is the in-repo record. Secrets are entered by the user in the dashboard. `RAILWAY_RUN_UID=0` is
   required for the volume; `scripts/serve.sh` chowns `/data` and drops to uid 1000.
4. **R21, pins.** `python:3.11-slim` stays a floating tag as the spec asked; requirements are exact pins.

## Trust model

- **One shared password** (`APP_PASSWORD`, constant-time comparison, 1 s delay after a failure); the app refuses to
  open when it is unset. No per-user accounts and no per-user audit: the registry records runs, not people.
- **One service, one process.** Training runs as a subprocess (`python -m app.job`) on the same machine, with a
  filtered environment (no `APP_PASSWORD`, no `ANTHROPIC_*`); runs interrupted by a restart are marked failed.
- **No ground truth.** The image excludes `data/**/ground_truth*` (`.dockerignore`); the app refuses uploads named
  `ground_truth*` by name without opening them; `grep -rn ground_truth app/` returns only that check.
- User-supplied text (file names, errors, dataset names) is escaped at the HTML boundary (`app.components.esc`).

## Consequences

- The results page runs the frozen baseline script on the run's dataset (a few seconds, cached per process).
- Anyone with the password can train, read every run and delete nothing; datasets and runs are removed on the volume.
- Moving to multi-user or multi-customer use needs real authentication and an audit log first (ADR 0017).
