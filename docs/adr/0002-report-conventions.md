# 0002. Report conventions: recorded deal value, tie-averaged top-20%, canonical weights comparison

- Status: Accepted
- Date: 2026-09-25
- Source: `reports/phase0.md`, "Orchestrator decisions (review of Phase 0)" and "How the status quo is reconstructed"

## Context

Phase 0 had to make `make report` reproduce the status quo at AUC 0.639, top-20% wins 0.392, revenue
0.454, and make `make baseline` match the committed `weights.csv`. Three problems appeared:

1. The baseline's `top20_revenue` fills blank deal values on Won rows with the model's own
   `deal_value_hat`. That figure moves whenever the value model changes, so it cannot compare candidates;
   under it the status quo gets 0.449, not 0.454.
2. The status quo has 32 distinct values and 203 test leads tie at the top-20% cut. numpy's default
   (unstable) argsort gives 0.392 / 0.454; a stable sort gives 0.409 / 0.460; the tie-averaged expectation
   is 0.397 / 0.454. The unstable result depends on the CPU.
3. The regenerated `weights.csv` has identical values, but two rows tied at log-odds 0.16
   (`business_hours=wkday_9-18`, `channel=organic_direct`) come out swapped on arm64 under every
   numpy/pandas/scikit-learn combination tried. The committed file was probably made on x86.

## Decision

- Revenue in the standard report = **recorded** deal value of won test leads, blank = 0, for every model.
  The baseline script's own figure (0.795) is printed in the notes and still asserted by `make baseline`.
- Top-k keeps numpy's default argsort (as the baseline does); whenever scores tie at the cut the report
  prints a footnote with the tie-averaged value. Tests assert the sort-independent values (status quo
  AUC 0.639, tie-averaged wins 0.397, revenue 0.454) so they pass on any CPU.
- `weights.csv` is compared canonically (same row names and columns, values equal to 3 dp, row order
  ignored; `emva/eval/regression.py`), replacing "byte-identical" in plan 0.2. Byte identity is still
  required between the emva run and the baseline run on the same machine.
- Bootstrap CIs use 1000 resamples, seed 0 (`emva/eval/bootstrap.py`).
- The status-quo `country` rule reads the typed answer (`a_country`).

## Consequences

- Under recorded revenue the baseline scores 0.796 by p×value; reports quote this, not 0.795, in tables.
- Any new metric that ranks must print a tie footnote or use the tie-averaged value.
- pandas stays at 3.0.6: no pin makes the tie order portable.
