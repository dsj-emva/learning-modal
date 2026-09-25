# 0013. Rolling-origin headline: training mature at the split, one-month mature test windows, mean AUC over sufficient splits

- Status: Proposed
- Date: 2026-09-25
- Source: plan 4.1; `reports/phase4.md`, section 4.1; `emva/eval/rolling.py`; branch `phase4-eval-hardening`

## Context

ADR 0006 made Phase 4's rolling-origin evaluation the headline number but did not define the split
for horizon labels. A horizon label is only known once `created_at + H` has passed, so "train on
everything before S" either leaks labels from after S (the frozen split trains on leads created before
2026-05-01 whose labels matured as late as 2026-08-29) or must drop immature rows. The review's
baseline spread (0.814 to 0.836 on v1) used legacy labels read at AS_OF and tested on every lead from
S to the end of the data; `emva.eval.rolling.LEGACY_TO_END` with legacy features reproduces it exactly.

With H = 120 and AS_OF = 2026-09-24 the last mature lead was created 2026-05-26, so the test windows
from 2026-06-01 on are empty, and training at S = 2026-02-01 has only the leads created before
2025-10-04 (218 on v1, 182 on v2).

## Decision

At split S in {2026-02-01, 03-01, 04-01, 05-01, 06-01, 07-01}, horizon H = 120:

- **train** = leads with a default horizon label (`emva.labels.HORIZON`, ghosted-at-H excluded, stalled
  censored) created before S **and mature at S**: `created_at + H <= S`;
- **test** = labelled leads created in `[S, S + 1 calendar month)` and mature at AS_OF
  (`created_at + H <= AS_OF`);
- model = the pipeline's defaults (v2 features, L2 LR C = 0.5), refitted at every split.

A split is **sufficient** when its test set has at least 100 labelled leads and at least 10 of each
class. Insufficient splits stay in every table with the reason; they are left out of the mean.

**The headline number** is the mean of the per-split AUCs over the sufficient splits, on data/v2
(the v1 figure is kept for continuity, never quoted externally), with a 95% percentile bootstrap CI from
resampling each split's test set independently (1000 resamples, seed 0 + split index), plus the
per-split range. Comparisons between models use the mean of per-split paired differences with the same
resampling (`rolling.paired_headline`).

The labels are computed once at AS_OF. For a lead mature at S the outcome and "ghosted at H" are
already fixed at S; only the stalled censoring reads the CRM state at AS_OF (no stage history at S is
available in the frame), which can drop a training row but never change a label.

Reported next to the headline: *relaxed* training (created before S, mature at AS_OF: the frozen split's
rule), legacy labels with the same one-month windows, and the baseline reproduction.

## Consequences

- Under H = 120 only four splits (2026-02-01 to 2026-05-01) are sufficient on v1 and v2; 2026-05-01 covers
  26 days. The 2026-06-01 and 2026-07-01 splits are listed as having no mature test lead.
- The strict rule is what a customer would see at S, and it costs AUC: early splits train on a few hundred
  leads (the v2 headline is about 0.04 below the relaxed rule; section 4.1 of `reports/phase4.md`).
- The frozen test sets of ADR 0006 are unchanged and still reported.
- Phase 7 and any external claim quote this number, with its CI and "on simulated data".
- Open: whether a later AS_OF (more mature months) or a shorter H should move the split dates; not
  changed here (ground rule 5: no tuning to an acceptance number).
