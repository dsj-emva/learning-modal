# 0005. Mature still-open non-wins are 0 under a fixed horizon; stalled stays censored (R2)

- Status: Accepted
- Date: 2026-09-25
- Source: `reports/phase1.md`, "Orchestrator decisions (review of Phase 1)" R2; `emva/labels.py` module docstring

## Context

Under `won_within_h`, a mature lead that is still in an open stage at AS_OF and was not Won by
`created_at + H` demonstrably did not win within H. A stalled lead (open stage idle ≥ 90 days) is in the
same position, but plan 1.4 says stalled deals are censored, not lost. The review asked whether the two
should be treated alike.

## Decision

Keep the asymmetry deliberately. Mature open non-wins are 0. Stalled leads stay censored (NaN) by
default, because stalling is read as sales neglect rather than a buyer decision. `--stalled-as-lost`
shows the effect of treating them like any other mature non-win. No code change; the rule is documented
in the `emva/labels.py` docstring.

## Consequences

- On v1 the horizon test set keeps 30 still-open mature leads as 0s.
- `--stalled-as-lost` adds 513 training zeros, moves the 201-1000 labelled win rate from 33.5% to 29.2%,
  and leaves AUC unchanged.
- Censoring only ever turns a 0 into NaN; the two censoring groups (ghosted-at-H, stalled) are controlled
  independently of `label_source` precedence.
