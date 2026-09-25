# 0008. Persona must always be readable from text; assigned only to non-vague, non-copy-paste texts

- Status: Accepted, implementation in progress
- Date: 2026-09-25
- Source: orchestrator ruling on Phase 5.2-5.8, branch `phase5-generator-v2` (in progress). The report
  on that branch at ae5b7c6 (`reports/phase5.md`, "Deviations and judgement calls") still describes the
  earlier behaviour and is being updated.

## Context

Plan 5.7 plants a context-only signal: a hidden persona (job seeker, agency, nonprofit, competitor) with
effect −1.0, readable only from the meaning of the free text and invisible to the regex. Phase 6's
acceptance is that the context agent recovers more than half of this planted gap. As first implemented,
personas were drawn for about 8% of genuine-human leads and a persona sentence was written only onto
non-vague answers, so about 28% of persona leads had no textual trace at all. Those leads carry the −1.0
effect but no reader, human or LLM, could detect it, which caps what Phase 6 can recover for reasons
unrelated to the agent.

## Decision

A persona must always be readable from the lead's text. Persona is assigned only to leads whose text is
neither vague nor copy-paste, and every persona lead carries a persona sentence. Persona paraphrases stay
filtered by `persona_safe` so they never give the regex a category keyword (raw Haiku output stays in the
cache).

## Consequences

- The planted context-only gap becomes fully recoverable in principle, so a Phase 6 shortfall is
  attributable to the agent, not to invisible labels.
- Persona is correlated with the true text category by construction (never vague or copy-paste). Phase 6
  must report persona effects conditional on the text category, not only marginally.
- The persona counts and effect in `reports/phase5.md` (7.9% of leads, outcome LR −0.94) must be
  re-measured after the change; `data/v2/` is regenerated. Verify the new figures on merge.
