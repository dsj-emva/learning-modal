# 0014. Context contract: enum judgments, a four-field lead card, stamped and cached output

- Status: Accepted (Phase 6 review; persona pooling amended by ADR 0016)
- Date: 2026-09-25
- Source: Phase 6 (`phase6-context-agent`, `reports/phase6.md`); plan 6.1 to 6.4

## Context

The Phase 0 agent returned one 0-1 `context_score` parsed with a regex from free text, and its lead card
included enrichment bands, ad spend, CRM and hiring: features the formula already uses. A single score
cannot be audited or weighted per reason, and a card that repeats formula inputs makes the context
feature correlated with the formula score, so its "uplift" is not evidence of new information. Parse
failures and network failures were retried together, and outputs carried no record of which brief,
prompt or model produced them.

## Decision

- **Contract** (`emva/context/contract.py`): the model returns exactly five enum judgments,
  `is_real_business` (yes / no / unclear), `persona` (buyer / vendor / student / job_seeker / competitor /
  nonprofit / agency_pitching / unclear), `problem_specificity` (specific / vague / boilerplate /
  unrelated), `urgency` (now / this_quarter / researching / none), `brief_fit` (strong / partial / weak /
  none), plus a `reason` of at most 30 words. No numeric score. The JSON schema is sent as the API's
  structured-output format; `validate` checks keys, enums and the word limit.
- **Lead card** (`emva/context/card.py`): free text, typed company name, job title string, email domain
  string. Nothing else: no enrichment, spend, CRM, hiring, bucketed form answers (budget, timeline,
  company size, country), behavioural telemetry, CRM stage or outcome. The brief is the system prompt.
- **Errors**: a reply that breaks the contract is a *parse error* (recorded, cached, never retried); a
  connection failure, timeout, 408/409/429 or 5xx is a *transport error* (retried with backoff 1, 2, 4,
  8 s, never cached); other API errors (400/401/403/404) are configuration errors and raise.
- **Stamp and cache**: every output row carries `(brief_hash, prompt_version, model_id)` and the card
  hash; the cache key is (card hash, brief hash, prompt version, prompt fingerprint, model id), the
  fingerprint hashing the system template, response schema and max tokens. A judgments file must carry
  one stamp. A brief edit gives a new `brief_hash`, every lead becomes a miss (`--dry-run` reports it
  without calling), and the context weights are refit on the new judgments; weights never carry over
  between stamps.
- **Features**: each judgment is a categorical `ctx_<judgment>` with the declared levels above and
  reference levels yes / buyer / specific / none / partial (ADR 0009 fixed schema). Persona enters the
  design pooled as `ctx_persona_group` (buyer / non_buyer / unclear; ADR 0016), giving 13 columns. Rows
  without an ok judgment have no context and are left out of both models in a comparison.
- Adding or renaming a level is a contract change: bump `PROMPT_VERSION` and the feature levels together.

## Consequences

- `python -m emva --context` accepts the new judgments file; the legacy `context_score` file still works
  so the baseline's context mode stays byte-identical.
- On data/v2 the judgments correlate weakly with the formula logit (largest single column |r| 0.33,
  combined context score r 0.14), and Haiku reads the planted persona well, but the planted persona's
  AUC gap is too small to measure at n = 1,000; acceptance is coefficient-level (ADR 0016).
- The card is deliberately thin. If a later phase wants the agent to use enrichment it must show the
  added correlation with the formula is acceptable and supersede this ADR.
