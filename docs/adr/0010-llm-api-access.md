# 0010. API access: key in gitignored .env, workspace header required, Haiku 4.5, committed caches, no fake LLM output

- Status: Accepted
- Date: 2026-09-25
- Source: orchestrator ruling during Phase 5.2-5.8 (`phase5-generator-v2:reports/phase5.md`, "Paraphrase
  (5.2 i): done with the real API"; `scripts/paraphrase_templates.py`); commit 869805a ("Ignore .env");
  `emva/context/agent.py` (`MODEL`)

## Context

Two parts of the plan call Claude: generator v2 paraphrases free-text templates (5.2), and the context
agent reads leads (Phase 6; Phase 0 version in `emva/context/agent.py`). An API key became available
during Phase 5. The key is not workspace-scoped, so the workspace has to be named on every request. Results must be reproducible offline and in review, and synthetic data built on LLM output must
not be passed off with placeholder text.

## Decision

- Credentials: `ANTHROPIC_API_KEY` and `ANTHROPIC_WORKSPACE_ID` live in `.env` at the repo root, which is
  gitignored. They are never printed, logged or committed. Scripts read the environment first, then fall back
  to the first `.env` found walking up from the script (as `scripts/paraphrase_templates.py` does).
- Every client sends the `anthropic-workspace-id` header with the workspace ID.
- Model for all LLM calls: `claude-haiku-4-5-20251001`.
- Every LLM result is cached on disk, keyed by a hash that includes the model id and prompt version (the
  paraphrase cache: sha256 of template, model id, prompt version), and the cache is committed so reruns
  reproduce without the API. Changing the model or prompt makes every entry a miss.
- No fake LLM output. If the API is unavailable the code fails loudly (`MissingParaphraseError`) or the
  feature is switched off explicitly (`generate_data_v2.py --no-paraphrase`); nothing mocks a response
  into data or reports.

## Consequences

- `scripts/paraphrase_cache.json` holds 62 templates × 5 paraphrases; `data/v2/` is generated from it.
- `emva/context/agent.py` (moved unchanged from the baseline) uses `requests` with `x-api-key` only and
  does not yet send the workspace header; its cache key is sha256(system prompt + lead card) without the
  model id. Phase 6 (Anthropic SDK, `(brief_hash, prompt_version, model_id)` stamped per row) must bring
  it in line with this ADR.
- Using another model needs a new ADR.
