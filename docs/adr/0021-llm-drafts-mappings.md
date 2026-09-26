# 0021. The LLM only drafts a dataset mapping, from redacted column profiles; a person confirms it and conversion is deterministic

- Status: Proposed (item 2 amended by ADR 0025: the reply names `primary_file` and `joins`)
- Date: 2026-09-26
- Source: Phase 9 (a), `reports/phase9.md`; `emva/ingest/draft.py`, `emva/ingest/profile.py`, `emva/ingest/mapping.py`,
  `app/ingest.py`; branch `claude/epic-edison-onl83z`

## Context

Phase 9 converts foreign CSVs (a CRM export, a Kaggle dataset) into EMVA's five-file format with a TOML mapping per
data source. Writing a mapping by hand is slow: a user has to match every column to the lead schema and define the
outcome. A model can propose it quickly. But a wrong outcome mapping silently corrupts every label, and a row sent
to an external API can hold personal data (emails, names, phone numbers, IPs). The context agent (ADR 0010, ADR
0014) already set the rules for LLM calls in this repo: Haiku 4.5, the workspace header, structured output, and a
committed cache keyed so that a prompt or model change is a miss.

## Decision

1. **Drafts only.** `emva.ingest.draft.draft_mapping` returns a `DatasetMapping` with `outcome_confirmed = false`.
   The model never converts rows, fills values or scores. `emva.ingest.convert` (and `python -m emva.ingest convert`,
   and Keel's Convert button) refuses a mapping unless `outcome_confirmed = true`, and only a person sets it: the
   TOML flag, or Keel's "Outcome mapping confirmed" checkbox (`app.ingest.confirm`).
2. **Structured output with a closed enum.** Per profiled column the reply gives a `target` from `DRAFT_TARGETS`
   (a lead role such as `lead.created_at`, a schema column, `value_map`, or `ignore`), a `reason` and a
   `confidence` (high / medium / low). The reply also names the sources with their join columns and the outcome
   (kind `stage` / `won_flag` / `presence`, column, stage map or won / lost values). `check_reply` refuses anything
   the schema cannot express, such as unknown files or columns or a missing `created_at`, as a `ContractError`
   (parse error: cached, never retried).
3. **Not drafted by the model:**
   - **Derived expressions** (`date_from_parts`, `minus_days`, `product`, `sum`): lead roles are plain column
     references, and a person adds expressions while reviewing.
   - **Dates:** `as_of` / `test_from` are passed in, or computed from the profile of the column drafted as
     `created_at` (`default_dates`) and then reviewed like everything else. The split stays a human decision
     frozen in the mapping (ADR 0020).
4. **Privacy: profiles, never rows.** The model sees only `emva.ingest.profile` output (`render_profiles`): file and
   column names, the inferred type, share missing, distinct count, the min / max of dates and numbers, and up to 10
   examples for columns with at most 20 distinct values. Redaction is applied to every example, min and max:
   emails, phone-like digit runs and IP addresses become `<email>`, `<phone>`, `<ip>`, and a column whose name
   suggests a person's name gets only `<name>`. Redaction errs towards over-redacting. The Keel page shows this
   profile table before the button is pressed.
5. **Runtime and cache as ADR 0010.** The model is `claude-haiku-4-5-20251001` (`emva.context.agent.MODEL_ID`),
   called through `emva.context.agent.make_client` (the `anthropic-workspace-id` header) at temperature 0.
   Transport errors are retried with logged backoff, and other API errors are raised. Replies go in an
   `emva.context.cache.ReplyCache` keyed by `cache_key(profiles_hash, DRAFT_BRIEF_HASH, PROMPT_VERSION,
   prompt_fingerprint(), MODEL_ID)`, where the fingerprint covers the system prompt, response schema and max tokens.
   `ok` and parse-error replies are cached, and transport errors never are. The CLI cache defaults to
   `.draft_cache.json` next to the mapping and is committed with it. Keel's cache is
   `DATA_DIR/ingest/draft_cache.json`. A cached draft needs no client. Without a key, Keel says so and offers the
   table to fill by hand. Nothing fakes a reply.
6. **The model call runs in the server process.** The training job keeps its filtered environment without
   `ANTHROPIC_*` (ADR 0019).

## Consequences

- A wrong draft costs review time, not a wrong dataset. The outcome and dates are always a person's decision, and
  every field row keeps its reason and confidence in the TOML, where low-confidence rows are highlighted in Keel.
- The three committed mappings (`mappings/*.toml`) were written by hand because no API key was available in the
  Phase 9 environment, and their headers say so. The live draft run and cache-hit check were
  done in the Phase 10 live draft check (ADR 0025, `reports/phase10.md`).
- Value maps can be drafted, but only onto schema columns with their listed options. A draft value outside the
  options is refused at conversion (R25), not dropped.
- ADR 0017 decision 5 (ingestion adapters per CRM platform) becomes a set of mappings: a mapping written once per
  CRM platform or export layout is reused through Keel's saved and built-in mapping picker, with no model call.
- Changing the prompt, schema or model invalidates every cached draft. Using another model needs a new ADR.
