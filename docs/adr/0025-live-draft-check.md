# 0025. Live mapping-draft check: the reply names its primary file; every live reply is committed; semantic errors are left to review

- Status: Proposed
- Date: 2026-09-26
- Source: Phase 10 live draft check, `reports/phase10.md` "Live draft check" and D34-D37; `emva/env.py`,
  `emva/ingest/draft.py`, `mappings/drafts/`; branch `phase10-generic-features`

## Context

ADR 0021 introduced the LLM mapping draft, but until now it had only run against a fake client: no API key was
available in the Phase 9 and Phase 10 sessions. The first live run, on the three Kaggle sources, broke the
contract on two of the three sources (prompt v2: no `lead.created_at`). After guidance was added (v3), the third
source broke a different rule: the primary file carried a `join_on`, and the old encoding marked the primary by an
empty `join_on`. Replies that passed the contract still had semantic errors: extras only won leads have, and a
post-outcome date as `created_at`. The cloud session also does not pass `ANTHROPIC_API_KEY` through.

## Decision

1. **Key alias.** `emva.env.load_env_var` reads `EMVA_ANTHROPIC_API_KEY` (environment, then `.env`) when
   `ANTHROPIC_API_KEY` is unset. The real name wins. ADR 0010 otherwise unchanged.
2. **Explicit primary file** (amends ADR 0021 item 2). The reply schema has `primary_file` and `joins` (file + join
   column) in place of `sources` with an empty `join_on` marking the primary. The contract refuses a primary that
   is also joined, and a file joined twice. `PROMPT_VERSION` is `mapping-draft-v4`.
3. **Prompt guidance, then stop.** The prompt now says that the primary file holds every lead (never a won-only
   file), that such a file's columns are never features, and that `created_at` is the earliest date known at
   arrival and never a post-outcome date. Iteration stopped at the first version whose three replies pass the
   contract. The semantic errors that remain are the reviewer's to catch (ADR 0021: a wrong draft costs review
   time, not a wrong dataset). They are not prompted away source by source.
4. **Every live reply is committed.** The draft cache keeps superseded replies, contract failures included, as
   evidence. They are never served, because the key includes the prompt fingerprint.

## Consequences

- The three live drafts are committed next to the hand-written mappings (`mappings/drafts/`), unconfirmed. The hand
  mappings stay the ones in use.
- Known limits, each caught at review or refused by convert:
  - A draft cannot express a derived `created_at`, so on a source without a single creation-date column (hotel) the
    model picks a wrong column.
  - Won-only columns may still be proposed as extras (Olist). The missingness screen (ADR 0024) flags them after
    training.
  - `lost_without_close`, `drop_rows_without_created_at` and the dates are outside the reply schema.
- The draft's confidence did not separate right from wrong rows on the live run, so review cannot rely on it alone.
