# 0023. Conversion rulings of Phase 9 and its fix round, in one place

- Status: Proposed
- Date: 2026-09-26
- Source: `reports/phase9.md` "Orchestrator decisions" R22-R40 (R22-R33 from the phase, R34-R40 from the review fix
  round); `emva/ingest/`, `emva/eval/evaluation_only.py`, `emva/eval/report.py`, `app/ingest.py`,
  `app/validation.py`, `app/views/upload.py`; branch `claude/epic-edison-onl83z`

## Context

CLAUDE.md requires an ADR for every orchestrator ruling. Phase 9 made twelve rulings (R22-R33) while building the
dataset converter and Keel's Map & convert; three already have their own ADR (0020 per-dataset dates, 0021 the LLM
only drafts, 0022 results without a baseline). The two-axis review (standards, spec) found no blockers but seven
should-fix items; the fix round ruled on them (R34-R40). This ADR records the rest of the set so each ruling has a
written home; where a ruling has its own ADR, this one points to it.

## Decision

Phase 9 (build):

1. **R22** (ADR 0022 decision 4): Keel's store record is `keel_meta.json`; `dataset.json` means the converter's
   metadata only, and `init_root` migrates legacy records.
2. **R23** (ADR 0020): a `dataset.json` without valid `as_of` / `test_from` is refused, never read as "no dates".
3. **R24**: required fields that are not mapped get placeholders, counted as unfilled in the coverage table:
   email `<lead_id>@unmapped.invalid`, `form_variant` "A", `lead_id` `<name>-000001`, ...
4. **R25**: values outside the form options, non-booleans and non-numbers in typed columns are refused with the
   values listed, never dropped; a value map translates them.
5. **R26**: `companies.csv` and `people.csv` are written header-only (company and person columns are not mapped).
6. **R27** (Olist): an MQL without a closed deal is Lost at as_of; closed-deal attributes are ignored (leakage).
7. **R28** (hotel): a kept booking is Won at arrival, a cancellation Lost at `reservation_status_date`; H is not
   changed (ground rule 5).
8. **R29**: the training job runs the standard report on converted datasets without status-quo rules.
9. **R30**: source-format scoring (`convert_leads`, `python -m emva.ingest score`, the Score page's mode).
10. **R31**: developed on `claude/epic-edison-onl83z` in a Linux container, not `phase9-ingest` / Docker.
11. **R32**: the Dockerfile ships `mappings/`.
12. **R33**: the scope split into (a) library, CLI and dates and (b) Keel, both on one branch.

Fix round:

13. **R34** (ADR 0022 decision 5): `emva.eval.report.baseline_or_note` omits the baseline row only when the
    dataset carries `dataset.json`; on data/v1, data/v2 and v1-format uploads a failing baseline script raises, as
    before. The app page keeps logging and noting a failure (ADR 0022 decision 1).
14. **R35**: the converter never reads an evaluation-only file. `emva.eval.evaluation_only.is_evaluation_only` holds
    the name check inside `emva/eval/` (so the grep rule stays clean); the draft CLI refuses ground truth and more
    than 3 CSVs before reading anything, `Source.file` must be a plain `.csv` file name (no directory part, not
    ground truth), and `frames_from_bytes` and the score CLI refuse ground-truth names.
15. **R36**: `won_at` is optional (the spec's "optional won_at"). A Won lead without one is dated at its
    `close_at`; a Won lead with neither is refused, naming example leads. The hotel mapping is unchanged.
16. **R37**: adjustments stay, but are visible. The convert CLI prints every count of derived or adjusted values
    (`DERIVED_COUNTS`: rows dropped without created_at, placeholder emails, Lost dated at as_of, non-positive deal
    values blanked, CRM ties separated, CRM times reordered); Keel's coverage card lists the non-zero ones under
    "Derived during conversion"; `dataset.json` keeps the first 10 reordered lead ids (`crm_times_reordered_ids`)
    and validation turns a reorder (an event before the previous one, e.g. a win before created_at) into a warning
    naming them.
17. **R38**: Map & convert takes an optional `status_quo_rules.json`, checked by the same validation as an upload
    in EMVA's format and saved with the dataset, so a converted dataset can have a status-quo row.
18. **R39**: person-name redaction decides on the last token of the column name (`account_manager` is a person,
    `company_name` and `lead_id` are not) and on the values (2-3 capitalised words with distinct last words become
    `<name>`). It errs towards redacting: a two-word company name may be hidden.
19. **R40**: the CRM opportunities licence is Apache 2.0, from the Kaggle dataset metadata; Olist is CC BY-NC-SA 4.0
    (non-commercial: no commercial or marketing use of the data or its numbers).

## Consequences

- Nothing about a conversion is silent any more: every value the converter invents or moves is counted where the
  user looks (CLI output, the coverage card, a validation warning with lead ids).
- A report on data/v1 or data/v2 fails loudly when the frozen baseline breaks, instead of losing its reference row.
- Mappings whose source has no won date convert without a workaround expression.
- Open follow-ups (not ruled here): a join key must have the same name in both files; `convert_leads` refuses a
  whole batch on one bad row (consistent with R25); the live API draft run is pending (no key in this environment);
  the CRM and hotel conversions exercise the plumbing only (0 of 39 signals with data).
