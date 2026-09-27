# Keel QA fix round 2 (M5, M7, minor and cosmetic)

This round fixes the rest of the findings from the 2026-09-27 hands-on QA of the live Keel, the items the first round
(`reports/keel-fixes.md`, ADR 0026) left open. The first round already fixed two items, which were not redone: the
breakdown lists no 0-point rows, and its parts add up. The rulings are in ADR 0027 (Proposed).

Three tasks ran in parallel, each on its own branch in a worktree. The orchestrator reviewed each branch on two axes
(standards and spec), made one fix round, merged them in the order state, source, polish, and resolved the conflicts
in `app/views/upload.py`, `app/ingest.py` and `tests/test_app_smoke.py`. Every Olist number here is **on public
data** (CC BY-NC-SA 4.0, non-commercial; not for quoting).

| issue | fix | branch, commits |
|---|---|---|
| M5: Map & convert work and the training options were lost on a page switch | Each upload slot's file is held in session state, one copy per slot (`app.uploads`), and shown as "using <file> … kept while you use other pages · Replace". The mapping under review is kept with its edits and rebuilt after the switch. Dataset, labels, features, the saved-mapping choice and the confirmation ticks are mirrored (`ui.restore` / `ui.keep`). Saving releases every held file. | `keel-fix2-state` f91dc4f, ecd295a |
| m1: stale check results after refused files | Validation compares the uploaded and the refused names with those it checked. A change asks the person to validate again; removing every file clears the results. | `keel-fix2-state` f91dc4f |
| m10: "(value map)" offered without a value map; a stale conversion stayed convertible | "(value map)" is offered only when a row shows it. An edit that cannot be applied makes the mapping incomplete. A conversion from another mapping, other rules or an incomplete mapping says "Convert again" and withdraws its check and save steps. | `keel-fix2-state` f91dc4f |
| m2: Generic vs v2 briefly showed the previous test set | Every section that depends on the test set is computed and drawn in one `st.empty` slot, so there are spinners until both sets of numbers are ready. | `keel-fix2-state` 975342b |
| m7: an unknown "Start from a lead" id left the previous score | The EMVA-form result belongs to its model and starting lead; changing either clears it. | `keel-fix2-state` a553213 |
| M7: the one-lead form and the CSV path disagreed on unlisted values | `convert_leads` returns `NewLeads(leads, refused)`. A lead whose value-map column holds an unlisted value is refused on its own, in plain words (leads, column, value, allowed values), and the rest are scored; Keel and `python -m emva.ingest score` behave the same. "other (any value not listed)" appears only for an extra's `other` level; Olist's raw `origin` value "other" shows as plain "other". An unseen extra category scores as `other` with a note naming the leads. | `keel-fix2-source` f05ca05 |
| m6: the generic EMVA-form lead's 24.2% silently assumed the reference landing page | The result card says "Assumes landing_page = <reference> (blank; no training lead had it missing)" under the headline. Scoring is unchanged. | `keel-fix2-source` 24dd827 |
| m8: Apply TOML errors appeared ~1,900 px above the editor | The error is shown under the TOML editor, the expander stays open, and the text is kept. | `keel-fix2-source` 1ad4365 |
| m9: stale review text after edits; the derived counts disagreed with validation | A row re-mapped in the table or through TOML while keeping the draft's reason gets "edited by the reviewer" (stored in `mapping.toml`), and an "Edited by you" note lists those rows. `dataset.json` gains `won_deal_values_missing`, which adds up with `non_positive_deal_values_blanked` to validation's "N Won row(s) have no deal value". | `keel-fix2-source` 7be450c, fe94bfb |
| m3: results text promised a status quo and baseline the run did not have; data note at the bottom | Neutral subtitle; the report expander is titled from the standard table's rows ("model only", "model vs status quo", …); the data note sits under the run header. | `keel-fix2-polish` 48539b8 |
| c1: the sidebar stayed open over the content on a phone | Streamlit's own sidebar navigation (`st.navigation(position="sidebar")`, logo via `st.logo`), which collapses after a page is chosen. | `keel-fix2-polish` ebaff9a |
| c2: clipped test-set segment and axis titles | Shorter segment labels (details in the control's help), axis title "Points (+20 = 2× odds)", long signal labels cut to 32 characters with the full text on hover. | `keel-fix2-polish` 23ccb69 |
| c3: raw floats in the source-format table | **Not reproduced**, no change (below). | – |
| c4: "None" cells, upload slots in column order, step numbers 02 → 05 | Blank cells show blank; the slots are laid out one row per three, so they read Leads, CRM history, Companies, People, Rules; the training step is numbered after the steps shown (01, 02, 03 before anything is validated). | `keel-fix2-polish` 6f42f76 |
| c5: the "+ Add files" icon on single-file uploaders; the log's top20_revenue unexplained | A scoped CSS rule hides "Add files" where the uploader takes one file. A note under the training log says its summary is the pipeline's own, which imputes blank deal values (ADR 0002). The chip for a wrong-type file is Streamlit's own refusal and stays (declined, below). | `keel-fix2-polish` c30792b |

Merges: dbc7269 (state), 7193f40 (source), 58b4623 (polish, including line-length fixes in the merge).

## Checks (on public data where Olist)

- **M5**: the QA repro was run in Chromium against the merged main on the real Olist files. The steps: upload both
  raw files, load `olist_funnel`, rename it, confirm, Convert, pick Legacy labels, open "Score a lead", come back.
  Both files were still there ("using …"), as were the mapping name `olist_handson`, "03 Check results" and "Save as
  a dataset". Labels stayed Legacy (POC) and both confirmation ticks stayed ticked.
- AppTest coverage in `tests/test_app_state.py`:
  - the widget keys are gone after a page switch, and the state is still there on return;
  - Save and Replace release the held files;
  - the m1, m10 and m7 repros.
  All six tests fail on 420fdf0.
- **m2**: on a generic Olist run, sampled every 250 ms after the switch to Legacy: for about 10 s only the spinners
  showed, then the headline and Generic vs v2 appeared together, both on 3,829 test leads.
- **M7**, `python -m emva.ingest score` on a generic Olist run with two new MQLs:
  - `new1` (origin `tiktok`) was refused on stderr: "Lead(s) new1: origin 'tiktok' is not one of the values this
    dataset's mapping knows (paid_search, social, …, other_publicities); this lead was not scored."
  - `new2` was scored.
  - In Keel, the same CSV showed the refusal above the table, and an unseen `landing_page_id` got its "scores as
    other" note.
- **m9**: the unchanged `olist_funnel` conversion gives `non_positive_deal_values_blanked` 797 and
  `won_deal_values_missing` 0. Together they match validation's "797 Won row(s) have no deal value". With
  `deal_value` switched to `product()` through TOML, the smoke test checks that the two counts add up to the validation
  figure and that the row shows "edited by the reviewer".
- **c1, c2**: checked in Chromium at 390 px (iPhone 13 emulation) and 1440 px. After navigation the sidebar is
  collapsed (`aria-expanded=false`), and no segment label or axis title is clipped.
- **c3**: at both widths, on sample, v2, legacy and generic runs, the source-format table shows `3.91%` and
  `BRL 682583`. The frame is float64 and the column-config keys match; the format was the same in the deployed
  31dd8b8. The raw `0.10130628…` was most likely the Scores CSV download, which is raw by design, like `scores.csv`.
- Merged main: `make test` TESTCOUNT; `make baseline` PASS (byte-identical weights and scores). The v1 standard
  report below is byte-identical to 420fdf0's (`cmp`).
- Branch runs:

  | branch | `make test` | `make baseline` |
  |---|---|---|
  | state | 852 passed, 1 skipped | PASS |
  | source | 848 passed, 1 skipped | PASS; v1 report byte-identical |
  | polish | 849 passed, 1 skipped | PASS |
  | 420fdf0 (before) | 841 passed, 1 skipped | PASS |

## Orchestrator decisions

- **M5**: the uploaded bytes are kept rather than only warning before they are lost; a warning still loses the work.
  Memory is bounded at one copy per slot. The widget's own copy is dropped because the uploader is not rendered while
  its slot holds a file.
- **M7**: an unlisted value-map value is refused per lead, not scored with a fallback. A value map has no documented
  fallback, and inventing one at scoring time would score a lead on a meaning nobody confirmed. Training conversion
  keeps its whole-file refusal, worded for the mapping author.
- **m6**: the smallest honest option was taken, a line on the card; defaulting the extras differently would change
  scores.
- **m9**: the "edited by the reviewer" marker is stored in the saved mapping, not only shown.
- **c1**: the custom sidebar links were replaced by Streamlit's navigation, the supported way to collapse on a phone,
  rather than injected JS.
- Review fix: `_raw_uploads` no longer returns an unused list of refused names.

## Declined or deferred

- **c3** was not reproduced, so there is no change. `money_format` has no thousands separator (`BRL 682583` in
  tables vs `BRL 682,583` on cards) because Streamlit's printf format has none. This is left as is.
- **c5**, the red chip for a wrong-type file in a single-file slot: it is Streamlit's own refusal, and hiding it would
  hide why the file was refused.
- The Score page's "Start from a lead" box is not kept across page switches (outside M5's scope).
- The review table's Reason cell keeps showing the old text until the form is rebuilt. Updating it would mean
  committing the form on every edit. The "Edited by you" note names those rows and says what is saved.
- A value outside the form options in a copied (not value-mapped) field still refuses the whole batch. The M7 ruling
  covers value maps only.
- Carried over: rotate the Anthropic key and split the Railway variables (the user); `VALUE_FLOOR_GBP` naming.
- **Not deployed.** Railway's push trigger was found disconnected: the service has no branch set.

<details>
<summary>Standard report: data/v1 (on simulated data; identical to 420fdf0)</summary>

REPORTBODY

</details>
