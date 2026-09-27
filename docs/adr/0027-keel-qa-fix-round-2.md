# 0027. Keel QA fix round 2: Map & convert work kept across pages, no stale results, unlisted source values refused per lead, edited review rows marked

- Status: Proposed
- Date: 2026-09-27
- Source: hands-on QA of the live Keel (2026-09-27), the issues ADR 0026 left open (M5, M7, m1-m3, m6-m10, c1-c5);
  `reports/keel-fixes-2.md`; branches `keel-fix2-state`, `keel-fix2-source`, `keel-fix2-polish`

## Context

The first fix round (ADR 0026) left one major issue on each of two paths, plus the minor and cosmetic findings:

- **M5.** Map & convert work (raw files, the mapping under review, the conversion, its check results) and the
  training choices were lost when the person opened another page. Streamlit deletes the state of widgets that are not
  rendered, and file uploaders lose their files.
- **M7.** Source-format scoring disagreed with itself on values outside a value map:
  - the one-lead form offered "other (any value not listed)";
  - a CSV with one such value refused the whole batch, worded for a mapping author;
  - an unseen extra category was scored as `other` with no note.
- Several pages showed results from the previous input next to the current one:
  - m1: check results after a refused file was removed;
  - m2: Generic vs v2 while the test set changed;
  - m7: the previous lead's score after the prefill changed;
  - m10: a conversion after a failed mapping edit.
- Other text on the pages claimed more than was true:
  - m3: "status quo" and "baseline" promised on runs without them;
  - m6: a headline P(close) that silently assumed reference levels;
  - m9: a draft's reason kept on a row the reviewer changed.

## Decision

1. **Uploads and review state survive page switches (M5).**
   - `app.uploads` holds each upload slot's file in session state as one `HeldFile` (name, bytes, SHA-256). This
     covers the three raw slots and the rules slot of Map & convert, and the five EMVA-format slots.
   - Memory is bounded: one copy per slot, never a history.
     - A new upload replaces the slot's file.
     - While a slot holds a file, its uploader is not rendered, so Streamlit drops the widget's own copy.
     - Replace empties the slot, and saving a dataset releases every slot.
     - At most 8 slots of at most 200 MB each.
   - A `ground_truth*` upload is held by its name only, never its bytes, and is shown as refused.
   - Simple widgets (training dataset, label mode, feature set, saved-mapping choice, the two confirmation ticks) use
     the documented separate-key pattern: `ui.restore` before the widget and `ui.keep` after it. The confirmation
     ticks stay keyed by the mapping digest or features signature, so they are never ticked for a different mapping.
   - The mapping under review is stored with its edits applied (`mc_edited`). When Streamlit has dropped the review
     widgets, the page rebuilds them from it. Data-editor deltas cannot be restored through session state.
   - Warning before the state is lost was rejected: it would still lose the work.
2. **Stale results are withdrawn, never shown next to new inputs.**
   - m1: validation compares the uploaded and the refused names with those it checked. A change asks the person to
     validate again, and removing every file clears the results.
   - m10:
     - "(value map)" is offered as a target only when some row of the table shows it.
     - An edit that cannot be applied makes the mapping incomplete.
     - A conversion made from another mapping, other rules, or a mapping that is now incomplete is marked
       "Convert again", and its check and save steps are withdrawn.
   - m2: everything on the results page that depends on the test set is computed and drawn in one slot, so numbers
     from two test sets are never on screen together.
   - m7: the Score page's result belongs to a model and a starting lead. Changing either one clears the result.
3. **Unlisted value-map values are refused per lead at scoring time (M7; amends the source-format rule of
   ADR 0023).**
   - A value map has no documented fallback: every raw value must be listed. For new leads,
     `emva.ingest.convert_leads` now returns `NewLeads(leads, refused)`. A row whose value-map column holds an unlisted
     value is left out and reported as an `UnlistedValue`. Its message names the leads (or row numbers when there is
     no id), the column, the values and the allowed values. The other rows are converted and scored. When every row
     is refused, the call raises.
   - Keel shows the refusal above the results table. `python -m emva.ingest score` prints it to stderr and scores the
     rest.
   - `convert` (training data) still refuses the whole file, worded for the mapping author.
   - The one-lead form offers exactly the listed values. "other (any value not listed)" is shown only for a generic
     extra's `other` level, where it is true. Olist's `origin` value map lists a raw value "other", shown as plain
     "other".
   - An extra's value that is not a kept training level scores as `other` (ADR 0024 unchanged). The page now names
     those leads and values in a note per extra.
4. **The result card says what a blank extra assumes (m6).** When an extra is blank on the lead and no training lead
   had it missing, it scores as its reference level (ADR 0026 open item). The card then carries "Assumes
   <extra> = <reference> (blank; no training lead had it missing)" under the headline. Scoring is unchanged.
5. **Edited review rows lose the draft's reason (m9).**
   - This covers a lead role, the outcome, a field row or a feature that the reviewer re-mapped, in the table or
     through TOML, while keeping the old reason and confidence. That review is replaced by "edited by the reviewer"
     (no confidence).
   - The marker is **stored** in `mapping.toml`, so a saved mapping never carries a reason written for another
     mapping. A reason the reviewer types is kept.
   - `dataset.json` gains `won_deal_values_missing`: Won leads left without a deal value because the source is blank,
     an input of its expression is blank, or `deal_value` is unmapped. With `non_positive_deal_values_blanked`, it
     adds up to validation's "N Won row(s) have no deal value". Older `dataset.json` files read it as 0.
6. **The page text matches the run (m3, c5).**
   - The results subtitle no longer promises a status-quo comparison.
   - The report expander is titled from the rows its standard table has, e.g. "(model only)" or "(model vs status
     quo)".
   - The data note (simulated sample, or a converted source's data with its licence) sits under the run header, not
     in the footer.
   - The training log says its summary table is the pipeline's own, whose `top20_revenue` imputes blank deal values,
     while the results page counts recorded values only (ADR 0002). The pipeline's output is unchanged.
7. **Keel's navigation is Streamlit's own sidebar navigation (c1).** `st.navigation(position="sidebar")` with
   `st.logo` replaces the custom page links, because only Streamlit's nav collapses the sidebar after a page is chosen
   on a phone. The wordmark on the sign-in screen is unchanged.

## Consequences

- Scores and training are unchanged. `make baseline` passes and the v1 standard report is byte-identical.
- `convert_leads` returns `NewLeads` rather than a DataFrame; callers read `.leads`.
- A converted dataset's `dataset.json` gains one derived count. Its training files are unchanged.
- A mapping saved after edits may carry the reason "edited by the reviewer".
- Session memory holds each uploaded file for as long as it stays in its slot (at most 8 × 200 MB per session).
- Declined, with reasons in the report:
  - c3 (raw floats in the source-format table) did not reproduce: the table shows % and the currency; the raw floats
    were most likely the Scores CSV download, which is raw by design.
  - Hiding Streamlit's own chip for a file of the wrong type (c5) would hide why the file was refused.
- Open items:
  - The Score page's "Start from a lead" box is not kept across page switches.
  - `money_format` has no thousands separator (a Streamlit printf limit), unlike the cards.
  - The review table's Reason cell shows the old text until the form is rebuilt; the "Edited by you" note names those
    rows and says what is saved.
