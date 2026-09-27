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

Merges: dbc7269 (state), 7193f40 (source), 58b4623 (polish, with line-length fixes in the merge); c9fecae fixes one test after the merge.

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
- Merged main: `make test` 867 passed, 1 skipped (after c9fecae, which updates the m2 test for the polish branch's shorter test-set labels; the merge alone had that one failure); `make baseline` PASS (byte-identical weights and scores). The v1 standard
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

### EMVA standard report

Data: `data/v1`. baseline = frozen `baseline/emva_score.py`; candidate = `emva` pipeline trained with `horizon H=120` labels and `v2` features; status quo = `status_quo_rules.json` reconstructed. Every model is scored on the same rows under two test definitions. AUC CI: percentile bootstrap, 1000 resamples, seed 0. Top-20% capture ranks by p (status quo: by its value) or by p×value.

#### Label definitions

Counts over all 9311 scored leads, as wins / losses / unlabelled. `label_source` is the CRM state at 2026-09-24. legacy y = baseline rules. won_within_h = Won within 120 days of created_at (NaN if younger and not Won). horizon y = won_within_h with ghosted-at-H leads excluded and stalled leads censored (the default training and test label).

| label_source | leads | legacy y | won_within_h | horizon y |
|---|---|---|---|---|
| won | 1199 | 1199 / 0 / 0 | 1099 / 100 / 0 | 1099 / 100 / 0 |
| crm_lost | 4896 | 0 / 4896 / 0 | 0 / 3505 / 1391 | 0 / 3505 / 1391 |
| stalled | 618 | 0 / 618 / 0 | 0 / 564 / 54 | 0 / 0 / 618 |
| ghosted | 1207 | 0 / 1207 / 0 | 0 / 1207 / 0 | 0 / 0 / 1207 |
| open | 1391 | 0 / 121 / 1270 | 0 / 70 / 1321 | 0 / 70 / 1321 |
| all | 9311 | 1199 / 6842 / 1270 | 1099 / 5446 / 2766 | 1099 / 3675 / 4537 |

Mature = created_at + 120 days <= 2026-09-24T00:00:00+00:00: the latest mature lead was created 2026-05-26T20:06:58+00:00. Train = created before 2026-05-01 (all mature); test = created on or after 2026-05-01.

| definition | train (wins) | test (wins) |
|---|---|---|
| legacy | 5588 (847) | 2453 (352) |
| horizon H=120 | 4049 (752) | 458 (80) |

Candidate trained with: `horizon H=120`.

#### (a) Legacy labels, legacy test set

2453 leads labelled by the baseline rules, created on or after 2026-05-01 (352 won). Label = legacy y.

##### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.814 [0.789, 0.836] | 0.1006 | 0.571 | 0.742 | 0.796 |
| candidate | 0.814 [0.790, 0.837] | 0.1020 | 0.571 | 0.733 | 0.815 |
| status quo | 0.639 [0.605, 0.673] | n/a | 0.398 | 0.447 | 0.447 |

##### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.000 | [-0.006, +0.006] | 0.972 |
| status quo | -0.174 | [-0.209, -0.141] | 0.000 |

##### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 246 | 0.009 | 0.016 | 0.009 | 0.012 |
| 2 | 245 | 0.020 | 0.012 | 0.022 | 0.012 |
| 3 | 245 | 0.033 | 0.016 | 0.040 | 0.029 |
| 4 | 245 | 0.050 | 0.053 | 0.061 | 0.049 |
| 5 | 246 | 0.075 | 0.077 | 0.092 | 0.065 |
| 6 | 245 | 0.109 | 0.135 | 0.132 | 0.135 |
| 7 | 245 | 0.150 | 0.131 | 0.184 | 0.147 |
| 8 | 245 | 0.213 | 0.176 | 0.254 | 0.167 |
| 9 | 245 | 0.316 | 0.327 | 0.363 | 0.347 |
| 10 | 246 | 0.511 | 0.492 | 0.559 | 0.472 |

Status quo has no probability, so it has no Brier score or calibration.

##### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 754 | 103 | 0.800 | 0.801 | 0.641 |
| 2026-06 | 665 | 93 | 0.797 | 0.806 | 0.631 |
| 2026-07 | 541 | 88 | 0.833 | 0.826 | 0.602 |
| 2026-08 | 371 | 55 | 0.813 | 0.812 | 0.664 |
| 2026-09 | 122 | 13 | 0.920 | 0.914 | 0.772 |

##### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 2453 | 346 | 17167 | 45562 | 131.8× | 11.8% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 2453 | 414 | 17889 | 44625 | 107.8× | 10.9% |
| candidate | all scored leads | 9311 | 446 | 18646 | 44625 | 100.0× | 10.6% |
| status quo | test set | 2453 | 96 | 1440 | 1872 | 19.5× | 6.7% |
| status quo | all scored leads | 9311 | 96 | 1440 | 1872 | 19.5× | 6.7% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

##### Notes

- status quo: 203 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.397.
- status quo: 203 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454.
- status quo: 203 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.454.

#### (b) Horizon labels, mature test set

458 mature leads created on or after 2026-05-01 and up to 2026-05-26T20:06:58+00:00 that the default horizon definition labels (80 won). Label = won within 120 days; ghosted-at-H leads excluded, stalled leads censored.

##### Headline

| model | AUC [95% CI] | Brier | top-20% wins | top-20% revenue (by p) | top-20% revenue (by p×value) |
|---|---|---|---|---|---|
| baseline | 0.778 [0.726, 0.827] | 0.1258 | 0.487 | 0.670 | 0.752 |
| candidate | 0.778 [0.730, 0.827] | 0.1265 | 0.450 | 0.608 | 0.774 |
| status quo | 0.644 [0.571, 0.717] | n/a | 0.425 | 0.525 | 0.525 |

##### Paired AUC comparison vs baseline

| model | AUC − baseline | 95% CI | bootstrap p |
|---|---|---|---|
| candidate | +0.000 | [-0.015, +0.015] | 0.974 |
| status quo | -0.134 | [-0.199, -0.069] | 0.000 |

##### Calibration by decile of p

| decile | n | baseline mean p | baseline observed | candidate mean p | candidate observed |
|---|---|---|---|---|---|
| 1 | 46 | 0.008 | 0.022 | 0.009 | 0.000 |
| 2 | 46 | 0.020 | 0.000 | 0.023 | 0.043 |
| 3 | 46 | 0.038 | 0.022 | 0.045 | 0.000 |
| 4 | 45 | 0.060 | 0.111 | 0.072 | 0.089 |
| 5 | 46 | 0.090 | 0.174 | 0.106 | 0.109 |
| 6 | 46 | 0.124 | 0.152 | 0.151 | 0.261 |
| 7 | 45 | 0.171 | 0.178 | 0.208 | 0.222 |
| 8 | 46 | 0.237 | 0.239 | 0.280 | 0.239 |
| 9 | 46 | 0.357 | 0.391 | 0.391 | 0.348 |
| 10 | 46 | 0.555 | 0.457 | 0.592 | 0.435 |

Status quo has no probability, so it has no Brier score or calibration.

##### AUC by test month

| month | n | wins | baseline | candidate | status quo |
|---|---|---|---|---|---|
| 2026-05 | 458 | 80 | 0.778 | 0.778 | 0.644 |

##### Value scale

| model | scope | n | median | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline | test set | 458 | 432 | 18577 | 34763 | 80.5× | 10.9% |
| baseline | all scored leads | 9311 | 374 | 17841 | 45562 | 121.7× | 11.5% |
| candidate | test set | 458 | 518 | 20121 | 32268 | 62.3× | 9.7% |
| candidate | all scored leads | 9311 | 446 | 18646 | 44625 | 100.0× | 10.6% |
| status quo | test set | 458 | 96 | 1440 | 1872 | 19.5× | 6.4% |
| status quo | all scored leads | 9311 | 96 | 1440 | 1872 | 19.5× | 6.7% |

Value = p × expected deal value (models), rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.

##### Notes

- status quo: 37 rows tie at the top-20% cut when ranking wins; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.398.
- status quo: 37 rows tie at the top-20% cut when ranking revenue by p; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.506.
- status quo: 37 rows tie at the top-20% cut when ranking revenue by p×value; the table uses numpy's default argsort (as the baseline does), tie-averaged value is 0.506.

#### Bottom-decile ghosted share

Share of leads in the 10% of each population with the lowest p that are *ghosted* (`label_source`: mature and not contacted within 120 days) or *still New* at 2026-09-24 whatever their age (the definition behind the plan's baseline figure of 27%). The horizon test set (b) excludes ghosted leads, so the comparison uses populations that keep them.

| population | n | ghosted: overall | ghosted: baseline bottom decile | ghosted: candidate bottom decile | still New: overall | still New: baseline bottom decile | still New: candidate bottom decile |
|---|---|---|---|---|---|---|---|
| all leads labelled by legacy rules (train + test) | 8041 | 15.0% | 24.5% | 24.4% | 16.5% | 27.5% | 27.5% |
| (a) legacy test set | 2453 | 5.7% | 9.4% | 9.0% | 10.7% | 19.2% | 19.2% |
| all mature leads, every label_source | 6278 | 19.2% | 30.6% | 31.3% | 19.2% | 30.6% | 31.3% |
| mature leads created on or after 2026-05-01, every label_source | 650 | 21.7% | 29.2% | 27.7% | 21.7% | 29.2% | 27.7% |

#### Value transforms (plan 3.2)

Candidate value = `value_formula` (p × E[deal value] × margin) through each transform, fitted on the candidate's 4049 training leads. Top-20% revenue = tie-averaged share of recorded won deal value in the top 20% by value; vs baseline = that share over the baseline's own (p×value, identity). value / revenue = total value over total recorded revenue. Every transform is monotone, so capture moves only through ties at the cut.

##### (a) Legacy labels, legacy test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | vs baseline | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (identity) | 11 | 346 | 5670 | 17167 | 45562 | 131.8× | 11.8% | 0.796 | 1 | 100.0% | 1.14 |
| candidate: identity | 6 | 414 | 6875 | 17889 | 44625 | 107.8× | 10.9% | 0.815 | 1 | 102.5% | 1.30 |
| candidate: cap p97 | 6 | 414 | 6875 | 13221 | 13221 | 31.9× | 6.6% | 0.815 | 1 | 102.5% | 1.21 |
| candidate: cap p97 + floor £25 | 25 | 414 | 6875 | 13221 | 13221 | 31.9× | 6.6% | 0.815 | 1 | 102.5% | 1.22 |
| candidate: cap p97 + sqrt | 156 | 1274 | 5193 | 7201 | 7201 | 5.7× | 3.5% | 0.815 | 1 | 102.5% | 1.26 |
| candidate: cap p97 + log | 25 | 1208 | 5412 | 6664 | 6664 | 5.5× | 3.2% | 0.815 | 1 | 102.5% | 1.25 |
| candidate: tiers 5 | 44 | 541 | 8910 | 8910 | 8910 | 16.5× | 4.2% | 0.792 | 479 | 99.5% | 1.29 |
| candidate: tiers 10 | 23 | 384 | 5471 | 12350 | 12350 | 32.2× | 5.8% | 0.802 | 227 | 100.8% | 1.29 |
| candidate: cap p97 + log + floor £25 | 25 | 1208 | 5412 | 6664 | 6664 | 5.5× | 3.2% | 0.815 | 1 | 102.5% | 1.25 |

##### (b) Horizon labels, mature test set

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share | top-20% revenue | ties at cut | vs baseline | value / revenue |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (identity) | 11 | 432 | 7048 | 18577 | 34763 | 80.5× | 10.9% | 0.752 | 1 | 100.0% | 0.99 |
| candidate: identity | 6 | 518 | 7873 | 20121 | 32268 | 62.3× | 9.7% | 0.774 | 1 | 103.0% | 1.11 |
| candidate: cap p97 | 6 | 518 | 7873 | 13221 | 13221 | 25.5× | 5.0% | 0.774 | 1 | 103.0% | 1.02 |
| candidate: cap p97 + floor £25 | 25 | 518 | 7873 | 13221 | 13221 | 25.5× | 5.0% | 0.774 | 1 | 103.0% | 1.02 |
| candidate: cap p97 + sqrt | 152 | 1425 | 5557 | 7201 | 7201 | 5.1× | 2.8% | 0.774 | 1 | 103.0% | 0.99 |
| candidate: cap p97 + log | 23 | 1425 | 5668 | 6664 | 6664 | 4.7× | 2.6% | 0.774 | 1 | 103.0% | 0.99 |
| candidate: tiers 5 | 44 | 541 | 8910 | 8910 | 8910 | 16.5× | 3.3% | 0.742 | 105 | 98.7% | 1.03 |
| candidate: tiers 10 | 23 | 698 | 12350 | 12350 | 12350 | 17.7× | 4.5% | 0.747 | 47 | 99.3% | 1.07 |
| candidate: cap p97 + log + floor £25 | 25 | 1425 | 5668 | 6664 | 6664 | 4.7× | 2.6% | 0.774 | 1 | 103.0% | 0.99 |

##### All scored leads

| value | p1 | p50 | p90 | p99 | max | max/median | top-1% share |
|---|---|---|---|---|---|---|---|
| baseline (identity) | 11 | 374 | 6317 | 17841 | 45562 | 121.7× | 11.5% |
| candidate: identity | 6 | 446 | 7332 | 18646 | 44625 | 100.0× | 10.6% |
| candidate: cap p97 | 6 | 446 | 7332 | 13221 | 13221 | 29.6× | 6.3% |
| candidate: cap p97 + floor £25 | 25 | 446 | 7332 | 13221 | 13221 | 29.6× | 6.3% |
| candidate: cap p97 + sqrt | 152 | 1323 | 5363 | 7201 | 7201 | 5.4× | 3.4% |
| candidate: cap p97 + log | 23 | 1278 | 5534 | 6664 | 6664 | 5.2× | 3.1% |
| candidate: tiers 5 | 44 | 541 | 8910 | 8910 | 8910 | 16.5× | 4.0% |
| candidate: tiers 10 | 23 | 384 | 5471 | 12350 | 12350 | 32.2× | 5.5% |
| candidate: cap p97 + log + floor £25 | 25 | 1278 | 5534 | 6664 | 6664 | 5.2× | 3.1% |

##### Acceptance (plan 3.2): ≥ 95% of the baseline's capture on every test set and max/median < 20× on every test set and on all scored leads

| transform | result |
|---|---|
| identity | fail |
| cap p97 | fail |
| cap p97 + floor £25 | fail |
| cap p97 + sqrt | PASS |
| cap p97 + log | PASS |
| tiers 5 | PASS |
| tiers 10 | fail |
| cap p97 + log + floor £25 | PASS |

#### Click-ID coverage (plan 3.5)

Scored leads per paid channel by the identifier an upload could match on (see `docs/platform_contract.md`): a click id (gclid, gbraid, wbraid, fbclid, li_fat_id, oppref), else the lead-form id, else nothing but hashed email / phone (Meta: plus the `fbp` browser cookie).

| channel | leads | click id | lead-form id | no id: fbp present | no id at all | email present | phone present |
|---|---|---|---|---|---|---|---|
| google | 2763 | 72.3% | 0.0% | 22.0% | 27.7% | 100.0% | 25.2% |
| meta | 2733 | 70.7% | 0.0% | 22.6% | 29.3% | 100.0% | 24.4% |
| linkedin | 1485 | 70.8% | 0.0% | 24.5% | 29.2% | 100.0% | 23.7% |
| chatgpt | 473 | 69.6% | 0.0% | 24.9% | 30.4% | 100.0% | 24.5% |
| meta_leadads | 932 | 0.0% | 100.0% | 0.0% | 0.0% | 100.0% | 80.8% |
| landing-page paid (all but meta_leadads) | 7454 | 71.2% | 0.0% | 22.9% | 28.8% | 100.0% | 24.6% |
| all paid | 8386 | 63.3% | 11.1% | 20.4% | 25.6% | 100.0% | 30.8% |

#### Design collinearity (plan 2.7): FAIL

Candidate design (`v2` features, 39 columns) on its 4049 training rows; fails when two columns have |corr| > 0.95. The report exits 1 on a failure only with `--strict`; `python -m emva.eval.collinearity` always does.

- FAIL (1 pairs with |corr| > 0.95): max |corr| = 1.000 between channel=meta_leadads and session_missing=yes
  - channel=meta_leadads ~ session_missing=yes: +1.000

#### Baseline script output

- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, which fills blank deal values with its predicted value, printed:

```
  model   auc  brier  top20_wins  top20_revenue
formula 0.814 0.1006       0.571          0.795
```

</details>
