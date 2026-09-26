"""Page: Upload & train. Map and convert a foreign export (or upload files in EMVA's format), validate, save them as
a dataset, and train a model on any dataset."""
from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from app import charts, ingest, storage, training, ui
from app import components as C
from app.storage import StorageError
from app.validation import ValidationReport, validate_files
from emva.ingest.convert import DEFAULT_FORM_VARIANT
from emva.ingest.draft import source_name
from emva.ingest.mapping import (
    CONFIDENCES,
    LEAD_ROLES,
    LOST_WITHOUT_CLOSE,
    OUTCOME_KINDS,
    DatasetMapping,
    dump_mapping,
)
from emva.ingest.profile import ColumnProfile

SLOTS: list[tuple[str, str, str]] = [
    (storage.LEADS_FILE, "Leads", "One row per form submission, with the answers JSON and on-site session."),
    (storage.CRM_FILE, "CRM history", "Every stage change per lead (New, Contacted, ..., Won/Lost) with deal value."),
    (storage.COMPANIES_FILE, "Companies", "Firmographics joined on the company domain."),
    (storage.PEOPLE_FILE, "People", "Contact enrichment. Optional: the model never reads it."),
    (storage.RULES_FILE, "Status-quo rules", "Optional JSON: today's bucket values, for the benchmark."),
]
# Raw-file slots of Map & convert: the primary file (one row per lead) first, then up to two joined files.
RAW_SLOTS: list[tuple[str, str, str]] = [
    ("raw_primary", "Primary file", "One row per lead (e.g. the leads, MQL or opportunities export)."),
    ("raw_join_1", "Joined file (optional)", "Joined to the primary file on a shared column (e.g. closed deals)."),
    ("raw_join_2", "Joined file (optional)", "A second joined file (e.g. accounts)."),
]
ROLE_LABELS: dict[str, str] = {
    "lead_id": "Lead id", "created_at": "Created at *", "contacted_at": "First contacted at", "won_at": "Won at",
    "close_at": "Lost / closed at", "deal_value": "Deal value"}
OUTCOME_LABELS: dict[str, str] = {"stage": "A column of CRM stages", "won_flag": "A won / lost flag column",
                                  "presence": "Won when a column is filled"}
LOST_LABELS: dict[str, str] = {"error": "Refuse the conversion", "as_of": "Date the Lost row at as_of"}
# Session-state keys of Map & convert (prefix mc_) and of the shared check / save steps.
MC_SIG, MC_FORM, MC_VERSION, MC_MSG, MC_CONV = "mc_sig", "mc_form", "mc_version", "mc_msg", "mc_conv"
VALIDATION, VALIDATED_FILES, VALIDATED_ORIGIN = "validation", "validated_files", "validated_origin"
LABEL_MODES = {"horizon": "Horizon", "legacy": "Legacy (POC)"}
FEATURE_SETS = {"v2": "v2", "generic": "Generic (v2 + extras)", "legacy": "Legacy (POC)"}
PROCS_KEY, ACTIVE_KEY = "training_procs", "active_run"
LOG_TAIL_LINES = 60


def _collect() -> tuple[dict[str, bytes], list[str]]:
    """The uploader slots: files keyed by canonical training-file name, plus names of ground-truth-looking uploads
    (passed to validation under their own name so they are refused without being parsed)."""
    files: dict[str, bytes] = {}
    refused: list[str] = []
    cols = st.columns(3, gap="medium")
    for i, (name, label, help_text) in enumerate(SLOTS):
        with cols[i % 3]:
            up = st.file_uploader(f"{label} · {name}", type=[Path(name).suffix[1:]], key=f"up_{name}", help=help_text)
            if up is None:
                continue
            if up.name.lower().startswith(storage.FORBIDDEN_PREFIX):
                refused.append(up.name)
            elif up.name in storage.TRAINING_FILES and up.name != name:
                ui.html(C.callout("belongs in its own slot, not here.", "warn", lead=up.name))
            else:
                files[name] = up.getvalue()
    return files, refused


def _render_report(report: ValidationReport, provided: set[str]) -> None:
    """Per-file cards with errors and warnings, then the preview summary."""
    slots = [n for n, _, _ in SLOTS]
    meta = [n for n in (*storage.METADATA_FILES, storage.EXTRA_FEATURES_FILE)
            if n in provided or report.for_file(n) != ([], [])]
    names = slots + meta + sorted({i.file for i in report.errors} - set(slots) - set(meta))
    labels = {n: lbl for n, lbl, _ in SLOTS} | {storage.MAPPING_FILE: "Confirmed mapping",
                                                 storage.DATASET_META_FILE: "Converter metadata (dates, coverage)",
                                                 storage.EXTRA_FEATURES_FILE: "Extra features (generic feature set)"}
    left, right = st.columns(2, gap="medium")
    for i, name in enumerate(names):
        errors, warnings = report.for_file(name)
        with (left if i % 2 == 0 else right):
            ui.html(C.file_card(name, labels.get(name, "Refused upload"), errors, warnings, name in provided))
    s = report.summary
    if s.get("leads"):
        ui.html(C.section("Preview", "What these files contain, before anything is saved."))
        ui.html(C.stats([("Leads", f"{s['leads']:,}"), ("Won leads", f"{s.get('won_leads', 0):,}"),
                         ("First lead", s.get("created_from", "–")), ("Last lead", s.get("created_to", "–")),
                         ("CRM rows", f"{s['rows'].get(storage.CRM_FILE, 0):,}"),
                         ("Companies", f"{s['rows'].get(storage.COMPANIES_FILE, 0):,}")]))
        a, _ = st.columns([3, 2], gap="large")
        with a:
            if s.get("final_stage_counts"):
                st.caption("Leads by current CRM stage")
                st.plotly_chart(charts.stage_chart(s["final_stage_counts"]), config=charts.CONFIG, theme=None,
                                width="stretch", key="stage_chart")


# --- 01 Map & convert ------------------------------------------------------------------------------------------------

def _raw_uploads() -> tuple[dict[str, bytes], list[str]]:
    """The raw-file slots: file name -> bytes (primary first), plus refused ground-truth-looking names."""
    files: dict[str, bytes] = {}
    refused: list[str] = []
    cols = st.columns(3, gap="medium")
    for (key, label, help_text), col in zip(RAW_SLOTS, cols):
        with col:
            up = st.file_uploader(label, type=["csv"], key=key, help=help_text)
        if up is None:
            continue
        if up.name.lower().startswith(storage.FORBIDDEN_PREFIX):
            refused.append(up.name)
        elif up.name in files:
            with col:
                ui.html(C.callout("is already uploaded in another slot.", "warn", lead=up.name))
        else:
            files[up.name] = up.getvalue()
    return files, refused


def _signature(files: dict[str, bytes]) -> tuple[tuple[str, str], ...]:
    """Names and content hashes of the raw files, in slot order (a change resets the mapping under review)."""
    return tuple((n, hashlib.sha256(b).hexdigest()) for n, b in files.items())


@st.cache_data(show_spinner="Reading and profiling the files…", max_entries=4)
def _parsed(sig: tuple[tuple[str, str], ...], _files: dict[str, bytes]) -> tuple[dict[str, pd.DataFrame],
                                                                                 list[ColumnProfile]]:
    """The raw frames and their column profiles (cached by ``sig``, the files' names and hashes)."""
    frames = ingest.raw_frames(_files)
    return frames, ingest.profile(frames)


def _set_form(form: ingest.MappingForm, kind: str, text: str, lead: str = "") -> None:
    """Replace the mapping under review (new widget keys) and the message about where it came from."""
    st.session_state[MC_FORM] = form
    st.session_state[MC_VERSION] = st.session_state.get(MC_VERSION, 0) + 1
    st.session_state[MC_MSG] = (kind, text, lead)
    st.session_state.pop(MC_CONV, None)


def _reset_mapping() -> None:
    """Forget the mapping under review and its conversion (the raw files changed)."""
    for k in (MC_FORM, MC_MSG, MC_CONV):
        st.session_state.pop(k, None)
    if st.session_state.get(VALIDATED_ORIGIN) == "convert":
        for k in (VALIDATION, VALIDATED_FILES, VALIDATED_ORIGIN):
            st.session_state.pop(k, None)


def _start_buttons(root: Path, frames: dict[str, pd.DataFrame], profiles: list[ColumnProfile]) -> None:
    """Three ways to start the mapping: an AI draft, a saved or built-in mapping, or the table by hand."""
    ui.html(C.section("Start the mapping", "Draft one with AI from the column profile (never the rows), load a "
                      "mapping saved with an earlier dataset from the same source (no model call), or fill the "
                      "table by hand.", step="1c"))
    a, b, c = st.columns([2, 3, 2], gap="medium", vertical_alignment="bottom")
    cached = ingest.draft_is_cached(profiles, root)
    with a:
        if st.button("Draft mapping with AI", icon=":material/auto_awesome:", key="mc_draft", width="stretch"):
            with st.spinner("Drafting the mapping from the column profile…"):
                out = ingest.draft(profiles, root, source_name(next(iter(frames))))
            if out.mapping is not None:
                form = ingest.form_from_mapping(out.mapping, frames, out.origin)
                if out.dates_note:
                    form = replace(form, as_of="", test_from="")
                _set_form(form, "warn" if out.dates_note else "info",
                          (out.dates_note or "") + " Every row is a proposal: review targets, the outcome and the "
                          "dates, then confirm.", lead=out.origin + ".")
            else:
                _set_form(ingest.blank_form(frames), "warn", out.error or "",
                          lead="No draft; the table is ready to fill by hand.")
        st.caption("A cached draft exists for these columns: no model call." if cached else
                   "Sends the column profile (names, types, redacted examples) to Claude Haiku.")
    with b:
        choices = ingest.mapping_choices(root)
        by_key = {ch.key: ch for ch in choices}
        key = st.selectbox("Use saved mapping", list(by_key), key="mc_choice", index=None,
                           placeholder="Choose a saved or built-in mapping", format_func=lambda k: by_key[k].label)
    with c:
        if st.button("Load mapping", icon=":material/upload_file:", key="mc_load", disabled=key is None,
                     width="stretch"):
            try:
                m = ingest.load_choice(by_key[key])
            except ValueError as e:
                st.session_state[MC_MSG] = ("bad", str(e), "This mapping file is not valid.")
            else:
                missing = ingest.missing_files(m, frames)
                if missing:
                    st.session_state[MC_MSG] = (
                        "bad", f"It reads {', '.join(missing)}, which is not uploaded (files are matched by name; "
                        f"uploaded: {', '.join(frames)}).", "This mapping does not fit these files.")
                else:
                    _set_form(ingest.form_from_mapping(m, frames, by_key[key].label), "info",
                              "No model call. Review it against these files, then confirm.",
                              lead=f"Loaded {by_key[key].label}.")
        if st.button("Fill by hand", icon=":material/edit_note:", key="mc_blank", width="stretch"):
            _set_form(ingest.blank_form(frames), "info", "Every column starts as ignore.",
                      lead="Empty mapping.")


def _text_inputs(form: ingest.MappingForm, v: int) -> ingest.MappingForm:
    """Name, dates, source URL and licence."""
    c1, c2, c3 = st.columns(3, gap="medium")
    name = c1.text_input("Mapping name", form.name, key=f"mc_name_{v}", help="Letters, digits, _ and -.")
    as_of = c2.text_input("as_of (snapshot date) *", form.as_of, key=f"mc_asof_{v}", placeholder="2018-11-15",
                          help="The date the outcomes were read at; every event must be on or before it.")
    test_from = c3.text_input("test_from (train/test boundary) *", form.test_from, key=f"mc_testfrom_{v}",
                              placeholder="2018-03-01", help="Leads created on or after it are the test set; "
                              "frozen for this dataset once saved.")
    c4, c5, c6 = st.columns(3, gap="medium")
    url = c4.text_input("Source URL", form.source_url, key=f"mc_url_{v}")
    licence = c5.text_input("Licence", form.licence, key=f"mc_licence_{v}")
    with c6:
        drop = st.checkbox("Drop rows without created_at", form.drop_rows_without_created_at, key=f"mc_drop_{v}",
                           help="Otherwise a blank created_at refuses the conversion.")
    return replace(form, name=name.strip(), as_of=as_of.strip(), test_from=test_from.strip(), source_url=url.strip(),
                   licence=licence.strip(), drop_rows_without_created_at=drop)


def _pick(label: str, options: list[str], current: str, key: str, help_text: str | None = None) -> str:
    """A select box over column references with a "not mapped" first option; returns the choice ("" = none)."""
    opts = ["", *options] if current in options or not current else ["", current, *options]
    return st.selectbox(label, opts, index=opts.index(current), key=key, help=help_text,
                        format_func=lambda o: "— not mapped —" if o == "" else o)


def _review_note(form: ingest.MappingForm, key: str) -> None:
    """The draft's or reviewer's reason and confidence for a lead role or the outcome, as plain text (``st.text``: an
    LLM's or a person's words are never rendered as markdown)."""
    r = form.review.get(key)
    if r is not None and (r.reason or r.confidence):
        st.text(f"{r.confidence or 'no'} confidence · {r.reason}")  # plain text: a reason is never markdown


def _lead_and_outcome(form: ingest.MappingForm, edited: ingest.MappingForm, frames: dict[str, pd.DataFrame],
                      v: int) -> ingest.MappingForm:
    """Join columns, the lead columns and the outcome."""
    refs = form.refs()
    f = edited
    joined = list(form.sources[1:])
    if joined:
        cols = st.columns(len(joined), gap="medium")
        for s, col in zip(joined, cols):
            with col:
                options = list(frames[s.file].columns) if s.file in frames else [s.join_on]
                j = _pick(f"{s.file} joins on", options, s.join_on or "", f"mc_join_{v}_{s.name}",
                          "A column with the same name in the primary file; unique in this file.")
                f = ingest.with_join(f, s.file, j)
    st.markdown("**Lead columns** (* required)")
    cols = st.columns(3, gap="medium")
    for i, role in enumerate(LEAD_ROLES):
        e = form.lead.get(role)
        with cols[i % 3]:
            if ingest.is_plain(e):
                ref = _pick(ROLE_LABELS[role], refs, e.column if e is not None else "", f"mc_role_{v}_{role}")
                f = ingest.with_role(f, role, ref)
            else:
                st.text_input(ROLE_LABELS[role], ingest.expr_text(e), key=f"mc_role_{v}_{role}", disabled=True,
                              help="A derived expression: edit it in the TOML below.")
            _review_note(form, role)
    st.markdown("**Outcome** (how a lead's final CRM stage is read)")
    c1, c2, c3 = st.columns(3, gap="medium")
    kind = c1.selectbox("Outcome kind", list(OUTCOME_KINDS), index=list(OUTCOME_KINDS).index(form.outcome_kind),
                        format_func=OUTCOME_LABELS.get, key=f"mc_okind_{v}")
    with c2:
        column = _pick("Outcome column *", refs, form.outcome_column, f"mc_ocol_{v}")
    lost = c3.selectbox("A Lost lead without a close date", list(LOST_WITHOUT_CLOSE),
                        index=list(LOST_WITHOUT_CLOSE).index(form.lost_without_close), format_func=LOST_LABELS.get,
                        key=f"mc_olost_{v}")
    _review_note(form, "outcome")
    table = None
    if kind in ("stage", "won_flag") and column:
        same = kind == form.outcome_kind and column == form.outcome_column
        base = replace(f, outcome_kind=kind, outcome_column=column,
                       outcome_values=form.outcome_values if same else {})
        values, truncated = ingest.outcome_frame(frames, base)
        if truncated:
            ui.html(C.callout(f"{column} has more than {ingest.MAX_OUTCOME_VALUES} distinct values; is it really "
                              "an outcome column?", "warn"))
        st.caption("Give every raw value its meaning (\"\" is a blank cell). A value left without one is refused "
                   "by the conversion if it occurs." if kind == "stage" else
                   "Mark each raw value Won or Lost (\"\" is a blank cell).")
        table = st.data_editor(values, key=f"mc_ovalues_{v}_{kind}_{column}", hide_index=True, num_rows="dynamic",
                               width="stretch", column_config={
                                   "value": st.column_config.TextColumn("Raw value"),
                                   "meaning": st.column_config.SelectboxColumn(
                                       "Meaning", options=["", *ingest.meanings(kind)])})
    try:
        return ingest.apply_outcome(f, kind, column, table, lost)
    except ValueError as e:
        ui.html(C.callout(str(e), "bad"))
        return replace(f, outcome_kind=kind, outcome_column=column, lost_without_close=lost)


def _field_table(form: ingest.MappingForm, edited: ingest.MappingForm, v: int) -> ingest.MappingForm:
    """The review table of field rows and extra features (low-confidence rows highlighted), the read-only value maps
    and the read-only derived features."""
    st.markdown("**Fields and extra features** (one row per uploaded column; ignore = not used as a field)")
    st.caption("Tick Feature to use a column as an extra feature of the generic feature set (v2 + extras): numeric "
               "columns are cut into 5 bins, categorical ones keep levels with at least 30 training leads. Only "
               "columns known when the lead is submitted: an outcome column here leaks the label. A column can be a "
               "field and a feature at once. Reason and confidence describe the field when it has a target, else "
               "the feature.")
    rf = ingest.review_frame(form)
    low = rf.source[rf.check != ""].tolist()
    styled = rf.style.apply(lambda r: ["background-color: rgba(214, 150, 40, 0.22)" if r.check else ""] * len(r),
                            axis=1)
    table = st.data_editor(styled, key=f"mc_fields_{v}", hide_index=True, width="stretch",
                           height=min(38 + 35 * len(rf), 460), disabled=["source", "check"], column_config={
                               "source": st.column_config.TextColumn("Source column"),
                               "target": st.column_config.SelectboxColumn("Target", options=[
                                   *ingest.TARGET_OPTIONS, ingest.VALUE_MAP_TARGET], required=True),
                               "reason": st.column_config.TextColumn("Reason", width="medium"),
                               "confidence": st.column_config.SelectboxColumn(
                                   "Confidence", options=["", *CONFIDENCES]),
                               "feature": st.column_config.CheckboxColumn(
                                   "Feature", help="An extra feature of the generic feature set"),
                               "kind": st.column_config.SelectboxColumn("Kind", options=["", *ingest.FEATURE_KINDS]),
                               "name": st.column_config.TextColumn(
                                   "Feature name", help="Lower-case letters, digits and _; blank = from the column"),
                               "check": st.column_config.TextColumn("Check", width="small")})
    if low:
        ui.html(C.callout(", ".join(low), "warn", lead="Low confidence, check these rows:"))
    try:
        edited = ingest.apply_review(edited, table)
    except ValueError as e:
        ui.html(C.callout(str(e), "bad"))
    vm = ingest.value_map_frame(form)
    if not vm.empty:
        st.markdown("**Value maps** (read-only here; edit them as TOML below)")
        st.dataframe(vm, hide_index=True, width="stretch", height=min(38 + 35 * len(vm), 320))
    derived = ingest.derived_features_frame(form)
    if not derived.empty:
        st.markdown("**Derived extra features** (read-only here; edit them as TOML below)")
        st.dataframe(derived, hide_index=True, width="stretch", height=min(38 + 35 * len(derived), 320))
    return edited


def _toml_editor(edited: ingest.MappingForm, frames: dict[str, pd.DataFrame]) -> None:
    """The whole mapping as TOML text, editable; "Apply TOML" loads it into the form (``load_mapping`` checks it)."""
    text = ingest.form_toml(edited)
    with st.expander("Edit the whole mapping as TOML (value maps, derived expressions)"):
        st.caption("The schema is in emva/ingest/mapping.py. Applying replaces the form above.")
        start = text if text is not None else ("# The form is not complete yet; complete it above or paste a whole "
                                               "mapping here.\n")
        digest = hashlib.sha256(start.encode("utf-8")).hexdigest()[:12]
        new = st.text_area("Mapping TOML", start, height=320, key=f"mc_toml_{digest}", label_visibility="collapsed")
        if st.button("Apply TOML", icon=":material/check:", key="mc_toml_apply"):
            try:
                form = ingest.form_from_toml(new, frames)
            except ValueError as e:
                st.session_state[MC_MSG] = ("bad", str(e), "The TOML was not applied.")
            else:
                _set_form(form, "info", "Review it, then confirm.", lead="TOML applied.")
            st.rerun()


def _convert_step(root: Path, mapping: DatasetMapping | None, error: str | None,
                  frames: dict[str, pd.DataFrame]) -> None:
    """Confirm the outcome mapping, convert, and show the coverage card; the result goes to the check step."""
    ui.html(C.section("Confirm and convert", "The conversion is deterministic code: every value comes from a "
                      "mapped column or a documented rule, and anything the mapping does not cover is refused.",
                      step="1e"))
    if mapping is None:
        ui.html(C.callout(error or "", "warn", lead="The mapping is not complete:"))
        return
    toml = dump_mapping(mapping)
    up = st.file_uploader(f"Status-quo rules (optional) · {storage.RULES_FILE}", type=["json"], key="mc_rules",
                          help="Today's bucket values for this source, in the sample's format: saved with the dataset "
                               "so the standard report gets a status-quo row. Converted leads all have form variant "
                               f"{DEFAULT_FORM_VARIANT}, so base_value_by_form needs it.")
    rules = None
    if up is not None and up.name.lower().startswith(storage.FORBIDDEN_PREFIX):
        ui.html(C.callout(f"Refused {up.name}: evaluation-only ground-truth files are never accepted.", "bad"))
    elif up is not None:
        rules = up.getvalue()
    digest = hashlib.sha256(toml.encode("utf-8") + (rules or b"")).hexdigest()[:12]
    ok = st.checkbox("Outcome mapping confirmed: I checked which leads count as won and lost, and the dates.",
                     key=f"mc_confirm_{digest}")
    features_ok = True
    if mapping.features:
        ui.html(C.callout(", ".join(f"{f.name} ({f.kind})" for f in mapping.features), "info",
                          lead=f"{len(mapping.features)} extra feature(s) declared:"))
        # keyed by the features alone: any change to them (source, kind, name) clears the tick
        features_ok = st.checkbox("Extra features are known when the lead is submitted",
                                  key=f"mc_features_confirm_{ingest.features_signature(mapping.features)}",
                                  help="None of them is filled in or changed after the lead arrives (a column like "
                                       "that would leak the outcome into the model). Required to convert.")
    if st.button("Convert", type="primary", icon=":material/transform:", key="mc_convert",
                 disabled=not (ok and features_ok)):
        try:
            with st.spinner("Converting…"):
                conv = ingest.convert_raw(frames, ingest.confirm(mapping, features_confirmed=features_ok),
                                          datetime.now(timezone.utc).date().isoformat(), rules)
        except ValueError as e:
            st.session_state.pop(MC_CONV, None)
            ui.html(C.callout(str(e), "bad", lead="The conversion refused these files:"))
            return
        st.session_state[MC_CONV] = (digest, conv)
        st.session_state[VALIDATION] = conv.report
        st.session_state[VALIDATED_FILES] = conv.files
        st.session_state[VALIDATED_ORIGIN] = "convert"
    held = st.session_state.get(MC_CONV)
    if held is None:
        return
    if held[0] != digest:
        ui.html(C.callout("Convert again before saving.", "warn",
                          lead="The mapping or the rules changed since the conversion."))
        return
    _coverage(held[1].meta)


def _coverage(meta: dict) -> None:
    """The coverage card: how many of the model's signals the converted data fill."""
    cov = ingest.coverage_summary(meta)
    extras = ingest.extras_summary(meta)
    ui.html(C.section("Coverage", "Which of the model's input signals (design columns of the v2 features) the "
                      "converted leads fill. Unfilled signals are blank on every lead, so the model cannot use them."))
    cards = [
        C.kpi("Signals with data", f"{cov.data} of {cov.total}", "vary across leads",
              note=f"fills {cov.data} of {cov.total} signals"),
        C.kpi("Constant", str(cov.constant), "mapped, but the same on every lead"),
        C.kpi("Unfilled", str(cov.unfilled), "no source column mapped"),
        C.kpi("Leads", f"{meta['leads']:,}", ", ".join(f"{k} {v:,}" for k, v in meta["final_stage_counts"].items()),
              note=f"as_of {meta['as_of']} · test from {meta['test_from']}")]
    if extras.n:
        cards.append(C.kpi("Extra features", str(extras.n), f"{extras.numeric} numeric, {extras.categorical} "
                           "categorical", note="design columns: fixed at training"))
    ui.html(C.kpi_row(cards))
    if extras.n:
        ui.html(C.callout(f"{', '.join(extras.names)}. How many design columns they become is known only when a "
                          "model is trained with the generic feature set (v2 + extras): categorical levels with at "
                          "least 30 training leads plus other and missing, numeric values in 5 quantile bins plus "
                          "missing, all fitted on the training leads.", "info", lead="Extra features:"))
    derived = ingest.derived_counts(meta)
    if derived:
        st.markdown("**Derived during conversion** (counts in dataset.json)")
        st.text("\n".join(f"{n:,}  {what}" for _, n, what in derived))
    with st.expander("Coverage by signal"):
        st.dataframe(ingest.coverage_frame(meta), hide_index=True, width="stretch", column_config={
            "column": st.column_config.TextColumn("Signal (design column)"),
            "status": st.column_config.TextColumn("Status"),
            "inputs": st.column_config.TextColumn("Mapped inputs"),
            "leads_with_1": st.column_config.NumberColumn("Leads with it", format="%d"),
            "leads": st.column_config.NumberColumn("Leads", format="%d")})


def _map_section() -> None:
    """Step 1: raw CSVs -> column profile -> mapping (AI draft, saved, or by hand) -> review -> confirm -> convert."""
    ui.html(C.section("Map & convert a foreign export", "Bring 1-3 CSVs in any layout (a CRM, a Kaggle dataset). "
                      "Map their columns onto the lead schema, confirm the outcome, and convert them into the "
                      "model's format. Nothing is stored until you save.", step="01"))
    ui.html(C.section("Raw files", "The primary file has one row per lead; joined files add columns to it.",
                      step="1a"))
    root = ui.data_root()
    files, refused = _raw_uploads()
    if refused:
        ui.html(C.callout(f"Refused {', '.join(refused)}: evaluation-only ground-truth files are never accepted.",
                          "bad"))
    if not files:
        st.session_state.pop(MC_SIG, None)
        _reset_mapping()
        return
    sig = _signature(files)
    if st.session_state.get(MC_SIG) != sig:
        st.session_state[MC_SIG] = sig
        _reset_mapping()
    try:
        frames, profiles = _parsed(sig, files)
    except ValueError as e:
        ui.html(C.callout(str(e), "bad", lead="Cannot read the files."))
        return
    ui.html(C.section("Column profile", "What the drafter sees: per column its type, share missing, distinct "
                      "values and, for low-cardinality columns, redacted examples. Rows are never sent.", step="1b"))
    st.dataframe(ingest.profile_frame(profiles), hide_index=True, width="stretch", height=260, column_config={
        "missing": st.column_config.NumberColumn("Missing", format="percent"),
        "distinct": st.column_config.NumberColumn("Distinct", format="%d")})
    _start_buttons(root, frames, profiles)
    msg = st.session_state.get(MC_MSG)
    if msg:
        ui.html(C.callout(msg[1], msg[0], lead=msg[2]))
    form: ingest.MappingForm | None = st.session_state.get(MC_FORM)
    if form is None:
        return
    v = st.session_state.get(MC_VERSION, 0)
    ui.html(C.section("Review the mapping", f"Started from: {form.origin}. Check every target, the lead columns, "
                      "the outcome and the dates.", step="1d"))
    edited = _text_inputs(form, v)
    edited = _lead_and_outcome(form, edited, frames, v)
    edited = _field_table(form, edited, v)
    _toml_editor(edited, frames)
    try:
        mapping, error = ingest.mapping_from_form(edited), None
    except ValueError as e:
        mapping, error = None, str(e)
    _convert_step(root, mapping, error, frames)


# --- 02 files in EMVA's format, 03 check, 04 save ----------------------------------------------------------------

def _upload_section() -> None:
    """Step 2: upload files in EMVA's format and validate them."""
    ui.html(C.section("Or add exports in EMVA's format", "Already in the sample's format (data/v1)? Drop the files "
                      "in here instead.", step="02"))
    files, refused = _collect()
    if st.button("Validate files", type="primary", icon=":material/fact_check:", disabled=not (files or refused),
                 key="validate"):
        st.session_state[VALIDATION] = validate_files({**files, **{n: b"" for n in refused}})
        st.session_state[VALIDATED_FILES] = files
        st.session_state[VALIDATED_ORIGIN] = "upload"
        st.session_state["validated_refused"] = refused
    if st.session_state.get(VALIDATED_ORIGIN) == "upload" and st.session_state.get(VALIDATED_FILES) != files:
        ui.html(C.callout("Validate again before saving.", "warn", lead="The files changed since the last check."))
        st.session_state.pop(VALIDATION, None)


def _check_and_save() -> None:
    """Steps 3 and 4: the check results of the converted or uploaded files, then save them as a dataset."""
    report: ValidationReport | None = st.session_state.get(VALIDATION)
    if report is None:
        return
    files: dict[str, bytes] = st.session_state[VALIDATED_FILES]
    converted = st.session_state.get(VALIDATED_ORIGIN) == "convert"
    ui.html(C.section("Check results", ("The converted files, checked like any upload. " if converted else "")
                      + "Errors (red) must be fixed; notes (amber) are worth a look but do not block training.",
                      step="03"))
    _render_report(report, set(files) | set(st.session_state.get("validated_refused", []) if not converted else []))
    if not report.ok:
        ui.html(C.callout("before this can be saved.", "bad", lead=f"{len(report.errors)} problem(s) to fix"))
        return
    ui.html(C.section("Save as a dataset", "Give it a short name, e.g. crm-2026-09. Lower-case letters, digits, "
                      "- and _." + (" The confirmed mapping and dataset.json are saved with it." if converted
                                    else ""), step="04"))
    with st.form("save_dataset", border=False):
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        name = c1.text_input("Dataset name", placeholder="crm-2026-09", key="dataset_name")
        save = c2.form_submit_button("Save dataset", type="primary", icon=":material/save:", width="stretch")
    if save:
        try:
            ds = storage.save_dataset(ui.data_root(), name.strip(), files)
        except StorageError as e:
            ui.html(C.callout(str(e), "bad"))
        else:
            st.session_state["train_dataset"] = ds.name
            st.session_state["train_ds"] = ds.name  # the picker keeps its own state; point it at the new dataset
            for k in (VALIDATION, VALIDATED_FILES, VALIDATED_ORIGIN):
                st.session_state.pop(k, None)
            st.toast(f"Saved {ds.name}: {ds.n_leads:,} leads", icon=":material/check_circle:")
            st.rerun()


@st.cache_data(show_spinner="Counting training and test leads…")
def _summary(dataset_path: str, label_mode: str, feature_set: str, _stamp: str) -> training.TrainingSummary:
    """Cached ``pre_training_summary`` (``_stamp`` = dataset creation time, so a replaced dataset recomputes)."""
    return training.pre_training_summary(dataset_path, training.TrainingConfig(label_mode, feature_set))


def _train_section() -> None:
    """Step 4: choose a dataset and options, see what training will use, train and follow the log."""
    ui.html(C.section("Train a model", "Runs the same command as a local training run and keeps every run, so "
                      "you can compare them on the results page.", step="05"))
    root = ui.data_root()
    datasets = storage.list_datasets(root)
    names = [d.name for d in datasets]
    by_name = {d.name: d for d in datasets}
    wanted = st.session_state.get("train_dataset")
    c1, c2, c3 = st.columns([3, 2, 3], gap="medium")
    with c1:
        ds_name = st.selectbox("Dataset", names, index=names.index(wanted) if wanted in names else 0, key="train_ds",
                               format_func=lambda n: f"{n} · {by_name[n].n_leads:,} leads"
                               + (" · bundled sample, read-only" if by_name[n].read_only else ""))
    with c2:
        label_mode = st.segmented_control("Labels", list(LABEL_MODES), default="horizon", required=True,
                                          format_func=LABEL_MODES.get, key="label_mode")
    ds = by_name[ds_name]
    options = training.feature_set_options(ds)
    with c3:  # keyed per offer, so a choice of generic does not outlive a switch to a dataset without extras
        feature_set = st.segmented_control("Features", options, default="v2", required=True,
                                           format_func=FEATURE_SETS.get, key=f"feature_set_{len(options)}")
    st.caption("Recommended: horizon labels and v2 features. Horizon labels count a lead as won only if it closed "
               "within 120 days, and leave out leads too young to know. Legacy reproduces the frozen proof of "
               "concept, which counted slow or neglected leads as lost and let missing data fall into the "
               "reference level."
               + (" This dataset declares extra features: generic trains on v2 plus them, and its results compare "
                  "it with v2 on the same leads." if ds.has_extras else ""))
    try:
        s = _summary(ds.path, label_mode, feature_set, ds.created_at)
    except ValueError as e:
        ui.html(C.callout(str(e), "bad", lead="This dataset cannot be trained as it is."))
        return
    extra_stats = [("Extra features", f"{len(s.extras)} ({sum(k == 'numeric' for _, k in s.extras)} numeric)")] \
        if s.extras else []
    ui.html(C.stats([("Leads after bot/duplicate removal", f"{s.scored:,}"),
                     ("Training leads (won)", f"{s.train:,} ({s.train_wins:,})"),
                     ("Test leads (won)", f"{s.test:,} ({s.test_wins:,})"),
                     ("Mature test set (won)", f"{s.mature_test:,} ({s.mature_test_wins:,})"), *extra_stats]))
    with st.expander("Labels by CRM state (label_source)"):
        t = s.label_counts_frame()
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "label_source": st.column_config.TextColumn("CRM state at the snapshot"),
            "leads": st.column_config.NumberColumn("Leads", format="%d"),
            "wins": st.column_config.NumberColumn("Labelled won", format="%d"),
            "losses": st.column_config.NumberColumn("Labelled lost", format="%d"),
            "unlabelled": st.column_config.NumberColumn("Left out", format="%d")})
        st.caption("won = closed won; crm_lost = marked lost in the CRM; stalled = no stage change for 90+ days "
                   "(left out by horizon labels); ghosted = never contacted within the horizon; open = still "
                   "in progress.")
    if s.train == 0 or s.train_wins == 0:
        ui.html(C.callout("The model cannot be trained on this dataset with these labels.", "bad",
                          lead="No labelled training leads with wins."))
        return
    if st.button("Train model", type="primary", icon=":material/play_arrow:", key="train"):
        cfg = training.TrainingConfig(label_mode, feature_set)
        run = storage.register_run(root, storage.new_run(root, ds, cfg.as_dict()))
        st.session_state.setdefault(PROCS_KEY, {})[run.run_id] = training.start_training(root, run,
                                                                                         python=sys.executable)
        st.session_state[ACTIVE_KEY] = run.run_id
    active = st.session_state.get(ACTIVE_KEY)
    if active:
        _progress(root, active)
    _history(root)


def _tail(path: str) -> str:
    """The last ``LOG_TAIL_LINES`` lines of the log (empty before it exists)."""
    p = Path(path)
    if not p.exists():
        return ""
    return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_TAIL_LINES:])


@st.fragment(run_every=1.5)
def _live(root: Path, run_id: str) -> None:
    """Re-renders every 1.5 s while the run is active: status pill and log tail; stops polling when done."""
    run = storage.get_run(root, run_id)
    proc = st.session_state.get(PROCS_KEY, {}).get(run_id)
    if proc is not None and proc.poll() is not None:
        st.session_state[PROCS_KEY].pop(run_id)
    ui.html(C.run_header(run.status, run.run_id))
    st.code(_tail(run.log_path) or "Starting…", language="text", height=280, wrap_lines=True)
    if run.is_done:
        st.rerun(scope="app")


def _progress(root: Path, run_id: str) -> None:
    """The active run: live log while running; outcome and a link to its results when done."""
    run = storage.get_run(root, run_id)
    if not run.is_done:
        _live(root, run_id)
        return
    ui.html(C.run_header(run.status, run.run_id))
    if run.status == "succeeded":
        auc = run.metrics.get("auc")
        ui.html(C.callout(f"Test AUC {auc:.3f} on this run's own test leads." if auc is not None else "", "info",
                          lead="Model trained."))
        if st.button("Open results", type="primary", icon=":material/insights:", key="open_results"):
            st.session_state[ui.SELECTED_RUN_KEY] = run.run_id
            st.switch_page("views/results.py")
    else:
        ui.html(C.callout(run.error or "", "bad", lead="Training failed."))
    with st.expander("Training log"):
        st.code(_tail(run.log_path), language="text", wrap_lines=True)


def _history(root: Path) -> None:
    """Recent runs as a compact table."""
    runs = storage.list_runs(root)
    if not runs:
        return
    ui.html(C.section("Recent runs", "Newest first. Open any finished run on the results page."))
    t = pd.DataFrame([{"started": r.created_at[:16].replace("T", " "), "dataset": r.dataset,
                       "labels": r.args.get("label_mode"), "features": r.args.get("feature_set"),
                       "status": r.status, "auc": r.metrics.get("auc")} for r in runs[:15]])
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "started": st.column_config.TextColumn("Started (UTC)"), "dataset": st.column_config.TextColumn("Dataset"),
        "labels": st.column_config.TextColumn("Labels"), "features": st.column_config.TextColumn("Features"),
        "status": st.column_config.TextColumn("Status"),
        "auc": st.column_config.NumberColumn("Test AUC", format="%.3f")})


def render() -> None:
    """The page."""
    ui.html(C.page_header("Upload & train", "Bring your CRM data",
                          "Convert an export in any layout (or upload one in the model's format), check it against "
                          "what the model needs, keep it as a named dataset, and train a model on it (or on the "
                          "bundled sample) with live progress."))
    _map_section()
    _upload_section()
    _check_and_save()
    _train_section()


render()
