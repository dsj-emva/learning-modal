"""Page: Score a lead. Fill in one lead's form answers and session, and see what the model makes of it and why. For a
model trained on a converted dataset, leads can also be given in the source's own format (a form or a CSV)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app import charts, scoring, storage, ui
from app import components as C
from app.scoring import FormField
from emva.generic import OTHER
from emva.ingest.mapping import DatasetMapping
from emva.model import INTERCEPT
from emva.persist import BUNDLE_FILE, ModelBundle
from emva.scoring import FieldType, UnknownLevelError

LONG_TEXT = {"what_to_solve", "user_agent", "landing_url"}
MODES = {"emva": "EMVA form", "source": "Source format"}


def _widget(f: FormField, default: object, key: str) -> object:
    """The input widget for one field; returns its value (None = blank)."""
    label = f.label + ("" if not f.spec.required else " *")
    if f.options is not None:
        opts = list(f.options)
        index = opts.index(default) if default in opts else 0
        return st.selectbox(label, opts, index=index, key=key, help=f.spec.description,
                            format_func=lambda o: "— not answered —" if o == "" else o)
    if f.spec.dtype is FieldType.BOOL:
        return st.toggle(label, value=bool(default), key=key, help=f.spec.description)
    if f.spec.dtype.is_numeric:
        is_int = f.spec.dtype is FieldType.INT
        value = f.spec.coerce(default)
        return st.number_input(label, value=value, min_value=0 if is_int else 0.0, step=1 if is_int else 1.0,
                               format="%d" if is_int else "%.0f", key=key, help=f.spec.description,
                               placeholder="blank")
    if f.name == "what_to_solve":
        return st.text_area(label, value=str(default or ""), key=key, help=f.spec.description, height=96)
    return st.text_input(label, value=str(default or ""), key=key, help=f.spec.description)


def _form(defaults: dict[str, object], form_key: str) -> dict[str, object] | None:
    """The lead form, grouped by section; returns the values when submitted. ``form_key`` names the widgets, so a
    different starting lead gets fresh widgets."""
    values: dict[str, object] = {}
    with st.form(f"lead_form_{form_key}", border=False):
        for title, fields in scoring.form_sections().items():
            ui.html(C.section(title))
            if title == "Session":
                st.caption("Filled with a typical visit. Leave any of these blank and the model treats the visitor "
                           "as having no on-site session at all, which changes the score.")
            cols = st.columns(2, gap="medium")
            for i, f in enumerate(fields):
                target = st.container() if f.name in LONG_TEXT else cols[i % 2]
                with target:
                    values[f.name] = _widget(f, defaults.get(f.name, f.default), key=f"f_{form_key}_{f.name}")
        submitted = st.form_submit_button("Score this lead", type="primary", icon=":material/target:",
                                          width="stretch")
    return values if submitted else None


def _result(res: scoring.LeadScore, base: float, transform: str, batch: bool = False) -> None:
    """The result card, then the points breakdown (``batch``: the lead was scored with others from a file)."""
    flags = [C.pill("Looks like a bot", "bad") if res.is_bot else C.pill("Not a bot", "ok")]
    if batch:
        flags.append(C.pill("Duplicate within the file", "warn") if res.is_duplicate else
                     C.pill("No duplicate within the file", "ok"))
    else:
        flags.append(C.pill("Scored alone: duplicates are checked in batches", "info"))
    ui.html(C.result_card(res.p, base, res.deal_value, res.value_at_submit, res.value_formula, transform,
                          "".join(flags)))
    if res.blank_session:
        ui.html(C.callout(f"Blank: {', '.join(res.blank_session)}. The model scored this lead as having no on-site "
                          "session (session_missing).", "warn", lead="No session data."))
    if res.is_bot:
        ui.html(C.callout("Under 15 seconds on the page or a headless browser: the batch pipeline would drop this "
                          "lead before training and scoring. It is scored here anyway.", "warn"))
    pts = res.points
    icpt = pts[pts.feature == INTERCEPT]
    rest = pts[pts.feature != INTERCEPT].sort_values("points", ascending=False, kind="stable")
    ui.html(C.section("Why this score", "Every signal that moved this lead away from the starting score. Signals at "
                                        "their comparison level add nothing and are not listed."))
    ui.html(C.stats([("Starting score", C.points(float(icpt.points.iloc[0])) + " pts"),
                     ("Signals", C.points(float(rest.points.sum())) + " pts"),
                     ("Total", C.points(float(pts.points.sum())) + " pts")]))
    labels = [f"{lbl} · {lvl}" for lbl, lvl in zip(rest.label, rest.level)]
    st.plotly_chart(charts.points_chart(labels, rest.points.tolist()), config=charts.CONFIG, theme=None, width="stretch",
                    key="lead_points")


def _option_label(o: str) -> str:
    """A source-form option: "" is a blank cell; ``other`` stands for any value the model was not trained with."""
    return "— blank —" if o == "" else "other (any value not listed)" if o == OTHER else o


def _source_form(mapping: DatasetMapping, bundle: ModelBundle, run_id: str) -> dict[str, object] | None:
    """The source-format form: only the raw columns the mapping reads at submit time (a generic bundle's extras as a
    number input or their training levels); returns values on submit."""
    values: dict[str, object] = {}
    with st.form(f"source_form_{run_id}", border=False):
        cols = st.columns(2, gap="medium")
        for i, f in enumerate(scoring.source_fields(mapping, bundle)):
            with cols[i % 2]:
                label, key, help_text = f.column, f"src_{run_id}_{f.key}", f"{f.file} · feeds {f.feeds}"
                if f.numeric:
                    values[f.key] = scoring.number_text(st.number_input(label, value=None, key=key, help=help_text,
                                                                        placeholder="blank", format="%g"))
                elif f.options is not None:
                    values[f.key] = st.selectbox(label, ["", *f.options], key=key, help=help_text,
                                                 format_func=_option_label)
                else:
                    values[f.key] = st.text_input(label, key=key, help=help_text)
        submitted = st.form_submit_button("Score this lead", type="primary", icon=":material/target:",
                                          width="stretch", key=f"src_submit_{run_id}")
    return values if submitted else None


def _source_mode(run: storage.Run, bundle: ModelBundle, mapping: DatasetMapping) -> None:
    """Leads in the source format: a one-lead form or a CSV, converted with the dataset's mapping and scored."""
    state = f"source_{run.run_id}"
    left, right = st.columns([3, 2], gap="large")
    frames = None
    with left:
        st.caption(f"Leads as {mapping.name} sends them, converted with the dataset's confirmed mapping "
                   "(emva.ingest.convert_leads) and scored with the same model. Outcome columns are not needed.")
        one, many = st.tabs(["One lead", "Upload a CSV"])
        try:
            with one:
                values = _source_form(mapping, bundle, run.run_id)
                if values is not None:
                    frames = scoring.frames_from_source_form(mapping, values)
            with many:
                files = st.file_uploader("New leads in the source format", type=["csv"], accept_multiple_files=True,
                                         key=f"src_upload_{run.run_id}",
                                         help=f"The primary file {mapping.sources[0].file} (any name if it is the "
                                              "only file), plus any joined file the mapping reads at submit time.")
                refused = [f.name for f in files or [] if f.name.lower().startswith(storage.FORBIDDEN_PREFIX)]
                if refused:
                    ui.html(C.callout(f"Refused {', '.join(refused)}: ground-truth files are never accepted.", "bad"))
                elif st.button("Score these leads", type="primary", icon=":material/target:", disabled=not files,
                               key=f"src_score_file_{run.run_id}"):
                    frames = scoring.frames_from_source_uploads(mapping, {f.name: f.getvalue() for f in files})
            if frames is not None:
                st.session_state[state] = scoring.score_source(bundle, mapping, frames, run.dataset_path)
        except ValueError as e:  # includes UnknownLevelError and convert_leads' refusals
            st.session_state.pop(state, None)
            ui.html(C.callout(str(e), "bad", lead="These leads cannot be scored:"))
    with right:
        result: scoring.SourceScores | None = st.session_state.get(state)
        if result is None:
            ui.html(C.empty_state("Ready when you are", "Fill in a lead as the source sends it, or upload a CSV of "
                                                        "new leads, and score it."))
            return
        table = result.table()
        st.dataframe(table, hide_index=True, width="stretch", height=min(38 + 35 * len(table), 250), column_config={
            "lead_id": st.column_config.TextColumn("Lead"),
            "p": st.column_config.NumberColumn("P(close)", format="percent"),
            "deal_value": st.column_config.NumberColumn("Deal value if won", format="£%.0f"),
            "value_at_submit": st.column_config.NumberColumn("Value sent", format="£%.0f"),
            "value_formula": st.column_config.NumberColumn("p × value", format="£%.0f"),
            "is_bot": st.column_config.CheckboxColumn("Bot"), "is_duplicate": st.column_config.CheckboxColumn("Dup")})
        st.download_button("Scores CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"{run.run_id}-source-scores.csv", mime="text/csv", icon=":material/download:",
                           key=f"src_dl_{run.run_id}")
        ids = table.lead_id.tolist()
        lead_id = st.selectbox("Why this score: lead", ids, key=f"src_pick_{run.run_id}") if len(ids) > 1 else ids[0]
        res = scoring.source_lead_score(bundle, result, lead_id, run.dataset_path)
        _result(res, ui.base_rate(run.run_id, run.out_dir, run.dataset_path), scoring.describe_transform(bundle),
                batch=len(ids) > 1)


def render() -> None:
    """The page."""
    ui.html(C.page_header("Score a lead", "What is this lead worth?",
                          "Enter one lead as it arrives from the form. The trained model returns its chance of "
                          "closing, the value it would send to the ad platforms, and the signals behind it."))
    root = ui.data_root()
    runs = storage.list_runs(root, ui.has_bundle)
    if not runs:
        ui.no_runs_state("Scoring needs a trained model. Train one on the Upload & train page, then come back.")
        return
    run = ui.pick_run(runs, key="score_run", label="Model (completed runs)")
    bundle = ui.bundle(run.run_id, str(Path(run.out_dir) / BUNDLE_FILE))
    try:
        mapping = scoring.run_mapping(root, run.dataset)
    except ValueError as e:
        mapping = None
        ui.html(C.callout(str(e), "warn", lead="The dataset's mapping.toml cannot be read; source format is off."))
    if mapping is not None:
        mode = st.segmented_control("Lead format", list(MODES), default="emva", required=True,
                                    format_func=MODES.get, key=f"score_mode_{run.run_id}",
                                    help="This model was trained on a converted dataset: leads can be given in the "
                                         "source's own format too.")
        if mode == "source":
            _source_mode(run, bundle, mapping)
            return
    extras = scoring.extra_names(bundle)
    if extras:
        ui.html(C.callout(f"This model also reads {len(extras)} extra feature(s) ({', '.join(extras)}), which the EMVA "
                          "form has no inputs for, so a lead scored here has every extra set to missing."
                          + (" Switch the lead format to Source format to give them." if mapping is not None else ""),
                          "info", lead="Extra features are scored as missing."))
    left, right = st.columns([3, 2], gap="large")
    with left:
        lead_id = st.text_input("Start from a lead in the training data (optional)", key=f"prefill_{run.run_id}",
                                placeholder="lead_id, e.g. L00042",
                                help="Prefills the form with that lead's answers and session, to see how the model "
                                     "scored a real submission.").strip()
        defaults = scoring.defaults_for(run.dataset_path)
        if lead_id:
            row = ui.raw_lead(run.dataset_path, lead_id)
            if row is None:
                ui.html(C.callout("Showing the example lead instead.", "warn", lead=f"No lead {lead_id} in this dataset."))
            else:
                defaults = scoring.values_from_lead(row)
        values = _form(defaults, f"{run.run_id}_{lead_id or 'example'}")
    with right:
        if values is None and f"score_{run.run_id}" not in st.session_state:
            ui.html(C.empty_state("Ready when you are",
                                  "Adjust the example lead on the left and press Score this lead."))
            return
        if values is not None:
            try:
                st.session_state[f"score_{run.run_id}"] = scoring.score_form(bundle, values, run.dataset_path)
            except UnknownLevelError as e:
                st.session_state.pop(f"score_{run.run_id}", None)
                ui.html(C.callout(f"{e}. Choose one of the allowed values, or retrain on data that contains it.",
                                  "bad", lead="This model has not seen one of these values."))
                return
            except ValueError as e:
                st.session_state.pop(f"score_{run.run_id}", None)
                ui.html(C.callout(str(e), "bad", lead="Check the form."))
                return
        res = st.session_state[f"score_{run.run_id}"]
        _result(res, ui.base_rate(run.run_id, run.out_dir, run.dataset_path), scoring.describe_transform(bundle))


render()
