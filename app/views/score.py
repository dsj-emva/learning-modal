"""Page: Score a lead. Fill in one lead's form answers and session, and see what the model makes of it and why."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from app import charts, scoring, storage, ui
from app import components as C
from app.scoring import FormField
from emva.persist import BUNDLE_FILE
from emva.scoring import UnknownLevelError, is_blank

LONG_TEXT = {"what_to_solve", "user_agent", "landing_url"}


def _widget(f: FormField, default: object, key: str) -> object:
    """The input widget for one field; returns its value (None = blank)."""
    label = f.label + ("" if not f.spec.required else " *")
    if f.options is not None:
        opts = list(f.options)
        index = opts.index(default) if default in opts else 0
        return st.selectbox(label, opts, index=index, key=key, help=f.spec.description,
                            format_func=lambda o: "— not answered —" if o == "" else o)
    if f.spec.dtype == "bool":
        return st.toggle(label, value=bool(default), key=key, help=f.spec.description)
    if f.spec.dtype in ("int", "float"):
        is_int = f.spec.dtype == "int"
        value = None if is_blank(default) else (int(default) if is_int else float(default))
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


def _result(res: scoring.LeadScore, base: float, transform: str) -> None:
    """The result card, then the points breakdown."""
    flags = [C.pill("Looks like a bot", "bad") if res.is_bot else C.pill("Not a bot", "ok")]
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
    icpt = pts[pts.feature == "(intercept)"]
    rest = pts[pts.feature != "(intercept)"].sort_values("points", ascending=False, kind="stable")
    ui.html(C.section("Why this score", "Every signal that moved this lead away from the starting score. Signals at "
                                        "their comparison level add nothing and are not listed."))
    ui.html(C.stats([("Starting score", C.points(float(icpt.points.iloc[0])) + " pts"),
                     ("Signals", C.points(float(rest.points.sum())) + " pts"),
                     ("Total", C.points(float(pts.points.sum())) + " pts")]))
    labels = [f"{lbl} · {lvl}" for lbl, lvl in zip(rest.label, rest.level)]
    st.plotly_chart(charts.points_chart(labels, rest.points.tolist()), config=charts.CONFIG, theme=None, width="stretch",
                    key="lead_points")


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
