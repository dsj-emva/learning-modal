"""Page: Upload & train. Validate CRM exports, save them as a dataset, and train a model on any dataset."""
from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from app import charts, storage, training, ui
from app import components as C
from app.storage import StorageError
from app.validation import ValidationReport, validate_files

SLOTS: list[tuple[str, str, str]] = [
    (storage.LEADS_FILE, "Leads", "One row per form submission, with the answers JSON and on-site session."),
    (storage.CRM_FILE, "CRM history", "Every stage change per lead (New, Contacted, ..., Won/Lost) with deal value."),
    (storage.COMPANIES_FILE, "Companies", "Firmographics joined on the company domain."),
    (storage.PEOPLE_FILE, "People", "Contact enrichment. Optional: the model never reads it."),
    (storage.RULES_FILE, "Status-quo rules", "Optional JSON: today's bucket values, for the benchmark."),
]
LABEL_MODES = {"horizon": "Horizon", "legacy": "Legacy (POC)"}
FEATURE_SETS = {"v2": "v2", "legacy": "Legacy (POC)"}
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
                ui.html(C.callout(f"<b>{escape(up.name)}</b> belongs in its own slot, not here.", "warn"))
            else:
                files[name] = up.getvalue()
    return files, refused


def _render_report(report: ValidationReport, provided: set[str]) -> None:
    """Per-file cards with errors and warnings, then the preview summary."""
    names = [n for n, _, _ in SLOTS] + sorted({i.file for i in report.errors} - {n for n, _, _ in SLOTS})
    labels = {n: lbl for n, lbl, _ in SLOTS}
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


def _upload_section() -> None:
    """Step 1 and 2: upload, validate, save."""
    ui.html(C.section("Add your CRM exports", "Drop in the exports; nothing is stored until you validate and save. "
                      "Column names must match the sample data (data/v1).", step="01"))
    files, refused = _collect()
    provided = set(files)
    if st.button("Validate files", type="primary", icon=":material/fact_check:", disabled=not (files or refused),
                 key="validate"):
        st.session_state["validation"] = validate_files({**files, **{n: b"" for n in refused}})
        st.session_state["validated_files"] = files
    report: ValidationReport | None = st.session_state.get("validation")
    if report is None:
        return
    if st.session_state.get("validated_files") != files:
        ui.html(C.callout("The files changed since the last check. <b>Validate again</b> before saving.", "warn"))
        return
    ui.html(C.section("Check results", "Errors (red) must be fixed in the export; notes (amber) are worth a look "
                      "but do not block training.", step="02"))
    _render_report(report, provided | set(refused))
    if not report.ok:
        ui.html(C.callout(f"<b>{len(report.errors)} problem(s) to fix</b> before this can be saved.", "bad"))
        return
    ui.html(C.section("Save as a dataset", "Give it a short name, e.g. crm-2026-09. Lower-case letters, digits, "
                      "- and _.", step="03"))
    with st.form("save_dataset", border=False):
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        name = c1.text_input("Dataset name", placeholder="crm-2026-09", key="dataset_name")
        save = c2.form_submit_button("Save dataset", type="primary", icon=":material/save:", width="stretch")
    if save:
        try:
            ds = storage.save_dataset(ui.data_root(), name.strip(), files)
        except StorageError as e:
            ui.html(C.callout(str(e).replace("<", "&lt;"), "bad"))
        else:
            st.session_state["train_dataset"] = ds.name
            st.session_state.pop("validation", None)
            st.toast(f"Saved {ds.name}: {ds.n_leads:,} leads", icon=":material/check_circle:")
            st.rerun()


@st.cache_data(show_spinner="Counting training and test leads…")
def _summary(dataset_path: str, label_mode: str, feature_set: str, _stamp: str) -> dict[str, object]:
    """Cached ``pre_training_summary`` (``_stamp`` = dataset creation time, so a replaced dataset recomputes)."""
    return training.pre_training_summary(dataset_path, training.TrainingConfig(label_mode, feature_set))


def _train_section() -> None:
    """Step 4: choose a dataset and options, see what training will use, train and follow the log."""
    ui.html(C.section("Train a model", "Runs the same command as a local training run and keeps every run, so "
                      "you can compare them on the results page.", step="04"))
    root = ui.data_root()
    datasets = storage.list_datasets(root)
    names = [d.name for d in datasets]
    by_name = {d.name: d for d in datasets}
    wanted = st.session_state.get("train_dataset")
    c1, c2, c3 = st.columns([3, 2, 2], gap="medium")
    with c1:
        ds_name = st.selectbox("Dataset", names, index=names.index(wanted) if wanted in names else 0, key="train_ds",
                               format_func=lambda n: f"{n} · {by_name[n].n_leads:,} leads"
                               + (" · bundled sample, read-only" if by_name[n].read_only else ""))
    with c2:
        label_mode = st.segmented_control("Labels", list(LABEL_MODES), default="horizon", required=True,
                                          format_func=LABEL_MODES.get, key="label_mode")
    with c3:
        feature_set = st.segmented_control("Features", list(FEATURE_SETS), default="v2", required=True,
                                           format_func=FEATURE_SETS.get, key="feature_set")
    st.caption("Recommended: horizon labels and v2 features. Horizon labels count a lead as won only if it closed "
               "within 120 days, and leave out leads too young to know. Legacy reproduces the frozen proof of "
               "concept, which counted slow or neglected leads as lost and let missing data fall into the "
               "reference level.")
    ds = by_name[ds_name]
    try:
        s = _summary(ds.path, label_mode, feature_set, ds.created_at)
    except ValueError as e:
        ui.html(C.callout(f"<b>This dataset cannot be trained as it is.</b> {str(e).replace('<', '&lt;')}", "bad"))
        return
    ui.html(C.stats([("Leads after bot/duplicate removal", f"{s['scored']:,}"),
                     ("Training leads (won)", f"{s['train']:,} ({s['train_wins']:,})"),
                     ("Test leads (won)", f"{s['test']:,} ({s['test_wins']:,})"),
                     ("Mature test set (won)", f"{s['mature_test']:,} ({s['mature_test_wins']:,})")]))
    with st.expander("Labels by CRM state (label_source)"):
        t = training.label_counts_frame(s)
        st.dataframe(t, hide_index=True, width="stretch", column_config={
            "label_source": st.column_config.TextColumn("CRM state at the snapshot"),
            "leads": st.column_config.NumberColumn("Leads", format="%d"),
            "wins": st.column_config.NumberColumn("Labelled won", format="%d"),
            "losses": st.column_config.NumberColumn("Labelled lost", format="%d"),
            "unlabelled": st.column_config.NumberColumn("Left out", format="%d")})
        st.caption("won = closed won; crm_lost = marked lost in the CRM; stalled = no stage change for 90+ days "
                   "(left out by horizon labels); ghosted = never contacted within the horizon; open = still "
                   "in progress.")
    if s["train"] == 0 or s["train_wins"] == 0:
        ui.html(C.callout("<b>No labelled training leads with wins.</b> The model cannot be trained on this "
                          "dataset with these labels.", "bad"))
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
    run = training.reconcile(root, run) if proc is None else run
    ui.html(f'<div style="display:flex;gap:12px;align-items:center;margin:16px 0 8px">{C.status_pill(run.status)}'
            f'<span style="font-family:var(--mono);font-size:13px;color:var(--muted)">run {run.run_id}</span></div>')
    st.code(_tail(run.log_path) or "Starting…", language="text", height=280)
    if run.is_done:
        st.rerun(scope="app")


def _progress(root: Path, run_id: str) -> None:
    """The active run: live log while running; outcome and a link to its results when done."""
    run = storage.get_run(root, run_id)
    if not run.is_done:
        _live(root, run_id)
        return
    ui.html(f'<div style="display:flex;gap:12px;align-items:center;margin:16px 0 8px">{C.status_pill(run.status)}'
            f'<span style="font-family:var(--mono);font-size:13px;color:var(--muted)">run {run.run_id}</span></div>')
    if run.status == "succeeded":
        auc = run.metrics.get("auc")
        ui.html(C.callout(f"<b>Model trained.</b> Test AUC {auc:.3f} on this run's own test leads."
                          if auc is not None else "<b>Model trained.</b>", "info"))
        if st.button("Open results", type="primary", icon=":material/insights:", key="open_results"):
            st.session_state[ui.SELECTED_RUN_KEY] = run.run_id
            st.switch_page("views/results.py")
    else:
        ui.html(C.callout(f"<b>Training failed.</b> {run.error or ''}", "bad"))
    with st.expander("Training log"):
        st.code(_tail(run.log_path), language="text")


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
                          "Check your exports against what the model needs, keep them as a named dataset, and train "
                          "a model on it (or on the bundled sample) with live progress."))
    _upload_section()
    _train_section()


render()
