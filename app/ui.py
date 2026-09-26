"""Streamlit helpers shared by the pages: the data root, cached loaders, run pickers and HTML rendering.

Caching: evaluation frames per (run, test set) with ``st.cache_data``; the model bundle per run with
``st.cache_resource`` (one unpickled object per process, shared by sessions; bundles are read-only).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app import components as C
from app import results, storage
from app.storage import Run
from emva.persist import BUNDLE_FILE, ModelBundle, load_bundle

DATA_DIR_ENV: str = "DATA_DIR"
DEFAULT_DATA_DIR: str = "runs/app"
SELECTED_RUN_KEY: str = "selected_run"


def html(markup: str) -> None:
    """Render trusted component HTML (``app.components`` escapes all user text)."""
    st.markdown(markup, unsafe_allow_html=True)


def data_root() -> Path:
    """The data root from ``DATA_DIR`` (default ``runs/app``), created with its layout on first use."""
    return storage.init_root(os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR))


def run_label(run: Run) -> str:
    """One line for a run picker: date, dataset, options, AUC, status."""
    when = run.created_at[:16].replace("T", " ")
    auc = f"AUC {run.metrics['auc']:.3f}" if run.metrics.get("auc") is not None else "no AUC yet"
    opts = f"{run.args.get('label_mode', '?')} labels · {run.args.get('feature_set', '?')} features"
    return f"{when} · {run.dataset} · {opts} · {auc} · {run.status}"


def pick_run(runs: list[Run], key: str, label: str = "Run") -> Run:
    """A select box over ``runs`` (newest first) preselecting the session's selected run; returns the choice."""
    ids = [r.run_id for r in runs]
    wanted = st.session_state.get(SELECTED_RUN_KEY)
    index = ids.index(wanted) if wanted in ids else 0
    by_id = {r.run_id: r for r in runs}
    choice = st.selectbox(label, ids, index=index, key=key, format_func=lambda i: run_label(by_id[i]))
    st.session_state[SELECTED_RUN_KEY] = choice
    return by_id[choice]


def has_bundle(run: Run) -> bool:
    """True when the run succeeded and wrote ``model.joblib``."""
    return run.status == "succeeded" and (Path(run.out_dir) / BUNDLE_FILE).exists()


@st.cache_resource(show_spinner=False)
def bundle(run_id: str, path: str) -> ModelBundle:
    """The run's model bundle, loaded once per process (keyed by run id and path)."""
    return load_bundle(path)


@st.cache_data(show_spinner=False)
def leads(dataset_path: str) -> pd.DataFrame:
    """``emva.io.load`` of a dataset (cached per path)."""
    from emva.io import load
    return load(dataset_path)


@st.cache_data(show_spinner="Evaluating on the frozen test set…")
def evaluation(run_id: str, out_dir: str, dataset_path: str, test_set: str) -> dict[str, object] | None:
    """Every results-page frame for one run and test set (None when the test set cannot be evaluated)."""
    ev = results.evaluate(out_dir, dataset_path, test_set, leads=leads(dataset_path))
    if ev is None:
        return None
    head, paired = results.standard_table(ev)
    return {"headline": head, "paired": paired, "notes": ev.notes, "baseline_missing": ev.baseline_missing,
            "calibration": results.calibration(ev),
            "auc_month": results.auc_month(ev), "base_rate": ev.base_rate, "n": len(ev.test.y),
            "wins": int(ev.test.y.sum())}


@st.cache_data(show_spinner=False)
def _raw_leads(dataset_path: str) -> pd.DataFrame:
    """The dataset's ``historical_leads.csv`` as ``emva.io.read_leads`` reads it (cached per path)."""
    from emva.io import read_leads
    return read_leads(Path(dataset_path) / storage.LEADS_FILE)


def raw_lead(dataset_path: str, lead_id: str) -> pd.Series | None:
    """One raw ``historical_leads.csv`` row, or None when the dataset has no such lead."""
    leads_ = _raw_leads(dataset_path)
    return leads_.loc[lead_id] if lead_id in leads_.index else None


@st.cache_data(show_spinner=False)
def base_rate(run_id: str, out_dir: str, dataset_path: str) -> float:
    """Win rate of the run's training labels (``results.training_base_rate``)."""
    return results.training_base_rate(results.load_scores(out_dir), _raw_leads(dataset_path).created_at)


def no_runs_state(message: str) -> None:
    """The empty state shown when there is nothing to display yet, with a link to the upload page."""
    html(C.empty_state("Nothing trained yet", message))
    st.page_link("views/upload.py", label="Upload data and train a model", icon=":material/arrow_forward:")


__all__ = ["bundle", "base_rate", "data_root", "evaluation", "has_bundle", "html", "no_runs_state", "pick_run", "raw_lead",
           "run_label"]
