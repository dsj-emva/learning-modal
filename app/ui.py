"""Streamlit helpers shared by the pages: the data root, cached loaders, run pickers and HTML rendering.

Caching: evaluation frames per (run, test set) and the Score page's example lead per run with ``st.cache_data``; the
model bundle per run with ``st.cache_resource`` (one unpickled object per process, shared by sessions; bundles are
read-only).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app import components as C
from app import results, scoring, storage
from app.storage import Run
from emva.dataset_meta import dataset_currency, dataset_dates
from emva.features import FeatureSet
from emva.labels import LabelMode
from emva.persist import BUNDLE_FILE, ModelBundle, load_bundle

DATA_DIR_ENV: str = "DATA_DIR"
DEFAULT_DATA_DIR: str = "runs/app"
SELECTED_RUN_KEY: str = "selected_run"
# Prefix of the plain session keys that mirror widget values across page switches (``restore`` / ``keep``).
KEPT_PREFIX: str = "kept:"


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


@st.cache_data(show_spinner="Training a v2 model on the same leads for the comparison…")
def generic_section(run_id: str, out_dir: str, dataset_path: str, label_mode: str,
                    test_set: str) -> results.GenericSection | None:
    """``results.generic_comparison`` for a generic-feature-set run and test set (cached per run and test set)."""
    return results.generic_comparison(out_dir, dataset_path, label_mode, test_set)


@st.cache_data(show_spinner="Checking which signals vary across training leads…")
def constant_columns(run_id: str, out_dir: str, dataset_path: str, label_mode: str,
                     feature_set: str) -> dict[str, float]:
    """``results.constant_columns`` for one run (cached per run): design columns the same on every training lead; a
    generic run's extras are encoded with the run's bundle."""
    extras = bundle(run_id, str(Path(out_dir) / BUNDLE_FILE)).extras \
        if feature_set == FeatureSet.GENERIC.value else None
    return results.constant_columns(dataset_path, label_mode, feature_set, extras)


def run_constant_columns(run: Run) -> dict[str, float]:
    """``constant_columns`` for ``run`` with its stored options (``Run.args``; the CLI defaults when absent)."""
    return constant_columns(run.run_id, run.out_dir, run.dataset_path,
                            run.args.get("label_mode", LabelMode.HORIZON.value),
                            run.args.get("feature_set", FeatureSet.V2.value))


@st.cache_data(show_spinner=False)
def currency(dataset_path: str) -> str | None:
    """The dataset's currency code (``emva.dataset_meta.dataset_currency``: GBP without ``dataset.json``, None when a
    converted dataset's mapping recorded none), cached per path."""
    return dataset_currency(dataset_path)


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
def example(run_id: str, bundle_path: str, dataset_path: str) -> scoring.Example:
    """The Score page's starting lead for the run (``scoring.example_for``; cached per run, since it may featurise
    the whole dataset)."""
    return scoring.example_for(bundle(run_id, bundle_path), dataset_path)


@st.cache_data(show_spinner=False)
def base_rate(run_id: str, out_dir: str, dataset_path: str) -> float:
    """Win rate of the run's training labels (``results.training_base_rate``, split at the dataset's
    ``test_from``)."""
    return results.training_base_rate(results.load_scores(out_dir), _raw_leads(dataset_path).created_at,
                                      dataset_dates(dataset_path)[1])


def restore(key: str, default: object, options: list | None = None) -> None:
    """Before rendering the widget ``key`` (without its own default): set its value to the one it last had, else
    ``default``. Streamlit deletes the state of a widget whose page is not shown, so ``keep`` mirrors each value into
    a plain session key (``KEPT_PREFIX`` + ``key``) that survives page switches; a value no longer among ``options``
    falls back to ``default``."""
    value = st.session_state[key] if key in st.session_state else st.session_state.get(KEPT_PREFIX + key, default)
    st.session_state[key] = value if options is None or value in options else default


def keep(key: str) -> None:
    """After rendering the widget ``key``: mirror its value for ``restore``."""
    st.session_state[KEPT_PREFIX + key] = st.session_state[key]


def forget_kept(prefix: str) -> None:
    """Drop the mirrored values (``keep``) of every widget key starting with ``prefix``."""
    for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(KEPT_PREFIX + prefix)]:
        del st.session_state[k]


def no_runs_state(message: str) -> None:
    """The empty state shown when there is nothing to display yet, with a link to the upload page."""
    html(C.empty_state("Nothing trained yet", message))
    st.page_link("views/upload.py", label="Upload data and train a model", icon=":material/arrow_forward:")


__all__ = ["bundle", "base_rate", "constant_columns", "currency", "data_root", "evaluation", "example", "forget_kept",
           "generic_section", "has_bundle", "html", "keep", "no_runs_state", "pick_run", "raw_lead", "restore",
           "run_constant_columns", "run_label"]
