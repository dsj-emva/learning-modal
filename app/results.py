"""Frames for the "Model results" page, computed from a run's outputs with ``emva.eval`` code. No Streamlit.

Inputs are a run directory (``scores.csv``, ``weights.csv`` from ``python -m emva``) and the dataset it was
trained on. The test sets are the standard report's frozen definitions (``emva.eval.report.frozen_test_labels``):
``mature`` (horizon labels, the headline since Phase 1) and ``legacy`` (the POC's labels). Metrics come from
``emva.eval.metrics`` / ``emva.eval.bootstrap`` and value scale from ``emva.eval.value_report.scale_stats``;
nothing is parsed out of ``report.txt``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from app.storage import RULES_FILE
from emva.constants import CATS, CATS_V2_CANDIDATES, TEST_FROM, TOP_FRACTION
from emva.eval.bootstrap import N_RESAMPLES, auc_ci
from emva.eval.metrics import auc_by_month, calibration_by_decile, top_share
from emva.eval.report import TestSet, frozen_test_labels
from emva.eval.status_quo import load_rules, status_quo_value
from emva.eval.value_report import scale_stats
from emva.io import load

TEST_SETS: tuple[str, ...] = ("mature", "legacy")
CANDIDATE, STATUS_QUO = "Model", "Status quo"
# Months with fewer test leads than this get no bootstrap band (the interval would be meaningless).
MIN_MONTH_FOR_CI: int = 30

# Plain-language names of the model features (``weights.csv`` rows are ``feature=level``).
FEATURE_LABELS: dict[str, str] = {
    "channel": "Channel", "form_variant": "Form variant", "session_missing": "No on-site session",
    "enrichment_missing": "Company not in database", "band": "Company size", "email": "Email address",
    "text": "What they want to solve", "seniority": "Seniority", "spend": "Monthly ad spend", "crm": "CRM platform",
    "hiring": "Company hiring", "time_on_page": "Time on page", "hesitation_90s": "Paused over 90 s",
    "sessions_3plus": "3+ visits before converting", "viewed_pricing": "Viewed pricing",
    "search_term": "Search term", "business_hours": "Submitted", "ip_country": "IP vs typed country",
    "ip_type": "IP address type", "c_budget": "Budget answer", "c_timeline": "Timeline answer",
    "edits_1_4": "Edited 1-4 form fields", "no_company": "No company given", "(intercept)": "Starting score",
}
REFERENCE_LEVELS: dict[str, str] = {**CATS, **CATS_V2_CANDIDATES}


def feature_label(feature: str) -> str:
    """Plain-language name of a model feature (the raw name when it has none)."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


def split_design_column(column: str) -> tuple[str, str]:
    """``"channel=linkedin"`` -> ``("channel", "linkedin")``."""
    feature, _, level = column.partition("=")
    return feature, level


def load_scores(run_dir: str | Path) -> pd.DataFrame:
    """The run's ``scores.csv`` indexed by ``lead_id``, parsed with ``float_precision="round_trip"`` so every float
    is exactly the one the pipeline wrote (pandas' default parser can be off by 100+ ulp)."""
    return pd.read_csv(Path(run_dir) / "scores.csv", index_col="lead_id", float_precision="round_trip")


def scorecard(run_dir: str | Path) -> pd.DataFrame:
    """The run's ``weights.csv`` as a readable scorecard, sorted by points (highest first).

    Columns: ``feature`` (plain name), ``level``, ``reference`` (the level it is compared with, which scores 0),
    ``points`` (20 points = odds of closing double), ``odds_multiplier``, ``log_odds``, ``column`` (raw name).
    """
    w = pd.read_csv(Path(run_dir) / "weights.csv", index_col=0, float_precision="round_trip")
    parts = [split_design_column(c) for c in w.index]
    out = pd.DataFrame({
        "feature": [feature_label(f) for f, _ in parts], "level": [lvl for _, lvl in parts],
        "reference": [REFERENCE_LEVELS.get(f, "") for f, _ in parts], "points": w.points.to_numpy(),
        "odds_multiplier": w.odds_multiplier.to_numpy(), "log_odds": w.log_odds.to_numpy(), "column": list(w.index)})
    return out.sort_values(["points", "column"], ascending=[False, True], kind="stable").reset_index(drop=True)


@dataclass
class Evaluation:
    """A run's scores joined to one frozen test set: ``test`` (``TestSet``: labels, raw lead rows, revenue),
    ``p`` and ``value`` (the model's P(close) and p × deal value on the test leads), ``sq_value`` (status-quo value
    on the test leads, None without rules) and ``base_rate`` (win rate of the run's training labels)."""

    name: str
    test: TestSet
    p: pd.Series
    value: pd.Series
    sq_value: pd.Series | None
    base_rate: float


def training_base_rate(scores: pd.DataFrame, created_at: pd.Series) -> float:
    """Win rate of the run's training labels: ``scores.y`` of labelled leads created before ``TEST_FROM``
    (``created_at`` indexed by ``lead_id``). The same rows ``emva.pipeline.run`` trains on: in horizon mode ``y``
    is already NaN for leads that are not mature."""
    train = scores.y.notna() & (created_at.reindex(scores.index) < pd.Timestamp(TEST_FROM, tz="UTC"))
    return float(scores.y[train].mean())


def evaluate(run_dir: str | Path, dataset: str | Path, test_set: str = "mature",
             leads: pd.DataFrame | None = None) -> Evaluation | None:
    """Join the run's scores to ``test_set`` (``mature`` or ``legacy``) of ``dataset``; None when that test set
    is empty or has a single class (nothing to evaluate). ``leads`` is ``emva.io.load(dataset)`` if already loaded.
    Raises ``ValueError`` if a test lead has no score (the run and dataset do not match)."""
    if test_set not in TEST_SETS:
        raise ValueError(f"test set must be one of {TEST_SETS}, got {test_set!r}")
    L = load(dataset) if leads is None else leads
    _, _, y_a, y_b = frozen_test_labels(L)
    y = y_b if test_set == "mature" else y_a
    if len(y) == 0 or y.nunique() < 2:
        return None
    s = load_scores(run_dir)
    missing = y.index.difference(s.index[s.p_formula.notna()])
    if len(missing):
        raise ValueError(f"the run has no score for {len(missing)} test leads (e.g. {list(missing[:3])}); "
                         "was it trained on this dataset?")
    rules = Path(dataset) / RULES_FILE
    sq = status_quo_value(L.loc[y.index], load_rules(rules)).sq_value if rules.exists() else None
    return Evaluation(name=test_set, test=TestSet(test_set, "", y, L.loc[y.index]), p=s.p_formula.loc[y.index],
                      value=s.value_formula.loc[y.index], sq_value=sq,
                      base_rate=training_base_rate(s, L.created_at))


def headline(ev: Evaluation, n_resamples: int = N_RESAMPLES) -> pd.DataFrame:
    """The standard table's headline rows (model, then status quo when available), numeric.

    Columns: ``model``, ``auc``, ``auc_lo``, ``auc_hi`` (percentile bootstrap, ``n_resamples``, seed 0), ``brier``
    (NaN for the status quo, which has no probability), ``top20_wins``, ``top20_revenue_p`` (revenue captured by
    the top 20% ranked by p; status quo by its value), ``top20_revenue_value`` (ranked by p × value), ``n``, ``wins``.
    """
    y, rev = ev.test.y, ev.test.revenue
    rows = []
    models = [(CANDIDATE, ev.p, ev.value)] + ([(STATUS_QUO, None, ev.sq_value)] if ev.sq_value is not None else [])
    for name, p, value in models:
        rank = (value if p is None else p).to_numpy()
        ci = auc_ci(y.to_numpy(), rank, n_resamples)
        rows.append({"model": name, "auc": ci.point, "auc_lo": ci.lo, "auc_hi": ci.hi,
                     "brier": np.nan if p is None else brier_score_loss(y, np.clip(p, 0, 1)),
                     "top20_wins": top_share(y, rank), "top20_revenue_p": top_share(rev, rank),
                     "top20_revenue_value": top_share(rev, value.to_numpy()), "n": len(y), "wins": int(y.sum())})
    return pd.DataFrame(rows)


def calibration(ev: Evaluation) -> pd.DataFrame:
    """Mean predicted vs observed win rate per decile of p on the test set (``calibration_by_decile``)."""
    return calibration_by_decile(ev.test.y, ev.p.to_numpy())


def auc_month(ev: Evaluation, n_resamples: int = N_RESAMPLES) -> pd.DataFrame:
    """AUC per test month (``auc_by_month``) plus ``auc_lo`` / ``auc_hi``: a bootstrap CI for months with both
    classes and at least ``MIN_MONTH_FOR_CI`` leads, NaN otherwise."""
    y, created = ev.test.y, ev.test.leads.created_at
    out = auc_by_month(y, ev.p.to_numpy(), created)
    months = created.dt.strftime("%Y-%m")
    lo, hi = [], []
    for m, n in zip(out.month, out.n):
        mask = (months == m).to_numpy()
        ym = y.to_numpy()[mask]
        if n >= MIN_MONTH_FOR_CI and 0 < ym.sum() < len(ym):
            ci = auc_ci(ym, ev.p.to_numpy()[mask], n_resamples)
            lo.append(ci.lo)
            hi.append(ci.hi)
        else:
            lo.append(np.nan)
            hi.append(np.nan)
    return out.assign(auc_lo=lo, auc_hi=hi)


VALUE_COLUMNS: dict[str, str] = {"value_at_submit": "Value sent at submit", "value_formula": "p × expected deal value"}


def value_distribution(run_dir: str | Path) -> pd.DataFrame:
    """Scale of the run's value columns over every scored lead (``emva.eval.value_report.scale_stats``).

    One row per column present (``value_at_submit`` is absent for legacy-label runs): ``column``, ``label``, ``p1``,
    ``p50``, ``p90``, ``p99``, ``max``, ``max_over_median``, ``top1pct_share``, ``n``.
    """
    s = load_scores(run_dir)
    rows = []
    for col, label in VALUE_COLUMNS.items():
        if col in s and s[col].notna().any():
            v = s[col].dropna()
            rows.append({"column": col, "label": label, **scale_stats(v), "n": len(v)})
    return pd.DataFrame(rows)


def top_fraction_label() -> str:
    """``"top 20%"`` from ``TOP_FRACTION``."""
    return f"top {int(TOP_FRACTION * 100)}%"


__all__ = ["CANDIDATE", "Evaluation", "FEATURE_LABELS", "STATUS_QUO", "TEST_SETS", "VALUE_COLUMNS", "auc_month",
           "calibration", "evaluate", "feature_label", "headline", "load_scores", "scorecard", "training_base_rate",
           "value_distribution"]
