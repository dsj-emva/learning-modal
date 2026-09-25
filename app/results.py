"""Frames for the "Model results" page, computed from a run's outputs with ``emva.eval`` code. No Streamlit.

Inputs are a run directory (``scores.csv``, ``weights.csv`` from ``python -m emva``) and the dataset it was
trained on. The test sets are the standard report's frozen definitions (``emva.eval.report.frozen_test_labels``):
``mature`` (horizon labels, the headline since Phase 1) and ``legacy`` (the POC's labels). The standard table is
``emva.eval.report.headline_frame`` / ``paired_frame`` over the report's own models (the frozen baseline via
``run_baseline``, this run's scores, the status quo from the dataset's rules); calibration and AUC by month come
from ``emva.eval.metrics`` / ``emva.eval.bootstrap``, value scale from ``emva.eval.value_report.scale_stats``.
Nothing is parsed out of ``report.txt``.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.storage import RULES_FILE
from emva.constants import CATS, CATS_V2_CANDIDATES, TEST_FROM, TOP_FRACTION
from emva.eval.bootstrap import N_RESAMPLES, SEED, auc_ci
from emva.eval.metrics import auc_by_month, calibration_by_decile
from emva.eval.report import ScoredModel, TestSet, frozen_test_labels, headline_frame, paired_frame, run_baseline
from emva.eval.status_quo import load_rules, status_quo_value
from emva.eval.value_report import scale_stats
from emva.io import load

log = logging.getLogger(__name__)

TEST_SETS: tuple[str, ...] = ("mature", "legacy")
BASELINE, CANDIDATE, STATUS_QUO = "Baseline (frozen POC)", "This model", "Status quo"
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
    """A run's scores joined to one frozen test set, as the standard report sees it.

    ``test`` is the report's ``TestSet`` (labels, raw lead rows, revenue); ``models`` the report's ``ScoredModel``
    list: the frozen baseline (when its script ran on this dataset), this run's model, and the status quo (when the
    dataset has rules). ``notes`` say why a row is missing; ``base_rate`` is the win rate of the run's training labels.
    """

    name: str
    test: TestSet
    models: list[ScoredModel]
    notes: list[str]
    base_rate: float

    def model(self, name: str) -> ScoredModel | None:
        """The scored model called ``name`` (``BASELINE``, ``CANDIDATE``, ``STATUS_QUO``), or None if absent."""
        return next((m for m in self.models if m.name == name), None)

    @property
    def candidate(self) -> ScoredModel:
        """This run's model."""
        return self.model(CANDIDATE)


def training_base_rate(scores: pd.DataFrame, created_at: pd.Series) -> float:
    """Win rate of the run's training labels: ``scores.y`` of labelled leads created before ``TEST_FROM``
    (``created_at`` indexed by ``lead_id``). The same rows ``emva.pipeline.run`` trains on: in horizon mode ``y``
    is already NaN for leads that are not mature."""
    train = scores.y.notna() & (created_at.reindex(scores.index) < pd.Timestamp(TEST_FROM, tz="UTC"))
    return float(scores.y[train].mean())


def _baseline(dataset: str | Path) -> tuple[pd.DataFrame | None, str | None]:
    """The frozen baseline's scores on ``dataset`` (``emva.eval.report.run_baseline``), or None and why not."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            return run_baseline(dataset, tmp)[0], None
        except RuntimeError as e:
            log.warning("baseline script failed on %s: %s", dataset, e)
            return None, "The frozen baseline script could not run on this dataset, so it has no row."


def evaluate(run_dir: str | Path, dataset: str | Path, test_set: str = "mature",
             leads: pd.DataFrame | None = None) -> Evaluation | None:
    """Join the run's scores to ``test_set`` (``mature`` or ``legacy``) of ``dataset``, with the baseline and status
    quo scored as ``emva.eval.report.build_report`` scores them; None when that test set is empty or has a single
    class. ``leads`` is ``emva.io.load(dataset)`` if already loaded. Raises ``ValueError`` if a test lead has no score
    (the run and dataset do not match)."""
    if test_set not in TEST_SETS:
        raise ValueError(f"test set must be one of {TEST_SETS}, got {test_set!r}")
    L = load(dataset) if leads is None else leads
    X, _, y_a, y_b = frozen_test_labels(L)
    y = y_b if test_set == "mature" else y_a
    if len(y) == 0 or y.nunique() < 2:
        return None
    s = load_scores(run_dir)
    missing = y.index.difference(s.index[s.p_formula.notna()])
    if len(missing):
        raise ValueError(f"the run has no score for {len(missing)} test leads (e.g. {list(missing[:3])}); "
                         "was it trained on this dataset?")
    models, notes = [], []
    base, why = _baseline(dataset)
    if base is None:
        notes.append(why)
    else:
        models.append(ScoredModel(BASELINE, base.p_formula, base.value_formula))
    models.append(ScoredModel(CANDIDATE, s.p_formula.loc[X.index], s.value_formula.loc[X.index]))
    rules = Path(dataset) / RULES_FILE
    if rules.exists():
        models.append(ScoredModel(STATUS_QUO, None, status_quo_value(L.loc[X.index], load_rules(rules)).sq_value))
    else:
        notes.append(f"The dataset has no {RULES_FILE}, so there is no status-quo row.")
    return Evaluation(name=test_set, test=TestSet(test_set, "", y, L.loc[y.index]), models=models, notes=notes,
                      base_rate=training_base_rate(s, L.created_at))


def standard_table(ev: Evaluation, n_resamples: int = N_RESAMPLES) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """``(headline, paired)`` from ``emva.eval.report.headline_frame`` / ``paired_frame``, the functions behind the
    standard report; ``paired`` (each model vs the baseline) is None without a baseline row."""
    head = headline_frame(ev.test, ev.models, n_resamples, SEED)
    paired = paired_frame(ev.test, ev.models, n_resamples, SEED) if ev.model(BASELINE) is not None else None
    return head, paired


def calibration(ev: Evaluation) -> pd.DataFrame:
    """Mean predicted vs observed win rate per decile of p on the test set (``calibration_by_decile``)."""
    return calibration_by_decile(ev.test.y, ev.candidate.p.loc[ev.test.ids].to_numpy())


def auc_month(ev: Evaluation, n_resamples: int = N_RESAMPLES) -> pd.DataFrame:
    """AUC per test month (``auc_by_month``) plus ``auc_lo`` / ``auc_hi``: a bootstrap CI for months with both
    classes and at least ``MIN_MONTH_FOR_CI`` leads, NaN otherwise."""
    y, created = ev.test.y, ev.test.leads.created_at
    p = ev.candidate.p.loc[ev.test.ids].to_numpy()
    out = auc_by_month(y, p, created)
    months = created.dt.strftime("%Y-%m")
    lo, hi = [], []
    for m, n in zip(out.month, out.n):
        mask = (months == m).to_numpy()
        ym = y.to_numpy()[mask]
        if n >= MIN_MONTH_FOR_CI and 0 < ym.sum() < len(ym):
            ci = auc_ci(ym, p[mask], n_resamples)
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
    ``p50``, ``p90``, ``p99``, ``max``, ``max_over_median``, ``top1pct_share``, ``n`` and ``skipped`` (True, stats
    NaN, when the median or total is not positive).
    """
    s = load_scores(run_dir)
    rows = []
    for col, label in VALUE_COLUMNS.items():
        if col in s and s[col].notna().any():
            v = s[col].dropna()
            if np.median(v) <= 0 or v.sum() <= 0:  # scale_stats' ratios are undefined; say so instead of raising
                rows.append({"column": col, "label": label, "n": len(v), "skipped": True})
                continue
            rows.append({"column": col, "label": label, **scale_stats(v), "n": len(v), "skipped": False})
    return pd.DataFrame(rows)


def top_fraction_label() -> str:
    """``"top 20%"`` from ``TOP_FRACTION``."""
    return f"top {int(TOP_FRACTION * 100)}%"


__all__ = ["BASELINE", "CANDIDATE", "Evaluation", "FEATURE_LABELS", "STATUS_QUO", "TEST_SETS", "VALUE_COLUMNS", "auc_month",
           "calibration", "evaluate", "feature_label", "load_scores", "scorecard", "standard_table", "training_base_rate",
           "value_distribution"]
