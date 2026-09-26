"""Frames for the "Model results" page, computed from a run's outputs with ``emva.eval`` code. No Streamlit.

Inputs are a run directory (``scores.csv``, ``weights.csv`` from ``python -m emva``) and the dataset it was
trained on. The test sets are the standard report's frozen definitions (``emva.eval.report.frozen_test_labels``):
``mature`` (horizon labels, the headline since Phase 1) and ``legacy`` (the POC's labels). The standard table is
``emva.eval.report.headline_frame`` / ``paired_frame`` over the report's own models (the frozen baseline via
``run_baseline``, this run's scores, the status quo from the dataset's rules); calibration and AUC by month come
from ``emva.eval.metrics`` / ``emva.eval.bootstrap``, value scale from ``emva.eval.value_report.scale_stats``.
Nothing is parsed out of ``report.txt``.

A run trained with the generic feature set (Phase 10, ADR 0024) also gets ``generic_comparison``: a v2 model trained
by ``emva.pipeline.run`` on the same data, labels and split, compared on the selected test set with
``emva.eval.generic_report.compare`` (paired bootstrap AUC difference), the generic design's collinearity check
(``emva.eval.collinearity``) and the extras' leakage and missingness screen, the numbers of the standard report's
"Generic vs v2"
section. The scorecard names ``x_`` columns as extras and takes their reference levels from the run's bundle.

The baseline row is optional (ADR 0022): a converted dataset (one carrying ``dataset.json``) never gets one, and a
dataset the frozen script cannot score loses it; ``Evaluation.baseline_missing`` then says why, and the page shows
the model and status-quo rows without the paired comparison.
"""
from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.storage import DATASET_META_FILE, RULES_FILE, is_converted
from emva.constants import AS_OF, CATS, CATS_V2_CANDIDATES, TEST_FROM, TOP_FRACTION
from emva.dataset_meta import dataset_dates
from emva.eval.bootstrap import N_RESAMPLES, SEED, PairedComparison, auc_ci
from emva.eval.collinearity import CollinearityResult, check_collinearity
from emva.constants import GENERIC_MISSINGNESS_FLAG
from emva.eval.generic_report import LEAKAGE_AUC_FLAG, MISSINGNESS_FLAG_TEXT, compare
from emva.eval.generic_report import flagged as screen_flagged
from emva.eval.metrics import auc_by_month, calibration_by_decile
from emva.eval.report import ScoredModel, TestSet, frozen_test_labels, headline_frame, paired_frame, run_baseline
from emva.eval.status_quo import load_rules, status_quo_value
from emva.eval.value_report import scale_stats
from emva.features import FeatureSet
from emva.generic import EXTRA_PREFIX, GenericEncoder
from emva.io import load
from emva.labels import LabelConfig
from emva.model import INTERCEPT, split_design_column
from emva.pipeline import run as pipeline_run

log = logging.getLogger(__name__)

TEST_SETS: tuple[str, ...] = ("mature", "legacy")
BASELINE, CANDIDATE, STATUS_QUO = "Baseline (frozen POC)", "This model", "Status quo"
# What the KPI strip compares this model with, in order of preference (none when neither row exists).
KPI_REFERENCES: tuple[str, ...] = (STATUS_QUO, BASELINE)
# Why a baseline row can be missing (ADR 0022); the frozen script hard-codes the v1 snapshot dates and file format.
BASELINE_CONVERTED: str = (
    f"No baseline row: this dataset was converted from another source (it carries {DATASET_META_FILE} with its own "
    f"dates). The frozen baseline script hard-codes the sample's file format and dates (as of {AS_OF.date()}, test "
    f"from {TEST_FROM}), so its scores would mean nothing here; the comparison with the baseline is left out.")
BASELINE_FAILED: str = ("No baseline row: the frozen baseline script could not score this dataset, so the comparison "
                        "with the baseline is left out.")
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
    "edits_1_4": "Edited 1-4 form fields", "no_company": "No company given", INTERCEPT: "Starting score",
}
REFERENCE_LEVELS: dict[str, str] = {**CATS, **CATS_V2_CANDIDATES}


# Prefix of an extra feature's plain name (generic feature set): "Extra · landing_page".
EXTRA_LABEL: str = "Extra · "


def feature_label(feature: str) -> str:
    """Plain-language name of a model feature (the raw name when it has none); an extra of the generic feature set
    (``x_<name>``) is ``EXTRA_LABEL`` plus its declared name."""
    if feature.startswith(EXTRA_PREFIX):
        return EXTRA_LABEL + feature[len(EXTRA_PREFIX):]
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


def load_scores(run_dir: str | Path) -> pd.DataFrame:
    """The run's ``scores.csv`` indexed by ``lead_id``, parsed with ``float_precision="round_trip"`` so every float
    is exactly the one the pipeline wrote (pandas' default parser can be off by 100+ ulp)."""
    return pd.read_csv(Path(run_dir) / "scores.csv", index_col="lead_id", float_precision="round_trip")


def scorecard(run_dir: str | Path, extras: GenericEncoder | None = None) -> pd.DataFrame:
    """The run's ``weights.csv`` as a readable scorecard, sorted by points (highest first).

    Columns: ``feature`` (plain name), ``level``, ``reference`` (the level it is compared with, which scores 0),
    ``points`` (20 points = odds of closing double), ``odds_multiplier``, ``log_odds``, ``column`` (raw name).
    ``extras`` is the run's bundle's encoder (``ModelBundle.extras``, generic feature set): it gives the ``x_``
    columns' reference levels (fitted on the training leads).
    """
    w = pd.read_csv(Path(run_dir) / "weights.csv", index_col=0, float_precision="round_trip")
    parts = [split_design_column(c) for c in w.index]
    refs = {**REFERENCE_LEVELS, **({e.feature.column: e.reference for e in extras.extras} if extras else {})}
    out = pd.DataFrame({
        "feature": [feature_label(f) for f, _ in parts], "level": [lvl for _, lvl in parts],
        "reference": [refs.get(f, "") for f, _ in parts], "points": w.points.to_numpy(),
        "odds_multiplier": w.odds_multiplier.to_numpy(), "log_odds": w.log_odds.to_numpy(), "column": list(w.index)})
    return out.sort_values(["points", "column"], ascending=[False, True], kind="stable").reset_index(drop=True)


@dataclass
class Evaluation:
    """A run's scores joined to one frozen test set, as the standard report sees it.

    ``test`` is the report's ``TestSet`` (labels, raw lead rows, revenue); ``models`` the report's ``ScoredModel``
    list: the frozen baseline (when its script ran on this dataset), this run's model, and the status quo (when the
    dataset has rules). ``notes`` say why the status-quo row is missing; ``baseline_missing`` why the baseline row is
    (None when it is there); ``base_rate`` is the win rate of the run's training labels.
    """

    name: str
    test: TestSet
    models: list[ScoredModel]
    notes: list[str]
    base_rate: float
    baseline_missing: str | None = None

    def model(self, name: str) -> ScoredModel | None:
        """The scored model called ``name`` (``BASELINE``, ``CANDIDATE``, ``STATUS_QUO``), or None if absent."""
        return next((m for m in self.models if m.name == name), None)

    @property
    def candidate(self) -> ScoredModel:
        """This run's model."""
        return self.model(CANDIDATE)


def training_base_rate(scores: pd.DataFrame, created_at: pd.Series, test_from: str = TEST_FROM) -> float:
    """Win rate of the run's training labels: ``scores.y`` of labelled leads created before ``test_from``
    (``created_at`` indexed by ``lead_id``). The same rows ``emva.pipeline.run`` trains on: in horizon mode ``y``
    is already NaN for leads that are not mature."""
    train = scores.y.notna() & (created_at.reindex(scores.index) < pd.Timestamp(test_from, tz="UTC"))
    return float(scores.y[train].mean())


def baseline_scores(dataset: str | Path, test_ids: pd.Index) -> tuple[pd.DataFrame | None, str | None]:
    """The frozen baseline's scores on ``dataset`` (``emva.eval.report.run_baseline``), or None and why not.

    None with ``BASELINE_CONVERTED`` for a converted dataset (the script is not run); None with ``BASELINE_FAILED``
    (logged) when the script fails, its output cannot be read, or it leaves a lead of ``test_ids`` unscored.
    """
    if is_converted(dataset):
        return None, BASELINE_CONVERTED
    with tempfile.TemporaryDirectory() as tmp:
        try:
            base = run_baseline(dataset, tmp)[0]
        except (RuntimeError, OSError, ValueError) as e:  # script exit != 0, no or unparsable scores.csv
            log.warning("baseline script failed on %s: %s", dataset, e)
            return None, BASELINE_FAILED
    unscored = test_ids.difference(base.index[base.p_formula.notna()]) if "p_formula" in base else test_ids
    if len(unscored):
        log.warning("baseline script left %d test leads of %s unscored (e.g. %s)", len(unscored), dataset,
                    list(unscored[:3]))
        return None, BASELINE_FAILED
    return base, None


def kpi_reference(models: Iterable[str]) -> str | None:
    """The row the KPI strip compares this model with: the first of ``KPI_REFERENCES`` among ``models``, else None."""
    present = set(models)
    return next((n for n in KPI_REFERENCES if n in present), None)


def evaluate(run_dir: str | Path, dataset: str | Path, test_set: str = "mature",
             leads: pd.DataFrame | None = None) -> Evaluation | None:
    """Join the run's scores to ``test_set`` (``mature`` or ``legacy``) of ``dataset``, with the baseline and status
    quo scored as ``emva.eval.report.build_report`` scores them; None when that test set is empty or has a single
    class. ``leads`` is ``emva.io.load(dataset)`` if already loaded. The test sets and the training base rate use the
    dataset's own ``as_of`` / ``test_from`` (``emva.dataset_meta.dataset_dates``, ADR 0020). Raises ``ValueError`` if a
    test lead has no score (the run and dataset do not match) or the dataset's ``dataset.json`` is malformed."""
    if test_set not in TEST_SETS:
        raise ValueError(f"test set must be one of {TEST_SETS}, got {test_set!r}")
    as_of, test_from = dataset_dates(dataset)
    L = load(dataset) if leads is None else leads
    X, _, y_a, y_b = frozen_test_labels(L, as_of, test_from)
    y = y_b if test_set == "mature" else y_a
    if len(y) == 0 or y.nunique() < 2:
        return None
    s = load_scores(run_dir)
    missing = y.index.difference(s.index[s.p_formula.notna()])
    if len(missing):
        raise ValueError(f"the run has no score for {len(missing)} test leads (e.g. {list(missing[:3])}); "
                         "was it trained on this dataset?")
    models, notes = [], []
    base, baseline_missing = baseline_scores(dataset, y.index)
    if base is not None:
        models.append(ScoredModel(BASELINE, base.p_formula, base.value_formula))
    models.append(ScoredModel(CANDIDATE, s.p_formula.loc[X.index], s.value_formula.loc[X.index]))
    rules = Path(dataset) / RULES_FILE
    if rules.exists():
        models.append(ScoredModel(STATUS_QUO, None, status_quo_value(L.loc[X.index], load_rules(rules)).sq_value))
    else:
        notes.append(f"The dataset has no {RULES_FILE}, so there is no status-quo row.")
    return Evaluation(name=test_set, test=TestSet(test_set, "", y, L.loc[y.index]), models=models, notes=notes,
                      base_rate=training_base_rate(s, L.created_at, test_from),
                      baseline_missing=baseline_missing)


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


@dataclass(frozen=True)
class GenericSection:
    """The "Generic vs v2" numbers of a generic-feature-set run on one test set (``generic_comparison``):
    ``paired`` (a = v2, b = generic: AUCs and the bootstrap difference generic − v2), ``n`` test leads,
    ``collinearity`` (the generic design on its training rows), ``screen`` (``emva.eval.generic_report.leakage_screen``:
    one row per extra, ``flag`` set above ``LEAKAGE_AUC_FLAG``, ``missingness flag`` when the share missing differs by
    ``GENERIC_MISSINGNESS_FLAG`` or more between closed and open leads), and the design sizes ``v2_columns`` /
    ``x_columns``."""

    test_set: str
    paired: PairedComparison
    n: int
    collinearity: CollinearityResult
    screen: pd.DataFrame
    v2_columns: int
    x_columns: int

    @property
    def flagged(self) -> list[str]:
        """Extras with either screen flag set (``emva.eval.generic_report.flagged``)."""
        return screen_flagged(self.screen)


def generic_comparison(run_dir: str | Path, dataset: str | Path, label_mode: str, test_set: str = "mature",
                       n_resamples: int = N_RESAMPLES) -> GenericSection | None:
    """Generic vs v2 for a run trained with the generic feature set and ``label_mode`` on ``dataset``, on ``test_set``
    (``mature`` or ``legacy``, the standard report's frozen definitions); None when that test set is empty or has a
    single class.

    Both models are retrained with ``emva.pipeline.run`` on the same data, labels and split (deterministic, so the
    generic one reproduces the run); ``emva.eval.generic_report.compare`` gives the paired difference (seed
    ``SEED``) and the leakage screen. Raises ``ValueError`` when the dataset has no extras, or when the retrained
    generic model does not reproduce the run's ``scores.csv`` on the test leads (the dataset changed since the run).
    """
    if test_set not in TEST_SETS:
        raise ValueError(f"test set must be one of {TEST_SETS}, got {test_set!r}")
    as_of, test_from = dataset_dates(dataset)
    _, _, y_a, y_b = frozen_test_labels(load(dataset), as_of, test_from)
    y = y_b if test_set == "mature" else y_a
    if len(y) == 0 or y.nunique() < 2:
        return None
    labels = LabelConfig(mode=label_mode)
    generic = pipeline_run(dataset, labels=labels, features=FeatureSet.GENERIC)
    s = load_scores(run_dir)
    if not np.allclose(generic.X.p_formula.loc[y.index], s.p_formula.reindex(y.index), rtol=1e-9, atol=0):
        raise ValueError("retraining the generic model does not reproduce this run's scores; the dataset changed "
                         "since the run was trained")
    v2 = pipeline_run(dataset, labels=labels, features=FeatureSet.V2)
    collinearity = check_collinearity(generic.design[generic.train])
    cmp = compare(generic, v2, [(test_set, y)], test_set, collinearity, n_resamples, SEED)
    return GenericSection(test_set=test_set, paired=cmp.paired[test_set], n=len(y), collinearity=collinearity,
                          screen=cmp.screen, v2_columns=v2.design.shape[1], x_columns=len(generic.extras.columns()))


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


__all__ = ["BASELINE", "BASELINE_CONVERTED", "BASELINE_FAILED", "CANDIDATE", "EXTRA_LABEL", "Evaluation",
           "FEATURE_LABELS", "GENERIC_MISSINGNESS_FLAG", "GenericSection", "KPI_REFERENCES", "LEAKAGE_AUC_FLAG",
           "MISSINGNESS_FLAG_TEXT", "STATUS_QUO", "TEST_SETS",
           "VALUE_COLUMNS", "auc_month", "baseline_scores", "calibration", "evaluate", "feature_label",
           "generic_comparison", "kpi_reference", "load_scores", "scorecard", "standard_table",
           "training_base_rate", "value_distribution"]
