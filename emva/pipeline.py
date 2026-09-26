"""End-to-end pipeline: load -> clean -> label -> features -> design -> fit -> value -> summary.

With ``labels=LEGACY`` and ``features=FeatureSet.LEGACY`` ``run`` reproduces
``baseline/emva_score.py::main`` step for step, byte for byte. The defaults are the fixed-horizon
label over mature leads (plan 1.2 to 1.5) and the v2 feature set (plan Phase 2): explicit missing
levels and indicators, company-name enrichment, similarity-based boilerplate detection, and a
deal-value correction estimated from training residuals. In horizon mode ``run`` also adds the two-stage
upload columns (plan 3.3): ``value_at_submit`` (the expected value through a fitted ``ValueTransform``) and
``value_at_close``, each with its timestamp, and ``value_at_close_status``. ``emva/__main__.py`` does the printing and file writing.

The snapshot date (``as_of``) and the train/test boundary (``test_from``) are the dataset's own when it carries a
``dataset.json`` (a converted dataset, ADR 0020; ``emva.dataset_meta.dataset_dates``), else ``AS_OF`` / ``TEST_FROM``.

With ``features=FeatureSet.GENERIC`` (Phase 10, ADR 0024) ``run`` also reads the dataset's declared extras
(``emva.generic.read_extra_features``: ``extra_features.csv`` joined on ``lead_id``, refused when absent), fits their
encoding on the training leads only and appends the ``x_`` design columns to the v2 design.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from emva.constants import AS_OF, TEST_FROM
from emva.context.features import context_features
from emva.dataset_meta import dataset_dates
from emva.eval.metrics import summary
from emva.feature_spec import feature_spec
from emva.features import FeatureSet
from emva.generic import GenericEncoder, fit_encoder, read_extra_features
from emva.io import clean, load
from emva.labels import HORIZON, LabelConfig, assign_labels, split_masks
from emva.model import fit_lr, predict, scorecard
from emva.value import DealValueModel, expected_value, fit_deal_value, predict_deal_value, realised_revenue, value_at_close
from emva.value_transform import FittedValueTransform, ValueTransform

# Columns written to scores.csv, in this order, when present (legacy mode: exactly the baseline's).
SCORE_COLUMNS: tuple[str, ...] = ("p_formula", "deal_value_hat", "value_formula", "p_combined", "value_combined", "y")
# Two-stage upload columns (plan 3.3), written after the label columns in horizon mode only (ADR 0007).
VALUE_SCORE_COLUMNS: tuple[str, ...] = ("value_at_submit", "value_at_submit_ts", "value_at_close", "value_at_close_ts",
                                        "value_at_close_status")


@dataclass
class PipelineResult:
    """Everything ``run`` produces.

    ``X`` holds one row per cleaned lead with features, label columns and all score columns;
    ``train``/``test`` are boolean masks over ``X``; ``summary`` is the printed metrics table
    (empty when there are no labelled test leads, with a line in ``messages`` saying so);
    ``messages`` are the context-mode lines printed before it; ``labels`` and ``features`` are
    the label definition and feature set used; ``model`` is the fitted formula model (its columns are
    ``design.columns``); ``deal_value`` is the fitted value model;
    ``value_transform`` the transform fitted on the training leads' values (None in legacy mode); ``as_of`` and
    ``test_from`` the snapshot date and train/test boundary the run used (ADR 0020); ``extras`` the extras' encoder
    fitted on the training leads (generic feature set only, else None).
    """

    X: pd.DataFrame
    train: pd.Series
    test: pd.Series
    design: pd.DataFrame
    weights: pd.DataFrame
    summary: pd.DataFrame
    labels: LabelConfig
    features: FeatureSet
    model: LogisticRegression
    deal_value: DealValueModel
    value_transform: FittedValueTransform | None = None
    messages: list[str] = field(default_factory=list)
    as_of: pd.Timestamp = AS_OF
    test_from: str = TEST_FROM
    extras: GenericEncoder | None = None

    def scores(self) -> pd.DataFrame:
        """The scores.csv frame (indexed by ``lead_id``): ``SCORE_COLUMNS``, the label config's extra columns and,
        in horizon mode, ``VALUE_SCORE_COLUMNS``."""
        cols = SCORE_COLUMNS + self.labels.extra_score_columns() + (() if self.labels.is_legacy else VALUE_SCORE_COLUMNS)
        return self.X[[c for c in cols if c in self.X]]


def build(L: pd.DataFrame, labels: LabelConfig = HORIZON, features: FeatureSet = FeatureSet.V2,
          as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    """Clean, label (at ``as_of``) and featurise loaded leads (``baseline/emva_score.py::build`` with both legacy
    options)."""
    X = clean(L)
    assign_labels(X, labels, as_of)
    return feature_spec(features).featurise(X)


def run(data: str | Path, margin: float = 1.0, context: str | Path | None = None,
        test_from: str | None = None, labels: LabelConfig = HORIZON,
        features: FeatureSet = FeatureSet.V2, value_transform: ValueTransform = ValueTransform(),
        as_of: pd.Timestamp | None = None) -> PipelineResult:
    """Train the formula and deal-value models on ``data`` and score every cleaned lead.

    ``labels`` picks the label definition; in horizon mode only mature leads enter the
    train and test sets. ``features`` picks the feature set; the legacy set also keeps the
    baseline's fixed deal-value residual sd, v2 estimates it from training residuals. With
    ``context`` (an agent output CSV: v2 judgments or the legacy ``context_score``; see
    ``emva.context.features.context_features``), also fit formula-only, context-only and formula+context
    models on the rows that have context and summarise all three. In horizon mode
    ``value_transform`` is fitted on the training leads' ``value_formula`` and gives ``value_at_submit``
    (legacy mode ignores it: its outputs stay the baseline's). ``test_from`` and ``as_of`` default to the dataset's
    dates (``emva.dataset_meta.dataset_dates``: its ``dataset.json``, else ``TEST_FROM`` / ``AS_OF``); labels,
    maturity and ``value_at_close`` are read at ``as_of``. The generic feature set also joins the dataset's raw
    extras onto ``X`` (columns ``x_<name>``) and raises ``ValueError`` when ``data`` has none, or when
    ``extra_features.csv`` and ``historical_leads.csv`` do not hold the same leads (either direction).
    """
    meta_as_of, meta_test_from = dataset_dates(data)
    as_of = meta_as_of if as_of is None else as_of
    test_from = meta_test_from if test_from is None else test_from
    spec = feature_spec(features)
    features = spec.feature_set
    L = load(data)
    X = build(L, labels, features, as_of)
    tr, te = split_masks(X, test_from, labels.eligible(X, as_of))
    D = spec.design(X)
    encoder = None
    if spec.extras:
        declared, E = read_extra_features(data)
        unknown, absent = E.index.difference(L.index), L.index.difference(E.index)
        if len(unknown):
            raise ValueError(f"extra_features.csv has {len(unknown)} lead(s) not in historical_leads.csv, e.g. "
                             f"{list(unknown[:5])}")
        if len(absent):
            raise ValueError(f"extra_features.csv lacks {len(absent)} lead(s) of historical_leads.csv, e.g. "
                             f"{list(absent[:5])} (every lead needs a row; blank values mean missing)")
        X = X.join(E)
        encoder = fit_encoder(declared, X[tr])
        D = D.join(encoder.design(X))
    lr = fit_lr(D, X.y, tr)
    X["p_formula"] = predict(lr, D)

    # expected deal value (fit on train wins)
    M = spec.value_design(X)
    dv = fit_deal_value(M, X.deal_value, tr, log_residual_sd=spec.fixed_log_residual_sd)
    X["deal_value_hat"] = predict_deal_value(dv, M)
    X["value_formula"] = expected_value(X.p_formula, X.deal_value_hat, margin)
    fitted = None
    if not labels.is_legacy:
        fitted = value_transform.fit(X.value_formula[tr])
        X["value_at_submit"] = fitted.apply(X.value_formula)
        X["value_at_submit_ts"] = X.created_at
        close = value_at_close(X, labels.horizon_days, as_of)
        X[list(close.columns)] = close

    weights = scorecard(lr, D.columns)

    T = X[te]
    messages: list[str] = []
    if te.any():
        rows = [summary("formula", T.y, T.p_formula.values, realised_revenue(T), T.value_formula.values)]
    else:
        # e.g. a horizon so long that no lead created on or after test_from is mature yet
        rows = []
        messages.append(f"no labelled test leads ({labels.describe()} labels, created on or after {test_from}); "
                        "test metrics skipped")

    if context:
        C = context_features(context, X.index)
        have = C.notna().all(axis=1)
        X[list(C.columns)] = C
        messages.append(f"context scores for {have.sum()} of {len(X)} leads")
        # same rows for both models so the comparison is fair
        tr2, te2 = tr & have, te & have
        if not te2.any():
            raise ValueError("no labelled test leads have a context score; cannot compare the context models")
        base = fit_lr(D, X.y, tr2)
        both = fit_lr(D.join(C), X.y, tr2)
        X["p_base"] = predict(base, D)
        X.loc[have, "p_combined"] = predict(both, D.join(C)[have])
        ctx_only = LogisticRegression().fit(C[tr2], X.y[tr2])
        X.loc[have, "p_context_only"] = ctx_only.predict_proba(C[have])[:, 1]
        T2 = X[te2]
        rev2 = realised_revenue(T2)
        rows = [summary(n, T2.y, T2[c].values, rev2, (T2[c] * T2.deal_value_hat).values) for n, c in
                [("formula", "p_base"), ("context_only", "p_context_only"), ("formula+context", "p_combined")]]
        w = both.coef_[0][-C.shape[1]:]
        if list(C.columns) == ["context_logit"]:
            messages.append(f"context weight in combined model: {w[0]:.3f} (0 = context adds nothing)")
        else:
            messages.append("context weights in combined model: " + ", ".join(f"{c} {v:+.3f}" for c, v in zip(C.columns, w)))
        X["value_combined"] = expected_value(X.p_combined, X.deal_value_hat, margin)

    return PipelineResult(X=X, train=tr, test=te, design=D, weights=weights, summary=pd.DataFrame(rows),
                          labels=labels, features=features, model=lr, deal_value=dv, value_transform=fitted,
                          messages=messages, as_of=as_of, test_from=test_from, extras=encoder)


def write_outputs(result: PipelineResult, out: str | Path) -> None:
    """Write ``weights.csv`` and ``scores.csv`` into ``out`` (created if missing; same format as the baseline)."""
    Path(out).mkdir(parents=True, exist_ok=True)
    result.weights.to_csv(f"{out}/weights.csv")
    result.scores().to_csv(f"{out}/scores.csv")


__all__ = ["PipelineResult", "SCORE_COLUMNS", "VALUE_SCORE_COLUMNS", "build", "run", "write_outputs"]
