"""End-to-end pipeline: load -> clean -> label -> features -> design -> fit -> value -> summary.

With ``LabelConfig(mode="legacy")`` ``run`` reproduces ``baseline/emva_score.py::main`` step
for step, byte for byte. The default (``horizon``, plan 1.2 to 1.5) trains and evaluates on
the fixed-horizon label over mature leads only. ``emva/__main__.py`` does the printing and
file writing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from emva.constants import TEST_FROM
from emva.context.features import context_logit
from emva.design import design
from emva.eval.metrics import summary
from emva.features import add_features
from emva.io import clean, load
from emva.labels import HORIZON, LabelConfig, assign_labels, eligible_rows, split_masks
from emva.model import fit_lr, predict, scorecard
from emva.value import deal_value_design, expected_value, fit_deal_value, predict_deal_value, realised_revenue

# Columns written to scores.csv, in this order, when present (legacy mode: exactly the baseline's).
SCORE_COLUMNS: tuple[str, ...] = ("p_formula", "deal_value_hat", "value_formula", "p_combined", "value_combined", "y")
# Appended in horizon mode (plan 1.1, 1.2): the raw outcome, where the label came from, and when it matured.
HORIZON_SCORE_COLUMNS: tuple[str, ...] = ("won_within_h", "label_source", "matured_at")


@dataclass
class PipelineResult:
    """Everything ``run`` produces.

    ``X`` holds one row per cleaned lead with features, label columns and all score columns;
    ``train``/``test`` are boolean masks over ``X``; ``summary`` is the printed metrics table
    (empty when there are no labelled test leads, with a line in ``messages`` saying so);
    ``messages`` are the context-mode lines printed before it; ``labels`` is the label
    definition used.
    """

    X: pd.DataFrame
    train: pd.Series
    test: pd.Series
    design: pd.DataFrame
    weights: pd.DataFrame
    summary: pd.DataFrame
    labels: LabelConfig
    messages: list[str] = field(default_factory=list)

    def scores(self) -> pd.DataFrame:
        """The scores.csv frame (indexed by ``lead_id``); horizon mode adds ``HORIZON_SCORE_COLUMNS``."""
        cols = SCORE_COLUMNS + (HORIZON_SCORE_COLUMNS if self.labels.mode == "horizon" else ())
        return self.X[[c for c in cols if c in self.X]]


def build(L: pd.DataFrame, labels: LabelConfig = HORIZON) -> pd.DataFrame:
    """Clean, label and featurise loaded leads (``baseline/emva_score.py::build`` in legacy mode)."""
    X = clean(L)
    assign_labels(X, labels)
    return add_features(X)


def run(data: str | Path, margin: float = 1.0, context: str | Path | None = None,
        test_from: str = TEST_FROM, labels: LabelConfig = HORIZON) -> PipelineResult:
    """Train the formula and deal-value models on ``data`` and score every cleaned lead.

    ``labels`` picks the label definition; in horizon mode only mature leads enter the
    train and test sets. With ``context`` (an agent output CSV), also fit formula-only,
    context-only and formula+context models on the rows that have a context score and
    summarise all three.
    """
    X = build(load(data), labels)
    tr, te = split_masks(X, test_from, eligible_rows(X, labels))
    D = design(X)
    lr = fit_lr(D, X.y, tr)
    X["p_formula"] = predict(lr, D)

    # expected deal value (fit on train wins)
    M = deal_value_design(X)
    rg = fit_deal_value(M, X.deal_value, tr)
    X["deal_value_hat"] = predict_deal_value(rg, M)
    X["value_formula"] = expected_value(X.p_formula, X.deal_value_hat, margin)

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
        X["context_logit"] = context_logit(context, X.index)
        have = X.context_logit.notna()
        messages.append(f"context scores for {have.sum()} of {len(X)} leads")
        # same rows for both models so the comparison is fair
        tr2, te2 = tr & have, te & have
        if not te2.any():
            raise ValueError("no labelled test leads have a context score; cannot compare the context models")
        base = fit_lr(D, X.y, tr2)
        both = fit_lr(D.join(X.context_logit), X.y, tr2)
        X["p_base"] = predict(base, D)
        X.loc[have, "p_combined"] = predict(both, D.join(X.context_logit)[have])
        ctx_only = LogisticRegression().fit(X.loc[tr2, ["context_logit"]], X.y[tr2])
        X.loc[have, "p_context_only"] = ctx_only.predict_proba(X.loc[have, ["context_logit"]])[:, 1]
        T2 = X[te2]
        rev2 = realised_revenue(T2)
        rows = [summary(n, T2.y, T2[c].values, rev2, (T2[c] * T2.deal_value_hat).values) for n, c in
                [("formula", "p_base"), ("context_only", "p_context_only"), ("formula+context", "p_combined")]]
        messages.append(f"context weight in combined model: {both.coef_[0][-1]:.3f} (0 = context adds nothing)")
        X["value_combined"] = expected_value(X.p_combined, X.deal_value_hat, margin)

    return PipelineResult(X=X, train=tr, test=te, design=D, weights=weights,
                          summary=pd.DataFrame(rows), labels=labels, messages=messages)


def write_outputs(result: PipelineResult, out: str | Path) -> None:
    """Write ``weights.csv`` and ``scores.csv`` into ``out`` (same format as the baseline)."""
    result.weights.to_csv(f"{out}/weights.csv")
    result.scores().to_csv(f"{out}/scores.csv")


__all__ = ["HORIZON_SCORE_COLUMNS", "PipelineResult", "SCORE_COLUMNS", "build", "run", "write_outputs"]
