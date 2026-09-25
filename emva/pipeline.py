"""End-to-end pipeline: load -> clean -> label -> features -> design -> fit -> value -> summary.

``run`` reproduces ``baseline/emva_score.py::main`` step for step (Phase 0: no behaviour
change); ``emva/__main__.py`` does the printing and file writing.
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
from emva.labels import label, split_masks
from emva.model import fit_lr, predict, scorecard
from emva.value import deal_value_design, expected_value, fit_deal_value, predict_deal_value, realised_revenue

# Columns written to scores.csv, in this order, when present.
SCORE_COLUMNS: tuple[str, ...] = ("p_formula", "deal_value_hat", "value_formula", "p_combined", "value_combined", "y")


@dataclass
class PipelineResult:
    """Everything ``run`` produces.

    ``X`` holds one row per cleaned lead with features, ``y`` and all score columns;
    ``train``/``test`` are boolean masks over ``X``; ``summary`` is the printed metrics table;
    ``messages`` are the context-mode lines printed before it.
    """

    X: pd.DataFrame
    train: pd.Series
    test: pd.Series
    design: pd.DataFrame
    weights: pd.DataFrame
    summary: pd.DataFrame
    messages: list[str] = field(default_factory=list)

    def scores(self) -> pd.DataFrame:
        """The scores.csv frame (indexed by ``lead_id``)."""
        return self.X[[c for c in SCORE_COLUMNS if c in self.X]]


def build(L: pd.DataFrame) -> pd.DataFrame:
    """Clean, label and featurise loaded leads (``baseline/emva_score.py::build``)."""
    X = clean(L)
    X["y"] = label(X)
    return add_features(X)


def run(data: str | Path, margin: float = 1.0, context: str | Path | None = None,
        test_from: str = TEST_FROM) -> PipelineResult:
    """Train the formula and deal-value models on ``data`` and score every cleaned lead.

    With ``context`` (an agent output CSV), also fit formula-only, context-only and
    formula+context models on the rows that have a context score and summarise all three.
    """
    X = build(load(data))
    tr, te = split_masks(X, test_from)
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
    rows = [summary("formula", T.y, T.p_formula.values, realised_revenue(T), T.value_formula.values)]
    messages: list[str] = []

    if context:
        X["context_logit"] = context_logit(context, X.index)
        have = X.context_logit.notna()
        messages.append(f"context scores for {have.sum()} of {len(X)} leads")
        # same rows for both models so the comparison is fair
        tr2, te2 = tr & have, te & have
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
                          summary=pd.DataFrame(rows), messages=messages)


def write_outputs(result: PipelineResult, out: str | Path) -> None:
    """Write ``weights.csv`` and ``scores.csv`` into ``out`` (same format as the baseline)."""
    result.weights.to_csv(f"{out}/weights.csv")
    result.scores().to_csv(f"{out}/scores.csv")


__all__ = ["PipelineResult", "SCORE_COLUMNS", "build", "run", "write_outputs"]
