"""Formula model: logistic regression fit, prediction and scorecard export."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from emva.constants import LR_C, LR_MAX_ITER, POINTS_TO_DOUBLE_ODDS


def fit_lr(D: pd.DataFrame, y: pd.Series, tr: pd.Series) -> LogisticRegression:
    """Fit the L2 logistic regression on the rows selected by the boolean mask ``tr``."""
    return LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER).fit(D[tr], y[tr])


def predict(lr: LogisticRegression, D: pd.DataFrame) -> np.ndarray:
    """Return P(won) for every row of ``D``."""
    return lr.predict_proba(D)[:, 1]


def scorecard(lr: LogisticRegression, columns: pd.Index) -> pd.DataFrame:
    """Return the weights table written to ``weights.csv``, sorted by rounded log-odds.

    Columns: ``log_odds`` (3 dp), ``odds_multiplier`` (2 dp) and ``points`` where
    ``POINTS_TO_DOUBLE_ODDS`` points = odds of closing double. Rows with equal rounded
    log-odds come out in an order that depends on numpy's unstable sort (baseline behaviour).
    """
    w = pd.Series(lr.coef_[0], index=columns)
    return pd.DataFrame({"log_odds": w.round(3), "odds_multiplier": np.exp(w).round(2),
                         "points": (w * POINTS_TO_DOUBLE_ODDS / np.log(2)).round(0)}).sort_values("log_odds")
