"""Formula model: logistic regression fit, prediction and scorecard export."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from emva.constants import LR_C, LR_MAX_ITER, POINTS_TO_DOUBLE_ODDS


def make_lr() -> LogisticRegression:
    """The unfitted formula model: L2 logistic regression with ``LR_C`` and ``LR_MAX_ITER``."""
    return LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER)


def fit_lr(D: pd.DataFrame, y: pd.Series, tr: pd.Series) -> LogisticRegression:
    """Fit the formula model on the rows selected by the boolean mask ``tr``."""
    return make_lr().fit(D[tr], y[tr])


def predict(lr: LogisticRegression, D: pd.DataFrame) -> np.ndarray:
    """Return P(won) for every row of ``D``."""
    return lr.predict_proba(D)[:, 1]


# Row label of the intercept in ``intercept_row``.
INTERCEPT: str = "(intercept)"


def _points_table(w: pd.Series) -> pd.DataFrame:
    """``log_odds`` = ``w``, ``odds_multiplier`` = exp(w), ``points`` (``POINTS_TO_DOUBLE_ODDS`` points = odds double)."""
    return pd.DataFrame({"log_odds": w, "odds_multiplier": np.exp(w), "points": w * POINTS_TO_DOUBLE_ODDS / np.log(2)})


def coefficients(lr: LogisticRegression, columns: pd.Index) -> pd.DataFrame:
    """Unrounded coefficient table indexed by design column, in ``columns`` order.

    Columns: ``log_odds`` (the coefficient), ``odds_multiplier`` (``exp(log_odds)``) and ``points``
    (``POINTS_TO_DOUBLE_ODDS`` points = odds of closing double). ``scorecard`` is this table rounded and sorted.
    """
    return _points_table(pd.Series(lr.coef_[0], index=columns))


def intercept_row(lr: LogisticRegression) -> pd.DataFrame:
    """The intercept as a one-row ``coefficients``-style table indexed by ``INTERCEPT``."""
    return _points_table(pd.Series(lr.intercept_[:1], index=[INTERCEPT]))


def scorecard(lr: LogisticRegression, columns: pd.Index) -> pd.DataFrame:
    """Return the weights table written to ``weights.csv``, sorted by rounded log-odds.

    ``coefficients`` rounded: ``log_odds`` (3 dp), ``odds_multiplier`` (2 dp) and ``points`` (0 dp).
    Rows with equal rounded log-odds come out in an order that depends on numpy's unstable sort (baseline
    behaviour).
    """
    c = coefficients(lr, columns)
    return pd.DataFrame({"log_odds": c.log_odds.round(3), "odds_multiplier": c.odds_multiplier.round(2),
                         "points": c.points.round(0)}).sort_values("log_odds")
