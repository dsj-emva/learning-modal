"""Deal-value model and expected value (the deal-value block of ``baseline/emva_score.py::main``).

The value sent to ad platforms is ``p × E[deal value] × margin``. Phase 3 adds capping,
flooring and compression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from emva.constants import DEAL_VALUE_FEATURES, DEAL_VALUE_RIDGE_ALPHA

# Lognormal mean correction exp(sd**2 / 2) uses the residual sd on log deal value below.
# This number is the generator's planted noise sd (from the generator documentation), not something
# estimated from training data, so it violates ground rule 2. Kept verbatim in Phase 0 for
# exact reproduction of the baseline.
# TODO(plan 2.4): estimate the residual variance from training residuals and delete this.
DEAL_LOG_RESIDUAL_SD_FROM_GENERATOR = 0.45


def deal_value_design(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot matrix of ``DEAL_VALUE_FEATURES`` (all levels kept, ``_`` separator)."""
    return pd.concat([pd.get_dummies(X[c].astype(str), prefix=c) for c in DEAL_VALUE_FEATURES], axis=1).astype(float)


def fit_deal_value(M: pd.DataFrame, deal_value: pd.Series, tr: pd.Series) -> Ridge:
    """Fit ridge regression of log deal value on training rows that have a recorded deal value."""
    m = tr & deal_value.notna()
    return Ridge(alpha=DEAL_VALUE_RIDGE_ALPHA).fit(M[m], np.log(deal_value[m]))


def predict_deal_value(rg: Ridge, M: pd.DataFrame) -> np.ndarray:
    """Expected deal value: exp(predicted log value) times the lognormal mean correction."""
    return np.exp(rg.predict(M)) * np.exp(DEAL_LOG_RESIDUAL_SD_FROM_GENERATOR ** 2 / 2)


def expected_value(p: pd.Series, deal_value_hat: pd.Series, margin: float) -> pd.Series:
    """Value to send: P(won) × expected deal value × margin."""
    return p * deal_value_hat * margin


def realised_revenue(T: pd.DataFrame) -> pd.Series:
    """Baseline revenue per test lead: deal value if won (predicted value when blank), else 0.

    This is the definition behind the baseline's ``top20_revenue``. Because it imputes with
    the model's own ``deal_value_hat`` it moves when the value model changes; the standard
    report (``emva.eval``) uses recorded deal value only.
    """
    return pd.Series(np.where(T.y == 1, T.deal_value.fillna(T.deal_value_hat), 0), index=T.index)
