"""Deal-value model and expected value (the deal-value block of ``baseline/emva_score.py::main``).

The value sent to ad platforms is ``p × E[deal value] × margin``. E[deal value] is
``exp(ridge prediction of log value) × exp(s² / 2)``, the lognormal mean correction, where ``s``
is the residual sd of the log-value fit estimated from its training residuals (plan 2.4). A
caller may pass a fixed ``s`` instead (only the legacy feature set does, to reproduce the
baseline). Phase 3 adds capping, flooring and compression.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from emva.constants import DEAL_VALUE_FEATURES, DEAL_VALUE_RIDGE_ALPHA


@dataclass(frozen=True)
class DealValueModel:
    """A fitted log-value ridge model and the residual sd used for its lognormal mean correction.

    ``estimated`` is True when ``log_residual_sd`` came from the training residuals, False when
    the caller fixed it; ``n_train`` is the number of training deals.
    """

    ridge: Ridge
    log_residual_sd: float
    estimated: bool
    n_train: int

    @property
    def correction(self) -> float:
        """Lognormal mean correction ``exp(sd² / 2)``: E[value] / exp(E[log value])."""
        return np.exp(self.log_residual_sd ** 2 / 2)


def residual_sd(residuals: np.ndarray | pd.Series, ddof: int = 1) -> float:
    """Sample standard deviation of ``residuals`` (``ddof`` degrees of freedom removed).

    Raises ``ValueError`` when there are not more residuals than ``ddof``.
    """
    r = np.asarray(residuals, dtype=float)
    if len(r) <= ddof:
        raise ValueError(f"need more than {ddof} residuals to estimate an sd, got {len(r)}")
    return float(np.std(r, ddof=ddof))


def deal_value_design(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot matrix of ``DEAL_VALUE_FEATURES`` (all levels kept, ``_`` separator)."""
    return pd.concat([pd.get_dummies(X[c].astype(str), prefix=c) for c in DEAL_VALUE_FEATURES], axis=1).astype(float)


def fit_deal_value(M: pd.DataFrame, deal_value: pd.Series, tr: pd.Series,
                   log_residual_sd: float | None = None) -> DealValueModel:
    """Fit ridge regression of log deal value on training rows that have a recorded deal value.

    With ``log_residual_sd=None`` (the default) the correction's sd is the sample sd of the
    in-sample training residuals of that fit; otherwise the given value is used as is.
    """
    m = tr & deal_value.notna()
    log_value = np.log(deal_value[m])
    rg = Ridge(alpha=DEAL_VALUE_RIDGE_ALPHA).fit(M[m], log_value)
    if log_residual_sd is None:
        return DealValueModel(rg, residual_sd(log_value - rg.predict(M[m])), estimated=True, n_train=int(m.sum()))
    return DealValueModel(rg, log_residual_sd, estimated=False, n_train=int(m.sum()))


def predict_deal_value(model: DealValueModel, M: pd.DataFrame) -> np.ndarray:
    """Expected deal value: exp(predicted log value) times the lognormal mean correction."""
    return np.exp(model.ridge.predict(M)) * model.correction


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
