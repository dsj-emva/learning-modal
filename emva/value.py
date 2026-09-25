"""Deal-value model and expected value (the deal-value block of ``baseline/emva_score.py::main``).

The value sent to ad platforms is ``p × E[deal value] × margin``. E[deal value] is
``exp(ridge prediction of log value) × exp(s² / 2)``, the lognormal mean correction, where ``s``
is the residual sd of the log-value fit estimated from its training residuals (plan 2.4). A
caller may pass a fixed ``s`` instead (only the legacy feature set does, to reproduce the
baseline). The transform from expected value to uploaded value is ``emva.value_transform``.

Two-stage upload (plan 3.3, horizon mode): ``value_at_submit`` is the transformed expected value, sent
when the lead is created; ``value_at_close`` (function ``value_at_close``) is what the lead turned out to be worth,
sent as an adjustment once known.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from emva.constants import AS_OF, DEAL_VALUE_FEATURES, DEAL_VALUE_LEVELS, DEAL_VALUE_RIDGE_ALPHA, HORIZON_DAYS
from emva.design import fixed_columns, fixed_design
from emva.labels import is_mature, matured_at


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


def residual_sd(residuals: np.ndarray | pd.Series) -> float:
    """Sample standard deviation of ``residuals`` with one degree of freedom removed (ddof = 1).

    ddof = 1 is a deliberate choice: the residuals are in-sample residuals of a ridge fit on dummy
    columns, whose shrinkage makes them larger than least-squares residuals and roughly offsets the
    fit's effective degrees of freedom, so no further correction is applied (slightly conservative;
    with hundreds of training deals the difference is small). Raises ``ValueError`` for fewer than two
    residuals.
    """
    r = np.asarray(residuals, dtype=float)
    if len(r) < 2:
        raise ValueError(f"need at least 2 residuals to estimate an sd, got {len(r)}")
    return float(np.std(r, ddof=1))


def deal_value_design(X: pd.DataFrame) -> pd.DataFrame:
    """Legacy (data-driven) one-hot matrix of ``DEAL_VALUE_FEATURES``: all levels present kept, ``_`` separator.

    The baseline's design, kept for ``--feature-set legacy`` (byte identity); levels absent from ``X``
    make no column. The v2 feature set uses ``fixed_deal_value_design``.
    """
    return pd.concat([pd.get_dummies(X[c].astype(str), prefix=c) for c in DEAL_VALUE_FEATURES], axis=1).astype(float)


# The ridge needs no reference level: every declared level of every deal-value feature gets a column.
_ALL_LEVELS: dict[str, None] = dict.fromkeys(DEAL_VALUE_FEATURES)


def fixed_deal_value_columns(levels: dict[str, tuple[str, ...]] = DEAL_VALUE_LEVELS) -> list[str]:
    """The columns ``fixed_deal_value_design`` emits: ``<feature>=<level>`` for every declared level, in order."""
    return fixed_columns(_ALL_LEVELS, levels)


def fixed_deal_value_design(X: pd.DataFrame, levels: dict[str, tuple[str, ...]] = DEAL_VALUE_LEVELS) -> pd.DataFrame:
    """Fixed-schema one-hot matrix of ``DEAL_VALUE_FEATURES`` (ADR 0009 applied to the value model, plan 3.1).

    ``design.fixed_design`` with every level kept, whatever occurs in ``X``, so a one-row frame gets the full
    schema. A value outside a feature's declared levels raises ``ValueError`` naming the feature and the values.
    """
    return fixed_design(X, _ALL_LEVELS, levels)


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


# ``value_at_close_status`` values (R10): the amount is known, the win is known but its amount is not, or not yet known.
CLOSE_KNOWN, CLOSE_UNKNOWN_AMOUNT, CLOSE_PENDING = "known", "unknown_amount", "pending"


def value_at_close(X: pd.DataFrame, horizon_days: int = HORIZON_DAYS,
                   as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    """The close-stage value of each lead and when it became known (plan 3.3); needs ``created_at``, ``won_at``, ``deal_value``.

    Returns ``value_at_close``, ``value_at_close_ts`` and ``value_at_close_status``:

    - Won at or before ``as_of`` (whenever, also after H or before maturity), at ``won_at``: the recorded deal
      value, status ``known``; with a blank deal value, NaN and status ``unknown_amount`` (ruling R10: the win
      is known, the amount is not; ADR 0002's blank = 0 is an evaluation convention only);
    - otherwise mature (``created_at + horizon_days <= as_of``): 0 at ``matured_at``, status ``known``;
    - otherwise (immature, not Won): NaN and NaT, status ``pending``.
    """
    won = X.won_at <= as_of  # NaT compares False
    mature = is_mature(X, horizon_days, as_of)
    value = np.where(won, X.deal_value, np.where(mature, 0.0, np.nan))
    ts = X.won_at.where(won, matured_at(X, horizon_days).where(mature))
    status = np.select([won & X.deal_value.isna(), won | mature], [CLOSE_UNKNOWN_AMOUNT, CLOSE_KNOWN], CLOSE_PENDING)
    return pd.DataFrame({"value_at_close": value, "value_at_close_ts": ts, "value_at_close_status": status},
                        index=X.index)


def realised_revenue(T: pd.DataFrame) -> pd.Series:
    """Baseline revenue per test lead: deal value if won (predicted value when blank), else 0.

    This is the definition behind the baseline's ``top20_revenue``. Because it imputes with
    the model's own ``deal_value_hat`` it moves when the value model changes; the standard
    report (``emva.eval``) uses recorded deal value only.
    """
    return pd.Series(np.where(T.y == 1, T.deal_value.fillna(T.deal_value_hat), 0), index=T.index)
