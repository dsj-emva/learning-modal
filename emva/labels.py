"""Label rules, ``label_source``, the maturity mask and the train/test split.

Two label definitions exist side by side (``--label-mode``, plan 1.5):

``legacy`` (``label``)
    The baseline's rules, unchanged from the "labels (decision log)" block of
    ``baseline/emva_score.py::build``: Won = 1; Lost = 0; open deals idle ``STALLED_DAYS``+
    = 0; never-contacted (stage New) leads ``GHOSTED_DAYS``+ old = 0; everything else NaN.

``horizon`` (``won_within_h`` then ``horizon_label``, plan 1.2 to 1.4)
    A fixed-horizon outcome, the same H (``HORIZON_DAYS``) for every lead, measured from
    ``created_at``. ``matured_at = created_at + H``; a lead is *mature* when
    ``matured_at <= as_of``.

    ``won_within_h`` (the outcome):
      - 1 if the Won stage change happened at or before ``matured_at`` (so a lead that is
        already Won counts as 1 even when it is not mature yet);
      - 0 if the lead is mature and was not Won by ``matured_at`` (including leads Won
        later, which is what "fixed horizon" means);
      - NaN if the lead is not mature and not Won: too young to know, never Lost.

    ``horizon_label`` (the training/evaluation label ``y``) is ``won_within_h`` with two
    groups of mature non-wins censored (set to NaN) by default:
      - ghosted at H: still at stage New at ``matured_at`` (no change to another stage by
        then). ``include_ghosted=True`` (``--include-ghosted``) counts them as 0 instead.
      - ``label_source == "stalled"``. ``stalled_as_lost=True`` (``--stalled-as-lost``)
        counts them as 0 instead.

``label_source`` describes each lead's CRM state at ``as_of`` (plan 1.1; horizon mode only);
see ``label_source`` for the exact definitions. Its ``ghosted`` is exactly the ghosted-at-H
group above, restricted to mature leads; a lead still at New but younger than H is ``open``.

Only mature leads may be used to train or evaluate horizon labels (``split_masks`` with
``eligible=LabelConfig.eligible(X)``): young leads can only be labelled 1, so including them
would bias the win rate upwards.

Asymmetry between open and stalled (orchestrator ruling R2): a mature lead that is still in
an open stage at ``as_of`` and was not Won by H is 0, because it demonstrably did not win
within H. A stalled lead is in the same position but is censored by default (plan 1.4):
stalling is read as sales neglect rather than a buyer decision. ``--stalled-as-lost`` shows
the effect of treating it like any other mature non-win.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from emva.constants import AS_OF, GHOSTED_DAYS, HORIZON_DAYS, OPEN_STAGES, STALLED_DAYS, TEST_FROM


class LabelMode(str, Enum):
    """``--label-mode``: ``legacy`` = the baseline rules, ``horizon`` = won within H days (default)."""

    LEGACY = "legacy"
    HORIZON = "horizon"


# Columns the horizon mode adds to scores.csv (plan 1.1, 1.2): raw outcome, where the label came from, maturity date.
HORIZON_SCORE_COLUMNS: tuple[str, ...] = ("won_within_h", "label_source", "matured_at")


@dataclass(frozen=True)
class LabelConfig:
    """Which label definition to use, and its options.

    ``mode`` accepts a ``LabelMode`` or its string value. ``horizon_days``, ``include_ghosted``
    and ``stalled_as_lost`` apply to the horizon mode only; setting either flag in legacy mode
    is an error (legacy already counts ghosted and stalled leads as Lost).
    """

    mode: LabelMode = LabelMode.HORIZON
    horizon_days: int = HORIZON_DAYS
    include_ghosted: bool = False
    stalled_as_lost: bool = False

    def __post_init__(self) -> None:
        """Coerce ``mode`` to ``LabelMode`` and reject option combinations that mean nothing."""
        object.__setattr__(self, "mode", LabelMode(self.mode))  # ValueError for an unknown mode
        if self.is_legacy and (self.include_ghosted or self.stalled_as_lost):
            raise ValueError("include_ghosted / stalled_as_lost only apply to the horizon label mode")
        if self.horizon_days < 1:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days}")

    @property
    def is_legacy(self) -> bool:
        """True for the baseline label rules."""
        return self.mode is LabelMode.LEGACY

    def describe(self) -> str:
        """Short human-readable name, e.g. ``horizon H=120`` or ``horizon H=120 +ghosted``."""
        if self.is_legacy:
            return "legacy"
        flags = (" +ghosted" if self.include_ghosted else "") + (" +stalled-as-lost" if self.stalled_as_lost else "")
        return f"horizon H={self.horizon_days}{flags}"

    def eligible(self, X: pd.DataFrame, as_of: pd.Timestamp = AS_OF) -> pd.Series:
        """Rows allowed into training and evaluation: all rows (legacy) or mature rows (horizon)."""
        if self.is_legacy:
            return pd.Series(True, index=X.index)
        return is_mature(X, self.horizon_days, as_of)

    def extra_score_columns(self) -> tuple[str, ...]:
        """Label columns written to scores.csv after the baseline's: none in legacy mode (byte identity)."""
        return () if self.is_legacy else HORIZON_SCORE_COLUMNS


LEGACY = LabelConfig(mode=LabelMode.LEGACY)
HORIZON = LabelConfig(mode=LabelMode.HORIZON)


def _idle_days(X: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Whole days (floored) since the last CRM change, as the legacy stalled rule counts them."""
    return (as_of - X.last_change).dt.days


def _stalled(X: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Last stage is one of ``OPEN_STAGES`` and unchanged for ``STALLED_DAYS`` or more whole days."""
    return X.final_stage.isin(list(OPEN_STAGES)) & (_idle_days(X, as_of) >= STALLED_DAYS)


def label(X: pd.DataFrame, as_of: pd.Timestamp = AS_OF) -> pd.Series:
    """Return the legacy (baseline) label (1.0, 0.0 or NaN) for each row.

    Needs ``created_at``, ``last_change`` and ``final_stage``. Later rules override earlier
    ones in this order: Won, Lost, stalled, ghosted (the rules are disjoint on stage anyway).
    An unrecognised final stage (NaN) is simply unlabelled, as in the baseline.
    """
    age = (as_of - X.created_at).dt.days
    y = pd.Series(np.nan, index=X.index, name="y")
    y.loc[X.final_stage == "Won"] = 1
    y.loc[X.final_stage == "Lost"] = 0
    y.loc[_stalled(X, as_of)] = 0          # stalled 90+ days = Lost
    y.loc[(X.final_stage == "New") & (age >= GHOSTED_DAYS)] = 0  # ghosted 90+ days = Lost
    return y


def label_source(X: pd.DataFrame, horizon_days: int = HORIZON_DAYS, as_of: pd.Timestamp = AS_OF) -> pd.Series:
    """Classify every lead by its CRM state at ``as_of``; first matching rule wins.

    Needs ``final_stage``, ``last_change``, ``created_at`` and ``first_contact_at``.

    - ``won``: last CRM stage is Won.
    - ``crm_lost``: last CRM stage is Lost (the CRM recorded the loss).
    - ``ghosted``: a mature lead (``created_at + horizon_days <= as_of``) that was not
      contacted by ``matured_at`` (no change from New to another stage by then).
    - ``stalled``: last CRM stage is one of ``OPEN_STAGES`` (Contacted, Qualified, Demo booked,
      Proposal) and it has not changed for ``STALLED_DAYS`` (90) or more whole days at ``as_of``.
    - ``open``: everything else still undecided: an open stage changed less than
      ``STALLED_DAYS`` whole days ago, or a lead still at New that is younger than H.

    Raises ``ValueError`` if any lead has no recognised final stage, rather than guessing a
    source for it (legacy mode does not call this, so it never fails where the baseline runs).
    """
    stage = X.final_stage
    known = stage.isin(["Won", "Lost", "New", *OPEN_STAGES])
    if not known.all():
        bad = X.index[~known]
        raise ValueError(f"{len(bad)} leads have no recognised final CRM stage, e.g. {list(bad[:5])}")
    ghosted = is_mature(X, horizon_days, as_of) & ghosted_at_horizon(X, horizon_days)
    src = np.select([stage == "Won", stage == "Lost", ghosted, _stalled(X, as_of)], ["won", "crm_lost", "ghosted", "stalled"],
                    default="open")
    return pd.Series(src, index=X.index, name="label_source")


def matured_at(X: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> pd.Series:
    """``created_at + horizon_days``: when the lead's fixed-horizon outcome is known."""
    return (X.created_at + pd.Timedelta(days=horizon_days)).rename("matured_at")


def is_mature(X: pd.DataFrame, horizon_days: int = HORIZON_DAYS, as_of: pd.Timestamp = AS_OF) -> pd.Series:
    """True where ``created_at + horizon_days <= as_of``."""
    return (matured_at(X, horizon_days) <= as_of).rename("mature")


def ghosted_at_horizon(X: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> pd.Series:
    """True where the lead was still at stage New at ``matured_at``.

    That is: no change to another recognised stage (``first_contact_at``) at or before
    ``created_at + horizon_days``. Only meaningful for mature leads.
    """
    return (~(X.first_contact_at <= matured_at(X, horizon_days))).rename("ghosted_at_h")  # NaT compares False


def won_within_h(X: pd.DataFrame, horizon_days: int = HORIZON_DAYS, as_of: pd.Timestamp = AS_OF) -> pd.Series:
    """Fixed-horizon outcome (1.0, 0.0 or NaN); see the module docstring. Needs ``created_at``, ``won_at``."""
    won = X.won_at <= matured_at(X, horizon_days)  # NaT (never Won) compares False
    mature = is_mature(X, horizon_days, as_of)
    return pd.Series(np.where(won, 1.0, np.where(mature, 0.0, np.nan)), index=X.index, name="won_within_h")


def horizon_label(X: pd.DataFrame, config: LabelConfig = HORIZON, as_of: pd.Timestamp = AS_OF) -> pd.Series:
    """``won_within_h`` with ghosted-at-H and stalled leads censored unless the config keeps them.

    Needs ``created_at``, ``won_at``, ``first_contact_at``, ``final_stage``, ``last_change``.
    Censoring only ever turns a 0 into NaN: a ghosted-at-H lead cannot have been Won by H,
    and a stalled lead was never Won. The two groups are censored independently of
    ``label_source``'s precedence, so each flag controls exactly its own group.
    """
    y = won_within_h(X, config.horizon_days, as_of).rename("y")
    censor = pd.Series(False, index=X.index)
    if not config.include_ghosted:
        censor |= ghosted_at_horizon(X, config.horizon_days)
    if not config.stalled_as_lost:
        censor |= _stalled(X, as_of)
    y[censor & (y == 0)] = np.nan
    return y


def assign_labels(X: pd.DataFrame, config: LabelConfig, as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    """Add label columns to ``X`` in place and return it.

    ``legacy`` adds ``y`` = ``label`` only. ``horizon`` adds ``label_source``, ``matured_at``,
    ``won_within_h`` (the outcome) and ``y`` = ``horizon_label`` (the outcome with the
    configured censoring).
    """
    if config.is_legacy:
        X["y"] = label(X, as_of)
        return X
    X["label_source"] = label_source(X, config.horizon_days, as_of)
    X["matured_at"] = matured_at(X, config.horizon_days)
    X["won_within_h"] = won_within_h(X, config.horizon_days, as_of)
    X["y"] = horizon_label(X, config, as_of)
    return X


def split_masks(X: pd.DataFrame, test_from: str = TEST_FROM,
                eligible: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """Return ``(train, test)`` boolean masks: labelled rows created before / on-or-after ``test_from``.

    ``eligible`` (for example ``LabelConfig.eligible(X)``) further restricts both masks; None keeps every
    labelled row (the legacy split).
    """
    lab = X.y.notna()
    if eligible is not None:
        lab = lab & eligible
    return lab & (X.created_at < test_from), lab & (X.created_at >= test_from)
