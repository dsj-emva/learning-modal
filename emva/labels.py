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

``label_source`` describes each lead's CRM state at ``as_of`` (plan 1.1); see
``label_source`` for the exact definitions. Because a lead that is still New at
``matured_at`` is, on data where stages never go back to New, either still New at ``as_of``
or contacted after H, "ghosted at H" and ``label_source == "ghosted"`` coincide for mature
leads unless the first contact came after H (never on v1, where first contact is at most
10 days after creation).

Only mature leads may be used to train or evaluate horizon labels (``split_masks`` with
``eligible=is_mature(...)``): young leads can only be labelled 1, so including them would
bias the win rate upwards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from emva.constants import (
    AS_OF,
    DEFAULT_LABEL_MODE,
    GHOSTED_DAYS,
    HORIZON_DAYS,
    LABEL_MODES,
    OPEN_STAGES,
    STALLED_DAYS,
    TEST_FROM,
)


@dataclass(frozen=True)
class LabelConfig:
    """Which label definition to use, and its options.

    ``horizon_days``, ``include_ghosted`` and ``stalled_as_lost`` apply to the ``horizon``
    mode only; setting either flag in ``legacy`` mode is an error (legacy already counts
    ghosted and stalled leads as Lost).
    """

    mode: str = DEFAULT_LABEL_MODE
    horizon_days: int = HORIZON_DAYS
    include_ghosted: bool = False
    stalled_as_lost: bool = False

    def __post_init__(self) -> None:
        if self.mode not in LABEL_MODES:
            raise ValueError(f"label mode must be one of {LABEL_MODES}, got {self.mode!r}")
        if self.mode == "legacy" and (self.include_ghosted or self.stalled_as_lost):
            raise ValueError("include_ghosted / stalled_as_lost only apply to the horizon label mode")
        if self.horizon_days < 1:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days}")

    def describe(self) -> str:
        """Short human-readable name, e.g. ``horizon H=120`` or ``horizon H=120 +ghosted``."""
        if self.mode == "legacy":
            return "legacy"
        flags = (" +ghosted" if self.include_ghosted else "") + (" +stalled-as-lost" if self.stalled_as_lost else "")
        return f"horizon H={self.horizon_days}{flags}"


LEGACY = LabelConfig(mode="legacy")
HORIZON = LabelConfig(mode="horizon")


def label(X: pd.DataFrame, as_of: pd.Timestamp = AS_OF) -> pd.Series:
    """Return the legacy (baseline) label (1.0, 0.0 or NaN) for each row.

    Needs ``created_at``, ``last_change`` and ``final_stage``. Later rules override earlier
    ones in this order: Won, Lost, stalled, ghosted (the rules are disjoint on stage anyway).
    """
    age = (as_of - X.created_at).dt.days
    idle = (as_of - X.last_change).dt.days
    open_ = X.final_stage.isin(list(OPEN_STAGES))
    y = pd.Series(np.nan, index=X.index, name="y")
    y.loc[X.final_stage == "Won"] = 1
    y.loc[X.final_stage == "Lost"] = 0
    y.loc[open_ & (idle >= STALLED_DAYS)] = 0          # stalled 90+ days = Lost
    y.loc[(X.final_stage == "New") & (age >= GHOSTED_DAYS)] = 0  # ghosted 90+ days = Lost
    return y


def label_source(X: pd.DataFrame, as_of: pd.Timestamp = AS_OF) -> pd.Series:
    """Classify every lead by its CRM state at ``as_of`` (needs ``final_stage``, ``last_change``).

    - ``won``: last CRM stage is Won.
    - ``crm_lost``: last CRM stage is Lost (the CRM recorded the loss).
    - ``stalled``: last CRM stage is one of ``OPEN_STAGES`` (Contacted, Qualified, Demo booked,
      Proposal) and it has not changed for ``STALLED_DAYS`` (90) or more whole days at ``as_of``.
    - ``ghosted``: last CRM stage is still New (sales never moved it on), whatever its age.
    - ``open``: last CRM stage is one of ``OPEN_STAGES`` and it changed less than
      ``STALLED_DAYS`` whole days before ``as_of``: active pipeline, not decided yet.

    Idle days are whole days (floored), as in the legacy stalled rule. Raises ``ValueError``
    if any lead has no recognised final stage, rather than guessing a source for it.
    """
    stage = X.final_stage
    known = stage.isin(["Won", "Lost", "New", *OPEN_STAGES])
    if not known.all():
        bad = X.index[~known]
        raise ValueError(f"{len(bad)} leads have no recognised final CRM stage, e.g. {list(bad[:5])}")
    idle = (as_of - X.last_change).dt.days
    open_ = stage.isin(list(OPEN_STAGES))
    # Every stage is recognised (checked above), so what is left is an open stage idle < STALLED_DAYS.
    src = np.select(
        [stage == "Won", stage == "Lost", open_ & (idle >= STALLED_DAYS), stage == "New"],
        ["won", "crm_lost", "stalled", "ghosted"],
        default="open",
    )
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
    and a stalled lead was never Won.
    """
    y = won_within_h(X, config.horizon_days, as_of).rename("y")
    censor = pd.Series(False, index=X.index)
    if not config.include_ghosted:
        censor |= ghosted_at_horizon(X, config.horizon_days)
    if not config.stalled_as_lost:
        censor |= label_source(X, as_of) == "stalled"
    y[censor & (y == 0)] = np.nan
    return y


def assign_labels(X: pd.DataFrame, config: LabelConfig, as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    """Add label columns to ``X`` in place and return it.

    Always adds ``label_source``. ``legacy`` adds ``y`` = ``label``. ``horizon`` adds
    ``matured_at``, ``won_within_h`` (the outcome) and ``y`` = ``horizon_label`` (the outcome
    with the configured censoring).
    """
    X["label_source"] = label_source(X, as_of)
    if config.mode == "legacy":
        X["y"] = label(X, as_of)
        return X
    X["matured_at"] = matured_at(X, config.horizon_days)
    X["won_within_h"] = won_within_h(X, config.horizon_days, as_of)
    X["y"] = horizon_label(X, config, as_of)
    return X


def eligible_rows(X: pd.DataFrame, config: LabelConfig, as_of: pd.Timestamp = AS_OF) -> pd.Series:
    """Rows allowed into training and evaluation: all rows (legacy) or mature rows (horizon)."""
    if config.mode == "legacy":
        return pd.Series(True, index=X.index)
    return is_mature(X, config.horizon_days, as_of)


def split_masks(X: pd.DataFrame, test_from: str = TEST_FROM,
                eligible: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """Return ``(train, test)`` boolean masks: labelled rows created before / on-or-after ``test_from``.

    ``eligible`` (for example ``is_mature``) further restricts both masks; None keeps every
    labelled row (the legacy split).
    """
    lab = X.y.notna()
    if eligible is not None:
        lab = lab & eligible
    return lab & (X.created_at < test_from), lab & (X.created_at >= test_from)
