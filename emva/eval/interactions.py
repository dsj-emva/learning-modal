"""Interaction detectability (Phase 5 hand-off): can the formula model see the two planted v2 interactions?

Generator v2 plants LinkedIn × company size 51+ (+0.8) and senior title × form D (−0.8) on close
log-odds (``data/v2/ground_truth.md``, 5.5). This check adds the two products of the pipeline's *own*
bucketed features to the v2 design (no ground truth: ``channel == linkedin`` × ``band`` in 51-200,
201-1000, 1000+; ``seniority == senior`` × ``form_variant == D``) and asks whether

1. their coefficients' 95% bootstrap CIs (1,000 refits of the training rows, seed 0) exclude zero, and
2. held-out AUC improves (paired bootstrap on the frozen test sets and on the rolling-origin headline).

On v1 the terms are not planted, so v1 is the null control.

The coefficient check uses no ground truth. ``null_coefficients`` does (evaluation only, ground rule 2):
it simulates the horizon labels of the training rows from the generator's documented outcome mechanism
(``scripts/generate_data_v1.py``: ``draw_outcome`` and ``make_crm``) and refits, giving the sampling
distribution of each interaction coefficient under the planted truth. On v1 that is the null
distribution; on v2 it is the distribution expected given the planted +0.8 / −0.8. Per training row and draw:

- a generator-stalled lead (``ground_truth_labels.csv`` column ``stalled``: stalled and still open) is never
  Won, so its label is 0. Stalling and ghosting are drawn from the status-quo tier only, independently of the
  win draw, so the set of training rows (mature, not ghosted, not censored as stalled) is held fixed;
- otherwise Won ~ Bernoulli(``p_close_true``). A Won deal closes ``close_days`` after creation: lognormal with
  the median and log-sd stated in ``ground_truth.md`` ("Time to close"), times the stated multiplier for true
  band 1000+, clipped to [``CLOSE_DAYS_MIN``, ``CLOSE_DAYS_MAX``] as the generator does. (The close cannot come
  before the first reply + 1 h, at most about 10 days after the lead; that never matters at H = 120.)
- label = Won and ``close_days <= H``: the pipeline's horizon label for that row.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from emva.eval.bootstrap import N_RESAMPLES, SEED, BootstrapCI, coef_bootstrap
from emva.model import make_lr

INTERACTION_BANDS: tuple[str, ...] = ("51-200", "201-1000", "1000+")
INTERACTION_COLUMNS: tuple[str, ...] = ("linkedin_x_51plus", "senior_x_formD")
PLANTED_V2: dict[str, float] = {"linkedin_x_51plus": 0.8, "senior_x_formD": -0.8}


def interaction_columns(X: pd.DataFrame) -> pd.DataFrame:
    """The two interaction columns (0/1 floats) from the pipeline's v2 features ``channel``, ``band``, ``seniority``, ``form_variant``."""
    return pd.DataFrame({
        "linkedin_x_51plus": (X.channel.eq("linkedin") & X.band.isin(INTERACTION_BANDS)).astype(float),
        "senior_x_formD": (X.seniority.eq("senior") & X.form_variant.astype(str).eq("D")).astype(float),
    }, index=X.index)


def with_interactions(D: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """``D`` with ``interaction_columns(X)`` appended."""
    return D.join(interaction_columns(X))


def interaction_coef_cis(D: pd.DataFrame, y: pd.Series, train: pd.Series, n_resamples: int = N_RESAMPLES,
                         seed: int = SEED) -> dict[str, BootstrapCI]:
    """Bootstrap CIs of the interaction coefficients of the formula model fitted on ``train`` rows of ``D``.

    ``D`` must already contain ``INTERACTION_COLUMNS`` (``with_interactions``).
    """
    pos = [D.columns.get_loc(c) for c in INTERACTION_COLUMNS]
    cis = coef_bootstrap(D[train].to_numpy(dtype=float), y[train].to_numpy(dtype=float),
                         lambda A, b: make_lr().fit(A, b).coef_[0][pos], n_resamples, seed=seed)
    return dict(zip(INTERACTION_COLUMNS, cis))


# ---------------------------------------------------------------------------------------------------------------
# Sampling distribution under the generator's outcome mechanism (reads ground truth)
# ---------------------------------------------------------------------------------------------------------------

N_NULL_DRAWS: int = 1000
# scripts/generate_data_v1.py::draw_outcome clips a Won deal's days to close to this range.
CLOSE_DAYS_MIN: float = 2.0
CLOSE_DAYS_MAX: float = 330.0
SLOW_BAND: str = "1000+"
TIME_TO_CLOSE_SENTENCE: str = (r"Won deals take a lognormal number of days \(median ([\d.]+), sd ([\d.]+)\); 1000\+\s+"
                               r"companies take ([\d.]+)x longer")
NULL_TRUTH_COLUMNS: tuple[str, ...] = ("lead_id", "p_close_true", "employee_band", "stalled")


@dataclass(frozen=True)
class TimeToWin:
    """Planted days from creation to a Won close: lognormal(log median, log_sd), times ``slow_mult`` for 1000+."""

    median_days: float
    log_sd: float
    slow_mult: float

    def draw(self, slow: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Days to close for rows flagged ``slow`` (true band 1000+) or not, clipped as the generator does."""
        days = np.exp(rng.normal(np.log(self.median_days), self.log_sd, len(slow)))
        return np.clip(days * np.where(slow, self.slow_mult, 1.0), CLOSE_DAYS_MIN, CLOSE_DAYS_MAX)


def read_time_to_win(data: str | Path) -> TimeToWin:
    """Parse the "Time to close" sentence of ``ground_truth.md``; raises ``ValueError`` if it is missing."""
    m = re.search(TIME_TO_CLOSE_SENTENCE, (Path(data) / "ground_truth.md").read_text())
    if m is None:
        raise ValueError(f"time-to-close sentence not found in {data}/ground_truth.md")
    return TimeToWin(float(m.group(1)), float(m.group(2)), float(m.group(3)))


@dataclass(frozen=True)
class NullTruth:
    """Per-row truth the simulation needs (``p_close_true``, true band 1000+, generator-stalled), row-aligned."""

    p_close: np.ndarray
    slow: np.ndarray
    stalled: np.ndarray
    time_to_win: TimeToWin

    def subset(self, mask: np.ndarray) -> NullTruth:
        """The rows selected by boolean ``mask``."""
        return NullTruth(self.p_close[mask], self.slow[mask], self.stalled[mask], self.time_to_win)


def read_null_truth(data: str | Path, index: pd.Index) -> NullTruth:
    """``NullTruth`` for the leads in ``index`` (in that order); raises if a lead has no ground-truth row."""
    t = pd.read_csv(Path(data) / "ground_truth_labels.csv", usecols=list(NULL_TRUTH_COLUMNS)).set_index("lead_id")
    t = t.reindex(index)
    if t.p_close_true.isna().any():
        raise ValueError(f"{int(t.p_close_true.isna().sum())} leads have no ground-truth row")
    return NullTruth(t.p_close_true.to_numpy(dtype=float), t.employee_band.eq(SLOW_BAND).to_numpy(),
                     (t.stalled == True).to_numpy(), read_time_to_win(data))  # noqa: E712 (NaN-safe)


def simulate_horizon_labels(truth: NullTruth, horizon_days: int, rng: np.random.Generator) -> np.ndarray:
    """One draw of the horizon label (0/1 floats) for every row of ``truth`` (see the module docstring)."""
    won = rng.random(len(truth.p_close)) < truth.p_close
    days = truth.time_to_win.draw(truth.slow, rng)
    return (won & (days <= horizon_days) & ~truth.stalled).astype(float)


@dataclass(frozen=True)
class NullDistribution:
    """Simulated coefficients (draws × terms), the observed coefficients, and simulated vs observed win rates."""

    columns: tuple[str, ...]
    draws: np.ndarray
    observed: np.ndarray
    sim_rate: float
    observed_rate: float

    def summary(self) -> pd.DataFrame:
        """Per term: observed, simulated mean and sd, central 95% interval, two-sided share at least as extreme."""
        rows = []
        for j, c in enumerate(self.columns):
            d, o = self.draws[:, j], float(self.observed[j])
            lo, hi = np.quantile(d, [0.025, 0.975])
            rows.append({"term": c, "observed": o, "sim_mean": float(d.mean()), "sim_sd": float(d.std(ddof=1)),
                         "sim_lo": float(lo), "sim_hi": float(hi),
                         "share_as_extreme": float((np.abs(d - d.mean()) >= abs(o - d.mean())).mean()),
                         "inside_95": bool(lo <= o <= hi)})
        return pd.DataFrame(rows)


def null_coefficients(D: pd.DataFrame, y: pd.Series, train: pd.Series, truth: NullTruth, horizon_days: int,
                      columns: tuple[str, ...] = INTERACTION_COLUMNS, n_draws: int = N_NULL_DRAWS,
                      seed: int = SEED) -> NullDistribution:
    """Refit the formula model on ``n_draws`` simulated horizon-label sets for the ``train`` rows of ``D``.

    ``truth`` must be aligned to ``D``'s rows (``read_null_truth(data, D.index)``); ``y`` gives the observed
    coefficients and win rate on the same rows. Draws use ``numpy.random.default_rng(seed)`` in sequence.
    """
    tr = train.to_numpy(dtype=bool)
    A = D.to_numpy(dtype=float)[tr]
    pos = [D.columns.get_loc(c) for c in columns]
    sub = truth.subset(tr)
    rng = np.random.default_rng(seed)
    draws, rates = [], []
    for _ in range(n_draws):
        ys = simulate_horizon_labels(sub, horizon_days, rng)
        draws.append(make_lr().fit(A, ys).coef_[0][pos])
        rates.append(ys.mean())
    yo = y[train].to_numpy(dtype=float)
    observed = make_lr().fit(A, yo).coef_[0][pos]
    return NullDistribution(tuple(columns), np.array(draws), np.asarray(observed), float(np.mean(rates)),
                            float(yo.mean()))
