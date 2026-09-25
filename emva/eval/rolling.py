"""Rolling-origin evaluation (plan 4.1) and the evaluation frame the Phase 4 sections share.

The split (ADR 0013). At split date S, with horizon H (default 120 days):

- **train** = labelled leads created before S whose horizon label is mature at S
  (``created_at + H <= S``): exactly what a model fitted on S could have learned from;
- **test** = labelled leads created in ``[S, S + 1 month)`` whose label is mature at ``AS_OF``
  (``created_at + H <= AS_OF``), scored by that model.

Labels are the pipeline's default horizon labels (``emva.labels.HORIZON``: won within H, ghosted-at-H
excluded, stalled censored), computed once at ``AS_OF``. For a lead mature at S the outcome
``won_within_h`` is already fixed at S (a Won change after ``created_at + H`` counts as 0), and so is
"ghosted at H"; only the stalled censoring reads the CRM state at ``AS_OF``, because the frame has
no stage history at S. That can only drop a training row, never flip its label.

``RollingSpec`` also expresses the comparison variants: *relaxed* training (created before S, mature
at ``AS_OF``, which is how the frozen split trains) and legacy labels (every legacy-labelled lead
created before S; test = legacy-labelled leads in the window, or every lead from S onwards to
reproduce the review's baseline spread). A split whose test set has fewer than ``MIN_TEST_LEADS``
labelled leads or fewer than ``MIN_TEST_CLASS`` of either class is reported as insufficient and left
out of the headline mean, never dropped silently.

The headline is the mean of the per-split AUCs over sufficient splits, with a percentile bootstrap CI
from resampling each split's test set independently (``headline``).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from emva.constants import AS_OF, HORIZON_DAYS, TEST_FROM
from emva.eval.bootstrap import CI_LEVEL, N_RESAMPLES, SEED, BootstrapCI, auc_ci, paired_auc
from emva.feature_spec import feature_spec
from emva.features import FeatureSet
from emva.io import load
from emva.labels import HORIZON, label, split_masks
from emva.model import make_lr
from emva.pipeline import build

SPLITS: tuple[pd.Timestamp, ...] = tuple(pd.Timestamp(d, tz="UTC") for d in (
    "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01"))
TEST_WINDOW: pd.DateOffset = pd.DateOffset(months=1)
MIN_TEST_LEADS: int = 100
MIN_TEST_CLASS: int = 10

# fit_predict(D, y, train_mask) -> P(won) for every row of D
FitPredict = Callable[[pd.DataFrame, pd.Series, pd.Series], np.ndarray]


def lr_fit_predict(D: pd.DataFrame, y: pd.Series, train: pd.Series) -> np.ndarray:
    """The pipeline's formula model (``emva.model.make_lr``) fitted on ``train`` rows; P(won) for every row."""
    return make_lr().fit(D[train], y[train]).predict_proba(D)[:, 1]


@dataclass
class EvalFrame:
    """One dataset prepared once for every Phase 4 section.

    ``X`` = cleaned leads with default horizon labels (``y``, ``won_within_h``) and v2 features;
    ``legacy_y`` = the baseline's label for the same rows; ``D`` = the v2 design (fixed schema);
    ``D_legacy`` = the baseline's legacy-feature design (for the baseline-continuity rows).
    """

    data: Path
    X: pd.DataFrame
    legacy_y: pd.Series
    D: pd.DataFrame
    D_legacy: pd.DataFrame
    horizon_days: int = HORIZON_DAYS

    @property
    def name(self) -> str:
        """The data directory's name (``v1``, ``v2``)."""
        return self.data.name

    def frozen_horizon(self) -> tuple[pd.Series, pd.Series]:
        """The pipeline's horizon split: ``(train, test)`` = labelled mature leads before / from ``TEST_FROM``.

        The test mask is the frozen mature test set (b) of ``emva.eval.report``.
        """
        return split_masks(self.X, TEST_FROM, HORIZON.eligible(self.X))

    def frozen_legacy(self) -> tuple[pd.Series, pd.Series]:
        """The baseline's split on legacy labels: ``(train, test)``; the test mask is the frozen legacy test set (a)."""
        lab, before = self.legacy_y.notna(), self.X.created_at < TEST_FROM
        return lab & before, lab & ~before


def prepare(data: str | Path) -> EvalFrame:
    """Load, clean, label (horizon + legacy) and featurise ``data`` (v2 and legacy feature sets)."""
    L = load(data)
    X = build(L, HORIZON, FeatureSet.V2)
    X_leg = build(L, HORIZON, FeatureSet.LEGACY)
    return EvalFrame(data=Path(data), X=X, legacy_y=label(X), D=feature_spec(FeatureSet.V2).design(X),
                     D_legacy=feature_spec(FeatureSet.LEGACY).design(X_leg), horizon_days=HORIZON.horizon_days)


@dataclass(frozen=True)
class RollingSpec:
    """One rolling-origin variant: which label, which training rule and which test window.

    ``horizon_days`` None = legacy labels (no maturity rule). ``strict`` (horizon only): training
    rows must be mature at the split (True) or at ``as_of`` (False). ``test_window`` None = test on
    every labelled lead from the split onwards.
    """

    name: str
    horizon_days: int | None
    strict: bool = True
    test_window: pd.DateOffset | None = field(default=TEST_WINDOW)

    def masks(self, created_at: pd.Series, y: pd.Series, split: pd.Timestamp,
              as_of: pd.Timestamp = AS_OF) -> tuple[pd.Series, pd.Series]:
        """``(train, test)`` boolean masks at ``split``; see the module docstring for the rules."""
        labelled = y.notna()
        before = created_at < split
        in_window = created_at >= split
        if self.test_window is not None:
            in_window &= created_at < split + self.test_window
        if self.horizon_days is None:
            return labelled & before, labelled & in_window
        h = pd.Timedelta(days=self.horizon_days)
        mature_at_split = created_at + h <= split
        mature_now = created_at + h <= as_of
        train = labelled & before & (mature_at_split if self.strict else mature_now)
        return train, labelled & in_window & mature_now


def horizon_strict(horizon_days: int = HORIZON_DAYS) -> RollingSpec:
    """The headline variant (ADR 0013): horizon labels, training mature at the split, one-month test window."""
    return RollingSpec(f"horizon H={horizon_days}, train mature at S", horizon_days, strict=True)


def horizon_relaxed(horizon_days: int = HORIZON_DAYS) -> RollingSpec:
    """Comparison: horizon labels, training rows created before S but mature only at AS_OF (look-ahead)."""
    return RollingSpec(f"horizon H={horizon_days}, train mature at AS_OF", horizon_days, strict=False)


LEGACY_WINDOW = RollingSpec("legacy labels, one-month test", None)
LEGACY_TO_END = RollingSpec("legacy labels, test = everything from S", None, test_window=None)


@dataclass
class SplitResult:
    """One split's sizes, AUC (with CI when computed) and test scores."""

    split: pd.Timestamp
    n_train: int
    wins_train: int
    n_test: int
    wins_test: int
    auc: float | None
    ci: BootstrapCI | None
    y_test: np.ndarray
    score_test: np.ndarray

    @property
    def sufficient(self) -> bool:
        """Enough test leads of both classes for the headline (``MIN_TEST_LEADS``, ``MIN_TEST_CLASS``)."""
        return (self.n_test >= MIN_TEST_LEADS and self.wins_test >= MIN_TEST_CLASS
                and self.n_test - self.wins_test >= MIN_TEST_CLASS and self.auc is not None)

    def status(self) -> str:
        """Human-readable reason the split is or is not in the headline."""
        if self.n_train == 0 or self.wins_train in (0, self.n_train):
            return "no fit: training set lacks a class"
        if self.n_test == 0:
            return "no labelled test leads"
        if not self.sufficient:
            return f"insufficient test set (< {MIN_TEST_LEADS} leads or < {MIN_TEST_CLASS} of a class)"
        return "ok"


def rolling_origin(spec: RollingSpec, D: pd.DataFrame, y: pd.Series, created_at: pd.Series,
                   fit_predict: FitPredict = lr_fit_predict, splits: Sequence[pd.Timestamp] = SPLITS,
                   with_ci: bool = True, n_resamples: int = N_RESAMPLES, seed: int = SEED,
                   as_of: pd.Timestamp = AS_OF) -> list[SplitResult]:
    """Fit at every split and score its test window; one ``SplitResult`` per split, in order.

    A split whose training set lacks a class is not fitted (AUC None); one whose test set lacks a
    class has AUC None. CIs (percentile bootstrap, ``n_resamples``, ``seed``) only with ``with_ci``.
    """
    out = []
    for s in splits:
        tr, te = spec.masks(created_at, y, s, as_of)
        ytr, yte = y[tr], y[te]
        n_tr, w_tr, n_te, w_te = int(tr.sum()), int((ytr == 1).sum()), int(te.sum()), int((yte == 1).sum())
        auc, ci, score = None, None, np.array([])
        if 0 < w_tr < n_tr and n_te:
            score = fit_predict(D, y, tr)[te.values]
            if 0 < w_te < n_te:
                auc = float(roc_auc_score(yte, score))
                ci = auc_ci(yte.values, score, n_resamples, seed=seed) if with_ci else None
        out.append(SplitResult(s, n_tr, w_tr, n_te, w_te, auc, ci, yte.values.astype(float), score))
    return out


@dataclass(frozen=True)
class Headline:
    """Mean per-split AUC over sufficient splits, its bootstrap CI, and the range of those AUCs."""

    mean: float
    lo: float
    hi: float
    min: float
    max: float
    n_splits: int

    @property
    def point(self) -> float:
        """The mean AUC (named like ``BootstrapCI.point`` so both format the same way)."""
        return self.mean

    @property
    def spread(self) -> float:
        """``max - min`` of the per-split AUCs."""
        return self.max - self.min


def headline(results: Sequence[SplitResult], n_resamples: int = N_RESAMPLES, seed: int = SEED,
             level: float = CI_LEVEL) -> Headline | None:
    """Mean AUC over the sufficient splits; CI by resampling each split's test set independently.

    Split ``i`` (in order among the sufficient ones) uses bootstrap seed ``seed + i``, so the splits'
    resamples are independent. None when no split is sufficient.
    """
    ok = [r for r in results if r.sufficient]
    if not ok:
        return None
    samples = np.mean([auc_ci(r.y_test, r.score_test, n_resamples, level, seed + i).samples
                       for i, r in enumerate(ok)], axis=0)
    aucs = [r.auc for r in ok]
    alpha = (1 - level) / 2
    lo, hi = np.quantile(samples, [alpha, 1 - alpha])
    return Headline(float(np.mean(aucs)), float(lo), float(hi), float(min(aucs)), float(max(aucs)), len(ok))


def paired_headline(a: Sequence[SplitResult], b: Sequence[SplitResult], n_resamples: int = N_RESAMPLES,
                    seed: int = SEED, level: float = CI_LEVEL) -> BootstrapCI | None:
    """Mean over splits of AUC(b) − AUC(a), on splits sufficient in both, with a paired bootstrap CI.

    ``a`` and ``b`` must come from the same spec and splits (identical test rows). Split ``i`` resamples
    with seed ``seed + i``; the CI is the percentile interval of the mean of the per-split differences.
    """
    pairs = [(ra, rb) for ra, rb in zip(a, b, strict=True) if ra.sufficient and rb.sufficient]
    for ra, rb in pairs:
        if ra.split != rb.split or not np.array_equal(ra.y_test, rb.y_test):
            raise ValueError(f"split {ra.split.date()}: the two results do not share test rows")
    if not pairs:
        return None
    comps = [paired_auc(ra.y_test, ra.score_test, rb.score_test, n_resamples, level, seed + i)
             for i, (ra, rb) in enumerate(pairs)]
    samples = np.mean([c.diff.samples for c in comps], axis=0)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(samples, [alpha, 1 - alpha])
    return BootstrapCI(float(np.mean([c.diff.point for c in comps])), float(lo), float(hi), samples)


def mean_auc(results: Sequence[SplitResult]) -> float | None:
    """Mean AUC over the sufficient splits (the headline point estimate, without a CI); None if there are none."""
    ok = [r.auc for r in results if r.sufficient]
    return float(np.mean(ok)) if ok else None


def results_table(results: Sequence[SplitResult]) -> pd.DataFrame:
    """Per-split table: split, train/test sizes and wins, AUC [95% CI], status."""
    rows = []
    for r in results:
        auc = "n/a" if r.auc is None else (f"{r.auc:.3f}" if r.ci is None else f"{r.auc:.3f} [{r.ci.lo:.3f}, {r.ci.hi:.3f}]")
        rows.append({"split S": r.split.date().isoformat(), "train (wins)": f"{r.n_train} ({r.wins_train})",
                     "test (wins)": f"{r.n_test} ({r.wins_test})", "AUC [95% CI]": auc, "status": r.status()})
    return pd.DataFrame(rows)


def format_headline(h: Headline | None) -> str:
    """``0.812 [0.790, 0.833] over 4 splits (range 0.80 to 0.83)``, or a sentence saying there is none."""
    if h is None:
        return "no split has a sufficient test set"
    return (f"{h.mean:.3f} [{h.lo:.3f}, {h.hi:.3f}] over {h.n_splits} splits "
            f"(per-split range {h.min:.3f} to {h.max:.3f}, spread {h.max - h.min:.3f})")
