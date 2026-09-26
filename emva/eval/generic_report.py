"""The numbers of the standard report's "generic vs v2" section (Phase 10, ADR 0024; ``emva.eval.report`` renders it),
computed when the candidate is ``generic``.

Both models are trained by ``emva.pipeline.run`` on the same data, labels and split (so the same rows): the v2 fixed
design, and the v2 design plus the declared extras' ``x_`` columns. The section shows, on both frozen test sets, each
model's AUC and the paired bootstrap AUC difference generic − v2 (``emva.eval.bootstrap.paired_auc``: 1000 resamples,
seed 0); the generic design's collinearity check (the candidate's, from ``emva.eval.collinearity``); and a leakage
screen: per extra, the single-feature training AUC of its encoded levels (each level scored by its training win
rate), folded as ``|AUC − 0.5| + 0.5``; above ``LEAKAGE_AUC_FLAG`` it is flagged "check for leakage" (a flag only:
nothing is dropped).

The Phase 10 acceptance criterion (pre-registered in ``reports/phase10.md``) is printed as a verdict: PASS iff on the
horizon test set (b) the 95% CI of generic − v2 lies entirely above 0 and the generic design passes the collinearity
check.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score

from emva.eval.bootstrap import PairedComparison, paired_auc
from emva.eval.collinearity import CollinearityResult
from emva.pipeline import PipelineResult

# A single extra whose folded training AUC exceeds this is flagged "check for leakage" (never dropped).
LEAKAGE_AUC_FLAG: float = 0.90
LEAKAGE_FLAG_TEXT: str = "check for leakage"


@dataclass(frozen=True)
class GenericComparison:
    """``paired`` maps each test set's title to its paired comparison (a = v2, b = generic) and ``sizes`` to its
    number of leads; ``passed`` is the Phase 10 criterion on the horizon test set; ``screen`` is ``leakage_screen``'s
    table."""

    paired: dict[str, PairedComparison]
    sizes: dict[str, int]
    passed: bool
    screen: pd.DataFrame


def leakage_screen(result: PipelineResult) -> pd.DataFrame:
    """One row per extra of a generic-feature-set ``result``: ``extra``, ``kind``, ``levels``, ``design columns``,
    ``training AUC`` (folded single-feature AUC on the training leads, NaN when they hold one class) and ``flag``.

    Raises ``ValueError`` when ``result`` has no extras encoder (not the generic feature set).
    """
    if result.extras is None:
        raise ValueError("leakage_screen needs a result trained with the generic feature set")
    T = result.X[result.train]
    y = T.y.astype(float)
    levels = result.extras.levels(T)
    rows = []
    for e in result.extras.extras:
        lv = levels[e.feature.column]
        rate = y.groupby(lv).mean()
        auc = float(roc_auc_score(y, lv.map(rate).astype(float))) if y.nunique() == 2 else float("nan")
        folded = abs(auc - 0.5) + 0.5
        rows.append({"extra": e.feature.name, "kind": e.feature.kind.value, "levels": len(e.levels),
                     "design columns": len(e.columns()), "training AUC": folded,
                     "flag": LEAKAGE_FLAG_TEXT if folded > LEAKAGE_AUC_FLAG else ""})
    return pd.DataFrame(rows, columns=["extra", "kind", "levels", "design columns", "training AUC", "flag"])


def compare(generic: PipelineResult, v2: PipelineResult, tests: list[tuple[str, pd.Series]], horizon_title: str,
            collinearity: CollinearityResult, n_resamples: int, seed: int) -> GenericComparison:
    """Paired AUC comparisons of ``generic`` against ``v2`` on each ``(title, y)`` test set (``y`` indexed by
    ``lead_id``), the criterion on the test set titled ``horizon_title`` with ``collinearity`` (the generic design's
    check), and the leakage screen."""
    paired = {title: paired_auc(y.values, v2.X.p_formula.loc[y.index].values, generic.X.p_formula.loc[y.index].values,
                                n_resamples, seed=seed) for title, y in tests}
    passed = paired[horizon_title].diff.lo > 0 and collinearity.passed
    return GenericComparison(paired, {title: len(y) for title, y in tests}, bool(passed), leakage_screen(generic))


__all__ = ["GenericComparison", "LEAKAGE_AUC_FLAG", "LEAKAGE_FLAG_TEXT", "compare", "leakage_screen"]
