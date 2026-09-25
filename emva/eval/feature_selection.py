"""Plan 2.6: keep or drop ``c_budget``, ``c_timeline``, ``ip_type`` and ``edits_1_4``.

``python -m emva.eval.feature_selection [--data data/v1] [--n-refits 1000] [label options]``
builds the v2 design with *every* candidate feature (``CATS_V2_CANDIDATES``, before
``emva.features.V2_DROPPED`` is applied), fits the formula model on the training rows, and
bootstraps each coefficient (``emva.eval.bootstrap.coef_bootstrap``: training rows resampled
with replacement, model refitted). A feature is kept when at least one of its levels has a 95%
CI that excludes zero, otherwise dropped. The decision is then hard-coded in
``emva.features.V2_DROPPED``; this script is the evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from emva.cli import add_label_arguments, label_config
from emva.constants import CATS_V2_CANDIDATES, TEST_FROM
from emva.design import fixed_columns, fixed_design
from emva.eval.bootstrap import N_RESAMPLES, SEED, coef_bootstrap
from emva.eval.report import md_table
from emva.features import FeatureSet
from emva.io import load
from emva.labels import HORIZON, LabelConfig, split_masks
from emva.model import make_lr
from emva.pipeline import build

SELECTION_CANDIDATES: tuple[str, ...] = ("c_budget", "c_timeline", "ip_type", "edits_1_4")
MIN_REFITS: int = 200


def _lr_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coefficients of the formula model (``emva.model.make_lr``) fitted on ``X``, ``y``."""
    return make_lr().fit(X, y).coef_[0]


def selection_table(data: str | Path, n_refits: int = N_RESAMPLES, seed: int = SEED,
                    labels: LabelConfig = HORIZON,
                    candidates: tuple[str, ...] = SELECTION_CANDIDATES) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Per-level coefficient CIs for ``candidates`` and the keep (True) / drop (False) decision per feature.

    Raises ``ValueError`` for fewer than ``MIN_REFITS`` refits (the plan's minimum for this decision).
    """
    if n_refits < MIN_REFITS:
        raise ValueError(f"need at least {MIN_REFITS} refits, got {n_refits}")
    X = build(load(data), labels, FeatureSet.V2)
    tr, _ = split_masks(X, TEST_FROM, labels.eligible(X))
    D = fixed_design(X, CATS_V2_CANDIDATES)
    cis = dict(zip(D.columns, coef_bootstrap(D[tr].values, X.y[tr].values, _lr_coef, n_resamples=n_refits, seed=seed),
                   strict=True))
    rows, keep = [], {}
    for feat in candidates:
        kept_any = False
        for col in fixed_columns({feat: CATS_V2_CANDIDATES[feat]}):
            n_train = int(X.loc[tr, feat].astype(str).eq(col.split("=", 1)[1]).sum())
            ci = cis[col]
            excludes = ci.lo > 0 or ci.hi < 0
            kept_any |= excludes
            rows.append({"feature": feat, "level": col, "log-odds [95% CI]": f"{ci.point:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]",
                         "excludes 0": "yes" if excludes else "no", "n train": n_train})
        keep[feat] = kept_any
    return pd.DataFrame(rows), keep


def build_selection(data: str | Path, n_refits: int = N_RESAMPLES, seed: int = SEED,
                    labels: LabelConfig = HORIZON) -> str:
    """The selection table and per-feature decisions as markdown."""
    table, keep = selection_table(data, n_refits, seed, labels)
    decisions = pd.DataFrame([{"feature": f, "decision": "keep" if k else "drop",
                               "reason": "a level's CI excludes 0" if k else "every level's CI contains 0"}
                              for f, k in keep.items()])
    return "\n".join([
        "## Plan 2.6: feature selection by coefficient bootstrap", "",
        f"Data `{data}`, labels `{labels.describe()}`, v2 design with every candidate feature, training rows "
        f"(created before {TEST_FROM}). 95% percentile CI from {n_refits} refits on resampled training rows "
        f"(seed {seed}).", "",
        md_table(table), "", md_table(decisions), ""])


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: print the selection."""
    ap = argparse.ArgumentParser(prog="python -m emva.eval.feature_selection")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--n-refits", type=int, default=N_RESAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    add_label_arguments(ap)
    a = ap.parse_args(argv)
    print(build_selection(a.data, a.n_refits, a.seed, label_config(ap, a)))


if __name__ == "__main__":
    main()
