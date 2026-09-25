"""Phase 1 label study: what the label definition does to the training set and the learned weights.

``python -m emva.eval.label_study [--data data/v1] [--n-refits 1000]`` prints markdown with:

1. Company-size weights under legacy and default horizon labels, with coefficient bootstrap
   CIs (``coef_bootstrap``: training rows resampled, formula model refitted).
2. Every label definition and flag combination (legacy, horizon, ``--include-ghosted``,
   ``--stalled-as-lost``, both): training-set size and wins, labelled win rate by company
   size in training, size weights, and AUC of the resulting model on both frozen test sets.
3. Horizon sensitivity (H = 90 to 240 days): training size, labelled win rates and size
   weights, plus how many training wins arrive after H, by company size.

Uses only the pipeline's own inputs; ground-truth reference numbers come from
``python -m emva.eval.ground_truth_reference``. Every model here uses the legacy feature set
(``FEATURES``), so the study isolates the label change and reproduces ``reports/phase1.md``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from emva.constants import TEST_FROM
from emva.eval.bootstrap import N_RESAMPLES, SEED, BootstrapCI, auc_ci, coef_bootstrap
from emva.eval.report import md_table, frozen_test_labels
from emva.features import FeatureSet
from emva.io import load
from emva.labels import HORIZON, LEGACY, LabelConfig
from emva.model import make_lr
from emva.pipeline import run

# Phase 1 study: features held at the baseline's so only the labels change.
FEATURES: FeatureSet = FeatureSet.LEGACY

BANDS: tuple[str, ...] = ("11-50", "51-200", "201-1000", "1000+")
FLAG_CONFIGS: tuple[LabelConfig, ...] = (
    LEGACY, HORIZON, LabelConfig(include_ghosted=True), LabelConfig(stalled_as_lost=True),
    LabelConfig(include_ghosted=True, stalled_as_lost=True),
)
SENSITIVITY_HORIZONS: tuple[int, ...] = (90, 120, 150, 180, 240)


def _lr_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coefficients of the formula model (``emva.model.make_lr``) fitted on ``X``, ``y``."""
    return make_lr().fit(X, y).coef_[0]


def _rate(y: pd.Series) -> str:
    """``"33.5% (n=692)"``: mean label and row count."""
    return f"{y.mean():.1%} (n={len(y)})"


def size_weight_cis(data: str | Path, configs: tuple[LabelConfig, ...], n_refits: int,
                    seed: int) -> dict[LabelConfig, tuple[int, dict[str, BootstrapCI]]]:
    """For each config: training-set size and a bootstrap CI for every ``band=<size>`` weight in ``BANDS``."""
    out = {}
    for cfg in configs:
        r = run(data, labels=cfg, features=FEATURES)
        D, tr = r.design, r.train
        cis = coef_bootstrap(D[tr].values, r.X.y[tr].values, _lr_coef, n_resamples=n_refits, seed=seed)
        by_name = dict(zip(D.columns, cis, strict=True))
        out[cfg] = (int(tr.sum()), {b: by_name[f"band={b}"] for b in BANDS})
    return out


def fmt_ci(ci: BootstrapCI) -> str:
    """``"0.971 [0.680, 1.275]"``."""
    return f"{ci.point:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]"


def band_weight_cis(data: str | Path, n_refits: int, seed: int) -> str:
    """Section 1: size weights with bootstrap CIs under legacy and default horizon labels."""
    rows = []
    for cfg, (n, cis) in size_weight_cis(data, (LEGACY, HORIZON), n_refits, seed).items():
        rows.append({"labels": cfg.describe(), "train n": n, **{f"band={b}": fmt_ci(ci) for b, ci in cis.items()}})
    return "\n".join([
        "## 1. Company-size weights with bootstrap CIs", "",
        f"Log-odds vs band 1-10. 95% percentile CI from {n_refits} refits of the formula model on training rows "
        f"resampled with replacement (seed {seed}).", "",
        md_table(pd.DataFrame(rows)), ""])


def flag_combinations(data: str | Path, n_resamples: int, seed: int) -> str:
    """Section 2: training sets, labelled win rates, weights and frozen-test AUC for each flag combination."""
    L = load(data)
    _, _, y_a, y_b = frozen_test_labels(L)
    rows, rate_rows = [], []
    for cfg in FLAG_CONFIGS:
        r = run(data, labels=cfg, features=FEATURES)
        X, tr = r.X, r.train
        w = pd.Series(r.weights.log_odds)
        auc_a = auc_ci(y_a.values, X.p_formula.loc[y_a.index].values, n_resamples, seed=seed)
        auc_b = auc_ci(y_b.values, X.p_formula.loc[y_b.index].values, n_resamples, seed=seed)
        rows.append({
            "labels": cfg.describe(), "train n": int(tr.sum()), "train wins": int(X.y[tr].sum()),
            "train win rate": f"{X.y[tr].mean():.1%}",
            "band=201-1000": f"{w['band=201-1000']:.3f}", "band=1000+": f"{w['band=1000+']:.3f}",
            "AUC (a) legacy test": f"{auc_a.point:.3f} [{auc_a.lo:.3f}, {auc_a.hi:.3f}]",
            "AUC (b) mature test": f"{auc_b.point:.3f} [{auc_b.lo:.3f}, {auc_b.hi:.3f}]",
        })
        rate_rows.append({"labels": cfg.describe(),
                          **{b: _rate(X.y[tr & (X.band == b)]) for b in ("1-10", *BANDS)}})
    return "\n".join([
        "## 2. Label definitions and flags", "",
        "Training set = labelled rows created before " + TEST_FROM + " (mature rows only for horizon labels). "
        "Weights are the fitted log-odds (weights.csv). AUC is for the model trained with each definition, on "
        f"the two frozen test sets ({len(y_a)} and {len(y_b)} leads), with a {n_resamples}-resample percentile CI.",
        "", md_table(pd.DataFrame(rows)), "",
        "Labelled win rate in training by company size (`band` feature), computed from each definition's labels:",
        "", md_table(pd.DataFrame(rate_rows)), ""])


def horizon_sensitivity(data: str | Path) -> str:
    """Section 3: training size, labelled win rates and size weights for a range of horizons."""
    rows = []
    late_rows = []
    for h in SENSITIVITY_HORIZONS:
        cfg = LabelConfig(horizon_days=h)
        r = run(data, labels=cfg, features=FEATURES)
        X, tr = r.X, r.train
        w = pd.Series(r.weights.log_odds)
        rows.append({
            "H (days)": h, "train n": int(tr.sum()), "train wins": int(X.y[tr].sum()),
            "rate 201-1000": _rate(X.y[tr & (X.band == "201-1000")]),
            "rate 1000+": _rate(X.y[tr & (X.band == "1000+")]),
            "band=201-1000": f"{w['band=201-1000']:.3f}", "band=1000+": f"{w['band=1000+']:.3f}",
            "mature test n": int(r.test.sum()),
        })
        won = tr & (X.label_source == "won")
        days = (X.won_at - X.created_at).dt.total_seconds() / 86400
        late_rows.append({"H (days)": h, **{b: f"{(days[won & (X.band == b)] > h).mean():.1%}"
                                            for b in ("1-10", *BANDS)}})
    return "\n".join([
        "## 3. Horizon sensitivity", "",
        "Default flags (ghosted excluded, stalled censored). Training rows must be mature, so a longer H trains "
        "on fewer, older leads; `mature test n` is the size of the frozen-style test set that H would allow.", "",
        md_table(pd.DataFrame(rows)), "",
        "Share of mature training leads that were eventually Won whose Won date is after H (these count as 0 "
        "under won_within_H), by company size:", "",
        md_table(pd.DataFrame(late_rows)), ""])


def build_study(data: str | Path, n_refits: int = N_RESAMPLES, n_resamples: int = N_RESAMPLES,
                seed: int = SEED) -> str:
    """All three sections as one markdown document."""
    return "\n".join([
        "# Phase 1 label study", "",
        f"Data: `{data}`.", "",
        band_weight_cis(data, n_refits, seed),
        flag_combinations(data, n_resamples, seed),
        horizon_sensitivity(data),
    ])


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: print the study."""
    ap = argparse.ArgumentParser(prog="python -m emva.eval.label_study")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--n-refits", type=int, default=N_RESAMPLES, help="coefficient bootstrap refits")
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES, help="AUC bootstrap resamples")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    print(build_study(a.data, a.n_refits, a.n_resamples, a.seed))


if __name__ == "__main__":
    main()
