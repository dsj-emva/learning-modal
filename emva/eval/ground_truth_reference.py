"""Ground-truth reference numbers for Phase 1 (evaluation only, ground rule 2).

``python -m emva.eval.ground_truth_reference [--data data/v1] [--n-refits 1000]`` prints:

1. The mean planted close probability (``p_close_true``) of training-period leads by company
   size (the pipeline's ``band`` feature): the plan's "39.4% true rate" for band 201-1000.
2. The orchestrator's restated size-weight criterion (ruling R1): the true effect of company
   size on closing *within H = 120 days*, against the learned legacy and horizon weights with
   their coefficient bootstrap CIs, and a per-band pass/fail.

The true 120-day close probability of a lead is derived from the planted truth, not from the
observed labels: ``p120 = p_close_true × P(time to close <= 120 days)``, where the planted
time to close is lognormal (median ``m``, log-sd ``s``, ×``k`` for true band 1000+) as
``ground_truth.md`` ("Time to close") states; those three numbers are parsed from that file.
This ignores the reply delay before the close clock can end (at most 10 days after the lead,
and the close cannot come earlier than one hour after contact), which only matters for deals
closing within hours. The observed won-within-120 rates by band are printed next to the
derived ones as a check.

Models use the legacy feature set (``label_study.FEATURES``) so the numbers match
``reports/phase1.md``.

The ground-truth readers are this module, ``emva/eval/phase2_study.py`` (Phase 2 validation) and
``emva/eval/ceiling.py`` (Phase 4 oracle ceiling, used only by ``emva.eval.hardening``). The pipeline
never imports any of them.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from emva.constants import TEST_FROM
from emva.eval.bootstrap import N_RESAMPLES, SEED, BootstrapCI
from emva.eval.label_study import BANDS as SIZE_BANDS
from emva.eval.label_study import FEATURES, fmt_ci, size_weight_cis
from emva.eval.report import md_table
from emva.io import load
from emva.labels import HORIZON, LEGACY, label
from emva.model import make_lr
from emva.pipeline import build, run

REFERENCE_BAND: str = "1-10"
BANDS: tuple[str, ...] = (REFERENCE_BAND, *SIZE_BANDS)
TIME_TO_CLOSE_PATTERN: str = (r"Won deals take a lognormal number of days \(median ([\d.]+), sd ([\d.]+)\); 1000\+\s+"
                              r"companies take ([\d.]+)x longer")


@dataclass(frozen=True)
class TimeToClose:
    """Planted time to close for Won deals: lognormal with this median and log-sd, ×``slow_mult`` for 1000+."""

    median_days: float
    log_sd: float
    slow_mult: float

    def p_within(self, days: float, slow: bool) -> float:
        """P(a Won deal closes within ``days`` of creation); ``slow`` = true band 1000+."""
        median = self.median_days * (self.slow_mult if slow else 1.0)
        return 0.5 * (1 + math.erf((math.log(days) - math.log(median)) / (self.log_sd * math.sqrt(2))))


def read_time_to_close(data: str | Path) -> TimeToClose:
    """Parse the time-to-close parameters from ``ground_truth.md``; raises if the sentence is not found."""
    text = (Path(data) / "ground_truth.md").read_text()
    m = re.search(TIME_TO_CLOSE_PATTERN, text)
    if m is None:
        raise ValueError(f"time-to-close sentence not found in {data}/ground_truth.md")
    return TimeToClose(float(m.group(1)), float(m.group(2)), float(m.group(3)))


def read_truth(data: str | Path) -> pd.DataFrame:
    """``ground_truth_labels.csv`` indexed by ``lead_id``."""
    return pd.read_csv(Path(data) / "ground_truth_labels.csv", index_col="lead_id")


def true_rates(data: str | Path) -> pd.DataFrame:
    """Mean ``p_close_true`` by ``band`` over three training-period populations, as a table."""
    truth = read_truth(data).p_close_true
    X = build(load(data), HORIZON, FEATURES)  # features do not depend on the labels; the legacy label is computed alongside
    legacy_y = label(X)
    pre = X.created_at < TEST_FROM
    pops = {
        f"training rows labelled by {LEGACY.describe()}": pre & legacy_y.notna(),
        f"training rows labelled by {HORIZON.describe()}": pre & X.y.notna() & HORIZON.eligible(X),
        f"all cleaned leads created before {TEST_FROM} (every label_source)": pre,
    }
    p = truth.reindex(X.index)
    rows = [{"population": name, **{b: f"{p[mask & (X.band == b)].mean():.1%} (n={int((mask & (X.band == b)).sum())})"
                                    for b in BANDS}} for name, mask in pops.items()]
    return pd.DataFrame(rows)


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _soft_label_fit(D: pd.DataFrame, p: pd.Series) -> pd.Series:
    """Fit the formula model to probabilities ``p`` (each row once as 1 with weight p, once as 0 with weight 1-p)."""
    X2 = pd.concat([D, D])
    y2 = np.r_[np.ones(len(D)), np.zeros(len(D))]
    w2 = np.r_[p.values, 1 - p.values]
    return pd.Series(make_lr().fit(X2, y2, sample_weight=w2).coef_[0], index=D.columns)


def _verdict(true: float, legacy: BootstrapCI, horizon: BootstrapCI) -> str:
    """R1 rule: pass if the horizon CI contains ``true`` or the horizon point is closer to it than legacy's."""
    if horizon.lo <= true <= horizon.hi:
        return "pass (in CI)"
    if abs(horizon.point - true) < abs(legacy.point - true):
        return "pass (closer)"
    return "fail"


def r1_tables(data: str | Path, n_refits: int, seed: int) -> str:
    """The restated size-weight criterion: true 120-day effects vs learned weights, per band."""
    ttc = read_time_to_close(data)
    truth = read_truth(data)
    r = run(data, labels=HORIZON, features=FEATURES)
    X, D, tr = r.X, r.design, r.train
    t = truth.reindex(X.index)
    f120 = pd.Series([ttc.p_within(HORIZON.horizon_days, band == "1000+") for band in t.employee_band], index=X.index)
    p120 = t.p_close_true * f120
    learned = size_weight_cis(data, (LEGACY, HORIZON), n_refits, seed)
    (n_leg, leg), (n_hor, hor) = learned[LEGACY], learned[HORIZON]

    Xt, pt, pe = X[tr], p120[tr], t.p_close_true[tr]
    marg = {b: _logit(pt[Xt.band == b].mean()) - _logit(pt[Xt.band == REFERENCE_BAND].mean()) for b in SIZE_BANDS}
    cond120 = _soft_label_fit(D[tr], pt)
    cond_ev = _soft_label_fit(D[tr], pe)

    check = pd.DataFrame([{
        "band": b, "rows": int((Xt.band == b).sum()),
        "mean p_close_true (eventual)": f"{pe[Xt.band == b].mean():.1%}",
        "derived p120": f"{pt[Xt.band == b].mean():.1%}",
        "observed won within 120 d (horizon label)": f"{Xt.y[Xt.band == b].mean():.1%}",
    } for b in BANDS])
    rows = []
    for b in SIZE_BANDS:
        c = f"band={b}"
        rows.append({
            "band": b,
            "true 120-day effect, conditional": f"{cond120[c]:.3f}",
            "true 120-day effect, marginal": f"{marg[b]:.3f}",
            "legacy weight [95% CI]": fmt_ci(leg[b]),
            "horizon weight [95% CI]": fmt_ci(hor[b]),
            "R1 (conditional)": _verdict(cond120[c], leg[b], hor[b]),
            "R1 (marginal)": _verdict(marg[b], leg[b], hor[b]),
            "eventual-close effect, conditional (check)": f"{cond_ev[c]:.3f}",
        })
    return "\n".join([
        "## R1: true 120-day size effects vs learned weights", "",
        f"Time to close parsed from ground_truth.md: median {ttc.median_days:g} days, log-sd {ttc.log_sd:g}, "
        f"×{ttc.slow_mult:g} for 1000+. P(close within {HORIZON.horizon_days} d | Won) = "
        f"{ttc.p_within(HORIZON.horizon_days, False):.3f} (other bands), "
        f"{ttc.p_within(HORIZON.horizon_days, True):.3f} (1000+). p120 = p_close_true × that, per lead, using the "
        "true band. Rows: the horizon training set "
        f"({n_hor} leads; the legacy weights come from its own {n_leg} training rows).", "",
        "Derivation check (horizon training rows, `band` feature):", "",
        md_table(check), "",
        "True effects are log-odds relative to band 1-10. *Conditional* = the formula model (same design, same "
        "rows, same regularisation) fitted to p120 as soft labels: the weight a noise-free 120-day label would "
        "give, holding the other features fixed, which is what the learned weights estimate. *Marginal* = "
        "logit(mean p120 in band) − logit(mean p120 in 1-10) on the same rows, as the ruling literally states; "
        "it includes every feature correlated with size (free email, missing company, spend), so it is not "
        "comparable with a multivariable weight. The last column fits p_close_true the same way, as a check "
        "that the conditional method recovers the planted eventual effects (11-50 +0.5, 51-200 and "
        "201-1000 +1.3, 1000+ +1.0) up to feature mismatch and regularisation.", "",
        f"Learned CIs: {n_refits} bootstrap refits, seed {seed}. R1 pass = the horizon CI contains the true "
        "effect, or the horizon point is closer to it than the legacy point.", "",
        md_table(pd.DataFrame(rows)), ""])


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: print both reference sections."""
    ap = argparse.ArgumentParser(prog="python -m emva.eval.ground_truth_reference")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--n-refits", type=int, default=N_RESAMPLES, help="coefficient bootstrap refits")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    print("## Mean planted close probability (p_close_true) by company size (band feature)\n")
    print(md_table(true_rates(a.data)))
    print()
    print(r1_tables(a.data, a.n_refits, a.seed))


if __name__ == "__main__":
    main()
