"""Standard report (ground rule 3): baseline vs candidate vs status quo on the frozen test set.

``python -m emva.eval.report [--data data/v1]`` prints a markdown report:

- baseline  = the frozen ``baseline/emva_score.py``, run in a temp dir (never edited)
- candidate = the current ``emva`` pipeline (``emva.pipeline.run``)
- status quo = ``status_quo_rules.json`` reconstructed by ``emva.eval.status_quo``

Frozen test set (ground rule 4): leads the baseline labels, created on or after
``TEST_FROM``, scored with the baseline's labels. Every model is evaluated on those rows.

Revenue is recorded deal value for won test leads, 0 when blank or not won. This differs
from the baseline script's own ``top20_revenue``, which fills blank deal values with its own
predicted value; that would move with each candidate's value model, so the report does not
use it (the baseline's own figure is printed in the notes for continuity).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from emva.constants import TEST_FROM, TOP_FRACTION
from emva.eval.bootstrap import N_RESAMPLES, SEED, auc_ci, paired_auc
from emva.eval.metrics import (
    auc_by_month,
    calibration_by_decile,
    tie_averaged_top_share,
    ties_at_cut,
    top_share,
    value_scale,
)
from emva.eval.status_quo import load_rules, status_quo_value
from emva.io import load
from emva.labels import LEGACY
from emva.pipeline import run

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = REPO_ROOT / "baseline" / "emva_score.py"


@dataclass
class ScoredModel:
    """One row of the report: ``p`` (None for rule-based scores) and the value used for ranking."""

    name: str
    p: np.ndarray | None
    value: np.ndarray
    value_all: np.ndarray  # value over every scored lead, for value-scale stats

    @property
    def rank_score(self) -> np.ndarray:
        """Score used where the table ranks "by p" (the value itself when there is no p)."""
        return self.value if self.p is None else self.p


def run_baseline(data: str | Path, out: str | Path) -> tuple[pd.DataFrame, str]:
    """Run the frozen baseline script into ``out``; return its scores.csv and stdout."""
    proc = subprocess.run([sys.executable, str(BASELINE_SCRIPT), "--data", str(Path(data).resolve()), "--out", str(out)],
                          cwd=BASELINE_SCRIPT.parent, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"baseline failed ({proc.returncode}):\n{proc.stderr}")
    return pd.read_csv(Path(out) / "scores.csv", index_col="lead_id"), proc.stdout


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (all cells as given)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _fmt_num(x: float | None, dp: int = 3) -> str:
    """Format ``x`` to ``dp`` decimals; ``None`` or NaN becomes "n/a"."""
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dp}f}"


def build_report(data: str | Path, n_resamples: int = N_RESAMPLES, seed: int = SEED) -> str:
    """Compute every section of the standard report and return it as markdown."""
    L = load(data)
    with tempfile.TemporaryDirectory() as tmp:
        base, base_stdout = run_baseline(data, tmp)
    cand = run(data, labels=LEGACY).X

    test_ids = base.index[base.y.notna() & (L.created_at.reindex(base.index) >= TEST_FROM)]
    missing = test_ids.difference(cand.index[cand.p_formula.notna()])
    if len(missing):
        raise ValueError(f"candidate has no score for {len(missing)} frozen test leads, e.g. {list(missing[:5])}")
    y = base.y.loc[test_ids]
    Lt = L.loc[test_ids]
    revenue = pd.Series(np.where(y == 1, Lt.deal_value.fillna(0), 0), index=test_ids)
    sq_all = status_quo_value(L.loc[base.index], load_rules(Path(data) / "status_quo_rules.json"))

    models = [
        ScoredModel("baseline", base.p_formula.loc[test_ids].values, base.value_formula.loc[test_ids].values,
                    base.value_formula.values),
        ScoredModel("candidate", cand.p_formula.loc[test_ids].values, cand.value_formula.loc[test_ids].values,
                    cand.value_formula.values),
        ScoredModel("status quo", None, sq_all.sq_value.loc[test_ids].values, sq_all.sq_value.values),
    ]

    out: list[str] = [
        "# EMVA standard report",
        "",
        f"Data: `{data}`. Frozen test set: {len(test_ids)} labelled leads created on or after {TEST_FROM} "
        f"({int(y.sum())} won). AUC CI: percentile bootstrap, {n_resamples} resamples, seed {seed}. "
        f"Top-{int(TOP_FRACTION * 100)}% capture ranks by p (status quo: by its value) or by p×value.",
        "",
        "## Headline",
        "",
    ]
    head = []
    for m in models:
        ci = auc_ci(y.values, m.rank_score, n_resamples, seed=seed)
        head.append({
            "model": m.name,
            "AUC [95% CI]": f"{ci.point:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]",
            "Brier": "n/a" if m.p is None else f"{brier_score_loss(y, np.clip(m.p, 0, 1)):.4f}",
            "top-20% wins": _fmt_num(top_share(y, m.rank_score)),
            "top-20% revenue (by p)": _fmt_num(top_share(revenue, m.rank_score)),
            "top-20% revenue (by p×value)": _fmt_num(top_share(revenue, m.value)),
        })
    out += [_md_table(pd.DataFrame(head)), "", "## Paired AUC comparison vs baseline", ""]

    paired = []
    for m in models[1:]:
        c = paired_auc(y.values, models[0].rank_score, m.rank_score, n_resamples, seed=seed)
        paired.append({"model": m.name, "AUC − baseline": f"{c.diff.point:+.3f}",
                       "95% CI": f"[{c.diff.lo:+.3f}, {c.diff.hi:+.3f}]", "bootstrap p": f"{c.p_value:.3f}"})
    out += [_md_table(pd.DataFrame(paired)), "", "## Calibration by decile of p", ""]

    cal = None
    for m in models:
        if m.p is None:
            continue
        d = calibration_by_decile(y, m.p)
        d = d.rename(columns={"mean_p": f"{m.name} mean p", "observed": f"{m.name} observed"})
        cal = d if cal is None else cal.merge(d.drop(columns="n"), on="decile")
    for c in cal.columns:
        if c.endswith(("mean p", "observed")):
            cal[c] = cal[c].map(_fmt_num)
    out += [_md_table(cal), "", "Status quo has no probability, so it has no Brier score or calibration.",
            "", "## AUC by test month", ""]

    month = None
    for m in models:
        d = auc_by_month(y, m.rank_score, Lt.created_at).rename(columns={"auc": m.name})
        month = d if month is None else month.merge(d[["month", m.name]], on="month")
    for m in models:
        month[m.name] = month[m.name].map(_fmt_num)
    out += [_md_table(month), "", "## Value scale", ""]

    scale = []
    for m in models:
        for scope, v in [("test set", m.value), ("all scored leads", m.value_all)]:
            s = value_scale(v)
            scale.append({"model": m.name, "scope": scope, "n": len(v), "median": f"{s['p50']:.0f}",
                          "p99": f"{s['p99']:.0f}", "max": f"{s['max']:.0f}",
                          "max/median": f"{s['max_over_median']:.1f}×", "top-1% share": f"{s['top1pct_share']:.1%}"})
    out += [_md_table(pd.DataFrame(scale)), "", "Value = p × expected deal value (models), "
            "rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.",
            "", "## Notes", ""]

    for m in models:
        for label_, target, score in [("wins", y, m.rank_score), ("revenue by p", revenue, m.rank_score),
                                      ("revenue by p×value", revenue, m.value)]:
            k = ties_at_cut(score)
            if k > 1:
                out.append(f"- {m.name}: {k} rows tie at the top-20% cut when ranking {label_}; the table uses numpy's "
                           f"default argsort (as the baseline does), tie-averaged value is "
                           f"{tie_averaged_top_share(target, score):.3f}.")
    out.append("- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, "
               "which fills blank deal values with its predicted value, printed:")
    out += ["", "```", base_stdout.rstrip(), "```"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: print the standard report."""
    ap = argparse.ArgumentParser(prog="python -m emva.eval.report")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    print(build_report(a.data, a.n_resamples, a.seed))


if __name__ == "__main__":
    main()
