"""Standard report (ground rule 3): baseline vs candidate vs status quo, under both label definitions.

``python -m emva.eval.report [--data data/v1] [label options]`` prints a markdown report:

- baseline  = the frozen ``baseline/emva_score.py``, run in a temp dir (never edited)
- candidate = the current ``emva`` pipeline (``emva.pipeline.run``) trained with the label
  options given (default: horizon labels, ``emva.labels.HORIZON``)
- status quo = ``status_quo_rules.json`` reconstructed by ``emva.eval.status_quo``

Every model is scored on the same rows under two frozen test definitions (plan 1.6, reported
side by side for one release):

(a) legacy: leads the baseline labels, created on or after ``TEST_FROM``, scored with the
    baseline's labels (the Phase 0 frozen test set, ground rule 4);
(b) horizon: mature leads (``created_at + HORIZON_DAYS <= AS_OF``) created on or after
    ``TEST_FROM`` that the default horizon definition labels, scored with ``won_within_h``
    (ghosted leads excluded, stalled censored). This definition is fixed to the defaults
    whatever label options the candidate was trained with.

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

from emva.cli import add_label_arguments, label_config
from emva.constants import AS_OF, LABEL_SOURCES, TEST_FROM, TOP_FRACTION
from emva.eval.bootstrap import N_RESAMPLES, SEED, auc_ci, paired_auc
from emva.eval.metrics import (
    auc_by_month,
    bottom_share,
    calibration_by_decile,
    tie_averaged_top_share,
    ties_at_cut,
    top_share,
    value_scale,
)
from emva.eval.status_quo import load_rules, status_quo_value
from emva.io import clean, load
from emva.labels import HORIZON, LabelConfig, assign_labels, is_mature, label
from emva.pipeline import run

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = REPO_ROOT / "baseline" / "emva_score.py"
BOTTOM_FRACTION: float = 0.1


@dataclass
class ScoredModel:
    """One model's scores over every scored lead (indexed by ``lead_id``).

    ``p`` is None for rule-based scores; ``value`` is the value used for ranking by p×value.
    """

    name: str
    p: pd.Series | None
    value: pd.Series

    def rank_score(self, ids: pd.Index) -> np.ndarray:
        """Score used where the table ranks "by p" (the value itself when there is no p)."""
        return (self.value if self.p is None else self.p).loc[ids].values


@dataclass
class TestSet:
    """A frozen test definition: its rows, labels and the leads' raw columns."""

    title: str
    description: str
    y: pd.Series
    leads: pd.DataFrame

    @property
    def ids(self) -> pd.Index:
        """The test rows' ``lead_id``s."""
        return self.y.index

    @property
    def revenue(self) -> pd.Series:
        """Recorded deal value for won rows (blank = 0), 0 otherwise."""
        return pd.Series(np.where(self.y == 1, self.leads.deal_value.fillna(0), 0), index=self.ids)


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


def _counts(y: pd.Series) -> str:
    """``"wins / losses / unlabelled"`` for a label vector."""
    return f"{int((y == 1).sum())} / {int((y == 0).sum())} / {int(y.isna().sum())}"


def standard_sections(ts: TestSet, models: list[ScoredModel], n_resamples: int, seed: int) -> list[str]:
    """The standard table (ground rule 3) for one test definition, as markdown lines under ``##``/``###`` headings."""
    y, ids, revenue = ts.y, ts.ids, ts.revenue
    out: list[str] = [f"## {ts.title}", "", ts.description, "", "### Headline", ""]
    head = []
    for m in models:
        s = m.rank_score(ids)
        ci = auc_ci(y.values, s, n_resamples, seed=seed)
        head.append({
            "model": m.name,
            "AUC [95% CI]": f"{ci.point:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]",
            "Brier": "n/a" if m.p is None else f"{brier_score_loss(y, np.clip(m.p.loc[ids], 0, 1)):.4f}",
            "top-20% wins": _fmt_num(top_share(y, s)),
            "top-20% revenue (by p)": _fmt_num(top_share(revenue, s)),
            "top-20% revenue (by p×value)": _fmt_num(top_share(revenue, m.value.loc[ids].values)),
        })
    out += [_md_table(pd.DataFrame(head)), "", "### Paired AUC comparison vs baseline", ""]

    paired = []
    for m in models[1:]:
        c = paired_auc(y.values, models[0].rank_score(ids), m.rank_score(ids), n_resamples, seed=seed)
        paired.append({"model": m.name, "AUC − baseline": f"{c.diff.point:+.3f}",
                       "95% CI": f"[{c.diff.lo:+.3f}, {c.diff.hi:+.3f}]", "bootstrap p": f"{c.p_value:.3f}"})
    out += [_md_table(pd.DataFrame(paired)), "", "### Calibration by decile of p", ""]

    cal = None
    for m in models:
        if m.p is None:
            continue
        d = calibration_by_decile(y, m.p.loc[ids].values)
        d = d.rename(columns={"mean_p": f"{m.name} mean p", "observed": f"{m.name} observed"})
        cal = d if cal is None else cal.merge(d.drop(columns="n"), on="decile")
    for c in cal.columns:
        if c.endswith(("mean p", "observed")):
            cal[c] = cal[c].map(_fmt_num)
    out += [_md_table(cal), "", "Status quo has no probability, so it has no Brier score or calibration.",
            "", "### AUC by test month", ""]

    month = None
    for m in models:
        d = auc_by_month(y, m.rank_score(ids), ts.leads.created_at).rename(columns={"auc": m.name})
        month = d if month is None else month.merge(d[["month", m.name]], on="month")
    for m in models:
        month[m.name] = month[m.name].map(_fmt_num)
    out += [_md_table(month), "", "### Value scale", ""]

    scale = []
    for m in models:
        for scope, v in [("test set", m.value.loc[ids].values), ("all scored leads", m.value.values)]:
            s = value_scale(v)
            scale.append({"model": m.name, "scope": scope, "n": len(v), "median": f"{s['p50']:.0f}",
                          "p99": f"{s['p99']:.0f}", "max": f"{s['max']:.0f}",
                          "max/median": f"{s['max_over_median']:.1f}×", "top-1% share": f"{s['top1pct_share']:.1%}"})
    out += [_md_table(pd.DataFrame(scale)), "", "Value = p × expected deal value (models), "
            "rule-based bucket value in GBP (status quo). All scored leads = leads left after bot/duplicate removal.",
            "", "### Notes", ""]

    notes = []
    for m in models:
        for label_, target, score in [("wins", y, m.rank_score(ids)), ("revenue by p", revenue, m.rank_score(ids)),
                                      ("revenue by p×value", revenue, m.value.loc[ids].values)]:
            k = ties_at_cut(score)
            if k > 1:
                notes.append(f"- {m.name}: {k} rows tie at the top-20% cut when ranking {label_}; the table uses "
                             f"numpy's default argsort (as the baseline does), tie-averaged value is "
                             f"{tie_averaged_top_share(target, score):.3f}.")
    out += notes or ["- No ties at the top-20% cut."]
    return out + [""]


def label_sections(X: pd.DataFrame, legacy_y: pd.Series, test_a: TestSet, test_b: TestSet,
                   candidate_labels: LabelConfig) -> list[str]:
    """Label counts by ``label_source`` under both definitions, and train/test sizes."""
    rows = []
    for src in [*LABEL_SOURCES, "all"]:
        m = X.label_source == src if src != "all" else pd.Series(True, index=X.index)
        rows.append({"label_source": src, "leads": int(m.sum()), "legacy y": _counts(legacy_y[m]),
                     "won_within_h": _counts(X.won_within_h[m]), "horizon y": _counts(X.y[m])})
    pre = X.created_at < TEST_FROM
    mature = is_mature(X, HORIZON.horizon_days)
    sizes = pd.DataFrame([
        {"definition": "legacy", "train (wins)": f"{int((legacy_y.notna() & pre).sum())} ({int((legacy_y[pre] == 1).sum())})",
         "test (wins)": f"{len(test_a.ids)} ({int(test_a.y.sum())})"},
        {"definition": HORIZON.describe(),
         "train (wins)": f"{int((X.y.notna() & mature & pre).sum())} ({int((X.y[mature & pre] == 1).sum())})",
         "test (wins)": f"{len(test_b.ids)} ({int(test_b.y.sum())})"},
    ])
    last = X.created_at[mature].max()
    return [
        "## Label definitions", "",
        f"Counts over all {len(X)} scored leads, as wins / losses / unlabelled. `label_source` is the CRM state at "
        f"{AS_OF.date()}. legacy y = baseline rules. won_within_h = Won within {HORIZON.horizon_days} days of "
        "created_at (NaN if younger and not Won). horizon y = won_within_h with ghosted-at-H leads excluded and "
        "stalled leads censored (the default training and test label).", "",
        _md_table(pd.DataFrame(rows)), "",
        f"Mature = created_at + {HORIZON.horizon_days} days <= {AS_OF.isoformat()}: the latest mature lead was created "
        f"{last.isoformat()}. Train = created before {TEST_FROM} (all mature); test = created on or after {TEST_FROM}.",
        "",
        _md_table(sizes), "",
        f"Candidate trained with: `{candidate_labels.describe()}`.", "",
    ]


def ghosted_share_section(X: pd.DataFrame, legacy_y: pd.Series, test_a: TestSet,
                          models: list[ScoredModel]) -> list[str]:
    """Share of ghosted leads (``label_source == "ghosted"``) in the bottom decile of each model's p."""
    mature = is_mature(X, HORIZON.horizon_days)
    window = X.created_at >= TEST_FROM
    populations = [
        ("all leads labelled by legacy rules (train + test)", legacy_y.notna()),
        ("(a) legacy test set", X.index.isin(test_a.ids)),
        ("all mature leads, every label_source", mature),
        ("mature leads created on or after " + TEST_FROM + ", every label_source", mature & window),
    ]
    gh = X.label_source == "ghosted"
    rows = []
    for name, mask in populations:
        mask = pd.Series(mask, index=X.index)
        row = {"population": name, "n": int(mask.sum()), "ghosted share overall": f"{gh[mask].mean():.1%}"}
        for m in models:
            if m.p is not None:
                row[f"{m.name} bottom decile"] = f"{bottom_share(gh[mask], m.p.loc[X.index[mask]].values, BOTTOM_FRACTION):.1%}"
        rows.append(row)
    return ["## Bottom-decile ghosted share", "",
            "Share of leads still at stage New (`label_source == ghosted`) among the 10% of each population with the "
            "lowest p. The horizon test set (b) excludes ghosted leads, so the comparison uses populations that "
            "keep them.", "", _md_table(pd.DataFrame(rows)), ""]


def build_report(data: str | Path, n_resamples: int = N_RESAMPLES, seed: int = SEED,
                 labels: LabelConfig = HORIZON) -> str:
    """Compute every section of the standard report and return it as markdown."""
    L = load(data)
    with tempfile.TemporaryDirectory() as tmp:
        base, base_stdout = run_baseline(data, tmp)
    cand = run(data, labels=labels).X

    # Both test definitions are computed here from the cleaned leads, independent of the candidate's options.
    X = assign_labels(clean(L), HORIZON)
    legacy_y = label(X)
    if not X.index.equals(base.index) or not legacy_y.equals(base.y):
        raise ValueError("legacy labels computed here differ from the baseline script's")
    ids_a = X.index[legacy_y.notna() & (X.created_at >= TEST_FROM)]
    ids_b = X.index[X.y.notna() & is_mature(X, HORIZON.horizon_days) & (X.created_at >= TEST_FROM)]
    for name, ids in [("legacy", ids_a), ("horizon", ids_b)]:
        missing = ids.difference(cand.index[cand.p_formula.notna()])
        if len(missing):
            raise ValueError(f"candidate has no score for {len(missing)} {name} test leads, e.g. {list(missing[:5])}")
    last_mature = X.created_at[is_mature(X, HORIZON.horizon_days)].max()
    test_a = TestSet(
        "(a) Legacy labels, legacy test set",
        f"{len(ids_a)} leads labelled by the baseline rules, created on or after {TEST_FROM} "
        f"({int(legacy_y[ids_a].sum())} won). Label = legacy y.",
        legacy_y.loc[ids_a], L.loc[ids_a])
    test_b = TestSet(
        "(b) Horizon labels, mature test set",
        f"{len(ids_b)} mature leads created on or after {TEST_FROM} and up to {last_mature.isoformat()} that the "
        f"default horizon definition labels ({int(X.y[ids_b].sum())} won). Label = won within "
        f"{HORIZON.horizon_days} days; ghosted-at-H leads excluded, stalled leads censored.",
        X.y.loc[ids_b], L.loc[ids_b])

    sq = status_quo_value(L.loc[X.index], load_rules(Path(data) / "status_quo_rules.json")).sq_value
    models = [
        ScoredModel("baseline", base.p_formula, base.value_formula),
        ScoredModel("candidate", cand.p_formula.loc[X.index], cand.value_formula.loc[X.index]),
        ScoredModel("status quo", None, sq),
    ]

    out: list[str] = [
        "# EMVA standard report",
        "",
        f"Data: `{data}`. baseline = frozen `baseline/emva_score.py`; candidate = `emva` pipeline trained with "
        f"`{labels.describe()}` labels; status quo = `status_quo_rules.json` reconstructed. Every model is scored "
        f"on the same rows under two test definitions. AUC CI: percentile bootstrap, {n_resamples} resamples, "
        f"seed {seed}. Top-{int(TOP_FRACTION * 100)}% capture ranks by p (status quo: by its value) or by p×value.",
        "",
    ]
    out += label_sections(X, legacy_y, test_a, test_b, labels)
    out += standard_sections(test_a, models, n_resamples, seed)
    out += standard_sections(test_b, models, n_resamples, seed)
    out += ghosted_share_section(X, legacy_y, test_a, models)
    out += ["## Baseline script output", "",
            "- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, "
            "which fills blank deal values with its predicted value, printed:",
            "", "```", base_stdout.rstrip(), "```"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: print the standard report."""
    ap = argparse.ArgumentParser(prog="python -m emva.eval.report")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    add_label_arguments(ap)
    a = ap.parse_args(argv)
    print(build_report(a.data, a.n_resamples, a.seed, label_config(ap, a)))


if __name__ == "__main__":
    main()
