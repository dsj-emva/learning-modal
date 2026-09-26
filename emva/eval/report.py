"""Standard report (ground rule 3): baseline vs candidate vs status quo, under both label definitions.

``python -m emva.eval.report [--data data/v1] [label options] [--feature-set v2]`` prints a
markdown report:

- baseline  = the frozen ``baseline/emva_score.py``, run in a temp dir (never edited)
- candidate = the current ``emva`` pipeline (``emva.pipeline.run``) trained with the label
  options and feature set given (default: horizon labels, ``emva.labels.HORIZON``; v2 features)
- status quo = ``status_quo_rules.json`` reconstructed by ``emva.eval.status_quo``

After the standard tables come the value-transform comparison and click-ID coverage (plan 3.2 / 3.5,
``emva.eval.value_report``). It ends with the candidate's design collinearity check (plan 2.7, ``emva.eval.collinearity``) as a
PASS/FAIL section; with ``--strict`` the command exits 1 after printing if that check fails (on
data/v1 the v2 design fails it because of a data property, ADR 0011, so the default does not).

Every model is scored on the same rows under two frozen test definitions (plan 1.6, reported
side by side for one release):

(a) legacy: leads the baseline labels, created on or after ``TEST_FROM``, scored with the
    baseline's labels (the Phase 0 frozen test set, ground rule 4);
(b) horizon (``FROZEN_HORIZON``): mature leads (``created_at + 120 days <= AS_OF``) created on or after
    ``TEST_FROM`` that the default horizon definition labels, scored with ``won_within_h``
    (ghosted leads excluded, stalled censored). This definition is fixed to the defaults
    whatever label options the candidate was trained with.

Dates (ADR 0020): ``AS_OF`` and ``TEST_FROM`` above are the dataset's own when it carries a ``dataset.json``
(``emva.dataset_meta.dataset_dates``). Such a converted dataset (``emva.ingest``) is not in the format or period the
frozen baseline script hard-codes, so the report omits the baseline row, the paired comparison against it and the
"legacy labels equal the baseline's" check, and says so (ADR 0022); it does the same when the baseline script fails.
Without ``status_quo_rules.json`` (converted datasets have none) the status-quo row is omitted with a note.

With ``--feature-set generic`` (Phase 10, ADR 0024; a converted dataset with declared extras) a v2 model is trained
on the same data, labels and split, and a "Generic vs v2" section follows the collinearity check: the paired AUC
difference generic − v2 on both test sets, the Phase 10 criterion's verdict and the extras' leakage screen
(``emva.eval.generic_report``).

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

from emva.cli import add_feature_arguments, add_label_arguments, feature_set, label_config
from emva.constants import AS_OF, LABEL_SOURCES, TEST_FROM, TOP_FRACTION
from emva.dataset_meta import dataset_dates, has_dataset_meta
from emva.eval.bootstrap import N_RESAMPLES, SEED, auc_ci, paired_auc
from emva.eval.collinearity import CollinearityResult, check_collinearity, format_result
from emva.eval.generic_report import LEAKAGE_AUC_FLAG, LEAKAGE_FLAG_TEXT, GenericComparison, compare
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
from emva.eval.value_report import value_report_sections
from emva.features import FeatureSet
from emva.io import clean, load
from emva.labels import HORIZON, LabelConfig, assign_labels, is_mature, label
from emva.pipeline import PipelineResult, run

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = REPO_ROOT / "baseline" / "emva_score.py"
BOTTOM_FRACTION: float = 0.1
# The horizon test definition (b) is frozen at the Phase 1 defaults (H = 120, ghosted excluded, stalled
# censored) whatever options the candidate is trained with: ground rule 4 in REBUILD_PLAN.md.
FROZEN_HORIZON: LabelConfig = HORIZON


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


# The status quo's rules file; a dataset without it (e.g. a converted one) gets no status-quo row.
RULES_FILE = "status_quo_rules.json"


def run_baseline(data: str | Path, out: str | Path) -> tuple[pd.DataFrame, str]:
    """Run the frozen baseline script into ``out``; return its scores.csv and stdout."""
    proc = subprocess.run([sys.executable, str(BASELINE_SCRIPT), "--data", str(Path(data).resolve()), "--out", str(out)],
                          cwd=BASELINE_SCRIPT.parent, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"baseline failed ({proc.returncode}):\n{proc.stderr}")
    return pd.read_csv(Path(out) / "scores.csv", index_col="lead_id"), proc.stdout


def md_table(df: pd.DataFrame) -> str:
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


def headline_frame(ts: TestSet, models: list[ScoredModel], n_resamples: int = N_RESAMPLES,
                   seed: int = SEED) -> pd.DataFrame:
    """The standard table's headline for one test definition, numeric, one row per model in ``models`` order.

    Columns: ``model``, ``auc`` / ``auc_lo`` / ``auc_hi`` (percentile bootstrap), ``brier`` (NaN for a model without
    p), ``top20_wins`` (ranked by p, or value without p), ``top20_revenue_p`` (same ranking) and
    ``top20_revenue_value`` (ranked by p×value); recorded revenue. ``standard_sections`` formats it; the app shows it.
    """
    y, ids, revenue = ts.y, ts.ids, ts.revenue
    rows = []
    for m in models:
        s = m.rank_score(ids)
        ci = auc_ci(y.values, s, n_resamples, seed=seed)
        rows.append({"model": m.name, "auc": ci.point, "auc_lo": ci.lo, "auc_hi": ci.hi,
                     "brier": np.nan if m.p is None else brier_score_loss(y, np.clip(m.p.loc[ids], 0, 1)),
                     "top20_wins": top_share(y, s), "top20_revenue_p": top_share(revenue, s),
                     "top20_revenue_value": top_share(revenue, m.value.loc[ids].values)})
    return pd.DataFrame(rows)


def paired_frame(ts: TestSet, models: list[ScoredModel], n_resamples: int = N_RESAMPLES,
                 seed: int = SEED) -> pd.DataFrame:
    """Paired bootstrap AUC comparison of each of ``models[1:]`` against ``models[0]`` (the baseline) on shared
    resamples: ``model``, ``auc_diff``, ``diff_lo``, ``diff_hi``, ``p_value``."""
    y, ids = ts.y, ts.ids
    rows = []
    for m in models[1:]:
        c = paired_auc(y.values, models[0].rank_score(ids), m.rank_score(ids), n_resamples, seed=seed)
        rows.append({"model": m.name, "auc_diff": c.diff.point, "diff_lo": c.diff.lo, "diff_hi": c.diff.hi,
                     "p_value": c.p_value})
    return pd.DataFrame(rows, columns=["model", "auc_diff", "diff_lo", "diff_hi", "p_value"])


def standard_sections(ts: TestSet, models: list[ScoredModel], n_resamples: int, seed: int,
                      paired_note: str | None = None) -> list[str]:
    """The standard table (ground rule 3) for one test definition, as markdown lines under ``##``/``###`` headings.

    ``models[0]`` is the baseline the paired comparison is made against; with ``paired_note`` (there is no baseline)
    the paired table is replaced by that note.
    """
    y, ids, revenue = ts.y, ts.ids, ts.revenue
    out: list[str] = [f"## {ts.title}", "", ts.description, "", "### Headline", ""]
    h = headline_frame(ts, models, n_resamples, seed)
    head = pd.DataFrame({
        "model": h.model,
        "AUC [95% CI]": [f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in zip(h.auc, h.auc_lo, h.auc_hi)],
        "Brier": ["n/a" if np.isnan(b) else f"{b:.4f}" for b in h.brier],
        "top-20% wins": h.top20_wins.map(_fmt_num),
        "top-20% revenue (by p)": h.top20_revenue_p.map(_fmt_num),
        "top-20% revenue (by p×value)": h.top20_revenue_value.map(_fmt_num),
    })
    out += [md_table(head), "", "### Paired AUC comparison vs baseline", ""]

    if paired_note is None:
        pf = paired_frame(ts, models, n_resamples, seed)
        paired = pd.DataFrame({"model": pf.model, "AUC − baseline": [f"{d:+.3f}" for d in pf.auc_diff],
                               "95% CI": [f"[{lo:+.3f}, {hi:+.3f}]" for lo, hi in zip(pf.diff_lo, pf.diff_hi)],
                               "bootstrap p": [f"{p:.3f}" for p in pf.p_value]})
        out += [md_table(paired), "", "### Calibration by decile of p", ""]
    else:
        out += [paired_note, "", "### Calibration by decile of p", ""]

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
    out += [md_table(cal), ""]
    if any(m.p is None for m in models):
        out += ["Status quo has no probability, so it has no Brier score or calibration.", ""]
    out += ["### AUC by test month", ""]

    month = None
    for m in models:
        d = auc_by_month(y, m.rank_score(ids), ts.leads.created_at).rename(columns={"auc": m.name})
        month = d if month is None else month.merge(d[["month", m.name]], on="month")
    for m in models:
        month[m.name] = month[m.name].map(_fmt_num)
    out += [md_table(month), "", "### Value scale", ""]

    scale = []
    for m in models:
        for scope, v in [("test set", m.value.loc[ids].values), ("all scored leads", m.value.values)]:
            s = value_scale(v)
            scale.append({"model": m.name, "scope": scope, "n": len(v), "median": f"{s['p50']:.0f}",
                          "p99": f"{s['p99']:.0f}", "max": f"{s['max']:.0f}",
                          "max/median": f"{s['max_over_median']:.1f}×", "top-1% share": f"{s['top1pct_share']:.1%}"})
    out += [md_table(pd.DataFrame(scale)), "", "Value = p × expected deal value (models), "
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
                   candidate_labels: LabelConfig, as_of: pd.Timestamp = AS_OF, test_from: str = TEST_FROM) -> list[str]:
    """Label counts by ``label_source`` under both definitions (read at ``as_of``), and train/test sizes (split at
    ``test_from``)."""
    rows = []
    for src in [*LABEL_SOURCES, "all"]:
        m = X.label_source == src if src != "all" else pd.Series(True, index=X.index)
        rows.append({"label_source": src, "leads": int(m.sum()), "legacy y": _counts(legacy_y[m]),
                     "won_within_h": _counts(X.won_within_h[m]), "horizon y": _counts(X.y[m])})
    pre = X.created_at < test_from
    mature = is_mature(X, FROZEN_HORIZON.horizon_days, as_of)
    sizes = pd.DataFrame([
        {"definition": "legacy",
         "train (wins)": f"{int((legacy_y.notna() & pre).sum())} ({int((legacy_y[pre] == 1).sum())})",
         "test (wins)": f"{len(test_a.ids)} ({int(test_a.y.sum())})"},
        {"definition": FROZEN_HORIZON.describe(),
         "train (wins)": f"{int((X.y.notna() & mature & pre).sum())} ({int((X.y[mature & pre] == 1).sum())})",
         "test (wins)": f"{len(test_b.ids)} ({int(test_b.y.sum())})"},
    ])
    last = X.created_at[mature].max()
    return [
        "## Label definitions", "",
        f"Counts over all {len(X)} scored leads, as wins / losses / unlabelled. `label_source` is the CRM state at "
        f"{as_of.date()}. legacy y = baseline rules. won_within_h = Won within {FROZEN_HORIZON.horizon_days} days of "
        "created_at (NaN if younger and not Won). horizon y = won_within_h with ghosted-at-H leads excluded and "
        "stalled leads censored (the default training and test label).", "",
        md_table(pd.DataFrame(rows)), "",
        f"Mature = created_at + {FROZEN_HORIZON.horizon_days} days <= {as_of.isoformat()}: the latest mature lead was "
        f"created {last.isoformat()}. Train = created before {test_from} (all mature); test = created on or after "
        f"{test_from}.",
        "",
        md_table(sizes), "",
        f"Candidate trained with: `{candidate_labels.describe()}`.", "",
    ]


def ghosted_share_section(X: pd.DataFrame, legacy_y: pd.Series, test_a: TestSet, models: list[ScoredModel],
                          as_of: pd.Timestamp = AS_OF, test_from: str = TEST_FROM) -> list[str]:
    """Share of ghosted and of still-New leads in the bottom decile of each model's p, on several populations
    (maturity at ``as_of``, test window from ``test_from``)."""
    mature = is_mature(X, FROZEN_HORIZON.horizon_days, as_of)
    window = X.created_at >= test_from
    populations = [
        ("all leads labelled by legacy rules (train + test)", legacy_y.notna()),
        ("(a) legacy test set", X.index.isin(test_a.ids)),
        ("all mature leads, every label_source", mature),
        ("mature leads created on or after " + test_from + ", every label_source", mature & window),
    ]
    flags = [("ghosted", X.label_source == "ghosted"), ("still New", X.final_stage == "New")]
    rows = []
    for name, mask in populations:
        mask = pd.Series(mask, index=X.index)
        row: dict[str, object] = {"population": name, "n": int(mask.sum())}
        for flag_name, flag in flags:
            row[f"{flag_name}: overall"] = f"{flag[mask].mean():.1%}"
            for m in models:
                if m.p is not None:
                    share = bottom_share(flag[mask], m.p.loc[X.index[mask]].values, BOTTOM_FRACTION)
                    row[f"{flag_name}: {m.name} bottom decile"] = f"{share:.1%}"
        rows.append(row)
    return ["## Bottom-decile ghosted share", "",
            "Share of leads in the 10% of each population with the lowest p that are *ghosted* (`label_source`: "
            f"mature and not contacted within {FROZEN_HORIZON.horizon_days} days) or *still New* at "
            f"{as_of.date()} whatever their age (the definition behind the plan's baseline figure of 27%). The "
            "horizon test set (b) excludes ghosted leads, so the comparison uses populations that keep them.", "",
            md_table(pd.DataFrame(rows)), ""]


def frozen_test_labels(L: pd.DataFrame, as_of: pd.Timestamp = AS_OF,
                       test_from: str = TEST_FROM) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Both frozen test definitions, independent of any candidate's label options.

    Returns ``(X, legacy_y, y_a, y_b)``: the cleaned leads with default horizon labels
    (``emva.labels.assign_labels(..., FROZEN_HORIZON)``), the legacy label for every cleaned lead,
    and the labels of the (a) legacy and (b) horizon test sets, indexed by their ``lead_id``s. Labels and maturity
    are read at ``as_of``; the test sets are leads created on or after ``test_from`` (the dataset's dates, ADR 0020;
    the defaults are the frozen v1 ones).
    """
    X = assign_labels(clean(L), FROZEN_HORIZON, as_of)
    legacy_y = label(X, as_of)
    window = X.created_at >= test_from
    y_a = legacy_y[legacy_y.notna() & window]
    y_b = X.y[X.y.notna() & is_mature(X, FROZEN_HORIZON.horizon_days, as_of) & window]
    return X, legacy_y, y_a, y_b


def collinearity_section(result: PipelineResult) -> tuple[list[str], CollinearityResult]:
    """The candidate's design collinearity check (plan 2.7) on its training rows, as markdown lines."""
    res = check_collinearity(result.design[result.train])
    return ([f"## Design collinearity (plan 2.7): {'PASS' if res.passed else 'FAIL'}", "",
             f"Candidate design (`{result.features.value}` features, {result.design.shape[1]} columns) on its "
             f"{int(result.train.sum())} training rows; fails when two columns have |corr| > {res.threshold}. "
             "The report exits 1 on a failure only with `--strict`; `python -m emva.eval.collinearity` always does.",
             "", *format_result(res), ""], res)


def generic_sections(cmp: GenericComparison, generic: PipelineResult, collinearity: CollinearityResult,
                     n_resamples: int, seed: int) -> list[str]:
    """The "Generic vs v2" section (``emva.eval.generic_report``) as markdown lines: the paired comparison on each test
    set, the generic design's collinearity verdict, the Phase 10 criterion and the leakage screen."""
    n_x = len(generic.extras.columns())
    table = pd.DataFrame([{"test set": title, "n": cmp.sizes[title], "v2 AUC": f"{c.auc_a:.3f}",
                           "generic AUC": f"{c.auc_b:.3f}", "generic − v2": f"{c.diff.point:+.3f}",
                           "95% CI": f"[{c.diff.lo:+.3f}, {c.diff.hi:+.3f}]", "bootstrap p": f"{c.p_value:.3f}"}
                          for title, c in cmp.paired.items()])
    screen = cmp.screen.assign(**{"training AUC": cmp.screen["training AUC"].map(_fmt_num)})
    flagged = [str(x) for x in cmp.screen.extra[cmp.screen.flag != ""]]
    return ["## Generic vs v2 (Phase 10, ADR 0024)", "",
            f"Both models trained on the same rows and split with the same labels: `v2` (the fixed "
            f"{generic.design.shape[1] - n_x}-column design) and `generic` (the same plus {n_x} `x_` columns from "
            f"{len(cmp.screen)} declared extras, encoded on the {int(generic.train.sum())} training leads only). "
            f"Paired bootstrap of AUC(generic) − AUC(v2) on shared resamples ({n_resamples} resamples, seed {seed}).",
            "", md_table(table), "",
            f"Generic design collinearity: {collinearity.describe()} (details in the collinearity section above).", "",
            f"**Phase 10 criterion (pre-registered): {'PASS' if cmp.passed else 'FAIL'}.** PASS iff on the horizon test "
            "set (b) the 95% CI of generic − v2 lies entirely above 0 and the generic design passes the collinearity "
            "check.", "",
            "### Leakage screen", "",
            "Single-feature training AUC of each extra's encoded levels (each level scored by its training win rate), "
            f"folded as |AUC − 0.5| + 0.5. Above {LEAKAGE_AUC_FLAG:.2f}: flagged \"{LEAKAGE_FLAG_TEXT}\" (a flag only; "
            "nothing is dropped).", "", md_table(screen), "",
            f"Flagged: {', '.join(flagged)}." if flagged else "Flagged: none.", ""]


def build_report(data: str | Path, n_resamples: int = N_RESAMPLES, seed: int = SEED,
                 labels: LabelConfig = HORIZON, features: FeatureSet = FeatureSet.V2) -> str:
    """Compute every section of the standard report and return it as markdown."""
    return build_report_with_check(data, n_resamples, seed, labels, features)[0]


def baseline_or_note(data: str | Path) -> tuple[pd.DataFrame | None, str, str | None]:
    """The frozen baseline's ``(scores, stdout, None)`` on ``data``, or ``(None, "", note)`` saying why it is omitted.

    Omitted only when ``data`` carries a ``dataset.json`` (a converted dataset: the baseline script hard-codes the v1
    snapshot date, test boundary and file format, ADR 0022); the script is then not run. On any other dataset
    (data/v1, data/v2, uploads in the v1 format) a failing script raises ``RuntimeError`` (``run_baseline``), as
    before Phase 9: the baseline row is never dropped silently.
    """
    if has_dataset_meta(data):
        return None, "", ("Baseline omitted: this dataset carries its own dates in `dataset.json` (a converted "
                          "dataset, ADR 0020). The frozen `baseline/emva_score.py` hard-codes the v1 snapshot date, "
                          "test boundary and file format, so its scores would not be comparable (ADR 0022). The "
                          "paired comparison against it and the check that the legacy labels equal the baseline's "
                          "are omitted too.")
    with tempfile.TemporaryDirectory() as tmp:
        base, stdout = run_baseline(data, tmp)
    return base, stdout, None


def build_report_with_check(data: str | Path, n_resamples: int = N_RESAMPLES, seed: int = SEED,
                            labels: LabelConfig = HORIZON,
                            features: FeatureSet = FeatureSet.V2) -> tuple[str, CollinearityResult]:
    """``build_report`` plus the candidate's collinearity result (so the CLI can exit 1 on failure).

    The dates are the dataset's (``emva.dataset_meta.dataset_dates``); the baseline row is omitted as
    ``baseline_or_note`` says, and the status-quo row when ``data`` has no ``status_quo_rules.json``, each with a note.
    """
    as_of, test_from = dataset_dates(data)
    L = load(data)
    base, base_stdout, baseline_note = baseline_or_note(data)
    cand_result = run(data, labels=labels, features=features)
    cand = cand_result.X

    X, legacy_y, y_a, y_b = frozen_test_labels(L, as_of, test_from)
    if base is not None and (not X.index.equals(base.index) or not legacy_y.equals(base.y)):
        raise ValueError("legacy labels computed here differ from the baseline script's")
    ids_a, ids_b = y_a.index, y_b.index
    for name, ids in [("legacy", ids_a), ("horizon", ids_b)]:
        missing = ids.difference(cand.index[cand.p_formula.notna()])
        if len(missing):
            raise ValueError(f"candidate has no score for {len(missing)} {name} test leads, e.g. {list(missing[:5])}")
    last_mature = X.created_at[is_mature(X, FROZEN_HORIZON.horizon_days, as_of)].max()
    test_a = TestSet(
        "(a) Legacy labels, legacy test set",
        f"{len(ids_a)} leads labelled by the baseline rules, created on or after {test_from} "
        f"({int(legacy_y[ids_a].sum())} won). Label = legacy y.",
        y_a, L.loc[ids_a])
    test_b = TestSet(
        "(b) Horizon labels, mature test set",
        f"{len(ids_b)} mature leads created on or after {test_from} and up to {last_mature.isoformat()} that the "
        f"default horizon definition labels ({int(X.y[ids_b].sum())} won). Label = won within "
        f"{FROZEN_HORIZON.horizon_days} days; ghosted-at-H leads excluded, stalled leads censored.",
        y_b, L.loc[ids_b])

    rules = Path(data) / RULES_FILE
    models = [] if base is None else [ScoredModel("baseline", base.p_formula, base.value_formula)]
    models.append(ScoredModel("candidate", cand.p_formula.loc[X.index], cand.value_formula.loc[X.index]))
    sq_note = None
    if rules.exists():
        models.append(ScoredModel("status quo", None, status_quo_value(L.loc[X.index], load_rules(rules)).sq_value))
    else:
        sq_note = (f"Status quo omitted: the dataset has no `{RULES_FILE}` (the rules the status quo is "
                   "reconstructed from; converted datasets have none).")

    described = [] if base is None else ["baseline = frozen `baseline/emva_score.py`"]
    described.append(f"candidate = `emva` pipeline trained with `{labels.describe()}` labels and "
                     f"`{cand_result.features.value}` features")
    if sq_note is None:
        described.append("status quo = `status_quo_rules.json` reconstructed")
    by = "p (status quo: by its value)" if sq_note is None else "p"
    rank = f"Top-{int(TOP_FRACTION * 100)}% capture ranks by {by} or by p×value."
    out: list[str] = [
        "# EMVA standard report",
        "",
        f"Data: `{data}`. " + "; ".join(described) + ". Every model is scored "
        f"on the same rows under two test definitions. AUC CI: percentile bootstrap, {n_resamples} resamples, "
        f"seed {seed}. " + rank,
        "",
    ]
    notes = [n for n in (baseline_note, sq_note) if n is not None]
    if notes:
        out += [f"Dates: as_of {as_of.isoformat()}, test_from {test_from}"
                f"{' (from dataset.json)' if has_dataset_meta(data) else ''}.", ""]
        out += [f"- {n}" for n in notes] + [""]
    paired_note = None if base is not None else "Omitted: there is no baseline row (see the note at the top)."
    out += label_sections(X, legacy_y, test_a, test_b, labels, as_of, test_from)
    out += standard_sections(test_a, models, n_resamples, seed, paired_note)
    out += standard_sections(test_b, models, n_resamples, seed, paired_note)
    out += ghosted_share_section(X, legacy_y, test_a, models, as_of, test_from)
    out += value_report_sections(cand_result, [(t.title, t.revenue) for t in (test_a, test_b)],
                                 None if base is None else base.value_formula)
    collinearity_lines, collinearity = collinearity_section(cand_result)
    out += collinearity_lines
    if cand_result.features is FeatureSet.GENERIC:
        v2_result = run(data, labels=labels, features=FeatureSet.V2)
        cmp = compare(cand_result, v2_result, [(t.title, t.y) for t in (test_a, test_b)], test_b.title,
                      collinearity, n_resamples, seed)
        out += generic_sections(cmp, cand_result, collinearity, n_resamples, seed)
    if base is None:
        out += ["## Baseline script output", "", f"- {baseline_note}"]
    else:
        out += ["## Baseline script output", "",
                "- Revenue = recorded deal value of won test leads (blank = 0). The baseline script's own summary, "
                "which fills blank deal values with its predicted value, printed:",
                "", "```", base_stdout.rstrip(), "```"]
    return "\n".join(out), collinearity


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: print the standard report; with ``--strict``, exit 1 if the collinearity check fails."""
    ap = argparse.ArgumentParser(prog="python -m emva.eval.report")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--strict", action="store_true", help="exit 1 if the candidate design fails the collinearity check")
    add_label_arguments(ap)
    add_feature_arguments(ap)
    a = ap.parse_args(argv)
    text, collinearity = build_report_with_check(a.data, a.n_resamples, a.seed, label_config(ap, a), feature_set(a))
    print(text)
    if not collinearity.passed:
        print(f"\n{'FAIL' if a.strict else 'WARNING'}: candidate design collinearity check: {collinearity.describe()}",
              file=sys.stderr)
        if a.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
