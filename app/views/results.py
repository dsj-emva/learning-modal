"""Page: Model results. How well a trained run ranks leads on the frozen test set, and why it scores what it does."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app import charts, results, storage, ui
from app import components as C
from emva.dataset_meta import dataset_dates
from emva.features import FeatureSet
from emva.generic import GenericEncoder
from emva.labels import LabelMode
from emva.persist import BUNDLE_FILE

TEST_SET_LABELS = {"mature": "Mature leads (horizon labels)", "legacy": "Legacy labels (frozen POC)"}
# A generic run's scorecard can hold hundreds of x_ columns: the chart and table show the strongest this many.
SCORECARD_TOP = 25


def _kpis(head: pd.DataFrame) -> None:
    """The KPI strip: this model against the status quo, else the frozen baseline, else nothing (ADR 0022)."""
    m = head.set_index("model")
    model = m.loc[results.CANDIDATE]
    ref_name = results.kpi_reference(m.index)
    ref = m.loc[ref_name] if ref_name else None
    vs = f"vs {ref_name.split(' (')[0].lower()}" if ref_name else ""

    def d(col: str, fmt: str) -> str:
        return C.delta(model[col] - ref[col], fmt) if ref is not None else ""

    ui.html(C.kpi_row([
        C.kpi("AUC · ranking quality", C.num(model.auc), vs, d("auc", "auc"),
              note=f"95% CI {model.auc_lo:.3f} – {model.auc_hi:.3f}"),
        C.kpi("Brier score", C.num(model.brier, 4), "lower is better", note="0 = perfect probabilities"),
        C.kpi("Wins in the top 20%", C.pct(model.top20_wins), vs, d("top20_wins", "pts"),
              note="ranked by chance of closing"),
        C.kpi("Revenue in the top 20%", C.pct(model.top20_revenue_value), vs,
              d("top20_revenue_value", "pts"), note="ranked by value per lead"),
    ]))


def _standard_table(head: pd.DataFrame, paired: pd.DataFrame | None, notes: list[str],
                    baseline_missing: str | None) -> None:
    """The standard report's headline and paired comparison (``emva.eval.report`` frames), formatted; without a
    baseline row the paired comparison is replaced by ``baseline_missing`` (ADR 0022)."""
    rows = [[r.model, C.num(r.auc), f"{r.auc_lo:.3f} – {r.auc_hi:.3f}", C.num(r.brier, 4), C.pct(r.top20_wins),
             C.pct(r.top20_revenue_p), C.pct(r.top20_revenue_value)] for r in head.itertuples()]
    ui.html(C.table(["Scored by", "AUC", "95% interval", "Brier", "Top-20% wins", "Top-20% revenue (by p)",
                     "Top-20% revenue (by value)"], rows))
    caption = ["Revenue is the recorded deal value of won test leads."]
    if results.STATUS_QUO in set(head.model):
        caption.insert(0, "Brier needs a probability, so the status quo (a bucket value) has none.")
    st.caption(" ".join(caption + notes))
    if baseline_missing:
        ui.html(C.callout(baseline_missing, "info"))
    elif paired is not None and not paired.empty:
        ui.html(C.section("Compared with the frozen baseline", "AUC difference on the same bootstrap resamples; "
                          "an interval that excludes 0 is a real difference, not noise."))
        ui.html(C.table(["Scored by", "AUC − baseline", "95% interval", "Bootstrap p"],
                        [[r.model, f"{r.auc_diff:+.3f}", f"{r.diff_lo:+.3f} – {r.diff_hi:+.3f}", f"{r.p_value:.3f}"]
                         for r in paired.itertuples()]))


def _scorecard_table(card: pd.DataFrame, height: int, key: str) -> None:
    """A sortable scorecard table with a strength bar per row."""
    t = card.assign(strength=card.points.abs() / max(card.points.abs().max(), 1),
                    effect=np.where(card.points >= 0, "raises", "lowers"))
    pos, neg = "color: #0B4644; font-weight: 600", "color: #B5543C; font-weight: 600"
    styled = t[["feature", "level", "reference", "points", "strength", "odds_multiplier", "effect"]].style.map(
        lambda v: pos if v >= 0 else neg, subset=["points"])
    st.dataframe(styled, hide_index=True, width="stretch", height=height, key=key, column_config={
        "feature": st.column_config.TextColumn("Signal"),
        "level": st.column_config.TextColumn("When the lead is"),
        "reference": st.column_config.TextColumn("Compared with"),
        "points": st.column_config.NumberColumn("Points", format="%+d"),
        "strength": st.column_config.ProgressColumn("Strength", min_value=0.0, max_value=1.0, format=" "),
        "odds_multiplier": st.column_config.NumberColumn("Odds ×", format="%.2f×"),
        "effect": st.column_config.TextColumn("Effect"),
    })


def _scorecard(run: storage.Run, extras: GenericEncoder | None) -> None:
    """Scorecard: signed bar chart and a sortable table. A generic run's long list (extras' ``x_`` columns) shows the
    ``SCORECARD_TOP`` rows with the largest |points|, with every row in an expander."""
    card = results.scorecard(run.out_dir, extras)
    long = extras is not None and len(card) > SCORECARD_TOP
    shown = card.loc[card.points.abs().sort_values(ascending=False, kind="stable").index[:SCORECARD_TOP]] \
        .sort_values("points", ascending=False, kind="stable") if long else card
    if long:
        st.caption(f"The {SCORECARD_TOP} strongest of {len(card)} signals ({sum(card.column.str.startswith('x_'))} "
                   "from extra features); every signal is in the list below.")
    labels = [f"{f} · {lvl}" for f, lvl in zip(shown.feature, shown.level)]
    chart_tab, table_tab = st.tabs(["Chart", "Table"])
    with chart_tab:
        st.plotly_chart(charts.points_chart(labels, shown.points.tolist()), config=charts.CONFIG, theme=None,
                        width="stretch", key="scorecard_chart")
    with table_tab:
        _scorecard_table(shown, 460, "scorecard_table")
    if long:
        with st.expander(f"All {len(card)} signals"):
            _scorecard_table(card, 520, "scorecard_all")


def _generic(run: storage.Run, test_set: str) -> None:
    """Generic vs v2 (a generic-feature-set run): the paired AUC difference on this test set, the generic design's
    collinearity check and the extras' leakage screen (``results.generic_comparison``)."""
    ui.html(C.section("Generic vs v2", "The same leads, labels and split trained twice: v2 (the fixed design) and "
                      "generic (v2 plus the extra features). AUC difference on shared bootstrap resamples; an "
                      "interval that excludes 0 is a real difference, not noise."))
    try:
        g = ui.generic_section(run.run_id, run.out_dir, run.dataset_path,
                               run.args.get("label_mode", LabelMode.HORIZON.value), test_set)
    except ValueError as e:
        ui.html(C.callout(str(e), "bad", lead="The comparison cannot be computed:"))
        return
    if g is None:
        ui.html(C.callout("This test set is empty or has a single class.", "warn", lead="No comparison."))
        return
    p = g.paired
    ui.html(C.kpi_row([
        C.kpi("v2 AUC", C.num(p.auc_a), f"{g.v2_columns} design columns"),
        C.kpi("Generic AUC", C.num(p.auc_b), f"v2 + {g.x_columns} extra columns"),
        C.kpi("Generic − v2", f"{p.diff.point:+.3f}", "excludes 0" if p.diff.lo > 0 or p.diff.hi < 0 else
              "includes 0: no clear difference", note=f"95% CI {p.diff.lo:+.3f} – {p.diff.hi:+.3f}"),
        C.kpi("Test leads", f"{g.n:,}", TEST_SET_LABELS[test_set], note=f"bootstrap p {p.p_value:.3f}")]))
    col = g.collinearity
    ui.html(C.callout(col.describe(), "info" if col.passed else "warn",
                      lead="Collinearity of the generic design (training leads, |corr| > 0.95 fails):"))
    ui.html(C.section("Leakage screen", "Each extra feature alone, on the training leads: the AUC of its levels "
                      f"scored by their win rate, folded so that 0.5 is no signal. Above {results.LEAKAGE_AUC_FLAG:.2f} "
                      "a single column separates wins from losses suspiciously well: check that it is known when the "
                      "lead is submitted. The missingness columns compare the share of leads with the extra blank "
                      "among closed leads (won or lost in the CRM) and among open ones (open, stalled or ghosted); a "
                      f"difference of {results.GENERIC_MISSINGNESS_FLAG:.2f} or more means blank stands for \"not "
                      "closed yet\" rather than for the lead. A flag only; nothing is dropped."))
    ui.html(C.table(["Extra feature", "Kind", "Levels", "Design columns", "Training AUC", "Flag", "Missing: closed",
                     "Missing: open", "Missingness flag"],
                    [[str(r["extra"]), str(r["kind"]), str(r["levels"]), str(r["design columns"]),
                      C.num(r["training AUC"]), str(r["flag"]), C.num(r["missing: closed"]),
                      C.num(r["missing: open"]), str(r["missingness flag"])] for _, r in g.screen.iterrows()],
                    numeric_from=2))
    if g.flagged:
        ui.html(C.callout(", ".join(g.flagged), "bad", lead="Check for leakage:"))


def _values(run: storage.Run) -> None:
    """Value distribution: stats per value column and a log-scale histogram."""
    vd = results.value_distribution(run.out_dir)
    if vd.empty:
        st.caption("No value columns in this run's scores.")
        return
    skipped = vd[vd.skipped]
    for _, r in skipped.iterrows():
        st.caption(f"{r.label}: no scale statistics (its median or total is not positive).")
    vd = vd[~vd.skipped]
    if vd.empty:
        return
    for _, r in vd.iterrows():
        st.markdown(f"**{r.label}** · {int(r.n):,} scored leads")
        ui.html(C.stats([("p1", C.gbp(r.p1)), ("median", C.gbp(r.p50)), ("p90", C.gbp(r.p90)), ("p99", C.gbp(r.p99)),
                         ("max", C.gbp(r["max"])), ("max / median", f"{r.max_over_median:.1f}×"),
                         ("top 1% share", C.pct(r.top1pct_share))]))
    scores = results.load_scores(run.out_dir)
    series = {r.label: scores[r.column].dropna().to_numpy() for _, r in vd.iterrows()}
    st.plotly_chart(charts.value_histogram(series), config=charts.CONFIG, theme=None, width="stretch", key="value_hist")


def render() -> None:
    """The page."""
    ui.html(C.page_header("Model results", "How well the model ranks leads",
                          "Each trained run scored on held-out leads it never saw: ranking quality against the "
                          "status-quo rules, calibration, stability by month, and the scorecard behind every score."))
    root = ui.data_root()
    runs = storage.list_runs(root)
    if not runs:
        ui.no_runs_state("Train a model on the Upload & train page (the bundled sample works) and its results "
                         "will appear here.")
        return
    top = st.columns([3, 2], vertical_alignment="bottom")
    with top[0]:
        run = ui.pick_run(runs, key="results_run")
    default_test = "legacy" if run.args.get("label_mode") == "legacy" else "mature"
    with top[1]:
        test_set = st.segmented_control("Test set", list(TEST_SET_LABELS), default=default_test, required=True,
                                        format_func=TEST_SET_LABELS.get, key=f"test_set_{run.run_id}")
    ui.html(C.run_header(run.status, run.run_id, f"dataset {run.dataset} · {run.args.get('label_mode')} labels · "
                                                 f"{run.args.get('feature_set')} features"))
    if run.status != "succeeded":
        msg = run.error or "This run has not finished yet; its progress is on the Upload & train page."
        ui.html(C.callout(msg, "bad" if run.status == "failed" else "info", lead="No results for this run."))
        return

    generic = run.args.get("feature_set") == FeatureSet.GENERIC.value
    ev = ui.evaluation(run.run_id, run.out_dir, run.dataset_path, test_set)
    if ev is None:
        test_from = dataset_dates(run.dataset_path)[1]
        ui.html(C.callout(f"It needs labelled leads created on or after {test_from} with both wins and losses.",
                          "warn", lead="This test set is empty for this dataset."))
    else:
        head = ev["headline"]
        ui.html(C.section("Headline", f"{ev['n']:,} test leads, {ev['wins']:,} won. AUC is the chance the model "
                                      "ranks a random winner above a random loser (0.5 = coin flip); the top-20% "
                                      "figures are the share of wins and recorded revenue in the leads it ranks "
                                      "highest."))
        _kpis(head)
        ui.html(C.section("Standard table", "The evaluation every change to the model is judged by, on this test set."))
        _standard_table(head, ev["paired"], ev["notes"], ev["baseline_missing"])
        if generic:
            _generic(run, test_set)
        left, right = st.columns(2, gap="large")
        with left:
            ui.html(C.section("Calibration", "Predicted chance vs what actually happened, by tenth of the test "
                                             "set. On the dotted line the probabilities can be taken at face value."))
            st.plotly_chart(charts.calibration_chart(ev["calibration"]), config=charts.CONFIG, theme=None, width="stretch",
                            key="cal_chart")
        with right:
            ui.html(C.section("Stability by month", "AUC for leads created in each test month, with a 95% "
                                                    "bootstrap interval where the month is large enough."))
            st.plotly_chart(charts.auc_month_chart(ev["auc_month"]), config=charts.CONFIG, theme=None, width="stretch",
                            key="month_chart")

    ui.html(C.section("Scorecard", "What moves a lead's score. Each signal adds or removes points relative to its "
                                   "comparison level; 20 points doubles the odds of closing."))
    extras = ui.bundle(run.run_id, str(Path(run.out_dir) / BUNDLE_FILE)).extras if generic else None
    _scorecard(run, extras)
    ui.html(C.section("Value per lead", "The value the ad platforms would receive per lead across every scored "
                                        "lead. A long right tail (high max / median) makes bidding erratic, which "
                                        "is why the value sent at submit is capped and compressed."))
    _values(run)

    ui.html(C.section("Files", "The run's raw outputs, identical to a local python -m emva run."))
    out = Path(run.out_dir)
    cols = st.columns(3)
    for col, (name, label) in zip(cols, [("scores.csv", "Scores per lead"), ("weights.csv", "Scorecard weights")]):
        with col:
            st.download_button(label, (out / name).read_bytes(), file_name=f"{run.run_id}-{name}", mime="text/csv",
                               icon=":material/download:", width="stretch", key=f"dl_{name}")
    if run.has_report:
        # the report names the dataset by its server path; show the dataset name instead
        report_text = run.report_path.read_text(encoding="utf-8").replace(run.dataset_path, run.dataset)
        with cols[2]:
            st.download_button("Standard report", report_text.encode("utf-8"), file_name=f"{run.run_id}-report.md",
                               mime="text/markdown", icon=":material/description:", width="stretch", key="dl_report")
        with st.expander("Full standard report (baseline vs model vs status quo)"):
            st.markdown(report_text)
    else:
        st.caption("No standard report for this run (the dataset is in the sample's format without "
                   "status_quo_rules.json, or the report failed; see the training log).")
    if run.dataset == storage.SAMPLE_DATASET_NAME:
        st.caption("Trained on the bundled synthetic sample: every number here is on simulated data.")
    elif storage.is_converted(run.dataset_path):
        st.caption("Trained on a converted dataset (dataset.json): every number here is on that source's data, not "
                   "on simulated data. For a public export, say \"on public data\"; check its licence before quoting.")


render()
