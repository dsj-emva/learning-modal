"""Page: Model results. How well a trained run ranks leads on the frozen test set, and why it scores what it does."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app import charts, results, storage, ui
from app import components as C
from emva.constants import TEST_FROM

TEST_SET_LABELS = {"mature": "Mature leads (horizon labels)", "legacy": "Legacy labels (frozen POC)"}


def _kpis(head: pd.DataFrame) -> None:
    """The KPI strip: this model against the status quo (or, without rules, the frozen baseline)."""
    m = head.set_index("model")
    model = m.loc[results.CANDIDATE]
    ref_name = next((n for n in (results.STATUS_QUO, results.BASELINE) if n in m.index), None)
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


def _standard_table(head: pd.DataFrame, paired: pd.DataFrame | None, notes: list[str]) -> None:
    """The standard report's headline and paired comparison (``emva.eval.report`` frames), formatted."""
    rows = [[r.model, C.num(r.auc), f"{r.auc_lo:.3f} – {r.auc_hi:.3f}", C.num(r.brier, 4), C.pct(r.top20_wins),
             C.pct(r.top20_revenue_p), C.pct(r.top20_revenue_value)] for r in head.itertuples()]
    ui.html(C.table(["Scored by", "AUC", "95% interval", "Brier", "Top-20% wins", "Top-20% revenue (by p)",
                     "Top-20% revenue (by value)"], rows))
    caption = ["Revenue is the recorded deal value of won test leads."]
    if results.STATUS_QUO in set(head.model):
        caption.insert(0, "Brier needs a probability, so the status quo (a bucket value) has none.")
    st.caption(" ".join(caption + notes))
    if paired is not None and not paired.empty:
        ui.html(C.section("Compared with the frozen baseline", "AUC difference on the same bootstrap resamples; "
                          "an interval that excludes 0 is a real difference, not noise."))
        ui.html(C.table(["Scored by", "AUC − baseline", "95% interval", "Bootstrap p"],
                        [[r.model, f"{r.auc_diff:+.3f}", f"{r.diff_lo:+.3f} – {r.diff_hi:+.3f}", f"{r.p_value:.3f}"]
                         for r in paired.itertuples()]))


def _scorecard(run: storage.Run) -> None:
    """Scorecard: signed bar chart and a sortable table."""
    card = results.scorecard(run.out_dir)
    labels = [f"{f} · {lvl}" for f, lvl in zip(card.feature, card.level)]
    chart_tab, table_tab = st.tabs(["Chart", "Table"])
    with chart_tab:
        st.plotly_chart(charts.points_chart(labels, card.points.tolist()), config=charts.CONFIG, theme=None,
                        width="stretch", key="scorecard_chart")
    with table_tab:
        t = card.assign(strength=card.points.abs() / max(card.points.abs().max(), 1),
                        effect=np.where(card.points >= 0, "raises", "lowers"))
        pos, neg = "color: #0B4644; font-weight: 600", "color: #B5543C; font-weight: 600"
        styled = t[["feature", "level", "reference", "points", "strength", "odds_multiplier", "effect"]].style.map(
            lambda v: pos if v >= 0 else neg, subset=["points"])
        st.dataframe(styled, hide_index=True, width="stretch", height=460, column_config={
            "feature": st.column_config.TextColumn("Signal"),
            "level": st.column_config.TextColumn("When the lead is"),
            "reference": st.column_config.TextColumn("Compared with"),
            "points": st.column_config.NumberColumn("Points", format="%+d"),
            "strength": st.column_config.ProgressColumn("Strength", min_value=0.0, max_value=1.0, format=" "),
            "odds_multiplier": st.column_config.NumberColumn("Odds ×", format="%.2f×"),
            "effect": st.column_config.TextColumn("Effect"),
        })


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

    ev = ui.evaluation(run.run_id, run.out_dir, run.dataset_path, test_set)
    if ev is None:
        ui.html(C.callout(f"It needs labelled leads created on or after {TEST_FROM} with both wins and losses.",
                          "warn", lead="This test set is empty for this dataset."))
    else:
        head = ev["headline"]
        ui.html(C.section("Headline", f"{ev['n']:,} test leads, {ev['wins']:,} won. AUC is the chance the model "
                                      "ranks a random winner above a random loser (0.5 = coin flip); the top-20% "
                                      "figures are the share of wins and recorded revenue in the leads it ranks "
                                      "highest."))
        _kpis(head)
        ui.html(C.section("Standard table", "The evaluation every change to the model is judged by, on this test set."))
        _standard_table(head, ev["paired"], ev["notes"])
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
    _scorecard(run)
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
        st.caption("No standard report for this run (the dataset has no status_quo_rules.json, or the report failed; "
                   "see the training log).")
    if run.dataset == storage.SAMPLE_DATASET_NAME:
        st.caption("Trained on the bundled synthetic sample: every number here is on simulated data.")


render()
