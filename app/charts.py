"""Plotly charts for Keel, all on one registered template (``template()``): Keel's fonts and palette, no gridline
clutter, no chart titles (sections carry the titles), and ``CONFIG`` hides the mode bar. Pages pass
``theme=None`` to ``st.plotly_chart`` so Streamlit's own chart theme does not override the template; the
backgrounds are set on each figure (Streamlit replaces a template-only plot background with its secondary one). Points bars: positive
in the accent, negative in the muted warm red.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from app.theme import FONT_MONO, FONT_UI, PALETTE

TEMPLATE_NAME: str = "keel"
# Passed to st.plotly_chart: no mode bar, no Plotly logo, no scroll zoom.
CONFIG: dict[str, object] = {"displayModeBar": False, "displaylogo": False, "scrollZoom": False, "responsive": True}


def template() -> str:
    """Register the ``keel`` template (idempotent) and return its name."""
    if TEMPLATE_NAME not in pio.templates:
        p = PALETTE
        axis = dict(automargin=True, showgrid=False, zeroline=False, showline=True, linecolor=p["hairline"], linewidth=1,
                    ticks="outside", tickcolor=p["hairline"], ticklen=4, tickfont=dict(family=FONT_MONO, size=11,
                    color=p["muted"]), title=dict(font=dict(family=FONT_UI, size=12, color=p["muted"])))
        pio.templates[TEMPLATE_NAME] = go.layout.Template(layout=go.Layout(
            font=dict(family=FONT_UI, size=13, color=p["ink"]), paper_bgcolor=p["paper"],
            plot_bgcolor=p["paper"], colorway=[p["accent"], p["sand"], p["negative"], p["muted"]],
            xaxis=axis, yaxis={**axis, "showgrid": True, "gridcolor": "#EFEBE3", "showline": False},
            margin=dict(l=8, r=8, t=16, b=8), hoverlabel=dict(bgcolor=p["surface"], bordercolor=p["hairline"],
            font=dict(family=FONT_UI, size=12, color=p["ink"])),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12, color=p["muted"])),
            bargap=0.28))
    return TEMPLATE_NAME


def _fig(height: int) -> go.Figure:
    """An empty figure on the Keel template."""
    # the backgrounds are repeated outside the template: Streamlit replaces a template-only plot_bgcolor
    return go.Figure(layout=dict(template=template(), height=height, paper_bgcolor=PALETTE["paper"],
                                 plot_bgcolor=PALETTE["paper"]))


def calibration_chart(cal: pd.DataFrame) -> go.Figure:
    """Mean predicted vs observed win rate per decile, with the diagonal (perfect calibration)."""
    p = PALETTE
    top = float(max(cal.mean_p.max(), cal.observed.max(), 0.05)) * 1.08
    fig = _fig(340)
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=p["hairline"], dash="dot", width=1.5),
                             hoverinfo="skip", name="Perfectly calibrated"))
    fig.add_trace(go.Scatter(
        x=cal.mean_p, y=cal.observed, mode="lines+markers", name="Model",
        line=dict(color=p["accent"], width=2), marker=dict(size=8, color=p["accent"], line=dict(color="#fff", width=1.5)),
        customdata=np.stack([cal.decile, cal.n], axis=1),
        hovertemplate="Decile %{customdata[0]} · %{customdata[1]} leads<br>Predicted %{x:.1%} · observed %{y:.1%}"
                      "<extra></extra>"))
    fig.update_xaxes(title="Predicted chance of closing", tickformat=".0%", range=[0, top])
    fig.update_yaxes(title="Observed win rate", tickformat=".0%", range=[0, top])
    return fig


def auc_month_chart(month: pd.DataFrame) -> go.Figure:
    """AUC per test month with 95% bootstrap error bars where available, and the 0.5 (coin flip) line."""
    p = PALETTE
    fig = _fig(340)
    fig.add_hline(y=0.5, line=dict(color=p["hairline"], dash="dot", width=1.5))
    up = (month.auc_hi - month.auc).fillna(0)
    down = (month.auc - month.auc_lo).fillna(0)
    fig.add_trace(go.Scatter(x=month.month, y=month.auc, mode="lines+markers", name="AUC",
                             line=dict(color=p["accent"], width=2),
                             marker=dict(size=9, color=p["accent"], line=dict(color="#fff", width=1.5)),
                             error_y=dict(type="data", symmetric=False, array=up, arrayminus=down,
                                          color="rgba(15,92,90,0.45)", thickness=1.5, width=6),
                             customdata=np.stack([month.n, month.wins, month.auc_lo, month.auc_hi], axis=1),
                             hovertemplate="%{x} · AUC %{y:.3f} [%{customdata[2]:.3f}, %{customdata[3]:.3f}]<br>"
                                           "%{customdata[0]} leads, %{customdata[1]} won<extra></extra>"))
    fig.update_xaxes(type="category", title=None)
    fig.update_yaxes(title="AUC", range=[0.3, 1.0])
    fig.update_layout(showlegend=False)
    return fig


def points_chart(labels: list[str], values: list[float], height_per_bar: int = 26) -> go.Figure:
    """Signed horizontal bars (points), first label at the top; positive accent, negative warm red."""
    p = PALETTE
    colors = [p["accent"] if v >= 0 else p["negative"] for v in values]
    text = [f"+{v:.0f}" if v > 0 else f"−{abs(v):.0f}" if v < 0 else "0" for v in values]
    fig = _fig(max(160, 40 + height_per_bar * len(values)))
    fig.add_trace(go.Bar(x=values, y=labels, orientation="h", marker=dict(color=colors), text=text,
                         textposition="outside", textfont=dict(family=FONT_MONO, size=11, color=p["muted"]),
                         cliponaxis=False, hovertemplate="%{y}: %{x:+.0f} points<extra></extra>"))
    span = max((abs(v) for v in values), default=1) * 1.45
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False, ticks="",
                     tickfont=dict(family=FONT_UI, size=12, color=p["ink"]))
    fig.update_xaxes(range=[-span, span], showgrid=True, gridcolor="#EFEBE3", zeroline=True, zerolinecolor=p["muted"],
                     zerolinewidth=1, title="Points (+20 doubles the odds of closing)")
    fig.update_layout(bargap=0.35)
    return fig


def value_histogram(series: dict[str, np.ndarray]) -> go.Figure:
    """Overlaid histograms of value columns on a log x axis (GBP); first series in the accent."""
    p = PALETTE
    colors = [p["accent"], p["sand"]]
    fig = _fig(320)
    for (name, v), c in zip(series.items(), colors):
        v = np.asarray(v, dtype=float)
        v = v[v > 0]
        fig.add_trace(go.Histogram(x=np.log10(v), name=name, marker=dict(color=c, line=dict(width=0)), opacity=0.75,
                                   nbinsx=48, hovertemplate="%{y} leads<extra>%{fullData.name}</extra>"))
    lo = int(np.floor(min(np.log10(np.asarray(v)[np.asarray(v) > 0]).min() for v in series.values())))
    hi = int(np.ceil(max(np.log10(np.asarray(v)[np.asarray(v) > 0]).max() for v in series.values())))
    ticks = list(range(lo, hi + 1))
    fig.update_xaxes(tickvals=ticks, ticktext=[f"£{10 ** t:,.0f}" for t in ticks], title="Value per lead (log scale)")
    fig.update_yaxes(title="Leads")
    fig.update_layout(barmode="overlay")
    return fig


def stage_chart(counts: dict[str, int]) -> go.Figure:
    """Horizontal bars of leads per final CRM stage, largest first."""
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    fig = _fig(max(140, 36 + 26 * len(items)))
    fig.add_trace(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                         marker=dict(color=[PALETTE["accent"] if k == "Won" else "#C9CFC9" for k, _ in items]),
                         text=[f"{v:,}" for _, v in items], textposition="outside", cliponaxis=False,
                         textfont=dict(family=FONT_MONO, size=11, color=PALETTE["muted"]),
                         hovertemplate="%{y}: %{x:,} leads<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    fig.update_xaxes(showgrid=False, showticklabels=False, showline=False, ticks="",
                     range=[0, max((v for _, v in items), default=1) * 1.18])
    return fig


__all__ = ["CONFIG", "auc_month_chart", "calibration_chart", "points_chart", "stage_chart", "template",
           "value_histogram"]
