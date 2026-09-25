"""tROAS eligibility calculator (plan 3.4): are there enough conversions per campaign for value bidding?

Given leads per month, the positive (win) rate and the number of campaigns (or a per-campaign split of the
leads), ``eligibility`` gives each campaign's conversions per 30 days and per week at two upload stages and
flags those below the platform threshold:

- **submit stage**: every lead is a conversion carrying ``value_at_submit`` (conversions = leads);
- **close stage**: only wins count (conversions = leads × positive rate), i.e. optimising on
  ``value_at_close`` alone.

``python -m emva.troas --leads-per-month N --positive-rate R (--campaigns K | --split a,b,...)`` prints the
table; ``--data DIR`` instead derives leads per month per campaign (by channel, the last 12 months of scored
leads) and the positive rate (``won_within_h`` over mature leads) from a dataset, and adds a row pooling all of a
platform's campaigns (what a portfolio bid strategy or a single consolidated campaign would see).
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from emva.constants import AS_OF
from emva.features import channel
from emva.io import clean, load
from emva.labels import is_mature, won_within_h

DAYS_PER_MONTH: float = 365.25 / 12

# Platform thresholds as of the author's knowledge (mid 2026); VERIFY against the current docs before use.
# Google Ads: target ROAS bidding is recommended once a campaign has at least 30 conversions in the past
# 30 days (Google Ads Help, "About Target ROAS bidding" / "Smart Bidding" data requirements; the published
# minimum has varied between 15 and 50 over time, and portfolio strategies pool campaigns).
GOOGLE_TROAS_MIN_CONVERSIONS_30D: float = 30.0
# Meta: an ad set leaves the learning phase after about 50 optimisation events in the 7 days after its last
# significant edit (Meta Business Help Center, "About the learning phase"). VERIFY.
META_MIN_EVENTS_PER_WEEK: float = 50.0


@dataclass(frozen=True)
class Threshold:
    """A platform's minimum: ``minimum`` conversions per ``days`` days, per ``unit`` (campaign or ad set)."""

    platform: str
    minimum: float
    days: int
    unit: str


GOOGLE = Threshold("Google Ads tROAS", GOOGLE_TROAS_MIN_CONVERSIONS_30D, 30, "campaign")
META = Threshold("Meta value optimisation", META_MIN_EVENTS_PER_WEEK, 7, "ad set")
THRESHOLDS: dict[str, Threshold] = {"google": GOOGLE, "meta": META}


def campaign_leads(leads_per_month: float, campaigns: int | None = None,
                   split: Sequence[float] | None = None) -> np.ndarray:
    """Leads per month per campaign: an even split over ``campaigns``, or ``split`` normalised to sum to 1.

    Exactly one of ``campaigns`` (>= 1) and ``split`` (non-empty, all > 0) must be given; ``leads_per_month``
    must be >= 0. Raises ``ValueError`` otherwise.
    """
    if leads_per_month < 0:
        raise ValueError(f"leads_per_month must be >= 0, got {leads_per_month}")
    if (campaigns is None) == (split is None):
        raise ValueError("give exactly one of campaigns and split")
    if campaigns is not None:
        if campaigns < 1:
            raise ValueError(f"campaigns must be >= 1, got {campaigns}")
        return np.full(campaigns, leads_per_month / campaigns)
    w = np.asarray(split, dtype=float)
    if len(w) == 0 or (w <= 0).any():
        raise ValueError("split must be a non-empty list of positive shares")
    return leads_per_month * w / w.sum()


def eligibility(leads: Sequence[float] | np.ndarray, positive_rate: float, threshold: Threshold,
                names: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per campaign: leads per month, conversions per threshold window at both stages, and flags.

    ``leads`` is leads per month per campaign; ``positive_rate`` in [0, 1]. Columns: ``campaign``,
    ``leads_per_month``, ``submit_conversions`` and ``close_conversions`` (per ``threshold.days`` days),
    ``submit_ok`` and ``close_ok`` (at or above ``threshold.minimum``) and ``close_shortfall`` (how many times
    the close-stage volume would have to grow to reach the threshold; <= 1 when it already does).
    """
    if not 0 <= positive_rate <= 1:
        raise ValueError(f"positive_rate must be in [0, 1], got {positive_rate}")
    per_month = np.asarray(leads, dtype=float)
    window = per_month * threshold.days / DAYS_PER_MONTH
    close = window * positive_rate
    shortfall = np.divide(threshold.minimum, close, out=np.full_like(close, np.inf), where=close > 0)
    return pd.DataFrame({
        "campaign": list(names) if names is not None else [f"campaign {i + 1}" for i in range(len(per_month))],
        "leads_per_month": per_month,
        "submit_conversions": window,
        "close_conversions": close,
        "submit_ok": window >= threshold.minimum,
        "close_ok": close >= threshold.minimum,
        "close_shortfall": shortfall,
    })


def volumes_from_data(data: str | Path, as_of: pd.Timestamp = AS_OF) -> tuple[dict[str, pd.Series], float]:
    """Leads per month per campaign for the ``google`` and ``meta`` platforms, and the positive rate, from ``data``.

    Uses scored leads (after bot and duplicate removal) created in the 12 months before ``as_of``; Meta
    campaigns pool the website (``meta``) and lead-form (``meta_leadads``) channels of the same
    ``utm_campaign``. The positive rate is the mean ``won_within_h`` over mature leads (every mature non-win
    counts as 0, as the platform would see it).
    """
    X = clean(load(data))
    X = X[X.created_at > as_of - pd.DateOffset(years=1)]
    ch = pd.Series(channel(X.utm_medium, X.utm_source), index=X.index)
    platform = ch.map({"google": "google", "meta": "meta", "meta_leadads": "meta"})
    counts = X.groupby([platform, X.utm_campaign]).size() / 12
    rate = float(won_within_h(X, as_of=as_of)[is_mature(X, as_of=as_of)].mean())
    return {p: counts.loc[p] for p in THRESHOLDS if p in counts.index.get_level_values(0)}, rate


def format_table(df: pd.DataFrame, threshold: Threshold) -> str:
    """Markdown table of an ``eligibility`` frame, with the threshold in the header."""
    per = f"per {threshold.days} days"
    lines = [f"{threshold.platform}: threshold {threshold.minimum:g} conversions per {threshold.unit} {per} (verify)",
             "", f"| campaign | leads / month | lead conversions {per} | win conversions {per} | submit stage | "
             "close stage | close volume needed × |", "|---|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        lines.append(f"| {r.campaign} | {r.leads_per_month:.1f} | {r.submit_conversions:.1f} | {r.close_conversions:.1f} | "
                     f"{'ok' if r.submit_ok else 'BELOW'} | {'ok' if r.close_ok else 'BELOW'} | {r.close_shortfall:.1f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: print the eligibility tables (see the module docstring)."""
    ap = argparse.ArgumentParser(prog="python -m emva.troas")
    ap.add_argument("--data", default=None, help="derive per-campaign volumes and the positive rate from this dataset")
    ap.add_argument("--leads-per-month", type=float, default=None)
    ap.add_argument("--positive-rate", type=float, default=None, help="share of leads that become wins (0-1)")
    ap.add_argument("--campaigns", type=int, default=None, help="number of campaigns sharing the leads evenly")
    ap.add_argument("--split", default=None, help="comma-separated per-campaign shares, e.g. 0.5,0.3,0.2")
    ap.add_argument("--platform", choices=[*THRESHOLDS, "both"], default="both")
    a = ap.parse_args(argv)
    platforms = list(THRESHOLDS) if a.platform == "both" else [a.platform]

    if a.data is not None:
        if any(v is not None for v in (a.leads_per_month, a.campaigns, a.split)):
            ap.error("--data derives volumes itself; do not combine it with --leads-per-month/--campaigns/--split")
        volumes, rate = volumes_from_data(a.data)
        rate = rate if a.positive_rate is None else a.positive_rate
        print(f"{a.data}: positive rate {rate:.3f} (won within H over mature leads)\n")
        for p in platforms:
            v = volumes[p]
            leads, names = [*v.values, v.sum()], [*v.index, "all campaigns pooled (portfolio)"]
            print(format_table(eligibility(leads, rate, THRESHOLDS[p], names), THRESHOLDS[p]) + "\n")
        return

    if a.leads_per_month is None or a.positive_rate is None:
        ap.error("give --leads-per-month and --positive-rate (or --data)")
    try:
        split = None if a.split is None else [float(x) for x in a.split.split(",")]
        leads = campaign_leads(a.leads_per_month, a.campaigns, split)
        for p in platforms:
            print(format_table(eligibility(leads, a.positive_rate, THRESHOLDS[p]), THRESHOLDS[p]) + "\n")
    except ValueError as e:
        ap.error(str(e))


if __name__ == "__main__":
    main()
