"""Reconstruct the status-quo value from ``status_quo_rules.json`` (EMVA's benchmark).

Value at submit = ``base_value_by_form[form_variant]``
× each ``value_rules`` multiplier whose field value is in its list
× the lead-score tier multiplier.

Lead score = job-title points + company-size points + pages-visited points:

- job title: lower-cased substring match against ``keywords`` in listed order, first match
  wins; a non-blank title with no match gets ``default_points``; blank or not asked gets
  ``missing_points``.
- company size: the typed dropdown answer (``a_company_size``), not the enrichment band;
  blank or not asked scores 0.
- pages visited: points of the first ``min_pages`` threshold met (thresholds listed high to
  low); missing (Lead Ads) scores 0.
- tier: the first tier whose ``min_points`` the score reaches (listed high to low).

Rule fields map to lead columns via ``RULE_FIELDS``; ``channel`` uses ``emva.features.channel``.
``stage_values`` and ``won_value`` describe values an advertiser uploads later as the CRM
stage moves; they are not known at submit time and are not used for ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from emva.features import channel

RULE_FIELDS: dict[str, str] = {"country": "a_country", "channel": "channel"}


def load_rules(path: str | Path) -> dict:
    """Read ``status_quo_rules.json``."""
    with open(path) as f:
        return json.load(f)


def job_title_points(title: object, rules: dict) -> int:
    """Points for one job title under ``rules['lead_score']['job_title']``."""
    jt = rules["lead_score"]["job_title"]
    if not isinstance(title, str) or not title.strip():
        return jt["missing_points"]
    t = title.lower()
    for kw in jt["keywords"]:
        if kw["keyword"] in t:
            return kw["points"]
    return jt["default_points"]


def pages_visited_points(pages: float, rules: dict) -> int:
    """Points for a pages-visited count (NaN = no page, 0 points)."""
    if pd.isna(pages):
        return 0
    for rule in rules["lead_score"]["pages_visited_points"]:
        if pages >= rule["min_pages"]:
            return rule["points"]
    return 0


def lead_score(X: pd.DataFrame, rules: dict) -> pd.Series:
    """Status-quo lead score per row. Needs ``a_job_title``, ``a_company_size``, ``pages_visited``."""
    size_pts = X.a_company_size.map(rules["lead_score"]["company_size_points"]).fillna(0)
    return (X.a_job_title.apply(lambda t: job_title_points(t, rules)) + size_pts
            + X.pages_visited.apply(lambda n: pages_visited_points(n, rules))).astype(float).rename("sq_points")


def tier(points: float, rules: dict) -> dict:
    """The first tier (``name``, ``min_points``, ``multiplier``) whose threshold ``points`` reaches."""
    for t in rules["lead_score"]["tiers"]:
        if points >= t["min_points"]:
            return t
    raise ValueError(f"no tier covers {points} points")


def status_quo_value(X: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Per-lead status quo: ``sq_points``, ``sq_tier`` and ``sq_value`` (GBP at submit).

    Needs ``form_variant``, ``a_country``, ``utm_medium``, ``utm_source``, ``a_job_title``,
    ``a_company_size`` and ``pages_visited`` (all present after ``emva.io.load``).
    """
    fields = pd.DataFrame({"a_country": X.a_country, "channel": channel(X.utm_medium, X.utm_source)}, index=X.index)
    value = X.form_variant.map(rules["base_value_by_form"]).astype(float)
    if value.isna().any():
        raise ValueError(f"form variants without a base value: {sorted(X.form_variant[value.isna()].unique())}")
    for rule in rules["value_rules"]:
        col = RULE_FIELDS[rule["field"]]
        value = value * np.where(fields[col].isin(rule["in"]), rule["multiplier"], 1.0)
    points = lead_score(X, rules)
    tiers = points.apply(lambda p: tier(p, rules))
    return pd.DataFrame({"sq_points": points, "sq_tier": tiers.map(lambda t: t["name"]),
                         "sq_value": value * tiers.map(lambda t: t["multiplier"])}, index=X.index)
