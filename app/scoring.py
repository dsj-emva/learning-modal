"""Form adapter over ``emva.scoring`` for the "Score a lead" page. No Streamlit, no feature code.

The page renders ``form_sections()`` (every ``emva.scoring.submit_time_fields`` field, grouped, with defaults and
labels), collects a dict of values and calls ``score_form``, which builds the lead with ``lead_from_form`` and
scores it with ``score_leads`` / ``points_breakdown`` against the run's bundle, so the app never re-implements a
feature. Session fields default to typical values because a blank one sets ``session_missing=yes`` (v2 features).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.results import feature_label
from emva.features import SESSION_INPUTS
from emva.persist import ModelBundle
from emva.scoring import FieldSpec, is_blank, lead_from_form, points_breakdown, score_leads, submit_time_fields

# Sections of the form, in display order: title -> field names.
SECTIONS: dict[str, tuple[str, ...]] = {
    "Form answers": ("what_to_solve", "job_title", "company_size", "budget", "timeline", "country", "form_variant"),
    "Contact": ("email", "company", "company_name", "company_domain"),
    "Attribution": ("utm_source", "utm_medium", "utm_term", "landing_url"),
    "Session": ("time_on_page_s", "hesitation_ms", "sessions_before_convert", "viewed_pricing", "field_edit_count",
                "submitted_weekday", "local_submit_hour", "ip_country", "is_datacenter_ip", "user_agent"),
}
# A typical, plausible lead: every session field filled (a blank one would mark the whole session as missing).
DEFAULTS: dict[str, object] = {
    "what_to_solve": "We need to attribute pipeline to paid social; team of 6 marketers, budget agreed for Q4",
    "job_title": "Head of Marketing", "company_size": "51-200", "budget": "", "timeline": "", "country": "UK",
    "form_variant": "A", "email": "jane.doe@acme-analytics.example", "company": "Acme Analytics",
    "company_name": "Acme Analytics", "company_domain": "acme-analytics.example", "utm_source": "google",
    "utm_medium": "cpc", "utm_term": "attribution software", "landing_url": "https://lp.example/lp.html?variant=A",
    "time_on_page_s": 120.0, "hesitation_ms": 9000.0, "sessions_before_convert": 2, "viewed_pricing": False,
    "field_edit_count": 2, "submitted_weekday": "Tuesday", "local_submit_hour": 11, "ip_country": "UK",
    "is_datacenter_ip": False,
    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Safari/605.1.15",
}
LABELS: dict[str, str] = {
    "what_to_solve": "What do they want to solve?", "job_title": "Job title", "company_size": "Company size (typed)",
    "budget": "Budget", "timeline": "Timeline", "country": "Country (typed)", "form_variant": "Form variant",
    "email": "Email", "company": "Company (typed in form)", "company_name": "Company name (CRM)",
    "company_domain": "Company domain", "utm_source": "UTM source", "utm_medium": "UTM medium",
    "utm_term": "Search term", "landing_url": "Landing page URL", "time_on_page_s": "Time on page (s)",
    "hesitation_ms": "Longest pause (ms)", "sessions_before_convert": "Visits before converting",
    "viewed_pricing": "Viewed pricing page", "field_edit_count": "Form fields edited",
    "submitted_weekday": "Weekday (local)", "local_submit_hour": "Hour (local, 0-23)", "ip_country": "IP country",
    "is_datacenter_ip": "Data-centre IP", "user_agent": "Browser user agent",
}
SESSION_FIELDS: frozenset[str] = frozenset(SESSION_INPUTS) | {"landing_url"}


def defaults_for(data: str | Path) -> dict[str, object]:
    """``DEFAULTS`` with the company taken from the first row of ``data/companies.csv``, so the example lead is
    enriched like a known company (the built-in example domain is in no dataset)."""
    co = pd.read_csv(Path(data) / "companies.csv", nrows=1)
    if co.empty:
        return dict(DEFAULTS)
    name, domain = str(co.company_name.iloc[0]), str(co.domain.iloc[0])
    local = DEFAULTS["email"].split("@")[0]
    return {**DEFAULTS, "company": name, "company_name": name, "company_domain": domain, "email": f"{local}@{domain}"}


def values_from_lead(row: pd.Series) -> dict[str, object]:
    """Form values for one ``historical_leads.csv`` row (as read by ``emva.io.read_leads``): its scoring columns
    and ``answers`` keys, blanks as None. Used to prefill the form with a historical lead."""
    answers = json.loads(row["answers"]) if isinstance(row.get("answers"), str) else {}
    out: dict[str, object] = {}
    for f in submit_time_fields():
        v = answers.get(f.name) if f.source == "answers" else row.get(f.name)
        if is_blank(v):
            out[f.name] = None
        elif f.dtype == "bool":
            out[f.name] = bool(v)
        elif f.dtype == "int":
            out[f.name] = int(v)
        elif f.dtype == "float":
            out[f.name] = float(v)
        else:
            out[f.name] = str(v)
    return out


@dataclass(frozen=True)
class FormField:
    """A ``FieldSpec`` with its form label and default."""

    spec: FieldSpec
    label: str
    default: object

    @property
    def name(self) -> str:
        """The field name (``submit_time_fields``)."""
        return self.spec.name

    @property
    def options(self) -> tuple[str, ...] | None:
        """Choices for a categorical field, a blank first for optional ones."""
        if self.spec.allowed is None:
            return None
        return self.spec.allowed if self.spec.required else ("", *self.spec.allowed)


def form_sections() -> dict[str, list[FormField]]:
    """Every scoring field, grouped by ``SECTIONS``; a field missing from ``SECTIONS`` lands in the last section,
    so a new ``submit_time_fields`` entry is never silently left out of the form."""
    specs = {f.name: f for f in submit_time_fields()}
    out = {title: [FormField(specs[n], LABELS.get(n, n), DEFAULTS.get(n, "")) for n in names if n in specs]
           for title, names in SECTIONS.items()}
    placed = {n for names in SECTIONS.values() for n in names}
    last = list(SECTIONS)[-1]
    out[last] += [FormField(s, LABELS.get(n, n), DEFAULTS.get(n, "")) for n, s in specs.items() if n not in placed]
    return out


def clean_values(values: dict[str, object]) -> dict[str, object]:
    """Form widget values as ``lead_from_form`` expects them: strings stripped, blank strings dropped from
    numeric fields, whole-number floats as ints for integer fields."""
    specs = {f.name: f for f in submit_time_fields()}
    out: dict[str, object] = {}
    for k, v in values.items():
        if isinstance(v, str):
            v = v.strip()
        if specs[k].dtype == "int" and isinstance(v, float) and v.is_integer():
            v = int(v)
        if specs[k].dtype in ("int", "float") and is_blank(v):
            v = None
        out[k] = v
    return out


def blank_session_fields(values: dict[str, object]) -> list[str]:
    """Session fields left blank (each one makes the model treat the whole session as missing)."""
    return [k for k in sorted(SESSION_FIELDS) if is_blank(values.get(k))]


@dataclass(frozen=True)
class LeadScore:
    """One scored lead: ``p`` (P(close)), ``deal_value`` (expected deal value if won, GBP), ``value_formula``
    (p × deal value × margin), ``value_at_submit`` (after the run's value transform; NaN for legacy labels),
    ``is_bot`` / ``is_duplicate``, ``points`` (the intercept row then active features, with plain labels) and
    ``blank_session`` (session fields left blank, each of which makes the model treat the session as missing)."""

    p: float
    deal_value: float
    value_formula: float
    value_at_submit: float
    is_bot: bool
    is_duplicate: bool
    points: pd.DataFrame
    blank_session: tuple[str, ...]


def score_form(bundle: ModelBundle, values: dict[str, object], data: str | Path) -> LeadScore:
    """Score the lead described by form ``values`` with ``bundle``; ``data`` holds ``companies.csv``.

    Raises ``ValueError`` for bad form input (``lead_from_form``) and ``emva.scoring.UnknownLevelError`` (a
    ``ValueError`` subclass) for a value the model was not trained with.
    """
    cleaned = clean_values(values)
    lead = lead_from_form(cleaned)
    s = score_leads(bundle, lead, data).iloc[0]
    pts = points_breakdown(bundle, lead, data)
    pts = pts.assign(label=pts.feature.map(feature_label))
    return LeadScore(p=float(s.p_formula), deal_value=float(s.deal_value_hat), value_formula=float(s.value_formula),
                     value_at_submit=float(s.value_at_submit), is_bot=bool(s.is_bot),
                     is_duplicate=bool(s.is_duplicate), points=pts,
                     blank_session=tuple(blank_session_fields(cleaned)))


def describe_transform(bundle: ModelBundle) -> str:
    """One sentence on how ``value_at_submit`` is derived from p × deal value for this bundle."""
    vt = bundle.value_transform
    if vt is None:
        return "This run used legacy labels, so no value is sent at submit; p × deal value is the value."
    steps = []
    if vt.cap is not None:
        steps.append(f"capped at £{vt.cap:,.0f} (the {vt.spec.cap_percentile:g}th percentile of training leads)")
    if vt.anchor is not None:
        steps.append(f"{vt.spec.compression.value}-compressed around the median £{vt.anchor:,.0f} and rescaled so "
                     "totals stay on the revenue scale")
    if vt.spec.floor is not None:
        steps.append(f"floored at £{vt.spec.floor:,.0f}")
    if vt.tier_values:
        steps.append(f"mapped to {len(vt.tier_values)} value tiers")
    return "p × expected deal value, " + ", then ".join(steps) + "." if steps else "p × expected deal value."


__all__ = ["DEFAULTS", "FormField", "LeadScore", "SECTIONS", "blank_session_fields", "clean_values",
           "defaults_for", "describe_transform", "form_sections", "score_form", "values_from_lead"]
