"""Form adapter over ``emva.scoring`` for the "Score a lead" page. No Streamlit, no feature code.

The page renders ``form_sections()`` (every ``emva.scoring.submit_time_fields`` field, grouped, with defaults and
labels), collects a dict of values and calls ``score_form``, which builds the lead with ``lead_from_form`` and
scores it with ``score_leads`` / ``points_breakdown`` against the run's bundle, so the app never re-implements a
feature. Session fields default to typical values because a blank one sets ``session_missing=yes`` (v2 features).

Source format (Phase 9): when the run's dataset was converted with a mapping (``mapping.toml``, ``run_mapping``),
leads can also be given as the source sends them: ``source_fields`` lists the raw columns the mapping reads at
submit time (``DatasetMapping.source_columns(submit_time_only=True)``), ``frames_from_source_form`` /
``frames_from_source_uploads`` build raw frames from a form or uploaded CSVs, and ``score_source`` converts them
with ``emva.ingest.convert_leads`` (the same field rules as the dataset's conversion) and scores them with the same
``score_leads`` / ``points_breakdown`` as the form.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.results import feature_label
from app.storage import MAPPING_FILE
from emva.features import SESSION_INPUTS
from emva.ingest.convert import convert_leads, frames_from_bytes
from emva.ingest.mapping import IGNORE, DatasetMapping, load_mapping
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
        out[f.name] = f.coerce(answers.get(f.name) if f.source == "answers" else row.get(f.name))
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
    """Form widget values as ``lead_from_form`` expects them: strings stripped, then ``FieldSpec.coerce`` (blank
    numbers None, whole-number floats as ints for integer fields)."""
    specs = {f.name: f for f in submit_time_fields()}
    out: dict[str, object] = {}
    for k, v in values.items():
        out[k] = specs[k].coerce(v.strip() if isinstance(v, str) else v)
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
    return _lead_score(bundle, lead, s, data, tuple(blank_session_fields(cleaned)))


def _lead_score(bundle: ModelBundle, lead: pd.DataFrame, s: pd.Series, data: str | Path,
                blank: tuple[str, ...]) -> LeadScore:
    """A ``LeadScore`` from one lead's row of ``score_leads`` and its ``points_breakdown``."""
    pts = points_breakdown(bundle, lead, data)
    pts = pts.assign(label=pts.feature.map(feature_label))
    return LeadScore(p=float(s.p_formula), deal_value=float(s.deal_value_hat), value_formula=float(s.value_formula),
                     value_at_submit=float(s.value_at_submit), is_bot=bool(s.is_bot),
                     is_duplicate=bool(s.is_duplicate), points=pts, blank_session=blank)


# --- source format ---------------------------------------------------------------------------------------------------

def run_mapping(dataset_path: str | Path) -> DatasetMapping | None:
    """The confirmed mapping the dataset was converted with (its ``mapping.toml``), or None when it has none.

    Raises ``ValueError`` when the file is not a valid confirmed mapping (``load_mapping``).
    """
    path = Path(dataset_path) / MAPPING_FILE
    if not path.is_file():
        return None
    return load_mapping(path.read_text(encoding="utf-8"), require_confirmed=True)


@dataclass(frozen=True)
class SourceField:
    """One raw column a new lead in the source format carries: its ``file`` and ``column``, the value-map keys as
    ``options`` (None = free text) and ``feeds``, what the mapping makes of it (for the form's help text)."""

    file: str
    column: str
    options: tuple[str, ...] | None
    feeds: str

    @property
    def key(self) -> str:
        """``<file>:<column>``, unique per field."""
        return f"{self.file}:{self.column}"


def _feeds(mapping: DatasetMapping) -> dict[tuple[str, str], tuple[str, tuple[str, ...] | None]]:
    """(file, column) -> (what it feeds, value-map keys or None) for every submit-time column of ``mapping``."""
    by_name = {s.name: s for s in mapping.sources}
    out: dict[tuple[str, str], tuple[str, tuple[str, ...] | None]] = {}

    def add(ref: str, feeds: str, options: tuple[str, ...] | None = None) -> None:
        name, column = ref.split(".", 1)
        out.setdefault((by_name[name].file, column), (feeds, options))

    for role in ("lead_id", "created_at"):
        e = getattr(mapping.lead, role)
        for c in e.columns() if e is not None else []:
            add(c, f"lead {role} (optional: blank = {'a new id' if role == 'lead_id' else 'now'})")
    for f in mapping.fields:
        if f.target == IGNORE:
            continue
        options = tuple(f.value_map) if f.value_map is not None else None
        add(f.source, ", ".join(f.targets()) or "nothing (every listed value sets nothing)", options)
    return out


def source_fields(mapping: DatasetMapping) -> list[SourceField]:
    """The raw columns a new lead needs, per file in ``DatasetMapping.source_columns(submit_time_only=True)`` order.

    A joined file's join column is not listed (the form copies the primary file's value into it); a column with a
    value map offers its keys, since any other value is refused by the conversion.
    """
    feeds = _feeds(mapping)
    joins = {s.file: s.join_on for s in mapping.sources[1:]}
    primary = mapping.sources[0].file
    out = []
    for file, columns in mapping.source_columns(submit_time_only=True).items():
        for c in columns:
            if joins.get(file) == c:
                continue
            what, options = feeds.get((file, c), ("join key of a joined file" if file == primary else "", None))
            out.append(SourceField(file, c, options, what))
    return out


def frames_from_source_form(mapping: DatasetMapping, values: dict[str, object]) -> dict[str, pd.DataFrame]:
    """One new lead as raw frames (file -> one-row frame of text) from form ``values`` keyed by ``SourceField.key``.

    Blank values are missing cells; a blank lead id leaves its column out (``convert_leads`` then numbers the lead),
    and each joined file's join column gets the primary file's value.
    """
    fields = source_fields(mapping)
    rows: dict[str, dict[str, object]] = {file: {} for file in mapping.source_columns(submit_time_only=True)}
    for f in fields:
        v = values.get(f.key)
        rows[f.file][f.column] = (v.strip() or None) if isinstance(v, str) else v
    id_cols = mapping.lead.lead_id.columns() if mapping.lead.lead_id is not None else []
    by_name = {s.name: s for s in mapping.sources}
    for ref in id_cols:
        name, column = ref.split(".", 1)
        if rows.get(by_name[name].file, {}).get(column) is None:
            rows[by_name[name].file].pop(column, None)
    primary = rows.get(mapping.sources[0].file, {})
    for s in mapping.sources[1:]:
        if s.file in rows:
            rows[s.file][s.join_on] = primary.get(s.join_on)
    return {file: pd.DataFrame([row], dtype=object) for file, row in rows.items()}


def frames_from_source_uploads(mapping: DatasetMapping, uploads: dict[str, bytes]) -> dict[str, pd.DataFrame]:
    """Raw frames from uploaded CSVs of new leads in the source format (outcome columns may be present; ignored).

    Files are matched to the mapping's files by name; a single upload with another name is taken as the primary
    file. Raises ``ValueError`` naming the files the mapping's submit-time part needs that are missing, or a file
    that is not a readable CSV.
    """
    needed = list(mapping.source_columns(submit_time_only=True))
    primary = mapping.sources[0].file
    named = dict(uploads)
    if len(named) == 1 and next(iter(named)) not in needed:
        named = {primary: next(iter(named.values()))}
    missing = [f for f in needed if f not in named]
    if missing:
        raise ValueError(f"missing source file(s) {missing}; the mapping reads them at submit time (uploaded: "
                         f"{sorted(uploads)})")
    try:
        return frames_from_bytes({f: named[f] for f in needed})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise ValueError(f"not a readable CSV file ({type(e).__name__}: {str(e).splitlines()[0]})") from e


@dataclass(frozen=True)
class SourceScores:
    """New leads in the source format, converted (``leads``: ``historical_leads`` rows, ``lead_id`` column) and
    scored (``scores``: ``emva.scoring.score_leads`` output indexed by ``lead_id``)."""

    leads: pd.DataFrame
    scores: pd.DataFrame

    def table(self) -> pd.DataFrame:
        """Per lead: ``lead_id``, ``p``, ``deal_value``, ``value_at_submit``, ``value_formula``, ``is_bot``,
        ``is_duplicate`` (the page's table and the scores CSV)."""
        s = self.scores
        return pd.DataFrame({"lead_id": s.index, "p": s.p_formula.to_numpy(),
                             "deal_value": s.deal_value_hat.to_numpy(), "value_at_submit": s.value_at_submit.to_numpy(),
                             "value_formula": s.value_formula.to_numpy(), "is_bot": s.is_bot.to_numpy(),
                             "is_duplicate": s.is_duplicate.to_numpy()})


def score_source(bundle: ModelBundle, mapping: DatasetMapping, frames: dict[str, pd.DataFrame],
                 data: str | Path) -> SourceScores:
    """Convert raw source ``frames`` with ``mapping`` (``emva.ingest.convert_leads``) and score them with ``bundle``
    (``score_leads``; ``data`` holds the ``companies.csv`` the enrichment join reads).

    Raises ``ValueError`` for what ``convert_leads`` refuses (e.g. a value outside a value map, a repeated lead id)
    and ``UnknownLevelError`` (a ``ValueError``) for a value the model was not trained with.
    """
    leads = convert_leads(frames, mapping)
    return SourceScores(leads, score_leads(bundle, leads, data))


def source_lead_score(bundle: ModelBundle, result: SourceScores, lead_id: str, data: str | Path) -> LeadScore:
    """The ``LeadScore`` of one scored source-format lead (``points_breakdown`` computed for that lead only)."""
    lead = result.leads[result.leads.lead_id == lead_id]
    blank = tuple(k for k in sorted(SESSION_FIELDS) if k in lead and is_blank(lead[k].iloc[0]))
    return _lead_score(bundle, lead, result.scores.loc[lead_id], data, blank)


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


__all__ = ["DEFAULTS", "FormField", "LeadScore", "SECTIONS", "SourceField", "SourceScores", "blank_session_fields",
           "clean_values", "defaults_for", "describe_transform", "form_sections", "frames_from_source_form",
           "frames_from_source_uploads", "run_mapping", "score_form", "score_source", "source_fields",
           "source_lead_score", "values_from_lead"]
