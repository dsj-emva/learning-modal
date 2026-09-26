"""Score new leads with a saved ``ModelBundle`` through the training code path (Phase 8 app seam).

``score_leads`` runs exactly the functions ``emva.pipeline.run`` runs on a lead, minus everything that needs
the future (CRM history, labels): ``io.enrich`` (answers, domain and name enrichment from ``companies.csv``),
``io.flag_bots_and_duplicates`` (the flags ``io.clean`` drops on), ``FeatureSpec.featurise``,
``FeatureSpec.design`` / ``value_design``, ``model.predict``, ``value.predict_deal_value``,
``value.expected_value`` and ``FittedValueTransform.apply``. A lead taken from the training data therefore
gets bit-for-bit the score the batch run gave it (``tests/test_scoring.py``).

Input is a frame with ``historical_leads.csv`` columns (``lead_from_form`` builds one from form fields, so the
app never assembles the schema). Absent columns are blank; values are coerced to the training file's dtypes,
except that a text field stays text in a column that was blank on every training lead (read as float64).

Rulings:

- **Unknown categorical levels raise** ``UnknownLevelError`` listing the leads, the values and the allowed
  levels (``ModelBundle.levels``), rather than mapping them to a missing level: a level the model never saw has
  no coefficient, and guessing one would silently misprice the lead. For the legacy feature set the allowed
  levels are those seen in training (its design is data-driven).
- **Extras of the generic feature set** (Phase 10, ADR 0024): a bundle trained with ``--feature-set generic`` also
  takes one raw column ``x_<name>`` per declared extra (``emva.ingest.convert.convert_leads`` adds them; an absent one
  is blank, i.e. ``missing``). They are encoded with the bundle's frozen ``GenericEncoder``: an unseen categorical
  value is ``other`` (a defined level, so it is not refused, amending ADR 0009 for extras only), a number outside the
  training range falls into the edge bin, and a non-number in a numeric extra raises ``ValueError``. Like the
  ``historical_leads.csv`` columns the models do not read, ``x_`` columns a bundle does not use are accepted and not
  read (a v2 bundle scores ``convert_leads`` output of a mapping that declares features).
- **Bots and duplicates are flagged, not dropped** (``is_bot``, ``is_duplicate``), and still scored. The
  duplicate check sees only the batch passed in: a lead scored alone is never a duplicate, even if its email is
  in the training history. In the batch run such leads were dropped before scoring, so they have no batch score.
"""
from __future__ import annotations

import json
import numbers
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from emva.constants import ANSWER_KEYS, CHANNEL_SOURCES, LEAD_ADS_MEDIUM, MISSING, V2_LEVELS, WEEKDAYS
from emva.feature_spec import feature_spec
from emva.generic import EXTRA_PREFIX
from emva.io import enrich, flag_bots_and_duplicates, read_companies
from emva.model import coefficients, intercept_row, predict, split_design_column
from emva.persist import ModelBundle
from emva.value import expected_value, predict_deal_value


class UnknownLevelError(ValueError):
    """A lead has a categorical feature value the saved models were not trained with."""


class FieldType(str, Enum):
    """A form field's type. The one place that knows how each type is checked and coerced (``FieldSpec`` delegates
    here; the app's form uses the same methods)."""

    STR = "str"
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"

    @property
    def kind(self) -> str:
        """How the type is named in an error message."""
        return {"str": "a string", "float": "a number", "int": "an integer", "bool": "a bool"}[self.value]

    @property
    def is_numeric(self) -> bool:
        """True for ``FLOAT`` and ``INT``."""
        return self in (FieldType.FLOAT, FieldType.INT)

    def accepts(self, v: object) -> bool:
        """True when non-blank ``v`` is of this type (an integral float counts as an int; a bool is not a number)."""
        is_bool = isinstance(v, (bool, np.bool_))
        if self is FieldType.STR:
            return isinstance(v, str)
        if self is FieldType.BOOL:
            return is_bool
        if self is FieldType.FLOAT:
            return isinstance(v, numbers.Real) and not is_bool
        return (isinstance(v, numbers.Integral) and not is_bool) or (isinstance(v, float) and v.is_integer())

    def coerce(self, v: object) -> object:
        """A widget or CSV value in the form ``lead_from_form`` expects: blanks (``is_blank``, and whitespace in a
        numeric field) as None, bools as ``bool``, integral numbers as ``int`` for ``INT``, numbers as ``float`` for
        ``FLOAT``, anything else for ``STR`` as text (not stripped: text features read it as typed)."""
        if is_blank(v) or (self.is_numeric and isinstance(v, str) and not v.strip()):
            return None
        if self is FieldType.STR:
            return v if isinstance(v, str) else str(v)
        if self is FieldType.BOOL:
            return bool(v)
        if self is FieldType.INT and isinstance(v, numbers.Real) and float(v).is_integer():
            return int(v)
        if self is FieldType.FLOAT and isinstance(v, numbers.Real):
            return float(v)
        return v


@dataclass(frozen=True)
class FieldSpec:
    """One input the app collects for a lead.

    ``dtype`` is a ``FieldType``; ``source`` is ``"answers"`` (a key of the ``answers`` JSON) or ``"column"`` (a
    top-level ``historical_leads.csv`` column); ``allowed`` lists the valid values of a categorical field (None =
    free input); ``required`` fields must be present and non-blank.
    """

    name: str
    dtype: FieldType
    source: str
    description: str
    allowed: tuple[str, ...] | None = None
    required: bool = False

    def check(self, v: object) -> None:
        """Raise ``ValueError`` if non-blank ``v`` does not fit this field's type or allowed values."""
        if not self.dtype.accepts(v):
            raise ValueError(f"form field {self.name!r} must be {self.dtype.kind}, got {v!r}")
        if self.allowed is not None and v not in self.allowed and v != "":
            raise ValueError(f"form field {self.name!r} has value {v!r}; allowed: {list(self.allowed)}")

    def coerce(self, v: object) -> object:
        """``FieldType.coerce`` for this field."""
        return self.dtype.coerce(v)


S, F, I, B = FieldType.STR, FieldType.FLOAT, FieldType.INT, FieldType.BOOL
# Top-level columns the scoring path reads. ``features.SESSION_INPUTS`` and ``landing_url`` are the on-site session:
# any of them blank means session_missing (v2 features).
_COLUMN_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("email", S, "column", "Email address (free-mail domain, duplicate check)", required=True),
    FieldSpec("company_name", S, "column", "Company name (name enrichment when the domain is unknown)"),
    FieldSpec("company_domain", S, "column", "Company domain (enrichment join with companies.csv)"),
    FieldSpec("form_variant", S, "column", "Landing-page form variant", V2_LEVELS["form_variant"]),
    FieldSpec("utm_source", S, "column", "UTM source; blank or anything else = organic/direct",
              tuple(s for sources in CHANNEL_SOURCES.values() for s in sources)),
    FieldSpec("utm_medium", S, "column", f"UTM medium; '{LEAD_ADS_MEDIUM}' = Meta lead ads"),
    FieldSpec("utm_term", S, "column", "Search term (Google only; brand vs generic)"),
    FieldSpec("landing_url", S, "column", "Landing page URL; blank = no on-site session"),
    FieldSpec("time_on_page_s", F, "column", "Seconds on the page before submitting (< 15 s is flagged as a bot)"),
    FieldSpec("hesitation_ms", F, "column", "Longest pause while filling the form, in ms"),
    FieldSpec("sessions_before_convert", I, "column", "Sessions before this submission"),
    FieldSpec("viewed_pricing", B, "column", "Viewed the pricing page"),
    FieldSpec("ip_country", S, "column", "Country of the IP address"),
    FieldSpec("is_datacenter_ip", B, "column", "IP address is a data-centre IP"),
    FieldSpec("submitted_weekday", S, "column", "Weekday of submission in the lead's local time", WEEKDAYS),
    FieldSpec("local_submit_hour", I, "column", "Hour of submission in the lead's local time, 0-23"),
    FieldSpec("field_edit_count", I, "column", "Number of form fields edited (legacy features)"),
    FieldSpec("user_agent", S, "column", "Browser user agent ('Headless' is flagged as a bot)"),
)
# Form answers (``constants.ANSWER_KEYS``), keys of the ``answers`` JSON.
_ANSWER_FIELDS: dict[str, FieldSpec] = {f.name: f for f in (
    FieldSpec("country", S, "answers", "Country the lead typed (e.g. UK); compared with ip_country"),
    FieldSpec("what_to_solve", S, "answers",
              "Free text: what they want to solve (scored as specific / vague / boilerplate)"),
    FieldSpec("job_title", S, "answers", "Job title (bucketed into seniority)"),
    FieldSpec("company_size", S, "answers", "Typed company size; used when the company is not found in companies.csv",
              tuple(b for b in V2_LEVELS["band"] if b != MISSING)),
    FieldSpec("budget", S, "answers", "Budget answer (legacy features)",
              tuple(b for b in V2_LEVELS["c_budget"] if b != "not_asked")),
    FieldSpec("timeline", S, "answers", "Timeline answer (legacy features)",
              tuple(t for t in V2_LEVELS["c_timeline"] if t != "not_asked")),
    FieldSpec("company", S, "answers", "Company name typed in the form (name enrichment fallback)"),
)}
del S, F, I, B


def is_blank(value: object) -> bool:
    """True for a missing form value: None, NaN (float or numpy), ``pd.NA``, ``NaT`` or the empty string.

    The one blank test shared by ``lead_from_form`` and the app (pandas 3 gives None, NaN or ``pd.NA`` for a missing
    cell depending on the column dtype). Whitespace-only strings are not blank.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, str):
        return value == ""
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)) and bool(np.isnan(value))


def submit_time_fields() -> list[FieldSpec]:
    """The fields the app must collect to score a lead: every input the scoring path reads, columns first.

    ``lead_id`` and ``created_at`` are not fields (``lead_from_form`` generates them); only ``email`` is
    required. Fields the models never read (consent, device, pages visited, click IDs, ...) are not listed.
    """
    return [*_COLUMN_FIELDS, *(_ANSWER_FIELDS[k] for k in ANSWER_KEYS)]


def lead_from_form(fields: dict[str, object]) -> pd.DataFrame:
    """A one-row ``historical_leads`` frame (``lead_id`` column) from form fields, ready for ``score_leads``.

    Keys are ``submit_time_fields`` names; None, NaN, absent or ``""`` means blank (a blank answer is kept as
    ``""`` in the JSON, like the historical forms). Answer fields are JSON-encoded into ``answers``; ``lead_id`` is
    a new ``form-<uuid>``; ``created_at`` is now (UTC). Raises ``ValueError`` for an unknown key, a missing
    ``email``, a value of the wrong type or outside a field's allowed values.
    """
    spec = {f.name: f for f in submit_time_fields()}
    unknown = sorted(set(fields) - set(spec))
    if unknown:
        raise ValueError(f"unknown form field(s) {unknown}; valid fields: {list(spec)}")
    # a blank column reads as NaN from historical_leads.csv; a blank answer stays "" in the JSON, as forms send it
    given = {k: v for k, v in fields.items()
             if not is_blank(v) or (v == "" and spec[k].source == "answers")}
    for name, f in spec.items():
        if f.required and not (isinstance(given.get(name), str) and given[name].strip()):
            raise ValueError(f"form field {name!r} is required")
    for k, v in given.items():
        spec[k].check(v)
    row: dict[str, object] = {"lead_id": f"form-{uuid.uuid4().hex}", "created_at": pd.Timestamp.now(tz="UTC"),
                              "answers": json.dumps({k: v for k, v in given.items() if spec[k].source == "answers"})}
    row.update({k: v for k, v in given.items() if spec[k].source == "column"})
    return pd.DataFrame([row])


def _coerce(s: pd.Series, dtype: object) -> pd.Series:
    """``s`` in the training file's ``dtype``; integer and bool columns with blanks become float / object,
    as ``pd.read_csv`` makes them."""
    if isinstance(dtype, pd.DatetimeTZDtype):
        return pd.to_datetime(s, utc=True).astype(dtype)
    if dtype == object:
        return s.astype(object)
    if pd.api.types.is_bool_dtype(dtype):
        return s.astype(object) if s.isna().any() else s.astype(bool)
    if pd.api.types.is_integer_dtype(dtype):
        n = pd.to_numeric(s)
        return n.astype(dtype) if n.notna().all() else n.astype("float64")
    if pd.api.types.is_float_dtype(dtype):
        return pd.to_numeric(s).astype(dtype)
    return s.astype(dtype)


def _as_training_schema(bundle: ModelBundle, leads: pd.DataFrame) -> pd.DataFrame:
    """``leads`` indexed by ``lead_id`` with every training column, in training order and dtypes, then the bundle's raw
    extra columns (``x_<name>``, as given; blank when absent) for a generic-feature-set bundle."""
    if "lead_id" in leads.columns:
        L = leads.set_index("lead_id")
    elif leads.index.name == "lead_id":
        L = leads
    else:
        raise ValueError("leads need a lead_id column (or index)")
    if L.index.duplicated().any():
        raise ValueError(f"duplicate lead_id values: {sorted(L.index[L.index.duplicated()].unique())[:5]}")
    schema = bundle.lead_schema
    extras = () if bundle.extras is None else bundle.extras.raw_columns
    unknown = sorted(c for c in set(L.columns) - set(schema.columns) if not str(c).startswith(EXTRA_PREFIX))
    if unknown:
        raise ValueError(f"unknown columns {unknown}; leads take historical_leads.csv columns and x_<name> extras only")
    if "email" not in L or L.email.isna().any():
        raise ValueError("every lead needs an email")
    blank = pd.Series(None, index=L.index, dtype=object)
    # a text column blank on every training lead (e.g. a converted dataset that maps no landing_url) reads as float64:
    # keep a form's text in it as text rather than failing to parse it as a number
    text = {f.name for f in submit_time_fields() if f.source == "column" and f.dtype is FieldType.STR}
    X = pd.DataFrame({c: _coerce(L[c] if c in L else blank,
                                 object if c in text and pd.api.types.is_numeric_dtype(dtype) else dtype)
                      for c, dtype in schema.dtypes.items()}, index=L.index)
    X["answers"] = X.answers.fillna("{}")
    for c in extras:
        X[c] = L[c].astype(object) if c in L else blank
    return X


def check_levels(bundle: ModelBundle, X: pd.DataFrame) -> None:
    """Raise ``UnknownLevelError`` if any featurised lead in ``X`` has a model feature value outside
    ``bundle.levels``; the message names the feature, the values, the leads and the allowed levels."""
    for feature, allowed in bundle.levels.items():
        values = X[feature].astype(str)
        bad = ~values.isin(allowed)
        if bad.any():
            raise UnknownLevelError(f"feature {feature!r} has value(s) {sorted(set(values[bad]))} for lead(s) "
                                    f"{list(X.index[bad])[:5]} that the model was not trained with; "
                                    f"allowed: {list(allowed)}")


def prepare(bundle: ModelBundle, leads: pd.DataFrame, data: str | Path
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The featurised leads ``X``, their formula design ``D`` and deal-value design ``M`` (bundle column order).

    The shared front half of ``score_leads`` and ``points_breakdown``; public so the path can be audited
    (``D`` and ``M`` of a training lead equal the batch run's rows exactly). Same inputs and errors as
    ``score_leads``.
    """
    spec = feature_spec(bundle.feature_set)
    X = enrich(_as_training_schema(bundle, leads), read_companies(data))
    X = spec.featurise(flag_bots_and_duplicates(X))
    check_levels(bundle, X)
    D = spec.design(X)
    if bundle.extras is not None:
        D = D.join(bundle.extras.design(X))
    # the legacy designs are data-driven: a batch lacks absent levels' columns (zero) and may carry the reference's
    D = D.reindex(columns=list(bundle.design_columns), fill_value=0.0)
    M = spec.value_design(X).reindex(columns=list(bundle.value_columns), fill_value=0.0)
    return X, D, M


def score_leads(bundle: ModelBundle, leads: pd.DataFrame, data: str | Path) -> pd.DataFrame:
    """Score ``leads`` (``historical_leads.csv`` columns, ``lead_id`` column or index) with ``bundle``.

    ``data`` is a dataset directory holding ``companies.csv`` (the enrichment join; normally the training
    data's). Returns, indexed by ``lead_id`` in input order: ``p_formula``, ``deal_value_hat``,
    ``value_formula`` (``p × deal value × bundle.margin``), ``value_at_submit`` (NaN when the bundle was trained
    with legacy labels), ``is_bot`` and ``is_duplicate`` (within this batch only; see the module docstring).
    Raises ``UnknownLevelError`` for a categorical value the model was not trained with and ``ValueError``
    for malformed input (no ``lead_id`` or ``email``, unknown columns, duplicate ids).
    """
    X, D, M = prepare(bundle, leads, data)
    out = pd.DataFrame(index=X.index)
    out["p_formula"] = predict(bundle.model, D)
    out["deal_value_hat"] = predict_deal_value(bundle.deal_value, M)
    out["value_formula"] = expected_value(out.p_formula, out.deal_value_hat, bundle.margin)
    vt = bundle.value_transform
    out["value_at_submit"] = np.nan if vt is None else vt.apply(out.value_formula)
    out["is_bot"] = X.bot.to_numpy(dtype=bool)
    out["is_duplicate"] = X.dup.to_numpy(dtype=bool)
    return out


def points_breakdown(bundle: ModelBundle, leads: pd.DataFrame, data: str | Path) -> pd.DataFrame:
    """Why each lead scored what it did: the intercept plus one row per active design column.

    Returns columns ``lead_id``, ``feature``, ``level``, ``log_odds``, ``odds_multiplier``, ``points`` (unrounded
    ``model.coefficients``; ``weights.csv`` is the same table rounded), leads in input order, each lead's
    intercept row (feature ``emva.model.INTERCEPT``, level ``""``) first and then its active columns in design order.
    A feature at its reference level has no row (it adds 0). Per lead, ``log_odds`` sums to
    ``logit(p_formula)`` up to floating-point rounding. Same inputs and errors as ``score_leads``.
    """
    _, D, _ = prepare(bundle, leads, data)
    coef = coefficients(bundle.model, D.columns)
    icpt = intercept_row(bundle.model)
    parts = []
    for lead_id, row in D.iterrows():
        t = pd.concat([icpt, coef[row.to_numpy() == 1.0]])
        feature, level = zip(*map(split_design_column, t.index))
        parts.append(t.reset_index(drop=True).assign(lead_id=lead_id, feature=list(feature), level=list(level)))
    return pd.concat(parts, ignore_index=True)[["lead_id", "feature", "level", "log_odds", "odds_multiplier",
                                                "points"]]


__all__ = ["FieldSpec", "FieldType", "UnknownLevelError", "check_levels", "is_blank", "lead_from_form", "points_breakdown", "prepare",
           "score_leads", "submit_time_fields"]
