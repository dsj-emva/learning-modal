"""Deterministic conversion of foreign CSVs into EMVA's five-file format plus ``dataset.json``, driven by a confirmed
``DatasetMapping``. No LLM is involved here, and nothing is guessed: every value comes from a mapped column, a declared
operation or a documented derived rule below, and anything the mapping does not cover is refused with a
``ValueError`` listing the offending values.

``convert(frames, mapping)`` returns ``{file name: bytes}`` in the form ``app.validation.validate_files`` and the app's
dataset store take:

- ``historical_leads.csv``: the full v1 header in v1 column order (``mapping.LEAD_COLUMNS``). Mapped columns carry the
  source values; unmapped columns are blank, so the v2 features' ``session_missing`` / ``enrichment_missing``
  indicators fire instead of reference levels being invented. ``answers`` is a JSON object of the mapped
  ``answers.<key>`` values (blank answers omitted). Values are checked against the lead fields' types and form options
  (``emva.scoring.submit_time_fields``): a categorical value outside the options, a non-boolean in a True/False
  column or a non-number in a numeric column is refused (use a value map to translate source values).
- Derived, never invented: when ``email`` is unmapped (or blank on a row) it is ``<lead_id>@unmapped.invalid``
  (``.invalid`` is reserved and never delivers). The pipeline needs an email on every lead (duplicate key); this one is
  unique per lead, so no lead is dropped as a duplicate, and it reads as a business address, so ``email=free`` stays 0
  (listed as unfilled in the coverage table). ``form_variant`` unmapped = ``DEFAULT_FORM_VARIANT`` ("A", the
  reference level: one form) on every lead, because the v2 design refuses a blank variant (ADR 0009); a mapped
  ``form_variant`` with blanks is refused. ``lead_id`` unmapped = ``<mapping name>-000001``, ... in the primary
  source's row order (rows dropped for a blank ``created_at`` keep their numbers out).
- ``crm_history.csv``: per lead a ``New`` row at ``created_at``; if the final stage is not New, a ``Contacted`` row at
  ``contacted_at`` (when mapped and present); then the final stage: ``Won`` at ``won_at`` (with the deal value),
  ``Lost`` at ``close_at`` (or at ``as_of`` when ``lost_without_close = "as_of"``), an open stage at ``contacted_at``
  else ``created_at``. Rows of one lead are strictly ordered in time: a row that would not come after the previous one
  is moved to one second after it (the pipeline sorts CRM rows with an unstable sort, so ties could reorder);
  ``dataset.json`` counts rows moved off a tie (``crm_ties_separated``, e.g. ``contacted_at`` = ``created_at``) and rows
  that were earlier than the previous one (``crm_times_reordered``, e.g. a win dated before the lead). Every event
  must be at or before ``as_of`` (a later event means
  ``as_of`` is wrong: refused). A non-positive deal value is left blank (``log`` of the value model is undefined) and
  counted.
- ``companies.csv`` and ``people.csv``: header only (company and person columns are not mapped).
- ``dataset.json``: ``emva_dataset_format``, ``mapping``, ``as_of``, ``test_from``, ``source_url``, ``licence``, row
  counts, the derived-value counts above, ``mapped_targets``, and ``coverage``: one entry per v2 design column (the 39
  of ``emva.design.fixed_design`` over ``emva.features.CATS_V2``) with its feature, the mapped inputs feeding it, the
  number of cleaned leads with a 1, and a status: ``data`` (varies across leads), ``constant`` (inputs mapped, but no
  variation) or ``unfilled`` (no input mapped).
"""
from __future__ import annotations

import calendar
import io
import json
import operator
from functools import reduce

import numpy as np
import pandas as pd

from emva.constants import ANSWER_KEYS
from emva.dataset_meta import DATASET_META_FILE, FORMAT_KEY, FORMAT_VERSION, parse_as_of
from emva.design import fixed_design
from emva.eval.evaluation_only import is_evaluation_only
from emva.features import CATS_V2, SESSION_INPUTS, add_features_v2
from emva.ingest.mapping import (
    ANSWER_PREFIX,
    IGNORE,
    LEAD_COLUMNS,
    COLUMN_OP,
    DatasetMapping,
    Expr,
    check_confirmed,
)
from emva.io import add_crm_outcomes, clean, enrich, read_leads
from emva.scoring import FieldType, submit_time_fields

LEADS_FILE, CRM_FILE, COMPANIES_FILE, PEOPLE_FILE = (
    "historical_leads.csv", "crm_history.csv", "companies.csv", "people.csv")
CRM_COLUMNS: tuple[str, ...] = ("lead_id", "stage", "comment", "deal_value", "changed_at")
COMPANIES_COLUMNS: tuple[str, ...] = (
    "company_name", "domain", "sector", "employee_band", "country", "hq_city", "annual_revenue_band", "founded_year",
    "funding_stage", "crm_platform", "monthly_ad_spend_band", "ad_platforms", "is_hiring", "open_roles")
PEOPLE_COLUMNS: tuple[str, ...] = (
    "email", "age_band", "household_income_band", "country", "region", "city", "job_title", "seniority", "department",
    "years_in_role", "is_homeowner", "match_confidence")
# Domain of derived placeholder emails (RFC 2606: .invalid never resolves).
PLACEHOLDER_DOMAIN: str = "unmapped.invalid"
# form_variant of every lead when the mapping does not map it: the v2 reference level (one form).
DEFAULT_FORM_VARIANT: str = "A"
TIMESTAMP_FORMAT: str = "%Y-%m-%dT%H:%M:%SZ"
MIN_STEP: pd.Timedelta = pd.Timedelta(seconds=1)
MAX_LISTED: int = 5

_FIELDS = {f.name if f.source == "column" else ANSWER_PREFIX + f.name: f for f in submit_time_fields()}
# historical_leads columns the scoring path does not read but whose v1 values are True/False or numbers.
BOOL_TARGETS: frozenset[str] = frozenset({*(n for n, f in _FIELDS.items() if f.dtype is FieldType.BOOL),
                                          "consent", "pasted_text", "returned_visitor"})
INT_TARGETS: frozenset[str] = frozenset({*(n for n, f in _FIELDS.items() if f.dtype is FieldType.INT), "pages_visited",
                                         "days_since_first_visit"})
FLOAT_TARGETS: frozenset[str] = frozenset({*(n for n, f in _FIELDS.items() if f.dtype is FieldType.FLOAT),
                                           "scroll_depth_pct"})
ALLOWED_VALUES: dict[str, tuple[str, ...]] = {n: f.allowed for n, f in _FIELDS.items() if f.allowed}
HOUR_RANGE: tuple[int, int] = (0, 23)

_ENRICHMENT_INPUTS = ("company_domain", "company_name", "answers.company")
# v2 feature -> the schema inputs it is computed from (for the coverage table).
FEATURE_INPUTS: dict[str, tuple[str, ...]] = {
    "channel": ("utm_source", "utm_medium"),
    "form_variant": ("form_variant",),
    "session_missing": ("landing_url", *SESSION_INPUTS),
    "enrichment_missing": _ENRICHMENT_INPUTS,
    "band": (*_ENRICHMENT_INPUTS, "answers.company_size"),
    "email": ("email",),
    "text": ("answers.what_to_solve",),
    "seniority": ("answers.job_title",),
    "spend": _ENRICHMENT_INPUTS, "crm": _ENRICHMENT_INPUTS, "hiring": _ENRICHMENT_INPUTS,
    "time_on_page": ("time_on_page_s",),
    "hesitation_90s": ("hesitation_ms",),
    "sessions_3plus": ("sessions_before_convert",),
    "viewed_pricing": ("viewed_pricing",),
    "search_term": ("utm_term", "utm_source", "utm_medium"),
    "business_hours": ("submitted_weekday", "local_submit_hour"),
    "ip_country": ("ip_country", "answers.country"),
}
_MONTHS = {**{m.lower(): i for i, m in enumerate(calendar.month_name) if m},
           **{m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}}


def frames_from_bytes(files: dict[str, bytes]) -> dict[str, pd.DataFrame]:
    """Raw CSVs (file name -> bytes) as frames with every cell read as text (the form ``convert`` expects).

    pandas' default missing markers (empty, ``NA``, ``NULL``, ``NaN``, ...) become blanks. Raises ``ValueError``
    for an evaluation-only ground-truth file name (``emva.eval.evaluation_only``; ground rule 2) without parsing it.
    """
    refused = sorted(n for n in files if is_evaluation_only(n))
    if refused:
        raise ValueError(f"refused {refused}: evaluation-only ground-truth files are never converted (ground rule 2)")
    return {name: pd.read_csv(io.BytesIO(content), dtype=str) for name, content in files.items()}


def _examples(values: pd.Series) -> list[str]:
    """Up to ``MAX_LISTED`` distinct values, sorted, as strings."""
    return sorted(map(str, values.dropna().unique()))[:MAX_LISTED]


def _text(df: pd.DataFrame) -> pd.DataFrame:
    """Every cell as stripped text, blanks (missing or whitespace) as missing (test with ``isinstance(v, str)``:
    pandas may hold a missing cell as None or NaN)."""
    return df.map(lambda v: v.strip() or None if isinstance(v, str) else (None if pd.isna(v) else str(v).strip()))


def join_sources(frames: dict[str, pd.DataFrame], mapping: DatasetMapping,
                 files: list[str] | None = None) -> pd.DataFrame:
    """The primary source left-joined with the others on their ``join_on`` columns, columns named ``<source>.<col>``.

    ``files`` limits the join to those source files (the primary is always read); None joins every source. One row per
    primary row, in its order. Raises ``ValueError`` for a missing file, a missing join column or a join key that
    repeats in a joined source (a lead would match several rows).
    """
    primary = mapping.sources[0]
    wanted = [s for s in mapping.sources if files is None or s is primary or s.file in files]
    missing = [s.file for s in wanted if s.file not in frames]
    if missing:
        raise ValueError(f"missing source file(s) {missing}; got {sorted(frames)}")
    out = _text(frames[primary.file]).add_prefix(primary.name + ".").reset_index(drop=True)
    for s in wanted[1:]:
        right = _text(frames[s.file])
        for file, df, name in ((primary.file, out, f"{primary.name}.{s.join_on}"), (s.file, right, s.join_on)):
            if name not in df:
                raise ValueError(f"join column {s.join_on!r} is not in source file {file}")
        keys = right[s.join_on]
        dup = keys.notna() & keys.duplicated(keep=False)
        if dup.any():
            raise ValueError(f"{s.file}: join key {s.join_on!r} repeats, e.g. {_examples(keys[dup])}")
        right = right[keys.notna()].add_prefix(s.name + ".")
        out = out.merge(right, how="left", left_on=f"{primary.name}.{s.join_on}", right_on=f"{s.name}.{s.join_on}",
                        validate="many_to_one")
    return out


def _dates(s: pd.Series, what: str) -> pd.Series:
    """``s`` as UTC timestamps (ISO 8601 text, or already datetimes); non-blank values that do not parse raise."""
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s
    parsed = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
    bad = parsed.isna() & s.notna()
    if bad.any():
        raise ValueError(f"{what}: {int(bad.sum())} value(s) are not ISO dates, e.g. {_examples(s[bad])}")
    return parsed


def _numbers(s: pd.Series, what: str) -> pd.Series:
    """``s`` as floats; non-blank values that are not numbers raise."""
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.astype(float)
    parsed = pd.to_numeric(s, errors="coerce")
    bad = parsed.isna() & s.notna()
    if bad.any():
        raise ValueError(f"{what}: {int(bad.sum())} value(s) are not numbers, e.g. {_examples(s[bad])}")
    return parsed.astype(float)


def _months(s: pd.Series, what: str) -> pd.Series:
    """Month numbers from numbers or English month names (full or three letters, any case)."""
    as_num = pd.to_numeric(s, errors="coerce")
    named = s.map(lambda v: _MONTHS.get(v.lower()) if isinstance(v, str) else None)
    out = as_num.fillna(pd.to_numeric(named, errors="coerce"))
    bad = s.notna() & (out.isna() | ~out.between(1, 12))
    if bad.any():
        raise ValueError(f"{what}: {int(bad.sum())} value(s) are not months, e.g. {_examples(s[bad])}")
    return out


def evaluate(expr: Expr, raw: pd.DataFrame, what: str = "expression") -> pd.Series:
    """The value of ``expr`` on every row of the joined frame ``raw`` (``join_sources``); blanks stay blank.

    A column reference gives its text; ``date_from_parts`` and ``minus_days`` give UTC timestamps; ``product`` and
    ``sum`` give floats (any blank argument makes the row blank). Raises ``ValueError`` for an unknown column or a
    value an operation cannot read (listed).
    """
    if expr.op == COLUMN_OP:
        if expr.column not in raw:
            raise ValueError(f"{what}: column {expr.column!r} is not in the source files")
        return raw[expr.column]
    args = [evaluate(a, raw, f"{what} ({expr.op} argument {i + 1})") for i, a in enumerate(expr.args)]
    if expr.op == "date_from_parts":
        y, m, d = (_numbers(args[0], f"{what} year"), _months(args[1], f"{what} month"),
                   _numbers(args[2], f"{what} day"))
        present = y.notna() & m.notna() & d.notna()
        parts = pd.DataFrame({"year": y.where(present, 2000), "month": m.where(present, 1), "day": d.where(present, 1)})
        dates = pd.to_datetime(parts, errors="coerce")
        bad = present & dates.isna()
        if bad.any():
            raise ValueError(f"{what}: {int(bad.sum())} impossible date(s), e.g. row {int(np.flatnonzero(bad)[0]) + 1}")
        return dates.where(present).dt.tz_localize("UTC")
    if expr.op == "minus_days":
        return _dates(args[0], what) - pd.to_timedelta(_numbers(args[1], f"{what} days"), unit="D")
    nums = [_numbers(a, what) for a in args]
    return reduce(operator.mul if expr.op == "product" else operator.add, nums)


def _final_stage(raw: pd.DataFrame, mapping: DatasetMapping) -> pd.Series:
    """The canonical final CRM stage of every row; raw values the outcome mapping does not cover raise."""
    o = mapping.outcome
    v = evaluate(Expr.col(o.column), raw, "outcome")
    key = v.fillna("")
    if o.kind == "presence":
        return pd.Series(np.where(v.notna(), "Won", "Lost"), index=raw.index)
    table = dict(o.stage_map) if o.kind == "stage" else {**dict.fromkeys(o.won_values, "Won"),
                                                         **dict.fromkeys(o.lost_values, "Lost")}
    unmapped = ~key.isin(list(table))
    if unmapped.any():
        shown = sorted(set(key[unmapped].map(lambda x: x or "<blank>")))
        raise ValueError(f"outcome column {o.column!r} has value(s) the mapping does not cover: {shown}; add them to "
                         f"{'stage_map' if o.kind == 'stage' else 'won_values / lost_values'}")
    return key.map(table)


def _field_values(raw: pd.DataFrame, mapping: DatasetMapping) -> dict[str, pd.Series]:
    """Target column -> text values from the field rows (copies and value maps), checked against ``_check_target``."""
    out: dict[str, pd.Series] = {}
    for f in mapping.fields:
        if f.target == IGNORE:
            continue
        src = evaluate(Expr.col(f.source), raw, f"field {f.source}")
        if f.value_map is None:
            out[f.target] = src
            continue
        unmapped = src.notna() & ~src.isin(list(f.value_map))
        if unmapped.any():
            raise ValueError(f"field {f.source!r}: value(s) {_examples(src[unmapped])} are not in its value_map; "
                             "list every value (map one to {} to set nothing)")
        for t in f.targets():
            out[t] = src.map(lambda v, t=t: f.value_map[v].get(t) if isinstance(v, str) else None)
    return {t: _check_target(t, s) for t, s in out.items()}


def _check_target(target: str, s: pd.Series) -> pd.Series:
    """``s`` in the target column's v1 form: booleans as ``True`` / ``False``, integers without a decimal point;
    raises ``ValueError`` for values outside the form options, non-booleans, non-numbers or hours outside 0-23."""
    present = s.notna()
    if target in ALLOWED_VALUES:
        bad = present & ~s.isin(ALLOWED_VALUES[target])
        if bad.any():
            raise ValueError(f"{target}: value(s) {_examples(s[bad])} are not form options "
                             f"{list(ALLOWED_VALUES[target])}; translate them with a value_map")
    if target in BOOL_TARGETS:
        low = s.str.lower()
        bad = present & ~low.isin(["true", "false"])
        if bad.any():
            raise ValueError(f"{target}: value(s) {_examples(s[bad])} are not True/False; use a value_map")
        return low.map({"true": "True", "false": "False"})
    if target in INT_TARGETS or target in FLOAT_TARGETS:
        n = _numbers(s, target)
        if target in INT_TARGETS:
            bad = present & (n % 1 != 0)
            if bad.any():
                raise ValueError(f"{target}: value(s) {_examples(s[bad])} are not whole numbers")
            if target == "local_submit_hour" and (present & ~n.between(*HOUR_RANGE)).any():
                raise ValueError(f"local_submit_hour: values outside {HOUR_RANGE}")
            return n.map(lambda v: None if np.isnan(v) else str(int(v)))
        return s
    return s


def _listed(mask: pd.Series, ids: pd.Series) -> str:
    """``N lead(s), e.g. [...]`` for the rows where ``mask`` is True."""
    return f"{int(mask.sum())} lead(s), e.g. {list(ids[mask][:MAX_LISTED])}"


def _fmt(ts: pd.Series) -> pd.Series:
    """UTC timestamps as ``2026-05-01T09:30:00Z`` text (None for NaT)."""
    return ts.dt.strftime(TIMESTAMP_FORMAT).where(ts.notna(), None)


def _csv(df: pd.DataFrame) -> bytes:
    """``df`` as CSV bytes without the index (blanks as empty cells)."""
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _lead_rows(raw: pd.DataFrame, mapping: DatasetMapping, lead_id: pd.Series,
               created: pd.Series) -> tuple[pd.DataFrame, int]:
    """The ``historical_leads.csv`` rows (text cells, ``LEAD_COLUMNS`` order) of the joined rows ``raw``, and the
    number of placeholder emails. Reads only the field rows (submit-time data), never the outcome."""
    values = _field_values(raw, mapping)
    email = values.get("email", pd.Series(None, index=raw.index, dtype=object))
    no_email = email.isna()
    email = email.where(~no_email, lead_id.map(lambda i: f"{i}@{PLACEHOLDER_DOMAIN}"))
    leads = pd.DataFrame({c: values.get(c, pd.Series(None, index=raw.index, dtype=object))
                          for c in LEAD_COLUMNS}, index=raw.index)
    leads["lead_id"], leads["created_at"], leads["email"] = lead_id, _fmt(created), email
    if "form_variant" not in values:
        leads["form_variant"] = DEFAULT_FORM_VARIANT
    elif leads.form_variant.isna().any():
        raise ValueError(f"form_variant is blank for {_listed(leads.form_variant.isna(), lead_id)}; the v2 design "
                         "needs a variant on every lead (map the blanks with a value_map)")
    answer_cols = {k: values[ANSWER_PREFIX + k] for k in ANSWER_KEYS if ANSWER_PREFIX + k in values}
    leads["answers"] = [json.dumps({k: s.iloc[i] for k, s in answer_cols.items() if isinstance(s.iloc[i], str)},
                                   ensure_ascii=False) for i in range(len(raw))]
    return leads, int(no_email.sum())


def convert_leads(frames: dict[str, pd.DataFrame], mapping: DatasetMapping) -> pd.DataFrame:
    """New leads in the source format (file name -> frame, as ``frames_from_bytes`` reads them) as ``historical_leads``
    rows ready for ``emva.scoring.score_leads``: one row per primary-source row, a ``lead_id`` column, typed values.

    The same field rules as ``convert`` (one code path, ``_lead_rows``): mapped values checked against the form
    options, unmapped columns blank, ``form_variant`` "A" when unmapped, email placeholder
    ``<lead_id>@unmapped.invalid``.
    Only the submit-time part of the mapping is read (``DatasetMapping.source_columns``): no outcome, won / close /
    contacted date or deal value, so those columns may be absent, and only the source files it names are needed.
    ``lead_id``: the mapped id when its columns are present (blank or repeated ids raise), else ``new-000001``, ... in
    row order. ``created_at``: the mapped time when its columns are present and filled, else the current UTC time
    (like ``emva.scoring.lead_from_form``; the models do not read it). Booleans come back as ``bool``, numbers as
    floats, ``created_at`` as a UTC timestamp. Raises ``ValueError`` for a draft mapping or a value the mapping refuses.
    """
    check_confirmed(mapping)
    cols = mapping.source_columns(submit_time_only=True)
    raw = join_sources(frames, mapping, files=list(cols))
    row_number = pd.Series(np.arange(1, len(raw) + 1), index=raw.index)
    lead = mapping.lead
    if lead.lead_id is not None and all(c in raw for c in lead.lead_id.columns()):
        lead_id = evaluate(lead.lead_id, raw, "lead_id")
        if lead_id.isna().any() or lead_id.duplicated().any():
            raise ValueError("lead_id must be filled and unique on every new lead (or leave its column out)")
    else:
        lead_id = row_number.map(lambda i: f"new-{i:06d}")
    now = pd.Timestamp.now(tz="UTC").floor("s")
    created = pd.Series(now, index=raw.index)
    if all(c in raw for c in lead.created_at.columns()):
        created = _dates(evaluate(lead.created_at, raw, "created_at"), "created_at").fillna(now)
    leads, _ = _lead_rows(raw, mapping, lead_id, created)
    out = leads.astype(object)
    out["created_at"] = created
    for c in BOOL_TARGETS & set(out.columns):
        out[c] = out[c].map({"True": True, "False": False})
    for c in (INT_TARGETS | FLOAT_TARGETS) & set(out.columns):
        out[c] = pd.to_numeric(out[c]).astype(float)
    return out.reset_index(drop=True)


def convert(frames: dict[str, pd.DataFrame], mapping: DatasetMapping) -> dict[str, bytes]:
    """Convert raw source ``frames`` (file name -> frame, cells as text: ``frames_from_bytes``) with ``mapping``.

    Returns the five files plus ``dataset.json`` (module docstring). Raises ``ValueError`` when the mapping is a draft
    (``outcome_confirmed = false``), and for anything the mapping does not cover: a missing file or column, an outcome
    value not mapped, a blank ``created_at`` (unless ``drop_rows_without_created_at``), a blank or repeated lead id, a
    Won lead without ``won_at``, a Lost lead without ``close_at`` (unless ``lost_without_close = "as_of"``), an event
    after ``as_of``, or a field value outside its column's type or options.
    """
    check_confirmed(mapping)
    as_of = parse_as_of(mapping.as_of)
    raw = join_sources(frames, mapping)
    row_number = pd.Series(np.arange(1, len(raw) + 1), index=raw.index)
    created = _dates(evaluate(mapping.lead.created_at, raw, "created_at"), "created_at")
    no_created = created.isna()
    if no_created.any() and not mapping.drop_rows_without_created_at:
        raise ValueError(f"created_at is blank on {int(no_created.sum())} row(s) (e.g. rows "
                         f"{list(row_number[no_created][:MAX_LISTED])}); set drop_rows_without_created_at = true "
                         "to drop them")
    keep = ~no_created
    raw, created, row_number = raw[keep].reset_index(drop=True), created[keep].reset_index(drop=True), \
        row_number[keep].reset_index(drop=True)

    if mapping.lead.lead_id is None:
        lead_id = row_number.map(lambda i: f"{mapping.name}-{i:06d}")
    else:
        lead_id = evaluate(mapping.lead.lead_id, raw, "lead_id")
        if lead_id.isna().any():
            raise ValueError(f"lead_id is blank on {int(lead_id.isna().sum())} row(s)")
    dup = lead_id.duplicated(keep=False)
    if dup.any():
        raise ValueError(f"lead_id repeats: {_listed(dup, lead_id)}; each lead needs one row (two rows could give "
                         "a lead two Won CRM rows)")

    stage = _final_stage(raw, mapping)
    lead = mapping.lead
    blank = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    contacted = blank if lead.contacted_at is None else _dates(evaluate(lead.contacted_at, raw, "contacted_at"),
                                                               "contacted_at")
    won_at = _dates(evaluate(lead.won_at, raw, "won_at"), "won_at")
    close_at = blank if lead.close_at is None else _dates(evaluate(lead.close_at, raw, "close_at"), "close_at")
    won, lost = stage == "Won", stage == "Lost"
    if (won & won_at.isna()).any():
        raise ValueError(f"won_at is blank for {_listed(won & won_at.isna(), lead_id)} whose outcome is Won")
    lost_no_close = lost & close_at.isna()
    if lost_no_close.any():
        if mapping.outcome.lost_without_close == "error":
            raise ValueError(f"close_at is blank for {_listed(lost_no_close, lead_id)} whose outcome is Lost; map "
                             "close_at or set lost_without_close = \"as_of\"")
        close_at = close_at.mask(lost_no_close, as_of)
    worked = stage != "New"
    contact_t = contacted.where(worked)
    final_t = pd.Series(np.where(won, won_at, np.where(lost, close_at, contacted.fillna(created))),
                        index=raw.index).astype("datetime64[ns, UTC]").where(worked)
    for name, t in (("created_at", created), ("contacted_at", contact_t), ("the final stage", final_t)):
        late = t > as_of
        if late.any():
            raise ValueError(f"{name} is after as_of {mapping.as_of} for {_listed(late, lead_id)}; as_of must be "
                             "the snapshot date the outcomes were read at")

    deal_value = pd.Series(np.nan, index=raw.index)
    non_positive = pd.Series(False, index=raw.index)
    if lead.deal_value is not None:
        deal_value = _numbers(evaluate(lead.deal_value, raw, "deal_value"), "deal_value").where(won)
        non_positive = deal_value <= 0
        deal_value = deal_value.where(~non_positive)

    # CRM rows: New, then Contacted (when mapped), then the final stage, strictly increasing per lead
    own_contact = contact_t.notna() & (stage != "Contacted")
    t_contact = contact_t.where(~(contact_t.notna() & (stage == "Contacted")))  # a Contacted final row carries it
    t_contact_adj = t_contact.where(t_contact.isna() | (t_contact >= created + MIN_STEP), created + MIN_STEP)
    prev = t_contact_adj.fillna(created)
    t_final_adj = final_t.where(final_t.isna() | (final_t >= prev + MIN_STEP), prev + MIN_STEP)
    prev_raw = t_contact.fillna(created)
    ties = int((t_contact == created).sum() + (final_t == prev_raw).sum())
    reordered = int((t_contact < created).sum() + (final_t < prev_raw).sum())
    parts = [pd.DataFrame({"lead_id": lead_id, "stage": "New", "deal_value": np.nan, "changed_at": created, "o": 0}),
             pd.DataFrame({"lead_id": lead_id, "stage": "Contacted", "deal_value": np.nan, "changed_at": t_contact_adj,
                           "o": 1})[own_contact],
             pd.DataFrame({"lead_id": lead_id, "stage": stage, "deal_value": deal_value, "changed_at": t_final_adj,
                           "o": 2})[worked]]
    crm = pd.concat(parts).sort_values(["lead_id", "o"], kind="stable")
    crm = crm.assign(comment=None, changed_at=_fmt(crm.changed_at),
                     deal_value=crm.deal_value.map(lambda v: None if np.isnan(v) else f"{v:.2f}"))[list(CRM_COLUMNS)]

    leads, placeholders = _lead_rows(raw, mapping, lead_id, created)
    files = {LEADS_FILE: _csv(leads), CRM_FILE: _csv(crm),
             COMPANIES_FILE: _csv(pd.DataFrame(columns=COMPANIES_COLUMNS)),
             PEOPLE_FILE: _csv(pd.DataFrame(columns=PEOPLE_COLUMNS))}

    mapped = mapping.mapped_targets()
    meta = {FORMAT_KEY: FORMAT_VERSION, "mapping": mapping.name, "as_of": mapping.as_of,
            "test_from": mapping.test_from, "source_url": mapping.source_url, "licence": mapping.licence,
            "leads": len(raw), "rows_dropped_without_created_at": int(no_created.sum()),
            "final_stage_counts": {k: int(v) for k, v in stage.value_counts().sort_index().items()},
            "placeholder_emails": placeholders, "lost_dated_at_as_of": int(lost_no_close.sum()),
            "non_positive_deal_values_blanked": int(non_positive.sum()), "crm_ties_separated": ties,
            "crm_times_reordered": reordered,
            "mapped_targets": mapped, "coverage": coverage(files, mapped)}
    files[DATASET_META_FILE] = (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return files


def coverage(files: dict[str, bytes], mapped_targets: list[str]) -> list[dict[str, object]]:
    """One entry per v2 design column over the converted leads after bot and duplicate removal (``emva.io.clean``):
    ``column``, ``feature``, ``inputs_mapped`` (the feature's ``FEATURE_INPUTS`` among ``mapped_targets``),
    ``leads_with_1``, ``leads`` and ``status`` (``data`` / ``constant`` / ``unfilled``; module docstring)."""
    L = read_leads(io.BytesIO(files[LEADS_FILE]))
    C = pd.read_csv(io.BytesIO(files[CRM_FILE]), parse_dates=["changed_at"])
    CO = pd.read_csv(io.BytesIO(files[COMPANIES_FILE]))
    X = add_features_v2(clean(enrich(add_crm_outcomes(L, C), CO)))
    D = fixed_design(X, CATS_V2)
    rows = []
    for col in D.columns:
        feature = col.split("=", 1)[0]
        inputs = [i for i in FEATURE_INPUTS[feature] if i in mapped_targets]
        ones = int(D[col].sum())
        status = "data" if 0 < ones < len(D) else ("constant" if inputs else "unfilled")
        rows.append({"column": col, "feature": feature, "inputs_mapped": inputs, "leads_with_1": ones,
                     "leads": len(D), "status": status})
    return rows


__all__ = ["COMPANIES_COLUMNS", "CRM_COLUMNS", "FEATURE_INPUTS", "PEOPLE_COLUMNS", "PLACEHOLDER_DOMAIN", "convert",
           "convert_leads", "coverage", "evaluate", "frames_from_bytes", "join_sources"]
