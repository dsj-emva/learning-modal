"""Schema validation of uploaded training files, before they are stored or trained on. Pure Python, no Streamlit.

``validate_files`` checks what the pipeline actually reads, so a dataset that passes does not fail later in
``python -m emva``:

- **historical_leads.csv**: the columns ``emva.scoring.submit_time_fields`` lists (every lead input the models
  read, ``company_name`` optional), plus ``lead_id``, ``created_at``, ``answers`` and ``pages_visited`` (status
  quo). ``lead_id`` unique; ``created_at`` ISO datetimes; ``email`` on every row; ``answers`` a JSON object per
  row whose categorical answers are form options; numbers numeric or blank; booleans True/False or blank;
  ``form_variant`` and ``submitted_weekday`` known levels (the v2 design refuses others, ADR 0009).
- **crm_history.csv** (``emva.io.add_crm_outcomes``): ``lead_id``, ``stage``, ``deal_value``, ``changed_at``;
  stages in ``emva.constants.STAGE`` (any case); ``changed_at`` ISO datetimes; ``deal_value`` numeric or blank;
  at most one Won row per lead; every CRM ``lead_id`` exists in the leads and every lead has a CRM row
  (``emva.labels.label_source`` refuses leads without a final stage).
- **companies.csv** (``emva.io.read_companies`` / ``enrich``): ``company_name``, ``domain`` (unique, any case)
  and ``ENRICHMENT_COLUMNS``; bands, spend bands and sectors in the model's level lists.
- **people.csv**: ``email`` (never read by the model; optional).
- **status_quo_rules.json** (``emva.eval.status_quo``): the keys it reads; optional (without it the status-quo
  benchmark and the standard report are skipped); every lead's ``form_variant`` needs a base value.

Nothing here raises on user input: every problem becomes an ``Issue`` in the returned ``ValidationReport``.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.storage import (
    COMPANIES_FILE,
    CRM_FILE,
    LEADS_FILE,
    PEOPLE_FILE,
    REQUIRED_FILES,
    RULES_FILE,
    TRAINING_FILES,
    StorageError,
    check_file_name,
)
from emva.constants import AS_OF, DEAL_VALUE_LEVELS, ENRICHMENT_COLUMNS, MISSING, STAGE, V2_LEVELS, WEEKDAYS
from emva.eval.status_quo import RULE_FIELDS
from emva.scoring import submit_time_fields

# Fewer leads than this cannot give a train and a test set worth reporting.
MIN_LEADS: int = 200
# At most this many row numbers are listed per issue.
MAX_ROWS_LISTED: int = 10

_LEAD_FIELDS = [f for f in submit_time_fields() if f.source == "column"]
OPTIONAL_LEAD_COLUMNS: frozenset[str] = frozenset({"company_name"})
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    LEADS_FILE: ("lead_id", "created_at", "answers", *(f.name for f in _LEAD_FIELDS
                                                       if f.name not in OPTIONAL_LEAD_COLUMNS), "pages_visited"),
    CRM_FILE: ("lead_id", "stage", "deal_value", "changed_at"),
    COMPANIES_FILE: ("company_name", "domain", *ENRICHMENT_COLUMNS),
    PEOPLE_FILE: ("email",),
}
_NUMERIC_LEAD = tuple(f.name for f in _LEAD_FIELDS if f.dtype in ("float", "int")) + ("pages_visited",)
_BOOL_LEAD = tuple(f.name for f in _LEAD_FIELDS if f.dtype == "bool")
_ANSWER_OPTIONS = {f.name: f.allowed for f in submit_time_fields() if f.source == "answers" and f.allowed}
_BOOL_TEXT = frozenset({"true", "false"})
RULES_KEYS: tuple[str, ...] = ("base_value_by_form", "value_rules", "lead_score")
LEAD_SCORE_KEYS: tuple[str, ...] = ("job_title", "company_size_points", "pages_visited_points", "tiers")
COMPANY_LEVELS: dict[str, tuple[str, ...]] = {
    "employee_band": tuple(b for b in V2_LEVELS["band"] if b != MISSING),
    "monthly_ad_spend_band": V2_LEVELS["spend"],
    "sector": tuple(s for s in DEAL_VALUE_LEVELS["sector"] if s != MISSING),
}


@dataclass(frozen=True)
class Issue:
    """One problem: ``file``, ``column`` (None for a file-level issue), a plain-language ``message`` and up to
    ``MAX_ROWS_LISTED`` 1-based data row numbers (row 1 = the first line after the header)."""

    file: str
    column: str | None
    message: str
    rows: tuple[int, ...] = ()


@dataclass
class ValidationReport:
    """``errors`` block saving and training; ``warnings`` do not; ``summary`` is the preview (rows per file,
    leads, date range, CRM stage counts, won leads)."""

    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when there are no errors."""
        return not self.errors

    def for_file(self, name: str) -> tuple[list[Issue], list[Issue]]:
        """``(errors, warnings)`` about file ``name``."""
        return [i for i in self.errors if i.file == name], [i for i in self.warnings if i.file == name]


def _rows(mask: pd.Series | np.ndarray) -> tuple[int, ...]:
    """1-based data row numbers where ``mask`` is True (positional), at most ``MAX_ROWS_LISTED``."""
    return tuple(int(i) + 1 for i in np.flatnonzero(np.asarray(mask))[:MAX_ROWS_LISTED])


def _n(mask: pd.Series | np.ndarray) -> int:
    """Number of True values."""
    return int(np.asarray(mask).sum())


def _blank(s: pd.Series) -> pd.Series:
    """True where a raw (string-read) cell is missing or whitespace only."""
    return s.isna() | s.astype(str).str.strip().eq("")


def _parse_dates(s: pd.Series) -> pd.Series:
    """ISO 8601 datetimes as UTC timestamps; NaT where a cell does not parse."""
    return pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")


class _Checker:
    """Collects issues for one validation run."""

    def __init__(self) -> None:
        self.report = ValidationReport()

    def error(self, file: str, column: str | None, message: str, mask: pd.Series | np.ndarray | None = None) -> None:
        self.report.errors.append(Issue(file, column, message, () if mask is None else _rows(mask)))

    def warn(self, file: str, column: str | None, message: str, mask: pd.Series | np.ndarray | None = None) -> None:
        self.report.warnings.append(Issue(file, column, message, () if mask is None else _rows(mask)))

    def read_csv(self, name: str, content: bytes) -> pd.DataFrame | None:
        """Parse ``content`` with every cell as text; None (and an error) if it is not a readable CSV."""
        try:
            return pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=True)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as e:
            self.error(name, None, f"not a readable CSV file ({type(e).__name__}: {str(e).splitlines()[0]})")
            return None

    def columns(self, name: str, df: pd.DataFrame) -> bool:
        """Error per missing required column; True when all are present."""
        missing = [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]
        for c in missing:
            self.error(name, c, "required column is missing")
        return not missing

    def dates(self, name: str, df: pd.DataFrame, col: str) -> pd.Series:
        """Error for blank or unparsable datetimes in ``col``; return the parsed column."""
        parsed = _parse_dates(df[col])
        blank = _blank(df[col])
        if blank.any():
            self.error(name, col, f"{_n(blank)} row(s) have no date", blank)
        bad = parsed.isna() & ~blank
        if bad.any():
            self.error(name, col, f"{_n(bad)} value(s) are not ISO dates like 2026-05-01T09:30:00Z "
                                  f"(e.g. {df[col][bad].iloc[0]!r})", bad)
        return parsed

    def numeric(self, name: str, df: pd.DataFrame, col: str) -> None:
        """Error for non-blank cells of ``col`` that are not numbers."""
        bad = pd.to_numeric(df[col], errors="coerce").isna() & ~_blank(df[col])
        if bad.any():
            self.error(name, col, f"{_n(bad)} value(s) are not numbers (e.g. {df[col][bad].iloc[0]!r}); "
                                  "leave the cell blank when unknown", bad)

    def levels(self, name: str, df: pd.DataFrame, col: str, allowed: tuple[str, ...], what: str) -> None:
        """Error for non-blank cells of ``col`` outside ``allowed``."""
        bad = ~df[col].isin(allowed) & ~_blank(df[col])
        if bad.any():
            seen = sorted(df[col][bad].unique())[:5]
            self.error(name, col, f"unknown {what} {seen}; allowed: {list(allowed)}", bad)


def _check_leads(c: _Checker, L: pd.DataFrame, rules: dict | None) -> None:
    """Row-level checks of ``historical_leads.csv`` (all required columns present)."""
    f = LEADS_FILE
    blank_id = _blank(L.lead_id)
    if blank_id.any():
        c.error(f, "lead_id", f"{_n(blank_id)} row(s) have no lead_id", blank_id)
    dup = L.lead_id.duplicated(keep="first") & ~blank_id
    if dup.any():
        c.error(f, "lead_id", f"{_n(dup)} duplicate lead_id value(s) (e.g. {L.lead_id[dup].iloc[0]!r})", dup)
    created = c.dates(f, L, "created_at")
    no_email = _blank(L.email)
    if no_email.any():
        c.error(f, "email", f"{_n(no_email)} row(s) have no email address", no_email)
    for col in _NUMERIC_LEAD:
        c.numeric(f, L, col)
    for col in _BOOL_LEAD:
        bad = ~L[col].astype(str).str.strip().str.lower().isin(_BOOL_TEXT) & ~_blank(L[col])
        if bad.any():
            c.error(f, col, f"{_n(bad)} value(s) are not True/False (e.g. {L[col][bad].iloc[0]!r})", bad)
    c.levels(f, L, "form_variant", V2_LEVELS["form_variant"], "form variant(s)")
    c.levels(f, L, "submitted_weekday", WEEKDAYS, "weekday(s)")
    hour = pd.to_numeric(L.local_submit_hour, errors="coerce")
    odd_hour = hour.notna() & ~hour.between(0, 23)
    if odd_hour.any():
        c.error(f, "local_submit_hour", f"{_n(odd_hour)} hour(s) outside 0-23", odd_hour)

    parsed: list[object] = []
    bad_json = np.zeros(len(L), dtype=bool)
    for i, text in enumerate(L.answers):
        try:
            d = json.loads(text) if isinstance(text, str) and text.strip() else {}
        except json.JSONDecodeError:
            d = None
        if not isinstance(d, dict):
            bad_json[i] = True
            d = {}
        parsed.append(d)
    if bad_json.any():
        c.error(f, "answers", f"{_n(bad_json)} row(s) are not a JSON object like "
                              '{"country": "UK", "job_title": "CMO"}', bad_json)
    for key, allowed in _ANSWER_OPTIONS.items():
        vals = pd.Series([d.get(key) for d in parsed], dtype=object)
        present = vals.map(lambda v: v is not None)
        bad = present & ~vals.isin(allowed) & ~((key == "company_size") & vals.eq(""))
        if bad.any():
            c.error(f, "answers", f"answer '{key}' has value(s) outside the form options "
                                  f"{sorted(map(str, vals[bad].unique()))[:5]}; allowed: {list(allowed)}", bad)
    if rules is not None and isinstance(rules.get("base_value_by_form"), dict):
        no_base = ~L.form_variant.isin(list(rules["base_value_by_form"])) & ~_blank(L.form_variant)
        if no_base.any():
            c.error(f, "form_variant", f"form variant(s) {sorted(L.form_variant[no_base].unique())} have no base "
                                       f"value in {RULES_FILE}", no_base)
    future = created > AS_OF
    if future.any():
        c.warn(f, "created_at", f"{_n(future)} lead(s) are dated after {AS_OF.date()}, the pipeline's snapshot date; "
                                "their outcomes are treated as unknown", future)
    if len(L) < MIN_LEADS:
        c.error(f, None, f"only {len(L)} leads; training needs at least {MIN_LEADS}")
    if "company_name" not in L.columns:
        c.warn(f, "company_name", "optional column absent: company-name enrichment falls back to the typed answer")


def _check_crm(c: _Checker, C: pd.DataFrame, lead_ids: pd.Series | None) -> None:
    """Row-level checks of ``crm_history.csv`` and its references to the leads."""
    f = CRM_FILE
    stage_n = C.stage.astype(str).str.strip().str.lower().map(STAGE)
    unknown = stage_n.isna()
    if unknown.any():
        c.error(f, "stage", f"unknown CRM stage(s) {sorted(C.stage[unknown].astype(str).unique())[:5]}; known "
                            f"(any case): {sorted(set(STAGE))}", unknown)
    c.dates(f, C, "changed_at")
    c.numeric(f, C, "deal_value")
    won = stage_n.eq("Won")
    won_ids = C.lead_id[won]
    multi = won & C.lead_id.isin(won_ids[won_ids.duplicated()])
    if multi.any():
        c.error(f, "stage", f"{C.lead_id[multi].nunique()} lead(s) have more than one Won row "
                            f"(e.g. {C.lead_id[multi].iloc[0]!r}); the deal value would be ambiguous", multi)
    no_value = won & _blank(C.deal_value)
    if no_value.any():
        c.warn(f, "deal_value", f"{_n(no_value)} Won row(s) have no deal value; they count as £0 revenue", no_value)
    if lead_ids is not None:
        orphan = ~C.lead_id.isin(set(lead_ids))
        if orphan.any():
            c.error(f, "lead_id", f"{_n(orphan)} CRM row(s) refer to lead_id(s) not in {LEADS_FILE} "
                                  f"(e.g. {C.lead_id[orphan].iloc[0]!r})", orphan)
        no_crm = ~lead_ids.isin(set(C.lead_id))
        if no_crm.any():
            c.report.errors.append(Issue(LEADS_FILE, "lead_id", f"{_n(no_crm)} lead(s) have no row in {CRM_FILE}; "
                                         "every lead needs at least its New row", _rows(no_crm)))


def _check_companies(c: _Checker, CO: pd.DataFrame) -> None:
    """Row-level checks of ``companies.csv``."""
    f = COMPANIES_FILE
    dom = CO.domain.astype(str).str.strip().str.lower()
    blank = _blank(CO.domain)
    if blank.any():
        c.error(f, "domain", f"{_n(blank)} row(s) have no domain", blank)
    dup = dom.duplicated(keep="first") & ~blank
    if dup.any():
        c.error(f, "domain", f"{_n(dup)} duplicate domain(s) (e.g. {CO.domain[dup].iloc[0]!r}); each company once",
                dup)
    for col, allowed in COMPANY_LEVELS.items():
        c.levels(f, CO, col, allowed, col.replace("_", " ") + " value(s)")
    bad = ~CO.is_hiring.astype(str).str.strip().str.lower().isin(_BOOL_TEXT) & ~_blank(CO.is_hiring)
    if bad.any():
        c.error(f, "is_hiring", f"{_n(bad)} value(s) are not True/False", bad)


def _check_rules(c: _Checker, content: bytes) -> dict | None:
    """Structure of ``status_quo_rules.json``; the parsed rules, or None when unusable."""
    f = RULES_FILE
    try:
        rules = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        c.error(f, None, f"not valid JSON ({e})")
        return None
    if not isinstance(rules, dict):
        c.error(f, None, "must be a JSON object")
        return None
    for k in RULES_KEYS:
        if k not in rules:
            c.error(f, k, "required key is missing")
    ls = rules.get("lead_score")
    if isinstance(ls, dict):
        for k in LEAD_SCORE_KEYS:
            if k not in ls:
                c.error(f, f"lead_score.{k}", "required key is missing")
    elif "lead_score" in rules:
        c.error(f, "lead_score", "must be an object")
    if "base_value_by_form" in rules and not isinstance(rules["base_value_by_form"], dict):
        c.error(f, "base_value_by_form", "must map form variant -> GBP value")
    for i, rule in enumerate(rules.get("value_rules") or []):
        if not isinstance(rule, dict) or rule.get("field") not in RULE_FIELDS or not {"in", "multiplier"} <= set(rule):
            c.error(f, "value_rules", f"rule {i + 1} needs 'field' (one of {sorted(RULE_FIELDS)}), 'in' and "
                                      "'multiplier'")
    return rules if not [i for i in c.report.errors if i.file == f] else None


def _summary(L: pd.DataFrame | None, C: pd.DataFrame | None, frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Preview numbers: rows per file, leads, created_at range, final CRM stage counts, won leads."""
    out: dict[str, object] = {"rows": {n: len(df) for n, df in frames.items()}}
    if L is not None and "created_at" in L:
        created = _parse_dates(L.created_at)
        out["leads"] = len(L)
        if created.notna().any():
            out["created_from"] = created.min().date().isoformat()
            out["created_to"] = created.max().date().isoformat()
    if C is not None and {"lead_id", "stage", "changed_at"} <= set(C.columns):
        crm = C.assign(stage_n=C.stage.astype(str).str.strip().str.lower().map(STAGE),
                       t=_parse_dates(C.changed_at)).sort_values(["lead_id", "t"], kind="stable")
        final = crm.groupby("lead_id").tail(1).stage_n.fillna("unknown")
        out["final_stage_counts"] = {str(k): int(v) for k, v in final.value_counts().items()}
        out["won_leads"] = int((crm.stage_n == "Won").groupby(crm.lead_id).any().sum())
    return out


def validate_files(files: dict[str, bytes]) -> ValidationReport:
    """Validate uploaded training files (file name -> content); never raises on user input.

    Unknown or ground-truth file names become errors without being parsed. Required files missing, missing
    columns, unparsable values and broken references become errors; things that train but deserve a look
    (Won rows without a value, leads dated after the snapshot, no rules file) become warnings.
    """
    c = _Checker()
    usable: dict[str, bytes] = {}
    for name, content in files.items():
        try:
            check_file_name(name)
            usable[name] = content
        except StorageError as e:
            c.error(Path(name).name, None, str(e))
    for name in REQUIRED_FILES:
        if name not in usable:
            c.error(name, None, "required file is missing")
    if PEOPLE_FILE not in usable:
        c.warn(PEOPLE_FILE, None, "not provided; the model never reads it, so training is unaffected")
    rules = None
    if RULES_FILE in usable:
        rules = _check_rules(c, usable[RULES_FILE])
    else:
        c.warn(RULES_FILE, None, "not provided; the status-quo benchmark and the standard report will be skipped")

    frames: dict[str, pd.DataFrame] = {}
    for name in (LEADS_FILE, CRM_FILE, COMPANIES_FILE, PEOPLE_FILE):
        if name in usable:
            df = c.read_csv(name, usable[name])
            if df is not None:
                frames[name] = df
    ok = {n: c.columns(n, df) for n, df in frames.items()}
    L = frames.get(LEADS_FILE) if ok.get(LEADS_FILE) else None
    C = frames.get(CRM_FILE) if ok.get(CRM_FILE) else None
    if L is not None:
        _check_leads(c, L, rules)
    if C is not None:
        _check_crm(c, C, L.lead_id if L is not None else None)
    if ok.get(COMPANIES_FILE):
        _check_companies(c, frames[COMPANIES_FILE])
    if ok.get(PEOPLE_FILE):
        no_email = _blank(frames[PEOPLE_FILE].email)
        if no_email.any():
            c.warn(PEOPLE_FILE, "email", f"{_n(no_email)} row(s) have no email", no_email)
    c.report.summary = _summary(L, C, frames)
    return c.report


def validate_dir(path: str | Path) -> ValidationReport:
    """``validate_files`` on the training files present in directory ``path`` (other files are not opened)."""
    p = Path(path)
    return validate_files({n: (p / n).read_bytes() for n in TRAINING_FILES if (p / n).exists()})


__all__ = ["Issue", "MIN_LEADS", "REQUIRED_COLUMNS", "ValidationReport", "validate_dir", "validate_files"]
