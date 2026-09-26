"""A dataset mapping: how one foreign data source (1-3 CSV files) becomes EMVA's five-file format.

A mapping is a TOML file per data source (``mappings/*.toml``), read by ``load_mapping`` and written by
``dump_mapping``. It is either a *draft* (``emva.ingest.draft``: an LLM proposed it, ``outcome_confirmed = false``) or
*confirmed* by a person who reviewed it; ``emva.ingest.convert`` refuses a draft. Conversion itself is deterministic
code: the mapping names columns and a small closed set of operations, never code to evaluate.

TOML schema (every key below; unknown keys are refused so typos fail loudly)::

    name = "hotel_bookings"                # slug: letters, digits, _ and -
    as_of = "2017-09-01"                   # snapshot date (ISO; naive = UTC) -> dataset.json (ADR 0020)
    test_from = "2017-01-01"               # frozen train/test boundary (YYYY-MM-DD) -> dataset.json
    source_url = "https://..."
    licence = "CC BY 4.0"
    outcome_confirmed = false              # true only after a person reviewed the outcome mapping
    drop_rows_without_created_at = false   # true: rows with a blank created_at are dropped (counted), else refused

    [[sources]]                            # 1-3 files; the first is the primary (one row per lead)
    name = "b"                             # no dots; columns are referenced as "<source>.<column>"
    file = "hotel_bookings.csv"
    # join_on = "account"                  # sources after the first: left-joined on this column (same name in the
                                           # primary source; must be unique in this source)

    [lead]                                 # expressions (see below); only created_at is required
    lead_id = "b.booking_id"               # optional: absent = "<name>-000001", ... in primary-source row order
    created_at = { op = "minus_days", args = [{ op = "date_from_parts", args = ["b.arrival_date_year",
                   "b.arrival_date_month", "b.arrival_date_day_of_month"] }, "b.lead_time"] }
    contacted_at = "..."                   # optional: a Contacted CRM row at this time (the lead was worked)
    won_at = "..."                         # optional: time of the Won CRM row (absent or blank = close_at)
    close_at = "b.reservation_status_date" # optional: time of the Lost CRM row (and of a Won one without won_at)
    deal_value = { op = "product", args = ["b.adr", { op = "sum", args = ["b.stays_in_weekend_nights",
                   "b.stays_in_week_nights"] }] }

    [outcome]                              # exactly one of three kinds
    kind = "won_flag"                      # "stage" | "won_flag" | "presence"
    column = "b.is_canceled"
    won_values = ["0"]                     # won_flag: raw values meaning Won / Lost ("" = blank)
    lost_values = ["1"]
    lost_without_close = "error"           # a Lost lead with a blank close_at: "error" (refuse) or "as_of"
    # kind = "stage": [outcome.stage_map] "Raw value" = "Won" | "Lost" | "New" | "Contacted" | "Qualified" |
    #                 "Demo booked" | "Proposal" (every raw value in the data must be listed; "" = blank)
    # kind = "presence": a non-blank value means Won, a blank one Lost

    [review]                               # optional reviewer notes on the [lead] and [outcome] choices
    created_at = { reason = "...", confidence = "high" }

    [[fields]]                             # one per source column used; target = a schema column or "ignore"
    source = "b.country"
    target = "answers.country"
    reason = "..."                         # optional, from the draft or the reviewer
    confidence = "high"                    # optional: high | medium | low

    [[fields]]                             # or a value map: each raw value sets one or more targets
    source = "m.origin"
    [fields.value_map]
    paid_search = { utm_source = "google", utm_medium = "cpc" }
    direct_traffic = {}                    # a listed value that sets nothing

Expressions: a string is a column reference ``"<source>.<column>"``; an inline table ``{ op = ..., args = [...] }``
applies one of ``OPS`` to its arguments (each itself an expression):

- ``date_from_parts`` (year, month, day): a date; month is a number or an English month name (full or 3 letters);
- ``minus_days`` (date, days): the date minus a whole or fractional number of days;
- ``product`` (a, b, ...) and ``sum`` (a, b, ...): numbers (two or more arguments).

Targets (``TARGETS``) are the ``historical_leads.csv`` columns other than ``lead_id``, ``created_at`` and ``answers``
(which the converter builds), plus ``answers.<key>`` for each of ``emva.constants.ANSWER_KEYS``. Companies and people
columns are not mapped (the converter writes those files header-only). A target set by two field rows is refused.
"""
from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePath

from emva.constants import ANSWER_KEYS, STAGE
from emva.dataset_meta import check_test_from, parse_as_of
from emva.eval.evaluation_only import is_evaluation_only

# historical_leads.csv header, in file order (the v1 format; the converter writes exactly these columns).
LEAD_COLUMNS: tuple[str, ...] = (
    "lead_id", "created_at", "form_variant", "answers", "email", "phone", "name", "company_name", "company_domain",
    "consent", "utm_source", "utm_medium", "utm_campaign", "utm_content", "fbclid", "fbc", "fbp", "gclid", "gbraid",
    "wbraid", "li_fat_id", "oppref", "meta_lead_id", "landing_url", "time_on_page_s", "fields_edited",
    "field_edit_count", "hesitation_ms", "pasted_text", "returned_visitor", "device", "ip", "user_agent",
    "pages_visited", "utm_term", "referrer", "ip_country", "ip_city", "is_datacenter_ip", "timezone",
    "browser_language", "local_submit_hour", "submitted_weekday", "sessions_before_convert", "days_since_first_visit",
    "viewed_pricing", "scroll_depth_pct")
# Columns the converter builds itself; they cannot be field targets.
BUILT_COLUMNS: tuple[str, ...] = ("lead_id", "created_at", "answers")
ANSWER_PREFIX: str = "answers."
IGNORE: str = "ignore"
TARGETS: tuple[str, ...] = (*(c for c in LEAD_COLUMNS if c not in BUILT_COLUMNS),
                            *(ANSWER_PREFIX + k for k in ANSWER_KEYS))

# Canonical CRM stages (``emva.constants.STAGE`` values), in funnel order.
STAGES: tuple[str, ...] = ("New", "Contacted", "Qualified", "Demo booked", "Proposal", "Won", "Lost")
if set(STAGES) != set(STAGE.values()):
    raise RuntimeError(f"mapping.STAGES {sorted(STAGES)} differ from emva.constants.STAGE {sorted(set(STAGE.values()))}")
OPEN_OR_NEW: tuple[str, ...] = tuple(s for s in STAGES if s not in ("Won", "Lost"))

# A mapping reads 1 to this many source files (the first is the primary one).
MAX_SOURCES: int = 3
COLUMN_OP: str = "column"
# op -> (minimum, maximum) number of arguments; None = no maximum.
OPS: dict[str, tuple[int, int | None]] = {"date_from_parts": (3, 3), "minus_days": (2, 2), "product": (2, None),
                                          "sum": (2, None)}
OUTCOME_KINDS: tuple[str, ...] = ("stage", "won_flag", "presence")
LOST_WITHOUT_CLOSE: tuple[str, ...] = ("error", "as_of")
CONFIDENCES: tuple[str, ...] = ("high", "medium", "low")
LEAD_ROLES: tuple[str, ...] = ("lead_id", "created_at", "contacted_at", "won_at", "close_at", "deal_value")
REVIEW_KEYS: tuple[str, ...] = (*LEAD_ROLES, "outcome")

_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class Expr:
    """A deterministic expression over source columns: ``op == "column"`` reads ``column`` (``"<source>.<col>"``);
    any other ``op`` (one of ``OPS``) applies to ``args``. Build a column reference with ``Expr.col``."""

    op: str
    column: str | None = None
    args: tuple[Expr, ...] = ()

    def __post_init__(self) -> None:
        """Refuse an unknown op, a wrong argument count, or a column reference without ``<source>.``."""
        if self.op == COLUMN_OP:
            if not isinstance(self.column, str) or "." not in self.column or self.args:
                raise ValueError(f"a column reference must be '<source>.<column>', got {self.column!r}")
            return
        if self.op not in OPS:
            raise ValueError(f"unknown op {self.op!r}; allowed: {sorted(OPS)} (or a column reference string)")
        lo, hi = OPS[self.op]
        if self.column is not None or len(self.args) < lo or (hi is not None and len(self.args) > hi):
            want = f"{lo}" if lo == hi else f"at least {lo}"
            raise ValueError(f"op {self.op!r} takes {want} arguments, got {len(self.args)}")

    @staticmethod
    def col(ref: str) -> Expr:
        """A column reference ``"<source>.<column>"``."""
        return Expr(COLUMN_OP, column=ref)

    def columns(self) -> list[str]:
        """Every column this expression reads, in order."""
        return [self.column] if self.op == COLUMN_OP else [c for a in self.args for c in a.columns()]


@dataclass(frozen=True)
class Source:
    """One input CSV: its short ``name`` (used in column references), ``file`` name, and for every source after the
    first the ``join_on`` column (same name in the primary source; unique in this one)."""

    name: str
    file: str
    join_on: str | None = None

    def __post_init__(self) -> None:
        """Refuse a ``file`` that is not a plain ``.csv`` file name (no directory part, no ``..``) and an
        evaluation-only ground-truth file (``emva.eval.evaluation_only``; ground rule 2): the converter reads
        ``<raw dir>/<file>`` and must never reach outside that directory or open ground truth."""
        f = self.file
        if not isinstance(f, str) or not f or PurePath(f).name != f or "\\" in f or f in (".", ".."):
            raise ValueError(f"source {self.name!r}: file {f!r} must be a plain file name (no directory part)")
        if PurePath(f).suffix.lower() != ".csv":
            raise ValueError(f"source {self.name!r}: file {f!r} must be a .csv file")
        if is_evaluation_only(f):
            raise ValueError(f"source {self.name!r}: {f!r} is an evaluation-only ground-truth file; a mapping never "
                             "reads ground truth (ground rule 2)")


@dataclass(frozen=True)
class LeadColumns:
    """Where each lead's id and CRM times come from; see the module docstring. ``lead_id`` None = row numbers;
    ``won_at`` None = a Won lead is dated at its ``close_at``."""

    created_at: Expr
    won_at: Expr | None = None
    lead_id: Expr | None = None
    contacted_at: Expr | None = None
    close_at: Expr | None = None
    deal_value: Expr | None = None


@dataclass(frozen=True)
class Outcome:
    """How a lead's final CRM stage is read: ``kind`` is ``stage`` (``stage_map``: raw value -> canonical stage),
    ``won_flag`` (``won_values`` / ``lost_values``) or ``presence`` (non-blank = Won). ``lost_without_close``: what a
    Lost lead without a ``close_at`` value gets (``error`` refuses, ``as_of`` dates the Lost row at the snapshot)."""

    kind: str
    column: str
    stage_map: Mapping[str, str] = field(default_factory=dict)
    won_values: tuple[str, ...] = ()
    lost_values: tuple[str, ...] = ()
    lost_without_close: str = "error"

    def __post_init__(self) -> None:
        """Refuse an unknown kind, options that do not belong to it, or stages outside ``STAGES``."""
        Expr.col(self.column)
        if self.kind not in OUTCOME_KINDS:
            raise ValueError(f"outcome kind {self.kind!r}; allowed: {list(OUTCOME_KINDS)}")
        if self.lost_without_close not in LOST_WITHOUT_CLOSE:
            raise ValueError(f"lost_without_close {self.lost_without_close!r}; allowed: {list(LOST_WITHOUT_CLOSE)}")
        if self.kind == "stage":
            if not self.stage_map or self.won_values or self.lost_values:
                raise ValueError("outcome kind 'stage' needs a stage_map and no won_values / lost_values")
            bad = sorted(set(self.stage_map.values()) - set(STAGES))
            if bad:
                raise ValueError(f"stage_map maps to unknown stage(s) {bad}; allowed: {list(STAGES)}")
        elif self.kind == "won_flag":
            if self.stage_map or not self.won_values or not self.lost_values:
                raise ValueError("outcome kind 'won_flag' needs won_values and lost_values and no stage_map")
            both = sorted(set(self.won_values) & set(self.lost_values))
            if both:
                raise ValueError(f"value(s) {both} are in both won_values and lost_values")
        elif self.stage_map or self.won_values or self.lost_values:
            raise ValueError("outcome kind 'presence' takes no stage_map, won_values or lost_values")


@dataclass(frozen=True)
class Review:
    """A reviewer note: ``reason`` (free text) and ``confidence`` (``high`` / ``medium`` / ``low`` or None)."""

    reason: str = ""
    confidence: str | None = None

    def __post_init__(self) -> None:
        """Refuse a confidence outside ``CONFIDENCES``."""
        if self.confidence is not None and self.confidence not in CONFIDENCES:
            raise ValueError(f"confidence {self.confidence!r}; allowed: {list(CONFIDENCES)}")


@dataclass(frozen=True)
class FieldMap:
    """One source column used for lead fields: copied to ``target`` (a ``TARGETS`` entry or ``"ignore"``), or through
    ``value_map`` (raw value -> {target: value}; every non-blank raw value in the data must be listed). Exactly one of
    ``target`` and ``value_map``. ``review`` carries the draft's or reviewer's reason and confidence."""

    source: str
    target: str | None = None
    value_map: Mapping[str, Mapping[str, str]] | None = None
    review: Review = Review()

    def __post_init__(self) -> None:
        """Refuse both or neither of ``target`` / ``value_map``, and unknown targets."""
        Expr.col(self.source)
        if (self.target is None) == (self.value_map is None):
            raise ValueError(f"field {self.source!r}: give exactly one of target and value_map")
        for t in self.targets():
            if t not in TARGETS:
                raise ValueError(f"field {self.source!r}: unknown target column {t!r}; allowed: {list(TARGETS)} "
                                 f"or {IGNORE!r}")
        if self.target is not None and self.target != IGNORE and self.target not in TARGETS:
            raise ValueError(f"field {self.source!r}: unknown target column {self.target!r}")

    def targets(self) -> list[str]:
        """The schema columns this row can set (none for ``ignore``), in first-seen order."""
        if self.value_map is None:
            return [] if self.target == IGNORE else [self.target]
        return list(dict.fromkeys(t for sets in self.value_map.values() for t in sets))


@dataclass(frozen=True)
class DatasetMapping:
    """A complete mapping (module docstring). Constructing one validates it (``ValueError`` on anything unknown or
    inconsistent); ``outcome_confirmed`` is checked by ``emva.ingest.convert``, not here, so drafts can be built."""

    name: str
    sources: tuple[Source, ...]
    lead: LeadColumns
    outcome: Outcome
    as_of: str
    test_from: str
    fields: tuple[FieldMap, ...] = ()
    source_url: str = ""
    licence: str = ""
    outcome_confirmed: bool = False
    drop_rows_without_created_at: bool = False
    review: Mapping[str, Review] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate names, sources and joins, column references, targets, dates and review keys."""
        if not _SLUG.match(self.name):
            raise ValueError(f"mapping name {self.name!r} must be letters, digits, '_' or '-'")
        if not 1 <= len(self.sources) <= MAX_SOURCES:
            raise ValueError(f"a mapping has 1 to {MAX_SOURCES} sources, got {len(self.sources)}")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names) or any(not _SLUG.match(n) for n in names):
            raise ValueError(f"source names must be unique slugs without dots, got {names}")
        if self.sources[0].join_on is not None:
            raise ValueError(f"the first source ({names[0]}) is the primary one and has no join_on")
        for s in self.sources[1:]:
            if not s.join_on:
                raise ValueError(f"source {s.name!r} needs join_on (a column it shares with {names[0]!r})")
        for ref in self.column_refs():
            if ref.split(".", 1)[0] not in names:
                raise ValueError(f"column {ref!r} names no source; sources: {names}")
        seen: dict[str, str] = {}
        for f in self.fields:
            for t in f.targets():
                if t in seen:
                    raise ValueError(f"target {t!r} is set by both {seen[t]!r} and {f.source!r}")
                seen[t] = f.source
        as_of = parse_as_of(self.as_of)
        check_test_from(self.test_from)
        if parse_as_of(self.test_from) > as_of:
            raise ValueError(f"test_from {self.test_from} is after as_of {self.as_of}")
        unknown = sorted(set(self.review) - set(REVIEW_KEYS))
        if unknown:
            raise ValueError(f"review has unknown key(s) {unknown}; allowed: {list(REVIEW_KEYS)}")

    def column_refs(self) -> list[str]:
        """Every column the mapping reads (lead expressions, outcome, fields), in order, duplicates kept."""
        exprs = [getattr(self.lead, r) for r in LEAD_ROLES]
        refs = [c for e in exprs if e is not None for c in e.columns()]
        return refs + [self.outcome.column] + [f.source for f in self.fields]

    def source_columns(self, submit_time_only: bool = True) -> dict[str, list[str]]:
        """Source file -> the raw column names the mapping reads from it, in first-use order.

        ``submit_time_only`` (the default) keeps what a new lead has when it arrives: the ``lead_id`` and
        ``created_at`` expressions and every field row that is not ``ignore`` (``emva.ingest.convert.convert_leads``
        reads exactly these; the first two are optional there). False adds the outcome, the other CRM times, the
        deal value and ignored fields: everything ``convert`` reads. Join columns are included for joined files.
        """
        if submit_time_only:
            exprs = [e for e in (self.lead.lead_id, self.lead.created_at) if e is not None]
            refs = [c for e in exprs for c in e.columns()] + [f.source for f in self.fields if f.target != IGNORE]
        else:
            refs = self.column_refs()
        by_name = {s.name: s for s in self.sources}
        out: dict[str, list[str]] = {}
        for ref in refs:
            name, column = ref.split(".", 1)
            src = by_name[name]
            cols = out.setdefault(src.file, [])
            for c in ([src.join_on] if src.join_on else []) + [column]:
                if c not in cols:
                    cols.append(c)
        primary = self.sources[0]
        for src in self.sources[1:]:
            if src.file in out and src.join_on not in out.setdefault(primary.file, []):
                out[primary.file].append(src.join_on)
        return out

    def mapped_targets(self) -> list[str]:
        """Schema columns some field row sets, in mapping order."""
        return [t for f in self.fields for t in f.targets()]


# --- TOML -> DatasetMapping ------------------------------------------------------------------------------------------

_TOP_KEYS = {"name", "as_of", "test_from", "source_url", "licence", "outcome_confirmed", "drop_rows_without_created_at",
             "sources", "lead", "outcome", "review", "fields"}
_REQUIRED_TOP = ("name", "as_of", "test_from", "sources", "lead", "outcome")


def _check_keys(table: Mapping[str, object], allowed: set[str] | tuple[str, ...], where: str) -> None:
    """Refuse keys of ``table`` outside ``allowed`` (typos must fail, not be ignored)."""
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _expr(obj: object, where: str) -> Expr:
    """An ``Expr`` from its TOML form (a column string or ``{op, args}``)."""
    if isinstance(obj, str):
        return Expr.col(obj)
    if isinstance(obj, Mapping):
        _check_keys(obj, {"op", "args"}, where)
        args = obj.get("args", [])
        if not isinstance(args, list):
            raise ValueError(f"{where}: args must be a list")
        return Expr(str(obj.get("op")), args=tuple(_expr(a, f"{where}.args[{i}]") for i, a in enumerate(args)))
    raise ValueError(f"{where}: an expression is a column string or {{ op = ..., args = [...] }}, got {obj!r}")


def _review(obj: object, where: str) -> Review:
    """A ``Review`` from ``{reason, confidence}``."""
    if not isinstance(obj, Mapping):
        raise ValueError(f"{where}: expected a table with reason / confidence")
    _check_keys(obj, {"reason", "confidence"}, where)
    return Review(str(obj.get("reason", "")), obj.get("confidence"))


def _strings(obj: object, where: str) -> tuple[str, ...]:
    """A list of strings as a tuple (``ValueError`` otherwise)."""
    if not isinstance(obj, list) or not all(isinstance(v, str) for v in obj):
        raise ValueError(f"{where} must be a list of strings")
    return tuple(obj)


def _str_map(obj: object, where: str) -> dict[str, str]:
    """A table of string -> string."""
    if not isinstance(obj, Mapping) or not all(isinstance(v, str) for v in obj.values()):
        raise ValueError(f"{where} must be a table of strings")
    return dict(obj)


def load_mapping(text: str, require_confirmed: bool = False) -> DatasetMapping:
    """Parse and validate mapping TOML ``text`` (schema in the module docstring).

    Raises ``ValueError`` for invalid TOML, unknown keys, a missing ``created_at``, unknown target
    columns, and (``require_confirmed=True``, as ``python -m emva.ingest convert`` loads) ``outcome_confirmed = false``.
    Drafts load with the default so they can be reviewed and saved.
    """
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"mapping is not valid TOML: {e}") from e
    _check_keys(doc, _TOP_KEYS, "mapping")
    missing = [k for k in _REQUIRED_TOP if k not in doc]
    if missing:
        raise ValueError(f"mapping is missing {missing}")
    lead = doc["lead"]
    if not isinstance(lead, Mapping):
        raise ValueError("[lead] must be a table")
    _check_keys(lead, LEAD_ROLES, "[lead]")
    if "created_at" not in lead:
        raise ValueError("[lead] has no created_at (required)")
    sources = []
    for i, s in enumerate(doc["sources"]):
        _check_keys(s, {"name", "file", "join_on"}, f"[[sources]] {i + 1}")
        if "name" not in s or "file" not in s:
            raise ValueError(f"[[sources]] {i + 1} needs name and file")
        sources.append(Source(s["name"], s["file"], s.get("join_on")))
    out = doc["outcome"]
    _check_keys(out, {"kind", "column", "stage_map", "won_values", "lost_values", "lost_without_close"}, "[outcome]")
    if "kind" not in out or "column" not in out:
        raise ValueError("[outcome] needs kind and column")
    outcome = Outcome(out["kind"], out["column"], _str_map(out.get("stage_map", {}), "[outcome.stage_map]"),
                      _strings(out.get("won_values", []), "won_values"),
                      _strings(out.get("lost_values", []), "lost_values"), out.get("lost_without_close", "error"))
    fields_ = []
    for i, f in enumerate(doc.get("fields", [])):
        where = f"[[fields]] {i + 1}"
        _check_keys(f, {"source", "target", "value_map", "reason", "confidence"}, where)
        if "source" not in f:
            raise ValueError(f"{where} needs source")
        vm = f.get("value_map")
        if vm is not None:
            vm = {k: _str_map(v, f"{where}.value_map.{k}") for k, v in _str_map_of_tables(vm, where).items()}
        fields_.append(FieldMap(f["source"], f.get("target"), vm, Review(f.get("reason", ""), f.get("confidence"))))
    review = doc.get("review", {})
    m = DatasetMapping(
        name=doc["name"], sources=tuple(sources),
        lead=LeadColumns(**{r: _expr(lead[r], f"[lead].{r}") for r in LEAD_ROLES if r in lead}),
        outcome=outcome, as_of=str(doc["as_of"]), test_from=str(doc["test_from"]), fields=tuple(fields_),
        source_url=doc.get("source_url", ""), licence=doc.get("licence", ""),
        outcome_confirmed=_bool(doc.get("outcome_confirmed", False), "outcome_confirmed"),
        drop_rows_without_created_at=_bool(doc.get("drop_rows_without_created_at", False),
                                           "drop_rows_without_created_at"),
        review={k: _review(v, f"[review].{k}") for k, v in review.items()})
    if require_confirmed:
        check_confirmed(m)
    return m


def _str_map_of_tables(obj: object, where: str) -> dict[str, object]:
    """A table whose values are tables (a value map)."""
    if not isinstance(obj, Mapping) or not all(isinstance(v, Mapping) for v in obj.values()):
        raise ValueError(f"{where}: value_map must map each raw value to a table {{ target = value, ... }}")
    return dict(obj)


def _bool(v: object, name: str) -> bool:
    """``v`` if it is a TOML boolean."""
    if not isinstance(v, bool):
        raise ValueError(f"{name} must be true or false, got {v!r}")
    return v


def check_confirmed(m: DatasetMapping) -> None:
    """Raise ``ValueError`` unless a person confirmed the outcome mapping (``outcome_confirmed = true``)."""
    if not m.outcome_confirmed:
        raise ValueError(f"mapping {m.name!r} is a draft: outcome_confirmed is false. Review the [outcome] and "
                         "[lead] sections, then set outcome_confirmed = true before converting.")


# --- DatasetMapping -> TOML ------------------------------------------------------------------------------------------

def _s(v: str) -> str:
    """A TOML basic string. JSON escaping is valid TOML and escapes the C0 controls; DEL (U+007F), which TOML also
    forbids raw, is escaped here. Raises ``ValueError`` for a lone surrogate (no TOML or UTF-8 form exists)."""
    bad = [f"U+{ord(c):04X}" for c in v if 0xD800 <= ord(c) <= 0xDFFF]
    if bad:
        raise ValueError(f"cannot write {v!r} as TOML: lone surrogate(s) {bad}")
    return json.dumps(v, ensure_ascii=False).replace("\x7f", "\\u007f")


def _k(key: str) -> str:
    """A TOML key: bare when possible, else quoted (so ``answers.country`` stays one key)."""
    return key if _BARE_KEY.match(key) else _s(key)


def _expr_toml(e: Expr) -> str:
    """An expression's TOML form."""
    if e.op == COLUMN_OP:
        return _s(e.column)
    return f'{{ op = {_s(e.op)}, args = [{", ".join(_expr_toml(a) for a in e.args)}] }}'


def _inline(table: Mapping[str, str]) -> str:
    """An inline table of strings (``{}`` when empty)."""
    return "{ " + ", ".join(f"{_k(k)} = {_s(v)}" for k, v in table.items()) + " }" if table else "{}"


def _review_toml(r: Review) -> str:
    """A review as an inline table."""
    return _inline({"reason": r.reason, **({"confidence": r.confidence} if r.confidence else {})})


def dump_mapping(m: DatasetMapping, header: str = "") -> str:
    """``m`` as TOML that ``load_mapping`` reads back to an equal mapping. Deterministic: same mapping, same text.

    ``header`` lines are written first as ``#`` comments (comments in a hand-edited file are not kept).
    """
    out = [f"# {line}".rstrip() for line in header.splitlines()]
    if out:
        out.append("")
    out += [f"name = {_s(m.name)}", f"as_of = {_s(m.as_of)}", f"test_from = {_s(m.test_from)}",
            f"source_url = {_s(m.source_url)}", f"licence = {_s(m.licence)}",
            f"outcome_confirmed = {str(m.outcome_confirmed).lower()}",
            f"drop_rows_without_created_at = {str(m.drop_rows_without_created_at).lower()}"]
    for s in m.sources:
        out += ["", "[[sources]]", f"name = {_s(s.name)}", f"file = {_s(s.file)}"]
        if s.join_on is not None:
            out.append(f"join_on = {_s(s.join_on)}")
    out += ["", "[lead]"]
    out += [f"{r} = {_expr_toml(getattr(m.lead, r))}" for r in LEAD_ROLES if getattr(m.lead, r) is not None]
    o = m.outcome
    out += ["", "[outcome]", f"kind = {_s(o.kind)}", f"column = {_s(o.column)}",
            f"lost_without_close = {_s(o.lost_without_close)}"]
    if o.kind == "won_flag":
        out += [f"won_values = [{', '.join(map(_s, o.won_values))}]",
                f"lost_values = [{', '.join(map(_s, o.lost_values))}]"]
    if o.kind == "stage":
        out += ["", "[outcome.stage_map]"] + [f"{_k(k)} = {_s(v)}" for k, v in o.stage_map.items()]
    if m.review:
        out += ["", "[review]"] + [f"{k} = {_review_toml(r)}" for k, r in m.review.items()]
    for f in m.fields:
        out += ["", "[[fields]]", f"source = {_s(f.source)}"]
        if f.target is not None:
            out.append(f"target = {_s(f.target)}")
        if f.review.reason:
            out.append(f"reason = {_s(f.review.reason)}")
        if f.review.confidence:
            out.append(f"confidence = {_s(f.review.confidence)}")
        if f.value_map is not None:
            out += ["[fields.value_map]"] + [f"{_k(k)} = {_inline(v)}" for k, v in f.value_map.items()]
    return "\n".join(out) + "\n"


__all__ = ["ANSWER_PREFIX", "CONFIDENCES", "DatasetMapping", "Expr", "FieldMap", "IGNORE", "LEAD_COLUMNS",
           "LEAD_ROLES", "LeadColumns", "MAX_SOURCES", "OPS", "OUTCOME_KINDS", "Outcome", "Review", "STAGES", "Source",
           "TARGETS",
           "check_confirmed", "dump_mapping", "load_mapping"]
