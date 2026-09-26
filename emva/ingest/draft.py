"""Draft a ``DatasetMapping`` with Claude Haiku from column profiles; a person must confirm it before any conversion.

The model sees only ``emva.ingest.profile`` profiles (names, types, share missing, distinct counts, min / max of dates
and numbers, redacted examples of low-cardinality columns), never rows. It assigns each column a target from a
closed enum (a schema column, a lead role such as ``lead.created_at``, ``value_map`` or ``ignore``) with a reason and a
high / medium / low confidence, and proposes the outcome column and its stage map or won / lost values. It never
converts rows or fills values, and it drafts no derived expressions (a person adds those while reviewing). The result
always has ``outcome_confirmed = False``; ``emva.ingest.convert`` refuses it until a person sets it to true.

``as_of`` and ``test_from`` are not asked of the model: pass them, or they are computed from the profile of the
column drafted as ``lead.created_at`` (``as_of`` = the day after the latest date among the drafted date columns,
``test_from`` = the first of the month at 80% of the ``created_at`` span), to be reviewed like everything else.

Runtime as ``emva.context.agent`` (ADR 0010): the client from ``emva.context.agent.make_client`` (workspace header),
model ``MODEL_ID``, structured output (``RESPONSE_SCHEMA``), temperature 0; transport errors retried with logged
backoff, other API errors raised. Replies are cached in an ``emva.context.cache.ReplyCache`` keyed by
``cache_key(profiles_hash, DRAFT_BRIEF_HASH, PROMPT_VERSION, prompt_fingerprint(), MODEL_ID)``; a reply that breaks the
contract is cached as a parse error and raised as ``ContractError`` (never retried), transport errors are never cached.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import PurePath
from typing import Any

import anthropic
import pandas as pd

from emva.context.agent import MAX_ATTEMPTS, MODEL_ID, is_transport_error
from emva.context.cache import ReplyCache, cache_key
from emva.context.contract import STATUS_OK, STATUS_PARSE_ERROR, ContractError
from emva.ingest.mapping import (
    CONFIDENCES,
    IGNORE,
    LEAD_ROLES,
    OUTCOME_KINDS,
    STAGES,
    TARGETS,
    DatasetMapping,
    Expr,
    FieldMap,
    LeadColumns,
    Outcome,
    Review,
    Source,
)
from emva.ingest.profile import ColumnProfile

log = logging.getLogger(__name__)

PROMPT_VERSION = "mapping-draft-v1"
MAX_TOKENS = 8000
# The cache key's second slot (the context agent's brief hash): drafts have no brief.
DRAFT_BRIEF_HASH = "no-brief"
ROLE_PREFIX = "lead."
VALUE_MAP = "value_map"
DRAFT_TARGETS: tuple[str, ...] = (*(ROLE_PREFIX + r for r in LEAD_ROLES), *TARGETS, VALUE_MAP, IGNORE)
# Placeholder dates used only to validate a reply's content before the real dates are known.
_CHECK_DATES = ("2000-01-02", "2000-01-01")
TEST_SHARE_OF_SPAN: float = 0.8

SYSTEM_PROMPT = """You map the columns of a data source (one to three CSV files, described by column profiles; you \
never see rows) onto the lead schema of a B2B lead-scoring pipeline. Each lead is one row of the primary file; other \
files are joined to it on a shared column.

For every profiled column choose exactly one target:
- lead.lead_id (unique id of the lead), lead.created_at (when the lead came in: required), lead.contacted_at (when \
it was first worked), lead.won_at (when it was won; optional), lead.close_at (when it was closed: lost, or won when \
there is no won_at column), lead.deal_value (amount of a won deal);
- a schema column (form and tracking fields; answers.<key> are form answers) when the column means the same thing;
- value_map when the column's values translate into one or more schema columns (list every source value you see, \
each with the target and value it sets; categorical schema columns only accept their listed options);
- ignore for anything else, including every column that is only known after the outcome (it would leak the label).
Give a reason (at most 25 words) and a confidence: high, medium or low.

Then name the outcome: kind stage (a column of CRM stages: map every value to New, Contacted, Qualified, Demo booked, \
Proposal, Won or Lost), won_flag (a column whose values mean won or lost: list both), or presence (won when the \
column is filled, lost when blank). List the files with the primary first (join_on empty) and the join column for \
the others. Use only the files and columns in the profiles."""


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    """A closed JSON-schema object with every property required."""
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


_CONFIDENCE = {"type": "string", "enum": list(CONFIDENCES)}
_STRINGS = {"type": "array", "items": {"type": "string"}}
RESPONSE_SCHEMA: dict[str, Any] = _object({
    "sources": {"type": "array", "items": _object({"file": {"type": "string"}, "join_on": {"type": "string"}})},
    "columns": {"type": "array", "items": _object({
        "file": {"type": "string"}, "column": {"type": "string"},
        "target": {"type": "string", "enum": list(DRAFT_TARGETS)},
        "value_map": {"type": "array", "items": _object({
            "source_value": {"type": "string"}, "target": {"type": "string", "enum": list(TARGETS)},
            "target_value": {"type": "string"}})},
        "reason": {"type": "string"}, "confidence": _CONFIDENCE})},
    "outcome": _object({
        "kind": {"type": "string", "enum": list(OUTCOME_KINDS)}, "file": {"type": "string"},
        "column": {"type": "string"},
        "stage_map": {"type": "array", "items": _object({"value": {"type": "string"},
                                                          "stage": {"type": "string", "enum": list(STAGES)}})},
        "won_values": _STRINGS, "lost_values": _STRINGS, "reason": {"type": "string"}, "confidence": _CONFIDENCE}),
})


def prompt_fingerprint() -> str:
    """First 16 hex digits of sha256 over ``SYSTEM_PROMPT``, ``RESPONSE_SCHEMA`` (sorted JSON) and ``MAX_TOKENS``."""
    blob = json.dumps([SYSTEM_PROMPT, RESPONSE_SCHEMA, MAX_TOKENS], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def render_profiles(profiles: list[ColumnProfile]) -> str:
    """The user message: the profiles as JSON, grouped by file (this text is all the model sees of the data)."""
    by_file: dict[str, list[dict[str, object]]] = {}
    for p in profiles:
        d = p.as_dict()
        by_file.setdefault(str(d.pop("file")), []).append(d)
    return json.dumps({"files": [{"file": f, "columns": cols} for f, cols in by_file.items()]}, ensure_ascii=False,
                      indent=1)


def profiles_hash(profiles: list[ColumnProfile]) -> str:
    """First 16 hex digits of the sha256 of ``render_profiles`` (the cache key's first slot)."""
    return hashlib.sha256(render_profiles(profiles).encode("utf-8")).hexdigest()[:16]


def source_name(file: str) -> str:
    """A source name from a file name: the stem, lower-cased, runs of other characters turned into ``_``."""
    return re.sub(r"[^a-z0-9]+", "_", PurePath(file).stem.lower()).strip("_") or "source"


def _ref(file: str, column: str, names: dict[str, str], columns: dict[str, set[str]]) -> str:
    """``<source>.<column>`` for a column the profiles list; ``ContractError`` otherwise."""
    if file not in names:
        raise ContractError(f"file {file!r} is not in the profiles ({sorted(names)})")
    if column not in columns[file]:
        raise ContractError(f"column {column!r} is not in {file}")
    return f"{names[file]}.{column}"


def _build(reply: dict[str, Any], profiles: list[ColumnProfile], name: str, as_of: str,
           test_from: str) -> DatasetMapping:
    """The draft mapping a validated reply describes; ``ContractError`` for anything inconsistent with the profiles."""
    columns: dict[str, set[str]] = {}
    for p in profiles:
        columns.setdefault(p.file, set()).add(p.name)
    names = {f: source_name(f) for f in columns}
    if len(set(names.values())) != len(names):
        raise ContractError(f"file names give clashing source names: {names}")
    primary = [s for s in reply["sources"] if not s["join_on"]]
    if len(primary) != 1:
        raise ContractError(f"expected exactly one primary source (empty join_on), got {len(primary)}")
    ordered = primary + [s for s in reply["sources"] if s["join_on"]]
    sources = []
    for s in ordered:
        if s["join_on"]:
            _ref(s["file"], s["join_on"], names, columns)
        sources.append(Source(names[_known(s["file"], names)], s["file"], s["join_on"] or None))
    roles: dict[str, Expr] = {}
    review: dict[str, Review] = {}
    fields = []
    for c in reply["columns"]:
        ref = _ref(c["file"], c["column"], names, columns)
        note = Review(c["reason"], c["confidence"])
        target = c["target"]
        if target.startswith(ROLE_PREFIX):
            role = target[len(ROLE_PREFIX):]
            if role in roles:
                raise ContractError(f"two columns drafted as {target}")
            roles[role], review[role] = Expr.col(ref), note
        elif target == VALUE_MAP:
            vm: dict[str, dict[str, str]] = {}
            for e in c["value_map"]:
                vm.setdefault(e["source_value"], {})[e["target"]] = e["target_value"]
            if not vm:
                raise ContractError(f"{ref}: value_map target with no entries")
            fields.append(FieldMap(ref, value_map=vm, review=note))
        else:
            fields.append(FieldMap(ref, target=target, review=note))
    if "created_at" not in roles:
        raise ContractError(f"no column drafted as {ROLE_PREFIX}created_at (required)")
    o = reply["outcome"]
    outcome = Outcome(o["kind"], _ref(o["file"], o["column"], names, columns),
                      stage_map={e["value"]: e["stage"] for e in o["stage_map"]} if o["kind"] == "stage" else {},
                      won_values=tuple(o["won_values"]) if o["kind"] == "won_flag" else (),
                      lost_values=tuple(o["lost_values"]) if o["kind"] == "won_flag" else ())
    review["outcome"] = Review(o["reason"], o["confidence"])
    return DatasetMapping(name=name, sources=tuple(sources), lead=LeadColumns(**roles), outcome=outcome, as_of=as_of,
                          test_from=test_from, fields=tuple(fields), outcome_confirmed=False, review=review)


def _known(file: str, names: dict[str, str]) -> str:
    """``file`` if the profiles list it; ``ContractError`` otherwise."""
    if file not in names:
        raise ContractError(f"file {file!r} is not in the profiles ({sorted(names)})")
    return file


def check_reply(obj: object, profiles: list[ColumnProfile]) -> dict[str, Any]:
    """``obj`` if it is a reply the profiles support (it builds a valid draft mapping); ``ContractError`` otherwise.

    The server enforces the schema's shape and enums; this adds everything the schema cannot say (files and columns
    exist, one primary source, one column per role, created_at present, targets not repeated).
    """
    if not isinstance(obj, dict):
        raise ContractError(f"reply is {type(obj).__name__}, not a JSON object")
    try:
        _build(obj, profiles, "draft", *_CHECK_DATES)
    except (KeyError, TypeError) as e:
        raise ContractError(f"reply does not follow the schema: {e!r}") from e
    except ContractError:
        raise
    except ValueError as e:
        raise ContractError(str(e)) from e
    return obj


def parse_reply(message: Any, profiles: list[ColumnProfile]) -> dict[str, Any]:
    """The checked reply object from an SDK ``Message``; ``ContractError`` for anything off-contract."""
    if message.stop_reason != "end_turn":
        raise ContractError(f"stop_reason {message.stop_reason!r}")
    texts = [b.text for b in message.content if b.type == "text"]
    if len(texts) != 1:
        raise ContractError(f"expected one text block, got {len(texts)}")
    try:
        obj = json.loads(texts[0])
    except json.JSONDecodeError as e:
        raise ContractError(f"invalid JSON: {e}") from e
    return check_reply(obj, profiles)


def _request(client: anthropic.Anthropic, profiles: list[ColumnProfile],
             sleep: Callable[[float], None]) -> dict[str, Any]:
    """One draft request, with logged retries of transport errors (backoff 1, 2, 4, 8 s): a cache entry (``ok`` or
    ``parse_error``). Raises ``RuntimeError`` after ``MAX_ATTEMPTS`` transport failures; other API errors propagate."""
    user = render_profiles(profiles)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            message = client.messages.create(
                model=MODEL_ID, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
                extra_body={"temperature": 0})
        except anthropic.APIError as exc:
            if not is_transport_error(exc):
                raise
            log.warning("mapping draft transport error (attempt %d/%d): %r", attempt, MAX_ATTEMPTS, exc)
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"mapping draft failed after {MAX_ATTEMPTS} attempts: {exc!r}") from exc
            sleep(2 ** (attempt - 1))
            continue
        texts = [b.text for b in message.content if b.type == "text"]
        try:
            return {"status": STATUS_OK, "reply": parse_reply(message, profiles), "raw": None, "error": ""}
        except ContractError as exc:
            log.warning("mapping draft broke the contract: %s", exc)
            return {"status": STATUS_PARSE_ERROR, "reply": None, "raw": texts[0] if texts else None,
                    "error": str(exc)}
    raise AssertionError("unreachable")


def default_dates(mapping: DatasetMapping, profiles: list[ColumnProfile]) -> tuple[str, str]:
    """``(as_of, test_from)`` from the profiles of the drafted date columns (module docstring).

    Raises ``ValueError`` when the ``created_at`` column has no date range (not an ISO date column): pass the dates.
    """
    by_ref = {f"{source_name(p.file)}.{p.name}": p for p in profiles}
    dated = [by_ref[e.column] for r in ("created_at", "contacted_at", "won_at", "close_at")
             if (e := getattr(mapping.lead, r)) is not None and e.op == "column" and by_ref[e.column].type == "date"]
    created = by_ref[mapping.lead.created_at.column] if mapping.lead.created_at.op == "column" else None
    if created is None or created.type != "date":
        raise ValueError("the column drafted as created_at is not an ISO date column; pass as_of and test_from")
    as_of = max(pd.Timestamp(p.max) for p in dated) + pd.Timedelta(days=1)
    lo, hi = pd.Timestamp(created.min), pd.Timestamp(created.max)
    cut = lo + (hi - lo) * TEST_SHARE_OF_SPAN
    return as_of.date().isoformat(), cut.replace(day=1).date().isoformat()


def draft_key(profiles: list[ColumnProfile]) -> str:
    """The cache key of the draft for ``profiles`` under the current prompt and model."""
    return cache_key(profiles_hash(profiles), DRAFT_BRIEF_HASH, PROMPT_VERSION, prompt_fingerprint(), MODEL_ID)


def is_cached(profiles: list[ColumnProfile], cache: ReplyCache) -> bool:
    """True when ``cache`` holds the draft for ``profiles`` (``draft_mapping`` then needs no client)."""
    return cache.get(draft_key(profiles)) is not None


def draft_mapping(profiles: list[ColumnProfile], client: anthropic.Anthropic | None, cache: ReplyCache, name: str,
                  as_of: str | None = None, test_from: str | None = None,
                  sleep: Callable[[float], None] = time.sleep) -> DatasetMapping:
    """A draft mapping named ``name`` for the source described by ``profiles`` (``outcome_confirmed = False``).

    A cache hit makes no request (``client`` may then be None). A miss sends one request with ``client`` (from
    ``emva.context.agent.make_client``) and caches the reply. Raises ``ContractError`` when the (cached) reply breaks
    the contract, ``ValueError`` on a miss without a client or when the dates cannot be derived (``default_dates``),
    ``RuntimeError`` after repeated transport errors. ``as_of`` / ``test_from`` override the derived dates.
    """
    fp = prompt_fingerprint()
    ph = profiles_hash(profiles)
    key = draft_key(profiles)
    entry = cache.get(key)
    if entry is None:
        if client is None:
            raise ValueError("no cached draft for these profiles and no client to request one")
        entry = {"card_hash": ph, "brief_hash": DRAFT_BRIEF_HASH, "prompt_version": PROMPT_VERSION,
                 "prompt_fingerprint": fp, "model_id": MODEL_ID, **_request(client, profiles, sleep)}
        cache.put(key, entry)
    if entry["status"] != STATUS_OK:
        raise ContractError(f"the model's draft broke the contract (cached): {entry['error']}")
    checked = _build(entry["reply"], profiles, name, *_CHECK_DATES)
    if as_of is None or test_from is None:
        derived = default_dates(checked, profiles)
        as_of, test_from = as_of or derived[0], test_from or derived[1]
    return _build(entry["reply"], profiles, name, as_of, test_from)


__all__ = ["DRAFT_TARGETS", "MAX_TOKENS", "PROMPT_VERSION", "RESPONSE_SCHEMA", "SYSTEM_PROMPT", "check_reply",
           "default_dates", "draft_key", "draft_mapping", "is_cached", "parse_reply", "profiles_hash",
           "prompt_fingerprint", "render_profiles", "source_name"]
