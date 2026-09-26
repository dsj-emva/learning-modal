"""Column profiles of raw source files: the only thing about the data ever sent to an LLM (``emva.ingest.draft``).

A ``ColumnProfile`` holds a column's file and name, inferred type, share missing, distinct count, the min and max of a
date or number column, and up to ``max_examples`` example values, only for low-cardinality columns
(``MAX_DISTINCT_FOR_EXAMPLES`` distinct values or fewer). Full rows are never sent. Every example, min and max passes
through ``redact``: emails, phone-like digit runs and IP addresses anywhere in a value become ``<email>``, ``<phone>``,
``<ip>``; a text column whose name suggests a person's name (``is_person_column``: ``first_name``, ``manager``,
``account_owner``, ``assigned_to``) or whose values look like people's names (``looks_like_names``: "Cara Losch")
gets no examples, only ``<name>``. Redaction errs towards over-redacting (a long number reads as a phone number, a
two-word company name such as "Acme Corporation" may read as a person's name).
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import asdict, dataclass

import pandas as pd

# Columns with more distinct values than this get no examples (ids, free text, names, timestamps).
MAX_DISTINCT_FOR_EXAMPLES: int = 20
MAX_EXAMPLES: int = 10
MAX_EXAMPLE_CHARS: int = 80
EMAIL_TOKEN, PHONE_TOKEN, IP_TOKEN, NAME_TOKEN = "<email>", "<phone>", "<ip>", "<name>"
TYPES: tuple[str, ...] = ("empty", "bool", "number", "date", "text")

_EMAIL = re.compile(r"[A-Za-z0-9._%+'-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*")
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
# IPv6 candidates (hex groups joined by colons); ``_ipv6`` keeps only those ``ipaddress`` accepts.
_IPV6 = re.compile(r"(?<![0-9A-Za-z:])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![0-9A-Za-z:])")
# A digit run with optional separators and a leading +: a phone number when it holds 7+ digits and is not an ISO date.
_PHONE_CANDIDATE = re.compile(r"\+?\(?\d[\d\s().\-/]*\d")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(\s\d{1,2})?$")
MIN_PHONE_DIGITS: int = 7
# Column-name tokens that mark a person (``is_person_column``): when one is the last token of the name, or the whole
# name, the column is redacted to <name>. A plural ("managers") counts too.
PERSON_NAME_TOKENS: frozenset[str] = frozenset({
    "name", "firstname", "lastname", "surname", "fullname", "forename", "contact", "person", "agent", "owner",
    "rep", "salesperson", "customer", "client", "guest", "user", "username", "manager", "seller", "assignee",
    "assigned", "sdr", "buyer", "employee", "author", "lead", "head"})
# A last token that says the column holds something about a person, not the person ("lead_id", "customer_type",
# "lead_time", "owner_status"): never a name column, by name or by value.
TRAILING_NON_PERSON_TOKENS: frozenset[str] = frozenset({
    "id", "ids", "type", "time", "date", "status", "source", "code", "count", "number", "stage", "segment", "score"})
# A last token after a verb that points at a person ("assigned_to", "created_by").
BY_TO_TOKENS: frozenset[str] = frozenset({"to", "by"})
# Qualifiers that make "<qualifier>_name" the name of something else ("company_name", "product_name", "hotel_name").
# Only ``name`` is qualified this way: any other person token as the last token wins ("account_manager").
NON_PERSON_TOKENS: frozenset[str] = frozenset({
    "company", "account", "business", "product", "campaign", "page", "domain", "file", "hotel", "type", "segment",
    "id", "count", "number", "country", "city", "brand", "group", "team", "office", "stage", "status", "event",
    "project", "deal", "opportunity", "channel", "plan", "region", "store", "site", "org", "organisation",
    "organization", "firm", "series", "list", "form", "source", "template", "field", "column", "table"})
# A value that looks like a person's name: 2-3 capitalised words ("Cara Losch", "Mary-Ann O'Neil").
_NAME_VALUE = re.compile(r"^[A-Z][a-z'’-]+(?: [A-Z][a-z'’-]+){1,2}$")
# Share of a text column's distinct values that must look like names for the column to be redacted by value.
NAME_VALUE_SHARE: float = 0.5
_BOOL_TEXT = frozenset({"true", "false", "yes", "no", "t", "f", "y", "n"})


@dataclass(frozen=True)
class ColumnProfile:
    """What the drafter may know about one source column (module docstring); ``examples`` are already redacted."""

    file: str
    name: str
    type: str
    share_missing: float
    distinct: int
    examples: tuple[str, ...] = ()
    min: str | None = None
    max: str | None = None

    def as_dict(self) -> dict[str, object]:
        """A JSON-ready dict (examples as a list)."""
        d = asdict(self)
        d["examples"] = list(self.examples)
        return d


def _ipv6(match: re.Match[str]) -> str:
    """``<ip>`` for a valid IPv6 address, the text itself otherwise (e.g. a time like 10:30:00)."""
    try:
        ipaddress.IPv6Address(match.group(0))
    except ValueError:
        return match.group(0)
    return IP_TOKEN


def _phone(match: re.Match[str]) -> str:
    """``<phone>`` for a phone-like digit run, the text itself otherwise."""
    s = match.group(0)
    if sum(c.isdigit() for c in s) < MIN_PHONE_DIGITS or _ISO_DATE.match(s.strip()):
        return s
    return PHONE_TOKEN


def redact(value: str) -> str:
    """``value`` with every email, IP address and phone-like digit run replaced by its token (in that order)."""
    out = _EMAIL.sub(EMAIL_TOKEN, value)
    out = _IPV6.sub(_ipv6, _IPV4.sub(IP_TOKEN, out))
    return _PHONE_CANDIDATE.sub(_phone, out)


def _tokens(column: str) -> list[str]:
    """Lower-case tokens of a column name in order (split on non-letters and camelCase)."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", column)
    return [t for t in re.split(r"[^a-z]+", spaced.lower()) if t]


def _is_person_token(t: str) -> bool:
    """True for a ``PERSON_NAME_TOKENS`` token or its plural."""
    return t in PERSON_NAME_TOKENS or (t.endswith("s") and t[:-1] in PERSON_NAME_TOKENS)


def is_person_column(column: str) -> bool:
    """True when the column name suggests a person's name.

    The last token decides: a ``TRAILING_NON_PERSON_TOKENS`` token means no (``lead_id``, ``customer_type``,
    ``lead_time``); ``to`` / ``by`` after another token means yes (``assigned_to``, ``created_by``); ``name`` means yes
    unless the token before it is a ``NON_PERSON_TOKENS`` qualifier (``first_name`` yes, ``company_name`` no); any
    other person token means yes (``manager``, ``sales_agent``, ``account_manager``, ``company_contact``). A person
    token elsewhere in the name does not count (``lead_behaviour_profile``).
    """
    t = _tokens(column)
    if not t or t[-1] in TRAILING_NON_PERSON_TOKENS:
        return False
    if t[-1] in BY_TO_TOKENS:
        return len(t) > 1
    if t[-1] in ("name", "names"):
        return len(t) == 1 or t[-2] not in NON_PERSON_TOKENS
    return _is_person_token(t[-1])


def looks_like_names(values: list[str]) -> bool:
    """True when at least ``NAME_VALUE_SHARE`` of the distinct ``values`` are 2-3 capitalised words and their last
    words never repeat (surnames differ; "Resort Hotel" / "City Hotel" share a word, so they are not names)."""
    distinct = sorted(set(values))
    names = [v for v in distinct if _NAME_VALUE.match(v)]
    if not names or len(names) < NAME_VALUE_SHARE * len(distinct):
        return False
    last = [v.rsplit(" ", 1)[1] for v in names]
    return len(set(last)) == len(last)


def redact_as_names(column: str, values: list[str]) -> bool:
    """True when a text column's examples must be replaced by ``<name>``: its name suggests a person
    (``is_person_column``), or its name does not rule one out and its values look like names (``looks_like_names``).
    Errs towards redacting: a company name like "Acme Corporation" may read as a person's name."""
    if is_person_column(column):
        return True
    t = _tokens(column)
    return not (t and t[-1] in TRAILING_NON_PERSON_TOKENS) and looks_like_names(values)


def infer_type(values: pd.Series) -> str:
    """``empty``, ``bool``, ``number``, ``date`` (ISO 8601) or ``text`` for the non-blank text ``values``."""
    if values.empty:
        return "empty"
    if values.str.lower().isin(_BOOL_TEXT).all():
        return "bool"
    if pd.to_numeric(values, errors="coerce").notna().all():
        return "number"
    if values.str.match(r"^\d{4}-\d{2}-\d{2}").all() and \
            pd.to_datetime(values, errors="coerce", format="ISO8601", utc=True).notna().all():
        return "date"
    return "text"


def _clip(s: str) -> str:
    """``s`` cut to ``MAX_EXAMPLE_CHARS`` characters (with an ellipsis when cut)."""
    return s if len(s) <= MAX_EXAMPLE_CHARS else s[:MAX_EXAMPLE_CHARS - 1] + "…"


def profile_column(file: str, name: str, s: pd.Series, max_examples: int = MAX_EXAMPLES) -> ColumnProfile:
    """The profile of one raw column ``s`` (module docstring)."""
    text = s.astype(object).where(s.notna(), None).map(lambda v: None if v is None else str(v).strip())
    present = text[text.notna() & text.ne("")]
    kind = infer_type(present.astype(str))
    distinct = int(present.nunique())
    lo = hi = None
    if kind == "number":
        n = pd.to_numeric(present)
        lo, hi = redact(f"{n.min():.15g}"), redact(f"{n.max():.15g}")
    elif kind == "date":
        d = pd.to_datetime(present, format="ISO8601", utc=True)
        lo, hi = d.min().date().isoformat(), d.max().date().isoformat()
    if kind == "text" and redact_as_names(name, present.tolist() if distinct <= MAX_DISTINCT_FOR_EXAMPLES else []):
        examples: tuple[str, ...] = (NAME_TOKEN,) if distinct else ()
    elif distinct <= MAX_DISTINCT_FOR_EXAMPLES:
        counts = present.value_counts()
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_examples]
        examples = tuple(dict.fromkeys(_clip(redact(v)) for v, _ in ranked))
    else:
        examples = ()
    share = 1.0 - len(present) / len(s) if len(s) else 1.0
    return ColumnProfile(file, name, kind, round(share, 4), distinct, examples, lo, hi)


def profile(frames: dict[str, pd.DataFrame], max_examples: int = MAX_EXAMPLES) -> list[ColumnProfile]:
    """Profiles of every column of every frame (file name -> raw frame), files and columns in their given order."""
    return [profile_column(file, str(col), df[col], max_examples) for file, df in frames.items() for col in df.columns]


__all__ = ["ColumnProfile", "MAX_DISTINCT_FOR_EXAMPLES", "infer_type", "is_person_column", "looks_like_names",
           "profile", "profile_column", "redact", "redact_as_names"]
