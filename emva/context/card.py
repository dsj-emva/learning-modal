"""The lead card: the only lead facts the context agent sees (plan 6.2, ADR 0014).

A card holds what the formula cannot use: the free text, the typed company name, the job title
string and the email domain string. The business brief goes in the system prompt, not the card.
No enrichment (bands, sector, spend, CRM, hiring), no form answers the formula already buckets
(budget, timeline, company size, country), no behavioural telemetry, no CRM stages, no outcome.
``CARD_FIELDS`` is the whole list; ``tests/test_context.py`` asserts that nothing else gets in.
"""
from __future__ import annotations

import hashlib
from dataclasses import astuple, dataclass, fields

import pandas as pd

BLANK = "(blank)"


@dataclass(frozen=True)
class LeadCard:
    """The four strings shown to the agent for one lead (``BLANK`` when absent)."""

    what_to_solve: str
    company_typed: str
    job_title: str
    email_domain: str


CARD_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(LeadCard))
_LABELS: dict[str, str] = {"what_to_solve": "What they want to solve", "company_typed": "Company typed",
                           "job_title": "Job title", "email_domain": "Email domain"}


def _text(value: object) -> str:
    """``value`` stripped, or ``BLANK`` for None / NaN / pd.NA / empty strings."""
    if not isinstance(value, str) or not value.strip():
        return BLANK
    return value.strip()


def lead_card(row: pd.Series) -> LeadCard:
    """Build the card from one row of ``emva.io.load`` output (``a_<answer>`` columns, ``company_name``, ``email``).

    The typed company is the ``company`` form answer, else the ``company_name`` field; the email
    domain is everything after the last ``@``.
    """
    company = _text(row.get("a_company"))
    if company == BLANK:
        company = _text(row.get("company_name"))
    email = _text(row.get("email"))
    domain = email.rsplit("@", 1)[-1].lower() if "@" in email else BLANK
    return LeadCard(what_to_solve=_text(row.get("a_what_to_solve")), company_typed=company,
                    job_title=_text(row.get("a_job_title")), email_domain=domain)


def render(card: LeadCard) -> str:
    """The user message for ``card``: one ``Label: value`` line per field, in ``CARD_FIELDS`` order."""
    return "\n".join(f"{_LABELS[name]}: {value}" for name, value in zip(CARD_FIELDS, astuple(card), strict=True))


def card_hash(card: LeadCard) -> str:
    """sha256 hex of the rendered card (the cache key's lead part)."""
    return hashlib.sha256(render(card).encode("utf-8")).hexdigest()


def cards(X: pd.DataFrame) -> pd.Series:
    """``LeadCard`` for every row of ``X``, indexed like ``X``."""
    return pd.Series([lead_card(r) for _, r in X.iterrows()], index=X.index, dtype="object")
