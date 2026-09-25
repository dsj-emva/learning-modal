"""The context agent's current output contract, made explicit (Phase 6 replaces it).

Nothing here validates or enforces anything yet: the Phase 0 agent parses the first
``{...}`` block in the model's reply and passes whatever keys it finds through. This module
only names the shape the rest of the code already assumes.

LLM reply (JSON object)::

    {"context_score": <number 0-1>, "fit": "<one short sentence>", "red_flags": ["..."]}

When every retry fails the agent records ``error_response()`` for the lead.

Agent output CSV (read by ``python -m emva --context``): one row per lead with
``OUTPUT_COLUMNS``; ``red_flags`` joined with "; ". The pipeline reads only ``lead_id`` and
``context_score`` (blank = no score for that lead).
"""
from __future__ import annotations

from typing import TypedDict


class ContextResponse(TypedDict):
    """Keys the agent reads from the parsed LLM reply."""

    context_score: float | None
    fit: str
    red_flags: list[str]


OUTPUT_COLUMNS: tuple[str, ...] = ("lead_id", "context_score", "fit", "red_flags")


def error_response() -> ContextResponse:
    """The record stored for a lead whose call failed after all retries."""
    return {"context_score": None, "fit": "error", "red_flags": []}
