"""The context agent's output contract (plan 6.1, ADR 0014): named enum judgments plus one short reason.

The LLM never returns a 0-1 score. It returns one value from a small declared enum for each of
``JUDGMENTS`` and a free-text ``reason`` of at most ``REASON_MAX_WORDS`` words. ``RESPONSE_SCHEMA`` is
the JSON schema sent as the API's structured-output format (enums are enforced server-side; the
word limit is not expressible there and is checked by ``validate``). ``validate`` is the single gate
between a raw reply and the rest of the code: anything that fails it is a *parse error*
(``ContractError``), recorded separately from transport errors.

Agent output CSV (read by ``python -m emva --context``): one row per lead with ``OUTPUT_COLUMNS``.
``status`` is one of ``STATUSES``; judgment and reason cells are blank unless ``status == "ok"``.
Every row carries the stamp ``(brief_hash, prompt_version, model_id)`` and the ``card_hash`` of the
lead card that was sent.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Judgment name -> allowed values, in declared order. Changing any list is a contract change: bump
# ``emva.context.agent.PROMPT_VERSION`` (every cache entry becomes a miss) and ``CONTEXT_LEVELS``.
JUDGMENTS: dict[str, tuple[str, ...]] = {
    "is_real_business": ("yes", "no", "unclear"),
    "persona": ("buyer", "vendor", "student", "job_seeker", "competitor", "nonprofit", "agency_pitching", "unclear"),
    "problem_specificity": ("specific", "vague", "boilerplate", "unrelated"),
    "urgency": ("now", "this_quarter", "researching", "none"),
    "brief_fit": ("strong", "partial", "weak", "none"),
}
REASON_MAX_WORDS: int = 30

STATUS_OK = "ok"
STATUS_PARSE_ERROR = "parse_error"          # the API answered, but the reply broke the contract
STATUS_TRANSPORT_ERROR = "transport_error"  # no usable HTTP response after every retry
STATUSES: tuple[str, ...] = (STATUS_OK, STATUS_PARSE_ERROR, STATUS_TRANSPORT_ERROR)

STAMP_COLUMNS: tuple[str, ...] = ("brief_hash", "prompt_version", "model_id")
OUTPUT_COLUMNS: tuple[str, ...] = ("lead_id", *JUDGMENTS, "reason", "status", "error", "card_hash", *STAMP_COLUMNS)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [*JUDGMENTS, "reason"],
    "properties": {**{name: {"type": "string", "enum": list(values)} for name, values in JUDGMENTS.items()},
                   "reason": {"type": "string"}},
}


class ContractError(ValueError):
    """A reply that does not satisfy the contract (a parse error, never retried)."""


def validate(reply: object) -> dict[str, str]:
    """Return the reply as ``{judgment: value, ..., "reason": text}`` or raise ``ContractError``.

    Requires a JSON object with exactly the keys of ``JUDGMENTS`` plus ``reason``; each judgment a
    string from its enum; ``reason`` a non-empty string of at most ``REASON_MAX_WORDS`` words.
    """
    if not isinstance(reply, Mapping):
        raise ContractError(f"reply is {type(reply).__name__}, not a JSON object")
    want = {*JUDGMENTS, "reason"}
    missing, extra = want - set(reply), set(reply) - want
    if missing or extra:
        raise ContractError(f"keys differ from the contract: missing {sorted(missing)}, unexpected {sorted(extra)}")
    for name, values in JUDGMENTS.items():
        if reply[name] not in values:
            raise ContractError(f"{name}={reply[name]!r} is not one of {list(values)}")
    reason = reply["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ContractError("reason must be a non-empty string")
    n_words = len(reason.split())
    if n_words > REASON_MAX_WORDS:
        raise ContractError(f"reason has {n_words} words (max {REASON_MAX_WORDS})")
    return {**{name: reply[name] for name in JUDGMENTS}, "reason": reason.strip()}
