"""Context agent v2 (plan 6.4): Claude Haiku reads each lead card against the business brief and
returns the named judgments of ``emva.context.contract``. Output feeds ``python -m emva --context``.

Runtime rules:
- Anthropic Python SDK, structured output (``output_config`` JSON schema from the contract), temperature 0,
  model ``MODEL_ID`` (ADR 0010). The client sends the ``anthropic-workspace-id`` header; the key and the
  workspace id come from the environment or the nearest ``.env`` and are never printed.
- Bots and repeat submissions are removed with ``emva.io.clean`` *before* any call.
- Every reply is cached by (card hash, brief hash, prompt version, prompt fingerprint, model id)
  (``emva.context.cache``); identical cards in one run share one request.
- Transport errors (connection failures, timeouts, 408/409/429/5xx) are retried with exponential backoff up
  to ``MAX_ATTEMPTS`` times, each failure logged; other API errors (400/401/403/404) are configuration
  errors and raise. A reply that breaks the contract is a *parse error*: recorded, cached, never retried.
- Every output row is stamped with ``(brief_hash, prompt_version, model_id)`` and its ``card_hash``.
- ``--dry-run`` counts cache hits and misses for the given brief and exits without creating a client, so a
  brief edit (new ``brief_hash``, every lead a miss, context weights must be refit) is reported for free.

Usage:
  python -m emva.context.agent --data data/v2 --brief baseline/business_brief.md \
      --cache data/v2/context/cache.json --out data/v2/context/context_judgments.csv \
      [--ids FILE] [--limit N] [--workers 8] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd

from emva.context.cache import ReplyCache, cache_key
from emva.context.card import LeadCard, card_hash, cards, render
from emva.context.contract import (
    JUDGMENTS,
    OUTPUT_COLUMNS,
    RESPONSE_SCHEMA,
    STATUS_OK,
    STATUS_PARSE_ERROR,
    STATUS_TRANSPORT_ERROR,
    ContractError,
    validate,
)
from emva.env import load_env_var
from emva.io import clean, load

log = logging.getLogger(__name__)

MODEL_ID = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "judgments-v1"
MAX_ATTEMPTS = 5
MAX_WORKERS = 8
MAX_TOKENS = 400
REQUEST_TIMEOUT_S = 60.0
RETRYABLE_STATUS = frozenset({408, 409, 429})

SYSTEM_TEMPLATE = """You qualify inbound B2B leads for the business described in the brief below, the way an \
experienced salesperson would. You see only what the person typed on the form. Judge the lead on each of these, \
choosing exactly one allowed value:

- is_real_business: does the lead come from a real, operating business? yes / no / unclear
- persona: who is this person, relative to the business in the brief?
  buyer (evaluating the product for their own organisation), vendor (selling something to us), student \
(studying or doing research), job_seeker (looking for a job or internship), competitor (works for or is building \
a similar product), nonprofit (a charity or non-profit organisation), agency_pitching (an agency offering its \
services to us), unclear (cannot tell)
- problem_specificity: the free text is specific (a concrete problem in their own words), vague (short or \
generic), boilerplate (pasted marketing or template copy), or unrelated (nothing to do with our product)
- urgency: now / this_quarter / researching / none (no sign of timing)
- brief_fit: how well the lead matches who buys, per the brief: strong / partial / weak / none

Also give a reason of at most 30 words. Base every judgment on the card and the brief only.

BUSINESS BRIEF:
{brief}"""


def prompt_fingerprint() -> str:
    """First 16 hex digits of sha256 over ``SYSTEM_TEMPLATE``, ``RESPONSE_SCHEMA`` (sorted JSON) and ``MAX_TOKENS``.

    Part of the cache key next to ``PROMPT_VERSION``, so editing the prompt, the schema or the token limit
    without bumping the version still makes every entry a miss.
    """
    blob = json.dumps([SYSTEM_TEMPLATE, RESPONSE_SCHEMA, MAX_TOKENS], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def brief_hash(brief_text: str) -> str:
    """First 16 hex digits of the sha256 of the brief text (stamped into every row and the cache key)."""
    return hashlib.sha256(brief_text.encode("utf-8")).hexdigest()[:16]


def system_prompt(brief_text: str) -> str:
    """The system prompt for ``brief_text``."""
    return SYSTEM_TEMPLATE.format(brief=brief_text)


# --- credentials and client -------------------------------------------------------------------------

def make_client() -> anthropic.Anthropic:
    """SDK client with the workspace header and SDK retries off (``judge`` does its own, logged, retries).

    Raises ``SystemExit`` when ``ANTHROPIC_API_KEY`` or ``ANTHROPIC_WORKSPACE_ID`` cannot be found.
    """
    key = load_env_var("ANTHROPIC_API_KEY")
    workspace = load_env_var("ANTHROPIC_WORKSPACE_ID")
    if not key or not workspace:
        raise SystemExit("ANTHROPIC_API_KEY and ANTHROPIC_WORKSPACE_ID must be set (environment or .env); "
                         "the key is not workspace-scoped, so the workspace header is required (ADR 0010).")
    return anthropic.Anthropic(api_key=key, default_headers={"anthropic-workspace-id": workspace},
                               max_retries=0, timeout=REQUEST_TIMEOUT_S)


# --- one call ---------------------------------------------------------------------------------------

@dataclass
class Outcome:
    """Result of judging one card: ``status`` (contract ``STATUSES``), the validated ``reply`` when ok,
    the model's ``raw`` text (parse errors), an ``error`` description, and ``attempts`` (API requests made)."""

    status: str
    reply: dict[str, str] | None = None
    raw: str | None = None
    error: str = ""
    attempts: int = 0


def is_transport_error(exc: BaseException) -> bool:
    """True for errors worth retrying: no HTTP response, timeouts, 408/409/429 and 5xx."""
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return True
    return isinstance(exc, anthropic.APIStatusError) and (exc.status_code in RETRYABLE_STATUS or exc.status_code >= 500)


def parse_reply(message: Any) -> dict[str, str]:
    """Validated judgments from an SDK ``Message``; raises ``ContractError`` for anything off-contract."""
    if message.stop_reason != "end_turn":
        raise ContractError(f"stop_reason {message.stop_reason!r}")
    texts = [b.text for b in message.content if b.type == "text"]
    if len(texts) != 1:
        raise ContractError(f"expected one text block, got {len(texts)}")
    try:
        obj = json.loads(texts[0])
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON: {exc}") from exc
    return validate(obj)


def judge(client: anthropic.Anthropic, system: str, card_text: str,
          sleep: Callable[[float], None] = time.sleep) -> Outcome:
    """Send one card; retry transport errors with backoff 1, 2, 4, 8 s; never retry parse errors.

    Non-retryable API errors (bad request, authentication, permission, not found) propagate.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            message = client.messages.create(
                model=MODEL_ID, max_tokens=MAX_TOKENS, system=system,
                messages=[{"role": "user", "content": card_text}],
                output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
                extra_body={"temperature": 0})
        except anthropic.APIError as exc:
            if not is_transport_error(exc):
                raise
            log.warning("context call transport error (attempt %d/%d): %r", attempt, MAX_ATTEMPTS, exc)
            if attempt == MAX_ATTEMPTS:
                return Outcome(STATUS_TRANSPORT_ERROR, error=f"{type(exc).__name__}: {exc}", attempts=attempt)
            sleep(2 ** (attempt - 1))
            continue
        texts = [b.text for b in message.content if b.type == "text"]
        raw = texts[0] if texts else None
        try:
            return Outcome(STATUS_OK, reply=parse_reply(message), attempts=attempt)
        except ContractError as exc:
            log.warning("context reply broke the contract: %s", exc)
            return Outcome(STATUS_PARSE_ERROR, raw=raw, error=str(exc), attempts=attempt)
    raise AssertionError("unreachable")


# --- a batch of leads -------------------------------------------------------------------------------

@dataclass
class RunStats:
    """Counts for one run: leads judged, cache hits, API requests (including retries) and outcome counts."""

    leads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    unique_misses: int = 0
    api_requests: int = 0
    by_status: dict[str, int] = field(default_factory=lambda: {STATUS_OK: 0, STATUS_PARSE_ERROR: 0,
                                                              STATUS_TRANSPORT_ERROR: 0})
    brief_hash: str = ""
    cached_brief_hashes: set[str] = field(default_factory=set)

    def describe(self) -> str:
        """One line for the console and the report."""
        s = self.by_status
        return (f"{self.leads} leads, brief_hash {self.brief_hash}: {self.cache_hits} cache hits, "
                f"{self.cache_misses} misses ({self.unique_misses} unique cards), {self.api_requests} API requests; "
                f"ok {s[STATUS_OK]}, "
                f"parse errors {s[STATUS_PARSE_ERROR]}, transport errors {s[STATUS_TRANSPORT_ERROR]}")


def select_leads(data: str | Path, ids: list[str] | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load ``data``, drop bots and duplicates (``emva.io.clean``), then keep ``ids`` (in order) and/or the first ``limit``.

    Requested ids that ``clean`` removed are logged and skipped; ids not in the data raise ``KeyError``.
    """
    L = load(data)
    X = clean(L)
    if ids is not None:
        unknown = [i for i in ids if i not in L.index]
        if unknown:
            raise KeyError(f"{len(unknown)} lead ids not in {data}: {unknown[:5]}")
        dropped = [i for i in ids if i not in X.index]
        if dropped:
            log.warning("%d requested leads are bots or duplicates and are not sent: %s", len(dropped), dropped[:5])
        X = X.loc[[i for i in ids if i in X.index]]
    return X.head(limit) if limit is not None else X


def _row(lead_id: str, card: LeadCard, entry: dict[str, Any] | None, transport_error: str, bh: str) -> dict[str, Any]:
    """One output row: judgments (when ok), status, error, card hash and the three stamps.

    ``entry`` is the cache entry (None when every attempt failed in transport; ``transport_error`` says why).
    """
    reply = (entry or {}).get("reply") or {}
    status = entry["status"] if entry else STATUS_TRANSPORT_ERROR
    error = entry.get("error", "") if entry else transport_error
    return {"lead_id": lead_id, **{k: reply.get(k) for k in (*JUDGMENTS, "reason")}, "status": status,
            "error": error, "card_hash": card_hash(card), "brief_hash": bh, "prompt_version": PROMPT_VERSION,
            "model_id": MODEL_ID}


def run_agent(X: pd.DataFrame, brief_text: str, cache: ReplyCache,
              client_factory: Callable[[], anthropic.Anthropic] = make_client, workers: int = MAX_WORKERS,
              dry_run: bool = False, sleep: Callable[[float], None] = time.sleep) -> tuple[pd.DataFrame, RunStats]:
    """Judge every row of the cleaned frame ``X``; return (output frame with ``OUTPUT_COLUMNS``, stats).

    Cache hits cost nothing. With ``dry_run`` no client is created and no request is made: the frame is
    empty and the stats give hits and misses. Otherwise misses are sent with at most ``workers`` (<=
    ``MAX_WORKERS``) concurrent requests, one request per unique cache key (leads with identical cards share
    it); ok and parse-error outcomes are cached as they arrive.
    """
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}, got {workers}")
    bh = brief_hash(brief_text)
    system = system_prompt(brief_text)
    C = cards(X)
    fp = prompt_fingerprint()
    keys = {lid: cache_key(card_hash(c), bh, PROMPT_VERSION, fp, MODEL_ID) for lid, c in C.items()}
    misses = [lid for lid in C.index if cache.get(keys[lid]) is None]
    to_call = {keys[lid]: C[lid] for lid in misses}  # one request per unique key, first card wins (identical anyway)
    stats = RunStats(leads=len(C), cache_hits=len(C) - len(misses), cache_misses=len(misses),
                     unique_misses=len(to_call), brief_hash=bh, cached_brief_hashes=cache.brief_hashes())
    if dry_run:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS)), stats

    transport_errors: dict[str, str] = {}  # cache key -> error
    lock = threading.Lock()
    if to_call:
        client = client_factory()

        def work(key: str) -> None:
            c = to_call[key]
            out = judge(client, system, render(c), sleep=sleep)
            with lock:
                stats.api_requests += out.attempts
                if out.status == STATUS_TRANSPORT_ERROR:
                    transport_errors[key] = out.error
            if out.status != STATUS_TRANSPORT_ERROR:
                cache.put(key, {"card_hash": card_hash(c), "brief_hash": bh, "prompt_version": PROMPT_VERSION,
                                "prompt_fingerprint": fp, "model_id": MODEL_ID, "status": out.status,
                                "reply": out.reply, "raw": out.raw, "error": out.error})

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, to_call))

    rows = [_row(lid, c, cache.get(keys[lid]), transport_errors.get(keys[lid], ""), bh) for lid, c in C.items()]
    for r in rows:
        stats.by_status[r["status"]] += 1
    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS)), stats


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (see the module docstring)."""
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m emva.context.agent")
    ap.add_argument("--data", default="data/v2")
    ap.add_argument("--brief", default="baseline/business_brief.md")
    ap.add_argument("--cache", default="data/v2/context/cache.json")
    ap.add_argument("--out", default="data/v2/context/context_judgments.csv")
    ap.add_argument("--ids", default=None, help="CSV with a lead_id column: judge these leads, in this order")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--dry-run", action="store_true", help="report cache hits/misses for this brief; no API calls")
    a = ap.parse_args(argv)
    ids = pd.read_csv(a.ids).lead_id.tolist() if a.ids else None
    brief = Path(a.brief).read_text(encoding="utf-8")
    out, stats = run_agent(select_leads(a.data, ids, a.limit), brief, ReplyCache(a.cache), workers=a.workers,
                           dry_run=a.dry_run)
    print(stats.describe())
    if a.dry_run:
        if stats.cache_misses:
            known = ", ".join(sorted(stats.cached_brief_hashes)) or "none"
            print(f"dry run: {stats.cache_misses} leads would call the API. Cached brief hashes: {known}. "
                  f"A new brief_hash means new judgments and a refit of the context weights (ADR 0014).")
        return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
