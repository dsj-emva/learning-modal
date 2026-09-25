"""Context agent: reads each lead the way a salesperson would, against a business brief,
and returns a 0-1 context score. Output feeds ``python -m emva --context``.

Moved from ``baseline/context_agent.py`` unchanged in behaviour, except that failed attempts
are logged to stderr instead of being swallowed silently, and the ``--data``/``--brief``
defaults point at this repo. Only submission-time information is shown to the agent. It
never sees CRM stages, comments, deal values or ground truth.

Usage:
  export ANTHROPIC_API_KEY=...
  python -m emva.context.agent --data data/v1 --brief baseline/business_brief.md \
      --out context_scores.csv [--limit 500]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from emva.context.contract import ContextResponse, error_response

log = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_ATTEMPTS = 5
SYSTEM = """You are an experienced B2B salesperson qualifying inbound leads for the business described below.
Judge how likely this lead is to become a paying customer, using judgement and context, not a checklist.
Reply with JSON only, no other text:
{"context_score": <number 0-1>, "fit": "<one short sentence>", "red_flags": ["..."]}

BUSINESS BRIEF:
"""


def lead_card(r: pd.Series) -> str:
    """Render the submission-time facts about one lead as the user message."""
    a = json.loads(r.answers)
    lines = [f"What they want to solve: {a.get('what_to_solve') or '(blank)'}",
             f"Job title: {a.get('job_title') or '(not asked)'}",
             f"Company typed: {a.get('company') or r.get('company_name') or '(none)'}",
             f"Email domain: {str(r.email).split('@')[-1]}",
             f"Country: {a.get('country')}"]
    for k in ["team_size", "budget", "timeline", "company_size"]:
        if a.get(k) not in (None, ""):
            lines.append(f"{k.replace('_', ' ').title()}: {a[k]}")
    if isinstance(r.get("co_sector"), str):
        lines.append(f"Company profile: {r.co_sector}, {r.co_employee_band} employees, ad spend {r.co_monthly_ad_spend_band}/month, "
                     f"CRM {r.co_crm_platform}, hiring {'yes' if r.co_is_hiring else 'no'}")
    return "\n".join(lines)


def call(system: str, card: str, key: str, cache: dict[str, ContextResponse]) -> ContextResponse:
    """Score one lead card, using ``cache`` keyed on sha256(system + card).

    Retries up to ``MAX_ATTEMPTS`` times with exponential backoff on any transport or parse
    failure (each failure is logged), then returns ``error_response()`` without caching it.
    """
    h = hashlib.sha256((system + card).encode()).hexdigest()
    if h in cache:
        return cache[h]
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
                "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": MODEL, "max_tokens": 200, "temperature": 0, "system": system,
                      "messages": [{"role": "user", "content": card}]})
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]
            out = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            cache[h] = out
            return out
        except Exception as exc:  # retried; Phase 6 separates parse from transport errors
            log.warning("context call failed (attempt %d/%d): %r", attempt + 1, MAX_ATTEMPTS, exc)
            time.sleep(2 ** attempt)
    return error_response()


def main() -> None:
    """CLI entry point: score leads and write ``lead_id,context_score,fit,red_flags``."""
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--brief", default="baseline/business_brief.md")
    ap.add_argument("--out", default="context_scores.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    key = os.environ["ANTHROPIC_API_KEY"]
    with open(a.brief) as f:
        system = SYSTEM + f.read()

    L = pd.read_csv(f"{a.data}/historical_leads.csv")
    CO = pd.read_csv(f"{a.data}/companies.csv")
    CO["dom"] = CO.domain.str.lower(); L["dom"] = L.company_domain.str.lower()
    L = L.merge(CO[["dom", "sector", "employee_band", "monthly_ad_spend_band", "crm_platform", "is_hiring"]]
                .add_prefix("co_").rename(columns={"co_dom": "dom"}), on="dom", how="left")
    if a.limit:
        L = L.sample(a.limit, random_state=0)
    cards = [lead_card(r) for _, r in L.iterrows()]

    cache_file = a.out + ".cache.json"
    cache: dict[str, ContextResponse] = {}
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda c: call(system, c, key, cache), cards))
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    pd.DataFrame({"lead_id": L.lead_id.values,
                  "context_score": [r.get("context_score") for r in res],
                  "fit": [r.get("fit") for r in res],
                  "red_flags": ["; ".join(r.get("red_flags") or []) for r in res]}).to_csv(a.out, index=False)
    print(f"wrote {len(res)} scores to {a.out}")


if __name__ == "__main__":
    main()
