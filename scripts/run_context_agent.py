"""Phase 6.6: run the context agent on the seeded 1,000-lead v2 sample.

  python scripts/run_context_agent.py --probe 20     # first 20 sample leads; prints the judgments
  python scripts/run_context_agent.py                # the whole sample -> data/v2/context/context_judgments.csv
  python scripts/run_context_agent.py --dry-run      # cache hits/misses only, no API calls

The sample is read from the committed ``data/v2/context/sample_ids.csv`` (``lead_id`` only). Its only writer
is ``python -m emva.eval.context_harness sample`` (evaluation side: it stratifies on the planted persona);
this script never reads ground truth (ground rule 2). The agent itself sees only the lead cards.
``--dry-run`` writes nothing. Replies are cached in ``data/v2/context/cache.json``
(committed), so a rerun with the same brief, prompt and model makes no API calls. Every run appends its
counts (cache hits, API requests including retries, parse and transport errors) to
``data/v2/context/runs.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from emva.context.agent import (  # noqa: E402
    MAX_WORKERS,
    MODEL_ID,
    PROMPT_VERSION,
    prompt_fingerprint,
    run_agent,
    select_leads,
)
from emva.context.cache import ReplyCache  # noqa: E402

DATA = REPO / "data" / "v2"
OUT_DIR = DATA / "context"
BRIEF = REPO / "baseline" / "business_brief.md"
SAMPLE_IDS = OUT_DIR / "sample_ids.csv"


def main(argv: list[str] | None = None) -> None:
    """Judge the committed sample (or its first ``--probe`` leads) and log the run (nothing written with ``--dry-run``)."""
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=None, help="judge only the first N sample leads and print them")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    a = ap.parse_args(argv)

    if not SAMPLE_IDS.exists():
        raise SystemExit(f"{SAMPLE_IDS} is missing; write it with `python -m emva.eval.context_harness sample`")
    ids = pd.read_csv(SAMPLE_IDS, usecols=["lead_id"]).lead_id.tolist()

    X = select_leads(DATA, ids, a.probe)
    brief = BRIEF.read_text(encoding="utf-8")
    out, stats = run_agent(X, brief, ReplyCache(OUT_DIR / "cache.json"), workers=a.workers, dry_run=a.dry_run)
    print(stats.describe())
    if a.dry_run:
        return
    with open(OUT_DIR / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "leads": stats.leads,
                            "probe": a.probe, "cache_hits": stats.cache_hits, "api_requests": stats.api_requests,
                            "by_status": stats.by_status, "brief_hash": stats.brief_hash,
                            "prompt_version": PROMPT_VERSION, "prompt_fingerprint": prompt_fingerprint(),
                            "model_id": MODEL_ID}) + "\n")
    if a.probe:
        with pd.option_context("display.max_colwidth", 80, "display.width", 250):
            print(out.drop(columns=["card_hash", "brief_hash", "prompt_version", "model_id"]).to_string(index=False))
        return
    out.to_csv(OUT_DIR / "context_judgments.csv", index=False)
    print(f"wrote {len(out)} rows to {OUT_DIR / 'context_judgments.csv'}")


if __name__ == "__main__":
    main()
