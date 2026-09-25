"""Synthetic lead data v2 (plan Phase 5): the v1 generator with every Phase 5.2-5.8 option switched on.

Usage:
  python scripts/generate_data_v2.py                      # data/v2, seed 20260924, n 10000, with paraphrase
  python scripts/generate_data_v2.py --no-paraphrase      # same, without the cached Haiku paraphrases
  python scripts/generate_data_v2.py --out DIR --n 2000 --seed 7

With paraphrase on (the default) every free-text template must already be in scripts/paraphrase_cache.json
(fill it with ``python scripts/paraphrase_templates.py --populate``); a missing template stops the run with
an error instead of silently falling back. What each option does and its planted size is written into the
output's ground_truth.md ("v2 mechanisms").
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from generate_data_v1 import Config, generate, write  # noqa: E402

V2_SETTINGS = {"text_paraphrase": True, "text_typo_share": 0.30, "text_language_mix_share": 0.35,
               "text_boilerplate_pool": True, "enrichment_dropout": 0.30, "consent_missing_share": 0.15,
               "interactions": True, "ghosting_follows_tier": True, "context_only_signal": True,
               "fast_human_share": 0.02}


def v2_config(seed: int = 20260924, as_of: str = "2026-09-24", n: int = 10000, paraphrase: bool = True) -> Config:
    """The v2 configuration: V2_SETTINGS on top of the v1 defaults (``paraphrase=False`` drops only 5.2 (i))."""
    return Config(seed=seed, as_of=as_of, n=n, **{**V2_SETTINGS, "text_paraphrase": paraphrase})


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--as-of", default="2026-09-24")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "data", "v2"))
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--no-paraphrase", action="store_true", help="skip the cached Claude Haiku paraphrases (5.2 i)")
    a = ap.parse_args(argv)
    cfg = v2_config(a.seed, a.as_of, a.n, paraphrase=not a.no_paraphrase)
    write(generate(cfg), a.out, cfg)


if __name__ == "__main__":
    main()
