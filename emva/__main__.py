"""CLI: ``python -m emva --data data/v1 --out DIR [--context context_scores.csv] [--margin 1.0]``.

Same arguments, stdout and output files (``weights.csv``, ``scores.csv``) as
``baseline/emva_score.py``, except ``--data`` defaults to ``data/v1``.
"""
from __future__ import annotations

import argparse

from emva.pipeline import run, write_outputs


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the pipeline, print the summary and write outputs."""
    ap = argparse.ArgumentParser(prog="python -m emva")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--context", default=None, help="CSV with lead_id,context_score (0-1) from emva.context.agent")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--out", default=".")
    a = ap.parse_args(argv)

    result = run(a.data, margin=a.margin, context=a.context)
    for line in result.messages:
        print(line)
    print(result.summary.to_string(index=False))
    write_outputs(result, a.out)


if __name__ == "__main__":
    main()
