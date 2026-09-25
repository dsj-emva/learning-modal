"""CLI: ``python -m emva --data data/v1 --out DIR [--context FILE] [--margin 1.0] [label options]``.

With ``--label-mode legacy`` the arguments, stdout and output files (``weights.csv``,
``scores.csv``) are the same as ``baseline/emva_score.py`` (except ``--data`` defaults to
``data/v1``). The default ``--label-mode horizon`` uses the fixed-horizon label
(``--horizon-days``, ``--include-ghosted``, ``--stalled-as-lost``; see ``emva.labels``) and
adds ``won_within_h``, ``label_source`` and ``matured_at`` to ``scores.csv``.
"""
from __future__ import annotations

import argparse

from emva.constants import DEFAULT_LABEL_MODE, HORIZON_DAYS, LABEL_MODES
from emva.labels import LabelConfig
from emva.pipeline import run, write_outputs


def add_label_arguments(ap: argparse.ArgumentParser) -> None:
    """Add ``--label-mode``, ``--horizon-days``, ``--include-ghosted`` and ``--stalled-as-lost``."""
    ap.add_argument("--label-mode", choices=LABEL_MODES, default=DEFAULT_LABEL_MODE,
                    help="legacy = baseline labels (byte-identical outputs); horizon = won within H days (default)")
    ap.add_argument("--horizon-days", type=int, default=None,
                    help=f"horizon H in days from created_at (horizon mode only, default {HORIZON_DAYS})")
    ap.add_argument("--include-ghosted", action="store_true",
                    help="horizon mode: count leads still New at H as 0 instead of excluding them")
    ap.add_argument("--stalled-as-lost", action="store_true",
                    help="horizon mode: count stalled open deals as 0 instead of censoring them")


def label_config(ap: argparse.ArgumentParser, a: argparse.Namespace) -> LabelConfig:
    """Build the ``LabelConfig`` from parsed arguments; horizon-only options with legacy are a usage error."""
    if a.label_mode == "legacy" and (a.horizon_days is not None or a.include_ghosted or a.stalled_as_lost):
        ap.error("--horizon-days, --include-ghosted and --stalled-as-lost need --label-mode horizon")
    try:
        return LabelConfig(mode=a.label_mode, horizon_days=HORIZON_DAYS if a.horizon_days is None else a.horizon_days,
                           include_ghosted=a.include_ghosted, stalled_as_lost=a.stalled_as_lost)
    except ValueError as e:
        ap.error(str(e))


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the pipeline, print the summary and write outputs."""
    ap = argparse.ArgumentParser(prog="python -m emva")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--context", default=None, help="CSV with lead_id,context_score (0-1) from emva.context.agent")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--out", default=".")
    add_label_arguments(ap)
    a = ap.parse_args(argv)

    result = run(a.data, margin=a.margin, context=a.context, labels=label_config(ap, a))
    for line in result.messages:
        print(line)
    print(result.summary.to_string(index=False))
    write_outputs(result, a.out)


if __name__ == "__main__":
    main()
