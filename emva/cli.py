"""Command-line options shared by ``python -m emva`` and ``python -m emva.eval.report``."""
from __future__ import annotations

import argparse

from emva.constants import DEFAULT_LABEL_MODE, HORIZON_DAYS, LABEL_MODES
from emva.labels import LabelConfig


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
