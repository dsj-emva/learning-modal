"""Command-line options shared by ``python -m emva``, ``python -m emva.eval.report`` and other eval scripts."""
from __future__ import annotations

import argparse

from emva.constants import HORIZON_DAYS, VALUE_CAP_PERCENTILE, VALUE_COMPRESSION, VALUE_FLOOR_GBP
from emva.features import FeatureSet
from emva.labels import LabelConfig, LabelMode
from emva.value_transform import Compression, ValueTransform


def add_feature_arguments(ap: argparse.ArgumentParser) -> None:
    """Add ``--feature-set``."""
    ap.add_argument("--feature-set", choices=[f.value for f in FeatureSet], default=FeatureSet.V2.value,
                    help="legacy = baseline features and fixed deal-value residual sd (with --label-mode legacy: "
                         "byte-identical outputs); v2 = plan Phase 2 features, residual sd estimated (default)")


def feature_set(a: argparse.Namespace) -> FeatureSet:
    """The ``FeatureSet`` chosen by ``--feature-set``."""
    return FeatureSet(a.feature_set)


def add_label_arguments(ap: argparse.ArgumentParser) -> None:
    """Add ``--label-mode``, ``--horizon-days``, ``--include-ghosted`` and ``--stalled-as-lost``."""
    ap.add_argument("--label-mode", choices=[m.value for m in LabelMode], default=LabelMode.HORIZON.value,
                    help="legacy = baseline labels (byte-identical outputs); horizon = won within H days (default)")
    ap.add_argument("--horizon-days", type=int, default=None,
                    help=f"horizon H in days from created_at (horizon mode only, default {HORIZON_DAYS})")
    ap.add_argument("--include-ghosted", action="store_true",
                    help="horizon mode: count leads still New at H as 0 instead of excluding them")
    ap.add_argument("--stalled-as-lost", action="store_true",
                    help="horizon mode: count stalled open deals as 0 instead of censoring them")


def label_config(ap: argparse.ArgumentParser, a: argparse.Namespace) -> LabelConfig:
    """Build the ``LabelConfig`` from parsed arguments; invalid combinations are a usage error.

    ``LabelConfig`` rejects the ghosted/stalled flags in legacy mode itself; only an explicit
    ``--horizon-days`` with legacy is checked here, because the config cannot tell it from the default.
    """
    if a.label_mode == LabelMode.LEGACY.value and a.horizon_days is not None:
        ap.error("--horizon-days needs --label-mode horizon")
    try:
        return LabelConfig(mode=a.label_mode, horizon_days=HORIZON_DAYS if a.horizon_days is None else a.horizon_days,
                           include_ghosted=a.include_ghosted, stalled_as_lost=a.stalled_as_lost)
    except ValueError as e:
        ap.error(str(e))


def add_value_arguments(ap: argparse.ArgumentParser) -> None:
    """Add the value transform options (plan 3.1): cap, floor, compression and tiers for ``value_at_submit``."""
    ap.add_argument("--value-cap-percentile", type=float, default=None,
                    help=f"cap value_at_submit at this percentile of the training leads' values "
                         f"(default {VALUE_CAP_PERCENTILE:g}; horizon mode only)")
    ap.add_argument("--no-value-cap", action="store_true", help="do not cap value_at_submit")
    ap.add_argument("--value-floor", type=float, default=None,
                    help=f"floor value_at_submit at this many GBP (default {VALUE_FLOOR_GBP:g}; horizon mode only)")
    ap.add_argument("--no-value-floor", action="store_true", help="do not floor value_at_submit")
    ap.add_argument("--value-compression", choices=[c.value for c in Compression], default=None,
                    help=f"compress value_at_submit after the cap (default {VALUE_COMPRESSION}; horizon mode only)")
    ap.add_argument("--value-tiers", type=int, default=None,
                    help="replace value_at_submit by the mean of its quantile tier among N tiers (horizon mode only)")


def value_transform(ap: argparse.ArgumentParser, a: argparse.Namespace, labels: LabelConfig) -> ValueTransform:
    """Build the ``ValueTransform`` from parsed arguments; invalid combinations are a usage error.

    Any value option with ``--label-mode legacy`` is an error: legacy mode writes no ``value_at_submit``.
    """
    given = [a.value_cap_percentile is not None, a.no_value_cap, a.value_floor is not None, a.no_value_floor,
             a.value_compression is not None, a.value_tiers is not None]
    if labels.is_legacy and any(given):
        ap.error("value transform options need --label-mode horizon (legacy mode writes no value_at_submit)")
    if a.no_value_cap and a.value_cap_percentile is not None:
        ap.error("--no-value-cap and --value-cap-percentile are mutually exclusive")
    if a.no_value_floor and a.value_floor is not None:
        ap.error("--no-value-floor and --value-floor are mutually exclusive")
    try:
        return ValueTransform(
            cap_percentile=None if a.no_value_cap else (VALUE_CAP_PERCENTILE if a.value_cap_percentile is None
                                                        else a.value_cap_percentile),
            floor=None if a.no_value_floor else (VALUE_FLOOR_GBP if a.value_floor is None else a.value_floor),
            compression=a.value_compression or VALUE_COMPRESSION,
            tiers=a.value_tiers)
    except ValueError as e:
        ap.error(str(e))
