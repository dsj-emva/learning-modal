"""Value transform (plan 3.1): turn the expected value ``p × E[deal value] × margin`` into the value uploaded.

A ``ValueTransform`` is a specification; ``ValueTransform.fit(reference)`` learns its data-dependent
parameters from a reference vector of expected values (the pipeline uses the training leads') and returns a
``FittedValueTransform`` whose ``apply`` maps any expected values, one lead or many, with those fixed
parameters. The steps run in this order, each optional:

1. **cap** at the ``cap_percentile``-th percentile of the reference values;
2. **compression** anchored at the median ``m`` of the capped reference values: ``sqrt``: ``sqrt(v × m)``;
   ``log``: ``m × log2(1 + v / m)``. Both are increasing and concave and map 0 to 0. The result is then
   multiplied by ``scale`` = (sum of capped reference values) / (sum of compressed reference values), so the
   reference total is unchanged and the value stays on the revenue scale in aggregate (without the rescale
   the v1 test sets' total value falls to about 0.4 × recorded revenue, which would misstate ROAS);
3. **floor** at ``floor`` GBP;
4. **tiers**: ``tiers`` quantile tiers of the reference values after steps 1 to 3; each value becomes the
   mean of the reference values in its tier (so the reference total is preserved).

Every step is monotone non-decreasing, so ranking by the transformed value never reverses an order; cap,
floor and tiers can only create ties.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from emva.constants import VALUE_CAP_PERCENTILE, VALUE_COMPRESSION, VALUE_FLOOR_GBP


class Compression(str, Enum):
    """``--value-compression``: none, ``log`` or ``sqrt`` (see the module docstring)."""

    NONE = "none"
    LOG = "log"
    SQRT = "sqrt"


@dataclass(frozen=True)
class ValueTransform:
    """Which transform steps to apply and their settings.

    ``cap_percentile`` in (0, 100] or None (no cap); ``floor`` in GBP (>= 0) or None; ``compression`` a
    ``Compression`` or its string value; ``tiers`` >= 2 or None. The default is plan 3.1's cap at p97
    and floor £25 plus log compression (ADR 0012, proposed), no tiers.
    """

    cap_percentile: float | None = VALUE_CAP_PERCENTILE
    floor: float | None = VALUE_FLOOR_GBP
    compression: Compression = Compression(VALUE_COMPRESSION)
    tiers: int | None = None

    def __post_init__(self) -> None:
        """Coerce ``compression`` and reject settings that mean nothing."""
        object.__setattr__(self, "compression", Compression(self.compression))  # ValueError for an unknown one
        if self.cap_percentile is not None and not 0 < self.cap_percentile <= 100:
            raise ValueError(f"cap_percentile must be in (0, 100], got {self.cap_percentile}")
        if self.floor is not None and self.floor < 0:
            raise ValueError(f"floor must be >= 0, got {self.floor}")
        if self.tiers is not None and self.tiers < 2:
            raise ValueError(f"tiers must be >= 2, got {self.tiers}")

    def describe(self) -> str:
        """Short name, e.g. ``identity``, ``cap p97 + floor £25`` or ``cap p97 + sqrt + tiers 5``."""
        parts = []
        if self.cap_percentile is not None:
            parts.append(f"cap p{self.cap_percentile:g}")
        if self.compression is not Compression.NONE:
            parts.append(self.compression.value)
        if self.floor is not None:
            parts.append(f"floor £{self.floor:g}")
        if self.tiers is not None:
            parts.append(f"tiers {self.tiers}")
        return " + ".join(parts) or "identity"

    def fit(self, reference: np.ndarray | pd.Series) -> FittedValueTransform:
        """Learn the cap, compression anchor and tiers from ``reference`` expected values (non-empty, finite, >= 0).

        Raises ``ValueError`` for an empty, non-finite or negative reference, or when the reference values
        after steps 1 to 3 cannot fill ``tiers`` non-empty tiers (too few distinct values).
        """
        ref = np.asarray(reference, dtype=float)
        if len(ref) == 0 or not np.isfinite(ref).all() or (ref < 0).any():
            raise ValueError("reference values must be a non-empty vector of finite values >= 0")
        cap = None if self.cap_percentile is None else float(np.percentile(ref, self.cap_percentile))
        capped = ref if cap is None else np.minimum(ref, cap)
        anchor = None if self.compression is Compression.NONE else float(np.median(capped))
        if anchor == 0:
            raise ValueError("compression needs a positive median reference value")
        scale = 1.0
        if anchor is not None:
            scale = float(capped.sum() / FittedValueTransform(self, cap, anchor, 1.0, (), ())._compress(capped).sum())
        fitted = FittedValueTransform(self, cap, anchor, scale, (), ())
        if self.tiers is None:
            return fitted
        pre = fitted.apply(ref)
        edges = np.quantile(pre, np.arange(1, self.tiers) / self.tiers)
        tier = np.searchsorted(edges, pre, side="right")
        counts = np.bincount(tier, minlength=self.tiers)
        if (counts == 0).any():
            raise ValueError(f"{self.tiers} tiers: the reference values leave {int((counts == 0).sum())} tier(s) "
                             "empty (too few distinct values)")
        means = np.bincount(tier, weights=pre, minlength=self.tiers) / counts
        return FittedValueTransform(self, cap, anchor, scale, tuple(float(e) for e in edges),
                                    tuple(float(m) for m in means))


@dataclass(frozen=True)
class FittedValueTransform:
    """A ``ValueTransform`` with its learned parameters: ``cap`` (GBP), compression ``anchor`` (GBP) and
    ``scale`` (1.0 without compression), and tiers.

    ``tier_edges`` are the ``tiers - 1`` interior quantile edges on the scale after steps 1 to 3, and
    ``tier_values`` the ``tiers`` values sent (empty tuples without tiers).
    """

    spec: ValueTransform
    cap: float | None
    anchor: float | None
    scale: float
    tier_edges: tuple[float, ...]
    tier_values: tuple[float, ...]

    def _compress(self, v: np.ndarray) -> np.ndarray:
        """Step 2 without the rescale: ``sqrt(v × anchor)`` or ``anchor × log2(1 + v / anchor)``; identity if none."""
        if self.spec.compression is Compression.SQRT:
            return np.sqrt(v * self.anchor)
        if self.spec.compression is Compression.LOG:
            return self.anchor * np.log2(1 + v / self.anchor)
        return v

    def apply(self, values: np.ndarray | pd.Series) -> np.ndarray:
        """Transform expected values (finite, >= 0) with the fitted parameters; raises ``ValueError`` otherwise."""
        v = np.asarray(values, dtype=float)
        if not np.isfinite(v).all() or (v < 0).any():
            raise ValueError("expected values must be finite and >= 0")
        if self.cap is not None:
            v = np.minimum(v, self.cap)
        if self.spec.compression is not Compression.NONE:
            v = self._compress(v) * self.scale
        if self.spec.floor is not None:
            v = np.maximum(v, self.spec.floor)
        if self.tier_values:
            v = np.asarray(self.tier_values)[np.searchsorted(np.asarray(self.tier_edges), v, side="right")]
        return v


IDENTITY = ValueTransform(cap_percentile=None, floor=None, compression=Compression.NONE)

__all__ = ["Compression", "FittedValueTransform", "IDENTITY", "ValueTransform"]
