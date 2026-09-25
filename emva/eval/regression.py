"""Regression checks against the frozen baseline (used by ``make baseline`` and the tests).

``weights.csv`` is compared canonically: same row names, values equal to 3 dp, row order
ignored. Byte identity is not required because rows with equal rounded log-odds are ordered
by numpy's unstable sort, whose tie order differs across CPU/numpy builds (the committed
file puts ``business_hours=wkday_9-18`` before ``channel=organic_direct``; arm64 macOS
produces the reverse with numpy 1.26 and 2.4 and pandas 2.2 and 3.0).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The baseline's published test-set numbers (REBUILD_PLAN.md, Phase 0 acceptance).
EXPECTED_SUMMARY: dict[str, float] = {"auc": 0.814, "brier": 0.1006, "top20_wins": 0.571, "top20_revenue": 0.795}
FROZEN_WEIGHTS: Path = Path(__file__).resolve().parents[2] / "baseline" / "weights.csv"


def parse_summary(stdout: str) -> dict[str, float]:
    """Parse the last two lines of the pipeline's printed summary (header + ``formula`` row)."""
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    header, values = lines[-2].split(), lines[-1].split()
    if header[0] != "model" or len(header) != len(values):
        raise ValueError(f"unexpected summary output:\n{stdout}")
    return {k: float(v) for k, v in zip(header[1:], values[1:])}


def summary_mismatches(got: dict[str, float], expected: dict[str, float] = EXPECTED_SUMMARY) -> list[str]:
    """Describe every expected metric that is missing or differs."""
    return [f"{k}: expected {v}, got {got.get(k)}" for k, v in expected.items() if got.get(k) != v]


def read_weights(path: str | Path) -> pd.DataFrame:
    """Read a weights.csv (index = design column name)."""
    return pd.read_csv(path, index_col=0)


def weights_mismatches(got: pd.DataFrame, expected: pd.DataFrame) -> list[str]:
    """Describe every difference between two weights tables, ignoring row order (3 dp)."""
    problems: list[str] = []
    if list(got.columns) != list(expected.columns):
        problems.append(f"columns differ: {list(got.columns)} vs {list(expected.columns)}")
        return problems
    extra, lost = got.index.difference(expected.index), expected.index.difference(got.index)
    if len(extra):
        problems.append(f"unexpected rows: {list(extra)}")
    if len(lost):
        problems.append(f"missing rows: {list(lost)}")
    common = got.index.intersection(expected.index).sort_values()
    a, b = got.loc[common].round(3), expected.loc[common].round(3)
    diff = ~np.isclose(a.values, b.values, rtol=0, atol=1e-9)
    for i, j in zip(*np.nonzero(diff)):
        problems.append(f"{common[i]} {a.columns[j]}: expected {b.iat[i, j]}, got {a.iat[i, j]}")
    return problems


def reordered_rows(got_path: str | Path, expected_path: str | Path) -> list[str]:
    """Row names whose position differs between two weights files (empty = same order)."""
    g, e = read_weights(got_path).index, read_weights(expected_path).index
    return [name for name, other in zip(g, e) if name != other]
