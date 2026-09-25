"""True close rates by company size, from ``ground_truth_labels.csv`` (evaluation only, ground rule 2).

``python -m emva.eval.ground_truth_reference [--data data/v1]`` prints the mean planted close
probability (``p_close_true``) of training-period leads by company size (the pipeline's
``band`` feature), for the reference figure in plan Phase 1 ("39.4% true rate" for band
201-1000). It is the target the labelled win rates in ``emva.eval.label_study`` are compared
against.

This is the only Phase 1 module that reads a ground-truth file. Nothing imports it; it is run
as a script so the pipeline and the rest of ``emva/eval`` stay free of ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emva.constants import TEST_FROM
from emva.eval.report import md_table
from emva.io import load
from emva.labels import HORIZON, LEGACY, eligible_rows
from emva.pipeline import build

BANDS: tuple[str, ...] = ("1-10", "11-50", "51-200", "201-1000", "1000+")


def true_rates(data: str | Path) -> pd.DataFrame:
    """Mean ``p_close_true`` by ``band`` over three training-period populations, as a table."""
    truth = pd.read_csv(Path(data) / "ground_truth_labels.csv", index_col="lead_id").p_close_true
    L = load(data)
    pops = {}
    for cfg in (LEGACY, HORIZON):
        X = build(L, cfg)
        pre = X.created_at < TEST_FROM
        pops[f"training rows labelled by {cfg.describe()}"] = (X, pre & X.y.notna() & eligible_rows(X, cfg))
    X = build(L, HORIZON)
    pops["all cleaned leads created before " + TEST_FROM + " (every label_source)"] = (X, X.created_at < TEST_FROM)
    rows = []
    for name, (X, mask) in pops.items():
        p = truth.reindex(X.index)
        rows.append({"population": name, **{b: f"{p[mask & (X.band == b)].mean():.1%} (n={int((mask & (X.band == b)).sum())})"
                                            for b in BANDS}})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: print the reference table."""
    ap = argparse.ArgumentParser(prog="python -m emva.eval.ground_truth_reference")
    ap.add_argument("--data", default="data/v1")
    a = ap.parse_args(argv)
    print("Mean planted close probability (p_close_true) by company size (band feature):\n")
    print(md_table(true_rates(a.data)))


if __name__ == "__main__":
    main()
