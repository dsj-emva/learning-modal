"""Collinearity check on the model design (plan 2.7).

``python -m emva.eval.collinearity [--data data/v1] [--feature-set v2] [label options]`` fits the
pipeline, computes the Pearson correlation between every pair of design columns on the training
rows, prints the largest |corr| and every pair above ``MAX_ABS_DESIGN_CORR`` (0.95), and exits 1
if there is any. Columns that are constant on the training rows have no correlation and are
listed separately (they get a zero weight from the L2 fit and cannot be collinear with anything).

Expected on v1: both feature sets fail. Legacy: ``email=free`` and ``no_company=yes`` are the same
column (a feature-design flaw, fixed in v2). v2: ``session_missing=yes`` and ``channel=meta_leadads`` are
the same column because every session-less v1 lead is a lead-ads lead (a property of the data; on
data/v2 consent-declined website leads separate them).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from emva.cli import add_feature_arguments, add_label_arguments, feature_set, label_config
from emva.constants import MAX_ABS_DESIGN_CORR
from emva.pipeline import run


class CollinearityError(ValueError):
    """Two or more design columns are correlated beyond the allowed limit."""


@dataclass(frozen=True)
class CollinearityResult:
    """Outcome of ``check_collinearity``.

    ``pairs`` lists ``(column a, column b, corr)`` for every pair with |corr| above ``threshold``,
    largest first; ``max_pair`` is the pair with the largest |corr| overall (None with fewer than two
    usable columns); ``constant`` lists the columns skipped because they have no variance.
    """

    threshold: float
    pairs: list[tuple[str, str, float]]
    max_pair: tuple[str, str, float] | None
    constant: list[str]

    @property
    def passed(self) -> bool:
        """True when no pair exceeds the threshold."""
        return not self.pairs

    def describe(self) -> str:
        """One line: PASS/FAIL, the largest |corr| pair and the number of offending pairs."""
        if self.max_pair is None:
            return "PASS: fewer than two non-constant columns"
        a, b, r = self.max_pair
        head = "PASS" if self.passed else f"FAIL ({len(self.pairs)} pairs with |corr| > {self.threshold})"
        return f"{head}: max |corr| = {abs(r):.3f} between {a} and {b}"


def check_collinearity(D: pd.DataFrame, threshold: float = MAX_ABS_DESIGN_CORR) -> CollinearityResult:
    """Pairwise Pearson correlation of the columns of ``D``; flag pairs with |corr| > ``threshold``."""
    values = D.to_numpy(dtype=float)
    sd = values.std(axis=0)
    usable = sd > 0
    cols = D.columns[usable]
    constant = [str(c) for c in D.columns[~usable]]
    if len(cols) < 2:
        return CollinearityResult(threshold, [], None, constant)
    corr = np.corrcoef(values[:, usable], rowvar=False)
    iu = np.triu_indices(len(cols), k=1)
    r = corr[iu]
    order = np.argsort(-np.abs(r), kind="stable")
    pairs = [(str(cols[iu[0][k]]), str(cols[iu[1][k]]), float(r[k])) for k in order if abs(r[k]) > threshold]
    k = order[0]
    return CollinearityResult(threshold, pairs, (str(cols[iu[0][k]]), str(cols[iu[1][k]]), float(r[k])), constant)


def assert_no_collinearity(D: pd.DataFrame, threshold: float = MAX_ABS_DESIGN_CORR) -> CollinearityResult:
    """``check_collinearity`` that raises ``CollinearityError`` listing the offending pairs on failure."""
    res = check_collinearity(D, threshold)
    if not res.passed:
        listed = ", ".join(f"{a} ~ {b} ({r:+.3f})" for a, b, r in res.pairs)
        raise CollinearityError(f"design columns with |corr| > {threshold}: {listed}")
    return res


def format_result(res: CollinearityResult) -> list[str]:
    """Markdown lines: the verdict, every offending pair and the constant columns."""
    lines = [f"- {res.describe()}"]
    lines += [f"  - {a} ~ {b}: {r:+.3f}" for a, b, r in res.pairs]
    if res.constant:
        lines.append(f"- Constant on the training rows (skipped): {', '.join(res.constant)}")
    return lines


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run the pipeline, print the check, exit 1 on failure."""
    ap =argparse.ArgumentParser(prog="python -m emva.eval.collinearity")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--threshold", type=float, default=MAX_ABS_DESIGN_CORR)
    add_label_arguments(ap)
    add_feature_arguments(ap)
    a = ap.parse_args(argv)
    r = run(a.data, labels=label_config(ap, a), features=feature_set(a))
    res = check_collinearity(r.design[r.train], a.threshold)
    print(f"feature set {r.features.value}, labels {r.labels.describe()}, {int(r.train.sum())} training rows, "
          f"{r.design.shape[1]} design columns")
    print("\n".join(format_result(res)))
    if not res.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
