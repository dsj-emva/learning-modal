"""Names of evaluation-only files (ground truth), so code outside ``emva/eval/`` can refuse them without naming them.

Ground rule 2 (CLAUDE.md): ground truth may be read only under ``emva/eval/`` and by the generator side, and the grep
rule keeps every mention of it inside this package. Code elsewhere (the dataset converter, its CLI) calls
``is_evaluation_only`` on a file name before reading the file and refuses it loudly; it never opens such a file.
"""
from __future__ import annotations

from pathlib import PurePath

# Every evaluation-only file starts with this (ground_truth_labels.csv, ground_truth.md, ground_truth_companies.csv).
EVALUATION_ONLY_PREFIX: str = "ground_truth"


def is_evaluation_only(name: str) -> bool:
    """True when the file name (or the last part of a path) ``name`` is an evaluation-only ground-truth file, in any
    letter case. Only the name is looked at; nothing is opened."""
    return PurePath(name).name.lower().startswith(EVALUATION_ONLY_PREFIX)


__all__ = ["EVALUATION_ONLY_PREFIX", "is_evaluation_only"]
