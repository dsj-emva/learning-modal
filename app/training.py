"""Training from the app: the pre-training summary, the job subprocess and run bookkeeping. No Streamlit.

The app trains exactly as an engineer does locally: ``python -m emva --data <dataset> --out <run dir>
--label-mode M --feature-set F`` (writes ``scores.csv``, ``weights.csv``, ``model.joblib``), then, when the dataset
has ``status_quo_rules.json``, ``python -m emva.eval.report`` with the same options into ``<run dir>/report.txt``.
``start_training`` launches ``python -m app.job`` in its own process, which runs those two commands with their
output streamed to ``train.log`` and then calls ``finish_run``; so a run finishes and is recorded even if nobody
keeps the browser open, and the Streamlit server never blocks on it.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.storage import REPO_ROOT, RULES_FILE, Run, get_run, update_run, utc_now
from emva.constants import LABEL_SOURCES, TEST_FROM
from emva.eval.bootstrap import N_RESAMPLES
from emva.eval.report import FROZEN_HORIZON, frozen_test_labels
from emva.features import FeatureSet
from emva.io import load
from emva.labels import LabelConfig, LabelMode, label_source, split_masks
from emva.persist import BUNDLE_FILE
from emva.pipeline import build

# Files a successful ``python -m emva`` run leaves in ``--out``.
RUN_OUTPUTS: tuple[str, ...] = ("scores.csv", "weights.csv", BUNDLE_FILE)
# Names of the summary columns ``python -m emva`` prints (``emva.eval.metrics.summary``).
SUMMARY_COLUMNS: tuple[str, ...] = ("auc", "brier", "top20_wins", "top20_revenue")


@dataclass(frozen=True)
class TrainingConfig:
    """The CLI options the app exposes: ``label_mode`` (``horizon`` default, ``legacy`` reproduces the frozen
    POC's labels) and ``feature_set`` (``v2`` default, ``legacy`` the POC's features)."""

    label_mode: str = LabelMode.HORIZON.value
    feature_set: str = FeatureSet.V2.value

    def __post_init__(self) -> None:
        """Reject unknown option values (``ValueError``)."""
        LabelMode(self.label_mode)
        FeatureSet(self.feature_set)

    @property
    def labels(self) -> LabelConfig:
        """The ``LabelConfig`` (default horizon options)."""
        return LabelConfig(mode=self.label_mode)

    def cli_args(self) -> list[str]:
        """``["--label-mode", M, "--feature-set", F]``, shared by the training and report commands."""
        return ["--label-mode", self.label_mode, "--feature-set", self.feature_set]

    def as_dict(self) -> dict[str, str]:
        """The options as stored in ``Run.args``."""
        return {"label_mode": self.label_mode, "feature_set": self.feature_set}


def training_command(dataset: str | Path, out_dir: str | Path, config: TrainingConfig,
                     python: str = sys.executable) -> list[str]:
    """The ``python -m emva`` argv for one run (paths passed as single arguments, so spaces are safe)."""
    return [python, "-m", "emva", "--data", str(dataset), "--out", str(out_dir), *config.cli_args()]


def report_command(dataset: str | Path, config: TrainingConfig, python: str = sys.executable,
                   n_resamples: int = N_RESAMPLES) -> list[str]:
    """The ``python -m emva.eval.report`` argv for one run's standard report."""
    extra = [] if n_resamples == N_RESAMPLES else ["--n-resamples", str(n_resamples)]
    return [python, "-m", "emva.eval.report", "--data", str(dataset), *config.cli_args(), *extra]


@dataclass(frozen=True)
class TrainingSummary:
    """What a training run on a dataset would use (``pre_training_summary``): raw ``leads``, ``scored`` leads after
    bot/duplicate removal, the run's own ``train`` / ``test`` sizes and wins, the standard report's frozen
    ``mature_test`` set and wins, ``label_counts`` (per ``label_source``: leads, wins, losses, unlabelled) and the
    label definition's name."""

    leads: int
    scored: int
    train: int
    train_wins: int
    test: int
    test_wins: int
    mature_test: int
    mature_test_wins: int
    label_counts: dict[str, dict[str, int]]
    labels: str

    def label_counts_frame(self) -> pd.DataFrame:
        """``label_counts`` as a table, one row per ``label_source``."""
        return pd.DataFrame([{"label_source": k, **v} for k, v in self.label_counts.items()])


def pre_training_summary(dataset_path: str | Path, config: TrainingConfig) -> TrainingSummary:
    """What training on ``dataset_path`` with ``config`` would use, computed by the CLI's own code.

    ``emva.pipeline.build`` (clean, label, featurise) and ``emva.labels.split_masks`` with the config's eligibility,
    exactly as ``emva.pipeline.run``; the mature test set is ``emva.eval.report.frozen_test_labels``. Label counts
    are per ``label_source`` under the config's label.
    Raises what ``emva.io.load`` and the label code raise on malformed data (validate first).
    """
    L = load(dataset_path)
    labels = config.labels
    X = build(L, labels, FeatureSet(config.feature_set))
    train, test = split_masks(X, TEST_FROM, labels.eligible(X))
    src = X.label_source if "label_source" in X else label_source(X, FROZEN_HORIZON.horizon_days)
    counts = {}
    for s in LABEL_SOURCES:
        m = src == s
        counts[s] = {"leads": int(m.sum()), "wins": int((X.y[m] == 1).sum()), "losses": int((X.y[m] == 0).sum()),
                     "unlabelled": int(X.y[m].isna().sum())}
    y_mature = frozen_test_labels(L)[3]
    return TrainingSummary(leads=len(L), scored=len(X), train=int(train.sum()),
                           train_wins=int((X.y[train] == 1).sum()), test=int(test.sum()),
                           test_wins=int((X.y[test] == 1).sum()), mature_test=len(y_mature),
                           mature_test_wins=int((y_mature == 1).sum()), label_counts=counts,
                           labels=labels.describe())


# Environment variables the job and ``python -m emva`` may see; everything else (APP_PASSWORD, ANTHROPIC_*, cloud
# credentials) stays in the server process.
CHILD_ENV_KEYS: frozenset[str] = frozenset({"PATH", "HOME", "LANG", "PYTHONPATH", "TMPDIR", "DATA_DIR"})


def child_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """The training job's environment: ``CHILD_ENV_KEYS`` and ``LC_*`` from ``env`` (default ``os.environ``),
    plus ``PYTHONUNBUFFERED=1`` so the log streams."""
    src = os.environ if env is None else env
    out = {k: v for k, v in src.items() if k in CHILD_ENV_KEYS or k.startswith("LC_")}
    return {**out, "PYTHONUNBUFFERED": "1"}


def start_training(root: str | Path, run: Run, python: str = sys.executable, cwd: str | Path = REPO_ROOT,
                   report_resamples: int = N_RESAMPLES) -> subprocess.Popen:
    """Launch the job for the registered ``run`` (``python -m app.job``) and return its process at once.

    The job's own output and both commands' output are appended to ``run.log_path``. The caller should keep the
    ``Popen`` and ``poll()`` it (reaps the process). The run is marked ``running`` with the job's pid at once; the
    job records the outcome itself.
    """
    log = open(run.log_path, "ab")
    try:
        proc = subprocess.Popen(
            [python, "-m", "app.job", "--root", str(Path(root).resolve()), "--run-id", run.run_id,
             "--report-resamples", str(report_resamples)],
            cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=child_env(), start_new_session=True)
    finally:
        log.close()  # the child holds its own copy of the descriptor
    update_run(root, run.run_id, status="running", pid=proc.pid)  # so a job that dies before it starts is seen
    return proc


def parse_summary(log_text: str) -> dict[str, float]:
    """The ``formula`` row of the summary table ``python -m emva`` prints, as floats keyed by ``SUMMARY_COLUMNS``;
    empty when the log has no such table (e.g. no labelled test leads)."""
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        if line.split() == ["model", *SUMMARY_COLUMNS]:
            for row in lines[i + 1:]:
                cells = row.split()
                if cells[:1] == ["formula"] and len(cells) == len(SUMMARY_COLUMNS) + 1:
                    return dict(zip(SUMMARY_COLUMNS, map(float, cells[1:])))
    return {}


def finish_run(root: str | Path, run_id: str, returncode: int) -> Run:
    """Record the outcome of run ``run_id``: ``succeeded`` when the command exited 0 and wrote every
    ``RUN_OUTPUTS`` file, else ``failed`` with a reason; stores the parsed summary metrics and whether
    ``report.txt`` exists. Returns the updated run."""
    run = get_run(root, run_id)
    out = Path(run.out_dir)
    missing = [f for f in RUN_OUTPUTS if not (out / f).exists()]
    log = Path(run.log_path).read_text(encoding="utf-8", errors="replace") if Path(run.log_path).exists() else ""
    error = None
    if returncode != 0:
        error = f"training exited with code {returncode}; see the log"
    elif missing:
        error = f"training wrote no {', '.join(missing)}"
    report = run.report_path
    return update_run(root, run_id, status="failed" if error else "succeeded", error=error, finished_at=utc_now(),
                      metrics=parse_summary(log), has_report=report.exists() and report.stat().st_size > 0, pid=None)


def rules_available(dataset_path: str | Path) -> bool:
    """True when the dataset has ``status_quo_rules.json`` (the standard report needs it)."""
    return (Path(dataset_path) / RULES_FILE).exists()


__all__ = ["CHILD_ENV_KEYS", "RUN_OUTPUTS", "TrainingConfig", "TrainingSummary", "child_env", "finish_run",
           "parse_summary", "pre_training_summary", "report_command", "rules_available", "start_training",
           "training_command"]
