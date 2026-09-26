"""The training job: ``python -m app.job --root DATA_DIR --run-id ID [--report-resamples N]``.

Started by ``app.training.start_training`` with stdout/stderr on the run's ``train.log``. Marks the run
``running``, runs ``python -m emva`` (output to the log), then, if training succeeded and the dataset has
status-quo rules or is a converted dataset (``app.training.report_available``), ``python -m emva.eval.report``
with the same options into ``report.txt`` (its stderr to the log), and finally records the outcome with ``app.training.finish_run``. The report is supplementary: if it
fails the run still succeeds, with the failure in the log.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from app.storage import Run, get_run, update_run
from app.training import TrainingConfig, finish_run, report_available, report_command, training_command
from emva.eval.bootstrap import N_RESAMPLES


def _say(line: str) -> None:
    """Write a job line to the log (stdout) immediately."""
    print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run the job for one registered run; returns the training command's exit code."""
    ap = argparse.ArgumentParser(prog="python -m app.job")
    ap.add_argument("--root", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--report-resamples", type=int, default=N_RESAMPLES)
    a = ap.parse_args(argv)
    update_run(a.root, a.run_id, status="running", pid=os.getpid())
    rc = 1
    try:
        rc = _train_and_report(a.root, a.run_id, a.report_resamples)
    finally:  # record the outcome even when the job itself raises (the traceback lands in the log)
        run = finish_run(a.root, a.run_id, rc)
        _say(f"[done] {run.status}")
    return rc


def _display(cmd: list[str], run: Run) -> str:
    """``cmd`` as a shell line for the log, with the dataset and run directories shown by name (the log is shown
    in the app, which has no business printing server paths)."""
    names = {run.dataset_path: f"<dataset {run.dataset}>", run.out_dir: f"<run {run.run_id}>"}
    return shlex.join(["python", *(names.get(a, a) for a in cmd[1:])])


def _train_and_report(root: str, run_id: str, report_resamples: int) -> int:
    """Run the training command and, when possible, the report; return the training exit code."""
    run = get_run(root, run_id)
    config = TrainingConfig(**run.args)
    cmd = training_command(run.dataset_path, run.out_dir, config)
    _say(f"$ {_display(cmd, run)}")
    rc = subprocess.run(cmd, stdin=subprocess.DEVNULL).returncode
    _say(f"[train] exit code {rc}")
    if rc == 0 and report_available(run.dataset_path):
        rep = report_command(run.dataset_path, config, n_resamples=report_resamples)
        _say(f"$ {_display(rep, run)} > report.txt")
        with open(run.report_path, "w", encoding="utf-8") as out:
            rrc = subprocess.run(rep, stdout=out, stdin=subprocess.DEVNULL).returncode
        _say(f"[report] exit code {rrc}" + ("" if rrc == 0 else " (the model is fine; the report failed)"))
    elif rc == 0:
        _say("[report] skipped: the dataset has no status_quo_rules.json and was not converted")
    return rc


if __name__ == "__main__":
    sys.exit(main())
