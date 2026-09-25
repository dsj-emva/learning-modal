"""Run a named experiment end to end: ``python scripts/run_experiments.py NAME [--data DIR]``.

Writes the pipeline's outputs to ``runs/NAME/`` and the standard report to
``runs/NAME/report.md``. Phase 0 has one experiment, ``baseline`` (the current ``emva``
pipeline with default settings); later phases register theirs in ``EXPERIMENTS``.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from emva.eval.report import build_report  # noqa: E402 (needs the repo on sys.path)
from emva.labels import LEGACY  # noqa: E402
from emva.pipeline import run, write_outputs  # noqa: E402


def baseline(data: Path, out: Path) -> None:
    """The ``emva`` pipeline with default settings (equal to the frozen baseline in Phase 0)."""
    result = run(data, labels=LEGACY)
    print(result.summary.to_string(index=False))
    write_outputs(result, out)


EXPERIMENTS: dict[str, Callable[[Path, Path], None]] = {"baseline": baseline}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(EXPERIMENTS))
    ap.add_argument("--data", default=str(REPO / "data" / "v1"))
    a = ap.parse_args()
    out = REPO / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS[a.name](Path(a.data), out)
    (out / "report.md").write_text(build_report(a.data) + "\n")
    print(f"wrote {out}/weights.csv, scores.csv, report.md")


if __name__ == "__main__":
    main()
