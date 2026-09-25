"""Run a named experiment end to end: ``python scripts/run_experiments.py NAME [--data DIR]``.

Writes the pipeline's outputs to ``runs/NAME/`` and the standard report (with that
experiment's model as the candidate) to ``runs/NAME/report.md``. Each experiment is a label
definition: ``baseline`` is the legacy labels (equal to the frozen baseline), ``horizon`` the
Phase 1 default, and the others the Phase 1 comparison flags.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from emva.eval.report import build_report  # noqa: E402 (needs the repo on sys.path)
from emva.labels import HORIZON, LEGACY, LabelConfig  # noqa: E402
from emva.pipeline import run, write_outputs  # noqa: E402

EXPERIMENTS: dict[str, LabelConfig] = {
    "baseline": LEGACY,
    "horizon": HORIZON,
    "horizon-include-ghosted": LabelConfig(include_ghosted=True),
    "horizon-stalled-as-lost": LabelConfig(stalled_as_lost=True),
    "horizon-both-flags": LabelConfig(include_ghosted=True, stalled_as_lost=True),
}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(EXPERIMENTS))
    ap.add_argument("--data", default=str(REPO / "data" / "v1"))
    a = ap.parse_args()
    out = REPO / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    labels = EXPERIMENTS[a.name]
    result = run(Path(a.data), labels=labels)
    print(result.summary.to_string(index=False))
    write_outputs(result, out)
    (out / "report.md").write_text(build_report(a.data, labels=labels) + "\n")
    print(f"wrote {out}/weights.csv, scores.csv, report.md")


if __name__ == "__main__":
    main()
