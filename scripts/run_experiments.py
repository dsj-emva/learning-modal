"""Run a named experiment end to end: ``python scripts/run_experiments.py NAME [--data DIR]``.

Writes the pipeline's outputs to ``runs/NAME/`` and the standard report (with that
experiment's model as the candidate) to ``runs/NAME/report.md``. Each experiment is a label
definition and a feature set: ``baseline`` is legacy labels and legacy features (equal to the
frozen baseline), ``horizon`` the current default (horizon labels, v2 features), the
``horizon-*`` flag variants use v2 features, ``legacy-labels`` isolates the Phase 2 feature
change and ``horizon-legacy-features`` the Phase 1 label change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from emva.eval.report import build_report  # noqa: E402 (needs the repo on sys.path)
from emva.features import FeatureSet  # noqa: E402
from emva.labels import HORIZON, LEGACY, LabelConfig  # noqa: E402
from emva.pipeline import run, write_outputs  # noqa: E402

EXPERIMENTS: dict[str, tuple[LabelConfig, FeatureSet]] = {
    "baseline": (LEGACY, FeatureSet.LEGACY),
    "horizon": (HORIZON, FeatureSet.V2),
    "horizon-include-ghosted": (LabelConfig(include_ghosted=True), FeatureSet.V2),
    "horizon-stalled-as-lost": (LabelConfig(stalled_as_lost=True), FeatureSet.V2),
    "horizon-both-flags": (LabelConfig(include_ghosted=True, stalled_as_lost=True), FeatureSet.V2),
    "legacy-labels": (LEGACY, FeatureSet.V2),
    "horizon-legacy-features": (HORIZON, FeatureSet.LEGACY),
}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(EXPERIMENTS))
    ap.add_argument("--data", default=str(REPO / "data" / "v1"))
    a = ap.parse_args()
    out = REPO / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    labels, features = EXPERIMENTS[a.name]
    result = run(Path(a.data), labels=labels, features=features)
    print(result.summary.to_string(index=False))
    write_outputs(result, out)
    (out / "report.md").write_text(build_report(a.data, labels=labels, features=features) + "\n")
    print(f"wrote {out}/weights.csv, scores.csv, report.md")


if __name__ == "__main__":
    main()
