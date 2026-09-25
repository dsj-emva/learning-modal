"""``make baseline``: prove the frozen baseline and the ``emva`` package reproduce the published numbers.

1. Run ``baseline/emva_score.py --data data/v1`` into a temp dir; assert AUC 0.814, Brier 0.1006,
   top-20% wins 0.571, top-20% revenue 0.795, and weights.csv canonically equal to
   ``baseline/weights.csv`` (see ``emva.eval.regression`` for why not byte-equal).
2. Run ``python -m emva --data data/v1 --label-mode legacy`` into another temp dir (the default
   horizon labels are a deliberate behaviour change, plan Phase 1); assert the same metrics, the
   same canonical weights, and byte-identical weights.csv and scores.csv to step 1's run.

Exits 1 with every failure listed on any mismatch.
"""
from __future__ import annotations

import argparse
import filecmp
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from emva.eval.regression import (  # noqa: E402 (needs the repo on sys.path)
    FROZEN_WEIGHTS,
    parse_summary,
    read_weights,
    reordered_rows,
    summary_mismatches,
    weights_mismatches,
)


def _run(cmd: list[str], cwd: Path) -> str:
    """Run ``cmd`` in ``cwd`` and return its stdout; exit with its stderr if it fails."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"FAIL: {' '.join(cmd)} exited {proc.returncode}\n{proc.stderr}")
    return proc.stdout


def check(data: Path) -> list[str]:
    """Run both pipelines and return a list of failures (empty = pass)."""
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as b_out, tempfile.TemporaryDirectory() as e_out:
        b_stdout = _run([sys.executable, "emva_score.py", "--data", str(data), "--out", b_out], REPO / "baseline")
        print("baseline/emva_score.py:\n" + b_stdout)
        failures += [f"baseline {m}" for m in summary_mismatches(parse_summary(b_stdout))]
        failures += [f"baseline weights.csv {m}" for m in weights_mismatches(read_weights(f"{b_out}/weights.csv"),
                                                                          read_weights(FROZEN_WEIGHTS))]
        moved = reordered_rows(f"{b_out}/weights.csv", FROZEN_WEIGHTS)
        print("baseline weights.csv vs committed: "
              + ("byte-identical" if filecmp.cmp(f"{b_out}/weights.csv", FROZEN_WEIGHTS, shallow=False)
                 else f"values equal, tie order differs for rows {moved}"))

        e_stdout = _run([sys.executable, "-m", "emva", "--data", str(data), "--out", e_out, "--label-mode", "legacy"], REPO)
        print("\npython -m emva:\n" + e_stdout)
        failures += [f"emva {m}" for m in summary_mismatches(parse_summary(e_stdout))]
        failures += [f"emva weights.csv {m}" for m in weights_mismatches(read_weights(f"{e_out}/weights.csv"),
                                                                      read_weights(FROZEN_WEIGHTS))]
        for name in ("weights.csv", "scores.csv"):
            same = filecmp.cmp(f"{b_out}/{name}", f"{e_out}/{name}", shallow=False)
            print(f"emva {name} vs baseline run: {'byte-identical' if same else 'DIFFERENT'}")
            if not same:
                failures.append(f"emva {name} is not byte-identical to the baseline run's")
    return failures


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data" / "v1"))
    a = ap.parse_args()
    failures = check(Path(a.data).resolve())
    if failures:
        print("\nFAIL: make baseline\n  " + "\n  ".join(failures))
        raise SystemExit(1)
    print("\nPASS: baseline and emva/ both reproduce AUC 0.814, Brier 0.1006, top-20% wins 0.571, "
          "revenue 0.795 and the frozen weights.")


if __name__ == "__main__":
    main()
