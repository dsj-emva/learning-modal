"""CLI: ``python -m emva --data data/v1 --out DIR [--context FILE] [--margin 1.0] [label options] [--feature-set]``.

With ``--label-mode legacy --feature-set legacy`` the arguments, stdout and output files
(``weights.csv``, ``scores.csv``) are the same as ``baseline/emva_score.py`` (except ``--data``
defaults to ``data/v1``). The default ``--label-mode horizon`` uses the fixed-horizon label
(``--horizon-days``, ``--include-ghosted``, ``--stalled-as-lost``; see ``emva.labels``) and
adds ``won_within_h``, ``label_source`` and ``matured_at`` to ``scores.csv``, then the two-stage upload
columns ``value_at_submit``, ``value_at_submit_ts``, ``value_at_close``, ``value_at_close_ts`` (plan 3.3;
value options ``--value-cap-percentile``, ``--value-floor``, ``--value-compression``, ``--value-tiers``, see
``emva.value_transform``). The default ``--feature-set v2`` uses the plan Phase 2 features (see ``emva.features``).
"""
from __future__ import annotations

import argparse

from emva.cli import (
    add_feature_arguments,
    add_label_arguments,
    add_value_arguments,
    feature_set,
    label_config,
    value_transform,
)
from emva.pipeline import run, write_outputs


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the pipeline, print the summary and write outputs."""
    ap = argparse.ArgumentParser(prog="python -m emva")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--context", default=None, help="CSV with lead_id,context_score (0-1) from emva.context.agent")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--out", default=".")
    add_label_arguments(ap)
    add_feature_arguments(ap)
    add_value_arguments(ap)
    a = ap.parse_args(argv)

    labels = label_config(ap, a)
    result = run(a.data, margin=a.margin, context=a.context, labels=labels, features=feature_set(a),
                 value_transform=value_transform(ap, a, labels))
    for line in result.messages:
        print(line)
    if not result.summary.empty:
        print(result.summary.to_string(index=False))
    write_outputs(result, a.out)


if __name__ == "__main__":
    main()
