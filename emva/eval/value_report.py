"""Value-scale report (plan 3.2) and click-ID coverage (plan 3.5), sections of the standard report.

``value_report_sections`` fits each transform in ``REPORT_TRANSFORMS`` on the candidate's training leads
(as ``emva.pipeline.run`` does for ``value_at_submit``) and, for every test set and for all scored leads,
tabulates the transformed value's scale (p1, p50, p90, p99, max, max/median, top-1% share) and, on test sets,
the top-20% revenue captured when ranking by it. Revenue is recorded deal value (ADR 0002); capture is the
tie-averaged share (cap, floor and tiers create ties), next to the baseline's own capture by p×value.

Every transform is monotone, so a transform changes the top-20% capture only through ties at the cut; the
table therefore also shows ``value / revenue`` (total transformed value over total recorded revenue on the
test set), which shows how far a transform moves the value off the revenue scale that tROAS bidding assumes.

``click_id_coverage`` counts, per paid channel, how scored leads could be matched back to the ad platform.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from emva.constants import TOP_FRACTION
from emva.eval.metrics import tie_averaged_top_share, ties_at_cut
from emva.features import channel
from emva.pipeline import PipelineResult
from emva.value_transform import IDENTITY, Compression, ValueTransform

# Plan 3.2's comparison, plus the default (ADR 0012) last.
REPORT_TRANSFORMS: tuple[ValueTransform, ...] = (
    IDENTITY,
    ValueTransform(floor=None, compression=Compression.NONE),
    ValueTransform(compression=Compression.NONE),
    ValueTransform(floor=None, compression=Compression.SQRT),
    ValueTransform(floor=None, compression=Compression.LOG),
    ValueTransform(cap_percentile=None, floor=None, compression=Compression.NONE, tiers=5),
    ValueTransform(cap_percentile=None, floor=None, compression=Compression.NONE, tiers=10),
    ValueTransform(),
)
# Plan 3.2 acceptance: keep this share of the baseline's top-20% revenue capture ...
MIN_CAPTURE_RETENTION: float = 0.95
# ... while max/median falls below this.
MAX_OVER_MEDIAN_TARGET: float = 20.0

# Paid channels (``emva.features.channel``) and the click identifiers each platform returns on the landing URL.
# Lead-form leads (``meta_leadads``) carry the platform's own lead id instead.
PAID_CHANNELS: tuple[str, ...] = ("google", "meta", "linkedin", "chatgpt", "meta_leadads")
CLICK_ID_COLUMNS: tuple[str, ...] = ("gclid", "gbraid", "wbraid", "fbclid", "li_fat_id", "oppref")
LEAD_FORM_ID_COLUMN: str = "meta_lead_id"


def scale_stats(values: np.ndarray | pd.Series) -> dict[str, float]:
    """p1, p50, p90, p99, max, max/median and the share of the total held by the top 1% of ``values``.

    Raises ``ValueError`` for an empty vector or one whose median or total is not positive (ratios undefined).
    """
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    if len(v) == 0 or np.median(v) <= 0 or v.sum() <= 0:
        raise ValueError("value scale needs a non-empty value vector with a positive median and total")
    k = max(1, int(len(v) * 0.01))
    p1, p50, p90, p99 = np.percentile(v, [1, 50, 90, 99])
    return {"p1": float(p1), "p50": float(p50), "p90": float(p90), "p99": float(p99), "max": float(v[0]),
            "max_over_median": float(v[0] / p50), "top1pct_share": float(v[:k].sum() / v.sum())}


def _scale_cells(values: np.ndarray | pd.Series) -> dict[str, str]:
    """``scale_stats`` formatted as table cells."""
    s = scale_stats(values)
    return {"p1": f"{s['p1']:.0f}", "p50": f"{s['p50']:.0f}", "p90": f"{s['p90']:.0f}", "p99": f"{s['p99']:.0f}",
            "max": f"{s['max']:.0f}", "max/median": f"{s['max_over_median']:.1f}×",
            "top-1% share": f"{s['top1pct_share']:.1%}"}


def transform_table(result: PipelineResult, ids: pd.Index, revenue: pd.Series | None,
                    baseline_value: pd.Series | None) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """One row per transform (plus the baseline's value first, unless ``baseline_value`` is None) on the leads ``ids``.

    Transforms are fitted on ``result``'s training leads. With ``revenue`` (recorded revenue indexed like
    ``ids``) the table adds top-20% capture, ties at the cut, retention vs the baseline's capture and value /
    revenue. Also returns, per transform row, the raw numbers the acceptance check needs. Raises ``ValueError``
    when the test set has no recorded revenue or the baseline captures none of it (ratios undefined). Without a
    baseline (a converted dataset, ADR 0022) the "vs baseline" column and the retention numbers are left out, and a
    transform that cannot be fitted on the candidate's values (e.g. tiers over a constant score) gets a row saying
    why instead of raising.
    """
    X, tr = result.X, result.train
    rows: list[dict[str, str]] = []
    raw: list[dict[str, float]] = []
    base_capture = None
    if revenue is not None:
        if revenue.sum() <= 0:
            raise ValueError("the test set has no recorded revenue; capture and value / revenue are undefined")
        if baseline_value is not None:
            base_capture = tie_averaged_top_share(revenue, baseline_value.loc[ids].values)
            if base_capture <= 0:
                raise ValueError("the baseline captures no revenue in its top 20%; retention vs baseline is undefined")
    entries = [] if baseline_value is None else [("baseline (identity)", baseline_value.loc[ids].values)]
    for t in REPORT_TRANSFORMS:
        try:
            entries.append((f"candidate: {t.describe()}", t.fit(X.value_formula[tr]).apply(X.value_formula.loc[ids])))
        except ValueError as e:
            if baseline_value is not None:
                raise
            entries.append((f"candidate: {t.describe()}", f"not fitted: {e}"))
    for name, v in entries:
        if isinstance(v, str):
            rows.append({"value": name, "p1": v})
            raw.append({})
            continue
        row = {"value": name, **_scale_cells(v)}
        stats = scale_stats(v)
        if revenue is not None:
            capture = tie_averaged_top_share(revenue, v)
            row["top-20% revenue"] = f"{capture:.3f}"
            row["ties at cut"] = str(ties_at_cut(v))
            if base_capture is not None:
                row["vs baseline"] = f"{capture / base_capture:.1%}"
            row["value / revenue"] = f"{v.sum() / revenue.sum():.2f}"
            stats |= {"capture": capture}
            if base_capture is not None:
                stats["retention"] = capture / base_capture
        rows.append(row)
        raw.append(stats)
    return pd.DataFrame(rows), raw if baseline_value is None else raw[1:]


def click_id_coverage(X: pd.DataFrame) -> pd.DataFrame:
    """Per paid channel: leads, and the share with a click id, with only ``fbp``, a lead-form id, or none of them.

    Needs the raw lead columns (``utm_medium``, ``utm_source``, ``CLICK_ID_COLUMNS``, ``fbp``,
    ``meta_lead_id``, ``email``). "none" leads can be matched only by hashed email (and phone when given).
    The last rows total the landing-page paid channels (all but lead forms) and all paid channels.
    """
    ch = pd.Series(channel(X.utm_medium, X.utm_source), index=X.index)
    click = X[list(CLICK_ID_COLUMNS)].notna().any(axis=1)
    lead_form = X[LEAD_FORM_ID_COLUMN].notna()
    fbp_only = ~click & ~lead_form & X.fbp.notna()
    none = ~click & ~lead_form
    groups = [(c, ch == c) for c in PAID_CHANNELS]
    groups += [("landing-page paid (all but meta_leadads)", ch.isin([c for c in PAID_CHANNELS if c != "meta_leadads"])),
               ("all paid", ch.isin(PAID_CHANNELS))]
    rows = []
    for name, m in groups:
        rows.append({"channel": name, "leads": int(m.sum()), "click id": f"{click[m].mean():.1%}",
                     "lead-form id": f"{lead_form[m].mean():.1%}", "no id: fbp present": f"{fbp_only[m].mean():.1%}",
                     "no id at all": f"{none[m].mean():.1%}", "email present": f"{X.email[m].notna().mean():.1%}",
                     "phone present": f"{X.phone[m].notna().mean():.1%}"})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    """GitHub markdown table of ``df`` (cells as given)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def value_report_sections(result: PipelineResult, tests: Sequence[tuple[str, pd.Series]],
                          baseline_value: pd.Series | None) -> list[str]:
    """Markdown lines: the value-scale comparison per test set and over all scored leads, an acceptance summary
    (plan 3.2), and click-ID coverage.

    ``tests`` are ``(title, recorded revenue indexed by lead_id)`` pairs; ``baseline_value`` is the baseline's
    ``value_formula`` for every scored lead, or None when the report has no baseline (a converted dataset,
    ADR 0022): then the baseline row and the acceptance summary (defined against the baseline) are replaced by a
    note, and a test set without recorded revenue gets a note instead of its table.
    """
    out = ["## Value transforms (plan 3.2)", "",
           f"Candidate value = `value_formula` (p × E[deal value] × margin) through each transform, fitted on the "
           f"candidate's {int(result.train.sum())} training leads. Top-20% revenue = tie-averaged share of recorded "
           f"won deal value in the top {TOP_FRACTION:.0%} by value; vs baseline = that share over the baseline's own "
           "(p×value, identity). value / revenue = total value over total recorded revenue. Every transform is "
           "monotone, so capture moves only through ties at the cut.", ""]
    all_ids = result.X.index
    all_table, all_raw = transform_table(result, all_ids, None, baseline_value)
    verdicts: dict[str, list[bool]] = {} if baseline_value is None else {
        t.describe(): [s["max_over_median"] < MAX_OVER_MEDIAN_TARGET] for t, s in zip(REPORT_TRANSFORMS, all_raw)}
    for title, revenue in tests:
        if baseline_value is None and revenue.sum() <= 0:
            out += [f"### {title}", "", "No recorded revenue on this test set: capture is undefined.", ""]
            continue
        table, raw = transform_table(result, revenue.index, revenue, baseline_value)
        out += [f"### {title}", "", _md(table), ""]
        if baseline_value is not None:
            for t, s in zip(REPORT_TRANSFORMS, raw):
                verdicts[t.describe()] += [s["retention"] >= MIN_CAPTURE_RETENTION,
                                           s["max_over_median"] < MAX_OVER_MEDIAN_TARGET]
    out += ["### All scored leads", "", _md(all_table), "",
            f"### Acceptance (plan 3.2): ≥ {MIN_CAPTURE_RETENTION:.0%} of the baseline's capture on every test set "
            f"and max/median < {MAX_OVER_MEDIAN_TARGET:g}× on every test set and on all scored leads", ""]
    if baseline_value is None:
        out += ["Not computed: the criterion is defined against the baseline, which this report omits.", ""]
    else:
        out += [_md(pd.DataFrame([{"transform": k, "result": "PASS" if all(v) else "fail"}
                                  for k, v in verdicts.items()])), ""]
    out += ["## Click-ID coverage (plan 3.5)", "",
            "Scored leads per paid channel by the identifier an upload could match on (see "
            "`docs/platform_contract.md`): a click id (" + ", ".join(CLICK_ID_COLUMNS) + "), else the lead-form id, "
            "else nothing but hashed email / phone (Meta: plus the `fbp` browser cookie).", "",
            _md(click_id_coverage(result.X)), ""]
    return out


__all__ = ["REPORT_TRANSFORMS", "click_id_coverage", "scale_stats", "transform_table", "value_report_sections"]
