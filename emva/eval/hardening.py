"""Phase 4 evaluation hardening: every section on every dataset, as one markdown report.

``python -m emva.eval.hardening [--data data/v2 data/v1] [--out reports/phase4.md] [--no-cache]``
(``make report-full``). Sections, each run on every ``--data`` directory (the first is the headline):

- 4.1 rolling-origin evaluation (``emva.eval.rolling``, ADR 0013), with legacy-label comparisons;
- 4.2 subsampling curve, LR vs monotone GBDT (``emva.eval.subsampling``);
- 4.3 customer-sized test set (``emva.eval.subsampling.customer_sized``);
- 4.4 oracle-feature ceiling and the planted context-only gap (``emva.eval.ceiling``, reads ground truth);
- 4.5 calibration decay (``emva.eval.calibration_decay``);
- 4.6 regularisation sweep (``emva.eval.regularisation``);
- interaction detectability (``emva.eval.interactions``);
- the standard report (ground rule 3, ``emva.eval.report``) for each dataset.

With ``--out`` the generated markdown replaces the block between ``MARK_BEGIN`` and ``MARK_END`` in that
file (hand-written text outside the block is kept; a missing file or block is created). Each
(section, dataset) result is cached under ``runs/phase4/cache/`` keyed by the data files, every
``emva``/``baseline`` source file and the parameters, so an unchanged rerun is instant and any code or
data change recomputes. The report states each section's compute time.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from emva.constants import TEST_FROM
from emva.eval import calibration_decay as cd
from emva.eval import ceiling as ce
from emva.eval import regularisation as rg
from emva.eval import rolling as ro
from emva.eval import subsampling as ss
from emva.eval.bootstrap import N_RESAMPLES, SEED, BootstrapCI, auc_ci, paired_auc
from emva.eval.interactions import INTERACTION_COLUMNS, PLANTED_V2, interaction_coef_cis, with_interactions
from emva.eval.report import build_report, md_table
from emva.labels import HORIZON

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "runs" / "phase4" / "cache"
DEFAULT_DATA: tuple[str, ...] = ("data/v2", "data/v1")
MARK_BEGIN = "<!-- BEGIN GENERATED: python -m emva.eval.hardening (make report-full); edits inside are overwritten -->"
MARK_END = "<!-- END GENERATED -->"


@dataclass
class Section:
    """One section's markdown lines for one dataset, key numbers for the summary, and its compute time."""

    lines: list[str]
    facts: dict[str, object] = field(default_factory=dict)
    seconds: float = 0.0


def _ci(ci: BootstrapCI | None, signed: bool = False) -> str:
    """``0.812 [0.790, 0.833]`` (``+0.012 [...]`` when ``signed``); ``n/a`` for None."""
    if ci is None:
        return "n/a"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(ci.point)} [{f.format(ci.lo)}, {f.format(ci.hi)}]"


def _hl(h: ro.Headline | None) -> str:
    return "n/a" if h is None else f"{h.mean:.3f} [{h.lo:.3f}, {h.hi:.3f}]"


def _frozen_score(D: pd.DataFrame, y: pd.Series, train: pd.Series, test: pd.Series,
                  fit_predict: ro.FitPredict = ro.lr_fit_predict) -> np.ndarray:
    """Scores on ``test`` rows of a model fitted on ``train`` rows."""
    return fit_predict(D, y, train)[test.to_numpy()]


# ---------------------------------------------------------------------------------------------------------------
# 4.1 rolling origin
# ---------------------------------------------------------------------------------------------------------------

def rolling_section(F: ro.EvalFrame, n_resamples: int) -> Section:
    """Rolling-origin AUC per split for the headline rule and three comparisons."""
    variants = [
        ("**headline**", ro.horizon_strict(F.horizon_days), "v2", F.D, F.X.y),
        ("comparison", ro.horizon_relaxed(F.horizon_days), "v2", F.D, F.X.y),
        ("comparison", ro.LEGACY_WINDOW, "v2", F.D, F.legacy_y),
        ("baseline reproduction", ro.LEGACY_TO_END, "legacy", F.D_legacy, F.legacy_y),
    ]
    summary, detail, facts = [], [], {}
    for role, spec, feats, D, y in variants:
        res = ro.rolling_origin(spec, D, y, F.X.created_at, n_resamples=n_resamples)
        h = ro.headline(res, n_resamples)
        excluded = [f"{r.split.date()} ({r.status()})" for r in res if not r.sufficient]
        summary.append({"variant": role, "labels / training rule / test window": spec.name, "features": feats,
                        "mean AUC [95% CI]": _hl(h),
                        "per-split range (spread)": "n/a" if h is None else f"{h.min:.3f} to {h.max:.3f} ({h.max - h.min:.3f})",
                        "splits used": 0 if h is None else h.n_splits,
                        "left out": "; ".join(excluded) or "none"})
        detail += [f"**{spec.name}, {feats} features**", "", md_table(ro.results_table(res)), ""]
        facts[spec.name] = None if h is None else (h.mean, h.lo, h.hi, h.min, h.max, h.n_splits)
    facts["headline"] = facts[ro.horizon_strict(F.horizon_days).name]
    facts["baseline_spread"] = facts[ro.LEGACY_TO_END.name]
    facts["legacy_window"] = facts[ro.LEGACY_WINDOW.name]
    return Section([md_table(pd.DataFrame(summary)), "", *detail], facts)


# ---------------------------------------------------------------------------------------------------------------
# 4.2 subsampling, 4.3 customer-sized test set
# ---------------------------------------------------------------------------------------------------------------

def subsampling_section(F: ro.EvalFrame, n_seeds: int) -> Section:
    """Subsampling curves: LR and both GBDTs on horizon labels (test b); LR legacy on legacy labels (test a)."""
    X = F.X
    trb, teb = F.frozen_horizon()
    tra, tea = F.frozen_legacy()
    models = {"LR (pipeline)": ss.lr_model(F.D, X.y), **ss.gbdt_models(X, X.y)}
    long_h = ss.subsample_curve(X.y, trb, teb, models, n_seeds=n_seeds)
    long_l = ss.subsample_curve(F.legacy_y, tra, tea, {"LR, legacy features": ss.lr_model(F.D_legacy, F.legacy_y)},
                                n_seeds=n_seeds)

    def at(long: pd.DataFrame, model: str, n: int) -> tuple[float, float] | None:
        v = long[(long.model == model) & (long.N == n) & ~long.full].auc
        return None if v.empty else (float(v.mean()), float(v.std()))
    facts = {"lr_2000": at(long_h, "LR (pipeline)", 2000), "gbdt_2000": at(long_h, "GBDT monotone", 2000),
             "gbdt_free_2000": at(long_h, "GBDT unconstrained", 2000),
             "legacy_lr_2000": at(long_l, "LR, legacy features", 2000),
             "lr_full": float(long_h[(long_h.model == "LR (pipeline)") & long_h.full].auc.iloc[0]),
             "gbdt_full": float(long_h[(long_h.model == "GBDT monotone") & long_h.full].auc.iloc[0])}
    return Section([
        f"Horizon labels, v2 features. Pool = the pipeline's horizon training set ({int(trb.sum())} leads, "
        f"{int(X.y[trb].sum())} won); test = frozen mature test set (b) ({int(teb.sum())} leads, "
        f"{int(X.y[teb].sum())} won). Mean ± sd AUC over {n_seeds} seeds:", "",
        md_table(ss.curve_table(long_h)), "",
        f"Baseline continuity: legacy labels, legacy features, pool = legacy training set ({int(tra.sum())} leads), "
        f"test = frozen legacy test set (a) ({int(tea.sum())} leads):", "",
        md_table(ss.curve_table(long_l)), ""], facts)


def customer_section(F: ro.EvalFrame, n_resamples: int) -> Section:
    """Customer-sized draws (N = 2000, latest 20% held out) and, for scale, the CI widths of the frozen test sets."""
    X = F.X
    draws = ss.customer_sized(F.D, X.y, HORIZON.eligible(X), X.created_at, n_resamples=n_resamples)
    w = np.array([d.width for d in draws])
    a = np.array([d.auc for d in draws])
    trb, teb = F.frozen_horizon()
    tra, tea = F.frozen_legacy()
    ci_b = auc_ci(X.y[teb].to_numpy(), _frozen_score(F.D, X.y, trb, teb), n_resamples)
    ci_a = auc_ci(F.legacy_y[tea].to_numpy(), _frozen_score(F.D, F.legacy_y, tra, tea), n_resamples)
    rows = [
        {"test set": f"customer-sized holdout (N = {ss.CUSTOMER_N}, latest {ss.CUSTOMER_HOLDOUT:.0%}), mean over "
                     f"{len(draws)} draws", "leads (wins)": f"{draws[0].n_test} ({np.mean([d.wins_test for d in draws]):.0f} mean)",
         "AUC": f"{a.mean():.3f} ± {a.std(ddof=1):.3f} (sd over draws)",
         "95% CI width": f"{w.mean():.3f} (draws {w.min():.3f} to {w.max():.3f})"},
        {"test set": "frozen mature test set (b), horizon labels", "leads (wins)": f"{int(teb.sum())} ({int(X.y[teb].sum())})",
         "AUC": f"{ci_b.point:.3f}", "95% CI width": f"{ci_b.width:.3f}"},
        {"test set": "frozen legacy test set (a), legacy labels", "leads (wins)": f"{int(tea.sum())} ({int(F.legacy_y[tea].sum())})",
         "AUC": f"{ci_a.point:.3f}", "95% CI width": f"{ci_a.width:.3f}"},
    ]
    return Section([md_table(pd.DataFrame(rows)), ""],
                   {"mean_width": float(w.mean()), "mean_auc": float(a.mean()), "sd_auc": float(a.std(ddof=1)),
                    "half_width": float(w.mean() / 2)})


# ---------------------------------------------------------------------------------------------------------------
# 4.4 oracle-feature ceiling
# ---------------------------------------------------------------------------------------------------------------

PIPELINE = "pipeline (v2 features)"
PIPELINE_PERSONA = "pipeline + true persona"


def ceiling_section(F: ro.EvalFrame, n_resamples: int) -> Section:
    """Oracle ceilings on the frozen test sets and the rolling-origin headline, with the planted persona gap."""
    X = F.X
    Fo, p_true = ce.oracle_frame(F)
    sets = ce.available_sets(Fo)
    designs: dict[str, pd.DataFrame] = {PIPELINE: F.D, **{s: ce.oracle_design(Fo, s) for s in sets}}
    persona = ce.ORACLE_PERSONA in sets
    if persona:
        designs[PIPELINE_PERSONA] = F.D.join(ce.persona_indicator(Fo))
    trb, teb = F.frozen_horizon()
    tra, tea = F.frozen_legacy()
    spec = ro.horizon_strict(F.horizon_days)
    scores: dict[str, dict[str, object]] = {}
    rows = []
    for name, D in designs.items():
        sb, sa = _frozen_score(D, X.y, trb, teb), _frozen_score(D, F.legacy_y, tra, tea)
        rr = ro.rolling_origin(spec, D, X.y, X.created_at, n_resamples=n_resamples)
        scores[name] = {"b": sb, "a": sa, "rolling": rr}
        h = ro.headline(rr, n_resamples)
        rows.append({"model": name, "design columns": D.shape[1], "rolling-origin mean [95% CI]": _hl(h),
                     "(b) mature test [95% CI]": _ci(auc_ci(X.y[teb].to_numpy(), sb, n_resamples)),
                     "(a) legacy test [95% CI]": _ci(auc_ci(F.legacy_y[tea].to_numpy(), sa, n_resamples))})
    const = lambda D, y, tr: p_true.to_numpy()  # noqa: E731 (a fixed score needs no fit)
    h_true = ro.headline(ro.rolling_origin(spec, F.D, X.y, X.created_at, const, n_resamples=n_resamples), n_resamples)
    rows.append({"model": "reference: p_close_true (includes noise and reply time)", "design columns": "",
                 "rolling-origin mean [95% CI]": _hl(h_true),
                 "(b) mature test [95% CI]": _ci(auc_ci(X.y[teb].to_numpy(), p_true[teb].to_numpy(), n_resamples)),
                 "(a) legacy test [95% CI]": _ci(auc_ci(F.legacy_y[tea].to_numpy(), p_true[tea].to_numpy(), n_resamples))})

    def gap(a: str, b: str) -> dict[str, object]:
        pb = paired_auc(X.y[teb].to_numpy(), scores[a]["b"], scores[b]["b"], n_resamples)
        pa = paired_auc(F.legacy_y[tea].to_numpy(), scores[a]["a"], scores[b]["a"], n_resamples)
        pr = ro.paired_headline(scores[a]["rolling"], scores[b]["rolling"], n_resamples)
        return {"comparison": f"{b} − {a}", "rolling-origin mean [95% CI]": _ci(pr, True),
                "(b) mature test [95% CI]": _ci(pb.diff, True), "(a) legacy test [95% CI]": _ci(pa.diff, True),
                "_rolling": None if pr is None else (pr.point, pr.lo, pr.hi), "_a": (pa.diff.point, pa.diff.lo, pa.diff.hi),
                "_b": (pb.diff.point, pb.diff.lo, pb.diff.hi)}
    gaps = [gap(PIPELINE, ce.ORACLE_INTERACTIONS), gap(ce.ORACLE_FORMULA, ce.ORACLE_INTERACTIONS)]
    if persona:
        gaps += [gap(ce.ORACLE_INTERACTIONS, ce.ORACLE_PERSONA), gap(PIPELINE, PIPELINE_PERSONA)]
    table = pd.DataFrame(rows)
    facts: dict[str, object] = {"rows": {r["model"]: r for r in rows}, "gaps": {g["comparison"]: g for g in gaps},
                                "persona": gaps[2] if persona else None, "pipeline_persona": gaps[3] if persona else None}
    gap_table = pd.DataFrame([{k: v for k, v in g.items() if not k.startswith("_")} for g in gaps])
    return Section([md_table(table), "", "Paired differences (same rows, shared resamples; rolling = mean of per-split "
                    "paired differences over the sufficient splits):", "", md_table(gap_table), ""], facts)


# ---------------------------------------------------------------------------------------------------------------
# 4.5 calibration decay, 4.6 regularisation, interactions
# ---------------------------------------------------------------------------------------------------------------

def calibration_section(F: ro.EvalFrame) -> Section:
    """Calibration on M+1..M+3 for the headline rule (horizon labels) and for legacy labels."""
    X = F.X
    hs = cd.calibration_decay(ro.horizon_strict(F.horizon_days), F.D, X.y, X.created_at)
    lg = cd.calibration_decay(ro.LEGACY_WINDOW, F.D, F.legacy_y, X.created_at)
    slopes = [s.slope for r in hs for _, s in r.evals if s.n]
    icpts = [s.intercept for r in hs for _, s in r.evals if s.n]
    return Section([
        f"**{ro.horizon_strict(F.horizon_days).name}, v2 features** (the headline rule):", "",
        md_table(cd.decay_table(hs)), "",
        "Deciles of p over each fit's pooled evaluation months, as mean p / observed:", "",
        md_table(cd.decile_table(hs)), "",
        "**Legacy labels (at AS_OF), training = every legacy-labelled lead created by the end of M, v2 features:**", "",
        md_table(cd.decay_table(lg)), ""],
        {"slope_range": (float(np.nanmin(slopes)), float(np.nanmax(slopes))) if slopes else None,
         "intercept_range": (float(np.nanmin(icpts)), float(np.nanmax(icpts))) if icpts else None})


def regularisation_section(F: ro.EvalFrame) -> Section:
    """AUC against C on both frozen test sets and the rolling-origin headline."""
    X = F.X
    tra, tea = F.frozen_legacy()
    trb, teb = F.frozen_horizon()
    a_leg, a_v2, b_v2 = "(a) legacy labels, legacy features", "(a) legacy labels, v2 features", "(b) horizon, v2 features"
    cases = {a_leg: (F.D_legacy, F.legacy_y, tra, tea), a_v2: (F.D, F.legacy_y, tra, tea), b_v2: (F.D, X.y, trb, teb)}
    t = rg.sweep(cases, (ro.horizon_strict(F.horizon_days), F.D, X.y, X.created_at), brier_for=(b_v2,))
    flat = {c: rg.flatness(t, c) for c in [a_leg, a_v2, b_v2, "rolling-origin mean"]}
    shown = t.copy()
    shown["C"] = [f"{c:g}" + (" (pipeline)" if c == rg.LR_C else "") for c in t.C]
    for c in shown.columns[1:]:
        shown[c] = [f"{v:.4f}" for v in t[c]]
    verdict = pd.DataFrame([{"column": c, f"AUC range over C in [{rg.FLAT_RANGE[0]:g}, {rg.FLAT_RANGE[1]:g}]":
                             f"{lo:.4f} to {hi:.4f} ({hi - lo:.4f})", f"flat (< {rg.FLAT_TOLERANCE})": "yes" if ok else "no",
                             "full-grid range": f"{t[c].min():.4f} to {t[c].max():.4f}"} for c, (lo, hi, ok) in flat.items()])
    return Section([md_table(shown), "", md_table(verdict), ""], {"flat": {c: v for c, v in flat.items()}})


def interactions_section(F: ro.EvalFrame, n_resamples: int) -> Section:
    """Coefficient CIs of the two interaction columns and the held-out AUC change from adding them."""
    X = F.X
    DI = with_interactions(F.D, X)
    trb, teb = F.frozen_horizon()
    tra, tea = F.frozen_legacy()
    cis = interaction_coef_cis(DI, X.y, trb, n_resamples)
    coef_rows = []
    for c in INTERACTION_COLUMNS:
        ci = cis[c]
        on = DI[c].eq(1)
        coef_rows.append({"term": c, "planted on v2": f"{PLANTED_V2[c]:+.1f}",
                          "training rows with term = 1 (wins)": f"{int((on & trb).sum())} ({int(X.y[on & trb].sum())})",
                          "coefficient [95% CI]": _ci(ci, True), "CI excludes 0": "yes" if ci.lo > 0 or ci.hi < 0 else "no"})
    spec = ro.horizon_strict(F.horizon_days)
    base_r = ro.rolling_origin(spec, F.D, X.y, X.created_at, with_ci=False)
    int_r = ro.rolling_origin(spec, DI, X.y, X.created_at, with_ci=False)
    pr = ro.paired_headline(base_r, int_r, n_resamples)
    pb = paired_auc(X.y[teb].to_numpy(), _frozen_score(F.D, X.y, trb, teb), _frozen_score(DI, X.y, trb, teb), n_resamples)
    pa = paired_auc(F.legacy_y[tea].to_numpy(), _frozen_score(F.D, F.legacy_y, tra, tea),
                    _frozen_score(DI, F.legacy_y, tra, tea), n_resamples)
    auc_rows = [
        {"evaluation": "rolling-origin mean (headline rule)", "AUC without": f"{ro.mean_auc(base_r):.3f}",
         "AUC with": f"{ro.mean_auc(int_r):.3f}", "with − without [95% CI]": _ci(pr, True)},
        {"evaluation": "(b) mature test, horizon labels", "AUC without": f"{pb.auc_a:.3f}", "AUC with": f"{pb.auc_b:.3f}",
         "with − without [95% CI]": _ci(pb.diff, True)},
        {"evaluation": "(a) legacy test, legacy labels", "AUC without": f"{pa.auc_a:.3f}", "AUC with": f"{pa.auc_b:.3f}",
         "with − without [95% CI]": _ci(pa.diff, True)},
    ]
    return Section([f"Coefficients: formula model + the two terms, fitted on the horizon training set ({int(trb.sum())} "
                    f"leads), {n_resamples} bootstrap refits, seed {SEED}.", "", md_table(pd.DataFrame(coef_rows)), "",
                    md_table(pd.DataFrame(auc_rows)), ""],
                   {"coefs": {c: (cis[c].point, cis[c].lo, cis[c].hi) for c in INTERACTION_COLUMNS},
                    "rolling_gain": None if pr is None else (pr.point, pr.lo, pr.hi),
                    "b_gain": (pb.diff.point, pb.diff.lo, pb.diff.hi), "a_gain": (pa.diff.point, pa.diff.lo, pa.diff.hi)})


def standard_section(F: ro.EvalFrame, n_resamples: int) -> Section:
    """The standard report for this dataset, headings demoted by two levels so it nests under this report."""
    text = build_report(F.data, n_resamples=n_resamples)
    out, fenced = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        out.append("##" + line if line.startswith("#") and not fenced else line)
    return Section([*out, ""])


# ---------------------------------------------------------------------------------------------------------------
# assembly, cache, CLI
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionSpec:
    """A report section: heading, the prose above the per-dataset tables, and how to compute one dataset."""

    key: str
    heading: str
    intro: str
    compute: Callable[[ro.EvalFrame, "Params"], Section]


@dataclass(frozen=True)
class Params:
    """Run parameters (part of the cache key)."""

    n_resamples: int = N_RESAMPLES
    n_seeds: int = ss.N_SEEDS


ROLLING_INTRO = f"""Split at S in {", ".join(s.date().isoformat() for s in ro.SPLITS)} (ADR 0013). **Headline rule:** train on
labelled leads created before S whose H = 120 horizon label is mature at S (`created_at + H <= S`); test on labelled
leads created in [S, S + 1 month) whose label is mature at AS_OF. Labels are the pipeline's default (ghosted-at-H
excluded, stalled censored; the stalled flag reads the CRM state at AS_OF, see `emva/eval/rolling.py`). A split is
left out of the mean when its test set has fewer than {ro.MIN_TEST_LEADS} labelled leads or fewer than
{ro.MIN_TEST_CLASS} of a class; it is still listed. The last mature lead was created 2026-05-27, so the 2026-05-01
window is partial (26 days) and the 2026-06-01 and 2026-07-01 windows have no mature test lead under H = 120.
Mean AUC CI: each split's test set resampled independently (1000 resamples, seeds 0, 1, ...).

Comparisons: *relaxed* = the same test windows but training on every lead created before S that is mature at
AS_OF (look-ahead: this is how the frozen split trains); *legacy labels, one-month test* = baseline labels at AS_OF;
*legacy labels, test = everything from S*, legacy features = the definition that reproduces the review's baseline
spread (0.814 to 0.836 on v1; at S = 2026-05-01 it is exactly the frozen legacy test set)."""

SUBSAMPLING_INTRO = """N training rows drawn without replacement from the training pool (seeds 0-19; the same rows for every
model within a seed), scored on a fixed test set; "full" is one fit on the whole pool. GBDT =
`HistGradientBoostingClassifier` (100 iterations, learning rate 0.1) on one column per v2 feature: the four
ordered features are ranks with a +1 monotone constraint, the rest categorical. The constraint vector (column order =
the GBDT design):

{constraints}

The planted effects of size (201-1000 +1.3 > 1000+ +1.0) and time on page (60-300s +0.4, >600s -0.4) are not
monotone, so the constraint is wrong at the top of those two scales; the unconstrained GBDT shows what it costs.
Baseline reference (review, v1): LR 0.807 ± 0.004 at N = 2000 (legacy labels and features, legacy test set)."""

CUSTOMER_INTRO = """A customer with 2,000 mature labelled leads validates on their most recent 20% (400 leads): draw 2,000
leads from the mature labelled pool (seeds 0-19), sort by `created_at`, fit the formula model on the first 1,600 and
bootstrap the AUC on the last 400 (1000 resamples, seed 0). The width is what that customer's CI would look like."""

CEILING_INTRO = """Formula model (same C, same rows, same splits) fitted on the generator's own inputs instead of the
pipeline's features (`emva/eval/ceiling.py`; ground truth, evaluation only). *truth file only* = the planted
categorical columns of `ground_truth_labels.csv` (true size, free email, true text category, channel, lead ads, true
seniority). *formula* = every planted main effect without the noise term and the sales reply time: those plus the true
company's spend / CRM / hiring, IP country vs the true country and the recorded session inputs at the planted cut
points (no telemetry = its own level: the generator scored consent-declined leads on hidden true behaviour).
*+ interactions* adds LinkedIn × true size 51+ and true senior × form D (planted on v2 only). *+ persona* (v2) adds the
hidden `context_persona`. **The formula + interactions row is the reachable ceiling that replaces the 0.836 figure;
the persona row minus it is the planted context-only gap Phase 6 has to recover.**"""

CALIBRATION_INTRO = """Fit at the end of month M (split = first day of M+1, same training rule as the rolling headline),
evaluate each of M+1, M+2, M+3. Slope = logistic regression of y on logit p (1 = well spread, < 1 = over-confident);
intercept = calibration-in-the-large (y on an offset of logit p; 0 = right level, < 0 = over-predicting). Months with no
mature labelled lead are listed as 0. The legacy-label table is for comparison only: legacy labels are read at AS_OF,
so the youngest months (2026-08, 2026-09) hold mostly fast decisions and are not comparable with older months."""

REGULARISATION_INTRO = f"""L2 strength C over a log grid; everything else as the pipeline's model (C = 0.5). Standing check:
flat = AUC range below {rg.FLAT_TOLERANCE} over C in [{rg.FLAT_RANGE[0]:g}, {rg.FLAT_RANGE[1]:g}]. Baseline (review, v1,
legacy test set): flat, 0.8107 to 0.8138."""

INTERACTIONS_INTRO = """Phase 5 planted LinkedIn × company size 51+ (+0.8) and senior title × form D (-0.8) on v2 and asked
whether the models can detect them. The two products of the pipeline's own features (`channel`, `band`, `seniority`,
`form_variant`; no ground truth) are added to the v2 design. v1 has neither term planted: it is the null control."""

STANDARD_INTRO = """`python -m emva.eval.report --data <dir>` (ground rule 3), unchanged model defaults. Headings demoted to
nest here."""


def _sections() -> list[SectionSpec]:
    return [
        SectionSpec("rolling", "4.1 Rolling-origin evaluation", ROLLING_INTRO, lambda F, p: rolling_section(F, p.n_resamples)),
        SectionSpec("subsampling", "4.2 Subsampling curve: LR vs monotone GBDT",
                    SUBSAMPLING_INTRO.format(constraints=md_table(ss.constraint_table())),
                    lambda F, p: subsampling_section(F, p.n_seeds)),
        SectionSpec("customer", "4.3 Customer-sized test set", CUSTOMER_INTRO, lambda F, p: customer_section(F, p.n_resamples)),
        SectionSpec("ceiling", "4.4 Oracle-feature ceiling and the planted context-only gap", CEILING_INTRO,
                    lambda F, p: ceiling_section(F, p.n_resamples)),
        SectionSpec("calibration", "4.5 Calibration decay", CALIBRATION_INTRO, lambda F, p: calibration_section(F)),
        SectionSpec("regularisation", "4.6 Regularisation sweep", REGULARISATION_INTRO, lambda F, p: regularisation_section(F)),
        SectionSpec("interactions", "Interaction detectability (Phase 5 hand-off)", INTERACTIONS_INTRO,
                    lambda F, p: interactions_section(F, p.n_resamples)),
        SectionSpec("standard", "Standard report", STANDARD_INTRO, lambda F, p: standard_section(F, p.n_resamples)),
    ]


SECTION_KEYS: tuple[str, ...] = tuple(s.key for s in _sections())


def _fingerprint(data: Path) -> str:
    """Hash of the data files (name, size, mtime) and of every emva/baseline source file's contents."""
    h = hashlib.sha256()
    for f in sorted(data.iterdir()):
        if f.is_file():
            st = f.stat()
            h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns};".encode())
    for f in sorted([*(REPO_ROOT / "emva").rglob("*.py"), REPO_ROOT / "baseline" / "emva_score.py"]):
        h.update(f.relative_to(REPO_ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _cache_path(key: str, data: Path, params: Params, fingerprint: str) -> Path:
    digest = hashlib.sha256(f"{key}|{data.resolve()}|{params}|{fingerprint}".encode()).hexdigest()[:20]
    return CACHE_DIR / f"{data.name}-{key}-{digest}.pkl"


def run_sections(datas: list[Path], params: Params, use_cache: bool = True,
                 only: tuple[str, ...] = SECTION_KEYS) -> dict[tuple[str, str], Section]:
    """Compute (or load from cache) every selected section for every dataset; keyed by (section key, data name)."""
    out: dict[tuple[str, str], Section] = {}
    for data in datas:
        fp = _fingerprint(data)
        frame: ro.EvalFrame | None = None
        for spec in _sections():
            if spec.key not in only:
                continue
            path = _cache_path(spec.key, data, params, fp)
            if use_cache and path.exists():
                sec = pickle.loads(path.read_bytes())
                out[(spec.key, data.name)] = sec
                log.info("%s %s: cached (%.1f s originally)", data.name, spec.key, sec.seconds)
                continue
            t0 = time.perf_counter()
            if frame is None:
                frame = ro.prepare(data)
            # Thousands of small fits: one thread each is far faster than OpenMP/BLAS thread pools
            # (HistGradientBoosting on 2,000 rows: 0.2 s single-threaded vs 2.7 s with the default pool).
            with threadpool_limits(limits=1):
                sec = spec.compute(frame, params)
            sec.seconds = time.perf_counter() - t0
            log.info("%s %s: %.1f s", data.name, spec.key, sec.seconds)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(sec))
            out[(spec.key, data.name)] = sec
    return out


def _fmt3(t: tuple | None, signed: bool = False) -> str:
    if t is None:
        return "n/a"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(t[0])} [{f.format(t[1])}, {f.format(t[2])}]"


def summary_lines(results: dict[tuple[str, str], Section], names: list[str]) -> list[str]:
    """The headline table at the top of the report: one row per key number, one column per dataset."""
    def fact(key: str, name: str) -> dict:
        sec = results.get((key, name))
        return {} if sec is None else sec.facts

    rows: list[dict[str, str]] = []

    def add(label: str, fn: Callable[[str], str]) -> None:
        rows.append({"quantity": label, **{f"{n}{' (headline)' if i == 0 else ''}": fn(n) for i, n in enumerate(names)}})

    add("4.1 rolling-origin AUC, headline rule: mean [95% CI]", lambda n: _fmt3(fact("rolling", n).get("headline")))
    add("4.1 ... per-split range (splits used)", lambda n: (lambda h: "n/a" if h is None else f"{h[3]:.3f} to {h[4]:.3f} ({h[5]})")(
        fact("rolling", n).get("headline")))
    add("4.1 legacy labels, one-month test: range", lambda n: (lambda h: "n/a" if h is None else f"{h[3]:.3f} to {h[4]:.3f}")(
        fact("rolling", n).get("legacy_window")))
    add("4.1 baseline reproduction (legacy, test to end): range", lambda n: (lambda h: "n/a" if h is None else f"{h[3]:.3f} to {h[4]:.3f}")(
        fact("rolling", n).get("baseline_spread")))

    def msd(t: tuple | None) -> str:
        return "n/a" if t is None else f"{t[0]:.3f} ± {t[1]:.3f}"
    add("4.2 LR at N = 2000, test (b)", lambda n: msd(fact("subsampling", n).get("lr_2000")))
    add("4.2 GBDT monotone at N = 2000, test (b)", lambda n: msd(fact("subsampling", n).get("gbdt_2000")))
    add("4.2 LR (legacy labels + features) at N = 2000, test (a)", lambda n: msd(fact("subsampling", n).get("legacy_lr_2000")))
    add("4.2 full pool: LR / GBDT monotone, test (b)", lambda n: (lambda f: "n/a" if not f else f"{f['lr_full']:.3f} / {f['gbdt_full']:.3f}")(
        fact("subsampling", n)))
    add("4.3 customer-sized (400-lead holdout) mean CI width", lambda n: (lambda f: "n/a" if not f else f"{f['mean_width']:.3f} (± {f['half_width']:.3f})")(
        fact("customer", n)))

    def ceil(n: str, model: str, col: str) -> str:
        r = fact("ceiling", n).get("rows", {}).get(model)
        return "n/a" if r is None else str(r[col])
    cols = [("(b)", "(b) mature test [95% CI]", "_b"), ("(a)", "(a) legacy test [95% CI]", "_a"),
            ("rolling", "rolling-origin mean [95% CI]", "_rolling")]
    for model, label_ in [(PIPELINE, "pipeline (v2 features)"), (ce.ORACLE_INTERACTIONS, "**ceiling = formula + interactions**"),
                          (ce.ORACLE_PERSONA, "formula + interactions + persona")]:
        for short, col, _ in cols:
            add(f"4.4 {label_}: {short}", lambda n, m=model, c=col: ceil(n, m, c))
    for fact_key, label_ in [("persona", "**planted context-only gap** (persona ceiling − formula ceiling)"),
                             ("pipeline_persona", "pipeline + true persona − pipeline")]:
        for short, _, k in cols:
            add(f"4.4 {label_}: {short}", lambda n, f=fact_key, k=k: _fmt3((fact("ceiling", n).get(f) or {}).get(k), True))
    for short, _, k in cols:
        add(f"4.4 ceiling − pipeline: {short}", lambda n, k=k: _fmt3(
            (fact("ceiling", n).get("gaps", {}).get(f"{ce.ORACLE_INTERACTIONS} − {PIPELINE}") or {}).get(k), True))
    add("4.5 calibration slope range (headline rule)", lambda n: (lambda r: "n/a" if r is None else f"{r[0]:.2f} to {r[1]:.2f}")(
        fact("calibration", n).get("slope_range")))
    add("4.5 calibration intercept range (headline rule)", lambda n: (lambda r: "n/a" if r is None else f"{r[0]:+.2f} to {r[1]:+.2f}")(
        fact("calibration", n).get("intercept_range")))
    add("4.6 flat over C in [0.1, 10]? (rolling / (a) legacy features)", lambda n: (lambda f: "n/a" if not f else
        f"{'yes' if f['rolling-origin mean'][2] else 'no'} / {'yes' if f['(a) legacy labels, legacy features'][2] else 'no'}")(
        fact("regularisation", n).get("flat")))
    for c in INTERACTION_COLUMNS:
        add(f"interaction {c}: coefficient [95% CI]", lambda n, c=c: _fmt3(fact("interactions", n).get("coefs", {}).get(c), True))
    add("interactions: rolling AUC gain", lambda n: _fmt3(fact("interactions", n).get("rolling_gain"), True))
    return ["## Summary (on simulated data)", "", md_table(pd.DataFrame(rows)), ""]


def timing_lines(results: dict[tuple[str, str], Section]) -> list[str]:
    """Compute time per section and dataset (from the run that produced each cached result)."""
    rows = [{"section": k, "data": n, "seconds": f"{s.seconds:.1f}"} for (k, n), s in results.items()]
    total = sum(s.seconds for s in results.values())
    return ["## Runtime", "", f"Total compute {total:.0f} s ({total / 60:.1f} min) on the machine that produced the cache "
            "entries (single-threaded fits); an unchanged rerun reads `runs/phase4/cache/` in seconds and writes the "
            "same text.", "", md_table(pd.DataFrame(rows)), ""]


def build(datas: list[Path], params: Params = Params(), use_cache: bool = True) -> str:
    """The whole generated block (without the markers) as markdown."""
    results = run_sections(datas, params, use_cache)
    names = [d.name for d in datas]
    out = ["# Phase 4 evaluation hardening (generated)", "",
           f"Data: {', '.join(f'`{d}`' for d in datas)} ({names[0]} is the headline). All numbers are on simulated data. "
           f"Model = the pipeline's defaults (horizon labels H = 120, v2 features, L2 LR C = 0.5) unless a row says "
           f"otherwise. (a) = frozen legacy test set (legacy labels, created on or after {TEST_FROM}); (b) = frozen mature "
           f"test set (`FROZEN_HORIZON`). CIs: percentile bootstrap, {params.n_resamples} resamples.", "",
           *summary_lines(results, names)]
    for spec in _sections():
        out += [f"## {spec.heading}", "", spec.intro, ""]
        for i, n in enumerate(names):
            sec = results.get((spec.key, n))
            if sec is not None:
                out += [f"### {n}{' (headline)' if i == 0 else ''}", "", *sec.lines]
    out += timing_lines(results)
    return "\n".join(out).rstrip() + "\n"


def write_block(path: Path, block: str) -> None:
    """Replace the generated block in ``path`` (or append it / create the file when there is none)."""
    new = f"{MARK_BEGIN}\n{block}{MARK_END}\n"
    if not path.exists():
        path.write_text(new)
        return
    text = path.read_text()
    if MARK_BEGIN in text and MARK_END in text:
        head, rest = text.split(MARK_BEGIN, 1)
        tail = rest.split(MARK_END, 1)[1].lstrip("\n")
        path.write_text(head + new + ("\n" + tail if tail else ""))
    else:
        path.write_text(text.rstrip("\n") + "\n\n" + new)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: print the report, or write it into ``--out`` between the markers."""
    ap = argparse.ArgumentParser(prog="python -m emva.eval.hardening")
    ap.add_argument("--data", nargs="+", default=list(DEFAULT_DATA), help="data directories; the first is the headline")
    ap.add_argument("--out", type=Path, default=None, help="markdown file whose generated block is replaced")
    ap.add_argument("--no-cache", action="store_true", help="recompute every section")
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--n-seeds", type=int, default=ss.N_SEEDS)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    block = build([Path(d) for d in a.data], Params(a.n_resamples, a.n_seeds), not a.no_cache)
    if a.out is None:
        print(block)
    else:
        write_block(a.out, block)
        log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
