"""Phase 4 evaluation hardening: every section on every dataset, as one markdown report.

``python -m emva.eval.hardening [--data data/v2 data/v1] [--out reports/phase4.md] [--no-cache]``
(``make report-full``). Sections, each run on every ``--data`` directory (the first is the headline):

- 4.1 rolling-origin evaluation (``emva.eval.rolling``, ADR 0013), with legacy-label comparisons;
- 4.2 subsampling curve, LR vs monotone GBDT (``emva.eval.subsampling``);
- 4.3 customer-sized test set (``emva.eval.subsampling.customer_sized``);
- 4.4 oracle-feature ceiling and the planted context-only gap (``emva.eval.ceiling``, reads ground truth);
- 4.5 calibration decay (``emva.eval.calibration_decay``);
- 4.6 regularisation sweep (``emva.eval.regularisation``);
- interaction detectability with its simulated sampling distribution (``emva.eval.interactions``, the
  simulation reads ground truth);
- the standard report (ground rule 3, ``emva.eval.report``) for each dataset.

With ``--out`` the generated markdown replaces the block between ``MARK_BEGIN`` and ``MARK_END`` in that
file (hand-written text outside the block is kept; a missing file or block is created). Each
(section, dataset) result is cached under ``runs/phase4/cache/``, keyed by the contents (sha256) of the
data directory's files, of ``requirements.txt``, of every ``emva``/``baseline`` source file, the installed
numpy / pandas / scikit-learn / scipy versions and the run parameters: an unchanged rerun is instant and any
code, data or library change recomputes (R16: ``make report-full`` is the inheritance mechanism). The report
states each section's compute time.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from emva.constants import AS_OF, HORIZON_DAYS, LR_C, TEST_FROM
from emva.eval import calibration_decay as cd
from emva.eval import ceiling as ce
from emva.eval import interactions as it
from emva.eval import regularisation as rg
from emva.eval import rolling as ro
from emva.eval import subsampling as ss
from emva.eval.bootstrap import N_RESAMPLES, SEED, auc_ci, paired_auc
from emva.eval.report import build_report, md_table
from emva.labels import HORIZON

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "runs" / "phase4" / "cache"
DEFAULT_DATA: tuple[str, ...] = ("data/v2", "data/v1")
MARK_BEGIN = "<!-- BEGIN GENERATED: python -m emva.eval.hardening (make report-full); edits inside are overwritten -->"
MARK_END = "<!-- END GENERATED -->"

# Section keys.
K_ROLLING, K_SUBSAMPLING, K_CUSTOMER, K_CEILING = "rolling", "subsampling", "customer", "ceiling"
K_CALIBRATION, K_REGULARISATION, K_INTERACTIONS, K_STANDARD = "calibration", "regularisation", "interactions", "standard"
# Fact keys (Section.facts) read by the summary table.
F_HEADLINE, F_RELAXED, F_LEGACY_WINDOW, F_BASELINE_SPREAD = "headline", "relaxed", "legacy_window", "baseline_spread"
F_LR_2000, F_GBDT_2000, F_GBDT_FREE_2000, F_LEGACY_LR_2000 = "lr_2000", "gbdt_2000", "gbdt_free_2000", "legacy_lr_2000"
F_LR_FULL, F_GBDT_FULL = "lr_full", "gbdt_full"
F_CUSTOMER = "customer"
F_CEILING_ROWS, F_GAPS = "rows", "gaps"
F_SLOPES, F_INTERCEPTS = "slopes", "intercepts"
F_FLAT = "flat"
F_COEFS, F_ROLLING_GAIN, F_NULL = "coefs", "rolling_gain", "null"
# Evaluation keys inside the ceiling rows and gaps.
E_B, E_A, E_ROLLING = "(b) mature test", "(a) legacy test", "rolling-origin mean"
SUMMARY_N: int = 2000            # the subsampling size the summary quotes (the review's reference point)
PIPELINE = "pipeline (v2 features)"
PIPELINE_PERSONA = "pipeline + true persona"
P_TRUE = "reference: p_close_true (includes noise and reply time)"
ROLLING_COL = "rolling-origin mean"


class HasCI(Protocol):
    """Anything with a point estimate and CI bounds (``BootstrapCI``, ``rolling.Headline``)."""

    point: float
    lo: float
    hi: float


@dataclass(frozen=True)
class MeanSd:
    """Mean and sd of AUC over subsampling seeds."""

    mean: float
    sd: float


@dataclass(frozen=True)
class Span:
    """Smallest and largest of a set of values."""

    lo: float
    hi: float


@dataclass(frozen=True)
class Customer:
    """Customer-sized draws: mean CI width, mean and sd of AUC over draws."""

    mean_width: float
    mean_auc: float
    sd_auc: float


@dataclass
class Section:
    """One section's markdown lines for one dataset, key numbers for the summary, and its compute time."""

    lines: list[str]
    facts: dict[str, object] = field(default_factory=dict)
    seconds: float = 0.0


def fmt_ci(x: HasCI | None, signed: bool = False) -> str:
    """``0.812 [0.790, 0.833]`` (``+0.012 [...]`` when ``signed``) for a CI or headline; ``n/a`` for None."""
    if x is None:
        return "n/a"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(x.point)} [{f.format(x.lo)}, {f.format(x.hi)}]"


def _frozen_score(D: pd.DataFrame, y: pd.Series, train: pd.Series, test: pd.Series,
                  fit_predict: ro.FitPredict = ro.lr_fit_predict) -> np.ndarray:
    """Scores on ``test`` rows of a model fitted on ``train`` rows."""
    return fit_predict(D, y, train)[test.to_numpy()]


def mature_cutoff(horizon_days: int = HORIZON_DAYS) -> pd.Timestamp:
    """Leads created before this instant are mature at AS_OF (``created_at + H <= AS_OF``)."""
    return AS_OF - pd.Timedelta(days=horizon_days)


# ---------------------------------------------------------------------------------------------------------------
# 4.1 rolling origin
# ---------------------------------------------------------------------------------------------------------------

def rolling_section(F: ro.EvalFrame, n_resamples: int) -> Section:
    """Rolling-origin AUC per split for the headline rule and three comparisons."""
    variants = [
        (F_HEADLINE, "**headline**", ro.horizon_strict(F.horizon_days), "v2", F.D, F.X.y),
        (F_RELAXED, "comparison", ro.horizon_relaxed(F.horizon_days), "v2", F.D, F.X.y),
        (F_LEGACY_WINDOW, "comparison", ro.LEGACY_WINDOW, "v2", F.D, F.legacy_y),
        (F_BASELINE_SPREAD, "baseline reproduction", ro.LEGACY_TO_END, "legacy", F.D_legacy, F.legacy_y),
    ]
    summary, detail, facts = [], [], {}
    for key, role, spec, feats, D, y in variants:
        res = ro.rolling_origin(spec, D, y, F.X.created_at, n_resamples=n_resamples)
        h = ro.headline(res, n_resamples)
        excluded = [f"{r.split.date()} ({r.status()})" for r in res if not r.sufficient]
        summary.append({"variant": role, "labels / training rule / test window": spec.name, "features": feats,
                        "mean AUC [95% CI]": fmt_ci(h),
                        "per-split range (spread)": "n/a" if h is None else f"{h.min:.3f} to {h.max:.3f} ({h.spread:.3f})",
                        "splits used": 0 if h is None else h.n_splits,
                        "left out": "; ".join(excluded) or "none"})
        detail += [f"**{spec.name}, {feats} features**", "", md_table(ro.results_table(res)), ""]
        facts[key] = h
    last = F.X.created_at[HORIZON.eligible(F.X)].max()
    return Section([f"Last mature lead in this data: created {last.isoformat()}.", "",
                    md_table(pd.DataFrame(summary)), "", *detail], facts)


# ---------------------------------------------------------------------------------------------------------------
# 4.2 subsampling, 4.3 customer-sized test set
# ---------------------------------------------------------------------------------------------------------------

def _mean_sd(long: pd.DataFrame, model: str, n: int) -> MeanSd | None:
    v = long[(long.model == model) & (long.N == n) & ~long.full].auc
    return None if v.empty else MeanSd(float(v.mean()), float(v.std()))


def _full(long: pd.DataFrame, model: str) -> float:
    return float(long[(long.model == model) & long.full].auc.iloc[0])


LR_MODEL, GBDT_MONO, GBDT_FREE, LR_LEGACY = "LR (pipeline)", "GBDT monotone", "GBDT unconstrained", "LR, legacy features"


def subsampling_section(F: ro.EvalFrame, n_seeds: int) -> Section:
    """Subsampling curves: LR and both GBDTs on horizon labels (test b); LR legacy on legacy labels (test a)."""
    X = F.X
    trb, teb = F.frozen_horizon()
    tra, tea = F.frozen_legacy()
    models = {LR_MODEL: ss.lr_model(F.D, X.y), **ss.gbdt_models(X, X.y)}
    long_h = ss.subsample_curve(X.y, trb, teb, models, n_seeds=n_seeds)
    long_l = ss.subsample_curve(F.legacy_y, tra, tea, {LR_LEGACY: ss.lr_model(F.D_legacy, F.legacy_y)}, n_seeds=n_seeds)
    facts = {F_LR_2000: _mean_sd(long_h, LR_MODEL, SUMMARY_N), F_GBDT_2000: _mean_sd(long_h, GBDT_MONO, SUMMARY_N),
             F_GBDT_FREE_2000: _mean_sd(long_h, GBDT_FREE, SUMMARY_N),
             F_LEGACY_LR_2000: _mean_sd(long_l, LR_LEGACY, SUMMARY_N),
             F_LR_FULL: _full(long_h, LR_MODEL), F_GBDT_FULL: _full(long_h, GBDT_MONO)}
    return Section([
        f"Horizon labels, v2 features. Pool = the pipeline's horizon training set ({int(trb.sum())} leads, "
        f"{int(X.y[trb].sum())} won); test = frozen mature test set (b) ({int(teb.sum())} leads, "
        f"{int(X.y[teb].sum())} won). Mean ± sd AUC over {n_seeds} seeds:", "",
        md_table(ss.curve_table(long_h)), "",
        f"Baseline continuity: legacy labels, legacy features, pool = legacy training set ({int(tra.sum())} leads), "
        f"test = frozen legacy test set (a) ({int(tea.sum())} leads):", "",
        md_table(ss.curve_table(long_l)), ""], facts)


def customer_section(F: ro.EvalFrame, n_resamples: int) -> Section:
    """Customer-sized draws and, for scale, the CI widths of the frozen test sets."""
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
                   {F_CUSTOMER: Customer(float(w.mean()), float(a.mean()), float(a.std(ddof=1)))})


# ---------------------------------------------------------------------------------------------------------------
# 4.4 oracle-feature ceiling
# ---------------------------------------------------------------------------------------------------------------

def gap_key(a: str, b: str) -> str:
    """Name of the paired comparison ``b − a``."""
    return f"{b} − {a}"


def ceiling_section(F: ro.EvalFrame, n_resamples: int) -> Section:
    """Oracle ceilings on the frozen test sets and the rolling-origin headline, with the planted persona gap.

    Facts: ``F_CEILING_ROWS`` = model -> {evaluation -> CI or headline, "columns": design width};
    ``F_GAPS`` = ``gap_key(a, b)`` -> {evaluation -> paired CI}.
    """
    X = F.X
    Fo, p_true = ce.oracle_frame(F)
    sets = ce.available_sets(Fo)
    designs: dict[str, pd.DataFrame] = {PIPELINE: F.D, **{s: ce.oracle_design(Fo, s) for s in sets}}
    persona = ce.ORACLE_PERSONA in sets
    if persona:
        designs[PIPELINE_PERSONA] = F.D.join(ce.persona_indicator(Fo))
    trb, teb = F.frozen_horizon()
    tra, tea = F.frozen_legacy()
    yb, ya = X.y[teb].to_numpy(), F.legacy_y[tea].to_numpy()
    spec = ro.horizon_strict(F.horizon_days)
    scores: dict[str, dict[str, object]] = {}
    facts_rows: dict[str, dict[str, object]] = {}
    for name, D in designs.items():
        sb, sa = _frozen_score(D, X.y, trb, teb), _frozen_score(D, F.legacy_y, tra, tea)
        rr = ro.rolling_origin(spec, D, X.y, X.created_at, n_resamples=n_resamples)
        scores[name] = {E_B: sb, E_A: sa, E_ROLLING: rr}
        facts_rows[name] = {"columns": D.shape[1], E_ROLLING: ro.headline(rr, n_resamples),
                            E_B: auc_ci(yb, sb, n_resamples), E_A: auc_ci(ya, sa, n_resamples)}
    const = lambda D, y, tr: p_true.to_numpy()  # noqa: E731 (a fixed score needs no fit)
    facts_rows[P_TRUE] = {"columns": None,
                          E_ROLLING: ro.headline(ro.rolling_origin(spec, F.D, X.y, X.created_at, const,
                                                                   n_resamples=n_resamples), n_resamples),
                          E_B: auc_ci(yb, p_true[teb].to_numpy(), n_resamples),
                          E_A: auc_ci(ya, p_true[tea].to_numpy(), n_resamples)}

    def gap(a: str, b: str) -> dict[str, object]:
        return {E_ROLLING: ro.paired_headline(scores[a][E_ROLLING], scores[b][E_ROLLING], n_resamples),
                E_B: paired_auc(yb, scores[a][E_B], scores[b][E_B], n_resamples).diff,
                E_A: paired_auc(ya, scores[a][E_A], scores[b][E_A], n_resamples).diff}
    pairs = [(PIPELINE, ce.ORACLE_INTERACTIONS), (ce.ORACLE_FORMULA, ce.ORACLE_INTERACTIONS)]
    if persona:
        pairs += [(ce.ORACLE_INTERACTIONS, ce.ORACLE_PERSONA), (PIPELINE, PIPELINE_PERSONA)]
    gaps = {gap_key(a, b): gap(a, b) for a, b in pairs}
    evals = [(ROLLING_COL + " [95% CI]", E_ROLLING), (E_B + " [95% CI]", E_B), (E_A + " [95% CI]", E_A)]
    table = pd.DataFrame([{"model": m, "design columns": "" if r["columns"] is None else r["columns"],
                           **{col: fmt_ci(r[k]) for col, k in evals}} for m, r in facts_rows.items()])
    gap_table = pd.DataFrame([{"comparison": g, **{col: fmt_ci(v[k], True) for col, k in evals}} for g, v in gaps.items()])
    return Section([md_table(table), "", "Paired differences (same rows, shared resamples; rolling = mean of per-split "
                    "paired differences over the sufficient splits):", "", md_table(gap_table), ""],
                   {F_CEILING_ROWS: facts_rows, F_GAPS: gaps})


# ---------------------------------------------------------------------------------------------------------------
# 4.5 calibration decay, 4.6 regularisation, interactions, standard report
# ---------------------------------------------------------------------------------------------------------------

def _span(values: Iterable[float]) -> Span | None:
    v = [x for x in values if not np.isnan(x)]
    return Span(float(min(v)), float(max(v))) if v else None


def calibration_section(F: ro.EvalFrame) -> Section:
    """Calibration on M+1..M+3 for the headline rule (horizon labels) and for legacy labels."""
    X = F.X
    hs = cd.calibration_decay(ro.horizon_strict(F.horizon_days), F.D, X.y, X.created_at)
    lg = cd.calibration_decay(ro.LEGACY_WINDOW, F.D, F.legacy_y, X.created_at)
    stats = [s for r in hs for _, s in r.evals if s.n]
    return Section([
        f"**{ro.horizon_strict(F.horizon_days).name}, v2 features** (the headline rule):", "",
        md_table(cd.decay_table(hs)), "",
        "Deciles of p over each fit's pooled evaluation months, as mean p / observed:", "",
        md_table(cd.decile_table(hs)), "",
        "**Legacy labels (at AS_OF), training = every legacy-labelled lead created by the end of M, v2 features:**", "",
        md_table(cd.decay_table(lg)), ""],
        {F_SLOPES: _span(s.slope for s in stats), F_INTERCEPTS: _span(s.intercept for s in stats)})


A_LEGACY, A_V2, B_V2 = "(a) legacy labels, legacy features", "(a) legacy labels, v2 features", "(b) horizon, v2 features"


def regularisation_section(F: ro.EvalFrame) -> Section:
    """AUC against C on both frozen test sets and the rolling-origin headline."""
    X = F.X
    tra, tea = F.frozen_legacy()
    trb, teb = F.frozen_horizon()
    cases = {A_LEGACY: (F.D_legacy, F.legacy_y, tra, tea), A_V2: (F.D, F.legacy_y, tra, tea), B_V2: (F.D, X.y, trb, teb)}
    t = rg.sweep(cases, (ro.horizon_strict(F.horizon_days), F.D, X.y, X.created_at), brier_for=(B_V2,))
    flat = {c: rg.flatness(t, c) for c in [A_LEGACY, A_V2, B_V2, ROLLING_COL]}
    shown = t.copy()
    shown["C"] = [f"{c:g}" + (" (pipeline)" if c == LR_C else "") for c in t.C]
    for c in shown.columns[1:]:
        shown[c] = [f"{v:.4f}" for v in t[c]]
    verdict = pd.DataFrame([{"column": c, f"AUC range over C in [{rg.FLAT_RANGE[0]:g}, {rg.FLAT_RANGE[1]:g}]":
                             f"{f.lo:.4f} to {f.hi:.4f} ({f.hi - f.lo:.4f})", f"flat (< {rg.FLAT_TOLERANCE})": "yes" if f.flat else "no",
                             "full-grid range": f"{t[c].min():.4f} to {t[c].max():.4f}"} for c, f in flat.items()])
    return Section([md_table(shown), "", md_table(verdict), ""], {F_FLAT: flat})


def interactions_section(F: ro.EvalFrame, n_resamples: int) -> Section:
    """Coefficient CIs of the two interaction columns, their simulated sampling distribution, and the AUC change."""
    X = F.X
    DI = it.with_interactions(F.D, X)
    trb, teb = F.frozen_horizon()
    tra, tea = F.frozen_legacy()
    cis = it.interaction_coef_cis(DI, X.y, trb, n_resamples)
    coef_rows = []
    for c in it.INTERACTION_COLUMNS:
        ci = cis[c]
        on = DI[c].eq(1)
        coef_rows.append({"term": c, "planted on v2": f"{it.PLANTED_V2[c]:+.1f}",
                          "training rows with term = 1 (wins)": f"{int((on & trb).sum())} ({int(X.y[on & trb].sum())})",
                          "coefficient [95% CI]": fmt_ci(ci, True), "CI excludes 0": "yes" if ci.lo > 0 or ci.hi < 0 else "no"})
    null = it.null_coefficients(DI, X.y, trb, it.read_null_truth(F.data, DI.index), F.horizon_days)
    ns = null.summary()
    null_rows = [{"term": r.term, "observed": f"{r.observed:+.3f}", "simulated mean ± sd": f"{r.sim_mean:+.3f} ± {r.sim_sd:.3f}",
                  "simulated central 95%": f"[{r.sim_lo:+.3f}, {r.sim_hi:+.3f}]",
                  "share of draws at least as far from the mean": f"{r.share_as_extreme:.3f}",
                  "observed inside the 95%": "yes" if r.inside_95 else "**no**"} for r in ns.itertuples()]
    spec = ro.horizon_strict(F.horizon_days)
    base_r = ro.rolling_origin(spec, F.D, X.y, X.created_at, with_ci=False)
    int_r = ro.rolling_origin(spec, DI, X.y, X.created_at, with_ci=False)
    pr = ro.paired_headline(base_r, int_r, n_resamples)
    pb = paired_auc(X.y[teb].to_numpy(), _frozen_score(F.D, X.y, trb, teb), _frozen_score(DI, X.y, trb, teb), n_resamples)
    pa = paired_auc(F.legacy_y[tea].to_numpy(), _frozen_score(F.D, F.legacy_y, tra, tea),
                    _frozen_score(DI, F.legacy_y, tra, tea), n_resamples)
    auc_rows = [
        {"evaluation": f"{ROLLING_COL} (headline rule)", "AUC without": f"{ro.mean_auc(base_r):.3f}",
         "AUC with": f"{ro.mean_auc(int_r):.3f}", "with − without [95% CI]": fmt_ci(pr, True)},
        {"evaluation": "(b) mature test, horizon labels", "AUC without": f"{pb.auc_a:.3f}", "AUC with": f"{pb.auc_b:.3f}",
         "with − without [95% CI]": fmt_ci(pb.diff, True)},
        {"evaluation": "(a) legacy test, legacy labels", "AUC without": f"{pa.auc_a:.3f}", "AUC with": f"{pa.auc_b:.3f}",
         "with − without [95% CI]": fmt_ci(pa.diff, True)},
    ]
    return Section([f"Coefficients: formula model + the two terms, fitted on the horizon training set ({int(trb.sum())} "
                    f"leads), {n_resamples} bootstrap refits, seed {SEED}.", "", md_table(pd.DataFrame(coef_rows)), "",
                    f"Simulated sampling distribution of the same coefficients: {len(null.draws)} horizon-label sets drawn "
                    "for the same training rows from `p_close_true`, the planted time to close and the generator's "
                    f"stalled flag (`emva.eval.interactions.null_coefficients`, seed {SEED}); on v1 this is the null. "
                    f"Mean simulated training win rate {null.sim_rate:.4f} vs observed {null.observed_rate:.4f}.", "",
                    md_table(pd.DataFrame(null_rows)), "",
                    md_table(pd.DataFrame(auc_rows)), ""],
                   {F_COEFS: cis, F_ROLLING_GAIN: pr, F_NULL: ns})


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
# prose (derived from the constants, never hard-coded)
# ---------------------------------------------------------------------------------------------------------------

def rolling_intro(horizon_days: int = HORIZON_DAYS) -> str:
    """The 4.1 introduction; split dates, H, the maturity cutoff and the empty or partial windows are derived."""
    cutoff = mature_cutoff(horizon_days)
    partial = [s for s in ro.SPLITS if s < cutoff < s + ro.TEST_WINDOW]
    empty = [s for s in ro.SPLITS if s >= cutoff]
    windows = ""
    if partial:
        windows += " " + "; ".join(f"the {s.date()} window is partial ({(cutoff - s).days} days)" for s in partial)
    if empty:
        windows += (", and" if partial else "") + " the " + " and ".join(str(s.date()) for s in empty) + \
            f" window{'s have' if len(empty) > 1 else ' has'} no mature test lead under H = {horizon_days}"
    return f"""Split at S in {", ".join(s.date().isoformat() for s in ro.SPLITS)} (ADR 0013). **Headline rule:** train on
labelled leads created before S whose H = {horizon_days} horizon label is mature at S (`created_at + H <= S`); test on
labelled leads created in [S, S + 1 month) whose label is mature at AS_OF ({AS_OF.date()}). Labels are the pipeline's
default (ghosted-at-H excluded, stalled censored; the stalled flag reads the CRM state at AS_OF, see
`emva/eval/rolling.py`). A split is left out of the mean when its test set has fewer than {ro.MIN_TEST_LEADS} labelled
leads or fewer than {ro.MIN_TEST_CLASS} of a class; it is still listed. Only leads created before
{cutoff.isoformat()} are mature, so{windows}. Mean AUC CI: each split's test set resampled independently
({N_RESAMPLES} resamples, seeds {SEED}, {SEED + 1}, ...).

Comparisons: *relaxed* = the same test windows but training on every lead created before S that is mature at
AS_OF (look-ahead: this is how the frozen split trains); *legacy labels, one-month test* = baseline labels at AS_OF;
*legacy labels, test = everything from S*, legacy features = the definition that reproduces the review's baseline
spread (0.814 to 0.836 on v1; at S = {TEST_FROM} it is exactly the frozen legacy test set)."""


SUBSAMPLING_INTRO = f"""N training rows drawn without replacement from the training pool (seeds 0-{ss.N_SEEDS - 1}; the same
rows for every model within a seed), scored on a fixed test set; "full" is one fit on the whole pool. GBDT =
`HistGradientBoostingClassifier` ({ss.GBDT_MAX_ITER} iterations, learning rate {ss.GBDT_LEARNING_RATE:g}) on one column
per v2 feature: the {len(ss.ORDERED_LEVELS)} ordered features are ranks with a +1 monotone constraint, the rest
categorical. The constraint vector (column order = the GBDT design):

{{constraints}}

The planted effects of size (201-1000 +1.3 > 1000+ +1.0) and time on page (60-300s +0.4, >600s -0.4) are not
monotone, so the constraint is wrong at the top of those two scales; the unconstrained GBDT shows what it costs.
Baseline reference (review, v1): LR 0.807 ± 0.004 at N = 2000 (legacy labels and features, legacy test set)."""

_HOLDOUT_N = int(round(ss.CUSTOMER_N * ss.CUSTOMER_HOLDOUT))
CUSTOMER_INTRO = f"""A customer with {ss.CUSTOMER_N:,} mature labelled leads validates on their most recent
{ss.CUSTOMER_HOLDOUT:.0%} ({_HOLDOUT_N} leads): draw {ss.CUSTOMER_N:,} leads from the mature labelled pool (seeds
0-{ss.CUSTOMER_DRAWS - 1}), sort by `created_at`, fit the formula model on the first {ss.CUSTOMER_N - _HOLDOUT_N:,} and
bootstrap the AUC on the last {_HOLDOUT_N} ({N_RESAMPLES} resamples, seed {SEED}). The width is what that customer's
CI would look like."""

CEILING_INTRO = f"""Formula model (same C = {LR_C:g}, same rows, same splits) fitted on the generator's own inputs instead
of the pipeline's features (`emva/eval/ceiling.py`; ground truth, evaluation only). *truth file only* = the planted
categorical columns of `ground_truth_labels.csv` (true size, free email, true text category, channel, lead ads, true
seniority). *formula* = every planted main effect without the noise term and the sales reply time: those plus the true
company's spend / CRM / hiring, IP country vs the true country and the recorded session inputs at the planted cut
points (no telemetry = its own level: the generator scored consent-declined leads on hidden true behaviour).
*+ interactions* adds LinkedIn × true size 51+ and true senior × form D (planted on v2 only). *+ persona* (v2) adds the
hidden `context_persona`. **The formula + interactions row is the reachable ceiling that replaces the 0.836 figure;
the persona row minus it is the planted context-only gap Phase 6 has to recover.**"""

CALIBRATION_INTRO = f"""Fit at the end of month M (split = first day of M+1, same training rule as the rolling headline),
evaluate each of M+1..M+{cd.DECAY_MONTHS}. Slope = logistic regression of y on logit p (1 = well spread, < 1 =
over-confident); intercept = calibration-in-the-large (y on an offset of logit p; 0 = right level, < 0 =
over-predicting). Months with no mature labelled lead are listed as 0. The legacy-label table is for comparison only:
legacy labels are read at AS_OF, so the youngest months hold mostly fast decisions and are not comparable with older
months."""

REGULARISATION_INTRO = f"""L2 strength C over a log grid; everything else as the pipeline's model (C = {LR_C:g}). Standing
check: flat = AUC range below {rg.FLAT_TOLERANCE} over C in [{rg.FLAT_RANGE[0]:g}, {rg.FLAT_RANGE[1]:g}]. Baseline
(review, v1, legacy test set): flat, 0.8107 to 0.8138."""

INTERACTIONS_INTRO = f"""Phase 5 planted LinkedIn × company size 51+ ({it.PLANTED_V2['linkedin_x_51plus']:+.1f}) and senior
title × form D ({it.PLANTED_V2['senior_x_formD']:+.1f}) on v2 and asked whether the models can detect them. The two
products of the pipeline's own features (`channel`, `band`, `seniority`, `form_variant`; no ground truth) are added to
the v2 design. v1 has neither term planted: it is the null control. The simulated sampling distribution (reads ground
truth) says how far an observed coefficient is from what the planted truth and the {HORIZON_DAYS}-day label produce."""

STANDARD_INTRO = """`python -m emva.eval.report --data <dir>` (ground rule 3), unchanged model defaults. Headings demoted to
nest here."""


# ---------------------------------------------------------------------------------------------------------------
# assembly, cache, CLI
# ---------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    """Run parameters (part of the cache key)."""

    n_resamples: int = N_RESAMPLES
    n_seeds: int = ss.N_SEEDS


@dataclass(frozen=True)
class SectionSpec:
    """A report section: heading, the prose above the per-dataset tables, and how to compute one dataset."""

    key: str
    heading: str
    intro: str
    compute: Callable[[ro.EvalFrame, Params], Section]


def _sections() -> list[SectionSpec]:
    return [
        SectionSpec(K_ROLLING, "4.1 Rolling-origin evaluation", rolling_intro(), lambda F, p: rolling_section(F, p.n_resamples)),
        SectionSpec(K_SUBSAMPLING, "4.2 Subsampling curve: LR vs monotone GBDT",
                    SUBSAMPLING_INTRO.format(constraints=md_table(ss.constraint_table())),
                    lambda F, p: subsampling_section(F, p.n_seeds)),
        SectionSpec(K_CUSTOMER, "4.3 Customer-sized test set", CUSTOMER_INTRO, lambda F, p: customer_section(F, p.n_resamples)),
        SectionSpec(K_CEILING, "4.4 Oracle-feature ceiling and the planted context-only gap", CEILING_INTRO,
                    lambda F, p: ceiling_section(F, p.n_resamples)),
        SectionSpec(K_CALIBRATION, "4.5 Calibration decay", CALIBRATION_INTRO, lambda F, p: calibration_section(F)),
        SectionSpec(K_REGULARISATION, "4.6 Regularisation sweep", REGULARISATION_INTRO, lambda F, p: regularisation_section(F)),
        SectionSpec(K_INTERACTIONS, "Interaction detectability (Phase 5 hand-off)", INTERACTIONS_INTRO,
                    lambda F, p: interactions_section(F, p.n_resamples)),
        SectionSpec(K_STANDARD, "Standard report", STANDARD_INTRO, lambda F, p: standard_section(F, p.n_resamples)),
    ]


SECTION_KEYS: tuple[str, ...] = tuple(s.key for s in _sections())


def source_files() -> list[Path]:
    """Files whose contents are part of the cache key: every emva source, the baseline script, requirements.txt."""
    return sorted([*(REPO_ROOT / "emva").rglob("*.py"), REPO_ROOT / "baseline" / "emva_score.py",
                   REPO_ROOT / "requirements.txt"])


def library_versions() -> str:
    """Installed versions of the numeric libraries the results depend on."""
    return f"numpy {np.__version__}; pandas {pd.__version__}; scikit-learn {sklearn.__version__}; scipy {scipy.__version__}"


def _fingerprint(data: Path, sources: Iterable[Path] | None = None) -> str:
    """sha256 over the contents of ``data``'s files (top level), of ``sources`` (default ``source_files()``) and the library versions."""
    h = hashlib.sha256()
    for f in sorted(p for p in data.iterdir() if p.is_file()):
        h.update(f.name.encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    for f in (source_files() if sources is None else sources):
        h.update(str(f).encode())
        h.update(hashlib.sha256(f.read_bytes()).digest())
    h.update(library_versions().encode())
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


def summary_lines(results: dict[tuple[str, str], Section], names: list[str]) -> list[str]:
    """The headline table at the top of the report: one row per key number, one column per dataset."""
    def fact(key: str, name: str, item: str) -> object:
        sec = results.get((key, name))
        return None if sec is None else sec.facts.get(item)

    rows: list[dict[str, str]] = []

    def add(label: str, fn: Callable[[str], str]) -> None:
        rows.append({"quantity": label, **{f"{n}{' (headline)' if i == 0 else ''}": fn(n) for i, n in enumerate(names)}})

    def rng_(h: ro.Headline | None, with_n: bool = False) -> str:
        if h is None:
            return "n/a"
        return f"{h.min:.3f} to {h.max:.3f}" + (f" ({h.n_splits})" if with_n else "")

    def msd(m: MeanSd | None) -> str:
        return "n/a" if m is None else f"{m.mean:.3f} ± {m.sd:.3f}"

    add("4.1 rolling-origin AUC, headline rule: mean [95% CI]", lambda n: fmt_ci(fact(K_ROLLING, n, F_HEADLINE)))
    add("4.1 ... per-split range (splits used)", lambda n: rng_(fact(K_ROLLING, n, F_HEADLINE), True))
    add("4.1 legacy labels, one-month test: range", lambda n: rng_(fact(K_ROLLING, n, F_LEGACY_WINDOW)))
    add("4.1 baseline reproduction (legacy, test to end): range", lambda n: rng_(fact(K_ROLLING, n, F_BASELINE_SPREAD)))
    add(f"4.2 LR at N = {SUMMARY_N}, test (b)", lambda n: msd(fact(K_SUBSAMPLING, n, F_LR_2000)))
    add(f"4.2 GBDT monotone at N = {SUMMARY_N}, test (b)", lambda n: msd(fact(K_SUBSAMPLING, n, F_GBDT_2000)))
    add(f"4.2 LR (legacy labels + features) at N = {SUMMARY_N}, test (a)", lambda n: msd(fact(K_SUBSAMPLING, n, F_LEGACY_LR_2000)))

    def full(n: str) -> str:
        lr, gb = fact(K_SUBSAMPLING, n, F_LR_FULL), fact(K_SUBSAMPLING, n, F_GBDT_FULL)
        return "n/a" if lr is None else f"{lr:.3f} / {gb:.3f}"
    add("4.2 full pool: LR / GBDT monotone, test (b)", full)

    def cust(n: str) -> str:
        c = fact(K_CUSTOMER, n, F_CUSTOMER)
        return "n/a" if c is None else f"{c.mean_width:.3f} (± {c.mean_width / 2:.3f})"
    add(f"4.3 customer-sized ({_HOLDOUT_N}-lead holdout) mean CI width", cust)

    evals = [("(b)", E_B), ("(a)", E_A), ("rolling", E_ROLLING)]

    def ceil(n: str, model: str, k: str) -> str:
        rows_ = fact(K_CEILING, n, F_CEILING_ROWS) or {}
        return fmt_ci(rows_.get(model, {}).get(k))

    def gapf(n: str, g: str, k: str) -> str:
        gaps = fact(K_CEILING, n, F_GAPS) or {}
        return fmt_ci(gaps.get(g, {}).get(k), True)
    for model, label_ in [(PIPELINE, PIPELINE), (ce.ORACLE_INTERACTIONS, "**ceiling = formula + interactions**"),
                          (ce.ORACLE_PERSONA, ce.ORACLE_PERSONA)]:
        for short, k in evals:
            add(f"4.4 {label_}: {short}", lambda n, m=model, k=k: ceil(n, m, k))
    for g, label_ in [(gap_key(ce.ORACLE_INTERACTIONS, ce.ORACLE_PERSONA),
                       "**planted context-only gap** (persona ceiling − formula ceiling)"),
                      (gap_key(PIPELINE, PIPELINE_PERSONA), "pipeline + true persona − pipeline"),
                      (gap_key(PIPELINE, ce.ORACLE_INTERACTIONS), "ceiling − pipeline")]:
        for short, k in evals:
            add(f"4.4 {label_}: {short}", lambda n, g=g, k=k: gapf(n, g, k))

    def span(s: Span | None, signed: bool) -> str:
        f = "{:+.2f}" if signed else "{:.2f}"
        return "n/a" if s is None else f"{f.format(s.lo)} to {f.format(s.hi)}"
    add("4.5 calibration slope range (headline rule)", lambda n: span(fact(K_CALIBRATION, n, F_SLOPES), False))
    add("4.5 calibration intercept range (headline rule)", lambda n: span(fact(K_CALIBRATION, n, F_INTERCEPTS), True))

    def flat(n: str) -> str:
        f = fact(K_REGULARISATION, n, F_FLAT)
        return "n/a" if f is None else f"{'yes' if f[ROLLING_COL].flat else 'no'} / {'yes' if f[A_LEGACY].flat else 'no'}"
    add(f"4.6 flat over C in [{rg.FLAT_RANGE[0]:g}, {rg.FLAT_RANGE[1]:g}]? (rolling / (a) legacy features)", flat)
    for c in it.INTERACTION_COLUMNS:
        add(f"interaction {c}: coefficient [95% CI]", lambda n, c=c: fmt_ci((fact(K_INTERACTIONS, n, F_COEFS) or {}).get(c), True))

        def null_cell(n: str, c: str = c) -> str:
            ns = fact(K_INTERACTIONS, n, F_NULL)
            if ns is None:
                return "n/a"
            r = ns.set_index("term").loc[c]
            return f"[{r.sim_lo:+.3f}, {r.sim_hi:+.3f}] (observed {'inside' if r.inside_95 else 'outside'})"
        add(f"interaction {c}: simulated central 95%", null_cell)
    add("interactions: rolling AUC gain", lambda n: fmt_ci(fact(K_INTERACTIONS, n, F_ROLLING_GAIN), True))
    return ["## Summary (on simulated data)", "", md_table(pd.DataFrame(rows)), ""]


def timing_lines(results: dict[tuple[str, str], Section]) -> list[str]:
    """Compute time per section and dataset (from the run that produced each cached result)."""
    rows = [{"section": k, "data": n, "seconds": f"{s.seconds:.1f}"} for (k, n), s in results.items()]
    total = sum(s.seconds for s in results.values())
    return ["## Runtime", "", f"Total compute {total:.0f} s ({total / 60:.1f} min) on the machine that produced the cache "
            f"entries (single-threaded fits; {library_versions()}); an unchanged rerun reads `runs/phase4/cache/` in "
            "seconds and writes the same text.", "", md_table(pd.DataFrame(rows)), ""]


def build(datas: list[Path], params: Params = Params(), use_cache: bool = True) -> str:
    """The whole generated block (without the markers) as markdown."""
    results = run_sections(datas, params, use_cache)
    names = [d.name for d in datas]
    out = ["# Phase 4 evaluation hardening (generated)", "",
           f"Data: {', '.join(f'`{d}`' for d in datas)} ({names[0]} is the headline). All numbers are on simulated data. "
           f"Model = the pipeline's defaults (horizon labels H = {HORIZON_DAYS}, v2 features, L2 LR C = {LR_C:g}) unless a "
           f"row says otherwise. (a) = frozen legacy test set (legacy labels, created on or after {TEST_FROM}); (b) = "
           f"frozen mature test set (`FROZEN_HORIZON`). CIs: percentile bootstrap, {params.n_resamples} resamples.", "",
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
