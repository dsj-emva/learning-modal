"""Context-agent evaluation harness (plan 6.5 and 6.6). Evaluation side: reads ground truth.

- ``sample_leads``: the seeded, persona-stratified 1,000-lead sample the real Haiku run uses.
- ``oracle_power_curve``: an oracle context feature built from the planted persona (log-odds effect
  ``PERSONA_EFFECT`` plus Gaussian noise of sd 0.25 / 0.5 / 1 / 2) is added to the formula design; the
  curve is the fraction of the oracle gap, ``AUC(formula + true persona) - AUC(formula)``, recovered at
  each noise level. It replaces the unreproducible "0.814 -> 0.818".
- ``evaluate_judgments``: formula vs formula+context on the same rows, paired bootstrap, per-judgment
  weights with bootstrap CIs, persona confusion against ``context_persona``, correlation of each context
  feature with the formula score, and boilerplate recall against the generator's ``copy_paste`` flag (R6).
- **Acceptance (R12, ADR 0016)**: ``coefficient_criterion``. The pooled persona feature's weight
  (``ctx_persona_group=non_buyer`` in formula + ``ctx_persona_group``) must have a 95% CI excluding zero
  and a point estimate inside the oracle weight's CI on the same rows. The AUC gap is reported but is
  recorded as unmeasurable at the planted effect size.

**Protocol.** Both models are always fit on identical rows. On a sample (1,000 leads) the comparison
is cross-fitted: ``N_FOLDS`` stratified folds (seed ``SEED``), every model gets the same folds, AUCs are
computed on the pooled out-of-fold predictions, and paired AUC differences come from shared bootstrap
resamples of those rows (``emva.eval.bootstrap``). The time split of ``python -m emva --context`` (train
before 2026-05-01, test after, sample rows only) is reported too, but a 1,000-lead sample leaves only
~100 test rows there.

CLI::

  python -m emva.eval.context_harness sample --data data/v2 --out data/v2/context/sample_ids.csv
  python -m emva.eval.context_harness evaluate --data data/v2 \
      --judgments data/v2/context/context_judgments.csv --plot reports/phase6_power_curve.png --md OUT.md
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from emva.context.contract import JUDGMENTS
from emva.context.features import CONTEXT_COLUMNS, PERSONA_ACCEPTANCE_COLUMN, PREFIX, context_design
from emva.eval.bootstrap import (
    CI_LEVEL,
    N_RESAMPLES,
    SEED,
    BootstrapCI,
    PairedComparison,
    _percentile_ci,
    _resample_indices,
    coef_bootstrap,
    paired_auc,
)
from emva.design import fixed_columns, fixed_design
from emva.feature_spec import feature_spec
from emva.features import FeatureSet
from emva.io import load
from emva.labels import HORIZON, split_masks
from emva.model import make_lr
from emva.pipeline import build, run

PERSONA_EFFECT: float = -1.0            # planted log-odds effect (reports/phase5.md, 5.7)
NOISE_SDS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
N_NOISE_DRAWS: int = 20
N_FOLDS: int = 5
SAMPLE_SIZE: int = 1000
SAMPLE_PERSONA: int = 300               # >= 150 required by the plan; ~75 per planted persona
# Secondary, underpowered (R13): the contract judgments without pooling, persona at its eight levels.
FINE_CATS: dict[str, str] = {"ctx_is_real_business": "yes", "ctx_persona": "buyer", "ctx_problem_specificity": "specific",
                             "ctx_urgency": "none", "ctx_brief_fit": "partial"}
FINE_LEVELS: dict[str, tuple[str, ...]] = {PREFIX + k: v for k, v in JUDGMENTS.items()}


@dataclass(frozen=True)
class EvalData:
    """Everything the harness needs from one data directory.

    ``X``: cleaned, labelled, featurised leads (horizon labels, v2 features); ``D``: the v2 design;
    ``labelled``: mature rows with a label (the union of the train and test masks); ``train``: its
    pre-2026-05-01 part; ``persona``: the generator's ``context_persona`` ("none" when absent);
    ``text_category``: the generator's true text category; ``p_true``: the generator's ``p_close_true``.
    """

    X: pd.DataFrame
    D: pd.DataFrame
    labelled: pd.Series
    train: pd.Series
    persona: pd.Series
    text_category: pd.Series
    p_true: pd.Series


def load_eval_data(data: str | Path) -> EvalData:
    """Build ``EvalData`` for ``data`` (reads ``ground_truth_labels.csv``)."""
    X = build(load(data), HORIZON, FeatureSet.V2)
    D = feature_spec(FeatureSet.V2).design(X)
    tr, te = split_masks(X, eligible=HORIZON.eligible(X))
    gt = pd.read_csv(Path(data) / "ground_truth_labels.csv").set_index("lead_id").reindex(X.index)
    return EvalData(X=X, D=D, labelled=tr | te, train=tr, persona=gt.context_persona.fillna("none"),
                    text_category=gt.text_category, p_true=gt.p_close_true)


def sample_leads(labelled: pd.Series, persona: pd.Series, n: int = SAMPLE_SIZE, n_persona: int = SAMPLE_PERSONA,
                 seed: int = SEED) -> pd.Index:
    """Seeded sample of ``n`` labelled leads, exactly ``n_persona`` of them with a planted persona.

    Draws without replacement from the labelled persona leads and, separately, from the labelled
    non-persona leads (``numpy.random.default_rng(seed)``); returns the ids sorted. Raises
    ``ValueError`` when either stratum is too small.
    """
    rng = np.random.default_rng(seed)
    pool_p = labelled.index[labelled & persona.ne("none")].sort_values()
    pool_n = labelled.index[labelled & persona.eq("none")].sort_values()
    if len(pool_p) < n_persona or len(pool_n) < n - n_persona:
        raise ValueError(f"need {n_persona} persona and {n - n_persona} other labelled leads; "
                         f"have {len(pool_p)} and {len(pool_n)}")
    pick = np.concatenate([rng.choice(pool_p, n_persona, replace=False), rng.choice(pool_n, n - n_persona, replace=False)])
    return pd.Index(sorted(pick), name="lead_id")


def folds(y: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified (train, held-out) index pairs, shared by every model compared on the same rows."""
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def crossfit(M: np.ndarray, y: np.ndarray, splits: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Pooled out-of-fold P(won) from the formula model (``make_lr``) fit on each fold's training rows."""
    out = np.empty(len(y))
    for fit_idx, held in splits:
        out[held] = make_lr().fit(M[fit_idx], y[fit_idx]).predict_proba(M[held])[:, 1]
    return out


def oracle_power_curve(D: np.ndarray, y: np.ndarray, persona: np.ndarray, sds: Sequence[float] = NOISE_SDS,
                       n_draws: int = N_NOISE_DRAWS, effect: float = PERSONA_EFFECT,
                       seed: int = SEED) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fraction of the oracle gap recovered by a noisy oracle feature, at each noise sd.

    ``D`` is the formula design, ``persona`` a 0/1 planted-persona indicator. The oracle column is
    ``effect * persona + N(0, sd)``, appended to ``D``; every model is cross-fitted on the same folds.
    Returns (one row per sd with the mean, 5th and 95th percentile over ``n_draws`` draws of the
    out-of-fold AUC and of the fraction recovered, and ``{"auc_formula", "auc_oracle", "gap"}``).
    """
    y = np.asarray(y, dtype=float)
    persona = np.asarray(persona, dtype=float)
    splits = folds(y, seed=seed)
    auc_f = roc_auc_score(y, crossfit(D, y, splits))
    auc_o = roc_auc_score(y, crossfit(np.column_stack([D, effect * persona]), y, splits))
    gap = auc_o - auc_f
    rng = np.random.default_rng(seed)
    rows = []
    for sd in sds:
        aucs = np.array([roc_auc_score(y, crossfit(np.column_stack([D, effect * persona + rng.normal(0, sd, len(y))]),
                                                   y, splits)) for _ in range(n_draws)])
        frac = (aucs - auc_f) / gap
        rows.append({"noise_sd": sd, "auc_mean": aucs.mean(), "frac_mean": frac.mean(),
                     "frac_p05": np.quantile(frac, 0.05), "frac_p95": np.quantile(frac, 0.95)})
    return pd.DataFrame(rows), {"auc_formula": auc_f, "auc_oracle": auc_o, "gap": gap}


def gap_recovered(y: np.ndarray, p_formula: np.ndarray, p_context: np.ndarray, p_oracle: np.ndarray,
                  n_resamples: int = N_RESAMPLES, level: float = CI_LEVEL, seed: int = SEED) -> BootstrapCI:
    """``(AUC(context) - AUC(formula)) / (AUC(oracle) - AUC(formula))`` with a CI from shared resamples."""
    y = np.asarray(y, dtype=float)

    def frac(i: np.ndarray) -> float:
        a_f = roc_auc_score(y[i], p_formula[i])
        return (roc_auc_score(y[i], p_context[i]) - a_f) / (roc_auc_score(y[i], p_oracle[i]) - a_f)

    rng = np.random.default_rng(seed)
    samples = np.array([frac(i) for i in _resample_indices(y, n_resamples, rng)])
    return _percentile_ci(frac(np.arange(len(y))), samples, level)


def true_probability_gap(y: np.ndarray, p_true: np.ndarray, persona: np.ndarray,
                         effect: float = PERSONA_EFFECT) -> dict[str, float]:
    """Ceiling on the persona's AUC value: AUC of the generator's true close probability with and without
    the planted persona term (``logit(p_true) - effect * persona``). No model, no estimation noise."""
    p = np.clip(np.asarray(p_true, dtype=float), 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p))
    with_p = roc_auc_score(y, lo)
    without = roc_auc_score(y, lo - effect * np.asarray(persona, dtype=float))
    return {"auc_true": with_p, "auc_true_without_persona": without, "true_gap": with_p - without}


def coefficient_criterion(weight: BootstrapCI, oracle: BootstrapCI) -> bool:
    """R12 (ADR 0016): ``weight``'s CI excludes zero and its point estimate lies inside ``oracle``'s CI."""
    return (weight.lo > 0 or weight.hi < 0) and oracle.lo <= weight.point <= oracle.hi


def _weights_table(columns: Sequence[str], M: np.ndarray, cis: Sequence[BootstrapCI]) -> pd.DataFrame:
    """Rows ``feature, n, weight, lo, hi, significant`` for design columns ``M`` and their CIs."""
    t = pd.DataFrame({"feature": list(columns), "n": M.sum(axis=0).astype(int), "weight": [c.point for c in cis],
                      "lo": [c.lo for c in cis], "hi": [c.hi for c in cis]})
    t["significant"] = (t.lo > 0) | (t.hi < 0)
    return t


@dataclass
class JudgmentEval:
    """Results of ``evaluate_judgments`` (all on the ok-status sample rows)."""

    n_rows: int
    n_wins: int
    n_persona: int
    status_counts: dict[str, int]
    ctx_vs_formula: PairedComparison
    oracle_vs_formula: PairedComparison
    persona_vs_formula: PairedComparison
    recovered: BootstrapCI
    weights: pd.DataFrame
    oracle_weight: BootstrapCI
    persona_weight: BootstrapCI
    persona_weight_combined: BootstrapCI
    r12_pass: bool
    fine_weights: pd.DataFrame
    confusion: pd.DataFrame
    correlations: pd.DataFrame
    context_score_corr: float
    boilerplate: pd.DataFrame
    time_split: pd.DataFrame
    time_split_rows: int


def _recall_precision(truth: pd.Series, flag: pd.Series) -> dict[str, float]:
    """Counts, recall and precision of boolean ``flag`` against boolean ``truth``."""
    tp = int((truth & flag).sum())
    return {"flagged": int(flag.sum()), "true": int(truth.sum()), "tp": tp,
            "recall": tp / truth.sum() if truth.sum() else float("nan"),
            "precision": tp / flag.sum() if flag.sum() else float("nan")}


def evaluate_judgments(ed: EvalData, judgments_path: str | Path, data: str | Path,
                       n_resamples: int = N_RESAMPLES, seed: int = SEED) -> JudgmentEval:
    """Evaluate a judgments CSV against ``ed`` (see the module docstring for the protocol)."""
    J = pd.read_csv(judgments_path)
    C_all = context_design(J, ed.X.index)
    rows = C_all.notna().all(axis=1) & ed.labelled
    idx = ed.X.index[rows]
    y = ed.X.y[idx].to_numpy(dtype=float)
    D, C = ed.D.loc[idx].to_numpy(), C_all.loc[idx].to_numpy()
    persona = ed.persona[idx]
    oracle = (persona != "none").to_numpy(dtype=float) * PERSONA_EFFECT
    splits = folds(y, seed=seed)
    p_f = crossfit(D, y, splits)
    p_c = crossfit(np.column_stack([D, C]), y, splits)
    p_o = crossfit(np.column_stack([D, oracle]), y, splits)

    def lr_coef(M: np.ndarray, t: np.ndarray) -> np.ndarray:
        return make_lr().fit(M, t).coef_[0]

    ci = coef_bootstrap(np.column_stack([D, C]), y, lr_coef, n_resamples=n_resamples, seed=seed)[D.shape[1]:]
    weights = _weights_table(CONTEXT_COLUMNS, C, ci)
    oracle_ci = coef_bootstrap(np.column_stack([D, oracle / PERSONA_EFFECT]), y, lr_coef,
                               n_resamples=n_resamples, seed=seed)[-1]

    # R12 acceptance: the pooled persona feature alone (its two non-reference columns) next to the formula
    pg_cols = [c for c in CONTEXT_COLUMNS if c.startswith("ctx_persona_group=")]
    P = C_all.loc[idx, pg_cols].to_numpy()
    p_pg = crossfit(np.column_stack([D, P]), y, splits)
    pg_ci = coef_bootstrap(np.column_stack([D, P]), y, lr_coef, n_resamples=n_resamples, seed=seed)[D.shape[1]:]
    persona_ci = pg_ci[pg_cols.index(PERSONA_ACCEPTANCE_COLUMN)]

    # secondary (underpowered): the unpooled judgments, persona at its eight contract levels
    ok = J[J.status == "ok"].set_index("lead_id")
    F = pd.DataFrame({PREFIX + k: ok[k] for k in JUDGMENTS}, index=ok.index).reindex(idx)
    Cf = fixed_design(F, FINE_CATS, FINE_LEVELS).to_numpy()
    fine_ci = coef_bootstrap(np.column_stack([D, Cf]), y, lr_coef, n_resamples=n_resamples, seed=seed)[D.shape[1]:]
    fine = _weights_table(fixed_columns(FINE_CATS, FINE_LEVELS), Cf, fine_ci)

    # correlation with the formula score: the formula fit on every pre-2026-05-01 labelled lead
    formula = make_lr().fit(ed.D[ed.train], ed.X.y[ed.train])
    f_logit = formula.decision_function(ed.D.loc[idx])
    corr = pd.DataFrame({"feature": CONTEXT_COLUMNS,
                         "r": [np.corrcoef(C[:, j], f_logit)[0, 1] if C[:, j].std() > 0 else np.nan
                               for j in range(C.shape[1])]})
    w_full = make_lr().fit(np.column_stack([D, C]), y).coef_[0][D.shape[1]:]
    ctx_score_r = float(np.corrcoef(C @ w_full, f_logit)[0, 1])

    conf = pd.crosstab(ok.persona.reindex(idx).rename("judged persona"), persona.rename("planted persona"))

    # boilerplate (R6): every ok sample row, labelled or not
    ok_ids = ok.index.intersection(ed.X.index)
    truth = ed.text_category[ok_ids].eq("copy_paste")
    boiler = pd.DataFrame([
        {"detector": "LLM problem_specificity=boilerplate",
         **_recall_precision(truth, ok.problem_specificity[ok_ids].eq("boilerplate"))},
        {"detector": "Jaccard >= 0.6 (text=copy_paste, Phase 2)",
         **_recall_precision(truth, ed.X.text[ok_ids].eq("copy_paste"))},
        {"detector": "LLM boilerplate or unrelated",
         **_recall_precision(truth, ok.problem_specificity[ok_ids].isin(["boilerplate", "unrelated"]))},
    ])

    res = run(data, context=judgments_path)
    return JudgmentEval(
        n_rows=len(idx), n_wins=int(y.sum()), n_persona=int((persona != "none").sum()),
        status_counts=J.status.value_counts().to_dict(),
        ctx_vs_formula=paired_auc(y, p_f, p_c, n_resamples=n_resamples, seed=seed),
        oracle_vs_formula=paired_auc(y, p_f, p_o, n_resamples=n_resamples, seed=seed),
        persona_vs_formula=paired_auc(y, p_f, p_pg, n_resamples=n_resamples, seed=seed),
        recovered=gap_recovered(y, p_f, p_c, p_o, n_resamples=n_resamples, seed=seed),
        weights=weights, oracle_weight=oracle_ci, persona_weight=persona_ci,
        persona_weight_combined=ci[CONTEXT_COLUMNS.index(PERSONA_ACCEPTANCE_COLUMN)],
        r12_pass=coefficient_criterion(persona_ci, oracle_ci), fine_weights=fine, confusion=conf,
        correlations=corr, context_score_corr=ctx_score_r, boilerplate=boiler, time_split=res.summary,
        time_split_rows=int((res.test & res.X.p_combined.notna()).sum()))


def plot_power_curve(curves: dict[str, pd.DataFrame], real: dict[str, BootstrapCI], path: str | Path) -> None:
    """Write the power-curve PNG: fraction recovered vs noise sd per population (mean and 5-95% band over
    noise draws), plus a dashed line per real run labelled with its bootstrap CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for (name, cur), color in zip(curves.items(), ("#2a6f97", "#c05621"), strict=False):
        ax.fill_between(cur.noise_sd, cur.frac_p05, cur.frac_p95, color=color, alpha=0.15, linewidth=0)
        ax.plot(cur.noise_sd, cur.frac_mean, marker="o", color=color, label=f"oracle + noise: {name}")
    for name, ci in real.items():
        ax.axhline(ci.point, color="#444444", linestyle="--", linewidth=1,
                   label=f"{name}: {ci.point:.2f} [95% CI {ci.lo:.1f}, {ci.hi:.1f}]")
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(NOISE_SDS), [str(s) for s in NOISE_SDS])
    ax.set_xlabel("noise sd on the oracle's log-odds")
    ax.set_ylabel("fraction of oracle AUC gap recovered")
    ax.set_title("Context power curve on data/v2 (simulated data)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fmt_ci(ci: BootstrapCI, dp: int = 3) -> str:
    """``point [lo, hi]``."""
    return f"{ci.point:.{dp}f} [{ci.lo:.{dp}f}, {ci.hi:.{dp}f}]"


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """A pipe table (floats formatted with ``floatfmt``)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, float | np.floating) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown(curves: dict[str, pd.DataFrame], refs: dict[str, dict[str, float]], ev: JudgmentEval) -> str:
    """The harness results as markdown sections for ``reports/phase6.md``."""
    out = ["### 6.5 Power curve (oracle persona feature + noise)", ""]
    for name, cur in curves.items():
        r = refs[name]
        out += [f"**{name}**: AUC(formula) {r['auc_formula']:.4f}, AUC(formula + true persona) {r['auc_oracle']:.4f}, "
                f"gap {r['gap']:.4f} (cross-fitted, {N_FOLDS} folds, {N_NOISE_DRAWS} noise draws per sd). "
                f"Ceiling from the generator's true probability: AUC {r['auc_true']:.4f} with the persona term, "
                f"{r['auc_true_without_persona']:.4f} without, gap {r['true_gap']:.4f}.", "",
                _md(cur), ""]
    c, o, pg = ev.ctx_vs_formula, ev.oracle_vs_formula, ev.persona_vs_formula
    w, ow = ev.persona_weight, ev.oracle_weight
    out += ["### R12 acceptance: pooled persona weight vs oracle weight (same rows)", "",
            "| model (formula + ...) | weight of the persona column [95% CI] |", "|---|---|",
            f"| `ctx_persona_group` (buyer / non_buyer / unclear): `{PERSONA_ACCEPTANCE_COLUMN}` | {_fmt_ci(w)} |",
            f"| true persona indicator (oracle) | {_fmt_ci(ow)} |", "",
            f"CI excludes zero: {w.lo > 0 or w.hi < 0}; point inside the oracle CI: {ow.lo <= w.point <= ow.hi}. "
            f"**R12: {'PASS' if ev.r12_pass else 'FAIL'}.** Formula + `ctx_persona_group` AUC {pg.auc_b:.4f}, "
            f"diff vs formula {_fmt_ci(pg.diff, 4)}, p {pg.p_value:.3f}. The same column in the full combined model "
            f"(all 13 context columns): {_fmt_ci(ev.persona_weight_combined)}.", "",
            "### 6.6 Real run: formula vs formula + context (same rows, cross-fitted)", "",
            f"Rows: {ev.n_rows} labelled sample leads with an ok judgment ({ev.n_wins} wins, {ev.n_persona} with a "
            f"planted persona). Status counts over the whole file: {ev.status_counts}.", "",
            "| model | AUC (out of fold) | diff vs formula [95% CI] | p |", "|---|---|---|---|",
            f"| formula | {c.auc_a:.4f} | | |",
            f"| formula + context | {c.auc_b:.4f} | {_fmt_ci(c.diff, 4)} | {c.p_value:.3f} |",
            f"| formula + true persona (oracle) | {o.auc_b:.4f} | {_fmt_ci(o.diff, 4)} | {o.p_value:.3f} |", "",
            f"**Fraction of the oracle gap recovered: {_fmt_ci(ev.recovered)}** (shared bootstrap resamples).", "",
            f"Oracle persona weight on the same rows: {_fmt_ci(ev.oracle_weight)}.", "",
            "#### Context weights (formula + all 13 context columns fit on all rows; 1,000 bootstrap refits)", "",
            _md(ev.weights), "",
            "#### Secondary, underpowered: unpooled judgments (persona at its eight contract levels)", "",
            _md(ev.fine_weights), "",
            "#### Persona judged vs planted (evaluation only)", "", _md(ev.confusion.reset_index()), "",
            "#### Correlation of each context feature with the formula logit", "",
            f"Formula fit on all pre-2026-05-01 labelled leads. Combined context score (context columns x their "
            f"weights) vs formula logit: r = {ev.context_score_corr:.3f}. "
            f"Largest |r| of a single column: {ev.correlations.r.abs().max():.3f}.", "",
            _md(ev.correlations), "",
            "#### Boilerplate detection (R6) against the generator's copy_paste flag, ok sample rows", "",
            _md(ev.boilerplate), "",
            f"#### Time split (`python -m emva --data data/v2 --context ...`, {ev.time_split_rows} test rows)", "",
            _md(ev.time_split, ".4f"), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    """CLI: ``sample`` writes the sample ids; ``evaluate`` writes the power curve PNG and markdown."""
    ap = argparse.ArgumentParser(prog="python -m emva.eval.context_harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--data", default="data/v2")
    s.add_argument("--out", default="data/v2/context/sample_ids.csv")
    e = sub.add_parser("evaluate")
    e.add_argument("--data", default="data/v2")
    e.add_argument("--judgments", default="data/v2/context/context_judgments.csv")
    e.add_argument("--plot", default="reports/phase6_power_curve.png")
    e.add_argument("--md", default=None, help="write the markdown here (default: stdout)")
    a = ap.parse_args(argv)
    ed = load_eval_data(a.data)
    if a.cmd == "sample":
        ids = sample_leads(ed.labelled, ed.persona)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"lead_id": ids, "planted_persona": (ed.persona[ids] != "none").to_numpy()}).to_csv(a.out, index=False)
        print(f"wrote {len(ids)} ids ({int((ed.persona[ids] != 'none').sum())} persona) to {a.out}")
        return
    ev = evaluate_judgments(ed, a.judgments, a.data)
    lab = ed.labelled
    J = pd.read_csv(a.judgments)
    sample_rows = ed.X.index.isin(J.lead_id[J.status == "ok"]) & lab
    curves, refs = {}, {}
    for name, m in (("all labelled leads", lab), ("the judged sample", pd.Series(sample_rows, index=ed.X.index))):
        y_m, persona_m = ed.X.y[m].to_numpy(dtype=float), (ed.persona[m] != "none").to_numpy()
        curves[name], refs[name] = oracle_power_curve(ed.D[m].to_numpy(), y_m, persona_m)
        refs[name] |= true_probability_gap(y_m, ed.p_true[m].to_numpy(), persona_m)
    plot_power_curve(curves, {"real Haiku run (sample)": ev.recovered}, a.plot)
    text = markdown(curves, refs, ev)
    if a.md:
        Path(a.md).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
