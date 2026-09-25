"""Phase 2 study: evidence for the feature and leakage changes (plan 2.1 to 2.5, 2.7 and the acceptance).

``python -m emva.eval.phase2_study [--data data/v1]`` prints markdown with:

1. 2.3 company-name enrichment: domain misses among cleaned leads, how many the typed company
   name matches, and (ground truth, evaluation only) whether the matched company is right.
2. 2.5 boilerplate detector: threshold, precision and recall against the generator's
   ``text_category`` (ground truth), next to the legacy prefix rule, and the similarity margin.
3. 2.1 Meta lead-ads leads: bucket distribution of every behavioural feature, legacy vs v2.
4. Weights before/after for every design column (baseline, v2 with legacy labels, v2 with
   horizon labels) and pairs of coefficients equal to three decimals.
5. 2.7 collinearity: max |corr| of each feature set's design.
6. 2.4 deal-value model: the fixed and the estimated residual sd, their corrections and what
   the change does to value and revenue capture.

Ground-truth readers: this module, Phase 1's ground-truth reference script and Phase 4's
``emva/eval/ceiling.py``; nothing imports this module.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from emva.boilerplate import boilerplate_similarity
from emva.constants import BOILERPLATE_SIMILARITY_THRESHOLD, MAX_ABS_DESIGN_CORR, MISSING
from emva.eval.bootstrap import N_RESAMPLES, SEED, paired_auc
from emva.eval.collinearity import check_collinearity, summary_row
from emva.eval.metrics import top_share
from emva.eval.regression import BASELINE_DEAL_LOG_RESIDUAL_SD, FROZEN_WEIGHTS, read_weights
from emva.eval.report import frozen_test_labels, md_table
from emva.features import BEHAVIOURAL_FEATURES, FeatureSet, add_features, add_features_v2, text_cat
from emva.io import clean, load
from emva.labels import HORIZON, LEGACY
from emva.pipeline import PipelineResult, run
from emva.value import deal_value_design, fit_deal_value, predict_deal_value, realised_revenue

TRUTH_FILE: str = "ground_truth_labels.csv"


def _truth(data: str | Path) -> pd.DataFrame:
    """The generator's per-lead truth, indexed by ``lead_id`` (evaluation only)."""
    return pd.read_csv(Path(data) / TRUTH_FILE, index_col="lead_id")


def _pct(n: int, d: int) -> str:
    """``"948 (32.6%)"``."""
    return f"{n} ({n / d:.1%})" if d else f"{n} (n/a)"


def name_enrichment_section(data: str | Path) -> list[str]:
    """2.3: domain misses, name matches by source column, and match correctness against the truth."""
    X = clean(load(data))
    miss = X.co_employee_band.isna()
    by_name = X.enrichment_source.eq("name")
    from_col = by_name & X.company_name.notna() & X.company_name.map(str).ne("")
    truth = _truth(data).reindex(X.index)
    matched = X[by_name]
    band_ok = matched.en_employee_band.eq(truth.employee_band[by_name])
    sector_ok = matched.en_sector.eq(truth.sector[by_name])
    typed = (X.company_name.notna() | X.a_company.fillna("").ne("")) & miss
    rows = [
        {"quantity": "cleaned leads", "value": len(X)},
        {"quantity": "email-domain misses (no companies.csv row by domain)", "value": int(miss.sum())},
        {"quantity": "... with a typed company name (company_name or the company answer)", "value": _pct(int(typed.sum()), int(miss.sum()))},
        {"quantity": "... matched by exact normalised name", "value": _pct(int(by_name.sum()), int(miss.sum()))},
        {"quantity": "... of which the historical_leads.company_name column matched", "value": int(from_col.sum())},
        {"quantity": "still without enrichment (enrichment_missing=yes)", "value": _pct(int(X.enrichment_source.isna().sum()), len(X))},
        {"quantity": "truth: matched company's employee band = generator's band (ground truth)", "value": _pct(int(band_ok.sum()), int(by_name.sum()))},
        {"quantity": "truth: matched company's sector = generator's sector (ground truth)", "value": _pct(int(sector_ok.sum()), int(by_name.sum()))},
        {"quantity": "truth: generator has_company among the matched", "value": _pct(int(truth.has_company[by_name].sum()), int(by_name.sum()))},
        {"quantity": "truth: generator has_company among unmatched domain misses", "value": _pct(int(truth.has_company[miss & ~by_name].sum()), int((miss & ~by_name).sum()))},
    ]
    unmatched = X.company_name[miss & ~by_name].map(lambda s: s if isinstance(s, str) else "(blank)")
    return ["## 2.3 Company-name enrichment", "",
            "Exact match after normalisation (lower-case, dots/apostrophes dropped, other punctuation to spaces, "
            "whitespace collapsed, trailing legal suffixes stripped); no fuzzy matching. The typed "
            "`historical_leads.company_name` is tried first, then the `company` form answer. Names shared by two "
            "companies after normalisation would be skipped (none on this data).", "",
            md_table(pd.DataFrame(rows)), "",
            "Most common unmatched typed names among domain misses: "
            + ", ".join(f"{k} ({v})" for k, v in unmatched.value_counts().head(5).items()) + ".", ""]


def boilerplate_section(data: str | Path) -> list[str]:
    """2.5: precision/recall of the similarity detector and the legacy prefix rule against the truth."""
    X = clean(load(data))
    truth = _truth(data).text_category.reindex(X.index).eq("copy_paste")
    text = X.a_what_to_solve
    sim = text.map(boilerplate_similarity)
    rows = []
    for name, pred in [(f"v2 similarity (Jaccard >= {BOILERPLATE_SIMILARITY_THRESHOLD})",
                        sim >= BOILERPLATE_SIMILARITY_THRESHOLD),
                       ("legacy prefix rule", text.map(text_cat).eq("copy_paste"))]:
        tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
        rows.append({"detector": name, "flagged": int(pred.sum()), "true copy_paste": int(truth.sum()), "TP": tp,
                     "FP": fp, "FN": fn, "precision": f"{tp / (tp + fp):.3f}" if tp + fp else "n/a",
                     "recall": f"{tp / (tp + fn):.3f}" if tp + fn else "n/a"})
    return ["## 2.5 Boilerplate detector", "",
            f"Cleaned leads ({len(X)}); truth = the generator's `text_category == copy_paste` (ground truth, "
            "evaluation only). Similarity = token-set Jaccard with the closest of the snippets in "
            "`emva/boilerplate.py`.", "",
            md_table(pd.DataFrame(rows)), "",
            f"Similarity margin: lowest similarity among true copy_paste texts {sim[truth].min():.3f}; highest "
            f"among all other texts {sim[~truth].max():.3f}. "
            + ("Any threshold between the two gives the same result on this data." if sim[truth].min() > sim[~truth].max()
               else "The two overlap: no threshold separates boilerplate from other text on this data."), ""]


def lead_ads_section(data: str | Path) -> list[str]:
    """2.1: bucket distribution of the behavioural features for Meta lead-ads leads, legacy vs v2."""
    base = clean(load(data))
    legacy, v2 = add_features(base.copy()), add_features_v2(base.copy())
    la = legacy.channel.eq("meta_leadads")
    rows = []
    for feat in (*BEHAVIOURAL_FEATURES, "session_missing"):
        for name, X in [("legacy", legacy), ("v2", v2)]:
            counts = X.loc[la, feat].value_counts() if feat in X else pd.Series(dtype=int)
            rows.append({"feature": feat, "feature set": name,
                         "buckets (lead-ads leads)": ", ".join(f"{k}: {v}" for k, v in counts.items()) or "(no feature)"})
    other = ~la & v2.session_missing.eq("yes")
    return ["## 2.1 Meta lead-ads leads: behavioural buckets", "",
            f"{int(la.sum())} cleaned Meta lead-ads leads. They have no on-site session, so every behavioural "
            "input is blank. Legacy silently puts them in each feature's reference level. v2 also leaves the seven "
            "behavioural features at their reference level but sets `session_missing=yes`, so the indicator carries "
            "the effect of the absent session (the intended encoding: the column set is fixed by the spec). "
            f"Other cleaned leads with `session_missing=yes` in this data: {int(other.sum())}.", "",
            md_table(pd.DataFrame(rows)), ""]


def _weight_table(results: dict[str, PipelineResult]) -> pd.DataFrame:
    """Log-odds per design column for each model (baseline from weights.csv)."""
    cols: dict[str, pd.Series] = {"baseline (weights.csv)": read_weights(FROZEN_WEIGHTS).log_odds}
    for name, r in results.items():
        cols[name] = r.weights.log_odds.map(lambda v: f"{v:.3f}")
    T = pd.DataFrame(cols)
    T["baseline (weights.csv)"] = T["baseline (weights.csv)"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return T.fillna("").rename_axis("column").reset_index().sort_values("column", kind="stable")


def _ties(weights: pd.DataFrame, r: PipelineResult) -> list[str]:
    """Pairs of design columns whose log-odds are equal to three decimals, with their training correlation."""
    w, D = weights.log_odds, r.design[r.train]
    return [f"{a} = {b} ({w[a]:.3f}; corr {D[a].corr(D[b]):+.3f})"
            for a, b in combinations(w.index, 2) if w[a] == w[b]]


def weights_section(results: dict[str, PipelineResult], baseline_run: PipelineResult) -> list[str]:
    """Before/after weights for every design column and coefficient ties.

    ``baseline_run`` is the legacy/legacy pipeline run (equal to the frozen baseline), used for
    the training correlation of the baseline's tied pairs.
    """
    out = ["## Weights before and after", "",
           "Log-odds per design column. Empty = the column does not exist in that design. The baseline column is "
           "the frozen `baseline/weights.csv` (fitted on data/v1, whatever `--data` is). The baseline has "
           "`no_company`; v2 has the `session_missing` and `enrichment_missing` indicators and `band=missing`, and "
           "drops c_budget, c_timeline, ip_type and edits_1_4 (2.6).", "", md_table(_weight_table(results)), "",
           "### Coefficients equal to three decimals", ""]
    base_ties = _ties(read_weights(FROZEN_WEIGHTS), baseline_run)
    out.append(f"- baseline (weights.csv): {'; '.join(base_ties) if base_ties else 'none'}")
    for name, r in results.items():
        t = _ties(r.weights, r)
        out.append(f"- {name}: {'; '.join(t) if t else 'none'}")
    return out + [""]


def collinearity_section(results: dict[str, PipelineResult]) -> list[str]:
    """2.7: max |corr| and the verdict for each model's design on its training rows."""
    rows = [{"model": name, "design columns": r.design.shape[1], "training rows": int(r.train.sum()),
             **summary_row(check_collinearity(r.design[r.train]))} for name, r in results.items()]
    return ["## 2.7 Collinearity", "",
            f"Threshold |corr| > {MAX_ABS_DESIGN_CORR} on the training rows of each model's design.", "",
            md_table(pd.DataFrame(rows)), ""]


def paired_features_section(data: str | Path, n_resamples: int, seed: int) -> list[str]:
    """AUC of v2 vs legacy features trained on the same labels, paired on both frozen test sets."""
    _, _, y_a, y_b = frozen_test_labels(load(data))
    rows = []
    for labels in (LEGACY, HORIZON):
        p_old = run(data, labels=labels, features=FeatureSet.LEGACY).X.p_formula
        p_new = run(data, labels=labels, features=FeatureSet.V2).X.p_formula
        for tname, y in [("(a) legacy test set", y_a), ("(b) mature test set", y_b)]:
            c = paired_auc(y.values, p_old.loc[y.index].values, p_new.loc[y.index].values, n_resamples, seed=seed)
            rows.append({"labels": labels.describe(), "test set": f"{tname} (n={len(y)})",
                         "AUC legacy features": f"{c.auc_a:.3f}", "AUC v2 features": f"{c.auc_b:.3f}",
                         "v2 − legacy [95% CI]": f"{c.diff.point:+.3f} [{c.diff.lo:+.3f}, {c.diff.hi:+.3f}]",
                         "bootstrap p": f"{c.p_value:.3f}"})
    return ["## Feature change alone: v2 vs legacy features, same labels", "",
            f"Paired bootstrap ({n_resamples} resamples, seed {seed}) of AUC(v2) − AUC(legacy features), both models "
            "trained on the same label definition and scored on the same frozen test rows.", "",
            md_table(pd.DataFrame(rows)), ""]


def value_section(data: str | Path, results: dict[str, PipelineResult]) -> list[str]:
    """2.4: fixed vs estimated residual sd, corrections, and the effect on value and revenue capture."""
    L = load(data)
    _, _, y_a, y_b = frozen_test_labels(L)
    rows = []
    for name, r in results.items():
        if not r.deal_value.estimated:  # only models that estimate the sd are compared with the fixed one
            continue
        X = r.X
        M = deal_value_design(X)
        fixed = fit_deal_value(M, X.deal_value, r.train, log_residual_sd=BASELINE_DEAL_LOG_RESIDUAL_SD)
        dv_fixed = pd.Series(predict_deal_value(fixed, M), index=X.index)
        T = X[r.test]
        for label_, dv_model, dv in [("fixed (baseline constant)", fixed, dv_fixed),
                                     ("estimated from training residuals", r.deal_value, X.deal_value_hat)]:
            value = X.p_formula * dv
            row = {"model": name, "residual sd": label_, "sd": f"{dv_model.log_residual_sd:.4f}",
                   "correction": f"{dv_model.correction:.4f}", "training deals": dv_model.n_train}
            for tname, y in [("(a)", y_a), ("(b)", y_b)]:
                rev = pd.Series(np.where(y == 1, L.deal_value.reindex(y.index).fillna(0), 0), index=y.index)
                row[f"top-20% revenue by p×value {tname}"] = f"{top_share(rev, value.loc[y.index].values):.4f}"
            won = T.y.eq(1) & T.deal_value.notna()
            row["mean predicted / mean recorded deal value, won test leads"] = f"{dv[won.index[won]].mean() / T.deal_value[won].mean():.3f}"
            imputed = T.assign(deal_value_hat=dv.loc[T.index])
            row["baseline-style top20_revenue (imputes blanks)"] = f"{top_share(realised_revenue(imputed), value.loc[T.index].values):.4f}"
            rows.append(row)
    return ["## 2.4 Deal-value model: residual sd", "",
            "The correction is exp(sd²/2). It multiplies every lead's expected deal value by the same factor, so "
            "it cannot change any ranking: top-20% capture by p×value is identical by construction. It changes the "
            "level of the value sent, checked here against recorded deal values of won test leads. Revenue "
            "(a)/(b) = recorded deal value on the frozen legacy / mature test sets; the baseline-style figure fills "
            "blank deal values with the model's own prediction (so it can move slightly).", "",
            md_table(pd.DataFrame(rows)), ""]


def build_study(data: str | Path, n_resamples: int = N_RESAMPLES, seed: int = SEED) -> str:
    """Every section as one markdown document."""
    results: dict[str, PipelineResult] = {
        "legacy features, horizon labels": run(data, labels=HORIZON, features=FeatureSet.LEGACY),
        "v2, legacy labels": run(data, labels=LEGACY, features=FeatureSet.V2),
        "v2, horizon labels (default)": run(data, labels=HORIZON, features=FeatureSet.V2),
    }
    legacy_legacy = run(data, labels=LEGACY, features=FeatureSet.LEGACY)
    return "\n".join([
        "# Phase 2 study", "", f"Data: `{data}`. Missing level name: `{MISSING}`.", "",
        *name_enrichment_section(data),
        *boilerplate_section(data),
        *lead_ads_section(data),
        *weights_section(results, legacy_legacy),
        *collinearity_section({"legacy features, legacy labels (baseline)": legacy_legacy, **results}),
        *paired_features_section(data, n_resamples, seed),
        *value_section(data, results),
    ])


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: print the study."""
    ap = argparse.ArgumentParser(prog="python -m emva.eval.phase2_study")
    ap.add_argument("--data", default="data/v1")
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    print(build_study(a.data, a.n_resamples, a.seed))


if __name__ == "__main__":
    main()
