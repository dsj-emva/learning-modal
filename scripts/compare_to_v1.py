"""Distributional fidelity of a regenerated directory against data/v1.

Usage:
  python scripts/compare_to_v1.py --ref data/v1 --gen <dir> [--md out.md]
  python scripts/compare_to_v1.py --gen data/v2 --gen-label v2 --md v2_vs_v1.md    # fidelity of v2 vs v1

Prints (and optionally writes as markdown):
  1. every ground_truth.md table, v1 vs generated, with absolute differences (share of decided leads
     and win rate, in percentage points);
  2. scalar statistics (outcome mix, bot/duplicate share, deal values, time to close, data problems);
  3. the baseline model via emva.pipeline.run (Phase 0 split of the frozen POC, no behaviour change):
     AUC, top-20% wins, top-20% revenue, and the learned weights for the planted effects;
  4. the pipeline's own cleaning view via emva.io / emva.pipeline.build: bot share, duplicate share,
     label balance.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from ground_truth_report import load as gt_load, measure  # noqa: E402
from emva.constants import TEST_FROM  # noqa: E402
from emva.features import FeatureSet  # noqa: E402
from emva.io import flag_bots_and_duplicates, load as emva_load  # noqa: E402
from emva.labels import LEGACY  # noqa: E402
from emva.pipeline import build, run  # noqa: E402

PLANTED = ["band=11-50", "band=51-200", "band=201-1000", "band=1000+", "email=free", "text=specific", "text=vague",
           "text=copy_paste", "time_on_page=60-300s", "time_on_page=>600s", "hesitation_90s=yes", "edits_1_4=yes",
           "form_variant=B", "form_variant=C", "form_variant=D", "channel=meta", "channel=meta_leadads",
           "channel=linkedin", "channel=chatgpt", "channel=organic_direct", "seniority=senior", "seniority=mid",
           "seniority=student", "viewed_pricing=yes", "sessions_3plus=yes", "search_term=brand",
           "business_hours=wkday_9-18", "ip_country=mismatch", "spend=none", "spend=£5k-£25k", "spend=£25k-£100k",
           "spend=£100k+", "crm=hubspot_sf", "hiring=hiring", "no_company=yes"]


def run_baseline(data_dir: str) -> tuple[dict, pd.Series]:
    """Baseline pipeline (emva.pipeline.run) on ``data_dir``: summary metrics and learned log-odds weights."""
    r = run(data_dir, labels=LEGACY, features=FeatureSet.LEGACY)  # frozen baseline labels and features
    return r.summary.iloc[0].to_dict(), r.weights["log_odds"]


def cleaning_view(data_dir: str) -> dict:
    """Bot/duplicate shares and label counts as the baseline pipeline's cleaning step sees them."""
    L = emva_load(data_dir)
    F = flag_bots_and_duplicates(L)
    B = build(L, LEGACY, FeatureSet.LEGACY)
    return {"rows": len(L), "bot_share": F.bot.mean(), "dup_share": (F.dup & ~F.bot).mean(), "kept": len(B),
            "labelled": int(B.y.notna().sum()), "label_pos_rate": B.y.mean(),
            "test_labelled": int((B.y.notna() & (B.created_at >= TEST_FROM)).sum())}


def side_by_side(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Share of decided leads and win rate per level for two tables, with absolute gaps in pp."""
    A = a.set_index("level")
    B_ = b.set_index("level")
    out = pd.DataFrame({"v1_share": A.decided / A.decided.sum(), "gen_share": B_.decided / B_.decided.sum(),
                        "v1_win": A.win_rate, "gen_win": B_.win_rate})
    out["d_share_pp"] = (100 * (out.gen_share - out.v1_share)).abs()
    out["d_win_pp"] = (100 * (out.gen_win - out.v1_win)).abs()
    return out


def to_md(df: pd.DataFrame) -> str:
    """Minimal markdown table (pandas' to_markdown needs tabulate, which is not installed)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def fmt_pct(x: float) -> str:
    """Percentage with one decimal, "n/a" for NaN."""
    return "n/a" if x != x else f"{100 * x:.1f}%"


def seed_spread(ref_dir: str, seeds: list[int], n: int) -> str:
    """Regenerate with several seeds; report where v1 falls inside the seed-to-seed spread."""
    import generate_data_v1 as g
    metrics, weights, wins = [], [], []
    for sd in seeds:
        with tempfile.TemporaryDirectory() as d:
            cfg = g.Config(seed=sd, n=n)
            g.write(g.generate(cfg), d, cfg)
            res, w = run_baseline(d)
            metrics.append({k: res[k] for k in ["auc", "top20_wins", "top20_revenue"]})
            weights.append(w)
            m = measure(gt_load(d))
            wins.append({f"{k}:{lvl}": wr for k, t in m["tables"].items() for lvl, wr in zip(t.level, t.win_rate)})
    br, wr = run_baseline(ref_dir)
    ref = measure(gt_load(ref_dir))
    rwin = {f"{k}:{lvl}": w for k, t in ref["tables"].items() for lvl, w in zip(t.level, t.win_rate)}
    rdec = {f"{k}:{lvl}": d for k, t in ref["tables"].items() for lvl, d in zip(t.level, t.decided)}
    M, W, Wi = pd.DataFrame(metrics), pd.DataFrame(weights), pd.DataFrame(wins)
    out = [f"### Seed-to-seed spread ({len(seeds)} seeds: {seeds}, n = {n})\n",
           "| metric | v1 | generated mean | generated sd | min | max |", "|---|---|---|---|---|---|"]
    for k in M:
        out.append(f"| {k} | {br[k]} | {M[k].mean():.3f} | {M[k].std():.3f} | {M[k].min():.3f} | {M[k].max():.3f} |")
    out += ["", "Planted-effect weights: v1 value, generated mean and sd across seeds, z = (v1 - mean) / sd:", "",
            "| weight | v1 | gen mean | gen sd | z |", "|---|---|---|---|---|"]
    for k in PLANTED:
        mu, sdv = W[k].mean(), W[k].std()
        out.append(f"| {k} | {wr[k]:.3f} | {mu:.3f} | {sdv:.3f} | {(wr[k] - mu) / sdv:+.1f} |")
    z = pd.Series({k: (rwin[k] - Wi[k].mean()) / Wi[k].std() for k in Wi if rdec[k] >= 100 and Wi[k].std() > 0})
    gap = pd.Series({k: 100 * abs(rwin[k] - Wi[k].mean()) for k in z.index})
    out += ["", f"Win-rate cells with >=100 decided v1 leads: {len(z)}. |z| > 2 in {int((z.abs() > 2).sum())} cells, "
            f"|z| > 3 in {int((z.abs() > 3).sum())}. Mean abs gap to the seed mean: {gap.mean():.2f} pp.", "",
            "Cells with |z| > 2:", "", "| cell | v1 win rate | gen mean | gen sd | z |", "|---|---|---|---|---|"]
    for k in z[z.abs() > 2].sort_values(key=abs, ascending=False).index:
        out.append(f"| {k} | {fmt_pct(rwin[k])} | {fmt_pct(Wi[k].mean())} | {100 * Wi[k].std():.1f} pp | {z[k]:+.1f} |")
    return "\n".join(out) + "\n"


def relabel(text: str, label: str) -> str:
    """Rename the compared directory from "generated"/"gen" to ``label`` in the markdown headers."""
    for a, b in [("generated", label), ("| gen share |", f"| {label} share |"), ("| gen win rate |", f"| {label} win rate |"),
                 ("gen_share", f"{label}_share"), ("gen_win", f"{label}_win"), ("| gen mean |", f"| {label} mean |"),
                 ("| gen sd |", f"| {label} sd |"), ('"gen"', f'"{label}"')]:
        text = text.replace(a, b)
    return text


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(ROOT, "data", "v1"))
    ap.add_argument("--gen", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--seeds", default="", help="comma-separated seeds for a seed-spread section, e.g. 1,2,3,4,5")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--gen-label", default="generated", help='name for --gen in the tables, e.g. "v2"')
    a = ap.parse_args(argv)
    ref, gen = measure(gt_load(a.ref)), measure(gt_load(a.gen))
    md = []
    worst = []
    md.append("### Ground-truth tables: share of decided leads and win rate (v1 vs generated)\n")
    for k in ref["tables"]:
        s = side_by_side(ref["tables"][k], gen["tables"][k])
        md.append(f"**{k}**\n")
        md.append("| level | v1 share | gen share | abs diff (pp) | v1 win rate | gen win rate | abs diff (pp) |")
        md.append("|---|---|---|---|---|---|---|")
        for lvl, r in s.iterrows():
            md.append(f"| {lvl} | {fmt_pct(r.v1_share)} | {fmt_pct(r.gen_share)} | {r.d_share_pp:.1f} | "
                      f"{fmt_pct(r.v1_win)} | {fmt_pct(r.gen_win)} | {r.d_win_pp:.1f} |")
            worst.append((k, lvl, r.d_share_pp, r.d_win_pp, ref["tables"][k].set_index("level").decided.get(lvl, 0)))
        md.append("")
    W = pd.DataFrame(worst, columns=["table", "level", "d_share_pp", "d_win_pp", "v1_decided"])
    md.append("### Largest deviations\n")
    md.append("Largest win-rate gaps among levels with at least 100 decided v1 leads:\n")
    md.append(to_md(W[W.v1_decided >= 100].sort_values("d_win_pp", ascending=False).head(10).round(1)))
    md.append("\nLargest share gaps:\n")
    md.append(to_md(W.sort_values("d_share_pp", ascending=False).head(10).round(1)))
    md.append(f"\nMean absolute win-rate gap over all levels with >=100 decided v1 leads: "
              f"{W[W.v1_decided >= 100].d_win_pp.mean():.2f} pp; mean share gap: {W.d_share_pp.mean():.2f} pp.\n")

    md.append("### Scalars\n")
    md.append("| statistic | v1 | generated | abs diff |")
    md.append("|---|---|---|---|")
    for k, v in ref["scalars"].items():
        g = gen["scalars"].get(k)
        if isinstance(v, dict):
            md.append(f"| {k} | {v} | {g} | |")
        else:
            diff = abs(float(g) - float(v))
            f = (lambda x: f"{x:.4f}") if isinstance(v, float) and abs(v) < 2 else (lambda x: f"{x:,.1f}" if isinstance(x, float) else f"{x:,}")
            md.append(f"| {k} | {f(v)} | {f(g)} | {f(diff) if isinstance(v, float) else f'{int(diff):,}'} |")
    md.append("")

    md.append("### Baseline pipeline cleaning view (emva.io.load / emva.pipeline.build)\n")
    cr, cg = cleaning_view(a.ref), cleaning_view(a.gen)
    md.append("| statistic | v1 | generated |")
    md.append("|---|---|---|")
    for k in cr:
        f = (lambda x: f"{x:.4f}") if isinstance(cr[k], float) else (lambda x: f"{x:,}")
        md.append(f"| {k} | {f(cr[k])} | {f(cg[k])} |")
    md.append("")

    md.append("### Baseline model (emva.pipeline.run = frozen baseline behaviour)\n")
    br, wr = run_baseline(a.ref)
    bg, wg = run_baseline(a.gen)
    md.append("| metric | v1 | generated | abs diff |")
    md.append("|---|---|---|---|")
    for k in ["auc", "brier", "top20_wins", "top20_revenue"]:
        md.append(f"| {k} | {br[k]} | {bg[k]} | {abs(bg[k] - br[k]):.3f} |")
    md.append("\nLearned weights (log-odds) for the planted effects:\n")
    md.append("| weight | v1 | generated | abs diff |")
    md.append("|---|---|---|---|")
    for k in PLANTED:
        v, g = wr.get(k, np.nan), wg.get(k, np.nan)
        md.append(f"| {k} | {v:.3f} | {g:.3f} | {abs(g - v):.3f} |")
    both = pd.concat([wr, wg], axis=1, keys=["v1", "gen"]).dropna()
    md.append(f"\nAll {len(both)} weights: mean abs diff {(both.v1 - both.gen).abs().mean():.3f}, "
              f"correlation {both.v1.corr(both.gen):.3f}.\n")
    if a.seeds:
        md.append(seed_spread(a.ref, [int(x) for x in a.seeds.split(",")], a.n))
    text = "\n".join(md)
    if a.gen_label != "generated":
        text = relabel(text, a.gen_label)
    print(text)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
