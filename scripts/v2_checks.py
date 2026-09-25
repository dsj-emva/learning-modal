"""Measurements of the Phase 5.2-5.8 mechanisms on a generated data directory.

Used by ground_truth_report.py (the "v2 mechanisms" section of ground_truth.md), by tests/test_generate_data_v2.py
and for reports/phase5.md (``python scripts/v2_checks.py --data data/v2 [--ref data/v1]``). Reads
ground_truth_labels.csv, so like the generator it sits on the evaluation side of ground rule 2; nothing under
emva/ may import it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emva.constants import BOT_MIN_TIME_ON_PAGE_S  # noqa: E402
from emva.features import text_cat  # noqa: E402
from emva.io import flag_bots_and_duplicates, load as emva_load  # noqa: E402
from ground_truth_report import load as gt_load, tier_of  # noqa: E402

if TYPE_CHECKING:
    from generate_data_v1 import Config

TEXT_ORDER = ["specific", "neutral", "vague", "copy_paste"]
LEGAL_SUFFIXES = {"ltd", "limited", "inc", "incorporated", "gmbh", "sas", "bv"}
CONSENT_COLUMNS = ["time_on_page_s", "hesitation_ms", "field_edit_count", "fields_edited", "pasted_text",
                   "sessions_before_convert", "days_since_first_visit", "returned_visitor", "pages_visited",
                   "viewed_pricing", "scroll_depth_pct", "fbp", "ip", "ip_country", "ip_city", "is_datacenter_ip"]


def merged(t: dict) -> pd.DataFrame:
    """historical_leads joined with ground_truth_labels (label columns prefixed ``g_``), plus the regex category."""
    X = t["historical_leads"].merge(t["ground_truth_labels"].add_prefix("g_"), left_on="lead_id", right_on="g_lead_id")
    X["ans"] = X.answers.apply(json.loads)
    X["regex_text"] = X.ans.apply(lambda a: text_cat(a.get("what_to_solve")))
    X["is_dup"] = X.g_duplicate_of.notna()
    X["web"] = X.utm_medium.ne("lead_form")
    return X


def _flag(s: pd.Series) -> pd.Series:
    """Boolean view of a CSV column that may hold True/False, 'True'/'False' or blanks."""
    return s.astype(str).eq("True")


# ---------------------------------------------------------------------------------------------
# 5.2 regex agreement and 5.8 bot rule (the two Phase 5 acceptance numbers)
# ---------------------------------------------------------------------------------------------

def regex_confusion(X: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Share of leads where ``emva.features.text_cat`` equals the generator's category, and the confusion matrix
    (rows = generator category, columns = regex category, counts)."""
    cm = pd.crosstab(X.g_text_category, X.regex_text).reindex(index=TEXT_ORDER, columns=TEXT_ORDER, fill_value=0)
    return float((X.g_text_category == X.regex_text).mean()), cm


def bot_rule(d: str) -> dict:
    """Precision and recall of ``emva.io``'s bot rule (headless UA or under BOT_MIN_TIME_ON_PAGE_S) vs is_bot."""
    F = flag_bots_and_duplicates(emva_load(d))
    G = pd.read_csv(os.path.join(d, "ground_truth_labels.csv")).set_index("lead_id")
    truth = G.is_bot.reindex(F.index).astype(bool)
    fast = _flag(G.fast_human.reindex(F.index)) if "fast_human" in G else pd.Series(False, index=F.index)
    consent = _flag(G.consent_declined.reindex(F.index)) if "consent_declined" in G else pd.Series(False, index=F.index)
    tp = int((F.bot & truth).sum())
    fp = int((F.bot & ~truth).sum())
    fn = int((~F.bot & truth).sum())
    return {"flagged": int(F.bot.sum()), "bots": int(truth.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "fp_fast_humans": int((F.bot & ~truth & fast).sum()),
            "fn_consent_declined": int((~F.bot & truth & consent).sum()),
            "threshold_s": BOT_MIN_TIME_ON_PAGE_S}


# ---------------------------------------------------------------------------------------------
# 5.3 enrichment dropout
# ---------------------------------------------------------------------------------------------

def normalise_company(name: object) -> str | None:
    """Lower-case, drop punctuation, '&' and hyphens, collapse spaces and strip a trailing legal suffix."""
    if not isinstance(name, str) or not name.strip():
        return None
    s = re.sub(r"[&\-]", " ", name.lower())
    s = re.sub(r"[.,]", "", s)
    words = s.split()
    while words and words[-1] in LEGAL_SUFFIXES:
        words = words[:-1]
    return " ".join(words) or None


def dropout_stats(t: dict, X: pd.DataFrame) -> dict:
    """Domain-miss leads (business email whose domain is not in companies.csv) and how many of them typed a
    company name that normalises to the right row of companies.csv; also the same for every lead without a
    domain match (free emails and bots included)."""
    CO = t["companies"]
    co_by_norm = CO.assign(norm=CO.company_name.map(normalise_company)).drop_duplicates("norm").set_index("norm")
    domains = set(CO.domain.str.lower())
    dom = X.company_domain.str.lower().str.strip()
    no_join = ~dom.isin(domains)
    business = ~X.g_is_free_email.astype(bool) & ~X.g_is_bot.astype(bool) & X.g_has_company.astype(bool)
    typed_norm = X.company_name.map(normalise_company)
    right = X.g_company_domain.map(t["gt_companies"].set_index("domain").company_name)
    match_any = typed_norm.isin(co_by_norm.index)
    match_right = typed_norm.notna() & (typed_norm == right.map(normalise_company))
    used = sorted(set(X.g_company_domain[business & ~X.is_dup]))
    in_csv = set(CO.domain)
    miss = business & no_join
    return {"business_domains": len(used), "business_domains_missing": sum(d not in in_csv for d in used),
            "business_leads": int(business.sum()), "domain_miss_leads": int(miss.sum()),
            "domain_miss_typed_name": int((miss & typed_norm.notna()).sum()),
            "domain_miss_matchable": int((miss & match_right).sum()),
            "no_join_leads": int(no_join.sum()), "no_join_matchable_any": int((no_join & match_any).sum()),
            "typed_names": int(X.company_name.notna().sum()),
            "typed_names_exact": int((X.company_name == right).sum())}


# ---------------------------------------------------------------------------------------------
# 5.4 consent, 5.7 persona, 5.8 fast humans
# ---------------------------------------------------------------------------------------------

def consent_stats(X: pd.DataFrame) -> dict:
    """Share of website rows with consent declined, and whether exactly the consent columns are blank on them."""
    cd = _flag(X.g_consent_declined)
    web = X.web
    blank = X.loc[cd, CONSENT_COLUMNS].isna().all(axis=1)
    return {"web_rows": int(web.sum()), "declined": int(cd.sum()), "share_of_web": float(cd[web].mean()),
            "declined_lead_ads": int((cd & ~web).sum()), "all_columns_blank": bool(blank.all()),
            "time_on_page_blank_otherwise": int((~cd & web & X.time_on_page_s.isna()).sum())}


def persona_stats(X: pd.DataFrame) -> dict:
    """Persona shares, decided win rates with and without a persona, and the regex category distribution of
    persona vs non-persona texts (genuine-human original leads)."""
    O = X[~X.is_dup]
    has = O.g_context_persona.notna()
    humans = O[~O.g_is_bot.astype(bool)]
    hp = humans.g_context_persona.notna()
    dist = pd.DataFrame({"persona": humans.regex_text[hp].value_counts(normalize=True),
                         "no_persona": humans.regex_text[~hp].value_counts(normalize=True)}).reindex(TEXT_ORDER).fillna(0)
    true_dist = pd.DataFrame({"persona": humans.g_text_category[hp].value_counts(normalize=True),
                              "no_persona": humans.g_text_category[~hp].value_counts(normalize=True)}).reindex(TEXT_ORDER).fillna(0)
    dec = O.g_outcome.isin(["won", "lost"])
    won = O.g_outcome.eq("won")
    return {"share_all_leads": float(X.g_context_persona.notna().mean()), "share_originals": float(has.mean()),
            "share_human_originals": float(hp.mean()), "by_persona": O.g_context_persona.value_counts().to_dict(),
            "win_rate_persona": float(won[dec & has].mean()), "win_rate_no_persona": float(won[dec & ~has].mean()),
            "decided_persona": int((dec & has).sum()), "regex_dist": dist, "true_dist": true_dist,
            "regex_dist_max_gap": float((dist.persona - dist.no_persona).abs().max())}


def fast_human_stats(X: pd.DataFrame) -> dict:
    """Fast humans as a share of genuine-human website sessions with recorded telemetry."""
    fast = _flag(X.g_fast_human)
    base = X.web & ~X.g_is_bot.astype(bool) & ~_flag(X.g_consent_declined)
    return {"fast_humans": int(fast.sum()), "eligible": int(base.sum()), "share": float(fast[base].mean()),
            "humans_under_15s": int((base & (X.time_on_page_s < 15)).sum())}


# ---------------------------------------------------------------------------------------------
# 5.6 ghosting and 5.5 interactions
# ---------------------------------------------------------------------------------------------

def ghosting_stats(t: dict, X: pd.DataFrame) -> dict:
    """Correlation of never-contacted with the status-quo tier (A=0, B=1, C=2) and with true p (originals)."""
    O = X[~X.is_dup]
    tier = pd.Series([tier_of(a, p, t["rules"]) for a, p in zip(O.ans, O.pages_visited)], index=O.index)
    rank = tier.map({"A": 0, "B": 1, "C": 2})
    nc = O.g_never_contacted.astype(float)
    p = O.g_p_close_true
    return {"never_contacted_share": float(nc.mean()), "corr_tier": float(nc.corr(rank)),
            "corr_p": float(nc.corr(p)), "corr_tier_vs_p": float(rank.corr(p)),
            "never_by_tier": nc.groupby(tier).mean().round(3).to_dict()}


def logit_fit(A: np.ndarray, y: np.ndarray, ridge: float = 1e-4, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by Newton's method (tiny ridge for stability): coefficients and standard errors."""
    b = np.zeros(A.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-(A @ b)))
        H = A.T @ (A * (mu * (1 - mu))[:, None]) + ridge * np.eye(A.shape[1])
        step = np.linalg.solve(H, A.T @ (y - mu) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    mu = 1 / (1 + np.exp(-(A @ b)))
    H = A.T @ (A * (mu * (1 - mu))[:, None]) + ridge * np.eye(A.shape[1])
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def interaction_stats(X: pd.DataFrame) -> pd.DataFrame:
    """Outcome logistic regression on decided original leads: main effects of channel, band, seniority, form,
    free email, true text category and persona, plus the two planted interaction terms. Returns the
    interaction rows (coefficient, 95% CI, cell size)."""
    D = X[~X.is_dup & X.g_outcome.isin(["won", "lost"])].copy()
    ch = np.where(D.g_is_lead_ads.astype(bool), "leadads", D.g_channel)
    parts = [pd.get_dummies(pd.Series(ch, index=D.index), prefix="ch").drop(columns="ch_google"),
             pd.get_dummies(D.g_employee_band, prefix="band").drop(columns="band_1-10"),
             pd.get_dummies(D.g_seniority, prefix="sen").drop(columns="sen_junior"),
             pd.get_dummies(D.form_variant, prefix="form").drop(columns="form_A"),
             pd.get_dummies(D.g_text_category, prefix="text").drop(columns="text_neutral")]
    M = pd.concat(parts, axis=1).astype(float)
    M["free"] = D.g_is_free_email.astype(float)
    if "g_context_persona" in D:
        M["persona"] = D.g_context_persona.notna().astype(float)
    M["linkedin_x_51plus"] = ((D.g_channel == "linkedin") & D.g_employee_band.isin(["51-200", "201-1000", "1000+"])).astype(float)
    M["senior_x_formD"] = ((D.g_seniority == "senior") & (D.form_variant == "D")).astype(float)
    A = np.column_stack([np.ones(len(M)), M.to_numpy()])
    b, se = logit_fit(A, D.g_outcome.eq("won").to_numpy(float))
    names = ["intercept"] + list(M.columns)
    out = pd.DataFrame({"coef": b, "se": se}, index=names)
    out["lo"], out["hi"] = out.coef - 1.96 * out.se, out.coef + 1.96 * out.se
    out["cell_decided"] = [len(M)] + [int(M[c].sum()) for c in M.columns]
    return out.loc[["linkedin_x_51plus", "senior_x_formD"] + (["persona"] if "persona" in M else [])]


# ---------------------------------------------------------------------------------------------
# Planted-effect recovery: regress logit(p_close_true) on every planted feature
# ---------------------------------------------------------------------------------------------

def planted_design(t: dict, X: pd.DataFrame, cfg: "Config") -> tuple[pd.DataFrame, pd.Series, dict]:
    """Design matrix of every planted feature for original leads whose true session is recorded (no consent
    decline, no fast-human edit, no duplicate), the logit of p_close_true, and the planted value per column."""
    O = X[~X.is_dup].copy()
    if "g_consent_declined" in O:
        O = O[~_flag(O.g_consent_declined) & ~_flag(O.g_fast_human)]
    O = O[(O.g_p_close_true > 0.001) & (O.g_p_close_true < 0.999)]     # p is rounded to 4 dp in the labels
    CO = t["gt_companies"].set_index("domain")
    has_co = O.g_company_domain.notna()
    spend = O.g_company_domain.map(CO.monthly_ad_spend_band)
    crm = O.g_company_domain.map(CO.crm_platform)
    hiring = _flag(O.g_company_domain.map(CO.is_hiring))
    top = O.time_on_page_s
    a_country = O.ans.apply(lambda a: a.get("country"))
    biz = ~O.submitted_weekday.isin(["Saturday", "Sunday"]) & O.local_submit_hour.between(9, 17)
    r = O.g_reply_hours
    cols, planted = {}, {}

    def add(name: str, values: pd.Series, eff: float) -> None:
        cols[name] = values.astype(float)
        planted[name] = eff
    for b in ["11-50", "51-200", "201-1000", "1000+"]:          # "none" = 1-10 apart from having no company
        add(f"band={b}", O.g_employee_band == b, cfg.band_eff[b] - cfg.band_eff["1-10"])
    add("free_email", O.g_is_free_email.astype(bool), cfg.free_eff)
    for c in ["specific", "vague", "copy_paste"]:
        add(f"text={c}", O.g_text_category == c, cfg.text_eff[c])
    for lo, hi, k in [(0, 15, "<15s"), (60, 300, "60-300s"), (300, 600.0001, "300-600s"), (600.0001, 1e9, ">600s")]:
        add(f"top={k}", (top >= lo) & (top < hi), cfg.top_eff[k])
    add("hesitation>90s", O.hesitation_ms > 90000, cfg.hes_eff)
    add("edits_1_4", O.field_edit_count.between(1, 4), cfg.edits_eff)
    for f in "BCD":
        add(f"form={f}", O.form_variant == f, cfg.form_eff[f])
    for c in ["meta", "linkedin", "chatgpt", "organic_direct"]:
        add(f"channel={c}", O.g_channel == c, cfg.channel_eff[c] - cfg.channel_eff["google"])
    add("lead_ads", O.g_is_lead_ads.astype(bool), cfg.lead_ads_eff)
    for sn in ["senior", "mid", "student"]:
        add(f"seniority={sn}", O.g_seniority == sn, cfg.seniority_eff[sn])
    add("viewed_pricing", _flag(O.viewed_pricing), cfg.pricing_eff)
    add("sessions_3plus", O.sessions_before_convert >= 3, cfg.sessions3_eff)
    add("brand_term", O.utm_term.fillna("").str.contains("northstar"), cfg.brand_eff)
    add("business_hours", biz, cfg.bizhours_eff)
    add("ip_mismatch", O.ip_country.notna() & (O.ip_country != a_country), cfg.ip_mismatch_eff)
    for sp in ["none", "£5k-£25k", "£25k-£100k", "£100k+"]:
        add(f"spend={sp}", has_co & (spend == sp), cfg.spend_eff[sp] - cfg.spend_eff["under £5k"])
    add("company_no_spend_ref", has_co, cfg.spend_eff["under £5k"])
    add("crm_hubspot_sf", has_co & crm.isin(["HubSpot", "Salesforce"]), cfg.crm_eff)
    add("hiring", has_co & hiring, cfg.hiring_eff)
    add("reply<1h", r < 1, cfg.reply_fast_eff)
    add("reply>48h", r > 48, cfg.reply_slow_eff)
    if cfg.interactions:
        add("linkedin_x_51plus", (O.g_channel == "linkedin") & O.g_employee_band.isin(["51-200", "201-1000", "1000+"]),
            cfg.linkedin_51plus_eff)
        add("senior_x_formD", (O.g_seniority == "senior") & (O.form_variant == "D"), cfg.senior_guide_eff)
    if cfg.context_only_signal:
        add("persona", O.g_context_persona.notna(), cfg.persona_eff)
    M = pd.DataFrame(cols, index=O.index)
    y = np.log(O.g_p_close_true / (1 - O.g_p_close_true))
    return M, y, planted


def planted_recovery(t: dict, X: pd.DataFrame, cfg: "Config") -> tuple[pd.DataFrame, float]:
    """OLS of logit(p_close_true) on the planted features: recovered vs planted coefficient, and residual sd
    (the planted noise sd is ``cfg.noise_sd``)."""
    M, y, planted = planted_design(t, X, cfg)
    A = np.column_stack([np.ones(len(M)), M.to_numpy()])
    b, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    resid = y.to_numpy() - A @ b
    out = pd.DataFrame({"planted": pd.Series(planted), "recovered": pd.Series(b[1:], index=M.columns)})
    out["abs_diff"] = (out.recovered - out.planted).abs()
    return out, float(resid.std())


# ---------------------------------------------------------------------------------------------
# All measures, and the ground_truth.md section
# ---------------------------------------------------------------------------------------------

def measure_v2(d: str, cfg: "Config | None" = None) -> dict:
    """Every v2 measurement for data directory ``d`` (planted recovery only when ``cfg`` is given)."""
    t = gt_load(d)
    X = merged(t)
    agree, cm = regex_confusion(X)
    humans = X[~X.g_is_bot.astype(bool)]
    out = {"regex_agreement": agree, "regex_confusion": cm,
           "regex_agreement_humans": float((humans.g_text_category == humans.regex_text).mean()),
           "bot_rule": bot_rule(d), "ghosting": ghosting_stats(t, X), "interactions": interaction_stats(X)}
    if "g_consent_declined" in X:
        out.update({"dropout": dropout_stats(t, X), "consent": consent_stats(X), "persona": persona_stats(X),
                    "fast_humans": fast_human_stats(X),
                    "language_mix": X[~X.is_dup].g_language_mix.value_counts().to_dict()})
    if cfg is not None:
        out["planted"], out["planted_resid_sd"] = planted_recovery(t, X, cfg)
    return out


def _md(df: pd.DataFrame, fmt: str = "{}") -> str:
    """Small markdown table with the index as the first column."""
    head = "| | " + " | ".join(str(c) for c in df.columns) + " |"
    rows = [head, "|---" * (len(df.columns) + 1) + "|"]
    for i, r in df.iterrows():
        rows.append(f"| {i} | " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(rows)


def render_v2_section(d: str, cfg: "Config") -> str:
    """The "v2 mechanisms" section of ground_truth.md, measured on the data in ``d``."""
    m = measure_v2(d, cfg)
    br, gh, cs, ps, fh, dr = m["bot_rule"], m["ghosting"], m["consent"], m["persona"], m["fast_humans"], m["dropout"]
    ix = m["interactions"]
    pl = m["planted"]
    para = "on (Claude Haiku paraphrases from scripts/paraphrase_cache.json)" if cfg.text_paraphrase else \
        "OFF for this output (no cached paraphrases were used; see reports/phase5.md)"
    return f"""## v2 mechanisms (plan items 5.2-5.8)

| Item | Mechanism | Planted size | Measured here |
|---|---|---|---|
| 5.2 | Free text: LLM paraphrase {para}; typos on {100 * cfg.text_typo_share:.0f}% of texts ({100 * cfg.typo_char_rate:.1f}% of letters, at least one); {100 * cfg.text_language_mix_share:.0f}% of DE/FR/NL answers mixed with a hand-written phrase bank (half wrapped in a native greeting/closing, half replaced by a native answer of the same meaning class); copy-paste drawn from {3 + 15} marketing snippets with varied openings, 35% mid-text | Words only: the true text category and its effect are unchanged | Regex agrees with the generator category on {100 * m['regex_agreement']:.1f}% of leads ({100 * m['regex_agreement_humans']:.1f}% of genuine humans). Language-mixed originals: {m['language_mix']} |
| 5.3 | Enrichment dropout: {100 * cfg.enrichment_dropout:.0f}% of business-email domains get a legacy domain in companies.csv (row kept, so typed-name matching can find it; the true row is in ground_truth_companies.csv); {100 * cfg.company_name_variation_share:.0f}% of typed company names get suffix/case/punctuation/spacing noise | No effect on closing (the company's firmographic effects still apply) | {dr['business_domains_missing']} of {dr['business_domains']} business domains missing; {dr['domain_miss_leads']} domain-miss leads, {dr['domain_miss_typed_name']} typed a company name, {dr['domain_miss_matchable']} of those normalise to the right company. Leads with no domain join at all: {dr['no_join_leads']} |
| 5.4 | Consent declined: {100 * cfg.consent_missing_share:.0f}% of website sessions record no analytics telemetry (columns: {', '.join(CONSENT_COLUMNS)}); flag `consent_declined` | No effect: closing uses the real behaviour; the status-quo tier sees the blank pages count | {cs['declined']} rows, {100 * cs['share_of_web']:.1f}% of website rows |
| 5.5 | Interactions: LinkedIn lead from a company with 51+ employees; senior title on form D | {cfg.linkedin_51plus_eff:+.1f} and {cfg.senior_guide_eff:+.1f} on close log-odds | Outcome regression on decided originals: {ix.loc['linkedin_x_51plus', 'coef']:+.2f} [{ix.loc['linkedin_x_51plus', 'lo']:+.2f}, {ix.loc['linkedin_x_51plus', 'hi']:+.2f}] (n = {ix.loc['linkedin_x_51plus', 'cell_decided']}) and {ix.loc['senior_x_formD', 'coef']:+.2f} [{ix.loc['senior_x_formD', 'lo']:+.2f}, {ix.loc['senior_x_formD', 'hi']:+.2f}] (n = {ix.loc['senior_x_formD', 'cell_decided']}) |
| 5.6 | Ghosting follows the status-quo tier: never-contacted log-odds = {cfg.ghost_intercept} + {cfg.ghost_tier_eff} by tier, nothing else; stalls {cfg.stall_by_tier} by tier | Sales neglect of low tiers | Never contacted {100 * gh['never_contacted_share']:.1f}% of originals, by tier {gh['never_by_tier']}; correlation with tier {gh['corr_tier']:.3f}, with true p {gh['corr_p']:.3f} |
| 5.7 | Hidden persona (nonprofit, job seeker, competitor, agency pitching) on {100 * cfg.persona_share:.0f}% of genuine-human leads, written as one sentence appended to non-vague answers, no regex keyword; `context_persona` | {cfg.persona_eff:+.1f} on close log-odds | {100 * ps['share_all_leads']:.1f}% of all leads, {ps['by_persona']}; decided win rate {100 * ps['win_rate_persona']:.1f}% vs {100 * ps['win_rate_no_persona']:.1f}%; largest regex-category share gap persona vs not {100 * ps['regex_dist_max_gap']:.1f} pp |
| 5.8 | Fast humans: {100 * cfg.fast_human_share:.0f}% of genuine-human website sessions with telemetry record 5-14.9s on the page; flag `fast_human` | None: they keep the close odds of their real session | {fh['fast_humans']} sessions ({100 * fh['share']:.1f}%). Bot rule (headless or under {br['threshold_s']}s): precision {100 * br['precision']:.1f}%, recall {100 * br['recall']:.1f}% ({br['fp']} false positives, {br['fp_fast_humans']} of them fast humans; {br['fn']} missed bots, {br['fn_consent_declined']} of them consent-declined) |

### Regex text category vs generator category (rows: generator, columns: `emva.features.text_cat`)

{_md(m['regex_confusion'])}

### Persona texts vs other genuine-human texts: regex category shares

{_md(ps['regex_dist'].round(3))}

### Planted-effect recovery

OLS of logit(p_close_true) on every planted feature (original leads with recorded sessions, 0.001 < p < 0.999;
p is rounded to 4 dp in the labels; bots sit near p = 0, so the truncation pulls the under-15s coefficient toward zero): residual sd {m['planted_resid_sd']:.3f} (planted noise sd {cfg.noise_sd}),
largest |recovered - planted| {pl.abs_diff.max():.3f} ({pl.abs_diff.idxmax()}).

{_md(pl.round(3))}
"""


def main(argv: list[str] | None = None) -> None:
    """CLI: print the v2 measurements for ``--data`` (and ``--ref`` for the ghosting and bot-rule comparison)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--ref", default=None, help="a v1 directory to compare ghosting correlations and the bot rule")
    a = ap.parse_args(argv)
    dirs = [("data", a.data)] + ([("ref", a.ref)] if a.ref else [])
    for name, d in dirs:
        m = measure_v2(d)
        print(f"== {name}: {d}")
        print(f"regex agreement {100 * m['regex_agreement']:.2f}% (humans {100 * m['regex_agreement_humans']:.2f}%)")
        print(m["regex_confusion"].to_string())
        print("bot rule", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m["bot_rule"].items()})
        print("ghosting", m["ghosting"])
        print(m["interactions"].round(3).to_string())
        for k in ["dropout", "consent", "fast_humans", "language_mix"]:
            if k in m:
                print(k, m[k])
        if "persona" in m:
            ps = m["persona"]
            print("persona", {k: v for k, v in ps.items() if not isinstance(v, pd.DataFrame)})
            print(ps["regex_dist"].round(3).to_string())
            print(ps["true_dist"].round(3).to_string())


if __name__ == "__main__":
    main()
