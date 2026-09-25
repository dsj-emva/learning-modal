"""Measures the ground_truth.md tables on a generated data directory and renders the markdown.

Used by generate_data_v1.py (to write ground_truth.md from its own output) and by
compare_to_v1.py (to put the same statistics for data/v1 and a regenerated directory side by side).
Reads ground_truth_labels.csv, so it belongs with the generator and evaluation code only.
"""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from generate_data_v1 import Config

AS_OF_DEFAULT = "2026-09-24"
STAGE = {"closed won": "Won", "won": "Won", "closed lost": "Lost", "lost": "Lost", "qualified": "Qualified",
         "demo booked": "Demo booked", "proposal": "Proposal", "new": "New", "contacted": "Contacted"}
CANON = {"New", "Contacted", "Qualified", "Demo booked", "Proposal", "Won", "Lost"}


def load(d: str) -> dict:
    """Read the five CSVs and status_quo_rules.json of a data directory.

    ``gt_companies`` is the full firmographics table: ground_truth_companies.csv when the directory has one
    (v2 with enrichment dropout, where companies.csv misses some domains), else companies.csv."""
    t = {n: pd.read_csv(os.path.join(d, f"{n}.csv")) for n in
         ["historical_leads", "crm_history", "companies", "people", "ground_truth_labels"]}
    gt_co = os.path.join(d, "ground_truth_companies.csv")
    t["gt_companies"] = pd.read_csv(gt_co) if os.path.exists(gt_co) else t["companies"]
    with open(os.path.join(d, "status_quo_rules.json")) as f:
        t["rules"] = json.load(f)
    return t


def tier_of(answers: dict, pages: float | None, rules: dict) -> str:
    """Status-quo tier (A/B/C) of one lead under ``rules``."""
    ls = rules["lead_score"]
    pts = 0
    jt = (answers.get("job_title") or "").lower()
    if jt:
        hit = next((k["points"] for k in ls["job_title"]["keywords"] if k["keyword"] in jt), None)
        pts += ls["job_title"]["default_points"] if hit is None else hit
    pts += ls["company_size_points"].get(answers.get("company_size") or "", 0)
    if pages == pages and pages is not None:
        pts += next((r["points"] for r in ls["pages_visited_points"] if pages >= r["min_pages"]), 0)
    return next(t["name"] for t in ls["tiers"] if pts >= t["min_points"])


def frame(t: dict) -> pd.DataFrame:
    """One row per lead with the hidden labels and the observed columns the tables need."""
    L, G, CO = t["historical_leads"], t["ground_truth_labels"], t["gt_companies"].set_index("domain")
    X = L.merge(G.add_prefix("g_"), left_on="lead_id", right_on="g_lead_id")
    X["ans"] = X.answers.apply(json.loads)
    X["a_country"] = X.ans.apply(lambda d: d.get("country"))
    for c in ["monthly_ad_spend_band", "crm_platform", "is_hiring"]:
        X["co_" + c] = X.g_company_domain.map(CO[c])
    X["decided"] = X.g_outcome.isin(["won", "lost"])
    X["won"] = X.g_outcome == "won"
    X["created"] = pd.to_datetime(X.created_at, utc=True)
    X["tier"] = [tier_of(a, p, t["rules"]) for a, p in zip(X.ans, X.pages_visited)]
    P = t["people"].set_index("email")
    X["email_l"] = X.email.str.lower()
    X["p_age"] = X.email_l.map(P.age_band)
    X["p_income"] = X.email_l.map(P.household_income_band)
    return X


def table(X: pd.DataFrame, levels: pd.Series, order: list) -> pd.DataFrame:
    """Decided-lead count and win rate per level, in ``order``."""
    D = X[X.decided]
    lv = levels[X.decided]
    rows = []
    for o in order:
        m = lv == o
        n = int(m.sum())
        rows.append({"level": o, "decided": n, "win_rate": (D.won[m].mean() if n else np.nan)})
    return pd.DataFrame(rows)


CONSENT_LEVEL = "blank (consent declined)"


def consent_table(X: pd.DataFrame, levels: pd.Series, order: list) -> pd.DataFrame:
    """``table`` for a telemetry column; in v2 data, consent-declined rows get their own level instead of
    falling into the Lead Ads / "under" level (v1 data has no ``consent_declined`` label, so is unchanged)."""
    if "g_consent_declined" not in X:
        return table(X, levels, order)
    cd = X.g_consent_declined.astype(str).eq("True")
    return table(X, levels.where(~cd, CONSENT_LEVEL), order + [CONSENT_LEVEL])


def top_bucket(s: pd.Series) -> pd.Series:
    """Time-on-page bucket labels as in ground_truth.md ("missing" for Lead Ads)."""
    return pd.Series(np.select([s < 15, s < 60, s < 300, s <= 600, s > 600],
                               ["<15s (bot-like)", "15-60s", "60-300s", "300-600s", ">600s"], "missing"), index=s.index)


def measure(t: dict, as_of: str = AS_OF_DEFAULT) -> dict:
    """Every ground_truth.md table (``tables``) and scalar statistic (``scalars``) for loaded tables ``t``."""
    X = frame(t)
    C = t["crm_history"].copy()
    tabs = {}
    tabs["band"] = table(X, X.g_employee_band, ["none", "1-10", "11-50", "51-200", "201-1000", "1000+"])
    tabs["email"] = table(X, pd.Series(np.where(X.g_is_free_email, "free email", "business email"), index=X.index),
                          ["business email", "free email"])
    tabs["text"] = table(X, X.g_text_category, ["specific", "neutral", "vague", "copy_paste"])
    tabs["time_on_page"] = consent_table(X, top_bucket(X.time_on_page_s), ["<15s (bot-like)", "15-60s", "60-300s", "300-600s", ">600s"])
    tabs["hesitation"] = consent_table(X, pd.Series(np.where(X.hesitation_ms > 90000, "over 90s", "under 90s"), index=X.index),
                               ["over 90s", "under 90s"])
    tabs["form_variant"] = table(X, X.form_variant, ["A", "B", "C", "D"])
    tabs["channel"] = table(X, X.g_channel, ["meta", "google", "linkedin", "chatgpt", "organic_direct"])
    tabs["seniority"] = table(X, X.g_seniority, ["senior", "mid", "junior", "student"])
    vp = pd.Series(np.where(X.viewed_pricing.isna(), "no page (Lead Ads)",
                            np.where(X.viewed_pricing.astype(str) == "True", "yes", "no")), index=X.index)
    tabs["viewed_pricing"] = consent_table(X, vp, ["yes", "no", "no page (Lead Ads)"])
    ss = pd.Series(np.select([X.sessions_before_convert == 1, X.sessions_before_convert == 2,
                              X.sessions_before_convert >= 3], ["1", "2", "3+"], "missing"), index=X.index)
    tabs["sessions"] = consent_table(X, ss, ["1", "2", "3+"])
    st = pd.Series(np.where(X.utm_source != "google", "not Google",
                            np.where(X.utm_term.fillna("").str.lower().str.contains("northstar"), "brand", "generic")),
                   index=X.index)
    tabs["search_term"] = table(X, st, ["brand", "generic", "not Google"])
    wk = ~X.submitted_weekday.isin(["Saturday", "Sunday"]) & X.local_submit_hour.between(9, 17)
    tabs["submitted"] = table(X, pd.Series(np.where(wk, "weekday 09-18", "outside"), index=X.index), ["weekday 09-18", "outside"])
    ipc = pd.Series(np.where(X.ip_country.isna(), "no IP (Lead Ads)",
                             np.where(X.ip_country != X.a_country, "mismatch", "match")), index=X.index)
    tabs["ip_country"] = consent_table(X, ipc, ["match", "mismatch", "no IP (Lead Ads)"])
    tabs["spend"] = table(X, X.co_monthly_ad_spend_band.fillna("no company"),
                          ["none", "under £5k", "£5k-£25k", "£25k-£100k", "£100k+", "no company"])
    tabs["crm"] = table(X, X.co_crm_platform.fillna("no company"),
                        ["HubSpot", "Salesforce", "Pipedrive", "Spreadsheet", "No CRM", "no company"])
    hi = pd.Series(np.where(X.co_is_hiring.isna(), "no company",
                            np.where(X.co_is_hiring.astype(str) == "True", "hiring", "not hiring")), index=X.index)
    tabs["hiring"] = table(X, hi, ["hiring", "not hiring", "no company"])
    r = X.g_reply_hours
    rb = pd.Series(np.select([r < 1, r <= 24, r <= 48, r > 48], ["under 1h", "1-24h", "24-48h", "over 48h"], "none"), index=X.index)
    tabs["first_reply"] = table(X, rb, ["under 1h", "1-24h", "24-48h", "over 48h"])
    tabs["age_band"] = table(X, X.p_age.fillna("no vendor match"),
                             ["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "no vendor match"])
    tabs["income"] = table(X, X.p_income.fillna("no vendor match"),
                           ["under £25k", "£25k-£50k", "£50k-£100k", "£100k-£150k", "£150k+", "no vendor match"])
    tabs["country"] = table(X, X.g_country, ["UK", "US", "DE", "FR", "NL"])
    pv = X.pages_visited
    pvb = pd.Series(np.where(pv.isna(), "none (Lead Ads)", np.where(pv >= 5, "5+", pv.fillna(0).astype(int).astype(str))),
                    index=X.index)
    tabs["pages_visited"] = consent_table(X, pvb, ["none (Lead Ads)", "1", "2", "3", "4", "5+"])
    tabs["status_quo_tier"] = table(X, X.tier, ["A", "B", "C"])

    s = {}
    s["n_leads"] = len(X)
    s["variant_counts"] = X.form_variant.value_counts().reindex(["A", "B", "C", "D"]).fillna(0).astype(int).to_dict()
    s["overall_won_share"] = X.won.mean()
    s["decided"] = int(X.decided.sum())
    s["decided_win_rate"] = X.won[X.decided].mean()
    for o in ["won", "lost", "open", "never_contacted", "duplicate"]:
        s["outcome_" + o] = int((X.g_outcome == o).sum())
    s["bots"] = int(X.g_is_bot.sum())
    s["stalled"] = int(X.g_stalled.sum())
    wv = X.g_deal_value[X.won]
    s["won_deal_median"], s["won_deal_mean"], s["won_deal_max"] = wv.median(), wv.mean(), wv.max()
    s["won_deal_median_by_channel"] = X[X.won].groupby("g_channel").g_deal_value.median().round(0).to_dict()
    # time to close from CRM
    C["sn"] = C.stage.str.lower().map(STAGE)
    C["t"] = pd.to_datetime(C.changed_at, utc=True)
    W = C[C.sn == "Won"].merge(X[["lead_id", "created", "g_employee_band"]], on="lead_id")
    days = (W.t - W.created).dt.total_seconds() / 86400
    s["won_days_median"] = days.median()
    s["won_within_14"] = (days <= 14).mean()
    s["won_past_90"] = (days > 90).mean()
    s["won_days_median_by_band"] = days.groupby(W.g_employee_band).median().round(0).to_dict()
    Lo = C[C.sn == "Lost"].merge(X[["lead_id", "created"]], on="lead_id")
    s["lost_days_median"] = ((Lo.t - Lo.created).dt.total_seconds() / 86400).median()
    # data problems
    web_paid = X.utm_source.notna() & (X.utm_medium != "lead_form")
    s["paid_web"] = int(web_paid.sum())
    s["click_id_missing"] = int((web_paid & X[["fbclid", "gclid", "gbraid", "wbraid", "li_fat_id", "oppref"]].isna().all(axis=1)).sum())
    s["duplicates"] = int(X.g_duplicate_of.notna().sum())
    s["dup_case_changed"] = int((X.g_duplicate_of.notna() & (X.email != X.email_l)).sum())
    s["messy_stage_rows"] = int((~C.stage.isin(CANON)).sum())
    s["open_leads"] = s["outcome_open"]
    s["won_no_value"] = int((C.sn == "Won").sum() - C[C.sn == "Won"].deal_value.notna().sum())
    s["people_rows"] = len(t["people"])
    s["no_person_match"] = int(X.p_age.isna().sum())
    s["ip_mismatch"] = int((X.ip_country.notna() & (X.ip_country != X.a_country)).sum())
    s["datacenter"] = int((X.is_datacenter_ip.astype(str) == "True").sum())
    s["never_contacted"] = s["outcome_never_contacted"]
    s["crm_rows"] = len(C)
    # reply time medians by tier
    s["reply_median_by_tier"] = X.groupby("tier").g_reply_hours.median().round(1).to_dict()
    nd = X[X.g_duplicate_of.isna()]
    bus, free = nd[~nd.g_is_free_email & ~nd.g_is_bot], nd[nd.g_is_free_email & ~nd.g_is_bot]
    s["vendor_cover_business"] = bus.p_age.notna().mean()
    s["vendor_cover_free"] = free.p_age.notna().mean()
    return {"tables": tabs, "scalars": s}


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------

def md_table(tab: pd.DataFrame, head: str) -> str:
    """Render one decided-leads / win-rate table as markdown."""
    out = [f"| {head} | decided leads | win rate |", "|---|---|---|"]
    for _, r in tab.iterrows():
        wr = "n/a" if r.win_rate != r.win_rate else f"{100 * r.win_rate:.1f}%"
        out.append(f"| {r.level} | {r.decided} | {wr} |")
    return "\n".join(out)


def render_ground_truth(d: str, cfg: "Config") -> str:
    """ground_truth.md for the data in ``d`` generated with ``cfg``."""
    from generate_data_v1 import v2_on  # sibling module; imported here because it imports this module's users
    m = measure(load(d), cfg.as_of)
    T, s = m["tables"], m["scalars"]
    band_base = {k: v for k, v in cfg.deal_base.items()}
    v2 = v2_on(cfg)
    if v2:
        written_by = (f"scripts/generate_data_v2.py, i.e. scripts/generate_data_v1.py with the Phase 5.2-5.8 options "
                      f"{v2} (seed {cfg.seed}, data as of {cfg.as_of}, n = {cfg.n}). Rerunning the script with the "
                      f"same arguments rewrites this file with identical numbers. Every v1 mechanism below still "
                      f"applies; the section \"v2 mechanisms\" at the end lists what the options add.")
        never_rule = "Stage stays New. v2: picked on the status quo tier only (see v2 mechanisms, 5.6)"
    else:
        written_by = (f"scripts/generate_data_v1.py (seed {cfg.seed}, data as of {cfg.as_of}, n = {cfg.n}). "
                      f"Rerunning the\nscript with the same arguments rewrites this file with identical numbers. "
                      f"This generator is a rewrite of the\nlost original from its ground_truth.md; it reproduces "
                      f"the v1 schema and distributions, not v1's rows.")
        never_rule = "Stage stays New. Picked with extra weight on status quo tier C, free emails and vague text"
    lines = f"""# Ground truth: planted signals and data problems

Written by {written_by}

Every lead has a hidden close log-odds: an intercept ({cfg.intercept}, calibrated so ~12% of all leads
end up Won) plus the effects below, plus random noise (sd {cfg.noise_sd}). Win or lose is then drawn
from that probability. A good model should rediscover these effects from the data alone.

"Win rate" in the tables below means Won / (Won + Lost), among leads with a final outcome. Open,
never-contacted and duplicate leads are left out. Observed rates mix several signals together (for
example Meta has more free emails), so they won't match the planted effect exactly.

Per-lead hidden truth is in `ground_truth_labels.csv`. Use it only for tests and evaluation, never as
model input.

Measured on this output: {s['n_leads']} leads, {s['outcome_won']} Won ({100 * s['overall_won_share']:.1f}% of all
leads), {s['decided']} decided leads with a {100 * s['decided_win_rate']:.1f}% win rate.

## Planted signals

| # | Signal | Planted effect on close log-odds | Strength |
|---|---|---|---|
| 1 | Company size | 1-10: 0, 11-50: +0.5, 51-200 and 201-1000: +1.3, 1000+: +1.0 | Strong. 51-1000 closes about 3x as often as 1-10 |
| 2 | Free email domain | -1.3 | Strong |
| 3 | Free text answer | mentions budget, timeline or team size: +0.75; vague: -0.75; copy-paste: -0.75 | Strong. Specific about 2x, vague about 0.5x |
| 4 | Time on page | under 15s: -1.5; 60-300s: +0.4; over 600s: -0.4 | Strong for bots, moderate otherwise |
| 5 | Hesitation | over 90s of pauses: -0.4 | Moderate |
| 6 | Field edits | 1-4 edits: +0.2 | Weak |
| 7 | Form variant | A: 0, B: +0.2, C (long form): +0.7, D (guide download): -0.3 | Moderate. C also gets more specific answers |
| 8 | Channel | Meta -0.4, Google +0.2, LinkedIn +0.2, ChatGPT 0, organic/direct +0.1. Meta Lead Ads a further -0.2 | Moderate. Meta also has more bots, free emails and vague text |
| 9 | Job title seniority | senior +0.3, mid +0.1, junior 0, student -1.2 | Moderate, strong for students. Applies whether or not the form asked for the title |

### Signal 1: company size (true band from companies.csv; "none" = individual with no company)

{md_table(T['band'], 'employee band')}

### Signal 2: email type

{md_table(T['email'], 'email')}

### Signal 3: free text answer

{md_table(T['text'], 'text category')}

### Signals 4 and 5: behaviour

{md_table(T['time_on_page'], 'time on page')}

{md_table(T['hesitation'], 'hesitation')}

### Signal 7: form variant

{md_table(T['form_variant'], 'form variant')}

Lead counts by variant: {s['variant_counts']}

### Signal 8: channel

{md_table(T['channel'], 'channel')}

### Signal 9: seniority

{md_table(T['seniority'], 'seniority')}

## Deal value

Deal value = band base x sector multiplier x channel multiplier x ad-spend multiplier x lognormal noise
(sd {cfg.deal_noise_sd}), rounded to GBP 50 and clipped to 500-60,000.

- Band base (GBP): {band_base}
- Sector multiplier: {cfg.deal_sector}
- Channel multiplier: {cfg.deal_channel}

Won deals in the data: median GBP {s['won_deal_median']:,.0f}, mean GBP {s['won_deal_mean']:,.0f},
max GBP {s['won_deal_max']:,.0f}.

Median won deal value by channel (GBP): {s['won_deal_median_by_channel']}

## Time to close

Won deals take a lognormal number of days (median {cfg.won_median_days:.0f}, sd {cfg.won_sd}); 1000+
companies take {cfg.won_1000_mult}x longer. Lost deals: median {cfg.lost_median_days:.0f} days.

Observed in crm_history.csv (Won rows only): median {s['won_days_median']:.0f} days,
{100 * s['won_within_14']:.0f}% within 14 days, {100 * s['won_past_90']:.0f}% past 90 days.
Long deals started recently are still open, so the observed tail is shorter than the planted one.

Median days to Won by band: {s['won_days_median_by_band']}

## More planted signals: intent, journey and company

| # | Signal | Planted effect | Strength |
|---|---|---|---|
| 10 | Viewed the pricing page | +0.5 | Moderate |
| 11 | 3 or more sessions before converting | +0.3 | Weak to moderate |
| 12 | Google search term was the brand ("northstar ...") | +0.6 | Moderate |
| 13 | Submitted on a weekday, 09:00-18:00 local time | +0.25 | Weak |
| 14 | IP country differs from the country typed | -0.3 | Weak |
| 15 | Company monthly ad spend (companies.csv) | none -0.7, under £5k -0.2, £5k-£25k +0.2, £25k+ +0.5 | Strong. The buyer must advertise to need EMVA |
| 16 | Company CRM is HubSpot or Salesforce | +0.3 | Weak to moderate |
| 17 | Company is hiring | +0.2 | Weak |
| 18 | Sales first reply (speed to lead) | under 1 hour +0.4, over 48 hours -0.4 | Moderate. Known only after the lead arrives, and confounded: sales reply faster to leads that look good (median {s['reply_median_by_tier'].get('A', float('nan'))}h for tier A, {s['reply_median_by_tier'].get('C', float('nan'))}h for tier C) |

Ad spend also raises deal value: {cfg.deal_spend}.

{md_table(T['viewed_pricing'], 'viewed pricing')}

{md_table(T['sessions'], 'sessions before converting')}

{md_table(T['search_term'], 'search term')}

{md_table(T['submitted'], 'submitted')}

{md_table(T['ip_country'], 'IP country')}

{md_table(T['spend'], 'monthly ad spend')}

{md_table(T['crm'], 'CRM platform')}

{md_table(T['hiring'], 'hiring')}

{md_table(T['first_reply'], 'first reply')}

## Demographics: proxies, not causes

people.csv is a fake person-data vendor: age band, household income band, region, city, seniority,
department, years in role and homeownership. It matches {s['people_rows']:,} people (about
{100 * s['vendor_cover_business']:.0f}% of business emails, {100 * s['vendor_cover_free']:.0f}% of free emails, no bots).

**Age and household income have NO planted effect.** They are generated from seniority (and country),
and seniority does matter. So in the tables below they look predictive, but only because a 45-year-old
is more likely to be a director. A model that controls for seniority and company should give them
roughly zero weight.

{md_table(T['age_band'], 'age band')}

{md_table(T['income'], 'household income')}

Deliberately **not** generated: gender, ethnicity, religion, health, sexual orientation, exact date
of birth, home address.

## Not signals (traps)

These have **no planted effect** on closing. Any rule or model that leans on them is wrong.

- **Country.** The status quo adds 20% for UK and US leads. Close rates don't depend on country.
- **Pages visited.** The status quo lead score gives points for it. It is pure noise here.
- **Company size dropdown** is a noisy copy of the true band (15% of answers are one band off).
- **City, region, browser language, scroll depth, weekday name** (beyond the business-hours signal):
  pure noise.
- **Homeowner, years in role** (people.csv): no effect.
- **Annual revenue band, founded year, funding stage** (companies.csv): no direct effect.

{md_table(T['country'], 'country')}

{md_table(T['pages_visited'], 'pages visited')}

## How the status quo tiers line up with reality

{md_table(T['status_quo_tier'], 'status quo tier')}

Variant A has no job title or company size questions, so every variant A lead lands in tier C,
whatever its real quality.

## Planted data problems (the readiness check must catch every one)

| Problem | How it was planted | Count |
|---|---|---|
| Missing click IDs | {100 * cfg.click_id_missing_share:.0f}% of paid website leads have every click ID blanked (fbp cookie kept) | {s['click_id_missing']} of {s['paid_web']} paid website leads |
| Duplicates | Same person submits again with a new lead_id; 35% have the email in a different case | {s['duplicates']} leads |
| Inconsistent stage names | "Closed Won", "won", "WON", "Closed Lost", "lost", "Demo Booked", "demo booked", "qualified", "QUALIFIED" | {s['messy_stage_rows']} CRM rows |
| Open leads | Still in progress on {cfg.as_of}: deals not closed yet, plus {100 * cfg.stall_share:.0f}% of contacted leads that stalled | {s['open_leads']} leads |
| Won without deal value | deal_value blank on {100 * cfg.won_blank_value_share:.0f}% of Won rows | {s['won_no_value']} rows |
| No person-data match | people.csv covers 75% of business emails, 40% of free emails and no bots | {s['no_person_match']} leads with no match |
| IP country mismatch | IP country differs from the country typed (VPN or travelling); about half of these on datacenter IPs | {s['ip_mismatch']} leads |
| Datacenter IPs | Bots and some VPN users | {s['datacenter']} leads |
| Never contacted | {never_rule} | {s['never_contacted']} leads (plus duplicates left at New) |

Other deliberate mess: phone numbers in mixed formats, some invalid ("12345"); company names typed with
or without the legal suffix, sometimes lowercase; bots share a handful of IP addresses.

**Selection bias warning:** never-contacted leads have no outcome, and they are mostly the ones that
looked weak. Training only on decided leads under-represents weak-looking leads.
"""
    if v2:
        from v2_checks import render_v2_section  # sibling module (evaluation side, reads ground truth)
        lines += "\n" + render_v2_section(d, cfg)
    return lines
