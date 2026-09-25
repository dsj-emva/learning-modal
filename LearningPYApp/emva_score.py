"""EMVA POC: scoring formula (logistic regression) + deal value + optional context-layer ablation.

Usage:
  python emva_score.py --data /mnt/project            # train formula, write scores + weights
  python emva_score.py --data /mnt/project --context context_scores.csv   # + measure context layer uplift
"""
import argparse, json, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss

AS_OF = pd.Timestamp("2026-09-24", tz="UTC")
TEST_FROM = "2026-05-01"
FREE = {"gmail.example", "yahoo.example", "hotmail.example", "outlook.example", "icloud.example"}
STAGE = {"closed won": "Won", "won": "Won", "closed lost": "Lost", "lost": "Lost", "qualified": "Qualified",
         "demo booked": "Demo booked", "proposal": "Proposal", "new": "New", "contacted": "Contacted"}
VAGUE = {"", "?", "n/a", "hi", "test", "info", "interested", "tell me more", "more info pls", "pricing"}
# feature -> reference level (weights are relative to this level)
CATS = {"channel": "google", "form_variant": "A", "band": "1-10", "email": "business", "text": "neutral",
        "seniority": "junior/ic", "spend": "under £5k", "crm": "other", "hiring": "not_hiring",
        "time_on_page": "15-60s", "hesitation_90s": "no", "sessions_3plus": "no", "viewed_pricing": "no",
        "search_term": "generic", "business_hours": "outside", "ip_country": "match", "ip_type": "residential",
        "c_budget": "not_asked", "c_timeline": "not_asked", "edits_1_4": "no", "no_company": "no"}


def text_cat(s):
    s = (s or "").strip().lower()
    if s.startswith(("lorem ipsum", "we are a leading provider", "our mission is to empower")):
        return "copy_paste"
    if s in VAGUE:
        return "vague"
    if re.search(r"£\s?\d|\d+\s*(seats|users|sdrs|marketers)|team of \d|budget|within \d+ weeks|by (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|before (q\d|end|the new)|contract ends", s):
        return "specific"
    return "neutral"


def seniority(t):
    if not isinstance(t, str) or not t.strip():
        return "not_asked/blank"
    t = t.lower()
    if "student" in t or "intern" in t:
        return "student"
    if re.search(r"ceo|founder|chief|vp|director|head|owner|partner|president|cmo|cto|cfo", t):
        return "senior"
    if re.search(r"manager|lead", t):
        return "mid"
    return "junior/ic"


def load(data):
    L = pd.read_csv(f"{data}/historical_leads.csv", parse_dates=["created_at"]).set_index("lead_id")
    C = pd.read_csv(f"{data}/crm_history.csv", parse_dates=["changed_at"])
    CO = pd.read_csv(f"{data}/companies.csv")
    C["stage_n"] = C.stage.str.lower().map(STAGE)
    C = C.sort_values(["lead_id", "changed_at"])
    last = C.groupby("lead_id").tail(1).set_index("lead_id")
    L["final_stage"] = last.stage_n
    L["last_change"] = last.changed_at
    L["deal_value"] = C[C.stage_n == "Won"].set_index("lead_id").deal_value
    ans = L.answers.apply(json.loads)
    for k in ["country", "what_to_solve", "job_title", "company_size", "budget", "timeline"]:
        L["a_" + k] = ans.apply(lambda d: d.get(k))
    CO["dom"] = CO.domain.str.lower()
    L["dom"] = L.company_domain.str.lower().str.strip()
    L = L.join(CO.set_index("dom")[["sector", "employee_band", "monthly_ad_spend_band", "crm_platform", "is_hiring"]].add_prefix("co_"), on="dom")
    return L


def build(L):
    X = L.copy()
    # cleaning: bots and repeat submissions
    X["email_l"] = X.email.str.strip().str.lower()
    X["bot"] = X.user_agent.fillna("").str.contains("Headless", case=False) | (X.time_on_page_s < 15)
    X["dup"] = X.groupby("email_l").created_at.rank(method="first") > 1
    X = X[~X.bot & ~X.dup].copy()
    # labels (decision log)
    age = (AS_OF - X.created_at).dt.days
    idle = (AS_OF - X.last_change).dt.days
    open_ = X.final_stage.isin(["Contacted", "Qualified", "Demo booked", "Proposal"])
    X["y"] = np.nan
    X.loc[X.final_stage == "Won", "y"] = 1
    X.loc[X.final_stage == "Lost", "y"] = 0
    X.loc[open_ & (idle >= 90), "y"] = 0                      # stalled 90+ days = Lost
    X.loc[(X.final_stage == "New") & (age >= 90), "y"] = 0     # ghosted 90+ days = Lost
    # features
    med = X.utm_medium; src = X.utm_source
    X["channel"] = np.select([med == "lead_form", src.isin(["facebook", "instagram"]), src == "google", src == "linkedin", src == "chatgpt"],
                             ["meta_leadads", "meta", "google", "linkedin", "chatgpt"], "organic_direct")
    lead_ads = X.channel == "meta_leadads"
    has_co = X.co_employee_band.notna()
    X["no_company"] = np.where(has_co, "no", "yes")
    X["band"] = X.co_employee_band.fillna(X.a_company_size.replace("", np.nan)).fillna("1-10")
    X["email"] = np.where(X.email_l.str.split("@").str[1].isin(FREE), "free", "business")
    X["text"] = X.a_what_to_solve.apply(text_cat)
    X["seniority"] = X.a_job_title.apply(seniority)
    X["spend"] = X.co_monthly_ad_spend_band.fillna("under £5k")
    X["crm"] = np.where(X.co_crm_platform.isin(["HubSpot", "Salesforce"]), "hubspot_sf", "other")
    X["hiring"] = np.where(X.co_is_hiring == True, "hiring", "not_hiring")
    top = X.time_on_page_s
    X["time_on_page"] = np.select([top < 60, top < 300, top < 600, top >= 600], ["15-60s", "60-300s", "300-600s", ">600s"], "15-60s")
    X["hesitation_90s"] = np.where(X.hesitation_ms > 90000, "yes", "no")
    X["sessions_3plus"] = np.where(X.sessions_before_convert >= 3, "yes", "no")
    X["viewed_pricing"] = np.where(X.viewed_pricing == True, "yes", "no")
    X["search_term"] = np.where(X.channel != "google", "not_google",
                                np.where(X.utm_term.fillna("").str.lower().str.contains("northstar"), "brand", "generic"))
    wk = ~X.submitted_weekday.isin(["Saturday", "Sunday"]) & X.local_submit_hour.between(9, 17)
    X["business_hours"] = np.where(wk & ~lead_ads, "wkday_9-18", "outside")
    X["ip_country"] = np.where(X.ip_country.notna() & (X.ip_country != X.a_country), "mismatch", "match")
    X["ip_type"] = np.where(X.is_datacenter_ip == True, "dc", "residential")
    X["c_budget"] = X.a_budget.fillna("not_asked")
    X["c_timeline"] = X.a_timeline.fillna("not_asked")
    X["edits_1_4"] = np.where(X.field_edit_count.between(1, 4), "yes", "no")
    X["sector"] = X.co_sector.fillna("none")
    return X


def design(X, extra=None):
    parts = []
    for c, ref in CATS.items():
        d = pd.get_dummies(X[c].astype(str), prefix=c, prefix_sep="=")
        parts.append(d.drop(columns=[f"{c}={ref}"], errors="ignore"))
    D = pd.concat(parts, axis=1).astype(float)
    if extra is not None:
        D = D.join(extra)
    return D


def fit_lr(D, y, tr):
    return LogisticRegression(C=0.5, max_iter=5000).fit(D[tr], y[tr])


def report(name, y, p, rev, value=None):
    n = int(len(y) * 0.2); o = np.argsort(-p)[:n]
    ov = o if value is None else np.argsort(-value)[:n]
    return {"model": name, "auc": round(roc_auc_score(y, p), 3), "brier": round(brier_score_loss(y, np.clip(p, 0, 1)), 4),
            "top20_wins": round(y.iloc[o].sum() / y.sum(), 3), "top20_revenue": round(rev.iloc[ov].sum() / rev.sum(), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/project")
    ap.add_argument("--context", default=None, help="CSV with lead_id,context_score (0-1) from context_agent.py")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    X = build(load(a.data))
    lab = X.y.notna()
    tr = lab & (X.created_at < TEST_FROM); te = lab & (X.created_at >= TEST_FROM)
    D = design(X)
    lr = fit_lr(D, X.y, tr)
    X["p_formula"] = lr.predict_proba(D)[:, 1]

    # expected deal value (fit on train wins)
    M = pd.concat([pd.get_dummies(X[c].astype(str), prefix=c) for c in ["band", "sector", "channel", "spend"]], axis=1).astype(float)
    m = tr & X.deal_value.notna()
    rg = Ridge(alpha=3).fit(M[m], np.log(X.deal_value[m]))
    X["deal_value_hat"] = np.exp(rg.predict(M)) * np.exp(0.45 ** 2 / 2)
    X["value_formula"] = X.p_formula * X.deal_value_hat * a.margin

    # weights as scorecard points: +20 points = odds of closing double
    w = pd.Series(lr.coef_[0], index=D.columns)
    pd.DataFrame({"log_odds": w.round(3), "odds_multiplier": np.exp(w).round(2), "points": (w * 20 / np.log(2)).round(0)}) \
        .sort_values("log_odds").to_csv(f"{a.out}/weights.csv")

    T = X[te]; rev = pd.Series(np.where(T.y == 1, T.deal_value.fillna(T.deal_value_hat), 0), index=T.index)
    rows = [report("formula", T.y, T.p_formula.values, rev, T.value_formula.values)]

    if a.context:
        cs = pd.read_csv(a.context).set_index("lead_id")["context_score"].clip(0.001, 0.999)
        X["context_logit"] = np.log(cs / (1 - cs)).reindex(X.index)
        have = X.context_logit.notna()
        print(f"context scores for {have.sum()} of {len(X)} leads")
        # same rows for both models so the comparison is fair
        tr2, te2 = tr & have, te & have
        base = fit_lr(D, X.y, tr2)
        both = fit_lr(D.join(X.context_logit), X.y, tr2)
        X["p_base"] = base.predict_proba(D)[:, 1]
        X.loc[have, "p_combined"] = both.predict_proba(D.join(X.context_logit)[have])[:, 1]
        ctx_only = LogisticRegression().fit(X.loc[tr2, ["context_logit"]], X.y[tr2])
        X.loc[have, "p_context_only"] = ctx_only.predict_proba(X.loc[have, ["context_logit"]])[:, 1]
        T2 = X[te2]; rev2 = pd.Series(np.where(T2.y == 1, T2.deal_value.fillna(T2.deal_value_hat), 0), index=T2.index)
        rows = [report(n, T2.y, T2[c].values, rev2, (T2[c] * T2.deal_value_hat).values) for n, c in
                [("formula", "p_base"), ("context_only", "p_context_only"), ("formula+context", "p_combined")]]
        print(f"context weight in combined model: {both.coef_[0][-1]:.3f} (0 = context adds nothing)")
        X["value_combined"] = X.p_combined * X.deal_value_hat * a.margin

    print(pd.DataFrame(rows).to_string(index=False))
    keep = [c for c in ["p_formula", "deal_value_hat", "value_formula", "p_combined", "value_combined", "y"] if c in X]
    X[keep].to_csv(f"{a.out}/scores.csv")


if __name__ == "__main__":
    main()
