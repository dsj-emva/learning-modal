"""EMVA POC: context agent. Reads each lead the way a salesperson would, against a business brief,
and returns a 0-1 context score. Output feeds emva_score.py --context.

Only submission-time information is shown to the agent. It never sees CRM stages, comments,
deal values or ground truth.

Usage:
  export ANTHROPIC_API_KEY=...
  python context_agent.py --data /mnt/project --brief business_brief.md --out context_scores.csv [--limit 500]
"""
import argparse, hashlib, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests

MODEL = "claude-haiku-4-5-20251001"
SYSTEM = """You are an experienced B2B salesperson qualifying inbound leads for the business described below.
Judge how likely this lead is to become a paying customer, using judgement and context, not a checklist.
Reply with JSON only, no other text:
{"context_score": <number 0-1>, "fit": "<one short sentence>", "red_flags": ["..."]}

BUSINESS BRIEF:
"""


def lead_card(r):
    a = json.loads(r.answers)
    lines = [f"What they want to solve: {a.get('what_to_solve') or '(blank)'}",
             f"Job title: {a.get('job_title') or '(not asked)'}",
             f"Company typed: {a.get('company') or r.get('company_name') or '(none)'}",
             f"Email domain: {str(r.email).split('@')[-1]}",
             f"Country: {a.get('country')}"]
    for k in ["team_size", "budget", "timeline", "company_size"]:
        if a.get(k) not in (None, ""):
            lines.append(f"{k.replace('_', ' ').title()}: {a[k]}")
    if isinstance(r.get("co_sector"), str):
        lines.append(f"Company profile: {r.co_sector}, {r.co_employee_band} employees, ad spend {r.co_monthly_ad_spend_band}/month, "
                     f"CRM {r.co_crm_platform}, hiring {'yes' if r.co_is_hiring else 'no'}")
    return "\n".join(lines)


def call(system, card, key, cache):
    h = hashlib.sha256((system + card).encode()).hexdigest()
    if h in cache:
        return cache[h]
    for attempt in range(5):
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
                "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": MODEL, "max_tokens": 200, "temperature": 0, "system": system,
                      "messages": [{"role": "user", "content": card}]})
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]
            out = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            cache[h] = out
            return out
        except Exception:
            time.sleep(2 ** attempt)
    return {"context_score": None, "fit": "error", "red_flags": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/project")
    ap.add_argument("--brief", default="business_brief.md")
    ap.add_argument("--out", default="context_scores.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    key = os.environ["ANTHROPIC_API_KEY"]
    system = SYSTEM + open(a.brief).read()

    L = pd.read_csv(f"{a.data}/historical_leads.csv")
    CO = pd.read_csv(f"{a.data}/companies.csv")
    CO["dom"] = CO.domain.str.lower(); L["dom"] = L.company_domain.str.lower()
    L = L.merge(CO[["dom", "sector", "employee_band", "monthly_ad_spend_band", "crm_platform", "is_hiring"]]
                .add_prefix("co_").rename(columns={"co_dom": "dom"}), on="dom", how="left")
    if a.limit:
        L = L.sample(a.limit, random_state=0)
    cards = [lead_card(r) for _, r in L.iterrows()]

    cache_file = a.out + ".cache.json"
    cache = json.load(open(cache_file)) if os.path.exists(cache_file) else {}
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda c: call(system, c, key, cache), cards))
    json.dump(cache, open(cache_file, "w"))

    pd.DataFrame({"lead_id": L.lead_id.values,
                  "context_score": [r.get("context_score") for r in res],
                  "fit": [r.get("fit") for r in res],
                  "red_flags": ["; ".join(r.get("red_flags") or []) for r in res]}).to_csv(a.out, index=False)
    print(f"wrote {len(res)} scores to {a.out}")


if __name__ == "__main__":
    main()
