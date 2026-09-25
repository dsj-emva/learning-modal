import json

import pandas as pd

from emva.io import clean, flag_bots_and_duplicates, load


def test_bot_and_duplicate_rules():
    L = pd.DataFrame({
        "email": ["A@x.example ", "a@x.example", "b@x.example", "c@x.example", "d@x.example"],
        "created_at": pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01"]),
        "user_agent": ["Mozilla", "Mozilla", "HeadlessChrome/1.0", None, "Mozilla"],
        "time_on_page_s": [100, 100, 100, 14.9, 15],
    }, index=["L1", "L2", "L3", "L4", "L5"])
    X = flag_bots_and_duplicates(L)
    assert X.bot.tolist() == [False, False, True, True, False]
    # L1 is a later submission of the same normalised email as L2
    assert X.dup.tolist() == [True, False, False, False, False]
    assert clean(L).index.tolist() == ["L2", "L5"]


def test_load_normalises_stages_and_joins_enrichment(tmp_path):
    answers = json.dumps({"country": "UK", "what_to_solve": "hi", "job_title": "CEO"})
    pd.DataFrame({"lead_id": ["L1", "L2"], "created_at": ["2026-01-01T00:00:00Z"] * 2, "answers": [answers] * 2,
                  "company_domain": [" Acme.Example ", "unknown.example"]}).to_csv(tmp_path / "historical_leads.csv", index=False)
    pd.DataFrame({"lead_id": ["L1", "L1", "L2"], "stage": ["New", "CLOSED WON", "demo booked"],
                  "deal_value": [None, 5000, None],
                  "changed_at": ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-01-05T00:00:00Z"]}
                 ).to_csv(tmp_path / "crm_history.csv", index=False)
    pd.DataFrame({"domain": ["acme.example"], "sector": ["Software"], "employee_band": ["51-200"],
                  "monthly_ad_spend_band": ["£5k-£25k"], "crm_platform": ["HubSpot"], "is_hiring": [True]}
                 ).to_csv(tmp_path / "companies.csv", index=False)
    L = load(tmp_path)
    assert L.final_stage.tolist() == ["Won", "Demo booked"]
    assert L.loc["L1", "deal_value"] == 5000 and pd.isna(L.loc["L2", "deal_value"])
    assert L.loc["L1", "a_job_title"] == "CEO" and L.loc["L1", "a_budget"] is None
    assert L.loc["L1", "co_employee_band"] == "51-200" and pd.isna(L.loc["L2", "co_sector"])
