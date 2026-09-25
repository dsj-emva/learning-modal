import json

import pandas as pd
import pytest

from emva.io import (
    clean,
    company_name_index,
    flag_bots_and_duplicates,
    load,
    match_company_names,
    normalise_company_name,
)


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


def _companies(**cols) -> pd.DataFrame:
    """A companies.csv frame; every enrichment column defaults to one value per row."""
    n = len(cols["domain"])
    base = {"company_name": [f"Company {i}" for i in range(n)], "sector": ["Software"] * n,
            "employee_band": ["51-200"] * n, "monthly_ad_spend_band": ["£5k-£25k"] * n,
            "crm_platform": ["HubSpot"] * n, "is_hiring": [True] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_load_normalises_stages_and_joins_enrichment(tmp_path):
    answers = json.dumps({"country": "UK", "what_to_solve": "hi", "job_title": "CEO"})
    pd.DataFrame({"lead_id": ["L1", "L2"], "created_at": ["2026-01-01T00:00:00Z"] * 2, "answers": [answers] * 2,
                  "company_domain": [" Acme.Example ", "unknown.example"]}
                 ).to_csv(tmp_path / "historical_leads.csv", index=False)
    pd.DataFrame({"lead_id": ["L1", "L1", "L2"], "stage": ["New", "CLOSED WON", "demo booked"],
                  "deal_value": [None, 5000, None],
                  "changed_at": ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z", "2026-01-05T00:00:00Z"]}
                 ).to_csv(tmp_path / "crm_history.csv", index=False)
    _companies(company_name=["Acme Ltd"], domain=["acme.example"]).to_csv(tmp_path / "companies.csv", index=False)
    L = load(tmp_path)
    assert L.final_stage.tolist() == ["Won", "Demo booked"]
    assert L.loc["L1", "deal_value"] == 5000 and pd.isna(L.loc["L2", "deal_value"])
    assert L.loc["L1", "a_job_title"] == "CEO" and L.loc["L1", "a_budget"] is None
    assert L.loc["L1", "co_employee_band"] == "51-200" and pd.isna(L.loc["L2", "co_sector"])
    assert L.loc["L1", "won_at"] == pd.Timestamp("2026-02-01T00:00:00Z") and pd.isna(L.loc["L2", "won_at"])
    assert L.loc["L1", "first_contact_at"] == pd.Timestamp("2026-02-01T00:00:00Z")
    assert L.loc["L2", "first_contact_at"] == pd.Timestamp("2026-01-05T00:00:00Z")
    assert L.enrichment_source.tolist()[0] == "domain" and pd.isna(L.enrichment_source["L2"])


def test_load_rejects_two_won_rows(tmp_path):
    pd.DataFrame({"lead_id": ["L1"], "created_at": ["2026-01-01T00:00:00Z"], "answers": ["{}"],
                  "company_domain": ["a.example"]}).to_csv(tmp_path / "historical_leads.csv", index=False)
    pd.DataFrame({"lead_id": ["L1", "L1"], "stage": ["Won", "closed won"], "deal_value": [1, 2],
                  "changed_at": ["2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"]}
                 ).to_csv(tmp_path / "crm_history.csv", index=False)
    _companies(domain=["a.example"]).to_csv(tmp_path / "companies.csv", index=False)
    with pytest.raises(ValueError, match="more than one Won row"):
        load(tmp_path)


@pytest.mark.parametrize("name,expected", [
    ("Acme Ltd.", "acme"), ("ACME Limited", "acme"), ("acme", "acme"), ("  Acme   LTD  ", "acme"),
    ("Foxglove Medical B.V.", "foxglove medical"), ("Kelvin Goods SAS", "kelvin goods"),
    ("Larkspur Haulage GmbH", "larkspur haulage"), ("Smith & Co", "smith"), ("Acme Holdings Co Ltd", "acme holdings"),
    ("Nimbus Market, Inc.", "nimbus market"), ("O'Brien Partners PLC", "obrien partners"),
    ("Co-operative Bank", "co operative bank"),  # only trailing suffixes are stripped
    ("Ltd", None), ("", None), ("   ", None), (None, None), (float("nan"), None),
])
def test_normalise_company_name(name, expected):
    assert normalise_company_name(name) == expected


def test_match_company_names_exact_only_and_skips_ambiguous_names():
    CO = pd.DataFrame({"company_name": ["Acme Ltd", "Beta GmbH", "Gamma Inc", "Gamma LLC"]})
    assert company_name_index(CO).to_dict() == {"acme": 0, "beta": 1}  # "gamma" is ambiguous
    typed = pd.Series(["ACME Limited", "Beta", "Gamma", "Acme Ltda", None, "Freelance"], index=list("abcdef"))
    answer = pd.Series([None, None, None, None, "beta gmbh", "Acme"], index=list("abcdef"))
    got = match_company_names([typed, answer], CO)
    # "Acme Ltda" is not an exact normalised match (no fuzzy matching); the answer is tried when the column misses
    assert got["a"] == 0 and got["b"] == 1
    assert pd.isna(got["c"]) and pd.isna(got["d"]) and got["e"] == 1 and got["f"] == 0


def test_load_adds_name_enrichment_only_where_the_domain_misses(tmp_path):
    answers = [json.dumps({"company": c}) for c in ["", "acme limited", "", ""]]
    pd.DataFrame({"lead_id": ["L1", "L2", "L3", "L4"], "created_at": ["2026-01-01T00:00:00Z"] * 4, "answers": answers,
                  "company_name": ["Beta GmbH", None, "Acme Ltd.", "Unknown Ltd"],
                  "company_domain": ["acme.example", None, "gmail.example", None]}
                 ).to_csv(tmp_path / "historical_leads.csv", index=False)
    pd.DataFrame({"lead_id": ["L1", "L2", "L3", "L4"], "stage": ["New"] * 4, "deal_value": [None] * 4,
                  "changed_at": ["2026-01-01T00:00:00Z"] * 4}).to_csv(tmp_path / "crm_history.csv", index=False)
    _companies(company_name=["Acme Ltd", "Beta GmbH"], domain=["acme.example", "beta.example"],
               employee_band=["51-200", "1000+"]).to_csv(tmp_path / "companies.csv", index=False)
    L = load(tmp_path)
    # L1: the domain wins even though the typed name is another company
    assert L.enrichment_source.fillna("none").tolist() == ["domain", "name", "name", "none"]
    assert L.en_employee_band.tolist()[:3] == ["51-200", "51-200", "51-200"] and pd.isna(L.en_employee_band["L4"])
    # the legacy (baseline) columns stay domain-only
    assert L.co_employee_band.notna().tolist() == [True, False, False, False]


def test_load_requires_company_names(tmp_path):
    pd.DataFrame({"lead_id": ["L1"], "created_at": ["2026-01-01T00:00:00Z"], "answers": ["{}"],
                  "company_domain": ["a.example"]}).to_csv(tmp_path / "historical_leads.csv", index=False)
    pd.DataFrame({"lead_id": ["L1"], "stage": ["New"], "deal_value": [None], "changed_at": ["2026-01-01T00:00:00Z"]}
                 ).to_csv(tmp_path / "crm_history.csv", index=False)
    _companies(domain=["a.example"]).drop(columns="company_name").to_csv(tmp_path / "companies.csv", index=False)
    with pytest.raises(ValueError, match="company_name"):
        load(tmp_path)
