import logging

import numpy as np
import pandas as pd
import pytest

from emva.context import agent
from emva.context.contract import OUTPUT_COLUMNS, error_response
from emva.context.features import context_logit


def test_lead_card_renders_submission_time_fields_only():
    r = pd.Series({"answers": '{"what_to_solve": "attribution", "company": "Acme", "country": "UK", "budget": "£5k"}',
                   "email": "a@acme.example", "company_name": "Acme Ltd", "co_sector": "Software",
                   "co_employee_band": "11-50", "co_monthly_ad_spend_band": "£5k-£25k",
                   "co_crm_platform": "HubSpot", "co_is_hiring": True})
    card = agent.lead_card(r)
    assert card.splitlines() == [
        "What they want to solve: attribution", "Job title: (not asked)", "Company typed: Acme",
        "Email domain: acme.example", "Country: UK", "Budget: £5k",
        "Company profile: Software, 11-50 employees, ad spend £5k-£25k/month, CRM HubSpot, hiring yes",
    ]


def test_call_uses_cache_without_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used")
    monkeypatch.setattr(agent.requests, "post", boom)
    import hashlib
    cache = {hashlib.sha256(b"sys" + b"card").hexdigest(): {"context_score": 0.7, "fit": "ok", "red_flags": []}}
    assert agent.call("sys", "card", "key", cache)["context_score"] == 0.7


def test_call_logs_failures_and_returns_error_record(monkeypatch, caplog):
    def fail(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(agent.requests, "post", fail)
    sleeps: list[float] = []
    monkeypatch.setattr(agent.time, "sleep", sleeps.append)
    cache: dict = {}
    with caplog.at_level(logging.WARNING, logger=agent.__name__):
        out = agent.call("sys", "card", "key", cache)
    assert out == error_response() and cache == {}
    assert len([r for r in caplog.records if "context call failed" in r.message]) == agent.MAX_ATTEMPTS
    assert sleeps == [2 ** i for i in range(agent.MAX_ATTEMPTS - 1)]  # no sleep after the last attempt


def test_contract_names_current_output():
    assert OUTPUT_COLUMNS == ("lead_id", "context_score", "fit", "red_flags")
    assert error_response() is not error_response()


def test_context_logit_clips_and_reindexes(tmp_path):
    path = tmp_path / "ctx.csv"
    pd.DataFrame({"lead_id": ["a", "b", "c"], "context_score": [0.5, 1.0, None]}).to_csv(path, index=False)
    out = context_logit(path, pd.Index(["a", "b", "c", "d"]))
    assert out.name == "context_logit"
    assert out["a"] == 0 and out["b"] == pytest.approx(np.log(0.999 / 0.001))
    assert np.isnan(out["c"]) and np.isnan(out["d"])
