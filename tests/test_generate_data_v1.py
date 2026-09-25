"""Tests for scripts/generate_data_v1.py: determinism, v1 schema equality, sanity ranges, pipeline smoke test."""
import filecmp
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "data", "v1")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import generate_data_v1 as gen  # noqa: E402
from emva.design import design  # noqa: E402
from emva.features import text_cat  # noqa: E402
from emva.io import load  # noqa: E402
from emva.labels import LEGACY  # noqa: E402
from emva.pipeline import build  # noqa: E402

N = 2000
CSVS = ["historical_leads", "crm_history", "companies", "people", "ground_truth_labels"]
FILES = [f"{n}.csv" for n in CSVS] + ["status_quo_rules.json", "ground_truth.md"]


def _generate(out_dir, seed):
    cfg = gen.Config(seed=seed, n=N)
    gen.write(gen.generate(cfg), str(out_dir), cfg)
    return str(out_dir)


@pytest.fixture(scope="session")
def run_a(tmp_path_factory):
    return _generate(tmp_path_factory.mktemp("a"), 20260924)


@pytest.fixture(scope="session")
def run_b(tmp_path_factory):
    return _generate(tmp_path_factory.mktemp("b"), 20260924)


@pytest.fixture(scope="session")
def run_c(tmp_path_factory):
    return _generate(tmp_path_factory.mktemp("c"), 7)


def read(d, name):
    return pd.read_csv(os.path.join(d, f"{name}.csv"))


@pytest.fixture(scope="session")
def v1():
    return {n: read(V1, n) for n in CSVS}


@pytest.fixture(scope="session")
def g(run_a):
    return {n: read(run_a, n) for n in CSVS}


# ---------------------------------------------------------------- determinism

def test_same_seed_identical_files(run_a, run_b):
    for f in FILES:
        assert filecmp.cmp(os.path.join(run_a, f), os.path.join(run_b, f), shallow=False), f


def test_different_seed_different_files(run_a, run_c):
    for f in [f"{n}.csv" for n in CSVS]:
        assert not filecmp.cmp(os.path.join(run_a, f), os.path.join(run_c, f), shallow=False), f


# ---------------------------------------------------------------- schema equality with data/v1

@pytest.mark.parametrize("name", CSVS)
def test_columns_and_dtypes_match_v1(v1, g, name):
    assert list(g[name].columns) == list(v1[name].columns)
    assert g[name].dtypes.astype(str).to_dict() == v1[name].dtypes.astype(str).to_dict()


def test_status_quo_rules_identical(run_a):
    with open(os.path.join(V1, "status_quo_rules.json")) as f1, open(os.path.join(run_a, "status_quo_rules.json")) as f2:
        assert json.load(f1) == json.load(f2)


def answer_keysets(L):
    return set(L.answers.apply(lambda s: tuple(json.loads(s).keys())))


def test_answer_json_keys_match_v1(v1, g):
    assert answer_keysets(g["historical_leads"]) == answer_keysets(v1["historical_leads"])
    for variant in "ABCD":
        a = answer_keysets(g["historical_leads"][g["historical_leads"].form_variant == variant])
        b = answer_keysets(v1["historical_leads"][v1["historical_leads"].form_variant == variant])
        assert a <= b, variant


def vocab(df, col):
    return set(df[col].dropna().astype(str))


CATEGORICAL = {
    "historical_leads": ["form_variant", "consent", "utm_source", "utm_medium", "utm_campaign", "utm_content",
                         "pasted_text", "returned_visitor", "device", "user_agent", "utm_term", "referrer", "ip_country",
                         "ip_city", "is_datacenter_ip", "timezone", "browser_language", "submitted_weekday",
                         "viewed_pricing"],
    "crm_history": ["stage"],
    "companies": ["sector", "employee_band", "country", "hq_city", "annual_revenue_band", "funding_stage",
                  "crm_platform", "monthly_ad_spend_band", "is_hiring"],
    "people": ["age_band", "household_income_band", "country", "region", "city", "job_title", "seniority",
               "department", "is_homeowner"],
    "ground_truth_labels": ["outcome", "channel", "is_lead_ads", "is_bot", "is_free_email", "has_company",
                            "employee_band", "sector", "seniority", "text_category", "country", "click_id_missing",
                            "never_contacted", "stalled"],
}


@pytest.mark.parametrize("name", CATEGORICAL)
def test_categorical_vocabularies_match_v1(v1, g, name):
    for col in CATEGORICAL[name]:
        assert vocab(g[name], col) == vocab(v1[name], col), (name, col)


def test_answer_vocabularies_within_v1(v1, g):
    def answers(L, k):
        return set(str(json.loads(s)[k]) for s in L.answers if k in json.loads(s))
    for k in ["country", "job_title", "company_size", "budget", "timeline"]:
        assert answers(g["historical_leads"], k) == answers(v1["historical_leads"], k), k


def test_crm_comment_vocabulary_within_v1(v1, g):
    def norm(s):
        return set(s.dropna().str.replace(r"spoke to [A-Z][a-z]+", "spoke to X", regex=True)
                   .str.replace(r"\d+", "N", regex=True))
    assert norm(g["crm_history"].comment) <= norm(v1["crm_history"].comment)


def test_regex_text_category_agrees_with_truth(g):
    """v1 property that Phase 5.2 will deliberately break: the POC regex recovers every category."""
    L = g["historical_leads"].merge(g["ground_truth_labels"], on="lead_id")
    cat = L.answers.apply(lambda s: text_cat(json.loads(s)["what_to_solve"]))
    assert (cat == L.text_category).all()


def test_formats(g):
    L, C = g["historical_leads"], g["crm_history"]
    assert L.lead_id.str.fullmatch(r"L\d{5}").all() and L.lead_id.is_unique
    assert L.created_at.str.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ").all()
    assert (pd.to_datetime(L.created_at).diff().dropna() >= pd.Timedelta(0)).all()      # ids follow time
    assert pd.to_datetime(L.created_at).max() < pd.Timestamp("2026-09-24", tz="UTC")
    assert pd.to_datetime(C.changed_at).max() <= pd.Timestamp("2026-09-24", tz="UTC")
    assert set(C.lead_id) == set(L.lead_id)                                             # every lead has a New row
    assert L.landing_url.dropna().str.startswith("https://lp.emva-poc.example/lp.html?variant=").all()
    assert (g["people"].email == g["people"].email.str.lower()).all()


# ---------------------------------------------------------------- sanity ranges vs v1

def test_sanity_ranges(v1, g):
    G, Gv = g["ground_truth_labels"], v1["ground_truth_labels"]
    won_share = (G.outcome == "won").mean()
    assert 0.095 <= won_share <= 0.145                          # "~12% of all leads end up Won"
    assert abs(G.is_bot.mean() - Gv.is_bot.mean()) < 0.015      # v1 3.9%
    assert abs(G.duplicate_of.notna().mean() - Gv.duplicate_of.notna().mean()) < 0.002   # v1 3.0%
    for o in ["never_contacted", "open", "lost"]:
        assert abs((G.outcome == o).mean() - (Gv.outcome == o).mean()) < 0.03, o
    def decided_win_rate(labels):
        dec = labels.outcome.isin(["won", "lost"])
        return (labels.outcome[dec] == "won").mean()
    assert abs(decided_win_rate(G) - decided_win_rate(Gv)) < 0.03
    assert G.deal_value.between(500, 60000).all() and (G.deal_value % 50 == 0).all()
    assert abs(G.is_free_email.mean() - Gv.is_free_email.mean()) < 0.03


def test_baseline_pipeline_runs_on_output(run_a):
    X = build(load(run_a), LEGACY)  # the baseline labels; horizon is the default since Phase 1
    assert len(X) > 0.9 * N * 0.9
    assert X.y.notna().sum() > 0.6 * len(X)
    assert 0.08 < X.y.mean() < 0.25
    D = design(X)
    assert D.shape[0] == len(X) and D.notna().all().all()


def test_ground_truth_md_written_from_output(run_a, g):
    md = open(os.path.join(run_a, "ground_truth.md")).read()
    G = g["ground_truth_labels"]
    assert "seed 20260924" in md and f"n = {N}" in md
    assert f"| {int(G.duplicate_of.notna().sum())} leads |" in md
    assert f"{int((G.outcome == 'never_contacted').sum())} leads (plus duplicates" in md


def test_v2_options_cover_all_seven_phase5_tasks():
    """Phase 5.2-5.8 are implemented (tests/test_generate_data_v2.py); every option is off by default."""
    tasks = {f.metadata["v2"] for f in gen.fields(gen.Config) if "v2" in f.metadata}
    assert tasks == {"5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8"}
    assert gen.v2_on(gen.Config()) == []


# ---------------------------------------------------------------- review fixes

@pytest.fixture(scope="session")
def run_3k(tmp_path_factory):
    cfg = gen.Config(seed=20260924, n=3000)
    d = str(tmp_path_factory.mktemp("n3000"))
    gen.write(gen.generate(cfg), d, cfg)
    return {n: read(d, n) for n in CSVS}


def test_duplicates_of_phoneless_originals_never_edit_phone(run_3k):
    L = run_3k["historical_leads"].set_index("lead_id")
    G = run_3k["ground_truth_labels"]
    dups = G[G.duplicate_of.notna()]
    phoneless = dups[L.phone.reindex(dups.duplicate_of).isna().to_numpy()]
    assert len(phoneless) > 20
    edited = L.fields_edited.reindex(phoneless.lead_id).dropna()
    assert not edited.apply(lambda s: "phone" in json.loads(s)).any()


def _lead_and_person(viewed_pricing):
    lead = pd.Series({"text_category": "neutral", "time_on_page_s": 120.0, "hesitation_ms": 5000, "field_edit_count": 0,
                      "form_variant": "A", "viewed_pricing": viewed_pricing, "sessions_before_convert": 1,
                      "utm_term": None, "submitted_weekday": "Tuesday", "local_submit_hour": 11, "ip_country": "UK"})
    person = pd.Series({"employee_band": "1-10", "is_free_email": False, "channel": "google", "is_lead_ads": False,
                        "seniority": "junior", "country": "UK"})
    return lead, person


@pytest.mark.parametrize("yes,no", [(True, False), (np.True_, np.False_),
                                    (pd.array([True], dtype="boolean")[0], pd.array([False], dtype="boolean")[0]),
                                    (True, np.nan), (True, None), (True, pd.NA)])
def test_pricing_effect_applied_for_any_bool_dtype(yes, no):
    cfg = gen.Config()
    z_yes = gen.close_log_odds(cfg, *_lead_and_person(yes), None, 5.0)
    z_no = gen.close_log_odds(cfg, *_lead_and_person(no), None, 5.0)
    assert z_yes - z_no == pytest.approx(cfg.pricing_eff)


def test_planted_effects_visible_in_decided_win_rates(run_3k):
    G = run_3k["ground_truth_labels"]
    D = G[G.outcome.isin(["won", "lost"])]

    def win(mask):
        return (D.outcome[mask] == "won").mean()
    assert win(D.employee_band == "51-200") > 2 * win(D.employee_band == "1-10")
    assert win(D.is_free_email) < win(~D.is_free_email)
    assert win(D.text_category == "specific") > win(D.text_category == "vague")
