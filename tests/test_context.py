"""Context agent v2: contract, lead card, cache, runtime (mocked client, no API calls), features, pipeline hook."""
import json
import logging
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import numpy as np
import pandas as pd
import pytest

from emva.context import agent
from emva.context.agent import brief_hash, judge, make_client, run_agent, select_leads
from emva.context.cache import ReplyCache, cache_key
from emva.context.card import BLANK, CARD_FIELDS, LeadCard, card_hash, lead_card, render
from emva.context.contract import (
    JUDGMENTS,
    OUTPUT_COLUMNS,
    REASON_MAX_WORDS,
    RESPONSE_SCHEMA,
    STAMP_COLUMNS,
    ContractError,
    validate,
)
from emva.context.features import (
    CONTEXT_CATS,
    CONTEXT_COLUMNS,
    CONTEXT_LEVELS,
    PERSONA_ACCEPTANCE_COLUMN,
    PERSONA_GROUP_LEVELS,
    PERSONA_GROUPS,
    context_design,
    context_features,
    context_logit,
    persona_group,
)
from emva.io import flag_bots_and_duplicates, load
from emva.pipeline import run

REPO = Path(__file__).resolve().parents[1]
DATA_V2 = REPO / "data" / "v2"
GOOD = {"is_real_business": "yes", "persona": "buyer", "problem_specificity": "specific", "urgency": "now",
        "brief_fit": "strong", "reason": "  Concrete attribution problem at a real advertiser.  "}
REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


# --- contract ---------------------------------------------------------------------------------------

def test_validate_accepts_a_good_reply_and_strips_the_reason():
    out = validate(GOOD)
    assert out["reason"] == "Concrete attribution problem at a real advertiser."
    assert {k: out[k] for k in JUDGMENTS} == {k: GOOD[k] for k in JUDGMENTS}


@pytest.mark.parametrize("change", [
    {"persona": "customer"},            # not in the enum
    {"urgency": "NOW"},                 # enums are case-sensitive
    {"brief_fit": None},
    {"reason": ""},
    {"reason": 3},
    {"reason": " ".join(["word"] * (REASON_MAX_WORDS + 1))},
    {"context_score": 0.7},             # extra key: no 0-1 score in the contract
])
def test_validate_rejects_off_contract_replies(change):
    with pytest.raises(ContractError):
        validate({**GOOD, **change})


def test_validate_rejects_missing_keys_and_non_objects():
    with pytest.raises(ContractError, match="missing"):
        validate({k: v for k, v in GOOD.items() if k != "persona"})
    with pytest.raises(ContractError, match="not a JSON object"):
        validate(["yes"])


def test_reason_at_the_word_limit_is_accepted():
    assert validate({**GOOD, "reason": " ".join(["w"] * REASON_MAX_WORDS)})


def test_schema_mirrors_the_judgments_and_has_no_score():
    props = RESPONSE_SCHEMA["properties"]
    assert set(props) == {*JUDGMENTS, "reason"} and set(RESPONSE_SCHEMA["required"]) == set(props)
    assert all(props[k]["enum"] == list(v) for k, v in JUDGMENTS.items())
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    assert JUDGMENTS["persona"] == ("buyer", "vendor", "student", "job_seeker", "competitor", "nonprofit",
                                    "agency_pitching", "unclear")
    assert set(STAMP_COLUMNS) <= set(OUTPUT_COLUMNS) and "context_score" not in OUTPUT_COLUMNS


# --- lead card --------------------------------------------------------------------------------------

FORBIDDEN = {"sector", "band", "employee", "spend", "crm", "hiring", "budget", "timeline", "company_size",
             "team_size", "country", "time_on_page", "session", "pricing", "ip", "stage", "deal", "channel", "utm"}


def test_card_has_only_the_four_allowed_fields():
    assert CARD_FIELDS == ("what_to_solve", "company_typed", "job_title", "email_domain")
    assert not {f.name for f in fields(LeadCard)} & FORBIDDEN


def test_card_leaks_nothing_the_formula_uses():
    row = pd.Series({"a_what_to_solve": "attribution from Meta", "a_company": "Acme", "a_job_title": "CMO",
                     "email": "Jo@Acme.Example", "a_budget": "£50k+", "a_timeline": "This month", "a_country": "UK",
                     "a_company_size": "51-200", "co_sector": "Software", "co_employee_band": "11-50",
                     "co_monthly_ad_spend_band": "£25k-£100k", "co_crm_platform": "HubSpot", "co_is_hiring": True,
                     "en_sector": "Software", "time_on_page_s": 431.0, "viewed_pricing": True, "final_stage": "Won",
                     "deal_value": 12000.0, "utm_source": "google"})
    text = render(lead_card(row))
    assert text.splitlines() == ["What they want to solve: attribution from Meta", "Company typed: Acme",
                                 "Job title: CMO", "Email domain: acme.example"]
    for leaked in ("£50k+", "This month", "UK", "51-200", "Software", "11-50", "£25k", "HubSpot", "True", "431",
                   "Won", "12000", "google"):
        assert leaked not in text


def test_card_falls_back_to_company_name_and_marks_blanks():
    row = pd.Series({"a_what_to_solve": None, "a_company": "  ", "company_name": "Beta Ltd", "a_job_title": np.nan,
                     "email": pd.NA})
    assert lead_card(row) == LeadCard(BLANK, "Beta Ltd", BLANK, BLANK)


def test_card_hash_is_stable_and_content_sensitive():
    a, b = LeadCard("x", "y", "z", "d.example"), LeadCard("x", "y", "z", "e.example")
    assert card_hash(a) == card_hash(LeadCard("x", "y", "z", "d.example")) != card_hash(b)


# --- cache ------------------------------------------------------------------------------------------

def test_cache_key_depends_on_all_five_identifiers():
    base = ("card", "brief", "v1", "fingerprint", "model")
    keys = {cache_key(*base)} | {cache_key(*(base[:i] + ("other",) + base[i + 1:])) for i in range(5)}
    assert len(keys) == 6


def test_concurrent_puts_lose_no_entries(tmp_path):
    import threading
    cache = ReplyCache(tmp_path / "c.json")

    def put_many(t: int) -> None:
        for i in range(50):
            cache.put(f"k{t}-{i}", {"brief_hash": "b", "status": "ok", "reply": GOOD})

    threads = [threading.Thread(target=put_many, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(cache.entries) == 400 and len(ReplyCache(tmp_path / "c.json").entries) == 400
    assert not list(tmp_path.glob("*.tmp"))


def test_prompt_edit_without_version_bump_is_a_cache_miss(tmp_path, leads, monkeypatch):
    cache = ReplyCache(tmp_path / "cache.json")
    run_agent(leads, "B", cache, client_factory=lambda: FakeClient(default=message(json.dumps(GOOD))))
    before = agent.prompt_fingerprint()
    monkeypatch.setattr(agent, "SYSTEM_TEMPLATE", agent.SYSTEM_TEMPLATE + " Be terse.")
    assert agent.prompt_fingerprint() != before
    _, stats = run_agent(leads, "B", cache, client_factory=no_client, dry_run=True)
    assert stats.cache_misses == 3 and stats.cache_hits == 0
    monkeypatch.setattr(agent, "MAX_TOKENS", agent.MAX_TOKENS + 1)
    monkeypatch.setattr(agent, "SYSTEM_TEMPLATE", agent.SYSTEM_TEMPLATE.removesuffix(" Be terse."))
    assert agent.prompt_fingerprint() != before


def test_cache_persists_and_rejects_unknown_format(tmp_path):
    path = tmp_path / "c.json"
    c = ReplyCache(path)
    c.put("k", {"brief_hash": "b", "status": "ok", "reply": GOOD})
    assert ReplyCache(path).get("k")["reply"] == GOOD and ReplyCache(path).brief_hashes() == {"b"}
    path.write_text(json.dumps({"format": 99, "entries": {}}))
    with pytest.raises(ValueError, match="format"):
        ReplyCache(path)


# --- runtime (mocked client) ------------------------------------------------------------------------

def message(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


class FakeClient:
    """Stands in for ``anthropic.Anthropic``: each call pops the next item of ``script`` (a reply or an exception)."""

    def __init__(self, script=None, default=None):
        self.script = list(script or [])
        self.default = default
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else self.default
        if isinstance(item, BaseException):
            raise item
        return item


def no_client():
    raise AssertionError("the API client must not be created")


@pytest.fixture
def leads() -> pd.DataFrame:
    return pd.DataFrame({"a_what_to_solve": ["Lead quality from Meta is poor", "need info", "Hiring?"],
                         "a_company": ["Acme", None, "Gamma"], "company_name": [None, "Beta", None],
                         "a_job_title": ["CMO", "Student", None],
                         "email": ["a@acme.example", "b@gmail.example", "c@gamma.example"]},
                        index=pd.Index(["L1", "L2", "L3"], name="lead_id"))


def test_run_stamps_every_row_and_rerun_reproduces_from_cache(tmp_path, leads):
    client = FakeClient(default=message(json.dumps(GOOD)))
    cache = ReplyCache(tmp_path / "cache.json")
    out, stats = run_agent(leads, "BRIEF A", cache, client_factory=lambda: client, workers=2)
    assert list(out.columns) == list(OUTPUT_COLUMNS) and len(client.calls) == 3
    assert (out.brief_hash == brief_hash("BRIEF A")).all()
    assert (out.prompt_version == agent.PROMPT_VERSION).all() and (out.model_id == agent.MODEL_ID).all()
    assert out.card_hash.tolist() == [card_hash(lead_card(r)) for _, r in leads.iterrows()]
    assert (out.status == "ok").all() and stats.api_requests == 3 and stats.cache_misses == 3
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5-20251001" and "BRIEF A" in call["system"]
    assert call["output_config"]["format"]["schema"] == RESPONSE_SCHEMA

    again, stats2 = run_agent(leads, "BRIEF A", ReplyCache(tmp_path / "cache.json"), client_factory=no_client)
    pd.testing.assert_frame_equal(out, again)
    assert stats2.cache_hits == 3 and stats2.api_requests == 0


def test_brief_edit_is_a_new_hash_and_dry_run_reports_misses_without_calling(tmp_path, leads):
    cache = ReplyCache(tmp_path / "cache.json")
    run_agent(leads, "BRIEF A", cache, client_factory=lambda: FakeClient(default=message(json.dumps(GOOD))))
    edited = tmp_path / "brief.md"
    edited.write_text("BRIEF A\n- Also: no agencies.")
    out, stats = run_agent(leads, edited.read_text(), cache, client_factory=no_client, dry_run=True)
    assert out.empty and stats.cache_misses == 3 and stats.cache_hits == 0
    assert stats.brief_hash != brief_hash("BRIEF A") and stats.cached_brief_hashes == {brief_hash("BRIEF A")}
    _, same = run_agent(leads, "BRIEF A", cache, client_factory=no_client, dry_run=True)
    assert same.cache_hits == 3 and same.cache_misses == 0


def test_dry_run_cli_reports_the_retrain_without_calling(tmp_path, monkeypatch, capsys, leads):
    monkeypatch.setattr(agent, "select_leads", lambda data, ids, limit: leads)
    monkeypatch.setattr(agent, "make_client", no_client)
    brief = tmp_path / "brief.md"
    brief.write_text("a temporary brief")
    agent.main(["--brief", str(brief), "--cache", str(tmp_path / "c.json"), "--dry-run"])
    printed = capsys.readouterr().out
    assert "3 misses (3 unique cards), 0 API requests" in printed and "refit of the context weights" in printed


@pytest.mark.parametrize("reply", [message("not json"), message(json.dumps({**GOOD, "persona": "customer"})),
                                   message(json.dumps(GOOD), stop_reason="max_tokens")])
def test_parse_errors_are_recorded_cached_and_not_retried(tmp_path, leads, reply):
    client = FakeClient(default=reply)
    cache = ReplyCache(tmp_path / "cache.json")
    out, stats = run_agent(leads.head(1), "B", cache, client_factory=lambda: client)
    assert out.status.tolist() == ["parse_error"] and out.error[0] and pd.isna(out.persona[0])
    assert len(client.calls) == 1 and stats.by_status["parse_error"] == 1 and stats.by_status["transport_error"] == 0
    again, _ = run_agent(leads.head(1), "B", cache, client_factory=no_client)
    assert again.status.tolist() == ["parse_error"] and again.error[0] == out.error[0]


def test_transport_errors_are_retried_with_backoff_logged_and_not_cached(tmp_path, leads, caplog):
    client = FakeClient(default=anthropic.APIConnectionError(request=REQ))
    sleeps: list[float] = []
    cache = ReplyCache(tmp_path / "cache.json")
    with caplog.at_level(logging.WARNING, logger=agent.__name__):
        out, stats = run_agent(leads.head(1), "B", cache, client_factory=lambda: client, sleep=sleeps.append)
    assert out.status.tolist() == ["transport_error"] and "APIConnectionError" in out.error[0]
    assert len(client.calls) == agent.MAX_ATTEMPTS == stats.api_requests and sleeps == [1, 2, 4, 8]
    assert len([r for r in caplog.records if "transport error" in r.message]) == agent.MAX_ATTEMPTS
    assert cache.entries == {} and stats.by_status == {"ok": 0, "parse_error": 0, "transport_error": 1}


def test_a_transient_overload_is_retried_then_succeeds():
    overloaded = anthropic.InternalServerError("overloaded", response=httpx2.Response(529, request=REQ), body=None)
    rate = anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=REQ), body=None)
    client = FakeClient([overloaded, rate, message(json.dumps(GOOD))])
    out = judge(client, "sys", "card", sleep=lambda s: None)
    assert out.status == "ok" and out.attempts == 3


def test_configuration_errors_raise_instead_of_being_retried():
    bad = anthropic.BadRequestError("bad schema", response=httpx2.Response(400, request=REQ), body=None)
    client = FakeClient([bad])
    with pytest.raises(anthropic.BadRequestError):
        judge(client, "sys", "card", sleep=lambda s: None)
    assert len(client.calls) == 1


def test_identical_cards_share_one_request(tmp_path, leads):
    twins = pd.concat([leads.head(1), leads.head(1).rename(index={"L1": "L1b"})])
    client = FakeClient(default=message(json.dumps(GOOD)))
    out, stats = run_agent(twins, "B", ReplyCache(tmp_path / "c.json"), client_factory=lambda: client)
    assert len(client.calls) == 1 and stats.api_requests == 1
    assert stats.cache_misses == 2 and stats.unique_misses == 1
    assert out.lead_id.tolist() == ["L1", "L1b"] and (out.status == "ok").all()


def test_transport_errors_reach_every_lead_sharing_the_card(tmp_path, leads):
    twins = pd.concat([leads.head(1), leads.head(1).rename(index={"L1": "L1b"})])
    client = FakeClient(default=anthropic.APIConnectionError(request=REQ))
    out, _ = run_agent(twins, "B", ReplyCache(tmp_path / "c.json"), client_factory=lambda: client, sleep=lambda s: None)
    assert out.status.tolist() == ["transport_error"] * 2 and len(client.calls) == agent.MAX_ATTEMPTS


def test_limit_zero_means_no_leads(data_v1):
    assert select_leads(data_v1, limit=0).empty
    assert len(select_leads(data_v1, limit=3)) == 3


def test_concurrency_is_capped(tmp_path, leads):
    with pytest.raises(ValueError, match="workers"):
        run_agent(leads, "B", ReplyCache(tmp_path / "c.json"), client_factory=no_client, workers=9)


def test_client_sends_the_workspace_header_and_does_its_own_retries(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    client = make_client()
    assert client.default_headers["anthropic-workspace-id"] == "wrkspc_test" and client.max_retries == 0


def test_client_refuses_to_run_without_a_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    monkeypatch.setattr(agent, "load_env_var", lambda name: "test-key" if name == "ANTHROPIC_API_KEY" else None)
    with pytest.raises(SystemExit, match="WORKSPACE"):
        make_client()


def make_repo(root: Path, git_file: str | None = None) -> Path:
    (root / "emva").mkdir(parents=True)
    if git_file is None:
        (root / ".git").mkdir()
    else:
        (root / ".git").write_text(git_file)
    return root


def test_load_env_var_reads_the_repo_dotenv(tmp_path, monkeypatch):
    from emva.env import find_dotenv, load_env_var
    monkeypatch.delenv("EMVA_TEST_VAR", raising=False)
    repo = make_repo(tmp_path / "repo")
    (repo / ".env").write_text("export EMVA_TEST_VAR='abc'\nOTHER=1\n")
    assert find_dotenv(repo / "emva") == repo / ".env"
    assert load_env_var("EMVA_TEST_VAR", repo / "emva") == "abc"
    assert load_env_var("MISSING_VAR", repo / "emva") is None
    monkeypatch.setenv("EMVA_TEST_VAR", "from-env")
    assert load_env_var("EMVA_TEST_VAR", repo / "emva") == "from-env"


def test_dotenv_search_stops_at_the_repo_root(tmp_path):
    from emva.env import find_dotenv
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=outside\n")      # above the repo: never read
    repo = make_repo(tmp_path / "repo")
    assert find_dotenv(repo / "emva") is None
    assert find_dotenv(tmp_path) is None                                   # not inside a repository


def test_dotenv_in_a_linked_worktree_falls_back_to_the_main_checkout(tmp_path):
    from emva.env import find_dotenv
    main = make_repo(tmp_path / "main")
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".env").write_text("X=1\n")
    wt = make_repo(main / ".claude" / "worktrees" / "wt", git_file=f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    assert find_dotenv(wt / "emva") == main / ".env"
    (wt / ".env").write_text("X=2\n")
    assert find_dotenv(wt / "emva") == wt / ".env"                         # the worktree's own file wins


def test_paraphrase_script_and_agent_share_the_loader(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import paraphrase_templates

    from emva import env
    assert paraphrase_templates.load_env_var is env.load_env_var is agent.load_env_var
    assert not hasattr(paraphrase_templates, "find_dotenv") and not hasattr(agent, "find_dotenv")


def test_bots_and_duplicates_are_dropped_before_calling(data_v1):
    F = flag_bots_and_duplicates(load(data_v1))
    bot, dup, human = F.index[F.bot][0], F.index[F.dup & ~F.bot][0], F.index[~F.bot & ~F.dup][0]
    assert select_leads(data_v1, [bot, human, dup]).index.tolist() == [human]
    with pytest.raises(KeyError):
        select_leads(data_v1, ["NOPE"])


# --- features ---------------------------------------------------------------------------------------

def judgments_frame(ids, values=None, status="ok", stamp=("b" * 16, "judgments-v1", "claude-haiku-4-5-20251001")):
    rows = [{"lead_id": i, **(values or {k: v for k, v in GOOD.items()}), "status": status, "error": "",
             "card_hash": "h", **dict(zip(STAMP_COLUMNS, stamp, strict=True))} for i in ids]
    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))


def test_context_levels_are_fixed():
    assert CONTEXT_LEVELS == {"ctx_is_real_business": ("yes", "no", "unclear"),
                              "ctx_persona_group": ("buyer", "non_buyer", "unclear"),
                              "ctx_problem_specificity": JUDGMENTS["problem_specificity"],
                              "ctx_urgency": JUDGMENTS["urgency"], "ctx_brief_fit": JUDGMENTS["brief_fit"]}
    assert CONTEXT_CATS == {"ctx_is_real_business": "yes", "ctx_persona_group": "buyer",
                            "ctx_problem_specificity": "specific", "ctx_urgency": "none", "ctx_brief_fit": "partial"}
    assert len(CONTEXT_COLUMNS) == 13 and PERSONA_ACCEPTANCE_COLUMN in CONTEXT_COLUMNS
    one = context_design(judgments_frame(["a"]), pd.Index(["a"]))
    assert list(one.columns) == CONTEXT_COLUMNS            # one row still gets the full schema
    assert one.loc["a", "ctx_urgency=now"] == 1 and one.loc["a", "ctx_brief_fit=strong"] == 1
    assert one.loc["a"].sum() == 2                          # yes / buyer / specific are reference levels


def test_context_design_nan_for_errors_and_absent_leads_and_raises_on_bad_input():
    J = pd.concat([judgments_frame(["a"]), judgments_frame(["b"], values=dict.fromkeys([*JUDGMENTS, "reason"]),
                                                         status="parse_error")])
    C = context_design(J, pd.Index(["a", "b", "c"]))
    assert C.loc["a"].notna().all() and C.loc["b"].isna().all() and C.loc["c"].isna().all()
    with pytest.raises(ValueError, match="outside its declared levels"):
        context_design(judgments_frame(["a"], values={**GOOD, "urgency": "soon"}), pd.Index(["a"]))
    with pytest.raises(ValueError, match="outside the contract"):
        context_design(judgments_frame(["a"], values={**GOOD, "persona": "customer"}), pd.Index(["a"]))
    with pytest.raises(ValueError, match="stamps"):
        context_design(pd.concat([judgments_frame(["a"]), judgments_frame(["b"], stamp=("c" * 16, "v", "m"))]),
                       pd.Index(["a"]))
    with pytest.raises(ValueError, match="duplicated"):
        context_design(judgments_frame(["a", "a"]), pd.Index(["a"]))


def test_every_contract_persona_maps_to_exactly_one_group():
    assert set(PERSONA_GROUPS) == set(JUDGMENTS["persona"]) and set(PERSONA_GROUPS.values()) == set(PERSONA_GROUP_LEVELS)
    grouped = persona_group(pd.Series(JUDGMENTS["persona"]))
    assert dict(zip(JUDGMENTS["persona"], grouped, strict=True)) == {
        "buyer": "buyer", "vendor": "non_buyer", "student": "non_buyer", "job_seeker": "non_buyer",
        "competitor": "non_buyer", "nonprofit": "non_buyer", "agency_pitching": "non_buyer", "unclear": "unclear"}
    with pytest.raises(ValueError, match="outside the contract"):
        persona_group(pd.Series(["buyer", "Buyer"]))


def test_persona_is_pooled_in_the_design():
    rows = [judgments_frame([p], values={**GOOD, "persona": p}) for p in JUDGMENTS["persona"]]
    C = context_design(pd.concat(rows), pd.Index(JUDGMENTS["persona"]))
    assert C["ctx_persona_group=non_buyer"].to_dict() == {p: float(PERSONA_GROUPS[p] == "non_buyer")
                                                          for p in JUDGMENTS["persona"]}
    assert C.loc["unclear", "ctx_persona_group=unclear"] == 1 and C.loc["buyer"].sum() == 2


def test_context_features_dispatches_on_format(tmp_path):
    legacy, judged, other = tmp_path / "l.csv", tmp_path / "j.csv", tmp_path / "o.csv"
    pd.DataFrame({"lead_id": ["a"], "context_score": [0.5]}).to_csv(legacy, index=False)
    judgments_frame(["a"]).to_csv(judged, index=False)
    pd.DataFrame({"lead_id": ["a"], "x": [1]}).to_csv(other, index=False)
    assert list(context_features(legacy, pd.Index(["a"])).columns) == ["context_logit"]
    assert list(context_features(judged, pd.Index(["a"])).columns) == CONTEXT_COLUMNS
    with pytest.raises(ValueError, match="neither"):
        context_features(other, pd.Index(["a"]))


def test_context_logit_clips_and_reindexes(tmp_path):
    path = tmp_path / "ctx.csv"
    pd.DataFrame({"lead_id": ["a", "b", "c"], "context_score": [0.5, 1.0, None]}).to_csv(path, index=False)
    out = context_logit(path, pd.Index(["a", "b", "c", "d"]))
    assert out.name == "context_logit"
    assert out["a"] == 0 and out["b"] == pytest.approx(np.log(0.999 / 0.001))
    assert np.isnan(out["c"]) and np.isnan(out["d"])


# --- pipeline hook ----------------------------------------------------------------------------------

def test_pipeline_context_path_joins_judgment_features(tmp_path, data_v1):
    ids = pd.read_csv(data_v1 / "historical_leads.csv").lead_id.sample(4000, random_state=3).tolist()
    rng = np.random.default_rng(0)
    J = judgments_frame(ids)
    for k, v in JUDGMENTS.items():
        J[k] = rng.choice(v, len(J))
    path = tmp_path / "judgments.csv"
    J.to_csv(path, index=False)
    res = run(data_v1, context=path)
    have = res.X.p_combined.notna()
    assert set(CONTEXT_COLUMNS) <= set(res.X.columns) and have.sum() > 0
    assert res.X.loc[have, CONTEXT_COLUMNS].notna().all().all() and res.X.loc[~have, CONTEXT_COLUMNS].isna().all().all()
    assert res.summary.model.tolist() == ["formula", "context_only", "formula+context"]
    assert any(m.startswith("context weights in combined model: ctx_is_real_business=no") for m in res.messages)


# --- the committed real run -------------------------------------------------------------------------

def test_committed_cache_reproduces_the_committed_judgments_without_the_api():
    ctx = DATA_V2 / "context"
    ids = pd.read_csv(ctx / "sample_ids.csv").lead_id.tolist()
    brief = (REPO / "baseline" / "business_brief.md").read_text(encoding="utf-8")
    out, stats = run_agent(select_leads(DATA_V2, ids), brief, ReplyCache(ctx / "cache.json"), client_factory=no_client)
    committed = pd.read_csv(ctx / "context_judgments.csv", keep_default_na=False)
    assert stats.cache_hits == len(ids) == 1000 and stats.api_requests == 0
    pd.testing.assert_frame_equal(out.fillna("").astype(str), committed.astype(str))


def test_committed_cache_is_rekeyed_with_the_current_prompt_fingerprint():
    data = json.loads((DATA_V2 / "context" / "cache.json").read_text(encoding="utf-8"))
    assert data["format"] == 2 and len(data["entries"]) == 998       # 1,000 leads, 998 unique cards
    fp = agent.prompt_fingerprint()
    for key, e in data["entries"].items():
        assert e["prompt_fingerprint"] == fp
        assert key == cache_key(e["card_hash"], e["brief_hash"], e["prompt_version"], fp, e["model_id"])


def test_committed_sample_file_holds_lead_ids_only():
    assert list(pd.read_csv(DATA_V2 / "context" / "sample_ids.csv").columns) == ["lead_id"]


def ground_truth_readers() -> set[str]:
    """``emva.eval`` modules whose source names a ground-truth file."""
    return {f"emva.eval.{p.stem}" for p in (REPO / "emva" / "eval").glob("*.py")
            if "ground_truth" in p.read_text(encoding="utf-8")}


def test_run_script_never_imports_ground_truth_readers():
    import subprocess
    import sys
    code = ("import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('rca', {str(REPO / 'scripts' / 'run_context_agent.py')!r}); "
            "spec.loader.exec_module(importlib.util.module_from_spec(spec)); "
            "print('\\n'.join(m for m in sys.modules if m.startswith('emva.eval')))")
    loaded = set(subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                                check=True).stdout.split())
    readers = ground_truth_readers()
    assert "emva.eval.context_harness" in readers and not loaded & readers


def test_run_script_dry_run_has_no_file_side_effects(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import run_context_agent
    ctx = DATA_V2 / "context"
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in ctx.iterdir()}
    run_context_agent.main(["--dry-run"])
    assert "1000 cache hits, 0 misses" in capsys.readouterr().out
    assert {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in ctx.iterdir()} == before
