"""Tests for the Phase 5.2-5.8 options of scripts/generate_data_v1.py (entry point scripts/generate_data_v2.py).

Covers: v1 output pinned byte for byte with every option off; v2 determinism; each option on its own changes only
what it should; the acceptance properties (regex agreement < 100%, bot-rule precision < 100%) and the planted
shares on a small v2 sample; the paraphrase option failing loudly without a cache.
"""
import filecmp
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import generate_data_v1 as gen  # noqa: E402
import generate_data_v2 as gen2  # noqa: E402
import paraphrase_templates as pt  # noqa: E402
import v2_checks  # noqa: E402
import v2_text  # noqa: E402
from emva.features import text_cat  # noqa: E402
from emva.io import load  # noqa: E402
from emva.pipeline import build  # noqa: E402

N_OPT = 1200      # per-option comparisons (in memory)
N_V2 = 3000       # full v2 sample (written to disk)
V1_PIN_N = 600
V1_PIN = {        # sha256 of the v1 output for seed 20260924, n = 600, with every v2 option off
    "companies.csv": "4cb4a9c794f75ae1ddf900ad24c32ecb6933bdc8b77964a4db2ac7e8d58b573d",
    "crm_history.csv": "8c2c17cf1f93830426abc7cb21fd04f277f3d4ef39293a91fa5e7dea88ba73ea",
    "ground_truth.md": "54044acccd7fab4ff06b8055e197c76b5f19ff60f3490c7271a3f0b7c18c4665",
    "ground_truth_labels.csv": "d9fdcc54f4c72e89664cb07f57134b9bf1cb0e92162fd4adaf6d86b7bd19be5b",
    "historical_leads.csv": "f6b19d18054775fd039bfad65c2035816e009c7ba1d55ec7cf8ed0f608ba0b6a",
    "people.csv": "437bcb68dc6691713bf8c083129e6e241ef116d86cf44aa88e7e995dec932b53",
    "status_quo_rules.json": "0376f9b7764d724d35e57f6ab5dc9608eae358dd8fd3a4abc033cc0baef13269",
}
NO_PARA = {**gen2.V2_SETTINGS, "text_paraphrase": False}


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------- v1 unchanged, v2 deterministic

def test_v1_output_pinned_with_options_off(tmp_path):
    cfg = gen.Config(n=V1_PIN_N)
    assert gen.v2_on(cfg) == []
    gen.write(gen.generate(cfg), str(tmp_path), cfg)
    assert sorted(os.listdir(tmp_path)) == sorted(V1_PIN)
    for f, h in V1_PIN.items():
        assert sha(tmp_path / f) == h, f


@pytest.fixture(scope="module")
def v2_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("v2")
    cfg = gen2.v2_config(n=N_V2, paraphrase=False)
    gen.write(gen.generate(cfg), str(d), cfg)
    return str(d)


def test_v2_same_seed_identical_files(v2_dir, tmp_path):
    cfg = gen2.v2_config(n=N_V2, paraphrase=False)
    gen.write(gen.generate(cfg), str(tmp_path), cfg)
    files = sorted(os.listdir(v2_dir))
    assert files == sorted(os.listdir(tmp_path)) and "ground_truth_companies.csv" in files
    for f in files:
        assert filecmp.cmp(os.path.join(v2_dir, f), tmp_path / f, shallow=False), f


def test_v2_schema(v2_dir):
    G = pd.read_csv(os.path.join(v2_dir, "ground_truth_labels.csv"))
    assert list(G.columns) == gen.LABEL_COLUMNS + gen.V2_LABEL_COLUMNS
    L = pd.read_csv(os.path.join(v2_dir, "historical_leads.csv"))
    assert list(L.columns) == gen.LEAD_COLUMNS


# ---------------------------------------------------------------- each option changes only what it should

@pytest.fixture(scope="module")
def base():
    return gen.generate(gen.Config(n=N_OPT))


def run_opt(**kw):
    return gen.generate(gen.Config(n=N_OPT, **kw))


def what(df):
    return df.answers.apply(lambda s: json.loads(s)["what_to_solve"])


def assert_same_except(a, b, cols):
    keep = [c for c in a.columns if c not in cols]
    pd.testing.assert_frame_equal(a[keep], b[keep])


def answers_without(df, key):
    return df.answers.apply(lambda s: {k: v for k, v in json.loads(s).items() if k != key})


@pytest.mark.parametrize("opt", [{"text_typo_share": 0.3}, {"text_language_mix_share": 0.35},
                                 {"text_boilerplate_pool": True}])
def test_text_options_change_only_the_free_text(base, opt):
    out = run_opt(**opt)
    L0, L1 = base["historical_leads"], out["historical_leads"]
    assert_same_except(L0, L1, ["answers"])
    assert (answers_without(L0, "what_to_solve") == answers_without(L1, "what_to_solve")).all()
    for name in ["crm_history", "companies", "people"]:
        pd.testing.assert_frame_equal(base[name], out[name])
    pd.testing.assert_frame_equal(base["ground_truth_labels"], out["ground_truth_labels"][gen.LABEL_COLUMNS])
    changed = what(L0) != what(L1)
    G = out["ground_truth_labels"]
    if "text_typo_share" in opt:
        nonempty = what(L0).str.contains(r"[A-Za-z]")
        assert 0.22 < changed[nonempty].mean() < 0.38
    if "text_language_mix_share" in opt:
        assert set(G.country[changed]) <= {"DE", "FR", "NL"}
        assert 0.25 < changed[G.country.isin(["DE", "FR", "NL"])].mean() < 0.45
    if "text_boilerplate_pool" in opt:
        assert (G.text_category[changed] == "copy_paste").all()
        assert changed[G.text_category == "copy_paste"].mean() > 0.6


def test_enrichment_dropout(base):
    out = run_opt(enrichment_dropout=0.3)
    C0, C1, H = base["companies"], out["companies"], out["ground_truth_companies"]
    assert_same_except(C0, C1, ["domain"])
    pd.testing.assert_frame_equal(C0, H[C0.columns])                       # hidden truth keeps every domain
    G = out["ground_truth_labels"]
    business = G[~G.is_free_email & ~G.is_bot & G.has_company & G.duplicate_of.isna()]
    used = set(business.company_domain)
    missing = used - set(C1.domain)
    assert len(missing) == round(0.3 * len(used))
    assert not set(C1.domain) & (set(C0.domain) - set(C1.domain))         # legacy domains are new strings
    assert (G.domain_in_companies[G.company_domain.isin(missing)] == False).all()  # noqa: E712
    pd.testing.assert_frame_equal(base["ground_truth_labels"], G[gen.LABEL_COLUMNS])
    L0, L1 = base["historical_leads"], out["historical_leads"]
    assert_same_except(L0, L1, ["answers", "company_name"])
    assert (answers_without(L0, "company") == answers_without(L1, "company")).all()
    varied = (L0.company_name != L1.company_name) & L0.company_name.notna()
    assert varied.sum() > 0.4 * L0.company_name.notna().sum()
    n0 = L0.company_name[varied].map(v2_checks.normalise_company)
    n1 = L1.company_name[varied].map(v2_checks.normalise_company)
    assert (n0 == n1).all()                                                # noise a matcher can undo


def test_consent_missingness(base):
    out = run_opt(consent_missing_share=0.15)
    L0, L1, G = base["historical_leads"], out["historical_leads"], out["ground_truth_labels"]
    assert_same_except(L0, L1, gen.CONSENT_COLUMNS)
    cd = G.consent_declined.astype(bool)
    web = L1.utm_medium.ne("lead_form")
    assert not (cd & ~web).any()
    assert 0.11 < cd[web].mean() < 0.19
    assert L1.loc[cd, gen.CONSENT_COLUMNS].isna().all().all()
    pd.testing.assert_frame_equal(L0.loc[~cd, gen.CONSENT_COLUMNS], L1.loc[~cd, gen.CONSENT_COLUMNS])


def test_fast_humans_change_only_time_and_hesitation(base):
    out = run_opt(fast_human_share=0.02)
    L0, L1, G = base["historical_leads"], out["historical_leads"], out["ground_truth_labels"]
    assert_same_except(L0, L1, ["time_on_page_s", "hesitation_ms"])
    fast = G.fast_human.astype(bool)
    assert ((L0.time_on_page_s.fillna(-1) != L1.time_on_page_s.fillna(-1)) == fast).all()
    assert (L1.time_on_page_s[fast] < 15).all() and not G.is_bot[fast].any()
    eligible = L1.utm_medium.ne("lead_form") & ~G.is_bot
    assert 0.008 < fast[eligible].mean() < 0.035
    pd.testing.assert_frame_equal(base["ground_truth_labels"], G[gen.LABEL_COLUMNS])   # close odds unchanged


def test_interactions_change_only_the_two_cells(base):
    out = run_opt(interactions=True)
    pd.testing.assert_frame_equal(base["historical_leads"], out["historical_leads"])
    G0, G1 = base["ground_truth_labels"], out["ground_truth_labels"]
    L = base["historical_leads"]
    orig = G0.duplicate_of.isna()
    li = (G0.channel == "linkedin") & G0.employee_band.isin(["51-200", "201-1000", "1000+"])
    sd = (G0.seniority == "senior") & (L.form_variant == "D")
    same = orig & ~li & ~sd
    assert (G0.p_close_true[same] == G1.p_close_true[same]).all()
    assert (G1.p_close_true[orig & li & ~sd] >= G0.p_close_true[orig & li & ~sd]).all()
    assert (G1.p_close_true[orig & sd & ~li] <= G0.p_close_true[orig & sd & ~li]).all()
    assert (orig & li).sum() > 30 and (orig & sd).sum() > 30


def test_ghosting_follows_tier_changes_contact_not_behaviour(base):
    out = run_opt(ghosting_follows_tier=True)
    pd.testing.assert_frame_equal(base["historical_leads"], out["historical_leads"])
    G0, G1 = base["ground_truth_labels"], out["ground_truth_labels"]
    orig = G0.duplicate_of.isna()
    assert (G0.never_contacted[orig] != G1.never_contacted[orig]).mean() > 0.1
    same_reply = orig & (G0.reply_hours.fillna(-1) == G1.reply_hours.fillna(-1))
    assert same_reply.mean() > 0.5
    assert (G0.p_close_true[same_reply] == G1.p_close_true[same_reply]).all()   # v1 noise draws kept aligned


def test_persona_changes_text_and_odds_of_persona_leads_only(base):
    out = run_opt(context_only_signal=True)
    L0, L1, G = base["historical_leads"], out["historical_leads"], out["ground_truth_labels"]
    assert_same_except(L0, L1, ["answers"])
    persona = G.context_persona.notna()
    changed = what(L0) != what(L1)
    assert (changed == (persona & (G.text_category != "vague"))).all()
    assert 0.05 < persona[G.duplicate_of.isna()].mean() < 0.11
    assert set(G.context_persona.dropna()) == set(v2_text.PERSONAS)
    G0 = base["ground_truth_labels"]
    orig = G.duplicate_of.isna()
    assert (G0.p_close_true[orig & ~persona] == G.p_close_true[orig & ~persona]).all()
    assert (G.p_close_true[orig & persona] < G0.p_close_true[orig & persona]).all()


def test_persona_sentences_invisible_to_the_regex():
    for persona, sentences in v2_text.PERSONA_SENTENCES.items():
        for s in sentences:
            assert text_cat(s) == "neutral", s
            for base_text in gen.NEUTRAL_TEXTS:
                assert text_cat(f"{base_text} {s}") == "neutral", s
            for cp in gen.COPY_PASTE_TEXTS:
                assert text_cat(f"{cp} {s}") == "copy_paste", s
    assert len(v2_text.BOILERPLATE_POOL) >= 15
    assert all(text_cat(b) == "neutral" for b in v2_text.BOILERPLATE_POOL)       # not one of the three prefixes


# ---------------------------------------------------------------- paraphrase option

def test_paraphrase_fails_loudly_without_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(pt, "CACHE_PATH", str(tmp_path / "empty_cache.json"))
    with pytest.raises(pt.MissingParaphraseError, match="--populate"):
        gen.generate(gen.Config(n=200, text_paraphrase=True))


def test_paraphrase_uses_cache_entries(monkeypatch, tmp_path, base):
    """Test fixture cache (placeholder-preserving dummy rewrites, not LLM output) drives the substitution."""
    path = str(tmp_path / "cache.json")
    entries = {pt.cache_key(t): {"template": t, "model": pt.MODEL_ID, "prompt_version": pt.PROMPT_VERSION,
                                 "paraphrases": [f"[p{i}] {t}" for i in range(pt.N_PARAPHRASES)]}
               for t in gen.all_text_templates()}
    pt.save_cache(entries, path)
    monkeypatch.setattr(pt, "CACHE_PATH", path)
    out = gen.generate(gen.Config(n=N_OPT, text_paraphrase=True))
    w = what(out["historical_leads"])
    share = w.str.startswith("[p").mean()
    assert 0.6 < share < 0.9                                     # 5 of 6 variants, minus empty and "?" answers
    stripped = w.str.replace(r"^\[p\d\] ", "", regex=True)
    assert (stripped == what(base["historical_leads"])).all()    # placeholders were filled with the same values


def test_cache_validation_rejects_lost_placeholders():
    t = gen.SPECIFIC_TEMPLATES[0]
    with pytest.raises(ValueError, match="placeholder"):
        pt.validate(t, ["Pricing for some seats please."] + [f"{t} {i}" for i in range(pt.N_PARAPHRASES - 1)])


# ---------------------------------------------------------------- acceptance properties on a v2 sample

@pytest.fixture(scope="module")
def v2m(v2_dir):
    return v2_checks.measure_v2(v2_dir, gen2.v2_config(n=N_V2, paraphrase=False))


def test_regex_agreement_below_100(v2m):
    assert 0.5 < v2m["regex_agreement"] < 1.0
    cm = v2m["regex_confusion"]
    assert cm.values.sum() == N_V2 and np.trace(cm.values) < N_V2


def test_bot_rule_precision_below_100(v2m):
    br = v2m["bot_rule"]
    assert br["precision"] < 1.0 and br["fp"] == br["fp_fast_humans"] > 0
    assert br["recall"] < 1.0 and br["fn"] == br["fn_consent_declined"]


def test_v2_planted_shares(v2m):
    assert 0.05 < v2m["persona"]["share_all_leads"] < 0.11
    assert v2m["persona"]["regex_dist_max_gap"] < 0.07
    assert 0.008 < v2m["fast_humans"]["share"] < 0.035
    assert 0.11 < v2m["consent"]["share_of_web"] < 0.19 and v2m["consent"]["all_columns_blank"]
    dr = v2m["dropout"]
    assert dr["business_domains_missing"] == round(0.3 * dr["business_domains"])
    assert dr["domain_miss_matchable"] > 0.8 * dr["domain_miss_typed_name"] > 0


def test_ghosting_tracks_tier_more_than_true_p(v2m):
    gh = v2m["ghosting"]
    assert gh["corr_tier"] > abs(gh["corr_p"])
    assert gh["never_by_tier"]["A"] < gh["never_by_tier"]["B"] < gh["never_by_tier"]["C"]


def test_planted_effects_recovered_on_v2(v2m):
    pl, sd = v2m["planted"], v2m["planted_resid_sd"]
    assert 0.45 < sd < 0.55
    assert (pl.drop(index="top=<15s").abs_diff < 0.25).all(), pl[pl.abs_diff >= 0.25]


def test_planted_effects_visible_in_decided_win_rates_on_v2(v2_dir):
    """Same check as the v1 test: the planted v1 effects are still there in v2."""
    G = pd.read_csv(os.path.join(v2_dir, "ground_truth_labels.csv"))
    D = G[G.outcome.isin(["won", "lost"])]

    def win(mask):
        return (D.outcome[mask] == "won").mean()
    assert win(D.employee_band == "51-200") > 2 * win(D.employee_band == "1-10")
    assert win(D.is_free_email) < win(~D.is_free_email)
    assert win(D.text_category == "specific") > win(D.text_category == "vague")
    assert win(D.context_persona.notna()) < win(D.context_persona.isna())


def test_baseline_pipeline_runs_on_v2(v2_dir):
    X = build(load(v2_dir))
    assert len(X) > 0.8 * N_V2 and 0.08 < X.y.mean() < 0.25


def test_ground_truth_md_has_v2_section(v2_dir):
    md = open(os.path.join(v2_dir, "ground_truth.md")).read()
    assert "## v2 mechanisms" in md and "generate_data_v2.py" in md
    assert "LLM paraphrase OFF" in md
