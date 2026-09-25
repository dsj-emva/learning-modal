"""Synthetic lead data generator, v1 behaviour (rewrite of the lost scripts/generate_data.py).

Usage:
  python scripts/generate_data_v1.py --seed 20260924 --as-of 2026-09-24 --out data/v1_regen --n 10000

Writes the six v1 files (historical_leads.csv, crm_history.csv, companies.csv, people.csv,
ground_truth_labels.csv, status_quo_rules.json) plus ground_truth.md with tables measured on the
output. The original generator is not in the repo; this file was rebuilt from data/v1/ground_truth.md
and a profile of the v1 files, so it reproduces v1's schema and distributions, not its rows.

Pipeline (one function per stage, each with its own RNG stream so later phases can add options to one
stage without shifting the random draws of the other stages):

  companies -> people (lead personas) -> leads (channel, form, answers skeleton, timestamps)
  -> behaviour -> text -> hidden log-odds -> outcome -> duplicates -> CRM history -> vendor people
  -> labels -> ground_truth.md

Random streams are independent per stage, not per row: a Phase 5.x option that adds a draw inside a
stage shifts every later row of that stage (other stages are untouched).

This is the one place in the repo that knows the hidden truth. Only emva/eval/ may read
ground_truth_labels.csv or ground_truth.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:          # ground_truth_report, v2_text, paraphrase_templates are siblings in scripts/
    sys.path.insert(0, _HERE)

_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:          # emva.features.text_cat: the regex the 5.7 persona sentences must not trigger
    sys.path.insert(0, _ROOT)

import paraphrase_templates  # noqa: E402
import v2_text  # noqa: E402
from emva.features import text_cat  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Configuration: every planted effect and every distribution constant lives here.
# ---------------------------------------------------------------------------------------------

BANDS = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
COUNTRIES = ["UK", "US", "DE", "FR", "NL"]
SECTORS = ["Education", "Financial services", "Healthcare", "Logistics", "Manufacturing",
           "Professional services", "Retail", "Software"]
CHANNELS = ["meta", "google", "linkedin", "chatgpt", "organic_direct"]
FREE_DOMAINS = ["gmail.example", "yahoo.example", "hotmail.example", "outlook.example", "icloud.example"]
SPEND = ["none", "under £5k", "£5k-£25k", "£25k-£100k", "£100k+"]
CRMS = ["HubSpot", "Salesforce", "Pipedrive", "Spreadsheet", "No CRM"]


@dataclass
class Config:
    seed: int = 20260924
    as_of: str = "2026-09-24"
    n: int = 10000
    n_companies: int = 2000
    dup_share: float = 0.03

    # hidden close log-odds (ground_truth.md "Planted signals")
    intercept: float = -3.12
    noise_sd: float = 0.5
    band_eff: dict = field(default_factory=lambda: {"none": 0.0, "1-10": 0.0, "11-50": 0.5, "51-200": 1.3,
                                                    "201-1000": 1.3, "1000+": 1.0})
    free_eff: float = -1.3
    text_eff: dict = field(default_factory=lambda: {"specific": 0.75, "neutral": 0.0, "vague": -0.75,
                                                    "copy_paste": -0.75})
    top_eff: dict = field(default_factory=lambda: {"<15s": -1.5, "15-60s": 0.0, "60-300s": 0.4,
                                                   "300-600s": 0.0, ">600s": -0.4})
    hes_eff: float = -0.4
    edits_eff: float = 0.2
    form_eff: dict = field(default_factory=lambda: {"A": 0.0, "B": 0.2, "C": 0.7, "D": -0.3})
    channel_eff: dict = field(default_factory=lambda: {"meta": -0.4, "google": 0.2, "linkedin": 0.2,
                                                       "chatgpt": 0.0, "organic_direct": 0.1})
    lead_ads_eff: float = -0.2
    seniority_eff: dict = field(default_factory=lambda: {"senior": 0.3, "mid": 0.1, "junior": 0.0,
                                                         "student": -1.2})
    pricing_eff: float = 0.5
    sessions3_eff: float = 0.3
    brand_eff: float = 0.6
    bizhours_eff: float = 0.25
    ip_mismatch_eff: float = -0.3
    spend_eff: dict = field(default_factory=lambda: {"none": -0.7, "under £5k": -0.2, "£5k-£25k": 0.2,
                                                     "£25k-£100k": 0.5, "£100k+": 0.5})
    crm_eff: float = 0.3
    hiring_eff: float = 0.2
    reply_fast_eff: float = 0.4    # first reply under 1 hour
    reply_slow_eff: float = -0.4   # first reply over 48 hours

    # deal value
    deal_base: dict = field(default_factory=lambda: {"none": 900, "1-10": 1500, "11-50": 3500, "51-200": 8000,
                                                     "201-1000": 15000, "1000+": 24000})
    deal_sector: dict = field(default_factory=lambda: {"Software": 1.3, "Logistics": 1.1, "Healthcare": 1.2,
                                                       "Financial services": 1.4, "Manufacturing": 1.0,
                                                       "Retail": 0.8, "Education": 0.6,
                                                       "Professional services": 0.9})
    deal_channel: dict = field(default_factory=lambda: {"meta": 0.8, "google": 1.25, "linkedin": 1.35,
                                                        "chatgpt": 1.0, "organic_direct": 1.0})
    deal_spend: dict = field(default_factory=lambda: {"none": 0.8, "under £5k": 0.9, "£5k-£25k": 1.0,
                                                      "£25k-£100k": 1.2, "£100k+": 1.4})
    deal_noise_sd: float = 0.45

    # time to close
    won_median_days: float = 40.0
    won_sd: float = 1.09
    won_1000_mult: float = 1.8
    lost_median_days: float = 18.0
    lost_sd: float = 0.9

    # sales process
    stall_share: float = 0.11
    won_blank_value_share: float = 0.08
    messy_stage_share: float = 0.15
    click_id_missing_share: float = 0.30

    # Phase 5.2-5.8 options (v1 behaviour = all off; scripts/generate_data_v2.py turns them on). Every option
    # draws from its own RNG stream, so switching one on never shifts the random draws of a v1 stage.
    text_paraphrase: bool = field(default=False, metadata={"v2": "5.2"})         # Haiku paraphrases (cache)
    text_typo_share: float = field(default=0.0, metadata={"v2": "5.2"})          # share of texts with typos
    text_language_mix_share: float = field(default=0.0, metadata={"v2": "5.2"})  # share of DE/FR/NL texts
    text_boilerplate_pool: bool = field(default=False, metadata={"v2": "5.2"})   # copy-paste from a wide pool
    enrichment_dropout: float = field(default=0.0, metadata={"v2": "5.3"})
    consent_missing_share: float = field(default=0.0, metadata={"v2": "5.4"})
    interactions: bool = field(default=False, metadata={"v2": "5.5"})
    ghosting_follows_tier: bool = field(default=False, metadata={"v2": "5.6"})
    context_only_signal: bool = field(default=False, metadata={"v2": "5.7"})
    fast_human_share: float = field(default=0.0, metadata={"v2": "5.8"})

    # Parameters of the v2 options (ignored while the matching option is off)
    typo_char_rate: float = 0.015             # 5.2: per-letter typo probability in a text chosen for typos
    company_name_variation_share: float = 0.7  # 5.3: typed company names given suffix/case/punctuation noise
    linkedin_51plus_eff: float = 0.8          # 5.5: LinkedIn lead from a company with 51+ employees
    senior_guide_eff: float = -0.8            # 5.5: senior title on form D (guide download)
    ghost_intercept: float = -5.3             # 5.6: never-contacted log-odds = intercept + tier term only
    ghost_tier_eff: dict = field(default_factory=lambda: {"A": 0.0, "B": 2.8, "C": 4.4})
    stall_by_tier: dict = field(default_factory=lambda: {"A": 0.05, "B": 0.10, "C": 0.17})
    persona_share: float = 0.08               # 5.7: share of genuine-human original leads with a persona
    persona_eff: float = -1.0                 # 5.7: persona effect on close log-odds

    @property
    def as_of_dt(self) -> datetime:
        """``as_of`` as a UTC datetime (midnight at the start of that day)."""
        return datetime.fromisoformat(self.as_of).replace(tzinfo=timezone.utc)


# Phase 5.2-5.8 options and their v1 (off) values, read from the Config field metadata.
V2_OPTIONS = {f.name: f.default for f in fields(Config) if "v2" in f.metadata}


def v2_on(cfg: Config) -> list[str]:
    """Names of the Phase 5.2-5.8 options that are switched on in ``cfg`` (empty list = pure v1)."""
    return [k for k, off in V2_OPTIONS.items() if getattr(cfg, k) != off]


def _present(x: Any) -> bool:
    """True unless ``x`` is None or NaN (pandas turns missing values into NaN in object/float columns)."""
    return x is not None and not (isinstance(x, float) and x != x) and not (x is pd.NA)


def _flag(x: Any) -> bool:
    """Truthiness that works for Python bools, numpy bools, pandas BooleanDtype and missing values."""
    return bool(x) if _present(x) else False


def rng_for(cfg: Config, stage: str) -> np.random.Generator:
    """Independent, reproducible RNG stream per stage (stable when other stages change)."""
    return np.random.default_rng([cfg.seed, zlib.crc32(stage.encode())])


def pick(rng: np.random.Generator, options: Sequence, p: Sequence[float] | None = None, size: int | None = None) -> Any:
    """Choose one element of ``options`` (or a list of ``size`` elements), optionally weighted by ``p``."""
    idx = rng.choice(len(options), p=None if p is None else np.asarray(p, float) / np.sum(p), size=size)
    if size is None:
        return options[idx]
    return [options[i] for i in idx]


# ---------------------------------------------------------------------------------------------
# Static vocabularies
# ---------------------------------------------------------------------------------------------

FIRST_NAMES = ["Ethan", "Jack", "Ruby", "Samuel", "Maya", "Oliver", "Mia", "Ben", "Emily", "Noah", "Zara", "Harry",
               "Fatima", "Jonas", "Lena", "Daniel", "Isla", "Arjun", "Sanne", "Priya", "Oscar", "Leo", "Chloe",
               "Grace", "Amelia", "Omar", "Ava", "James", "Freya", "George", "Daan", "Megan", "Hannah", "Adam",
               "Hugo", "Tyler", "Sophie", "Olivia", "Lucas", "Camille"]
LAST_NAMES = ["Weber", "Scott", "Taylor", "Hughes", "Anderson", "Williams", "Garcia", "Moore", "de Vries", "Walker",
              "Davies", "Baker", "Evans", "Khan", "Schmidt", "Muller", "Jansen", "King", "Lewis", "Bakker", "Green",
              "Hall", "Clarke", "Wilson", "Young", "White", "Murphy", "Thompson", "Wright", "Miller", "Johnson",
              "Patel", "Jones", "Wood", "Martin", "Brown", "Smith", "Dubois", "Roberts", "Bernard"]
COMPANY_WORDS = ["Alder", "Ashford", "Aurora", "Beacon", "Bluebell", "Bramble", "Brightwater", "Cedar", "Clearwater",
                 "Cobalt", "Dovecote", "Driftwood", "Dunmore", "Elmstead", "Emberly", "Evergreen", "Falcon",
                 "Fernhill", "Foxglove", "Glenmoor", "Granite", "Greystone", "Harbour", "Hawthorn", "Heron",
                 "Highgate", "Inkwell", "Ironbridge", "Ivywood", "Jasper", "Juniper", "Kelvin", "Kestrel",
                 "Keystone", "Kingsway", "Lantern", "Larkspur", "Lindenwood", "Marigold", "Meridian", "Moorland",
                 "Nettlefield", "Nimbus", "Norfolk", "Northwind", "Oakridge", "Oldfield", "Orchard", "Pebble",
                 "Pennine", "Pinecrest", "Quartz", "Quayside", "Redwood", "Riverside", "Rosewood", "Saltmarsh",
                 "Silverline", "Stonebridge", "Tamarind", "Thornbury", "Tidewater", "Umber", "Upland", "Valley",
                 "Vantage", "Westbrook", "Willow", "Yarrow", "Zephyr"]
SECTOR_WORDS = {
    "Education": ["Academy", "College", "Education", "Learning", "Training", "Tutors"],
    "Financial services": ["Capital", "Finance", "Insurance", "Lending", "Partners", "Wealth"],
    "Healthcare": ["Care", "Clinics", "Dental", "Health", "Medical", "Pharma"],
    "Logistics": ["Couriers", "Freight", "Haulage", "Logistics", "Shipping", "Transport"],
    "Manufacturing": ["Components", "Engineering", "Fabrication", "Industries", "Manufacturing", "Works"],
    "Professional services": ["Accountants", "Advisory", "Associates", "Consulting", "Legal", "Recruitment"],
    "Retail": ["Goods", "Market", "Outfitters", "Retail", "Stores", "Trading"],
    "Software": ["Cloud", "Data", "Digital", "Labs", "Software", "Systems"],
}
SUFFIX = {"UK": "Ltd", "US": "Inc", "DE": "GmbH", "FR": "SAS", "NL": "BV"}
CITIES = {"UK": ["London", "Manchester", "Birmingham", "Leeds", "Bristol", "Edinburgh"],
          "US": ["New York", "San Francisco", "Chicago", "Austin", "Boston", "Seattle"],
          "DE": ["Berlin", "Munich", "Hamburg", "Frankfurt"],
          "FR": ["Paris", "Lyon", "Toulouse", "Lille"],
          "NL": ["Amsterdam", "Rotterdam", "Utrecht", "Eindhoven"]}
REGION = {"London": "London", "Manchester": "North West", "Birmingham": "West Midlands", "Leeds": "Yorkshire",
          "Bristol": "South West", "Edinburgh": "Scotland", "New York": "New York", "San Francisco": "California",
          "Chicago": "Illinois", "Austin": "Texas", "Boston": "Massachusetts", "Seattle": "Washington",
          "Berlin": "Berlin", "Munich": "Bavaria", "Hamburg": "Hamburg", "Frankfurt": "Hesse",
          "Paris": "Ile-de-France", "Lyon": "Auvergne-Rhone-Alpes", "Toulouse": "Occitanie",
          "Lille": "Hauts-de-France", "Amsterdam": "North Holland", "Rotterdam": "South Holland",
          "Utrecht": "Utrecht", "Eindhoven": "North Brabant"}
TZ_BY_CITY_US = {"New York": "America/New_York", "Boston": "America/New_York",
                 "San Francisco": "America/Los_Angeles", "Seattle": "America/Los_Angeles",
                 "Chicago": "America/Chicago", "Austin": "America/Chicago"}
TZ = {"UK": "Europe/London", "DE": "Europe/Berlin", "FR": "Europe/Paris", "NL": "Europe/Amsterdam"}
LANG = {"UK": "en-GB", "US": "en-US", "DE": "de-DE", "FR": "fr-FR", "NL": "nl-NL"}

TITLES = {"junior": ["Analyst", "Assistant", "Marketing Coordinator", "Marketing Executive", "Sales Rep"],
          "mid": ["Demand Gen Manager", "Head of Growth", "Head of Marketing", "Marketing Manager",
                  "Operations Manager", "Sales Manager"],
          "senior": ["CEO", "Chief Revenue Officer", "Director of Sales", "Founder", "Managing Director",
                     "Marketing Director", "VP Marketing"],
          "student": ["Student"]}
ALL_TITLES = [t for v in TITLES.values() for t in v]
DEPARTMENT = {"Analyst": "Operations", "Assistant": "Operations", "Marketing Coordinator": "Marketing",
              "Marketing Executive": "Marketing", "Sales Rep": "Sales", "Demand Gen Manager": "Marketing",
              "Head of Growth": "Marketing", "Head of Marketing": "Marketing", "Marketing Manager": "Marketing",
              "Operations Manager": "Operations", "Sales Manager": "Sales", "CEO": "Executive",
              "Chief Revenue Officer": "Executive", "Director of Sales": "Sales", "Founder": "Executive",
              "Managing Director": "Executive", "Marketing Director": "Marketing", "VP Marketing": "Marketing",
              "Student": "Student"}
VENDOR_SENIORITY = {"senior": "Director or above", "mid": "Manager", "junior": "Individual contributor",
                    "student": "Student"}

NEUTRAL_TEXTS = ["Interested in better lead quality from ads.", "Improve how we report on paid social.",
                 "Want to understand which campaigns bring good leads.", "Trying to get more from our LinkedIn spend.",
                 "Looking at options to connect our CRM to Google Ads.", "Our sales team says the leads from Meta are poor."]
VAGUE_TEXTS = ["pricing", "more info pls", "", "?", "N/A", "interested", "tell me more", "hi", "test", "info"]
COPY_PASTE_TEXTS = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore.",
    "We are a leading provider of innovative solutions dedicated to delivering best-in-class value to our "
    "stakeholders across all verticals through synergy and excellence.",
    "Our mission is to empower businesses worldwide with cutting-edge, scalable, end-to-end solutions that drive "
    "growth, efficiency and customer delight."]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
TEAM_RANGE = {"none": (1, 3), "1-10": (2, 8), "11-50": (5, 30), "51-200": (10, 60), "201-1000": (20, 150),
              "1000+": (50, 400)}

UA = {"windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36",
      "mac": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
      "iphone": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
      "android": "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Mobile Safari/537.36",
      "ipad": "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
      "headless": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/120.0 Safari/537.36"}
REFERRER = {"google": "https://www.google.com/", "linkedin": "https://www.linkedin.com/",
            "facebook": "https://l.facebook.com/", "instagram": "https://l.instagram.com/",
            "chatgpt": "https://chatgpt.com/"}
CAMPAIGNS = {"facebook": ["guide_download", "prospecting_lookalike", "retargeting_30d", "uk_leadgen_q"],
             "instagram": ["guide_download", "prospecting_lookalike", "retargeting_30d", "uk_leadgen_q"],
             "google": ["attribution_software", "brand_search", "competitor_terms", "generic_lead_scoring"],
             "linkedin": ["abm_mid_market", "job_title_marketing_leaders", "thought_leadership"],
             "chatgpt": ["chatgpt_ads_pilot", "chatgpt_conversational"]}
BRAND_TERMS = ["northstar analytics", "northstar pricing", "northstar analytics reviews"]
GENERIC_TERMS = ["b2b lead quality", "crm to google ads integration", "lead scoring software", "value based bidding",
                 "offline conversion tracking", "marketing attribution tool"]
BOT_IPS = ["203.0.113.66", "203.0.113.67", "198.51.100.200", "192.0.2.251", "192.0.2.250"]
BOT_IP_COUNTRIES = ["DE", "NL", "RU", "SG", "US"]
ALNUM = np.array(list("abcdefghijklmnopqrstuvwxyz0123456789"))
B64 = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"))
HEX = np.array(list("0123456789abcdef"))

STATUS_QUO_RULES = {
    "description": "Rule-based bucket values a typical advertiser sends today. EMVA's benchmark.",
    "currency": "GBP",
    "base_value_by_form": {"A": 150, "B": 300, "C": 600, "D": 200},
    "value_rules": [
        {"name": "UK or US lead", "field": "country", "in": ["UK", "US"], "multiplier": 1.2},
        {"name": "LinkedIn traffic", "field": "channel", "in": ["linkedin"], "multiplier": 1.3}],
    "lead_score": {
        "job_title": {
            "keywords": [{"keyword": "student", "points": 0}, {"keyword": "ceo", "points": 30},
                         {"keyword": "founder", "points": 30}, {"keyword": "chief", "points": 30},
                         {"keyword": "vp", "points": 30}, {"keyword": "director", "points": 30},
                         {"keyword": "head", "points": 25}, {"keyword": "manager", "points": 15}],
            "default_points": 10, "missing_points": 0},
        "company_size_points": {"1-10": 5, "11-50": 10, "51-200": 20, "201-1000": 25, "1000+": 30},
        "pages_visited_points": [{"min_pages": 5, "points": 20}, {"min_pages": 3, "points": 10},
                                 {"min_pages": 1, "points": 5}],
        "tiers": [{"name": "A", "min_points": 55, "multiplier": 2.0}, {"name": "B", "min_points": 30, "multiplier": 1.0},
                  {"name": "C", "min_points": 0, "multiplier": 0.4}]},
    "stage_values": {"Qualified": 1000, "Demo booked": 2000, "Proposal": 4000, "Lost": 0},
    "won_value": "deal_value",
}

LEAD_COLUMNS = ["lead_id", "created_at", "form_variant", "answers", "email", "phone", "name", "company_name",
                "company_domain", "consent", "utm_source", "utm_medium", "utm_campaign", "utm_content", "fbclid", "fbc",
                "fbp", "gclid", "gbraid", "wbraid", "li_fat_id", "oppref", "meta_lead_id", "landing_url",
                "time_on_page_s", "fields_edited", "field_edit_count", "hesitation_ms", "pasted_text",
                "returned_visitor", "device", "ip", "user_agent", "pages_visited", "utm_term", "referrer",
                "ip_country", "ip_city", "is_datacenter_ip", "timezone", "browser_language", "local_submit_hour",
                "submitted_weekday", "sessions_before_convert", "days_since_first_visit", "viewed_pricing",
                "scroll_depth_pct"]
LABEL_COLUMNS = ["lead_id", "duplicate_of", "outcome", "p_close_true", "channel", "is_lead_ads", "is_bot",
                 "is_free_email", "has_company", "employee_band", "sector", "company_domain", "seniority",
                 "text_category", "country", "click_id_missing", "never_contacted", "stalled", "deal_value",
                 "reply_hours"]
# Hidden truth added by the v2 options (only written when at least one is on, so v1 keeps its schema).
V2_LABEL_COLUMNS = ["context_persona", "consent_declined", "fast_human", "domain_in_companies", "language_mix"]
INT_COLUMNS = ["field_edit_count", "hesitation_ms", "pages_visited", "sessions_before_convert",
               "days_since_first_visit", "scroll_depth_pct"]


def rand_str(rng: np.random.Generator, alphabet: np.ndarray, n: int) -> str:
    """Random string of ``n`` characters drawn from ``alphabet``."""
    return "".join(rng.choice(alphabet, n))


# ---------------------------------------------------------------------------------------------
# Stage 1: companies
# ---------------------------------------------------------------------------------------------

def make_companies(cfg: Config) -> pd.DataFrame:
    """companies.csv: firmographics for n_companies fake companies (band drives revenue, funding, CRM, spend, hiring)."""
    rng = rng_for(cfg, "companies")
    band_p = [0.263, 0.286, 0.2145, 0.155, 0.0815]
    country_p = [0.34, 0.296, 0.125, 0.1325, 0.1065]
    revenue = {"1-10": (["under £1m", "£1m-£10m"], [0.91, 0.09]),
               "11-50": (["under £1m", "£1m-£10m", "£10m-£50m"], [0.07, 0.85, 0.08]),
               "51-200": (["£1m-£10m", "£10m-£50m", "£50m-£250m"], [0.05, 0.87, 0.08]),
               "201-1000": (["£10m-£50m", "£50m-£250m", "£250m+"], [0.065, 0.85, 0.085]),
               "1000+": (["£50m-£250m", "£250m+"], [0.07, 0.93])}
    funding_levels = ["Bootstrapped", "Seed", "Series A", "Series B", "Series C+", "Private equity", "Public"]
    funding = {"1-10": [293, 204, 29, 0, 0, 0, 0], "11-50": [268, 153, 132, 19, 0, 0, 0],
               "51-200": [149, 33, 102, 107, 17, 21, 0], "201-1000": [80, 0, 15, 72, 86, 39, 18],
               "1000+": [33, 0, 0, 9, 26, 42, 53]}
    spend = {"1-10": [184, 253, 71, 18, 0], "11-50": [117, 230, 169, 56, 0], "51-200": [48, 89, 182, 87, 23],
             "201-1000": [20, 24, 107, 107, 52], "1000+": [4, 7, 39, 61, 52]}
    crm = {"1-10": [151, 27, 87, 173, 88], "11-50": [198, 88, 93, 130, 63], "51-200": [153, 143, 63, 48, 22],
           "201-1000": [68, 177, 29, 22, 14], "1000+": [21, 129, 5, 2, 6]}
    hiring_p = {"1-10": 0.211, "11-50": 0.383, "51-200": 0.497, "201-1000": 0.61, "1000+": 0.755}
    roles_lambda = {"1-10": 1.8, "11-50": 4.0, "51-200": 9.0, "201-1000": 20.7, "1000+": 63.4}
    platforms = ["Google Ads", "Meta Ads", "LinkedIn Ads", "Microsoft Ads", "TikTok Ads"]
    platform_w = np.array([0.632, 0.564, 0.364, 0.283, 0.144])

    used, rows = set(), []
    while len(rows) < cfg.n_companies:
        sector = pick(rng, SECTORS)
        w1, w2 = pick(rng, COMPANY_WORDS), pick(rng, SECTOR_WORDS[sector])
        if (w1, w2) in used:
            continue
        used.add((w1, w2))
        country = pick(rng, COUNTRIES, country_p)
        band = pick(rng, BANDS, band_p)
        rev_levels, rev_p = revenue[band]
        sp = pick(rng, SPEND, spend[band])
        n_plat = SPEND.index(sp)
        chosen = sorted(rng.choice(platforms, size=n_plat, replace=False, p=platform_w / platform_w.sum())) if n_plat else []
        hiring = bool(rng.random() < hiring_p[band])
        small = band in ("1-10", "11-50", "51-200")
        rows.append({
            "company_name": f"{w1} {w2} {SUFFIX[country]}",
            "domain": f"{w1.lower()}-{w2.lower()}.example",
            "sector": sector, "employee_band": band, "country": country,
            "hq_city": pick(rng, CITIES[country]),
            "annual_revenue_band": pick(rng, rev_levels, rev_p),
            "founded_year": int(rng.integers(1995 if small else 1965, 2025)),
            "funding_stage": pick(rng, funding_levels, funding[band]),
            "crm_platform": pick(rng, CRMS, crm[band]),
            "monthly_ad_spend_band": sp,
            "ad_platforms": json.dumps([str(p) for p in chosen]),
            "is_hiring": hiring,
            "open_roles": int(max(1, rng.poisson(roles_lambda[band]))) if hiring else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Stage 2: people (the persona behind each original lead)
# ---------------------------------------------------------------------------------------------

CHANNEL_P = [0.404, 0.294, 0.155, 0.050, 0.097]      # meta, google, linkedin, chatgpt, organic_direct
LEAD_ADS_SHARE_OF_META = 0.239
BOT_RATE = {"meta": 0.079, "google": 0.028, "linkedin": 0.022, "chatgpt": 0.036, "organic_direct": 0.012}
FREE_RATE = {"meta": 0.446, "lead_ads": 0.473, "google": 0.255, "linkedin": 0.097, "chatgpt": 0.216,
             "organic_direct": 0.318}
FREE_HAS_COMPANY = 0.60
SENIORITY_P = {"business": [0.345, 0.393, 0.240, 0.022], "free_company": [0.301, 0.329, 0.215, 0.154],
               "free_none": [0.291, 0.356, 0.216, 0.137], "bot": [0.30, 0.33, 0.27, 0.10]}
SENIORITIES = ["junior", "mid", "senior", "student"]
COUNTRY_P = [0.3426, 0.2998, 0.119, 0.1288, 0.1098]


def make_people(cfg: Config, companies: pd.DataFrame, n_orig: int) -> pd.DataFrame:
    """One row per original lead: who they are, independent of the session that brought them in."""
    rng = rng_for(cfg, "people")
    co = companies.reset_index(drop=True)
    by_country = {c: co.index[co.country == c].to_numpy() for c in COUNTRIES}
    used_emails, rows = set(), []
    for i in range(n_orig):
        ch_i = rng.choice(5, p=CHANNEL_P)
        channel = CHANNELS[ch_i]
        lead_ads = channel == "meta" and rng.random() < LEAD_ADS_SHARE_OF_META
        bot = (not lead_ads) and rng.random() < BOT_RATE[channel]
        country = pick(rng, COUNTRIES, COUNTRY_P)
        city = pick(rng, CITIES[country])
        first, last = pick(rng, FIRST_NAMES), pick(rng, LAST_NAMES)
        r = {"channel": channel, "is_lead_ads": lead_ads, "is_bot": bot, "country": country, "city": city,
             "first": first, "last": last}
        if bot:
            r["is_free_email"] = bool(rng.random() < 0.70)
            r["has_company"] = False
            r["company_idx"] = -1
            r["seniority"] = pick(rng, SENIORITIES, SENIORITY_P["bot"])
            name_kind = rng.random()
            r["name"] = rand_str(rng, ALNUM[:26], 6) if name_kind < 0.265 else pick(rng, ["asdf asdf", "Test Test", "John Doe"], [115, 96, 75])
            r["email"] = f"{rand_str(rng, ALNUM, 8)}@{pick(rng, FREE_DOMAINS)}"
            r["bot_fake_domain"] = None if r["is_free_email"] else \
                f"{pick(rng, COMPANY_WORDS).lower()}-{pick(rng, [w for ws in SECTOR_WORDS.values() for w in ws]).lower()}.example"
        else:
            free = bool(rng.random() < FREE_RATE["lead_ads" if lead_ads else channel])
            has_co = (not free) or rng.random() < FREE_HAS_COMPANY
            r["is_free_email"], r["has_company"] = free, has_co
            r["company_idx"] = int(rng.choice(by_country[country])) if has_co else -1
            key = "business" if not free else ("free_company" if has_co else "free_none")
            r["seniority"] = pick(rng, SENIORITIES, SENIORITY_P[key])
            r["name"] = f"{first} {last}"
            fl, ll = first.lower(), last.lower().replace(" ", "")
            if not free:
                dom = co.domain[r["company_idx"]]
                local = pick(rng, [f"{fl}.{ll}", f"{fl[0]}.{ll}", fl])
            else:
                dom = pick(rng, FREE_DOMAINS)
                local = pick(rng, [f"{fl}.{ll}", f"{fl}{ll}{rng.integers(1, 100):02d}", f"{fl[0]}{ll}{rng.integers(0, 1000):03d}"])
            email = f"{local}@{dom}"
            while email in used_emails:
                email = f"{local}{rng.integers(1, 100)}@{dom}"
            r["email"] = email
            r["bot_fake_domain"] = None
        used_emails.add(r["email"])
        r["job_title"] = pick(rng, TITLES[r["seniority"]])
        band = co.employee_band[r["company_idx"]] if r["company_idx"] >= 0 else "none"
        r["employee_band"] = band
        lo, hi = TEAM_RANGE[band]
        r["team_size"] = int(rng.integers(lo, hi + 1))
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Stage 3: leads (form, answers skeleton, timestamps, attribution)
# ---------------------------------------------------------------------------------------------

WEB_VARIANT_P = [0.357, 0.296, 0.10, 0.247]


def lead_timezone(country: str, city: str) -> str:
    """IANA timezone of a lead: by city in the US, by country elsewhere."""
    return TZ_BY_CITY_US[city] if country == "US" else TZ[country]


def draw_timestamp(rng: np.random.Generator, cfg: Config, tzname: str, bot: bool) -> tuple[datetime, datetime]:
    """Local submit time: humans mostly weekday office hours, weekend traffic partly shifted to Friday.

    Returns (UTC time, local time) within the year before ``as_of``."""
    as_of = cfg.as_of_dt
    lo, hi = as_of - timedelta(days=365), as_of
    tz = ZoneInfo(tzname)
    base = lo + timedelta(seconds=float(rng.uniform(0, (hi - lo).total_seconds())))
    local = base.astimezone(tz)
    day = local.date()
    if not bot:
        if day.weekday() >= 5 and rng.random() < 0.70:
            day = day - timedelta(days=day.weekday() - 4)
        p_office = 0.852 if day.weekday() < 5 else 0.395
        hour = int(rng.integers(9, 18)) if rng.random() < p_office else int(pick(rng, [h for h in range(24) if not 9 <= h <= 17]))
    else:
        hour = int(rng.integers(0, 24))
    local = datetime(day.year, day.month, day.day, hour, int(rng.integers(0, 60)), int(rng.integers(0, 60)), tzinfo=tz)
    utc = local.astimezone(timezone.utc)
    cap = as_of - timedelta(hours=1)
    if utc > cap:
        utc = cap
        local = utc.astimezone(tz)
    if utc < as_of - timedelta(days=365):
        utc = as_of - timedelta(days=365) + timedelta(seconds=float(rng.uniform(0, 3600)))
        local = utc.astimezone(tz)
    return utc, local


def typed_company(rng: np.random.Generator, name: str) -> str:
    """Company name as typed into the form: half drop the legal suffix, 10% lowercase."""
    if rng.random() < 0.5:
        name = name.rsplit(" ", 1)[0]
    if rng.random() < 0.1:
        name = name.lower()
    return name


SUFFIX_VARIANTS = {"Ltd": ["Ltd", "Ltd.", "LTD", "Limited", "ltd"], "Inc": ["Inc", "Inc.", "INC", "Incorporated", "inc."],
                   "GmbH": ["GmbH", "Gmbh", "GMBH", "G.m.b.H."], "SAS": ["SAS", "S.A.S.", "sas", "S.A.S"],
                   "BV": ["BV", "B.V.", "bv", "B.V"]}


def vary_company_name(rng: np.random.Generator, typed: str, legal_suffix: str) -> str:
    """5.3: typing noise on a company name that a normalising matcher can still undo.

    Suffix spelling (Ltd / Ltd. / LTD / Limited, Inc. / Incorporated, Gmbh, B.V., ...), sometimes a comma before
    it or a suffix added where the user left it off; an ampersand or hyphen between the two name words; upper or
    lower case; stray leading, trailing or doubled spaces."""
    words = typed.split(" ")
    has_suffix = words[-1].lower() == legal_suffix.lower()
    stem = words[:-1] if has_suffix else words
    if has_suffix or rng.random() < 0.3:
        suffix = pick(rng, SUFFIX_VARIANTS[legal_suffix])
        if rng.random() < 0.2:
            stem = stem[:-1] + [stem[-1] + ","]
        name_words = stem + [suffix]
    else:
        name_words = stem
    u = rng.random()
    joiner = " & " if u < 0.10 else "-" if u < 0.16 else " "
    name = joiner.join(name_words[:2]) + ("" if len(name_words) <= 2 else " " + " ".join(name_words[2:]))
    u = rng.random()
    name = name.upper() if u < 0.10 else name.lower() if u < 0.25 else name
    u = rng.random()
    if u < 0.15:
        name += " " * int(rng.integers(1, 3))
    elif u < 0.20:
        name = " " + name
    elif u < 0.28:
        name = name.replace(" ", "  ", 1)
    return name


def vary_company_names(cfg: Config, leads: pd.DataFrame, people: pd.DataFrame, companies: pd.DataFrame) -> pd.DataFrame:
    """5.3: apply ``vary_company_name`` to ``company_name_variation_share`` of typed names of real companies."""
    if cfg.enrichment_dropout <= 0:
        return leads
    rng = rng_for(cfg, "v2_company_names")
    for i, l in leads.iterrows():
        k = people.at[l.person, "company_idx"]
        typed = l.answers.get("company")
        if k < 0 or not typed or rng.random() >= cfg.company_name_variation_share:
            continue
        legal = SUFFIX[companies.at[k, "country"]]
        ans = dict(l.answers)
        ans["company"] = vary_company_name(rng, typed, legal)
        leads.at[i, "answers"] = ans
        leads.at[i, "company_name"] = ans["company"]
    return leads


def drop_enrichment(cfg: Config, companies: pd.DataFrame, people: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """5.3: hide ``enrichment_dropout`` of the business-email domains from the emitted companies.csv.

    The company row stays (so a typed-name match can still find it) but carries a legacy domain that no lead's
    email uses. Returns (emitted companies.csv, hidden full table with ``emitted_domain`` and
    ``domain_in_companies_csv``)."""
    rng = rng_for(cfg, "v2_dropout")
    used = sorted(set(people.company_idx[(~people.is_free_email) & (~people.is_bot) & (people.company_idx >= 0)]))
    n_drop = int(round(cfg.enrichment_dropout * len(used)))
    drop = set(int(k) for k in rng.choice(used, size=n_drop, replace=False))
    emitted = companies.copy()
    for k in sorted(drop):
        w1, w2 = companies.at[k, "domain"].removesuffix(".example").split("-")
        emitted.at[k, "domain"] = f"{w1}{w2[:4]}-group.example"
    if emitted.domain.duplicated().any():
        raise ValueError("legacy domains collide; change the legacy-domain rule in drop_enrichment")
    hidden = companies.copy()
    hidden["emitted_domain"] = emitted.domain
    hidden["domain_in_companies_csv"] = ~companies.index.isin(sorted(drop))
    return emitted, hidden


def typed_band(rng: np.random.Generator, band: str) -> str:
    """Company-size dropdown answer: the true band, 15% one band off; individuals answer "1-10" or blank."""
    if band == "none":
        return "1-10" if rng.random() < 0.67 else ""
    if rng.random() < 0.15:
        i = BANDS.index(band)
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(BANDS)]
        return BANDS[int(rng.choice(nb))]
    return band


def session_attribution(rng: np.random.Generator, channel: str, lead_ads: bool) -> dict[str, str | None]:
    """utm fields for one session (source, medium, campaign, content)."""
    if channel == "organic_direct":
        return {"utm_source": None, "utm_medium": None, "utm_campaign": None, "utm_content": None}
    if channel == "meta":
        src = "facebook" if rng.random() < 0.69 else "instagram"
        med = "lead_form" if lead_ads else "paid_social"
    elif channel == "google":
        src, med = "google", "cpc"
    elif channel == "linkedin":
        src, med = "linkedin", "paid_social"
    else:
        src, med = "chatgpt", "cpc"
    return {"utm_source": src, "utm_medium": med, "utm_campaign": pick(rng, CAMPAIGNS[src]),
            "utm_content": f"ad_{int(rng.integers(1, 13)):02d}"}


def make_leads(cfg: Config, people: pd.DataFrame, companies: pd.DataFrame) -> pd.DataFrame:
    """One lead per persona: submit time, form variant, UTM attribution, answers skeleton, phone, consent."""
    rng = rng_for(cfg, "leads")
    rows = []
    for i, p in people.iterrows():
        tzname = lead_timezone(p.country, p.city)
        utc, local = draw_timestamp(rng, cfg, tzname, p.is_bot)
        variant = "A" if p.is_lead_ads else pick(rng, ["A", "B", "C", "D"], WEB_VARIANT_P)
        r = {"person": i, "created": utc, "local_submit_hour": local.hour, "submitted_weekday": local.strftime("%A"),
             "timezone": tzname, "form_variant": variant}
        r.update(session_attribution(rng, p.channel, p.is_lead_ads))
        # answers skeleton; what_to_solve is filled by the text stage
        ans = {"country": p.country, "what_to_solve": None}
        co_name = companies.company_name[p.company_idx] if p.company_idx >= 0 else None
        if variant in ("B", "C", "D") and not p.is_lead_ads:
            if co_name is not None:
                ans["company"] = typed_company(rng, co_name)
            else:
                opts, w = (["", "n/a", "Self-employed", "Freelance"], [0.42, 0.24, 0.18, 0.16]) if p.is_bot else \
                          (["", "n/a", "Freelance", "Self-employed"], [0.41, 0.21, 0.19, 0.19])
                ans["company"] = pick(rng, opts, w)
            ans["job_title"] = p.job_title
        if p.is_lead_ads or variant in ("B", "C"):
            ans["company_size"] = typed_band(rng, p.employee_band)
        if variant == "C" and not p.is_lead_ads:
            ans["team_size"] = int(p.team_size)
            budget = {"none": (["Under £5k", "£5k-£20k"], [0.8, 0.2]),
                      "1-10": (["Under £5k", "£5k-£20k"], [0.8, 0.2]),
                      "11-50": (["Under £5k", "£5k-£20k", "£20k-£50k"], [0.22, 0.52, 0.26]),
                      "51-200": (["Under £5k", "£5k-£20k", "£20k-£50k"], [0.23, 0.47, 0.30]),
                      "201-1000": (["£5k-£20k", "£20k-£50k", "£50k+"], [0.23, 0.51, 0.26]),
                      "1000+": (["£20k-£50k", "£50k+"], [0.15, 0.85])}[p.employee_band]
            ans["budget"] = pick(rng, *budget)
            ans["timeline"] = None  # filled by the text stage (depends on text category)
        r["answers"] = ans
        r["company_name"] = ans.get("company")
        if p.is_bot:
            r["company_domain"] = p.bot_fake_domain
        else:
            r["company_domain"] = None if p.is_free_email else p.email.split("@")[1]
        # phone
        has_phone = (p.is_lead_ads and rng.random() < 0.808) or (not p.is_lead_ads and (variant == "C" or (variant == "B" and rng.random() < 0.5)))
        r["phone"] = make_phone(rng, p.country) if has_phone else None
        r["consent"] = bool(rng.random() < 0.9185)
        rows.append(r)
    return pd.DataFrame(rows)


def make_phone(rng: np.random.Generator, country: str) -> str:
    """Phone number in one of three national formats; about 4% invalid ("12345")."""
    u = rng.random()
    if u < 0.042:
        return pick(rng, ["12345", "0000 000000", "07700"], [0.45, 0.35, 0.20])
    k = int(rng.integers(0, 3))
    if country == "US":
        area = pick(rng, ["212", "415", "312", "617", "202", "206", "512"])
        tail = f"01{int(rng.integers(0, 100)):02d}"
        return [f"({area}) 555-{tail}", f"{area}.555.{tail}", f"+1 {area}-555-{tail}"][k]
    tail = f"{int(rng.integers(0, 1000)):03d}"
    return [f"+447700900{tail}", f"+44 7700 900{tail}", f"07700 900{tail}"][k]


# ---------------------------------------------------------------------------------------------
# Stage 4: behaviour (website session telemetry, click IDs)
# ---------------------------------------------------------------------------------------------

DEVICE_P = {"meta": [0.354, 0.604, 0.042], "google": [0.542, 0.419, 0.039], "linkedin": [0.597, 0.345, 0.058],
            "chatgpt": [0.600, 0.364, 0.036], "organic_direct": [0.594, 0.359, 0.047]}


def session_behaviour(rng: np.random.Generator, cfg: Config, channel: str, lead_ads: bool, bot: bool, country: str,
                      city: str, variant: str, has_phone: bool, attrib: dict[str, str | None],
                      created: datetime) -> dict[str, Any]:
    """All website telemetry for one session. Lead Ads leads have no website session."""
    b = {k: None for k in ["fbclid", "fbc", "fbp", "gclid", "gbraid", "wbraid", "li_fat_id", "oppref", "meta_lead_id",
                           "landing_url", "time_on_page_s", "fields_edited", "field_edit_count", "hesitation_ms",
                           "pasted_text", "returned_visitor", "ip", "user_agent", "pages_visited", "utm_term",
                           "referrer", "ip_country", "ip_city", "is_datacenter_ip", "browser_language",
                           "sessions_before_convert", "days_since_first_visit", "viewed_pricing", "scroll_depth_pct"]}
    b["click_id_missing"] = False
    ms = int(created.timestamp() * 1000)
    if lead_ads:
        b["device"] = "mobile" if rng.random() < 0.785 else "desktop"
        b["meta_lead_id"] = str(int(rng.integers(10 ** 14, 10 ** 16)))
        return b
    src = attrib["utm_source"]
    if bot:
        b["device"] = "desktop"
        b["user_agent"] = UA["headless"] if rng.random() < 0.51 else UA[pick(rng, ["windows", "mac"])]
        b["time_on_page_s"] = round(float(rng.uniform(2, 14)), 1)
        b["field_edit_count"], b["fields_edited"] = 0, "[]"
        b["hesitation_ms"] = int(rng.integers(0, 500))
        b["pasted_text"] = bool(rng.random() < 0.66)
        b["sessions_before_convert"], b["days_since_first_visit"], b["pages_visited"] = 1, 0, 1
        b["viewed_pricing"] = bool(rng.random() < 0.05)
        b["scroll_depth_pct"] = int(rng.integers(0, 20))
        k = int(rng.integers(0, len(BOT_IPS)))
        b["ip"], b["ip_country"], b["ip_city"], b["is_datacenter_ip"] = BOT_IPS[k], pick(rng, BOT_IP_COUNTRIES), None, True
        b["browser_language"] = "en-US"
    else:
        dev = pick(rng, ["desktop", "mobile", "tablet"], DEVICE_P[channel])
        b["device"] = dev
        b["user_agent"] = UA[pick(rng, ["windows", "mac"])] if dev == "desktop" else (
            UA["iphone" if rng.random() < 0.53 else "android"] if dev == "mobile" else UA["ipad"])
        # hesitation is part of the time on page: time = reading/typing time + pauses, floored at 16s
        b["hesitation_ms"] = int(rng.uniform(90000, 400000)) if rng.random() < 0.082 else int(np.exp(rng.normal(9.09, 0.78)))
        top = float(np.exp(rng.normal(4.54, 0.80))) + b["hesitation_ms"] / 1000
        top = max(top, 16.0)                 # Phase 5.8 hook: a share of genuine humans under 15s
        b["time_on_page_s"] = round(top, 1)
        n_edits = int(min(rng.poisson(1.8), 7))
        fields = ["name", "email", "country", "what_to_solve"]
        if variant in ("B", "C", "D"):
            fields += ["company", "job_title"]
        if variant in ("B", "C"):
            fields += ["company_size"]
        if variant == "C":
            fields += ["team_size", "budget", "timeline"]
        if has_phone:
            fields += ["phone"]
        b["field_edit_count"] = n_edits
        b["fields_edited"] = json.dumps(sorted(set(pick(rng, fields, size=n_edits)))) if n_edits else "[]"
        b["pasted_text"] = bool(rng.random() < 0.10)
        sess = 1 + int(rng.poisson(0.9))
        b["sessions_before_convert"] = sess
        b["days_since_first_visit"] = 0 if sess == 1 else int(min(120, max(1, round(np.exp(rng.normal(1.70, 1.04))))))
        b["pages_visited"] = 1 + int(rng.poisson(1.6))
        b["viewed_pricing"] = bool(rng.random() < {1: 0.255, 2: 0.352}.get(sess, 0.45))   # returning visitors compare prices
        b["scroll_depth_pct"] = int(rng.integers(30, 101))
        if rng.random() < 0.954:
            ipc = country
        else:
            ipc = pick(rng, [c for c in COUNTRIES if c != country])
        dc = ipc != country and rng.random() < 0.53
        b["ip_country"], b["is_datacenter_ip"] = ipc, dc
        if ipc == country:
            b["ip_city"] = city if rng.random() < 0.9 else pick(rng, CITIES[country])
        else:
            b["ip_city"] = pick(rng, CITIES[ipc])
        b["ip"] = f"198.51.100.{int(rng.integers(1, 255))}" if dc else \
            f"{pick(rng, ['192.0.2', '198.51.100', '203.0.113'])}.{int(rng.integers(1, 255))}"
        native = LANG[country]
        if country in ("UK", "US"):
            b["browser_language"] = native if rng.random() < 0.93 else ("en-US" if country == "UK" else "en-GB")
        else:
            u = rng.random()
            b["browser_language"] = native if u < 0.85 else ("en-GB" if u < 0.92 else "en-US")
    b["returned_visitor"] = b["sessions_before_convert"] > 1
    # attribution-specific fields
    if src is None:
        b["referrer"] = pick(rng, [None, "https://www.google.com/", "https://www.bing.com/", "https://news.example/best-tools"],
                             [0.40, 0.20, 0.22, 0.18])
    else:
        b["referrer"] = REFERRER[src]
    if src == "google":
        b["utm_term"] = pick(rng, BRAND_TERMS) if rng.random() < 0.247 else pick(rng, GENERIC_TERMS)
    if rng.random() < 0.80:
        b["fbp"] = f"fb.1.{ms - int(rng.uniform(1, 30) * 86400000)}.{int(rng.integers(0, 10 ** 10)):010d}"
    if src is not None:
        if rng.random() < cfg.click_id_missing_share:
            b["click_id_missing"] = True
        elif src in ("facebook", "instagram"):
            b["fbclid"] = "IwAR" + rand_str(rng, B64, 40)
            b["fbc"] = f"fb.1.{ms - int(rng.uniform(0.5, 15) * 60000)}.{b['fbclid']}"
        elif src == "google":
            iphone = b["user_agent"] == UA["iphone"]
            if iphone and rng.random() < 0.375:
                b["gbraid" if rng.random() < 0.45 else "wbraid"] = "0AAAAA" + rand_str(rng, B64, 30)
            else:
                b["gclid"] = "Cj0KCQ" + rand_str(rng, B64, 60)
        elif src == "linkedin":
            b["li_fat_id"] = f"{rand_str(rng, HEX, 8)}-{rand_str(rng, HEX, 4)}-{rand_str(rng, HEX, 12)}"
        elif src == "chatgpt":
            b["oppref"] = "opp_" + rand_str(rng, B64, 24)
    q = [("variant", variant)] + [(k, attrib[k]) for k in ["utm_source", "utm_medium", "utm_campaign", "utm_content"] if attrib[k]]
    q += [(k, b[k]) for k in ["fbclid", "gclid", "gbraid", "wbraid", "li_fat_id", "oppref"] if b[k]]
    b["landing_url"] = "https://lp.emva-poc.example/lp.html?" + "&".join(f"{k}={v}" for k, v in q)
    return b


# Telemetry that a consent-declined visit does not record (5.4): everything set by the analytics script or
# derived from its cookies and geolocation. User agent, device, referrer, landing URL and click IDs (which
# come from the request and the URL) are kept.
CONSENT_COLUMNS = ["time_on_page_s", "hesitation_ms", "field_edit_count", "fields_edited", "pasted_text",
                   "sessions_before_convert", "days_since_first_visit", "returned_visitor", "pages_visited",
                   "viewed_pricing", "scroll_depth_pct", "fbp", "ip", "ip_country", "ip_city", "is_datacenter_ip"]
# Session values the planted close log-odds read; kept as the truth when 5.4 or 5.8 changes what is recorded.
TRUE_BEHAVIOUR = ["time_on_page_s", "hesitation_ms", "field_edit_count", "viewed_pricing", "sessions_before_convert",
                  "ip_country"]


def observe_session(cfg: Config, b: dict[str, Any], lead_ads: bool, bot: bool, crng: np.random.Generator,
                    frng: np.random.Generator) -> None:
    """Turn the true session ``b`` into what is recorded (in place): 5.4 consent declined, 5.8 fast humans.

    Adds ``consent_declined``, ``fast_human`` and ``true_behaviour`` (the TRUE_BEHAVIOUR values before any change,
    None when nothing changed). Consent is drawn for every website session (bots included: a script that never
    runs the analytics tag looks the same); fast humans are drawn among genuine humans whose telemetry was
    recorded. Fast humans get 5-14.9s on the page and a pause share of at most 25% of it, but keep the close
    odds of their real session, so the bot rule's under-15s cut is wrong for them."""
    b["consent_declined"], b["fast_human"], b["true_behaviour"] = False, False, None
    if lead_ads:
        return
    truth = {c: b[c] for c in TRUE_BEHAVIOUR}
    if cfg.consent_missing_share > 0 and crng.random() < cfg.consent_missing_share:
        for c in CONSENT_COLUMNS:
            b[c] = None
        b["consent_declined"] = True
    elif not bot and cfg.fast_human_share > 0 and frng.random() < cfg.fast_human_share:
        top = round(float(frng.uniform(5.0, 14.9)), 1)
        b["time_on_page_s"] = top
        b["hesitation_ms"] = int(frng.uniform(0.0, 0.25) * top * 1000)
        b["fast_human"] = True
    if b["consent_declined"] or b["fast_human"]:
        b["true_behaviour"] = truth


def add_behaviour(cfg: Config, leads: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Append website telemetry, click IDs, IP and device columns (``session_behaviour``) to every lead, then
    what the v2 options change about what is recorded (``observe_session``)."""
    rng = rng_for(cfg, "behaviour")
    crng, frng = rng_for(cfg, "v2_consent"), rng_for(cfg, "v2_fast_humans")
    rows = []
    for _, l in leads.iterrows():
        p = people.loc[l.person]
        attrib = {k: (l[k] if isinstance(l[k], str) else None) for k in ["utm_source", "utm_medium", "utm_campaign", "utm_content"]}
        b = session_behaviour(rng, cfg, p.channel, p.is_lead_ads, p.is_bot, p.country, p.city, l.form_variant,
                              isinstance(l.phone, str), attrib, l.created)
        observe_session(cfg, b, p.is_lead_ads, p.is_bot, crng, frng)
        rows.append(b)
    return pd.concat([leads.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------------------------------------
# Stage 5: free text (what_to_solve) and the text-dependent C answers
# ---------------------------------------------------------------------------------------------

TEXT_CATS = ["copy_paste", "neutral", "specific", "vague"]
TEXT_LOGIT = {"intercept": [-0.96, 0.73, 0.04, 0.18], "C": [-0.52, 0.05, 0.94, -0.46], "meta": [0.24, -0.19, -0.32, 0.26]}


SPECIFIC_TEMPLATES = [
    "Looking for pricing for {team} seats, want to go live within {weeks} weeks.",
    "Budget of around £{k}k approved for a new attribution setup this quarter.",
    "Our paid budget is £{k}k a month and we can't tell which leads turn into revenue.",
    "We have a team of {team} SDRs and need better lead scoring before {when}.",
    "Replacing our current tool when the contract ends in {month}. {team} users.",
    "Need to fix our ad conversion tracking by {month}, team of {team} marketers.",
]


def specific_template(rng: np.random.Generator, team: int) -> tuple[str, dict[str, Any]]:
    """A "specific" what_to_solve answer (budget, timeline or team size) as (template, placeholder values).

    ``template.format(**values)`` is the v1 text, which the POC regex matches. The draws (and their order) are
    exactly those of the original v1 ``specific_text``."""
    k = int(rng.integers(0, 6))
    if k == 0:
        return SPECIFIC_TEMPLATES[0], {"team": team, "weeks": int(rng.integers(2, 10))}
    if k in (1, 2):
        return SPECIFIC_TEMPLATES[k], {"k": int(rng.integers(3, 60))}
    if k == 3:
        when = pick(rng, ["Q1", "Q2", "Q3", "Q4", "end of quarter", "the new year"], [48, 47, 60, 60, 64, 69])
        return SPECIFIC_TEMPLATES[3], {"team": team, "when": when}
    return SPECIFIC_TEMPLATES[k], {"team": team, "month": pick(rng, MONTHS)}


def add_text(cfg: Config, leads: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Draw the text category and what_to_solve answer (plus the C-form timeline) for every lead.

    Also records the template and placeholder values behind each answer (``text_template``, ``text_params``)
    so the v2 text stage (``v2text.rewrite_texts``) can paraphrase and perturb it without new draws here."""
    rng = rng_for(cfg, "text")
    cats, templates, params = [], [], []
    for i, l in leads.iterrows():
        p = people.loc[l.person]
        if p.is_bot:
            cat = "vague" if rng.random() < 0.58 else "copy_paste"
        else:
            z = np.array(TEXT_LOGIT["intercept"]) + (l.form_variant == "C") * np.array(TEXT_LOGIT["C"]) + \
                (p.channel == "meta") * np.array(TEXT_LOGIT["meta"])
            pr = np.exp(z) / np.exp(z).sum()
            cat = TEXT_CATS[int(rng.choice(4, p=pr))]
        if cat == "specific":
            tpl, val = specific_template(rng, int(p.team_size))
        elif cat == "neutral":
            tpl, val = pick(rng, NEUTRAL_TEXTS), {}
        elif cat == "vague":
            tpl, val = pick(rng, VAGUE_TEXTS), {}
        else:
            tpl, val = pick(rng, COPY_PASTE_TEXTS), {}
        ans = dict(l.answers)
        ans["what_to_solve"] = tpl.format(**val) if val else tpl
        if "timeline" in ans:
            ans["timeline"] = pick(rng, ["This month", "This quarter", "Next 6 months"], [0.22, 0.56, 0.22]) if cat == "specific" \
                else pick(rng, ["Just researching", "This quarter", "Next 6 months"], [0.49, 0.26, 0.25])
        leads.at[i, "answers"] = ans
        cats.append(cat)
        templates.append(tpl)
        params.append(val)
    leads["text_category"] = cats
    leads["text_template"] = templates
    leads["text_params"] = params
    return leads


# ---------------------------------------------------------------------------------------------
# Stage 5b (v2 only): hidden personas (5.7) and free-text rewriting (5.2)
# ---------------------------------------------------------------------------------------------

def all_text_templates() -> list[str]:
    """Every free-text template the v2 text stage can emit and that has words to paraphrase (sorted, unique).

    This is the list ``paraphrase_templates.py --populate`` sends to the API: v1 neutral / specific / vague /
    copy-paste templates, the boilerplate pool and the persona sentences. Empty and "?" answers are skipped, and
    so is the Lorem ipsum filler: it is pasted verbatim, and Haiku (rightly) returns it unchanged five times."""
    pool = NEUTRAL_TEXTS + SPECIFIC_TEMPLATES + VAGUE_TEXTS + COPY_PASTE_TEXTS + v2_text.BOILERPLATE_POOL
    pool += [t for ts in v2_text.PERSONA_SENTENCES.values() for t in ts]
    return sorted({t for t in pool if any(ch.isalpha() for ch in t) and not t.startswith("Lorem ipsum")})


def persona_safe(paraphrases: dict[str, list[str]]) -> dict[str, list[str]]:
    """Drop persona-sentence paraphrases that the POC regex would not call neutral.

    The persona must be readable only from meaning (5.7), but a paraphrase can add a keyword (Haiku turned
    "money is tight" into "our budget is limited"). Paraphrases of every other template are kept as they are."""
    persona = {t for ts in v2_text.PERSONA_SENTENCES.values() for t in ts}
    return {t: [q for q in ps if text_cat(q) == "neutral"] if t in persona else ps for t, ps in paraphrases.items()}


def assign_personas(cfg: Config, leads: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """5.7: give ``persona_share`` of genuine-human leads a hidden persona (and which persona sentence they write)."""
    leads["context_persona"] = None
    leads["persona_sentence"] = None
    if not cfg.context_only_signal:
        return leads
    rng = rng_for(cfg, "v2_persona")
    for i, l in leads.iterrows():
        if people.at[l.person, "is_bot"] or rng.random() >= cfg.persona_share:
            continue
        persona = pick(rng, v2_text.PERSONAS)
        leads.at[i, "context_persona"] = persona
        leads.at[i, "persona_sentence"] = pick(rng, v2_text.PERSONA_SENTENCES[persona])
    return leads


def _variant(rng: np.random.Generator | None, template: str, paraphrases: dict[str, list[str]]) -> str:
    """The template itself or, when paraphrasing, one of [template] + its cached paraphrases chosen uniformly."""
    if rng is None or template not in paraphrases:
        return template
    options = [template] + paraphrases[template]
    return options[int(rng.integers(0, len(options)))]


def boilerplate_text(rng: np.random.Generator, prng: np.random.Generator | None,
                     paraphrases: dict[str, list[str]]) -> str:
    """5.2 (iv): pasted marketing copy from the v1 texts plus the wider pool, with a varied opening; 35% sit
    mid-text between a human lead-in and tail, 15% paste two snippets."""
    pool = COPY_PASTE_TEXTS + v2_text.BOILERPLATE_POOL
    body = _variant(prng, pick(rng, pool), paraphrases)
    if rng.random() < 0.15:
        body += " " + _variant(prng, pick(rng, pool), paraphrases)
    if rng.random() < 0.35:
        return f"{pick(rng, v2_text.BOILERPLATE_LEAD_INS)} {body} {pick(rng, v2_text.BOILERPLATE_TAILS)}"
    return pick(rng, v2_text.BOILERPLATE_OPENERS) + body


def native_text(rng: np.random.Generator, country: str, cat: str, team: int) -> str:
    """5.2 (iii): a hand-written DE/FR/NL answer with the same meaning class as ``cat``."""
    bank = v2_text.LANG_BANK[country]
    tpl = pick(rng, bank[cat])
    return tpl.format(team=team, month=pick(rng, bank["months"]), k=int(rng.integers(3, 60)))


def rewrite_texts(cfg: Config, leads: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """v2 free text: paraphrase, boilerplate pool, persona sentence, language mixing, typos (in that order).

    Each step has its own RNG stream and runs only when its option is on; the true ``text_category`` never
    changes, only the words. Persona sentences are appended to non-vague answers (a one-word answer such as
    "pricing" cannot carry a persona), so the persona never changes which regex category a text falls in."""
    text_opts = [cfg.text_paraphrase, cfg.text_boilerplate_pool, cfg.text_language_mix_share > 0,
                 cfg.text_typo_share > 0, cfg.context_only_signal]
    if not any(text_opts):
        return leads
    paraphrases = persona_safe(paraphrase_templates.lookup(all_text_templates())) if cfg.text_paraphrase else {}
    prng = rng_for(cfg, "v2_paraphrase") if cfg.text_paraphrase else None
    brng, lrng, trng = rng_for(cfg, "v2_boilerplate"), rng_for(cfg, "v2_language"), rng_for(cfg, "v2_typos")
    lang_mode = []
    for i, l in leads.iterrows():
        p = people.loc[l.person]
        cat = l.text_category
        if cat == "copy_paste" and cfg.text_boilerplate_pool:
            base = boilerplate_text(brng, prng, paraphrases)
        else:
            tpl = _variant(prng, l.text_template, paraphrases)
            base = tpl.format(**l.text_params) if l.text_params else tpl
        persona = ""
        if _present(l.persona_sentence) and cat != "vague":
            persona = _variant(prng, l.persona_sentence, paraphrases)
        mode, closer = None, ""
        if cfg.text_language_mix_share > 0 and p.country in v2_text.LANG_BANK and \
                lrng.random() < cfg.text_language_mix_share:
            bank = v2_text.LANG_BANK[p.country]
            if lrng.random() < 0.5:
                mode = "wrap"
                base = pick(lrng, bank["openers"]) + base
                if lrng.random() < 0.5:
                    closer = pick(lrng, bank["closers"])
            else:
                mode = "native"
                base = native_text(lrng, p.country, cat, int(p.team_size))
        txt = " ".join(x for x in (base, persona) if x) + closer
        if mode:
            txt = txt.strip()
        if cfg.text_typo_share > 0 and trng.random() < cfg.text_typo_share:
            txt = v2_text.add_typos(trng, txt, cfg.typo_char_rate)
        ans = dict(l.answers)
        ans["what_to_solve"] = txt
        leads.at[i, "answers"] = ans
        lang_mode.append(mode)
    leads["language_mix"] = lang_mode
    return leads


# ---------------------------------------------------------------------------------------------
# Stage 6: status-quo tier, speed to lead, hidden close log-odds
# ---------------------------------------------------------------------------------------------

def status_quo_tier(answers: dict, pages_visited: float | None) -> str:
    """Status-quo lead-score tier (A/B/C) from STATUS_QUO_RULES, as a typical advertiser computes it."""
    ls = STATUS_QUO_RULES["lead_score"]
    pts = 0
    jt = (answers.get("job_title") or "").lower()
    if jt:
        hit = next((k["points"] for k in ls["job_title"]["keywords"] if k["keyword"] in jt), None)
        pts += ls["job_title"]["default_points"] if hit is None else hit
    pts += ls["company_size_points"].get(answers.get("company_size") or "", 0)
    if _present(pages_visited):
        pts += next((r["points"] for r in ls["pages_visited_points"] if pages_visited >= r["min_pages"]), 0)
    return next(t["name"] for t in ls["tiers"] if pts >= t["min_points"])


def top_bucket(t: float | None) -> str | None:
    """Time-on-page bucket used by the planted effect; None when there was no web session."""
    if not _present(t):
        return None
    return "<15s" if t < 15 else "15-60s" if t < 60 else "60-300s" if t < 300 else "300-600s" if t <= 600 else ">600s"


def close_log_odds(cfg: Config, l: pd.Series, p: pd.Series, co: pd.Series | None, reply: float | None) -> float:
    """Planted close log-odds of one lead before noise: intercept plus every effect in ground_truth.md.

    ``l`` is the lead row (with the true, not the recorded, session values), ``p`` its persona, ``co`` its company
    row (None for individuals), ``reply`` the first reply time in hours (None when never contacted). With the v2
    options on it adds the two 5.5 interaction terms and the 5.7 persona effect."""
    z = cfg.intercept + cfg.band_eff[p.employee_band] + cfg.free_eff * _flag(p.is_free_email) + cfg.text_eff[l.text_category]
    tb = top_bucket(l.time_on_page_s)
    if tb:
        z += cfg.top_eff[tb]
    if _present(l.hesitation_ms) and l.hesitation_ms > 90000:
        z += cfg.hes_eff
    if _present(l.field_edit_count) and 1 <= l.field_edit_count <= 4:
        z += cfg.edits_eff
    z += cfg.form_eff[l.form_variant] + cfg.channel_eff[p.channel] + cfg.lead_ads_eff * _flag(p.is_lead_ads)
    z += cfg.seniority_eff[p.seniority]
    z += cfg.pricing_eff * _flag(l.viewed_pricing)
    z += cfg.sessions3_eff * (_present(l.sessions_before_convert) and l.sessions_before_convert >= 3)
    z += cfg.brand_eff * (isinstance(l.utm_term, str) and "northstar" in l.utm_term)
    biz = l.submitted_weekday not in ("Saturday", "Sunday") and 9 <= l.local_submit_hour <= 17
    z += cfg.bizhours_eff * biz
    z += cfg.ip_mismatch_eff * (isinstance(l.ip_country, str) and l.ip_country != p.country)
    if co is not None:
        z += cfg.spend_eff[co.monthly_ad_spend_band] + cfg.crm_eff * (co.crm_platform in ("HubSpot", "Salesforce"))
        z += cfg.hiring_eff * _flag(co.is_hiring)
    if reply is not None:
        z += cfg.reply_fast_eff * (reply < 1) + cfg.reply_slow_eff * (reply > 48)
    if cfg.interactions:
        z += cfg.linkedin_51plus_eff * (p.channel == "linkedin" and p.employee_band in ("51-200", "201-1000", "1000+"))
        z += cfg.senior_guide_eff * (p.seniority == "senior" and l.form_variant == "D")
    if cfg.context_only_signal and _present(l.get("context_persona")):
        z += cfg.persona_eff
    return float(z)


def hidden_log_odds(cfg: Config, leads: pd.DataFrame, people: pd.DataFrame, companies: pd.DataFrame) -> pd.DataFrame:
    """Contact decision, reply time and the hidden close log-odds (the planted truth).

    The status-quo tier is computed from what sales can see (recorded pages visited). With
    ``ghosting_follows_tier`` the contact decision depends on that tier alone (5.6)."""
    rng = rng_for(cfg, "hidden")
    grng = rng_for(cfg, "v2_ghosting")
    out = []
    for _, l in leads.iterrows():
        p = people.loc[l.person]
        a = l.answers
        tier = status_quo_tier(a, l.pages_visited)
        free_or_vague = [p.is_free_email, l.text_category in ("vague", "copy_paste")]
        z_nc = -4.2 + 1.7 * (tier == "B") + 2.7 * (tier == "C") + 0.5 * free_or_vague[0] + 0.45 * free_or_vague[1]
        never = never_v1 = bool(rng.random() < 1 / (1 + np.exp(-z_nc)))
        mu = np.log(1.5) + {"A": 0.0, "B": 1.03, "C": 1.89}[tier]
        reply = None
        if not never:
            reply = round(float(np.clip(np.exp(rng.normal(mu, 1.2)), 0.05, 240.0)), 2)
        if cfg.ghosting_follows_tier:
            # 5.6: sales neglect follows the old score only. The v1 draws above are still consumed so the rest of
            # this stage's stream (noise, deal value) is unchanged; the v2 decision has its own stream.
            never = bool(grng.random() < 1 / (1 + np.exp(-(cfg.ghost_intercept + cfg.ghost_tier_eff[tier]))))
            if never:
                reply = None
            elif reply is None:
                reply = round(float(np.clip(np.exp(grng.normal(mu, 1.2)), 0.05, 240.0)), 2)
        co = companies.loc[p.company_idx] if p.company_idx >= 0 else None
        tb = l.get("true_behaviour")
        if _present(tb):                       # 5.4 / 5.8 changed the record, not the behaviour
            l = l.copy()
            for k, v in tb.items():
                l[k] = v
        z = close_log_odds(cfg, l, p, co, reply) + rng.normal(0, cfg.noise_sd)
        # deal value (drawn for every lead; only realised on Won)
        spend = co.monthly_ad_spend_band if co is not None else None
        mu_v = cfg.deal_base[p.employee_band] * (cfg.deal_sector[co.sector] if co is not None else 1.0) * \
            cfg.deal_channel[p.channel] * (cfg.deal_spend[spend] if spend else 1.0)
        dv = float(np.clip(round(mu_v * np.exp(rng.normal(0, cfg.deal_noise_sd)) / 50) * 50, 500, 60000))
        out.append({"tier": tier, "never_contacted": never, "never_contacted_v1": never_v1, "reply_hours": reply, "log_odds": z,
                    "p_close_true": 1 / (1 + np.exp(-z)), "deal_value_true": dv})
    return pd.concat([leads, pd.DataFrame(out, index=leads.index)], axis=1)


# ---------------------------------------------------------------------------------------------
# Stage 7: outcome (win draw, timing, stalls)
# ---------------------------------------------------------------------------------------------

def draw_outcome(cfg: Config, leads: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Draw win/lose from ``p_close_true``, the stall flag, time to close and the stage path for every lead."""
    rng = rng_for(cfg, "outcome")
    srng = rng_for(cfg, "v2_stalls")
    res = []
    for _, l in leads.iterrows():
        p = people.loc[l.person]
        won = bool(rng.random() < l.p_close_true)
        stalled = (not l.never_contacted_v1) and bool(rng.random() < cfg.stall_share)
        if cfg.ghosting_follows_tier:          # 5.6: stalls follow the tier too (own stream, v1 draw kept above)
            stalled = (not l.never_contacted) and bool(srng.random() < cfg.stall_by_tier[l.tier])
        if won:
            days = float(np.exp(rng.normal(np.log(cfg.won_median_days), cfg.won_sd)))
            days *= cfg.won_1000_mult if p.employee_band == "1000+" else 1.0
            days = float(np.clip(days, 2, 330))
            path = ["Qualified"] + (["Demo booked"] if rng.random() < 0.85 else []) + (["Proposal"] if rng.random() < 0.90 else [])
        else:
            days = float(np.clip(np.exp(rng.normal(np.log(cfg.lost_median_days), cfg.lost_sd)), 1, 300))
            depth = int(rng.choice(4, p=[0.46, 0.245, 0.145, 0.15]))
            path = ["Qualified", "Demo booked", "Proposal"][:depth]
        fracs = sorted(rng.uniform(0.10, 0.95, size=len(path)))
        res.append({"won_draw": won, "stalled": stalled, "close_days": days, "path": path, "path_fracs": fracs,
                    "new_offset_s": int(rng.integers(5, 120))})
    return pd.concat([leads, pd.DataFrame(res, index=leads.index)], axis=1)


# ---------------------------------------------------------------------------------------------
# Stage 8: duplicates (same person submits again in a new session)
# ---------------------------------------------------------------------------------------------

def add_duplicates(cfg: Config, leads: pd.DataFrame, people: pd.DataFrame, n_dup: int) -> pd.DataFrame:
    """Append ``n_dup`` repeat submissions: same person and answers, 1-45 days later, in a fresh session."""
    rng = rng_for(cfg, "duplicates")
    brng = rng_for(cfg, "duplicates_behaviour")
    crng, frng = rng_for(cfg, "v2_consent_duplicates"), rng_for(cfg, "v2_fast_humans_duplicates")
    as_of = cfg.as_of_dt
    is_bot = people.is_bot.reindex(leads.person).to_numpy(dtype=bool)
    eligible = leads.index[~is_bot & (leads.created < as_of - timedelta(days=2)).to_numpy()]
    src_idx = rng.choice(eligible, size=n_dup, replace=False)
    rows = []
    for j in sorted(src_idx):
        o = leads.loc[j]
        p = people.loc[o.person]
        d = o.to_dict()
        created = o.created + timedelta(seconds=float(rng.uniform(1.09, 45.0) * 86400))
        cap = as_of - timedelta(hours=1)
        created = min(created, cap)
        tz = ZoneInfo(o.timezone)
        local = created.astimezone(tz)
        if rng.random() < (0.852 if local.weekday() < 5 else 0.395):
            local = local.replace(hour=int(rng.integers(9, 18)))
            created = min(local.astimezone(timezone.utc), cap)
            local = created.astimezone(tz)
        d.update({"created": created, "local_submit_hour": local.hour, "submitted_weekday": local.strftime("%A"),
                  "duplicate_of_idx": j, "consent": bool(rng.random() < 0.9185)})
        # new session: fresh channel and telemetry, same person and same answers
        channel = CHANNELS[int(rng.choice(5, p=CHANNEL_P))]
        lead_ads = channel == "meta" and rng.random() < LEAD_ADS_SHARE_OF_META
        attrib = session_attribution(rng, channel, lead_ads)
        d.update(attrib)
        b = session_behaviour(brng, cfg, channel, lead_ads, False, p.country, p.city, o.form_variant,
                              isinstance(o.phone, str), attrib, created)
        observe_session(cfg, b, lead_ads, False, crng, frng)
        d.update(b)
        d["dup_channel"] = channel
        email = p.email
        if rng.random() < 0.35:
            email = email[0].upper() + email[1:]
        d["email_override"] = email
        d["dup_lost_after_h"] = float(rng.uniform(1, 120)) if rng.random() < 0.47 else None
        rows.append(d)
    dups = pd.DataFrame(rows)
    leads = leads.copy()
    leads["duplicate_of_idx"] = np.nan
    return pd.concat([leads, dups], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# Stage 9: CRM history
# ---------------------------------------------------------------------------------------------

STALL_COMMENTS = ["Chased twice, no reply yet", "Waiting on their budget sign-off", "Went quiet, chasing"]
GENERIC_LOST = ["Chose to build in-house", "NO_BUDGET", "NO_RESPONSE", "Price too high",
                "Timing not right, revisit next year", "Went with a competitor"]
PERSONAL_LOST = ["Just researching for a course", "Personal email, no company", "Student, not a buyer"]
MESSY = {"Qualified": ["QUALIFIED", "qualified"], "Demo booked": ["Demo Booked", "demo booked"],
         "Lost": ["Closed Lost", "lost"], "Won": ["Closed Won", "WON", "won"]}


def stage_comment(rng: np.random.Generator, stage: str, p: pd.Series, team: int) -> str | None:
    """Sales-rep comment for a CRM stage row (None = no comment)."""
    if stage == "Contacted":
        return pick(rng, [None, f"Called, spoke to {p.first}", "Intro call booked", "Left voicemail",
                          "Replied to email, wants a call", "Sent follow-up email"])
    if stage == "Qualified":
        k = int(rng.integers(0, 6))
        return [None, "Budget confirmed for this quarter", f"Current tool contract ends in {pick(rng, MONTHS)}",
                "Decision maker on the call", f"Evaluating {int(rng.integers(2, 4))} other vendors",
                f"Wants pricing for {team} seats"][k]
    if stage == "Demo booked":
        return pick(rng, ["Demo booked for next week", f"Demo with {int(rng.integers(1, min(team, 60) + 1))} people from their team",
                          "Wants to see CRM integration"])
    if stage == "Proposal":
        return pick(rng, ["Negotiating annual vs monthly", "Proposal sent, awaiting legal review",
                          f"Sent proposal for {min(team, 60)} seats"])
    if stage == "Won":
        return pick(rng, ["Closed, onboarding next week", "PO received", f"Signed {pick(rng, [12, 24, 36])}-month contract",
                          "Signed after pilot"])
    raise ValueError(stage)


def lost_reason(rng: np.random.Generator, p: pd.Series, team: int) -> str:
    """Lost-reason comment: spam for bots, personal reasons for individuals/students, size or generic otherwise."""
    if p.is_bot:
        return pick(rng, ["Spam submission", "Fake details, number doesn't work"])
    if not p.has_company:
        return pick(rng, PERSONAL_LOST)
    personal_p = 0.7 if p.seniority == "student" else (0.08 if p.is_free_email else 0.02)
    if rng.random() < personal_p:
        return pick(rng, PERSONAL_LOST)
    if p.employee_band == "1-10" and rng.random() < 0.44:
        return pick(rng, ["Too small for our product", f"Price too high for a team of {team}"])
    r = pick(rng, GENERIC_LOST)
    if r == "NO_BUDGET":
        return f"No budget until Q{int(rng.integers(1, 5))}"
    if r == "NO_RESPONSE":
        return f"No response after {int(rng.integers(3, 6))} follow-ups"
    return r


def fmt_ts(t: datetime) -> str:
    """ISO-8601 UTC timestamp with a Z suffix, as in v1."""
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_crm(cfg: Config, leads: pd.DataFrame, people: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Returns crm rows and the final outcome label per lead."""
    rng = rng_for(cfg, "crm")
    as_of = cfg.as_of_dt
    rows, outcome = [], {}
    for i, l in leads.iterrows():
        p = people.loc[l.person]
        team = int(p.team_size)
        created = l.created
        t_new = created + timedelta(seconds=int(l.new_offset_s))
        events = [(t_new, "New", None, None)]
        if _present(l.duplicate_of_idx):          # duplicate lead
            if _present(l.dup_lost_after_h):
                events.append((created + timedelta(hours=float(l.dup_lost_after_h)), "Lost",
                               "Duplicate - merged into existing contact", None))
            outcome[i] = "duplicate"
        elif l.never_contacted:
            outcome[i] = "never_contacted"
        else:
            t_c = t_new + timedelta(hours=float(l.reply_hours))
            events.append((t_c, "Contacted", stage_comment(rng, "Contacted", p, team), None))
            t_close = max(created + timedelta(days=float(l.close_days)), t_c + timedelta(hours=1))
            span = (t_close - t_c).total_seconds()
            for s, f in zip(l.path, l.path_fracs):
                dv = None
                if s == "Proposal":
                    dv = float(round(l.deal_value_true * rng.uniform(0.95, 1.20) / 50) * 50)
                events.append((t_c + timedelta(seconds=span * f), s, stage_comment(rng, s, p, team), dv))
            if l.won_draw:
                dv = None if rng.random() < cfg.won_blank_value_share else float(l.deal_value_true)
                final = (t_close, "Won", stage_comment(rng, "Won", p, team), dv)
            else:
                final = (t_close, "Lost", lost_reason(rng, p, team), None)
            if l.stalled:
                if rng.random() < 0.5:
                    t, s, _, dv = events[-1]
                    events[-1] = (t, s, pick(rng, STALL_COMMENTS), dv)
                outcome[i] = "open"
            else:
                events.append(final)
                outcome[i] = "won" if l.won_draw else "lost"
            if outcome[i] in ("won", "lost") and final[0] > as_of:
                outcome[i] = "open"
        for t, s, c, dv in events:
            if t > as_of:
                continue
            stage = s
            if s in MESSY and rng.random() < cfg.messy_stage_share:
                stage = pick(rng, MESSY[s])
            rows.append({"idx": i, "stage": stage, "comment": c, "deal_value": dv, "t": t})
    crm = pd.DataFrame(rows)
    return crm, pd.Series(outcome)


# ---------------------------------------------------------------------------------------------
# Stage 10: person-data vendor
# ---------------------------------------------------------------------------------------------

AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_P = {"senior": [0, 0.081, 0.365, 0.360, 0.167, 0.027], "junior": [0.207, 0.605, 0.142, 0.046, 0, 0],
         "mid": [0.021, 0.344, 0.441, 0.170, 0.024, 0], "student": [0.87, 0.13, 0, 0, 0, 0]}
INCOME = ["under £25k", "£25k-£50k", "£50k-£100k", "£100k-£150k", "£150k+"]
INCOME_P = {("senior", False): [0, 0.012, 0.24, 0.55, 0.198], ("senior", True): [0, 0.01, 0.13, 0.44, 0.42],
            ("junior", False): [0.26, 0.58, 0.16, 0, 0], ("junior", True): [0.14, 0.43, 0.35, 0.08, 0],
            ("mid", False): [0.01, 0.24, 0.58, 0.17, 0], ("mid", True): [0.01, 0.16, 0.43, 0.33, 0.07],
            ("student", False): [0.78, 0.22, 0, 0, 0], ("student", True): [0.54, 0.35, 0.11, 0, 0]}
HOME_P = [0.23, 0.41, 0.56, 0.60, 0.73, 0.71]


def make_vendor_people(cfg: Config, people: pd.DataFrame) -> pd.DataFrame:
    """people.csv: a fake person-data vendor covering 75% of business and 40% of free emails, no bots."""
    rng = rng_for(cfg, "vendor")
    rows = []
    for _, p in people.iterrows():
        if p.is_bot or rng.random() >= (0.40 if p.is_free_email else 0.75):
            continue
        age_i = int(rng.choice(6, p=np.array(AGE_P[p.seniority]) / sum(AGE_P[p.seniority])))
        inc = INCOME_P[(p.seniority, p.country == "US")]
        rows.append({
            "email": p.email.lower(), "age_band": AGE_BANDS[age_i],
            "household_income_band": pick(rng, INCOME, inc), "country": p.country,
            "region": REGION[p.city], "city": p.city, "job_title": p.job_title,
            "seniority": VENDOR_SENIORITY[p.seniority], "department": DEPARTMENT[p.job_title],
            "years_in_role": 0 if p.seniority == "student" else int(min(25, rng.negative_binomial(2, 2 / 5.3))),
            "is_homeowner": bool(rng.random() < HOME_P[age_i]),
            "match_confidence": round(float(rng.integers(60, 100)) / 100, 2)})
    return pd.DataFrame(rows).sort_values("email").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------

def generate(cfg: Config) -> dict[str, pd.DataFrame | dict]:
    """Run every stage and return the six v1 tables keyed by file stem (status_quo_rules is a dict).

    With any v2 option on, ground_truth_labels gets the V2_LABEL_COLUMNS, and with enrichment dropout a hidden
    ``ground_truth_companies`` table (the full firmographics, since companies.csv then misses some domains)."""
    n_dup = int(round(cfg.n * cfg.dup_share))
    n_orig = cfg.n - n_dup
    companies = make_companies(cfg)
    people = make_people(cfg, companies, n_orig)
    leads = make_leads(cfg, people, companies)
    leads = vary_company_names(cfg, leads, people, companies)
    leads = add_behaviour(cfg, leads, people)
    leads = add_text(cfg, leads, people)
    leads = assign_personas(cfg, leads, people)
    leads = rewrite_texts(cfg, leads, people)
    leads = hidden_log_odds(cfg, leads, people, companies)
    leads = draw_outcome(cfg, leads, people)
    leads = add_duplicates(cfg, leads, people, n_dup)

    # order by time and assign ids
    leads = leads.sort_values("created", kind="stable").reset_index(drop=False).rename(columns={"index": "row"})
    width = max(5, len(str(len(leads))))
    leads["lead_id"] = [f"L{k + 1:0{width}d}" for k in range(len(leads))]
    id_of_row = dict(zip(leads.row, leads.lead_id))

    crm, outcome = make_crm(cfg, leads, people)
    crm["lead_id"] = leads.lead_id.values[crm.idx.values]
    crm = crm.sort_values(["t", "lead_id"], kind="stable")
    crm_out = pd.DataFrame({"lead_id": crm.lead_id, "stage": crm.stage, "comment": crm.comment,
                            "deal_value": crm.deal_value.astype(float), "changed_at": crm.t.map(fmt_ts)})

    P = people.loc[leads.person].reset_index(drop=True)
    is_dup = leads.duplicate_of_idx.notna()
    H = pd.DataFrame({
        "lead_id": leads.lead_id,
        "created_at": leads.created.map(fmt_ts),
        "form_variant": leads.form_variant,
        "answers": leads.answers.map(lambda a: json.dumps(a, ensure_ascii=False)),
        "email": np.where(is_dup, leads.get("email_override"), P.email),
        "phone": leads.phone, "name": P.name, "company_name": leads.company_name,
        "company_domain": leads.company_domain,
    })
    # Lead Ads forms have no company question; the company name field stays blank
    for c in LEAD_COLUMNS:
        if c not in H:
            H[c] = leads[c].values if c in leads else None
    H = H[LEAD_COLUMNS]
    for c in INT_COLUMNS:
        H[c] = pd.array([None if v is None or v != v else int(v) for v in H[c]], dtype="Int64")
    for c in ["pasted_text", "returned_visitor", "is_datacenter_ip", "viewed_pricing"]:
        H[c] = H[c].astype(object)

    # labels
    src_row = leads.duplicate_of_idx.where(is_dup, leads.row).astype(int)
    orig = leads.set_index("row").loc[src_row.values].reset_index()
    comp = [companies.loc[k] if k >= 0 else None for k in P.company_idx]
    labels = pd.DataFrame({
        "lead_id": leads.lead_id,
        "duplicate_of": [id_of_row[int(r)] if d else None for r, d in zip(leads.duplicate_of_idx.fillna(-1), is_dup)],
        "outcome": outcome.reindex(leads.index).values,
        "p_close_true": orig.p_close_true.round(4).values,
        "channel": np.where(is_dup, leads.get("dup_channel"), P.channel),
        "is_lead_ads": P.is_lead_ads.values,
        "is_bot": P.is_bot.values, "is_free_email": P.is_free_email.values, "has_company": P.has_company.values,
        "employee_band": P.employee_band.values,
        "sector": [c.sector if c is not None else None for c in comp],
        "company_domain": [c.domain if c is not None else None for c in comp],
        "seniority": P.seniority.values, "text_category": orig.text_category.values, "country": P.country.values,
        "click_id_missing": leads.click_id_missing.astype(bool).values,
        "never_contacted": np.where(is_dup, False, leads.never_contacted.astype(bool)),
        "stalled": np.where(is_dup | (outcome.reindex(leads.index) != "open").values, False, leads.stalled.astype(bool)),
        "deal_value": orig.deal_value_true.astype(float).values,
        "reply_hours": np.where(is_dup, np.nan, leads.reply_hours.astype(float)),
    })[LABEL_COLUMNS]

    out = {"historical_leads": H, "crm_history": crm_out, "companies": companies, "people": make_vendor_people(cfg, people),
           "ground_truth_labels": labels, "status_quo_rules": STATUS_QUO_RULES}
    if cfg.enrichment_dropout > 0:
        out["companies"], out["ground_truth_companies"] = drop_enrichment(cfg, companies, people)
    if v2_on(cfg):
        in_csv = out.get("ground_truth_companies", companies.assign(domain_in_companies_csv=True))
        in_csv = in_csv.set_index("domain").domain_in_companies_csv
        labels["context_persona"] = orig.context_persona.values
        labels["consent_declined"] = leads.consent_declined.astype(bool).values
        labels["fast_human"] = leads.fast_human.astype(bool).values
        labels["domain_in_companies"] = [bool(in_csv[d]) if _present(d) else None for d in labels.company_domain]
        labels["language_mix"] = orig.language_mix.values if "language_mix" in orig else None
    return out


def write(out: dict[str, pd.DataFrame | dict], out_dir: str, cfg: Config) -> None:
    """Write the tables from ``generate`` plus ground_truth.md (measured on the written files) into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    for name in ["historical_leads", "crm_history", "companies", "people", "ground_truth_labels", "ground_truth_companies"]:
        if name in out:
            out[name].to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    with open(os.path.join(out_dir, "status_quo_rules.json"), "w") as f:
        json.dump(out["status_quo_rules"], f, indent=2, ensure_ascii=False)
    from ground_truth_report import render_ground_truth  # sibling module in scripts/ (path set at import)
    with open(os.path.join(out_dir, "ground_truth.md"), "w") as f:
        f.write(render_ground_truth(out_dir, cfg))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--as-of", default="2026-09-24")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000)
    a = ap.parse_args(argv)
    cfg = Config(seed=a.seed, as_of=a.as_of, n=a.n)
    write(generate(cfg), a.out, cfg)


if __name__ == "__main__":
    main()
