# Ground truth: planted signals and data problems

Written by scripts/generate_data_v2.py, i.e. scripts/generate_data_v1.py with the Phase 5.2-5.8 options ['text_paraphrase', 'text_typo_share', 'text_language_mix_share', 'text_boilerplate_pool', 'enrichment_dropout', 'consent_missing_share', 'interactions', 'ghosting_follows_tier', 'context_only_signal', 'fast_human_share'] (seed 20260924, data as of 2026-09-24, n = 10000). Rerunning the script with the same arguments rewrites this file with identical numbers. Every v1 mechanism below still applies; the section "v2 mechanisms" at the end lists what the options add.

Every lead has a hidden close log-odds: an intercept (-3.12, calibrated so ~12% of all leads
end up Won) plus the effects below, plus random noise (sd 0.5). Win or lose is then drawn
from that probability. A good model should rediscover these effects from the data alone.

"Win rate" in the tables below means Won / (Won + Lost), among leads with a final outcome. Open,
never-contacted and duplicate leads are left out. Observed rates mix several signals together (for
example Meta has more free emails), so they won't match the planted effect exactly.

Per-lead hidden truth is in `ground_truth_labels.csv`. Use it only for tests and evaluation, never as
model input.

Measured on this output: 10000 leads, 1153 Won (11.5% of all
leads), 5954 decided leads with a 19.4% win rate.

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

| employee band | decided leads | win rate |
|---|---|---|
| none | 890 | 2.6% |
| 1-10 | 1392 | 8.4% |
| 11-50 | 1354 | 14.8% |
| 51-200 | 1099 | 33.5% |
| 201-1000 | 877 | 37.3% |
| 1000+ | 342 | 34.5% |

### Signal 2: email type

| email | decided leads | win rate |
|---|---|---|
| business email | 4047 | 26.0% |
| free email | 1907 | 5.3% |

### Signal 3: free text answer

| text category | decided leads | win rate |
|---|---|---|
| specific | 1303 | 35.5% |
| neutral | 2339 | 19.6% |
| vague | 1713 | 10.5% |
| copy_paste | 599 | 8.7% |

### Signals 4 and 5: behaviour

| time on page | decided leads | win rate |
|---|---|---|
| <15s (bot-like) | 291 | 4.8% |
| 15-60s | 816 | 21.4% |
| 60-300s | 3013 | 22.3% |
| 300-600s | 493 | 16.4% |
| >600s | 71 | 12.7% |
| blank (consent declined) | 784 | 22.8% |

| hesitation | decided leads | win rate |
|---|---|---|
| over 90s | 357 | 16.2% |
| under 90s | 4813 | 19.0% |
| blank (consent declined) | 784 | 22.8% |

### Signal 7: form variant

| form variant | decided leads | win rate |
|---|---|---|
| A | 2246 | 14.2% |
| B | 1747 | 22.7% |
| C | 636 | 35.2% |
| D | 1325 | 16.2% |

Lead counts by variant: {'A': 4181, 'B': 2635, 'C': 952, 'D': 2232}

### Signal 8: channel

| channel | decided leads | win rate |
|---|---|---|
| meta | 2346 | 9.8% |
| google | 1755 | 23.5% |
| linkedin | 954 | 34.3% |
| chatgpt | 324 | 19.4% |
| organic_direct | 575 | 20.9% |

### Signal 9: seniority

| seniority | decided leads | win rate |
|---|---|---|
| senior | 1497 | 21.1% |
| mid | 2249 | 20.3% |
| junior | 1892 | 19.6% |
| student | 316 | 3.5% |

## Deal value

Deal value = band base x sector multiplier x channel multiplier x ad-spend multiplier x lognormal noise
(sd 0.45), rounded to GBP 50 and clipped to 500-60,000.

- Band base (GBP): {'none': 900, '1-10': 1500, '11-50': 3500, '51-200': 8000, '201-1000': 15000, '1000+': 24000}
- Sector multiplier: {'Software': 1.3, 'Logistics': 1.1, 'Healthcare': 1.2, 'Financial services': 1.4, 'Manufacturing': 1.0, 'Retail': 0.8, 'Education': 0.6, 'Professional services': 0.9}
- Channel multiplier: {'meta': 0.8, 'google': 1.25, 'linkedin': 1.35, 'chatgpt': 1.0, 'organic_direct': 1.0}

Won deals in the data: median GBP 9,950, mean GBP 14,156,
max GBP 60,000.

Median won deal value by channel (GBP): {'chatgpt': 13100.0, 'google': 11100.0, 'linkedin': 11250.0, 'meta': 7425.0, 'organic_direct': 6350.0}

## Time to close

Won deals take a lognormal number of days (median 40, sd 1.09); 1000+
companies take 1.8x longer. Lost deals: median 18 days.

Observed in crm_history.csv (Won rows only): median 32 days,
19% within 14 days, 15% past 90 days.
Long deals started recently are still open, so the observed tail is shorter than the planted one.

Median days to Won by band: {'1-10': 30.0, '1000+': 42.0, '11-50': 29.0, '201-1000': 33.0, '51-200': 30.0, 'none': 27.0}

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
| 18 | Sales first reply (speed to lead) | under 1 hour +0.4, over 48 hours -0.4 | Moderate. Known only after the lead arrives, and confounded: sales reply faster to leads that look good (median 1.5h for tier A, 9.7h for tier C) |

Ad spend also raises deal value: {'none': 0.8, 'under £5k': 0.9, '£5k-£25k': 1.0, '£25k-£100k': 1.2, '£100k+': 1.4}.

| viewed pricing | decided leads | win rate |
|---|---|---|
| yes | 1553 | 25.7% |
| no | 3131 | 17.6% |
| no page (Lead Ads) | 486 | 4.7% |
| blank (consent declined) | 784 | 22.8% |

| sessions before converting | decided leads | win rate |
|---|---|---|
| 1 | 2066 | 17.6% |
| 2 | 1602 | 21.3% |
| 3+ | 1016 | 24.1% |
| blank (consent declined) | 784 | 22.8% |

| search term | decided leads | win rate |
|---|---|---|
| brand | 415 | 30.6% |
| generic | 1340 | 21.3% |
| not Google | 4199 | 17.6% |

| submitted | decided leads | win rate |
|---|---|---|
| weekday 09-18 | 4518 | 20.4% |
| outside | 1436 | 16.0% |

| IP country | decided leads | win rate |
|---|---|---|
| match | 4272 | 21.2% |
| mismatch | 412 | 11.4% |
| no IP (Lead Ads) | 486 | 4.7% |
| blank (consent declined) | 784 | 22.8% |

| monthly ad spend | decided leads | win rate |
|---|---|---|
| none | 1001 | 11.5% |
| under £5k | 1475 | 15.1% |
| £5k-£25k | 1413 | 28.0% |
| £25k-£100k | 879 | 33.4% |
| £100k+ | 296 | 34.5% |
| no company | 890 | 2.6% |

| CRM platform | decided leads | win rate |
|---|---|---|
| HubSpot | 1509 | 20.6% |
| Salesforce | 1393 | 33.1% |
| Pipedrive | 665 | 18.0% |
| Spreadsheet | 935 | 14.3% |
| No CRM | 562 | 18.5% |
| no company | 890 | 2.6% |

| hiring | decided leads | win rate |
|---|---|---|
| hiring | 2132 | 27.7% |
| not hiring | 2932 | 18.4% |
| no company | 890 | 2.6% |

| first reply | decided leads | win rate |
|---|---|---|
| under 1h | 477 | 32.5% |
| 1-24h | 4547 | 19.3% |
| 24-48h | 568 | 12.7% |
| over 48h | 362 | 13.0% |

## Demographics: proxies, not causes

people.csv is a fake person-data vendor: age band, household income band, region, city, seniority,
department, years in role and homeownership. It matches 5,966 people (about
75% of business emails, 40% of free emails, no bots).

**Age and household income have NO planted effect.** They are generated from seniority (and country),
and seniority does matter. So in the tables below they look predictive, but only because a 45-year-old
is more likely to be a director. A model that controls for seniority and company should give them
roughly zero weight.

| age band | decided leads | win rate |
|---|---|---|
| 18-24 | 428 | 14.5% |
| 25-34 | 1277 | 23.5% |
| 35-44 | 1139 | 23.7% |
| 45-54 | 647 | 25.2% |
| 55-64 | 182 | 22.0% |
| 65+ | 26 | 23.1% |
| no vendor match | 2255 | 13.8% |

| household income | decided leads | win rate |
|---|---|---|
| under £25k | 371 | 16.7% |
| £25k-£50k | 993 | 22.1% |
| £50k-£100k | 1229 | 22.9% |
| £100k-£150k | 838 | 25.7% |
| £150k+ | 268 | 23.9% |
| no vendor match | 2255 | 13.8% |

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

| country | decided leads | win rate |
|---|---|---|
| UK | 2096 | 19.2% |
| US | 1742 | 18.5% |
| DE | 734 | 17.6% |
| FR | 760 | 23.0% |
| NL | 622 | 19.9% |

| pages visited | decided leads | win rate |
|---|---|---|
| none (Lead Ads) | 486 | 4.7% |
| 1 | 1113 | 17.7% |
| 2 | 1401 | 22.1% |
| 3 | 1193 | 20.4% |
| 4 | 599 | 19.9% |
| 5+ | 378 | 21.7% |
| blank (consent declined) | 784 | 22.8% |

## How the status quo tiers line up with reality

| status quo tier | decided leads | win rate |
|---|---|---|
| A | 384 | 49.0% |
| B | 2110 | 22.4% |
| C | 3460 | 14.2% |

Variant A has no job title or company size questions, so every variant A lead lands in tier C,
whatever its real quality.

## Planted data problems (the readiness check must catch every one)

| Problem | How it was planted | Count |
|---|---|---|
| Missing click IDs | 30% of paid website leads have every click ID blanked (fbp cookie kept) | 2397 of 8158 paid website leads |
| Duplicates | Same person submits again with a new lead_id; 35% have the email in a different case | 300 leads |
| Inconsistent stage names | "Closed Won", "won", "WON", "Closed Lost", "lost", "Demo Booked", "demo booked", "qualified", "QUALIFIED" | 2096 CRM rows |
| Open leads | Still in progress on 2026-09-24: deals not closed yet, plus 11% of contacted leads that stalled | 1747 leads |
| Won without deal value | deal_value blank on 8% of Won rows | 91 rows |
| No person-data match | people.csv covers 75% of business emails, 40% of free emails and no bots | 3844 leads with no match |
| IP country mismatch | IP country differs from the country typed (VPN or travelling); about half of these on datacenter IPs | 641 leads |
| Datacenter IPs | Bots and some VPN users | 496 leads |
| Never contacted | Stage stays New. v2: picked on the status quo tier only (see v2 mechanisms, 5.6) | 1999 leads (plus duplicates left at New) |

Other deliberate mess: phone numbers in mixed formats, some invalid ("12345"); company names typed with
or without the legal suffix, sometimes lowercase; bots share a handful of IP addresses.

**Selection bias warning:** never-contacted leads have no outcome, and they are mostly the ones that
looked weak. Training only on decided leads under-represents weak-looking leads.

## v2 mechanisms (plan items 5.2-5.8)

| Item | Mechanism | Planted size | Measured here |
|---|---|---|---|
| 5.2 | Free text: LLM paraphrase on (Claude Haiku paraphrases from scripts/paraphrase_cache.json); typos on 30% of texts (1.5% of letters, at least one); 35% of DE/FR/NL answers mixed with a hand-written phrase bank (half wrapped in a native greeting/closing, half replaced by a native answer of the same meaning class); copy-paste drawn from 18 marketing snippets with varied openings, 35% mid-text | Words only: the true text category and its effect are unchanged | Regex agrees with the generator category on 66.5% of leads (68.5% of genuine humans). Language-mixed originals: {'native': 602, 'wrap': 581} |
| 5.3 | Enrichment dropout: 30% of business-email domains get a legacy domain in companies.csv (row kept, so typed-name matching can find it; the true row is in ground_truth_companies.csv); 70% of typed company names get suffix/case/punctuation/spacing noise | No effect on closing (the company's firmographic effects still apply) | 576 of 1919 business domains missing; 2031 domain-miss leads, 1219 typed a company name, 1219 of those normalise to the right company. Leads with no domain join at all: 5315 |
| 5.4 | Consent declined: 15% of website sessions record no analytics telemetry (columns: time_on_page_s, hesitation_ms, field_edit_count, fields_edited, pasted_text, sessions_before_convert, days_since_first_visit, returned_visitor, pages_visited, viewed_pricing, scroll_depth_pct, fbp, ip, ip_country, ip_city, is_datacenter_ip); flag `consent_declined` | No effect: closing uses the real behaviour; the status-quo tier sees the blank pages count | 1347 rows, 14.8% of website rows |
| 5.5 | Interactions: LinkedIn lead from a company with 51+ employees; senior title on form D | +0.8 and -0.8 on close log-odds | Outcome regression on decided originals: +0.47 [+0.10, +0.84] (n = 417) and -0.99 [-1.44, -0.54] (n = 350) |
| 5.6 | Ghosting follows the status-quo tier: never-contacted log-odds = -5.3 + {'A': 0.0, 'B': 2.8, 'C': 4.4} by tier, nothing else; stalls {'A': 0.05, 'B': 0.1, 'C': 0.17} by tier | Sales neglect of low tiers | Never contacted 20.6% of originals, by tier {'A': 0.008, 'B': 0.081, 'C': 0.277}; correlation with tier 0.240, with true p -0.066 |
| 5.7 | Hidden persona (nonprofit, job seeker, competitor, agency pitching) on 8% of genuine-human leads, written as one sentence appended to non-vague answers, no regex keyword; `context_persona` | -1.0 on close log-odds | 7.9% of all leads, {'job_seeker': 193, 'agency_pitching': 191, 'nonprofit': 190, 'competitor': 188}; decided win rate 10.6% vs 20.2%; largest regex-category share gap persona vs not 2.4 pp |
| 5.8 | Fast humans: 2% of genuine-human website sessions with telemetry record 5-14.9s on the page; flag `fast_human` | None: they keep the close odds of their real session | 135 sessions (1.8%). Bot rule (headless or under 15s): precision 72.7%, recall 90.2% (135 false positives, 135 of them fast humans; 39 missed bots, 39 of them consent-declined) |

### Regex text category vs generator category (rows: generator, columns: `emva.features.text_cat`)

| | specific | neutral | vague | copy_paste |
|---|---|---|---|---|
| specific | 1917 | 265 | 0 | 0 |
| neutral | 95 | 3858 | 0 | 0 |
| vague | 0 | 1996 | 857 | 0 |
| copy_paste | 11 | 982 | 0 | 19 |

### Persona texts vs other genuine-human texts: regex category shares

| | persona | no_persona |
|---|---|---|
| specific | 0.234 | 0.209 |
| neutral | 0.685 | 0.707 |
| vague | 0.081 | 0.082 |
| copy_paste | 0.0 | 0.002 |

### Planted-effect recovery

OLS of logit(p_close_true) on every planted feature (original leads with recorded sessions, 0.001 < p < 0.999;
p is rounded to 4 dp in the labels; bots sit near p = 0, so the truncation pulls the under-15s coefficient toward zero): residual sd 0.497 (planted noise sd 0.5),
largest |recovered - planted| 0.174 (top=<15s).

| | planted | recovered | abs_diff |
|---|---|---|---|
| band=11-50 | 0.5 | 0.501 | 0.001 |
| band=51-200 | 1.3 | 1.301 | 0.001 |
| band=201-1000 | 1.3 | 1.285 | 0.015 |
| band=1000+ | 1.0 | 1.037 | 0.037 |
| free_email | -1.3 | -1.29 | 0.01 |
| text=specific | 0.75 | 0.768 | 0.018 |
| text=vague | -0.75 | -0.754 | 0.004 |
| text=copy_paste | -0.75 | -0.741 | 0.009 |
| top=<15s | -1.5 | -1.326 | 0.174 |
| top=60-300s | 0.4 | 0.416 | 0.016 |
| top=300-600s | 0.0 | 0.009 | 0.009 |
| top=>600s | -0.4 | -0.259 | 0.141 |
| hesitation>90s | -0.4 | -0.389 | 0.011 |
| edits_1_4 | 0.2 | 0.219 | 0.019 |
| form=B | 0.2 | 0.198 | 0.002 |
| form=C | 0.7 | 0.674 | 0.026 |
| form=D | -0.3 | -0.276 | 0.024 |
| channel=meta | -0.6 | -0.601 | 0.001 |
| channel=linkedin | 0.0 | 0.021 | 0.021 |
| channel=chatgpt | -0.2 | -0.207 | 0.007 |
| channel=organic_direct | -0.1 | -0.088 | 0.012 |
| lead_ads | -0.2 | -0.146 | 0.054 |
| seniority=senior | 0.3 | 0.297 | 0.003 |
| seniority=mid | 0.1 | 0.109 | 0.009 |
| seniority=student | -1.2 | -1.204 | 0.004 |
| viewed_pricing | 0.5 | 0.494 | 0.006 |
| sessions_3plus | 0.3 | 0.274 | 0.026 |
| brand_term | 0.6 | 0.591 | 0.009 |
| business_hours | 0.25 | 0.248 | 0.002 |
| ip_mismatch | -0.3 | -0.271 | 0.029 |
| spend=none | -0.5 | -0.523 | 0.023 |
| spend=£5k-£25k | 0.4 | 0.392 | 0.008 |
| spend=£25k-£100k | 0.7 | 0.684 | 0.016 |
| spend=£100k+ | 0.7 | 0.685 | 0.015 |
| company_no_spend_ref | -0.2 | -0.181 | 0.019 |
| crm_hubspot_sf | 0.3 | 0.3 | 0.0 |
| hiring | 0.2 | 0.197 | 0.003 |
| reply<1h | 0.4 | 0.408 | 0.008 |
| reply>48h | -0.4 | -0.387 | 0.013 |
| linkedin_x_51plus | 0.8 | 0.792 | 0.008 |
| senior_x_formD | -0.8 | -0.78 | 0.02 |
| persona | -1.0 | -0.991 | 0.009 |
