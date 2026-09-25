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

Measured on this output: 10000 leads, 1145 Won (11.5% of all
leads), 5929 decided leads with a 19.3% win rate.

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
| none | 885 | 1.5% |
| 1-10 | 1396 | 8.8% |
| 11-50 | 1358 | 17.0% |
| 51-200 | 1091 | 32.6% |
| 201-1000 | 862 | 36.1% |
| 1000+ | 337 | 32.9% |

### Signal 2: email type

| email | decided leads | win rate |
|---|---|---|
| business email | 4030 | 25.9% |
| free email | 1899 | 5.4% |

### Signal 3: free text answer

| text category | decided leads | win rate |
|---|---|---|
| specific | 1287 | 33.4% |
| neutral | 2340 | 20.0% |
| vague | 1707 | 11.0% |
| copy_paste | 595 | 9.7% |

### Signals 4 and 5: behaviour

| time on page | decided leads | win rate |
|---|---|---|
| <15s (bot-like) | 295 | 5.8% |
| 15-60s | 821 | 19.7% |
| 60-300s | 2990 | 22.7% |
| 300-600s | 488 | 19.1% |
| >600s | 68 | 14.7% |
| blank (consent declined) | 783 | 20.6% |

| hesitation | decided leads | win rate |
|---|---|---|
| over 90s | 359 | 18.1% |
| under 90s | 4787 | 19.2% |
| blank (consent declined) | 783 | 20.6% |

### Signal 7: form variant

| form variant | decided leads | win rate |
|---|---|---|
| A | 2233 | 15.4% |
| B | 1738 | 22.7% |
| C | 630 | 34.1% |
| D | 1328 | 14.5% |

Lead counts by variant: {'A': 4181, 'B': 2635, 'C': 952, 'D': 2232}

### Signal 8: channel

| channel | decided leads | win rate |
|---|---|---|
| meta | 2341 | 9.4% |
| google | 1742 | 24.5% |
| linkedin | 951 | 33.9% |
| chatgpt | 321 | 19.0% |
| organic_direct | 574 | 20.0% |

### Signal 9: seniority

| seniority | decided leads | win rate |
|---|---|---|
| senior | 1489 | 21.7% |
| mid | 2247 | 20.0% |
| junior | 1875 | 19.0% |
| student | 318 | 5.0% |

## Deal value

Deal value = band base x sector multiplier x channel multiplier x ad-spend multiplier x lognormal noise
(sd 0.45), rounded to GBP 50 and clipped to 500-60,000.

- Band base (GBP): {'none': 900, '1-10': 1500, '11-50': 3500, '51-200': 8000, '201-1000': 15000, '1000+': 24000}
- Sector multiplier: {'Software': 1.3, 'Logistics': 1.1, 'Healthcare': 1.2, 'Financial services': 1.4, 'Manufacturing': 1.0, 'Retail': 0.8, 'Education': 0.6, 'Professional services': 0.9}
- Channel multiplier: {'meta': 0.8, 'google': 1.25, 'linkedin': 1.35, 'chatgpt': 1.0, 'organic_direct': 1.0}

Won deals in the data: median GBP 9,600, mean GBP 13,793,
max GBP 60,000.

Median won deal value by channel (GBP): {'chatgpt': 10700.0, 'google': 10250.0, 'linkedin': 10925.0, 'meta': 7000.0, 'organic_direct': 7700.0}

## Time to close

Won deals take a lognormal number of days (median 40, sd 1.09); 1000+
companies take 1.8x longer. Lost deals: median 18 days.

Observed in crm_history.csv (Won rows only): median 33 days,
21% within 14 days, 14% past 90 days.
Long deals started recently are still open, so the observed tail is shorter than the planted one.

Median days to Won by band: {'1-10': 35.0, '1000+': 48.0, '11-50': 34.0, '201-1000': 29.0, '51-200': 33.0, 'none': 37.0}

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
| yes | 1565 | 26.9% |
| no | 3097 | 17.4% |
| no page (Lead Ads) | 484 | 5.0% |
| blank (consent declined) | 783 | 20.6% |

| sessions before converting | decided leads | win rate |
|---|---|---|
| 1 | 2047 | 18.1% |
| 2 | 1596 | 22.4% |
| 3+ | 1019 | 22.8% |
| blank (consent declined) | 783 | 20.6% |

| search term | decided leads | win rate |
|---|---|---|
| brand | 411 | 28.5% |
| generic | 1331 | 23.3% |
| not Google | 4187 | 17.1% |

| submitted | decided leads | win rate |
|---|---|---|
| weekday 09-18 | 4490 | 20.7% |
| outside | 1439 | 15.0% |

| IP country | decided leads | win rate |
|---|---|---|
| match | 4251 | 21.5% |
| mismatch | 411 | 10.7% |
| no IP (Lead Ads) | 484 | 5.0% |
| blank (consent declined) | 783 | 20.6% |

| monthly ad spend | decided leads | win rate |
|---|---|---|
| none | 987 | 10.9% |
| under £5k | 1489 | 16.5% |
| £5k-£25k | 1405 | 28.4% |
| £25k-£100k | 869 | 32.8% |
| £100k+ | 294 | 32.3% |
| no company | 885 | 1.5% |

| CRM platform | decided leads | win rate |
|---|---|---|
| HubSpot | 1511 | 21.5% |
| Salesforce | 1374 | 31.0% |
| Pipedrive | 657 | 20.2% |
| Spreadsheet | 939 | 15.7% |
| No CRM | 563 | 17.9% |
| no company | 885 | 1.5% |

| hiring | decided leads | win rate |
|---|---|---|
| hiring | 2125 | 26.6% |
| not hiring | 2919 | 19.4% |
| no company | 885 | 1.5% |

| first reply | decided leads | win rate |
|---|---|---|
| under 1h | 463 | 33.7% |
| 1-24h | 4545 | 19.3% |
| 24-48h | 564 | 11.9% |
| over 48h | 357 | 12.3% |

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
| 18-24 | 432 | 15.3% |
| 25-34 | 1251 | 23.9% |
| 35-44 | 1131 | 23.4% |
| 45-54 | 647 | 24.1% |
| 55-64 | 187 | 23.0% |
| 65+ | 26 | 15.4% |
| no vendor match | 2255 | 13.8% |

| household income | decided leads | win rate |
|---|---|---|
| under £25k | 383 | 19.1% |
| £25k-£50k | 973 | 21.7% |
| £50k-£100k | 1211 | 23.0% |
| £100k-£150k | 837 | 24.5% |
| £150k+ | 270 | 24.1% |
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
| UK | 2088 | 20.1% |
| US | 1725 | 17.9% |
| DE | 727 | 19.1% |
| FR | 759 | 20.4% |
| NL | 630 | 19.5% |

| pages visited | decided leads | win rate |
|---|---|---|
| none (Lead Ads) | 484 | 5.0% |
| 1 | 1107 | 17.0% |
| 2 | 1402 | 23.5% |
| 3 | 1176 | 20.2% |
| 4 | 603 | 20.4% |
| 5+ | 374 | 21.7% |
| blank (consent declined) | 783 | 20.6% |

## How the status quo tiers line up with reality

| status quo tier | decided leads | win rate |
|---|---|---|
| A | 383 | 47.3% |
| B | 2086 | 21.8% |
| C | 3460 | 14.7% |

Variant A has no job title or company size questions, so every variant A lead lands in tier C,
whatever its real quality.

## Planted data problems (the readiness check must catch every one)

| Problem | How it was planted | Count |
|---|---|---|
| Missing click IDs | 30% of paid website leads have every click ID blanked (fbp cookie kept) | 2397 of 8158 paid website leads |
| Duplicates | Same person submits again with a new lead_id; 35% have the email in a different case | 300 leads |
| Inconsistent stage names | "Closed Won", "won", "WON", "Closed Lost", "lost", "Demo Booked", "demo booked", "qualified", "QUALIFIED" | 2112 CRM rows |
| Open leads | Still in progress on 2026-09-24: deals not closed yet, plus 11% of contacted leads that stalled | 1772 leads |
| Won without deal value | deal_value blank on 8% of Won rows | 98 rows |
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
| 5.2 | Free text: LLM paraphrase on (Claude Haiku paraphrases from scripts/paraphrase_cache.json); typos on 30% of texts (1.5% of letters, at least one); 35% of DE/FR/NL answers mixed with a hand-written phrase bank (half wrapped in a native greeting/closing, half replaced by a native answer of the same meaning class); copy-paste drawn from 18 marketing snippets with varied openings, 35% mid-text | Words only: the true text category and its effect are unchanged | Regex agrees with the generator category on 66.5% of leads (68.5% of genuine humans). Language-mixed originals: {'native': 614, 'wrap': 602} |
| 5.3 | Enrichment dropout: 30% of business-email domains get a legacy domain in companies.csv (row kept, so typed-name matching can find it; the true row is in ground_truth_companies.csv); 70% of typed company names get suffix/case/punctuation/spacing noise | No effect on closing (the company's firmographic effects still apply) | 576 of 1919 business domains missing; 2031 domain-miss leads, 1219 typed a company name, 1219 of those normalise to the right company. Leads with no domain join at all: 5315 |
| 5.4 | Consent declined: 15% of website sessions record no analytics telemetry (columns: time_on_page_s, hesitation_ms, field_edit_count, fields_edited, pasted_text, sessions_before_convert, days_since_first_visit, returned_visitor, pages_visited, viewed_pricing, scroll_depth_pct, fbp, ip, ip_country, ip_city, is_datacenter_ip); flag `consent_declined` | No effect: closing uses the real behaviour; the status-quo tier sees the blank pages count | 1347 rows, 14.8% of website rows |
| 5.5 | Interactions: LinkedIn lead from a company with 51+ employees; senior title on form D | +0.8 and -0.8 on close log-odds | Outcome regression on decided originals: +0.87 [+0.50, +1.23] (n = 412) and -0.82 [-1.27, -0.38] (n = 345) |
| 5.6 | Ghosting follows the status-quo tier: never-contacted log-odds = -5.3 + {'A': 0.0, 'B': 2.8, 'C': 4.4} by tier, nothing else; stalls {'A': 0.05, 'B': 0.1, 'C': 0.17} by tier | Sales neglect of low tiers | Never contacted 20.6% of originals, by tier {'A': 0.008, 'B': 0.081, 'C': 0.277}; correlation with tier 0.240, with true p -0.065 |
| 5.7 | Hidden persona (nonprofit, job seeker, competitor, agency pitching) on 8% of original leads, assigned only among genuine humans with a neutral or specific answer so that every persona lead's text carries one persona sentence (no regex keyword). The sentence stays in English when a DE/FR/NL answer is replaced by a native one (code-switching). `context_persona` | -1.0 on close log-odds | 809 of 9700 original leads (8.3%), {'nonprofit': 213, 'competitor': 205, 'job_seeker': 197, 'agency_pitching': 194}; persona leads without a text trace: 0; decided win rate 15.2% vs 19.7%; largest regex-category share gap, persona vs other neutral/specific human texts: 0.4 pp |
| 5.8 | Fast humans: 2% of genuine-human website sessions with telemetry record 5-14.9s on the page; flag `fast_human` | None: they keep the close odds of their real session | 137 sessions (1.8%). Bot rule (headless or under 15s): precision 72.4%, recall 90.2% (137 false positives, 137 of them fast humans; 39 missed bots, 39 of them consent-declined) |

### Regex text category vs generator category (rows: generator, columns: `emva.features.text_cat`)

| | specific | neutral | vague | copy_paste |
|---|---|---|---|---|
| specific | 1904 | 278 | 0 | 0 |
| neutral | 91 | 3862 | 0 | 0 |
| vague | 0 | 1980 | 873 | 0 |
| copy_paste | 7 | 990 | 0 | 15 |

### Persona texts vs other genuine-human neutral/specific texts: regex category shares

| | persona | no_persona |
|---|---|---|
| specific | 0.33 | 0.326 |
| neutral | 0.67 | 0.674 |
| vague | 0.0 | 0.0 |
| copy_paste | 0.0 | 0.0 |

### Planted-effect recovery

OLS of logit(p_close_true) on every planted feature (original leads with recorded sessions, 0.001 < p < 0.999;
p is rounded to 4 dp in the labels; bots sit near p = 0, so the truncation pulls the under-15s coefficient toward zero): residual sd 0.496 (planted noise sd 0.5),
largest |recovered - planted| 0.167 (top=<15s).

| | planted | recovered | abs_diff |
|---|---|---|---|
| band=11-50 | 0.5 | 0.501 | 0.001 |
| band=51-200 | 1.3 | 1.298 | 0.002 |
| band=201-1000 | 1.3 | 1.283 | 0.017 |
| band=1000+ | 1.0 | 1.035 | 0.035 |
| free_email | -1.3 | -1.284 | 0.016 |
| text=specific | 0.75 | 0.769 | 0.019 |
| text=vague | -0.75 | -0.757 | 0.007 |
| text=copy_paste | -0.75 | -0.745 | 0.005 |
| top=<15s | -1.5 | -1.333 | 0.167 |
| top=60-300s | 0.4 | 0.413 | 0.013 |
| top=300-600s | 0.0 | 0.0 | 0.0 |
| top=>600s | -0.4 | -0.283 | 0.117 |
| hesitation>90s | -0.4 | -0.384 | 0.016 |
| edits_1_4 | 0.2 | 0.216 | 0.016 |
| form=B | 0.2 | 0.199 | 0.001 |
| form=C | 0.7 | 0.671 | 0.029 |
| form=D | -0.3 | -0.278 | 0.022 |
| channel=meta | -0.6 | -0.598 | 0.002 |
| channel=linkedin | 0.0 | 0.026 | 0.026 |
| channel=chatgpt | -0.2 | -0.205 | 0.005 |
| channel=organic_direct | -0.1 | -0.087 | 0.013 |
| lead_ads | -0.2 | -0.154 | 0.046 |
| seniority=senior | 0.3 | 0.3 | 0.0 |
| seniority=mid | 0.1 | 0.112 | 0.012 |
| seniority=student | -1.2 | -1.207 | 0.007 |
| viewed_pricing | 0.5 | 0.494 | 0.006 |
| sessions_3plus | 0.3 | 0.275 | 0.025 |
| brand_term | 0.6 | 0.587 | 0.013 |
| business_hours | 0.25 | 0.245 | 0.005 |
| ip_mismatch | -0.3 | -0.269 | 0.031 |
| spend=none | -0.5 | -0.521 | 0.021 |
| spend=£5k-£25k | 0.4 | 0.398 | 0.002 |
| spend=£25k-£100k | 0.7 | 0.691 | 0.009 |
| spend=£100k+ | 0.7 | 0.688 | 0.012 |
| company_no_spend_ref | -0.2 | -0.181 | 0.019 |
| crm_hubspot_sf | 0.3 | 0.3 | 0.0 |
| hiring | 0.2 | 0.198 | 0.002 |
| reply<1h | 0.4 | 0.41 | 0.01 |
| reply>48h | -0.4 | -0.379 | 0.021 |
| linkedin_x_51plus | 0.8 | 0.781 | 0.019 |
| senior_x_formD | -0.8 | -0.784 | 0.016 |
| persona | -1.0 | -1.005 | 0.005 |
