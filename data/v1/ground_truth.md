# Ground truth: planted signals and data problems

Written by scripts/generate_data.py (seed 20260924, data as of 2026-09-24). Rerunning the script
rewrites this file with identical numbers.

Every lead has a hidden close log-odds: an intercept (-3.12, calibrated so ~12% of all leads
end up Won) plus the effects below, plus random noise (sd 0.5). Win or lose is then drawn
from that probability. A good model should rediscover these effects from the data alone.

"Win rate" in the tables below means Won / (Won + Lost), among leads with a final outcome. Open,
never-contacted and duplicate leads are left out. Observed rates mix several signals together (for
example Meta has more free emails), so they won't match the planted effect exactly.

Per-lead hidden truth is in `ground_truth_labels.csv`. Use it only for tests and evaluation, never as
model input.

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
| none | 917 | 2.5% |
| 1-10 | 1426 | 10.6% |
| 11-50 | 1538 | 15.5% |
| 51-200 | 1211 | 32.9% |
| 201-1000 | 810 | 34.0% |
| 1000+ | 410 | 27.6% |

### Signal 2: email type

| email | decided leads | win rate |
|---|---|---|
| business email | 4364 | 25.0% |
| free email | 1948 | 5.6% |

### Signal 3: free text answer

| text category | decided leads | win rate |
|---|---|---|
| specific | 1352 | 31.2% |
| neutral | 2612 | 20.7% |
| vague | 1772 | 10.1% |
| copy_paste | 576 | 9.9% |

### Signals 4 and 5: behaviour

| time on page | decided leads | win rate |
|---|---|---|
| <15s (bot-like) | 217 | 0.0% |
| 15-60s | 1098 | 18.1% |
| 60-300s | 3634 | 23.9% |
| 300-600s | 715 | 14.0% |
| >600s | 74 | 9.5% |

| hesitation | decided leads | win rate |
|---|---|---|
| over 90s | 463 | 13.2% |
| under 90s | 5849 | 19.5% |

### Signal 7: form variant

| form variant | decided leads | win rate |
|---|---|---|
| A | 2476 | 14.0% |
| B | 1794 | 23.3% |
| C | 598 | 34.8% |
| D | 1444 | 15.7% |

Lead counts by variant: {'A': 4170, 'B': 2675, 'C': 900, 'D': 2255}

### Signal 8: channel

| channel | decided leads | win rate |
|---|---|---|
| meta | 2516 | 10.7% |
| google | 1845 | 25.3% |
| linkedin | 1022 | 26.3% |
| chatgpt | 333 | 21.0% |
| organic_direct | 596 | 20.8% |

### Signal 9: seniority

| seniority | decided leads | win rate |
|---|---|---|
| senior | 1515 | 21.9% |
| mid | 2364 | 20.0% |
| junior | 2064 | 18.7% |
| student | 369 | 2.4% |

## Deal value

Deal value = band base x sector multiplier x channel multiplier x lognormal noise (sd 0.45),
rounded to GBP 50 and clipped to 500-60,000.

- Band base (GBP): {'none': 900, '1-10': 1500, '11-50': 3500, '51-200': 8000, '201-1000': 15000, '1000+': 24000}
- Sector multiplier: {'Software': 1.3, 'Logistics': 1.1, 'Healthcare': 1.2, 'Financial services': 1.4, 'Manufacturing': 1.0, 'Retail': 0.8, 'Education': 0.6, 'Professional services': 0.9}
- Channel multiplier: {'meta': 0.8, 'google': 1.25, 'linkedin': 1.35, 'chatgpt': 1.0, 'organic_direct': 1.0}

Won deals in the data: median GBP 8,850, mean GBP 12,963,
max GBP 60,000.

Median won deal value by channel (GBP): {'chatgpt': 9600.0, 'google': 9900.0, 'linkedin': 10000.0, 'meta': 7300.0, 'organic_direct': 6700.0}

## Time to close

Won deals take a lognormal number of days (median 40, sd 1.09); 1000+
companies take 1.8x longer. Lost deals: median 18 days.

Observed in crm_history.csv (Won rows only): median 35 days,
21% within 14 days, 15% past 90 days.
Long deals started recently are still open, so the observed tail is shorter than the planted one.

Median days to Won by band: {'none': 34, '1-10': 35, '11-50': 34, '51-200': 31, '201-1000': 33, '1000+': 48}

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
| 18 | Sales first reply (speed to lead) | under 1 hour +0.4, over 48 hours -0.4 | Moderate. Known only after the lead arrives, and confounded: sales reply faster to leads that look good (median 1.5h for tier A, 10h for tier C) |

Ad spend also raises deal value: {'none': 0.8, 'under £5k': 0.9, '£5k-£25k': 1.0, '£25k-£100k': 1.2, '£100k+': 1.4}.

| viewed pricing | decided leads | win rate |
|---|---|---|
| yes | 1833 | 24.4% |
| no | 3905 | 18.6% |
| no page (Lead Ads) | 574 | 4.4% |

| sessions before converting | decided leads | win rate |
|---|---|---|
| 1 | 2423 | 18.1% |
| 2 | 2007 | 19.6% |
| 3+ | 1308 | 26.1% |

| search term | decided leads | win rate |
|---|---|---|
| brand | 435 | 31.5% |
| generic | 1410 | 23.4% |
| not Google | 4467 | 16.4% |

| submitted | decided leads | win rate |
|---|---|---|
| weekday 09-18 | 4888 | 20.0% |
| outside | 1424 | 15.4% |

| IP country | decided leads | win rate |
|---|---|---|
| match | 5262 | 21.4% |
| mismatch | 476 | 10.1% |
| no IP (Lead Ads) | 574 | 4.4% |

| monthly ad spend | decided leads | win rate |
|---|---|---|
| none | 1020 | 10.5% |
| under £5k | 1656 | 15.7% |
| £5k-£25k | 1531 | 25.9% |
| £25k-£100k | 895 | 34.0% |
| £100k+ | 293 | 36.9% |
| no company | 917 | 2.5% |

| CRM platform | decided leads | win rate |
|---|---|---|
| HubSpot | 1589 | 23.2% |
| Salesforce | 1463 | 28.7% |
| Pipedrive | 780 | 19.7% |
| Spreadsheet | 1027 | 14.9% |
| No CRM | 536 | 15.1% |
| no company | 917 | 2.5% |

| hiring | decided leads | win rate |
|---|---|---|
| hiring | 2340 | 27.4% |
| not hiring | 3055 | 17.5% |
| no company | 917 | 2.5% |

| first reply | decided leads | win rate |
|---|---|---|
| under 1h | 493 | 30.6% |
| 1-24h | 4814 | 19.1% |
| 24-48h | 634 | 13.7% |
| over 48h | 371 | 11.3% |

## Demographics: proxies, not causes

people.csv is a fake person-data vendor: age band, household income band, region, city, seniority,
department, years in role and homeownership. It matches 5,976 people (about
75% of business emails, 40% of free emails, no bots).

**Age and household income have NO planted effect.** They are generated from seniority (and country),
and seniority does matter. So in the tables below they look predictive, but only because a 45-year-old
is more likely to be a director. A model that controls for seniority and company should give them
roughly zero weight. That is the result to look for, and it is the answer to give a DPO: EMVA doesn't
need sensitive demographics to price a lead.

| age band | decided leads | win rate |
|---|---|---|
| 18-24 | 444 | 11.9% |
| 25-34 | 1415 | 21.1% |
| 35-44 | 1200 | 24.6% |
| 45-54 | 640 | 23.3% |
| 55-64 | 192 | 21.9% |
| 65+ | 26 | 26.9% |
| no vendor match | 2395 | 14.8% |

| household income | decided leads | win rate |
|---|---|---|
| under £25k | 399 | 13.0% |
| £25k-£50k | 1068 | 18.9% |
| £50k-£100k | 1301 | 21.9% |
| £100k-£150k | 858 | 26.2% |
| £150k+ | 291 | 27.5% |
| no vendor match | 2395 | 14.8% |

**Watch for proxy capture.** Household income is built almost directly from seniority, so a model
given both cannot tell them apart. It may hand seniority's credit to income, and income will then look
like a real driver even though it adds no predictive power (the validation run showed AUC 0.778 without
demographics, 0.779 with them). That is why demographics should be excluded from model features by
policy, not left for the model to ignore.

Deliberately **not** generated: gender, ethnicity, religion, health, sexual orientation, exact date
of birth, home address. These are special-category or high-risk data, and ad platforms restrict
using them.

## Not signals (traps)

These have **no planted effect** on closing. Any rule or model that leans on them is wrong.

- **Country.** The status quo adds 20% for UK and US leads. Close rates don't depend on country.
- **Pages visited.** The status quo lead score gives points for it. It is pure noise here.
- **Company size dropdown** is a noisy copy of the true band (15% of answers are one band off).
- **City, region, browser language, scroll depth, weekday name** (beyond the business-hours signal):
  pure noise.
- **Homeowner, years in role** (people.csv): no effect. Homeownership travels with seniority, like age.
- **Annual revenue band, founded year, funding stage** (companies.csv): no direct effect. Revenue
  band is almost a copy of company size, so it looks predictive for that reason only.

| country | decided leads | win rate |
|---|---|---|
| UK | 2158 | 19.7% |
| US | 1923 | 19.0% |
| DE | 754 | 17.9% |
| FR | 819 | 19.2% |
| NL | 658 | 17.6% |

| pages visited | decided leads | win rate |
|---|---|---|
| none (Lead Ads) | 574 | 4.4% |
| 1 | 1291 | 16.0% |
| 2 | 1843 | 22.5% |
| 3 | 1400 | 20.3% |
| 4 | 783 | 21.8% |
| 5+ | 421 | 23.0% |

## How the status quo tiers line up with reality

| status quo tier | decided leads | win rate |
|---|---|---|
| A | 402 | 45.5% |
| B | 2172 | 23.2% |
| C | 3738 | 13.7% |

Variant A has no job title or company size questions, so every variant A lead lands in tier C,
whatever its real quality.

## Planted data problems (the readiness check must catch every one)

| Problem | How it was planted | Count |
|---|---|---|
| Missing click IDs | 30% of paid website leads have every click ID blanked (fbp cookie kept) | 2340 of 8072 paid website leads |
| Duplicates | Same person submits again with a new lead_id; 30% have the email in a different case | 300 leads |
| Inconsistent stage names | "Closed Won", "won", "WON", "Closed Lost", "lost", "Demo Booked", "demo booked", "qualified", "QUALIFIED" | 2139 CRM rows |
| Open leads | Still in progress on 2026-09-24: deals not closed yet, plus 11% of contacted leads that stalled | 1538 leads |
| Won without deal value | deal_value blank on 8% of Won rows | 112 rows |
| No person-data match | people.csv covers 75% of business emails, 40% of free emails and no bots, like a real vendor | 3837 leads with no match |
| IP country mismatch | IP country differs from the country typed (VPN or travelling); half of these on datacenter IPs | 759 leads |
| Datacenter IPs | Bots and some VPN users | 609 leads |
| Never contacted | Stage stays New. Picked with extra weight on status quo tier C, free emails and vague text | 1850 leads (plus duplicates left at New) |

Other deliberate mess: phone numbers in mixed formats, some invalid ("12345"); company names typed with
or without the legal suffix, sometimes lowercase; bots share a handful of IP addresses.

**Selection bias warning:** never-contacted leads have no outcome, and they are mostly the ones that
looked weak. Training only on decided leads under-represents weak-looking leads.
