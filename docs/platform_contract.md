# Platform contract: how EMVA values reach Google Ads and Meta

Design only (plan 3.5). Nothing in this repo calls an ad platform. Numbers are from the synthetic data
(`data/v1`, `data/v2`), **on simulated data**, and are not for external quotation (claims policy).
Platform rules below are stated from the author's knowledge as of mid 2026 and are marked **verify**
where a limit or field name must be checked against the platform documentation before an upload job is built.

## 1. What EMVA produces

`python -m emva` in horizon mode (the default) writes `scores.csv` with, per scored lead (`lead_id`):

| column | meaning | when known |
|---|---|---|
| `value_at_submit` | `p × E[deal value] × margin` through the fitted value transform (default cap p97 + log + floor £25, ADR 0012). It is a bidding signal, not revenue: a median lead is sent about 2.5 to 2.8× its expected value, a capped lead about 0.5×, so ROAS reporting must use recorded revenue, never this value, GBP | at submit |
| `value_at_submit_ts` | `created_at` (the conversion time of the lead event) | at submit |
| `value_at_close` | recorded deal value if Won; blank for a win with no recorded amount; 0 for a mature non-win; blank while immature | at close or at maturity |
| `value_at_close_ts` | `won_at` for a win (also when the amount is unknown), `matured_at` for a mature non-win, empty while immature | same |
| `value_at_close_status` | `known` (send), `unknown_amount` (Won, amount blank: hold), `pending` (immature, not Won: wait) | every run |
| `label_source` | CRM state at the snapshot: `won`, `crm_lost`, `stalled`, `ghosted`, `open` | every run |
| `matured_at` | `created_at + H` (H = 120 days) | at submit |

`value_formula` (the untransformed expected value) stays in the file for comparison; the upload job
sends `value_at_submit`, never `value_formula`. The transform is fitted on the training leads and its
parameters (cap, compression anchor and scale) are fixed for scoring, so a single lead scored in a hosted
app gets the same value as in the batch run. On v1 the cap is £13,221 and the compressed values keep the
training leads' total value (section 4 of `reports/phase3.md`).

## 2. Two-stage upload

1. **Submit stage.** When the lead is created (or at the next batch), send one conversion per lead with
   value `value_at_submit` and conversion time `value_at_submit_ts`, keyed by the lead's click id or hashed
   identifiers (section 3) and by an order / transaction id = `lead_id` so later stages can refer to it.
2. **Close stage.** On each run, for every lead whose `value_at_close_status` became `known` since the previous
   run, send the adjustment:
   - **Google Ads**: a conversion adjustment of type RESTATEMENT on the original conversion (matched by
     `order_id` = `lead_id`) to `value_at_close`; a mature non-win (value 0) is a RETRACTION. Adjustments
     are accepted only for a limited time after the original conversion (**verify**; historically bounded by
     the conversion action's click-through window, at most 90 days). On v1, **15.2%** of wins close more
     than 90 days after `created_at` (v2 14.4%; median time to win 35 days, p90 114 days), so their
     restatement may be refused. Fallback: a second conversion action "closed-won" uploaded as an offline
     conversion at `won_at` with value `value_at_close`, used for reporting (secondary action) rather than bidding.
   - **Meta**: the Conversions API does not edit a sent event. Send a second event at close (a custom event
     such as `ClosedWon`, or Meta's CRM "conversion leads" stage event for lead-form leads) with
     `event_time = value_at_close_ts` and `value = value_at_close`. Meta rejects events whose `event_time` is
     more than 7 days old (**verify**), so the close event must be sent within a week of `won_at`; a mature
     non-win sends nothing (Meta has no retraction).
3. **Idempotency.** The upload job keeps a ledger keyed by (`lead_id`, stage) with the value sent, so a
   re-run never double-counts. A lead that moves from 0 (matured, not won) to Won later (a late win) gets a
   new close-stage row with `value_at_close_ts = won_at`; on Google this is a second restatement, on Meta a
   new close event.
4. **Unknown amounts (R10).** 112 of 1,199 wins on v1 (96 of 1,128 on v2) have no recorded deal value:
   `value_at_close` is blank, `value_at_close_ts` = `won_at` and `value_at_close_status` = `unknown_amount`.
   The upload job **holds** these rows (sends nothing, alerts the CRM owner) and sends the adjustment once an
   amount is recorded (the status then becomes `known`). It never sends 0 for them, which would retract a real
   win; ADR 0002's blank-as-0 is an evaluation convention only.

Which stage the platforms bid on: the submit-stage value is the bidding signal (every lead carries a value,
so there is volume, section 5). The close stage corrects the account's reported value and, where the
platform supports adjustment, what the bidder learns from.

## 3. Matching: click ids, hashed email, `fbp`

Scored leads by what an upload could match on (from `python -m emva.eval.report`, section
"Click-ID coverage"; click id = any of `gclid`, `gbraid`, `wbraid`, `fbclid`, `li_fat_id`, `oppref`):

| scored paid leads | v1 | v2 |
|---|---|---|
| landing-page paid leads (Google, Meta website, LinkedIn, ChatGPT) | 7,454 | 7,435 |
| ... with a click id | 71.2% | 70.8% |
| ... **with no click id** | **28.8%** | **29.2%** |
| ... no click id but a Meta `fbp` cookie | 22.9% | 20.1% |
| Meta lead-form leads (lead id, no click id needed) | 932 | 872 |
| all paid leads with no id at all | 25.6% | 26.2% |
| email present / phone present (landing-page paid) | 100% / 24.6% | 100% / 25.0% |

The plan quotes 29.7% of paid leads without a click id on v1. The same definition on the raw 10,000 v1
rows (before bot and duplicate removal) gives 29.0%, on scored leads 28.8%; I could not reproduce 29.7%
exactly (the review's exact channel and id definition is not recorded). By channel on v1 the no-click-id
share is 27.7% (Google), 29.3% (Meta website), 29.2% (LinkedIn), 30.4% (ChatGPT).

- **Google Ads: Enhanced Conversions for Leads.** Upload with `gclid` / `gbraid` / `wbraid` when present,
  and always with SHA-256 of the normalised email (trimmed, lower-cased; Gmail dots and `+suffix` handling
  per Google's normalisation rules, **verify**) and, when given, the E.164 phone. Google matches the hash to
  the signed-in user who clicked, so leads without a click id can still be attributed. Requires the
  Enhanced Conversions for Leads setting and customer-data terms accepted on the account, and the site tag
  capturing the hashed email at form submit (**verify** current setup).
- **Meta: Conversions API.** Send `lead_id` as `event_id`/`order_id` for deduplication with the pixel, and in
  `user_data`: `fbc` (built from `fbclid` when present), `fbp` (the `_fbp` browser cookie, present on
  78% of v1 Meta website leads), SHA-256 email and phone, client IP and user agent. Lead-form leads upload
  by `lead_id` through the CRM integration instead.
- **LinkedIn / ChatGPT.** Out of scope for the value upload in this plan; `li_fat_id` and `oppref` are
  recorded so a later integration can use them.
- **Consent.** Only leads with `consent == True` may be uploaded with hashed identifiers; EEA/UK uploads
  need the platform's consent signals (Google `ad_user_data` / `ad_personalization`, **verify**).

## 4. The unattributable share

After click id, lead-form id, hashed email and `fbp`, some conversions will still not match a platform user
(no signed-in Google account, email typed differently from the account email, consent missing). What to do:

1. **Do not reassign their value** to matched leads: inflating matched leads' values biases the bidder
   toward whatever audience happens to match best.
2. **Measure the match rate per platform** from the upload diagnostics (Google reports the Enhanced
   Conversions match rate; Meta reports Event Match Quality) and track it as a KPI next to the value upload.
3. **Report, not bid, on the gap:** the model's value for unmatched leads is still counted in EMVA's own
   pipeline value by channel and campaign (from `utm_*`), so the business sees total predicted revenue per
   channel even where the platform cannot attribute it.
4. **Reduce it at the source:** capture `gclid`/`gbraid`/`wbraid` and `fbclid` in hidden form fields on
   every landing page, keep `fbp` on the form post, and collect phone where the form allows (a second
   hashed identifier; 24.6% of v1 landing-page paid leads give one).

## 5. Volume: is there enough to bid on value?

`python -m emva.troas` (plan 3.4). On v1 each Google campaign has about 55-60 leads a month but only about
7-8 wins per 30 days, against Google's tROAS threshold of 30 conversions per campaign in 30 days (**verify**).
Uploading a value for every lead at submit gives each campaign enough conversions; bidding on wins alone
does not, except all four Google campaigns pooled in a portfolio strategy (30.1 wins per 30 days on v1,
29.1 on v2: at the threshold). Meta ad sets see about 17 leads a week against ~50 optimisation events a week
(**verify**): even the submit stage is below the learning threshold per campaign; pooled across the four Meta
campaigns (website + lead forms) it is 70 leads a week. A customer with 2,000 leads a year in one campaign
has 164 lead conversions and about 22 wins per 30 days. Full tables in `reports/phase3.md`.

Consequences for the contract: bid on the **submit-stage value**; consolidate campaigns (or use portfolio
strategies) rather than splitting budget; on Meta, optimise for leads with value (or use the CRM
"conversion leads" goal) at the account's pooled volume rather than per ad set.
