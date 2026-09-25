"""Hand-written phrase banks and text helpers for the v2 free-text stage (plan items 5.2 and 5.7).

Everything here is static data or a pure function of an RNG; the generator (``generate_data_v1.rewrite_texts``)
decides which lead gets what. Nothing here is machine translated or LLM written: the DE/FR/NL phrases, the
persona sentences and the boilerplate pool were written by hand for this generator. The LLM paraphrases live in
``paraphrase_cache.json`` (see ``paraphrase_templates.py``).

Persona sentences carry the hidden persona only through their meaning: none of them contains a keyword the POC
regex (``emva.constants``: vague list, copy-paste prefixes, SPECIFIC_TEXT_PATTERN) reacts to, which
``tests/test_generate_data_v2.py`` checks.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------------
# 5.7 hidden personas: one sentence each, appended to the lead's own answer
# ---------------------------------------------------------------------------------------------

PERSONAS = ["nonprofit", "job_seeker", "competitor", "agency_pitching"]
PERSONA_SENTENCES: dict[str, list[str]] = {
    "nonprofit": [
        "For context, we're a registered charity, so anything we take on has to be funded from donations.",
        "We're a small foundation supporting families in need and could only consider a nonprofit rate.",
        "I should mention we're a volunteer-run community trust and our trustees sign off every spend.",
        "We're a not-for-profit housing association rather than a commercial business, if that changes anything.",
        "Our organisation is a grant-funded charity helping young carers, so money is tight.",
        "Asking on behalf of an animal rescue society that relies entirely on public fundraising.",
        "We're an NGO working on clean water projects and don't really sell anything ourselves.",
    ],
    "job_seeker": [
        "Honestly I'm between roles at the moment and would love to know if you're hiring for growth positions.",
        "I'm exploring a career move into ad tech and wanted to understand your product before applying.",
        "Also, could you pass my details to your recruiter? My CV is on my LinkedIn profile.",
        "I'm preparing for an interview at a company that uses tools like yours and want to sound informed.",
        "Just finished a bootcamp and I'm looking for my first marketing job, any openings on your side?",
        "I was recently made redundant and am keen to join a company like yours.",
    ],
    "competitor": [
        "We build a similar lead scoring product and are curious how you handle value uploads to Meta.",
        "I work at a scoring vendor in the same space and wanted to compare notes on your approach.",
        "We're launching a rival attribution tool next year and are benchmarking what's already out there.",
        "Our product does much the same thing; I'd like to see how your onboarding flow is put together.",
        "I'm on the product team at another conversion-value platform and would like a walkthrough of yours.",
        "Doing some competitive research for our own ad-scoring startup.",
    ],
    "agency_pitching": [
        "We're a growth agency and could run your paid social for you, open to a quick intro call?",
        "I help software brands scale with outbound, could I send over a proposal for your team?",
        "Our studio builds high-converting landing pages and I think we could double your form fills.",
        "We offer white-label SEO services and have worked with several vendors like you.",
        "Quick one: we supply verified B2B contact lists and would love to have you as a client.",
        "I run a small performance marketing consultancy and am looking for products to resell.",
    ],
}

# ---------------------------------------------------------------------------------------------
# 5.2 (iv) boilerplate: pasted marketing copy beyond the three v1 texts, with varied framing
# ---------------------------------------------------------------------------------------------

BOILERPLATE_POOL = [
    "Founded on a passion for excellence, we deliver world-class services that exceed expectations every time.",
    "Trusted by customers worldwide, our award-winning platform transforms the way teams work.",
    "At our core, we believe in innovation, integrity and impact across everything we do.",
    "We are a dynamic, fast-growing company committed to exceptional results for clients of every size.",
    "Our holistic approach combines strategy, creativity and technology to unlock sustainable growth.",
    "Leveraging decades of combined experience, we provide tailored solutions that create lasting value.",
    "We partner with forward-thinking organisations to accelerate digital transformation at scale.",
    "Driven by data and powered by people, we help brands connect with the audiences that matter most.",
    "Your success is our success: we go above and beyond to deliver measurable outcomes.",
    "As an industry leader, we set the standard for quality, reliability and customer satisfaction.",
    "Headquartered in the heart of the city, we serve clients across a broad range of sectors and markets.",
    "We empower visionary leaders with next-generation tools designed for the modern enterprise.",
    "Combining best practice with cutting-edge thinking, we turn complex challenges into simple solutions.",
    "Proudly independent and customer obsessed, we have been redefining our category since day one.",
    "We deliver seamless, end-to-end experiences that delight customers and drive loyalty.",
]
BOILERPLATE_OPENERS = ["", "", "", "About us: ", "Company overview - ", "Hi, ", "Please see below. ",
                       "From our website: ", "Who we are: "]
BOILERPLATE_LEAD_INS = ["Hello, re your ad.", "Hi there.", "Good afternoon.", "Saw your ad, some background:",
                        "Hello team,"]
BOILERPLATE_TAILS = ["Kindly send more details.", "Looking forward to hearing from you.", "Please call me back.",
                     "Thanks.", "Let us know."]

# ---------------------------------------------------------------------------------------------
# 5.2 (iii) language mixing for DE / FR / NL leads (hand-written, not machine translated)
# ---------------------------------------------------------------------------------------------

LANG_BANK: dict[str, dict[str, list[str]]] = {
    "DE": {
        "openers": ["Hallo, ", "Guten Tag, ", "Hallo zusammen, ", "Moin, "],
        "closers": [" Danke!", " Vielen Dank.", " Beste Grüße", " Danke im Voraus."],
        "months": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
                   "Oktober", "November", "Dezember"],
        "neutral": ["Wir möchten die Qualität unserer Leads aus Anzeigen verbessern.",
                    "Wir wollen verstehen, welche Kampagnen gute Leads bringen.",
                    "Unser Vertrieb sagt, die Leads von Meta sind schlecht.",
                    "Wir suchen eine Möglichkeit, unser CRM mit Google Ads zu verbinden."],
        "specific": ["Wir sind ein Team von {team} Leuten und wollen bis {month} live gehen.",
                     "Unser Werbebudget liegt bei ca. {k} Tsd. Euro im Monat, aber wir wissen nicht, welche Leads Umsatz bringen.",
                     "Der Vertrag mit unserem aktuellen Anbieter läuft im {month} aus, {team} Nutzer."],
        "vague": ["Preise?", "mehr Infos bitte", "Infos", "Interesse", "hallo"],
        "copy_paste": ["Wir sind ein führender Anbieter innovativer Lösungen und stehen für Qualität und Kundenzufriedenheit.",
                       "Unsere Mission ist es, Unternehmen weltweit mit skalierbaren End-to-End-Lösungen zu stärken."],
    },
    "FR": {
        "openers": ["Bonjour, ", "Bonjour à tous, ", "Salut, "],
        "closers": [" Merci !", " Merci d'avance.", " Cordialement"],
        "months": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
                   "octobre", "novembre", "décembre"],
        "neutral": ["Nous voulons améliorer la qualité des leads issus de nos publicités.",
                    "On aimerait savoir quelles campagnes amènent de bons leads.",
                    "Notre équipe commerciale trouve que les leads Meta sont mauvais.",
                    "Nous cherchons à connecter notre CRM à Google Ads."],
        "specific": ["Nous sommes une équipe de {team} personnes et voulons démarrer avant {month}.",
                     "Notre budget pub est d'environ {k} k€ par mois et on ne sait pas quels leads rapportent.",
                     "Notre contrat actuel se termine en {month}, pour {team} utilisateurs."],
        "vague": ["tarifs ?", "plus d'infos svp", "infos", "intéressé", "bonjour"],
        "copy_paste": ["Nous sommes un fournisseur leader de solutions innovantes au service de l'excellence.",
                       "Notre mission est d'accompagner les entreprises du monde entier avec des solutions de bout en bout."],
    },
    "NL": {
        "openers": ["Hallo, ", "Goedemiddag, ", "Hoi, "],
        "closers": [" Bedankt!", " Alvast bedankt.", " Groet"],
        "months": ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september",
                   "oktober", "november", "december"],
        "neutral": ["We willen de kwaliteit van onze leads uit advertenties verbeteren.",
                    "We willen weten welke campagnes goede leads opleveren.",
                    "Ons salesteam zegt dat de leads van Meta slecht zijn.",
                    "We zoeken een manier om ons CRM aan Google Ads te koppelen."],
        "specific": ["We hebben een team van {team} mensen en willen voor {month} live.",
                     "Ons advertentiebudget is ongeveer {k}k euro per maand en we zien niet welke leads omzet opleveren.",
                     "Ons contract met de huidige leverancier loopt af in {month}, {team} gebruikers."],
        "vague": ["prijzen?", "meer info aub", "info", "interesse", "hallo"],
        "copy_paste": ["Wij zijn een toonaangevende leverancier van innovatieve oplossingen voor al uw uitdagingen.",
                       "Onze missie is bedrijven wereldwijd te versterken met schaalbare end-to-end oplossingen."],
    },
}

# ---------------------------------------------------------------------------------------------
# 5.2 (ii) typos
# ---------------------------------------------------------------------------------------------

_KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def _keyboard_neighbours() -> dict[str, str]:
    """Letters adjacent to each key on a QWERTY keyboard (same row and the rows above/below)."""
    out = {}
    for r, row in enumerate(_KEYBOARD_ROWS):
        for c, ch in enumerate(row):
            near = [row[j] for j in (c - 1, c + 1) if 0 <= j < len(row)]
            for rr in (r - 1, r + 1):
                if 0 <= rr < len(_KEYBOARD_ROWS):
                    near += [_KEYBOARD_ROWS[rr][j] for j in (c - 1, c, c + 1) if 0 <= j < len(_KEYBOARD_ROWS[rr])]
            out[ch] = "".join(near)
    return out


_NEIGHBOURS = _keyboard_neighbours()


def add_typos(rng: np.random.Generator, text: str, char_rate: float) -> str:
    """Return ``text`` with keyboard typos on letters only (digits and punctuation are left alone).

    Number of typos is max(1, Binomial(letters, char_rate)); each is one of: swap with the next letter, drop
    the letter, double it, or replace it with a neighbouring QWERTY key (case kept)."""
    letters = [i for i, ch in enumerate(text) if ch.isalpha()]
    if not letters:
        return text
    n = max(1, int(rng.binomial(len(letters), char_rate)))
    chars = list(text)
    # apply from the right so earlier positions stay valid after insertions/deletions
    for pos in sorted(rng.choice(letters, size=min(n, len(letters)), replace=False), reverse=True):
        ch = chars[pos]
        op = int(rng.integers(0, 4))
        if op == 0 and pos + 1 < len(chars) and chars[pos + 1].isalpha():
            chars[pos], chars[pos + 1] = chars[pos + 1], ch
        elif op == 1:
            del chars[pos]
        elif op == 2:
            chars.insert(pos, ch)
        else:
            near = _NEIGHBOURS.get(ch.lower())
            if near:
                sub = near[int(rng.integers(0, len(near)))]
                chars[pos] = sub.upper() if ch.isupper() else sub
            else:                                   # accented letter: double it instead
                chars.insert(pos, ch)
    return "".join(chars)
