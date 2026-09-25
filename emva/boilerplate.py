"""Boilerplate detector for the free-text answer (plan 2.5).

Replaces the legacy three-prefix match (``constants.COPY_PASTE_PREFIXES``) with a similarity
check: a text is boilerplate when its normalised token set has Jaccard similarity at or above a
threshold (``constants.BOILERPLATE_SIMILARITY_THRESHOLD``) with any snippet below. Word order,
punctuation, case and a few changed words no longer matter; sharing an opening phrase with a
snippet is not enough.

The first three snippets are the corporate filler observed verbatim in the v1 leads' answers
(observed input text, not generator output); the rest are generic mission-statement, marketing
and email filler of the kind pasted into "what do you want to solve" fields.
"""
from __future__ import annotations

import re
from functools import lru_cache

from emva.constants import BOILERPLATE_SIMILARITY_THRESHOLD

BOILERPLATE_SNIPPETS: tuple[str, ...] = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore.",
    "We are a leading provider of innovative solutions dedicated to delivering best-in-class value to our "
    "stakeholders across all verticals through synergy and excellence.",
    "Our mission is to empower businesses worldwide with cutting-edge, scalable, end-to-end solutions that "
    "drive growth, efficiency and customer delight.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "We are a customer-centric company committed to excellence, integrity and innovation in everything we do.",
    "Our vision is to be the most trusted partner for our clients by delivering world-class products and services.",
    "We leverage best-of-breed technologies to deliver holistic, future-proof solutions for the enterprise.",
    "Founded on a passion for quality, we help organisations of every size unlock their full potential.",
    "We are a dynamic, fast-growing team passionate about creating value for our customers and partners.",
    "Our award-winning team delivers bespoke, results-driven solutions tailored to your unique business needs.",
    "We pride ourselves on outstanding customer service and a commitment to continuous improvement.",
    "Synergistic, scalable and seamless: we transform the way businesses engage with their customers.",
    "This email and any attachments are confidential and intended solely for the use of the addressee.",
    "Please consider the environment before printing this email.",
    "Your satisfaction is our top priority and we look forward to building a long-lasting relationship.",
    "We deliver innovative, reliable and cost-effective solutions across a wide range of industries.",
)


def tokens(text: object) -> frozenset[str]:
    """Lower-cased alphanumeric word set of ``text`` (empty for non-strings)."""
    if not isinstance(text, str):
        return frozenset()
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|a ∩ b| / |a ∪ b|; 0 when both are empty."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


@lru_cache(maxsize=None)
def _snippet_tokens(snippets: tuple[str, ...]) -> tuple[frozenset[str], ...]:
    """Token sets of ``snippets``, computed once per snippet list."""
    return tuple(tokens(s) for s in snippets)


def boilerplate_similarity(text: object, snippets: tuple[str, ...] = BOILERPLATE_SNIPPETS) -> float:
    """Largest token-set Jaccard similarity between ``text`` and any of ``snippets`` (0 for empty text)."""
    t = tokens(text)
    return max((jaccard(t, s) for s in _snippet_tokens(snippets)), default=0.0) if t else 0.0


def is_boilerplate(text: object, threshold: float = BOILERPLATE_SIMILARITY_THRESHOLD,
                   snippets: tuple[str, ...] = BOILERPLATE_SNIPPETS) -> bool:
    """True when ``boilerplate_similarity(text) >= threshold``."""
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    return boilerplate_similarity(text, snippets) >= threshold
