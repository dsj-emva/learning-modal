"""Boilerplate similarity detector (plan 2.5)."""
import pytest

from emva.boilerplate import BOILERPLATE_SNIPPETS, boilerplate_similarity, is_boilerplate, jaccard, tokens
from emva.constants import BOILERPLATE_SIMILARITY_THRESHOLD

LEADING = ("We are a leading provider of innovative solutions dedicated to delivering best-in-class value to our "
           "stakeholders across all verticals through synergy and excellence.")


def test_at_least_fifteen_snippets():
    assert len(BOILERPLATE_SNIPPETS) >= 15 and len(set(BOILERPLATE_SNIPPETS)) == len(BOILERPLATE_SNIPPETS)


def test_tokens_and_jaccard():
    assert tokens("Best-in-class, SYNERGY!") == {"best", "in", "class", "synergy"}
    assert tokens(None) == frozenset() and tokens(float("nan")) == frozenset()
    assert jaccard(frozenset("ab"), frozenset("bc")) == pytest.approx(1 / 3)
    assert jaccard(frozenset(), frozenset()) == 0.0


@pytest.mark.parametrize("text", [
    LEADING,
    LEADING.upper().replace(",", ""),                                  # case and punctuation do not matter
    "we are a leading provider of innovative solutions dedicated to delivering best in class value to "
    "stakeholders across verticals through synergy and excellence",   # a few words dropped
    "Through synergy and excellence we are a leading provider delivering best-in-class value to our "
    "stakeholders across all verticals, dedicated to innovative solutions.",  # reordered
    "Please consider the environment before printing this e-mail.",
])
def test_positive_boilerplate(text):
    assert is_boilerplate(text)


@pytest.mark.parametrize("text", [
    "We are a leading provider of scaffolding in Leeds and need better leads from Google Ads.",
    "Our mission is to empower our SDR team with better lead scoring before Q3.",
    "Lorem ipsum is what our current landing page still says; we need help with attribution.",
])
def test_near_misses_are_not_boilerplate(text):
    # each shares an opening phrase with a snippet (the legacy prefix rule flagged the first two)
    assert 0 < boilerplate_similarity(text) < BOILERPLATE_SIMILARITY_THRESHOLD
    assert not is_boilerplate(text)


@pytest.mark.parametrize("text", ["Interested in better lead quality from ads.", "pricing", "", None, "?"])
def test_negative_text(text):
    assert not is_boilerplate(text)


def test_threshold_is_configurable_and_validated():
    text = "We are a leading provider of innovative solutions for stakeholders across all verticals."
    s = boilerplate_similarity(text)
    assert is_boilerplate(text, threshold=s) and not is_boilerplate(text, threshold=min(1.0, s + 0.01))
    with pytest.raises(ValueError):
        is_boilerplate(text, threshold=0)
    with pytest.raises(ValueError):
        is_boilerplate(text, threshold=1.5)


def test_custom_snippet_list():
    assert is_boilerplate("sent from my phone", snippets=("Sent from my phone.",))
    assert not is_boilerplate(LEADING, snippets=("Sent from my phone.",))
