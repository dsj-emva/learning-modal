import json

import numpy as np
import pandas as pd
import pytest

from conftest import DATA_V1
from emva.eval.status_quo import job_title_points, load_rules, pages_visited_points, status_quo_value

RULES = load_rules(DATA_V1 / "status_quo_rules.json")


def _lead(**kw) -> dict:
    base = dict(form_variant="A", a_country="DE", utm_medium="cpc", utm_source="google",
                a_job_title=None, a_company_size=None, pages_visited=np.nan)
    base.update(kw)
    return base


@pytest.mark.parametrize("title,points", [
    (None, 0), ("", 0), ("  ", 0), ("Student", 0), ("CEO", 30), ("Head of Growth", 25),
    ("Marketing Manager", 15), ("Analyst", 10),
    ("Head of Student Recruitment", 0),  # first listed keyword wins: "student" is listed first
    ("Sales Director", 30),
])
def test_job_title_points(title, points):
    assert job_title_points(title, RULES) == points


@pytest.mark.parametrize("pages,points", [(np.nan, 0), (1, 5), (2, 5), (3, 10), (4, 10), (5, 20), (10, 20)])
def test_pages_visited_points(pages, points):
    assert pages_visited_points(pages, RULES) == points


def test_hand_built_leads():
    X = pd.DataFrame([
        # A form, no title, no size, no pages: 0 points -> tier C (0.4): 150 * 0.4
        _lead(),
        # C form, UK, LinkedIn: 600*1.2*1.3; CEO 30 + 1000+ 30 + 5 pages 20 = 80 -> tier A (2.0)
        _lead(form_variant="C", a_country="UK", utm_medium="paid_social", utm_source="linkedin",
              a_job_title="CEO", a_company_size="1000+", pages_visited=6),
        # B form, US: 300*1.2; manager 15 + 11-50 10 + 3 pages 10 = 35 -> tier B (1.0)
        _lead(form_variant="B", a_country="US", a_job_title="Ops Manager", a_company_size="11-50", pages_visited=3),
        # D form, LinkedIn lead form is still channel meta_leadads, no LinkedIn uplift; 25 + 20 + 5 = 50 points -> tier B
        _lead(form_variant="D", utm_medium="lead_form", utm_source="linkedin", a_job_title="Head of Ops",
              a_company_size="51-200", pages_visited=1, a_country="NL"),
        # blank typed size scores 0; 55 points exactly is tier A
        _lead(form_variant="B", a_job_title="VP Sales", a_company_size="", pages_visited=5),
        _lead(form_variant="B", a_job_title="VP Sales", a_company_size="1-10", pages_visited=5),
    ])
    out = status_quo_value(X, RULES)
    assert out.sq_points.tolist() == [0, 80, 35, 50, 50, 55]
    assert out.sq_tier.tolist() == ["C", "A", "B", "B", "B", "A"]
    assert out.sq_value.tolist() == pytest.approx([60, 600 * 1.2 * 1.3 * 2, 360, 200, 300, 600])


def test_unknown_form_variant_is_an_error():
    with pytest.raises(ValueError):
        status_quo_value(pd.DataFrame([_lead(form_variant="Z")]), RULES)


def test_rules_file_shape_is_what_the_reconstruction_reads():
    with open(DATA_V1 / "status_quo_rules.json") as f:
        raw = json.load(f)
    assert {r["field"] for r in raw["value_rules"]} <= {"country", "channel"}
    tiers = [t["min_points"] for t in raw["lead_score"]["tiers"]]
    assert tiers == sorted(tiers, reverse=True) and tiers[-1] == 0
