"""emva.troas: tROAS eligibility calculator (plan 3.4)."""
import subprocess
import sys

import numpy as np
import pytest

from emva.troas import (
    DAYS_PER_MONTH,
    GOOGLE,
    GOOGLE_TROAS_MIN_CONVERSIONS_30D,
    META,
    META_MIN_EVENTS_PER_WEEK,
    campaign_leads,
    eligibility,
    volumes_from_data,
)

from conftest import REPO


def test_thresholds():
    assert (GOOGLE.minimum, GOOGLE.days) == (GOOGLE_TROAS_MIN_CONVERSIONS_30D, 30) == (30.0, 30)
    assert (META.minimum, META.days) == (META_MIN_EVENTS_PER_WEEK, 7) == (50.0, 7)


def test_campaign_leads_even_and_split():
    assert campaign_leads(300, campaigns=3).tolist() == [100, 100, 100]
    assert campaign_leads(300, split=[2, 1]).tolist() == [200, 100]
    assert campaign_leads(300, split=[0.5, 0.3, 0.2]) == pytest.approx([150, 90, 60])


@pytest.mark.parametrize("kwargs", [{}, {"campaigns": 2, "split": [1]}, {"campaigns": 0}, {"split": []},
                                    {"split": [1, 0]}])
def test_campaign_leads_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        campaign_leads(100, **kwargs)
    with pytest.raises(ValueError):
        campaign_leads(-1, campaigns=1)


def test_google_flags_close_stage_below_30_wins():
    # 2,000 leads a year, one campaign, 13% win rate: 21.4 wins per 30 days < 30
    df = eligibility(campaign_leads(2000 / 12, campaigns=1), 0.13, GOOGLE)
    wins = 2000 / 12 * 30 / DAYS_PER_MONTH * 0.13
    assert df.close_conversions.iloc[0] == pytest.approx(wins)
    assert df.submit_ok.iloc[0] and not df.close_ok.iloc[0]
    assert df.close_shortfall.iloc[0] == pytest.approx(30 / wins)


def test_meta_counts_per_week_and_flags_both_stages():
    df = eligibility(np.array([1000.0, 100.0]), 0.5, META, names=["big", "small"])
    per_week = np.array([1000.0, 100.0]) * 7 / DAYS_PER_MONTH
    assert df.submit_conversions.tolist() == pytest.approx(per_week.tolist())
    assert df.submit_ok.tolist() == [True, False]
    assert df.close_ok.tolist() == [True, False]  # 115 and 11.5 wins a week
    assert df.campaign.tolist() == ["big", "small"]


def test_exactly_at_threshold_is_ok_and_zero_rate_is_infinitely_short():
    leads = 30 * DAYS_PER_MONTH / 30  # exactly 30 lead conversions per 30 days
    df = eligibility([leads], 1.0, GOOGLE)
    assert df.submit_ok.iloc[0] and df.close_ok.iloc[0]
    assert np.isinf(eligibility([leads], 0.0, GOOGLE).close_shortfall.iloc[0])
    with pytest.raises(ValueError):
        eligibility([leads], 1.5, GOOGLE)


def test_volumes_from_v1(data_v1):
    volumes, rate = volumes_from_data(data_v1)
    assert 0.10 < rate < 0.16
    assert set(volumes) == {"google", "meta"}
    assert len(volumes["google"]) == 4 and (volumes["google"] > 40).all()


def test_cli_by_numbers_and_by_data(data_v1):
    proc = subprocess.run([sys.executable, "-m", "emva.troas", "--leads-per-month", "166.7", "--positive-rate", "0.13",
                           "--split", "0.5,0.3,0.2", "--platform", "google"], cwd=REPO, capture_output=True, text=True,
                          check=True)
    assert "campaign 3" in proc.stdout and "BELOW" in proc.stdout and "Meta" not in proc.stdout
    proc = subprocess.run([sys.executable, "-m", "emva.troas", "--data", str(data_v1)], cwd=REPO, capture_output=True,
                          text=True, check=True)
    assert "all campaigns pooled" in proc.stdout and "brand_search" in proc.stdout
    bad = subprocess.run([sys.executable, "-m", "emva.troas", "--leads-per-month", "10"], cwd=REPO,
                         capture_output=True, text=True)
    assert bad.returncode == 2
