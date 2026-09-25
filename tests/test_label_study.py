"""Phase 1 evaluation helpers: label study sections and the ground-truth reference script."""
import subprocess
import sys

import pytest

from emva.eval import label_study

from conftest import REPO


def test_band_weight_cis_reproduce_baseline_weights(data_v1):
    text = label_study.band_weight_cis(data_v1, n_refits=5, seed=0)
    rows = {line.split(" | ")[0].strip("| "): line for line in text.splitlines() if line.startswith("| ")}
    assert "| 5588 | " in rows["legacy"] and "| 0.873 [" in rows["legacy"] and "| 0.703 [" in rows["legacy"]
    assert "| 4049 | " in rows["horizon H=120"] and "| 0.971 [" in rows["horizon H=120"]


def test_flag_combinations_table(data_v1):
    text = label_study.flag_combinations(data_v1, n_resamples=5, seed=0)
    assert "| legacy | 5588 | 847 | 15.2% | 0.873 | 0.703 |" in text
    assert "| horizon H=120 | 4049 | 752 |" in text
    assert "| horizon H=120 +ghosted | 5115 | 752 |" in text
    assert "| horizon H=120 +stalled-as-lost | 4562 | 752 |" in text
    assert "| horizon H=120 +ghosted +stalled-as-lost | 5628 | 752 |" in text
    assert "| legacy | 6.2% (n=2357) | 12.7% (n=1260) | 27.8% (n=944) | 27.7% (n=692) |" in text
    assert "| horizon H=120 | 7.8% (n=1615) | 15.7% (n=922) | 33.2% (n=723) | 33.5% (n=514) |" in text


def test_horizon_sensitivity_handles_horizons_with_no_mature_test_leads(data_v1, monkeypatch):
    monkeypatch.setattr(label_study, "SENSITIVITY_HORIZONS", (120, 180))
    text = label_study.horizon_sensitivity(data_v1)
    assert "| 120 | 4049 | 752 |" in text and "| 458 |" in text
    assert "| 180 | 3461 |" in text and text.count("| 0 |") >= 1


@pytest.mark.parametrize("module", ["emva.eval.ground_truth_reference"])
def test_ground_truth_reference_reproduces_the_plan_figure(module, data_v1):
    # Run as a script (nothing may import it). Few refits: only deterministic numbers are asserted.
    proc = subprocess.run([sys.executable, "-m", module, "--data", str(data_v1), "--n-refits", "5"], cwd=REPO,
                          capture_output=True, text=True, check=True)
    lines = proc.stdout.splitlines()
    legacy = next(line for line in lines if line.startswith("| training rows labelled by legacy"))
    assert "| 39.4% (n=692) |" in legacy
    # R1: the planted time to close gives P(close within 120 d | Won) = 0.843, or 0.680 for 1000+ ...
    assert "= 0.843 (other bands), 0.680 (1000+)" in proc.stdout
    # ... and the derived 120-day rate matches the observed horizon label by band (the derivation check).
    assert "| 1000+ | 275 | 37.0% | 25.2% | 25.1% |" in lines
    # True conditional 120-day effects, then the legacy and horizon weights with their CIs, per band.
    for band, true_eff in [("11-50", "0.378"), ("51-200", "1.099"), ("201-1000", "1.043"), ("1000+", "0.532")]:
        row = next(line for line in lines if line.startswith(f"| {band} | {true_eff} | "))
        assert row.count("[") == 2


def test_nothing_imports_the_ground_truth_reference():
    hits = subprocess.run(["grep", "-rln", "--include=*.py", "ground_truth_reference", "emva", "scripts", "baseline"],
                          cwd=REPO, capture_output=True, text=True).stdout.split()
    assert sorted(hits) == ["emva/eval/ground_truth_reference.py", "emva/eval/label_study.py"]
    assert "import" not in next(line for line in (REPO / "emva/eval/label_study.py").read_text().splitlines()
                                if "ground_truth_reference" in line)


def test_ground_rule_2_grep():
    # Only emva/eval/ may mention ground truth; the one known 0.45 constant is removed in plan 2.4.
    # -I skips compiled __pycache__ files, which embed the strings of the modules they come from.
    out = subprocess.run(["grep", "-rnI", r"0.45\|ground_truth", "emva/"], cwd=REPO, capture_output=True,
                         text=True).stdout.splitlines()
    outside = [line for line in out if not line.startswith("emva/eval/")]
    assert outside == ["emva/value.py:19:DEAL_LOG_RESIDUAL_SD_FROM_GENERATOR = 0.45"]
