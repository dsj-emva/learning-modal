"""Phase 4 entry point: generated-block handling, cache key, derived prose, and the v2 ceiling section end to end."""
from __future__ import annotations

from pathlib import Path

import pytest

from emva.eval import ceiling as ce
from emva.eval import hardening as hd
from emva.eval import rolling as ro

REPO = Path(__file__).resolve().parents[1]


def test_write_block_creates_replaces_and_keeps_hand_written_text(tmp_path):
    p = tmp_path / "phase4.md"
    hd.write_block(p, "one\n")
    assert p.read_text() == f"{hd.MARK_BEGIN}\none\n{hd.MARK_END}\n"
    p.write_text("# Title\n\nintro\n\n" + p.read_text() + "\n## Orchestrator decisions\n\nkeep me\n")
    hd.write_block(p, "two\n")
    text = p.read_text()
    assert text.startswith("# Title\n\nintro\n\n") and "two\n" in text and "one\n" not in text
    assert text.rstrip().endswith("keep me") and text.count(hd.MARK_BEGIN) == 1
    q = tmp_path / "other.md"
    q.write_text("hand-written only\n")
    hd.write_block(q, "three\n")
    assert q.read_text() == f"hand-written only\n\n{hd.MARK_BEGIN}\nthree\n{hd.MARK_END}\n"


def test_cache_key_changes_with_data_contents_params_and_sources(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.csv").write_text("x\n1\n")
    src = tmp_path / "module.py"
    src.write_text("X = 1\n")
    fp = hd._fingerprint(data, [src])
    p1 = hd._cache_path("rolling", data, hd.Params(), fp)
    assert p1 == hd._cache_path("rolling", data, hd.Params(), hd._fingerprint(data, [src]))
    assert p1 != hd._cache_path("rolling", data, hd.Params(n_resamples=10), fp)
    assert p1 != hd._cache_path("ceiling", data, hd.Params(), fp)
    src.write_text("X = 2\n")                           # a source change moves the key
    assert hd._fingerprint(data, [src]) != fp
    src.write_text("X = 1\n")
    assert hd._fingerprint(data, [src]) == fp
    (data / "a.csv").write_text("x\n2\n")               # same size and name, different contents
    assert hd._fingerprint(data, [src]) != fp
    assert (hd.REPO_ROOT / "requirements.txt") in hd.source_files()
    assert "scikit-learn" in hd.library_versions()


def test_rolling_intro_is_derived_from_the_parameters():
    text = hd.rolling_intro(120)
    assert "H = 120" in text and "2026-05-27T00:00:00+00:00" in text
    assert "2026-05-01 window is partial (26 days)" in text
    assert "2026-06-01 and 2026-07-01 windows have no mature test lead" in text
    assert "no mature test lead" not in hd.rolling_intro(30)      # every window is mature with a short H


def test_every_plan_section_is_in_the_report():
    assert hd.SECTION_KEYS == (hd.K_ROLLING, hd.K_SUBSAMPLING, hd.K_CUSTOMER, hd.K_CEILING, hd.K_CALIBRATION,
                               hd.K_REGULARISATION, hd.K_INTERACTIONS, hd.K_STANDARD)
    assert [s.heading.split()[0] for s in hd._sections()[:6]] == ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6"]


@pytest.fixture(scope="module")
def v2_frame() -> ro.EvalFrame:
    return ro.prepare(REPO / "data" / "v2")


def test_v2_ceiling_section_reports_the_persona_gap(v2_frame):
    sec = hd.ceiling_section(v2_frame, n_resamples=20)
    rows = sec.facts[hd.F_CEILING_ROWS]
    assert {hd.PIPELINE, ce.ORACLE_FORMULA, ce.ORACLE_INTERACTIONS, ce.ORACLE_PERSONA, hd.PIPELINE_PERSONA} <= set(rows)
    assert rows[hd.PIPELINE_PERSONA]["columns"] == rows[hd.PIPELINE]["columns"] + 1     # only the persona column
    assert rows[ce.ORACLE_PERSONA]["columns"] == rows[ce.ORACLE_INTERACTIONS]["columns"] + 1
    gap = sec.facts[hd.F_GAPS][hd.gap_key(ce.ORACLE_INTERACTIONS, ce.ORACLE_PERSONA)]
    assert gap[hd.E_A].point > 0      # adding the true persona helps on the large legacy test set
    assert gap[hd.E_A].lo <= gap[hd.E_A].point <= gap[hd.E_A].hi
    assert any("persona" in line for line in sec.lines)
