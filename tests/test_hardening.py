"""Phase 4 entry point: generated-block handling, cache key, and the v2 ceiling section end to end."""
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


def test_cache_key_changes_with_data_and_params(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    fp = hd._fingerprint(tmp_path)
    p1 = hd._cache_path("rolling", tmp_path, hd.Params(), fp)
    assert p1 == hd._cache_path("rolling", tmp_path, hd.Params(), hd._fingerprint(tmp_path))
    assert p1 != hd._cache_path("rolling", tmp_path, hd.Params(n_resamples=10), fp)
    assert p1 != hd._cache_path("ceiling", tmp_path, hd.Params(), fp)
    (tmp_path / "a.csv").write_text("x\n1\n2\n")
    assert hd._fingerprint(tmp_path) != fp


def test_every_plan_section_is_in_the_report():
    assert hd.SECTION_KEYS == ("rolling", "subsampling", "customer", "ceiling", "calibration", "regularisation",
                               "interactions", "standard")
    assert [s.heading.split()[0] for s in hd._sections()[:6]] == ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6"]


@pytest.fixture(scope="module")
def v2_frame() -> ro.EvalFrame:
    return ro.prepare(REPO / "data" / "v2")


def test_v2_ceiling_section_reports_the_persona_gap(v2_frame):
    sec = hd.ceiling_section(v2_frame, n_resamples=20)
    rows = sec.facts["rows"]
    assert {hd.PIPELINE, ce.ORACLE_FORMULA, ce.ORACLE_INTERACTIONS, ce.ORACLE_PERSONA, hd.PIPELINE_PERSONA} <= set(rows)
    gap = sec.facts["persona"]
    assert gap["comparison"] == f"{ce.ORACLE_PERSONA} − {ce.ORACLE_INTERACTIONS}"
    assert gap["_a"][0] > 0          # adding the true persona helps on the large legacy test set
    assert any("persona" in line for line in sec.lines)
