"""app.ingest: the Map & convert state behind the upload page (forms, review table, outcome table, TOML, saved and
built-in mappings, drafting with a fake client, conversion). Hand-built fixtures; no API call is ever made."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2
import pandas as pd
import pytest

from app import ingest, storage
from emva.context.agent import MODEL_ID
from emva.ingest.mapping import IGNORE, Review, load_mapping

from conftest import INGEST_FIXTURES, OLIST_DEALS, OLIST_MQL, REPO, olist_raw

MAPPINGS = sorted((REPO / "mappings").glob("*.toml"))
OLIST = REPO / "mappings" / "olist_funnel.toml"


@pytest.fixture
def frames() -> dict[str, pd.DataFrame]:
    """The small Olist-shaped fixture, primary file first."""
    raw = INGEST_FIXTURES / "olist_funnel"
    return ingest.raw_frames({n: (raw / n).read_bytes() for n in (OLIST_MQL, OLIST_DEALS)})


def _olist() -> ingest.DatasetMapping:
    return load_mapping(OLIST.read_text(encoding="utf-8"))


def test_raw_frames_reads_text_and_refuses_bad_input() -> None:
    f = ingest.raw_frames({"a.csv": b"x,y\n1,\n"})
    assert f["a.csv"].x.iloc[0] == "1" and pd.isna(f["a.csv"].y.iloc[0])
    with pytest.raises(ValueError, match="1 to 3"):
        ingest.raw_frames({})
    with pytest.raises(ValueError, match="1 to 3"):
        ingest.raw_frames({f"{i}.csv": b"x\n1\n" for i in range(4)})
    with pytest.raises(ValueError, match="empty.csv is not a readable CSV"):
        ingest.raw_frames({"empty.csv": b""})


@pytest.mark.parametrize("path", MAPPINGS, ids=lambda p: p.stem)
def test_committed_mappings_round_trip_through_the_form(path: Path) -> None:
    m = load_mapping(path.read_text(encoding="utf-8"))
    form = ingest.form_from_mapping(m)
    assert ingest.confirm(ingest.mapping_from_form(form)) == m
    assert not ingest.mapping_from_form(form).outcome_confirmed  # a person must confirm again in the app
    assert load_mapping(ingest.form_toml(form)) == ingest.mapping_from_form(form)


def test_form_lists_every_uploaded_column_once(frames: dict[str, pd.DataFrame]) -> None:
    form = ingest.form_from_mapping(_olist(), frames, "built-in")
    rows = [f.source for f in form.fields]
    assert len(rows) == len(set(rows))
    used = set(_olist().column_refs())
    unused = {f"deal.{c}" for c in frames[OLIST_DEALS].columns} | {f"mql.{c}" for c in frames[OLIST_MQL].columns}
    assert unused - used <= set(rows)
    added = [f for f in form.fields if f.source not in used]
    assert added and all(f.target == IGNORE for f in added)
    assert [f.source for f in form.value_map_rows] == ["mql.origin"]
    assert ingest.value_map_frame(form).query("`raw value` == 'paid_search'").sets.iloc[0] == \
        "utm_source = google; utm_medium = cpc"


def test_blank_form_starts_with_everything_ignored(frames: dict[str, pd.DataFrame]) -> None:
    form = ingest.blank_form(frames)
    assert [s.file for s in form.sources] == [OLIST_MQL, OLIST_DEALS]
    assert form.sources[1].join_on == "mql_id"  # the one column both files share
    assert all(f.target == IGNORE for f in form.fields)
    assert len(form.fields) == sum(len(df.columns) for df in frames.values())
    assert ingest.form_toml(form) is None
    with pytest.raises(ValueError, match=r"choose the column for created_at \(required\)"):
        ingest.mapping_from_form(form)


def test_filling_a_blank_form_by_hand_gives_a_convertible_mapping(frames: dict[str, pd.DataFrame]) -> None:
    f = ingest.blank_form(frames)
    f = ingest.with_role(f, "created_at", f"{f.sources[0].name}.first_contact_date")
    f = ingest.with_role(f, "won_at", f"{f.sources[1].name}.won_date")
    with pytest.raises(ValueError, match="outcome column"):
        ingest.mapping_from_form(f)
    f = ingest.apply_outcome(f, "presence", f"{f.sources[1].name}.won_date", None, "as_of")
    with pytest.raises(ValueError, match="as_of"):
        ingest.mapping_from_form(f)
    f = replace(f, as_of="2018-11-15", test_from="2018-03-01")
    table = ingest.review_frame(f)
    table.loc[table.source.str.endswith(".origin"), "target"] = "utm_source"
    m = ingest.mapping_from_form(ingest.apply_review(f, table))
    with pytest.raises(ValueError, match="draft"):
        ingest.convert_raw(frames, m, "2026-09-26")
    with pytest.raises(ValueError, match="utm_source"):  # the raw values are not form options: a person must map
        ingest.convert_raw(frames, ingest.confirm(m), "2026-09-26")


def test_review_table_flags_low_confidence_and_applies_edits(frames: dict[str, pd.DataFrame]) -> None:
    form = ingest.form_from_mapping(_olist(), frames)
    form = replace(form, fields=tuple(replace(f, review=Review("an id?", "low")) if f.source == "mql.landing_page_id"
                                      else f for f in form.fields))
    table = ingest.review_frame(form)
    assert list(table.columns) == list(ingest.REVIEW_COLUMNS)
    assert table.set_index("source").loc["mql.landing_page_id", "check"] == "check"
    assert set(table.check) == {"", "check"}
    table.loc[table.source == "mql.landing_page_id", ["target", "confidence", "reason"]] = [IGNORE, "high", "an id"]
    edited = ingest.apply_review(form, table)
    row = next(f for f in edited.fields if f.source == "mql.landing_page_id")
    assert (row.target, row.review.confidence, row.review.reason) == (IGNORE, "high", "an id")
    assert edited.value_map_rows == form.value_map_rows  # value maps are not in the table
    table.loc[0, "target"] = "not_a_column"
    with pytest.raises(ValueError, match="unknown target"):
        ingest.apply_review(form, table)


def test_outcome_table_for_stage_and_won_flag(frames: dict[str, pd.DataFrame]) -> None:
    crm_raw = INGEST_FIXTURES / "crm_opportunities"
    crm = ingest.raw_frames({n: (crm_raw / n).read_bytes() for n in ("sales_pipeline.csv", "accounts.csv")})
    form = ingest.form_from_mapping(load_mapping((REPO / "mappings" / "crm_opportunities.toml").read_text()), crm)
    table, truncated = ingest.outcome_frame(crm, form)
    assert not truncated and set(table.value) >= {"Won", "Lost", "Engaging", "Prospecting"}
    assert dict(zip(table.value, table.meaning))["Engaging"] == "Contacted"
    table.loc[table.value == "Engaging", "meaning"] = "Lost"
    edited = ingest.apply_outcome(form, "stage", form.outcome_column, table, "error")
    assert edited.outcome_values["Engaging"] == "Lost"
    flag = pd.DataFrame({"value": ["Won", "Lost", "Engaging"], "meaning": ["Won", "Lost", ""]})
    m = ingest.mapping_from_form(ingest.apply_outcome(form, "won_flag", form.outcome_column, flag, "error"))
    assert (m.outcome.kind, m.outcome.won_values, m.outcome.lost_values) == ("won_flag", ("Won",), ("Lost",))
    with pytest.raises(ValueError, match="not one of"):
        ingest.apply_outcome(form, "won_flag", form.outcome_column,
                             pd.DataFrame({"value": ["Won"], "meaning": ["Proposal"]}), "error")
    values, _ = ingest.column_values(frames, ingest.form_from_mapping(_olist(), frames), "mql.origin")
    assert "paid_search" in values


def test_toml_edits_are_validated(frames: dict[str, pd.DataFrame]) -> None:
    with pytest.raises(ValueError, match="not valid TOML"):
        ingest.form_from_toml("name = ", frames)
    with pytest.raises(ValueError, match="unknown key"):
        ingest.form_from_toml(OLIST.read_text() + "\ntypo = 1\n", frames)
    form = ingest.form_from_toml(OLIST.read_text(), frames)
    assert form.origin == "edited as TOML" and form.name == "olist_funnel"
    assert ingest.expr_text(form.lead["created_at"]) == "mql.first_contact_date"
    hotel = ingest.form_from_mapping(load_mapping((REPO / "mappings" / "hotel_bookings.toml").read_text()))
    assert not ingest.is_plain(hotel.lead["created_at"])
    assert ingest.expr_text(hotel.lead["created_at"]).startswith("minus_days(date_from_parts(")


def test_mapping_choices_list_saved_then_builtin(tmp_path: Path) -> None:
    choices = ingest.mapping_choices(tmp_path)
    assert [c.key for c in choices] == [f"builtin:{p.stem}" for p in MAPPINGS] and all(c.builtin for c in choices)
    conv = ingest.convert_raw(ingest.raw_frames(olist_raw()), _olist(), "2026-09-26")
    storage.save_dataset(tmp_path, "olist-a", conv.files)
    choices = ingest.mapping_choices(tmp_path)
    assert choices[0].key == "saved:olist-a" and not choices[0].builtin
    assert choices[0].label == "olist-a · saved with a dataset · www.kaggle.com"
    assert ingest.load_choice(choices[0]) == _olist()  # the saved mapping is the confirmed one, unchanged
    assert ingest.missing_files(_olist(), {OLIST_MQL: pd.DataFrame()}) == [OLIST_DEALS]


def test_convert_raw_adds_the_confirmed_mapping_and_validates() -> None:
    conv = ingest.convert_raw(ingest.raw_frames(olist_raw()), _olist(), "2026-09-26")
    assert set(conv.files) == {*storage.REQUIRED_FILES, storage.PEOPLE_FILE, *storage.METADATA_FILES}
    text = conv.files[storage.MAPPING_FILE].decode()
    assert "Confirmed in Keel" in text and "2026-09-26" in text
    assert load_mapping(text, require_confirmed=True) == _olist()
    assert conv.report.ok and conv.meta["leads"] == 320 and conv.meta["as_of"] == "2018-11-15"
    cov = ingest.coverage_summary(conv.meta)
    assert cov.total == 39 == cov.data + cov.constant + cov.unfilled and cov.data >= 1
    frame = ingest.coverage_frame(conv.meta)
    assert len(frame) == 39 and (frame.status == "data").sum() == cov.data
    derived = ingest.derived_counts(conv.meta)
    assert derived and all(n > 0 and conv.meta[k] == n for k, n, _ in derived)
    keys = [k for k, _, _ in derived]
    assert "lost_dated_at_as_of" in keys and "crm_times_reordered" not in keys


def test_convert_raw_saves_uploaded_rules_after_checking_them() -> None:
    """R38: a Map & convert upload of status_quo_rules.json goes through the same checks and is saved with the data."""
    rules = (REPO / "data" / "v1" / "status_quo_rules.json").read_bytes()
    conv = ingest.convert_raw(ingest.raw_frames(olist_raw()), _olist(), "2026-09-26", rules)
    assert conv.files[storage.RULES_FILE] == rules and conv.report.ok
    assert not [w for w in conv.report.warnings if w.file == storage.RULES_FILE]
    bad = ingest.convert_raw(ingest.raw_frames(olist_raw()), _olist(), "2026-09-26", b'{"value_rules": []}')
    assert {e.column for e in bad.report.errors if e.file == storage.RULES_FILE} >= {"base_value_by_form",
                                                                                     "lead_score"}
    no_a = json.loads(rules)
    del no_a["base_value_by_form"]["A"]  # converted leads all have form variant A
    missing = ingest.convert_raw(ingest.raw_frames(olist_raw()), _olist(), "2026-09-26", json.dumps(no_a).encode())
    assert not missing.report.ok


# --- drafting: fake client, no API ------------------------------------------------------------------------------------

def _col(file: str, column: str, target: str, confidence: str = "high", vm: list | None = None) -> dict:
    return {"file": file, "column": column, "target": target, "feature_kind": "none", "value_map": vm or [],
            "reason": "hand-built test reply", "confidence": confidence}


def hand_built_reply(created_at: str = "first_contact_date") -> dict:
    """A reply in the draft contract, written by hand for these tests (not model output)."""
    ignored = ("seller_id", "sdr_id", "sr_id", "business_segment", "lead_type", "lead_behaviour_profile", "has_company",
               "has_gtin", "average_stock", "business_type", "declared_product_catalog_size")
    return {
        "sources": [{"file": OLIST_MQL, "join_on": ""}, {"file": OLIST_DEALS, "join_on": "mql_id"}],
        "columns": [_col(OLIST_MQL, "mql_id", "lead.lead_id"), _col(OLIST_MQL, created_at, "lead.created_at"),
                    _col(OLIST_MQL, "landing_page_id", "utm_content", "low"),
                    _col(OLIST_MQL, "origin" if created_at != "origin" else "first_contact_date", "ignore"),
                    _col(OLIST_DEALS, "won_date", "lead.won_at"),
                    _col(OLIST_DEALS, "declared_monthly_revenue", "lead.deal_value", "low")]
        + [_col(OLIST_DEALS, c, "ignore") for c in ignored],
        "outcome": {"kind": "presence", "file": OLIST_DEALS, "column": "won_date", "stage_map": [], "won_values": [],
                    "lost_values": [], "reason": "hand-built", "confidence": "medium"},
    }


class FakeClient:
    """Stands in for ``anthropic.Anthropic``: each ``messages.create`` pops the next scripted reply or exception."""

    def __init__(self, *script: object) -> None:
        self.script = list(script)
        self.calls = 0
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def reply_message(obj: dict) -> SimpleNamespace:
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(obj))])


def _no_client() -> anthropic.Anthropic:
    raise AssertionError("a cached draft must not build a client")


def test_draft_without_api_key_offers_manual_fill(tmp_path: Path, frames: dict[str, pd.DataFrame],
                                                  caplog: pytest.LogCaptureFixture) -> None:
    def missing_key() -> anthropic.Anthropic:
        raise SystemExit("ANTHROPIC_API_KEY and ANTHROPIC_WORKSPACE_ID must be set")

    with caplog.at_level(logging.WARNING, logger="app.ingest"):
        out = ingest.draft(ingest.profile(frames), tmp_path, "olist", client_factory=missing_key)
    assert out.mapping is None and out.no_key and out.error == ingest.NO_KEY_MESSAGE
    assert "must be set" in caplog.text  # logged, not swallowed
    assert not (tmp_path / ingest.DRAFT_CACHE).exists()


def test_draft_with_a_fake_client_then_from_the_cache(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> None:
    profiles = ingest.profile(frames)
    client = FakeClient(reply_message(hand_built_reply()))
    assert not ingest.draft_is_cached(profiles, tmp_path)
    out = ingest.draft(profiles, tmp_path, "olist_draft", client_factory=lambda: client)
    assert client.calls == 1 and out.error is None and not out.from_cache
    assert out.mapping.name == "olist_draft" and not out.mapping.outcome_confirmed
    assert (tmp_path / ingest.DRAFT_CACHE).is_file() and ingest.draft_is_cached(profiles, tmp_path)
    assert MODEL_ID in out.origin
    again = ingest.draft(profiles, tmp_path, "olist_draft", client_factory=_no_client)
    assert again.from_cache and again.mapping == out.mapping and "no request made" in again.origin
    form = ingest.form_from_mapping(out.mapping, frames, out.origin)
    low = ingest.review_frame(form).set_index("source").loc[f"{form.sources[0].name}.landing_page_id", "check"]
    assert low == "check"


def test_draft_api_error_is_logged_and_shown(tmp_path: Path, frames: dict[str, pd.DataFrame],
                                             caplog: pytest.LogCaptureFixture) -> None:
    denied = anthropic.APIStatusError("denied", response=httpx2.Response(
        401, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")), body=None)
    with caplog.at_level(logging.ERROR, logger="app.ingest"):
        out = ingest.draft(ingest.profile(frames), tmp_path, "x", client_factory=lambda: FakeClient(denied))
    assert out.mapping is None and not out.no_key and "APIStatusError" in out.error
    assert "mapping draft failed" in caplog.text


def test_draft_whose_dates_cannot_be_derived_asks_for_them(tmp_path: Path, frames: dict[str, pd.DataFrame]) -> None:
    client = FakeClient(reply_message(hand_built_reply(created_at="origin")))  # a text column as created_at
    out = ingest.draft(ingest.profile(frames), tmp_path, "x", client_factory=lambda: client)
    assert out.mapping is not None and out.error is None and "not an ISO date column" in out.dates_note
    assert (out.mapping.as_of, out.mapping.test_from) == ingest.PLACEHOLDER_DATES


def test_draft_lets_other_value_errors_through(tmp_path: Path, frames: dict[str, pd.DataFrame],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the dates are allowed to fail softly; any other ValueError is not mistaken for a date problem."""
    def broken(*args: object, **kwargs: object) -> None:
        raise ValueError("not a date problem")

    monkeypatch.setattr(ingest, "draft_mapping", broken)
    with pytest.raises(ValueError, match="not a date problem"):
        ingest.draft(ingest.profile(frames), tmp_path, "x", client_factory=lambda: FakeClient())
