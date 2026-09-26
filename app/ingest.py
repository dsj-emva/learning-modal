"""Map & convert from the app: raw foreign CSVs -> a reviewed mapping -> EMVA's five files. No Streamlit.

The upload page keeps one ``MappingForm`` in the Streamlit session: the editable state of a mapping under review,
which may be incomplete (a mapping filled by hand starts with every column set to ignore and no ``created_at``).
``mapping_from_form`` turns it into an ``emva.ingest.DatasetMapping`` (``ValueError`` with a message fit for the page
while it is incomplete); ``form_from_mapping`` goes the other way, for a draft, a saved or built-in mapping, or TOML
text edited by hand. Everything about the data goes through ``emva.ingest``: ``frames_from_bytes``, ``profile``,
``draft_mapping`` (Claude Haiku, drafts cached in ``DATA_DIR/ingest/draft_cache.json``), ``convert`` and
``load_mapping`` / ``dump_mapping``; the result is checked with the same ``app.validation.validate_files`` as an
upload in EMVA's format and saved with ``app.storage.save_dataset``, including ``mapping.toml`` and ``dataset.json``.

The model call runs in the server process (ADR 0010: ``emva.context.agent.make_client``); the training job keeps
its filtered environment without ``ANTHROPIC_*`` (ADR 0019). A missing key or a failed request is reported as a
``DraftOutcome`` with a message, and the page offers the table to fill by hand.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

import anthropic
import pandas as pd

from app.storage import DATASET_META_FILE, MAPPING_FILE, REPO_ROOT, RULES_FILE, list_mappings
from app.validation import ValidationReport, validate_files
from emva.context.agent import MODEL_ID, make_client
from emva.context.cache import ReplyCache
from emva.context.contract import ContractError
from emva.ingest.convert import DERIVED_COUNTS, convert, frames_from_bytes
from emva.generic import ExtraKind
from emva.ingest.draft import default_dates, draft_mapping, feature_name, is_cached, source_name
from emva.ingest.mapping import (
    COLUMN_OP,
    CONFIDENCES,
    IGNORE,
    LEAD_ROLES,
    MAX_SOURCES,
    OUTCOME_KINDS,
    STAGES,
    TARGETS,
    DatasetMapping,
    Expr,
    FeatureMap,
    FieldMap,
    LeadColumns,
    Outcome,
    Review,
    Source,
    dump_mapping,
    load_mapping,
)
from emva.ingest.profile import ColumnProfile, profile

log = logging.getLogger(__name__)

MAX_RAW_FILES: int = MAX_SOURCES
DRAFT_CACHE: Path = Path("ingest") / "draft_cache.json"
BUILTIN_MAPPINGS_DIR: Path = REPO_ROOT / "mappings"
# Choices in the review table's target column: every schema target, then "ignore".
TARGET_OPTIONS: tuple[str, ...] = (*TARGETS, IGNORE)
REQUIRED_ROLES: tuple[str, ...] = ("created_at",)
# The two outcome meanings of a won_flag column (its raw values map to one of these).
FLAG_MEANINGS: tuple[str, ...] = ("Won", "Lost")
# Distinct raw values listed for the outcome column's value table (more means the column is not a stage/flag column).
MAX_OUTCOME_VALUES: int = 50
# Dates a draft is built with when its own cannot be derived; the page blanks them for the person to fill.
PLACEHOLDER_DATES: tuple[str, str] = ("2000-01-02", "2000-01-01")
NO_KEY_MESSAGE: str = ("The Anthropic API key is not configured on this server (ANTHROPIC_API_KEY or "
                       "EMVA_ANTHROPIC_API_KEY, and ANTHROPIC_WORKSPACE_ID), so no draft can be requested. Fill the table by hand, or load a "
                       "saved mapping.")
SAVED_HEADER: str = "Confirmed in Keel (Map & convert) on {date}; the outcome mapping was reviewed by a person."


@dataclass(frozen=True)
class MappingForm:
    """The editable state of a mapping under review; any part may still be missing.

    ``sources`` as in ``DatasetMapping``; ``lead`` role -> expression (None = not mapped); ``outcome_column`` a column
    reference ("" = not chosen); ``outcome_values`` raw value -> stage (kind ``stage``) or ``Won`` / ``Lost`` (kind
    ``won_flag``; unused for ``presence``); ``fields`` every field row (plain targets are edited in the review table,
    value maps only as TOML); dates and the other top-level keys as strings; ``origin`` says where the form came from.
    ``features`` are the generic feature set's ``[[features]]`` rows (Phase 10): those whose source is a plain column
    are edited in the review table (``feature`` / ``kind`` / ``name`` columns), derived ones only as TOML. Like the
    outcome, the features are never confirmed in the form: a person ticks both confirmations before converting
    (``confirm``), so a loaded mapping's ``features_confirmed`` is not carried over.
    """

    name: str
    sources: tuple[Source, ...]
    lead: Mapping[str, Expr | None]
    outcome_kind: str = "stage"
    outcome_column: str = ""
    outcome_values: Mapping[str, str] = field(default_factory=dict)
    lost_without_close: str = "error"
    fields: tuple[FieldMap, ...] = ()
    as_of: str = ""
    test_from: str = ""
    source_url: str = ""
    licence: str = ""
    drop_rows_without_created_at: bool = False
    review: Mapping[str, Review] = field(default_factory=dict)
    origin: str = ""
    features: tuple[FeatureMap, ...] = ()

    def refs(self) -> list[str]:
        """Every ``<source>.<column>`` a field row, lead role or the outcome may use: the field rows' sources in order
        (each form lists every profiled column once), then any other column the lead or outcome reads."""
        out = [f.source for f in self.fields]
        extra = [c for e in self.lead.values() if e is not None for c in e.columns()] + \
            ([self.outcome_column] if self.outcome_column else [])
        return list(dict.fromkeys(out + extra))

    @property
    def target_rows(self) -> list[FieldMap]:
        """Field rows with a plain target (the review table)."""
        return [f for f in self.fields if f.value_map is None]

    @property
    def value_map_rows(self) -> list[FieldMap]:
        """Field rows with a value map (shown read-only; edited as TOML)."""
        return [f for f in self.fields if f.value_map is not None]


# --- raw files and profiles ------------------------------------------------------------------------------------------

def raw_frames(files: Mapping[str, bytes]) -> dict[str, pd.DataFrame]:
    """1-3 raw CSVs (file name -> bytes, primary first) as text frames (``emva.ingest.frames_from_bytes``).

    Raises ``ValueError`` naming the file for an unreadable CSV, and for no file or more than ``MAX_RAW_FILES``.
    """
    if not 1 <= len(files) <= MAX_RAW_FILES:
        raise ValueError(f"upload 1 to {MAX_RAW_FILES} CSV files, got {len(files)}")
    out = {}
    for name, content in files.items():
        try:
            out.update(frames_from_bytes({name: content}))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            raise ValueError(f"{name} is not a readable CSV file ({type(e).__name__}: {str(e).splitlines()[0]})") \
                from e
    return out


def profile_frame(profiles: list[ColumnProfile]) -> pd.DataFrame:
    """The column profiles as a table for the page (one row per column; examples joined)."""
    return pd.DataFrame([{"file": p.file, "column": p.name, "type": p.type, "missing": p.share_missing,
                          "distinct": p.distinct, "min": p.min, "max": p.max, "examples": ", ".join(p.examples)}
                         for p in profiles])


def _sources_for(files: list[str], frames: Mapping[str, pd.DataFrame]) -> tuple[Source, ...]:
    """Sources for the files in upload order (the first is the primary); each joined file joins on its first column
    that the primary file also has ("" when none: the person must choose)."""
    primary = files[0]
    out = [Source(source_name(primary), primary)]
    for f in files[1:]:
        shared = [c for c in frames[f].columns if c in frames[primary].columns]
        out.append(Source(source_name(f), f, shared[0] if shared else ""))
    return tuple(out)


def blank_form(frames: Mapping[str, pd.DataFrame], origin: str = "filled by hand") -> MappingForm:
    """A form to fill by hand: sources in upload order, every column of every file set to ignore, no lead role, no
    outcome, no dates. Raises ``ValueError`` when two file names give the same source name."""
    files = list(frames)
    sources = _sources_for(files, frames)
    if len({s.name for s in sources}) != len(sources):
        raise ValueError(f"file names give clashing source names: {[s.name for s in sources]}; rename a file")
    by_file = {s.file: s.name for s in sources}
    rows = tuple(FieldMap(f"{by_file[f]}.{c}", target=IGNORE) for f in files for c in frames[f].columns)
    name = source_name(files[0])
    return MappingForm(name=name, sources=sources, lead=dict.fromkeys(LEAD_ROLES), fields=rows, origin=origin)


def form_from_mapping(m: DatasetMapping, frames: Mapping[str, pd.DataFrame] | None = None,
                      origin: str = "") -> MappingForm:
    """The form of mapping ``m``. With the raw ``frames``, every column of a mapped file that ``m`` does not read is
    added as an ``ignore`` row, so the review table lists each uploaded column once."""
    o = m.outcome
    values: dict[str, str] = dict(o.stage_map) if o.kind == "stage" else \
        {**dict.fromkeys(o.won_values, "Won"), **dict.fromkeys(o.lost_values, "Lost")}
    rows = list(m.fields)
    if frames is not None:
        used = set(m.column_refs())
        for s in m.sources:
            if s.file in frames:
                rows += [FieldMap(f"{s.name}.{c}", target=IGNORE) for c in frames[s.file].columns
                         if f"{s.name}.{c}" not in used]
    return MappingForm(name=m.name, sources=m.sources, lead={r: getattr(m.lead, r) for r in LEAD_ROLES},
                       outcome_kind=o.kind, outcome_column=o.column, outcome_values=values,
                       lost_without_close=o.lost_without_close, fields=tuple(rows), as_of=m.as_of,
                       test_from=m.test_from, source_url=m.source_url, licence=m.licence,
                       drop_rows_without_created_at=m.drop_rows_without_created_at, review=dict(m.review),
                       origin=origin, features=m.features)


def mapping_from_form(f: MappingForm) -> DatasetMapping:
    """The ``DatasetMapping`` the form describes, as a draft (``outcome_confirmed = False`` and ``features_confirmed =
    False``; ``confirm`` sets them).

    Raises ``ValueError`` with a plain message for what is missing (``created_at``, the outcome column, a
    join column, the dates) and for anything ``DatasetMapping`` refuses.
    """
    missing = [r for r in REQUIRED_ROLES if f.lead.get(r) is None]
    if missing:
        raise ValueError(f"choose the column for {' and '.join(missing)} (required)")
    if not f.outcome_column:
        raise ValueError("choose the outcome column")
    for s in f.sources[1:]:
        if not s.join_on:
            raise ValueError(f"choose the column {s.file} joins on")
    if not f.as_of or not f.test_from:
        raise ValueError("enter as_of (the snapshot date) and test_from (the train/test boundary)")
    if f.outcome_kind == "stage":
        outcome = Outcome("stage", f.outcome_column, stage_map={k: v for k, v in f.outcome_values.items() if v},
                          lost_without_close=f.lost_without_close)
    elif f.outcome_kind == "won_flag":
        outcome = Outcome("won_flag", f.outcome_column,
                          won_values=tuple(k for k, v in f.outcome_values.items() if v == "Won"),
                          lost_values=tuple(k for k, v in f.outcome_values.items() if v == "Lost"),
                          lost_without_close=f.lost_without_close)
    else:
        outcome = Outcome(f.outcome_kind, f.outcome_column, lost_without_close=f.lost_without_close)
    lead = LeadColumns(**{r: e for r, e in f.lead.items() if e is not None})
    return DatasetMapping(name=f.name, sources=f.sources, lead=lead, outcome=outcome, as_of=f.as_of,
                          test_from=f.test_from, fields=f.fields, source_url=f.source_url, licence=f.licence,
                          outcome_confirmed=False, drop_rows_without_created_at=f.drop_rows_without_created_at,
                          review={k: v for k, v in f.review.items()}, features=f.features,
                          features_confirmed=False)


def confirm(m: DatasetMapping, features_confirmed: bool = False) -> DatasetMapping:
    """``m`` with ``outcome_confirmed = True``: call only once a person ticked "Outcome mapping confirmed". With
    ``features_confirmed`` (the person also ticked "Extra features are known when the lead is submitted") and declared
    ``[[features]]``, ``features_confirmed = True`` too; a mapping without features keeps it False."""
    return replace(m, outcome_confirmed=True, features_confirmed=features_confirmed and bool(m.features))


def form_from_toml(text: str, frames: Mapping[str, pd.DataFrame] | None = None) -> MappingForm:
    """The form of mapping TOML ``text`` edited by hand (``emva.ingest.load_mapping``; ``ValueError`` if invalid)."""
    return form_from_mapping(load_mapping(text), frames, origin="edited as TOML")


def form_toml(f: MappingForm) -> str | None:
    """The form as mapping TOML (``dump_mapping``), or None while it is incomplete (``mapping_from_form`` refuses)."""
    try:
        return dump_mapping(mapping_from_form(f))
    except ValueError:
        return None


# --- review table ----------------------------------------------------------------------------------------------------

REVIEW_COLUMNS: tuple[str, ...] = ("source", "target", "reason", "confidence", "feature", "kind", "name", "check")
LOW_CONFIDENCE: str = "low"
CHECK_FLAG: str = "check"
# Target shown for a column that has a value map (edited as TOML) but is listed in the table as an extra feature.
VALUE_MAP_TARGET: str = "(value map)"
# Kinds of an extra feature (``emva.generic.ExtraKind``), for the table's kind column.
FEATURE_KINDS: tuple[str, ...] = tuple(k.value for k in ExtraKind)


def _plain_features(f: MappingForm) -> dict[str, FeatureMap]:
    """The form's features whose source is a plain column, by column reference (edited in the review table)."""
    return {fm.source.column: fm for fm in f.features if fm.source.op == COLUMN_OP}


def derived_features(f: MappingForm) -> list[FeatureMap]:
    """The form's features whose source is a derived expression (shown read-only; edited as TOML)."""
    return [fm for fm in f.features if fm.source.op != COLUMN_OP]


def review_frame(f: MappingForm) -> pd.DataFrame:
    """The review table: one row per plain field row, then one per plain-column extra feature without such a row
    (target ``ignore``, or ``VALUE_MAP_TARGET`` when its column has a value map).

    Columns: ``source``, ``target``; ``reason`` / ``confidence`` ("" when none) describe the field row when it has a
    target, else the feature; ``feature`` (True when the column is an extra feature of the generic feature set),
    ``kind`` and ``name`` (the feature's, "" otherwise); ``check`` ("check" when the field's or the feature's confidence
    is low, for highlighting)."""
    feats = _plain_features(f)

    def row(source: str, target: str, field_review: Review | None) -> dict[str, object]:
        fm = feats.get(source)
        if field_review is not None and (target != IGNORE or fm is None):
            shown = field_review
        else:
            shown = fm.review if fm is not None else Review()
        low = shown.confidence == LOW_CONFIDENCE or (fm is not None and fm.review.confidence == LOW_CONFIDENCE)
        return {"source": source, "target": target, "reason": shown.reason, "confidence": shown.confidence or "",
                "feature": fm is not None, "kind": fm.kind if fm is not None else "",
                "name": fm.name if fm is not None else "", "check": CHECK_FLAG if low else ""}

    listed = {r.source for r in f.target_rows}
    value_mapped = {r.source for r in f.value_map_rows}
    rows = [row(r.source, r.target, r.review) for r in f.target_rows]
    rows += [row(src, VALUE_MAP_TARGET if src in value_mapped else IGNORE, None) for src in feats if src not in listed]
    return pd.DataFrame(rows, columns=list(REVIEW_COLUMNS))


def _cell(v: object) -> str:
    """A table cell as stripped text ("" for None / NaN / NA, which an edited table may hold)."""
    return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v).strip()


def _review_of(r: pd.Series, where: str) -> Review:
    """The row's ``reason`` / ``confidence`` as a ``Review``; ``ValueError`` for an unknown confidence."""
    conf = _cell(r["confidence"]) or None
    if conf is not None and conf not in CONFIDENCES:
        raise ValueError(f"{where}: confidence must be one of {list(CONFIDENCES)}")
    return Review(_cell(r["reason"]), conf)


def default_feature_name(source: str, taken: set[str]) -> str:
    """A feature name for column ``source`` (``<source>.<column>``): the column as a slug
    (``emva.ingest.draft.feature_name``), prefixed with the source name when that name is already ``taken``."""
    name = feature_name(source.split(".", 1)[1])
    return name if name not in taken else feature_name(source.replace(".", "_"))


def _feature_from_row(r: pd.Series, source: str, old: FeatureMap | None, review: Review | None,
                      taken: set[str]) -> FeatureMap:
    """The feature a ticked table row describes (``review`` None keeps ``old``'s review)."""
    kind = _cell(r["kind"])
    if kind not in FEATURE_KINDS:
        raise ValueError(f"{source}: choose the feature kind ({' or '.join(FEATURE_KINDS)})")
    name = _cell(r["name"]) or default_feature_name(source, taken)
    keep = old.review if old is not None else Review()
    return FeatureMap(Expr.col(source), kind, name, review if review is not None else keep)


def apply_review(f: MappingForm, table: pd.DataFrame) -> MappingForm:
    """``f`` with the plain field rows and the plain-column features taken from an edited review table
    (``review_frame`` columns; rows matched by ``source``; value-map rows and derived features untouched).

    A row's reason and confidence go to its field row when it has a target, else to its feature (``review_frame``).
    Features keep their order; newly ticked ones follow in table order; a blank name becomes
    ``default_feature_name``. Raises ``ValueError`` for an unknown target or confidence, ``VALUE_MAP_TARGET`` on a
    column without a value map, a ticked row without a kind, an invalid or repeated feature name."""
    edited = {_cell(r["source"]): r for _, r in table.iterrows()}
    value_mapped = {r.source for r in f.value_map_rows}
    rows, feature_reviews = [], {}
    for fm in f.fields:
        r = edited.get(fm.source)
        if fm.value_map is not None or r is None:
            rows.append(fm)
            continue
        target, review = _target(r, fm.source, fm.source in value_mapped), _review_of(r, fm.source)
        if target == IGNORE and _ticked(r):  # the row's reason is the feature's
            feature_reviews[fm.source] = review
            review = fm.review
        rows.append(FieldMap(fm.source, target=target, review=review))
    listed = {fm.source for fm in f.fields}
    for source, r in edited.items():
        if source in listed:
            continue
        target, review = _target(r, source, source in value_mapped), _review_of(r, source)
        if target in (IGNORE, VALUE_MAP_TARGET):
            feature_reviews[source] = review
        else:
            rows.append(FieldMap(source, target=target, review=review))
    return replace(f, fields=tuple(rows), features=_features_from_table(f, edited, feature_reviews))


def _target(r: pd.Series, source: str, value_mapped: bool) -> str:
    """The row's target (blank = ignore); ``ValueError`` for an unknown one or ``VALUE_MAP_TARGET`` on a column
    without a value map."""
    target = _cell(r["target"]) or IGNORE
    if target == VALUE_MAP_TARGET and value_mapped:
        return target
    if target == VALUE_MAP_TARGET:
        raise ValueError(f"{source}: only a column with a value map can show {VALUE_MAP_TARGET!r}")
    if target not in TARGET_OPTIONS:
        raise ValueError(f"{source}: unknown target {target!r}")
    return target


def _ticked(r: pd.Series) -> bool:
    """True when the row's ``feature`` cell is ticked (NaN-safe)."""
    return r["feature"] == True  # noqa: E712 (an edited table may hold NaN or None)


def _features_from_table(f: MappingForm, edited: Mapping[str, pd.Series],
                         reviews: Mapping[str, Review]) -> tuple[FeatureMap, ...]:
    """The form's features after the table edit (``apply_review``)."""
    out: list[FeatureMap] = []
    taken: set[str] = {fm.name for fm in derived_features(f)}
    plain = _plain_features(f)
    for fm in f.features:
        if fm.source.op != COLUMN_OP:
            out.append(fm)
            continue
        r = edited.get(fm.source.column)
        if r is None:
            out.append(fm)
            taken.add(fm.name)
        elif _ticked(r):  # an unticked row drops the feature
            out.append(_feature_from_row(r, fm.source.column, fm, reviews.get(fm.source.column), taken))
            taken.add(out[-1].name)
    for source, r in edited.items():
        if source not in plain and _ticked(r):
            out.append(_feature_from_row(r, source, None, reviews.get(source), taken))
            taken.add(out[-1].name)
    names = [fm.name for fm in out]
    if len(set(names)) != len(names):
        raise ValueError(f"feature names must be unique; repeated: {sorted({n for n in names if names.count(n) > 1})}")
    return tuple(out)


def derived_features_frame(f: MappingForm) -> pd.DataFrame:
    """The derived-expression features, read-only: ``name``, ``kind``, ``source`` (``expr_text``), ``reason``,
    ``confidence``."""
    return pd.DataFrame([{"name": fm.name, "kind": fm.kind, "source": expr_text(fm.source), "reason": fm.review.reason,
                          "confidence": fm.review.confidence or ""} for fm in derived_features(f)],
                        columns=["name", "kind", "source", "reason", "confidence"])


def features_signature(features: tuple[FeatureMap, ...]) -> str:
    """A short digest of the declared features (source, kind, name): the features confirmation resets when it
    changes."""
    text = "\n".join(f"{expr_text(fm.source)}|{fm.kind}|{fm.name}" for fm in features)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def value_map_frame(f: MappingForm) -> pd.DataFrame:
    """The value maps, read-only: one row per raw value with the targets it sets (``target = value; ...``)."""
    rows = [{"source": fm.source, "raw value": raw,
             "sets": "; ".join(f"{t} = {v}" for t, v in sets.items()) or "(nothing)"}
            for fm in f.value_map_rows for raw, sets in fm.value_map.items()]
    return pd.DataFrame(rows, columns=["source", "raw value", "sets"])


def expr_text(e: Expr | None) -> str:
    """An expression as one readable line, e.g. ``minus_days(date_from_parts(b.y, b.m, b.d), b.lead_time)``."""
    if e is None:
        return ""
    if e.op == COLUMN_OP:
        return str(e.column)
    return f"{e.op}({', '.join(expr_text(a) for a in e.args)})"


def is_plain(e: Expr | None) -> bool:
    """True for an unmapped role or a plain column reference (editable with a select box; a derived expression is
    shown read-only and edited as TOML)."""
    return e is None or e.op == COLUMN_OP


def with_role(f: MappingForm, role: str, ref: str) -> MappingForm:
    """``f`` with lead ``role`` set to column ``ref`` ("" = not mapped)."""
    if role not in LEAD_ROLES:
        raise ValueError(f"unknown lead role {role!r}")
    return replace(f, lead={**f.lead, role: Expr.col(ref) if ref else None})


def with_join(f: MappingForm, file: str, column: str) -> MappingForm:
    """``f`` with the joined source ``file`` joining on ``column``."""
    return replace(f, sources=tuple(replace(s, join_on=column) if s.file == file and s.join_on is not None else s
                                    for s in f.sources))


def column_values(frames: Mapping[str, pd.DataFrame], f: MappingForm, ref: str,
                  limit: int = MAX_OUTCOME_VALUES) -> tuple[list[str], bool]:
    """Distinct raw values of column ``ref`` (``<source>.<column>``) in the uploaded data, most frequent first, as
    ``(values, truncated)``; a blank cell is listed as "" (the key for blank in a stage map or won/lost list)."""
    name, column = ref.split(".", 1)
    src = next((s for s in f.sources if s.name == name), None)
    if src is None or src.file not in frames or column not in frames[src.file]:
        return [], False
    s = frames[src.file][column].map(lambda v: v.strip() if isinstance(v, str) else "")
    counts = s.value_counts()
    values = [str(v) for v in counts.index[:limit]]
    return values, len(counts) > limit


def outcome_frame(frames: Mapping[str, pd.DataFrame], f: MappingForm) -> tuple[pd.DataFrame, bool]:
    """The outcome value table (``value``, ``meaning``) for kinds ``stage`` and ``won_flag``: the form's values first,
    then the column's other raw values with no meaning yet; and whether the column had more than
    ``MAX_OUTCOME_VALUES`` distinct values (then it is probably not an outcome column)."""
    seen, truncated = column_values(frames, f, f.outcome_column) if f.outcome_column else ([], False)
    rows = [{"value": k, "meaning": v} for k, v in f.outcome_values.items()]
    rows += [{"value": v, "meaning": ""} for v in seen if v not in f.outcome_values]
    return pd.DataFrame(rows, columns=["value", "meaning"]), truncated


def meanings(kind: str) -> tuple[str, ...]:
    """Allowed meanings in the outcome value table: CRM stages for ``stage``, Won / Lost for ``won_flag``."""
    if kind not in OUTCOME_KINDS:
        raise ValueError(f"unknown outcome kind {kind!r}")
    return STAGES if kind == "stage" else FLAG_MEANINGS if kind == "won_flag" else ()


def apply_outcome(f: MappingForm, kind: str, column: str, table: pd.DataFrame | None,
                  lost_without_close: str) -> MappingForm:
    """``f`` with the outcome from the page: kind, column, the edited value table (rows with a meaning kept; None =
    no table, as for ``presence``) and what a Lost lead without ``close_at`` gets."""
    values: dict[str, str] = {}
    if table is not None:
        allowed = meanings(kind)
        for _, r in table.iterrows():
            meaning = _cell(r["meaning"])
            if meaning and meaning not in allowed:
                raise ValueError(f"outcome value {_cell(r['value'])!r}: {meaning!r} is not one of {list(allowed)}")
            if meaning:
                values[_cell(r["value"])] = meaning
    return replace(f, outcome_kind=kind, outcome_column=column, outcome_values=values,
                   lost_without_close=lost_without_close)


# --- mapping choices -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MappingChoice:
    """A mapping the page can load with no model call: a saved dataset's ``mapping.toml`` or a committed one in
    ``mappings/`` (``builtin``). ``key`` is unique; ``label`` is for the picker."""

    key: str
    label: str
    path: str
    builtin: bool


def mapping_choices(root: str | Path, builtin_dir: Path = BUILTIN_MAPPINGS_DIR) -> list[MappingChoice]:
    """Saved mappings (``app.storage.list_mappings``, newest dataset first; labelled with the dataset name and the
    host of the mapping's ``source_url``), then the built-in ones by name."""
    saved = [MappingChoice(f"saved:{m.name}", f"{m.name} · saved with a dataset"
                           + (f" · {urlparse(m.source_url).hostname or m.source_url}" if m.source_url else ""),
                           m.path, False)
             for m in list_mappings(root)]
    builtin = [MappingChoice(f"builtin:{p.stem}", f"{p.stem} · built-in", str(p), True)
               for p in sorted(builtin_dir.glob("*.toml"))]
    return saved + builtin


def load_choice(choice: MappingChoice) -> DatasetMapping:
    """The mapping of ``choice`` (``load_mapping``; ``ValueError`` if the file is not a valid mapping)."""
    return load_mapping(Path(choice.path).read_text(encoding="utf-8"))


def missing_files(m: DatasetMapping, frames: Mapping[str, pd.DataFrame]) -> list[str]:
    """Source files ``m`` names that were not uploaded (convert needs every one; matched by file name)."""
    return [s.file for s in m.sources if s.file not in frames]


# --- drafting --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DraftOutcome:
    """The result of "Draft mapping with AI": the draft ``mapping`` (None on failure), whether it came from the reply
    cache (``from_cache``: no request was made), and on failure an ``error`` to show; ``no_key`` when the server has
    no API key (the page then offers the table to fill by hand). ``dates_note`` is set when the draft is fine but
    its dates could not be derived (``emva.ingest.draft.default_dates``): the mapping then carries placeholder dates
    the page must blank out (``PLACEHOLDER_DATES``) for the person to enter."""

    mapping: DatasetMapping | None
    from_cache: bool = False
    error: str | None = None
    no_key: bool = False
    dates_note: str | None = None

    @property
    def origin(self) -> str:
        """Where the draft came from, for the page."""
        return f"AI draft by {MODEL_ID}" + (" (from the draft cache: no request made)" if self.from_cache else "")


def draft_cache(root: str | Path) -> ReplyCache:
    """The app's draft reply cache, ``DATA_DIR/ingest/draft_cache.json``."""
    return ReplyCache(Path(root) / DRAFT_CACHE)


def draft_is_cached(profiles: list[ColumnProfile], root: str | Path) -> bool:
    """True when a draft for these profiles is already cached (the button then makes no request)."""
    return is_cached(profiles, draft_cache(root))


def draft(profiles: list[ColumnProfile], root: str | Path, name: str,
          client_factory: Callable[[], anthropic.Anthropic] | None = None) -> DraftOutcome:
    """Draft a mapping called ``name`` from ``profiles`` (``emva.ingest.draft_mapping``; only profiles are sent).

    A cached draft needs no client. Otherwise ``client_factory`` (default ``emva.context.agent.make_client``, looked
    up at call time) builds one; its ``SystemExit`` for missing keys becomes ``no_key``. API errors, repeated
    transport errors and a reply that breaks the contract are logged and returned as ``error``. The dates are derived
    separately (``emva.ingest.draft.default_dates``); when they cannot be, the draft keeps ``PLACEHOLDER_DATES`` and
    ``dates_note`` says why. Nothing is swallowed silently: any other error propagates.
    """
    cache = draft_cache(root)
    cached = is_cached(profiles, cache)
    client = None
    if not cached:
        try:
            client = (client_factory or make_client)()
        except SystemExit as e:
            log.warning("mapping draft not requested: %s", e)
            return DraftOutcome(None, error=NO_KEY_MESSAGE, no_key=True)
    try:
        mapping = draft_mapping(profiles, client, cache, name, *PLACEHOLDER_DATES)
    except (anthropic.APIError, RuntimeError, ContractError) as e:
        log.exception("mapping draft failed")
        return DraftOutcome(None, from_cache=cached, error=f"The draft failed ({type(e).__name__}: {e}). Fill the "
                                                          "table by hand, or load a saved mapping.")
    try:
        as_of, test_from = default_dates(mapping, profiles)
    except ValueError as e:  # e.g. the column drafted as created_at is not an ISO date column
        log.warning("mapping draft has no derivable dates: %s", e)
        return DraftOutcome(mapping, from_cache=cached, dates_note=f"{e}. Enter as_of and test_from below.")
    return DraftOutcome(replace(mapping, as_of=as_of, test_from=test_from), from_cache=cached)


# --- convert ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Conversion:
    """A conversion: ``files`` (the training files, ``dataset.json``, ``mapping.toml`` and any uploaded
    ``status_quo_rules.json``, ready for ``save_dataset``), ``meta`` (``dataset.json`` parsed) and ``report``
    (``validate_files`` of ``files``)."""

    files: dict[str, bytes]
    meta: dict
    report: ValidationReport


def convert_raw(frames: Mapping[str, pd.DataFrame], mapping: DatasetMapping, date: str,
                rules: bytes | None = None) -> Conversion:
    """Convert the raw ``frames`` with a confirmed ``mapping`` (``emva.ingest.convert``), add the mapping as
    ``mapping.toml`` (``dump_mapping``, header noting it was confirmed in Keel on ``date``) and, when given, the
    uploaded status-quo ``rules`` as ``status_quo_rules.json`` (so the standard report gets a status-quo row), then
    validate the result (the rules by the same checks as an upload in EMVA's format).

    Raises ``ValueError`` for a draft mapping and for everything ``convert`` refuses (message fit for the page).
    """
    files = convert(dict(frames), mapping)
    files[MAPPING_FILE] = dump_mapping(mapping, SAVED_HEADER.format(date=date)).encode("utf-8")
    if rules is not None:
        files[RULES_FILE] = rules
    return Conversion(files, json.loads(files[DATASET_META_FILE]), validate_files(files))


@dataclass(frozen=True)
class CoverageSummary:
    """How many of the v2 design columns the converted data fill: ``data`` (vary across leads), ``constant``
    (inputs mapped, no variation), ``unfilled`` (no input mapped), of ``total``."""

    data: int
    constant: int
    unfilled: int
    total: int


def coverage_summary(meta: Mapping[str, object]) -> CoverageSummary:
    """Counts of ``dataset.json``'s ``coverage`` entries by status."""
    status = [c["status"] for c in meta["coverage"]]
    return CoverageSummary(status.count("data"), status.count("constant"), status.count("unfilled"), len(status))


def derived_counts(meta: Mapping[str, object]) -> list[tuple[str, int, str]]:
    """The non-zero counts of values the conversion derived or adjusted (``emva.ingest.convert.DERIVED_COUNTS``), as
    ``(dataset.json key, count, what it means)`` in that order, for the coverage card's "Derived during conversion"."""
    return [(k, int(meta.get(k, 0)), what) for k, what in DERIVED_COUNTS.items() if meta.get(k, 0)]


@dataclass(frozen=True)
class ExtrasSummary:
    """The extra features a conversion declared (``dataset.json`` ``features``): ``names`` in declaration order,
    ``numeric`` and ``categorical`` counts. Their design columns are only known at training (the encoding is fitted
    on the training leads)."""

    names: tuple[str, ...]
    numeric: int
    categorical: int

    @property
    def n(self) -> int:
        """Number of declared extras."""
        return len(self.names)


def extras_summary(meta: Mapping[str, object]) -> ExtrasSummary:
    """The declared extras of ``dataset.json`` (none when it has no ``features``)."""
    feats = list(meta.get("features", []))
    kinds = [str(x["kind"]) for x in feats]
    return ExtrasSummary(tuple(str(x["name"]) for x in feats), kinds.count(ExtraKind.NUMERIC.value),
                         kinds.count(ExtraKind.CATEGORICAL.value))


def coverage_frame(meta: Mapping[str, object]) -> pd.DataFrame:
    """``dataset.json``'s ``coverage`` as a table: design column, status, mapped inputs, leads with a 1."""
    return pd.DataFrame([{"column": c["column"], "status": c["status"], "inputs": ", ".join(c["inputs_mapped"]),
                          "leads_with_1": c["leads_with_1"], "leads": c["leads"]} for c in meta["coverage"]])


__all__ = ["BUILTIN_MAPPINGS_DIR", "Conversion", "CoverageSummary", "DRAFT_CACHE", "DraftOutcome", "ExtrasSummary",
           "FEATURE_KINDS", "MappingChoice", "MappingForm", "NO_KEY_MESSAGE", "PLACEHOLDER_DATES", "TARGET_OPTIONS",
           "VALUE_MAP_TARGET", "apply_outcome", "apply_review", "blank_form", "column_values", "confirm", "convert_raw",
           "coverage_frame", "coverage_summary", "default_feature_name", "derived_counts", "derived_features",
           "derived_features_frame", "draft", "draft_cache", "draft_is_cached", "expr_text", "extras_summary",
           "features_signature", "form_from_mapping", "form_from_toml", "form_toml", "is_plain", "load_choice",
           "mapping_choices", "mapping_from_form", "meanings", "missing_files", "outcome_frame", "profile",
           "profile_frame", "raw_frames", "review_frame", "value_map_frame", "with_join", "with_role"]
