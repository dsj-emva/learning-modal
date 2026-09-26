"""The generic feature set's extras (Phase 10, ADR 0024): source columns a mapping declares in ``[[features]]``.

A converted dataset carries its extras in ``extra_features.csv`` (``lead_id`` plus one column per declared extra,
raw values, no encoding) and declares them in ``dataset.json`` under ``features`` (``name``, ``kind``, ``source``);
``emva.ingest.convert`` writes both. ``read_extra_features`` reads them as raw columns ``x_<name>``.

``fit_encoder`` fits the encoding on the pipeline's training leads only and freezes it (``GenericEncoder``, stored in
the model bundle so ``emva.scoring`` reproduces the same design columns):

- **categorical**: values are stripped text (a blank is ``missing``). A level is kept when at least
  ``GENERIC_MIN_LEVEL_COUNT`` training leads have it; rarer values, and at scoring values never seen in training, are
  ``other``; a raw value spelled ``other`` or ``missing`` falls into that level. Reference = the most frequent kept
  level (ties: the smaller value); with no kept level, the more frequent of ``other`` / ``missing`` (tie: ``other``).
  Level order: the reference, the other kept levels by training count (descending, then value), ``other``,
  ``missing``. So an unseen value never raises: this amends ADR 0009's "unknown level raises" for extras only.
- **numeric**: edges are the training quantiles at 1/B, ..., (B-1)/B (B = ``GENERIC_NUMERIC_BINS``; numpy's
  ``inverted_cdf`` method, so every edge is an observed training value), de-duplicated. An edge at the training maximum would leave the top bin empty, so it is replaced by
  the largest training value below the maximum (if any) and the edges are de-duplicated again (a 0/1 column with 90%
  ones still gets the bins ``<=0`` and ``>0``). Bins are right-closed: ``<=e1``, ``(e1, e2]``, ..., ``>ek``; with no
  edge (one distinct training value) the single bin is ``all``. A blank is ``missing``; a value outside the training
  range falls into the edge bin. Reference = the first bin (``missing`` when no training lead has a value).

Design columns are ``x_<name>=<level>`` for every level but the reference, extras in declaration order. Everything is
deterministic: no data-dependent ordering is left to an unstable sort.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from emva.constants import GENERIC_MIN_LEVEL_COUNT, GENERIC_NUMERIC_BINS, MISSING
from emva.dataset_meta import read_dataset_meta

EXTRA_FEATURES_FILE: str = "extra_features.csv"
# Raw extra column in the featurised frame, and prefix of its design columns (``x_<name>=<level>``).
EXTRA_PREFIX: str = "x_"
# Level of a categorical extra for rare and unseen values; ``MISSING`` is the level of a blank.
OTHER: str = "other"
# Level of the single bin of a numeric extra with no edge (one distinct training value).
ALL: str = "all"
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class ExtraKind(str, Enum):
    """How an extra is encoded: ``numeric`` (quantile bins) or ``categorical`` (frequent levels + other)."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class ExtraFeature:
    """A declared extra: ``name`` (a slug; raw column ``x_<name>``), ``kind`` and ``source`` (the mapping's source
    expression as text, for the record)."""

    name: str
    kind: ExtraKind
    source: str = ""

    def __post_init__(self) -> None:
        """Refuse a name that is not a lower-case slug (letters, digits, ``_``) and an unknown kind."""
        if not isinstance(self.name, str) or not NAME_PATTERN.match(self.name):
            raise ValueError(f"extra feature name {self.name!r} must be lower-case letters, digits and '_' "
                             "(starting with a letter or digit)")
        object.__setattr__(self, "kind", ExtraKind(self.kind))

    @property
    def column(self) -> str:
        """The raw column in the featurised frame: ``x_<name>``."""
        return EXTRA_PREFIX + self.name


def _text(values: pd.Series) -> pd.Series:
    """Categorical values as stripped text; blanks (None, NaN, ``pd.NA``, empty or whitespace) as missing (pandas may
    hold them as None or NaN: test with ``isinstance(v, str)``)."""
    def one(v: object) -> str | None:
        if isinstance(v, str):
            return v.strip() or None
        return None if v is None or pd.isna(v) else str(v).strip() or None
    return values.map(one).astype(object)


def numeric_values(values: pd.Series, what: str) -> pd.Series:
    """``values`` as floats (blanks NaN); ``ValueError`` naming ``what`` when a non-blank value is not a finite number
    (``inf``, ``-inf`` and an overflow such as ``1e400`` are refused: they have no quantile bin)."""
    text = _text(values)
    parsed = pd.to_numeric(text, errors="coerce").astype(float)
    bad = (parsed.isna() | np.isinf(parsed)) & text.notna()
    if bad.any():
        raise ValueError(f"extra feature {what!r} is numeric but has values that are not finite numbers, e.g. "
                         f"{sorted(set(text[bad]))[:5]}")
    return parsed


def _fmt_edge(e: float) -> str:
    """An edge in a level name: 6 significant digits."""
    return f"{e:.6g}"


def bin_labels(edges: tuple[float, ...]) -> tuple[str, ...]:
    """Level names of the right-closed bins between ``edges``: ``<=e1``, ``(e1, e2]``, ..., ``>ek`` (``all`` with no
    edge). Edges that print alike at 6 significant digits are printed in full (``repr``)."""
    if not edges:
        return (ALL,)
    shown = [_fmt_edge(e) for e in edges]
    if len(set(shown)) < len(shown):
        shown = [repr(float(e)) for e in edges]
    return (f"<={shown[0]}", *(f"({a}, {b}]" for a, b in zip(shown, shown[1:])), f">{shown[-1]}")


def numeric_edges(values: np.ndarray, bins: int = GENERIC_NUMERIC_BINS) -> tuple[float, ...]:
    """Inner bin edges from training ``values`` (no NaN): the module docstring's rule."""
    if len(values) == 0:
        return ()
    edges = np.unique(np.quantile(values, np.arange(1, bins) / bins, method="inverted_cdf"))
    top = values.max()
    if len(edges) and edges[-1] >= top:
        below = values[values < top]
        edges = np.unique(np.concatenate([edges[edges < top], [below.max()] if len(below) else []]))
    return tuple(float(e) for e in edges)


@dataclass(frozen=True)
class FittedExtra:
    """One extra's frozen encoding: ``levels`` (reference first, then design-column order), ``reference``, the
    numeric ``edges`` (empty for a categorical extra) and ``train_counts`` (training leads per level)."""

    feature: ExtraFeature
    levels: tuple[str, ...]
    reference: str
    edges: tuple[float, ...]
    train_counts: dict[str, int]

    def encode(self, values: pd.Series) -> pd.Series:
        """The level of every raw value (module docstring). Raises ``ValueError`` for a non-number in a numeric
        extra."""
        f = self.feature
        if f.kind is ExtraKind.NUMERIC:
            v = numeric_values(values, f.name)
            labels = np.array(bin_labels(self.edges), dtype=object)
            idx = np.searchsorted(np.array(self.edges, dtype=float), v.fillna(0).to_numpy(), side="left")
            return pd.Series(np.where(v.isna(), MISSING, labels[idx]), index=values.index, dtype=object)
        t = _text(values)
        kept = set(self.levels) - {OTHER, MISSING}
        return t.map(lambda v: MISSING if not isinstance(v, str) else
                     (v if v in kept else (MISSING if v == MISSING else OTHER))).astype(object)

    def columns(self) -> list[str]:
        """Design columns: ``x_<name>=<level>`` for every level but the reference, in level order."""
        return [f"{self.feature.column}={lvl}" for lvl in self.levels if lvl != self.reference]


def _fit_one(feature: ExtraFeature, values: pd.Series, min_count: int, bins: int) -> FittedExtra:
    """Fit one extra on its training ``values``."""
    if feature.kind is ExtraKind.NUMERIC:
        v = numeric_values(values, feature.name).dropna().to_numpy()
        edges = numeric_edges(v, bins)
        levels = (*bin_labels(edges), MISSING)
        fitted = FittedExtra(feature, levels, levels[0] if len(v) else MISSING, edges, {})
    else:
        t = _text(values)
        counts = t.dropna().value_counts()
        ranked = sorted(((lvl, int(n)) for lvl, n in counts.items() if lvl not in (OTHER, MISSING)),
                        key=lambda kv: (-kv[1], kv[0]))
        kept = [lvl for lvl, n in ranked if n >= min_count]
        n_missing = int(t.isna().sum() + counts.get(MISSING, 0))
        reference = kept[0] if kept else (OTHER if len(t) - n_missing >= n_missing else MISSING)
        levels = (*kept, OTHER, MISSING)
        fitted = FittedExtra(feature, (reference, *(lvl for lvl in levels if lvl != reference)), reference, (), {})
    enc = fitted.encode(values)
    counts = {lvl: int((enc == lvl).sum()) for lvl in fitted.levels}
    return FittedExtra(feature, fitted.levels, fitted.reference, fitted.edges, counts)


@dataclass(frozen=True)
class GenericEncoder:
    """The frozen encoding of every extra, in declaration order (``fit_encoder``)."""

    extras: tuple[FittedExtra, ...]

    @property
    def raw_columns(self) -> tuple[str, ...]:
        """The raw extra columns (``x_<name>``) the encoder reads."""
        return tuple(e.feature.column for e in self.extras)

    def columns(self) -> list[str]:
        """Every design column, extras in declaration order."""
        return [c for e in self.extras for c in e.columns()]

    def levels(self, X: pd.DataFrame) -> pd.DataFrame:
        """The level of every extra for every row of ``X`` (columns ``x_<name>``; an absent raw column is blank)."""
        blank = pd.Series(None, index=X.index, dtype=object)
        return pd.DataFrame({e.feature.column: e.encode(X[e.feature.column] if e.feature.column in X else blank)
                             for e in self.extras}, index=X.index)

    def design(self, X: pd.DataFrame) -> pd.DataFrame:
        """The extras' design columns (``columns()``, 0/1 floats) for every row of ``X``."""
        lv = self.levels(X)
        data = {f"{e.feature.column}={lvl}": lv[e.feature.column].eq(lvl).astype(float)
                for e in self.extras for lvl in e.levels if lvl != e.reference}
        return pd.DataFrame(data, index=X.index, columns=self.columns())


def fit_encoder(features: tuple[ExtraFeature, ...], X_train: pd.DataFrame,
                min_count: int = GENERIC_MIN_LEVEL_COUNT, bins: int = GENERIC_NUMERIC_BINS) -> GenericEncoder:
    """Fit every extra's encoding on the training rows ``X_train`` (raw columns ``x_<name>``; module docstring).

    Raises ``ValueError`` when a raw column is missing or a numeric extra holds a non-number.
    """
    missing = [f.column for f in features if f.column not in X_train]
    if missing:
        raise ValueError(f"raw extra column(s) {missing} are not in the training frame")
    return GenericEncoder(tuple(_fit_one(f, X_train[f.column], min_count, bins) for f in features))


def parse_features(entries: object, where: str) -> tuple[ExtraFeature, ...]:
    """The ``features`` list of a ``dataset.json`` as ``ExtraFeature``s; ``ValueError`` naming ``where`` for a
    malformed entry or a repeated name."""
    if not isinstance(entries, list):
        raise ValueError(f"{where}: 'features' must be a list of {{name, kind, source}}")
    out = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or not {"name", "kind"} <= e.keys():
            raise ValueError(f"{where}: features[{i}] needs name and kind")
        try:
            out.append(ExtraFeature(e["name"], e["kind"], str(e.get("source", ""))))
        except ValueError as err:
            raise ValueError(f"{where}: features[{i}]: {err}") from err
    names = [f.name for f in out]
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: repeated feature names in {names}")
    return tuple(out)


def read_extra_features(data: str | Path) -> tuple[tuple[ExtraFeature, ...], pd.DataFrame]:
    """The declared extras of the dataset directory ``data`` and their raw values (index ``lead_id``, columns
    ``x_<name>`` in declaration order, text; numeric extras checked to be numbers).

    Raises ``ValueError`` (the generic feature set needs both) when ``data`` has no ``dataset.json`` with a
    ``features`` list, no ``extra_features.csv``, a file whose columns are not ``lead_id`` plus the declared names,
    a blank or repeated ``lead_id``, or a non-number in a numeric extra.
    """
    meta = read_dataset_meta(data)
    if meta is None or "features" not in meta:
        raise ValueError(f"{data}: the generic feature set needs a converted dataset whose dataset.json declares "
                         "'features' (a mapping with [[features]], python -m emva.ingest convert)")
    features = parse_features(meta["features"], f"{Path(data) / 'dataset.json'}")
    path = Path(data) / EXTRA_FEATURES_FILE
    if not path.is_file():
        raise ValueError(f"{data}: the generic feature set needs {EXTRA_FEATURES_FILE} (written by "
                         "python -m emva.ingest convert with a mapping that declares [[features]])")
    E = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    want = ["lead_id", *(f.name for f in features)]
    if list(E.columns) != want:
        raise ValueError(f"{path}: columns {list(E.columns)}, expected {want} (lead_id plus the declared features)")
    if E.lead_id.isna().any() or E.lead_id.duplicated().any():
        raise ValueError(f"{path}: lead_id must be filled and unique")
    E = E.set_index("lead_id")
    for f in features:
        if f.kind is ExtraKind.NUMERIC:
            numeric_values(E[f.name], f.name)
    return features, E.rename(columns={f.name: f.column for f in features})


__all__ = ["ALL", "EXTRA_FEATURES_FILE", "EXTRA_PREFIX", "ExtraFeature", "ExtraKind", "FittedExtra", "GenericEncoder",
           "OTHER", "bin_labels", "fit_encoder", "numeric_edges", "numeric_values", "parse_features",
           "read_extra_features"]
