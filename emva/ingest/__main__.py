"""CLI of the dataset converter.

  python -m emva.ingest draft --raw DIR --out FILE.toml [--name NAME] [--cache FILE] [--as-of DATE] [--test-from DATE]
  python -m emva.ingest convert --mapping FILE.toml --raw DIR --out DIR
  python -m emva.ingest score --mapping FILE.toml --raw FILE_OR_DIR --run RUN_DIR [--data DATASET_DIR] [--out FILE]

``draft`` profiles every ``*.csv`` in ``--raw`` (``emva.ingest.profile``), asks Claude Haiku for a draft mapping
(``emva.ingest.draft``; needs ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_WORKSPACE_ID``, ADR 0010, unless the reply is
cached) and writes it with ``outcome_confirmed = false``. The reply cache defaults to ``.draft_cache.json`` next to
``--out`` (commit it with the mapping so the draft is reproducible without the API). A person reviews the draft,
fixes it, and sets ``outcome_confirmed = true``.

``convert`` reads the source files a confirmed mapping names from ``--raw`` and writes the five training files,
``dataset.json`` and a copy of the mapping (``mapping.toml``) into ``--out``; it refuses a draft. It prints the
coverage summary.

``score`` scores new leads given in the source format: ``--raw`` is a CSV of primary-source rows (other source files
the mapping's submit-time part needs are read from the same directory) or a directory holding the source files. The
rows go through ``emva.ingest.convert.convert_leads`` (no outcome columns needed) and ``emva.scoring.score_leads``
with ``RUN_DIR/model.joblib`` (written by ``python -m emva --out RUN_DIR``). ``--data`` is the dataset whose
``companies.csv`` the enrichment join reads; without it an empty one is used, which is what every converted dataset
holds (the converter writes companies.csv header-only). Writes ``--out`` or prints the scores as CSV.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from emva.context.agent import MODEL_ID, make_client
from emva.context.cache import ReplyCache
from emva.ingest.convert import COMPANIES_COLUMNS, convert, convert_leads, frames_from_bytes
from emva.ingest.draft import draft_mapping, is_cached, source_name
from emva.ingest.mapping import dump_mapping, load_mapping
from emva.ingest.profile import profile
from emva.persist import BUNDLE_FILE, load_bundle
from emva.scoring import score_leads

MAPPING_COPY = "mapping.toml"
DRAFT_HEADER = ("DRAFT by {model} from column profiles of {raw} (no rows were sent). Review every row, add derived\n"
                "expressions where needed, check the outcome, then set outcome_confirmed = true.")


def _draft(a: argparse.Namespace) -> None:
    """``draft`` subcommand."""
    raw = Path(a.raw)
    files = sorted(raw.glob("*.csv"))
    if not files:
        raise SystemExit(f"no .csv files in {raw}")
    frames = frames_from_bytes({f.name: f.read_bytes() for f in files})
    out = Path(a.out)
    cache = ReplyCache(a.cache or out.parent / ".draft_cache.json")
    name = a.name or source_name(out.name)
    profiles = profile(frames)
    mapping = draft_mapping(profiles, None if is_cached(profiles, cache) else make_client(), cache, name,
                            a.as_of, a.test_from)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_mapping(mapping, DRAFT_HEADER.format(model=MODEL_ID, raw=raw.name)), encoding="utf-8")
    print(f"wrote draft {out} ({len(mapping.fields)} field rows); outcome_confirmed = false until reviewed")


def _convert(a: argparse.Namespace) -> None:
    """``convert`` subcommand."""
    text = Path(a.mapping).read_text(encoding="utf-8")
    mapping = load_mapping(text, require_confirmed=True)
    raw = Path(a.raw)
    missing = [s.file for s in mapping.sources if not (raw / s.file).is_file()]
    if missing:
        raise SystemExit(f"{raw} lacks {missing}")
    frames = frames_from_bytes({s.file: (raw / s.file).read_bytes() for s in mapping.sources})
    files = convert(frames, mapping)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out / name).write_bytes(content)
    (out / MAPPING_COPY).write_text(text, encoding="utf-8")
    meta = json.loads(files["dataset.json"])
    status = Counter(c["status"] for c in meta["coverage"])
    print(f"wrote {meta['leads']} leads to {out} (as_of {meta['as_of']}, test_from {meta['test_from']}); "
          f"final stages {meta['final_stage_counts']}; design columns: {status.get('data', 0)} with data, "
          f"{status.get('constant', 0)} constant, {status.get('unfilled', 0)} unfilled of {len(meta['coverage'])}")


def _score(a: argparse.Namespace) -> None:
    """``score`` subcommand."""
    mapping = load_mapping(Path(a.mapping).read_text(encoding="utf-8"), require_confirmed=True)
    raw = Path(a.raw)
    needed = mapping.source_columns(submit_time_only=True)
    primary = mapping.sources[0].file
    folder = raw.parent if raw.is_file() else raw
    paths = {f: (raw if raw.is_file() and f == primary else folder / f) for f in needed}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"missing source file(s) {missing}")
    leads = convert_leads(frames_from_bytes({f: p.read_bytes() for f, p in paths.items()}), mapping)
    bundle = load_bundle(Path(a.run) / BUNDLE_FILE)
    if a.data:
        scores = score_leads(bundle, leads, a.data)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(columns=COMPANIES_COLUMNS).to_csv(Path(tmp) / "companies.csv", index=False)
            scores = score_leads(bundle, leads, tmp)
    if a.out:
        scores.to_csv(a.out)
        print(f"wrote {len(scores)} scores to {a.out}")
    else:
        scores.to_csv(sys.stdout)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run ``draft`` or ``convert`` (module docstring)."""
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(prog="python -m emva.ingest")
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draft", help="draft a mapping from column profiles (LLM; outcome_confirmed = false)")
    d.add_argument("--raw", required=True, help="directory holding the source CSVs")
    d.add_argument("--out", required=True, help="mapping TOML to write")
    d.add_argument("--name", default=None, help="mapping name (default: from --out)")
    d.add_argument("--cache", default=None, help="reply cache (default: .draft_cache.json next to --out)")
    d.add_argument("--as-of", default=None, help="snapshot date (default: derived from the created_at column)")
    d.add_argument("--test-from", default=None, help="train/test boundary (default: derived from created_at)")
    c = sub.add_parser("convert", help="convert source CSVs with a confirmed mapping")
    c.add_argument("--mapping", required=True)
    c.add_argument("--raw", required=True, help="directory holding the source CSVs the mapping names")
    c.add_argument("--out", required=True, help="dataset directory to write")
    s = sub.add_parser("score", help="score new leads given in the source format with a trained run")
    s.add_argument("--mapping", required=True)
    s.add_argument("--raw", required=True, help="CSV of new primary-source rows, or a directory of source files")
    s.add_argument("--run", required=True, help="run directory holding model.joblib (python -m emva --out)")
    s.add_argument("--data", default=None, help="dataset directory whose companies.csv enrichment reads "
                                                "(default: an empty one, as in every converted dataset)")
    s.add_argument("--out", default=None, help="scores CSV to write (default: print)")
    a = ap.parse_args(argv)
    {"draft": _draft, "convert": _convert, "score": _score}[a.command](a)


if __name__ == "__main__":
    main()
