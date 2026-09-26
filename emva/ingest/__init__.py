"""Generic dataset converter (Phase 9 a): a TOML mapping per data source turns 1-3 foreign CSVs into EMVA's five-file
format plus ``dataset.json``.

- ``mapping``: ``DatasetMapping``, ``load_mapping`` / ``dump_mapping`` (the TOML schema is in its docstring);
- ``profile``: column profiles with redacted examples, the only thing about the data an LLM ever sees;
- ``draft``: Claude Haiku drafts a mapping from the profiles (``outcome_confirmed = false``, cached on disk);
- ``convert``: deterministic conversion of a confirmed mapping (``frames_from_bytes`` reads raw CSVs);
- ``python -m emva.ingest draft|convert``: the command line.
"""
