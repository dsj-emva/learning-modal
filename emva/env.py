"""Read settings from the environment, falling back to the nearest ``.env`` file (ADR 0010).

Shared by ``emva/context/agent.py`` and ``scripts/paraphrase_templates.py``. Values are returned, never
printed or logged. In a git worktree under ``.claude/worktrees/`` the walk up reaches the main checkout's
``.env``.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def find_dotenv(start: str | Path = PACKAGE_DIR) -> Path | None:
    """The first ``.env`` file found walking up from ``start`` to the filesystem root, else None."""
    d = Path(start).resolve()
    for cand_dir in (d, *d.parents):
        cand = cand_dir / ".env"
        if cand.is_file():
            return cand
    return None


def load_env_var(name: str, start: str | Path = PACKAGE_DIR) -> str | None:
    """``name`` from the environment, else from the nearest ``.env`` above ``start`` (``NAME=value`` lines,
    optional ``export`` and quotes), else None. Empty values count as unset."""
    value = os.environ.get(name)
    if value:
        return value
    path = find_dotenv(start)
    if path is None:
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ")
        key, sep, val = line.partition("=")
        if sep and key.strip() == name:
            return val.strip().strip('"').strip("'") or None
    return None
