"""Read settings from the environment, falling back to the repository's ``.env`` file (ADR 0010).

Shared by ``emva/context/agent.py`` and ``scripts/paraphrase_templates.py``. Values are returned, never
printed or logged. The ``.env`` search walks up from a start directory and stops at the repository root
(the first directory holding both ``emva/`` and ``.git``); it never looks above it. When that root is a
linked git worktree (``.git`` is a file, as under ``.claude/worktrees/``), the main checkout's root, where
the gitignored ``.env`` lives, is checked last.

Some hosts reserve or strip ``ANTHROPIC_API_KEY`` (the Claude Code cloud environment does), so a missing name
falls back to its ``EMVA_``-prefixed alias in ``ENV_ALIASES``, looked up the same way.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
# Setting name -> fallback name read when the setting is missing from both the environment and ``.env``.
ENV_ALIASES: dict[str, str] = {"ANTHROPIC_API_KEY": "EMVA_ANTHROPIC_API_KEY"}


def repo_root(start: str | Path) -> Path | None:
    """The first directory at or above ``start`` containing both ``emva/`` and ``.git``, else None."""
    d = Path(start).resolve()
    for cand in (d, *d.parents):
        if (cand / "emva").is_dir() and (cand / ".git").exists():
            return cand
    return None


def main_checkout(root: Path) -> Path | None:
    """For a linked worktree root (``.git`` file ``gitdir: <main>/.git/worktrees/<name>``), the main checkout."""
    git = root / ".git"
    if not git.is_file():
        return None
    first = git.read_text(encoding="utf-8").strip()
    if not first.startswith("gitdir:"):
        return None
    gitdir = Path(first.removeprefix("gitdir:").strip())
    gitdir = gitdir if gitdir.is_absolute() else (root / gitdir).resolve()
    return gitdir.parents[2] if gitdir.parent.name == "worktrees" else None


def find_dotenv(start: str | Path = PACKAGE_DIR) -> Path | None:
    """The first ``.env`` from ``start`` up to the repository root (then the main checkout for a worktree).

    None when ``start`` is not inside a repository or no ``.env`` is found within those bounds.
    """
    root = repo_root(start)
    if root is None:
        return None
    d = Path(start).resolve()
    dirs = [c for c in (d, *d.parents) if c == root or root in c.parents]
    main = main_checkout(root)
    if main is not None:
        dirs.append(main)
    for cand_dir in dirs:
        cand = cand_dir / ".env"
        if cand.is_file():
            return cand
    return None


def load_env_var(name: str, start: str | Path = PACKAGE_DIR) -> str | None:
    """``name`` from the environment, else from ``find_dotenv(start)`` (``NAME=value`` lines, optional
    ``export`` and quotes), else its ``ENV_ALIASES`` fallback looked up the same way, else None. Empty values
    count as unset."""
    value = _lookup(name, start)
    if value is None and name in ENV_ALIASES:
        value = _lookup(ENV_ALIASES[name], start)
    return value


def _lookup(name: str, start: str | Path) -> str | None:
    """``name`` from the environment, else from ``find_dotenv(start)``, else None (no alias fallback)."""
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
