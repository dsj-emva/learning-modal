"""Paraphrase every free-text template once through Claude Haiku and keep the results in an on-disk cache.

Usage:
  python scripts/paraphrase_templates.py --check       # list templates missing from the cache (no API call)
  python scripts/paraphrase_templates.py --populate    # call the API for every missing template
  python scripts/paraphrase_templates.py --probe       # one 5-token call to check credentials and model

The cache (``scripts/paraphrase_cache.json``, committed) is keyed by (template text, model id, prompt version):
changing the model or the prompt makes every entry a miss, so old paraphrases are never silently reused. One
API call per template, temperature 0, N paraphrases per call, returned as structured JSON and validated (right
count, distinct, non-empty, identical ``{placeholder}`` set). The generator only ever reads the cache; with
``Config.text_paraphrase`` on it raises ``MissingParaphraseError`` for any template not in it.

``ANTHROPIC_API_KEY`` (and, if set, ``ANTHROPIC_WORKSPACE_ID``, sent as the ``anthropic-workspace-id``
header) are read from the environment, falling back to a ``.env`` file found by walking up from this script's
directory. Neither value is ever printed or written anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "paraphrase_cache.json")
MODEL_ID = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "v2"   # v1 named "{team}" as an example and Haiku copied it into answers without placeholders
N_PARAPHRASES = 5
POPULATE_CMD = "python scripts/paraphrase_templates.py --populate"

PROMPT = """You are helping build realistic synthetic data for a B2B lead-scoring test. Below is one answer \
that a person typed into the free-text box "What do you want to solve?" on a web form.

Write {n} different paraphrases of it, as {n} different real people might have typed the same thing. Rules:
- Keep the meaning, tone and level of detail. Do not add facts, numbers, names or requests that are not there.
- Vary wording and sentence structure; casual, terse and formal styles are all fine.
- {placeholder_rule}
- If the answer is a short fragment, keep the paraphrases short too.
- Write in the same language as the original.

Answer:
{template}"""


class MissingParaphraseError(RuntimeError):
    """Raised when the paraphrase option is on and the cache lacks a template."""


def placeholders(text: str) -> set[str]:
    """Names of the ``{placeholder}`` fields in ``text`` (as used by ``str.format``)."""
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def cache_key(template: str, model: str = MODEL_ID, prompt_version: str = PROMPT_VERSION) -> str:
    """Stable cache key for (template text, model id, prompt version)."""
    return hashlib.sha256(json.dumps([template, model, prompt_version]).encode()).hexdigest()


def load_cache(path: str = CACHE_PATH) -> dict[str, dict]:
    """The cache entries keyed by ``cache_key``; empty dict if the file does not exist."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)["entries"]


def save_cache(entries: dict[str, dict], path: str = CACHE_PATH) -> None:
    """Write the cache deterministically (sorted keys) so re-saving an unchanged cache is a no-op diff."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_ID, "prompt_version": PROMPT_VERSION, "n_paraphrases": N_PARAPHRASES,
                   "entries": entries}, f, indent=1, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def validate(template: str, paraphrases: list[str], n: int = N_PARAPHRASES) -> None:
    """Raise ValueError unless ``paraphrases`` is ``n`` distinct non-empty strings with the template's placeholders."""
    if len(paraphrases) != n:
        raise ValueError(f"expected {n} paraphrases, got {len(paraphrases)} for {template!r}")
    if len({p.strip() for p in paraphrases}) != n or any(not p.strip() for p in paraphrases):
        raise ValueError(f"paraphrases must be distinct and non-empty for {template!r}: {paraphrases}")
    want = placeholders(template)
    for p in paraphrases:
        if placeholders(p) != want:
            raise ValueError(f"placeholder mismatch in {p!r} (want {sorted(want)}) for {template!r}")


def missing(templates: list[str], entries: dict[str, dict]) -> list[str]:
    """Templates without a cache entry for the current model and prompt version."""
    return [t for t in templates if cache_key(t) not in entries]


def lookup(templates: list[str], path: str | None = None) -> dict[str, list[str]]:
    """Paraphrases for every template, from the cache only (default: CACHE_PATH, read at call time).

    Raises MissingParaphraseError if any template is absent."""
    path = path or CACHE_PATH
    entries = load_cache(path)
    miss = missing(templates, entries)
    if miss:
        raise MissingParaphraseError(
            f"text_paraphrase is on but {len(miss)} of {len(templates)} templates are not in {path} for model "
            f"{MODEL_ID}, prompt {PROMPT_VERSION} (first missing: {miss[0]!r}). Set ANTHROPIC_API_KEY (or put it "
            f"in a .env file at the repo root) and run `{POPULATE_CMD}`, or generate without paraphrase "
            f"(--no-paraphrase).")
    out = {}
    for t in templates:
        paras = entries[cache_key(t)]["paraphrases"]
        validate(t, paras, len(paras))
        out[t] = paras
    return out


def find_dotenv(start: str = HERE) -> str | None:
    """Path of the first ``.env`` file found walking up from ``start`` to the filesystem root, else None."""
    d = os.path.abspath(start)
    while True:
        cand = os.path.join(d, ".env")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def load_env_var(name: str) -> str | None:
    """``name`` from the environment, else from the nearest ``.env`` (``NAME=value`` lines), else None."""
    value = os.environ.get(name)
    if value:
        return value
    path = find_dotenv()
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                return value.strip().strip('"').strip("'") or None
    return None


def make_client() -> "anthropic.Anthropic":  # noqa: F821
    """Anthropic client from ANTHROPIC_API_KEY, adding the ``anthropic-workspace-id`` header when
    ANTHROPIC_WORKSPACE_ID is set (keys that are not workspace-scoped need it). Exits if no key is found."""
    import anthropic
    key = load_env_var("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set and no .env with it was found; cannot call the API.")
    workspace = load_env_var("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace} if workspace else None
    return anthropic.Anthropic(api_key=key, default_headers=headers)


def probe(client: "anthropic.Anthropic") -> None:  # noqa: F821
    """A 5-token request to MODEL_ID; raises the SDK's typed error if the credentials or model are unusable."""
    client.messages.create(model=MODEL_ID, max_tokens=5, messages=[{"role": "user", "content": "ping"}])


def paraphrase(client: "anthropic.Anthropic", template: str, n: int = N_PARAPHRASES) -> list[str]:  # noqa: F821
    """One API call: ``n`` validated paraphrases of ``template`` from MODEL_ID at temperature 0."""
    schema = {"type": "object", "additionalProperties": False, "required": ["paraphrases"],
              "properties": {"paraphrases": {"type": "array", "items": {"type": "string"}}}}
    names = sorted(placeholders(template))
    rule = ("Keep the placeholders " + ", ".join("{" + x + "}" for x in names) + " exactly as written, once each."
            if names else "Do not use curly braces or placeholders.")
    resp = client.messages.create(
        model=MODEL_ID, max_tokens=2048,
        messages=[{"role": "user", "content": PROMPT.format(n=n, template=template, placeholder_rule=rule)}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
        extra_body={"temperature": 0},      # anthropic 1.x dropped the typed parameter; Haiku 4.5 still honours it
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"unexpected stop_reason {resp.stop_reason!r} for {template!r} (request {resp._request_id})")
    text = next(b.text for b in resp.content if b.type == "text")
    paras = [p.strip() for p in json.loads(text)["paraphrases"]]
    validate(template, paras, n)
    return paras


def populate(templates: list[str], path: str = CACHE_PATH) -> int:
    """Fill every missing template (saving after each call so an interruption loses nothing). Returns calls made."""
    client = make_client()
    probe(client)
    entries = load_cache(path)
    todo = missing(templates, entries)
    for i, t in enumerate(todo, 1):
        entries[cache_key(t)] = {"template": t, "model": MODEL_ID, "prompt_version": PROMPT_VERSION,
                                 "paraphrases": paraphrase(client, t)}
        save_cache(entries, path)
        print(f"[{i}/{len(todo)}] {t[:70]!r}")
    return len(todo)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--populate", action="store_true", help="call the API for every template missing from the cache")
    g.add_argument("--check", action="store_true", help="list templates missing from the cache")
    g.add_argument("--probe", action="store_true", help="one 5-token API call to check credentials and model")
    a = ap.parse_args(argv)
    if a.probe:
        import anthropic
        try:
            probe(make_client())
        except anthropic.APIError as e:           # report the failure without echoing credentials
            raise SystemExit(f"probe failed: {type(e).__name__}: {getattr(e, 'message', e)}")
        print(f"probe ok: {MODEL_ID} reachable")
        return
    sys.path.insert(0, HERE)
    from generate_data_v1 import all_text_templates  # the generator imports this module, so import lazily
    templates = all_text_templates()
    if a.check:
        miss = missing(templates, load_cache())
        print(f"{len(templates) - len(miss)} of {len(templates)} templates cached ({MODEL_ID}, prompt {PROMPT_VERSION}).")
        for t in miss:
            print("  missing:", repr(t))
        sys.exit(1 if miss else 0)
    n = populate(templates)
    print(f"{n} API calls; cache now covers all {len(templates)} templates.")


if __name__ == "__main__":
    main()
