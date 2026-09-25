"""Keel's custom components as HTML strings, plus number formatting. Pure Python (no Streamlit import).

Every builder escapes the text it is given; the pages pass the result to ``st.markdown(..., unsafe_allow_html=True)``.
Classes are styled by ``app.theme.inject_css``.
"""
from __future__ import annotations

import math
import html

from app.validation import Issue

# Keel glyph: a keel line under a hull, drawn in white on the accent square.
_GLYPH = ('<svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">'
          '<path d="M2 5h12l-2.5 4.5h-7z" fill="#fff"/><path d="M8 9.5V14" stroke="#fff" stroke-width="1.8" '
          'stroke-linecap="round"/></svg>')


def esc(value: object) -> str:
    """The one HTML-escaping helper: ``value`` as text with ``& < > " '`` escaped (None becomes "")."""
    return "" if value is None else html.escape(str(value), quote=True)


def _is_missing(x: float | None) -> bool:
    """True for None or NaN."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def gbp(x: float | None, dp: int = 0) -> str:
    """``£12,345`` (``–`` when missing)."""
    return "–" if _is_missing(x) else f"£{x:,.{dp}f}"


def pct(x: float | None, dp: int = 1) -> str:
    """``23.4%`` from a share 0-1 (``–`` when missing)."""
    return "–" if _is_missing(x) else f"{x * 100:.{dp}f}%"


def num(x: float | None, dp: int = 3) -> str:
    """Fixed decimals (``–`` when missing)."""
    return "–" if _is_missing(x) else f"{x:.{dp}f}"


def points(x: float | None) -> str:
    """Signed integer points, e.g. ``+12`` / ``−7`` (true minus sign)."""
    if _is_missing(x):
        return "–"
    v = int(round(x))
    return f"+{v}" if v > 0 else (f"−{abs(v)}" if v < 0 else "0")


def wordmark(name: str, tagline: str) -> str:
    """The sidebar wordmark: glyph, name in the display face, tagline in small caps."""
    return (f'<div class="k-wordmark"><div class="glyph">{_GLYPH}</div><div><div class="name">{esc(name)}</div>'
            f'<div class="tag">{esc(tagline)}</div></div></div>')


def page_header(eyebrow: str, title: str, lede: str) -> str:
    """Eyebrow, serif title and one muted sentence saying what the page shows."""
    return (f'<div class="k-eyebrow">{esc(eyebrow)}</div><div class="k-title" role="heading" aria-level="1">'
            f'{esc(title)}</div>'
            f'<p class="k-lede">{esc(lede)}</p>')


def section(title: str, text: str = "", step: str | None = None) -> str:
    """A section header with an optional step number and a short explanation in muted text."""
    s = f'<span class="step">{esc(step)}</span>' if step else ""
    p = f"<p>{esc(text)}</p>" if text else ""
    return f'<div class="k-section"><div class="h" role="heading" aria-level="3">{s}{esc(title)}</div>{p}</div>'


def delta(value: float | None, fmt: str = "pts", good_when: str = "up") -> str:
    """A delta chip: ``value`` formatted as percentage points (``pts``), ``auc`` (3 dp) or ``x`` (multiplier);
    teal when it moves in the ``good_when`` direction, warm red otherwise."""
    if _is_missing(value):
        return ""
    text = {"pts": f"{value * 100:+.1f} pts", "auc": f"{value:+.3f}", "x": f"{value:.1f}×"}[fmt]
    text = text.replace("-", "−")
    if abs(value) < 1e-12:
        cls = "flat"
    else:
        cls = "up" if (value > 0) == (good_when == "up") else "down"
    return f'<span class="k-delta {cls}">{esc(text)}</span>'


def kpi(label: str, value: str, sub: str = "", delta_html: str = "", unit: str = "", note: str = "") -> str:
    """One KPI card: label, value (mono) with an optional unit, a delta chip with a sub-label, and a mono note
    (e.g. a confidence interval)."""
    u = f"<small>{esc(unit)}</small>" if unit else ""
    n = f'<div class="note">{esc(note)}</div>' if note else ""
    return (f'<div class="k-kpi"><div class="label">{esc(label)}</div><div class="value">{esc(value)}{u}</div>'
            f'<div class="sub">{delta_html}{esc(sub)}</div>{n}</div>')


def kpi_row(cards: list[str]) -> str:
    """A responsive row of KPI cards."""
    return f'<div class="k-kpis">{"".join(cards)}</div>'


def stats(items: list[tuple[str, str]]) -> str:
    """A compact joined row of label/value stats."""
    cells = "".join(f'<div><div class="label">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
                    for k, v in items)
    return f'<div class="k-stats">{cells}</div>'


PILL_KINDS: dict[str, str] = {"succeeded": "ok", "running": "live", "queued": "info", "failed": "bad"}


def pill(text: str, kind: str = "info") -> str:
    """A status pill; ``kind`` is ``ok``, ``bad``, ``warn``, ``info`` or ``live`` (pulsing)."""
    return f'<span class="k-pill {esc(kind)}">{esc(text)}</span>'


def status_pill(status: str) -> str:
    """The pill for a run status (``RUN_STATUSES``)."""
    return pill(status.capitalize(), PILL_KINDS.get(status, "info"))


def table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    """A quiet HTML table; columns from ``numeric_from`` on are right-aligned in the mono face."""
    def cells(values: list[str], tag: str) -> str:
        return "".join(f'<{tag} class="{"n" if i >= numeric_from else ""}">{esc(v)}</{tag}>'
                       for i, v in enumerate(values))
    body = "".join(f"<tr>{cells(r, 'td')}</tr>" for r in rows)
    return f'<div class="k-table-wrap"><table class="k-table"><thead><tr>{cells(headers, "th")}</tr></thead>' \
           f"<tbody>{body}</tbody></table></div>"


def empty_state(title: str, text: str) -> str:
    """A dashed empty-state panel."""
    return f'<div class="k-empty"><div class="mark">{esc(title)}</div><p>{esc(text)}</p></div>'


def callout(text: str, kind: str = "info", lead: str = "") -> str:
    """A tinted note: an optional bold ``lead`` then ``text``; both are plain text and escaped here."""
    b = f"<b>{esc(lead)}</b> " if lead else ""
    return f'<div class="k-callout {esc(kind)}">{b}{esc(text)}</div>'


def run_header(status: str, run_id: str, detail: str = "") -> str:
    """A run's status pill, its id in mono and an optional muted detail line (all escaped)."""
    d = f'<span class="detail">{esc(detail)}</span>' if detail else ""
    return f'<div class="k-runhead">{status_pill(status)}<span class="id">run {esc(run_id)}</span>{d}</div>'


def file_card(name: str, label: str, errors: list[Issue], warnings: list[Issue], provided: bool) -> str:
    """A validation card for one file: name, status pill and its issues (errors red, warnings amber)."""
    if not provided and not errors:
        status = pill("Not provided", "info")
    elif errors:
        status = pill(f"{len(errors)} error{'s' * (len(errors) != 1)}", "bad")
    elif warnings:
        status = pill(f"Valid · {len(warnings)} note{'s' * (len(warnings) != 1)}", "warn")
    else:
        status = pill("Valid", "ok")
    body = ""
    for kind, issues in (("error", errors), ("warning", warnings)):
        for i in issues:
            col = f'<span class="col">{esc(i.column)}</span> ' if i.column else ""
            rows = (f'<div class="rows">rows {", ".join(map(str, i.rows))}{" …" if len(i.rows) >= 10 else ""}</div>'
                    if i.rows else "")
            body += (f'<div class="k-issue {kind}"><div class="bar"></div><div><div class="msg">{col}'
                     f'{esc(i.message)}</div>{rows}</div></div>')
    return (f'<div class="k-file"><div class="head"><div><div class="fname">{esc(name)}</div>'
            f'<div style="font-size:13px;color:var(--muted)">{esc(label)}</div></div>{status}</div>{body}</div>')


def result_card(p: float, base_rate: float, deal_value: float, value_at_submit: float, value_formula: float,
                transform: str, flags_html: str) -> str:
    """The scored-lead card: P(close) large with the base rate for context, expected deal value, the value sent at
    submit (or p × value for legacy-label runs) with how it is derived, and bot/duplicate flags."""
    ratio = p / base_rate if base_rate > 0 else math.nan
    rel = "" if _is_missing(ratio) else f" · {ratio:.1f}× the average lead"
    submit_label = "Value sent at submit" if not _is_missing(value_at_submit) else "p × deal value"
    submit_value = value_at_submit if not _is_missing(value_at_submit) else value_formula
    return (
        '<div class="k-result"><div class="eyebrow">Chance this lead closes</div>'
        f'<div class="big">{esc(pct(p))}</div>'
        f'<div class="context">Training win rate {esc(pct(base_rate))}{esc(rel)}</div>'
        '<div class="grid">'
        f'<div><div class="label">Expected deal value if won</div><div class="v">{esc(gbp(deal_value))}</div></div>'
        f'<div><div class="label">{esc(submit_label)}</div><div class="v">{esc(gbp(submit_value))}</div></div>'
        '</div>'
        f'<div class="note">{esc(transform)}</div>'
        f'<div class="flags">{flags_html}</div></div>')


__all__ = ["callout", "delta", "esc", "empty_state", "file_card", "gbp", "kpi", "kpi_row", "num", "page_header", "pct",
           "pill", "points", "result_card", "run_header", "section", "stats", "status_pill", "table", "wordmark"]
