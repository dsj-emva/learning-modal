"""Keel's visual identity: the palette, fonts and the one stylesheet (``inject_css``).

Direction: functional and quiet (Dieter Rams by way of a paper ledger). Warm paper background, white surfaces
with hairline borders instead of shadows, one accent (deep teal) used only for meaning (positive, primary
action, the model's line), a muted warm red for negative, amber for warnings. Instrument Sans for the interface,
Instrument Serif for page titles and the wordmark, IBM Plex Mono for every number. Spacing on an 8 px grid.
``.streamlit/config.toml`` sets the same colours for Streamlit's own widgets and loads the three font families.
"""
from __future__ import annotations

import streamlit as st

APP_NAME: str = "Keel"
APP_TAGLINE: str = "Lead scoring by EMVA"

PALETTE: dict[str, str] = {
    "paper": "#F6F4EF", "surface": "#FFFFFF", "sunken": "#F1EEE8", "hairline": "#E2DDD2", "ink": "#1B1F1E",
    "muted": "#6B6F6A", "faint": "#9A9C95", "accent": "#0F5C5A", "accent_dark": "#0B4644", "accent_tint": "#E2EEEB",
    "negative": "#B5543C", "negative_tint": "#F6E5DF", "warning": "#B7791F", "warning_tint": "#F7EBD3",
    "sand": "#C8A36A",
}
FONT_UI, FONT_DISPLAY, FONT_MONO = "Instrument Sans", "Instrument Serif", "IBM Plex Mono"


def _css() -> str:
    """The stylesheet: tokens, Streamlit chrome overrides and the custom components of ``app.components``."""
    p = PALETTE
    return f"""
:root {{
  --paper:{p['paper']}; --surface:{p['surface']}; --sunken:{p['sunken']}; --hairline:{p['hairline']};
  --ink:{p['ink']}; --muted:{p['muted']}; --faint:{p['faint']}; --accent:{p['accent']};
  --accent-dark:{p['accent_dark']}; --accent-tint:{p['accent_tint']}; --neg:{p['negative']};
  --neg-tint:{p['negative_tint']}; --warn:{p['warning']}; --warn-tint:{p['warning_tint']};
  --ui:'{FONT_UI}', system-ui, sans-serif; --display:'{FONT_DISPLAY}', Georgia, serif;
  --mono:'{FONT_MONO}', ui-monospace, monospace; --r:10px;
}}
/* chrome: no Streamlit menu, footer, deploy button or decoration bar */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"], [data-testid="stStatusWidget"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:2.5rem; padding-bottom:4rem; max-width:1240px; }}
[data-testid="stSidebar"] {{ border-right:1px solid var(--hairline); }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a {{ border-radius:8px; padding:6px 10px; }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a span {{ font-weight:500; color:var(--ink); }}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"],
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {{ background:var(--accent-tint); }}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] span {{ color:var(--accent-dark); }}
code, pre, [data-testid="stCode"] {{ font-family:var(--mono) !important; }}
[data-testid="stExpander"] details {{ border-color:var(--hairline); background:var(--surface); border-radius:var(--r); }}
[data-testid="stDataFrame"] {{ border-radius:var(--r); }}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primaryFormSubmit"],
.stDownloadButton button {{ font-weight:600; letter-spacing:.01em; }}
[data-testid="stFileUploaderDropzone"] {{ background:var(--surface); border:1px dashed var(--hairline); }}
hr {{ border-color:var(--hairline) !important; }}

/* wordmark */
.k-wordmark {{ display:flex; align-items:center; gap:10px; padding:4px 0 16px; }}
.k-wordmark .glyph {{ width:28px; height:28px; border-radius:8px; background:var(--accent); display:grid;
  place-items:center; }}
.k-wordmark .glyph svg {{ display:block; }}
.k-wordmark .name {{ font-family:var(--display); font-size:28px; line-height:1; color:var(--ink); }}
.k-wordmark .tag {{ font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
  margin-top:3px; }}

/* page header */
.k-eyebrow {{ font-size:12px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--accent); }}
.k-title {{ font-family:var(--display) !important; font-weight:400 !important; font-size:48px; line-height:1.05; color:var(--ink);
  margin:8px 0 8px; letter-spacing:-.01em; }}
.k-lede {{ color:var(--muted); font-size:16px; max-width:72ch; margin:0 0 24px; line-height:1.5; }}

/* section header */
.k-section {{ margin:40px 0 12px; }}
.k-section .h {{ font-family:var(--ui); font-size:18px; font-weight:600; margin:0; padding:0; color:var(--ink); }}
.k-section p {{ margin:4px 0 0; color:var(--muted); font-size:14px; line-height:1.5; max-width:80ch; }}
.k-section .step {{ font-family:var(--mono); font-size:12px; color:var(--accent); margin-right:8px; }}

/* KPI cards */
.k-kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:16px; margin:8px 0 8px; }}
.k-kpi {{ background:var(--surface); border:1px solid var(--hairline); border-radius:var(--r); padding:16px 20px 16px; }}
.k-kpi .label {{ font-size:13px; color:var(--muted); font-weight:500; }}
.k-kpi .value {{ font-family:var(--mono); font-size:32px; font-weight:500; color:var(--ink); margin:8px 0 4px;
  letter-spacing:-.02em; line-height:1.1; }}
.k-kpi .value small {{ font-size:14px; color:var(--muted); letter-spacing:0; margin-left:6px; }}
.k-kpi .sub {{ font-size:12px; color:var(--faint); line-height:1.4; display:flex; align-items:center; flex-wrap:wrap; gap:4px 0; }}
.k-kpi .note {{ font-size:12px; color:var(--faint); margin-top:6px; font-family:var(--mono); }}
.k-delta {{ display:inline-block; font-family:var(--mono); font-size:12px; padding:2px 8px; border-radius:999px;
  margin-right:6px; }}
.k-delta.up {{ background:var(--accent-tint); color:var(--accent-dark); }}
.k-delta.down {{ background:var(--neg-tint); color:var(--neg); }}
.k-delta.flat {{ background:var(--sunken); color:var(--muted); }}

/* pills */
.k-pill {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:600; padding:3px 10px;
  border-radius:999px; letter-spacing:.02em; white-space:nowrap; }}
.k-pill::before {{ content:""; width:6px; height:6px; border-radius:50%; background:currentColor; }}
.k-pill.ok {{ background:var(--accent-tint); color:var(--accent-dark); }}
.k-pill.bad {{ background:var(--neg-tint); color:var(--neg); }}
.k-pill.warn {{ background:var(--warn-tint); color:var(--warn); }}
.k-pill.info {{ background:var(--sunken); color:var(--muted); }}
.k-pill.live {{ background:var(--accent-tint); color:var(--accent-dark); }}
.k-pill.live::before {{ animation:k-pulse 1.2s ease-in-out infinite; }}
@keyframes k-pulse {{ 50% {{ opacity:.25; }} }}
@media (prefers-reduced-motion: reduce) {{ .k-pill.live::before {{ animation:none; }} }}

/* empty state */
.k-empty {{ border:1px dashed var(--hairline); border-radius:14px; padding:48px 32px; text-align:center;
  background:var(--surface); margin:16px 0; }}
.k-empty .mark {{ font-family:var(--display); font-size:34px; color:var(--ink); margin-bottom:8px; }}
.k-empty p {{ color:var(--muted); max-width:52ch; margin:0 auto; line-height:1.55; }}

/* validation issues */
.k-file {{ background:var(--surface); border:1px solid var(--hairline); border-radius:var(--r); padding:16px 20px;
  margin-bottom:12px; }}
.k-file .head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }}
.k-file .fname {{ font-family:var(--mono); font-size:14px; color:var(--ink); }}
.k-issue {{ display:grid; grid-template-columns:4px 1fr; gap:12px; margin-top:12px; }}
.k-issue .bar {{ border-radius:2px; }}
.k-issue.error .bar {{ background:var(--neg); }}
.k-issue.warning .bar {{ background:var(--warn); }}
.k-issue .col {{ font-family:var(--mono); font-size:12px; background:var(--sunken); padding:1px 6px;
  border-radius:4px; color:var(--ink); }}
.k-issue .msg {{ font-size:14px; color:var(--ink); line-height:1.45; }}
.k-issue .rows {{ font-size:12px; color:var(--faint); margin-top:2px; font-family:var(--mono); }}

/* scored lead */
.k-result {{ background:var(--surface); border:1px solid var(--hairline); border-radius:14px; padding:24px 24px 20px;
  border-top:4px solid var(--accent); }}
.k-result .eyebrow {{ font-size:12px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:600; }}
.k-result .big {{ font-family:var(--mono); font-size:64px; font-weight:500; letter-spacing:-.04em; line-height:1;
  color:var(--ink); margin:12px 0 8px; }}
.k-result .context {{ color:var(--muted); font-size:14px; line-height:1.5; }}
.k-result .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:20px; padding-top:20px;
  border-top:1px solid var(--hairline); }}
.k-result .grid .label {{ font-size:12px; color:var(--muted); }}
.k-result .grid .v {{ font-family:var(--mono); font-size:22px; color:var(--ink); margin-top:4px; }}
.k-result .note {{ font-size:12px; color:var(--faint); margin-top:16px; line-height:1.5; }}
.k-result .flags {{ margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; }}

/* callout */
.k-callout {{ border-radius:var(--r); padding:14px 16px; font-size:14px; line-height:1.5; margin:8px 0 16px; }}
.k-callout.info {{ background:var(--sunken); color:var(--ink); }}
.k-callout.bad {{ background:var(--neg-tint); color:#6E2E1E; }}
.k-callout.warn {{ background:var(--warn-tint); color:#6A4610; }}
.k-callout b {{ font-weight:600; }}

/* stat row */
.k-stats {{ display:flex; flex-wrap:wrap; background:var(--surface); border:1px solid var(--hairline);
  border-radius:var(--r); overflow:hidden; margin:8px 0 16px; }}
.k-stats > div {{ flex:1 1 140px; padding:12px 16px; box-shadow:inset -1px 0 0 var(--sunken), inset 0 -1px 0 var(--sunken); }}
.k-stats .label {{ font-size:12px; color:var(--muted); }}
.k-stats .v {{ font-family:var(--mono); font-size:18px; color:var(--ink); margin-top:4px; }}

.k-table-wrap {{ overflow-x:auto; background:var(--surface); border:1px solid var(--hairline); border-radius:var(--r); }}
.k-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
.k-table th {{ text-align:left; font-weight:500; font-size:12px; color:var(--muted); padding:12px 16px;
  border-bottom:1px solid var(--hairline); background:var(--sunken); vertical-align:bottom; line-height:1.3; }}
.k-table td {{ padding:12px 16px; border-bottom:1px solid var(--sunken); color:var(--ink); white-space:nowrap; }}
.k-table tr:last-child td {{ border-bottom:none; }}
.k-table .n {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.k-table td:first-child {{ font-weight:600; }}

/* login */
.k-login {{ max-width:420px; margin:12vh auto 24px; }}
.k-login .k-title {{ font-size:52px; }}
.k-footnote {{ font-size:12px; color:var(--faint); margin-top:32px; line-height:1.5; }}
@media (max-width: 640px) {{ .k-title {{ font-size:34px; }} .k-result .big {{ font-size:48px; }}
  .block-container {{ padding-left:16px; padding-right:16px; }} }}
"""


def inject_css() -> None:
    """Add the stylesheet to the current page (call once per script run, before any content)."""
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


__all__ = ["APP_NAME", "APP_TAGLINE", "FONT_DISPLAY", "FONT_MONO", "FONT_UI", "PALETTE", "inject_css"]
