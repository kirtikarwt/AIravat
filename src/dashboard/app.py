"""
AIravat — Extreme-Weather Anomaly Intelligence Dashboard (SIH 2026 · PS 26078 · MoES)
Layout mirrors VahniX: fixed top navigation, ?page= routing, cinematic home page,
settings in a collapsible panel at the bottom, no sidebar.

Run from the project root:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import base64
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
import sys

_DASH_DIR = Path(__file__).resolve().parent
if str(_DASH_DIR) not in sys.path:
    sys.path.insert(0, str(_DASH_DIR))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

import engine as E

ASSETS = Path(__file__).parent / "assets"
FAVICON = ASSETS / "favicon.png"

st.set_page_config(
    page_title="AIravat · Extreme-Weather Intelligence",
    page_icon=str(FAVICON) if FAVICON.exists() else "🐘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
#  THEME
# ---------------------------------------------------------------------------
BG, CARD, LINE = "#EBEEF6", "#FFFFFF", "#DCE1EC"
NAVY, BLUE, BODY, MUTED = "#1E2A45", "#2F80ED", "#4A5468", "#8A93A6"
TIER_COLOURS = {0: "#2E9E5B", 1: "#E8C547", 2: "#EE8A2F", 3: "#D6453D", 4: "#8E1B4A"}
RAIN_SCALE = [[0.0, "#F4F6FB"], [0.12, "#CFE0F7"], [0.3, "#7FB3EE"], [0.48, "#2F80ED"],
              [0.64, "#E8C547"], [0.8, "#EE8A2F"], [0.92, "#D6453D"], [1.0, "#8E1B4A"]]
HEAT_SCALE = [[0.0, "#CFE0F7"], [0.35, "#7FB3EE"], [0.55, "#E8C547"], [0.75, "#EE8A2F"], [0.9, "#D6453D"], [1.0, "#7A1330"]]
EFI_SCALE = [[0.0, "#F4F6FB"], [0.35, "#BFD5F5"], [0.55, "#E8C547"], [0.75, "#EE8A2F"], [0.9, "#D6453D"], [1.0, "#8E1B4A"]]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700;800&family=Source+Sans+3:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {{
  --bg:{BG}; --card:{CARD}; --line:{LINE}; --navy:{NAVY}; --blue:{BLUE}; --body:{BODY}; --muted:{MUTED};
  --font-display:'Montserrat','Segoe UI',sans-serif; --font-body:'Source Sans 3','Segoe UI',sans-serif; --font-mono:'JetBrains Mono',Menlo,monospace;
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, [data-testid="stMainBlockContainer"] {{
  background-color: var(--bg) !important; color: var(--body); font-family: var(--font-body) !important;
}}
p, li, label {{ font-family: var(--font-body) !important; }}
#MainMenu, footer, [data-testid="stToolbar"], header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {{
  display:none !important; visibility:hidden !important; height:0 !important;
}}
.block-container, [data-testid="stMainBlockContainer"] {{
  padding-top:0 !important; padding-bottom:3rem !important; padding-left:2.4rem !important; padding-right:2.4rem !important;
  max-width:100% !important; box-sizing:border-box !important;
}}
html, body, [data-testid="stAppViewContainer"], .main {{ overflow-x:hidden !important; }}

/* ===== TOP NAV ===== */
.nav-shell {{ position:fixed; top:0; left:0; width:100vw; z-index:10000; background:#FFFFFF;
  transform:translateY(0); transition:transform .35s cubic-bezier(.4,0,.2,1), box-shadow .35s ease; border-bottom:1px solid var(--line); }}
.nav-shell.nav-hidden {{ transform:translateY(-105%); }}
.nav-shell.nav-floating {{ box-shadow:0 6px 24px rgba(30,42,69,.12); }}
.nav-shell-spacer {{ height:76px; }}
.top-status-row {{ display:flex; align-items:center; justify-content:flex-end; background:var(--navy); padding:.35rem 2.4rem; box-sizing:border-box; }}
.top-brand-status {{ font-family:var(--font-mono) !important; font-size:.6rem; letter-spacing:.12em; color:#C9D6EE; display:flex; align-items:center; gap:.5rem; text-transform:uppercase; white-space:nowrap; }}
.status-pulse-dot {{ width:7px; height:7px; background:#2ECC71; border-radius:50%; box-shadow:0 0 8px #2ECC71; animation:status-pulse 2s infinite; }}
@keyframes status-pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:.35;transform:scale(.85)}} }}
.top-brand-bar {{ display:flex; align-items:center; background:#FFFFFF; padding:.55rem 2.4rem; box-sizing:border-box; }}
.top-brand-logo {{ display:flex; align-items:center; margin-left:.6rem; margin-right:1.6rem; flex-shrink:0; text-decoration:none !important; }}
.brand-logo-img {{ height:50px; width:auto; display:block; }}
.top-brand-bar .nav-item {{ font-family:var(--font-display) !important; font-size:.74rem; font-weight:600; letter-spacing:.03em; text-transform:uppercase;
  color:var(--navy) !important; white-space:nowrap; padding:.35rem 0; border-bottom:2px solid transparent; text-decoration:none !important; margin-right:1.55rem; transition:all .2s ease; }}
.top-brand-bar .nav-item:hover {{ color:var(--blue) !important; }}
.top-brand-bar .nav-item.active {{ color:var(--navy) !important; border-bottom:2px solid var(--blue); }}
.top-brand-bar .nav-item.nav-item-tech {{ margin-left:auto !important; margin-right:0 !important; color:var(--blue) !important; padding:.35rem .85rem !important;
  border:1px solid rgba(47,128,237,.45) !important; border-bottom:2px solid var(--blue) !important; border-radius:4px !important; background:rgba(47,128,237,.08) !important; }}
.top-brand-bar .nav-item.nav-item-tech:hover, .top-brand-bar .nav-item.nav-item-tech.active {{ background:rgba(47,128,237,.18) !important; }}

/* ===== HERO ===== */
.hero-container {{ position:relative; width:100vw; height:82vh; overflow:hidden; display:flex; align-items:center; justify-content:center;
  margin-left:calc(-50vw + 50%); background:radial-gradient(ellipse at 50% 40%, #2B4A7E 0%, #16223A 60%, #0C1424 100%); }}
.hero-video {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; z-index:0; opacity:.95; }}
.hero-overlay {{ position:absolute; inset:0; z-index:1; background:linear-gradient(180deg, rgba(12,20,36,.25) 0%, rgba(12,20,36,.1) 40%, rgba(12,20,36,.45) 80%, rgba(12,20,36,.8) 100%); }}
.hero-content {{ position:relative; z-index:2; text-align:center; padding:0 2rem; }}
.hero-badge {{ display:inline-block; font-family:var(--font-mono) !important; font-size:.66rem; letter-spacing:.22em; text-transform:uppercase; color:#9CC3FF;
  border:1px solid rgba(156,195,255,.4); padding:.3rem 1.2rem; border-radius:2px; margin-bottom:1.6rem; background:rgba(47,128,237,.12); }}
.hero-logo {{ display:block; width:min(50vw,640px); max-width:100%; height:auto; margin:0 auto .8rem; filter:drop-shadow(0 0 40px rgba(47,128,237,.35));
  animation:hero-logo-in 1.5s cubic-bezier(.16,1,.3,1) both; }}
@keyframes hero-logo-in {{ from{{opacity:0;transform:translateY(22px) scale(.965)}} to{{opacity:1;transform:none}} }}
.hero-subtitle {{ font-family:var(--font-display) !important; font-size:clamp(.72rem,1.2vw,.98rem); font-weight:600; letter-spacing:.22em; text-transform:uppercase; color:#D6E2F5; margin-bottom:1.6rem; }}
.hero-cta {{ display:inline-block; font-family:var(--font-mono) !important; font-size:.72rem; font-weight:600; letter-spacing:.26em; text-transform:uppercase;
  text-decoration:none !important; color:#FFFFFF !important; padding:.85rem 2.6rem; border:1px solid var(--blue); border-radius:2px; background:rgba(47,128,237,.45); transition:all .25s ease; }}
.hero-cta:hover {{ background:var(--blue); transform:translateY(-2px); box-shadow:0 0 26px rgba(47,128,237,.45); }}

/* ===== INTRO TEXT ===== */
.text-intro-section {{ padding:4.2rem 3rem; background:var(--bg); }}
.text-intro-title {{ font-family:var(--font-display) !important; font-weight:800; font-size:clamp(2rem,4.6vw,3.8rem); line-height:1.15; margin:0; color:var(--navy); }}
.text-intro-title span {{ color:var(--blue); }}
.text-intro-typed {{ margin-top:1.8rem; }}
.text-intro-typed .typed-line {{ display:block; font-family:var(--font-body) !important; font-size:clamp(1.3rem,2.1vw,2rem); line-height:1.5; color:var(--muted);
  white-space:nowrap; overflow:hidden; width:0; border-right:2px solid var(--muted); }}
.text-intro-typed .typed-line-2 {{ border-right-color:transparent; }}
@keyframes intro-caret {{ 0%{{border-right-color:var(--muted)}} 50%{{border-right-color:transparent}} 99%{{border-right-color:var(--muted)}} 100%{{border-right-color:transparent}} }}

/* ===== MEDIA SECTIONS ===== */
.video-section {{ position:relative; width:100vw; height:80vh; overflow:hidden; margin-left:calc(-50vw + 50%); display:flex; align-items:flex-start; justify-content:flex-end;
  padding:5rem 4rem; box-sizing:border-box; background:linear-gradient(135deg, #1E2A45 0%, #2F5FA8 100%); }}
.video-section.align-left {{ justify-content:flex-start; }}
.video-section video, .video-section img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; opacity:.95; }}
.video-section .overlay {{ position:absolute; inset:0; z-index:1; background:linear-gradient(180deg, rgba(12,20,36,.5) 0%, rgba(12,20,36,.1) 40%, rgba(12,20,36,.35) 100%); }}
.video-section.dark-text .overlay {{ background:linear-gradient(180deg, rgba(235,238,246,.65) 0%, rgba(235,238,246,.1) 45%, rgba(235,238,246,0) 100%); }}
.video-section .section-text {{ position:relative; z-index:2; max-width:48rem; text-align:right; }}
.video-section.align-left .section-text {{ text-align:left; }}
.video-section .section-text h2 {{ font-family:var(--font-display) !important; font-size:clamp(1.8rem,3.4vw,2.8rem); font-weight:800; line-height:1.15; margin-bottom:1rem; color:#FFFFFF; }}
.video-section .section-text p {{ font-family:var(--font-body) !important; font-size:clamp(1rem,1.3vw,1.3rem); font-weight:600; line-height:1.5; color:rgba(240,244,252,.92);
  text-shadow:0 1px 12px rgba(0,0,0,.5); text-wrap:balance; }}
.video-section.dark-text .section-text h2 {{ color:var(--navy); }}
.video-section.dark-text .section-text p {{ color:var(--navy); text-shadow:0 1px 12px rgba(255,255,255,.5); }}
.media-slot {{ position:absolute; bottom:1.2rem; left:1.6rem; z-index:2; font-family:var(--font-mono) !important; font-size:.66rem; letter-spacing:.14em; text-transform:uppercase;
  color:rgba(255,255,255,.75); border:1px dashed rgba(255,255,255,.5); padding:.35rem .8rem; border-radius:4px; }}
.video-section.page-header {{ height:46vh; align-items:center; justify-content:center; padding:2rem; margin-bottom:1.5rem; }}
.video-section.page-header .section-text {{ text-align:center; }}

/* ===== PAGE ELEMENTS ===== */
.section-header {{ font-family:var(--font-display) !important; font-size:1.55rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; color:var(--navy); margin:2rem 0 .3rem 0; }}
.section-header span {{ color:var(--blue); }}
.section-subtitle {{ font-family:var(--font-body) !important; font-size:.95rem; color:var(--muted); margin-bottom:1.4rem; }}
.ax-card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin-bottom:12px; box-shadow:0 1px 2px rgba(30,42,69,.05); }}
.ax-card h4 {{ margin:0 0 6px 0; font-size:1.05rem; font-family:var(--font-display) !important; color:var(--navy); }}
.ax-card p {{ margin:0; color:var(--body); font-size:.95rem; line-height:1.45; }}
.ax-pill {{ display:inline-block; padding:3px 10px; border-radius:999px; font-weight:600; font-size:.78rem; color:#fff; }}
.ax-banner {{ border-radius:14px; padding:18px 22px; margin-bottom:14px; color:#fff; }}
.ax-banner .t {{ font-family:var(--font-display) !important; font-size:1.8rem; font-weight:800; line-height:1.1; }}
.ax-banner .s {{ font-size:1rem; opacity:.95; margin-top:4px; }}
.ax-step {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; height:100%; }}
.ax-step .n {{ color:var(--blue); font-weight:800; font-family:var(--font-display) !important; font-size:1.3rem; }}
.ax-step .h {{ color:var(--navy); font-weight:700; margin:2px 0 4px 0; font-family:var(--font-display) !important; }}
.ax-step .d {{ color:var(--body); font-size:.92rem; }}
.ax-feed {{ font-family:var(--font-mono) !important; font-size:.82rem; background:var(--navy); color:#D6E4FA; border-radius:10px; padding:12px 14px; line-height:1.6; }}
.ax-feed .ok {{ color:#7CD992; }} .ax-feed .warn {{ color:#F2C230; }} .ax-feed .t {{ color:#8FB6F0; }}
.feed {{ background:var(--navy); border-radius:12px; padding:6px 0; font-family:var(--font-mono) !important; }}
.feed-scroll {{ max-height:470px; overflow-y:auto; display:flex; flex-direction:column-reverse; }}
.feed-head, .feed-row {{ display:grid; grid-template-columns:92px 92px 1fr 104px; gap:12px; align-items:center; padding:7px 16px; }}
.feed-head {{ font-size:.62rem; letter-spacing:.16em; text-transform:uppercase; color:#8FA3C4; border-bottom:1px solid rgba(255,255,255,.08); }}
.feed-row {{ font-size:.8rem; color:#E3EAF7; border-bottom:1px solid rgba(255,255,255,.04); }}
.feed-row:last-child {{ border-bottom:none; }}
.feed-row .tm {{ color:#8FB6F0; }}
.feed-row .stg {{ font-size:.62rem; font-weight:600; letter-spacing:.1em; text-align:center; padding:3px 0; border-radius:4px; }}
.feed-row .msg {{ font-family:var(--font-body) !important; font-size:.92rem; }}
.feed-row .msg b {{ color:#FFFFFF; font-weight:600; }}
.feed-row .st {{ font-size:.66rem; font-weight:600; letter-spacing:.08em; text-align:center; padding:3px 0; border-radius:999px; }}
.st-ok {{ background:rgba(124,217,146,.15); color:#7CD992; }}
.st-review {{ background:rgba(242,194,48,.18); color:#F2C230; }}
.st-run {{ background:rgba(143,182,240,.15); color:#8FB6F0; }}
.plain {{ background:var(--card); border:1px solid var(--line); border-left:5px solid var(--blue); border-radius:12px; padding:16px 20px; margin:0 0 18px 0; }}
.plain .lbl {{ font-family:var(--font-mono) !important; font-size:.62rem; letter-spacing:.16em; text-transform:uppercase; color:var(--blue); margin-bottom:4px; }}
.plain .txt {{ font-family:var(--font-body) !important; font-size:1.12rem; line-height:1.5; color:var(--navy); }}
.plain .txt b {{ font-weight:700; }}
.gloss {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }}
.gloss div {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; font-size:.92rem; color:var(--body); }}
.gloss b {{ color:var(--navy); font-family:var(--font-display) !important; }}
.ev {{ background:#0C1424; border:1px solid #22324F; border-radius:16px; padding:18px 18px 16px; color:#DCE6F7; }}
.ev-top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:10px; margin-bottom:12px; }}
.ev-k {{ font-family:var(--font-mono) !important; font-size:.62rem; letter-spacing:.18em; text-transform:uppercase; color:#8FA3C4; margin-bottom:4px; }}
.ev-id {{ font-family:var(--font-display) !important; font-size:1.55rem; font-weight:800; color:#FFFFFF; line-height:1.1; }}
.ev-sub {{ font-size:.85rem; color:#9FB3D1; margin-top:4px; }}
.ev-badge {{ font-family:var(--font-mono) !important; font-size:.68rem; font-weight:700; letter-spacing:.14em; border-radius:6px; padding:6px 11px; white-space:nowrap; box-shadow:0 0 0 1px rgba(255,255,255,.15); }}
.ev-box {{ background:#111C31; border:1px solid #22324F; border-radius:12px; padding:14px; margin-bottom:10px; }}
.ev-row {{ display:flex; align-items:center; gap:16px; }}
.ev-conf {{ flex:1; }}
.ev-big {{ font-family:var(--font-display) !important; font-size:2.1rem; font-weight:800; color:#8FB6F0; line-height:1; }}
.ev-bar {{ height:6px; background:rgba(255,255,255,.1); border-radius:3px; margin:8px 0 6px; }}
.ev-bar span {{ display:block; height:100%; background:#2F80ED; border-radius:3px; }}
.ev-small {{ font-size:.78rem; color:#9FB3D1; }}
.ev-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:10px; }}
.ev-t {{ background:#111C31; border:1px solid #22324F; border-radius:12px; padding:12px; }}
.ev-v {{ font-family:var(--font-display) !important; font-weight:700; font-size:1.05rem; color:#FFFFFF; }}
.ev-ul {{ margin:6px 0 0; padding-left:18px; font-size:.9rem; color:#DCE6F7; }}
.ev-ul li {{ margin-bottom:6px; }} .ev-ul li::marker {{ color:#F0955A; }}
.ev-f {{ margin:9px 0; }} .ev-fh {{ display:flex; justify-content:space-between; font-family:var(--font-mono) !important; font-size:.72rem; color:#C9D6EE; letter-spacing:.04em; }}
.ev-fh b {{ color:#FFFFFF; }} .ev-fb {{ height:4px; background:rgba(255,255,255,.08); border-radius:2px; margin-top:5px; }} .ev-fb span {{ display:block; height:100%; border-radius:2px; }}
.ev-btns {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.ev-btns a {{ text-align:center; font-family:var(--font-display) !important; font-weight:700; font-size:.75rem; letter-spacing:.1em; text-transform:uppercase; text-decoration:none !important;
  color:#8FB6F0 !important; border:1px solid #2F80ED; border-radius:8px; padding:10px 6px; }}
.ev-btns a.pri {{ background:#2F80ED; color:#FFFFFF !important; }}
.legend-bar {{ display:flex; flex-wrap:wrap; gap:14px; align-items:center; background:#0C1424; border:1px solid #22324F; border-radius:10px; padding:10px 14px; margin-top:-6px;
  font-size:.85rem; color:#DCE6F7; }}
.legend-bar .lk {{ font-family:var(--font-mono) !important; font-size:.66rem; letter-spacing:.16em; color:#8FA3C4; }}
.legend-bar i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }}
.hp-head {{ color:var(--navy); font-family:var(--font-display) !important; font-size:1.1rem; font-weight:700; padding:4px 4px 10px; }}
.hp-head .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; }}
.hp-meta {{ float:right; font-family:var(--font-mono) !important; font-size:.72rem; color:var(--muted); font-weight:400; margin-top:5px; }}
.hp-stats {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:10px; background:#F4F6FB; border:1px solid var(--line); border-radius:10px;
  padding:10px 16px; font-family:var(--font-mono) !important; font-size:.72rem; letter-spacing:.06em; color:var(--body); }}
.hp-stats b {{ color:var(--navy); }}
.hp-card {{ background:#FFFFFF; }}
[data-testid="stVerticalBlockBorderWrapper"]:has(.hp-head), div[data-testid="stVerticalBlock"]:has(> div .hp-head) {{ background:#FFFFFF !important; border-radius:14px; }}
.rule-t {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; font-size:.9rem; }}
.rule-t th {{ text-align:left; font-family:var(--font-mono) !important; font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); background:#F4F6FB; padding:10px 12px; border-bottom:1px solid var(--line); }}
.rule-t td {{ padding:10px 12px; border-bottom:1px solid var(--line); color:var(--body); }}
.rule-t tr:last-child td {{ border-bottom:none; }}
.stage-list {{ display:flex; flex-direction:column; gap:10px; }}
.stage-item {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px; display:grid; grid-template-columns:1fr auto; gap:4px 12px; align-items:center; }}
.stage-item .nm {{ font-family:var(--font-display) !important; font-weight:700; color:var(--navy); font-size:.95rem; }}
.stage-item .sub {{ grid-column:1 / 2; color:var(--muted); font-size:.85rem; }}
.stage-item .badge {{ grid-row:1 / 3; grid-column:2; font-family:var(--font-mono) !important; font-size:.66rem; font-weight:600; letter-spacing:.1em; padding:4px 10px; border-radius:999px; }}
.badge-done {{ background:#E4F3E8; color:#2E8B45; }} .badge-run {{ background:#E3EDFC; color:#2F80ED; }} .badge-wait {{ background:#F1F3F6; color:#8A93A6; }}
[data-testid="stMetric"] {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px 14px; }}
[data-testid="stMetricLabel"] p {{ font-family:var(--font-mono) !important; font-size:.62rem !important; letter-spacing:.14em; text-transform:uppercase; color:var(--muted) !important; }}
[data-testid="stMetricValue"] {{ font-family:var(--font-display) !important; color:var(--navy) !important; font-weight:700 !important; }}
[data-testid="stExpander"] {{ background:var(--card) !important; border:1px solid var(--line) !important; border-radius:10px !important; }}
[data-testid="stExpander"] summary p {{ font-family:var(--font-display) !important; font-size:.78rem !important; letter-spacing:.12em !important; text-transform:uppercase; color:var(--navy) !important; font-weight:700; }}
[data-testid="stButton"] button {{ font-family:var(--font-display) !important; font-weight:700 !important; letter-spacing:.1em !important; text-transform:uppercase; font-size:.72rem !important;
  color:var(--navy) !important; background:#FFFFFF !important; border:1px solid var(--line) !important; border-radius:6px !important; }}
[data-testid="stButton"] button:hover {{ border-color:var(--blue) !important; color:var(--blue) !important; }}
.site-footer {{ text-align:center; padding:2.6rem 0 1rem 0; margin-top:2.6rem; border-top:1px solid var(--line); }}
.site-footer p {{ font-family:var(--font-mono) !important; font-size:.62rem; letter-spacing:.26em; text-transform:uppercase; color:var(--muted); margin:.2rem 0; }}
</style>
"""


# ---------------------------------------------------------------------------
#  HELPERS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def b64(path: str) -> str:
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def asset(name: str) -> str:
    p = ASSETS / name
    if not p.exists():
        return ""
    if name.endswith(".mp4"):
        return base64.b64encode(p.read_bytes()).decode()
    return b64(str(p))


def html(markup: str) -> None:
    st.markdown("\n".join(ln for ln in markup.splitlines() if ln.strip()), unsafe_allow_html=True)


def pill(t: int) -> str:
    return f'<span class="ax-pill" style="background:{TIER_COLOURS[t]}">{E.TIERS[t][0].title()}</span>'


def section_header(dark: str, blue: str, subtitle: str = "") -> None:
    sub = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    html(f'<div class="section-header">{dark} <span>{blue}</span></div>{sub}')


def style_fig(fig, height=460):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Source Sans 3, sans-serif", color=NAVY, size=13),
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def render_hero(video: str, logo: str) -> None:
    hero_video_file = ASSETS / "hero_monsoon_earth.mp4"
    if hero_video_file.exists():
        vid = '<video class="hero-video" autoplay muted loop playsinline preload="auto"><source src="app/static/hero_monsoon_earth.mp4" type="video/mp4"></video>'
        slot = ""
    elif video and not video.endswith(".mp4"):
        vid = f'<video class="hero-video" autoplay muted loop playsinline><source src="data:video/mp4;base64,{video}" type="video/mp4"></video>'
        slot = ""
    else:
        vid = ""
        slot = '<div class="media-slot">hero video · assets/hero_monsoon_earth.mp4</div>'
    brand = f'<img src="data:image/png;base64,{logo}" class="hero-logo" alt="AIravat">' if logo else '<h1 style="color:#fff;font-family:Montserrat;font-size:5rem">AIravat</h1>'
    html(f"""
    <div class="hero-container">
      {vid}
      <div class="hero-overlay"></div>
      {slot}
      <div class="hero-content">
        <div class="hero-badge">SIH 2026 · Problem Statement 26078</div>
        {brand}
        <p class="hero-subtitle">Days ahead · kilometres close</p>
        <a href="{link('overview')}" target="_self" class="hero-cta">Explore</a>
      </div>
    </div>""")


def render_intro(dark: str, blue: str, line1: str, line2: str) -> None:
    css, spans, delay = "", "", 0.4
    for i, text in enumerate([line1, line2], start=1):
        w = len(text)
        dur = round(w * 0.06, 2)
        css += (f"@keyframes intro-type-{i} {{ from {{ width:0 }} to {{ width:{w}ch }} }}"
                f".typed-line-{i} {{ animation: intro-type-{i} {dur}s steps({w}, end) {delay}s forwards, intro-caret .8s step-end {delay}s {max(1, int(dur / .8))} forwards; }}")
        spans += f'<span class="typed-line typed-line-{i}">{text}</span>'
        delay += dur + 0.3
    html(f'<style>{css}</style><div class="text-intro-section"><h2 class="text-intro-title">{dark} <span>{blue}</span></h2><div class="text-intro-typed">{spans}</div></div>')


def render_media(fname: str, title: str, subtitle: str, align: str = "right", dark_text: bool = False, header: bool = False) -> None:
    p = ASSETS / fname
    is_video = fname.endswith(".mp4")
    classes = "video-section" + (" align-left" if align == "left" else "") + (" dark-text" if dark_text and p.exists() else "") + (" page-header" if header else "")
    if is_video and p.exists():
        media = f'<video autoplay muted loop playsinline preload="auto"><source src="app/static/{fname}" type="video/mp4"></video>'
    elif p.exists():
        data = asset(fname)
        mime = "image/png" if fname.endswith(".png") else "image/jpeg"
        media = f'<img src="data:{mime};base64,{data}">'
    else:
        media = f'<div class="media-slot">{"video" if is_video else "image"} · assets/{fname}</div>'
    html(f"""
    <div class="{classes}">
      {media}
      <div class="overlay"></div>
      <div class="section-text"><h2>{title}</h2><p>{subtitle}</p></div>
    </div>""")


def nav_scroll_script(page: str) -> None:
    script = ("""
    <script>
    (function () {
      const doc = window.parent.document; const PAGE = "__PAGE__";
      function scroller() { return doc.querySelector('section.main') || doc.querySelector('[data-testid="stMain"]') || doc.querySelector('[data-testid="stAppViewContainer"]') || doc.scrollingElement; }
      if (doc.__airavatPage !== PAGE) { doc.__airavatPage = PAGE; setTimeout(function () { const el = scroller(); if (el) el.scrollTo({top:0}); window.parent.scrollTo({top:0}); const sh = doc.getElementById('airavat-nav-shell'); if (sh) sh.classList.remove('nav-hidden'); }, 40); }
      if (doc.__airavatNavScroll) return; doc.__airavatNavScroll = true;
      let lastY = 0, ticking = false;
      function apply() { const sh = doc.getElementById('airavat-nav-shell'); const el = scroller(); if (!sh || !el) { ticking = false; return; }
        const y = el.scrollTop || window.parent.scrollY || 0; sh.classList.toggle('nav-floating', y > 90);
        if (Math.abs(y - lastY) > 6) { sh.classList.toggle('nav-hidden', y > lastY && y > 90); lastY = y; } ticking = false; }
      function onScroll() { if (!ticking) { ticking = true; window.requestAnimationFrame(apply); } }
      function attach() { const el = scroller(); if (el) el.addEventListener('scroll', onScroll, {passive:true}); window.parent.addEventListener('scroll', onScroll, {passive:true}); }
      attach(); new MutationObserver(attach).observe(doc.body, {childList:true, subtree:false});
    })();
    </script>""").replace("__PAGE__", page)
    if hasattr(st, "iframe"):
        st.iframe(script, height=1)
    else:
        components.html(script, height=0)


# ---------------------------------------------------------------------------
#  SETTINGS (kept in the URL so they survive page changes)
# ---------------------------------------------------------------------------
SCEN_NAMES = list(E.SCENARIOS.keys())
LAYERS = ["Extreme Forecast Index", "Shift of Tails", "Exceedance probability", "Ensemble mean"]
LAYER_PLAIN = {"Extreme Forecast Index": "How unusual (EFI)", "Shift of Tails": "Record-breaking signal (SOT)",
               "Exceedance probability": "Chance of crossing the danger level", "Ensemble mean": "Average of all forecast runs"}
qp = st.query_params


def _qp_int(key, default, lo, hi):
    try:
        return min(max(int(qp.get(key, default)), lo), hi)
    except ValueError:
        return default


if "set_scen" not in st.session_state:
    st.session_state.set_scen = SCEN_NAMES[_qp_int("sc", 0, 0, len(SCEN_NAMES) - 1)]
    lead_q = _qp_int("lead", 120, 72, 240)
    st.session_state.set_lead = lead_q if lead_q in E.LEADS else 120
    st.session_state.set_layer = LAYERS[_qp_int("layer", 0, 0, len(LAYERS) - 1)]
    st.session_state.set_thr = min(max(_qp_int("thr", 65, 50, 95), 50), 95) / 100


def link(page: str) -> str:
    return (f"?page={page}&sc={SCEN_NAMES.index(st.session_state.set_scen)}&lead={st.session_state.set_lead}"
            f"&layer={LAYERS.index(st.session_state.set_layer)}&thr={int(round(st.session_state.set_thr * 100))}")


scenario, lead = st.session_state.set_scen, st.session_state.set_lead
layer, efi_thr = st.session_state.set_layer, st.session_state.set_thr
cfg = E.SCENARIOS[scenario]
unit, crit = cfg["unit"], cfg["thresholds"][1]
cycle_dt = datetime.strptime(cfg["cycle"].replace(" UTC", ""), "%Y-%m-%d %H")


@st.cache_data(show_spinner=False)
def get_scenario(name, lead_h):
    return E.build_scenario(name, lead_h)


@st.cache_data(show_spinner=False)
def get_clusters(name, lead_h, thr):
    return E.cluster_anomalies(get_scenario(name, lead_h), efi_thresh=thr)


def cluster_tier(c):
    return E.severity(c["max_efi"], c["max_sot"], c["max_prob"])[1]


sc = get_scenario(scenario, lead)
clusters = get_clusters(scenario, lead, efi_thr)
top_tier = max([cluster_tier(c) for c in clusters], default=0)

# ---------------------------------------------------------------------------
#  PLAIN-LANGUAGE HELPERS
# ---------------------------------------------------------------------------
IST = timedelta(hours=5, minutes=30)
HAZARD_WORD = {
    "cyclone": "a cyclone",
    "cyclone_arabian": "a severe cyclonic storm",
    "cloudburst": "a mountain cloudburst",
    "assam_flood": "an extreme monsoon deluge",
    "chennai_flood": "a coastal torrential deluge",
    "heat": "a severe heatwave",
    "monsoon": "very heavy rain",
}.get(cfg.get("key", ""), "an extreme weather anomaly")
TIER_MEANING = {0: "no action needed", 1: "be aware", 2: "be prepared", 3: "take action", 4: "act now"}
VAR_WORD = "rain" if cfg["variable"] == "precip" else "temperature"
CRIT_TEXT = f"{crit:g} mm or more of rain in a day" if cfg["variable"] == "precip" else f"{crit:g} °C or more"
PEAK_WORD = "heaviest rain" if cfg["variable"] == "precip" else "highest temperature"
N_RUNS = E.N_MEMBERS


def day_label(lead_h: int) -> str:
    d = cycle_dt + timedelta(hours=lead_h)
    return f"{d:%a %d %b}"


def day_short(lead_h: int) -> str:
    return f"Day {lead_h // 24} · {(cycle_dt + timedelta(hours=lead_h)):%d %b}"


def near_place(lat: float, lon: float) -> str:
    best = min(E.PLACES.items(), key=lambda kv: E.haversine_km(lat, lon, *kv[1]))
    dist = float(E.haversine_km(lat, lon, *best[1]))
    if dist < 120:
        return f"near {best[0]}"
    if dist < 450:
        return f"about {dist:.0f} km from {best[0]}"
    return f"at {lat:.1f}°N, {lon:.1f}°E"


def efi_words(efi: float) -> str:
    if efi < 0.5:
        return "normal for this time of year"
    if efi < 0.7:
        return "unusual for this time of year"
    if efi < 0.85:
        return "very unusual for this time of year"
    return "extremely unusual: rarely or never seen here in 30 years"


def plain_box(text: str, label: str = "In plain words") -> None:
    html(f'<div class="plain"><div class="lbl">{label}</div><div class="txt">{text}</div></div>')


def cycle_story() -> str:
    issued = f"{cycle_dt:%d %b}, {(cycle_dt + IST):%H:%M} IST"
    if not clusters:
        return (f"The forecast run from <b>{issued}</b> shows <b>nothing dangerous</b> for <b>{day_label(lead)}</b> "
                f"(day {lead // 24} of the forecast). Conditions look normal for this time of year.")
    c = clusters[0]
    t = cluster_tier(c)
    where = near_place(*c["centroid"])
    return (f"The forecast run from <b>{issued}</b> shows <b>{HAZARD_WORD} {where}</b> on <b>{day_label(lead)}</b> "
            f"(day {lead // 24} of the forecast). <b>{c['max_prob'] * 100:.0f}% of the {N_RUNS} forecast runs</b> show {CRIT_TEXT} somewhere in the zone. "
            f"Conditions like this are {efi_words(c['max_efi'])}. Alert level: <b style=\"color:{TIER_COLOURS[t]}\">{E.TIERS[t][0].title()}</b>, meaning <b>{TIER_MEANING[t]}</b>.")


GLOSSARY = [
    ("Forecast run", "NCMRWF runs its weather model 23 times, each from slightly different starting conditions. If most runs agree, the forecast is confident."),
    ("How unusual (EFI)", "0 = normal for this place and season, 1 = beyond anything in the last 30 years. Below 0 means unusually low."),
    ("Record-breaking signal (SOT)", "Above 0: the worst forecast runs go past the 1-in-100-day level. Above 2: likely record-breaking."),
    ("Chance of crossing the danger level", f"Share of the {N_RUNS} runs that show {CRIT_TEXT}."),
    ("Danger zone", "A 640 × 640 km area where the forecast is extreme. Only these areas are zoomed to 5 km."),
    ("Alert levels", "Advisory = be aware · Watch = be prepared · Warning = take action · Emergency = act now (IMD colours)."),
]


# ---------------------------------------------------------------------------
#  CHARTS
# ---------------------------------------------------------------------------
# Satellite basemap (Esri World Imagery, no API key) + our own labels. No political boundaries are drawn.
MapTrace = getattr(go, "Scattermap", None) or getattr(go, "Scattermapbox")
MAP_KEY = "map" if hasattr(go, "Scattermap") else "mapbox"
SAT_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
CITY_LABELS = {"Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21), "Kolkata": (22.57, 88.36), "Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59),
               "Hyderabad": (17.39, 78.49), "Ahmedabad": (23.02, 72.57), "Pune": (18.52, 73.86), "Jaipur": (26.91, 75.79), "Lucknow": (26.85, 80.95),
               "Bhopal": (23.26, 77.41), "Patna": (25.59, 85.14), "Bhubaneswar": (20.30, 85.82), "Puri": (19.81, 85.83), "Visakhapatnam": (17.69, 83.22),
               "Raipur": (21.25, 81.63), "Guwahati": (26.14, 91.74), "Thiruvananthapuram": (8.52, 76.94), "Kochi": (9.93, 76.27), "Goa": (15.49, 73.83),
               "Ratnagiri": (16.99, 73.31), "Bikaner": (28.02, 73.31), "Jaisalmer": (26.92, 70.91), "Jodhpur": (26.24, 73.02), "Nagpur": (21.15, 79.09),
               "Dehradun": (30.32, 78.03), "Shimla": (31.10, 77.17), "Srinagar": (34.08, 74.80), "Gopalpur": (19.26, 84.91), "Mangaluru": (12.91, 74.86)}


@st.cache_data(show_spinner=False)
def field_overlay(name: str, lead_h: int, layer_name: str) -> tuple[str, float, float, str]:
    """Smooth, semi-transparent PNG of the chosen field, draped over the satellite map."""
    from io import BytesIO
    from PIL import Image
    from scipy.ndimage import zoom as _zoom
    from matplotlib.colors import LinearSegmentedColormap
    s_ = get_scenario(name, lead_h)
    cfg_ = E.SCENARIOS[name]
    if layer_name == "Extreme Forecast Index":
        z, lo, hi, stops, ttl = s_["efi"], 0.35, 1.0, ["#E8C547", "#EE8A2F", "#D6453D", "#8E1B4A"], "How unusual"
    elif layer_name == "Shift of Tails":
        z, lo, hi, stops, ttl = s_["sot"], 0.0, 4.0, ["#E8C547", "#EE8A2F", "#D6453D", "#8E1B4A"], "Record signal"
    elif layer_name == "Exceedance probability":
        z, lo, hi, stops, ttl = s_["p_exc"] * 100, 5, 100, ["#7FB3EE", "#2F80ED", "#8E4FD0", "#D6453D"], "Chance %"
    else:
        mean = s_["ens"].mean(axis=0)
        if cfg_["variable"] == "precip":
            z, lo, hi, stops, ttl = mean, 20, 220, ["#7FD1F0", "#2F80ED", "#6B3FA0", "#D6453D"], "mm/day"
        else:
            z, lo, hi, stops, ttl = mean, 36, 48, ["#E8C547", "#EE8A2F", "#D6453D", "#7A1330"], "°C"
    fine = _zoom(np.asarray(z, float), 8, order=3)
    a = np.clip((fine - lo) / (hi - lo), 0, 1)
    cm = LinearSegmentedColormap.from_list("ov", stops)
    rgba = cm(a)
    rgba[..., 3] = np.clip(a * 1.6, 0, 0.85)
    img = Image.fromarray((rgba[::-1] * 255).astype(np.uint8), "RGBA")
    buf = BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), lo, hi, ttl


def india_map(height=720, focus=None):
    img, lo, hi, ttl = field_overlay(scenario, lead, layer)
    fig = go.Figure()
    # colour key (invisible trace that only draws the colour bar)
    fig.add_trace(MapTrace(lat=[0], lon=[0], mode="markers", marker=dict(size=0.1, color=[lo], cmin=lo, cmax=hi, showscale=True,
                                                                        colorscale=[[0, "#E8C547"], [0.4, "#EE8A2F"], [0.75, "#D6453D"], [1, "#8E1B4A"]],
                                                                        colorbar=dict(title=dict(text=ttl, font=dict(color="#fff")), tickfont=dict(color="#fff"),
                                                                                      thickness=10, len=0.45, x=0.01, xanchor="left", y=0.25,
                                                                                      bgcolor="rgba(12,20,36,.55)")),
                           hoverinfo="skip", showlegend=False))
    if "cyclone" in cfg["key"]:
        for h in range(72, 241, 24):
            clat, clon = E.circle(*E.cyclone_track(h, cfg["key"]), E.track_cone(h))
            fig.add_trace(MapTrace(lat=list(clat), lon=list(clon), mode="lines", fill="toself", fillcolor="rgba(120,180,255,0.10)",
                                   line=dict(color="rgba(160,200,255,0.55)", width=1), hoverinfo="skip", showlegend=False))
        tr = [E.cyclone_track(h, cfg["key"]) for h in range(72, 241, 6)]
        fig.add_trace(MapTrace(lat=[p_[0] for p_ in tr], lon=[p_[1] for p_ in tr], mode="lines", line=dict(color="#FFFFFF", width=2.5),
                               name="Most likely storm path", hoverinfo="skip"))
        days = [E.cyclone_track(h, cfg["key"]) for h in E.LEADS]
        fig.add_trace(MapTrace(lat=[p_[0] for p_ in days], lon=[p_[1] for p_ in days], mode="markers+text",
                               marker=dict(size=7, color="#FFFFFF"), text=[day_label(h)[4:] for h in E.LEADS], textposition="middle left",
                               textfont=dict(color="#FFFFFF", size=10), hoverinfo="skip", showlegend=False))
    for c in clusters:
        la0, la1, lo0, lo1 = c["box"]
        col = TIER_COLOURS[cluster_tier(c)]
        fig.add_trace(MapTrace(lat=[la0, la0, la1, la1, la0], lon=[lo0, lo1, lo1, lo0, lo0], mode="lines",
                               line=dict(color=col, width=2.5), hoverinfo="skip", showlegend=False))
        for size, op in ((58, 0.18), (38, 0.35), (18, 0.95)):
            fig.add_trace(MapTrace(lat=[c["centroid"][0]], lon=[c["centroid"][1]], mode="markers",
                                   marker=dict(size=size, color=col, opacity=op), showlegend=False,
                                   hovertemplate=f"<b>{c['id']}</b><br>{E.TIERS[cluster_tier(c)][0].title()}<br>{near_place(*c['centroid'])}<extra></extra>"))
    names = list(CITY_LABELS)
    fig.add_trace(MapTrace(lat=[CITY_LABELS[n][0] for n in names], lon=[CITY_LABELS[n][1] for n in names], mode="markers+text",
                           marker=dict(size=5, color="#FFFFFF"), text=names, textposition="top right",
                           textfont=dict(color="#FFFFFF", size=11), hovertemplate="%{text}<extra></extra>", showlegend=False))
    center = dict(lat=focus[0], lon=focus[1]) if focus else dict(lat=21.5, lon=81.0)
    layers = [dict(sourcetype="raster", source=[SAT_TILES], below="traces", sourceattribution="Imagery © Esri, Maxar, Earthstar Geographics"),
              dict(sourcetype="image", source=img, coordinates=[[66, 37], [98, 37], [98, 5], [66, 5]], opacity=0.85, below="traces")]
    fig.update_layout(**{MAP_KEY: dict(style="white-bg", center=center, zoom=5.2 if focus else 3.9, layers=layers)},
                      height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0C1424",
                      legend=dict(orientation="h", x=0.01, y=0.99, bgcolor="rgba(12,20,36,.6)", font=dict(color="#fff")))
    return fig


def badge_style(t: int) -> str:
    fg = "#1E2A45" if t in (0, 1) else "#FFFFFF"
    bg = {4: "#C2185B", 3: "#D6453D", 2: "#EE8A2F", 1: "#E8C547", 0: "#2E9E5B"}[t]
    return f"background:{bg};color:{fg}"


def gauge_svg(score: float, colour: str) -> str:
    r, c = 52, 2 * math.pi * 52
    return (f'<svg viewBox="0 0 130 130" width="130" height="130"><circle cx="65" cy="65" r="{r}" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="12"/>'
            f'<circle cx="65" cy="65" r="{r}" fill="none" stroke="{colour}" stroke-width="12" stroke-linecap="round" '
            f'stroke-dasharray="{c * score:.1f} {c:.1f}" transform="rotate(-90 65 65)"/>'
            f'<text x="65" y="66" text-anchor="middle" font-family="Montserrat,sans-serif" font-size="30" font-weight="800" fill="#fff">{score * 100:.0f}</text>'
            f'<text x="65" y="84" text-anchor="middle" font-family="JetBrains Mono,monospace" font-size="10" fill="#9FB3D1">/ 100</text></svg>')


def event_panel(c):
    t = cluster_tier(c)
    col = TIER_COLOURS[t]
    i_, j_ = int(np.abs(E.LAT - c["centroid"][0]).argmin()), int(np.abs(E.LON - c["centroid"][1]).argmin())
    d = E.point_diagnostics(sc, float(E.LAT[i_]), float(E.LON[j_]))
    score = E.severity(c["max_efi"], c["max_sot"], c["max_prob"])[0]
    agree = c["max_prob"]
    lines = [f"{c['max_prob'] * 100:.0f}% of the {N_RUNS} runs show {CRIT_TEXT} in this zone",
             f"Conditions are {efi_words(c['max_efi']).split(':')[0]} (how unusual: {c['max_efi']:.2f})"]
    if c["max_sot"] > 1:
        lines.append(f"Worst runs go past the 1-in-100-day level (record signal {c['max_sot']:.1f})")
    lines.append(f"Most extreme run: {c['peak_value']:.0f} {unit}, {near_place(*c['centroid'])}")
    feats = [("How unusual (EFI)", f"{c['max_efi']:.2f}", min(1, max(c["max_efi"], 0))),
             ("Record signal (SOT)", f"{c['max_sot']:.1f}", min(1, max(c["max_sot"], 0) / 4)),
             ("Chance of danger level", f"{c['max_prob'] * 100:.0f}%", c["max_prob"]),
             (f"Bad case (1 in 10 runs)", f"{d['q90']:.0f} {unit}", min(1, d["q90"] / max(c["peak_value"], 1))),
             ("1-in-100-day level here", f"{d['clim_q99']:.0f} {unit}", min(1, d["clim_q99"] / max(c["peak_value"], 1)))]
    bars = "".join(f'<div class="ev-f"><div class="ev-fh"><span>{k + 1}. {n}</span><b>{v}</b></div><div class="ev-fb"><span style="width:{w * 100:.0f}%;background:{col}"></span></div></div>'
                   for k, (n, v, w) in enumerate(feats))
    html(f"""
    <div class="ev">
      <div class="ev-top"><div><div class="ev-k">Event intelligence</div><div class="ev-id">{c['id']}</div>
        <div class="ev-sub">{day_label(lead)} · NEPS-G {cycle_dt:%d %b %H} UTC run · {N_RUNS} forecast runs</div></div>
        <span class="ev-badge" style="{badge_style(t)}">{E.TIERS[t][0]}</span></div>
      <div class="ev-box"><div class="ev-k" style="color:{col}">{HAZARD_WORD.split(' ', 1)[1].upper()} · {TIER_MEANING[t].upper()}</div>
        <div class="ev-row">{gauge_svg(score, col)}
          <div class="ev-conf"><div class="ev-k">Runs that agree</div><div class="ev-big">{agree * 100:.0f}%</div>
          <div class="ev-bar"><span style="width:{agree * 100:.0f}%"></span></div><div class="ev-small">{int(round(agree * N_RUNS))} of {N_RUNS} runs cross the danger level</div></div></div>
        <div class="ev-small" style="margin-top:8px">Location: <b style="color:#8FB6F0">{near_place(*c['centroid'])[0].upper() + near_place(*c['centroid'])[1:]}</b></div></div>
      <div class="ev-grid">
        <div class="ev-t"><div class="ev-k">Hazard</div><div class="ev-v" style="color:{col}">{cfg['hazard'].split(':')[0]}</div></div>
        <div class="ev-t"><div class="ev-k">Heaviest in any run</div><div class="ev-v">{c['peak_value']:.0f} {unit}</div></div>
        <div class="ev-t"><div class="ev-k">Most likely</div><div class="ev-v">{d['q50']:.0f} {unit}</div></div>
        <div class="ev-t"><div class="ev-k">What to do</div><div class="ev-v" style="font-size:.85rem">{E.TIERS[t][2]}</div></div>
      </div>
      <div class="ev-box"><div class="ev-k">Why this was flagged</div><ul class="ev-ul">{''.join(f'<li>{ln}</li>' for ln in lines)}</ul></div>
      <div class="ev-box"><div class="ev-k">Top 5 measurements</div>{bars}</div>
      <div class="ev-btns"><a href="{link('copilot')}" target="_self">Ask Copilot</a><a class="pri" href="{link('reasons')}" target="_self">Full alert reasons →</a></div>
    </div>""")


def history_panel(c):
    i_, j_ = int(np.abs(E.LAT - c["centroid"][0]).argmin()), int(np.abs(E.LON - c["centroid"][1]).argmin())
    lat_, lon_ = float(E.LAT[i_]), float(E.LON[j_])
    rows = []
    for h in E.LEADS:
        m_ = get_scenario(scenario, h)["ens"][:, i_, j_]
        rows.append((h, np.percentile(m_, 10), np.percentile(m_, 50), np.percentile(m_, 90), m_.max()))
    d0 = E.point_diagnostics(sc, lat_, lon_)
    clim99 = d0["clim_q99"]
    xs = [day_short(r[0]) for r in rows]
    t = cluster_tier(c)
    col = TIER_COLOURS[t]
    peak_row = max(rows, key=lambda r: r[3])
    where = near_place(lat_, lon_)
    html(f"""
    <div class="hp-head"><span class="dot" style="background:{col}"></span>Forecast vs 30-year climate · <span style="color:{BLUE}">{c['id']}</span>
      <span class="ev-badge" style="{badge_style(t)};margin-left:10px">{E.TIERS[t][0]}</span>
      <span class="hp-meta">{where[0].upper() + where[1:]} · NEPS-G {cycle_dt:%d %b %Y %H} UTC</span></div>
    <div class="hp-stats"><span>BAD-CASE PEAK: <b style="color:#E0661F">{peak_row[3]:.0f} {unit}</b> ({day_label(peak_row[0])})</span>
      <span>1-IN-100-DAY LEVEL: <b>{clim99:.0f} {unit}</b></span><span>EXCESS: <b style="color:#D6453D">+{peak_row[3] - clim99:.0f} {unit}</b></span>
      <span>{N_RUNS} RUNS · DAY 3–10</span></div>""")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r[3] for r in rows], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=[r[1] for r in rows], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(214,69,61,0.14)",
                             name="Range of runs (10–90%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r[2] for r in rows], mode="lines+markers", line=dict(color="#D6453D", width=3), marker=dict(size=8, color="#D6453D"),
                             name="Most likely", hovertemplate="%{x}: %{y:.0f} " + unit + "<extra></extra>"))
    fig.add_hline(y=clim99, line_color=BLUE, line_dash="dash", annotation_text="1-in-100-day level for this place", annotation_font_color=BLUE,
                  annotation_position="top left")
    fig.add_hline(y=crit, line_color="#E0661F", line_dash="dot", annotation_text="Danger level", annotation_font_color="#E0661F", annotation_position="bottom left")
    fig.add_vline(x=day_short(lead), line_color=MUTED, line_dash="dot")
    fig.update_layout(height=340, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color=NAVY, family="JetBrains Mono, monospace", size=11),
                      margin=dict(l=50, r=20, t=36, b=40), legend=dict(orientation="h", y=1.14, x=1, xanchor="right", font=dict(color=BODY)))
    fig.update_xaxes(gridcolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, title=unit)
    st.plotly_chart(fig, width="stretch", key=f"hist_{c['id']}", config={"displayModeBar": False})


def alert_cards():
    if not clusters:
        html('<div class="ax-card"><p>Nothing dangerous is forecast for this day. Open <b>Mission Control</b> at the bottom to look at another day.</p></div>')
    for c in clusters:
        t = cluster_tier(c)
        html(f'<div class="ax-card"><h4>{HAZARD_WORD[0].upper() + HAZARD_WORD[1:]} {near_place(*c["centroid"])} &nbsp;{pill(t)}</h4>'
             f'<p><b>{day_label(lead)}</b> · {c["max_prob"] * 100:.0f}% of forecast runs show {CRIT_TEXT} somewhere in this zone.<br>'
             f'Conditions like this are {efi_words(c["max_efi"])}.</p>'
             f'<p style="margin-top:8px;color:{NAVY}"><b>What to do ({TIER_MEANING[t]}):</b> {E.TIERS[t][2]}</p>'
             f'<p style="margin-top:6px;font-size:.8rem;color:{MUTED}">{c["id"]} · EFI {c["max_efi"]:.2f} · SOT {c["max_sot"]:.2f}</p></div>')


# ---------------------------------------------------------------------------
#  NAVIGATION
# ---------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

PAGES = [("overview", "Tactical Overview"), ("live", "Live Feed"), ("tracking", "Anomaly Tracking"), ("downscaling", "5 km Downscaling"),
         ("pinpoint", "Pinpoint Alerts"), ("reasons", "Alert Reasons"), ("copilot", "AIravat Copilot"),
         ("technology", "Our Technology")]
page = str(qp.get("page", "home")).lower().strip()
if page not in [p for p, _ in PAGES] + ["home"]:
    page = "home"

items = ""
for pid, label in PAGES:
    cls = "nav-item" + (" active" if page == pid else "") + (" nav-item-tech" if pid == "technology" else "")
    items += f'<a href="{link(pid)}" class="{cls}" target="_self">{label}</a>'
logo_b64 = asset("logo.png")
logo_img = f'<img src="data:image/png;base64,{logo_b64}" class="brand-logo-img" alt="AIravat">' if logo_b64 else '<b style="font-family:Montserrat;font-size:1.4rem;color:#1E2A45">AIravat</b>'
html(f"""
<div class="nav-shell" id="airavat-nav-shell">
  <div class="top-status-row"><div class="top-brand-status"><span class="status-pulse-dot"></span>
    Forecast issued {cycle_dt:%d %b %Y} · showing {day_label(lead)} (day {lead // 24} ahead) · highest alert {E.TIERS[top_tier][0]}</div></div>
  <div class="top-brand-bar"><a href="{link('home')}" target="_self" class="top-brand-logo">{logo_img}</a>{items}</div>
</div>
<div class="nav-shell-spacer"></div>""")

# ---------------------------------------------------------------------------
#  PAGES
# ---------------------------------------------------------------------------
if page == "home":
    render_hero("hero_monsoon_earth.mp4", asset("logo_hero.png"))
    render_intro("How We See a Cloudburst", "Days Before It Hits",
                 "One 12 km forecast cannot show a 5 km storm cell,",
                 "so we find the extreme first, then sharpen it with physics-checked AI.")
    render_media("satellite_orbit.png", "ORBITAL WATCH",
                 "Twenty-three forecasts every twelve hours, each checked against thirty years of local weather.", align="right")
    render_media("ghats_monsoon.mp4", "TERRAIN MATTERS",
                 "The Western Ghats and the Himalaya decide where rain lands, so every 5 km forecast knows the ground beneath it.",
                 align="left", dark_text=True)
    render_media("rescue_response.png", "EARLY ACTION",
                 "Days of warning mean families reach the shelter before the first gust arrives.", align="right")

elif page == "overview":
    section_header("Tactical", "Overview", f"What the forecast says for {day_label(lead)}, day {lead // 24} of the forecast")
    plain_box(cycle_story())
    ordered = sorted(clusters, key=lambda c_: -cluster_tier(c_))
    left, right = st.columns([1.85, 1], gap="small")
    with right:
        if ordered:
            if len(ordered) > 1:
                sel = st.selectbox("Event", [c_["id"] for c_ in ordered], format_func=lambda z: f"{z} · {near_place(*next(c_ for c_ in ordered if c_['id'] == z)['centroid'])}")
                focus_c = next(c_ for c_ in ordered if c_["id"] == sel)
            else:
                focus_c = ordered[0]
            event_panel(focus_c)
        else:
            focus_c = None
            html('<div class="ev"><div class="ev-k">Event intelligence</div><div class="ev-id">All clear</div>'
                 f'<div class="ev-sub">Nothing dangerous is forecast for {day_label(lead)}. Open Mission Control below to look at another day.</div></div>')
    with left:
        zoom_in = st.toggle("Zoom to event", value=False, disabled=focus_c is None)
        st.plotly_chart(india_map(760, focus_c["centroid"] if (zoom_in and focus_c) else None), width="stretch", key="map_over",
                        config={"scrollZoom": True, "displayModeBar": False})
        html('<div class="legend-bar"><span class="lk">ALERT LEVEL:</span>' + "".join(
            f'<span><i style="background:{TIER_COLOURS[k]}"></i>{E.TIERS[k][0].title()}</span>' for k in (4, 3, 2, 1, 0)) +
            f'<span class="lk" style="margin-left:18px">MAP COLOUR:</span><span>{LAYER_PLAIN[layer]}</span></div>')
    if focus_c:
        st.write("")
        with st.container(border=True):
            history_panel(focus_c)
    with st.expander("What do these terms mean?"):
        html('<div class="gloss">' + "".join(f"<div><b>{a_}</b><br>{b_}</div>" for a_, b_ in GLOSSARY) + "</div>")

elif page == "live":
    import time as _time

    section_header("Live Forecast", "Feed", "What happens, step by step, when a new forecast arrives. Press Replay to watch it run.")
    rng = np.random.default_rng(cycle_dt.toordinal())
    t0 = cycle_dt + timedelta(hours=4, minutes=30)
    STAGE_COL = {"CYCLE": ("#2F80ED", "#E3EDFC"), "INGEST": ("#8FB6F0", "rgba(143,182,240,.14)"), "QC": ("#7CD992", "rgba(124,217,146,.14)"),
                 "ZARR": ("#B69CF0", "rgba(182,156,240,.16)"), "STAGE 1": ("#F2C230", "rgba(242,194,48,.16)"), "STAGE 2": ("#F0955A", "rgba(240,149,90,.16)"),
                 "PHYSICS": ("#7CD992", "rgba(124,217,146,.14)"), "ALERT": ("#FF8C82", "rgba(255,140,130,.16)"), "PUBLISH": ("#8FB6F0", "rgba(143,182,240,.14)")}
    n_box = len(clusters)
    box_word = "box" if n_box == 1 else "boxes"
    events = [(t0, "CYCLE", f"NEPS-G <b>{cycle_dt:%d %b %Y %H} UTC</b> cycle published · 23 members expected", "ok")]
    arrival = np.sort(np.cumsum(rng.uniform(1.4, 2.6, 23)))
    for m, sec in enumerate(arrival):
        label = "control" if m == 0 else f"member {m}"
        events.append((t0 + timedelta(seconds=float(sec)), "INGEST",
                       f"<b>nepsg_{cycle_dt:%Y%m%d}_{cycle_dt:%H}z_mem{m:02d}.grib2</b> · {label} · T+0 to T+240 h", "ok"))
    t = t0 + timedelta(seconds=float(arrival[-1]))
    step = lambda s_: t + timedelta(seconds=s_)
    events += [
        (step(2), "QC", "Checksums verified · physical bounds checked on all 23 files · <b>0 flagged</b>", "ok"),
        (step(9), "ZARR", "India slice 0–40°N, 60–100°E · <b>14 pressure levels</b> · staged to Zarr", "ok"),
        (step(27), "STAGE 1", "Spherical GNN pass on 40,962 mesh nodes · T+72 to T+240 h", "ok"),
        (step(31), "STAGE 1", "EFI and SOT computed against <b>930-sample</b> ERA5 climatology", "ok"),
    ]
    if n_box:
        top = clusters[0]
        events.append((step(33), "STAGE 1", f"<b>{n_box} anomaly {box_word}</b> found · strongest {top['id']} (EFI {top['max_efi']:.2f}, SOT {top['max_sot']:.2f})", "ok"))
        events.append((step(35), "STAGE 1", f"Track linked across T+72 → T+240 h for {n_box} {box_word}", "ok"))
        events.append((step(91), "STAGE 2", f"Diffusion downscaling · <b>30 realizations × {n_box} {box_word}</b> at 5 km", "ok"))
        events.append((step(97), "PHYSICS", "Mass, moisture and energy residuals <b>within tolerance</b>", "ok"))
        n_alerts = sum(1 for c in clusters if cluster_tier(c) >= 3)
        tier_name = E.TIERS[top_tier][0].title()
        if n_alerts:
            events.append((step(99), "ALERT", f"Highest tier <b>{tier_name}</b> · {n_alerts} CAP alert{'s' if n_alerts > 1 else ''} awaiting forecaster approval", "review"))
        else:
            events.append((step(99), "ALERT", f"Highest tier <b>{tier_name}</b> · no Warning or Emergency this cycle", "ok"))
    else:
        events.append((step(33), "STAGE 1", "<b>No anomaly boxes</b> above the EFI threshold · Stage 2 skipped", "ok"))
    events.append((events[-1][0] + timedelta(seconds=6), "PUBLISH", "Map tiles, pinpoint cache and SitRep refreshed", "ok"))

    tech = st.toggle("Show technical log", value=False, help="Switch between the plain summary and the full engineering log.")

    t_q = events[[e_[1] for e_ in events].index("QC")][0]
    t_s1 = [e_[0] for e_ in events if e_[1] == "STAGE 1"][-1]
    plain = [(t0, "1", "New forecast arrived", "NCMRWF published <b>23 forecast runs</b> covering the next 10 days.", "ok"),
             (t_q, "2", "All forecasts checked", "All 23 runs arrived <b>complete and error-free</b>.", "ok")]
    if n_box:
        top = clusters[0]
        plain += [(t_s1, "3", "Danger found", f"<b>{n_box} danger zone{'s' if n_box > 1 else ''} found</b>: {HAZARD_WORD} {near_place(*top['centroid'])}. "
                                               f"Conditions there are {efi_words(top['max_efi']).split(':')[0]}.", "ok"),
                  ([e_[0] for e_ in events if e_[1] == "STAGE 2"][-1], "4", "Zoomed to 5 km", f"Built <b>30 detailed 5 km versions</b> of each danger zone.", "ok"),
                  ([e_[0] for e_ in events if e_[1] == "PHYSICS"][-1], "5", "Physics check passed", "The detailed forecasts obey the laws for <b>air, water and energy</b>.", "ok")]
        a_ev = [e_ for e_ in events if e_[1] == "ALERT"][-1]
        tn = E.TIERS[top_tier][0].title()
        plain.append((a_ev[0], "6", "Alert ready", f"<b>{tn}</b> alert ({TIER_MEANING[top_tier]}) prepared"
                      + (" and waiting for an <b>IMD forecaster to approve</b>." if a_ev[3] == "review" else "."), a_ev[3]))
    else:
        plain.append((t_s1, "3", "Nothing dangerous", "No place crosses the danger level for this day. No alert needed.", "ok"))
    plain.append((events[-1][0], "7" if n_box else "4", "Published", "Maps, place look-ups and the situation report are updated.", "ok"))

    def feed_html(evts, running=False):
        rows = ""
        for k_, (tm, stg, msg, stt) in enumerate(evts):
            fg, bg = STAGE_COL[stg]
            live = running and k_ == len(evts) - 1
            badge = ('<span class="st st-run">RUNNING</span>' if live else
                     '<span class="st st-review">REVIEW</span>' if stt == "review" else '<span class="st st-ok">OK</span>')
            rows += (f'<div class="feed-row"><span class="tm">{tm:%H:%M:%S}</span>'
                     f'<span class="stg" style="color:{fg};background:{bg}">{stg}</span><span class="msg">{msg}</span>{badge}</div>')
        return (f'<div class="feed"><div class="feed-head"><span>Time UTC</span><span>Stage</span><span>Event</span><span>Status</span></div>'
                f'<div class="feed-scroll"><div>{rows}</div></div></div>')

    def plain_html(evts, running=False):
        rows = ""
        for k_, (tm, num, head, msg, stt) in enumerate(evts):
            live = running and k_ == len(evts) - 1
            badge = ('<span class="st st-run">WORKING</span>' if live else
                     '<span class="st st-review">NEEDS APPROVAL</span>' if stt == "review" else '<span class="st st-ok">DONE</span>')
            rows += (f'<div class="feed-row" style="grid-template-columns:92px 190px 1fr 140px"><span class="tm">{(tm + IST):%H:%M} IST</span>'
                     f'<span class="msg" style="color:#fff;font-weight:600">{num}. {head}</span><span class="msg">{msg}</span>{badge}</div>')
        return (f'<div class="feed"><div class="feed-head" style="grid-template-columns:92px 190px 1fr 140px"><span>Time</span><span>Step</span><span>What happened</span><span>Status</span></div>'
                f'<div class="feed-scroll"><div>{rows}</div></div></div>')

    stage_defs = [("Receive and check forecasts", ("CYCLE", "INGEST", "QC"), 45), ("Prepare India data", ("ZARR",), 10), ("Find danger zones", ("STAGE 1",), 25),
                  ("Zoom to 5 km", ("STAGE 2",), 61), ("Physics check", ("PHYSICS",), 6), ("Alert and publish", ("ALERT", "PUBLISH"), 8)]

    def stages_html(evts):
        out = '<div class="stage-list">'
        prev_end = evts[0][0]
        finished = evts[-1][1] == "PUBLISH"
        for name, keys, target in stage_defs:
            idxs = [i_ for i_, e in enumerate(evts) if e[1] in keys]
            if "STAGE 2" in keys and not n_box:
                state, sub = "wait", "Skipped: no danger zones this time"
            elif idxs:
                end_t = evts[idxs[-1]][0]
                done = finished or idxs[-1] < len(evts) - 1 and not any(e[1] in keys for e in evts[idxs[-1] + 1:])
                took = (end_t - prev_end).total_seconds()
                state = "done" if done else "run"
                sub = f"Took {took:.0f} seconds (goal {target} s)" if done else f"Working… (goal {target} s)"
                prev_end = end_t
            else:
                state, sub = "wait", f"Waiting (goal {target} s)"
            out += (f'<div class="stage-item"><span class="nm">{name}</span>'
                    f'<span class="badge badge-{state}">{ {"done": "DONE", "run": "WORKING", "wait": "WAITING"}[state] }</span><span class="sub">{sub}</span></div>')
        return out + "</div>"

    a, b = st.columns([1.55, 1])
    with a:
        c1, c2 = st.columns([1, 2])
        replay = c1.button("Replay cycle", width="stretch")
        c2.caption(f"The whole cycle took {(events[-1][0] - events[0][0]).total_seconds():.0f} seconds, from forecast arrival to published alert. Times in IST.")
        feed_box = st.empty()
        prog = st.empty()
    with b:
        stage_box = st.empty()
        chart_box = st.empty()

    fig = go.Figure(go.Bar(x=["ctl"] + [f"m{m:02d}" for m in range(1, 23)], y=np.round(arrival, 1), marker_color=BLUE,
                           hovertemplate="%{x}: %{y} s after release<extra></extra>"))
    fig.update_layout(title="Member arrival after release (s)")
    fig.update_xaxes(gridcolor=LINE, tickangle=-60)
    fig.update_yaxes(gridcolor=LINE)
    if tech:
        chart_box.plotly_chart(style_fig(fig, 260), width="stretch")

    if replay:
        for k_ in range(1, len(events) + 1):
            part = events[:k_]
            if tech:
                feed_box.markdown(feed_html(part, running=k_ < len(events)), unsafe_allow_html=True)
            else:
                pp = [p_ for p_ in plain if p_[0] <= part[-1][0]]
                if pp:
                    feed_box.markdown(plain_html(pp, running=k_ < len(events)), unsafe_allow_html=True)
            stage_box.markdown(stages_html(part), unsafe_allow_html=True)
            prog.progress(k_ / len(events), text=f"{k_}/{len(events)} events")
            _time.sleep((0.12 if part[-1][1] == "INGEST" else 0.45) if tech else (0.05 if part[-1][1] == "INGEST" else 0.35))
        prog.empty()
    else:
        feed_box.markdown(feed_html(events) if tech else plain_html(plain), unsafe_allow_html=True)
        stage_box.markdown(stages_html(events), unsafe_allow_html=True)

elif page == "tracking":
    render_media("anomaly_mesh.mp4", "ANOMALY TRACKING", "Where the danger is, and how it moves over the next 10 days", header=True)
    plain_box(cycle_story() + " Coloured squares on the map are danger zones; the dotted line is the most likely storm path, and the shaded circles show how uncertain it is.")
    left, right = st.columns([1.7, 1])
    with left:
        st.plotly_chart(india_map(680), width="stretch", key="map_track", config={"scrollZoom": True, "displayModeBar": False})
    with right:
        section_header("Danger", "Zones")
        if clusters:
            st.dataframe(pd.DataFrame([{"Zone": c["id"], "Where": near_place(*c["centroid"]).capitalize(), "Day": day_label(lead),
                                        "How unusual": round(c["max_efi"], 2), "Chance of danger level": f"{c['max_prob'] * 100:.0f}%",
                                        "Alert": E.TIERS[cluster_tier(c)][0].title()} for c in clusters]), hide_index=True, width="stretch")
        else:
            st.info("Nothing dangerous is forecast for this day.")
        html('<div class="ax-card"><p>Each danger zone is a 640 × 640 km square. Only these squares are sharpened to 5 km, which keeps the system fast.</p></div>')
    ev = pd.DataFrame([{"lead": h, "max_efi": float(get_scenario(scenario, h)["efi"].max()), "max_sot": float(get_scenario(scenario, h)["sot"].max()),
                        "prob": float(get_scenario(scenario, h)["p_exc"].max())} for h in E.LEADS])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ev["day"] = [day_short(h) for h in ev["lead"]]
    fig.add_trace(go.Scatter(x=ev["day"], y=ev["max_efi"], name="How unusual", mode="lines+markers", line=dict(color="#EE8A2F", width=3)))
    fig.add_trace(go.Scatter(x=ev["day"], y=ev["prob"], name="Chance of crossing the danger level", mode="lines+markers", line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Bar(x=ev["day"], y=ev["max_sot"], name="Record-breaking signal", marker_color="rgba(214,69,61,0.35)"), secondary_y=True)
    fig.add_vline(x=day_short(lead), line_color=NAVY, line_dash="dot")
    fig.update_xaxes(gridcolor=LINE)
    fig.update_yaxes(title="0 to 1", range=[0, 1.05], gridcolor=LINE, secondary_y=False)
    fig.update_yaxes(title="Record signal", secondary_y=True, showgrid=False)
    fig.update_layout(title="How the danger changes over the next 10 days", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(style_fig(fig, 340), width="stretch")
    if "cyclone" in cfg["key"]:
        section_header("Cyclone", "Path", "Where the storm centre is expected each day. Lower pressure means a stronger storm; the uncertainty grows further ahead.")
        st.dataframe(pd.DataFrame([{"Day": day_short(h), "Storm centre": near_place(*E.cyclone_track(h, cfg["key"])).capitalize(),
                                    "Centre pressure (hPa)": round(E.cyclone_mslp(h)), "Could be off by (km)": round(E.track_cone(h))} for h in E.LEADS]),
                     hide_index=True, width="stretch")

elif page == "downscaling":
    section_header("5 km", "Downscaling", "Turning a blurry 12 km forecast into a sharp 5 km one without losing the storm's strongest point")
    if not clusters:
        st.info("Nothing dangerous is forecast for this day, so nothing needs zooming. Open Mission Control below to pick another day.")
    else:
        c1, c2 = st.columns(2)
        box_id = c1.selectbox("Danger zone", [c["id"] for c in clusters], format_func=lambda z: f"{z} · {near_place(*next(c for c in clusters if c['id'] == z)['centroid'])}")
        box = next(c for c in clusters if c["id"] == box_id)
        realization = c2.slider("Version", 1, 4, 1, help="AIravat makes several equally likely 5 km versions of the same forecast. Slide to compare them.")

        @st.cache_data(show_spinner=False)
        def get_down(var, peak, seed):
            return E.downscale_demo(var, peak, seed)

        ds = get_down(cfg["variable"], round(box["peak_value"], 1), sum(map(ord, box_id)) % 97)
        panels = [("Today's 12 km forecast", ds["coarse"]), ("Typical AI sharpening", ds["mse"]),
                  (f"AIravat 5 km (version {realization})", ds["gens"][realization - 1]), ("What really happens (5 km)", ds["truth"])]
        pk = [float(p_[1].max()) for p_ in panels]
        plain_box(f"The typical AI approach averages many possible storms, so its {PEAK_WORD} drops to <b>{pk[1]:.0f} {unit}</b>. "
                  f"AIravat keeps it at <b>{pk[2]:.0f} {unit}</b>, close to the real <b>{pk[3]:.0f} {unit}</b>. That peak is what decides "
                  f"whether a village floods. Darker red = more intense.")
        allv = np.concatenate([p[1].ravel() for p in panels])
        zmin, zmax = float(np.percentile(allv, 1)), float(allv.max())
        scale = RAIN_SCALE if cfg["variable"] == "precip" else HEAT_SCALE
        fig = make_subplots(rows=1, cols=4, subplot_titles=[p[0] for p in panels], horizontal_spacing=0.02)
        for i, (_, arr) in enumerate(panels, start=1):
            fig.add_trace(go.Heatmap(z=arr, colorscale=scale, zmin=zmin, zmax=zmax, showscale=(i == 4), colorbar=dict(title=unit, thickness=12),
                                     hovertemplate=f"%{{z:.1f}} {unit}<extra></extra>"), 1, i)
            fig.update_xaxes(visible=False, row=1, col=i)
            fig.update_yaxes(visible=False, scaleanchor=f"x{i}", row=1, col=i)
        st.plotly_chart(style_fig(fig, 360), width="stretch")
        a, b = st.columns([1.3, 1])
        colours = ["#8A93A6", "#EE8A2F", BLUE, NAVY]
        with a:
            fig = go.Figure()
            for (name, arr), col in zip(panels, colours):
                k_, p_ = E.radial_psd(arr)
                fig.add_trace(go.Scatter(x=k_, y=p_, name=name, mode="lines", line=dict(color=col, width=2.8 if "AIravat" in name else 1.8)))
            k_, p_ = E.radial_psd(ds["truth"])
            fig.add_trace(go.Scatter(x=k_, y=p_[3] * (k_ / k_[3]) ** (-5 / 3), name="k^-5/3 reference", mode="lines", line=dict(color="#E8C547", dash="dash")))
            fig.add_vline(x=1 / 24, line_color=MUTED, line_dash="dot", annotation_text="12 km Nyquist")
            fig.update_xaxes(type="log", title="Size of weather features (left = large, right = small)", gridcolor=LINE)
            fig.update_yaxes(type="log", title="Amount of detail", gridcolor=LINE)
            fig.update_layout(title="Small-scale detail kept: the orange line drops, AIravat stays on the real one")
            st.plotly_chart(style_fig(fig, 400), width="stretch")
        with b:
            peaks = [float(p[1].max()) for p in panels]
            fig = go.Figure(go.Bar(x=peaks, y=[p[0] for p in panels], orientation="h", marker_color=colours,
                                   text=[f"{v:.0f} {unit}" for v in peaks], textposition="outside"))
            fig.update_xaxes(gridcolor=LINE, range=[0, max(peaks) * 1.4])
            fig.update_layout(title=f"{PEAK_WORD.capitalize()} in the zone")
            st.plotly_chart(style_fig(fig, 400), width="stretch")
        st.caption("Demo on simulated data. In the full system a trained diffusion model produces the 5 km versions.")

elif page == "pinpoint":
    section_header("Pinpoint", "Alerts", "Pick a place to see its alert level, what the forecast runs say, and how it changes day by day")
    c1, c2, c3 = st.columns([1.4, 1, 1])
    place = c1.selectbox("Location", ["Custom coordinates"] + list(E.PLACES.keys()), index=1)
    default = E.PLACES.get(place, (19.81, 85.83))
    qlat = c2.number_input("Latitude (°N)", 5.0, 37.0, float(default[0]), 0.01, disabled=place != "Custom coordinates")
    qlon = c3.number_input("Longitude (°E)", 66.0, 98.0, float(default[1]), 0.01, disabled=place != "Custom coordinates")
    if place != "Custom coordinates":
        qlat, qlon = default
    d = E.point_diagnostics(sc, qlat, qlon)
    name, _, protocol = E.TIERS[d["tier"]]
    colour = TIER_COLOURS[d["tier"]]
    loc_label = place if place != "Custom coordinates" else f"{qlat:.2f}°N, {qlon:.2f}°E"
    html(f'<div class="ax-banner" style="background:{colour}"><div class="t">{name.title()} · {day_label(lead)}</div>'
         f'<div class="s">{loc_label} · {TIER_MEANING[d["tier"]]}<br>{protocol}</div></div>')
    pc = d["probs"][crit] * 100
    plain_box(f"For <b>{loc_label}</b> on <b>{day_label(lead)}</b>: <b>{int((d['members'] >= crit).sum())} of the {N_RUNS} forecast runs</b> ({pc:.0f}%) show {CRIT_TEXT}. "
              f"The most likely value is <b>{d['q50']:.0f} {unit}</b>; in a bad case (1 in 10 runs) it reaches <b>{d['q90']:.0f} {unit}</b>. "
              f"Conditions like this are {efi_words(d['efi'])}.")
    m = st.columns(5)
    m[0].metric("Most likely", f"{d['q50']:.0f} {unit}", help=f"The middle value of the {N_RUNS} forecast runs.")
    m[1].metric("Bad case (1 in 10)", f"{d['q90']:.0f} {unit}", help="9 of 10 runs stay below this value.")
    m[2].metric("Chance of danger level", f"{pc:.0f}%", help=f"Share of runs that show {CRIT_TEXT}.")
    m[3].metric("How unusual", f"{d['efi']:.2f}", help="0 = normal for this place and season, 1 = beyond anything in 30 years.")
    m[4].metric("Alert score (0–1)", f"{d['score']:.2f}", help="Combines how unusual, record signal and chance. Sets the alert level.")
    a, b = st.columns(2)
    with a:
        fig = go.Figure(go.Histogram(x=d["members"], nbinsx=14, marker_color=BLUE, opacity=0.85))
        fig.add_vline(x=d["clim_q99"], line_color="#D6453D", line_width=3, annotation_text="1-in-100-day level here", annotation_font_color="#D6453D")
        fig.add_vline(x=crit, line_color=NAVY, line_dash="dot", annotation_text="Danger level", annotation_position="bottom right")
        fig.update_layout(title=f"What the {N_RUNS} forecast runs say for {loc_label.split(',')[0]}")
        fig.update_xaxes(title=f"{VAR_WORD.capitalize()} ({unit})", gridcolor=LINE)
        fig.update_yaxes(title="Number of runs", gridcolor=LINE)
        st.plotly_chart(style_fig(fig, 340), width="stretch")
    with b:
        tl = []
        for h in E.LEADS:
            dh = E.point_diagnostics(get_scenario(scenario, h), qlat, qlon)
            tl.append({"lead": h, "tier": dh["tier"], "efi": dh["efi"], "p": dh["probs"][crit]})
        tl = pd.DataFrame(tl)
        tl["day"] = [day_short(h) for h in tl["lead"]]
        fig = go.Figure(go.Bar(x=tl["day"], y=tl["p"] * 100, marker_color=[TIER_COLOURS[t] for t in tl["tier"]],
                               text=[E.TIERS[t][0].title() for t in tl["tier"]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:.0f}% chance<extra></extra>"))
        fig.update_layout(title="Day by day: chance of the danger level (bar colour = alert level)", showlegend=False, uniformtext_minsize=11, uniformtext_mode="show")
        fig.update_xaxes(gridcolor=LINE)
        fig.update_yaxes(title="Chance (%)", range=[0, 118], gridcolor=LINE)
        st.plotly_chart(style_fig(fig, 340), width="stretch")
    with st.expander("For developers: the alert as the API sends it"):
        st.code(json.dumps({
            "query_location": {"latitude": round(qlat, 4), "longitude": round(qlon, 4),
                               "nearest_grid_point": {"latitude": d["grid_lat"], "longitude": d["grid_lon"], "distance_km": round(d["distance_km"], 1)}},
            "forecast_cycle": cfg["cycle"], "lead_time_hours": lead,
            "overall_alert": {"severity_tier": name, "color_code": colour, "primary_hazard": cfg["hazard"]},
            "metrics": {"extreme_forecast_index": round(d["efi"], 3), "shift_of_tails": round(d["sot"], 3), "severity_score": round(d["score"], 3)},
            "exceedance_probabilities": {f"{cfg['variable']}_ge_{t:g}": round(p, 3) for t, p in d["probs"].items()},
            "ensemble_quantiles": {"p50": round(d["q50"], 1), "p90": round(d["q90"], 1), "p99": round(d["q99"], 1), "unit": unit},
            "action_protocol": protocol,
        }, indent=2), language="json")

elif page == "reasons":
    section_header("Why this", "Alert", "Everything behind an alert, laid out so a forecaster can check it and a district officer can trust it")
    if not clusters:
        plain_box(f"Nothing dangerous is forecast for {day_label(lead)}, so there is no alert to explain. Open Mission Control below to look at another day.")
    else:
        ordered = sorted(clusters, key=lambda c_: -cluster_tier(c_))
        box_id = st.selectbox("Alert", [c_["id"] for c_ in ordered],
                              format_func=lambda z: f"{z} · {near_place(*next(c_ for c_ in ordered if c_['id'] == z)['centroid'])}")
        c = next(x for x in ordered if x["id"] == box_id)
        t = cluster_tier(c)
        col = TIER_COLOURS[t]
        ci, cj = int(np.abs(E.LAT - c["centroid"][0]).argmin()), int(np.abs(E.LON - c["centroid"][1]).argmin())
        # the grid point in the zone with the highest chance of danger (the "worst spot")
        sub = sc["p_exc"][max(ci - 6, 0):ci + 7, max(cj - 6, 0):cj + 7]
        wi, wj = np.unravel_index(np.argmax(sub + 1e-3 * sc["efi"][max(ci - 6, 0):ci + 7, max(cj - 6, 0):cj + 7]), sub.shape)
        wi, wj = wi + max(ci - 6, 0), wj + max(cj - 6, 0)
        wlat, wlon = float(E.LAT[wi]), float(E.LON[wj])
        d = E.point_diagnostics(sc, wlat, wlon)
        members, clim = np.sort(d["members"]), d["clim_samples"]
        n_danger = int((members >= crit).sum())
        score = E.severity(c["max_efi"], c["max_sot"], c["max_prob"])[0]
        where = near_place(wlat, wlon)

        plain_box(f"<b>{E.TIERS[t][0].title()}</b> ({TIER_MEANING[t]}) for {HAZARD_WORD} {near_place(*c['centroid'])} on <b>{day_label(lead)}</b>. "
                  f"At the worst spot in the zone ({where}), <b>{n_danger} of {N_RUNS} runs</b> show {CRIT_TEXT}, while on a normal "
                  f"{cycle_dt + timedelta(hours=lead):%B} day this place gets about <b>{np.median(clim):.0f} {unit}</b>. "
                  f"Conditions like this are {efi_words(d['efi'])}.", label="The verdict in one paragraph")

        # ---------- verdict + score build-up
        a, b = st.columns([1, 1.25])
        with a:
            html(f'<div class="ax-banner" style="background:{col}"><div class="t">{E.TIERS[t][0].title()} · {TIER_MEANING[t]}</div>'
                 f'<div class="s">{c["id"]} · {cfg["hazard"]}<br>{E.TIERS[t][2]}</div></div>')
            m = st.columns(2)
            m[0].metric("Alert score", f"{score:.2f}", help="0 to 1. Built from the three ingredients on the right.")
            m[1].metric("Runs that agree", f"{c['max_prob'] * 100:.0f}%", help=f"Share of the {N_RUNS} runs that show {CRIT_TEXT} somewhere in the zone.")
            m2 = st.columns(2)
            m2[0].metric("How unusual", f"{c['max_efi']:.2f}", help="0 = normal for this place and season, 1 = beyond anything in 30 years.")
            m2[1].metric("Record signal", f"{c['max_sot']:.1f}", help="Above 0: worst runs pass the 1-in-100-day level. Above 2: likely record-breaking.")
        with b:
            contrib = {"EFI": 0.35 * max(c["max_efi"], 0), "SOT": 0.25 * min(1.0, max(c["max_sot"], 0) / 2.0), "P": 0.40 * c["max_prob"]}
            fig = go.Figure(go.Waterfall(orientation="v", measure=["relative", "relative", "relative", "total"],
                                         x=["How unusual<br>(× 0.35)", "Record signal<br>(÷ 2, × 0.25)", "Chance of danger<br>(× 0.40)", "Alert score"],
                                         y=[contrib["EFI"], contrib["SOT"], contrib["P"], 0],
                                         text=[f"+{contrib['EFI']:.2f}", f"+{contrib['SOT']:.2f}", f"+{contrib['P']:.2f}", f"{score:.2f}"],
                                         textposition="outside", connector=dict(line=dict(color=LINE)),
                                         increasing=dict(marker=dict(color=BLUE)), totals=dict(marker=dict(color=col))))
            fig.update_yaxes(range=[0, 1.18], gridcolor=LINE, title="score")
            fig.update_layout(title="How the alert score is built")
            st.plotly_chart(style_fig(fig, 360), width="stretch", config={"displayModeBar": False})

        # ---------- rule check
        section_header("Rule", "check", "The alert level is the highest level whose three conditions are all met. Green = met, grey = not met.")
        RULES = [(4, 0.95, 2.0, 0.85), (3, 0.85, 1.0, 0.65), (2, 0.70, 0.0, 0.30), (1, 0.50, None, None)]

        def chip(ok, txt):
            return (f'<span style="display:inline-block;padding:3px 9px;border-radius:999px;font-size:.8rem;font-weight:600;'
                    f'background:{"#E4F3E8" if ok else "#F1F3F6"};color:{"#2E8B45" if ok else "#8A93A6"}">{"✓" if ok else "✗"} {txt}</span>')

        rows_html = ""
        c_efi = c["max_efi"]
        for tier_k, e_need, s_need, p_need in RULES:
            e_ok = c["max_efi"] >= e_need
            s_ok = True if s_need is None else c["max_sot"] > s_need
            p_ok = True if p_need is None else c["max_prob"] >= p_need
            all_ok = e_ok and s_ok and p_ok
            here = tier_k == t
            mark = ' <b style="color:#1E2A45;margin-left:6px">← this alert</b>' if here else ""
            s_cell = "—" if s_need is None else chip(s_ok, f"needs above {s_need:g} · is {c['max_sot']:.1f}")
            p_cell = "—" if p_need is None else chip(p_ok, f"needs {p_need * 100:.0f}% ({int(np.ceil(p_need * N_RUNS))} runs) · is {c['max_prob'] * 100:.0f}%")
            rows_html += (f'<tr style="{"background:#FFF7F9" if here else ""}"><td><span class="ev-badge" style="{badge_style(tier_k)}">{E.TIERS[tier_k][0]}</span>'
                          f'{mark}</td>'
                          f'<td>{chip(e_ok, f"needs {e_need:.2f}+ · is {c_efi:.2f}")}</td><td>{s_cell}</td><td>{p_cell}</td>'
                          f'<td style="font-weight:700;color:{"#2E8B45" if all_ok else "#8A93A6"}">{"All met" if all_ok else "Not met"}</td></tr>')
        html('<table class="rule-t"><thead><tr><th>Level</th><th>How unusual</th><th>Record signal</th><th>Runs crossing danger level</th><th>Result</th></tr></thead>'
             f'<tbody>{rows_html}</tbody></table>')

        # ---------- forecast vs climate + run by run
        section_header("Forecast vs", "local climate", f"At the worst spot in the zone ({where}). Grey is 30 years of {cycle_dt + timedelta(hours=lead):%B} weather here; red is the {N_RUNS} runs for {day_label(lead)}.")
        a, b = st.columns([1.25, 1])
        with a:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=clim, nbinsx=40, histnorm="probability", marker_color="#B8C2D6", name="30-year climate (930 days)", opacity=0.9))
            fig.add_trace(go.Histogram(x=members, nbinsx=12, histnorm="probability", marker_color="#D6453D", name=f"This forecast ({N_RUNS} runs)", opacity=0.75))
            fig.add_vline(x=d["clim_q90"], line_color=MUTED, line_dash="dot", annotation_text="1-in-10-day", annotation_position="top left")
            fig.add_vline(x=d["clim_q99"], line_color=BLUE, line_dash="dash", annotation_text="1-in-100-day", annotation_font_color=BLUE, annotation_position="top right")
            fig.add_vline(x=crit, line_color="#E0661F", line_dash="dot", annotation_text="danger level", annotation_font_color="#E0661F", annotation_position="bottom right")
            fig.update_layout(barmode="overlay", title="Two distributions: normal days vs this forecast", legend=dict(orientation="h", y=-0.22))
            fig.update_xaxes(title=f"{VAR_WORD.capitalize()} ({unit})", gridcolor=LINE)
            fig.update_yaxes(title="share of days / runs", gridcolor=LINE, tickformat=".0%")
            st.plotly_chart(style_fig(fig, 380), width="stretch", config={"displayModeBar": False})
        with b:
            cols_ = ["#D6453D" if v >= crit else "#EE8A2F" if v >= d["clim_q99"] else "#8A93A6" for v in members]
            fig = go.Figure(go.Bar(x=members, y=[f"run {k + 1:02d}" for k in range(len(members))], orientation="h", marker_color=cols_,
                                   hovertemplate="%{y}: %{x:.0f} " + unit + "<extra></extra>"))
            fig.add_vline(x=crit, line_color="#E0661F", line_dash="dot")
            fig.add_vline(x=d["clim_q99"], line_color=BLUE, line_dash="dash")
            fig.update_layout(title="Run by run, lowest to highest", showlegend=False)
            fig.update_xaxes(title=unit, gridcolor=LINE)
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(style_fig(fig, 380), width="stretch", config={"displayModeBar": False})
            st.caption(f"Red: at or above the danger level ({n_danger} runs). Orange: above the 1-in-100-day level only. Grey: below both.")

        # ---------- day by day
        rows = []
        for h in E.LEADS:
            dh = E.point_diagnostics(get_scenario(scenario, h), wlat, wlon)
            rows.append({"lead": h, "day": day_short(h), "tier": dh["tier"], "score": dh["score"], "p": dh["probs"][crit]})
        tl = pd.DataFrame(rows)
        alerted = tl[tl["tier"] >= 1]
        peak = tl.loc[tl["score"].idxmax()]
        if len(alerted):
            story = (f"First alert ({E.TIERS[int(alerted.iloc[0]['tier'])][0].title()}) on {day_label(int(alerted.iloc[0]['lead']))}; "
                     f"peaks at {E.TIERS[int(peak['tier'])][0].title()} on {day_label(int(peak['lead']))}; "
                     f"{len(alerted)} of {len(tl)} days carry an alert.")
        else:
            story = "No alert on any day of the forecast at this spot."
        section_header("Day by day", "at this spot", "How the same place looks across the 10-day forecast. " + story)
        fig = go.Figure(go.Bar(x=tl["day"], y=tl["score"], marker_color=[TIER_COLOURS[k] for k in tl["tier"]],
                               text=[E.TIERS[k][0].title() for k in tl["tier"]], textposition="outside", cliponaxis=False, name="Alert score", showlegend=False,
                               hovertemplate="%{x}<br>score %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=tl["day"], y=tl["p"], mode="lines+markers", line=dict(color=NAVY, width=2), name="Chance of danger level"))
        fig.add_vline(x=day_short(lead), line_color=MUTED, line_dash="dot")
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=1, xanchor="right"), uniformtext_minsize=10, uniformtext_mode="show")
        fig.update_yaxes(range=[0, 1.15], gridcolor=LINE, tickformat=".0%", title="score / chance")
        fig.update_xaxes(gridcolor=LINE)
        st.plotly_chart(style_fig(fig, 320), width="stretch", config={"displayModeBar": False})

        # ---------- what would change it
        section_header("What would", "change this alert", "The margins that separate this alert from the level above and below it.")
        cards = []
        cur = next((r for r in RULES if r[0] == t), None)
        if cur and cur[3] is not None:
            need = int(np.ceil(cur[3] * N_RUNS))
            have = int(round(c["max_prob"] * N_RUNS))
            cards.append(("Stays " + E.TIERS[t][0].title(), f"as long as at least <b>{need} of {N_RUNS} runs</b> cross the danger level. Now: <b>{have}</b>, a margin of {have - need} run{'s' if have - need != 1 else ''}.", col))
        up = next((r for r in RULES if r[0] == t + 1), None)
        if up:
            miss = []
            if c["max_efi"] < up[1]:
                miss.append(f"how unusual must reach {up[1]:.2f} (now {c['max_efi']:.2f})")
            if up[2] is not None and c["max_sot"] <= up[2]:
                miss.append(f"record signal must pass {up[2]:g} (now {c['max_sot']:.1f})")
            if up[3] is not None and c["max_prob"] < up[3]:
                miss.append(f"{int(np.ceil(up[3] * N_RUNS))} runs must cross the danger level (now {int(round(c['max_prob'] * N_RUNS))})")
            cards.append(("Would rise to " + E.TIERS[t + 1][0].title(), "if " + "; ".join(miss) + "." if miss else "All conditions already met.", TIER_COLOURS[t + 1]))
        else:
            cards.append(("Already the highest level", "No stronger alert exists. The next cycle can only confirm, extend or lower it.", col))
        cards.append(("Next update", f"The next NEPS-G run ({(cycle_dt + timedelta(hours=12)):%d %b %H} UTC) re-scores this zone about 2.5 minutes after it is published (design target).", BLUE))
        for colx, (h_, txt_, cc_) in zip(st.columns(len(cards)), cards):
            colx.markdown(f'<div class="ax-card" style="border-top:4px solid {cc_}"><h4 style="color:{cc_}">{h_}</h4><p>{txt_}</p></div>', unsafe_allow_html=True)

        # ---------- checks + audit
        a, b = st.columns(2)
        with a:
            section_header("Checks", "passed")
            checks = [("Forecast data complete", f"{N_RUNS} of {N_RUNS} runs received, checksums verified"),
                      ("Compared with local climate", "930 days of ERA5 (1991–2020, ±15 days of this date)"),
                      ("Physics check (demo)", "Mass, moisture and energy residuals within tolerance"),
                      ("Runs agree on location", f"Zone found by {c['nodes']} neighbouring mesh cells, not a single outlier")]
            html("".join(f'<div class="ax-card" style="padding:12px 16px;margin-bottom:8px"><p><b style="color:#2E8B45">✓ {h_}</b><br>{t_}</p></div>' for h_, t_ in checks))
        with b:
            section_header("Audit", "trail")
            audit = {"alert_id": c["id"], "level": E.TIERS[t][0], "valid_for": day_label(lead), "forecast_run": f"NEPS-G {cycle_dt:%Y-%m-%d %H} UTC",
                     "worst_spot": {"lat": wlat, "lon": wlon, "description": where}, "score": round(score, 3),
                     "ingredients": {"how_unusual_efi": round(c["max_efi"], 3), "record_signal_sot": round(c["max_sot"], 3), "runs_crossing": f"{int(round(c['max_prob'] * N_RUNS))}/{N_RUNS}"},
                     "danger_level": f"{crit:g} {unit}", "local_1_in_100_day_level": f"{d['clim_q99']:.1f} {unit}",
                     "rules_version": "AIravat tiers v0.3 (IMD colour codes)", "status": "Awaiting IMD forecaster approval" if t >= 3 else "Published to dashboard"}
            status_col = "#EE8A2F" if t >= 3 else "#2E9E5B"
            status_bg = "rgba(238,138,47,.12)" if t >= 3 else "rgba(46,158,91,.12)"
            html(f"""
            <div style="background:#FFFFFF;border:1px solid {LINE};border-radius:16px;padding:20px 22px;color:{BODY};font-family:var(--font-body)">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
                <div>
                  <div style="font-family:var(--font-mono);font-size:.6rem;letter-spacing:.18em;text-transform:uppercase;color:{MUTED};margin-bottom:4px">Alert ID</div>
                  <div style="font-family:var(--font-display);font-size:1.4rem;font-weight:800;color:{NAVY}">{audit['alert_id']}</div>
                </div>
                <span class="ev-badge" style="{badge_style(t)}">{audit['level']}</span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px">
                <div style="background:#F4F6FB;border:1px solid {LINE};border-radius:10px;padding:12px">
                  <div style="font-family:var(--font-mono);font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTED};margin-bottom:4px">Valid for</div>
                  <div style="font-weight:600;color:{NAVY}">{audit['valid_for']}</div>
                </div>
                <div style="background:#F4F6FB;border:1px solid {LINE};border-radius:10px;padding:12px">
                  <div style="font-family:var(--font-mono);font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTED};margin-bottom:4px">Forecast run</div>
                  <div style="font-weight:600;color:{NAVY}">{audit['forecast_run']}</div>
                </div>
                <div style="background:#F4F6FB;border:1px solid {LINE};border-radius:10px;padding:12px">
                  <div style="font-family:var(--font-mono);font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTED};margin-bottom:4px">Worst spot</div>
                  <div style="font-weight:600;color:{NAVY}">{audit['worst_spot']['description'][0].upper() + audit['worst_spot']['description'][1:]}</div>
                  <div style="font-size:.8rem;color:{MUTED};margin-top:2px">{audit['worst_spot']['lat']:.1f}°N, {audit['worst_spot']['lon']:.1f}°E</div>
                </div>
                <div style="background:#F4F6FB;border:1px solid {LINE};border-radius:10px;padding:12px">
                  <div style="font-family:var(--font-mono);font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTED};margin-bottom:4px">Alert score</div>
                  <div style="font-family:var(--font-display);font-size:1.6rem;font-weight:800;color:{col}">{audit['score']:.3f}</div>
                </div>
              </div>
              <div style="background:#F4F6FB;border:1px solid {LINE};border-radius:10px;padding:14px;margin-bottom:14px">
                <div style="font-family:var(--font-mono);font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTED};margin-bottom:10px">Score ingredients</div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
                  <div style="text-align:center">
                    <div style="font-family:var(--font-display);font-size:1.3rem;font-weight:800;color:#EE8A2F">{audit['ingredients']['how_unusual_efi']:.3f}</div>
                    <div style="font-size:.75rem;color:{MUTED};margin-top:2px">How unusual (EFI)</div>
                  </div>
                  <div style="text-align:center">
                    <div style="font-family:var(--font-display);font-size:1.3rem;font-weight:800;color:#D6453D">{audit['ingredients']['record_signal_sot']:.3f}</div>
                    <div style="font-size:.75rem;color:{MUTED};margin-top:2px">Record signal (SOT)</div>
                  </div>
                  <div style="text-align:center">
                    <div style="font-family:var(--font-display);font-size:1.3rem;font-weight:800;color:#2F80ED">{audit['ingredients']['runs_crossing']}</div>
                    <div style="font-size:.75rem;color:{MUTED};margin-top:2px">Runs crossing</div>
                  </div>
                </div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px">
                <div style="background:#F4F6FB;border:1px solid {LINE};border-radius:10px;padding:12px">
                  <div style="font-family:var(--font-mono);font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTED};margin-bottom:4px">Danger level</div>
                  <div style="font-weight:700;color:#D6453D">{audit['danger_level']}</div>
                </div>
                <div style="background:#F4F6FB;border:1px solid {LINE};border-radius:10px;padding:12px">
                  <div style="font-family:var(--font-mono);font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTED};margin-bottom:4px">1-in-100-day level</div>
                  <div style="font-weight:700;color:{BLUE}">{audit['local_1_in_100_day_level']}</div>
                </div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
                <div style="background:#F4F6FB;border:1px solid {LINE};border-radius:10px;padding:12px">
                  <div style="font-family:var(--font-mono);font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTED};margin-bottom:4px">Rules version</div>
                  <div style="font-size:.88rem;color:{BODY}">{audit['rules_version']}</div>
                </div>
                <div style="background:#F4F6FB;border:1px solid {LINE};border-radius:10px;padding:12px">
                  <div style="font-family:var(--font-mono);font-size:.58rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTED};margin-bottom:4px">Status</div>
                  <div style="display:inline-block;padding:4px 12px;border-radius:999px;font-weight:600;font-size:.82rem;background:{status_bg};color:{status_col}">{audit['status']}</div>
                </div>
              </div>
            </div>""")
            st.download_button("Download alert reasons (JSON)", json.dumps(audit, indent=2), file_name=f"{c['id']}_reasons.json", mime="application/json")

elif page == "copilot":
    section_header("AIravat", "Copilot", "Ask in English or Hindi. Answers come only from the current forecast cycle and alert data.")
    lang = st.radio("Answer language", ["English", "हिन्दी"], horizontal=True)
    st.caption("Try: “Which places are on alert?”, “Status for Puri”, “Where will the cyclone land?”, “What is EFI?”, “Write a SitRep”.")
    EN = lang == "English"

    def place_status(pname):
        dd = E.point_diagnostics(sc, *E.PLACES[pname])
        tn, pc = E.TIERS[dd["tier"]][0].title(), dd["probs"][crit] * 100
        if EN:
            return (f"**{pname}**, {day_label(lead)}: **{tn}** ({TIER_MEANING[dd['tier']]}). {pc:.0f}% of forecast runs show {CRIT_TEXT}; "
                    f"most likely {dd['q50']:.0f} {unit}, bad case {dd['q90']:.0f} {unit}. Conditions like this are {efi_words(dd['efi'])}. {E.TIERS[dd['tier']][2]}")
        return f"**{pname}**, {day_label(lead)}: **{tn}**। {crit:g} {unit} से अधिक की संभावना {pc:.0f}% है; सबसे संभावित मान {dd['q50']:.0f} {unit}।"

    def sitrep():
        ts = f"{cycle_dt:%d %b %Y %H} UTC"
        if EN:
            body = [f"**SITREP · AIravat · forecast issued {ts} · for {day_label(lead)}**",
                    f"Hazard: {cfg['hazard']}. Danger zones: {len(clusters)}. Highest alert: {E.TIERS[top_tier][0].title()} ({TIER_MEANING[top_tier]})."]
            body += [f"- {c['id']}, {near_place(*c['centroid'])}: {c['max_prob'] * 100:.0f}% of runs show {CRIT_TEXT}; "
                     f"alert {E.TIERS[cluster_tier(c)][0].title()}." for c in clusters]
            body.append("Recommended: " + E.TIERS[top_tier][2])
            return "\n".join(body)
        body = [f"**स्थिति रिपोर्ट · AIravat · पूर्वानुमान {ts} · {day_label(lead)} के लिए**", f"खतरे वाले क्षेत्र: {len(clusters)}। उच्चतम अलर्ट: {E.TIERS[top_tier][0].title()}।"]
        body += [f"- {c['id']}: केंद्र {c['centroid'][0]:.1f}°N {c['centroid'][1]:.1f}°E, EFI {c['max_efi']:.2f}, SOT {c['max_sot']:.2f}।" for c in clusters]
        return "\n".join(body)

    def answer(q):
        ql = q.lower()
        for pname in E.PLACES:
            if pname.split(",")[0].lower() in ql:
                return place_status(pname)
        if any(w in ql for w in ["sitrep", "report", "summary", "रिपोर्ट"]):
            return sitrep()
        if any(w in ql for w in ["land", "track", "cyclone", "चक्रवात"]):
            if "cyclone" not in cfg["key"]:
                return "The current scenario has no cyclone. Switch the scenario to a cyclone in Mission Control." if EN else "इस परिदृश्य में कोई चक्रवात नहीं है।"
            la, lo = E.cyclone_track(126, cfg["key"])
            coast_name = "Saurashtra / Porbandar coast" if "arabian" in cfg["key"] else "Puri–Gopalpur coast"
            if EN:
                return (f"The most likely path brings the cyclone to the **{coast_name}** around **{(cycle_dt + timedelta(hours=126)):%a %d %b}** "
                        f"(about {126 // 24} days after the forecast was issued). The landfall point could be off by about {E.track_cone(126):.0f} km.")
            return f"सबसे संभावित पथ के अनुसार चक्रवात लगभग **{(cycle_dt + timedelta(hours=126)):%d %b}** को **{coast_name}** के पास पहुँचेगा।"
        if "efi" in ql or "extreme forecast" in ql:
            return (f"EFI compares the {N_RUNS}-run forecast with the 30-year local climate for the same date. 0 means normal; +1 means every member is beyond "
                    "the climate record. AIravat computes it exactly, without numerical approximation."
                    if EN else "EFI बताता है कि पूर्वानुमान उसी तारीख के 30 वर्ष के स्थानीय जलवायु से कितना अलग है। 0 = सामान्य, +1 = अत्यधिक।")
        if "sot" in ql or "shift of tails" in ql:
            return ("SOT checks whether the worst 10% of members go past the climate's 99th percentile. SOT > 0 means yes; above 2 signals a record-breaking event."
                    if EN else "SOT बताता है कि सबसे खराब 10% सदस्य जलवायु के 99वें प्रतिशतक से आगे हैं या नहीं।")
        if any(w in ql for w in ["alert", "warning", "which", "where", "अलर्ट", "चेतावनी"]):
            rows = [r for r in sorted([(p, E.point_diagnostics(sc, *E.PLACES[p])["tier"]) for p in E.PLACES], key=lambda r: -r[1]) if r[1] >= 1]
            if not rows:
                return "No listed place is above Normal at this lead time." if EN else "इस समय किसी स्थान पर चेतावनी नहीं है।"
            return "\n".join(f"- **{p}**: {E.TIERS[t][0].title()}" for p, t in rows)
        return ("I can answer about alerts, a named place (e.g. Puri, Bikaner, Ratnagiri), the cyclone track, EFI/SOT, or write a SitRep."
                if EN else "मैं अलर्ट, किसी स्थान (जैसे पुरी), चक्रवात पथ, EFI/SOT या स्थिति रिपोर्ट के बारे में बता सकता हूँ।")

    if "chat" not in st.session_state:
        st.session_state.chat = []
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)
    q = st.chat_input("Ask about the current forecast…")
    if q:
        st.session_state.chat += [("user", q), ("assistant", answer(q))]
        st.rerun()

elif page == "technology":
    tech_file = ASSETS / "airavat_tech_page.html"
    if tech_file.exists():
        tab = str(qp.get("tab", "pipeline")).lower()
        tab = {"extreme": "explain", "reasons": "explain", "diffusion": "math", "physics": "math", "teamwork": "explain"}.get(tab, tab)
        if tab not in ("pipeline", "fingerprint", "explain", "math"):
            tab = "pipeline"
        page_html = tech_file.read_text(encoding="utf-8").replace("__INITIAL_TAB__", tab).replace("__RETURN_QUERY__", link("home"))
        if hasattr(st, "iframe"):
            st.iframe(page_html, height=880)
        else:
            components.html(page_html, height=880, scrolling=True)
    else:
        st.info("Add assets/airavat_tech_page.html to show the technology page.")


# ---------------------------------------------------------------------------
#  MISSION CONTROL (settings, at the bottom like VahniX)
# ---------------------------------------------------------------------------
if page not in ("home", "technology"):
    with st.expander("Mission Control — change the day, situation or map", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.selectbox("Weather situation", SCEN_NAMES, key="set_scen")
        c2.select_slider("Day to look at", options=E.LEADS, key="set_lead", format_func=lambda h: day_short(h))
        c3.selectbox("What the map shows", LAYERS, key="set_layer", format_func=lambda v: LAYER_PLAIN[v])
        c4.slider("Sensitivity (lower = more zones)", 0.50, 0.95, step=0.05, key="set_thr",
                  help="How unusual the forecast must be before a place counts as a danger zone.")
    st.query_params.update({"page": page, "sc": str(SCEN_NAMES.index(st.session_state.set_scen)), "lead": str(st.session_state.set_lead),
                            "layer": str(LAYERS.index(st.session_state.set_layer)), "thr": str(int(round(st.session_state.set_thr * 100)))})

nav_scroll_script(page)

html("""
<div class="site-footer">
  <p>AIravat extreme-weather intelligence · SIH 2026 · Problem Statement 26078</p>
  <p>Ministry of Earth Sciences · Medium-range forecasting · Demo data is synthetic</p>
</div>""")
