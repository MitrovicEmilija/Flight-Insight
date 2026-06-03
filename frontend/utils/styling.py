"""
FlightInsight — skupni stilski modul.

Centralizira ves izgled aplikacije:
- apply_base_styles()  -> injecta CSS (kliči ENKRAT na vrhu vsake strani)
- page_header(...)     -> editorial naslov strani z ikono
- section_title(...)   -> podnaslov sekcije s tanko črto
- theme_fig(fig)       -> poenoten Plotly izgled za vse grafe
- barvne konstante     -> INK, MUTED, ACCENT, semantične barve, lestvice

Uporaba ikon: Material Symbols (https://fonts.google.com/icons).
"""

from __future__ import annotations

import streamlit as st


# ----------------------------------------------------------------------------
# Barvni tokeni (uporabljaj te povsod namesto hardcodanih hex vrednosti)
# ----------------------------------------------------------------------------
ACCENT = "#2B59FF"
ACCENT_SOFT = "#EAF0FF"
INK = "#0F1419"
MUTED = "#5B6573"
BORDER = "#E3E7ED"
GRID = "#EDEFF3"
CANVAS = "#F7F8FA"
SURFACE = "#FFFFFF"

SUCCESS = "#11845B"
WARNING = "#B7791F"
DANGER = "#C0362C"
INFO = "#0E7C86"

# Lestvice za grafe
CATEGORICAL = ["#2B59FF", "#11845B", "#B7791F", "#C0362C", "#6C5CE7", "#0E7C86", "#9B5DE5", "#5B6573"]
SEQ_BLUE = ["#EAF0FF", "#C3D3FF", "#90ACFF", "#5C82F7", "#2B59FF", "#1B3FB0", "#122A78"]
# "Toplotna" lestvica za zamude (več = slabše), umirjena namesto kričečega RdYlGn
SEQ_DELAY = ["#FDF1E7", "#F6C99A", "#E89A4E", "#D5602E", "#A8321F"]
# Fiksne barve za sentiment
SENTIMENT_COLORS = {"positive": SUCCESS, "neutral": MUTED, "negative": DANGER}


# ----------------------------------------------------------------------------
# CSS
# ----------------------------------------------------------------------------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

:root {{
  --fi-accent: {ACCENT};
  --fi-accent-soft: {ACCENT_SOFT};
  --fi-ink: {INK};
  --fi-muted: {MUTED};
  --fi-border: {BORDER};
  --fi-surface: {SURFACE};
}}

/* ---- Layout ---------------------------------------------------------- */
.block-container {{
  max-width: 1200px;
  padding-top: 2.2rem;
  padding-bottom: 4rem;
}}

/* Skrij privzeti Streamlit chrome (footer "Made with Streamlit") */
footer {{ visibility: hidden; height: 0; }}
#MainMenu {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* ---- Tipografija ----------------------------------------------------- */
html, body, [class*="css"] {{ -webkit-font-smoothing: antialiased; }}
h1, h2, h3, h4 {{ letter-spacing: -0.018em; color: var(--fi-ink); }}

/* ---- Naslov strani (page_header) ------------------------------------ */
.fi-header {{
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.4rem;
}}
.fi-header-icon {{
  flex: 0 0 auto;
  width: 48px; height: 48px;
  display: grid; place-items: center;
  border-radius: 14px;
  background: var(--fi-accent-soft);
  color: var(--fi-accent);
  border: 1px solid {ACCENT}22;
}}
.fi-header-icon .material-symbols-outlined {{ font-size: 26px; }}
.fi-title {{
  font-size: 1.85rem; font-weight: 800; line-height: 1.1;
  margin: 0; color: var(--fi-ink);
}}
.fi-subtitle {{
  margin: 0.15rem 0 0 0; color: var(--fi-muted);
  font-size: 0.98rem; font-weight: 400;
}}
.fi-rule {{
  border: none; border-top: 1px solid var(--fi-border);
  margin: 1.4rem 0 1.6rem 0;
}}

/* ---- Sekcijski naslovi (section_title) ------------------------------ */
.fi-section {{
  display: flex; align-items: center; gap: 0.5rem;
  margin: 1.6rem 0 0.8rem 0;
  font-size: 1.12rem; font-weight: 700; color: var(--fi-ink);
}}
.fi-section .material-symbols-outlined {{
  font-size: 20px; color: var(--fi-accent);
}}
.fi-eyebrow {{
  text-transform: uppercase; letter-spacing: 0.08em;
  font-size: 0.72rem; font-weight: 600; color: var(--fi-muted);
}}

/* ---- Metrike kot kartice -------------------------------------------- */
[data-testid="stMetric"] {{
  background: var(--fi-surface);
  border: 1px solid var(--fi-border);
  border-radius: 14px;
  padding: 1rem 1.1rem;
  transition: border-color .15s ease, transform .15s ease;
}}
[data-testid="stMetric"]:hover {{
  border-color: {ACCENT}55;
  transform: translateY(-1px);
}}
[data-testid="stMetricLabel"] p {{
  font-size: 0.74rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--fi-muted) !important;
}}
[data-testid="stMetricValue"] {{
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em;
  color: var(--fi-ink) !important;
}}

/* ---- Gumbi ----------------------------------------------------------- */
.stButton button, .stDownloadButton button, .stLinkButton a {{
  border-radius: 10px;
  font-weight: 600;
  border: 1px solid var(--fi-border);
  transition: all .15s ease;
}}
.stButton button[kind="primary"] {{
  background: var(--fi-accent);
  border-color: var(--fi-accent);
  box-shadow: 0 1px 2px {ACCENT}33;
}}
.stButton button[kind="primary"]:hover {{
  background: #1B3FB0; border-color: #1B3FB0;
}}
.stButton button[kind="secondary"]:hover {{
  border-color: {ACCENT}66; color: var(--fi-accent);
}}

/* ---- Zavihki (tabs) -------------------------------------------------- */
[data-baseweb="tab-list"] {{
  gap: 0.25rem;
  border-bottom: 1px solid var(--fi-border);
}}
[data-baseweb="tab"] {{
  font-weight: 600; color: var(--fi-muted);
  padding: 0.5rem 0.9rem;
}}
[data-baseweb="tab"][aria-selected="true"] {{ color: var(--fi-ink); }}
[data-baseweb="tab-highlight"] {{ background: var(--fi-accent); }}

/* ---- Expanderji, dataframe, info-boksi ------------------------------ */
[data-testid="stExpander"] {{
  border: 1px solid var(--fi-border);
  border-radius: 12px;
}}
[data-testid="stDataFrame"] {{
  border: 1px solid var(--fi-border);
  border-radius: 12px;
}}
[data-testid="stAlert"] {{ border-radius: 12px; }}

/* ---- Sidebar --------------------------------------------------------- */
[data-testid="stSidebar"] {{ border-right: 1px solid var(--fi-border); }}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ font-size: 1rem; }}

/* ---- Status pills (predict stran) ----------------------------------- */
.fi-pill {{
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .3rem .7rem; border-radius: 999px;
  font-size: .85rem; font-weight: 600;
}}
.fi-pill.green  {{ background: {SUCCESS}14; color: {SUCCESS}; }}
.fi-pill.amber  {{ background: {WARNING}1A; color: {WARNING}; }}
.fi-pill.orange {{ background: #D5602E1A; color: #D5602E; }}
.fi-pill.red    {{ background: {DANGER}14; color: {DANGER}; }}

/* ---- Kartica (poljubna vsebina) ------------------------------------- */
.fi-card {{
  background: var(--fi-surface);
  border: 1px solid var(--fi-border);
  border-radius: 14px;
  padding: 1.2rem 1.3rem;
}}
.material-symbols-outlined {{
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
  vertical-align: middle;
}}
</style>
"""


def apply_base_styles() -> None:
    """Injecta globalni CSS. Kliči enkrat na vrhu vsake strani (po set_page_config)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def _icon(name: str | None) -> str:
    if not name:
        return ""
    return f'<span class="material-symbols-outlined">{name}</span>'


def page_header(title: str, subtitle: str = "", icon: str | None = "insights") -> None:
    """Editorial naslov strani z ikono, podnaslovom in tanko ločilno črto."""
    st.markdown(
        f"""
        <div class="fi-header">
          <div class="fi-header-icon">{_icon(icon)}</div>
          <div>
            <div class="fi-title">{title}</div>
            {f'<div class="fi-subtitle">{subtitle}</div>' if subtitle else ''}
          </div>
        </div>
        <hr class="fi-rule"/>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, icon: str | None = None) -> None:
    """Podnaslov sekcije (nadomesti st.subheader + emoji)."""
    st.markdown(
        f'<div class="fi-section">{_icon(icon)}<span>{title}</span></div>',
        unsafe_allow_html=True,
    )


def divider() -> None:
    """Tiha hairline ločilna črta (nadomesti st.markdown('---'))."""
    st.markdown('<hr class="fi-rule"/>', unsafe_allow_html=True)


def status_pill(label: str, tone: str = "green") -> str:
    """Vrne HTML za status značko. tone: green | amber | orange | red."""
    return f'<span class="fi-pill {tone}">{label}</span>'


# ----------------------------------------------------------------------------
# Plotly tema
# ----------------------------------------------------------------------------
def theme_fig(fig, *, height: int | None = None, legend: bool = True):
    """
    Aplicira poenoten izgled na Plotly figuro. Vrne isto figuro (chainable).
    """
    bottom_margin = 64 if legend else 16
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=13, color=INK),
        title=dict(
            font=dict(size=15, color=INK),
            x=0, xanchor="left",
            y=0.98, yanchor="top", yref="container",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CATEGORICAL,
        margin=dict(l=12, r=16, t=58, b=bottom_margin),
        hoverlabel=dict(
            bgcolor=SURFACE, bordercolor=BORDER,
            font=dict(family="Inter, sans-serif", color=INK, size=12),
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.22, x=0,
            font=dict(size=12, color=MUTED), title=None,
        ),
        showlegend=legend,
    )
    fig.update_xaxes(
        gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
        tickfont=dict(color=MUTED, size=12),
        title_font=dict(color=MUTED, size=12),
    )
    fig.update_yaxes(
        gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER,
        tickfont=dict(color=MUTED, size=12),
        title_font=dict(color=MUTED, size=12),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig