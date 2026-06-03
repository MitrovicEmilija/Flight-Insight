"""
FlightInsight — Streamlit Dashboard (entry point).

Streamlit avtomatsko ustvari multi-page navigacijo iz pages/ mape.

Zagon:
    uv run streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from styling import (
    apply_base_styles,
    page_header,
    section_title,
    divider,
    ACCENT,
    MUTED,
)


st.set_page_config(
    page_title="FlightInsight",
    page_icon=":material/flight_takeoff:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_base_styles()


def capability_card(icon: str, title: str, desc: str, tag: str) -> str:
    return f"""
    <div class="fi-card" style="height:100%;">
      <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.6rem;">
        <span class="material-symbols-outlined" style="color:{ACCENT};font-size:22px;">{icon}</span>
        <span style="font-weight:700;color:#0F1419;">{title}</span>
      </div>
      <p style="color:{MUTED};font-size:.9rem;line-height:1.5;margin:0 0 .8rem 0;">{desc}</p>
      <span class="fi-eyebrow">{tag}</span>
    </div>
    """


def main():
    page_header(
        "FlightInsight",
        "Inteligentni sistem za napovedovanje in analizo zamud letov",
        icon="flight_takeoff",
    )

    section_title("Zmožnosti sistema", icon="apps")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(capability_card(
            "schedule", "Napoved zamude",
            "Regresijski model napove pričakovano zamudo odhoda v minutah na podlagi rute, časa in letalske družbe.",
            "XGBoost regresor",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(capability_card(
            "reviews", "Sentiment analiza",
            "Transformer klasificira komentarje potnikov v pozitivne, nevtralne ali negativne z oceno zaupanja.",
            "RoBERTa transformer",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(capability_card(
            "monitoring", "Analitika trendov",
            "Interaktivne vizualizacije zgodovinskih vzorcev zamud po družbah, urah, sezonah in rutah.",
            "~7M letov / 2024",
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(capability_card(
            "tune", "Nadzorna plošča",
            "Live monitoring napovedi, validacija podatkov, detekcija drifta in sledenje eksperimentov.",
            "MLOps observability",
        ), unsafe_allow_html=True)

    divider()

    # --- Uvod + tehnologije ------------------------------------------------
    col_main, col_side = st.columns([2, 1], gap="large")

    with col_main:
        section_title("Kako uporabljati", icon="route")
        st.markdown(
            f"""
            <ol style="color:#0F1419;line-height:2;padding-left:1.1rem;">
              <li><b>Predict Delay</b> — vnesi podatke o letu in pridobi napoved zamude.</li>
              <li><b>Review Sentiment</b> — vpiši komentar in pridobi sentiment analizo.</li>
              <li><b>Analytics</b> — preglej zgodovinske trende in statistike.</li>
              <li><b>Admin</b> — preveri zdravje sistema in produkcijski monitoring.</li>
            </ol>
            <p style="color:{MUTED};font-size:.9rem;">
              Uporabi navigacijo na levi za dostop do posameznih modulov.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with col_side:
        section_title("Tehnološki sklad", icon="layers")
        tech = ["DVC + DagsHub", "Great Expectations", "Evidently AI", "MLflow",
                "XGBoost", "HuggingFace Transformers", "GitHub Actions CI/CD"]
        chips = "".join(
            f'<span class="fi-pill" style="background:#F1F3F7;color:#0F1419;'
            f'margin:0 .3rem .4rem 0;">{t}</span>'
            for t in tech
        )
        st.markdown(f'<div style="margin-top:.2rem;">{chips}</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="fi-card" style="margin-top:1rem;">
              <span class="fi-eyebrow">Modeli v produkciji</span>
              <div style="margin-top:.6rem;display:flex;flex-direction:column;gap:.4rem;">
                <div style="display:flex;align-items:center;gap:.5rem;color:#0F1419;">
                  <span class="material-symbols-outlined" style="color:#11845B;font-size:18px;">check_circle</span>
                  XGBoost — napoved zamud
                </div>
                <div style="display:flex;align-items:center;gap:.5rem;color:#0F1419;">
                  <span class="material-symbols-outlined" style="color:#11845B;font-size:18px;">check_circle</span>
                  RoBERTa — sentiment
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    divider()

    # --- Hitre statistike --------------------------------------------------
    section_title("Hitre statistike", icon="query_stats")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Naučenih letov", "1M+", "2024")
    with m2:
        st.metric("Modelov", "2", "+0")
    with m3:
        st.metric("Letalskih družb", "13", "ZDA")
    with m4:
        st.metric("Test MAE", "21 min", "-25% YoY")


if __name__ == "__main__":
    main()