"""
FlightInsight — Streamlit Dashboard.

Main entry point. Streamlit avtomatsko ustvari multi-page navigation
iz pages/ mape.

Zagon:
    uv run streamlit run streamlit_app/app.py
"""

import streamlit as st


st.set_page_config(
    page_title="FlightInsight",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("✈️ FlightInsight")
    st.subheader("Inteligentni sistem za napovedovanje zamud letov")

    st.markdown("---")

    # Welcome content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Dobrodošli!

        **FlightInsight** je celovit inteligentni sistem za analizo letalskih zamud,
        ki združuje:

        - 🛫 **Napoved zamude letov** — XGBoost regresijski model
        - 💬 **Sentiment analiza komentarjev** — RoBERTa transformer
        - 📊 **Analitika in vizualizacije** — interaktivni grafi
        - 🔧 **Admin nadzorna plošča** — monitoring, validacija, drift detection

        Uporabi navigacijo na levi za dostop do funkcionalnosti.
        """)

        st.markdown("---")

        st.markdown("""
        ### Kako uporabljati

        1. **Predict Delay** — vnesi podatke o letu in dobi napoved zamude
        2. **Review Sentiment** — vpiši komentar in dobi sentiment analizo
        3. **Analytics** — preglej zgodovinske trende in statistike
        4. **Admin** — preveri zdravje sistema (samo za administratorje)
        """)

    with col2:
        st.info("""
        **Tehnologije**

        - DVC + DagsHub
        - Great Expectations
        - Evidently AI
        - MLflow
        - XGBoost
        - HuggingFace Transformers
        - GitHub Actions CI/CD
        """)

        st.success("""
        **Modeli v produkciji**

        ✅ XGBoost (zamude)
        ✅ RoBERTa (sentiment)
        """)

    st.markdown("---")

    # Quick stats
    st.markdown("### 📈 Hitre statistike")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Naučenih letov", "1M+", "2024")
    with col2:
        st.metric("Modelov", "2", "+0")
    with col3:
        st.metric("Letalskih družb", "13", "ZDA")
    with col4:
        st.metric("Test MAE", "21 min", "-25% YoY")


if __name__ == "__main__":
    main()