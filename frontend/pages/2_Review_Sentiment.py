import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from model_loader import load_review_analyzer
from monitoring import log_prediction
from styling import (
    apply_base_styles,
    page_header,
    section_title,
    divider,
    theme_fig,
    SENTIMENT_COLORS,
    SUCCESS,
    DANGER,
    MUTED,
)


st.set_page_config(page_title="Review Sentiment", page_icon=":material/reviews:", layout="wide")
apply_base_styles()

page_header(
    "Sentiment analiza letalskih komentarjev",
    "Vpiši komentar o letu in sistem klasificira sentiment (RoBERTa transformer).",
    icon="reviews",
)

with st.spinner("Nalagam RoBERTa model (lahko traja prvič)..."):
    analyzer = load_review_analyzer()

if analyzer is None:
    st.stop()

with st.sidebar:
    section_title("Model info", icon="info")
    st.write("**Tip:** RoBERTa (transformer)")
    st.write("**Vir:** HuggingFace")
    st.write("**Model:** cardiffnlp/twitter-roberta-base-sentiment-latest")
    st.write("**Razredi:** negative, neutral, positive")
    divider()
    st.markdown("""
    **Kako uporabljati?**

    Vnesi komentar o letu v angleščini.
    Model napove eno od treh kategorij.
    """)

section_title("Vnesi komentar", icon="rate_review")

if "review_input" not in st.session_state:
    st.session_state.review_input = ""

def set_positive():
    st.session_state.review_input = "Amazing flight! The crew was friendly and we arrived 10 minutes early. Highly recommend!"

def set_neutral():
    st.session_state.review_input = "Average flight. Nothing special but no major issues either."

def set_negative():
    st.session_state.review_input = "Terrible experience! Flight delayed 3 hours, lost luggage, and rude staff."

st.markdown("**Predloge za testiranje:**")
col1, col2, col3 = st.columns(3)

with col1:
    st.button("Pozitiven primer", use_container_width=True, on_click=set_positive,
              icon=":material/sentiment_satisfied:")
with col2:
    st.button("Nevtralen primer", use_container_width=True, on_click=set_neutral,
              icon=":material/sentiment_neutral:")
with col3:
    st.button("Negativen primer", use_container_width=True, on_click=set_negative,
              icon=":material/sentiment_dissatisfied:")

review_text = st.text_area(
    "Tvoj komentar:",
    height=100,
    placeholder="Type your review here in English...",
    key="review_input",
)

if st.button("Analiziraj sentiment", type="primary", use_container_width=True, icon=":material/insights:"):
    if not review_text.strip():
        st.warning("Prosim vnesi komentar.")
    else:
        with st.spinner("Analiziram..."):
            result = analyzer.predict(review_text)

        # === LOG PREDICTION ===
        try:
            log_prediction(
                model_type="roberta_sentiment",
                prediction=result["label"],
                features={"text": review_text[:200]},  # samo 200 znakov
                extra={
                    "confidence": round(result["score"], 4),
                    "scores": {k: round(v, 4) for k, v in result["scores"].items()},
                },
            )
        except Exception as log_err:
            st.warning(f"Logiranje neuspešno: {log_err}")

        divider()
        section_title("Rezultat", icon="analytics")

        col1, col2 = st.columns([1, 2])

        with col1:
            label = result["label"]
            score = result["score"]
            color = SENTIMENT_COLORS.get(label, MUTED)
            icon_map = {
                "positive": "sentiment_satisfied",
                "neutral": "sentiment_neutral",
                "negative": "sentiment_dissatisfied",
            }
            st.markdown(
                f"""
                <div class="fi-card">
                  <div style="display:flex;align-items:center;gap:.5rem;">
                    <span class="material-symbols-outlined" style="color:{color};font-size:30px;">
                      {icon_map.get(label, 'help')}
                    </span>
                    <span style="font-size:1.3rem;font-weight:800;color:{color};">
                      {label.upper()}
                    </span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("Zaupanje", f"{score * 100:.1f}%")

        with col2:
            scores = result["scores"]
            bar_colors = [SENTIMENT_COLORS.get(k, MUTED) for k in scores.keys()]
            fig = go.Figure(data=[
                go.Bar(
                    x=list(scores.values()),
                    y=list(scores.keys()),
                    orientation="h",
                    marker=dict(color=bar_colors),
                    text=[f"{v * 100:.1f}%" for v in scores.values()],
                    textposition="outside",
                )
            ])
            fig.update_layout(
                title="Verjetnost po kategoriji",
                xaxis_title="Verjetnost",
                xaxis=dict(range=[0, 1.1]),
            )
            st.plotly_chart(theme_fig(fig, height=300, legend=False), use_container_width=True)

        st.caption("Napoved zabeležena v monitoring log.")

# Batch analiza
divider()
section_title("Predogled batch analize", icon="dataset")
st.markdown("Rezultati iz `data/reviews/predictions.csv` (predhodno analiziranih 30 komentarjev):")

predictions_path = Path(__file__).parent.parent.parent / "data" / "reviews" / "predictions.csv"
if predictions_path.exists():
    df = pd.read_csv(predictions_path)

    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = df["sentiment"].value_counts().to_dict()
        pie_colors = [SENTIMENT_COLORS.get(k, MUTED) for k in sentiment_counts.keys()]
        fig = go.Figure(data=[
            go.Pie(
                labels=list(sentiment_counts.keys()),
                values=list(sentiment_counts.values()),
                hole=0.55,
                marker=dict(colors=pie_colors, line=dict(color="#FFFFFF", width=2)),
            )
        ])
        fig.update_layout(title="Distribucija sentiment")
        st.plotly_chart(theme_fig(fig, height=350), use_container_width=True)

    with col2:
        summary_path = Path(__file__).parent.parent.parent / "reports" / "sentiment_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            fig = go.Figure(data=[
                go.Bar(
                    x=summary["sentiment_score"],
                    y=summary["airline"],
                    orientation="h",
                    marker=dict(
                        color=[SUCCESS if s > 0 else DANGER for s in summary["sentiment_score"]],
                    ),
                )
            ])
            fig.update_layout(
                title="Sentiment score po družbi",
                xaxis_title="Score (-1 do +1)",
            )
            st.plotly_chart(theme_fig(fig, height=350, legend=False), use_container_width=True)

    with st.expander("Vsi analizirani komentarji"):
        st.dataframe(
            df[["airline", "text", "sentiment", "confidence"]].sort_values("confidence", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Batch rezultati niso na voljo. Najprej poženi: `uv run dvc repro analyze_reviews`")