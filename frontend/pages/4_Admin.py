"""
Stran 4 — Admin nadzorna plošča (z Live Monitoring tab-om).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from monitoring import load_predictions_log, get_stats
from styling import (
    apply_base_styles,
    page_header,
    section_title,
    divider,
    theme_fig,
    SENTIMENT_COLORS,
    ACCENT,
    MUTED,
)


st.set_page_config(page_title="Admin", page_icon=":material/tune:", layout="wide")
apply_base_styles()

page_header(
    "Nadzorna plošča",
    "Monitoring, validacija in performance ML sistema.",
    icon="tune",
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
GX_DOCS_DIR = PROJECT_ROOT / "gx" / "uncommitted" / "data_docs" / "local_site"

# === SIDEBAR ===
with st.sidebar:
    section_title("System status", icon="health_and_safety")

    model_path = PROJECT_ROOT / "models" / "model.pkl"
    if model_path.exists():
        st.success("XGBoost model", icon=":material/check_circle:")
    else:
        st.error("XGBoost model", icon=":material/cancel:")

    data_path = PROJECT_ROOT / "data" / "preprocessed" / "flights.csv"
    if data_path.exists():
        st.success("Preprocessed data", icon=":material/check_circle:")
    else:
        st.error("Preprocessed data", icon=":material/cancel:")

    if (REPORTS_DIR / "drift_report.html").exists():
        st.success("Drift report", icon=":material/check_circle:")
    else:
        st.warning("Drift report manjka", icon=":material/warning:")

    if (REPORTS_DIR / "feature_importance.png").exists():
        st.success("Feature importance", icon=":material/check_circle:")
    else:
        st.warning("Feature importance manjka", icon=":material/warning:")

    if GX_DOCS_DIR.exists():
        st.success("GX docs", icon=":material/check_circle:")
    else:
        st.warning("GX docs manjka", icon=":material/warning:")

# === TABS ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    ":material/sensors: Live Monitoring",
    ":material/speed: Model performance",
    ":material/target: Feature importance",
    ":material/verified: Data validation (GX)",
    ":material/trending_down: Drift detection",
    ":material/experiment: MLflow tracking",
])


# ====================================================
# TAB 1 — LIVE MONITORING
# ====================================================
with tab1:
    section_title("Live Production Monitoring", icon="sensors")
    st.markdown("Real-time pregled napovedi v produkciji.")

    df_log = load_predictions_log()
    stats = get_stats(df_log)

    if stats["total_predictions"] == 0:
        st.info("Še nobenih napovedi v produkciji. Naredi prvo napoved v 'Predict Delay' ali 'Review Sentiment'.")
    else:
        # Top metrike
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Skupaj napovedi", stats["total_predictions"])
        with col2:
            st.metric("Zadnjih 24h", stats["last_24h"])
        with col3:
            st.metric("Zadnjih 7 dni", stats["last_7d"])
        with col4:
            n_models = len(stats["by_model"])
            st.metric("Aktivnih modelov", n_models)

        divider()

        # Po modelu
        col1, col2 = st.columns(2)

        with col1:
            section_title("Napovedi po modelu", icon="donut_small")
            if stats["by_model"]:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(stats["by_model"].keys()),
                        values=list(stats["by_model"].values()),
                        hole=0.55,
                        marker=dict(line=dict(color="#FFFFFF", width=2)),
                    )
                ])
                st.plotly_chart(theme_fig(fig, height=350), use_container_width=True)

        with col2:
            section_title("Časovni trend (zadnjih 7 dni)", icon="show_chart")
            df_7d = df_log[df_log["timestamp"] > pd.Timestamp.now() - pd.Timedelta(days=7)]
            if len(df_7d) > 0:
                daily = df_7d.groupby([df_7d["timestamp"].dt.date, "model_type"]).size().reset_index(name="count")
                daily.columns = ["date", "model_type", "count"]
                fig = px.line(
                    daily, x="date", y="count", color="model_type",
                    markers=True,
                    title="Napovedi na dan",
                )
                st.plotly_chart(theme_fig(fig, height=350), use_container_width=True)
            else:
                st.info("Ni dovolj podatkov za graf.")

        divider()

        # XGBoost specifične metrike
        df_xgb = df_log[df_log["model_type"] == "xgboost_delay"]
        if len(df_xgb) > 0:
            section_title("XGBoost Delay Predictions", icon="schedule")
            df_xgb_pred = pd.to_numeric(df_xgb["prediction"], errors="coerce").dropna()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Skupaj napovedi", len(df_xgb_pred))
            with col2:
                st.metric("Povprečna napoved", f"{df_xgb_pred.mean():.1f} min")
            with col3:
                st.metric("Median napoved", f"{df_xgb_pred.median():.1f} min")

            fig = px.histogram(
                df_xgb_pred, nbins=30,
                title="Distribucija napovedi zamud (min)",
                labels={"value": "Napoved zamude (min)"},
            )
            fig.update_traces(marker_color=ACCENT)
            st.plotly_chart(theme_fig(fig, height=300, legend=False), use_container_width=True)

        # RoBERTa specifične metrike
        df_rob = df_log[df_log["model_type"] == "roberta_sentiment"]
        if len(df_rob) > 0:
            divider()
            section_title("RoBERTa Sentiment Predictions", icon="reviews")

            col1, col2 = st.columns(2)
            with col1:
                sent_counts = df_rob["prediction"].value_counts().to_dict()
                pie_colors = [SENTIMENT_COLORS.get(k, MUTED) for k in sent_counts.keys()]
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(sent_counts.keys()),
                        values=list(sent_counts.values()),
                        hole=0.55,
                        marker=dict(colors=pie_colors, line=dict(color="#FFFFFF", width=2)),
                    )
                ])
                fig.update_layout(title="Distribucija sentiment napovedi")
                st.plotly_chart(theme_fig(fig, height=350), use_container_width=True)

            with col2:
                st.metric("Skupaj sentiment napovedi", len(df_rob))
                st.write("**Distribucija:**")
                for label, count in sent_counts.items():
                    pct = count / len(df_rob) * 100
                    st.write(f"- {label}: {count} ({pct:.1f}%)")

        # Recent predictions
        divider()
        section_title("Zadnjih 20 napovedi", icon="history")
        df_recent = df_log.sort_values("timestamp", ascending=False).head(20)
        st.dataframe(
            df_recent[["timestamp", "model_type", "prediction"]],
            use_container_width=True,
            hide_index=True,
        )

        # Download log
        divider()
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Prenesi cel log (CSV)",
            csv,
            "predictions_log.csv",
            "text/csv",
            icon=":material/download:",
        )


# ====================================================
# TAB 2 — MODEL PERFORMANCE
# ====================================================
with tab2:
    section_title("Model Performance", icon="speed")

    import yaml
    metadata_path = PROJECT_ROOT / "models" / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**XGBoost Regressor**")
            st.write(f"**Model type:** {metadata.get('model_type', 'N/A')}")
            st.write(f"**Št. features:** {metadata.get('n_features', 'N/A')}")
            st.markdown("**Naloga:** Napoved zamude v minutah")

        with col2:
            metrics = metadata.get("metrics", {})
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Test MAE", f"{metrics.get('test_mae', 0):.2f} min")
            with col_b:
                st.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f} min")
            with col_c:
                st.metric("Test MSE", f"{metrics.get('test_mse', 0):.2f}")
            with col_d:
                st.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}")

            st.info("""
            **Interpretacija:**
            - **MAE 21 min** — povprečna napaka napovedi
            - **RMSE 53 min** — občutljiv na ekstreme
            - **R² 0.045** — inherentna stohastičnost letalskih zamud
            """)
    else:
        st.warning("Metadata ni najden.")

    divider()
    section_title("RoBERTa Sentiment Classifier", icon="reviews")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Vir:** HuggingFace")
        st.write("**Model:** cardiffnlp/twitter-roberta-base-sentiment-latest")
        st.markdown("**Naloga:** Klasifikacija sentiment (3 razredi)")
    with col2:
        st.info("""
        **Pretreniran model**: ~125M tweetov, fine-tuned na sentiment.
        Uporabljamo ga za analizo flight reviews.
        """)


# ====================================================
# TAB 3 — FEATURE IMPORTANCE
# ====================================================
with tab3:
    section_title("Feature Importance", icon="target")
    st.markdown("Kateri features najmočneje vplivajo na napoved.")

    fi_path = REPORTS_DIR / "feature_importance.png"
    fi_comp_path = REPORTS_DIR / "feature_importance_comparison.png"
    fi_csv_path = REPORTS_DIR / "feature_importance.csv"

    if fi_path.exists():
        st.image(str(fi_path), caption="Top 20 features po gain")

    if fi_csv_path.exists():
        df_fi = pd.read_csv(fi_csv_path)
        divider()
        section_title("Top 15 features", icon="table_rows")
        st.dataframe(df_fi.head(15), use_container_width=True, hide_index=True)

    if fi_comp_path.exists():
        divider()
        section_title("Primerjava metrik", icon="bar_chart")
        st.image(str(fi_comp_path))
        st.caption("Gain | Weight | Cover")


# ====================================================
# TAB 4 — GX
# ====================================================
with tab4:
    section_title("Data Validation — Great Expectations", icon="verified")
    gx_index = GX_DOCS_DIR / "index.html"

    if gx_index.exists():
        st.success("GX poročilo naloženo", icon=":material/check_circle:")
        with open(gx_index, "r") as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    else:
        st.warning("GX docs niso najdeni.")


# ====================================================
# TAB 5 — DRIFT
# ====================================================
with tab5:
    section_title("Drift Detection — Evidently", icon="trending_down")
    st.markdown("Primerjava jan 2024 vs jan 2025.")

    drift_path = REPORTS_DIR / "drift_report.html"
    if drift_path.exists():
        st.success("Drift poročilo naloženo", icon=":material/check_circle:")
        with open(drift_path, "r") as f:
            html_content = f.read()
        components.html(html_content, height=900, scrolling=True)
    else:
        st.warning("Drift report ni najden.")


# ====================================================
# TAB 6 — MLFLOW
# ====================================================
with tab6:
    section_title("MLflow Experiment Tracking", icon="experiment")
    mlflow_url = "https://dagshub.com/MitrovicEmilija/Flight-Insight.mlflow"

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**MLflow tracking URI:** `{mlflow_url}`")
        st.link_button("Odpri MLflow na DagsHub", mlflow_url, type="primary",
                       icon=":material/open_in_new:")
    with col2:
        st.info("""
        **Kaj se sledi:**
        - Parametri treninga
        - Test metrike
        - Artefakti
        - Run history
        """)