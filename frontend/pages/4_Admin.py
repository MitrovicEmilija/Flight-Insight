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


st.set_page_config(page_title="Admin", page_icon="🔧", layout="wide")

st.title("🔧 Admin nadzorna plošča")
st.markdown("Monitoring, validacija in performance ML sistema.")

PROJECT_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
GX_DOCS_DIR = PROJECT_ROOT / "gx" / "uncommitted" / "data_docs" / "local_site"

# === SIDEBAR ===
with st.sidebar:
    st.subheader("🔋 System status")

    model_path = PROJECT_ROOT / "models" / "model.pkl"
    if model_path.exists():
        st.success("✅ XGBoost model")
    else:
        st.error("❌ XGBoost model")

    data_path = PROJECT_ROOT / "data" / "preprocessed" / "flights.csv"
    if data_path.exists():
        st.success("✅ Preprocessed data")
    else:
        st.error("❌ Preprocessed data")

    if (REPORTS_DIR / "drift_report.html").exists():
        st.success("✅ Drift report")
    else:
        st.warning("⚠️ Drift report manjka")

    if (REPORTS_DIR / "feature_importance.png").exists():
        st.success("✅ Feature importance")
    else:
        st.warning("⚠️ Feature importance manjka")

    if GX_DOCS_DIR.exists():
        st.success("✅ GX docs")
    else:
        st.warning("⚠️ GX docs manjka")

# === TABS ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Live Monitoring",
    "📊 Model performance",
    "🎯 Feature importance",
    "✅ Data validation (GX)",
    "📉 Drift detection (Evidently)",
    "🔗 MLflow tracking",
])


# ====================================================
# TAB 1 — LIVE MONITORING (NEW!)
# ====================================================
with tab1:
    st.header("📡 Live Production Monitoring")
    st.markdown("Real-time pregled napovedi v produkciji.")

    df_log = load_predictions_log()
    stats = get_stats(df_log)

    if stats["total_predictions"] == 0:
        st.info("📭 Še nobenih napovedi v produkciji. Naredi prvo napoved v 'Predict Delay' ali 'Review Sentiment'!")
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

        st.markdown("---")

        # Po modelu
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Napovedi po modelu")
            if stats["by_model"]:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(stats["by_model"].keys()),
                        values=list(stats["by_model"].values()),
                        hole=0.4,
                    )
                ])
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Časovni trend (zadnjih 7 dni)")
            # Group po dnevu
            df_7d = df_log[df_log["timestamp"] > pd.Timestamp.now() - pd.Timedelta(days=7)]
            if len(df_7d) > 0:
                daily = df_7d.groupby([df_7d["timestamp"].dt.date, "model_type"]).size().reset_index(name="count")
                daily.columns = ["date", "model_type", "count"]
                fig = px.line(
                    daily, x="date", y="count", color="model_type",
                    markers=True,
                    title="Napovedi na dan",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Ni dovolj podatkov za graf.")

        st.markdown("---")

        # XGBoost specifične metrike
        df_xgb = df_log[df_log["model_type"] == "xgboost_delay"]
        if len(df_xgb) > 0:
            st.subheader("🛫 XGBoost Delay Predictions")
            df_xgb_pred = pd.to_numeric(df_xgb["prediction"], errors="coerce").dropna()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Skupaj napovedi", len(df_xgb_pred))
            with col2:
                st.metric("Povprečna napoved", f"{df_xgb_pred.mean():.1f} min")
            with col3:
                st.metric("Median napoved", f"{df_xgb_pred.median():.1f} min")

            # Histogram napovedi
            fig = px.histogram(
                df_xgb_pred, nbins=30,
                title="Distribucija napovedi zamud (min)",
                labels={"value": "Napoved zamude (min)"},
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # RoBERTa specifične metrike
        df_rob = df_log[df_log["model_type"] == "roberta_sentiment"]
        if len(df_rob) > 0:
            st.markdown("---")
            st.subheader("💬 RoBERTa Sentiment Predictions")

            col1, col2 = st.columns(2)
            with col1:
                sent_counts = df_rob["prediction"].value_counts().to_dict()
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(sent_counts.keys()),
                        values=list(sent_counts.values()),
                        hole=0.4,
                        marker=dict(colors=["#dc3545", "#6c757d", "#28a745"]),
                    )
                ])
                fig.update_layout(title="Distribucija sentiment napovedi", height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Skupaj sentiment napovedi", len(df_rob))
                st.write("**Distribucija:**")
                for label, count in sent_counts.items():
                    pct = count / len(df_rob) * 100
                    st.write(f"- {label}: {count} ({pct:.1f}%)")

        # Recent predictions
        st.markdown("---")
        st.subheader("📋 Zadnjih 20 napovedi")
        df_recent = df_log.sort_values("timestamp", ascending=False).head(20)
        st.dataframe(
            df_recent[["timestamp", "model_type", "prediction"]],
            use_container_width=True,
            hide_index=True,
        )

        # Download log
        st.markdown("---")
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Prenesi cel log (CSV)",
            csv,
            "predictions_log.csv",
            "text/csv",
        )


# ====================================================
# TAB 2 — MODEL PERFORMANCE
# ====================================================
with tab2:
    st.header("Model Performance")

    import yaml
    metadata_path = PROJECT_ROOT / "models" / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("XGBoost Regressor")
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
            - **MAE 21 min** = povprečna napaka napovedi  
            - **RMSE 53 min** = občutljiv na ekstreme  
            - **R² 0.045** = inherentna stohastičnost letalskih zamud
            """)
    else:
        st.warning("Metadata ni najden.")

    st.markdown("---")
    st.subheader("RoBERTa Sentiment Classifier")
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
    st.header("Feature Importance")
    st.markdown("Kateri features najmočneje vplivajo na napoved.")

    fi_path = REPORTS_DIR / "feature_importance.png"
    fi_comp_path = REPORTS_DIR / "feature_importance_comparison.png"
    fi_csv_path = REPORTS_DIR / "feature_importance.csv"

    if fi_path.exists():
        st.image(str(fi_path), caption="Top 20 features po gain")

    if fi_csv_path.exists():
        df_fi = pd.read_csv(fi_csv_path)
        st.markdown("---")
        st.subheader("📋 Top 15 features (tabela)")
        st.dataframe(df_fi.head(15), use_container_width=True, hide_index=True)

    if fi_comp_path.exists():
        st.markdown("---")
        st.subheader("📊 Primerjava metrik")
        st.image(str(fi_comp_path))
        st.caption("Gain | Weight | Cover")


# ====================================================
# TAB 4 — GX
# ====================================================
with tab4:
    st.header("Data Validation — Great Expectations")
    gx_index = GX_DOCS_DIR / "index.html"

    if gx_index.exists():
        st.success(f"✅ GX poročilo")
        with open(gx_index, "r") as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    else:
        st.warning("GX docs niso najdeni.")


# ====================================================
# TAB 5 — DRIFT
# ====================================================
with tab5:
    st.header("Drift Detection — Evidently")
    st.markdown("Apples-to-apples primerjava jan 2024 vs jan 2025.")

    drift_path = REPORTS_DIR / "drift_report.html"
    if drift_path.exists():
        st.success("✅ Drift poročilo")
        with open(drift_path, "r") as f:
            html_content = f.read()
        components.html(html_content, height=900, scrolling=True)
    else:
        st.warning("Drift report ni najden.")


# ====================================================
# TAB 6 — MLFLOW
# ====================================================
with tab6:
    st.header("MLflow Experiment Tracking")
    mlflow_url = "https://dagshub.com/MitrovicEmilija/Flight-Insight.mlflow"

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**MLflow tracking URI:** `{mlflow_url}`")
        st.link_button("🔗 Odpri MLflow na DagsHub", mlflow_url, type="primary")
    with col2:
        st.info("""
        **Kaj se sledi:**
        - Parametri treninga
        - Test metrike
        - Artefakti
        - Run history
        """)