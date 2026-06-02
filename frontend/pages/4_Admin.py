"""
Stran 4 — Admin nadzorna plošča.

Vključuje:
  - Monitoring metrike (live napovedi)
  - Great Expectations validacijo (HTML poročilo)
  - Evidently drift detection (HTML poročilo)
  - Feature importance grafe
  - MLflow link
  - Model info & metrike
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

st.set_page_config(page_title="Admin", page_icon="🔧", layout="wide")

st.title("🔧 Admin nadzorna plošča")
st.markdown("Monitoring, validacija in performance ML sistema.")

PROJECT_ROOT = Path(__file__).parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
GX_DOCS_DIR = PROJECT_ROOT / "gx" / "uncommitted" / "data_docs" / "local_site"

# === SIDEBAR — System status ===
with st.sidebar:
    st.subheader("🔋 System status")

    # Check models
    model_path = PROJECT_ROOT / "models" / "model.pkl"
    if model_path.exists():
        st.success("✅ XGBoost model")
    else:
        st.error("❌ XGBoost model")

    # Check data
    data_path = PROJECT_ROOT / "data" / "preprocessed" / "flights.csv"
    if data_path.exists():
        st.success("✅ Preprocessed data")
    else:
        st.error("❌ Preprocessed data")

    # Check reports
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


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model performance",
    "🎯 Feature importance",
    "✅ Data validation (GX)",
    "📉 Drift detection (Evidently)",
    "🔗 MLflow tracking",
])


with tab1:
    st.header("Model Performance")

    # Naloži metadata
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
            st.markdown("**Naloga:** Napoved zamude v minutah (regresija)")

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
            **Interpretacija metrik:**
            - **MAE 21 min** = povprečna napaka napovedi  
            - **RMSE 53 min** = občutljiv na ekstreme  
            - **R² 0.045** = inherentna stohastičnost letalskih zamud
            """)
    else:
        st.warning("Metadata ni najden. Poženi training.")

    st.markdown("---")

    # HuggingFace model info
    st.subheader("RoBERTa Sentiment Classifier")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Vir:** HuggingFace")
        st.write("**Model:** cardiffnlp/twitter-roberta-base-sentiment-latest")
        st.markdown("**Naloga:** Klasifikacija sentiment (3 razredi)")
    with col2:
        st.info("""
        **Pretreniran model**: ~125M tweetov, fine-tuned na sentiment.
        Uporabljamo ga za analizo flight reviews uporabnikov.
        Ne fine-tuninjamo (uporabljamo as-is).
        """)


with tab2:
    st.header("Feature Importance")
    st.markdown("Kateri features najmočneje vplivajo na napoved zamude (XGBoost gain).")

    fi_path = REPORTS_DIR / "feature_importance.png"
    fi_comp_path = REPORTS_DIR / "feature_importance_comparison.png"
    fi_csv_path = REPORTS_DIR / "feature_importance.csv"

    if fi_path.exists():
        st.image(str(fi_path), caption="Top 20 features po gain")
    else:
        st.warning("Feature importance graf ni najden.")

    if fi_csv_path.exists():
        df_fi = pd.read_csv(fi_csv_path)
        st.markdown("---")
        st.subheader("📋 Top 15 features (tabela)")
        st.dataframe(
            df_fi.head(15),
            use_container_width=True,
            hide_index=True,
        )

    if fi_comp_path.exists():
        st.markdown("---")
        st.subheader("📊 Primerjava metrik (gain vs weight vs cover)")
        st.image(str(fi_comp_path))
        st.caption("Gain = informacijski dobiček | Weight = pogostost uporabe | Cover = obseg vpliva")


with tab3:
    st.header("Data Validation — Great Expectations")
    st.markdown("Avtomatska validacija podatkov pred treningom.")

    # Poskusi najti GX index.html
    gx_index = GX_DOCS_DIR / "index.html"

    if gx_index.exists():
        st.success(f"✅ GX poročilo najdeno: {gx_index.relative_to(PROJECT_ROOT)}")

        # Embed HTML
        with open(gx_index, "r") as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)

        with st.expander("📂 Lokacija poročila"):
            st.code(str(gx_index))
            st.info("Odpri direktno v brskalniku za polno funkcionalnost.")
    else:
        st.warning(f"GX docs niso najdeni v {GX_DOCS_DIR}")
        st.info("Poženi: `cd gx && uv run python run_checkpoint.py`")


with tab4:
    st.header("Drift Detection — Evidently")
    st.markdown("Apples-to-apples primerjava: jan 2024 (reference) vs jan 2025 (current)")

    drift_path = REPORTS_DIR / "drift_report.html"

    if drift_path.exists():
        st.success(f"✅ Drift poročilo: {drift_path.relative_to(PROJECT_ROOT)}")

        st.info("""
        💡 **Strategija**: Apples-to-apples primerjava ISTEGA MESECA, RAZLIČNIH LET.
        To izolira sezonalnost in zazna SAMO resničen drift.
        """)

        # Embed HTML
        with open(drift_path, "r") as f:
            html_content = f.read()
        components.html(html_content, height=900, scrolling=True)
    else:
        st.warning("Drift report ni najden.")
        st.info("Poženi: `uv run python src/data/test_data.py`")


with tab5:
    st.header("MLflow Experiment Tracking")
    st.markdown("Vpogled v sledenje eksperimentov.")

    mlflow_url = "https://dagshub.com/MitrovicEmilija/Flight-Insight.mlflow"

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        **MLflow tracking URI:**  
        `{mlflow_url}`
        """)
        st.link_button("🔗 Odpri MLflow na DagsHub", mlflow_url, type="primary")

    with col2:
        st.info("""
        **Kaj se sledi:**
        - Parametri treninga
        - Test metrike (MAE, RMSE, R²)
        - Artefakti (model.pkl, ONNX, grafi)
        - Run history
        """)

    st.markdown("---")
    st.subheader("Sledenje eksperimentov")

    st.markdown("""
    Vsak trening run je zabeležen v MLflow z:

    - **Parametri**: n_estimators, max_depth, learning_rate, ...
    - **Metrike**: test_mae, test_rmse, test_r2
    - **Artefakti**: model.pkl, model.onnx, preprocessor.pkl, feature_importance.png/csv

    **DagsHub MLflow UI** omogoča:
    - Primerjavo run-ov
    - Filtriranje po metrikah
    - Download artefaktov
    - Reproducibilnost
    """)