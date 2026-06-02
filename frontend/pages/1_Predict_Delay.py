"""
Stran 1 — Napoved zamude leta z XGBoost modelom.
Z dodanim production monitoring (logiranje napovedi).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from model_loader import (
    load_xgboost_model,
    load_airports,
    load_airlines,
    load_model_metadata,
)
from monitoring import log_prediction


st.set_page_config(page_title="Predict Delay", page_icon="🛫", layout="wide")

st.title("🛫 Napoved zamude leta")
st.markdown("Vnesi podatke o letu in sistem ti napove pričakovano zamudo (XGBoost regresor).")

with st.spinner("Nalagam model..."):
    model = load_xgboost_model()
    metadata = load_model_metadata()

if model is None:
    st.stop()

with st.sidebar:
    st.subheader("ℹ️ Model info")
    if metadata:
        st.write(f"**Tip:** {metadata.get('model_type', 'N/A')}")
        st.write(f"**Features:** {metadata.get('n_features', 'N/A')}")
        metrics = metadata.get("metrics", {})
        st.metric("Test MAE", f"{metrics.get('test_mae', 0):.2f} min")
        st.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f} min")
        st.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}")

airports = load_airports()
airlines = load_airlines()

st.markdown("---")
st.subheader("📋 Podatki o letu")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Datum in čas**")
    flight_date = st.date_input("Datum leta", value=pd.Timestamp("2025-06-15"))
    dep_time_str = st.time_input("Načrtovan odhod", value=pd.Timestamp("2025-06-15 14:30").time())

with col2:
    st.markdown("**Letalska družba**")
    airline_code = st.selectbox(
        "Družba",
        options=list(airlines.keys()),
        format_func=lambda x: f"{x} — {airlines[x]}",
        index=0,
    )
    st.markdown("**Letališča**")
    origin = st.selectbox("Origin (izvor)", airports, index=airports.index("JFK") if "JFK" in airports else 0)
    dest = st.selectbox("Destination (cilj)", airports, index=airports.index("LAX") if "LAX" in airports else 1)

with col3:
    st.markdown("**Karakteristike leta**")
    distance = st.number_input("Razdalja (milje)", min_value=50, max_value=6000, value=2475, step=50)
    elapsed_time = st.number_input("Načrtovano trajanje (min)", min_value=20, max_value=900, value=380, step=10)

st.markdown("---")

if st.button("🔮 Napovej zamudo", type="primary", use_container_width=True):
    month = flight_date.month
    day_of_month = flight_date.day
    day_of_week = flight_date.isoweekday()
    crs_dep_time = dep_time_str.hour * 100 + dep_time_str.minute
    dep_hour = dep_time_str.hour
    is_weekend = 1 if day_of_week in [6, 7] else 0

    if 5 <= dep_hour <= 11:
        time_of_day = "morning"
    elif 12 <= dep_hour <= 16:
        time_of_day = "afternoon"
    elif 17 <= dep_hour <= 20:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    season_map = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "fall", 10: "fall", 11: "fall",
    }
    season = season_map[month]

    if distance < 500:
        distance_group = "short"
    elif distance < 1000:
        distance_group = "medium"
    elif distance < 2000:
        distance_group = "long"
    else:
        distance_group = "very_long"

    features = {
        "Month": month,
        "DayofMonth": day_of_month,
        "DayOfWeek": day_of_week,
        "Distance": distance,
        "CRSElapsedTime": elapsed_time,
        "dep_hour": dep_hour,
        "is_weekend": is_weekend,
        "Marketing_Airline_Network": airline_code,
        "time_of_day": time_of_day,
        "season": season,
        "distance_group": distance_group,
        "Origin": origin,
        "Dest": dest,
    }

    input_df = pd.DataFrame([features])

    try:
        with st.spinner("Računam napoved..."):
            prediction = float(model.predict(input_df)[0])

        # === LOG PREDICTION (production monitoring) ===
        try:
            log_prediction(
                model_type="xgboost_delay",
                prediction=round(prediction, 2),
                features=features,
                extra={
                    "airline_name": airlines[airline_code],
                    "flight_date": str(flight_date),
                    "dep_time": str(dep_time_str),
                },
            )
        except Exception as log_err:
            st.warning(f"⚠️ Logiranje napovedi neuspešno: {log_err}")

        # Display
        st.markdown("---")
        st.subheader("📊 Rezultat napovedi")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Pričakovana zamuda", f"{prediction:.1f} min")

        with col2:
            if prediction < 5:
                status = "✅ Pravočasno"
                status_color = "green"
            elif prediction < 15:
                status = "🟡 Manjša zamuda"
                status_color = "orange"
            elif prediction < 30:
                status = "🟠 Srednja zamuda"
                status_color = "orange"
            else:
                status = "🔴 Velika zamuda"
                status_color = "red"
            st.markdown(f"**Status:** :{status_color}[{status}]")

        with col3:
            if metadata:
                mae = metadata.get("metrics", {}).get("test_mae", 0)
                st.metric("Pričakovana napaka (±)", f"{mae:.1f} min")

        with st.expander("🔍 Podrobnosti napovedi"):
            st.write(f"**Let:** {airlines[airline_code]} ({airline_code})")
            st.write(f"**Ruta:** {origin} → {dest} ({distance} milj)")
            st.write(f"**Čas:** {flight_date} ob {dep_time_str}")
            st.write(f"**Del dneva:** {time_of_day}")
            st.write(f"**Sezona:** {season}")
            st.write(f"**Tip leta:** {distance_group}")
            st.write(f"**Vikend:** {'Da' if is_weekend else 'Ne'}")
            st.success("📝 Napoved zabeležena v monitoring log.")

        st.info(f"""
        💡 **Kaj to pomeni?**

        Napovedana zamuda {prediction:.1f} min temelji na zgodovinskih podatkih
        ~7M letov iz 2024. Pričakovana napaka napovedi je ±{metadata.get('metrics', {}).get('test_mae', 21):.1f} min.
        """)

    except Exception as e:
        st.error(f"❌ Napaka pri napovedi: {e}")
        st.exception(e)