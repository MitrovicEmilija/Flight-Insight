import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from model_loader import (  # type: ignore
    load_xgboost_model,
    load_airports,
    load_airlines,
    load_model_metadata,
    format_airport,
)
from monitoring import log_prediction  # type: ignore
from styling import (  # type: ignore
    apply_base_styles,
    page_header,
    section_title,
    divider,
    status_pill,
)


st.set_page_config(
    page_title="Predict Delay", page_icon=":material/schedule:", layout="wide"
)
apply_base_styles()

page_header(
    "Napoved zamude leta",
    "Vnesi podatke o letu in sistem napove pričakovano zamudo (XGBoost regresor).",
    icon="schedule",
)

with st.spinner("Nalagam model..."):
    model = load_xgboost_model()
    metadata = load_model_metadata()

if model is None:
    st.stop()

with st.sidebar:
    section_title("Model info", icon="info")
    if metadata:
        st.write(f"**Tip:** {metadata.get('model_type', 'N/A')}")
        st.write(f"**Features:** {metadata.get('n_features', 'N/A')}")
        metrics = metadata.get("metrics", {})
        st.metric("Test MAE", f"{metrics.get('test_mae', 0):.2f} min")
        st.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.2f} min")
        st.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}")


# ===== SENTIMENT INTEGRATION =====
@st.cache_data
def load_airline_sentiment():
    summary_path = (
        Path(__file__).parent.parent.parent / "reports" / "sentiment_summary.csv"
    )
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    return df.set_index("airline").to_dict("index")


def get_sentiment_pill(sentiment_score):
    if sentiment_score is None:
        return "Brez podatkov", "gray", "Sentiment ni dostopen za to družbo."
    if sentiment_score >= 0.4:
        return (
            "Odlične ocene",
            "green",
            "Visoko priporočena družba — pozitivne izkušnje potnikov.",
        )
    elif sentiment_score >= 0.0:
        return "Solidne ocene", "green", "Solidno ocenjena družba."
    elif sentiment_score >= -0.3:
        return (
            "Mešane ocene",
            "amber",
            "Mešane ocene potnikov — razmisli o alternativah.",
        )
    else:
        return (
            "Slabe ocene",
            "red",
            "Negativne ocene — potniki pogosto poročajo o problemih.",
        )


sentiment_data = load_airline_sentiment()

airports = load_airports()  # zdaj dict {IATA: City}
airport_codes = list(airports.keys())
airlines = load_airlines()


def airport_label(code):
    return format_airport(code, airports)


section_title("Podatki o letu", icon="edit_note")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Datum in čas**")
    flight_date = st.date_input("Datum leta", value=pd.Timestamp("2025-06-15"))
    dep_time_str = st.time_input(
        "Načrtovan odhod", value=pd.Timestamp("2025-06-15 14:30").time()
    )

with col2:
    st.markdown("**Letalska družba**")
    airline_code = st.selectbox(
        "Družba",
        options=list(airlines.keys()),
        format_func=lambda x: f"{x} — {airlines[x]}",
        index=0,
    )
    st.markdown("**Letališča**")
    origin = st.selectbox(
        "Origin (izvor)",
        airport_codes,
        index=airport_codes.index("JFK") if "JFK" in airport_codes else 0,
        format_func=airport_label,
    )
    dest = st.selectbox(
        "Destination (cilj)",
        airport_codes,
        index=airport_codes.index("LAX") if "LAX" in airport_codes else 1,
        format_func=airport_label,
    )

with col3:
    st.markdown("**Karakteristike leta**")
    distance = st.number_input(
        "Razdalja (milje)", min_value=50, max_value=6000, value=2475, step=50
    )
    elapsed_time = st.number_input(
        "Načrtovano trajanje (min)", min_value=20, max_value=900, value=380, step=10
    )

divider()

if st.button(
    "Napovej zamudo",
    type="primary",
    use_container_width=True,
    icon=":material/insights:",
):
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
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "fall",
        10: "fall",
        11: "fall",
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
            st.warning(f"Logiranje napovedi neuspešno: {log_err}")

        divider()
        section_title("Rezultat napovedi", icon="analytics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Pričakovana zamuda", f"{prediction:.1f} min")

        with col2:
            if prediction < 5:
                pill = status_pill("Pravočasno", "green")
            elif prediction < 15:
                pill = status_pill("Manjša zamuda", "amber")
            elif prediction < 30:
                pill = status_pill("Srednja zamuda", "orange")
            else:
                pill = status_pill("Velika zamuda", "red")
            st.markdown(
                f'<div class="fi-eyebrow">Status</div>'
                f'<div style="margin-top:.5rem;">{pill}</div>',
                unsafe_allow_html=True,
            )

        with col3:
            if metadata:
                mae = metadata.get("metrics", {}).get("test_mae", 0)
                st.metric("Pričakovana napaka (±)", f"{mae:.1f} min")

        # === SENTIMENT INFO O LETALSKI DRUŽBI ===
        divider()
        section_title("Mnenja potnikov o tej družbi", icon="chat")
        st.markdown(
            f"Analiza mnenj potnikov za **{airlines[airline_code]}** "
            f"(RoBERTa transformer)."
        )

        sentiment_score = None
        if sentiment_data and airline_code in sentiment_data:
            airline_info = sentiment_data[airline_code]
            sentiment_score = airline_info.get("sentiment_score")
            n_positive = airline_info.get("n_positive", 0)
            n_negative = airline_info.get("n_negative", 0)
            n_neutral = airline_info.get("n_neutral", 0)
            total = n_positive + n_negative + n_neutral

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sentiment score", f"{sentiment_score:+.2f}")
            with col2:
                pct_pos = (n_positive / total * 100) if total > 0 else 0
                st.metric("Pozitivnih", f"{n_positive} ({pct_pos:.0f}%)")
            with col3:
                pct_neg = (n_negative / total * 100) if total > 0 else 0
                st.metric("Negativnih", f"{n_negative} ({pct_neg:.0f}%)")
            with col4:
                st.metric("Skupaj reviews", total)

            label, color, recommendation = get_sentiment_pill(sentiment_score)
            st.markdown(
                f'<div class="fi-eyebrow">Ocena potnikov</div>'
                f'<div style="margin-top:.5rem;">{status_pill(label, color)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(recommendation)

            # === KOMBINIRANO PRIPOROČILO ===
            divider()
            section_title("Skupna ocena leta", icon="recommend")

            delay_severity = (
                "ok" if prediction < 15 else ("medium" if prediction < 30 else "high")
            )
            sentiment_quality = "good" if sentiment_score >= 0 else "bad"

            if delay_severity == "ok" and sentiment_quality == "good":
                pill_label = "Odlična izbira"
                pill_color = "green"
                summary = "Pričakuje se pravočasen let in družba ima pozitivne ocene potnikov."
            elif delay_severity == "ok" and sentiment_quality == "bad":
                pill_label = "Sprejemljiv let"
                pill_color = "amber"
                summary = "Verjetno pravočasen, vendar družba ima slabše ocene potnikov. Razmisli o alternativnih ponudnikih."
            elif delay_severity == "medium" and sentiment_quality == "good":
                pill_label = "Solidna izbira"
                pill_color = "amber"
                summary = "Možna zamuda, vendar družba je sicer dobro ocenjena."
            elif delay_severity == "high" and sentiment_quality == "good":
                pill_label = "Pričakuj zamudo"
                pill_color = "orange"
                summary = (
                    "Velika napoved zamude, vendar družba je sicer pozitivno ocenjena."
                )
            else:
                pill_label = "Razmisli o alternativi"
                pill_color = "red"
                summary = "Pričakovana zamuda IN slabe ocene potnikov za to družbo."

            st.markdown(
                f'<div class="fi-eyebrow">Skupna ocena</div>'
                f'<div style="margin-top:.5rem;">{status_pill(pill_label, pill_color)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{summary}**")
            st.caption(
                "Ocena temelji na kombinaciji XGBoost napovedi zamude in RoBERTa sentiment "
                "analize zgodovinskih mnenj potnikov za to družbo."
            )

        else:
            st.info(
                f"Za družbo **{airlines[airline_code]}** še ni dovolj podatkov o mnenjih potnikov."
            )

        with st.expander("Podrobnosti napovedi"):
            st.write(f"**Let:** {airlines[airline_code]} ({airline_code})")
            origin_label = format_airport(origin, airports)
            dest_label = format_airport(dest, airports)
            st.write(f"**Ruta:** {origin_label} → {dest_label} ({distance} milj)")
            st.write(f"**Čas:** {flight_date} ob {dep_time_str}")
            st.write(f"**Del dneva:** {time_of_day}")
            st.write(f"**Sezona:** {season}")
            st.write(f"**Tip leta:** {distance_group}")
            st.write(f"**Vikend:** {'Da' if is_weekend else 'Ne'}")
            st.caption("Napoved zabeležena v monitoring log.")

        st.info(f"""
        **Kaj to pomeni?**

        Napovedana zamuda {prediction:.1f} min temelji na zgodovinskih podatkih
        ~7M letov iz 2024. Pričakovana napaka napovedi je ±{metadata.get("metrics", {}).get("test_mae", 21):.1f} min.
        """)

    except Exception as e:
        st.error(f"Napaka pri napovedi: {e}")
        st.exception(e)
