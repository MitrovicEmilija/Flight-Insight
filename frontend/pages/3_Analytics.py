"""
Stran 3 — Analytics z zgodovinskimi trendi.

V Dockerju uporablja flights_sample.csv (500K vrstic).
Lokalno uporablja flights.csv (~7M vrstic).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from styling import (
    apply_base_styles,
    page_header,
    section_title,
    divider,
    theme_fig,
    SEQ_BLUE,
    SEQ_DELAY,
    ACCENT,
)


st.set_page_config(page_title="Analytics", page_icon=":material/monitoring:", layout="wide")
apply_base_styles()

page_header(
    "Analytics — zgodovinski trendi",
    "Analitika zamud na podlagi ~7M letov iz BTS podatkov za 2024.",
    icon="monitoring",
)


PROJECT_ROOT = Path(__file__).parent.parent.parent
FULL_DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "flights.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "preprocessed" / "flights_sample.csv"


@st.cache_data
def load_data():
    """Naloži podatke. Preferira polni dataset, fallback na sample."""
    if FULL_DATA_PATH.exists():
        data_path = FULL_DATA_PATH
        is_sample = False
    elif SAMPLE_DATA_PATH.exists():
        data_path = SAMPLE_DATA_PATH
        is_sample = True
    else:
        return None, None

    cols = [
        "Year", "Month", "DayOfWeek",
        "Marketing_Airline_Network", "Origin", "Dest",
        "Distance", "DepDelayMinutes",
        "dep_hour", "time_of_day", "season", "is_weekend",
        "CarrierDelay", "WeatherDelay", "NASDelay",
        "SecurityDelay", "LateAircraftDelay",
    ]

    df = pd.read_csv(data_path, usecols=cols, low_memory=False)
    if len(df) > 500_000:
        df = df.sample(n=500_000, random_state=42).reset_index(drop=True)
    return df, is_sample


with st.spinner("Nalagam podatke..."):
    df, is_sample = load_data()

if df is None:
    st.error("Podatki niso najdeni.")
    st.info("Poženi `uv run dvc repro` da generiraš podatke ali `uv run python scripts/create_sample.py` za vzorec.")
    st.stop()

if is_sample:
    st.info(f"Uporabljam vzorec ({len(df):,} vrstic). Za polno analitiko poženi pipeline lokalno.")

# Filtri
with st.sidebar:
    section_title("Filtri", icon="filter_alt")
    year_filter = st.selectbox(
        "Leto",
        options=sorted(df["Year"].unique()),
        index=0,
    )
    df_filtered = df[df["Year"] == year_filter]
    st.caption(f"Filtriranih {len(df_filtered):,} letov")

# Splošne metrike
section_title("Splošne metrike", icon="query_stats")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Skupaj letov (vzorec)", f"{len(df_filtered):,}")
with col2:
    avg_delay = df_filtered["DepDelayMinutes"].mean()
    st.metric("Povprečna zamuda", f"{avg_delay:.1f} min")
with col3:
    on_time_pct = (df_filtered["DepDelayMinutes"] < 15).mean() * 100
    st.metric("% pravočasnih (<15 min)", f"{on_time_pct:.1f}%")
with col4:
    major_delay_pct = (df_filtered["DepDelayMinutes"] > 60).mean() * 100
    st.metric("% velikih zamud (>60 min)", f"{major_delay_pct:.1f}%")

# AIRLINE PERFORMANCE
divider()
section_title("Zanesljivost po letalskih družbah", icon="flight")

airline_stats = df_filtered.groupby("Marketing_Airline_Network").agg(
    avg_delay=("DepDelayMinutes", "mean"),
    median_delay=("DepDelayMinutes", "median"),
    n_flights=("DepDelayMinutes", "count"),
    on_time_pct=("DepDelayMinutes", lambda x: (x < 15).mean() * 100),
).reset_index()
airline_stats = airline_stats.sort_values("avg_delay")

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        airline_stats,
        x="Marketing_Airline_Network",
        y="avg_delay",
        color="avg_delay",
        color_continuous_scale=SEQ_DELAY,
        title="Povprečna zamuda po družbi",
        labels={"avg_delay": "Povprečna zamuda (min)", "Marketing_Airline_Network": "Letalska družba"},
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(theme_fig(fig, height=400, legend=False), use_container_width=True)

with col2:
    fig = px.bar(
        airline_stats.sort_values("on_time_pct", ascending=False),
        x="Marketing_Airline_Network",
        y="on_time_pct",
        color="on_time_pct",
        color_continuous_scale=SEQ_BLUE,
        title="% pravočasnih letov po družbi",
        labels={"on_time_pct": "% pravočasnih", "Marketing_Airline_Network": "Letalska družba"},
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(theme_fig(fig, height=400, legend=False), use_container_width=True)

# TIME PATTERNS
divider()
section_title("Vzorci po času", icon="schedule")

col1, col2 = st.columns(2)

with col1:
    hour_stats = df_filtered.groupby("dep_hour")["DepDelayMinutes"].mean().reset_index()
    fig = px.line(
        hour_stats, x="dep_hour", y="DepDelayMinutes", markers=True,
        title="Povprečna zamuda po uri odhoda",
        labels={"DepDelayMinutes": "Povp. zamuda (min)", "dep_hour": "Ura odhoda"},
    )
    fig.update_traces(line=dict(color=ACCENT, width=2.5), marker=dict(color=ACCENT, size=7))
    st.plotly_chart(theme_fig(fig, height=350, legend=False), use_container_width=True)

with col2:
    day_map = {1: "Pon", 2: "Tor", 3: "Sre", 4: "Čet", 5: "Pet", 6: "Sob", 7: "Ned"}
    day_stats = df_filtered.groupby("DayOfWeek")["DepDelayMinutes"].mean().reset_index()
    day_stats["Day"] = day_stats["DayOfWeek"].map(day_map)
    fig = px.bar(
        day_stats, x="Day", y="DepDelayMinutes",
        title="Povprečna zamuda po dnevu v tednu",
        labels={"DepDelayMinutes": "Povp. zamuda (min)"},
        color="DepDelayMinutes", color_continuous_scale=SEQ_DELAY,
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(theme_fig(fig, height=350, legend=False), use_container_width=True)

# SEASONAL
divider()
section_title("Sezonski vzorci", icon="calendar_month")

col1, col2 = st.columns(2)

with col1:
    month_stats = df_filtered.groupby("Month")["DepDelayMinutes"].mean().reset_index()
    fig = px.line(
        month_stats, x="Month", y="DepDelayMinutes", markers=True,
        title="Povprečna zamuda po mesecu",
        labels={"DepDelayMinutes": "Povp. zamuda (min)"},
    )
    fig.update_traces(line=dict(color=ACCENT, width=2.5), marker=dict(color=ACCENT, size=7))
    st.plotly_chart(theme_fig(fig, height=350, legend=False), use_container_width=True)

with col2:
    season_stats = df_filtered.groupby("season")["DepDelayMinutes"].mean().reset_index()
    season_order = ["winter", "spring", "summer", "fall"]
    season_stats["season"] = pd.Categorical(season_stats["season"], categories=season_order, ordered=True)
    season_stats = season_stats.sort_values("season")

    fig = px.bar(
        season_stats, x="season", y="DepDelayMinutes",
        title="Povprečna zamuda po sezoni",
        labels={"DepDelayMinutes": "Povp. zamuda (min)", "season": "Sezona"},
        color="DepDelayMinutes", color_continuous_scale=SEQ_DELAY,
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(theme_fig(fig, height=350, legend=False), use_container_width=True)

# DELAY CAUSES
divider()
section_title("Vzroki zamud", icon="warning")

delay_cols = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
cause_totals = df_filtered[delay_cols].sum().sort_values(ascending=False)

col1, col2 = st.columns(2)

with col1:
    fig = px.pie(
        values=cause_totals.values, names=cause_totals.index,
        title="Distribucija vzrokov zamud (skupne minute)", hole=0.55,
    )
    fig.update_traces(marker=dict(line=dict(color="#FFFFFF", width=2)))
    st.plotly_chart(theme_fig(fig, height=400), use_container_width=True)

with col2:
    st.markdown("**Razlaga vzrokov:**")
    st.markdown("""
    - **LateAircraftDelay** — kaskadne zamude
    - **CarrierDelay** — zamuda letalske družbe
    - **NASDelay** — nacionalni zračni sistem
    - **WeatherDelay** — zamude zaradi vremena
    - **SecurityDelay** — varnostni razlogi
    """)

# TOP ROUTES
divider()
section_title("Top rute", icon="route")

col1, col2 = st.columns(2)

with col1:
    route_counts = df_filtered.groupby(["Origin", "Dest"]).size().reset_index(name="n")
    route_counts["route"] = route_counts["Origin"] + " → " + route_counts["Dest"]
    top10 = route_counts.nlargest(10, "n")

    fig = px.bar(
        top10[::-1], x="n", y="route", orientation="h",
        title="Top 10 najpogostejših rut",
        labels={"n": "Št. letov", "route": "Ruta"},
    )
    fig.update_traces(marker_color=ACCENT)
    st.plotly_chart(theme_fig(fig, height=400, legend=False), use_container_width=True)

with col2:
    route_delays = df_filtered.groupby(["Origin", "Dest"]).agg(
        avg_delay=("DepDelayMinutes", "mean"),
        n=("DepDelayMinutes", "count"),
    ).reset_index()
    route_delays = route_delays[route_delays["n"] >= 500]
    route_delays["route"] = route_delays["Origin"] + " → " + route_delays["Dest"]
    worst10 = route_delays.nlargest(10, "avg_delay")

    fig = px.bar(
        worst10[::-1], x="avg_delay", y="route", orientation="h",
        title="Top 10 najslabših rut (min 500 letov)",
        labels={"avg_delay": "Povp. zamuda (min)", "route": "Ruta"},
        color="avg_delay", color_continuous_scale=SEQ_DELAY,
    )
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(theme_fig(fig, height=400, legend=False), use_container_width=True)