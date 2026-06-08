import sys
from pathlib import Path

import joblib
import streamlit as st
import pandas as pd


# Dodaj root projekta v PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_SRC_PATH = PROJECT_ROOT / "src" / "model"
sys.path.insert(0, str(MODEL_SRC_PATH))


@st.cache_resource
def load_xgboost_model():
    """Naloži XGBoost sklearn pipeline. Cached da se naloži samo enkrat."""
    model_path = PROJECT_ROOT / "models" / "model.pkl"
    if not model_path.exists():
        st.error(f"❌ Model ni najden: {model_path}")
        st.info("Najprej poženi training: `uv run dvc repro train`")
        return None

    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except ImportError as e:
        st.error(f"❌ Ne morem importat preprocess module: {e}")
        st.info(f"Iščem v: {MODEL_SRC_PATH}")
        return None
    except Exception as e:
        st.error(f"Napaka pri nalaganju modela: {e}")
        st.exception(e)
        return None


@st.cache_resource
def load_review_analyzer():
    """Naloži HuggingFace ReviewAnalyzer. Cached da se model prenese samo enkrat."""
    try:
        from src.reviews.analyzer import ReviewAnalyzer

        analyzer = ReviewAnalyzer()
        return analyzer
    except Exception as e:
        st.error(f"Napaka pri nalaganju sentiment analyzer-ja: {e}")
        return None


@st.cache_data
def load_model_metadata():
    """Naloži metadata.yaml o XGBoost modelu."""
    import yaml

    metadata_path = PROJECT_ROOT / "models" / "metadata.yaml"
    if not metadata_path.exists():
        return None
    with open(metadata_path) as f:
        return yaml.safe_load(f)


FALLBACK_AIRPORTS = {
    "ATL": "Atlanta, GA",
    "LAX": "Los Angeles, CA",
    "ORD": "Chicago, IL",
    "DFW": "Dallas-Fort Worth, TX",
    "DEN": "Denver, CO",
    "JFK": "New York, NY",
    "SFO": "San Francisco, CA",
    "LAS": "Las Vegas, NV",
    "SEA": "Seattle, WA",
    "MCO": "Orlando, FL",
    "EWR": "Newark, NJ",
    "MIA": "Miami, FL",
    "BOS": "Boston, MA",
    "PHX": "Phoenix, AZ",
    "IAH": "Houston, TX",
    "MSP": "Minneapolis, MN",
    "DTW": "Detroit, MI",
    "LGA": "New York, NY",
    "PHL": "Philadelphia, PA",
    "FLL": "Fort Lauderdale, FL",
    "BWI": "Baltimore, MD",
    "DCA": "Washington, DC",
    "IAD": "Washington, DC",
    "MDW": "Chicago, IL",
    "SAN": "San Diego, CA",
    "TPA": "Tampa, FL",
    "PDX": "Portland, OR",
    "SLC": "Salt Lake City, UT",
    "STL": "St. Louis, MO",
    "HNL": "Honolulu, HI",
    "ANC": "Anchorage, AK",
    "AUS": "Austin, TX",
    "BNA": "Nashville, TN",
    "CLT": "Charlotte, NC",
    "RDU": "Raleigh-Durham, NC",
    "PIT": "Pittsburgh, PA",
    "CLE": "Cleveland, OH",
    "MCI": "Kansas City, MO",
    "MEM": "Memphis, TN",
    "OAK": "Oakland, CA",
    "SJC": "San Jose, CA",
    "SMF": "Sacramento, CA",
    "ABQ": "Albuquerque, NM",
}


@st.cache_data
def load_airports():

    candidates = [
        PROJECT_ROOT / "data" / "preprocessed" / "flights_sample.csv",
        PROJECT_ROOT / "data" / "preprocessed" / "flights.csv",
    ]

    for data_path in candidates:
        if not data_path.exists():
            continue
        try:
            cols = ["Origin", "OriginCityName", "Dest", "DestCityName"]
            df = pd.read_csv(data_path, usecols=cols, nrows=200_000, low_memory=False)

            # Origin mapping
            origin_map = (
                df[["Origin", "OriginCityName"]]
                .dropna()
                .drop_duplicates(subset=["Origin"])
                .set_index("Origin")["OriginCityName"]
                .to_dict()
            )
            # Dest mapping
            dest_map = (
                df[["Dest", "DestCityName"]]
                .dropna()
                .drop_duplicates(subset=["Dest"])
                .set_index("Dest")["DestCityName"]
                .to_dict()
            )

            combined = {**dest_map, **origin_map}

            if len(combined) > 10:
                return dict(sorted(combined.items()))
        except Exception:
            continue

    return FALLBACK_AIRPORTS


def format_airport(code, airports_dict):
    city = airports_dict.get(code, "")
    if city:
        return f"{code} — {city}"
    return code


@st.cache_data
def load_airlines():
    """Vrne seznam letalskih družb iz BTS."""
    return {
        "AA": "American Airlines",
        "DL": "Delta Air Lines",
        "UA": "United Airlines",
        "WN": "Southwest Airlines",
        "AS": "Alaska Airlines",
        "B6": "JetBlue Airways",
        "NK": "Spirit Airlines",
        "F9": "Frontier Airlines",
        "G4": "Allegiant Air",
        "HA": "Hawaiian Airlines",
    }
