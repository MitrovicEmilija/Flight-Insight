"""
FlightInsight — Model loading utilities.

Naloži XGBoost pipeline in HuggingFace ReviewAnalyzer.
Uporablja Streamlit cache za hitro ponovno nalaganje.
"""

import os
import sys
from pathlib import Path

import joblib
import streamlit as st


# Dodaj root projekta v PYTHONPATH (za importe iz src/)
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
        # Importiraj custom razred PREDEN naložimo model
        # Joblib potrebuje ta import da prepozna HighCardinalityEncoder
        from preprocess import HighCardinalityEncoder  # noqa: F401

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


@st.cache_data
def load_airports():
    """Naloži seznam unikatnih letališč iz preprocessed podatkov."""
    import pandas as pd
    data_path = PROJECT_ROOT / "data" / "preprocessed" / "flights.csv"
    if not data_path.exists():
        return ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "LAS", "SEA", "MCO",
                "EWR", "MIA", "BOS", "PHX", "IAH", "MSP", "DTW", "LGA", "PHL", "FLL"]

    try:
        df = pd.read_csv(data_path, usecols=["Origin", "Dest"], nrows=100_000)
        airports = sorted(set(df["Origin"].unique()) | set(df["Dest"].unique()))
        return airports
    except Exception:
        return ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "LAS"]


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