import json
from datetime import datetime
from pathlib import Path

import pandas as pd


# Lokacija log file-a
PROJECT_ROOT = Path(__file__).parent.parent.parent
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"
PREDICTIONS_LOG = MONITORING_DIR / "predictions_log.csv"


def log_prediction(
    model_type: str,
    prediction: float | str,
    features: dict,
    extra: dict = None,
) -> None:
    # Logiraj eno napoved v CSV.
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "prediction": prediction,
        "features": json.dumps(features),
    }

    if extra:
        row["extra"] = json.dumps(extra)
    else:
        row["extra"] = "{}"

    # Append v CSV (ustvari header če novo)
    file_exists = PREDICTIONS_LOG.exists()

    new_row = pd.DataFrame([row])
    new_row.to_csv(
        PREDICTIONS_LOG,
        mode="a",
        header=not file_exists,
        index=False,
    )


def load_predictions_log() -> pd.DataFrame:
    """Naloži log za analizo. Vrne prazen DataFrame če ne obstaja."""
    if not PREDICTIONS_LOG.exists():
        return pd.DataFrame(columns=["timestamp", "model_type", "prediction", "features", "extra"])

    df = pd.read_csv(PREDICTIONS_LOG)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_stats(df: pd.DataFrame) -> dict:
    """Izračunaj statistike za admin dashboard."""
    if len(df) == 0:
        return {
            "total_predictions": 0,
            "by_model": {},
            "last_24h": 0,
            "last_7d": 0,
        }

    now = pd.Timestamp.now()

    return {
        "total_predictions": len(df),
        "by_model": df["model_type"].value_counts().to_dict(),
        "last_24h": (df["timestamp"] > now - pd.Timedelta(days=1)).sum(),
        "last_7d": (df["timestamp"] > now - pd.Timedelta(days=7)).sum(),
        "first_prediction": df["timestamp"].min(),
        "last_prediction": df["timestamp"].max(),
    }