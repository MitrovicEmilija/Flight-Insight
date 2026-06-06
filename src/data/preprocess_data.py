import os
import sys
import gc

import yaml
import numpy as np
import pandas as pd

# Stolpci, ki jih dejansko rabimo iz BTS CSV
USECOLS = [
    "Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Marketing_Airline_Network",
    "Origin", "OriginCityName", "Dest", "DestCityName",
    "CRSDepTime", "CRSArrTime",
    "Distance", "CRSElapsedTime",
    "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
    "DepDelayMinutes",
    "Cancelled", "Diverted",
]

DTYPES = {
    "Year": "int16",
    "Month": "int8",
    "DayofMonth": "int8",
    "DayOfWeek": "int8",
    "Marketing_Airline_Network": "category",
    "Origin": "category",
    "OriginCityName": "category",
    "Dest": "category",
    "DestCityName": "category",
    "Cancelled": "float32",
    "Diverted": "float32",
}

CHUNK_SIZE = 500_000


def process_chunk(chunk: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Procesiraj en chunk podatkov."""
    # Strip column names
    chunk.columns = chunk.columns.str.strip()

    # Cleaning
    if params.get("drop_cancelled", True) and "Cancelled" in chunk.columns:
        chunk = chunk[chunk["Cancelled"] == 0]

    if params.get("drop_diverted", True) and "Diverted" in chunk.columns:
        chunk = chunk[chunk["Diverted"] == 0]

    if "DepDelayMinutes" in chunk.columns:
        chunk = chunk.dropna(subset=["DepDelayMinutes"])

    # Drop Cancelled/Diverted po čiščenju (več ne rabimo)
    cols_to_drop = [c for c in ["Cancelled", "Diverted"] if c in chunk.columns]
    if cols_to_drop:
        chunk = chunk.drop(columns=cols_to_drop)

    # Feature engineering
    if "CRSDepTime" in chunk.columns:
        chunk["dep_hour"] = (chunk["CRSDepTime"].fillna(0).astype("int16") // 100).clip(0, 23).astype("int8")

    if "dep_hour" in chunk.columns:
        conditions = [
            chunk["dep_hour"].between(5, 11),
            chunk["dep_hour"].between(12, 16),
            chunk["dep_hour"].between(17, 20),
        ]
        choices = ["morning", "afternoon", "evening"]
        chunk["time_of_day"] = pd.Categorical(
            np.select(conditions, choices, default="night"),
            categories=["morning", "afternoon", "evening", "night"]
        )

    if "DayOfWeek" in chunk.columns:
        chunk["is_weekend"] = (chunk["DayOfWeek"].isin([6, 7])).astype("int8")

    if "Month" in chunk.columns:
        season_map = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall",
        }
        chunk["season"] = pd.Categorical(
            chunk["Month"].map(season_map),
            categories=["winter", "spring", "summer", "fall"]
        )

    if "Origin" in chunk.columns and "Dest" in chunk.columns:
        # route ne pretvarjamo v category (preveč unikatnih vrednosti)
        chunk["route"] = chunk["Origin"].astype(str) + "-" + chunk["Dest"].astype(str)

    if "Distance" in chunk.columns:
        chunk["distance_group"] = pd.cut(
            chunk["Distance"],
            bins=[0, 500, 1000, 2000, float("inf")],
            labels=["short", "medium", "long", "very_long"],
        )

    # Delay cause: NaN → 0
    delay_causes = ["CarrierDelay", "WeatherDelay", "NASDelay",
                    "SecurityDelay", "LateAircraftDelay"]
    for col in delay_causes:
        if col in chunk.columns:
            chunk[col] = chunk[col].fillna(0).astype("float32")

    return chunk


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_dir = params["fetch"]["raw_dir"]
    output_path = params["preprocess"]["output_path"]
    preprocess_params = params["preprocess"]
    reference_year = preprocess_params.get("reference_year", 2024)
    current_year = preprocess_params.get("current_year", 2025)

    raw_path = os.path.join(raw_dir, "_combined.csv")
    if not os.path.exists(raw_path):
        print(f"NAPAKA: {raw_path} ne obstaja!")
        sys.exit(1)

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    reference_path = os.path.join(output_dir, "flights_reference.csv")
    current_path = os.path.join(output_dir, "flights_current.csv")

    # Pobriši stare izhode (pisali bomo append-style)
    for path in [output_path, reference_path, current_path]:
        if os.path.exists(path):
            os.remove(path)

    print(f"Berem v chunkih po {CHUNK_SIZE:,} vrstic iz: {raw_path}")
    print(f"  Reference year: {reference_year}")
    print(f"  Current year:   {current_year}")
    print()

    chunk_iter = pd.read_csv(
        raw_path,
        encoding="latin-1",
        usecols=lambda c: c.strip() in USECOLS,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    total_rows = 0
    reference_rows = 0
    current_rows = 0

    for i, chunk in enumerate(chunk_iter, start=1):
        n_in = len(chunk)

        # Procesiraj chunk
        chunk = process_chunk(chunk, preprocess_params)
        n_out = len(chunk)

        if n_out == 0:
            print(f"  Chunk {i}: {n_in:,} → 0 (vse filtrirano)")
            continue

        # Razdeli v 3 datoteke (append mode)
        # 1. flights.csv (vse)
        chunk.to_csv(
            output_path,
            mode="a",
            header=(not os.path.exists(output_path) or os.path.getsize(output_path) == 0),
            index=False,
        )
        total_rows += n_out

        # 2. flights_reference.csv (samo reference_year)
        ref_chunk = chunk[chunk["Year"] == reference_year]
        if len(ref_chunk) > 0:
            ref_chunk.to_csv(
                reference_path,
                mode="a",
                header=(not os.path.exists(reference_path) or os.path.getsize(reference_path) == 0),
                index=False,
            )
            reference_rows += len(ref_chunk)

        # 3. flights_current.csv (samo current_year)
        cur_chunk = chunk[chunk["Year"] == current_year]
        if len(cur_chunk) > 0:
            cur_chunk.to_csv(
                current_path,
                mode="a",
                header=(not os.path.exists(current_path) or os.path.getsize(current_path) == 0),
                index=False,
            )
            current_rows += len(cur_chunk)

        print(f"  Chunk {i}: {n_in:,} → {n_out:,} vrstic "
              f"(ref: {reference_rows:,}, cur: {current_rows:,})")

        # Sprosti pomnilnik
        del chunk
        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"Skupaj obdelanih:    {total_rows:,} vrstic")
    print(f"Reference ({reference_year}): {reference_rows:,} vrstic → {reference_path}")
    print(f"Current   ({current_year}): {current_rows:,} vrstic → {current_path}")
    print(f"Vsi podatki:         {output_path}")


if __name__ == "__main__":
    main()