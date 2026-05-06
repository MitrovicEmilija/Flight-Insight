import os
import sys

import numpy as np
import pandas as pd
import yaml


def load_raw_data(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "_combined.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} does not exist. Run fetch_data.py first.")
        sys.exit(1)

    print(f"Reading raw data from {path}")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"  Read {len(df):,} rows, {len(df.columns):,} columns.")
    return df



def clean_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    n_start = len(df)

    df.columns = df.columns.str.strip()

    # Remove cancelled flights
    if params.get("drop_cancelled", True) and "Cancelled" in df.columns:
        df = df[df["Cancelled"] == 0].copy()
        print(f"  After dropping Cancelled: {len(df):,} ({n_start - len(df):,} removed)")

    # Remove diverted flights
    if params.get("drop_diverted", True) and "Diverted" in df.columns:
        n_before = len(df)
        df = df[df["Diverted"] == 0].copy()
        print(f"  After dropping Diverted: {len(df):,} ({n_before - len(df):,} removed)")

    # Remove rows without target variable
    target = "DepDelayMinutes"
    if target in df.columns:
        n_before = len(df)
        df = df.dropna(subset=[target])
        print(f"  After removing NaN in {target}: {len(df):,} ({n_before - len(df):,} removed)")

    # Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    if n_before - len(df) > 0:
        print(f"  Removed {n_before - len(df):,} duplicate rows.")

    return df



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Hour of departure from CRSDepTime (format: hhmm)
    if "CRSDepTime" in df.columns:
        df["dep_hour"] = (df["CRSDepTime"].fillna(0).astype(int) // 100).clip(0, 23)
        print("  Added: dep_hour")

    # Part of the day
    if "dep_hour" in df.columns:
        conditions = [
            df["dep_hour"].between(5, 11),
            df["dep_hour"].between(12, 16),
            df["dep_hour"].between(17, 20),
        ]
        choices = ["morning", "afternoon", "evening"]
        df["time_of_day"] = np.select(conditions, choices, default="night")
        print("  Added: time_of_day")

    # Weekend
    if "DayOfWeek" in df.columns:
        df["is_weekend"] = (df["DayOfWeek"].isin([6, 7])).astype(int)
        print("  Added: is_weekend")

    # Season
    if "Month" in df.columns:
        season_map = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall",
        }
        df["season"] = df["Month"].map(season_map)
        print("  Added: season")

    # Route
    if "Origin" in df.columns and "Dest" in df.columns:
        df["route"] = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
        print("  Added: route")

    # Distance group
    if "Distance" in df.columns:
        df["distance_group"] = pd.cut(
            df["Distance"],
            bins=[0, 500, 1000, 2000, float("inf")],
            labels=["short", "medium", "long", "very_long"]
        )
        print("  Added: distance_group")

    # Delay cause: NaN -> 0 (no delay of that type)
    delay_causes = [
        "CarrierDelay", "WeatherDelay", "NASDelay",
        "SecurityDelay", "LateAircraftDelay"
    ]
    for col in delay_causes:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    print("  Filled NaN in delay cause columns")
    return df



def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        # Date and time
        "Year", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
        # Airline
        "Marketing_Airline_Network",
        # Airports
        "Origin", "OriginCityName", "Dest", "DestCityName",
        # Arrivals and departures
        "CRSDepTime", "CRSArrTime",
        # Distance and duration
        "Distance", "CRSElapsedTime",
        # Delay causes
        "CarrierDelay", "WeatherDelay", "NASDelay",
        "SecurityDelay", "LateAircraftDelay",
        # TARGET
        "DepDelayMinutes",
        # Engineered features
        "dep_hour", "time_of_day", "is_weekend", "season",
        "route", "distance_group",
    ]

    available = [c for c in keep_cols if c in df.columns]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        print(f"  Missing columns: {missing}")

    df = df[available].copy()
    print(f"  Selected {len(df.columns)} columns")
    return df


def split_by_year(df: pd.DataFrame, reference_year: int, current_year: int):
    if "Year" not in df.columns:
        print("  OPOZORILO: Year stolpec ne obstaja, ne morem narediti split-a")
        return df, df

    reference = df[df["Year"] == reference_year].copy()
    current = df[df["Year"] == current_year].copy()

    print(f"  Reference (leto {reference_year}): {len(reference):,} vrstic")
    print(f"  Current   (leto {current_year}): {len(current):,} vrstic")

    if len(current) == 0:
        print(f"  OPOZORILO: Ni podatkov za current_year={current_year}!")
        print("  Preveri, da si v params.yaml dodala mesec za to leto.")

    return reference, current


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_dir = params["fetch"]["raw_dir"]
    output_path = params["preprocess"]["output_path"]
    preprocess_params = params["preprocess"]
    reference_year = preprocess_params.get("reference_year", 2024)
    current_year = preprocess_params.get("current_year", 2025)

    # 1. Naloži
    df = load_raw_data(raw_dir)

    # 2. Čiščenje
    print("\nČiščenje podatkov:")
    df = clean_data(df, preprocess_params)

    # 3. Feature engineering
    print("\nFeature engineering:")
    df = engineer_features(df)

    # 4. Izberi končne stolpce
    print("\nIzbira stolpcev:")
    df = select_final_columns(df)

    # 5. Shrani VSE podatke (oba leta, za trening)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nShranjeno (vsi podatki, oba leta): {output_path}")
    print(f"  Vrstic: {len(df):,}, Stolpcev: {len(df.columns)}")

    # 6. Razdeli po LETU za drift detection
    print(f"\nRazdelitev po letu (reference={reference_year} vs current={current_year}):")
    reference, current = split_by_year(df, reference_year, current_year)

    output_dir = os.path.dirname(output_path)
    reference_path = os.path.join(output_dir, "flights_reference.csv")
    current_path = os.path.join(output_dir, "flights_current.csv")

    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"Reference ({reference_year}): {reference_path}")
    print(f"Current   ({current_year}): {current_path}")

    if len(current) > 0:
        print(f"\nTarget statistika (drift po letu):")
        print(f"  {reference_year} DepDelayMinutes: mean={reference['DepDelayMinutes'].mean():.2f}, "
              f"median={reference['DepDelayMinutes'].median():.2f}")
        print(f"  {current_year} DepDelayMinutes: mean={current['DepDelayMinutes'].mean():.2f}, "
              f"median={current['DepDelayMinutes'].median():.2f}")


if __name__ == "__main__":
    main()