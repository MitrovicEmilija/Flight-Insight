"""
FlightInsight — Priprava surovih BTS podatkov.

Vhodni CSV-ji shranjeni v data/raw/ (npr. flights_2024.csv).
Izhod: data/raw/_combined.csv (preimenovan zato, da ga ne prebere kot vhod).
"""

import os
import sys
import glob
import zipfile

import yaml
import pandas as pd

# Izhodna datoteka — nima pripone .csv da je ne pobere glob nazaj
OUTPUT_FILENAME = "_combined.csv"


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_dir = params["fetch"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Razpakiraj morebitne ZIP datoteke
    zip_files = glob.glob(os.path.join(raw_dir, "*.zip"))
    for zf_path in zip_files:
        print(f"Razpakiram: {os.path.basename(zf_path)}")
        with zipfile.ZipFile(zf_path, "r") as zf:
            csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            for member in csv_members:
                zf.extract(member, raw_dir)

    # 2. Poišči VSE CSV datoteke RAZEN izhodne
    output_path = os.path.join(raw_dir, OUTPUT_FILENAME)
    csv_files = [
        f for f in glob.glob(os.path.join(raw_dir, "*.csv"))
        if os.path.abspath(f) != os.path.abspath(output_path)
    ]

    if not csv_files:
        print(f"NAPAKA: V {raw_dir}/ ni vhodnih CSV datotek!")
        print("Prenesi BTS podatke iz Kaggle in jih shrani v data/raw/.")
        sys.exit(1)

    print(f"Najdenih {len(csv_files)} vhodnih CSV datotek")

    # 3. Združi vse CSV-je
    dfs = []
    for csv_path in sorted(csv_files):
        fname = os.path.basename(csv_path)
        print(f"  Berem: {fname}", end="")
        try:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
            print(f" → {len(df):,} vrstic, {len(df.columns)} stolpcev")
            dfs.append(df)
        except Exception as e:
            print(f" → NAPAKA: {e}")

    if not dfs:
        print("NAPAKA: Noben CSV se ni uspešno naložil!")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)

    # Odstrani 'Unnamed' stolpce (BTS trailing vejica)
    unnamed_cols = [c for c in combined.columns if "Unnamed" in str(c)]
    if unnamed_cols:
        combined.drop(columns=unnamed_cols, inplace=True)

    # 4. Shrani izhod
    combined.to_csv(output_path, index=False)
    print(f"\nShranjeno v: {output_path}")
    print(f"Skupaj vrstic: {len(combined):,}")
    print(f"Stolpcev: {len(combined.columns)}")


if __name__ == "__main__":
    main()