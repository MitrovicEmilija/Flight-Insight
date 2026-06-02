"""
FlightInsight — Naredi manjši vzorec flights.csv za Docker sliko.

Bere flights.csv in shrani vzorec 500K vrstic (~50 MB).
Ta vzorec se vključi v Docker sliko za Analytics stran.

uv run python scripts/create_sample.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

INPUT_PATH = "data/preprocessed/flights.csv"
OUTPUT_PATH = "data/preprocessed/flights_sample.csv"
SAMPLE_SIZE = 500_000


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"NAPAKA: {INPUT_PATH} ne obstaja!")
        sys.exit(1)

    print(f"Berem {INPUT_PATH}...")
    print(f"  (lahko traja ~30 sek)")

    # Beri samo Year stolpec da preverim koliko vrstic
    n_rows = sum(1 for _ in open(INPUT_PATH)) - 1  # -1 za header
    print(f"  Skupaj vrstic: {n_rows:,}")

    if n_rows <= SAMPLE_SIZE:
        print(f"  Datoteka je manjša od {SAMPLE_SIZE:,} vrstic - kopiram celo.")
        import shutil
        shutil.copy(INPUT_PATH, OUTPUT_PATH)
    else:
        print(f"\nVzorčim {SAMPLE_SIZE:,} vrstic (random_state=42)...")
        df = pd.read_csv(INPUT_PATH, low_memory=False)
        sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        sample.to_csv(OUTPUT_PATH, index=False)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nShranjeno: {OUTPUT_PATH}")
    print(f"Velikost:  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()