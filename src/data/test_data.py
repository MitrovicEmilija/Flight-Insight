import os
import sys
import shutil

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

REFERENCE_PATH = "data/preprocessed/flights_reference.csv"
CURRENT_PATH = "data/preprocessed/flights_current.csv"
REFERENCE_DIR = "data/reference"
REPORT_PATH = "reports/drift_report.html"

# Stolpci, ki niso primerni za drift detection
EXCLUDE_COLS = [
    "FlightDate",
    "Year",
    "Month",
    "season",
    "OriginCityName",
    "DestCityName",
    "route",
    "CRSDepTime",
    "CRSArrTime",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Naloži reference in current podatke."""
    if not os.path.exists(REFERENCE_PATH):
        print(f"NAPAKA: {REFERENCE_PATH} ne obstaja!")
        sys.exit(1)
    if not os.path.exists(CURRENT_PATH):
        print(f"NAPAKA: {CURRENT_PATH} ne obstaja!")
        sys.exit(1)

    print(f"Nalagam reference: {REFERENCE_PATH}")
    reference = pd.read_csv(REFERENCE_PATH)
    print(f"  → {len(reference):,} vrstic (vsi meseci)")

    print(f"Nalagam current: {CURRENT_PATH}")
    current = pd.read_csv(CURRENT_PATH)
    print(f"{len(current):,} vrstic")

    return reference, current


def filter_same_month(reference: pd.DataFrame, current: pd.DataFrame):
    if "Month" not in current.columns:
        print("  OPOZORILO: Month stolpec ne obstaja, primerjam vse")
        return reference, current

    # Najdi katere mesece vsebuje current
    current_months = current["Month"].unique().tolist()
    print(f"  Current vsebuje meseci: {current_months}")

    # Filtriraj reference na iste mesece
    reference_filtered = reference[reference["Month"].isin(current_months)].copy()
    print(f"  Reference pred filtrom: {len(reference):,} vrstic")
    print(
        f"  Reference po filtru:    {len(reference_filtered):,} vrstic (samo meseci {current_months})"
    )

    return reference_filtered, current


def prepare_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Pripravi DataFrame za Evidently — odstrani neprimerne stolpce."""
    cols_to_drop = [c for c in EXCLUDE_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def sample_for_speed(df: pd.DataFrame, max_rows: int = 50_000) -> pd.DataFrame:
    """Vzemi vzorec — Evidently na 250k+ vrsticah je počasen."""
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        print(f"  Vzorec: {len(df):,} vrstic (random_state=42)")
    return df


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Poženi Evidently drift report."""
    print("\nPriprava za drift detection...")
    reference = prepare_for_drift(reference)
    current = prepare_for_drift(current)

    print("Vzorčenje za hitrost...")
    reference = sample_for_speed(reference)
    current = sample_for_speed(current)

    print(
        f"  Reference končno: {len(reference):,} vrstic, {len(reference.columns)} stolpcev"
    )
    print(
        f"  Current   končno: {len(current):,} vrstic, {len(current.columns)} stolpcev"
    )

    print("\nGeneriranje Evidently poročila...")
    report = Report(
        [
            DataSummaryPreset(),
            DataDriftPreset(),
        ],
        include_tests=True,
    )

    snapshot = report.run(reference_data=reference, current_data=current)

    # Shrani HTML
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    snapshot.save_html(REPORT_PATH)
    print(f"Poročilo shranjeno: {REPORT_PATH}")

    return snapshot.dict()


def analyze_results(result_dict: dict) -> None:
    """Izpiši povzetek rezultatov."""
    print("\n" + "=" * 60)
    print("DRIFT REPORT POVZETEK (apples-to-apples)")
    print("=" * 60)

    tests = result_dict.get("tests", [])
    if not tests:
        print("(Ni testov v poročilu)")
        return

    passed = sum(1 for t in tests if t.get("status") == "SUCCESS")
    failed = sum(1 for t in tests if t.get("status") != "SUCCESS")
    total = len(tests)

    print(f"Skupaj testov: {total}")
    print(f"  Uspešnih:   {passed}")
    print(f"  Neuspešnih: {failed}")

    if failed > 0:
        for t in tests:
            if t.get("status") != "SUCCESS":
                name = t.get("name", "unknown")
                desc = t.get("description", "")[:80]
                print(f"  - {name}: {desc}")


def update_reference(current_path: str) -> None:
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    dest = os.path.join(REFERENCE_DIR, "flights_reference.csv")
    shutil.copy(current_path, dest)
    print(f"\nReferenca posodobljena: {dest}")


def main():
    print("=" * 60)
    print("FlightInsight — Drift Detection")
    print("=" * 60)

    reference, current = load_data()

    reference, current = filter_same_month(reference, current)

    if len(current) == 0:
        print("\nNAPAKA: Po filtriranju je current prazen!")
        sys.exit(0)
    if len(reference) == 0:
        print("\nNAPAKA: Po filtriranju je reference prazen!")
        sys.exit(0)

    # Poženi drift report
    try:
        result_dict = run_drift_report(reference, current)
        analyze_results(result_dict)
    except Exception as e:
        print(f"\nNAPAKA pri generiranju poročila: {e}")
        print("(Pipeline nadaljuje, ker so failures pričakovani)")

    # Kopiraj current kot novo referenco za naslednji run
    update_reference(CURRENT_PATH)

    print("\nDrift detection končana.")
    sys.exit(0)


if __name__ == "__main__":
    main()
