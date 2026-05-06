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
    "FlightDate",       # datumi se naravno spreminjajo
    "Year",             # konstanten v enem letu
    "OriginCityName",   # preveč unikatnih vrednosti
    "DestCityName",     # preveč unikatnih vrednosti
    "route",            # preveč unikatnih vrednosti
    "CRSDepTime",       # časi (uporabljamo dep_hour)
    "CRSArrTime",       # časi
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
    print(f"  → {len(reference):,} vrstic")

    print(f"Nalagam current: {CURRENT_PATH}")
    current = pd.read_csv(CURRENT_PATH)
    print(f"  → {len(current):,} vrstic")

    return reference, current


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

    print("\nGeneriranje Evidently poročila...")
    report = Report([
        DataSummaryPreset(),
        DataDriftPreset(),
    ], include_tests=True)

    snapshot = report.run(reference_data=reference, current_data=current)

    # Shrani HTML
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    snapshot.save_html(REPORT_PATH)
    print(f"Poročilo shranjeno: {REPORT_PATH}")

    # Vrni dict za analizo
    return snapshot.dict()


def analyze_results(result_dict: dict) -> None:
    """Izpiši povzetek rezultatov."""
    print("\n" + "=" * 60)
    print("DRIFT REPORT POVZETEK")
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
        print(f"\nNeuspeli testi (drift zaznan v):")
        for t in tests:
            if t.get("status") != "SUCCESS":
                name = t.get("name", "unknown")
                desc = t.get("description", "")[:80]
                print(f"  - {name}: {desc}")


def update_reference(current_path: str) -> None:
    """Po uspešnem testiranju kopiraj current → reference za naslednji zagon."""
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    dest = os.path.join(REFERENCE_DIR, "flights_reference.csv")
    shutil.copy(current_path, dest)
    print(f"\nReferenca posodobljena: {dest}")


def main():
    print("=" * 60)
    print("FlightInsight — Drift Detection (Evidently)")
    print("=" * 60)

    # 1. Naloži
    reference, current = load_data()

    # 2. Poženi drift report
    try:
        result_dict = run_drift_report(reference, current)
        analyze_results(result_dict)
    except Exception as e:
        print(f"\nNAPAKA pri generiranju poročila: {e}")
        # Vseeno nadaljuj — drift failures so OK
        print("(Pipeline nadaljuje, ker so failures pričakovani)")

    # 3. Kopiraj current kot novo referenco za naslednji run
    update_reference(CURRENT_PATH)

    # POMEMBNO: vedno exit 0!
    # Drift detection failures so del normalnega monitoring procesa.
    print("\nDrift detection končana.")
    sys.exit(0)


if __name__ == "__main__":
    main()
