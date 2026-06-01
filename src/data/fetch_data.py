"""
FlightInsight — Inkrementalni prenos BTS Marketing Carrier podatkov.

Strategija:
  1. Preveri kateri (year, month) pari so ŽE v obstoječem _combined.csv
  2. Iz params.yaml `year_months` izberi SAMO manjkajoče
  3. Prenesi samo manjkajoče → pripni k obstoječemu _combined.csv

Tako se v cloud-u (GitHub Actions) prenese samo nov mesec, ne celega leta.
"""

import os
import sys
import glob
import time
import zipfile

import requests
import yaml
import pandas as pd

BTS_URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip"
)

OUTPUT_FILENAME = "_combined.csv"


def get_existing_months(combined_path: str) -> set:
    """Vrne set (year, month) parov ki so že v _combined.csv."""
    if not os.path.exists(combined_path):
        return set()

    try:
        # Bere samo Year + Month stolpca da prihrani RAM
        df = pd.read_csv(
            combined_path,
            encoding="latin-1",
            usecols=["Year", "Month"],
            low_memory=False,
        )
        existing = set(zip(df["Year"].astype(int), df["Month"].astype(int)))
        print(f"Obstoječi (year, month) v {OUTPUT_FILENAME}: {len(existing)}")
        return existing
    except Exception as e:
        print(f"OPOZORILO: Ne morem prebrati obstoječega {combined_path}: {e}")
        return set()


def download_bts_month(year: int, month: int, raw_dir: str) -> str | None:
    """Prenese eno mesečno BTS ZIP datoteko."""
    url = BTS_URL_TEMPLATE.format(year=year, month=month)
    zip_filename = f"BTS_{year}_{month:02d}.zip"
    zip_path = os.path.join(raw_dir, zip_filename)

    if os.path.exists(zip_path):
        print(f"  ✓ {zip_filename} že obstaja lokalno, preskočim prenos")
        return zip_path

    print(f"  ⬇ Prenašam: {zip_filename}", end=" ", flush=True)
    try:
        response = requests.get(url, stream=True, timeout=180, verify=False)
        response.raise_for_status()

        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        size_mb = downloaded / 1024 / 1024
        print(f"→ {size_mb:.1f} MB")
        return zip_path

    except requests.exceptions.RequestException as e:
        print(f"→ NAPAKA: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None


def extract_zip(zip_path: str, raw_dir: str) -> list[str]:
    """Razpakira ZIP datoteko, vrne seznam izvlečenih CSV-jev."""
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        for member in csv_members:
            zf.extract(member, raw_dir)
            extracted.append(os.path.join(raw_dir, member))
    return extracted


def load_new_csv(csv_paths: list[str]) -> pd.DataFrame:
    """Naloži in združi nove CSV datoteke v en DataFrame."""
    dfs = []
    for csv_path in sorted(csv_paths):
        fname = os.path.basename(csv_path)
        try:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
            print(f"  Berem: {fname} → {len(df):,} vrstic")
            dfs.append(df)
        except Exception as e:
            print(f"  Berem: {fname} → NAPAKA: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Odstrani Unnamed stolpce
    unnamed_cols = [c for c in combined.columns if "Unnamed" in str(c)]
    if unnamed_cols:
        combined.drop(columns=unnamed_cols, inplace=True)

    return combined


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_dir = params["fetch"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    year_months = params["fetch"].get("year_months", [])
    if not year_months:
        print("NAPAKA: year_months ni nastavljen v params.yaml")
        sys.exit(1)

    output_path = os.path.join(raw_dir, OUTPUT_FILENAME)

    # 1. Pridobi obstoječe (year, month) pari
    existing = get_existing_months(output_path)

    # 2. Izračunaj manjkajoče
    requested = {(entry["year"], entry["month"]) for entry in year_months}
    missing = requested - existing

    print(f"\nIz params.yaml zahtevanih: {len(requested)} (year, month) parov")
    print(f"Že imamo:                  {len(existing & requested)}")
    print(f"Manjkajočih za prenos:     {len(missing)}")

    if not missing:
        print("\n✓ Nič novega za prenesti! Vsi meseci so že v _combined.csv")
        return

    print("-" * 60)
    print(f"Prenašam {len(missing)} manjkajočih mesecev:")

    # 3. Onemogoči SSL warnings
    requests.packages.urllib3.disable_warnings(
        requests.packages.urllib3.exceptions.InsecureRequestWarning
    )

    new_csv_paths = []
    failed = []

    for year, month in sorted(missing):
        zip_path = download_bts_month(year, month, raw_dir)
        if zip_path:
            extracted = extract_zip(zip_path, raw_dir)
            for csv_file in extracted:
                print(f"    → {os.path.basename(csv_file)}")
                new_csv_paths.append(csv_file)
        else:
            failed.append((year, month))
        time.sleep(1)

    print("-" * 60)

    if failed:
        print(f"OPOZORILO: Neuspešno {len(failed)} mesecev → {failed}")

    if not new_csv_paths:
        print("Ni novih podatkov za pripenjanje.")
        return

    # 4. Naloži nove CSV-je
    print(f"\nNalagam {len(new_csv_paths)} novih CSV datotek...")
    new_data = load_new_csv(new_csv_paths)

    if len(new_data) == 0:
        print("NAPAKA: Novi podatki so prazni!")
        sys.exit(1)

    # 5. Pripni k obstoječemu _combined.csv
    if os.path.exists(output_path):
        print(f"\nPripenjam k obstoječemu {OUTPUT_FILENAME}...")
        # Append mode (brez headerja) - pisanje brez RAM hog-a
        new_data.to_csv(output_path, mode="a", header=False, index=False)
    else:
        print(f"\nUstvarjam nov {OUTPUT_FILENAME}...")
        new_data.to_csv(output_path, index=False)

    # 6. Verify (samo preberi metapodatke)
    df_check = pd.read_csv(
        output_path,
        encoding="latin-1",
        usecols=["Year", "Month"],
        low_memory=False,
    )
    final_months = set(zip(df_check["Year"].astype(int), df_check["Month"].astype(int)))

    print(f"\n{'=' * 60}")
    print(f"Shranjeno v: {output_path}")
    print(f"Skupaj vrstic: {len(df_check):,}")
    print(f"Skupaj mesecev: {len(final_months)}")


if __name__ == "__main__":
    main()