import os
import sys
import glob
import time
import zipfile

import requests
import yaml
import pandas as pd

BTS_URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip"
)

OUTPUT_FILENAME = "_combined.csv"

def download_bts_month(year: int, month: int, raw_dir: str) -> str | None:
    url = BTS_URL_TEMPLATE.format(year=year, month=month)
    zip_filename = f"BTS_{year}_{month:02d}.zip"
    zip_path = os.path.join(raw_dir, zip_filename)

    if os.path.exists(zip_path):
        print(f"  ✓ {zip_filename} že obstaja, preskočim prenos")
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
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        for member in csv_members:
            zf.extract(member, raw_dir)
            extracted.append(os.path.join(raw_dir, member))
    return extracted


def auto_fetch(year_months: list, raw_dir: str) -> None:
    print(f"AUTO mode: prenašam {len(year_months)} mesecev iz BTS Marketing Carrier")
    print("-" * 60)

    requests.packages.urllib3.disable_warnings(
        requests.packages.urllib3.exceptions.InsecureRequestWarning
    )

    successful = []
    failed = []

    for entry in year_months:
        year = entry["year"]
        month = entry["month"]
        zip_path = download_bts_month(year, month, raw_dir)

        if zip_path:
            successful.append((year, month, raw_dir))
            extracted = extract_zip(zip_path, raw_dir)
            for csv_file in extracted:
                print(f"    → {os.path.basename(csv_file)}")
        else:
            failed.append((year, month))

        time.sleep(1)

    print("-" * 60)
    print(f"Uspešno: {len(successful)} mesecev")
    if failed:
        print(f"Neuspešno: {len(failed)} mesecev → {failed}")


def combine_csvs(raw_dir: str) -> None:
    output_path = os.path.join(raw_dir, OUTPUT_FILENAME)

    csv_files = [
        f for f in glob.glob(os.path.join(raw_dir, "*.csv"))
        if os.path.abspath(f) != os.path.abspath(output_path)
    ]

    if not csv_files:
        print(f"NAPAKA: V {raw_dir}/ ni CSV datotek!")
        sys.exit(1)

    print(f"\nZdružujem {len(csv_files)} CSV datotek:")
    dfs = []
    for csv_path in sorted(csv_files):
        fname = os.path.basename(csv_path)
        try:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
            print(f"  {fname} → {len(df):,} vrstic, {len(df.columns)} stolpcev")
            dfs.append(df)
        except Exception as e:
            print(f"  {fname} → NAPAKA: {e}")

    if not dfs:
        print("NAPAKA: Noben CSV se ni uspešno naložil!")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)

    unnamed_cols = [c for c in combined.columns if "Unnamed" in str(c)]
    if unnamed_cols:
        combined.drop(columns=unnamed_cols, inplace=True)

    combined.to_csv(output_path, index=False)
    print(f"\nShranjeno v: {output_path}")
    print(f"Skupaj vrstic: {len(combined):,}")
    print(f"Stolpcev: {len(combined.columns)}")


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_dir = params["fetch"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    year_months = params["fetch"].get("year_months", None)

    if year_months:
        auto_fetch(year_months, raw_dir)
    else:
        print("MANUAL mode: uporabljam obstoječe CSV/ZIP datoteke v data/raw/")
        zip_files = glob.glob(os.path.join(raw_dir, "*.zip"))
        for zf_path in zip_files:
            print(f"Razpakiram: {os.path.basename(zf_path)}")
            extract_zip(zf_path, raw_dir)

    combine_csvs(raw_dir)

if __name__ == "__main__":
    main()