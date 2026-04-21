import os
import sys
import glob
import zipfile

import yaml
import pandas as pd

def main():
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)

    raw_dir = params["fetch"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Unpack ZIP files
    zip_files = glob.glob(os.path.join(raw_dir, "*.zip"))
    for zf_path in zip_files:
        print(f"Extracting: {os.path.basename(zf_path)}")
        with zipfile.ZipFile(zf_path, "r") as zf:
            csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            for member in csv_members:
                zf.extract(member, raw_dir)
                print(f"  -> {member}")

    # 2. Find all csv files (except flights_raw.csv)
    csv_files = [
        f for f in glob.glob(os.path.join(raw_dir, "*.csv"))
        if not f.endswith(".flights_raw.csv")
    ]

    if not csv_files:
        print(f"ERROR: In {raw_dir} there are no CSV files!")
        print("Download data from kaggle")
        print("and place them in data/raw and run the script again.")
        sys.exit(1)

    print(f"Extracting {len(csv_files)} CSV files...")

    #3. Join all csv files
    dfs = []
    for csv_path in sorted(csv_files):
        fname = os.path.basename(csv_path)
        print(f"  Reading: {fname}", end="")
        try:
            df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
            print(f" -> {len(df):,} rows, {len(df.columns)} columns")
            dfs.append(df)
        except Exception as e:
            print(f"ERROR: {e}")

    if not dfs:
        print(f"ERROR: No CSV files found!")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)

    # Remove Unnamed column
    unnamed_cols = [c for c in combined.columns if "Unnamed" in str(c)]
    if unnamed_cols:
        combined.drop(columns=unnamed_cols, inplace=True)

    # 4. Save
    output_path = os.path.join(raw_dir, "flights_raw.csv")
    combined.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")
    print(f"All rows: {len(combined):,}")
    print(f"All columns: {len(combined.columns)}")

if __name__ == "__main__":
    main()