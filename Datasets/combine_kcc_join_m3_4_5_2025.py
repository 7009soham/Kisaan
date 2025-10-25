#!/usr/bin/env python3
"""
Combine these three files located in the SAME folder as this script:
  - 3may2025.csv
  - 4may2025.csv
  - 5may2025.csv

What it does:
- Reads the three files (CSV; falls back to XLSX if CSV not found)
- Normalizes headers to a canonical schema
- Parses CreatedOn (UTC)
- Stacks rows and removes duplicates
- Saves: KCC_MarMay2025_combined.csv in the same folder
- Prints row counts by month

Run (from the Datasets folder):
  python .\combine_kcc_join_m3_4_5_2025.py
"""

from pathlib import Path
import pandas as pd

# Canonical schema to keep consistent ordering
CANONICAL_COLS = [
    "StateName","DistrictName","BlockName","Season","Sector","Category","Crop",
    "QueryType","QueryText","KccAns","CreatedOn","year","month"
]

def find_input(base_dir: Path, stem: str) -> Path:
    """
    Try to find stem.csv, else stem.xlsx in base_dir.
    Raise a clear error if not found.
    """
    csv = base_dir / f"{stem}.csv"
    xlsx = base_dir / f"{stem}.xlsx"
    if csv.exists():
        return csv
    if xlsx.exists():
        return xlsx
    # Suggest what’s actually in the folder
    listing = "\n  - " + "\n  - ".join(sorted(p.name for p in base_dir.iterdir()))
    raise FileNotFoundError(f"Missing input file: {csv.name} (or {xlsx.name}) in {base_dir}\nFiles present:{listing}")

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize to lower+spaces; then map to canonical names
    norm = {c: c.strip().lower().replace("_"," ").replace("-"," ").strip() for c in df.columns}
    df = df.rename(columns=norm)

    mapping = {
        "state name": "StateName", "statename": "StateName",
        "district name": "DistrictName", "districtname": "DistrictName",
        "block name": "BlockName", "blockname": "BlockName",
        "season": "Season",
        "sector": "Sector",
        "category": "Category",
        "crop": "Crop",
        "query type": "QueryType", "querytype": "QueryType",
        "query text": "QueryText", "querytext": "QueryText",
        "kcc ans": "KccAns", "kccans": "KccAns",
        "created on": "CreatedOn", "createdon": "CreatedOn",
        "year": "year",
        "month": "month",
    }
    df = df.rename(columns=mapping)

    # Ensure all canonical columns exist; keep extras too
    for col in CANONICAL_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    extras = [c for c in df.columns if c not in CANONICAL_COLS]
    return df[CANONICAL_COLS + extras]

def parse_created_on(df: pd.DataFrame) -> pd.DataFrame:
    if "CreatedOn" in df.columns:
        df["CreatedOn"] = pd.to_datetime(df["CreatedOn"], errors="coerce", utc=True)
    return df

def main():
    base_dir = Path(__file__).parent.resolve()  # use this script's folder
    print(f"Working directory: {base_dir}")

    # Your exact file stems
    stems = ["3may2025", "4may2025", "5may2025"]
    input_paths = [find_input(base_dir, s) for s in stems]
    print("Input files:")
    for p in input_paths:
        print(f"  - {p.name}")

    frames = []
    for p in input_paths:
        df = read_any(p)
        df = normalize_headers(df)
        df = parse_created_on(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Deduplicate on robust keys
    dedup_keys = ["StateName","DistrictName","BlockName","QueryText","CreatedOn"]
    before = len(combined)
    combined = combined.drop_duplicates(subset=[k for k in dedup_keys if k in combined.columns], keep="first")
    after = len(combined)
    print(f"Removed {before - after} duplicates. Final rows: {after}")

    # Derive year/month from CreatedOn if missing
    if "CreatedOn" in combined.columns and combined["CreatedOn"].notna().any():
        if "year" in combined.columns:
            combined.loc[combined["year"].isna(), "year"] = combined["CreatedOn"].dt.year
        if "month" in combined.columns:
            combined.loc[combined["month"].isna(), "month"] = combined["CreatedOn"].dt.month

    # Sort for stable output
    if "CreatedOn" in combined.columns:
        combined = combined.sort_values("CreatedOn", na_position="last")

    out_path = base_dir / "KCC_MarMay2025_combined.csv"
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Wrote: {out_path}")

    # Monthly counts
    try:
        if "CreatedOn" in combined.columns and combined["CreatedOn"].notna().any():
            counts = combined.set_index("CreatedOn").resample("M").size()
            print("\nRows per month (from CreatedOn):")
            print(counts)
        elif {"year","month"}.issubset(set(combined.columns)):
            counts = combined.groupby(["year","month"]).size().sortindex()
            print("\nRows per month (from year/month columns):")
            print(counts)
    except Exception as e:
        print(f"Note: Could not compute monthly counts: {e}")

if __name__ == "__main__":
    main()