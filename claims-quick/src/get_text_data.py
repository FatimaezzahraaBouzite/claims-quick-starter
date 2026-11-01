"""Download vehicle complaint descriptions from the free NHTSA API.

Each complaint has a free-text description written by the driver, plus a
yes/no field saying whether a crash happened. That field is our label.

Run: python -m src.get_text_data
Output: data/complaints.csv
"""
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
OUT = Path("data/complaints.csv")

# Popular vehicles in Canada, several model years each
VEHICLES = [
    ("honda", "civic"), ("honda", "cr-v"), ("toyota", "corolla"), ("toyota", "rav4"),
    ("ford", "f-150"), ("ford", "escape"), ("hyundai", "elantra"), ("mazda", "cx-5"),
    ("chevrolet", "equinox"), ("nissan", "rogue"), ("kia", "sportage"), ("jeep", "grand cherokee"),
]
YEARS = range(2015, 2025)


def yes(value) -> bool:
    """The API returns true/false or 'YES'/'NO' depending on the record."""
    return value is True or str(value).strip().upper() in {"YES", "Y", "TRUE"}


def fetch(make: str, model: str, year: int) -> list[dict]:
    r = requests.get(URL, params={"make": make, "model": model, "modelYear": year}, timeout=30)
    r.raise_for_status()
    rows = []
    for c in r.json().get("results", []):
        rows.append({
            "odi_number": c.get("odiNumber"),
            "make": make, "model": model, "year": year,
            "component": c.get("components"),
            "description": c.get("summary"),
            "crash": int(yes(c.get("crash"))),
            "fire": int(yes(c.get("fire"))),
            "injuries": int(c.get("numberOfInjuries") or 0),
        })
    return rows


def main() -> None:
    rows = []
    for make, model in VEHICLES:
        for year in YEARS:
            try:
                got = fetch(make, model, year)
                rows += got
                print(f"{make} {model} {year}: {len(got)} complaints")
            except Exception as e:
                print(f"{make} {model} {year}: skipped ({e})")
            time.sleep(0.3)          # be polite to the free API

    df = pd.DataFrame(rows).drop_duplicates("odi_number")
    df = df[df["description"].fillna("").str.len() > 50]
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {len(df):,} complaints to {OUT}")
    print(f"Crash reported in {df['crash'].mean():.1%} of them")


if __name__ == "__main__":
    main()
