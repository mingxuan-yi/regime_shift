"""
Download Gürkaynak-Sack-Wright (2006) US Treasury zero-coupon yield dataset
from the Federal Reserve and save selected maturities + derived spread/
curvature features to data/raw/gsw_yields.parquet.
"""

import io
import pathlib

import pandas as pd
import requests

URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

# The CSV has 9 preamble rows before the actual column-header row.
SKIP_ROWS = 9

# Zero-coupon yield columns we care about (continuously compounded, % p.a.)
RAW_COLS = {
    "SVENY01": "yield_1y",
    "SVENY02": "yield_2y",
    "SVENY05": "yield_5y",
    "SVENY10": "yield_10y",
    "SVENY30": "yield_30y",
}

DATE_START = "1960-01-01"
DATE_END = "2026-04-20"

OUTPUT_PATH = pathlib.Path(__file__).parents[2] / "data" / "raw" / "gsw_yields.parquet"


def fetch() -> pd.DataFrame:
    print(f"Downloading GSW yield-curve data from:\n  {URL}")
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(
        io.BytesIO(resp.content),
        skiprows=SKIP_ROWS,
        index_col=0,
        parse_dates=True,
        na_values=["NA", "N/A", ""],
    )
    df.index.name = "date"

    # The dataset uses -999.99 as a second sentinel for missing values.
    df = df.replace(-999.99, pd.NA)

    # Keep only the five maturity columns we need.
    df = df[list(RAW_COLS.keys())].rename(columns=RAW_COLS)

    # Filter to requested date window (data only begins ~1961-06-14).
    df = df.loc[DATE_START:DATE_END]

    # Derived spread / curvature features.
    df["slope_2s10s"] = df["yield_10y"] - df["yield_2y"]
    df["slope_5s30s"] = df["yield_30y"] - df["yield_5y"]
    df["curvature"] = 2 * df["yield_5y"] - df["yield_2y"] - df["yield_10y"]

    return df


def main() -> None:
    df = fetch()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)

    print(f"\nSaved {len(df):,} rows × {len(df.columns)} columns → {OUTPUT_PATH}")
    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nSample (last 5 rows):\n{df.tail()}")
    missing = df.isna().sum()
    if missing.any():
        print(f"\nMissing values per column:\n{missing[missing > 0]}")


if __name__ == "__main__":
    main()
