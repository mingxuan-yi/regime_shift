"""
Download the Adrian-Crump-Moench (ACM) term premium decomposition from the
NY Fed and save the 2y / 5y / 10y term premium and expected-short-rate series
to data/raw/acm_term_premium.parquet.

Column naming convention
------------------------
tp_Xy    : ACM term premium at maturity X
rny_Xy   : Risk-neutral (expected short-rate) component at maturity X
"""

import io
import pathlib

import pandas as pd
import requests

URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
SHEET = "ACM Daily"

RAW_COLS = {
    "ACMTP02":  "tp_2y",
    "ACMTP05":  "tp_5y",
    "ACMTP10":  "tp_10y",
    "ACMRNY02": "rny_2y",
    "ACMRNY05": "rny_5y",
    "ACMRNY10": "rny_10y",
}

OUTPUT_PATH = pathlib.Path(__file__).parents[2] / "data" / "raw" / "acm_term_premium.parquet"


def fetch() -> pd.DataFrame:
    print(f"Downloading ACM term-premium data from:\n  {URL}")
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()

    df = pd.read_excel(
        io.BytesIO(resp.content),
        sheet_name=SHEET,
        engine="xlrd",
    )

    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)
    df = df.set_index("DATE")
    df.index.name = "date"

    df = df[list(RAW_COLS.keys())].rename(columns=RAW_COLS)
    df = df.sort_index()

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
