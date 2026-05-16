"""
Download macro time-series from FRED via fredapi, apply standard
transformations (YoY %, MoM change), resample everything to a daily
calendar with forward-fill, and save to data/raw/fred_macro.parquet.

Requires FRED_API_KEY in a .env file (or as an environment variable).
"""

import os
import pathlib

import certifi
import pandas as pd
from dotenv import load_dotenv

# Point requests (used internally by fredapi) at certifi's CA bundle —
# required on macOS where Python's SSL store is separate from the keychain.
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import time

from fredapi import Fred  # noqa: E402 — must come after env vars are set

load_dotenv(pathlib.Path(__file__).parents[2] / ".env")

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise EnvironmentError(
        "FRED_API_KEY not set. Add it to .env in the project root:\n"
        "  FRED_API_KEY=your_key_here\n"
        "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
    )

OUTPUT_PATH = pathlib.Path(__file__).parents[2] / "data" / "raw" / "fred_macro.parquet"

# ---------------------------------------------------------------------------
# Series catalogue
# Each entry: (fred_id, output_name, transform, frequency_hint)
#   transform : None | "yoy" | "mom_diff" | "mom_pct"
#   frequency_hint : "D" | "M" | "Q"  (used only for display)
# ---------------------------------------------------------------------------
SERIES = [
    # --- rates (daily) ---
    ("DFF",           "ff_rate",          None,       "D"),
    ("DFEDTARU",      "ff_target_upper",  None,       "D"),
    # --- inflation (monthly → daily) ---
    ("CPIAUCSL",      "cpi_yoy",          "yoy",      "M"),
    ("PCEPILFE",      "core_pce_yoy",     "yoy",      "M"),
    # --- labour (monthly → daily) ---
    ("UNRATE",        "unemp_rate",       None,       "M"),
    ("PAYEMS",        "nfp_mom",          "mom_diff", "M"),
    # --- activity (monthly → daily) ---
    ("RSAFS",         "retail_sales_mom", "mom_pct",  "M"),
    # --- breakeven / spread (daily) ---
    ("T10YIE",        "breakeven_10y",    None,       "D"),
    ("T10Y2Y",        "spread_10y2y",     None,       "D"),
    # --- credit / vol (daily) ---
    ("BAMLC0A0CM",    "ig_oas",           None,       "D"),
    ("BAMLH0A0HYM2",  "hy_oas",           None,       "D"),
    ("BAMLHE00EHYIEY","hy_vol_proxy",     None,       "D"),
    # --- NEW: bond-market CPD additions (2026-05-12) ---
    ("VIXCLS",        "vix",              None,       "D"),   # equity implied vol (CBOE)
    ("WALCL",         "fed_assets",       None,       "W"),   # Fed total assets (H.4.1, weekly Wed)
]

# ISM PMI: NAPM is the preferred ticker; MANEMP (mfg employment) is the fallback
ISM_TICKERS = [("NAPM", "ism_pmi"), ("MANEMP", "ism_pmi")]

# TED spread: TEDRATE discontinued 2023-03; reconstruct from T-bill minus LIBOR
TED_PRIMARY  = "TEDRATE"
TED_TBILL    = "DTB3"
TED_LIBOR    = "USD3MTD156N"


def _transform(s: pd.Series, transform: str | None) -> pd.Series:
    if transform is None:
        return s
    if transform == "yoy":
        return s.pct_change(12) * 100          # monthly series → 12 lags
    if transform == "mom_diff":
        return s.diff(1)
    if transform == "mom_pct":
        return s.pct_change(1) * 100
    raise ValueError(f"Unknown transform: {transform}")


def _to_daily(s: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    return s.reindex(daily_index).ffill()


def _get_series(fred: Fred, ticker: str, retries: int = 3, delay: float = 2.0) -> pd.Series:
    for attempt in range(retries):
        try:
            s = fred.get_series(ticker)
            time.sleep(0.3)   # stay well under FRED's rate limit
            return s
        except Exception as exc:
            if attempt < retries - 1 and "Internal Server Error" in str(exc):
                wait = delay * (attempt + 1)
                print(f"    retry {attempt + 1}/{retries - 1} after {wait:.0f}s ({exc})")
                time.sleep(wait)
            else:
                raise


def fetch_all() -> pd.DataFrame:
    fred = Fred(api_key=FRED_API_KEY)

    daily_index = pd.date_range("1960-01-01", pd.Timestamp.today().normalize(), freq="D")
    frames: dict[str, pd.Series] = {}

    # --- main catalogue ---
    for ticker, name, transform, freq in SERIES:
        print(f"  {ticker:20s} → {name}")
        try:
            raw = _get_series(fred, ticker)
            raw.index = pd.DatetimeIndex(raw.index).normalize()
            transformed = _transform(raw, transform)
            frames[name] = _to_daily(transformed, daily_index)
        except Exception as exc:
            print(f"    WARNING: {ticker} failed ({exc}) — skipping")

    # --- ISM PMI (try NAPM then MANEMP) ---
    for ticker, name in ISM_TICKERS:
        if name in frames:
            break
        print(f"  {ticker:20s} → {name}")
        try:
            raw = _get_series(fred, ticker)
            raw.index = pd.DatetimeIndex(raw.index).normalize()
            frames[name] = _to_daily(raw, daily_index)
            break
        except Exception as exc:
            print(f"    WARNING: {ticker} failed ({exc})")

    # --- TED spread ---
    print(f"  {TED_PRIMARY:20s} → ted_spread")
    try:
        raw = _get_series(fred, TED_PRIMARY)
        raw.index = pd.DatetimeIndex(raw.index).normalize()
        frames["ted_spread"] = _to_daily(raw, daily_index)
        print("    (using TEDRATE)")
    except Exception:
        print(f"    TEDRATE unavailable — reconstructing from {TED_TBILL} - {TED_LIBOR}")
        try:
            tbill = _get_series(fred, TED_TBILL)
            libor  = _get_series(fred, TED_LIBOR)
            tbill.index = pd.DatetimeIndex(tbill.index).normalize()
            libor.index  = pd.DatetimeIndex(libor.index).normalize()
            ted = (libor - tbill).dropna()
            frames["ted_spread"] = _to_daily(ted, daily_index)
            print("    (reconstructed from LIBOR - T-bill)")
        except Exception as exc2:
            print(f"    WARNING: TED spread fallback also failed ({exc2}) — skipping")

    df = pd.DataFrame(frames, index=daily_index)
    df.index.name = "date"

    # Drop leading all-NaN rows (before any series has data)
    df = df.loc[df.notna().any(axis=1)]
    return df


def main() -> None:
    print("Fetching FRED series…")
    df = fetch_all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)

    print(f"\nSaved {len(df):,} rows × {len(df.columns)} columns → {OUTPUT_PATH}")
    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"\nColumns and first non-NaN date:")
    for col in df.columns:
        first = df[col].first_valid_index()
        print(f"  {col:25s}  first obs: {first.date() if first else 'all NaN'}")


if __name__ == "__main__":
    main()
