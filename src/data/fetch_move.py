"""Fetch the ICE BofAML MOVE Index (Treasury implied volatility) from Yahoo Finance.

The MOVE Index is the bond-market analogue of the VIX: a 30-day implied
volatility of Treasury options. Not freely available on FRED (proprietary
ICE BofAML data), but Yahoo Finance distributes daily closing values under
ticker `^MOVE`.

Output
------
data/raw/move.parquet : DatetimeIndex × 1 column ('move'), daily Close values.
"""
import pathlib

import pandas as pd
import yfinance as yf

ROOT       = pathlib.Path(__file__).parents[2]
OUTPUT     = ROOT / "data" / "raw" / "move.parquet"
TICKER     = "^MOVE"
START_DATE = "2003-01-01"   # MOVE has data back to ~2003


def fetch_move() -> pd.DataFrame:
    print(f"Fetching {TICKER} from Yahoo Finance (start={START_DATE}) …")
    t = yf.Ticker(TICKER)
    hist = t.history(start=START_DATE, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"No data returned for {TICKER}")
    # Keep the Close column only; rename to `move`.
    out = hist[["Close"]].rename(columns={"Close": "move"})
    # Strip timezone (yfinance returns tz-aware index; rest of repo is tz-naive).
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
    out.index.name = "date"
    return out


def main() -> None:
    df = fetch_move()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT)
    print(f"\nSaved {len(df):,} rows × {len(df.columns)} columns → {OUTPUT}")
    print(f"Date range : {df.index.min().date()} → {df.index.max().date()}")
    print(f"First non-NaN: {df['move'].first_valid_index().date()}")
    print(f"Summary:")
    print(df["move"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
