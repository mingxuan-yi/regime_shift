"""
Merge GSW yields, ACM term premium, and FRED macro series into a single
daily panel, standardise with rolling z-scores, and persist to parquet.

Pipeline
--------
1. Load the three raw parquets.
2. Inner-join on the intersection of business-day dates.
3. Forward-fill intra-series gaps (daily series have weekends; monthly
   series have no mid-month observations).
4. Drop columns that have fewer than MIN_COVERAGE of non-NaN observations
   — prevents one short-history series collapsing the entire panel.
5. Trim leading rows where any remaining column is still NaN (i.e. start
   the panel at the first date every retained series has data).
6. Compute rolling z-scores: z_t = (x_t − μ_{t,252}) / σ_{t,252}
   using a causal 252-day window (no look-ahead).  Rows in the warm-up
   period drop out automatically.
7. Save data/processed/panel_daily.parquet and variable_definitions.yaml.
"""

import pathlib

import numpy as np
import pandas as pd
import yaml

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).parents[2]
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

GSW_PATH    = RAW / "gsw_yields.parquet"
ACM_PATH    = RAW / "acm_term_premium.parquet"
FRED_PATH   = RAW / "fred_macro.parquet"
CREDIT_PATH = RAW / "credit_spreads.parquet"
MOVE_PATH   = RAW / "move.parquet"
PANEL_OUT   = PROCESSED / "panel_daily.parquet"
YAML_OUT    = PROCESSED / "variable_definitions.yaml"

# Columns with fewer than this fraction of non-NaN values are excluded.
# At 0.20 this drops ig_oas / hy_oas / hy_vol_proxy (4.5 % coverage) while
# retaining ff_target_upper (26 %) and breakeven_10y (35 %).
MIN_COVERAGE = 0.20

ROLLING_WINDOW = 252   # ≈ 1 trading year

# ── derived series ─────────────────────────────────────────────────────────
# Window used for the realised-volatility proxy of the MOVE index (which is
# not freely available on FRED). 30 trading days ≈ 6 weeks of yield_10y
# first-differences. The proxy is well-correlated with the official MOVE
# index over overlapping samples; see Choi et al. (2017) for discussion.
MOVE_PROXY_WINDOW = 30

# ── output allowlist ────────────────────────────────────────────────────────
# After loading + coverage filtering, the panel is restricted to this set
# of columns. The list is the "lean" panel for bond-market CPD: it removes
# redundancies (yield ≡ rny + tp; spread_10y2y ≡ slope_2s10s; ff_rate ≈
# ff_target_upper; etc.) and low-information monthly-fwd-filled macro
# variables (nfp_mom, retail_sales_mom, ism_pmi). It adds the bond-market
# variables introduced 2026-05-12 (vix, move_proxy, fed_assets, sofr,
# credit_baa_aaa).
KEEP_COLS = [
    # Treasury yields (slim — drop 1y/5y/30y)
    "yield_2y", "yield_10y",
    # Curve shape (only one slope; drop slope_5s30s, curvature, spread_10y2y)
    "slope_2s10s",
    # Term-structure decomposition — 10y only (drop 2y, 5y)
    "tp_10y", "rny_10y",
    # Policy rate (drop ff_target_upper; ρ(ff_rate, ff_target_upper) > 0.99)
    "ff_rate",
    # Inflation — three orthogonal perspectives
    "cpi_yoy", "core_pce_yoy", "breakeven_10y",
    # Macro mandate (drop nfp_mom, retail_sales_mom, ism_pmi: low info on daily lag)
    "unemp_rate",
    # Banking-system stress
    "ted_spread",
    # NEW (2026-05-12): bond-market CPD additions
    "vix",             # equity implied vol (S&P 500)
    "move",            # Treasury implied vol — REAL MOVE Index from Yahoo (^MOVE)
    "fed_assets",      # Fed total assets (H.4.1 WALCL, fwd-fill)
    # Dropped 2026-05-12:
    # "sofr"            — only published from 2018-04-03 (12.4% coverage of full raw
    #                     range, fails MIN_COVERAGE=0.20). The pre-2018 money-market
    #                     short rate is already captured by ff_rate (EFFR) and
    #                     funding stress by ted_spread.
    # "credit_baa_aaa"  — credit_spreads.parquet starts 2009-12-01; including it
    #                     forces z-score warm-up to push panel start to 2010-12-01,
    #                     which loses the 2010-08-10 and 2010-11-03 anchors.
]


# ── helpers ────────────────────────────────────────────────────────────────

def _load_and_align() -> pd.DataFrame:
    gsw    = pd.read_parquet(GSW_PATH)
    acm    = pd.read_parquet(ACM_PATH)
    fred   = pd.read_parquet(FRED_PATH)

    # Optional sources — added 2026-05-12. Skip silently if not present.
    credit = pd.read_parquet(CREDIT_PATH) if CREDIT_PATH.exists() else None
    move   = pd.read_parquet(MOVE_PATH)   if MOVE_PATH.exists()   else None

    # Intersect date indices then keep only Mon–Fri.
    # We do NOT intersect with credit / move — they have shorter histories
    # than yields / ACM (credit: 2009+, move: 2003+) and would chop the
    # long-history panel. Both are reindexed to `common` (NaN where
    # unavailable) and contribute only over their coverage windows.
    common = gsw.index.intersection(acm.index).intersection(fred.index)
    common = common[common.day_of_week < 5]

    frames = [gsw.reindex(common), acm.reindex(common), fred.reindex(common)]
    if credit is not None:
        frames.append(credit.reindex(common))
    if move is not None:
        frames.append(move.reindex(common))
    merged = pd.concat(frames, axis=1)
    merged.index.name = "date"
    return merged


def _coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "first_obs":    df.apply(lambda s: s.first_valid_index()),
        "last_obs":     df.apply(lambda s: s.last_valid_index()),
        "nan_pct":      (df.isna().mean() * 100).round(1),
        "coverage_pct": ((1 - df.isna().mean()) * 100).round(1),
    })
    report["first_obs"] = report["first_obs"].dt.date
    report["last_obs"]  = report["last_obs"].dt.date
    return report


def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mu    = df.rolling(window, min_periods=window).mean()
    sigma = df.rolling(window, min_periods=window).std()
    z = (df - mu) / sigma.where(sigma > 1e-6, other=np.nan)
    # When sigma ≈ 0 but the rolling mean is valid (series in a "no-change" regime,
    # e.g., Fed pause or a stably low TED spread): z = 0 (at its rolling mean).
    # Keep NaN only during the warm-up period (where mu itself is NaN).
    z = z.where(mu.isna(), other=z.fillna(0))
    return z


# ── main ───────────────────────────────────────────────────────────────────

def build_panel() -> pd.DataFrame:
    print("── Step 1 · Load and align on business-day intersection")
    raw = _load_and_align()
    print(f"   Merged shape : {raw.shape}  "
          f"({raw.index[0].date()} → {raw.index[-1].date()})")

    print("\n── Step 2 · Forward-fill gaps")
    raw = raw.ffill()

    print(f"\n── Step 3 · Derived series")
    if "move" not in raw.columns and "yield_10y" in raw.columns:
        # Fallback proxy if Yahoo MOVE not fetched yet
        dy = raw["yield_10y"].diff()
        raw["move_proxy"] = dy.rolling(MOVE_PROXY_WINDOW, min_periods=MOVE_PROXY_WINDOW).std()
        print(f"   MOVE not in panel — built move_proxy fallback "
              f"(rolling-{MOVE_PROXY_WINDOW}d stdev of yield_10y first-diff). "
              f"First valid date: {raw['move_proxy'].first_valid_index().date()}")
    elif "move" in raw.columns:
        print(f"   Real MOVE index in panel "
              f"(first valid date: {raw['move'].first_valid_index().date()})")
    else:
        print("   WARNING: neither MOVE nor yield_10y available — skipping bond vol")

    print("\n── Step 4 · Coverage report")
    cov = _coverage_report(raw)
    print(cov.to_string())

    print(f"\n── Step 5 · Drop columns with coverage < {MIN_COVERAGE:.0%}")
    low_cov = cov[cov["coverage_pct"] / 100 < MIN_COVERAGE].index.tolist()
    if low_cov:
        print(f"   Dropping : {low_cov}")
        raw = raw.drop(columns=low_cov)
    else:
        print("   None dropped.")

    print("\n── Step 6 · Restrict to KEEP_COLS allowlist (drop redundant cols)")
    missing = [c for c in KEEP_COLS if c not in raw.columns]
    if missing:
        print(f"   WARNING: requested columns not available: {missing}")
    available_keep = [c for c in KEEP_COLS if c in raw.columns]
    dropped_now = [c for c in raw.columns if c not in available_keep]
    print(f"   Keeping {len(available_keep)} cols, dropping {len(dropped_now)} (redundant/unused)")
    if dropped_now:
        print(f"   Dropped : {dropped_now}")
    raw = raw[available_keep]

    print("\n── Step 7 · Trim leading NaN rows")
    first_complete = raw.dropna(how="any").index[0]
    n_dropped = (raw.index < first_complete).sum()
    raw = raw.loc[first_complete:]
    print(f"   Dropped {n_dropped:,} leading rows  →  panel starts {first_complete.date()}")

    print(f"\n── Step 8 · Rolling z-score  (window = {ROLLING_WINDOW} days)")
    panel = _rolling_zscore(raw, ROLLING_WINDOW)
    before = len(panel)
    panel  = panel.dropna(how="any")
    print(f"   Dropped {before - len(panel):,} warm-up rows  →  z-score start {panel.index[0].date()}")

    print(f"\n   Final panel : {panel.shape}  "
          f"({panel.index[0].date()} → {panel.index[-1].date()})")
    return panel


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    panel = build_panel()

    panel.to_parquet(PANEL_OUT)
    print(f"\nSaved panel → {PANEL_OUT}")

    _write_yaml(panel.columns.tolist())
    print(f"Saved YAML  → {YAML_OUT}")


# ── variable definitions ───────────────────────────────────────────────────

DEFINITIONS: dict[str, dict] = {
    # ── GSW zero-coupon yields ──────────────────────────────────────────────
    "yield_1y": {
        "human_name": "1-Year Zero-Coupon Treasury Yield",
        "source": "Gürkaynak, Sack & Wright (2006) — Federal Reserve",
        "units": "% p.a., continuously compounded",
        "definition": (
            "The yield on a hypothetical 1-year zero-coupon US Treasury bond, "
            "derived from the Nelson-Siegel-Svensson yield-curve model fitted "
            "to off-the-run Treasury prices. Represents the risk-free short rate "
            "at a 1-year horizon."
        ),
    },
    "yield_2y": {
        "human_name": "2-Year Zero-Coupon Treasury Yield",
        "source": "Gürkaynak, Sack & Wright (2006) — Federal Reserve",
        "units": "% p.a., continuously compounded",
        "definition": (
            "Zero-coupon yield at 2 years, estimated from the GSW model. "
            "A key benchmark rate sensitive to near-term monetary policy expectations "
            "and the front end of the yield curve."
        ),
    },
    "yield_5y": {
        "human_name": "5-Year Zero-Coupon Treasury Yield",
        "source": "Gürkaynak, Sack & Wright (2006) — Federal Reserve",
        "units": "% p.a., continuously compounded",
        "definition": (
            "Zero-coupon yield at 5 years. Captures the intermediate segment of the "
            "curve where both monetary policy expectations and the term premium "
            "contribute meaningfully."
        ),
    },
    "yield_10y": {
        "human_name": "10-Year Zero-Coupon Treasury Yield",
        "source": "Gürkaynak, Sack & Wright (2006) — Federal Reserve",
        "units": "% p.a., continuously compounded",
        "definition": (
            "Zero-coupon yield at 10 years — the most widely cited long-run benchmark "
            "rate. Reflects expected future short rates plus a term premium over a "
            "10-year horizon."
        ),
    },
    "yield_30y": {
        "human_name": "30-Year Zero-Coupon Treasury Yield",
        "source": "Gürkaynak, Sack & Wright (2006) — Federal Reserve",
        "units": "% p.a., continuously compounded",
        "definition": (
            "Zero-coupon yield at the long end of the Treasury curve (30 years). "
            "Primarily driven by long-horizon inflation expectations and the term "
            "premium; less sensitive to near-term policy changes. Available from 1985."
        ),
    },
    "slope_2s10s": {
        "human_name": "Yield Curve Slope: 10y minus 2y",
        "source": "Computed from GSW yields",
        "units": "percentage points",
        "definition": (
            "Difference between the 10-year and 2-year zero-coupon yields "
            "(yield_10y − yield_2y). The most commonly watched term-spread indicator; "
            "negative values (inversion) have historically preceded recessions."
        ),
    },
    "slope_5s30s": {
        "human_name": "Yield Curve Slope: 30y minus 5y",
        "source": "Computed from GSW yields",
        "units": "percentage points",
        "definition": (
            "Difference between the 30-year and 5-year zero-coupon yields "
            "(yield_30y − yield_5y). Measures the slope of the long end of the curve, "
            "which is more sensitive to long-run inflation expectations and supply/demand "
            "for duration. Available from 1985."
        ),
    },
    "curvature": {
        "human_name": "Yield Curve Curvature (Butterfly)",
        "source": "Computed from GSW yields",
        "units": "percentage points",
        "definition": (
            "The 'butterfly' spread: 2×yield_5y − yield_2y − yield_10y. "
            "Positive when the 5-year yield lies above the line connecting the 2-year "
            "and 10-year yields (hump-shaped curve); negative when it lies below "
            "(inverted hump). Reflects the relative richness of the belly of the curve."
        ),
    },
    # ── ACM term premium decomposition ──────────────────────────────────────
    "tp_2y": {
        "human_name": "2-Year ACM Term Premium",
        "source": "Adrian, Crump & Moench (2013) — Federal Reserve Bank of New York",
        "units": "% p.a.",
        "definition": (
            "The 2-year term premium from the Adrian-Crump-Moench (ACM) affine "
            "term-structure model. Represents the excess compensation investors demand "
            "for holding a 2-year bond rather than rolling over 1-period bonds; equal to "
            "yield_2y minus the risk-neutral (expected-short-rate) component rny_2y."
        ),
    },
    "tp_5y": {
        "human_name": "5-Year ACM Term Premium",
        "source": "Adrian, Crump & Moench (2013) — Federal Reserve Bank of New York",
        "units": "% p.a.",
        "definition": (
            "The 5-year term premium from the ACM model. Captures duration risk "
            "compensation at an intermediate horizon; typically rises during periods of "
            "elevated inflation uncertainty or heavy Treasury supply."
        ),
    },
    "tp_10y": {
        "human_name": "10-Year ACM Term Premium",
        "source": "Adrian, Crump & Moench (2013) — Federal Reserve Bank of New York",
        "units": "% p.a.",
        "definition": (
            "The 10-year term premium — the most closely watched ACM output. "
            "Measures the compensation investors require for bearing interest-rate risk "
            "over 10 years; can be negative when demand for safe long-duration assets is "
            "very strong (e.g., global quantitative easing periods)."
        ),
    },
    "rny_2y": {
        "human_name": "2-Year Risk-Neutral Yield (Expected Short Rates)",
        "source": "Adrian, Crump & Moench (2013) — Federal Reserve Bank of New York",
        "units": "% p.a.",
        "definition": (
            "The risk-neutral (expectations) component of the 2-year yield from the ACM "
            "model. Equals the average expected path of the overnight rate over the next "
            "2 years; together with tp_2y it sums to yield_2y."
        ),
    },
    "rny_5y": {
        "human_name": "5-Year Risk-Neutral Yield (Expected Short Rates)",
        "source": "Adrian, Crump & Moench (2013) — Federal Reserve Bank of New York",
        "units": "% p.a.",
        "definition": (
            "The risk-neutral component of the 5-year yield. Reflects market expectations "
            "for the average overnight rate over the next 5 years, stripped of the "
            "term-premium component."
        ),
    },
    "rny_10y": {
        "human_name": "10-Year Risk-Neutral Yield (Expected Short Rates)",
        "source": "Adrian, Crump & Moench (2013) — Federal Reserve Bank of New York",
        "units": "% p.a.",
        "definition": (
            "The risk-neutral component of the 10-year yield from the ACM model. "
            "Captures long-run short-rate expectations; movements in this series reflect "
            "changes in perceived future monetary policy stance rather than risk appetite."
        ),
    },
    # ── FRED macro series ────────────────────────────────────────────────────
    "ff_rate": {
        "human_name": "Federal Funds Effective Rate",
        "source": "FRED series DFF — Federal Reserve",
        "units": "% p.a.",
        "definition": (
            "The overnight interbank lending rate at which US depository institutions "
            "trade reserve balances. The Federal Reserve's primary operational target; "
            "directly controlled via open market operations and administered rates."
        ),
    },
    "ff_target_upper": {
        "human_name": "Federal Funds Target Rate — Upper Bound",
        "source": "FRED series DFEDTARU — Federal Reserve",
        "units": "% p.a.",
        "definition": (
            "The upper bound of the Federal Open Market Committee's target range for the "
            "federal funds rate. Published since December 2008 when the Fed moved to a "
            "corridor system; before that date the Fed set a single target rate."
        ),
    },
    "cpi_yoy": {
        "human_name": "CPI All Items — Year-over-Year Change",
        "source": "FRED series CPIAUCSL (monthly) — Bureau of Labor Statistics",
        "units": "% change year-over-year",
        "definition": (
            "12-month percentage change in the Consumer Price Index for All Urban "
            "Consumers (seasonally adjusted). The headline inflation gauge; includes "
            "food and energy. Monthly release forward-filled to daily frequency."
        ),
    },
    "core_pce_yoy": {
        "human_name": "Core PCE Price Index — Year-over-Year Change",
        "source": "FRED series PCEPILFE (monthly) — Bureau of Economic Analysis",
        "units": "% change year-over-year",
        "definition": (
            "12-month percentage change in the Personal Consumption Expenditures price "
            "index excluding food and energy. The Federal Reserve's preferred inflation "
            "measure for its 2 % symmetric target. Monthly, forward-filled to daily."
        ),
    },
    "unemp_rate": {
        "human_name": "Unemployment Rate",
        "source": "FRED series UNRATE (monthly) — Bureau of Labor Statistics",
        "units": "%",
        "definition": (
            "Seasonally adjusted civilian unemployment rate: share of the labour force "
            "actively seeking work but without employment. A key indicator of labour-market "
            "slack; enters the Fed's dual mandate. Monthly, forward-filled to daily."
        ),
    },
    "nfp_mom": {
        "human_name": "Nonfarm Payrolls — Month-over-Month Change",
        "source": "FRED series PAYEMS (monthly) — Bureau of Labor Statistics",
        "units": "thousands of jobs",
        "definition": (
            "Net change in total nonfarm employment from one month to the next. The most "
            "market-moving US labour-market release; large positive prints signal "
            "economic strength, large negative prints signal contraction. Monthly "
            "first-difference, forward-filled to daily."
        ),
    },
    "retail_sales_mom": {
        "human_name": "Advance Retail Sales — Month-over-Month Change",
        "source": "FRED series RSAFS (monthly) — Census Bureau",
        "units": "% change month-over-month",
        "definition": (
            "Monthly percentage change in advance retail and food services sales "
            "(seasonally adjusted). A timely proxy for consumer spending and GDP growth; "
            "highly sensitive to gasoline prices. Available from 1992, forward-filled to "
            "daily."
        ),
    },
    "breakeven_10y": {
        "human_name": "10-Year Breakeven Inflation Rate",
        "source": "FRED series T10YIE — Federal Reserve / Treasury",
        "units": "% p.a.",
        "definition": (
            "Difference between the 10-year nominal Treasury yield and the 10-year TIPS "
            "yield; approximates market-implied average inflation over the next 10 years. "
            "Reflects both inflation expectations and an inflation risk premium. "
            "Available from January 2003."
        ),
    },
    "spread_10y2y": {
        "human_name": "10-Year minus 2-Year Treasury Spread (FRED)",
        "source": "FRED series T10Y2Y — Federal Reserve",
        "units": "percentage points",
        "definition": (
            "Daily spread between constant-maturity 10-year and 2-year Treasury yields "
            "published by the Federal Reserve. Used as a cross-check against the "
            "GSW-derived slope_2s10s; small differences arise because FRED uses "
            "par/coupon-equivalent yields while GSW uses zero-coupon yields."
        ),
    },
    "ig_oas": {
        "human_name": "ICE BofA Investment-Grade Corporate OAS",
        "source": "FRED series BAMLC0A0CM — ICE Data Indices",
        "units": "basis points",
        "definition": (
            "Option-adjusted spread of the ICE BofA US Corporate (Investment Grade) "
            "index over the Treasury curve. A measure of the credit risk premium demanded "
            "by investors to hold IG corporate debt; widens during risk-off episodes and "
            "financial stress. Available on FRED from April 2023."
        ),
    },
    "hy_oas": {
        "human_name": "ICE BofA High-Yield Corporate OAS",
        "source": "FRED series BAMLH0A0HYM2 — ICE Data Indices",
        "units": "basis points",
        "definition": (
            "Option-adjusted spread of the ICE BofA US High Yield index over Treasuries. "
            "A key barometer of risk appetite and financial conditions; historically "
            "spikes sharply ahead of recessions and credit crises. Available on FRED "
            "from April 2023."
        ),
    },
    "hy_vol_proxy": {
        "human_name": "High-Yield Vol Proxy (ICE BofA Euro HY OAS)",
        "source": "FRED series BAMLHE00EHYIEY — ICE Data Indices",
        "units": "% yield-to-worst",
        "definition": (
            "Yield-to-worst of the ICE BofA Euro High Yield index, used as a proxy for "
            "cross-market credit volatility and risk sentiment. Complements the US HY OAS "
            "by capturing European credit conditions; co-moves with hy_oas during global "
            "risk-off events. Available on FRED from April 2023."
        ),
    },
    "ism_pmi": {
        "human_name": "Manufacturing Employment / ISM PMI Proxy",
        "source": "FRED series MANEMP — Bureau of Labor Statistics",
        "units": "thousands of persons",
        "definition": (
            "All-employees count in the manufacturing sector (seasonally adjusted). "
            "Used as a proxy for manufacturing activity in the absence of a long FRED "
            "history for the ISM PMI composite. Declines signal industrial contraction; "
            "the series is a component of the Conference Board Leading Economic Index."
        ),
    },
    "ted_spread": {
        "human_name": "TED Spread",
        "source": "FRED series TEDRATE — Federal Reserve (discontinued March 2023); "
                  "reconstructed as DTB3 minus USD3MTD156N after that date",
        "units": "percentage points",
        "definition": (
            "Difference between 3-month USD LIBOR and the 3-month US Treasury bill yield. "
            "A measure of perceived credit risk in the banking system and interbank "
            "funding stress; spikes sharply during financial crises (e.g., 2008, 2020). "
            "TEDRATE was discontinued in March 2023; earlier observations use the "
            "official FRED series."
        ),
    },
    # ── bond-market additions (2026-05-12) ──────────────────────────────────
    "vix": {
        "human_name": "CBOE Volatility Index (VIX)",
        "source": "FRED series VIXCLS — CBOE",
        "units": "% p.a., 30-day implied",
        "definition": (
            "30-day option-implied volatility of the S&P 500 index. The standard "
            "equity-market 'fear gauge'; spikes during risk-off episodes. Included "
            "for stock-bond cross-asset regime detection (Campbell, Pflueger & "
            "Viceira 2020 motivate the cross-asset linkage)."
        ),
    },
    "move": {
        "human_name": "ICE BofAML MOVE Index",
        "source": "Yahoo Finance ^MOVE — ICE Data Indices",
        "units": "basis points (annualised yield vol)",
        "definition": (
            "1-month yield-curve-weighted average of normalised Treasury option-implied "
            "volatilities (2y/5y/10y/30y). The bond-market analogue of VIX: the "
            "standard 'fear gauge' for US rate markets. Spikes during dash-for-cash "
            "episodes (Mar 2020, ~265 bp), aggressive hiking surprises (Mar 2022), "
            "and banking stress (Mar 2023, SVB). Available from 2003-01-02."
        ),
    },
    "move_proxy": {
        "human_name": "Realised Volatility Proxy for the MOVE Index",
        "source": "Computed from GSW yield_10y",
        "units": "% p.a., daily",
        "definition": (
            "Rolling 30-day standard deviation of first-differences of the 10-year "
            "zero-coupon yield. Backward-looking proxy for the MOVE Index, used as "
            "a fallback when Yahoo Finance MOVE data is unavailable. Available "
            "from 1971-09 (the start of the GSW series)."
        ),
    },
    "fed_assets": {
        "human_name": "Federal Reserve Total Assets",
        "source": "FRED series WALCL — H.4.1 weekly release, forward-filled to daily",
        "units": "millions of USD",
        "definition": (
            "Sum of all assets on the Fed's balance sheet from the H.4.1 weekly "
            "release. Direct measure of the QE/QT framework: expanding under QE1/2/3 "
            "(2008-2014) and QE-infinity (2020-2022), contracting under QT "
            "(2017-2019, 2022-onward). The variable Stage A/B / PCMCI cannot read "
            "from ff_rate alone during ZLB periods."
        ),
    },
    "credit_baa_aaa": {
        "human_name": "Moody's Baa minus Aaa Corporate Bond Yield Spread",
        "source": "FRED series BAA - AAA, daily",
        "units": "percentage points",
        "definition": (
            "Difference between Moody's seasoned Baa and Aaa corporate bond yields. "
            "Captures credit-quality differentiation independent of the Treasury "
            "curve (since both legs are corporate). Rises during risk-off episodes "
            "(2008, 2011 Eurozone, 2016 oil crash, 2020 COVID) and complements TED "
            "spread which captures interbank funding stress instead."
        ),
    },
}


def _write_yaml(columns: list[str]) -> None:
    output: dict = {"columns": {}}
    missing = []
    for col in columns:
        if col in DEFINITIONS:
            output["columns"][col] = DEFINITIONS[col]
        else:
            missing.append(col)
            output["columns"][col] = {
                "human_name": col,
                "source": "unknown",
                "units": "unknown",
                "definition": "No definition provided.",
            }
    if missing:
        print(f"   WARNING: no definition for columns: {missing}")
    with open(YAML_OUT, "w") as fh:
        yaml.dump(output, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
