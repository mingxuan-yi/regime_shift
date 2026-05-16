"""
Rolling-window structural break detection using PCMCI (tigramite).

For each rolling window of WINDOW trading days (stepped by STEP):
  1. Run PCMCI + ParCorr to find significant lagged causal links.
  2. Compare to the previous window's edge set via Jaccard distance.
  3. Flag a regime change when Jaccard distance exceeds THRESHOLD.

Outputs
-------
data/processed/regimes.parquet      — one row per window (scores + flags)
data/processed/regime_periods.json  — list of {start, end, n_windows}
data/processed/pcmci_checkpoint.pkl — resumable intermediate results

Usage
-----
    uv run python src/causal/regime_detection.py
    uv run python src/causal/regime_detection.py --threshold 0.4
    uv run python src/causal/regime_detection.py --reset   # clear checkpoint
"""

import argparse
import json
import pathlib
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = pathlib.Path(__file__).parents[2]
PANEL_PATH  = ROOT / "data" / "processed" / "panel_daily.parquet"
REGIMES_OUT = ROOT / "data" / "processed" / "regimes.parquet"
PERIODS_OUT = ROOT / "data" / "processed" / "regime_periods.json"
CHECKPOINT  = ROOT / "data" / "processed" / "pcmci_checkpoint.pkl"

# ── defaults ───────────────────────────────────────────────────────────────
WINDOW    = 200    # trading days per window
STEP      = 20     # days to advance between windows
TAU_MAX   = 5      # maximum lag for PCMCI
ALPHA     = 0.05   # significance threshold (PC step + MCI test)
THRESHOLD = 0.80   # Jaccard distance threshold for flagging a regime change
# NOTE: adjacent 200-day windows shifted by 20 days have a natural Jaccard
# distance baseline of ~0.38–0.66 (mean ≈ 0.66) due to edge-set churn from
# the rolling step.  A threshold of 0.30 flags almost every window.
# Empirically, 0.80 (≈ top 10th percentile) isolates genuinely anomalous
# structural breaks.  Use --threshold to override.


# ── PCMCI helpers ──────────────────────────────────────────────────────────

def _run_pcmci(window_data: np.ndarray, var_names: list[str]
               ) -> tuple[set[tuple[int, int, int]], dict]:
    """
    Run PCMCI on a single window.

    Returns
    -------
    links : set of (parent_idx, child_idx, lag) for significant links
    pvals : dict mapping the same tuples to their p-values
    """
    T, N = window_data.shape
    dataframe = pp.DataFrame(
        window_data,
        datatime=np.arange(T),
        var_names=var_names,
    )
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    results = pcmci.run_pcmci(
        tau_min=1,
        tau_max=TAU_MAX,
        pc_alpha=ALPHA,
        alpha_level=ALPHA,
    )

    # p_matrix[j, i, tau] = p-value for X_i(t-tau) -> X_j(t)
    p_matrix = results["p_matrix"]
    links: set[tuple[int, int, int]] = set()
    pvals: dict[tuple[int, int, int], float] = {}

    for j in range(N):          # child (effect)
        for i in range(N):      # parent (cause)
            if i == j:
                continue        # skip self-links
            for tau in range(1, TAU_MAX + 1):   # lagged links only
                p = p_matrix[j, i, tau]
                if not np.isnan(p) and p < ALPHA:
                    links.add((i, j, tau))
                    pvals[(i, j, tau)] = float(p)

    return links, pvals


# ── Jaccard distance ────────────────────────────────────────────────────────

def jaccard_distance(a: set, b: set) -> float:
    """1 − |A ∩ B| / |A ∪ B|.  Returns 0 when both sets are empty."""
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


# ── window grid ────────────────────────────────────────────────────────────

def _make_windows(n: int) -> list[tuple[int, int]]:
    """Return (start, end) index pairs covering `n` observations."""
    wins = []
    start = 0
    while start + WINDOW <= n:
        wins.append((start, start + WINDOW))
        start += STEP
    return wins


# ── checkpoint I/O ─────────────────────────────────────────────────────────

def _load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "rb") as f:
            return pickle.load(f)
    return {}


def _save_checkpoint(data: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "wb") as f:
        pickle.dump(data, f)


# ── regime period extraction ────────────────────────────────────────────────

def _build_periods(regimes: pd.DataFrame) -> list[dict]:
    """
    Convert a boolean is_regime_change column into a list of stable periods.

    Each period runs from either the panel start (or previous change date)
    up to (but not including) the next regime change date.
    """
    panel_start = regimes["window_start"].iloc[0]
    panel_end   = regimes.index[-1]

    change_dates = [panel_start] + list(
        regimes.loc[regimes["is_regime_change"]].index
    ) + [panel_end]

    periods = []
    for i in range(len(change_dates) - 1):
        s, e = change_dates[i], change_dates[i + 1]
        mask = (regimes.index >= s) & (regimes.index <= e)
        periods.append({
            "start":     str(s.date()),
            "end":       str(e.date()),
            "n_windows": int(mask.sum()),
        })
    return periods


# ── main ───────────────────────────────────────────────────────────────────

def run(threshold: float = THRESHOLD, reset: bool = False) -> pd.DataFrame:
    print("Loading panel …")
    df        = pd.read_parquet(PANEL_PATH)
    data      = df.values.astype(np.float64)
    index     = df.index
    var_names = list(df.columns)
    N         = data.shape[1]

    windows   = _make_windows(len(data))
    n_windows = len(windows)
    print(f"Panel : {df.shape}   {index[0].date()} → {index[-1].date()}")
    print(f"Windows : {n_windows}  (size={WINDOW}, step={STEP}, "
          f"tau_max={TAU_MAX}, alpha={ALPHA})")

    # Load / reset checkpoint
    if reset and CHECKPOINT.exists():
        CHECKPOINT.unlink()
        print("Checkpoint cleared.")
    cache: dict[int, dict] = _load_checkpoint()
    n_cached = len(cache)
    if n_cached:
        print(f"Resuming: {n_cached}/{n_windows} windows already computed.")

    # ── main loop ──────────────────────────────────────────────────────────
    t_total = time.time()
    for idx, (start, end) in enumerate(windows):
        if idx in cache:
            continue

        t0    = time.time()
        links, pvals = _run_pcmci(data[start:end], var_names)
        elapsed = time.time() - t0

        cache[idx] = {
            "start":  index[start],
            "end":    index[end - 1],
            "links":  links,
            "pvals":  pvals,
            "n_links": len(links),
        }

        done      = sum(1 for k in cache if k >= n_cached)
        remaining = n_windows - (idx + 1)
        rate      = (time.time() - t_total) / max(done, 1)
        eta_min   = rate * remaining / 60

        print(
            f"  [{idx+1:3d}/{n_windows}]"
            f"  {index[start].date()} → {index[end-1].date()}"
            f"  links={len(links):3d}"
            f"  {elapsed:.1f}s"
            f"  ETA {eta_min:.0f} min"
        )

        if (idx + 1) % 10 == 0:
            _save_checkpoint(cache)

    _save_checkpoint(cache)
    print(f"\nAll windows done in {(time.time() - t_total)/60:.1f} min")

    # ── build results dataframe ────────────────────────────────────────────
    rows = []
    for idx in range(n_windows):
        w    = cache[idx]
        prev = cache[idx - 1] if idx > 0 else None

        jd = (
            jaccard_distance(prev["links"], w["links"])
            if prev is not None
            else np.nan
        )
        rows.append({
            "window_idx":       idx,
            "window_start":     w["start"],
            "window_end":       w["end"],
            "n_links":          w["n_links"],
            "jaccard_dist":     round(jd, 4) if not np.isnan(jd) else np.nan,
            "is_regime_change": bool(not np.isnan(jd) and jd > threshold),
        })

    regimes = pd.DataFrame(rows)
    regimes.index = pd.DatetimeIndex(regimes["window_end"])
    regimes.index.name = "date"

    # ── regime periods ─────────────────────────────────────────────────────
    periods = _build_periods(regimes)

    # ── save ───────────────────────────────────────────────────────────────
    REGIMES_OUT.parent.mkdir(parents=True, exist_ok=True)
    regimes.to_parquet(REGIMES_OUT)
    with open(PERIODS_OUT, "w") as f:
        json.dump(periods, f, indent=2, default=str)

    n_changes = regimes["is_regime_change"].sum()
    print(f"\nRegime changes detected : {n_changes}")
    print(f"Stable periods          : {len(periods)}")
    print(f"\nChange dates (Jaccard > {threshold}):")
    for _, row in regimes[regimes["is_regime_change"]].iterrows():
        print(f"  {row['window_end'].date()}  Jaccard = {row['jaccard_dist']:.3f}"
              f"  links {int(cache[int(row['window_idx'])-1]['n_links'])}"
              f" → {int(row['n_links'])}")

    print(f"\nSaved → {REGIMES_OUT}")
    print(f"Saved → {PERIODS_OUT}")

    return regimes


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rolling PCMCI structural break detection"
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Jaccard distance threshold for regime change (default {THRESHOLD})"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear the checkpoint and recompute from scratch"
    )
    args = parser.parse_args()
    run(threshold=args.threshold, reset=args.reset)
