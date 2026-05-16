"""Smoke test: import every baseline, run it on the 5-variable FOMC panel,
verify each returns a sorted list of pd.Timestamp and time the call."""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.baselines import bai_perron, binseg, bocpd, cusum, markov_switching, pelt

WINDOW_START = pd.Timestamp("2010-01-01")
WINDOW_END   = pd.Timestamp("2024-12-31")
KEY_VARS = ["yield_2y", "yield_10y", "ff_rate", "cpi_yoy", "breakeven_10y"]

panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
panel = panel[KEY_VARS].dropna()
panel = panel.loc[(panel.index >= WINDOW_START) & (panel.index <= WINDOW_END)]
print(f"Panel: {panel.shape}  {panel.index.min().date()} → {panel.index.max().date()}\n")

baselines = [
    ("PELT",           pelt),
    ("BinSeg",         binseg),
    ("Bai-Perron",     bai_perron),
    ("CUSUM",          cusum),
    ("BOCPD",          bocpd),
    ("MarkovSwitch",   markov_switching),
]

print(f"{'baseline':14s}  {'runtime':>10s}  {'n_CP':>5s}  first / last CP")
print("-" * 80)
for name, mod in baselines:
    t0 = time.time()
    cps = mod.detect(panel)
    dt = time.time() - t0
    assert isinstance(cps, list), f"{name} did not return a list"
    assert all(isinstance(d, pd.Timestamp) for d in cps), f"{name}: non-timestamp in output"
    assert cps == sorted(cps), f"{name}: output not sorted"
    first = cps[0].strftime("%Y-%m-%d") if cps else "—"
    last  = cps[-1].strftime("%Y-%m-%d") if cps else "—"
    print(f"{name:14s}  {dt:>9.2f}s  {len(cps):>5d}  {first}  →  {last}")
