"""
Ground-truth DAGs and variable definitions for synthetic bond-market regime
validation.

Two regimes capture a structural shift in bond-market causal mechanics:

  - Regime 1 (ZLB / QE, 15 edges): monetary base expansion is the dominant
    driver, long-term yields drive short-term yields via duration-channel
    transmission, the Taylor rule is broken at ZLB so policy rate does not
    respond to inflation. QE is also a commitment signal that lifts growth
    expectations. Asset valuations are supported by liquidity (money_growth,
    term_prem, credit_sprd) but also reflect fundamentals (output, expectations).

  - Regime 2 (Tightening / QT, 18 edges): policy rate is the dominant driver,
    short-term yields lead long-term yields (curve flattening / inversion),
    Taylor rule is active so output, inflation, and growth expectations all
    feed into the policy rate. QT mirrors QE: balance-sheet runoff raises
    long yields and rebuilds the term premium. Asset valuations reflect both
    higher discount rates and earnings/growth fundamentals.

Both edge lists are validated as DAGs at import time. The DAG encodes the
structural causal links; SEM coefficient *signs* (e.g. QE → lower long
yields vs QT → higher long yields) are a property of the data-generating
process, not of the DAG.
"""

from __future__ import annotations

import networkx as nx

# ── variable definitions ──────────────────────────────────────────────────────

VARIABLES: dict[str, str] = {
    "pol_rate":     "Policy Rate",
    "short_yield":  "Short-Term Bond Yield",
    "long_yield":   "Long-Term Bond Yield",
    "term_prem":    "Term Premium",
    "inflation":    "Price Growth Rate",
    "output":       "Output Growth Rate",
    "credit_sprd":  "Credit Risk Spread",
    "asset_val":    "Asset Valuation Index",
    "money_growth": "Monetary Base Growth",
    "expectations": "Growth Expectations Index",
}

_DEFINITIONS: list[str] = [
    "The central bank's overnight benchmark lending rate, set by the monetary authority.",
    "The yield on government bonds with maturity of 1–2 years, reflecting near-term rate expectations.",
    "The yield on government bonds with maturity of 10 years, incorporating term premia and inflation expectations.",
    "The compensation demanded by investors for bearing duration risk; long minus short yield adjusted for expectations.",
    "The broad rate of increase in the general price level, typically measured on a 12-month basis.",
    "The rate of change in aggregate real economic activity, measured over a quarter or year.",
    "The yield spread between investment-grade corporate bonds and equivalent sovereign bonds.",
    "A composite index of equity and real-asset valuations relative to fundamentals.",
    "The rate of growth of the central bank's balance sheet / monetary base.",
    "A survey- or market-based measure of expected future economic growth over a 12-month horizon.",
]

VARIABLE_DEFINITIONS: dict[str, dict[str, str]] = {
    var: {
        "human_name": human,
        "units":      "standardised z-score",
        "definition": _def,
    }
    for (var, human), _def in zip(VARIABLES.items(), _DEFINITIONS)
}

VARS: list[str] = list(VARIABLES.keys())
N_VAR: int = len(VARS)
VAR_IDX: dict[str, int] = {v: i for i, v in enumerate(VARS)}

# ── ground-truth DAG: Regime 1 (ZLB / QE) ────────────────────────────────────
# money_growth dominates; long-term yield drives short-term; term premium is
# the main transmission channel; policy rate is pinned near zero and does NOT
# respond to inflation (Taylor rule is broken at ZLB).

EDGES_R1: list[tuple[str, str]] = [
    # ── monetary policy / liquidity channels ──────────────────────────────
    ("money_growth", "long_yield"),     # QE suppresses long-term yields (duration absorption)
    ("money_growth", "term_prem"),      # QE compresses term premium
    ("money_growth", "asset_val"),      # portfolio rebalancing lifts asset prices
    ("money_growth", "expectations"),   # QE size signals commitment, lifts growth expectations
    ("pol_rate",     "short_yield"),    # policy rate anchors short end (pinned at ZLB but still structural)
    # ── yield curve & term structure ──────────────────────────────────────
    ("term_prem",    "long_yield"),     # term premium feeds into long yields
    ("long_yield",   "short_yield"),    # key: QE/forward-guidance — long drives short
    ("expectations", "long_yield"),     # growth expectations → real-rate component of long yields
    # ── financial conditions ──────────────────────────────────────────────
    ("long_yield",   "credit_sprd"),    # lower yields → reach-for-yield compresses spreads
    # ── asset valuations: liquidity + fundamentals ────────────────────────
    ("term_prem",    "asset_val"),      # lower term premium → lower discount rate → higher valuations
    ("credit_sprd",  "asset_val"),      # compressed spreads support corporate valuations
    ("output",       "asset_val"),      # current earnings drive valuations (V = CF / (r-g))
    ("expectations", "asset_val"),      # forward earnings / sentiment drive valuations
    # ── real economy ──────────────────────────────────────────────────────
    ("output",       "inflation"),      # Phillips curve (weakened at ZLB but present)
    ("output",       "expectations"),   # stronger growth raises forward expectations
]

# ── ground-truth DAG: Regime 2 (Tightening / QT) ─────────────────────────────
# policy rate dominant; short-term leads long-term (yield curve inversion risk);
# several edges reverse direction vs Regime 1; Taylor rule is active so output,
# inflation, and growth expectations all feed into the policy rate. QT mirrors
# QE: balance-sheet runoff raises long yields and rebuilds the term premium.

EDGES_R2: list[tuple[str, str]] = [
    # ── real economy & expectations ───────────────────────────────────────
    ("output",       "inflation"),      # Phillips curve (active in tightening regime)
    ("output",       "expectations"),   # stronger growth raises forward growth expectations
    ("inflation",    "expectations"),   # high inflation signals stagflation risk to growth (adaptive)
    # ── Taylor rule (active in tightening regime) ─────────────────────────
    ("output",       "pol_rate"),       # output gap: strong growth triggers tightening
    ("inflation",    "pol_rate"),       # inflation gap: high inflation triggers hikes
    ("expectations", "pol_rate"),       # forward-looking output-gap component
    # ── monetary policy / liquidity channels (QT mirrors QE) ──────────────
    ("pol_rate",     "short_yield"),    # rate hikes anchor short end
    ("pol_rate",     "credit_sprd"),    # higher rates widen spreads (signaling / risk premium)
    ("money_growth", "long_yield"),     # QT raises long yields (reverse of QE absorption)
    ("money_growth", "term_prem"),      # QT rebuilds term premium (reverse of QE compression)
    # ── yield curve reversal ──────────────────────────────────────────────
    ("short_yield",  "long_yield"),     # key reversal: short now drives long (curve flattens / inverts)
    ("long_yield",   "term_prem"),      # rising long yields rebuild term premium (R1 direction reversed)
    # ── financial conditions ──────────────────────────────────────────────
    ("short_yield",  "credit_sprd"),    # rising bank funding costs widen spreads
    # ── asset valuations: discount rate + fundamentals ────────────────────
    ("long_yield",   "asset_val"),      # higher discount rate depresses valuations
    ("credit_sprd",  "asset_val"),      # wider spreads tighten financial conditions
    ("money_growth", "asset_val"),      # QT removes balance-sheet asset price support
    ("output",       "asset_val"),      # current earnings drive valuations
    ("expectations", "asset_val"),      # forward earnings / sentiment drive valuations
]

# ── DAG validity check at import time ────────────────────────────────────────

def _assert_dag(edges: list[tuple[str, str]], name: str) -> None:
    G = nx.DiGraph()
    G.add_nodes_from(VARS)
    G.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f"{name} is not a DAG; cycles: {cycles}")


_assert_dag(EDGES_R1, "EDGES_R1")
_assert_dag(EDGES_R2, "EDGES_R2")


__all__ = [
    "VARIABLES",
    "VARIABLE_DEFINITIONS",
    "VARS",
    "N_VAR",
    "VAR_IDX",
    "EDGES_R1",
    "EDGES_R2",
]
