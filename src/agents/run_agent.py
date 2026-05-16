"""
Quick runner: execute the CausalAgent on one regime and save results to JSON.

Usage
-----
    uv run python src/agents/run_agent.py
    uv run python src/agents/run_agent.py --regime 9   # COVID onset
"""

import argparse
import json
import logging
import pathlib
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()  # reads .env before CausalAgent imports os.environ

from causal_agent import CausalAgent  # noqa: E402 (after load_dotenv)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = pathlib.Path(__file__).parents[2]

REGIME_DESCRIPTIONS = {
    1:  "Post-GFC recovery: gradual deleveraging, suppressed volatility, QE1/QE2, low rates",
    2:  "US downgrade / Eurozone crisis: sovereign stress, risk-off, safe-haven demand",
    3:  "Taper tantrum aftermath: rising long-end rates, EM outflows, curve steepening",
    4:  "Oil crash / EM stress: commodity collapse, USD strength, EM currency pressure",
    5:  "CNY devaluation / Brexit / Trump: political uncertainty, reflation trade",
    6:  "Rate hike cycle / Vol shock: VIX spike (Feb 2018), flattening yield curve",
    7:  "Trade war / Q4 2018 selloff: credit spread widening, risk-off, Fed pivot",
    8:  "Pre-COVID slowdown: late-cycle, muted volatility, repo stress (Sep 2019)",
    9:  "COVID onset: extreme volatility, flight to safety, Fed emergency cuts, "
        "credit spreads spiking, yield curve collapse",
    10: "COVID peak causal density: massive fiscal/monetary stimulus, yield curve control fears",
    11: "COVID regime unwind I: vaccine optimism, reflation, steepening curve",
    12: "COVID regime unwind II: breakeven inflation surge, taper speculation",
    13: "COVID regime unwind III: Fed pivot to tightening, hiking cycle begins",
    14: "Post-COVID / SVB crisis: banking stress, credit tightening, disinflation",
    15: "Soft landing / carry unwind: Yen carry unwind (Aug 2024), rate cuts begin",
    16: "Late-cycle links collapse: low volatility, narrow spreads, AI-driven equities",
    17: "Tariff shock recovery: tariff uncertainty, growth fears, renewed safe-haven bid",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=int, default=9,
                        help="Regime number 1-17 (default: 9 — COVID onset)")
    args = parser.parse_args()

    regime_n = args.regime
    if regime_n not in REGIME_DESCRIPTIONS:
        print(f"Unknown regime {regime_n}. Choose 1-17.")
        sys.exit(1)

    regime_desc = REGIME_DESCRIPTIONS[regime_n]
    print(f"\nRegime {regime_n}: {regime_desc}\n")

    # Load variable definitions
    yaml_path = ROOT / "data" / "processed" / "variable_definitions.yaml"
    with open(yaml_path) as f:
        var_defs = yaml.safe_load(f)
    if "columns" in var_defs:
        var_defs = var_defs["columns"]

    agent = CausalAgent()
    result = agent.run(
        variable_definitions=var_defs,
        regime_description=regime_desc,
    )

    # Save results
    out_dir = ROOT / "data" / "processed" / "causal_dags"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dag_regime_{regime_n:02d}.json"

    payload = {
        "regime":      regime_n,
        "description": regime_desc,
        "n_edges":     result["n_edges"],
        "is_dag":      result["is_dag"],
        "edges":       result["edges"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Save intermediates for notebook visualisation
    inter_path = out_dir / f"intermediates_regime_{regime_n:02d}.json"
    with open(inter_path, "w") as f:
        json.dump(result["intermediates"], f, indent=2)

    print(f"\n{'='*60}")
    print(f"Regime {regime_n} DAG — {result['n_edges']} edges  |  is_dag={result['is_dag']}")
    print(f"{'='*60}")
    for e in result["edges"]:
        print(f"  {e['from']:20s} → {e['to']:20s}  {e.get('justification','')[:70]}")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
