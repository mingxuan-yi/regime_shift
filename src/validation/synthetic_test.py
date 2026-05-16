"""
Ground-truth validation of the LLM causal discovery pipeline on synthetic data.

Two ground-truth DAGs with 10 generic-named variables are defined:
  - Regime 1 (~11 edges): ZLB/QE-like structure where long-term yield drives
    short-term, duration premium is the main transmission channel.
  - Regime 2 (~13 edges): Tightening-like structure where policy rate is the
    dominant driver, short-term leads long-term, some edges reverse direction.

Data is generated via a linear SEM with 500 time steps per regime.
The pipeline (CausalAgent → CausalValidator) is run on each regime's data
and compared against ground truth.

Outputs
-------
  outputs/results/synthetic_validation.csv
  outputs/figures/synthetic_gt_dags.png
  outputs/figures/synthetic_discovered_dags.png
  outputs/figures/synthetic_metrics.png

Usage
-----
    uv run python src/validation/synthetic_test.py
    uv run python src/validation/synthetic_test.py --no-cache   # force re-run
"""

import argparse
import json
import logging
import pathlib
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")

# Adjust path so imports from src/ work when run directly
ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src" / "agents"))
sys.path.insert(0, str(ROOT / "src" / "causal"))

from causal_agent import CausalAgent          # noqa: E402
from validate_dag import CausalValidator      # noqa: E402

from regime_dags import (                     # noqa: E402
    EDGES_R1,
    EDGES_R2,
    N_VAR,
    VAR_IDX,
    VARIABLE_DEFINITIONS,
    VARIABLES,
    VARS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── directories ───────────────────────────────────────────────────────────────

OUT_RESULTS = ROOT / "outputs" / "results"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_DAGS    = ROOT / "outputs" / "dags"
CACHE_DIR   = ROOT / "outputs" / "dags"

for d in (OUT_RESULTS, OUT_FIGURES, OUT_DAGS):
    d.mkdir(parents=True, exist_ok=True)

# ── SEM data generation ───────────────────────────────────────────────────────

def _edges_to_beta(edges: list[tuple[str, str]], rng: np.random.Generator) -> np.ndarray:
    """Build NxN coefficient matrix B where B[i,j] means j → i."""
    B = np.zeros((N_VAR, N_VAR))
    for src, tgt in edges:
        i, j = VAR_IDX[tgt], VAR_IDX[src]
        B[i, j] = rng.uniform(0.08, 0.30)
    return B


def generate_sem(
    edges: list[tuple[str, str]],
    n_steps: int = 500,
    ar_coef: float = 0.75,
    noise_std: float = 0.50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a linear SEM with one lag:
        X_i(t) = ar_coef * X_i(t-1) + sum_j B[i,j] * X_j(t-1) + eps_i(t)

    Returns a DataFrame with columns = VARS and n_steps rows (burn-in of 200
    steps discarded so the series is stationary before the returned sample).
    """
    rng  = np.random.default_rng(seed)
    B    = _edges_to_beta(edges, rng)
    burn = 200
    X    = np.zeros((burn + n_steps, N_VAR))
    eps  = rng.normal(0, noise_std, size=(burn + n_steps, N_VAR))

    for t in range(1, burn + n_steps):
        X[t] = ar_coef * X[t - 1] + B @ X[t - 1] + eps[t]

    data = pd.DataFrame(X[burn:], columns=VARS)
    # z-score so the agent sees standardised input (as with real panel data)
    data = (data - data.mean()) / data.std()
    return data

# ── regime description (hand-crafted, data-grounded but generic) ──────────────

def _build_regime_description(regime: int, data: pd.DataFrame) -> str:
    means = data.mean()
    top   = means.abs().nlargest(4)
    parts = []
    for var, val in top.items():
        human = VARIABLES[var]
        direction = "elevated" if val > 0 else "depressed"
        parts.append(f"{human} ({val:+.2f}σ, {direction})")
    cond_str = "; ".join(parts)

    if regime == 1:
        return (
            f"Regime 1 (ZLB/QE): monetary base expansion is the dominant driver with {cond_str}. "
            "Long-term yields are compressed by asset purchases, pulling short-term yields lower "
            "through duration-channel transmission. Policy rate is effectively pinned at zero with "
            "growth expectations subdued."
        )
    else:
        return (
            f"Regime 2 (Tightening): aggressive policy rate increases anchor the short end with {cond_str}. "
            "Short-term yields lead long-term yields upward, flattening then inverting the curve. "
            "Credit spreads widen as policy tightens, depressing asset valuations while inflation "
            "expectations feed back into the policy rate."
        )

# ── metrics ───────────────────────────────────────────────────────────────────

def compute_edge_metrics(
    gt_edges: list[tuple[str, str]],
    pred_edges: list[tuple[str, str]],
) -> dict:
    """
    Precision, Recall, F1 treating edge identity as (from, to) pair.
    Direction accuracy: among correctly identified undirected edges, fraction
    that have the correct direction.
    SHD (Structural Hamming Distance): |missing| + |extra| + |reversed|.
    """
    gt_set   = set(gt_edges)
    pred_set = set(pred_edges)

    gt_undir   = {frozenset(e) for e in gt_edges}
    pred_undir = {frozenset(e) for e in pred_edges}

    tp_dir = gt_set & pred_set
    fp_dir = pred_set - gt_set
    fn_dir = gt_set - pred_set

    # Reversed: pred has (B,A) but GT has (A,B)
    reversed_edges = {(b, a) for (a, b) in gt_set if (b, a) in pred_set}
    n_reversed     = len(reversed_edges)

    precision = len(tp_dir) / max(len(pred_set), 1)
    recall    = len(tp_dir) / max(len(gt_set), 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    # Direction accuracy: correct direction among shared undirected edges
    shared_undir = gt_undir & pred_undir
    if shared_undir:
        correctly_directed = sum(
            1 for e in shared_undir
            if tuple(e) in gt_set or (
                len(e) == 2 and list(e) == list(e)  # always True; check forward
            )
        )
        # recount properly
        correctly_directed = sum(
            1 for a, b in gt_edges
            if frozenset((a, b)) in shared_undir and (a, b) in pred_set
        )
        dir_accuracy = correctly_directed / len(shared_undir)
    else:
        dir_accuracy = float("nan")

    n_missing = len(fn_dir) - n_reversed   # fn_dir includes reversed as fn
    n_missing = max(n_missing, 0)
    n_extra   = len(fp_dir) - n_reversed
    n_extra   = max(n_extra, 0)
    shd       = n_missing + n_extra + n_reversed

    return {
        "n_gt":          len(gt_edges),
        "n_pred":        len(pred_edges),
        "n_tp":          len(tp_dir),
        "n_fp":          len(fp_dir),
        "n_fn":          len(fn_dir),
        "n_reversed":    n_reversed,
        "precision":     round(precision, 4),
        "recall":        round(recall, 4),
        "f1":            round(f1, 4),
        "dir_accuracy":  round(dir_accuracy, 4) if not np.isnan(dir_accuracy) else float("nan"),
        "shd":           shd,
    }


def compute_regime_change_accuracy(
    gt_r1: list[tuple],
    gt_r2: list[tuple],
    pred_r1: list[tuple],
    pred_r2: list[tuple],
) -> dict:
    """
    Regime-change detection: measure how well the pipeline captures the
    structural shift between R1 and R2.

    Reports two complementary metrics:
      ── Skeleton (undirected XOR) ──
        Treats each edge as an unordered pair.  Catches edges that
        appeared / disappeared between regimes.  Direction reversals
        (e.g. A→B in R1, B→A in R2) collapse to the same frozenset and
        are NOT counted as a change.

      ── Directed (directed XOR) ──
        Operates on (from, to) tuples directly.  A reversal contributes
        TWO changes (the old direction disappears AND the new direction
        appears), so a correctly-predicted reversal scores as 2 TP, while
        a missed reversal contributes 2 FN.  This is the metric that
        actually rewards detecting causal-direction shifts between regimes.
    """
    # ── skeleton (undirected) — original metric, kept for backward compat ──
    gt_change_undir   = {frozenset(e) for e in gt_r1} ^ {frozenset(e) for e in gt_r2}
    pred_change_undir = {frozenset(e) for e in pred_r1} ^ {frozenset(e) for e in pred_r2}

    tp_u = gt_change_undir & pred_change_undir
    p_u  = len(tp_u) / max(len(pred_change_undir), 1)
    r_u  = len(tp_u) / max(len(gt_change_undir), 1)
    f1_u = 2 * p_u * r_u / max(p_u + r_u, 1e-9)

    # ── directed — also flags reversals as changes ─────────────────────────
    gt_change_dir   = set(gt_r1) ^ set(gt_r2)
    pred_change_dir = set(pred_r1) ^ set(pred_r2)

    tp_d = gt_change_dir & pred_change_dir
    p_d  = len(tp_d) / max(len(pred_change_dir), 1)
    r_d  = len(tp_d) / max(len(gt_change_dir), 1)
    f1_d = 2 * p_d * r_d / max(p_d + r_d, 1e-9)

    return {
        # skeleton (undirected)  — legacy keys
        "n_gt_changes":     len(gt_change_undir),
        "n_pred_changes":   len(pred_change_undir),
        "n_tp_changes":     len(tp_u),
        "change_precision": round(p_u, 4),
        "change_recall":    round(r_u, 4),
        "change_f1":        round(f1_u, 4),
        # directed — new keys, catch direction reversals
        "n_gt_changes_dir":     len(gt_change_dir),
        "n_pred_changes_dir":   len(pred_change_dir),
        "n_tp_changes_dir":     len(tp_d),
        "change_precision_dir": round(p_d, 4),
        "change_recall_dir":    round(r_d, 4),
        "change_f1_dir":        round(f1_d, 4),
    }

# ── visualisation helpers ─────────────────────────────────────────────────────

_LAYOUT_LEVELS = {
    # top: monetary policy drivers
    "pol_rate":     (0.5,  1.0),
    "money_growth": (0.0,  1.0),
    # second: yield curve
    "short_yield":  (0.15, 0.70),
    "long_yield":   (0.5,  0.70),
    "term_prem":    (0.85, 0.70),
    # third: financial conditions
    "credit_sprd":  (0.0,  0.40),
    "asset_val":    (0.35, 0.40),
    "inflation":    (0.65, 0.40),
    "expectations": (1.0,  0.40),
    # bottom: real activity
    "output":       (0.5,  0.10),
}

_NODE_COLORS = {
    "pol_rate":     "#4e79a7",
    "money_growth": "#4e79a7",
    "short_yield":  "#76b7b2",
    "long_yield":   "#76b7b2",
    "term_prem":    "#76b7b2",
    "credit_sprd":  "#f28e2b",
    "asset_val":    "#f28e2b",
    "inflation":    "#e15759",
    "expectations": "#e15759",
    "output":       "#59a14f",
}


def _draw_dag(
    ax: plt.Axes,
    gt_edges: list[tuple],
    pred_edges: list[tuple],
    title: str,
) -> None:
    """Draw pred DAG with colour-coded edge correctness vs ground truth."""
    gt_set   = set(gt_edges)
    pred_set = set(pred_edges)

    G = nx.DiGraph()
    G.add_nodes_from(VARS)
    G.add_edges_from(pred_set)

    pos    = _LAYOUT_LEVELS
    labels = {v: VARIABLES[v].replace(" ", "\n") for v in VARS}
    colors = [_NODE_COLORS[v] for v in VARS]

    nx.draw_networkx_nodes(ax=ax, G=G, pos=pos, node_color=colors,
                           node_size=900, alpha=0.9)
    nx.draw_networkx_labels(ax=ax, G=G, pos=pos, labels=labels,
                            font_size=5.5, font_color="white", font_weight="bold")

    # colour edges
    correct   = [e for e in pred_set if e in gt_set]
    reversed_ = [(b, a) for (a, b) in gt_set if (b, a) in pred_set]
    extra     = [e for e in pred_set if e not in gt_set
                 and (e[1], e[0]) not in gt_set]

    def _draw_edges(edges, color, style="solid", width=2.0):
        if edges:
            nx.draw_networkx_edges(ax=ax, G=G, pos=pos, edgelist=edges,
                                   edge_color=color, style=style,
                                   width=width, arrows=True,
                                   arrowsize=14, connectionstyle="arc3,rad=0.05")

    _draw_edges(correct,   "#2ca02c", width=2.2)
    _draw_edges(reversed_, "#ff7f0e", width=2.2)
    _draw_edges(extra,     "#d62728", width=1.6, style="dashed")

    # missing ground-truth edges drawn as thin grey
    missing = [e for e in gt_set if e not in pred_set and (e[1], e[0]) not in pred_set]
    G_miss  = nx.DiGraph()
    G_miss.add_nodes_from(VARS)
    G_miss.add_edges_from(missing)
    nx.draw_networkx_edges(ax=ax, G=G_miss, pos=pos, edgelist=missing,
                           edge_color="#aaaaaa", style="dotted",
                           width=1.2, arrows=True, arrowsize=10,
                           connectionstyle="arc3,rad=0.05")

    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")


def _draw_gt_dag(ax: plt.Axes, edges: list[tuple], title: str) -> None:
    G = nx.DiGraph()
    G.add_nodes_from(VARS)
    G.add_edges_from(edges)
    pos    = _LAYOUT_LEVELS
    labels = {v: VARIABLES[v].replace(" ", "\n") for v in VARS}
    colors = [_NODE_COLORS[v] for v in VARS]
    nx.draw_networkx_nodes(ax=ax, G=G, pos=pos, node_color=colors,
                           node_size=900, alpha=0.9)
    nx.draw_networkx_labels(ax=ax, G=G, pos=pos, labels=labels,
                            font_size=5.5, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(ax=ax, G=G, pos=pos,
                           edge_color="#333333", width=2.0,
                           arrows=True, arrowsize=14,
                           connectionstyle="arc3,rad=0.05")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")

# ── legend patches ────────────────────────────────────────────────────────────

def _edge_legend(ax: plt.Axes) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    patches = [
        mlines.Line2D([], [], color="#2ca02c", linewidth=2,   label="Correct"),
        mlines.Line2D([], [], color="#ff7f0e", linewidth=2,   label="Reversed"),
        mlines.Line2D([], [], color="#d62728", linewidth=1.5, linestyle="--", label="False positive"),
        mlines.Line2D([], [], color="#aaaaaa", linewidth=1,   linestyle=":",  label="Missing (GT)"),
    ]
    ax.legend(handles=patches, loc="lower center", fontsize=7,
              ncol=2, framealpha=0.8)

# ── main experiment class ─────────────────────────────────────────────────────

class SyntheticExperiment:
    """
    Runs the full pipeline (CausalAgent → CausalValidator) on synthetic data
    and evaluates against ground-truth DAGs.
    """

    def __init__(
        self,
        use_cache: bool = True,
        use_prev_dag: bool = True,
        config_path: pathlib.Path = ROOT / "config" / "agent_config.yaml",
    ) -> None:
        self.use_cache    = use_cache
        self.use_prev_dag = use_prev_dag
        if config_path.exists():
            logger.info("Loading hyperparameters from %s", config_path)
            self.agent     = CausalAgent.from_config(config_path)
            self.validator = CausalValidator.from_config(config_path)
            # Synthetic experiment intentionally uses fewer refuter simulations
            # for speed — override only that one knob.
            self.validator.n_simulations = 50
        else:
            logger.info("No config found at %s — using built-in defaults", config_path)
            self.agent     = CausalAgent()
            self.validator = CausalValidator(alpha=0.05, n_simulations=50, min_obs=30)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _cache_path(self, regime: int, with_prev: bool = False) -> pathlib.Path:
        """
        Cache file name distinguishes runs that received a prev_dag context
        from runs that did not — otherwise an old (prev-less) cached result
        would silently shadow a freshly-wired prev-aware run.
        """
        suffix = "_with_prev" if with_prev else ""
        return CACHE_DIR / f"synthetic_regime_{regime:02d}{suffix}.json"

    def _run_agent(
        self,
        regime: int,
        data: pd.DataFrame,
        desc: str,
        prev_dag: list[dict] | None = None,
        broken_edges: list[tuple[str, str]] | None = None,
    ) -> tuple[list[tuple[str, str]], list[dict]]:
        """
        Run CausalAgent (with optional caching) for one regime.

        Parameters
        ----------
        regime : int
            Regime index (1 or 2).
        data : pd.DataFrame
            Synthetic SEM panel for this regime (data-blind to the agent).
        desc : str
            Hand-crafted regime description string.
        prev_dag : list of edge dicts, optional
            Edges from the previous regime's DAG (with justifications). When
            provided, the agent's Stage 1 prompt is augmented with this
            context — this is the "dynamic" link the project is built on.
        broken_edges : list of (str, str), optional
            Variable pairs whose causal link is suspected to have broken
            across the regime boundary (currently passed through but not
            populated by this experiment).

        Returns
        -------
        edge_tuples : list of (from, to) tuples — used by the validator.
        full_edges  : list of edge dicts (from / to / justification) — kept
                      so it can be threaded into the next regime as prev_dag.
        """
        has_prev = prev_dag is not None
        cache    = self._cache_path(regime, with_prev=has_prev)

        if self.use_cache and cache.exists():
            logger.info(
                "Loading cached agent result for regime %d (with_prev=%s)",
                regime, has_prev,
            )
            with open(cache) as f:
                payload = json.load(f)
            full = payload["edges"]
            return [(e["from"], e["to"]) for e in full], full

        logger.info(
            "Running CausalAgent for regime %d%s …",
            regime, " (with prev_dag)" if has_prev else "",
        )
        result = self.agent.run(
            variable_definitions=VARIABLE_DEFINITIONS,
            regime_description=desc,
            prev_dag=prev_dag,
            broken_edges=broken_edges,
        )
        # cache
        with open(cache, "w") as f:
            json.dump(
                {
                    "regime":    regime,
                    "edges":     result["edges"],
                    "n_edges":   result["n_edges"],
                    "is_dag":    result["is_dag"],
                    "with_prev": has_prev,
                },
                f, indent=2,
            )

        # also save intermediates (separate file per with_prev variant)
        inter_path = CACHE_DIR / (
            f"synthetic_intermediates_{regime:02d}"
            f"{'_with_prev' if has_prev else ''}.json"
        )
        with open(inter_path, "w") as f:
            json.dump(result["intermediates"], f, indent=2)

        return [(e["from"], e["to"]) for e in result["edges"]], result["edges"]

    def _run_validator(
        self, regime: int, pred_edges: list[tuple], data: pd.DataFrame
    ) -> list[tuple[str, str]]:
        """Run CausalValidator and return validated edge list."""
        G = nx.DiGraph()
        G.add_nodes_from(VARS)
        G.add_edges_from(pred_edges)

        # Only run on variables actually in the data
        data_clean = data[VARS].dropna()
        if len(data_clean) < 30:
            logger.warning("Regime %d: only %d obs — skipping DoWhy validation", regime, len(data_clean))
            return pred_edges

        validated_dag, results_df = self.validator.validate(G, data_clean)

        # save validation results
        results_df.to_csv(
            OUT_RESULTS / f"synthetic_validation_r{regime}.csv", index=False
        )
        return list(validated_dag.edges())

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        logger.info("=== Synthetic ground-truth validation ===")

        # ── generate data ──────────────────────────────────────────────────
        logger.info("Generating SEM data …")
        data_r1 = generate_sem(EDGES_R1, n_steps=500, seed=42)
        data_r2 = generate_sem(EDGES_R2, n_steps=500, seed=99)

        desc_r1 = _build_regime_description(1, data_r1)
        desc_r2 = _build_regime_description(2, data_r2)
        logger.info("Regime 1 description: %s", desc_r1[:80])
        logger.info("Regime 2 description: %s", desc_r2[:80])

        # ── run pipeline ───────────────────────────────────────────────────
        # R1 runs from scratch.  R2 receives R1's full DAG (with justifications)
        # via prev_dag — this is the "dynamic" link the project is named after.
        # `--no-prev-dag` ablates it so R2 runs independently (useful for A/B
        # comparison and for reusing the pre-wiring R2 cache).
        pred_r1_raw, full_r1_edges = self._run_agent(1, data_r1, desc_r1)
        pred_r2_raw, _              = self._run_agent(
            2, data_r2, desc_r2,
            prev_dag=(full_r1_edges if self.use_prev_dag else None),
            # broken_edges left None until we add a detector for it.
        )

        logger.info("R1 agent discovered %d edges; R2 agent discovered %d edges",
                    len(pred_r1_raw), len(pred_r2_raw))

        # ── DoWhy validation ───────────────────────────────────────────────
        logger.info("Running DoWhy validation …")
        pred_r1_val = self._run_validator(1, pred_r1_raw, data_r1)
        pred_r2_val = self._run_validator(2, pred_r2_raw, data_r2)

        logger.info("R1 validated: %d edges; R2 validated: %d edges",
                    len(pred_r1_val), len(pred_r2_val))

        # ── compute metrics ────────────────────────────────────────────────
        m1_raw = compute_edge_metrics(EDGES_R1, pred_r1_raw)
        m1_val = compute_edge_metrics(EDGES_R1, pred_r1_val)
        m2_raw = compute_edge_metrics(EDGES_R2, pred_r2_raw)
        m2_val = compute_edge_metrics(EDGES_R2, pred_r2_val)
        m_chg  = compute_regime_change_accuracy(EDGES_R1, EDGES_R2, pred_r1_val, pred_r2_val)

        rows = []
        for stage, m1, m2 in [("pre_validation", m1_raw, m2_raw),
                               ("post_validation", m1_val, m2_val)]:
            for regime, m in [(1, m1), (2, m2)]:
                row = {"stage": stage, "regime": regime}
                row.update(m)
                rows.append(row)

        # regime-change row
        chg_row = {"stage": "regime_change", "regime": "1→2"}
        chg_row.update(m_chg)
        rows.append(chg_row)

        results_df = pd.DataFrame(rows)
        out_csv    = OUT_RESULTS / "synthetic_validation.csv"
        results_df.to_csv(out_csv, index=False)
        logger.info("Saved metrics → %s", out_csv)

        # ── figures ────────────────────────────────────────────────────────
        self._plot_gt_dags()
        self._plot_discovered_dags(pred_r1_val, pred_r2_val)
        self._plot_metrics(results_df, m_chg)

        # ── console summary ────────────────────────────────────────────────
        self._print_summary(m1_raw, m1_val, m2_raw, m2_val, m_chg)

        return results_df

    # ── plotting ──────────────────────────────────────────────────────────────

    def _plot_gt_dags(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        _draw_gt_dag(axes[0], EDGES_R1,
                     f"Ground Truth — Regime 1 (ZLB/QE)\n{len(EDGES_R1)} edges")
        _draw_gt_dag(axes[1], EDGES_R2,
                     f"Ground Truth — Regime 2 (Tightening)\n{len(EDGES_R2)} edges")
        fig.suptitle("Synthetic Ground-Truth DAGs", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = OUT_FIGURES / "synthetic_gt_dags.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", out)

    def _plot_discovered_dags(
        self,
        pred_r1: list[tuple],
        pred_r2: list[tuple],
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        _draw_dag(axes[0], EDGES_R1, pred_r1,
                  f"Discovered (post-validation) — Regime 1\n{len(pred_r1)} edges")
        _draw_dag(axes[1], EDGES_R2, pred_r2,
                  f"Discovered (post-validation) — Regime 2\n{len(pred_r2)} edges")
        _edge_legend(axes[1])
        fig.suptitle("Discovered vs Ground-Truth DAGs", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = OUT_FIGURES / "synthetic_discovered_dags.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", out)

    def _plot_metrics(self, results_df: pd.DataFrame, m_chg: dict) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # ── bar chart: precision / recall / F1 by regime × stage ──────────
        ax = axes[0]
        sub = results_df[results_df["stage"].isin(["pre_validation", "post_validation"])].copy()
        sub["label"] = sub.apply(
            lambda r: f"R{r['regime']} {'pre' if r['stage']=='pre_validation' else 'post'}",
            axis=1,
        )
        x    = np.arange(len(sub))
        w    = 0.25
        ax.bar(x - w,   sub["precision"],    w, label="Precision", color="#4e79a7")
        ax.bar(x,        sub["recall"],       w, label="Recall",    color="#f28e2b")
        ax.bar(x + w,   sub["f1"],            w, label="F1",        color="#59a14f")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"].tolist(), fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Precision / Recall / F1\nby Regime and Validation Stage")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # ── SHD and n_reversed ──────────────────────────────────────────────
        ax = axes[1]
        colors_shd = ["#4e79a7", "#aec7e8", "#f28e2b", "#ffbb78"]
        for idx, (_, row) in enumerate(sub.iterrows()):
            ax.bar(idx, row["shd"],       color=colors_shd[idx], alpha=0.85,
                   label=row["label"])
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["label"].tolist(), fontsize=8)
        ax.set_ylabel("SHD")
        ax.set_title("Structural Hamming Distance")
        ax.grid(axis="y", alpha=0.3)
        # annotate reversed
        for idx, (_, row) in enumerate(sub.iterrows()):
            ax.text(idx, row["shd"] + 0.2, f"↕{int(row['n_reversed'])}",
                    ha="center", fontsize=7, color="#d62728")

        # ── regime-change detection ─────────────────────────────────────────
        ax = axes[2]
        chg_vals   = [m_chg["change_precision"], m_chg["change_recall"], m_chg["change_f1"]]
        chg_labels = ["Precision", "Recall", "F1"]
        chg_colors = ["#4e79a7", "#f28e2b", "#59a14f"]
        bars = ax.bar(chg_labels, chg_vals, color=chg_colors, alpha=0.85)
        for bar, val in zip(bars, chg_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(
            f"Regime-Change Detection (R1→R2)\n"
            f"GT changes: {m_chg['n_gt_changes']}  |  Detected: {m_chg['n_pred_changes']}"
        )
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Synthetic Validation Metrics", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = OUT_FIGURES / "synthetic_metrics.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved → %s", out)

    # ── console summary ───────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(m1_raw, m1_val, m2_raw, m2_val, m_chg) -> None:
        sep = "=" * 65
        print(f"\n{sep}")
        print(" SYNTHETIC VALIDATION RESULTS")
        print(sep)
        header = f"{'Metric':<22} {'R1 pre':>8} {'R1 post':>8} {'R2 pre':>8} {'R2 post':>8}"
        print(header)
        print("-" * 65)
        for key in ("n_gt", "n_pred", "n_tp", "n_reversed", "precision",
                    "recall", "f1", "dir_accuracy", "shd"):
            def _fmt(v):
                if v != v:   # nan check
                    return "   nan"
                if isinstance(v, float):
                    return f"{v:8.4f}"
                return f"{v:8d}"
            print(
                f"  {key:<20} {_fmt(m1_raw.get(key, float('nan')))} "
                f"{_fmt(m1_val.get(key, float('nan')))} "
                f"{_fmt(m2_raw.get(key, float('nan')))} "
                f"{_fmt(m2_val.get(key, float('nan')))}"
            )
        print(sep)
        print(" REGIME-CHANGE DETECTION (R1→R2, post-validation DAGs)")
        print(f"  GT structural changes  : {m_chg['n_gt_changes']}")
        print(f"  Detected changes       : {m_chg['n_pred_changes']}")
        print(f"  Precision              : {m_chg['change_precision']:.4f}")
        print(f"  Recall                 : {m_chg['change_recall']:.4f}")
        print(f"  F1                     : {m_chg['change_f1']:.4f}")
        print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    matplotlib.use("Agg")  # non-interactive backend only needed when saving figures to disk
    parser = argparse.ArgumentParser(description="Synthetic ground-truth validation")
    parser.add_argument(
        "--no-cache", dest="use_cache", action="store_false",
        help="Force re-run of CausalAgent even if cached results exist",
    )
    parser.add_argument(
        "--no-prev-dag", dest="use_prev_dag", action="store_false",
        help="Run R2 without prev_dag from R1 (ablation; lets you reuse the "
             "no-prev R2 cache from before the dynamic-link wiring was added)",
    )
    args = parser.parse_args()

    experiment = SyntheticExperiment(
        use_cache=args.use_cache,
        use_prev_dag=args.use_prev_dag,
    )
    experiment.run()


if __name__ == "__main__":
    main()
