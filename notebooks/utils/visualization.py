"""Visualization helpers: trajectories, DAG comparison, level mapping."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D


# ── shared style ───────────────────────────────────────────────────────────

TYPE_COLORS = {
    "unchanged":       "#7d7d7d",
    "appear":          "#1D9E75",
    "remove":          "#D85A30",
    "reversal":        "#9C5BCB",
    "magnitude_shift": "#1F77B4",
    "sign_flip":       "#E1B12C",
}

NODE_POS = {
    "output":      (0.0,  2.0),
    "inflation":   (0.0,  1.0),
    "pol_rate":    (1.2,  1.5),
    "short_yield": (2.4,  2.0),
    "long_yield":  (2.4,  1.0),
}

NODE_COLORS = {
    "output":      "#59a14f",
    "inflation":   "#e15759",
    "pol_rate":    "#4e79a7",
    "short_yield": "#76b7b2",
    "long_yield":  "#76b7b2",
}

# FRED-calibrated display levels (2020-03 → 2024-08 windows)
LEVEL_MEAN = {
    1: {"pol_rate": 0.10, "inflation": 2.50, "output": 1.50, "short_yield": 0.35, "long_yield": 1.30},
    2: {"pol_rate": 4.20, "inflation": 5.00, "output": 2.50, "short_yield": 4.20, "long_yield": 3.85},
}
LEVEL_STD = {
    1: {"pol_rate": 0.05, "inflation": 1.50, "output": 1.20, "short_yield": 0.30, "long_yield": 0.40},
    2: {"pol_rate": 1.50, "inflation": 1.80, "output": 0.30, "short_yield": 0.75, "long_yield": 0.55},
}

DAG_LEGEND = [
    Line2D([0], [0], color="#2ca02c", lw=2.4,            label="correct (TP)"),
    Line2D([0], [0], color="#d62728", lw=1.6, ls="--",   label="false positive (FP)"),
    Line2D([0], [0], color="#aaaaaa", lw=1.4, ls=":",    label="missing (FN)"),
]


# ── display-only affine transform (does NOT modify ds.X_seeds) ────────────

def to_display_levels(X, regime_seq, var_names):
    """Per-regime within-regime z-score, then apply target (mean, std) for plotting."""
    X_disp = X.copy()
    for r in [1, 2]:
        mask = regime_seq == r
        if mask.sum() == 0:
            continue
        for j, name in enumerate(var_names):
            seg = X[mask, j]
            seg_z = (seg - seg.mean()) / max(seg.std(), 1e-9)
            X_disp[mask, j] = LEVEL_MEAN[r][name] + LEVEL_STD[r][name] * seg_z
    return X_disp


# ── trajectory plot ───────────────────────────────────────────────────────

def plot_trajectories(ds, human, seed_idx=0, display_levels=True):
    X = (
        to_display_levels(ds.X_seeds[seed_idx], ds.regime_seq, ds.config.var_names)
        if display_levels
        else ds.X_seeds[seed_idx]
    )
    T = ds.config.T; switch = ds.config.switch_at
    var_names = ds.config.var_names; N = len(var_names)
    fig, axes = plt.subplots(
        N + 1, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [0.5] + [1] * N},
    )
    axes[0].axvspan(0, switch, color="#5DCAA5", alpha=0.4, label="R1 (ZLB / QE)")
    axes[0].axvspan(switch, T, color="#F0997B", alpha=0.4, label="R2 (Tightening)")
    axes[0].set_yticks([])
    axes[0].set_ylabel("Regime", rotation=0, ha="right", va="center")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    axes[0].set_title(
        f"Seed {seed_idx} — " + ("display levels" if display_levels else "raw SCM"),
        fontsize=10, loc="left",
    )

    line_colors = ["#534AB7", "#BA7517", "#0F6E56", "#185FA5", "#993C1D"]
    for j, name in enumerate(var_names):
        ax = axes[j + 1]
        ax.axvspan(0, switch, color="#5DCAA5", alpha=0.08)
        ax.axvspan(switch, T, color="#F0997B", alpha=0.08)
        ax.plot(X[:, j], color=line_colors[j % len(line_colors)], linewidth=0.8)
        ax.set_ylabel(human[name].replace(" ", "\n"),
                      rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("Time t")
    plt.tight_layout()
    return fig


# ── DAG plot with edge values + correctness colour ────────────────────────

def draw_dag_with_values(
    ax, edges, title, human,
    ref_edges=None, color_by_correctness=True, exclude_self_loops=True,
):
    """Draw a DAG with coefficient labels.

    If ``ref_edges`` is given, edges are coloured: green=correct, red dashed=FP,
    grey dotted=missing (FN drawn from ``ref_edges`` with the GT coefficient).
    Otherwise edges are drawn in a uniform dark colour (use this for GT panels).
    """
    edge_dict = {(p, c, l): coef for (p, c, l, coef) in edges
                 if not (exclude_self_loops and p == c)}
    if ref_edges is None:
        ref_set = set()
    else:
        ref_set = {(p, c, l) for (p, c, l, _) in ref_edges
                   if not (exclude_self_loops and p == c)}
    pred_set = set(edge_dict.keys())
    correct  = pred_set & ref_set
    fp       = pred_set - ref_set
    missing  = ref_set - pred_set if ref_edges is not None else set()
    ref_dict = {(p, c, l): coef for (p, c, l, coef) in (ref_edges or [])}

    def _arrow(p, c, color, style, lw, label=None):
        x0, y0 = NODE_POS[p]; x1, y1 = NODE_POS[c]
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=14, color=color,
            linewidth=lw, alpha=0.9, shrinkA=22, shrinkB=22,
            connectionstyle="arc3,rad=0.07", linestyle=style,
        ))
        if label is not None:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            length = max(np.hypot(dx, dy), 1e-6)
            ox, oy = -dy / length * 0.10, dx / length * 0.10
            ax.text(
                mx + ox, my + oy, label, fontsize=7,
                ha="center", va="center", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )

    if color_by_correctness and ref_edges is not None:
        for k in correct: _arrow(*k[:2], "#2ca02c", "solid",  2.2, label=f"{edge_dict[k]:+.2f}")
        for k in fp:      _arrow(*k[:2], "#d62728", "dashed", 1.6, label=f"{edge_dict[k]:+.2f}")
        for k in missing: _arrow(*k[:2], "#aaaaaa", "dotted", 1.4, label=f"{ref_dict[k]:+.2f}")
    else:
        for k, coef in edge_dict.items():
            _arrow(*k[:2], "#333333", "solid", 2.0, label=f"{coef:+.2f}")

    for name, (x, y) in NODE_POS.items():
        ax.scatter(x, y, s=2400, c=NODE_COLORS[name], edgecolors="#444", linewidth=0.8, zorder=3)
        ax.text(x, y, human[name].replace(" ", "\n"),
                fontsize=7, ha="center", va="center",
                color="white", fontweight="bold", zorder=4)

    xs = [p[0] for p in NODE_POS.values()]; ys = [p[1] for p in NODE_POS.values()]
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax.set_ylim(min(ys) - 0.4, max(ys) + 0.4)
    ax.set_aspect("equal"); ax.axis("off")
    if ref_edges is None:
        ax.set_title(title, fontsize=10, pad=8)
    else:
        ax.set_title(
            f"{title}\nTP={len(correct)}, FP={len(fp)}, FN={len(missing)}",
            fontsize=10, pad=8,
        )
