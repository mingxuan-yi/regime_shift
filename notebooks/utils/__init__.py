"""Utilities for the yield-curve regime-reversal benchmark."""
from .dataset import BenchmarkConfig, BenchmarkDataset
from .pcmci_helpers import run_pcmci_segment, fit_phi_ols
from .baselines import regime_pcmci, ruptures_pcmci
from .evaluation import (
    edges_to_set,
    regime_label_accuracy,
    changepoint_error,
    shd,
    edge_prf1,
    coefficient_rmse,
    per_type_recall,
    evaluate_full,
    METRIC_DESCRIPTIONS,
    TYPE_DESCRIPTIONS,
)
from .visualization import (
    TYPE_COLORS,
    NODE_POS,
    NODE_COLORS,
    LEVEL_MEAN,
    LEVEL_STD,
    DAG_LEGEND,
    to_display_levels,
    plot_trajectories,
    draw_dag_with_values,
)
from .text_regime import (
    FOMCDoc,
    CandidateChangepoint,
    ValidatedChangepoint,
    DetectionResult,
    TextRegimeDetector,
    likelihood_ratio_test,
    match_anchors,
    summarize_match,
    load_fomc_corpus,
    FED_ANCHOR_EVENTS,
)

__all__ = [
    "BenchmarkConfig", "BenchmarkDataset",
    "run_pcmci_segment", "fit_phi_ols",
    "regime_pcmci", "ruptures_pcmci",
    "edges_to_set", "regime_label_accuracy", "changepoint_error", "shd",
    "edge_prf1", "coefficient_rmse", "per_type_recall", "evaluate_full",
    "METRIC_DESCRIPTIONS", "TYPE_DESCRIPTIONS",
    "TYPE_COLORS", "NODE_POS", "NODE_COLORS", "LEVEL_MEAN", "LEVEL_STD", "DAG_LEGEND",
    "to_display_levels", "plot_trajectories", "draw_dag_with_values",
    "FOMCDoc", "CandidateChangepoint", "ValidatedChangepoint", "DetectionResult",
    "TextRegimeDetector", "likelihood_ratio_test",
    "match_anchors", "summarize_match", "load_fomc_corpus", "FED_ANCHOR_EVENTS",
]
