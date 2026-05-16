"""BenchmarkDataset class for the yield-curve regime-reversal benchmark.

Mirrors the schema used by 10.2_yield_curve_dataset.ipynb. Loads from .npz + .json.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

EdgeKey = Tuple[str, str, int]
Edge    = Tuple[str, str, int, float]


@dataclass
class BenchmarkConfig:
    name: str = "yield_curve_reversal"
    var_names: List[str] = field(default_factory=list)
    T: int = 1000
    switch_at: int = 500
    n_seeds: int = 10
    sigma: float = 0.20
    burn_in: int = 200
    magnitude_threshold: float = 0.20


@dataclass
class BenchmarkDataset:
    config: BenchmarkConfig
    regime_seq: np.ndarray
    X_seeds: np.ndarray
    Phi_dict: Dict[int, np.ndarray]
    edges_dict: Dict[int, List[Edge]]
    tags: Dict[EdgeKey, str]

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = np.load(path.with_suffix(".npz"))
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        config = BenchmarkConfig(**meta["config"])
        edges_dict = {
            1: [tuple(e) for e in meta["edges_R1"]],
            2: [tuple(e) for e in meta["edges_R2"]],
        }
        Phi_dict = {1: data["Phi_R1"], 2: data["Phi_R2"]}
        tags = {}
        for k, v in meta["tags"].items():
            p, c, l = k.split("|")
            tags[(p, c, int(l))] = v
        return cls(
            config=config,
            regime_seq=data["regime_seq"],
            X_seeds=data["X_seeds"],
            Phi_dict=Phi_dict,
            edges_dict=edges_dict,
            tags=tags,
        )

    def __repr__(self):
        cfg = self.config
        return (
            f"BenchmarkDataset(name={cfg.name!r}, n_seeds={cfg.n_seeds}, "
            f"T={cfg.T}, switch_at={cfg.switch_at}, N_X={len(cfg.var_names)})"
        )
