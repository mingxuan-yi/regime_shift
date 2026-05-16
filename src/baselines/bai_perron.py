"""Bai-Perron multiple structural-break detection (DP approximation).

Reference
---------
Bai, J., & Perron, P. (2003). Computation and Analysis of Multiple
Structural Change Models. Journal of Applied Econometrics, 18(1), 1-22.

Notes
-----
Strict Bai-Perron uses sequential SupF tests to determine the number of
breaks and the dynamic-programming (DP) algorithm to place them. In Python
without `strucchange`, we use ruptures' DP backend (`rpt.Dynp`) with an
L2 cost as a faithful approximation:

  - Given K, ruptures.Dynp returns the exact L2-optimal placement of
    K breakpoints — identical to Bai-Perron's DP step under Gaussian
    likelihood.
  - For unknown K, we sweep K ∈ {1, ..., K_max} and select by BIC.

The minimum-segment-length `min_size` corresponds to Bai-Perron's
"trimming" parameter h.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import ruptures as rpt

DEFAULT_COST     = "l2"
DEFAULT_MIN_SIZE = 30
DEFAULT_K_MAX    = 40   # maximum K to consider when sweeping
DEFAULT_JUMP     = 1    # DP grid stride; 1 = exact, >1 = approximation


def _bic(rss: float, n: int, d: int, k: int) -> float:
    """BIC criterion for K break points.

    Parameters scale: per-segment mean has d params + d(d+1)/2 covariance
    params, K+1 segments → K(d + d(d+1)/2) extra params over the no-break model.
    """
    extra_params = k * (d + d * (d + 1) // 2)
    return n * np.log(max(rss / n, 1e-30)) + extra_params * np.log(n)


def detect(
    panel: pd.DataFrame,
    n_bkps: Optional[int] = None,
    k_max: int = DEFAULT_K_MAX,
    cost: str = DEFAULT_COST,
    min_size: int = DEFAULT_MIN_SIZE,
    jump: int = DEFAULT_JUMP,
) -> List[pd.Timestamp]:
    """Detect change points via Bai-Perron-style DP placement.

    Parameters
    ----------
    panel : DatetimeIndex DataFrame
    n_bkps : int, optional
        Fixed number of breaks. If given, DP returns that many. If None,
        sweep K=1..k_max and pick by BIC.
    k_max : int
        Maximum K when sweeping.
    cost : str
    min_size : int

    Returns
    -------
    list of pd.Timestamp
    """
    df = panel.dropna()
    X = df.values
    n, d = X.shape

    algo = rpt.Dynp(model=cost, min_size=min_size, jump=jump).fit(X)

    if n_bkps is not None:
        cp_idx = algo.predict(n_bkps=n_bkps)
    else:
        # Sweep K, pick by BIC
        best_bic = np.inf
        best_cp = []
        # K=0 baseline (single segment) is implicit; start at K=1.
        for k in range(1, min(k_max, max(1, n // (2 * min_size))) + 1):
            try:
                cp = algo.predict(n_bkps=k)
            except Exception:
                continue
            # Compute RSS from segment means
            segments = [(0 if i == 0 else cp[i - 1], cp[i]) for i in range(len(cp))]
            rss = 0.0
            for s, e in segments:
                seg = X[s:e]
                rss += float(np.sum((seg - seg.mean(axis=0)) ** 2))
            score = _bic(rss, n, d, k)
            if score < best_bic:
                best_bic = score
                best_cp = cp
        cp_idx = best_cp

    cp_idx = [i for i in cp_idx if i < n]
    return [df.index[i] for i in cp_idx]
