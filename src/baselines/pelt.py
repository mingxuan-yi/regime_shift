"""PELT — Pruned Exact Linear Time change-point detection.

Reference
---------
Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal Detection of
Changepoints with a Linear Computational Cost. Journal of the American
Statistical Association, 107(500), 1590-1598.

Notes
-----
Wraps `ruptures.Pelt`. With separable costs PELT returns the exact MAP
segmentation in O(n) under regularity conditions; in pathological cases
behaviour degrades to O(n^2).

The default penalty is BIC-scaled: pen = pen_bic_mult * d * log(n), where
`d` is the number of variables. The default multiplier 10.0 was chosen by
F1-against-anchor sweep over {0.5, 1, 2, 5, 10, 20} on the FOMC 5-variable
panel; users running on other data should re-sweep.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import ruptures as rpt

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_COST         = "l2"
DEFAULT_MIN_SIZE     = 30
DEFAULT_PEN_BIC_MULT = 10.0


def detect(
    panel: pd.DataFrame,
    pen: Optional[float] = None,
    pen_bic_mult: float = DEFAULT_PEN_BIC_MULT,
    cost: str = DEFAULT_COST,
    min_size: int = DEFAULT_MIN_SIZE,
) -> List[pd.Timestamp]:
    """Detect change points via PELT.

    Parameters
    ----------
    panel : DatetimeIndex DataFrame
        Variables along columns.
    pen : float, optional
        Penalty. If None, set to `pen_bic_mult * d * log(n)`.
    pen_bic_mult : float
        BIC multiplier when `pen` is None.
    cost : str
        ruptures cost name ("l2", "l1", "rbf", "normal", "ar", ...).
    min_size : int
        Minimum segment length.

    Returns
    -------
    list of pd.Timestamp
        Sorted change-point timestamps (excludes the trailing endpoint
        ruptures appends).
    """
    df = panel.dropna()
    X = df.values
    n, d = X.shape
    if pen is None:
        pen = float(pen_bic_mult) * d * np.log(n)

    algo = rpt.Pelt(model=cost, min_size=min_size).fit(X)
    cp_idx = [i for i in algo.predict(pen=pen) if i < n]
    return [df.index[i] for i in cp_idx]
