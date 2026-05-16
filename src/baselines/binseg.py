"""Binary Segmentation — recursive change-point detection.

Reference
---------
Scott, A. J., & Knott, M. (1974). A cluster analysis method for grouping
means in the analysis of variance. Biometrics, 30(3), 507-512.

Notes
-----
Greedy: find the single change point minimising the cost, recurse on each
segment, stop when the cost reduction is below the penalty. Faster than
PELT/Dynp but only approximately optimal.

Wraps `ruptures.Binseg`.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import ruptures as rpt

DEFAULT_COST         = "l2"
DEFAULT_MIN_SIZE     = 30
DEFAULT_PEN_BIC_MULT = 10.0


def detect(
    panel: pd.DataFrame,
    pen: Optional[float] = None,
    pen_bic_mult: float = DEFAULT_PEN_BIC_MULT,
    n_bkps: Optional[int] = None,
    cost: str = DEFAULT_COST,
    min_size: int = DEFAULT_MIN_SIZE,
) -> List[pd.Timestamp]:
    """Detect change points via binary segmentation.

    Parameters
    ----------
    panel : DatetimeIndex DataFrame
    pen : float, optional
        Penalty for unknown-K mode. Ignored if `n_bkps` is given.
    pen_bic_mult : float
        Default multiplier when `pen` is None and `n_bkps` is None.
    n_bkps : int, optional
        Fixed number of breakpoints. If given, overrides `pen`.
    cost, min_size : ruptures hyperparameters.

    Returns
    -------
    list of pd.Timestamp
    """
    df = panel.dropna()
    X = df.values
    n, d = X.shape

    algo = rpt.Binseg(model=cost, min_size=min_size).fit(X)
    if n_bkps is not None:
        cp_idx = algo.predict(n_bkps=n_bkps)
    else:
        if pen is None:
            pen = float(pen_bic_mult) * d * np.log(n)
        cp_idx = algo.predict(pen=pen)
    cp_idx = [i for i in cp_idx if i < n]
    return [df.index[i] for i in cp_idx]
