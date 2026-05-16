"""CUSUM — Page (1954) two-sided cumulative-sum statistic.

Reference
---------
Page, E. S. (1954). Continuous Inspection Schemes. Biometrika, 41(1/2),
100-115.

Notes
-----
Univariate two-sided Page statistic per variable; per-variable alarms are
unioned and deduplicated within `dedup_days`. Suitable as a baseline for
the bidirectional shift detection problem, though it does not natively
exploit covariance.

Default hyperparameters chosen for daily standardised series:
- k_ref      = 0.5   (reference value; "slack")
- h_thresh   = 5.0   (alarm threshold in standardised units)
- refractory = 30    (min gap between successive alarms on a single var)
- dedup_days = 14    (cross-variable merge window)
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

DEFAULT_K_REF      = 0.5
DEFAULT_H_THRESH   = 5.0
DEFAULT_REFRACTORY = 30
DEFAULT_DEDUP_DAYS = 14


def _page_cusum_1d(
    x: np.ndarray,
    k_ref: float,
    h_thresh: float,
    refractory: int,
) -> list[int]:
    """Two-sided Page CUSUM on a 1D series. Returns alarm indices."""
    n = len(x)
    mu = float(np.mean(x))
    sigma = float(np.std(x)) + 1e-12
    z = (x - mu) / sigma

    S_pos = S_neg = 0.0
    last_alarm = -refractory - 1
    alarms: list[int] = []
    for t in range(n):
        S_pos = max(0.0, S_pos + z[t] - k_ref)
        S_neg = max(0.0, S_neg - z[t] - k_ref)
        if (S_pos > h_thresh or S_neg > h_thresh) and (t - last_alarm) > refractory:
            alarms.append(t)
            S_pos = 0.0
            S_neg = 0.0
            last_alarm = t
    return alarms


def detect(
    panel: pd.DataFrame,
    k_ref: float = DEFAULT_K_REF,
    h_thresh: float = DEFAULT_H_THRESH,
    refractory_days: int = DEFAULT_REFRACTORY,
    dedup_days: int = DEFAULT_DEDUP_DAYS,
) -> List[pd.Timestamp]:
    """Detect change points via per-variable two-sided Page CUSUM, then union.

    Parameters
    ----------
    panel : DatetimeIndex DataFrame
    k_ref : float
        Page reference value (drift parameter).
    h_thresh : float
        Alarm threshold.
    refractory_days : int
        Minimum gap (in time-series rows) between successive alarms per variable.
    dedup_days : int
        Cross-variable alarm dedup window (calendar days).

    Returns
    -------
    list of pd.Timestamp
    """
    df = panel.dropna()
    alarms_set: set[pd.Timestamp] = set()
    for col in df.columns:
        x = df[col].to_numpy()
        for i in _page_cusum_1d(x, k_ref, h_thresh, refractory_days):
            alarms_set.add(df.index[i])

    if not alarms_set:
        return []

    dates = sorted(alarms_set)
    deduped = [dates[0]]
    for d in dates[1:]:
        if (d - deduped[-1]).days > dedup_days:
            deduped.append(d)
    return deduped
