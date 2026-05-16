"""BOCPD — Bayesian Online Change-Point Detection (Adams & MacKay 2007).

Reference
---------
Adams, R. P., & MacKay, D. J. C. (2007). Bayesian Online Changepoint
Detection. arXiv:0710.3742.

Algorithm
---------
Maintains a posterior over run length r_t given observations x_{1:t}:

    P(r_t = r | x_{1:t}) ∝ \sum_{r_{t-1}} P(x_t | r_{t-1}, x_{(t-r_{t-1}):t-1})
                              × P(r_t | r_{t-1}) × P(r_{t-1} | x_{1:t-1})

with a constant hazard P(r_t = 0 | r_{t-1}) = 1/λ.

For a univariate Gaussian observation model with Normal-Inverse-Gamma
prior (μ_0, κ_0, α_0, β_0), the posterior predictive is a Student-t.

Multivariate adaptation: run univariate BOCPD on each column independently,
union the alarm timestamps across columns, dedup within `dedup_days`.

Run length is truncated to `r_max` (default 500) to bound the O(T·r_max)
runtime; this is conservative for monetary-policy regimes that rarely run
longer than ~6 months × 21 trading days = 126.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

DEFAULT_LAMBDA       = 250    # prior expected run length (days)
DEFAULT_MU0          = 0.0
DEFAULT_KAPPA0       = 1.0
DEFAULT_ALPHA0       = 1.0
DEFAULT_BETA0        = 1.0
DEFAULT_R_MAX        = 500    # truncate run length for compute
DEFAULT_DEDUP_DAYS   = 14
DEFAULT_REFRACTORY   = 30
DEFAULT_RL_DROP_TO   = 5      # alarm when MAP run-length drops back below this


def _bocpd_1d(
    x: np.ndarray,
    hazard: float,
    mu0: float,
    kappa0: float,
    alpha0: float,
    beta0: float,
    r_max: int,
) -> np.ndarray:
    """Run BOCPD on a 1D series; return MAP run length at each step (length T)."""
    T = len(x)
    # R[r, t] = P(r_t = r | x_{1:t}); we slide a vector instead of storing full matrix
    R = np.zeros(r_max + 1)
    R[0] = 1.0

    mu = np.full(r_max + 1, mu0)
    kappa = np.full(r_max + 1, kappa0)
    alpha = np.full(r_max + 1, alpha0)
    beta = np.full(r_max + 1, beta0)

    map_rl = np.zeros(T, dtype=int)

    for t in range(T):
        # Predictive: x_t ~ T(2α; μ, β(κ+1)/(ακ))
        df_t = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pred = student_t.pdf(x[t], df=df_t, loc=mu, scale=scale)

        # New run-length distribution
        # Growth: r_t = r_{t-1} + 1, prob = R[r_{t-1}] * pred * (1 - H)
        growth = R * pred * (1.0 - hazard)
        # Change: r_t = 0, prob = sum over r_{t-1} of R[r_{t-1}] * pred * H
        cp = float(np.sum(R * pred * hazard))

        # Shift growth by 1 (r_{t-1} → r_t = r_{t-1} + 1), drop oldest
        R_new = np.empty_like(R)
        R_new[0] = cp
        R_new[1:] = growth[:-1]
        s = R_new.sum()
        if s > 0:
            R_new /= s
        R = R_new

        # Sufficient-statistic update (shifted by 1, prepend prior)
        new_kappa = np.empty_like(kappa)
        new_mu = np.empty_like(mu)
        new_alpha = np.empty_like(alpha)
        new_beta = np.empty_like(beta)
        new_kappa[0] = kappa0
        new_mu[0] = mu0
        new_alpha[0] = alpha0
        new_beta[0] = beta0
        new_kappa[1:] = kappa[:-1] + 1
        new_mu[1:] = (kappa[:-1] * mu[:-1] + x[t]) / (kappa[:-1] + 1)
        new_alpha[1:] = alpha[:-1] + 0.5
        new_beta[1:] = beta[:-1] + kappa[:-1] * (x[t] - mu[:-1]) ** 2 / (2 * (kappa[:-1] + 1))
        kappa, mu, alpha, beta = new_kappa, new_mu, new_alpha, new_beta

        map_rl[t] = int(R.argmax())

    return map_rl


def detect(
    panel: pd.DataFrame,
    hazard_lambda: int = DEFAULT_LAMBDA,
    r_max: int = DEFAULT_R_MAX,
    rl_drop_to: int = DEFAULT_RL_DROP_TO,
    refractory_days: int = DEFAULT_REFRACTORY,
    dedup_days: int = DEFAULT_DEDUP_DAYS,
    mu0: float = DEFAULT_MU0,
    kappa0: float = DEFAULT_KAPPA0,
    alpha0: float = DEFAULT_ALPHA0,
    beta0: float = DEFAULT_BETA0,
) -> List[pd.Timestamp]:
    """Detect change points via per-variable BOCPD union.

    A change point is recorded when the MAP run length on any variable drops
    back below `rl_drop_to` (from a higher value), subject to a per-variable
    refractory period of `refractory_days` and a cross-variable dedup window
    of `dedup_days`.

    Parameters
    ----------
    panel : DatetimeIndex DataFrame
    hazard_lambda : int
        Prior expected run length, in observations (constant hazard = 1/λ).
    r_max : int
        Run-length truncation for compute (run lengths above this are merged).
    rl_drop_to : int
        Threshold for alarm: MAP run length falling below this triggers a CP.
    refractory_days : int
        Min gap between successive alarms on a single variable.
    dedup_days : int
        Cross-variable alarm dedup window (calendar days).
    mu0, kappa0, alpha0, beta0 : float
        Normal-Inverse-Gamma prior hyperparameters.

    Returns
    -------
    list of pd.Timestamp
    """
    df = panel.dropna()
    hazard = 1.0 / hazard_lambda

    alarms: set[pd.Timestamp] = set()
    for col in df.columns:
        x = df[col].to_numpy()
        # Standardize to keep prior reasonable across columns
        x = (x - x.mean()) / (x.std() + 1e-12)
        map_rl = _bocpd_1d(x, hazard, mu0, kappa0, alpha0, beta0, r_max)

        last_alarm = -refractory_days - 1
        for t in range(1, len(map_rl)):
            if map_rl[t] <= rl_drop_to and map_rl[t - 1] > rl_drop_to:
                if t - last_alarm > refractory_days:
                    alarms.add(df.index[t])
                    last_alarm = t

    if not alarms:
        return []
    dates = sorted(alarms)
    deduped = [dates[0]]
    for d in dates[1:]:
        if (d - deduped[-1]).days > dedup_days:
            deduped.append(d)
    return deduped
