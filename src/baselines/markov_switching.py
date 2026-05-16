"""Markov-Switching baseline — Hamilton (1989).

Reference
---------
Hamilton, J. D. (1989). A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle. Econometrica, 57(2),
357-384.

Notes
-----
Multivariate adaptation: project the panel onto its first `n_pcs`
principal components and fit a `k_regimes`-state Markov-switching
autoregression on the (concatenated) PC series. Change points are
transitions in the smoothed state path (Viterbi-like argmax).

This is a deliberately simple multivariate handle on what is natively a
univariate model — statsmodels does not ship a multivariate Markov-switching
VAR. If you need MS-VAR proper, use a custom HMM (e.g., hmmlearn) with
Gaussian emissions of dimension d, but expect slower convergence and more
local optima.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

DEFAULT_K_REGIMES   = 4
DEFAULT_N_PCS       = 1
DEFAULT_ORDER       = 1
DEFAULT_DEDUP_DAYS  = 14


def detect(
    panel: pd.DataFrame,
    k_regimes: int = DEFAULT_K_REGIMES,
    n_pcs: int = DEFAULT_N_PCS,
    order: int = DEFAULT_ORDER,
    dedup_days: int = DEFAULT_DEDUP_DAYS,
    random_state: int = 0,
) -> List[pd.Timestamp]:
    """Detect change points as Markov-switching state transitions.

    Parameters
    ----------
    panel : DatetimeIndex DataFrame
    k_regimes : int
        Number of latent regimes.
    n_pcs : int
        Number of principal components to summarise the panel by.
        Used only if `n_pcs == 1` for the univariate AR; for n_pcs > 1
        we currently run one AR per PC and union alarms.
    order : int
        AR order in each regime.
    dedup_days : int
        Dedup window for unioned alarms across PCs.
    random_state : int
        PCA random state. (statsmodels EM is deterministic given init.)

    Returns
    -------
    list of pd.Timestamp
    """
    from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

    df = panel.dropna()
    X = df.values
    pca = PCA(n_components=n_pcs, random_state=random_state)
    PCs = pca.fit_transform(X)

    alarms: set[pd.Timestamp] = set()
    for k in range(n_pcs):
        y = PCs[:, k]
        try:
            model = MarkovAutoregression(
                y, k_regimes=k_regimes, order=order, switching_ar=False,
            )
            result = model.fit(disp=False)
            smoothed = np.asarray(result.smoothed_marginal_probabilities)
            # statsmodels returns shape (T, k_regimes); MAP state per row
            states = smoothed.argmax(axis=1)
        except Exception:
            # MS estimation can fail on awkward series; skip this PC silently
            continue

        # Transitions (offset by `order` for the AR burn-in)
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                alarms.add(df.index[i + order])

    if not alarms:
        return []
    dates = sorted(d for d in alarms if d <= df.index[-1])
    deduped = [dates[0]]
    for d in dates[1:]:
        if (d - deduped[-1]).days > dedup_days:
            deduped.append(d)
    return deduped
