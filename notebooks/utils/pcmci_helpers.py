"""PCMCI segment fitting and OLS Phi estimation."""

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


def run_pcmci_segment(X_seg, var_names, tau_max=1, alpha=0.05, min_obs=30):
    """Run PCMCI on a single time-series segment.

    Returns
    -------
    list of (parent, child, lag, coef)
        Each significant lagged link with the val-matrix coefficient.

    Notes
    -----
    Tigramite's indexing convention is::

        val_matrix[i, j, tau]  ==  link  X_i(t - tau) -> X_j(t)

    so the FIRST index is the parent and the SECOND is the child.
    (The comment in ``src/causal/regime_detection.py`` reverses this; the bug
    is hidden there because Jaccard distance is direction-symmetric.)
    """
    if len(X_seg) < min_obs:
        return []
    N = X_seg.shape[1]
    dataframe = pp.DataFrame(X_seg, var_names=var_names)
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    res = pcmci.run_pcmci(
        tau_min=1, tau_max=tau_max,
        pc_alpha=alpha, alpha_level=alpha,
    )
    p_matrix   = res["p_matrix"]
    val_matrix = res["val_matrix"]
    edges = []
    for parent in range(N):
        for child in range(N):
            if parent == child:
                continue
            for tau in range(1, tau_max + 1):
                p = p_matrix[parent, child, tau]
                if not np.isnan(p) and p < alpha:
                    edges.append((
                        var_names[parent],
                        var_names[child],
                        tau,
                        float(val_matrix[parent, child, tau]),
                    ))
    return edges


def fit_phi_ols(X_seg, indices=None):
    """OLS lag-1 VAR fit. Returns Phi shape (N, N) with Phi[child, parent].

    If ``indices`` is given, only consecutive pairs within that index list are used.
    """
    N = X_seg.shape[1]
    if indices is None:
        Y = X_seg[1:];  Xlag = X_seg[:-1]
    else:
        idx = np.asarray(sorted(indices))
        cons = idx[np.where(np.diff(idx) == 1)[0]]
        if len(cons) < 30:
            return np.zeros((N, N))
        Y    = X_seg[cons + 1]
        Xlag = X_seg[cons]
    return np.linalg.lstsq(Xlag, Y, rcond=None)[0].T
