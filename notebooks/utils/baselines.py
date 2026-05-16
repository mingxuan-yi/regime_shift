"""Two baselines: Regime-PCMCI (joint EM) and Ruptures + PCMCI (two-stage)."""

import numpy as np
from scipy.ndimage import median_filter
import ruptures as rpt

from .pcmci_helpers import run_pcmci_segment, fit_phi_ols


def regime_pcmci(
    X, var_names,
    n_regimes=2, max_iter=10, smooth_window=21,
    init="midpoint", tau_max=1, alpha=0.05,
    seed=0, verbose=False,
):
    """Joint regime detection + per-regime PCMCI via EM.

    Iterates between (a) per-regime OLS Phi fit and (b) timepoint reassignment
    by one-step prediction loss; median-filters labels to suppress single-step
    flips. After convergence, runs PCMCI on the longest contiguous run of each
    regime to extract the directed edge set.
    """
    T, N = X.shape
    rng = np.random.default_rng(seed)
    if init == "midpoint":
        labels = np.ones(T, dtype=int); labels[T // 2:] = 2
    elif init == "random":
        labels = rng.choice(np.arange(1, n_regimes + 1), size=T)
    elif init == "thirds":
        labels = np.ones(T, dtype=int)
        labels[T // 3 : 2 * T // 3] = 2
        labels[2 * T // 3:] = 1
    elif init == "equal_segments":
        # Split into n_regimes equal-width contiguous segments
        labels = np.zeros(T, dtype=int)
        seg = T // n_regimes
        for k in range(n_regimes):
            s = k * seg
            e = (k + 1) * seg if k < n_regimes - 1 else T
            labels[s:e] = k + 1
    elif init == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10)
        labels = km.fit_predict(X) + 1   # map to 1..n_regimes
    else:
        raise ValueError(f"Unknown init: {init!r}")

    Phi_dict = {}
    for it in range(max_iter):
        # M-step
        for r in range(1, n_regimes + 1):
            mask_idx = np.where(labels == r)[0]
            if len(mask_idx) < 30:
                continue
            Phi_dict[r] = fit_phi_ols(X, indices=mask_idx)
        if not Phi_dict:
            break

        # E-step
        new_labels = labels.copy()
        for t in range(1, T):
            losses = []
            for r in range(1, n_regimes + 1):
                if r not in Phi_dict:
                    losses.append(np.inf); continue
                pred = Phi_dict[r] @ X[t - 1]
                losses.append(((X[t] - pred) ** 2).sum())
            new_labels[t] = 1 + int(np.argmin(losses))
        new_labels[0] = new_labels[1]
        new_labels = median_filter(new_labels, size=smooth_window, mode="nearest")

        if verbose:
            n_changes = (new_labels != labels).sum()
            print(f"  iter {it+1}: {n_changes} timepoints reassigned")
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    # Final structure step: PCMCI on longest run of each regime
    edges_dict = {}
    for r in range(1, n_regimes + 1):
        mask = labels == r
        if mask.sum() < 30:
            edges_dict[r] = []; continue
        runs = []
        in_run = False; start = 0
        for t in range(T):
            if mask[t] and not in_run:
                start = t; in_run = True
            elif not mask[t] and in_run:
                runs.append((start, t)); in_run = False
        if in_run:
            runs.append((start, T))
        runs = sorted(runs, key=lambda ab: ab[1] - ab[0], reverse=True)
        s, e = runs[0]
        edges_dict[r] = run_pcmci_segment(X[s:e], var_names, tau_max=tau_max, alpha=alpha)

    return labels, edges_dict, Phi_dict


def ruptures_pcmci(X, var_names, n_bkps=1, model="l2", tau_max=1, alpha=0.05):
    """Two-stage: Binseg changepoint detection, then PCMCI per segment.

    Returns ``(pred_seq, edges_dict, Phi_dict, cp)`` — the trailing ``cp`` is
    the detected changepoint index (useful for diagnostics).
    """
    T, N = X.shape
    algo = rpt.Binseg(model=model).fit(X)
    breakpoints = algo.predict(n_bkps=n_bkps)
    cps = breakpoints[:-1]
    cp = cps[0] if cps else T // 2

    pred_seq = np.ones(T, dtype=int)
    pred_seq[cp:] = 2

    edges_R1 = run_pcmci_segment(X[:cp], var_names, tau_max=tau_max, alpha=alpha)
    edges_R2 = run_pcmci_segment(X[cp:], var_names, tau_max=tau_max, alpha=alpha)
    Phi_R1   = fit_phi_ols(X[:cp])
    Phi_R2   = fit_phi_ols(X[cp:])

    return pred_seq, {1: edges_R1, 2: edges_R2}, {1: Phi_R1, 2: Phi_R2}, cp
