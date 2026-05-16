"""Change-point detection baselines for the IJCAI paper.

Every module exposes a single function with a uniform signature:

    detect(panel: pd.DataFrame, **hyperparams) -> list[pd.Timestamp]

The argument `panel` is a DatetimeIndex DataFrame whose columns are the
variables to monitor (already pre-processed; see preprocess.py). The
function returns a chronologically sorted list of change-point timestamps.

Modules
-------
- pelt              : Killick, Fearnhead & Eckley (2012) PELT
- binseg            : Binary segmentation (Scott & Knott 1974; via ruptures)
- bocpd             : Adams & MacKay (2007) Bayesian online change-point detection
- cusum             : Page (1954) two-sided cumulative-sum statistic (multivariate union)
- bai_perron        : Bai & Perron (2003) — dynamic-programming approximation
- markov_switching  : Hamilton (1989) Markov-switching autoregression (via statsmodels)
"""
from . import bai_perron, binseg, bocpd, cusum, markov_switching, pelt

__all__ = [
    "bai_perron",
    "binseg",
    "bocpd",
    "cusum",
    "markov_switching",
    "pelt",
]
