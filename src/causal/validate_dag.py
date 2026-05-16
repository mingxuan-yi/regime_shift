"""
Statistical validation of LLM-proposed causal edges using DoWhy.

For each edge X → Y in the input DAG:
  1. Build a DoWhy CausalModel from the full DAG structure.
  2. Identify the causal effect (backdoor adjustment set).
  3. Estimate via OLS; the regression specification (contemporaneous /
     lag1_ar / first_diff / lag_sweep) is configurable — see _fit_spec.
  4. Run a block-permutation placebo refuter on the spec-aligned frame.
     Replaces DoWhy's iid placebo_treatment_refuter, which destroys
     autocorrelation and is mis-specified for time-series data.
  5. Run a block-bootstrap stability refuter on the spec-aligned frame.
     Replaces DoWhy's data_subset_refuter for the same reason.
  6. Flag as valid if:
       p_value       < alpha   (the edge has a significant linear effect)
       placebo_p    >= alpha   (block-permuted treatment does NOT look causal)
       subset_p     >= alpha   (estimate is stable under block resampling)
  7. Remove invalid edges from the DAG.

Outputs
-------
  outputs/dags/dag_validated_{start}_{end}.json
  outputs/results/validation_{start}_{end}.csv

Usage
-----
    uv run python src/causal/validate_dag.py \\
        --dag  data/processed/causal_dags/dag_regime_09.json \\
        --start 2020-03-12 \\
        --end   2020-04-09

    # or via Python:
    from src.causal.validate_dag import CausalValidator
    import networkx as nx, pandas as pd

    G     = nx.DiGraph(...)
    panel = pd.read_parquet("data/processed/panel_daily.parquet")
    data  = panel.loc["2020-03-12":"2020-04-09"].dropna()

    validator       = CausalValidator()
    validated_dag, results_df = validator.validate(G, data)
"""

import argparse
import json
import logging
import pathlib
import warnings
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Suppress DoWhy's chatty internal loggers
for _log in ("dowhy", "dowhy.causal_model", "dowhy.causal_identifier",
             "dowhy.causal_estimators", "dowhy.causal_refuters"):
    logging.getLogger(_log).setLevel(logging.CRITICAL)

ROOT = pathlib.Path(__file__).parents[2]

# ── defaults ─────────────────────────────────────────────────────────────────
ALPHA              = 0.05    # significance threshold for both main test and refutations
N_SIMULATIONS      = 100     # permutation / bootstrap simulations per refuter
MIN_OBS            = 30      # warn (but still try) below this observation count
MAX_CONTROLS       = 10      # cap adjustment-set size to avoid underdetermined regression
REGRESSION_SPEC    = "contemporaneous"  # default per-edge regression spec
LAG_SWEEP_MAX      = 3       # max lag tried when regression_spec == "lag_sweep"
BLOCK_SIZE         = 10      # block length for block-bootstrap refuters (preserves AR structure)
BOOTSTRAP_SEED     = None    # set to an int for reproducibility; None = system entropy

# Allowed regression specifications.
SUPPORTED_SPECS = ("contemporaneous", "lag1_ar", "first_diff", "lag_sweep")

# Allowed control-selection / regression strategies.
#   corr_y           : trim by |corr(C, Y)|, then OLS                [original behaviour]
#   corr_xy_product  : trim by |corr(C, X)| × |corr(C, Y)|, then OLS [Solution 1]
#   partial_corr     : trim by |partial_corr(C, Y | X)|, then OLS    [Solution 2]
#   ridge            : no trim; RidgeCV regression with bootstrap SE [Solution 3a]
#   dml              : no trim; double machine learning via FWL      [Solution 3b]
SUPPORTED_CONTROL_STRATEGIES = (
    "corr_y", "corr_xy_product", "partial_corr", "ridge", "dml",
)
CONTROL_STRATEGY = "corr_xy_product"   # default = Solution 1
TRIM_STRATEGIES  = ("corr_y", "corr_xy_product", "partial_corr")
NO_TRIM_STRATEGIES = ("ridge", "dml")

# Multiple-testing correction applied across all edges in a single validate()
# call.  Adjusts the headline p_value column; refuter p-values are left raw
# (they are robustness gates, not discovery tests).
#   bh         : Benjamini–Hochberg FDR correction      [default]
#   bonferroni : Bonferroni FWER correction (conservative)
#   none       : no adjustment (single-test α per edge)
SUPPORTED_FDR_METHODS = ("bh", "bonferroni", "none")
FDR_METHOD = "bh"


class CausalValidator:
    """
    Validates a causal DAG against panel data using DoWhy + statsmodels OLS.

    Parameters
    ----------
    alpha : float
        Significance threshold.  Edges with p_value >= alpha are removed.
        Refutation passes if refuter p_value >= alpha (i.e., the spurious/
        unstable signal is NOT statistically significant).
    n_simulations : int
        Number of permutations / bootstrap samples for each refuter.
    min_obs : int
        Minimum observations to attempt validation.  Below this threshold a
        warning is logged and validation still runs, but results may be
        unreliable.
    max_controls : int
        Maximum number of backdoor-adjustment controls included in the OLS.
        If the identified set is larger, controls are ranked by absolute
        Pearson correlation with the outcome and the top max_controls are kept.
    """

    def __init__(
        self,
        alpha: float = ALPHA,
        n_simulations: int = N_SIMULATIONS,
        min_obs: int = MIN_OBS,
        max_controls: int = MAX_CONTROLS,
        regression_spec: str = REGRESSION_SPEC,
        lag_sweep_max: int = LAG_SWEEP_MAX,
        block_size: int = BLOCK_SIZE,
        bootstrap_seed: Optional[int] = BOOTSTRAP_SEED,
        control_strategy: str = CONTROL_STRATEGY,
        fdr_method: str = FDR_METHOD,
    ) -> None:
        if regression_spec not in SUPPORTED_SPECS:
            raise ValueError(
                f"regression_spec={regression_spec!r} not supported; "
                f"choose one of {SUPPORTED_SPECS}"
            )
        if control_strategy not in SUPPORTED_CONTROL_STRATEGIES:
            raise ValueError(
                f"control_strategy={control_strategy!r} not supported; "
                f"choose one of {SUPPORTED_CONTROL_STRATEGIES}"
            )
        if fdr_method not in SUPPORTED_FDR_METHODS:
            raise ValueError(
                f"fdr_method={fdr_method!r} not supported; "
                f"choose one of {SUPPORTED_FDR_METHODS}"
            )
        self.alpha            = alpha
        self.n_simulations    = n_simulations
        self.min_obs          = min_obs
        self.max_controls     = max_controls
        self.regression_spec  = regression_spec
        self.lag_sweep_max    = int(lag_sweep_max)
        self.block_size       = int(block_size)
        self.bootstrap_seed   = bootstrap_seed
        self.control_strategy = control_strategy
        self.fdr_method       = fdr_method

    @classmethod
    def from_config(cls, config: Union[str, pathlib.Path, dict]) -> "CausalValidator":
        """
        Build a CausalValidator from a YAML file path or a config dict.

        Accepts either:
          - the full top-level config (with a top-level "validator" section), or
          - just the inner validator-section dict.

        The optional `regression_spec_choices` field (a list of allowed spec
        names) is treated as informational and is cross-checked against the
        selected `regression_spec` if present, then stripped before init.
        """
        if isinstance(config, (str, pathlib.Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise TypeError(f"config must be path or dict, got {type(config).__name__}")
        cfg = dict(config.get("validator", config))   # copy so we can mutate

        # Optional documentation field listing all candidate specs by name.
        # Keep it out of __init__ kwargs but use it to validate the selection.
        spec_choices = cfg.pop("regression_spec_choices", None)
        chosen = cfg.get("regression_spec", REGRESSION_SPEC)
        if spec_choices is not None and chosen not in spec_choices:
            raise ValueError(
                f"regression_spec={chosen!r} is not listed in "
                f"regression_spec_choices={list(spec_choices)}"
            )

        # Same pattern for control_strategy_choices.
        cs_choices = cfg.pop("control_strategy_choices", None)
        chosen_cs = cfg.get("control_strategy", CONTROL_STRATEGY)
        if cs_choices is not None and chosen_cs not in cs_choices:
            raise ValueError(
                f"control_strategy={chosen_cs!r} is not listed in "
                f"control_strategy_choices={list(cs_choices)}"
            )

        # Same pattern for fdr_method_choices.
        fdr_choices = cfg.pop("fdr_method_choices", None)
        chosen_fdr = cfg.get("fdr_method", FDR_METHOD)
        if fdr_choices is not None and chosen_fdr not in fdr_choices:
            raise ValueError(
                f"fdr_method={chosen_fdr!r} is not listed in "
                f"fdr_method_choices={list(fdr_choices)}"
            )
        return cls(**cfg)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_refutation_pvalue(refutation) -> float:
        """
        Extract a numeric p-value from a DoWhy 0.14 CausalRefutation object.

        DoWhy 0.14 stores results in `refutation_result` as a dict:
          {'p_value': float, 'is_statistically_significant': bool}

        Interpretation (consistent across both refuters):
          - placebo_treatment_refuter p_value: significance of the refuted
            (permuted-treatment) estimate.  If significant (p < alpha), the
            permuted treatment still "causes" Y → original edge may be spurious.
            We KEEP edges where placebo_p >= alpha.
          - data_subset_refuter p_value: significance of the deviation between
            the subset estimate and the original.  If significant (p < alpha),
            the estimate is unstable across subsets.
            We KEEP edges where subset_p >= alpha.
        """
        result = getattr(refutation, "refutation_result", None)
        if isinstance(result, dict) and "p_value" in result:
            return float(result["p_value"])
        # older DoWhy: scalar float
        if isinstance(result, (int, float)) and not isinstance(result, bool):
            return float(result)
        return np.nan

    def _trim_controls(
        self,
        controls: list[str],
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
    ) -> list[str]:
        """
        Rank candidate controls and keep at most max_controls of them.

        The ranking metric is chosen by self.control_strategy:
          - corr_y          : |corr(C, Y)|                    [predictive — original]
          - corr_xy_product : |corr(C, X)| × |corr(C, Y)|     [Solution 1: confounder proxy]
          - partial_corr    : |partial_corr(C, Y | X)|        [Solution 2: marginal-on-X impact]
          - ridge / dml     : no trim — return controls unchanged

        For ridge / dml the regression itself handles many controls, so the
        cap is irrelevant.
        """
        if self.control_strategy in NO_TRIM_STRATEGIES:
            return controls
        if len(controls) <= self.max_controls:
            return controls

        if self.control_strategy == "corr_y":
            score = data[controls].corrwith(data[outcome]).abs()

        elif self.control_strategy == "corr_xy_product":
            cx = data[controls].corrwith(data[treatment]).abs()
            cy = data[controls].corrwith(data[outcome]).abs()
            score = cx * cy

        elif self.control_strategy == "partial_corr":
            # |partial_corr(C, Y | X)| via residualisation (FWL):
            #   res_y = Y residualised on X
            #   for each control C: res_c = C residualised on X
            #   score[C] = |corr(res_c, res_y)|
            X_const = sm.add_constant(data[[treatment]])
            res_y   = sm.OLS(data[outcome], X_const).fit().resid
            score   = pd.Series(index=controls, dtype=float)
            for c in controls:
                res_c    = sm.OLS(data[c], X_const).fit().resid
                score[c] = float(abs(np.corrcoef(res_c, res_y)[0, 1]))

        else:
            # Should not reach here given __init__ validation.
            raise ValueError(f"unsupported control_strategy: {self.control_strategy!r}")

        return list(score.nlargest(self.max_controls).index)

    # ── multiple-testing correction across all edges in this validate() ───────

    def _apply_fdr_correction(self, p_values: pd.Series) -> pd.Series:
        """
        Adjust the per-edge headline p-values for multiple testing across all
        edges submitted to validate().  NaNs (skipped edges) are passed
        through unchanged.

        - "bh"         → Benjamini-Hochberg, controls FDR at self.alpha.
        - "bonferroni" → Bonferroni, controls family-wise error rate.
        - "none"       → returns the input unchanged.
        """
        if self.fdr_method == "none":
            return p_values.copy()

        from statsmodels.stats.multitest import multipletests
        method_map = {"bh": "fdr_bh", "bonferroni": "bonferroni"}

        result = p_values.copy().astype(float)
        mask   = result.notna()
        if mask.sum() == 0:
            return result
        _, p_adj, _, _ = multipletests(
            result[mask].values,
            alpha=self.alpha,
            method=method_map[self.fdr_method],
        )
        result.loc[mask] = p_adj
        return result

    # ── coefficient + p-value dispatch (depends on control_strategy) ──────────

    def _fit_headline(
        self,
        df: pd.DataFrame,
        outcome: str,
        treatment: str,
        predictors: list[str],
    ) -> tuple[float, float]:
        """
        Return (effect_size, p_value) for the treatment using whichever
        regression matches self.control_strategy:

          - OLS-based strategies (corr_y, corr_xy_product, partial_corr):
                ordinary least squares of `outcome` on `predictors`
                (predictors already include treatment + chosen controls)

          - ridge:
                RidgeCV regression on the same predictors; p-value via
                block-bootstrap of the treatment coefficient against H0=0

          - dml:
                Frisch-Waugh-Lovell with Ridge residualisation:
                  res_Y = Y − Ridge(Y ~ confounders).predict(confounders)
                  res_X = X − Ridge(X ~ confounders).predict(confounders)
                OLS of res_Y on res_X gives the structural treatment effect
                with valid SE / p-value (via OLS asymptotics on residuals)
        """
        strategy = self.control_strategy

        if strategy in ("corr_y", "corr_xy_product", "partial_corr"):
            X_mat = sm.add_constant(df[predictors])
            ols   = sm.OLS(df[outcome], X_mat).fit()
            return float(ols.params[treatment]), float(ols.pvalues[treatment])

        if strategy == "ridge":
            from sklearn.linear_model import RidgeCV
            alphas = np.logspace(-2, 2, 20)
            X_arr  = df[predictors].to_numpy()
            y_arr  = df[outcome].to_numpy()
            model  = RidgeCV(alphas=alphas).fit(X_arr, y_arr)
            t_idx  = predictors.index(treatment)
            coef   = float(model.coef_[t_idx])
            # block-bootstrap p-value against H0: beta_treatment = 0
            p = self._ridge_bootstrap_pvalue(
                df, outcome, treatment, predictors, alpha=float(model.alpha_)
            )
            return coef, p

        if strategy == "dml":
            from sklearn.linear_model import RidgeCV
            confounders = [c for c in predictors if c != treatment]
            if not confounders:
                # nothing to residualise on — degenerate to plain OLS
                X_mat = sm.add_constant(df[[treatment]])
                ols   = sm.OLS(df[outcome], X_mat).fit()
                return float(ols.params[treatment]), float(ols.pvalues[treatment])

            alphas = np.logspace(-2, 2, 20)
            Z      = df[confounders].to_numpy()
            y_arr  = df[outcome].to_numpy()
            t_arr  = df[treatment].to_numpy()
            res_y  = y_arr - RidgeCV(alphas=alphas).fit(Z, y_arr).predict(Z)
            res_t  = t_arr - RidgeCV(alphas=alphas).fit(Z, t_arr).predict(Z)
            X_mat  = sm.add_constant(res_t.reshape(-1, 1))
            ols    = sm.OLS(res_y, X_mat).fit()
            return float(ols.params[1]), float(ols.pvalues[1])

        raise ValueError(f"unsupported control_strategy: {strategy!r}")

    def _ridge_bootstrap_pvalue(
        self,
        df: pd.DataFrame,
        outcome: str,
        treatment: str,
        predictors: list[str],
        alpha: float,
        n_boot: int = 200,
    ) -> float:
        """
        Block-bootstrap two-sided p-value for the treatment coefficient in a
        Ridge regression, against H0: beta_treatment = 0. Uses the same block
        size as the refuters so AR structure is preserved.
        """
        from sklearn.linear_model import Ridge
        from scipy import stats
        rng    = np.random.default_rng(self.bootstrap_seed)
        n      = len(df)
        t_idx  = predictors.index(treatment)
        coefs  = []
        for _ in range(n_boot):
            idx    = self._block_bootstrap_indices(n, self.block_size, rng)
            sample = df.iloc[idx].reset_index(drop=True)
            try:
                model = Ridge(alpha=alpha).fit(
                    sample[predictors].to_numpy(), sample[outcome].to_numpy(),
                )
                coefs.append(float(model.coef_[t_idx]))
            except Exception:
                continue
        if len(coefs) < 5:
            return float("nan")
        coefs = np.array(coefs)
        sd    = float(np.std(coefs))
        if sd < 1e-12:
            return 1.0 if abs(np.mean(coefs)) < 1e-12 else 0.0
        z = abs(float(np.mean(coefs))) / sd
        return float(2.0 * (1.0 - stats.norm.cdf(z)))

    # ── regression-spec dispatch ──────────────────────────────────────────────

    def _fit_spec(
        self,
        X: str,
        Y: str,
        controls: list[str],
        data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Fit the configured regression spec for edge X → Y and return:
            effect_size      : float — coefficient on the treatment
            p_value          : float — p-value on the treatment (Bonferroni-adjusted for lag_sweep)
            regression_frame : pd.DataFrame — fully-aligned frame the OLS used
                               (post-shift / post-diff; includes AR column for lag1_ar)
            outcome_col      : str — name of the outcome column in regression_frame
            treatment_col    : str — name of the treatment column in regression_frame
            predictor_cols   : list[str] — columns used as predictors in the OLS
                               (X first, then AR control if any, then controls)
            best_lag         : int   — lag actually used
            n_obs            : int   — rows used after dropna / shift
            spec_extras      : dict  — optional metadata (e.g. AR coefficient for lag1_ar)
        """
        spec = self.regression_spec

        if spec == "contemporaneous":
            df = data[[Y, X] + controls].dropna()
            predictor_cols = [X] + controls
            coef, p = self._fit_headline(df, Y, X, predictor_cols)
            return {
                "effect_size":      coef,
                "p_value":          p,
                "regression_frame": df,
                "outcome_col":      Y,
                "treatment_col":    X,
                "predictor_cols":   predictor_cols,
                "best_lag":         0,
                "n_obs":            len(df),
                "spec_extras":      {},
            }

        if spec == "lag1_ar":
            ar_col = f"__{Y}_lag1"
            df = pd.concat(
                [
                    data[[Y]],
                    data[[X] + controls].shift(1),
                    data[[Y]].shift(1).rename(columns={Y: ar_col}),
                ],
                axis=1,
            ).dropna()
            predictor_cols = [X, ar_col] + controls
            coef, p = self._fit_headline(df, Y, X, predictor_cols)
            # AR coefficient is informative regardless of fitting method, but
            # only available when the headline fit was OLS-based; for ridge/dml
            # we just skip it.
            extras: dict = {}
            if self.control_strategy in ("corr_y", "corr_xy_product", "partial_corr"):
                X_mat = sm.add_constant(df[predictor_cols])
                ols   = sm.OLS(df[Y], X_mat).fit()
                extras["ar_coef"] = float(ols.params[ar_col])
            return {
                "effect_size":      coef,
                "p_value":          p,
                "regression_frame": df,
                "outcome_col":      Y,
                "treatment_col":    X,
                "predictor_cols":   predictor_cols,
                "best_lag":         1,
                "n_obs":            len(df),
                "spec_extras":      extras,
            }

        if spec == "first_diff":
            df = data[[Y, X] + controls].diff().dropna()
            predictor_cols = [X] + controls
            coef, p = self._fit_headline(df, Y, X, predictor_cols)
            return {
                "effect_size":      coef,
                "p_value":          p,
                "regression_frame": df,
                "outcome_col":      Y,
                "treatment_col":    X,
                "predictor_cols":   predictor_cols,
                "best_lag":         0,
                "n_obs":            len(df),
                "spec_extras":      {"differenced": True},
            }

        if spec == "lag_sweep":
            # For each lag k, fit using the configured control_strategy and
            # record its p-value; pick the lag with the smallest p-value.
            best: Optional[dict] = None
            for k in range(0, self.lag_sweep_max + 1):
                if k == 0:
                    df_k = data[[Y, X] + controls].dropna()
                else:
                    df_k = pd.concat(
                        [data[[Y]], data[[X] + controls].shift(k)], axis=1
                    ).dropna()
                if len(df_k) < len(controls) + 2:
                    continue
                try:
                    coef_k, p_k = self._fit_headline(df_k, Y, X, [X] + controls)
                except Exception:
                    continue
                if best is None or p_k < best["p_raw"]:
                    best = {"lag": k, "p_raw": p_k, "coef": coef_k, "df": df_k}
            n_lags_tested = self.lag_sweep_max + 1
            if best is None:
                return {
                    "effect_size":      np.nan,
                    "p_value":          np.nan,
                    "regression_frame": data[[Y, X] + controls].dropna(),
                    "outcome_col":      Y,
                    "treatment_col":    X,
                    "predictor_cols":   [X] + controls,
                    "best_lag":         -1,
                    "n_obs":            0,
                    "spec_extras":      {"raw_p": np.nan, "n_lags_tested": 0},
                }
            return {
                "effect_size":      best["coef"],
                "p_value":          min(1.0, best["p_raw"] * n_lags_tested),  # Bonferroni
                "regression_frame": best["df"],
                "outcome_col":      Y,
                "treatment_col":    X,
                "predictor_cols":   [X] + controls,
                "best_lag":         best["lag"],
                "n_obs":            len(best["df"]),
                "spec_extras":      {"raw_p": best["p_raw"], "n_lags_tested": n_lags_tested},
            }

        raise ValueError(f"Unknown regression_spec: {spec!r}")

    # ── block-bootstrap refuters (replace DoWhy's iid refuters) ───────────────
    #
    # Time-series validity: DoWhy's placebo_treatment_refuter (np.random.choice
    # row-shuffle) and data_subset_refuter (data.sample(frac=...) random row
    # drop) both assume i.i.d. observations — they destroy the autocorrelation
    # structure of the data.  The two helpers below sample contiguous *blocks*
    # of length `block_size`, which preserves local AR persistence while still
    # producing valid resamples.

    @staticmethod
    def _block_permute(x: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
        """
        Build a length-N permutation of x by concatenating random contiguous
        blocks of length `block_size` sampled with replacement from x.
        Local autocorrelation inside each block is preserved.
        """
        n = len(x)
        b = max(1, min(block_size, n))
        n_blocks = (n + b - 1) // b
        out = np.empty(n_blocks * b, dtype=x.dtype)
        starts = rng.integers(0, n - b + 1, size=n_blocks)
        for i, s in enumerate(starts):
            out[i * b : (i + 1) * b] = x[s : s + b]
        return out[:n]

    @staticmethod
    def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
        """Indices for a block-bootstrap resample of length n."""
        b = max(1, min(block_size, n))
        n_blocks = (n + b - 1) // b
        starts = rng.integers(0, n - b + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + b) for s in starts])
        return idx[:n]

    def _block_bootstrap_placebo_p(
        self,
        frame: pd.DataFrame,
        outcome: str,
        treatment: str,
        predictors: list[str],
        rng: np.random.Generator,
    ) -> float:
        """
        Block-permutation placebo test.

        For each simulation: replace the treatment column with a block-
        permutation of itself (preserves AR, breaks alignment with Y), refit
        Y ~ predictors, and record the p-value of the (placebo) treatment
        coefficient.  Return the mean placebo p-value across simulations.

        Convention (matches DoWhy):
            high p (≥ alpha) → placebos look null → original signal is real → KEEP edge
            low  p (< alpha) → placebos often look causal → original is suspicious → DROP edge
        """
        frame = frame.reset_index(drop=True)
        x_orig = frame[treatment].to_numpy()
        ps: list[float] = []
        for _ in range(self.n_simulations):
            permuted = self._block_permute(x_orig, self.block_size, rng)
            tmp = frame.copy()
            tmp[treatment] = permuted
            try:
                X_mat = sm.add_constant(tmp[predictors])
                ols   = sm.OLS(tmp[outcome], X_mat).fit()
                ps.append(float(ols.pvalues[treatment]))
            except Exception:
                continue
        if not ps:
            return float("nan")
        return float(np.mean(ps))

    def _block_bootstrap_subset_p(
        self,
        frame: pd.DataFrame,
        outcome: str,
        treatment: str,
        predictors: list[str],
        original_coef: float,
        rng: np.random.Generator,
    ) -> float:
        """
        Block-bootstrap stability test.

        For each simulation: resample contiguous blocks (with replacement) from
        the rows of `frame` to length n, refit, record the treatment coefficient.
        Return a two-sided empirical p-value of `original_coef` in that
        bootstrap distribution.

        Convention (matches DoWhy data_subset_refuter):
            high p (≥ alpha) → original_coef sits in the bulk → estimate stable → KEEP
            low  p (< alpha) → original_coef in the tail → estimate unstable → DROP
        """
        frame = frame.reset_index(drop=True)
        n = len(frame)
        coefs: list[float] = []
        for _ in range(self.n_simulations):
            idx    = self._block_bootstrap_indices(n, self.block_size, rng)
            sample = frame.iloc[idx].reset_index(drop=True)
            try:
                X_mat = sm.add_constant(sample[predictors])
                ols   = sm.OLS(sample[outcome], X_mat).fit()
                coefs.append(float(ols.params[treatment]))
            except Exception:
                continue
        if len(coefs) < 5:
            return float("nan")
        coefs = np.sort(np.array(coefs))
        # rank of original_coef in the bootstrap distribution → two-sided p
        rank   = float(np.searchsorted(coefs, original_coef)) / len(coefs)
        p_two  = 2.0 * min(rank, 1.0 - rank)
        return float(min(1.0, p_two))

    # ── single-edge validation ────────────────────────────────────────────────

    def _validate_edge(
        self,
        X: str,
        Y: str,
        dag: nx.DiGraph,
        data: pd.DataFrame,
    ) -> dict:
        """
        Validate edge X → Y using the full DAG for causal identification.

        Returns a dict with keys:
          from, to, effect_size, p_value,
          placebo_refuter_p, subset_refuter_p,
          n_controls, n_obs, valid, skip_reason
        """
        row: dict = {
            "from": X, "to": Y,
            "effect_size": np.nan, "p_value": np.nan,
            "placebo_refuter_p": np.nan, "subset_refuter_p": np.nan,
            "n_controls": np.nan, "n_obs": len(data),
            "regression_spec": self.regression_spec, "best_lag": np.nan,
            "block_size": self.block_size,
            "control_strategy": self.control_strategy,
            "valid": False, "skip_reason": None,
        }

        # Skip if either variable is missing from data
        for var in (X, Y):
            if var not in data.columns:
                row["skip_reason"] = f"variable '{var}' not in data"
                return row

        # Skip if edge not actually in the DAG (safety check)
        if not dag.has_edge(X, Y):
            row["skip_reason"] = "edge not in DAG"
            return row

        if len(data) < self.min_obs:
            logger.warning(
                "  %s → %s: only %d obs (< %d) — results may be unreliable",
                X, Y, len(data), self.min_obs,
            )

        try:
            # ── import here to keep startup fast ────────────────────────────
            from dowhy import CausalModel

            # ── 1. Identify (on original data + DAG) ────────────────────────
            id_model = CausalModel(
                data=data,
                treatment=X,
                outcome=Y,
                graph=dag,
                logging_level=logging.CRITICAL,
            )
            estimand = id_model.identify_effect(proceed_when_unidentifiable=True)

            # Backdoor adjustment set (may be empty if no confounders)
            controls = [
                v for v in estimand.get_backdoor_variables()
                if v in data.columns and v not in (X, Y)
            ]
            controls = self._trim_controls(controls, data, Y, X)

            # Guard: need at least 2 df for the regression
            if len(data) < len(controls) + 2:
                extra = len(controls) + 2 - len(data)
                controls = controls[:-extra] if extra < len(controls) else []

            row["n_controls"] = len(controls)

            # ── 2. Headline OLS via configured regression spec ──────────────
            fit = self._fit_spec(X, Y, controls, data)
            row["effect_size"] = round(fit["effect_size"], 6) if not np.isnan(fit["effect_size"]) else np.nan
            row["p_value"]     = round(fit["p_value"], 6)     if not np.isnan(fit["p_value"])     else np.nan
            row["best_lag"]    = fit["best_lag"]
            row["n_obs"]       = fit["n_obs"]
            if fit["spec_extras"]:
                for k, v in fit["spec_extras"].items():
                    row[f"spec_{k}"] = v

            # ── 3. Block-bootstrap refuters on the regression frame ─────────
            # Replaces DoWhy's iid placebo_treatment_refuter and
            # data_subset_refuter, which destroy temporal structure.  Both
            # helpers operate on the spec-aligned frame, so they honour the
            # configured regression_spec automatically.
            rng = np.random.default_rng(self.bootstrap_seed)

            row["placebo_refuter_p"] = round(
                self._block_bootstrap_placebo_p(
                    frame=fit["regression_frame"],
                    outcome=fit["outcome_col"],
                    treatment=fit["treatment_col"],
                    predictors=fit["predictor_cols"],
                    rng=rng,
                ),
                6,
            )

            row["subset_refuter_p"] = round(
                self._block_bootstrap_subset_p(
                    frame=fit["regression_frame"],
                    outcome=fit["outcome_col"],
                    treatment=fit["treatment_col"],
                    predictors=fit["predictor_cols"],
                    original_coef=fit["effect_size"],
                    rng=rng,
                ),
                6,
            )

            # ── 6. Validity decision ────────────────────────────────────────
            placebo_ok = (
                np.isnan(row["placebo_refuter_p"])  # can't refute → keep
                or row["placebo_refuter_p"] >= self.alpha
            )
            subset_ok = (
                np.isnan(row["subset_refuter_p"])
                or row["subset_refuter_p"] >= self.alpha
            )
            row["valid"] = (
                row["p_value"] < self.alpha
                and placebo_ok
                and subset_ok
            )

        except Exception as exc:
            row["skip_reason"] = str(exc)[:120]
            logger.warning("  %s → %s: EXCEPTION — %s", X, Y, exc)

        return row

    # ── public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        dag: nx.DiGraph,
        data: pd.DataFrame,
    ) -> tuple[nx.DiGraph, pd.DataFrame]:
        """
        Validate every edge in `dag` against `data`.

        Parameters
        ----------
        dag : nx.DiGraph
            Causal DAG to validate (typically the LLM-generated one).
        data : pd.DataFrame
            Panel slice for the regime period (already z-score standardised).

        Returns
        -------
        validated_dag : nx.DiGraph
            Pruned DAG containing only edges that pass validation.
        results_df : pd.DataFrame
            One row per edge with effect_size, p_value, refuter p-values, valid flag.
        """
        data = data.dropna()
        edges = list(dag.edges())
        n = len(edges)
        logger.info(
            "Validating %d edges  |  %d obs  |  %d variables",
            n, len(data), len(data.columns),
        )
        if len(data) < self.min_obs:
            logger.warning(
                "Only %d observations — validation results will be unreliable "
                "(regime period too short for reliable regression).", len(data)
            )

        rows = []
        for i, (X, Y) in enumerate(edges):
            logger.info("  [%d/%d] %s → %s", i + 1, n, X, Y)
            row = self._validate_edge(X, Y, dag, data)
            # Per-edge `valid` from _validate_edge is single-test; we will
            # overwrite it below using FDR-adjusted p-values across all edges.
            row["valid_raw"] = row["valid"]
            rows.append(row)
            _fmt = lambda v: float(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else float("nan")
            logger.info(
                "         p=%.4f  placebo_p=%.4f  subset_p=%.4f  raw_valid=%s",
                _fmt(row.get("p_value")),
                _fmt(row.get("placebo_refuter_p")),
                _fmt(row.get("subset_refuter_p")),
                row["valid_raw"],
            )

        results_df = pd.DataFrame(rows)

        # ── Multiple-testing correction across all edges ──────────────────────
        # Adjusts the headline p_value column; refuter p-values stay raw.
        # Final `valid` uses adjusted p-value AND refuter pass conditions.
        results_df["p_value_adjusted"] = self._apply_fdr_correction(results_df["p_value"])
        results_df["fdr_method"]       = self.fdr_method

        p_adj_pass   = results_df["p_value_adjusted"].notna() & (
            results_df["p_value_adjusted"] < self.alpha
        )
        placebo_pass = results_df["placebo_refuter_p"].isna() | (
            results_df["placebo_refuter_p"] >= self.alpha
        )
        subset_pass  = results_df["subset_refuter_p"].isna() | (
            results_df["subset_refuter_p"] >= self.alpha
        )
        results_df["valid"] = p_adj_pass & placebo_pass & subset_pass

        n_raw_valid = int(results_df["valid_raw"].sum())
        n_valid     = int(results_df["valid"].sum())
        if self.fdr_method != "none" and n_raw_valid != n_valid:
            logger.info(
                "FDR adjustment (%s, α=%.3f): %d → %d edges valid (%d dropped by multiple-testing)",
                self.fdr_method, self.alpha, n_raw_valid, n_valid, n_raw_valid - n_valid,
            )

        # Build pruned DAG: keep all nodes, only valid edges
        validated_dag = nx.DiGraph()
        validated_dag.add_nodes_from(dag.nodes())
        for _, row in results_df[results_df["valid"]].iterrows():
            validated_dag.add_edge(row["from"], row["to"])

        n_removed = n - n_valid
        logger.info(
            "Validation complete: %d / %d edges kept  |  %d removed  (fdr=%s)",
            n_valid, n, n_removed, self.fdr_method,
        )
        if not nx.is_directed_acyclic_graph(validated_dag):
            logger.warning("Validated DAG contains cycles — check for issues.")

        return validated_dag, results_df


# ── output helpers ────────────────────────────────────────────────────────────

def save_results(
    validated_dag: nx.DiGraph,
    results_df: pd.DataFrame,
    start: str,
    end: str,
    out_root: pathlib.Path = ROOT / "outputs",
) -> tuple[pathlib.Path, pathlib.Path]:
    """
    Persist the validated DAG and the validation table.

    Returns (dag_path, csv_path).
    """
    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")

    dag_dir = out_root / "dags"
    res_dir = out_root / "results"
    dag_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    dag_path = dag_dir / f"dag_validated_{s}_{e}.json"
    csv_path = res_dir / f"validation_{s}_{e}.csv"

    # DAG JSON
    payload = {
        "start":   str(start),
        "end":     str(end),
        "n_edges": validated_dag.number_of_edges(),
        "is_dag":  nx.is_directed_acyclic_graph(validated_dag),
        "edges": [
            {"from": u, "to": v}
            for u, v in validated_dag.edges()
        ],
    }
    with open(dag_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Validation table CSV
    results_df.to_csv(csv_path, index=False)

    logger.info("Saved validated DAG → %s", dag_path)
    logger.info("Saved validation table → %s", csv_path)

    return dag_path, csv_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def run(
    dag_path: pathlib.Path,
    start: str,
    end: str,
    panel_path: pathlib.Path = ROOT / "data" / "processed" / "panel_daily.parquet",
    alpha: float = ALPHA,
    n_simulations: int = N_SIMULATIONS,
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Load a DAG JSON + panel slice, validate, and save outputs."""
    # Load DAG
    with open(dag_path) as f:
        dag_data = json.load(f)
    dag = nx.DiGraph()
    for e in dag_data["edges"]:
        dag.add_edge(e["from"], e["to"])

    # Slice panel
    panel = pd.read_parquet(panel_path)
    data  = panel.loc[pd.Timestamp(start): pd.Timestamp(end)].copy()

    logger.info(
        "Loaded DAG: %d edges  |  Panel slice: %d obs (%s → %s)",
        dag.number_of_edges(), len(data), start, end,
    )

    validator = CausalValidator(alpha=alpha, n_simulations=n_simulations)
    validated_dag, results_df = validator.validate(dag, data)
    save_results(validated_dag, results_df, start, end)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Validation summary  {start} → {end}")
    print(f"{'='*60}")
    print(f"  Input edges     : {dag.number_of_edges()}")
    print(f"  Valid edges     : {results_df['valid'].sum()}")
    print(f"  Removed edges   : {(~results_df['valid']).sum()}")
    print(f"  Skipped (error) : {results_df['skip_reason'].notna().sum()}")
    print()

    removed = results_df[~results_df["valid"]][
        ["from", "to", "p_value", "placebo_refuter_p", "subset_refuter_p", "skip_reason"]
    ]
    if len(removed):
        print("Removed edges:")
        for _, r in removed.iterrows():
            reason = (
                r.get("skip_reason")
                or f"p={r['p_value']:.3f}, placebo_p={r['placebo_refuter_p']:.3f}, "
                   f"subset_p={r['subset_refuter_p']:.3f}"
            )
            print(f"  {r['from']:20s} → {r['to']:20s}  ({reason})")

    return validated_dag, results_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Validate LLM-proposed causal edges with DoWhy + OLS"
    )
    parser.add_argument(
        "--dag", type=pathlib.Path,
        default=ROOT / "data" / "processed" / "causal_dags" / "dag_regime_09.json",
        help="Path to the DAG JSON file (default: dag_regime_09.json)",
    )
    parser.add_argument(
        "--start", type=str, default="2020-03-12",
        help="Regime start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default="2020-04-09",
        help="Regime end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA,
        help=f"Significance threshold (default {ALPHA})",
    )
    parser.add_argument(
        "--simulations", type=int, default=N_SIMULATIONS,
        help=f"Refuter simulations (default {N_SIMULATIONS})",
    )
    args = parser.parse_args()

    run(
        dag_path=args.dag,
        start=args.start,
        end=args.end,
        alpha=args.alpha,
        n_simulations=args.simulations,
    )
