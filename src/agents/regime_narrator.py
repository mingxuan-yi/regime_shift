"""
Regime narrator: generates a 2-3 sentence natural language description of a
macroeconomic regime from its summary statistics.

The description is consumed by CausalAgent as the `regime_description` context
string, grounding the LLM's causal reasoning in the actual data for that period.

Inputs
------
- Date range (start, end)
- Per-variable summary statistics (mean / std / min / max z-score) for the period
- Per-variable summary statistics for the *previous* regime (optional — used to
  identify the largest structural shifts at the regime boundary)
- Variable definitions from variable_definitions.yaml

Output
------
- A single string: 2-3 sentences in the voice of a macro strategist.

Usage
-----
    import pandas as pd
    import yaml
    from src.agents.regime_narrator import RegimeNarrator

    panel = pd.read_parquet('data/processed/panel_daily.parquet')
    with open('data/processed/variable_definitions.yaml') as f:
        var_defs = yaml.safe_load(f)['columns']

    narrator = RegimeNarrator()

    # From raw panel slices (recommended)
    desc = narrator.describe_from_panel(
        panel=panel,
        start='2020-03-12',
        end='2020-04-09',
        variable_definitions=var_defs,
        prev_start='2019-10-16',
        prev_end='2020-03-12',
    )

    # From pre-computed stats
    desc = narrator.describe(
        start='2020-03-12',
        end='2020-04-09',
        variable_stats={'yield_10y': {'mean': -2.1, 'std': 0.4, 'min': -3.0, 'max': -1.2}, ...},
        variable_definitions=var_defs,
    )
"""

import logging
import os
import time
from typing import Optional

import anthropic
import pandas as pd

logger = logging.getLogger(__name__)

MODEL            = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS       = 512     # description is short — no need for large budget
TEMPERATURE      = 0.3     # low: we want consistent, factual descriptions
N_TOP_EXTREME    = 8       # most extreme variables to include in the prompt
N_TOP_SHIFTS     = 6       # largest shifts from previous regime


class RegimeNarrator:
    """
    Generates a concise natural language regime description from data statistics.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
    model : str
        Claude model identifier.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "Anthropic API key required. "
                "Pass api_key= or set the ANTHROPIC_API_KEY environment variable."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model  = model

    # ── API call ──────────────────────────────────────────────────────────────

    def _call(self, prompt: str, retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=(
                        "You are a senior macro strategist. "
                        "Write precise, data-grounded regime descriptions."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip()
            except anthropic.RateLimitError:
                wait = 5 * (2 ** attempt)
                logger.warning("Rate limit; waiting %ds …", wait)
                time.sleep(wait)
            except anthropic.APIError as exc:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError("API call failed after retries.")

    # ── prompt construction ───────────────────────────────────────────────────

    @staticmethod
    def _format_extreme(
        variable_stats: dict[str, dict],
        variable_definitions: dict,
        n: int,
    ) -> str:
        """Top-N variables by |mean z-score|, formatted for the prompt."""
        ranked = sorted(
            variable_stats.items(),
            key=lambda x: abs(x[1].get("mean", 0.0)),
            reverse=True,
        )[:n]
        lines = []
        for var, stats in ranked:
            meta      = variable_definitions.get(var, {})
            human     = meta.get("human_name", var) if isinstance(meta, dict) else var
            mean_z    = stats.get("mean", 0.0)
            direction = "elevated" if mean_z > 0 else "depressed"
            lines.append(
                f"  {human} ({var}): mean = {mean_z:+.2f}σ  [{direction}]"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_shifts(
        variable_stats: dict[str, dict],
        prev_stats: dict[str, dict],
        variable_definitions: dict,
        n: int,
    ) -> str:
        """Top-N variables by |Δmean z-score| vs previous regime."""
        deltas = []
        for var, stats in variable_stats.items():
            if var not in prev_stats:
                continue
            delta    = stats.get("mean", 0.0) - prev_stats[var].get("mean", 0.0)
            curr     = stats.get("mean", 0.0)
            deltas.append((var, delta, curr))
        deltas.sort(key=lambda x: abs(x[1]), reverse=True)

        lines = []
        for var, delta, curr in deltas[:n]:
            meta      = variable_definitions.get(var, {})
            human     = meta.get("human_name", var) if isinstance(meta, dict) else var
            direction = "↑" if delta > 0 else "↓"
            lines.append(
                f"  {human} ({var}): {direction} {abs(delta):.2f}σ  (now {curr:+.2f}σ)"
            )
        return "\n".join(lines) if lines else "  (no previous regime data provided)"

    def _build_prompt(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        variable_stats: dict[str, dict],
        variable_definitions: dict,
        prev_stats: Optional[dict[str, dict]],
    ) -> str:
        n_days = (end - start).days

        extreme_text = self._format_extreme(
            variable_stats, variable_definitions, N_TOP_EXTREME
        )
        shift_text = (
            self._format_shifts(
                variable_stats, prev_stats, variable_definitions, N_TOP_SHIFTS
            )
            if prev_stats is not None
            else "  (no previous regime data provided)"
        )

        return f"""You are writing a 2-3 sentence macroeconomic regime description for use as
context in a causal inference model. The description must be specific and data-grounded.

Regime date range : {start.date()} → {end.date()}  ({n_days} days)

Most extreme conditions in this regime (mean z-score vs rolling 252-day baseline):
{extreme_text}

Largest shifts vs the previous regime (Δ mean z-score):
{shift_text}

Instructions:
- Write exactly 2-3 sentences.
- State which variables are most elevated or depressed and in which direction.
- Name the dominant macro dynamic (e.g. risk-off flight-to-safety, aggressive tightening,
  disinflation, credit stress, reflation) and its likely cause given the date range.
- Be specific — avoid vague phrases like "mixed environment" or "uncertain conditions".
- Write as a macro strategist briefing a quant team, not as an academic.

Return only the description. No preamble, no bullet points."""

    # ── public API ────────────────────────────────────────────────────────────

    def describe(
        self,
        start,
        end,
        variable_stats: dict[str, dict],
        variable_definitions: dict,
        prev_stats: Optional[dict[str, dict]] = None,
    ) -> str:
        """
        Generate a 2-3 sentence regime description from pre-computed statistics.

        Parameters
        ----------
        start, end : str or pd.Timestamp
            Regime date boundaries.
        variable_stats : dict
            {var_name: {"mean": float, "std": float, "min": float, "max": float}}
            All values are z-scores (panel is assumed already standardised).
        variable_definitions : dict
            YAML-loaded variable metadata (human_name, units, definition per var).
        prev_stats : dict, optional
            Same structure as variable_stats but for the preceding regime.
            Used to surface the largest structural shifts at the boundary.

        Returns
        -------
        str
            2-3 sentence natural language regime description.
        """
        start = pd.Timestamp(start)
        end   = pd.Timestamp(end)
        prompt = self._build_prompt(start, end, variable_stats, variable_definitions, prev_stats)
        description = self._call(prompt)
        logger.info(
            "Narrator: %s → %s  (%d chars)",
            start.date(), end.date(), len(description),
        )
        return description

    def describe_from_panel(
        self,
        panel: pd.DataFrame,
        start,
        end,
        variable_definitions: dict,
        prev_start=None,
        prev_end=None,
    ) -> str:
        """
        Convenience wrapper: slice the panel, compute stats, then describe.

        Parameters
        ----------
        panel : pd.DataFrame
            Rolling z-score standardised panel (panel_daily.parquet).
        start, end : str or pd.Timestamp
            Date boundaries of the current regime.
        variable_definitions : dict
            YAML-loaded variable metadata.
        prev_start, prev_end : str or pd.Timestamp, optional
            Date boundaries of the previous regime. When provided, the narrator
            includes the largest z-score shifts across the boundary.

        Returns
        -------
        str
            2-3 sentence natural language regime description.
        """
        start = pd.Timestamp(start)
        end   = pd.Timestamp(end)

        def _stats(df_slice: pd.DataFrame) -> dict[str, dict]:
            return {
                col: {
                    "mean": float(df_slice[col].mean()),
                    "std":  float(df_slice[col].std()),
                    "min":  float(df_slice[col].min()),
                    "max":  float(df_slice[col].max()),
                }
                for col in df_slice.columns
                if df_slice[col].notna().any()
            }

        curr_stats = _stats(panel.loc[start:end])

        prev_stats = None
        if prev_start is not None and prev_end is not None:
            prev_start = pd.Timestamp(prev_start)
            prev_end   = pd.Timestamp(prev_end)
            prev_stats = _stats(panel.loc[prev_start:prev_end])

        return self.describe(start, end, curr_stats, variable_definitions, prev_stats)
