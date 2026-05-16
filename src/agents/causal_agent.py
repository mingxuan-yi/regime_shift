"""
LLM-driven causal DAG construction agent (Sokolov et al. three-stage pipeline).

Stage 1 — Causal Exploration
    Prompt Claude N_EXPLORATION_ROUNDS times to propose plausible causal edges
    with justifications.  Collect the union into an undirected candidate set.

Stage 2 — Causal Inference
    Give the LLM the undirected candidate set and ask it to orient each edge.
    Validate acyclicity with networkx; ask the LLM to resolve any cycles.

Stage 3 — Causal Validation
    Give the LLM the full DAG and ask it to correct errors (missing edges,
    false positives, wrong directions).  Apply corrections.  Repeat once.

Usage
-----
    import yaml
    from src.agents.causal_agent import CausalAgent

    with open("data/processed/variable_definitions.yaml") as f:
        var_defs = yaml.safe_load(f)

    # API key via env var (recommended):
    #   export ANTHROPIC_API_KEY=sk-ant-...
    # or pass explicitly:
    #   agent = CausalAgent(api_key="sk-ant-...")

    agent = CausalAgent()
    result = agent.run(
        variable_definitions=var_defs,
        regime_description="COVID onset: extreme volatility, flight to safety",
    )
    print(result["n_edges"], "edges,", "is_dag =", result["is_dag"])
"""

import json
import logging
import os
import pathlib
import re
import time
from collections import Counter
from typing import Optional, Union

import anthropic
import networkx as nx
import yaml

logger = logging.getLogger(__name__)

# ── configuration ────────────────────────────────────────────────────────────
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

MAX_TOKENS           = 8192
TEMPERATURE_EXPLORE  = 0.7   # diversity across exploration rounds
TEMPERATURE_ORIENT   = 0.2   # deterministic for edge orientation
TEMPERATURE_VALIDATE = 0.2

N_EXPLORATION_ROUNDS  = 5
MIN_VOTES_STAGE1      = 2   # an undirected pair must appear in this many Stage-1 rounds to survive
MAX_CYCLE_RETRIES     = 3
N_VALIDATION_ROUNDS   = 2   # Stage 3 correction passes
MFAS_BRUTE_FORCE_MAX  = 20  # max edges per SCC for brute-force MFAS; above this fall back to Eades

# ── API key placeholder ───────────────────────────────────────────────────────
# Set your key in the environment:  export ANTHROPIC_API_KEY=sk-ant-...
# or in your .env file:             ANTHROPIC_API_KEY=sk-ant-...
# The agent reads from os.environ["ANTHROPIC_API_KEY"] by default.


class CausalAgent:
    """
    Three-stage LLM agent for causal DAG discovery over macroeconomic variables.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key.  Defaults to ANTHROPIC_API_KEY env var.
    model : str
        Claude model identifier.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
        temperature_explore: float = TEMPERATURE_EXPLORE,
        temperature_orient: float = TEMPERATURE_ORIENT,
        temperature_validate: float = TEMPERATURE_VALIDATE,
        n_exploration_rounds: int = N_EXPLORATION_ROUNDS,
        min_votes_stage1: int = MIN_VOTES_STAGE1,
        max_cycle_retries: int = MAX_CYCLE_RETRIES,
        n_validation_rounds: int = N_VALIDATION_ROUNDS,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "Anthropic API key required. "
                "Pass api_key= or set the ANTHROPIC_API_KEY environment variable."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model                = model
        self.max_tokens           = int(max_tokens)
        self.temperature_explore  = float(temperature_explore)
        self.temperature_orient   = float(temperature_orient)
        self.temperature_validate = float(temperature_validate)
        self.n_exploration_rounds = int(n_exploration_rounds)
        self.min_votes_stage1     = int(min_votes_stage1)
        self.max_cycle_retries    = int(max_cycle_retries)
        self.n_validation_rounds  = int(n_validation_rounds)
        self._intermediates: dict       = {}   # populated during run()
        self._cycle_history: list[dict] = []   # appended to during _resolve_cycles

    @classmethod
    def from_config(
        cls,
        config: Union[str, pathlib.Path, dict],
        api_key: Optional[str] = None,
    ) -> "CausalAgent":
        """
        Build a CausalAgent from a YAML file path or a config dict.

        Accepts either:
          - the full top-level config (with a top-level "agent" section), or
          - just the inner agent-section dict.

        `api_key` is kept as a separate parameter so secrets stay out of YAML.
        """
        if isinstance(config, (str, pathlib.Path)):
            with open(config) as f:
                config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise TypeError(f"config must be path or dict, got {type(config).__name__}")
        cfg = config.get("agent", config)
        return cls(api_key=api_key, **cfg)

    # ── low-level API call ────────────────────────────────────────────────────

    def _call(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        temperature: float = TEMPERATURE_EXPLORE,
        retries: int = 3,
    ) -> str:
        """Call Claude with exponential-backoff retry on transient errors."""
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        for attempt in range(retries):
            try:
                response = self.client.messages.create(**kwargs)
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = 5 * (2 ** attempt)
                logger.warning("Rate limit hit; waiting %ds …", wait)
                time.sleep(wait)
            except anthropic.APIError as exc:
                if attempt < retries - 1:
                    logger.warning("API error (attempt %d): %s", attempt + 1, exc)
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise RuntimeError("API call failed after all retries.")

    # ── JSON parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> object:
        """
        Extract and parse the first JSON block from an LLM response.
        Handles ```json ... ``` fences and bare [...] / {...}.
        """
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            return json.loads(m.group(1).strip())
        m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if m:
            return json.loads(m.group(1))
        raise ValueError(f"No JSON found in LLM response:\n{text[:600]}")

    # ── prompt helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _format_variables(variable_definitions: dict) -> str:
        """Render variable definitions as a numbered list for prompts."""
        lines = []
        for i, (var, meta) in enumerate(variable_definitions.items(), 1):
            if isinstance(meta, dict):
                name  = meta.get("human_name", var)
                units = meta.get("units", "")
                defn  = meta.get("definition", "")
                lines.append(f"  {i:2d}. {var}  ({name}, {units})\n      {defn}")
            else:
                lines.append(f"  {i:2d}. {var}")
        return "\n".join(lines)

    @staticmethod
    def _edges_to_text(edges: list[dict]) -> str:
        return "\n".join(
            f"  {e['from']} → {e['to']}: {e.get('justification', '')}"
            for e in edges
        )

    @staticmethod
    def _validate_variable_names(edges: list[dict], valid_vars: set[str]) -> list[dict]:
        """Drop any edge whose endpoints are not in the declared variable set."""
        clean = []
        for e in edges:
            if e.get("from") in valid_vars and e.get("to") in valid_vars:
                clean.append(e)
            else:
                logger.warning("Dropping edge with unknown variable: %s", e)
        return clean

    # ── DAG utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_digraph(edges: list[dict]) -> nx.DiGraph:
        G = nx.DiGraph()
        for e in edges:
            G.add_edge(e["from"], e["to"], justification=e.get("justification", ""))
        return G

    # ── minimum feedback arc set (MFAS) helpers ──────────────────────────────

    @staticmethod
    def _mfas_brute_force(sub: nx.DiGraph) -> list[tuple[str, str]]:
        """
        Exact minimum feedback arc set for a small SCC by enumerating edge
        subsets in ascending size: try removing 0, then 1, then 2 ... edges,
        stop at the first size that makes the subgraph acyclic.

        Returns the smallest set of edges (as a list of (u, v) tuples) whose
        removal makes `sub` a DAG.  For an SCC with ≤ 20 edges this completes
        in well under a second.
        """
        from itertools import combinations
        edge_list = list(sub.edges())
        n = len(edge_list)
        for k in range(0, n + 1):
            for subset in combinations(edge_list, k):
                test = sub.copy()
                test.remove_edges_from(subset)
                if nx.is_directed_acyclic_graph(test):
                    return list(subset)
        return edge_list   # unreachable on a finite graph but keeps mypy quiet

    @staticmethod
    def _mfas_eades(sub: nx.DiGraph) -> list[tuple[str, str]]:
        """
        Eades–Lin–Smyth (1993) heuristic for feedback arc set.

        Builds a linear ordering of the vertices by repeatedly peeling sinks
        (appended to the right) and sources (appended to the left), then
        removing the vertex with the largest out_degree − in_degree from
        what remains.  Edges that point from a later vertex back to an
        earlier vertex in this ordering are the feedback arc set.

        Worst-case 3/8 approximation; linear-time; used here only when the
        SCC is too large for brute force (> MFAS_BRUTE_FORCE_MAX edges).
        """
        g = sub.copy()
        s1: list[str] = []
        s2: list[str] = []

        while g.number_of_nodes() > 0:
            changed = True
            while changed:
                changed = False
                for n in [n for n in g.nodes() if g.out_degree(n) == 0]:
                    s2.append(n)
                    g.remove_node(n)
                    changed = True
            changed = True
            while changed:
                changed = False
                for n in [n for n in g.nodes() if g.in_degree(n) == 0]:
                    s1.append(n)
                    g.remove_node(n)
                    changed = True
            if g.number_of_nodes() > 0:
                pick = max(g.nodes(), key=lambda n: g.out_degree(n) - g.in_degree(n))
                s1.append(pick)
                g.remove_node(pick)

        order = s1 + list(reversed(s2))
        pos   = {n: i for i, n in enumerate(order)}
        # back-edges (u → v where pos[u] >= pos[v]) constitute the FAS
        return [(u, v) for u, v in sub.edges() if pos[u] >= pos[v]]

    def _minimum_feedback_arc_set(self, G: nx.DiGraph) -> list[tuple[str, str]]:
        """
        Find a minimum (or near-minimum) set of edges whose removal makes G
        a DAG.  Decomposes G into strongly-connected components first: any
        edge not inside an SCC of size > 1 is *not* on any cycle and is
        kept automatically.  Within each non-trivial SCC, uses brute-force
        MFAS for ≤ MFAS_BRUTE_FORCE_MAX edges, otherwise Eades heuristic.

        Returns a list of (u, v) edges to remove.
        """
        removed: list[tuple[str, str]] = []
        for scc_nodes in nx.strongly_connected_components(G):
            if len(scc_nodes) <= 1:
                continue
            sub = G.subgraph(scc_nodes).copy()
            n_edges = sub.number_of_edges()
            if n_edges <= MFAS_BRUTE_FORCE_MAX:
                fas = self._mfas_brute_force(sub)
            else:
                logger.warning(
                    "    SCC has %d edges (> %d); using Eades heuristic instead of brute force",
                    n_edges, MFAS_BRUTE_FORCE_MAX,
                )
                fas = self._mfas_eades(sub)
            removed.extend(fas)
        return removed

    def _resolve_cycles(
        self,
        edges: list[dict],
        var_text: str,
        regime_desc: str,
    ) -> list[dict]:
        """
        Iteratively ask the LLM to break cycles; fall back to greedy removal
        if the LLM cannot converge within MAX_CYCLE_RETRIES attempts.
        """
        G = self._build_digraph(edges)

        for attempt in range(self.max_cycle_retries):
            if nx.is_directed_acyclic_graph(G):
                return edges

            cycles = list(nx.simple_cycles(G))
            self._cycle_history.append({
                "edges_before": [dict(e) for e in edges],
                "cycles":       [list(c) for c in cycles[:5]],
            })
            cycle_strs = "\n".join(
                " → ".join(c + [c[0]]) for c in cycles[:5]
            )
            prompt = f"""The proposed DAG contains directed cycles, violating acyclicity.

Cycles detected:
{cycle_strs}

Current edges:
{self._edges_to_text(edges)}

Variables:
{var_text}

Regime: {regime_desc}

Remove or reverse the minimum set of edges to eliminate all cycles while \
preserving as much causal structure as possible.

Return the FULL corrected edge list as JSON:
[
  {{"from": "var_a", "to": "var_b", "justification": "..."}},
  ...
]"""
            logger.warning("  Cycle detected (attempt %d); asking LLM to resolve …", attempt + 1)
            text  = self._call(
                [{"role": "user", "content": prompt}],
                temperature=self.temperature_orient,
            )
            try:
                edges = self._extract_json(text)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("  Cycle-resolution parse error: %s", exc)
                break
            G = self._build_digraph(edges)

        if not nx.is_directed_acyclic_graph(G):
            logger.warning(
                "LLM could not resolve cycles; using minimum feedback arc set."
            )
            fas = self._minimum_feedback_arc_set(G)
            removed_records = [
                {
                    "from": u,
                    "to":   v,
                    "justification": G.edges[u, v].get("justification", ""),
                }
                for u, v in fas
            ]
            for rec in removed_records:
                logger.warning(
                    "    MFAS remove: %s → %s  (%s)",
                    rec["from"], rec["to"],
                    (rec["justification"][:60] + "…")
                    if len(rec["justification"]) > 60 else rec["justification"],
                )
            self._cycle_history.append({
                "fallback":      "minimum_feedback_arc_set",
                "n_removed":     len(removed_records),
                "edges_removed": removed_records,
            })

            fas_set = set(fas)
            edges = [
                {
                    "from": u,
                    "to":   v,
                    "justification": G.edges[u, v].get("justification", ""),
                }
                for u, v in G.edges()
                if (u, v) not in fas_set
            ]

        return edges

    # ── Stage 1: Causal Exploration ───────────────────────────────────────────

    def _stage1_explore(
        self,
        var_text: str,
        regime_desc: str,
        valid_vars: set[str],
        prev_dag: Optional[list[dict]],
        broken_edges: Optional[list[tuple[str, str]]],
    ) -> tuple[set[frozenset], list[list[dict]], Counter]:
        """
        Prompt the LLM N_EXPLORATION_ROUNDS times to propose plausible causal
        edges.  Aggregate by *voting*: an undirected pair survives only if it
        appears in at least MIN_VOTES_STAGE1 rounds.  Same-round duplicates
        count once.

        Returns
        -------
        candidates : set of frozenset
            Undirected pairs that cleared the vote threshold.
        per_round : list[list[dict]]
            Raw edge proposals from each round (for inspection).
        votes : Counter
            {frozenset({a, b}): n_rounds_proposed} for every pair seen at
            least once across all rounds (pre-threshold).
        """
        prev_context = ""
        if prev_dag:
            prev_context = (
                "\n\n**Previous regime DAG** (for reference — structure may have shifted):\n"
                + self._edges_to_text(prev_dag)
            )
        broken_context = ""
        if broken_edges:
            broken_strs = "\n".join(f"  {a} — {b}" for a, b in broken_edges)
            broken_context = (
                "\n\n**Edges that broke down in this new regime** "
                "(high Jaccard distance — reconsider whether these still hold):\n"
                + broken_strs
            )

        system = (
            "You are an expert macroeconomist and causal inference specialist. "
            "Think carefully about direct causal mechanisms, not correlations."
        )
        user_prompt = f"""You are analysing causal relationships among macroeconomic and \
financial variables during a specific market regime.

**Causal relationship definition**: A causes B means a change in A directly leads \
to a change in B, holding all other listed variables constant (ceteris paribus). \
Propose only direct causal links, not indirect ones mediated by other variables \
in this list.

**Variables**:
{var_text}

**Regime context**: {regime_desc}{prev_context}{broken_context}

**Task**: Propose ALL plausible direct causal edges between the variables above. \
Think carefully about which economic mechanisms are most active in this specific regime.

Return a JSON list — use the exact variable names from the list above:
[
  {{"from": "var_a", "to": "var_b", "justification": "one-sentence economic mechanism"}},
  ...
]"""

        votes: Counter             = Counter()
        per_round: list[list[dict]] = []
        logger.info(
            "Stage 1: %d exploration rounds (vote threshold ≥ %d) …",
            self.n_exploration_rounds, self.min_votes_stage1,
        )

        for i in range(self.n_exploration_rounds):
            logger.info("  Round %d/%d", i + 1, self.n_exploration_rounds)
            text = self._call(
                [{"role": "user", "content": user_prompt}],
                system=system,
                temperature=self.temperature_explore,
            )
            try:
                proposed = self._extract_json(text)
                proposed = self._validate_variable_names(proposed, valid_vars)
                per_round.append(proposed)
                # de-duplicate within a round so each pair contributes at most
                # one vote per round, regardless of how many times the LLM
                # proposed it (or proposed both directions)
                round_pairs = {
                    frozenset({e["from"], e["to"]})
                    for e in proposed
                    if e.get("from") != e.get("to")
                }
                votes.update(round_pairs)
                logger.info(
                    "    → %d edges proposed  |  %d unique pairs ever seen",
                    len(proposed), len(votes),
                )
            except (ValueError, json.JSONDecodeError, KeyError) as exc:
                logger.warning("  Round %d parse error: %s", i + 1, exc)
                per_round.append([])

        candidates = {pair for pair, n in votes.items() if n >= self.min_votes_stage1}
        n_dropped  = len(votes) - len(candidates)
        logger.info(
            "Stage 1 voting: %d unique pairs → %d survive (≥ %d votes); %d single-round pairs dropped",
            len(votes), len(candidates), self.min_votes_stage1, n_dropped,
        )
        return candidates, per_round, votes

    # ── Stage 2: Causal Inference (orientation) ───────────────────────────────

    def _stage2_orient(
        self,
        undirected_edges: set[frozenset],
        var_text: str,
        regime_desc: str,
        valid_vars: set[str],
    ) -> list[dict]:
        """
        Give the LLM the undirected candidate set and ask it to orient each
        edge.  Validate acyclicity and repair if needed.
        """
        edge_list_str = "\n".join(
            f"  {{ {', '.join(sorted(e))} }}"
            for e in sorted(undirected_edges, key=lambda x: sorted(x)[0])
        )

        prompt = f"""You have identified the following pairs of macroeconomic variables \
as potentially causally related:

**Candidate undirected edges** ({len(undirected_edges)} pairs):
{edge_list_str}

**Variables**:
{var_text}

**Regime context**: {regime_desc}

**Task**: For each candidate edge {{A, B}}, choose one of:
  • A → B  (A directly causes B)
  • B → A  (B directly causes A)
  • drop it  (not causally linked in this regime)

The result MUST be a DAG (no directed cycles). Justify each direction \
with the economic mechanism active in this regime.

Return a JSON list using exact variable names:
[
  {{"from": "var_a", "to": "var_b", "justification": "economic reason for this direction"}},
  ...
]"""

        logger.info("Stage 2: orienting %d undirected edges …", len(undirected_edges))
        text  = self._call(
            [{"role": "user", "content": prompt}],
            temperature=self.temperature_orient,
        )
        edges = self._extract_json(text)
        edges = self._validate_variable_names(edges, valid_vars)
        self._intermediates["stage2_raw"]    = [dict(e) for e in edges]
        self._cycle_history                  = []
        edges = self._resolve_cycles(edges, var_text, regime_desc)
        self._intermediates["stage2_cycles"] = list(self._cycle_history)

        logger.info(
            "Stage 2 complete: %d directed edges  |  is_dag=%s",
            len(edges),
            nx.is_directed_acyclic_graph(self._build_digraph(edges)),
        )
        return edges

    # ── Stage 3 helpers: corrections + conflict detection ────────────────────

    @staticmethod
    def _edge_key(e: dict) -> Optional[tuple[str, str]]:
        """Return (from, to) tuple for a correction-edge dict, or None if malformed."""
        f, t = e.get("from"), e.get("to")
        if f is None or t is None:
            return None
        return (f, t)

    def _apply_stage3_corrections(
        self,
        edges: list[dict],
        corrections: dict,
    ) -> tuple[list[dict], list[dict]]:
        """
        Apply add / remove / reverse corrections with conflict detection.

        Conflicts surfaced (logged + recorded, not silently absorbed):
          - add_and_remove        : same edge listed in both add and remove
          - add_and_reverse       : same edge in both add and reverse-target
          - remove_and_reverse    : same edge in both remove and reverse-target
          - reverse_nonexistent   : reverse of an edge whose source direction
                                     doesn't exist in the current DAG
          - reverse_already_in_target_direction : reverse no-op (target already
                                     in the requested direction)

        Returns
        -------
        new_edges : list[dict]   — edges after applying non-conflicting ops
        conflicts : list[dict]   — one entry per conflict for logging / audit
        """
        add_list     = corrections.get("add", [])     or []
        remove_list  = corrections.get("remove", [])  or []
        reverse_list = corrections.get("reverse", []) or []

        add_keys     = {k for k in (self._edge_key(e) for e in add_list)     if k is not None}
        remove_keys  = {k for k in (self._edge_key(e) for e in remove_list)  if k is not None}
        reverse_keys = {k for k in (self._edge_key(e) for e in reverse_list) if k is not None}

        conflicts: list[dict] = []
        skip_add: set      = set()
        skip_remove: set   = set()
        skip_reverse: set  = set()

        for k in sorted(add_keys & remove_keys):
            conflicts.append({
                "type": "add_and_remove",
                "edge": list(k),
                "resolution": "skip both — contradictory",
            })
            skip_add.add(k)
            skip_remove.add(k)
            logger.warning(
                "    CONFLICT: %s → %s in both add and remove — skipping both",
                *k,
            )

        for k in sorted(add_keys & reverse_keys):
            conflicts.append({
                "type": "add_and_reverse",
                "edge": list(k),
                "resolution": "drop add — reverse already produces it",
            })
            skip_add.add(k)
            logger.warning(
                "    CONFLICT: %s → %s in both add and reverse-target — dropping add",
                *k,
            )

        for k in sorted(remove_keys & reverse_keys):
            conflicts.append({
                "type": "remove_and_reverse",
                "edge": list(k),
                "resolution": "skip both — contradictory",
            })
            skip_remove.add(k)
            skip_reverse.add(k)
            logger.warning(
                "    CONFLICT: %s → %s in both remove and reverse-target — skipping both",
                *k,
            )

        # Apply with explicit handling of mid-application discrepancies.
        index        = {(e["from"], e["to"]): e for e in edges}
        current_keys = set(index.keys())

        for e in remove_list:
            key = self._edge_key(e)
            if key is None or key in skip_remove:
                continue
            if key in index:
                del index[key]
                logger.info("    Removed  : %s → %s", *key)

        for e in reverse_list:
            new_key = self._edge_key(e)
            if new_key is None or new_key in skip_reverse:
                continue
            old_key = (new_key[1], new_key[0])
            if old_key in index:
                del index[old_key]
                index[new_key] = e
                logger.info("    Reversed : %s → %s", *new_key)
            elif new_key in current_keys:
                conflicts.append({
                    "type": "reverse_already_in_target_direction",
                    "edge": list(new_key),
                    "resolution": "no-op",
                })
                logger.warning(
                    "    Reverse no-op: edge already in target direction %s → %s",
                    *new_key,
                )
            else:
                conflicts.append({
                    "type":         "reverse_nonexistent",
                    "edge":         list(new_key),
                    "missing_edge": [old_key[0], old_key[1]],
                    "resolution":   "treated as add",
                })
                logger.warning(
                    "    Reverse of non-existent edge — neither %s→%s nor %s→%s "
                    "exists; treating as add of %s → %s",
                    old_key[0], old_key[1], new_key[0], new_key[1], *new_key,
                )
                index[new_key] = e

        for e in add_list:
            key = self._edge_key(e)
            if key is None or key in skip_add:
                continue
            if key not in index:
                index[key] = e
                logger.info("    Added    : %s → %s", *key)

        return list(index.values()), conflicts

    # ── Stage 3: Causal Validation ────────────────────────────────────────────

    def _stage3_validate(
        self,
        edges: list[dict],
        var_text: str,
        regime_desc: str,
        valid_vars: set[str],
    ) -> list[dict]:
        """
        Ask the LLM to review the full DAG for errors; apply corrections;
        repeat N_VALIDATION_ROUNDS times.
        """
        logger.info("Stage 3: %d validation rounds …", self.n_validation_rounds)
        s3_rounds: list[dict] = []

        for round_n in range(self.n_validation_rounds):
            dag_text = self._edges_to_text(edges)

            prompt = f"""Critically review the following causal DAG for macroeconomic variables.

**Current DAG** ({len(edges)} edges):
{dag_text}

**Variables**:
{var_text}

**Regime context**: {regime_desc}

**Task**: Identify errors in this DAG:
  1. **Missing edges** — important direct causal links that are absent
  2. **False positives** — edges that should not be there (spurious or indirect)
  3. **Wrong directions** — edges pointing the wrong way

Return corrections as JSON (use empty lists if no changes needed for a category):
{{
  "add":     [{{"from": "var_a", "to": "var_b", "justification": "why this link is missing"}}],
  "remove":  [{{"from": "var_a", "to": "var_b", "justification": "why spurious"}}],
  "reverse": [{{"from": "var_a", "to": "var_b", "justification": "corrected direction"}}]
}}

For "reverse", specify the NEW (corrected) direction as from/to.
Be conservative — only make corrections you are confident about given the regime context."""

            logger.info("  Validation round %d/%d", round_n + 1, self.n_validation_rounds)
            text = self._call(
                [{"role": "user", "content": prompt}],
                temperature=self.temperature_validate,
            )
            edges_before = [dict(e) for e in edges]
            try:
                corrections = self._extract_json(text)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("  Parse error in validation round %d: %s; skipping.", round_n + 1, exc)
                continue

            edges, conflicts = self._apply_stage3_corrections(edges, corrections)
            if conflicts:
                logger.warning(
                    "    Stage 3 round %d had %d correction conflict(s)",
                    round_n + 1, len(conflicts),
                )

            edges = self._validate_variable_names(edges, valid_vars)
            self._cycle_history = []
            edges = self._resolve_cycles(edges, var_text, regime_desc)
            s3_rounds.append({
                "edges_before": edges_before,
                "corrections":  corrections,
                "conflicts":    conflicts,
                "edges_after":  [dict(e) for e in edges],
                "cycles":       list(self._cycle_history),
            })

        self._intermediates["stage3_rounds"] = s3_rounds
        logger.info("Stage 3 complete: %d edges", len(edges))
        return edges

    # ── public API ────────────────────────────────────────────────────────────

    def run(
        self,
        variable_definitions: dict,
        regime_description: str,
        prev_dag: Optional[list[dict]] = None,
        broken_edges: Optional[list[tuple[str, str]]] = None,
    ) -> dict:
        """
        Run the full three-stage causal DAG discovery pipeline.

        Parameters
        ----------
        variable_definitions : dict
            Variable metadata as loaded from variable_definitions.yaml.
            Keys are variable names; values are dicts with human_name,
            source, units, definition.
        regime_description : str
            Natural language description of the current macro regime
            (e.g. "COVID onset: extreme volatility, flight to safety, …").
        prev_dag : list of edge dicts, optional
            Directed edges from the previous regime's DAG.
            Each dict: {"from": str, "to": str, "justification": str}.
        broken_edges : list of (str, str), optional
            Variable pairs whose causal link broke down at this regime
            boundary (high Jaccard distance).

        Returns
        -------
        dict
            edges            : list of {"from", "to", "justification"}
            graph            : networkx.DiGraph
            n_edges          : int
            is_dag           : bool
            undirected_edges : set of frozensets (Stage 1 output)
        """
        valid_vars = set(variable_definitions.keys())
        var_text   = self._format_variables(variable_definitions)

        logger.info("=== CausalAgent pipeline start ===")
        logger.info("Variables  : %d", len(valid_vars))
        logger.info("Regime     : %s", regime_description)
        if prev_dag:
            logger.info("Prev DAG   : %d edges", len(prev_dag))
        if broken_edges:
            logger.info("Broken edges: %d", len(broken_edges))

        # ── Stage 1 ────────────────────────────────────────────────────────
        self._intermediates = {}
        self._cycle_history = []
        undirected, per_round, votes = self._stage1_explore(
            var_text, regime_description, valid_vars, prev_dag, broken_edges
        )
        self._intermediates["stage1_per_round"]   = per_round
        self._intermediates["stage1_undirected"]  = [sorted(e) for e in undirected]
        self._intermediates["stage1_votes"]       = [
            {"pair": sorted(pair), "votes": n}
            for pair, n in sorted(votes.items(), key=lambda kv: (-kv[1], sorted(kv[0])))
        ]
        self._intermediates["stage1_min_votes"]   = self.min_votes_stage1
        logger.info("Stage 1 → %d undirected candidate edges", len(undirected))

        # ── Stage 2 ────────────────────────────────────────────────────────
        edges = self._stage2_orient(undirected, var_text, regime_description, valid_vars)

        # ── Stage 3 ────────────────────────────────────────────────────────
        edges = self._stage3_validate(edges, var_text, regime_description, valid_vars)

        G = self._build_digraph(edges)
        is_dag = nx.is_directed_acyclic_graph(G)
        logger.info("=== Pipeline complete: %d edges, is_dag=%s ===", len(edges), is_dag)

        return {
            "edges":            edges,
            "graph":            G,
            "n_edges":          len(edges),
            "is_dag":           is_dag,
            "undirected_edges": undirected,
            "intermediates":    self._intermediates,
        }
