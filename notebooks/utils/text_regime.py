"""Layer 1 — Text-Driven Regime Detection.

Pipeline:
    FOMC corpus  →  LLM proposes candidate changepoints
                 →  Likelihood-ratio test on time-series data
                 →  Validated regime boundaries

Data flow:
    docs: List[FOMCDoc]                       # input
        ↓
    candidates: List[CandidateChangepoint]    # LLM stage
        ↓
    validated: List[ValidatedChangepoint]     # statistical stage
        ↓
    rejected: List[(CandidateChangepoint, reason)]

The LLM is NEVER asked about edges, directions, or per-regime DAG structure.
It only proposes changepoint dates from text comparison.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from numpy.linalg import lstsq, slogdet
from scipy.stats import chi2

logger = logging.getLogger(__name__)


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class FOMCDoc:
    """One FOMC minutes / statement document."""
    date: pd.Timestamp
    text: str
    title: str = ""

    def __post_init__(self):
        if not isinstance(self.date, pd.Timestamp):
            self.date = pd.Timestamp(self.date)

    @property
    def text_hash(self) -> str:
        return hashlib.md5(self.text.encode("utf-8")).hexdigest()[:12]


@dataclass
class CandidateChangepoint:
    """LLM-proposed regime boundary, before statistical validation."""
    date: pd.Timestamp
    confidence: float          # LLM-self-reported in [0, 1]
    reasoning: str             # 1-3 sentence justification quoting prior phrasing changes
    triggered_doc: str         # title or date of the doc that triggered the proposal

    def to_dict(self):
        d = asdict(self)
        d["date"] = self.date.strftime("%Y-%m-%d")
        return d


@dataclass
class ValidatedChangepoint:
    """LLM candidate that passed the statistical likelihood-ratio test."""
    date: pd.Timestamp
    llm_reasoning: str
    llm_confidence: float
    lr_statistic: float
    p_value: float
    pre_window: Tuple[pd.Timestamp, pd.Timestamp]
    post_window: Tuple[pd.Timestamp, pd.Timestamp]

    def to_dict(self):
        d = {
            "date": self.date.strftime("%Y-%m-%d"),
            "llm_reasoning": self.llm_reasoning,
            "llm_confidence": self.llm_confidence,
            "lr_statistic": self.lr_statistic,
            "p_value": self.p_value,
            "pre_window": [t.strftime("%Y-%m-%d") for t in self.pre_window],
            "post_window": [t.strftime("%Y-%m-%d") for t in self.post_window],
        }
        return d


@dataclass
class DetectionResult:
    """Full pipeline output: kept candidates, rejected ones, plus diagnostics."""
    candidates:  List[CandidateChangepoint]
    validated:   List[ValidatedChangepoint]
    rejected:    List[Tuple[CandidateChangepoint, str]]      # (cand, reason)

    @property
    def n_validated(self) -> int:
        return len(self.validated)

    @property
    def validated_dates(self) -> List[pd.Timestamp]:
        return [v.date for v in self.validated]

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "candidates":  [c.to_dict() for c in self.candidates],
                "validated":   [v.to_dict() for v in self.validated],
                "rejected":    [{"cand": c.to_dict(), "reason": r} for c, r in self.rejected],
            }, f, indent=2)
        logger.info("Saved detection result → %s", path)


# ── likelihood-ratio test ────────────────────────────────────────────────────

def _var1_loglik(X_seg: np.ndarray) -> float:
    """Gaussian log-likelihood of a lag-1 VAR fit by OLS on X_seg.

    Model: X[t] = c + Phi @ X[t-1] + eps, eps ~ N(0, Sigma).
    Returns log L(X | OLS estimates).
    """
    n_total, N = X_seg.shape
    if n_total < max(N + 5, 15):
        return -np.inf
    Y    = X_seg[1:]
    Xlag = X_seg[:-1]
    X_with_const = np.column_stack([np.ones(len(Xlag)), Xlag])
    Phi, *_ = lstsq(X_with_const, Y, rcond=None)
    residuals = Y - X_with_const @ Phi
    n = residuals.shape[0]
    Sigma = (residuals.T @ residuals) / n + 1e-9 * np.eye(N)
    sign, log_det = slogdet(Sigma)
    if sign <= 0:
        return -np.inf
    return -n / 2.0 * (N * np.log(2 * np.pi) + log_det + N)


def likelihood_ratio_test(
    X: pd.DataFrame,
    cp_date: pd.Timestamp,
    window_days: int = 90,
    min_obs: int = 30,
) -> Tuple[float, float, Tuple, Tuple]:
    """Test whether the lag-1 VAR parameters differ before vs after `cp_date`.

    H0: same (c, Phi, Sigma) on both sides   (restricted)
    H1: separate (c, Phi, Sigma) on each side (unrestricted, the implementation
        fits per-segment Sigma)

    LR = -2 * (loglik_full - loglik_pre - loglik_post)  ~  chi^2(df)

    Unrestricted has, per segment, N intercept entries + N^2 autoregression
    entries + N*(N+1)/2 covariance entries; restricted has one copy. Hence the
    extra parameters in unrestricted relative to restricted are
        df = N + N^2 + N*(N+1)/2.

    Returns
    -------
    lr_statistic : float
    p_value      : float (under H0)
    pre_window   : (start, end) timestamps
    post_window  : (start, end) timestamps
    """
    if not isinstance(cp_date, pd.Timestamp):
        cp_date = pd.Timestamp(cp_date)

    # Find nearest available date
    if cp_date not in X.index:
        loc = X.index.get_indexer([cp_date], method="nearest")[0]
        cp_date = X.index[loc]

    pre_start  = cp_date - pd.Timedelta(days=window_days)
    post_end   = cp_date + pd.Timedelta(days=window_days)
    # pre_seg keeps cp_date as its last observation; post_seg starts strictly
    # after cp_date so that the partition pre/post is disjoint.
    pre_seg    = X.loc[pre_start:cp_date].values
    post_seg   = X.loc[cp_date:post_end].iloc[1:].values

    if len(pre_seg) < min_obs or len(post_seg) < min_obs:
        return np.nan, np.nan, (pre_start, cp_date), (cp_date, post_end)

    full_seg = np.vstack([pre_seg, post_seg])
    L_pre  = _var1_loglik(pre_seg)
    L_post = _var1_loglik(post_seg)
    L_full = _var1_loglik(full_seg)

    if not all(np.isfinite([L_pre, L_post, L_full])):
        return np.nan, np.nan, (pre_start, cp_date), (cp_date, post_end)

    lr = -2.0 * (L_full - (L_pre + L_post))
    N = pre_seg.shape[1]
    df = N + N * N + N * (N + 1) // 2
    p = float(chi2.sf(max(lr, 0.0), df))
    return float(lr), p, (pre_start, cp_date), (cp_date, post_end)


# ── LLM prompt ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an expert macro-financial economist. You analyze Federal Reserve "
    "communications to identify substantive shifts in monetary policy stance. "
    "You distinguish material regime change from incremental adjustments."
)

_USER_PROMPT_TEMPLATE = """You are identifying MAJOR REGIME SHIFTS in monetary policy, not incremental
adjustments within an existing regime.

PRIOR DOCUMENT (date: {prev_date}):
\"\"\"{prev_text}\"\"\"

CURRENT DOCUMENT (date: {curr_date}):
\"\"\"{curr_text}\"\"\"

DEFINITIONS

A REGIME is a multi-meeting monetary stance characterized by:
- A stable reaction function (which macro variables drive decisions)
- A stable dominant policy tool (rate cuts vs hikes vs QE vs QT vs forward guidance)
- A stable trajectory (easing / pausing / tightening / accommodation)

A "MAJOR_PIVOT" is a TRANSITION between such stances. Examples:
- ✅ Pivot from accommodation to tightening (e.g., 2021-11 taper start, 2022-03 first hike)
- ✅ Pivot from tightening to easing (e.g., 2018-12 → 2019-07 first cut, 2024-09 first cut)
- ✅ Crisis emergency response (e.g., 2020-03 COVID cuts and QE infinity)
- ✅ ZLB onset or exit
- ✅ Introduction or termination of a QE/QT program (e.g., 2012-09 QE3, 2014-10 taper end)
- ✅ "Transitory inflation" → "Persistent inflation" reframing

An "INCREMENTAL" change is a continuation along an established trajectory:
- ❌ Hike size step (50 bp → 75 bp within an ongoing hike cycle)
- ❌ Routine SEP / dot-plot updates without strategy change
- ❌ Acknowledging incoming data within the established stance
- ❌ Forward guidance language polish that does not change the underlying stance
- ❌ Refining technical balance-sheet runoff caps within an already-announced QT program

"NO_CHANGE" means the two meetings are operationally the same regime with no
material trajectory or tool shift.

CALIBRATION PRIOR
The Federal Reserve typically experiences only 0-1 MAJOR_PIVOTs per year (rarely 2).
Most consecutive minutes pairs (about 6 of 8 per year) are NO_CHANGE.
Be strict — when in doubt between MAJOR_PIVOT and INCREMENTAL, choose INCREMENTAL.

OUTPUT JSON only, no preamble:
{{
  "regime_change_type": "major_pivot" | "incremental" | "no_change",
  "confidence": 0.0-1.0,
  "reasoning": "1-3 sentences citing exact phrasing changes that justify the classification",
  "key_quote": "verbatim phrase from the current document that signals the shift, or empty string"
}}"""


def _truncate(text: str, max_chars: int = 4000) -> str:
    """Shorten text to fit prompt budget while preserving start + end."""
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2):]
    return f"{head}\n... [truncated] ...\n{tail}"


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from LLM response."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}...")


# ── detector ────────────────────────────────────────────────────────────────

class TextRegimeDetector:
    """Two-stage regime detector: LLM proposal + statistical validation.

    Parameters
    ----------
    llm_call : callable, optional
        Function (prev_doc, curr_doc) -> dict. If None, instantiates an
        Anthropic client using ANTHROPIC_API_KEY env var.
    model : str
        Anthropic model identifier when using the default client.
    cache_dir : str or Path, optional
        Directory for caching LLM responses by (prev_hash, curr_hash). If
        None, no caching.
    confidence_threshold : float
        Minimum LLM confidence for an edge to be retained at the candidate stage.
    p_value_threshold : float
        Maximum p-value for an edge to pass statistical validation.
    """

    def __init__(
        self,
        llm_call=None,
        model: str = "claude-opus-4-7",
        cache_dir: Optional[str] = None,
        confidence_threshold: float = 0.5,
        p_value_threshold: float = 0.05,
        max_tokens: int = 512,
        temperature: float = 0.2,
        max_excerpt_chars: int = 4000,
    ):
        self.llm_call = llm_call or self._default_llm_call
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.confidence_threshold = confidence_threshold
        self.p_value_threshold = p_value_threshold
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_excerpt_chars = max_excerpt_chars

        self._client = None  # lazy init for default client

    # ── default Anthropic client ─────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. "
                    "Either pass llm_call= explicitly or set the env var."
                )
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    # Bump this when the prompt schema changes — old cache entries become stale
    PROMPT_VERSION = "v2_major_pivot"

    def _default_llm_call(self, prev_doc: FOMCDoc, curr_doc: FOMCDoc) -> dict:
        """Default LLM call using Anthropic client. Cached by content hash + prompt version."""
        cache_key = f"{self.PROMPT_VERSION}_{prev_doc.text_hash}_{curr_doc.text_hash}"
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{cache_key}.json"
            if cache_path.exists():
                with open(cache_path) as f:
                    payload = json.load(f)
                return self._normalize_payload(payload)

        prompt = _USER_PROMPT_TEMPLATE.format(
            prev_date=prev_doc.date.strftime("%Y-%m-%d"),
            curr_date=curr_doc.date.strftime("%Y-%m-%d"),
            prev_text=_truncate(prev_doc.text, self.max_excerpt_chars),
            curr_text=_truncate(curr_doc.text, self.max_excerpt_chars),
        )

        client = self._get_client()
        # Newer models (e.g., claude-opus-4-7) deprecate temperature; older ones accept it.
        # We probe once and adapt for the rest of the session.
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        for attempt in range(3):
            try:
                resp = client.messages.create(**kwargs)
                payload = _extract_json(resp.content[0].text)
                if self.cache_dir is not None:
                    with open(cache_path, "w") as f:
                        json.dump(payload, f, indent=2)
                return self._normalize_payload(payload)
            except Exception as exc:
                msg = str(exc)
                # Adapt to deprecated-temperature errors (Claude opus-4-7 etc.)
                if "temperature" in msg.lower() and "deprecated" in msg.lower() and "temperature" in kwargs:
                    logger.info("Model rejects temperature param; dropping and retrying.")
                    kwargs.pop("temperature", None)
                    self.temperature = None
                    continue
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                logger.warning("LLM call failed for %s vs %s: %s",
                               prev_doc.date.date(), curr_doc.date.date(), exc)
                return {"is_changepoint": False, "confidence": 0.0,
                        "reasoning": f"(LLM error: {exc})", "key_quote": ""}

    @staticmethod
    def _normalize_payload(payload: dict) -> dict:
        """Map the v2 schema (regime_change_type) onto the legacy boolean is_changepoint.

        Only `major_pivot` is treated as a candidate changepoint; `incremental`
        and `no_change` are not.  We keep `regime_change_type` in the payload
        for downstream inspection.
        """
        rct = payload.get("regime_change_type")
        if rct is not None:
            payload["is_changepoint"] = (rct == "major_pivot")
        return payload

    # ── stage 1: LLM proposal ────────────────────────────────────────────

    def propose_changepoints(
        self, docs: List[FOMCDoc], verbose: bool = False,
    ) -> List[CandidateChangepoint]:
        """Slide through consecutive doc pairs; LLM judges each."""
        docs = sorted(docs, key=lambda d: d.date)
        candidates: List[CandidateChangepoint] = []
        for i in range(1, len(docs)):
            prev_d, curr_d = docs[i - 1], docs[i]
            payload = self.llm_call(prev_d, curr_d)
            is_cp = bool(payload.get("is_changepoint", False))
            conf = float(payload.get("confidence", 0.0))
            if verbose:
                marker = "✓" if is_cp else " "
                logger.info("  %s %s → %s  conf=%.2f  %s",
                            marker, prev_d.date.date(), curr_d.date.date(),
                            conf, payload.get("reasoning", "")[:60])
            if not is_cp or conf < self.confidence_threshold:
                continue
            candidates.append(CandidateChangepoint(
                date=curr_d.date,
                confidence=conf,
                reasoning=str(payload.get("reasoning", "")),
                triggered_doc=curr_d.title or curr_d.date.strftime("%Y-%m-%d"),
            ))
        return candidates

    # ── stage 2: statistical validation ──────────────────────────────────

    def validate_with_data(
        self,
        candidates: List[CandidateChangepoint],
        time_series: pd.DataFrame,
        window_days: int = 90,
    ) -> Tuple[List[ValidatedChangepoint], List[Tuple[CandidateChangepoint, str]]]:
        """Run LR test on each candidate; partition into validated / rejected."""
        validated: List[ValidatedChangepoint] = []
        rejected:  List[Tuple[CandidateChangepoint, str]] = []
        for cand in candidates:
            lr, p, pre_w, post_w = likelihood_ratio_test(
                time_series, cand.date, window_days=window_days,
            )
            if np.isnan(p):
                rejected.append((cand, "insufficient data around changepoint"))
                continue
            if p > self.p_value_threshold:
                rejected.append((cand, f"LR p={p:.3f} > {self.p_value_threshold}"))
                continue
            validated.append(ValidatedChangepoint(
                date=cand.date,
                llm_reasoning=cand.reasoning,
                llm_confidence=cand.confidence,
                lr_statistic=lr,
                p_value=p,
                pre_window=pre_w,
                post_window=post_w,
            ))
        return validated, rejected

    # ── full pipeline ────────────────────────────────────────────────────

    def detect(
        self,
        docs: List[FOMCDoc],
        time_series: pd.DataFrame,
        window_days: int = 90,
        verbose: bool = False,
    ) -> DetectionResult:
        """Run full Layer 1 pipeline: LLM propose → LR validate."""
        if verbose:
            logger.info("Stage 1: LLM proposing changepoints over %d docs ...", len(docs))
        candidates = self.propose_changepoints(docs, verbose=verbose)
        if verbose:
            logger.info("Stage 1 → %d candidates", len(candidates))
            logger.info("Stage 2: LR-validating against time series (window=%d days) ...", window_days)
        validated, rejected = self.validate_with_data(
            candidates, time_series, window_days=window_days,
        )
        if verbose:
            logger.info("Stage 2 → %d validated, %d rejected", len(validated), len(rejected))
        return DetectionResult(
            candidates=candidates, validated=validated, rejected=rejected,
        )


# ── evaluation: compare detected vs anchor events ────────────────────────────

def match_anchors(
    detected_dates: List[pd.Timestamp],
    anchor_events: List[Tuple[pd.Timestamp, str]],
    tolerance_days: int = 30,
) -> Tuple[List[Dict], List[Tuple[pd.Timestamp, str]], List[pd.Timestamp]]:
    """Match detected changepoints against ground-truth anchor events.

    Returns
    -------
    hits : list of dict
        {anchor_date, anchor_label, matched_detected_date, offset_days}
        offset_days = matched - anchor (negative = detected ahead, positive = lagging)
    misses : list of (anchor_date, anchor_label)
        Anchor events with no matching detection within tolerance.
    false_alarms : list of pd.Timestamp
        Detected dates not matching any anchor.
    """
    detected = sorted([pd.Timestamp(d) for d in detected_dates])
    anchors  = sorted([(pd.Timestamp(a), lbl) for a, lbl in anchor_events])
    used_detected = set()
    hits: List[Dict] = []
    misses: List[Tuple[pd.Timestamp, str]] = []

    for a_date, a_label in anchors:
        best, best_offset = None, None
        for d in detected:
            if d in used_detected:
                continue
            offset = (d - a_date).days
            if abs(offset) <= tolerance_days:
                if best is None or abs(offset) < abs(best_offset):
                    best, best_offset = d, offset
        if best is not None:
            hits.append({
                "anchor_date":  a_date,
                "anchor_label": a_label,
                "matched_detected_date": best,
                "offset_days": best_offset,
            })
            used_detected.add(best)
        else:
            misses.append((a_date, a_label))

    false_alarms = [d for d in detected if d not in used_detected]
    return hits, misses, false_alarms


def summarize_match(
    hits: List[Dict],
    misses: List,
    false_alarms: List,
    n_anchors: int,
) -> Dict[str, Any]:
    """Compute paper-ready summary metrics from match results."""
    n_hits = len(hits)
    recall = n_hits / max(n_anchors, 1)
    precision = n_hits / max(n_hits + len(false_alarms), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    offsets = np.array([h["offset_days"] for h in hits]) if hits else np.array([])
    return {
        "n_anchors":     n_anchors,
        "n_hits":        n_hits,
        "n_misses":      len(misses),
        "n_false_alarms": len(false_alarms),
        "recall":        round(recall, 3),
        "precision":     round(precision, 3),
        "f1":            round(f1, 3),
        "mean_offset_days":   float(offsets.mean()) if len(offsets) else float("nan"),
        "median_offset_days": float(np.median(offsets)) if len(offsets) else float("nan"),
        "abs_mean_offset":    float(np.abs(offsets).mean()) if len(offsets) else float("nan"),
    }


# ── canonical anchor event list (US Fed, 2008-2024) ──────────────────────────

FED_ANCHOR_EVENTS: List[Tuple[pd.Timestamp, str]] = [
    (pd.Timestamp("2008-09-15"), "Lehman bankruptcy / GFC onset"),
    (pd.Timestamp("2008-12-16"), "Fed funds → 0-0.25% (ZLB onset)"),
    (pd.Timestamp("2010-11-03"), "QE2 announced"),
    (pd.Timestamp("2012-09-13"), "QE3 announced"),
    (pd.Timestamp("2013-05-22"), "Bernanke 'taper tantrum' speech"),
    (pd.Timestamp("2014-10-29"), "QE3 ends / asset purchases concluded"),
    (pd.Timestamp("2015-12-16"), "First post-GFC hike"),
    (pd.Timestamp("2018-12-19"), "Last hike of cycle / Powell hawkish-then-pivot"),
    (pd.Timestamp("2019-07-31"), "First post-2018 cut (mid-cycle adjustment)"),
    (pd.Timestamp("2020-03-15"), "COVID emergency cut to 0% + QE infinity"),
    (pd.Timestamp("2021-03-17"), "FOMC SEP first shows 2023 hike dots"),
    (pd.Timestamp("2021-11-03"), "Taper announced; 'transitory' framing fades"),
    (pd.Timestamp("2022-03-16"), "First hike of 2022 tightening cycle"),
    (pd.Timestamp("2023-03-22"), "SVB-era hike + financial stress acknowledgement"),
    (pd.Timestamp("2024-09-18"), "First cut of 2024 easing cycle"),
]


# ── corpus loader ────────────────────────────────────────────────────────────

def load_fomc_corpus(directory: str, pattern: str = "*.txt") -> List[FOMCDoc]:
    """Load FOMC docs from a directory; expects filenames containing YYYY-MM-DD.

    Each file's first line may be the title; remaining lines are the body.
    """
    directory = Path(directory)
    docs: List[FOMCDoc] = []
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")
    for path in sorted(directory.glob(pattern)):
        m = date_re.search(path.name)
        if not m:
            logger.warning("Skipping %s — no YYYY-MM-DD in filename", path.name)
            continue
        date = pd.Timestamp(m.group(1))
        text = path.read_text(encoding="utf-8")
        title = path.stem
        docs.append(FOMCDoc(date=date, text=text, title=title))
    docs.sort(key=lambda d: d.date)
    return docs


__all__ = [
    "FOMCDoc",
    "CandidateChangepoint",
    "ValidatedChangepoint",
    "DetectionResult",
    "TextRegimeDetector",
    "likelihood_ratio_test",
    "match_anchors",
    "summarize_match",
    "load_fomc_corpus",
    "FED_ANCHOR_EVENTS",
]
