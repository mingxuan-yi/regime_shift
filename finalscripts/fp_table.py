#!/usr/bin/env python
"""Reproduce the false-positive list for paper Table `tab:fp`.

Same 14-variable consistent setup as finalscripts/table1.py (0 API, all
cached). Builds the Cross-Validation + PCMCI detection set
(LLM-validated  U  Stage-C-ratified PCMCI candidates, 14-day dedup) at the
reported operating point theta_C = 0.8, matches it against the 26-anchor
list at +/-90 days (anchor-priority greedy), and prints:

  - total detections, hits, false alarms
  - every false-alarm date, with the channel it came from
    (LLM-validated text candidate vs Stage-C-ratified PCMCI candidate)

so the paper's tab:fp rows are reproduced from data, not hand-written.

Run:  uv run python finalscripts/fp_table.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from utils.text_regime import (  # noqa: E402
    TextRegimeDetector, load_fomc_corpus, match_anchors, summarize_match,
    _truncate, _extract_json, _SYSTEM_PROMPT,
)

WS, WE = pd.Timestamp("2010-01-01"), pd.Timestamp("2024-12-31")
KEY14 = ["yield_2y", "yield_10y", "slope_2s10s", "tp_10y", "rny_10y",
         "ff_rate", "cpi_yoy", "core_pce_yoy", "breakeven_10y",
         "unemp_rate", "ted_spread", "vix", "move", "fed_assets"]
THETA_A, THETA_C, DEDUP, TOL = 0.6, 0.8, 14, 90
CACHE = ROOT / "outputs" / "llm_regime_cache"

ANCHORS = [
    ("2010-08-10", "Reinvestment of MBS principal"),
    ("2010-11-03", "2nd quantitative easing announced"),
    ("2011-08-09", "Calendar-based forward guidance"),
    ("2011-09-21", "Operation Twist"),
    ("2012-09-13", "3rd quantitative easing announced"),
    ("2012-12-12", "Threshold guidance (Evans rule)"),
    ("2013-05-22", "Bernanke taper-tantrum speech"),
    ("2013-12-18", "Taper officially begins"),
    ("2014-10-29", "3rd quantitative easing ends"),
    ("2014-12-17", "Patient language introduced"),
    ("2015-12-16", "First post-GFC hike"),
    ("2017-06-14", "Balance-sheet normalization principles"),
    ("2017-09-20", "Balance-sheet runoff announcement"),
    ("2018-12-19", "Last hike / Powell hawkish-then-pivot"),
    ("2019-01-30", "Powell pivot to patient"),
    ("2019-07-31", "First post-2018 cut"),
    ("2020-03-15", "COVID emergency cut + open-ended QE"),
    ("2020-08-27", "Jackson Hole: FAIT adopted"),
    ("2020-12-16", "Outcome-based forward guidance"),
    ("2021-03-17", "FOMC SEP first shows 2023 hike dots"),
    ("2021-11-03", "Taper announced; transitory fades"),
    ("2021-12-15", "Transitory dropped + accelerated taper"),
    ("2022-03-16", "First hike of 2022 cycle"),
    ("2022-05-04", "50bp hike + balance-sheet runoff start"),
    ("2023-03-22", "Silicon-Valley-Bank-era hike"),
    ("2024-09-18", "First cut of 2024 cycle"),
]
anchors = [(pd.Timestamp(d), e) for d, e in ANCHORS]


def dedup(dates, w=DEDUP):
    s = sorted(set(pd.Timestamp(x) for x in dates))
    if not s:
        return []
    out = [s[0]]
    for d in s[1:]:
        if (d - out[-1]).days > w:
            out.append(d)
    return out


def main():
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    p14 = panel[KEY14].dropna().loc[WS:WE]  # noqa: F841 (kept for parity)
    docs_all = load_fomc_corpus(str(ROOT / "data" / "fomc_texts" / "txts"), pattern="*.txt")
    docs = [d for d in docs_all if WS <= d.date <= WE]
    sd = sorted(docs, key=lambda d: d.date)
    det = TextRegimeDetector(
        model="claude-sonnet-4-6", cache_dir=str(CACHE),
        confidence_threshold=THETA_A, p_value_threshold=0.05,
        max_excerpt_chars=12000, temperature=0.2,
    )

    # Stage A (cached)
    for i in range(1, len(docs)):
        det.llm_call(docs[i - 1], docs[i])

    # Stage B: 14-var residual bootstrap -> all 23 pass (reuse saved CSV)
    b = pd.read_csv(ROOT / "2026-05-14" / "bootstrap_14v_vs_5v.csv", parse_dates=["cp"])
    b14 = b[b["panel"] == "14v"]
    llm_validated = sorted(b14[b14["pass_boot"]]["cp"].tolist())

    # PCMCI candidates on the 14-var panel
    r14 = pd.read_parquet(ROOT / "2026-05-14" / "regimes_14var.parquet")
    pcmci14 = sorted(d for d in r14[r14["is_regime_change"]].index.tolist()
                     if WS <= d <= WE)

    lenient = (
        "A separate statistical detector flagged a structural break in financial "
        "market data on {cp_date}.\n\nLook at the two FOMC documents straddling that "
        "date and decide whether either contains SUBSTANTIVE monetary-policy content "
        "that could plausibly explain such a market break.\n\nPRIOR DOCUMENT (date: "
        "{prev_date}):\n{prev_text}\n\nCURRENT DOCUMENT (date: {curr_date}):\n"
        "{curr_text}\n\nBe LENIENT --- answer yes if there is any plausible content."
        "\n\nOutput JSON only:\n"
        '{{"explains_signal": true | false, "confidence": 0.0-1.0, '
        '"explanation": "1-2 sentences"}}'
    )

    def ratify(cands, theta=THETA_C, mx=120):
        pc, kept = {}, []
        for cp in cands:
            bef = [d for d in sd if d.date < cp]
            aft = [d for d in sd if d.date > cp]
            if not bef or not aft:
                continue
            pv, cu = bef[-1], aft[0]
            if (cp - pv.date).days > mx or (cu.date - cp).days > mx:
                continue
            pk = (pv.text_hash, cu.text_hash)
            if pk in pc:
                payload = pc[pk]
            else:
                f = CACHE / f"lenient_v1_{pv.text_hash}_{cu.text_hash}_{cp.strftime('%Y%m%d')}.json"
                if f.exists():
                    payload = json.loads(f.read_text())
                else:
                    g = list(CACHE.glob(f"lenient_v1_{pv.text_hash}_{cu.text_hash}_*.json"))
                    payload = json.loads(g[0].read_text())
                pc[pk] = payload
            if (bool(payload.get("explains_signal", False))
                    and float(payload.get("confidence", 0)) >= theta):
                kept.append(cp)
        return kept

    kept = ratify(pcmci14)
    final = dedup(list(llm_validated) + kept)
    h, m, fa = match_anchors(final, anchors, tolerance_days=TOL)
    s = summarize_match(h, m, fa, len(anchors))

    lv = set(pd.Timestamp(x) for x in llm_validated)
    kp = set(pd.Timestamp(x) for x in kept)

    def source(d):
        d = pd.Timestamp(d)
        in_lv = any(abs((d - x).days) <= DEDUP for x in lv)
        in_kp = any(abs((d - x).days) <= DEDUP for x in kp)
        if in_lv and in_kp:
            return "LLM+PCMCI"
        if in_lv:
            return "LLM"
        if in_kp:
            return "PCMCI"
        return "?"

    fa_dates = sorted(pd.Timestamp(x["pred"] if isinstance(x, dict) else x) for x in fa)
    print(f"\nCross-Validation + PCMCI  (theta_C = {THETA_C}, 14-var panel)")
    print(f"  total detections : {len(final)}")
    print(f"  hits / 26        : {s['n_hits']}")
    print(f"  false alarms     : {s['n_false_alarms']}\n")
    print(f"{'FP date':<12} {'source':<10}")
    print("-" * 24)
    for d in fa_dates:
        print(f"{d.strftime('%Y-%m-%d'):<12} {source(d):<10}")
    print("-" * 24)
    print("\nfull detection set:")
    print(", ".join(d.strftime("%Y-%m-%d") for d in final))


if __name__ == "__main__":
    main()
