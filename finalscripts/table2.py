#!/usr/bin/env python
"""Reproduce and print paper Table 2 (per-anchor signed offsets).

Same single consistent 14-variable setup as finalscripts/table1.py
(0 API, all cached):

  - Panel : 14-variable, rolling-252d z-scored
  - Stage A: strict major_pivot LLM prompt (cached)
  - Stage B: 14-variable residual-bootstrap LR test; the Stage-A
             candidates that pass (reused from bootstrap_14v_vs_5v.csv)
             -> the "LLM only" column
  - Standalone : PELT / BinSeg / Bai-Perron / rolling-PCMCI, each run
                 alone on the same panel
  - Cross Validation : LLM-validated  U  Stage-C-ratified detector
                       candidates (theta_C = 0.8), 14-day dedup
  - Each column is matched to the 26-anchor list at +/-90 days with the
    SAME anchor-priority greedy assignment as the paper
    (utils.text_regime.match_anchors: each detection serves one anchor;
     offset_days = matched - anchor, + = detection lags the anchor).

For every anchor the signed offset (days) is printed per column, "---"
if no detection falls within +/-90 d; the bottom row is hits / 26.
This matches Table~\\ref{tab:per-anchor} exactly.

Run:  uv run python finalscripts/table2.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from utils.text_regime import (  # noqa: E402
    TextRegimeDetector, load_fomc_corpus, match_anchors,
    _truncate, _extract_json, _SYSTEM_PROMPT,
)
from src.baselines import pelt, binseg, bai_perron  # noqa: E402

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


def offsets_by_anchor(dets):
    """anchor_date -> signed offset (int) for hits; missing anchors absent."""
    h, _m, _fa = match_anchors(dedup(dets), anchors, tolerance_days=TOL)
    return {pd.Timestamp(x["anchor_date"]): int(x["offset_days"]) for x in h}


def main():
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    p14 = panel[KEY14].dropna().loc[WS:WE]
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

    # Stage B: 14-var residual bootstrap -> passing Stage-A set (= LLM only)
    b = pd.read_csv(ROOT / "2026-05-14" / "bootstrap_14v_vs_5v.csv", parse_dates=["cp"])
    b14 = b[b["panel"] == "14v"]
    llm_validated = sorted(b14[b14["pass_boot"]]["cp"].tolist())

    # Data-channel detectors on the same 14-variable panel
    r14 = pd.read_parquet(ROOT / "2026-05-14" / "regimes_14var.parquet")
    pcmci14 = sorted(d for d in r14[r14["is_regime_change"]].index.tolist()
                     if WS <= d <= WE)
    STANDALONE = {
        "PELT": pelt.detect(p14),
        "BinSeg": binseg.detect(p14),
        "Bai-P": bai_perron.detect(p14, n_bkps=25, jump=5),
        "PCMCI": pcmci14,
    }

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

    # Build the 9 method columns
    cols = ["LLM only"]
    omap = {"LLM only": offsets_by_anchor(llm_validated)}
    for name, cands in STANDALONE.items():
        cols.append(name)
        omap[name] = offsets_by_anchor(cands)
    for name, cands in STANDALONE.items():
        cv = f"CV+{name}"
        cols.append(cv)
        kept = ratify(cands)
        omap[cv] = offsets_by_anchor(dedup(list(llm_validated) + kept))

    def cell(v):
        if v is None:
            return "---"
        return f"+{v}" if v >= 0 else f"{v}"

    w = 8
    hdr = f"{'Anchor':<12} {'Event':<40} " + "".join(f"{c:>{w}}" for c in cols)
    print("\n# Paper Table 2  (per-anchor signed offset, 14-var, "
          f"theta_C={THETA_C}; '+' = detection lags anchor)\n")
    print(hdr)
    print("-" * len(hdr))
    hits = {c: 0 for c in cols}
    for a_date, a_lbl in anchors:
        row = f"{a_date.strftime('%Y-%m-%d'):<12} {a_lbl[:40]:<40} "
        for c in cols:
            v = omap[c].get(a_date)
            if v is not None:
                hits[c] += 1
            row += f"{cell(v):>{w}}"
        print(row)
    print("-" * len(hdr))
    foot = f"{'Hits / 26':<12} {'':<40} " + "".join(f"{hits[c]:>{w}}" for c in cols)
    print(foot)
    print("\n(LLM only = Stage A + Stage B 14-var bootstrap; "
          "CV+X = LLM-validated U Stage-C-ratified X, 14-day dedup.)")


if __name__ == "__main__":
    main()
