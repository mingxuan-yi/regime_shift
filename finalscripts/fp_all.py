#!/usr/bin/env python
"""False positives for ALL four Cross-Validation instantiations.

Same 14-variable consistent setup as finalscripts/table1.py (0 API, all
cached), theta_C = 0.8. For each data channel (PELT / BinSeg / Bai-Perron /
PCMCI) builds the Cross-Validation set (LLM-validated U Stage-C-ratified
detector candidates, 14-day dedup), matches vs the 26-anchor list at
+/-90d, and prints every false alarm split by source:

  - LLM   : the FP comes from a bootstrap-validated text candidate
            (this set is IDENTICAL across all four instantiations)
  - <det> : the FP comes from the Stage-C-ratified data-side candidate

so we can check (a) whether the LLM-side FPs are common to all four and
(b) how many extra data-side FPs each detector adds.

Run:  uv run python finalscripts/fp_all.py
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
    TextRegimeDetector, load_fomc_corpus, match_anchors, summarize_match,
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

    for i in range(1, len(docs)):
        det.llm_call(docs[i - 1], docs[i])

    b = pd.read_csv(ROOT / "2026-05-14" / "bootstrap_14v_vs_5v.csv", parse_dates=["cp"])
    b14 = b[b["panel"] == "14v"]
    llm_validated = sorted(b14[b14["pass_boot"]]["cp"].tolist())

    r14 = pd.read_parquet(ROOT / "2026-05-14" / "regimes_14var.parquet")
    pcmci14 = sorted(d for d in r14[r14["is_regime_change"]].index.tolist()
                     if WS <= d <= WE)
    DETS = {
        "PELT": pelt.detect(p14),
        "BinSeg": binseg.detect(p14),
        "Bai-Perron": bai_perron.detect(p14, n_bkps=25, jump=5),
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

    lv = set(pd.Timestamp(x) for x in llm_validated)

    def src(d, kp):
        d = pd.Timestamp(d)
        in_lv = any(abs((d - x).days) <= DEDUP for x in lv)
        in_kp = any(abs((d - x).days) <= DEDUP for x in kp)
        if in_lv and in_kp:
            return "LLM+det"
        if in_lv:
            return "LLM"
        if in_kp:
            return "det"
        return "?"

    per_det_fps = {}
    print(f"\ntheta_C = {THETA_C}, 14-var panel, +/-{TOL}d match, {DEDUP}d dedup\n")
    for name, cands in DETS.items():
        kp = ratify(cands)
        final = dedup(list(llm_validated) + kp)
        h, m, fa = match_anchors(final, anchors, tolerance_days=TOL)
        s = summarize_match(h, m, fa, len(anchors))
        fa_dates = sorted(pd.Timestamp(x["pred"] if isinstance(x, dict) else x) for x in fa)
        tagged = [(d, src(d, kp)) for d in fa_dates]
        per_det_fps[name] = tagged
        n_llm = sum(1 for _, sname in tagged if sname.startswith("LLM"))
        n_det = sum(1 for _, sname in tagged if sname == "det")
        print(f"=== Cross-Validation + {name} ===")
        print(f"  detections={len(final)}  hits={s['n_hits']}/26  "
              f"FA={s['n_false_alarms']}  (LLM-side={n_llm}, det-side={n_det})")
        for d, sname in tagged:
            print(f"    {d.strftime('%Y-%m-%d')}  [{sname}]")
        print()

    # Common LLM-side FPs across all four instantiations
    sets = []
    for name, tagged in per_det_fps.items():
        sets.append(set(d.strftime('%Y-%m-%d') for d, sname in tagged
                        if sname.startswith("LLM")))
    common = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    print("LLM-side FPs common to ALL four instantiations:",
          sorted(common))
    print("LLM-side FPs appearing in ANY instantiation        :",
          sorted(union))


if __name__ == "__main__":
    main()
