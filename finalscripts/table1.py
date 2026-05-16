#!/usr/bin/env python
"""Reproduce and print paper Table 1 (14-variable consistent experiment).

Single consistent setup (no panel-mismatch confound):
  - Panel : 14-variable, rolling-252d z-scored (data/processed/panel_daily.parquet)
  - Stage A: strict major_pivot LLM prompt (cached verdicts -> 0 API)
  - Stage B: 14-variable residual-bootstrap LR test; the 23 Stage-A
             candidates all pass (reused from 2026-05-14/bootstrap_14v_vs_5v.csv)
  - Data channel: PELT / BinSeg / Bai-Perron / rolling-PCMCI on the same panel
  - Stage C : generic lenient ratifier, single global theta_C = 0.8
              (cached verdicts -> 0 API)
  - Combined = LLM-validated set  U  Stage-C-ratified detector candidates,
               14-day dedup, scored vs the 26-anchor list at +/-90 days.

R/P/F1 are formatted to 2 decimals DIRECTLY from the raw values (no
intermediate rounding), so the printout matches the paper Table 1 exactly.

Run:  uv run python finalscripts/table1.py
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
from src.baselines import pelt, binseg, bai_perron  # noqa: E402

WS, WE = pd.Timestamp("2010-01-01"), pd.Timestamp("2024-12-31")
KEY14 = ["yield_2y", "yield_10y", "slope_2s10s", "tp_10y", "rny_10y",
         "ff_rate", "cpi_yoy", "core_pce_yoy", "breakeven_10y",
         "unemp_rate", "ted_spread", "vix", "move", "fed_assets"]
THETA_A, THETA_C, DEDUP, TOL = 0.6, 0.8, 14, 90   # theta_C = 0.8 = Table 1 operating point
CACHE = ROOT / "outputs" / "llm_regime_cache"

# 26 anchors (date, label) -- identical to the paper's anchor list.
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


def score(dets):
    """Recall/precision/F1 computed DIRECTLY from raw integer counts.

    summarize_match() pre-rounds recall/precision/F1 to 3 dp, which then
    double-rounds wrongly at 2 dp (e.g. 16/26 = 0.6154 -> stored 0.615 ->
    printed 0.61 instead of the correct 0.62). We therefore recompute from
    the raw hit / false-alarm counts so the 2-dp printout is exact.
    """
    h, m, fa = match_anchors(dedup(dets), anchors, tolerance_days=TOL)
    s = summarize_match(h, m, fa, len(anchors))
    hits = s["n_hits"]
    false_alarms = s["n_false_alarms"]
    n_anchor = len(anchors)
    R = hits / n_anchor
    P = hits / (hits + false_alarms) if (hits + false_alarms) else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) else 0.0
    offs = [hh["offset_days"] for hh in h]
    off = float(np.mean(offs)) if offs else float("nan")
    return R, P, F1, hits, false_alarms, off


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
    llm_strict = []
    for i in range(1, len(docs)):
        p = det.llm_call(docs[i - 1], docs[i])
        if p.get("is_changepoint") and float(p.get("confidence", 0)) >= THETA_A:
            llm_strict.append(docs[i].date)

    # Stage B: 14-var residual bootstrap -> all 23 pass (reuse saved CSV)
    b = pd.read_csv(ROOT / "2026-05-14" / "bootstrap_14v_vs_5v.csv", parse_dates=["cp"])
    b14 = b[b["panel"] == "14v"]
    llm_validated = sorted(b14[b14["pass_boot"]]["cp"].tolist())

    # Data-channel detectors on the same 14-variable panel
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
        pc, kept, api = {}, [], 0
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
                    if g:
                        payload = json.loads(g[0].read_text())
                    else:
                        pr = lenient.format(
                            cp_date=cp.strftime("%Y-%m-%d"),
                            prev_date=pv.date.strftime("%Y-%m-%d"),
                            curr_date=cu.date.strftime("%Y-%m-%d"),
                            prev_text=_truncate(pv.text, det.max_excerpt_chars),
                            curr_text=_truncate(cu.text, det.max_excerpt_chars))
                        cl = det._get_client()
                        kw = dict(model=det.model, max_tokens=512,
                                  system=_SYSTEM_PROMPT,
                                  messages=[{"role": "user", "content": pr}])
                        if det.temperature is not None:
                            kw["temperature"] = det.temperature
                        try:
                            rp = cl.messages.create(**kw)
                        except Exception as e:
                            if "temperature" in str(e).lower():
                                kw.pop("temperature")
                                rp = cl.messages.create(**kw)
                            else:
                                raise
                        payload = _extract_json(rp.content[0].text)
                        f.write_text(json.dumps(payload, indent=2))
                        api += 1
                pc[pk] = payload
            if (bool(payload.get("explains_signal", False))
                    and float(payload.get("confidence", 0)) >= theta):
                kept.append(cp)
        return kept, api

    # Build Table 1 rows (RAW values; format to 2 dp only at print time)
    rows = []
    rR, rP, rF, rH, rFA, rOff = score(llm_validated)
    rows.append(("LLM only", rR, rP, rF, rOff, rH, rFA))
    combined_rows, api_total = [], 0
    for name, cands in DETS.items():
        sR, sP, sF, sH, sFA, sOff = score(cands)               # standalone
        rows.append((name, sR, sP, sF, sOff, sH, sFA))
        kept, api = ratify(cands)
        api_total += api
        cR, cP, cF, cH, cFA, cOff = score(dedup(list(llm_validated) + kept))
        combined_rows.append((f"Cross Validation + {name}", cR, cP, cF, cOff, cH, cFA))
    rows += combined_rows

    f1s = [r[3] for r in rows]
    best = max(f1s)
    print(f"\n# Paper Table 1  (14-variable consistent panel; "
          f"theta_C = {THETA_C} for Cross Validation; new API calls = {api_total})\n")
    print(f"{'Method':<30} {'R':>5} {'P':>5} {'F1':>6} {'Off':>7}  {'hits':>4} {'FA':>3}")
    print("-" * 64)
    for name, R, P, F1, Off, H, FA in rows:
        star = " *" if abs(F1 - best) < 1e-9 else ""
        print(f"{name:<30} {R:5.2f} {P:5.2f} {F1:6.2f} {Off:+7.1f}  {H:4d} {FA:3d}{star}")
    print("-" * 64)
    print("* = best F1.  R/P/F1 computed directly from raw counts (exact 2-dp).\n"
          "Standalone PELT/BinSeg/Bai-Perron/PCMCI = data-only baselines;\n"
          "Cross Validation = our detector-agnostic pipeline on the same panel.")


if __name__ == "__main__":
    main()
