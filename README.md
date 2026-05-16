# Enhancing Regime Shift Detection Using Unstructured Data

Reproduction code and data for the paper
**"Enhancing Regime Shift Detection Using Unstructured Data: A Study on the
Treasury Market."**

The pipeline pairs an LLM text channel (FOMC minutes) with a data channel
(a 14-variable Treasury / macro panel) and cross-validates the two:
LLM-proposed regime boundaries are checked by a bootstrap likelihood-ratio
VAR test, and any data-driven detector's candidates are ratified by a
lenient LLM text check.

This repository is **fully self-contained and 0-API**: every LLM verdict is
cached under `outputs/llm_regime_cache/`, so all tables and the figure
reproduce exactly with no API key and no network access.

## Layout

```
finalscripts/
  table1.py          # Table 1  — detection metrics (R/P/F1/Off), 14-var panel
  table2.py          # Table 2  — per-anchor signed offsets (anchor-priority greedy)
  fp_table.py        # Table 3  — the 7 Cross-Validation+PCMCI false positives
  fp_all.py          # Sec 4.4  — false-positive breakdown across all four detectors
  figure1_dag.ipynb  # Figure 1 — Fed lift-off lagged-causal-DAG (rolling PCMCI)
src/                 # detection baselines (PELT, BinSeg, Bai-Perron, ...) + project src
notebooks/utils/     # text_regime.py (LLM stages, VAR LR test, anchor matching) + utils
data/
  processed/panel_daily.parquet   # the 14-variable bond-market / macro panel
  raw/credit_spreads.parquet      # credit series (Figure 1 6-var extension)
  fomc_texts/txts/                # FOMC minutes corpus (2010--2024)
outputs/llm_regime_cache/         # cached LLM verdicts (Stage A strict + Stage C lenient)
2026-05-14/
  bootstrap_14v_vs_5v.csv         # Stage-B 14-var residual-bootstrap results
  regimes_14var.parquet           # rolling-PCMCI change-point candidates
anchors.csv          # the 26-event predefined ground-truth anchor list
requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The core table/FP scripts need only `numpy pandas scipy pyarrow ruptures
scikit-learn statsmodels python-dotenv`. `figure1_dag.ipynb` additionally
needs `tigramite matplotlib networkx` (PCMCI). `anthropic` is listed only
to *regenerate* the cache; it is not needed for reproduction.

## Reproduce

Run from the repository root (each script resolves all paths relative to
the repo, so the working directory does not matter):

```bash
python finalscripts/table1.py     # -> Table 1
python finalscripts/table2.py     # -> Table 2
python finalscripts/fp_table.py   # -> Table 3 (7 false positives)
python finalscripts/fp_all.py     # -> Sec 4.4 per-detector FP breakdown
jupyter nbconvert --to notebook --execute finalscripts/figure1_dag.ipynb   # -> Figure 1
```

All four `.py` scripts make **0 API calls** (cached verdicts) and print
tables that match the paper exactly.

## The anchor list

`anchors.csv` is the predefined ground truth: 26 US monetary-policy regime
boundaries over 2010--2024. It was first proposed by prompting a separate
LLM (GPT-5.5) for canonical Fed regime boundaries and then verified by the
authors event-by-event against federalreserve.gov primary sources; events
without primary-source confirmation were dropped. The proposing LLM is
distinct from the pipeline's proposer/ratifier LLM, so the evaluation does
not score a model against its own outputs. The same 26 (date, label) pairs
are embedded in each `finalscripts/*.py` and exported here as a standalone
file.

## Data provenance

Treasury yields (GSW), ACM term premium, FRED macro series, the MOVE
index, and credit spreads are aligned into `data/processed/panel_daily.parquet`
(14 variables, 2010--2024, 3,752 daily observations). FOMC minutes under
`data/fomc_texts/txts/` are sourced from federalreserve.gov.

## Citation

If you use this code, please cite the paper (see the project page linked
from the publication).
