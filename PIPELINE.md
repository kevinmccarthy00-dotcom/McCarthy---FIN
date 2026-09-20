# News-Based Trading Pipeline

End-to-end pipeline: CNBC headlines -> Ken French Mkt-RF -> LLM predictions -> performance report.

See `reports/DATA_LIMITATIONS.md` for why this submission uses a 30-day
sample rather than the full 365-day assignment scope.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # required for scripts/llm_predict.py (skip with --dry-run)
```

## Run order

```bash
# 1. Headlines (30-day walkthrough; use --days 365 for the full assignment scope)
python scripts/gdelt_cnbc_headlines.py --days 30
# If a run gets rate-limited partway through, rerun the same command to resume,
# or export what's already checkpointed without making more requests:
python scripts/gdelt_cnbc_headlines.py --days 30 --export-only

# 2. Matching Ken French daily Mkt-RF for the exact same date range
python scripts/fetch_ff_mkt_rf.py --from-gdelt data/gdelt/cnbc_headlines_<start>_<end>.csv

# 3. LLM predictions (LONG/SHORT + sentiment, structured JSON, logged prompts/responses/cost)
python scripts/llm_predict.py \
    --headlines data/gdelt/cnbc_headlines_<start>_<end>.csv \
    --mkt-rf data/famafrench/mkt_rf_<start>_<end>.csv
# Preview prompts and cost-free plumbing before spending real API budget:
python scripts/llm_predict.py --headlines ... --mkt-rf ... --dry-run

# 4. Performance analysis: 5 strategies, all required metrics, gross + net, report + plots
python scripts/performance_analysis.py \
    --predictions data/predictions/predictions_<start>_<end>.csv \
    --mkt-rf data/famafrench/mkt_rf_<start>_<end>.csv \
    --gdelt-meta data/gdelt/cnbc_headlines_<start>_<end>.csv.meta.json
```

## Outputs

- `data/gdelt/cnbc_headlines_*.csv` (+ `.meta.json` completeness sidecar, `checkpoints/`)
- `data/famafrench/mkt_rf_*.csv`
- `data/predictions/predictions_*.csv` (+ `logs/*.jsonl` full prompt/response/token logs)
- `reports/performance_report.md`, `reports/performance_metrics.csv`, `reports/plots/*.png`
