# Data Limitations

**This submission uses a 30-day sample, not the full 365-day dataset the assignment specifies.**

The assignment calls for one year of CNBC headlines pulled from the GDELT 2.0
DOC API. GDELT rate-limited the full-year pull (HTTP 429) heavily enough,
even after adding adaptive date-window splitting, slow exponential backoff,
jitter, and Retry-After handling, that the full 365-day run could not be
completed before this submission was due.

Rather than fabricate, backfill, or synthesize a year of missing headlines
or returns, this submission runs the complete end-to-end pipeline
(headlines -> Mkt-RF -> LLM predictions -> performance analysis) on the
subset of days GDELT actually returned within the available time. Every
headline, prediction, and return figure in this submission comes from a
real GDELT/Ken French/Claude API response for a real date in that window --
nothing is invented to fill the gap between the 30-day sample and the
365-day target.

`scripts/gdelt_cnbc_headlines.py` tracks this explicitly: every headlines
CSV it writes has a `.meta.json` sidecar recording whether the dataset is
`complete` for its requested date range and how many date windows are
unresolved due to rate limiting. `scripts/performance_analysis.py` reads
that file (via `--gdelt-meta`) and reports the same completeness figures in
`performance_report.md`.

To reproduce or extend this run with more data, once GDELT's rate limit
allows it: `python scripts/gdelt_cnbc_headlines.py --days 365` (resumable
via its checkpoint, or `--export-only` to see what's collected so far
without making more requests).
