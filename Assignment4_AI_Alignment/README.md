# Assignment 4 — AI Alignment in Financial Advising

Compares three LLM advisor agents (baseline, homo economicus, homo moralis) on
50 "Unsuitable Product Dilemma" scenarios, following Lu, Chen & Hansen (2025).

- **Model:** `claude-haiku-4-5-20251001` (same model for all three conditions)
- **Temperature:** 1.0
- **Theory:** kappa = 0.5, tau = 0.2 → moralis recommends S iff C_s > 0.6 × C_u (ratio threshold ≈ 1.667)
- **Bonus (fine-tuning):** not attempted

## Layout

```
scripts/
  generate_scenarios.py    Part 1: 50 scenarios (seeded, 5 ratio bands x 2 harm levels x 5)
  validate_scenarios.py    Part 1: checks ranges, coverage, and recomputes theory
  prompts.py               Part 2.1-2.3: the three system prompts + shared user message
  check_moralis_prompt.py  Part 2.3: verifies all five required moralis elements
  test_api_connection.py   one-call pre-flight check (key, network, model)
  run_evaluation.py        Part 2.4: 150 calls, crash-safe logging, resumable
  analyze.py               Part 3: rates, theory match, sensitivity tables and plots
data/
  scenarios.csv            50 scenarios with theoretical predictions
  raw_responses.jsonl      every API attempt, incl. raw text and failures (source of truth)
  llm_responses.csv        deliverable: scenario_id, agent_type, recommendation, justification
output/                    tables, plots, sensitivity_summary.md, alignment_assessment.md
```

## Setup

```bash
cd Assignment4_AI_Alignment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Configure the API key

The scripts read `ANTHROPIC_API_KEY` from the environment only. The key is never
printed, logged, or written to any file. Do not paste it into code, chat, or a
committed file.

- **Claude Code on the web:** add `ANTHROPIC_API_KEY=<key>` as an environment
  variable in the cloud environment's settings, then start a new session.
- **Local terminal:** `read -rs ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY`
  (prompts without echoing; lasts only for that shell).

## Run order

```bash
.venv/bin/python scripts/generate_scenarios.py      # already done; deterministic (seeded)
.venv/bin/python scripts/validate_scenarios.py
.venv/bin/python scripts/check_moralis_prompt.py
.venv/bin/python scripts/test_api_connection.py     # 1 tiny call
.venv/bin/python scripts/run_evaluation.py --limit 2  # optional pilot: 6 calls
.venv/bin/python scripts/run_evaluation.py          # remaining calls; rerun to resume
.venv/bin/python scripts/analyze.py
```

### Resume and failure handling

Every attempt is appended to `data/raw_responses.jsonl` and fsync'd before the
next call. On rerun, pairs with a successful response are skipped; pairs that
only failed (API error or unparseable reply) are retried, up to `--max-attempts`
(default 3) logged attempts per pair. `data/llm_responses.csv` is rebuilt from
the log at the end of every run, including after Ctrl-C. The pilot run's
responses count toward the full run; they are not repeated.
