"""Orchestrates the five-stage AI Financial Advisor prompt chain.

Each stage is a separate LLM call with its own system prompt (defined in
prompts.py). The output of each stage is fed forward as context to the
next. External market data is retrieved deterministically via yfinance
(market_data.py) and then handed to a dedicated Stage 3 LLM call whose only
job is to organize/interpret it — never to invent figures.

The full chain output (every stage's raw response, parsed JSON, and the
raw market data) is printed and saved to output/<client>_<timestamp>.json
so the entire reasoning chain is auditable.
"""

import json
import os
import re
from datetime import datetime, timezone

from llm_client import call_llm, parse_json_response
from market_data import get_market_data
from prompts import (
    INTAKE_SYSTEM_PROMPT,
    MARKET_DATA_INTERPRETATION_SYSTEM_PROMPT,
    PORTFOLIO_CONSTRUCTION_SYSTEM_PROMPT,
    RECOMMENDATION_LETTER_SYSTEM_PROMPT,
    RISK_PROFILE_SYSTEM_PROMPT,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "client"


def _log_stage(stage_number: int, title: str, content: str):
    print(f"\n{'=' * 20} Stage {stage_number}: {title} {'=' * 20}")
    print(content)


def run_advisor_chain(client_profile_text: str, client_name: str = "client") -> dict:
    """Runs all five stages in sequence and returns the full chain output."""

    generated_at = datetime.now(timezone.utc).isoformat()

    # Stage 1: Client Intake
    stage1_raw = call_llm(INTAKE_SYSTEM_PROMPT, client_profile_text)
    stage1_parsed = parse_json_response(stage1_raw)
    _log_stage(1, "Client Intake", stage1_raw)

    # Stage 2: Risk Profiling
    stage2_user_message = (
        "Client intake summary (JSON):\n" + json.dumps(stage1_parsed, indent=2)
    )
    stage2_raw = call_llm(RISK_PROFILE_SYSTEM_PROMPT, stage2_user_message)
    stage2_parsed = parse_json_response(stage2_raw)
    _log_stage(2, "Risk Profiling", stage2_raw)

    # Stage 3: External Data Retrieval (deterministic) + LLM interpretation
    raw_market_data = get_market_data()
    _log_stage(3, "Raw Retrieved Market Data (yfinance, no LLM)", json.dumps(raw_market_data, indent=2))

    stage3_user_message = (
        "Client intake summary (JSON):\n"
        + json.dumps(stage1_parsed, indent=2)
        + "\n\nRisk profile assessment (JSON):\n"
        + json.dumps(stage2_parsed, indent=2)
        + "\n\nRaw market data retrieved via yfinance moments ago (JSON) — "
        "treat every number here as ground truth; do not add, estimate, or "
        "supplement any missing values:\n"
        + json.dumps(raw_market_data, indent=2)
    )
    stage3_raw = call_llm(MARKET_DATA_INTERPRETATION_SYSTEM_PROMPT, stage3_user_message)
    stage3_parsed = parse_json_response(stage3_raw)
    _log_stage(3, "Market Data Interpretation (LLM)", stage3_raw)

    # Stage 4: Portfolio Construction
    stage4_user_message = (
        "Client intake summary (JSON):\n"
        + json.dumps(stage1_parsed, indent=2)
        + "\n\nRisk profile assessment (JSON):\n"
        + json.dumps(stage2_parsed, indent=2)
        + "\n\nInterpreted market data (JSON):\n"
        + json.dumps(stage3_parsed, indent=2)
    )
    stage4_raw = call_llm(PORTFOLIO_CONSTRUCTION_SYSTEM_PROMPT, stage4_user_message)
    stage4_parsed = parse_json_response(stage4_raw)
    _log_stage(4, "Portfolio Construction", stage4_raw)

    # Stage 5: Recommendation Letter
    stage5_user_message = (
        "Client intake summary (JSON):\n"
        + json.dumps(stage1_parsed, indent=2)
        + "\n\nRisk profile assessment (JSON):\n"
        + json.dumps(stage2_parsed, indent=2)
        + "\n\nInterpreted market data (JSON):\n"
        + json.dumps(stage3_parsed, indent=2)
        + "\n\nProposed portfolio allocation (JSON):\n"
        + json.dumps(stage4_parsed, indent=2)
    )
    stage5_raw = call_llm(RECOMMENDATION_LETTER_SYSTEM_PROMPT, stage5_user_message, max_tokens=4000)
    _log_stage(5, "Recommendation Letter", stage5_raw)

    result = {
        "client_name": client_name,
        "generated_at": generated_at,
        "client_profile_input": client_profile_text,
        "stage_1_intake": {
            "system_prompt": INTAKE_SYSTEM_PROMPT,
            "raw_response": stage1_raw,
            "parsed": stage1_parsed,
        },
        "stage_2_risk_profile": {
            "system_prompt": RISK_PROFILE_SYSTEM_PROMPT,
            "raw_response": stage2_raw,
            "parsed": stage2_parsed,
        },
        "raw_market_data": raw_market_data,
        "stage_3_market_interpretation": {
            "system_prompt": MARKET_DATA_INTERPRETATION_SYSTEM_PROMPT,
            "raw_response": stage3_raw,
            "parsed": stage3_parsed,
        },
        "stage_4_portfolio": {
            "system_prompt": PORTFOLIO_CONSTRUCTION_SYSTEM_PROMPT,
            "raw_response": stage4_raw,
            "parsed": stage4_parsed,
        },
        "stage_5_recommendation_letter": {
            "system_prompt": RECOMMENDATION_LETTER_SYSTEM_PROMPT,
            "text": stage5_raw,
        },
    }

    result["saved_output_path"] = _save_output(result, client_name)
    return result


def _save_output(result: dict, client_name: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{_slugify(client_name)}_{timestamp}.json"
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull chain output saved to: {path}")
    return path


if __name__ == "__main__":
    from client_profiles import SAMPLE_PROFILE

    run_advisor_chain(SAMPLE_PROFILE, client_name="sample_client")
