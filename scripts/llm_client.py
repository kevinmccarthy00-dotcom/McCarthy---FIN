"""Thin wrapper around the Anthropic API used for every stage's LLM call."""

import json
import os
import re

from anthropic import Anthropic

DEFAULT_MODEL = os.environ.get("ADVISOR_LLM_MODEL", "claude-sonnet-5")

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Set it as an environment "
                "variable before running the advisor agent."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def call_llm(system_prompt: str, user_message: str, max_tokens: int = 3000) -> str:
    """Make a single LLM call with a distinct system prompt. Returns raw text."""
    client = _get_client()
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def parse_json_response(raw_text: str) -> dict:
    """Best-effort parse of a stage's JSON response.

    Strips markdown code fences if the model added them despite instructions,
    then attempts json.loads. On failure, returns the raw text under a
    "raw_text" key with a "parse_error" note instead of raising, so a single
    malformed stage doesn't crash the whole chain.
    """
    cleaned = raw_text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*)\s*```$", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return {"raw_text": raw_text, "parse_error": str(exc)}
