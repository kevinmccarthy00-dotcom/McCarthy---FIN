"""Show logged replies and how the current parser classifies them (no API calls).

By default prints every attempt whose logged status was not "ok", with the raw
model text, stop_reason and token count, so format problems, truncation and
refusals can be told apart.

Usage:
  python scripts/inspect_responses.py                     # all non-ok attempts
  python scripts/inspect_responses.py --agent economicus  # every economicus attempt
  python scripts/inspect_responses.py --log data/pilot_raw_responses_old_prompts.jsonl
"""

import argparse
import json
from pathlib import Path

from response_parser import classify

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "data" / "raw_responses.jsonl"))
    ap.add_argument("--agent", choices=["baseline", "economicus", "moralis"])
    args = ap.parse_args()

    shown = 0
    with open(args.log) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if args.agent:
                if r["agent_type"] != args.agent:
                    continue
            elif r["status"] == "ok":
                continue
            shown += 1
            now = classify(r["raw_text"], r.get("stop_reason")) if r.get("raw_text") is not None else None
            print("=" * 78)
            print(f"scenario {r['scenario_id']} | {r['agent_type']} | attempt {r.get('attempt')} | "
                  f"logged status: {r['status']}")
            if now:
                print(f"current parser: {now['status']} / {now['recommendation']} | "
                      f"refusal language: {now['refusal_language']}")
            print(f"stop_reason: {r.get('stop_reason')} | output tokens: {r.get('output_tokens')} | "
                  f"prompt hash: {r.get('prompt_hash', 'n/a (pre-fix)')}")
            if r.get("error_type"):
                print(f"API error: {r['error_type']} {r.get('error_status')} {r.get('error_message')}")
            print("-" * 78)
            print(r.get("raw_text") if r.get("raw_text") is not None else "(no reply text)")
    print("=" * 78)
    print(f"{shown} record(s) shown from {args.log}")


if __name__ == "__main__":
    main()
