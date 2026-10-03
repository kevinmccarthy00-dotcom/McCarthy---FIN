"""Part 2.4: query Claude Haiku 4.5 for every scenario x agent type (50 x 3 = 150).

Persistence and resume
----------------------
* data/raw_responses.jsonl is the append-only source of truth. Every attempt
  (success, unparseable reply, or API error) is written as one JSON line and
  flushed + fsync'd to disk before the next call, so nothing is lost if the
  run is interrupted.
* Each reply is classified by response_parser.classify as ok (S/U),
  refusal, ambiguous, or parse_error. Refusals and ambiguous replies are
  genuine outcomes: they are kept, never retried, and never counted as U.
* A logged final outcome (ok / refusal / ambiguous) is frozen: it is never
  re-classified, so a refusal can never later become S or U. Only replies
  logged as parse errors, or by the pre-fix parser (no parser_version), are
  re-parsed from their stored raw text on restart. Any pair with a final
  outcome is skipped. Pairs with only parse or API errors are retried, up to
  --max-attempts total attempts per pair (failures are kept in the log).
* Every record carries a hash of the prompts. The runner refuses to add to a
  log made with different prompts, so one dataset never mixes prompt versions,
  and refuses to call the API at all if the prompts no longer match the frozen
  hash in prompts.lock (the final experiment's prompts).
* data/llm_responses.csv (the deliverable) is rebuilt from the JSONL at the
  end of every run, including after Ctrl-C or an error.

The API key is read by the Anthropic SDK from the ANTHROPIC_API_KEY
environment variable. It is never printed, logged, or written to disk.

Usage
-----
  python scripts/run_evaluation.py --dry-run          # show prompts, no API calls
  python scripts/run_evaluation.py --limit 2          # pilot: first 2 scenarios (6 calls)
  python scripts/run_evaluation.py                    # full run / resume
  python scripts/run_evaluation.py --rebuild-csv      # only regenerate the CSV
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from prompts import SYSTEM_PROMPTS, USER_TEMPLATE, build_user_message
from response_parser import PARSER_VERSION, TERMINAL_STATUSES, classify

MODEL = "claude-haiku-4-5-20251001"
TEMPERATURE = 1.0
MAX_TOKENS = 300
AGENTS = ["baseline", "economicus", "moralis"]

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_CSV = ROOT / "data" / "scenarios.csv"
RAW_JSONL = ROOT / "data" / "raw_responses.jsonl"
RESPONSES_CSV = ROOT / "data" / "llm_responses.csv"

# recommendation is S, U, REFUSED, or AMBIGUOUS; status is ok / refusal / ambiguous
CSV_COLUMNS = [
    "scenario_id", "agent_type", "recommendation", "justification",
    "status", "refusal_language", "prompt_hash", "model", "attempts", "timestamp",
]

PROMPT_HASH = hashlib.sha256(
    json.dumps({"system": SYSTEM_PROMPTS, "user": USER_TEMPLATE}, sort_keys=True).encode()
).hexdigest()[:12]
PROMPT_LOCK = ROOT / "prompts.lock"


def check_prompt_lock():
    """Return an error message if the prompts differ from the frozen hash, else None."""
    if not PROMPT_LOCK.exists():
        return f"{PROMPT_LOCK.name} is missing; the prompts are not frozen."
    locked = PROMPT_LOCK.read_text(encoding="utf-8").strip()
    if locked != PROMPT_HASH:
        return (f"Prompts changed: current hash {PROMPT_HASH} != frozen hash {locked} in "
                f"{PROMPT_LOCK.name}. The final experiment must use the frozen prompts.")
    return None


def load_scenarios():
    with SCENARIOS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reclassify(record):
    """Re-parse a logged reply only if it has no final outcome yet.

    Final outcomes logged by this parser (ok / refusal / ambiguous) are frozen,
    so a recorded refusal is never turned into S or U. Parse errors and records
    from the pre-fix parser (no parser_version) are re-parsed from their raw
    text. The log on disk is never rewritten; the logged status is kept as
    logged_status. API errors (no raw text) are left as they are.
    """
    if record.get("raw_text") is None:
        return record
    if record.get("parser_version") is not None and record["status"] in TERMINAL_STATUSES:
        return record
    out = dict(record)
    out["logged_status"] = record.get("logged_status", record["status"])
    out.update(classify(record["raw_text"], record.get("stop_reason")))
    return out


def load_log():
    """Read all logged attempts (re-classified). A truncated final line is skipped."""
    records = []
    if RAW_JSONL.exists():
        with RAW_JSONL.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(reclassify(json.loads(line)))
                except json.JSONDecodeError:
                    print(f"  warning: skipping corrupt log line {line_no}", file=sys.stderr)
    return records


def append_record(record):
    """Append one record and force it to disk before returning."""
    RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)
    # ASCII-only JSON (non-ASCII characters escaped) so the file is identical on every OS
    with RAW_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rebuild_csv(records):
    """Write the deliverable CSV: one row per (scenario, agent) with a final outcome."""
    attempts = {}
    ok = {}
    for r in records:
        key = (int(r["scenario_id"]), r["agent_type"])
        attempts[key] = attempts.get(key, 0) + 1
        if r["status"] in TERMINAL_STATUSES and key not in ok:
            ok[key] = r  # first final outcome is the one we keep
    rows = []
    for key in sorted(ok, key=lambda k: (k[0], AGENTS.index(k[1]))):
        r = ok[key]
        rows.append({
            "scenario_id": key[0],
            "agent_type": key[1],
            "recommendation": r["recommendation"],
            "justification": r["justification"],
            "status": r["status"],
            "refusal_language": r.get("refusal_language", False),
            "prompt_hash": r.get("prompt_hash", ""),
            "model": r.get("response_model") or r["model"],
            "attempts": attempts[key],
            "timestamp": r["timestamp"],
        })
    tmp = RESPONSES_CSV.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    try:
        tmp.replace(RESPONSES_CSV)  # atomic swap: the CSV is never half-written
    except PermissionError:
        # Windows: the CSV is open in another program (e.g. Excel). The log is safe.
        print(f"  warning: could not replace {RESPONSES_CSV.name} (open in another program?). "
              f"Close it and run with --rebuild-csv.", file=sys.stderr)
    return len(rows)


def summarize(records, scenarios):
    expected = {(int(s["scenario_id"]), a) for s in scenarios for a in AGENTS}
    ok = {(int(r["scenario_id"]), r["agent_type"]) for r in records if r["status"] in TERMINAL_STATUSES}
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nProgress: {len(ok & expected)}/{len(expected)} pairs complete. "
          f"Logged attempts by outcome: {dict(sorted(counts.items()))}")
    missing = sorted(expected - ok)
    if missing:
        preview = ", ".join(f"{s}/{a}" for s, a in missing[:10])
        print(f"Still missing ({len(missing)}): {preview}{' ...' if len(missing) > 10 else ''}")


def call_model(client, system_prompt, user_message):
    return client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        extra_body={"temperature": TEMPERATURE},  # SDK 1.x removed the kwarg; Haiku 4.5 still honours it
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print prompts for the first scenario; no API calls")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N scenarios (pilot)")
    ap.add_argument("--max-attempts", type=int, default=3, help="max logged attempts per pair (default 3)")
    ap.add_argument("--allow-mixed-prompts", action="store_true",
                    help="allow adding to a log made with different prompts (not recommended)")
    ap.add_argument("--rebuild-csv", action="store_true", help="regenerate llm_responses.csv from the log and exit")
    args = ap.parse_args()

    scenarios = load_scenarios()
    if args.limit:
        scenarios = scenarios[: args.limit]

    if args.dry_run:
        s = scenarios[0]
        for agent in AGENTS:
            print(f"===== {agent} | system prompt =====\n{SYSTEM_PROMPTS[agent]}\n")
        print(f"===== user message (scenario {s['scenario_id']}) =====\n{build_user_message(s)}\n")
        print(f"Would make {len(scenarios) * len(AGENTS)} calls to {MODEL} at temperature {TEMPERATURE}.")
        lock_error = check_prompt_lock()
        print(f"Prompt hash {PROMPT_HASH}: " + (lock_error or "matches prompts.lock (frozen)."))
        return

    if args.rebuild_csv:
        n = rebuild_csv(load_log())
        print(f"Rebuilt {RESPONSES_CSV} with {n} rows.")
        return

    lock_error = check_prompt_lock()
    if lock_error:
        sys.exit(lock_error)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. See README.md > 'Configure the API key'.")

    import anthropic  # imported late so --dry-run works without the SDK
    client = anthropic.Anthropic(max_retries=4, timeout=60.0)  # SDK retries 429/5xx with backoff

    records = load_log()
    other = sorted({r.get("prompt_hash", "none") for r in records} - {PROMPT_HASH})
    if other and not args.allow_mixed_prompts:
        sys.exit(f"{RAW_JSONL.name} contains responses made with different prompts "
                 f"(prompt hash {', '.join(other)}; current {PROMPT_HASH}).\n"
                 "Archive it (move it into data/archive/) so the dataset uses one prompt "
                 "version, or pass --allow-mixed-prompts to override (not recommended).")
    done = {(int(r["scenario_id"]), r["agent_type"]) for r in records if r["status"] in TERMINAL_STATUSES}
    tries = {}
    for r in records:
        key = (int(r["scenario_id"]), r["agent_type"])
        tries[key] = tries.get(key, 0) + 1

    todo = [(s, a) for s in scenarios for a in AGENTS
            if (int(s["scenario_id"]), a) not in done
            and tries.get((int(s["scenario_id"]), a), 0) < args.max_attempts]
    print(f"Model {MODEL}, temperature {TEMPERATURE}. "
          f"{len(done)} pair(s) already complete; {len(todo)} to run.")

    try:
        for i, (scenario, agent) in enumerate(todo, start=1):
            sid = int(scenario["scenario_id"])
            key = (sid, agent)
            user_message = build_user_message(scenario)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "scenario_id": sid,
                "agent_type": agent,
                "attempt": tries.get(key, 0) + 1,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "prompt_hash": PROMPT_HASH,
                "parser_version": PARSER_VERSION,
                "system_prompt": SYSTEM_PROMPTS[agent],
                "user_message": user_message,
            }
            try:
                resp = call_model(client, SYSTEM_PROMPTS[agent], user_message)
                text = "".join(b.text for b in resp.content if b.type == "text")
                record.update(classify(text, resp.stop_reason))
                record.update({
                    "raw_text": text,
                    "response_model": resp.model,
                    "response_id": resp.id,
                    "stop_reason": resp.stop_reason,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                })
            except anthropic.AuthenticationError:
                sys.exit("\nAuthentication failed (401). Check ANTHROPIC_API_KEY; stopping without logging.")
            except anthropic.PermissionDeniedError as e:
                sys.exit(f"\nPermission denied (403): {type(e).__name__}. Stopping.")
            except anthropic.APIError as e:
                # Logged so the failure is preserved; the pair is retried on the next run
                record.update({
                    "status": "api_error",
                    "error_type": type(e).__name__,
                    "error_status": getattr(e, "status_code", None),
                    "error_message": str(getattr(e, "message", e))[:500],
                })

            append_record(record)
            tries[key] = tries.get(key, 0) + 1
            shown = record["status"] if record["status"] != "ok" else record["recommendation"]
            print(f"[{i}/{len(todo)}] scenario {sid:>2} {agent:<10} -> {shown}")
            if record["status"] == "api_error":
                time.sleep(2)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed responses are saved; rerun the same command to resume.")
    finally:
        records = load_log()
        n = rebuild_csv(records)
        print(f"Wrote {n} rows to {RESPONSES_CSV}")
        summarize(records, load_scenarios() if not args.limit else scenarios)


if __name__ == "__main__":
    main()
