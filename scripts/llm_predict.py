"""Predict next-trading-day LONG/SHORT position + sentiment from CNBC headlines.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/llm_predict.py \\
        --headlines data/gdelt/cnbc_headlines_20260821_20260920.csv \\
        --mkt-rf data/famafrench/mkt_rf_20260821_20260920.csv

    # See exactly what would be sent to the API, at zero cost, before spending
    # real money on a run:
    python scripts/llm_predict.py --headlines ... --mkt-rf ... --dry-run

For each trading day D in the Mkt-RF calendar (except the last), this script:
  1. Gathers every headline with `seendate` after the previous trading day's
     4:00pm ET cutoff and at or before D's 4:00pm ET cutoff (i.e. everything
     that would have been public before the market close on day D).
  2. Asks Claude for a structured JSON prediction: a LONG/SHORT position and
     a -5..+5 sentiment score, to be held on the *next* trading day in the
     calendar (target_date).
  3. Logs the full prompt, the full response, and token usage/estimated cost
     to a JSONL log file, and appends one row to the predictions CSV.

The predictions CSV is written incrementally (one row per day, flushed
immediately) and a rerun skips target dates already present in it, so an
interrupted run -- by a rate limit, a crash, or hitting a cost budget -- can
simply be resumed with the same output path.
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("llm_predict")

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "low"
DEFAULT_MAX_HEADLINES_PER_DAY = 40
DEFAULT_MAX_TOKENS = 1024

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "predictions"
LOG_DIR = OUTPUT_DIR / "logs"

# Anthropic first-party pricing, USD per 1M tokens (cached 2026-06-24; verify
# against https://www.anthropic.com/pricing before relying on cost totals for
# anything beyond a rough estimate -- pass --input-price-per-mtok /
# --output-price-per-mtok to override).
PRICING_PER_MTOK = {
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "position": {"type": "string", "enum": ["LONG", "SHORT"]},
            "sentiment_score": {"type": "integer", "minimum": -5, "maximum": 5},
            "rationale": {"type": "string"},
        },
        "required": ["position", "sentiment_score", "rationale"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """\
You are a disciplined quantitative news-sentiment analyst for a systematic \
equity trading desk. You will be given a batch of CNBC headlines that were \
all published before 4:00pm ET on a given trading day. Based only on these \
headlines, decide:

1. `position`: whether the desk should be LONG or SHORT the broad U.S. \
   equity market (proxied by SPY) for the *next* trading session.
2. `sentiment_score`: an integer from -5 (extremely bearish) to +5 \
   (extremely bullish) summarizing the overall tone of the headlines toward \
   the market.
3. `rationale`: one or two sentences explaining the call.

Base your judgment only on the headlines provided. Do not assume access to \
any information published after the stated cutoff time. Respond only with \
the requested JSON.\
"""

CSV_COLUMNS = [
    "target_date", "as_of_date", "num_headlines", "position", "sentiment_score",
    "rationale", "model", "input_tokens", "output_tokens", "estimated_cost_usd",
]


def load_mkt_rf_calendar(path):
    dates = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(datetime.strptime(row["date"], "%Y-%m-%d").date())
    dates.sort()
    if len(dates) < 2:
        logger.error("Need at least 2 trading days in %s to predict any next-day position", path)
        sys.exit(1)
    return dates


def load_headlines(path):
    """Return a list of (seendate_utc, title) sorted ascending by time."""
    headlines = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seendate = row.get("seendate", "")
            title = row.get("title", "")
            if not seendate or not title:
                continue
            try:
                dt = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            except ValueError:
                logger.warning("Skipping headline with unparseable seendate %r", seendate)
                continue
            headlines.append((dt, title))
    headlines.sort(key=lambda h: h[0])
    return headlines


def et_close_cutoff_utc(trading_day):
    return datetime.combine(trading_day, datetime.min.time(), tzinfo=ET).replace(hour=16).astimezone(UTC)


def headlines_before_close(headlines, lower_bound_utc, upper_bound_utc):
    return [h for h in headlines if (lower_bound_utc is None or h[0] > lower_bound_utc) and h[0] <= upper_bound_utc]


def build_user_prompt(as_of_date, target_date, headlines, max_headlines):
    truncated = len(headlines) > max_headlines
    used = headlines[-max_headlines:] if truncated else headlines
    lines = [
        f"As-of trading day: {as_of_date.isoformat()} (headlines below are everything "
        f"published before 4:00pm ET on this date since the prior trading day's close).",
        f"Predict the position to hold for the next trading day: {target_date.isoformat()}.",
        "",
        f"Headlines ({len(used)}{' of ' + str(len(headlines)) + ', most recent kept' if truncated else ''}):",
    ]
    for dt, title in used:
        lines.append(f"- [{dt.strftime('%Y-%m-%d %H:%M UTC')}] {title}")
    if not used:
        lines.append("(No headlines were found in this window.)")
    return "\n".join(lines), truncated


def estimate_cost(model, input_tokens, output_tokens, price_override):
    if price_override:
        input_price, output_price = price_override
    elif model in PRICING_PER_MTOK:
        input_price = PRICING_PER_MTOK[model]["input"]
        output_price = PRICING_PER_MTOK[model]["output"]
    else:
        return None
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


def call_claude(client, model, effort, user_prompt, max_tokens):
    import anthropic

    for attempt in range(1, 4):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                output_config={"format": RESPONSE_SCHEMA, "effort": effort},
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response
        except anthropic.RateLimitError as exc:
            retry_after = int(exc.response.headers.get("retry-after", "30")) if exc.response is not None else 30
            logger.warning("Rate limited by Claude API (attempt %d/3); waiting %ds", attempt, retry_after)
            time.sleep(retry_after)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500 and attempt < 3:
                logger.warning("Claude API server error (attempt %d/3): %s; retrying", attempt, exc)
                time.sleep(5 * attempt)
                continue
            raise
    raise RuntimeError("Giving up on Claude API call after 3 rate-limit retries")


def already_predicted(output_path):
    if not output_path.exists():
        return set()
    done = set()
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["target_date"])
    return done


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict next-trading-day LONG/SHORT + sentiment from CNBC headlines using Claude.",
    )
    parser.add_argument("--headlines", required=True, help="Path to a GDELT headlines CSV.")
    parser.add_argument("--mkt-rf", required=True, help="Path to a Ken French Mkt-RF CSV (supplies the trading-day calendar).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model ID. Default: {DEFAULT_MODEL}.")
    parser.add_argument(
        "--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"],
        help=f"Claude output_config effort level. Default: {DEFAULT_EFFORT} (classification-shaped task).",
    )
    parser.add_argument("--max-headlines-per-day", type=int, default=DEFAULT_MAX_HEADLINES_PER_DAY)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--input-price-per-mtok", type=float, default=None)
    parser.add_argument("--output-price-per-mtok", type=float, default=None)
    parser.add_argument("--output", default=None, help="Predictions CSV path. Default: auto-named under data/predictions/.")
    parser.add_argument("--log-dir", default=None, help="Directory for prompt/response JSONL logs. Default: data/predictions/logs/.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build every prompt and log it, but do not call the Claude API or spend money. "
             "Writes placeholder rows so the rest of the pipeline can be exercised for free.",
    )
    args = parser.parse_args()

    price_override = None
    if args.input_price_per_mtok is not None or args.output_price_per_mtok is not None:
        if args.input_price_per_mtok is None or args.output_price_per_mtok is None:
            parser.error("--input-price-per-mtok and --output-price-per-mtok must be given together")
        price_override = (args.input_price_per_mtok, args.output_price_per_mtok)
    args.price_override = price_override
    return args


def main():
    args = parse_args()

    calendar = load_mkt_rf_calendar(args.mkt_rf)
    headlines = load_headlines(args.headlines)
    logger.info("Loaded %d trading day(s) and %d headline(s)", len(calendar), len(headlines))

    if args.output:
        output_path = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"predictions_{calendar[0].strftime('%Y%m%d')}_{calendar[-1].strftime('%Y%m%d')}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir) if args.log_dir else LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"llm_predict_{run_id}.jsonl"

    done_targets = already_predicted(output_path)
    if done_targets:
        logger.info("Resuming: %d target date(s) already in %s will be skipped", len(done_targets), output_path)

    write_header = not output_path.exists()
    csv_file = open(output_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    client = None
    if not args.dry_run:
        import anthropic
        client = anthropic.Anthropic()

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    rows_written = 0

    previous_cutoff = None
    for i in range(len(calendar) - 1):
        as_of_date = calendar[i]
        target_date = calendar[i + 1]
        cutoff = et_close_cutoff_utc(as_of_date)
        window = headlines_before_close(headlines, previous_cutoff, cutoff)
        previous_cutoff = cutoff

        if target_date.isoformat() in done_targets:
            logger.info("Skipping %s -> %s (already in %s)", as_of_date, target_date, output_path)
            continue

        user_prompt, truncated = build_user_prompt(as_of_date, target_date, window, args.max_headlines_per_day)
        if truncated:
            logger.warning(
                "Day %s: %d headlines exceeds --max-headlines-per-day=%d; kept the most recent %d",
                as_of_date, len(window), args.max_headlines_per_day, args.max_headlines_per_day,
            )

        log_entry = {
            "target_date": target_date.isoformat(),
            "as_of_date": as_of_date.isoformat(),
            "num_headlines": len(window),
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        }

        if args.dry_run:
            logger.info(
                "[DRY RUN] %s -> %s: %d headline(s) in window (no API call made)",
                as_of_date, target_date, len(window),
            )
            position, sentiment_score, rationale = "LONG", 0, "DRY RUN - no real LLM call made."
            model_used = "dry-run"
            input_tokens = output_tokens = 0
            cost = 0.0
            log_entry["response_text"] = None
            log_entry["dry_run"] = True
        else:
            logger.info("Day %s -> %s: calling %s with %d headline(s)", as_of_date, target_date, args.model, len(window))
            response = call_claude(client, args.model, args.effort, user_prompt, args.max_tokens)
            response_text = next((b.text for b in response.content if b.type == "text"), "")
            log_entry["response_text"] = response_text
            log_entry["stop_reason"] = response.stop_reason
            log_entry["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            if response.stop_reason == "refusal":
                logger.error("Day %s -> %s: Claude refused; skipping this day (no fabricated prediction)", as_of_date, target_date)
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(log_entry) + "\n")
                continue

            try:
                parsed = json.loads(response_text)
                position = parsed["position"]
                sentiment_score = int(parsed["sentiment_score"])
                rationale = parsed["rationale"]
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.error("Day %s -> %s: could not parse structured response (%s); skipping", as_of_date, target_date, exc)
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(log_entry) + "\n")
                continue

            model_used = args.model
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = estimate_cost(args.model, input_tokens, output_tokens, args.price_override)
            log_entry["estimated_cost_usd"] = cost

        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_entry) + "\n")

        writer.writerow({
            "target_date": target_date.isoformat(),
            "as_of_date": as_of_date.isoformat(),
            "num_headlines": len(window),
            "position": position,
            "sentiment_score": sentiment_score,
            "rationale": rationale,
            "model": model_used,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": f"{cost:.6f}" if cost is not None else "",
        })
        csv_file.flush()
        rows_written += 1
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        if cost is not None:
            total_cost += cost

    csv_file.close()

    logger.info(
        "Done. Wrote %d new row(s) to %s (%d skipped as already done).",
        rows_written, output_path, len(done_targets),
    )
    logger.info(
        "Token usage this run: %d input, %d output. Estimated cost: $%.4f%s",
        total_input_tokens, total_output_tokens, total_cost,
        " (dry run, no real spend)" if args.dry_run else "",
    )
    logger.info("Full prompts/responses logged to: %s", log_path)

    print(f"PREDICTIONS_CSV={output_path}")
    print(f"ROWS_WRITTEN={rows_written}")
    print(f"ESTIMATED_COST_USD={total_cost:.6f}")


if __name__ == "__main__":
    main()
