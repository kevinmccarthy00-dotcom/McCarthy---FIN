"""Download cnbc.com headlines from the GDELT 2.0 DOC API.

Usage:
    python scripts/gdelt_cnbc_headlines.py --days 30    # walkthrough mode
    python scripts/gdelt_cnbc_headlines.py --days 365   # full assignment mode

Writes a deduplicated CSV (by article URL) with columns:
seendate, title, domain, url
into data/gdelt/.

Request strategy
-----------------
GDELT's DOC 2.0 API caps every response at 250 records and offers no
cursor/offset pagination, so the only way to retrieve more than 250 articles
for a date range is to split that range into smaller windows and query each
one separately.

Rather than requesting one day at a time (365 requests for a full year, and
the main cause of the 429s seen in the local run), this script queries the
*entire* requested date range in a single request first. Only if a window's
response comes back at the 250-record cap (meaning it was truncated and
articles may be missing) does the script split that window in half and
re-query each half, recursing until either a window's result is under the
cap or the window has shrunk to one hour (MIN_SPLIT_WINDOW), at which point
it accepts what GDELT returned and logs a warning that some articles for
that hour may be missing.

This means low-news-volume periods cost a single request, and only
genuinely high-volume periods incur the extra requests needed to page
through them -- keeping the total request count roughly proportional to
article volume rather than to the number of days requested.
"""

import argparse
import csv
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
DOMAIN = "cnbc.com"
DEFAULT_DAYS = 30
MAX_RECORDS_PER_REQUEST = 250
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
RATE_LIMIT_BACKOFF_SECONDS = [30, 60, 120]
DELAY_BETWEEN_REQUESTS_SECONDS = 3
MIN_SPLIT_WINDOW = timedelta(hours=1)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gdelt"
CSV_COLUMNS = ["seendate", "title", "domain", "url"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gdelt_cnbc_headlines")


def fetch_raw(start_dt, end_dt, stats):
    """Issue one GDELT request for a window, retrying on transient failures.

    HTTP 429 is retried separately from other transient failures, using a
    longer exponential backoff and honoring the Retry-After header when
    GDELT provides one. Returns the list of articles on success (which may
    be empty), or None if the window could not be fetched after exhausting
    retries.
    """
    params = {
        "query": f"domain:{DOMAIN}",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": MAX_RECORDS_PER_REQUEST,
        "sort": "DateDesc",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_dt.strftime("%Y%m%d%H%M%S"),
    }

    attempt = 0
    rate_limit_attempt = 0

    while True:
        attempt += 1
        stats["total_requests"] += 1
        try:
            response = requests.get(GDELT_DOC_API, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Request failed for window %s -> %s (attempt %d/%d): %s",
                start_dt, end_dt, attempt, MAX_RETRIES, exc,
            )
            if attempt >= MAX_RETRIES:
                break
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        if response.status_code == 429:
            rate_limit_attempt += 1
            if rate_limit_attempt > len(RATE_LIMIT_BACKOFF_SECONDS):
                logger.error(
                    "Giving up on window %s -> %s after %d rate-limit retries",
                    start_dt, end_dt, len(RATE_LIMIT_BACKOFF_SECONDS),
                )
                return None

            wait_seconds = RATE_LIMIT_BACKOFF_SECONDS[rate_limit_attempt - 1]
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    wait_seconds = max(wait_seconds, float(retry_after))
                except ValueError:
                    logger.warning("Ignoring unparseable Retry-After header: %r", retry_after)

            logger.warning(
                "Rate limited (429) for window %s -> %s (retry %d/%d), waiting %.0fs",
                start_dt, end_dt, rate_limit_attempt, len(RATE_LIMIT_BACKOFF_SECONDS), wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        try:
            response.raise_for_status()
            payload = response.json()
            return payload.get("articles", [])
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Request failed for window %s -> %s (attempt %d/%d): %s",
                start_dt, end_dt, attempt, MAX_RETRIES, exc,
            )
        except ValueError as exc:
            logger.warning(
                "Could not parse JSON for window %s -> %s (attempt %d/%d): %s",
                start_dt, end_dt, attempt, MAX_RETRIES, exc,
            )

        if attempt >= MAX_RETRIES:
            break
        time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    logger.error("Giving up on window %s -> %s after %d attempts", start_dt, end_dt, MAX_RETRIES)
    return None


def fetch_window_recursive(start_dt, end_dt, stats):
    """Fetch all articles in [start_dt, end_dt), splitting the window if the
    250-record cap was hit so no articles are silently dropped.
    """
    logger.info("Querying window %s -> %s", start_dt, end_dt)
    articles = fetch_raw(start_dt, end_dt, stats)

    if articles is None:
        logger.error(
            "Skipping window %s -> %s after repeated failures; results will be incomplete",
            start_dt, end_dt,
        )
        return []

    if len(articles) < MAX_RECORDS_PER_REQUEST:
        logger.info("Window %s -> %s: retrieved %d articles", start_dt, end_dt, len(articles))
        return articles

    window_length = end_dt - start_dt
    if window_length <= MIN_SPLIT_WINDOW:
        logger.warning(
            "Window %s -> %s returned the maximum %d records at the smallest allowed "
            "granularity (%s); some articles in this window may be missing",
            start_dt, end_dt, MAX_RECORDS_PER_REQUEST, MIN_SPLIT_WINDOW,
        )
        return articles

    midpoint = start_dt + window_length / 2
    stats["windows_split"] += 1
    logger.info(
        "Window %s -> %s hit the %d-record cap; splitting at %s and re-querying both halves",
        start_dt, end_dt, MAX_RECORDS_PER_REQUEST, midpoint,
    )

    time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)
    left = fetch_window_recursive(start_dt, midpoint, stats)
    time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)
    right = fetch_window_recursive(midpoint, end_dt, stats)
    return left + right


def collect_articles(days):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    logger.info(
        "Downloading %s headlines from %s to %s (UTC) using adaptive date-window splitting",
        DOMAIN, start_time.isoformat(), end_time.isoformat(),
    )

    stats = {"total_requests": 0, "windows_split": 0}
    raw_articles = fetch_window_recursive(start_time, end_time, stats)

    articles_by_url = {}
    for article in raw_articles:
        url = article.get("url")
        if not url:
            continue
        articles_by_url.setdefault(url, article)

    logger.info(
        "Finished: %d API request(s), %d window split(s), %d raw records, %d unique articles",
        stats["total_requests"], stats["windows_split"], len(raw_articles), len(articles_by_url),
    )

    return list(articles_by_url.values()), start_time, end_time


def write_csv(articles, start_time, end_time):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"cnbc_headlines_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
    )
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for article in sorted(articles, key=lambda a: a.get("seendate", "")):
            writer.writerow({
                "seendate": article.get("seendate", ""),
                "title": article.get("title", ""),
                "domain": article.get("domain", ""),
                "url": article.get("url", ""),
            })

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download cnbc.com headlines from the GDELT 2.0 DOC API.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=(
            "Number of days to look back from now (e.g. 30 for a walkthrough "
            f"run, 365 for the full assignment). Default: {DEFAULT_DAYS}."
        ),
    )
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be a positive integer")
    return args


def main():
    args = parse_args()
    articles, start_time, end_time = collect_articles(args.days)

    if not articles:
        logger.error("No articles collected. Exiting without writing a CSV.")
        sys.exit(1)

    output_path = write_csv(articles, start_time, end_time)

    logger.info("Done. Collected %d unique articles.", len(articles))
    logger.info("CSV saved to: %s", output_path)

    print(f"UNIQUE_ARTICLES={len(articles)}")
    print(f"CSV_PATH={output_path}")


if __name__ == "__main__":
    main()
