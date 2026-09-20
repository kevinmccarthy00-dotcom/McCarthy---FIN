"""Download the last 30 days of cnbc.com headlines from the GDELT 2.0 DOC API.

Usage:
    python scripts/gdelt_cnbc_headlines.py

Writes a deduplicated CSV (by article URL) with columns:
seendate, title, domain, url
into data/gdelt/.
"""

import csv
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
DOMAIN = "cnbc.com"
LOOKBACK_DAYS = 30
MAX_RECORDS_PER_REQUEST = 250
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
RATE_LIMIT_BACKOFF_SECONDS = [30, 60, 120]
DELAY_BETWEEN_REQUESTS_SECONDS = 3
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gdelt"
CSV_COLUMNS = ["seendate", "title", "domain", "url"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gdelt_cnbc_headlines")


def fetch_window(start_dt, end_dt):
    """Fetch articles for a single time window, retrying on transient failures.

    HTTP 429 (rate limited) is retried separately from other transient
    failures, using a longer exponential backoff and honoring the
    Retry-After header when GDELT provides one.
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
                return []

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
    return []


def collect_articles():
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=LOOKBACK_DAYS)

    logger.info(
        "Downloading %s headlines from %s to %s (UTC)",
        DOMAIN, start_time.isoformat(), end_time.isoformat(),
    )

    articles_by_url = {}
    window_start = start_time
    day_number = 0
    total_days = LOOKBACK_DAYS

    while window_start < end_time:
        day_number += 1
        window_end = min(window_start + timedelta(days=1), end_time)

        logger.info(
            "Day %d/%d: querying %s -> %s",
            day_number, total_days, window_start.date(), window_end.date(),
        )

        articles = fetch_window(window_start, window_end)
        new_count = 0
        for article in articles:
            url = article.get("url")
            if not url:
                continue
            if url not in articles_by_url:
                articles_by_url[url] = article
                new_count += 1

        logger.info(
            "Day %d/%d: retrieved %d articles (%d new, %d running total)",
            day_number, total_days, len(articles), new_count, len(articles_by_url),
        )

        window_start = window_end
        if window_start < end_time:
            time.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

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


def main():
    articles, start_time, end_time = collect_articles()

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
