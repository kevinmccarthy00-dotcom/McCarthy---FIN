"""Download cnbc.com headlines from the GDELT 2.0 DOC API.

Usage:
    python scripts/gdelt_cnbc_headlines.py --days 30    # walkthrough mode
    python scripts/gdelt_cnbc_headlines.py --days 365   # full assignment mode

    # Export whatever has already been checkpointed to CSV without making
    # any new requests (useful when a run got heavily rate-limited):
    python scripts/gdelt_cnbc_headlines.py --days 30 --export-only

    # Quick partial sample: exactly one GDELT request, minimal retry, for
    # when there's no time left for the adaptive/checkpointed modes above:
    python scripts/gdelt_cnbc_headlines.py --quick-sample 7

Writes a deduplicated CSV (by article URL) with columns:
seendate, title, domain, url
into data/gdelt/, plus a "<csv name>.meta.json" sidecar recording whether
the dataset is complete for the requested date range.

Request strategy
-----------------
GDELT's DOC 2.0 API caps every response at 250 records and offers no
cursor/offset pagination, so the only way to retrieve more than 250 articles
for a date range is to split that range into smaller windows and query each
one separately.

Rather than requesting one day at a time, this script queries the *entire*
requested date range in a single request first. Only if a window's response
comes back at the 250-record cap (meaning it was truncated and articles may
be missing) does the script split that window in half and re-query each
half, recursing until either a window's result is under the cap or the
window has shrunk to one hour (MIN_SPLIT_WINDOW), at which point it accepts
what GDELT returned and logs a warning that some articles for that hour may
be missing. This keeps total request count roughly proportional to article
volume rather than to the number of days requested.

Rate-limit handling
--------------------
GDELT can still return HTTP 429 even on the first request. Each window is
retried up to RATE_LIMIT_MAX_RETRIES times with a slow exponential backoff
(60s, 120s, 240s, 480s, capped at RATE_LIMIT_MAX_BACKOFF_SECONDS), with
random jitter so retries from a single run don't land on perfectly
predictable intervals. If GDELT sends a Retry-After header, that value is
always honored even when it's longer than the scheduled backoff.

Checkpointing and partial exports
----------------------------------
Every window that finishes successfully (either under the record cap, or a
window GDELT is known to require splitting) is saved to a checkpoint file
under data/gdelt/checkpoints/ as soon as it completes. If the script is
interrupted -- by a long rate-limit wait, a crash, or a manual stop -- simply
rerunning it with the same --days value resumes from the checkpoint instead
of re-fetching windows that already succeeded. Delete the checkpoint file
for a --days value to force a fully fresh run.

If GDELT rate-limiting prevents a run from ever finishing, `--export-only`
writes a CSV from exactly the windows that have already resolved in the
checkpoint -- no new requests, no fabricated or backfilled articles. The
CSV's .meta.json sidecar records `complete: false` and the count of
unresolved windows in that case, so downstream scripts and the final report
can say plainly that the dataset is a partial sample.

Quick-sample mode
------------------
`--quick-sample N` skips the adaptive/checkpointed machinery entirely: it
makes exactly one GDELT request for the most recent N days, accepts
whatever real records come back (even if that's the 250-record cap), and on
a 429 retries only once after a short fixed wait before failing fast. It
writes a CSV immediately from whatever GDELT actually returned -- never
fabricating or filling in missing articles -- with a `.meta.json` sidecar
that always marks the dataset as a partial "quick sample" and states plainly
whether the 250-record cap was hit. Use it when there's no time left for a
full adaptive run; use `--days`/`--export-only` above for a more complete
pull.
"""

import argparse
import csv
import json
import logging
import random
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

RATE_LIMIT_MAX_RETRIES = 7
RATE_LIMIT_BASE_BACKOFF_SECONDS = 60
RATE_LIMIT_MAX_BACKOFF_SECONDS = 480
JITTER_FRACTION = 0.25

DELAY_BETWEEN_REQUESTS_SECONDS = 3
MIN_SPLIT_WINDOW = timedelta(hours=1)

QUICK_SAMPLE_RETRY_WAIT_SECONDS = 15

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "gdelt"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CSV_COLUMNS = ["seendate", "title", "domain", "url"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gdelt_cnbc_headlines")


def with_jitter(seconds):
    """Apply +/- JITTER_FRACTION random jitter to a wait time, floored at 0."""
    jitter = seconds * JITTER_FRACTION
    return max(0.0, seconds + random.uniform(-jitter, jitter))


def rate_limit_backoff_seconds(attempt_number):
    """Exponential backoff for the given 1-indexed 429 retry attempt."""
    backoff = RATE_LIMIT_BASE_BACKOFF_SECONDS * (2 ** (attempt_number - 1))
    return min(backoff, RATE_LIMIT_MAX_BACKOFF_SECONDS)


class Checkpoint:
    """Tracks which date windows have already been resolved for a run.

    A window is either "done" (its final article list is known, because the
    response came in under the record cap or the split floor was hit) or a
    known "split" (its midpoint, once we've learned it needs to be divided).
    Both are persisted to disk immediately so a rerun with the same --days
    value can skip straight past already-resolved windows, and so a partial
    checkpoint can be exported even if the run never finished.
    """

    def __init__(self, path, domain, days, default_start, default_end):
        self.path = path
        self.domain = domain
        self.days = days
        self.start_time = default_start
        self.end_time = default_end
        self.done_windows = {}
        self.split_windows = {}
        self._load()
        self._save()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read checkpoint file %s (%s); starting fresh", self.path, exc)
            return

        if data.get("domain") != self.domain or data.get("days") != self.days:
            logger.warning(
                "Checkpoint %s is for different parameters; starting fresh", self.path,
            )
            return

        self.start_time = datetime.fromisoformat(data["start_time"])
        self.end_time = datetime.fromisoformat(data["end_time"])
        for entry in data.get("done_windows", []):
            self.done_windows[(entry["start"], entry["end"])] = entry["articles"]
        for entry in data.get("split_windows", []):
            self.split_windows[(entry["start"], entry["end"])] = entry["midpoint"]

        logger.info(
            "Resuming from checkpoint %s: %d window(s) already done, %d known split(s)",
            self.path, len(self.done_windows), len(self.split_windows),
        )

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "domain": self.domain,
            "days": self.days,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "done_windows": [
                {"start": s, "end": e, "articles": articles}
                for (s, e), articles in self.done_windows.items()
            ],
            "split_windows": [
                {"start": s, "end": e, "midpoint": midpoint}
                for (s, e), midpoint in self.split_windows.items()
            ],
        }
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        tmp_path.write_text(json.dumps(data))
        tmp_path.replace(self.path)

    @staticmethod
    def _key(start_dt, end_dt):
        return (start_dt.isoformat(), end_dt.isoformat())

    def get_done(self, start_dt, end_dt):
        return self.done_windows.get(self._key(start_dt, end_dt))

    def mark_done(self, start_dt, end_dt, articles):
        self.done_windows[self._key(start_dt, end_dt)] = articles
        self._save()

    def get_split(self, start_dt, end_dt):
        midpoint = self.split_windows.get(self._key(start_dt, end_dt))
        return datetime.fromisoformat(midpoint) if midpoint else None

    def mark_split(self, start_dt, end_dt, midpoint_dt):
        self.split_windows[self._key(start_dt, end_dt)] = midpoint_dt.isoformat()
        self._save()

    def gather_done_articles(self, start_dt, end_dt):
        """Recursively collect articles from resolved windows only.

        Returns (articles, complete, unresolved_windows). A window counts as
        resolved if it's directly "done", or it's a known "split" whose both
        halves are (recursively) resolved. Anything else -- never attempted,
        or attempted and gave up after retries -- comes back in
        unresolved_windows with no articles contributed, so nothing is ever
        fabricated for a gap.
        """
        cached = self.get_done(start_dt, end_dt)
        if cached is not None:
            return list(cached), True, []

        midpoint = self.get_split(start_dt, end_dt)
        if midpoint is not None:
            left_articles, left_complete, left_unresolved = self.gather_done_articles(start_dt, midpoint)
            right_articles, right_complete, right_unresolved = self.gather_done_articles(midpoint, end_dt)
            return (
                left_articles + right_articles,
                left_complete and right_complete,
                left_unresolved + right_unresolved,
            )

        return [], False, [(start_dt, end_dt)]


def fetch_raw(start_dt, end_dt, stats):
    """Issue one GDELT request for a window, retrying on transient failures.

    HTTP 429 is retried separately from other transient failures, using a
    slow exponential backoff with jitter and honoring the Retry-After header
    when GDELT provides one (even if it's longer than the scheduled wait).
    Returns the list of articles on success (which may be empty), or None if
    the window could not be fetched after exhausting retries.
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
            if rate_limit_attempt > RATE_LIMIT_MAX_RETRIES:
                logger.error(
                    "Giving up on window %s -> %s after %d rate-limit retries",
                    start_dt, end_dt, RATE_LIMIT_MAX_RETRIES,
                )
                return None

            wait_seconds = with_jitter(rate_limit_backoff_seconds(rate_limit_attempt))
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    retry_after_seconds = float(retry_after)
                    logger.info(
                        "GDELT sent Retry-After: %.0fs for window %s -> %s",
                        retry_after_seconds, start_dt, end_dt,
                    )
                    wait_seconds = max(wait_seconds, retry_after_seconds)
                except ValueError:
                    logger.warning("Ignoring unparseable Retry-After header: %r", retry_after)

            logger.warning(
                "Rate limited (429) for window %s -> %s (retry %d/%d), waiting %.0fs",
                start_dt, end_dt, rate_limit_attempt, RATE_LIMIT_MAX_RETRIES, wait_seconds,
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


def fetch_window_recursive(start_dt, end_dt, stats, checkpoint):
    """Fetch all articles in [start_dt, end_dt), splitting the window if the
    250-record cap was hit so no articles are silently dropped. Windows
    already resolved in the checkpoint are served from disk without making
    a request.
    """
    cached = checkpoint.get_done(start_dt, end_dt)
    if cached is not None:
        logger.info("Window %s -> %s already completed (checkpoint): %d articles", start_dt, end_dt, len(cached))
        return cached

    known_midpoint = checkpoint.get_split(start_dt, end_dt)
    if known_midpoint is not None:
        logger.info(
            "Window %s -> %s already known to require splitting (checkpoint); resuming both halves",
            start_dt, end_dt,
        )
        left = fetch_window_recursive(start_dt, known_midpoint, stats, checkpoint)
        right = fetch_window_recursive(known_midpoint, end_dt, stats, checkpoint)
        return left + right

    time.sleep(with_jitter(DELAY_BETWEEN_REQUESTS_SECONDS))
    logger.info("Querying window %s -> %s", start_dt, end_dt)
    articles = fetch_raw(start_dt, end_dt, stats)

    if articles is None:
        stats["failed_windows"] += 1
        logger.error(
            "Skipping window %s -> %s after repeated failures; results will be incomplete "
            "(rerun with the same --days to retry just this window)",
            start_dt, end_dt,
        )
        return []

    if len(articles) < MAX_RECORDS_PER_REQUEST:
        logger.info("Window %s -> %s: retrieved %d articles", start_dt, end_dt, len(articles))
        checkpoint.mark_done(start_dt, end_dt, articles)
        return articles

    window_length = end_dt - start_dt
    if window_length <= MIN_SPLIT_WINDOW:
        logger.warning(
            "Window %s -> %s returned the maximum %d records at the smallest allowed "
            "granularity (%s); some articles in this window may be missing",
            start_dt, end_dt, MAX_RECORDS_PER_REQUEST, MIN_SPLIT_WINDOW,
        )
        checkpoint.mark_done(start_dt, end_dt, articles)
        return articles

    midpoint = start_dt + window_length / 2
    stats["windows_split"] += 1
    logger.info(
        "Window %s -> %s hit the %d-record cap; splitting at %s and re-querying both halves",
        start_dt, end_dt, MAX_RECORDS_PER_REQUEST, midpoint,
    )
    checkpoint.mark_split(start_dt, end_dt, midpoint)

    left = fetch_window_recursive(start_dt, midpoint, stats, checkpoint)
    right = fetch_window_recursive(midpoint, end_dt, stats, checkpoint)
    return left + right


def dedupe_by_url(articles):
    articles_by_url = {}
    for article in articles:
        url = article.get("url")
        if not url:
            continue
        articles_by_url.setdefault(url, article)
    return articles_by_url


def collect_articles(days):
    """Run a live download (using and updating the checkpoint), returning
    (articles_by_url, start_time, end_time, complete, unresolved_count,
    failed_count).
    """
    now = datetime.now(timezone.utc)
    proposed_start = now - timedelta(days=days)

    checkpoint_path = CHECKPOINT_DIR / f"cnbc_headlines_{days}d.checkpoint.json"
    checkpoint = Checkpoint(checkpoint_path, DOMAIN, days, proposed_start, now)
    start_time = checkpoint.start_time
    end_time = checkpoint.end_time

    logger.info(
        "Downloading %s headlines from %s to %s (UTC) using adaptive date-window splitting",
        DOMAIN, start_time.isoformat(), end_time.isoformat(),
    )
    logger.info("Checkpoint file: %s", checkpoint_path)

    stats = {"total_requests": 0, "windows_split": 0, "failed_windows": 0}
    raw_articles = fetch_window_recursive(start_time, end_time, stats, checkpoint)
    articles_by_url = dedupe_by_url(raw_articles)

    _, complete, unresolved = checkpoint.gather_done_articles(start_time, end_time)

    logger.info(
        "Finished: %d API request(s), %d window split(s), %d failed window(s), "
        "%d raw records, %d unique articles",
        stats["total_requests"], stats["windows_split"], stats["failed_windows"],
        len(raw_articles), len(articles_by_url),
    )

    return articles_by_url, start_time, end_time, complete, len(unresolved), stats["failed_windows"]


def export_only(days):
    """Export whatever is already checkpointed to CSV, making zero requests."""
    checkpoint_path = CHECKPOINT_DIR / f"cnbc_headlines_{days}d.checkpoint.json"
    if not checkpoint_path.exists():
        logger.error(
            "No checkpoint found at %s for --days %d; nothing to export. "
            "Run without --export-only first to create one.",
            checkpoint_path, days,
        )
        sys.exit(1)

    now = datetime.now(timezone.utc)
    checkpoint = Checkpoint(checkpoint_path, DOMAIN, days, now - timedelta(days=days), now)
    raw_articles, complete, unresolved = checkpoint.gather_done_articles(
        checkpoint.start_time, checkpoint.end_time,
    )
    articles_by_url = dedupe_by_url(raw_articles)

    logger.info(
        "Export-only mode: %d window(s) unresolved, %d raw records, %d unique articles "
        "loaded from checkpoint (no network requests made)",
        len(unresolved), len(raw_articles), len(articles_by_url),
    )

    return articles_by_url, checkpoint.start_time, checkpoint.end_time, complete, len(unresolved), None


def quick_sample_request(start_time, end_time):
    """Make exactly one GDELT request, retrying a 429 exactly once after a
    short fixed wait, then failing fast. Returns the raw articles list from
    GDELT on success. Exits the process on any failure -- no fabricated data
    is ever produced for a quick sample.
    """
    params = {
        "query": f"domain:{DOMAIN}",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": MAX_RECORDS_PER_REQUEST,
        "sort": "DateDesc",
        "startdatetime": start_time.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_time.strftime("%Y%m%d%H%M%S"),
    }

    for attempt in (1, 2):
        try:
            response = requests.get(GDELT_DOC_API, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.RequestException as exc:
            logger.error("Quick-sample request failed (attempt %d/2): %s", attempt, exc)
            if attempt == 2:
                sys.exit(1)
            time.sleep(QUICK_SAMPLE_RETRY_WAIT_SECONDS)
            continue

        if response.status_code == 429:
            if attempt == 2:
                logger.error(
                    "Quick-sample request rate-limited (429) again after one retry; "
                    "failing fast rather than fabricating data.",
                )
                sys.exit(1)
            wait_seconds = QUICK_SAMPLE_RETRY_WAIT_SECONDS
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    wait_seconds = max(wait_seconds, float(retry_after))
                except ValueError:
                    logger.warning("Ignoring unparseable Retry-After header: %r", retry_after)
            logger.warning(
                "Quick-sample request rate-limited (429); waiting %.0fs then retrying once",
                wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        try:
            response.raise_for_status()
            payload = response.json()
            return payload.get("articles", [])
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.error("Quick-sample request failed (attempt %d/2): %s", attempt, exc)
            if attempt == 2:
                sys.exit(1)
            time.sleep(QUICK_SAMPLE_RETRY_WAIT_SECONDS)
            continue

    logger.error("Quick-sample request did not succeed.")
    sys.exit(1)


def write_quick_sample_csv(articles, start_time, end_time):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = (
        f"cnbc_headlines_quicksample_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}.csv"
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


def write_quick_sample_meta(csv_path, days, start_time, end_time, raw_count, unique_count, hit_cap):
    if hit_cap:
        note = (
            "QUICK-SAMPLE PARTIAL DATASET: exactly one GDELT request was made for this date "
            "range. The response hit the 250-record cap, so additional real articles in this "
            "range almost certainly exist but were not retrieved -- no further requests were "
            "made to keep this a single quick-sample call. No articles were fabricated or "
            "backfilled to fill the gap."
        )
    else:
        note = (
            "QUICK-SAMPLE PARTIAL DATASET: exactly one GDELT request was made for this date "
            "range, with only one retry attempted on a rate limit. The response came in under "
            "the 250-record cap, but because only a single request was made (no adaptive "
            "re-querying or verification), this dataset should still be treated as a partial "
            "sample rather than a confirmed-complete pull. No articles were fabricated or "
            "backfilled."
        )

    meta = {
        "mode": "quick_sample",
        "domain": DOMAIN,
        "requested_days": days,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "raw_articles_returned": raw_count,
        "unique_articles": unique_count,
        "hit_record_cap": hit_cap,
        "complete": False,
        "note": note,
    }
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def run_quick_sample(days):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)

    logger.info(
        "Quick-sample mode: exactly one GDELT request for the most recent %d day(s): %s to %s (UTC)",
        days, start_time.isoformat(), end_time.isoformat(),
    )

    raw_articles = quick_sample_request(start_time, end_time)
    hit_cap = len(raw_articles) >= MAX_RECORDS_PER_REQUEST
    if hit_cap:
        logger.warning(
            "Quick-sample response hit the %d-record cap; real articles beyond this cap "
            "were not retrieved (quick-sample mode makes only one request by design)",
            MAX_RECORDS_PER_REQUEST,
        )

    articles_by_url = dedupe_by_url(raw_articles)
    if not articles_by_url:
        logger.error("Quick-sample returned zero articles for this date range. Exiting without writing a CSV.")
        sys.exit(1)

    articles = list(articles_by_url.values())
    output_path = write_quick_sample_csv(articles, start_time, end_time)
    meta_path = write_quick_sample_meta(
        output_path, days, start_time, end_time, len(raw_articles), len(articles), hit_cap,
    )

    logger.warning(
        "Done. Quick-sample collected %d unique article(s) (%d raw) from ONE request. This is a "
        "PARTIAL dataset by design (%s) -- no articles were fabricated or backfilled.",
        len(articles), len(raw_articles),
        "the 250-record cap was hit" if hit_cap else "only a single request was made",
    )
    logger.info("CSV saved to: %s", output_path)
    logger.info("Metadata saved to: %s", meta_path)

    print(f"UNIQUE_ARTICLES={len(articles)}")
    print(f"CSV_PATH={output_path}")
    print(f"HIT_RECORD_CAP={hit_cap}")
    print("COMPLETE=False")
    return output_path


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


def write_meta(csv_path, start_time, end_time, unique_articles, complete, unresolved_count, failed_count):
    meta = {
        "domain": DOMAIN,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "unique_articles": unique_articles,
        "complete": complete,
        "unresolved_windows": unresolved_count,
        "failed_windows": failed_count,
    }
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def finalize(articles_by_url, start_time, end_time, complete, unresolved_count, failed_count):
    if not articles_by_url:
        logger.error("No articles collected. Exiting without writing a CSV.")
        sys.exit(1)

    articles = list(articles_by_url.values())
    output_path = write_csv(articles, start_time, end_time)
    meta_path = write_meta(
        output_path, start_time, end_time, len(articles), complete, unresolved_count, failed_count,
    )

    if complete:
        logger.info(
            "Done. Collected %d unique articles (dataset COMPLETE for the requested range).",
            len(articles),
        )
    elif failed_count is None:
        logger.warning(
            "Done. Collected %d unique articles, but the dataset is INCOMPLETE: %d window(s) "
            "are not yet resolved in the checkpoint (never attempted or previously gave up), "
            "most likely due to GDELT rate limiting. No missing articles were fabricated or "
            "backfilled.",
            len(articles), unresolved_count,
        )
    else:
        logger.warning(
            "Done. Collected %d unique articles, but the dataset is INCOMPLETE: %d window(s) "
            "unresolved (%d of them failed after retries this run), most likely due to GDELT "
            "rate limiting. No missing articles were fabricated or backfilled.",
            len(articles), unresolved_count, failed_count,
        )

    logger.info("CSV saved to: %s", output_path)
    logger.info("Metadata saved to: %s", meta_path)

    print(f"UNIQUE_ARTICLES={len(articles)}")
    print(f"CSV_PATH={output_path}")
    print(f"COMPLETE={complete}")
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
    parser.add_argument(
        "--export-only",
        action="store_true",
        help=(
            "Export whatever has already been checkpointed for --days to CSV "
            "without making any new GDELT requests. Fails if no checkpoint "
            "exists yet for that --days value."
        ),
    )
    parser.add_argument(
        "--quick-sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Make exactly one GDELT request for the most recent N days and write "
            "whatever real articles it returns to CSV immediately (a 429 is retried "
            "only once before failing fast). Always a partial sample -- see the "
            ".meta.json sidecar. Cannot be combined with --export-only."
        ),
    )
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be a positive integer")
    if args.quick_sample is not None:
        if args.quick_sample <= 0:
            parser.error("--quick-sample must be a positive integer")
        if args.export_only:
            parser.error("--quick-sample cannot be combined with --export-only")
    return args


def main():
    args = parse_args()

    if args.quick_sample is not None:
        run_quick_sample(args.quick_sample)
        return

    if args.export_only:
        articles_by_url, start_time, end_time, complete, unresolved_count, failed_count = export_only(args.days)
    else:
        articles_by_url, start_time, end_time, complete, unresolved_count, failed_count = collect_articles(args.days)

    finalize(articles_by_url, start_time, end_time, complete, unresolved_count, failed_count)


if __name__ == "__main__":
    main()
