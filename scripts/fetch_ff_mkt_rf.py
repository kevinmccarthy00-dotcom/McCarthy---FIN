"""Download Ken French daily Mkt-RF (and RF) factor returns for a date range.

Usage:
    # Explicit date range:
    python scripts/fetch_ff_mkt_rf.py --start 2026-08-21 --end 2026-09-20

    # Or derive the range automatically from a GDELT headlines CSV produced
    # by scripts/gdelt_cnbc_headlines.py (uses its .meta.json sidecar if
    # present, else the min/max seendate in the CSV itself):
    python scripts/fetch_ff_mkt_rf.py --from-gdelt data/gdelt/cnbc_headlines_20260821_20260920.csv

Downloads the "F-F Research Data Factors [Daily]" zip from Ken French's data
library, parses just the daily factors table (the file also contains an
annual table further down, which is skipped), and writes a trimmed CSV with
columns:
    date, mkt_rf, rf
(both in decimal form, e.g. 0.0012 for 0.12%) covering the requested date
range, into data/famafrench/.
"""

import argparse
import csv
import io
import logging
import sys
import time
import zipfile
from datetime import date, datetime
from pathlib import Path

import requests

FF_DAILY_FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "famafrench"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fetch_ff_mkt_rf")


def download_zip():
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Downloading Ken French daily factors (attempt %d/%d)", attempt, MAX_RETRIES)
            response = requests.get(FF_DAILY_FACTORS_URL, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            logger.warning("Download failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    logger.error("Giving up downloading Ken French daily factors after %d attempts: %s", MAX_RETRIES, last_exc)
    sys.exit(1)


def parse_daily_factors(zip_bytes):
    """Parse the daily Mkt-RF/SMB/HML/RF table out of the Ken French zip.

    Returns a list of (date, mkt_rf, rf) tuples with returns in decimal form.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            logger.error("No CSV file found inside the downloaded zip (found: %s)", zf.namelist())
            sys.exit(1)
        raw_text = zf.read(csv_names[0]).decode("utf-8", errors="replace")

    rows = []
    header_seen = False
    for line in raw_text.splitlines():
        if not line.strip():
            if header_seen and rows:
                # A genuinely blank line after we've started collecting daily
                # rows marks the end of the daily table (an annual table
                # follows further down the file).
                break
            continue

        parts = [p.strip() for p in line.split(",")]

        if not header_seen:
            if "mkt-rf" in line.lower():
                header_seen = True
            continue

        date_token = parts[0]
        if not (date_token.isdigit() and len(date_token) == 8):
            # Not a daily YYYYMMDD row -- either the header repeated or the
            # start of the annual table. Stop parsing daily data.
            if rows:
                break
            continue

        try:
            row_date = datetime.strptime(date_token, "%Y%m%d").date()
            mkt_rf = float(parts[1]) / 100.0
            rf = float(parts[4]) / 100.0
        except (IndexError, ValueError) as exc:
            logger.warning("Skipping unparseable row %r: %s", line, exc)
            continue

        rows.append((row_date, mkt_rf, rf))

    if not rows:
        logger.error("Parsed zero daily factor rows; the file format may have changed.")
        sys.exit(1)

    logger.info("Parsed %d daily factor rows (%s to %s)", len(rows), rows[0][0], rows[-1][0])
    return rows


def date_range_from_gdelt(csv_path):
    csv_path = Path(csv_path)
    meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    if meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        start_dt = datetime.fromisoformat(meta["start_time"])
        end_dt = datetime.fromisoformat(meta["end_time"])
        logger.info("Derived date range from %s: %s to %s", meta_path, start_dt.date(), end_dt.date())
        return start_dt.date(), end_dt.date()

    logger.warning("No .meta.json found next to %s; falling back to min/max seendate in the CSV", csv_path)
    seendates = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seendate = row.get("seendate", "")
            if len(seendate) >= 8 and seendate[:8].isdigit():
                seendates.append(datetime.strptime(seendate[:8], "%Y%m%d").date())
    if not seendates:
        logger.error("Could not determine a date range from %s", csv_path)
        sys.exit(1)
    return min(seendates), max(seendates)


def write_csv(rows, start_date, end_date):
    filtered = [r for r in rows if start_date <= r[0] <= end_date]
    if not filtered:
        logger.error(
            "No Ken French rows found in [%s, %s]. Check the date range and try again.",
            start_date, end_date,
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"mkt_rf_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "mkt_rf", "rf"])
        for row_date, mkt_rf, rf in sorted(filtered, key=lambda r: r[0]):
            writer.writerow([row_date.isoformat(), f"{mkt_rf:.6f}", f"{rf:.6f}"])

    return output_path, len(filtered)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Ken French daily Mkt-RF factor returns for a date range.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-gdelt", help="Path to a GDELT headlines CSV; derive the date range from it.")
    group.add_argument(
        "--start", help="Start date (YYYY-MM-DD). Must be used together with --end.",
    )
    parser.add_argument("--end", help="End date (YYYY-MM-DD). Must be used together with --start.")
    args = parser.parse_args()

    if args.start and not args.end:
        parser.error("--start requires --end")
    if args.end and not args.start:
        parser.error("--end requires --start")
    return args


def main():
    args = parse_args()

    if args.from_gdelt:
        start_date, end_date = date_range_from_gdelt(args.from_gdelt)
    else:
        try:
            start_date = date.fromisoformat(args.start)
            end_date = date.fromisoformat(args.end)
        except ValueError as exc:
            logger.error("Invalid date: %s", exc)
            sys.exit(1)

    if start_date > end_date:
        logger.error("--start (%s) is after --end (%s)", start_date, end_date)
        sys.exit(1)

    zip_bytes = download_zip()
    rows = parse_daily_factors(zip_bytes)
    output_path, count = write_csv(rows, start_date, end_date)

    logger.info("Done. Wrote %d daily row(s) for %s to %s.", count, start_date, end_date)
    logger.info("CSV saved to: %s", output_path)

    print(f"ROWS={count}")
    print(f"CSV_PATH={output_path}")


if __name__ == "__main__":
    main()
