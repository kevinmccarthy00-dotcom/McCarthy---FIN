"""Deterministic external market data retrieval via yfinance.

No LLM involvement here. This module only fetches and lightly computes
(returns from real historical prices) — it never estimates or guesses a
value. Anything that can't be retrieved is marked "unavailable" with a
reason, and is never backfilled later in the chain.
"""

from datetime import datetime, timezone

import yfinance as yf

# Fixed representative basket of ETFs covering major asset classes.
ETF_BASKET = {
    "VTI": "U.S. equities",
    "VXUS": "International equities",
    "BND": "U.S. investment-grade bonds",
    "BNDX": "International bonds",
    "VNQ": "Real estate (REITs)",
    "SHY": "Short-term Treasuries",
    "IEF": "Intermediate Treasuries",
}

# Treasury yield proxies available as Yahoo Finance indices.
# Yahoo quotes these indices at 10x the actual yield percentage
# (e.g. a quoted value of 42.5 represents an approximate yield of 4.25%).
# This is documented explicitly in the output so later stages/humans can
# audit the conversion rather than treating it as an exact par yield.
TREASURY_YIELD_PROXIES = {
    "^IRX": "13-week T-bill yield proxy",
    "^FVX": "5-year Treasury yield proxy",
    "^TNX": "10-year Treasury yield proxy",
    "^TYX": "30-year Treasury yield proxy",
}


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _trailing_return_pct(close_series, years):
    """Cumulative % return over the trailing N years using adjusted close.

    Returns None if there isn't enough history to compute it.
    """
    if close_series.empty:
        return None
    end_date = close_series.index[-1]
    target_date = end_date - _year_offset(years)
    history_before_target = close_series[close_series.index <= target_date]
    if history_before_target.empty:
        return None
    start_price = float(history_before_target.iloc[-1])
    end_price = float(close_series.iloc[-1])
    if start_price == 0:
        return None
    return round((end_price / start_price - 1) * 100, 2)


def _year_offset(years):
    import pandas as pd

    return pd.DateOffset(years=years)


def _fetch_expense_ratio(ticker_obj):
    try:
        info = ticker_obj.get_info()
    except Exception as exc:  # noqa: BLE001 - we want to record any failure
        return None, f"expense ratio lookup failed: {exc}"

    raw = info.get("annualReportExpenseRatio") if info else None
    if raw is None:
        return None, "expense ratio not provided by yfinance for this ticker"
    return round(raw * 100, 4), None


def _fetch_etf(symbol, label):
    entry = {
        "ticker": symbol,
        "label": label,
        "status": "unavailable",
        "as_of_date": None,
        "last_adjusted_close": None,
        "trailing_return_1y_pct": None,
        "trailing_return_3y_pct": None,
        "expense_ratio_pct": None,
        "unavailable_fields": [],
        "error": None,
    }

    try:
        ticker_obj = yf.Ticker(symbol)
        hist = ticker_obj.history(period="3y", interval="1d", auto_adjust=True)
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"price history retrieval failed: {exc}"
        entry["unavailable_fields"] = [
            "last_adjusted_close",
            "trailing_return_1y_pct",
            "trailing_return_3y_pct",
        ]
        return entry

    if hist is None or hist.empty:
        entry["error"] = "no price history returned by yfinance"
        entry["unavailable_fields"] = [
            "last_adjusted_close",
            "trailing_return_1y_pct",
            "trailing_return_3y_pct",
        ]
        return entry

    close = hist["Close"].dropna()
    entry["status"] = "ok"
    entry["as_of_date"] = close.index[-1].strftime("%Y-%m-%d")
    entry["last_adjusted_close"] = round(float(close.iloc[-1]), 4)

    return_1y = _trailing_return_pct(close, 1)
    return_3y = _trailing_return_pct(close, 3)
    if return_1y is None:
        entry["unavailable_fields"].append("trailing_return_1y_pct")
    else:
        entry["trailing_return_1y_pct"] = return_1y
    if return_3y is None:
        entry["unavailable_fields"].append("trailing_return_3y_pct")
    else:
        entry["trailing_return_3y_pct"] = return_3y

    expense_ratio, expense_error = _fetch_expense_ratio(ticker_obj)
    if expense_ratio is None:
        entry["unavailable_fields"].append("expense_ratio_pct")
        if expense_error:
            entry["error"] = (
                f"{entry['error']}; {expense_error}" if entry["error"] else expense_error
            )
    else:
        entry["expense_ratio_pct"] = expense_ratio

    return entry


def _fetch_yield_proxy(symbol, label):
    entry = {
        "ticker": symbol,
        "label": label,
        "status": "unavailable",
        "as_of_date": None,
        "raw_quote": None,
        "approx_yield_pct": None,
        "conversion_note": (
            "Yahoo Finance quotes this index at 10x the approximate yield "
            "percentage; approx_yield_pct = raw_quote / 10. This is a proxy, "
            "not an exact par yield."
        ),
        "error": None,
    }

    try:
        ticker_obj = yf.Ticker(symbol)
        hist = ticker_obj.history(period="5d", interval="1d", auto_adjust=False)
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"yield proxy retrieval failed: {exc}"
        return entry

    if hist is None or hist.empty:
        entry["error"] = "no data returned by yfinance for this yield proxy"
        return entry

    close = hist["Close"].dropna()
    if close.empty:
        entry["error"] = "no closing quote available for this yield proxy"
        return entry

    entry["status"] = "ok"
    entry["as_of_date"] = close.index[-1].strftime("%Y-%m-%d")
    raw_quote = float(close.iloc[-1])
    entry["raw_quote"] = round(raw_quote, 4)
    entry["approx_yield_pct"] = round(raw_quote / 10, 4)

    return entry


def get_market_data():
    """Retrieve the full fixed basket of market data via yfinance.

    Every ticker is fetched independently; a failure on one ticker never
    prevents the others from being retrieved, and is recorded explicitly
    rather than silently dropped.
    """
    retrieved_at = _now_iso()

    etfs = {symbol: _fetch_etf(symbol, label) for symbol, label in ETF_BASKET.items()}
    treasury_yield_proxies = {
        symbol: _fetch_yield_proxy(symbol, label)
        for symbol, label in TREASURY_YIELD_PROXIES.items()
    }

    return {
        "retrieved_at": retrieved_at,
        "source": "yfinance",
        "return_calculation_method": (
            "Trailing returns are cumulative percentage changes computed from "
            "yfinance auto-adjusted daily close prices (dividends/splits "
            "adjusted), comparing the most recent close to the close closest "
            "to (but not after) N years earlier in the retrieved history."
        ),
        "etfs": etfs,
        "treasury_yield_proxies": treasury_yield_proxies,
    }
