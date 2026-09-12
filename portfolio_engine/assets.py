"""Baseline asset-class universe and fallback capital-market assumptions.

The fallback numbers below are rough, long-run, illustrative assumptions
(not a live forecast) used only when historical ETF data cannot be
retrieved. They exist so the engine keeps working offline / in class demos.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetClass:
    key: str
    name: str
    ticker: str
    fallback_return: float
    fallback_volatility: float


ASSET_CLASSES: list[AssetClass] = [
    AssetClass("US_STOCKS", "US Stocks", "VTI", 0.075, 0.16),
    AssetClass("INTL_DEV", "International Developed Stocks", "VEA", 0.070, 0.17),
    AssetClass("EM", "Emerging Markets Stocks", "VWO", 0.085, 0.22),
    AssetClass("US_BONDS", "US Bonds", "BND", 0.040, 0.06),
    AssetClass("TIPS", "TIPS", "SCHP", 0.035, 0.06),
    AssetClass("REITS", "REITs", "VNQ", 0.070, 0.19),
]

ASSET_KEYS: list[str] = [a.key for a in ASSET_CLASSES]
TICKER_BY_KEY: dict[str, str] = {a.key: a.ticker for a in ASSET_CLASSES}
KEY_BY_TICKER: dict[str, str] = {a.ticker: a.key for a in ASSET_CLASSES}

# Equity vs. fixed-income grouping, used by the lifecycle heuristic to
# translate a single "equity %" glide-path number into per-asset weights.
EQUITY_KEYS: list[str] = ["US_STOCKS", "INTL_DEV", "EM", "REITS"]
FIXED_INCOME_KEYS: list[str] = ["US_BONDS", "TIPS"]

# How the equity sleeve and the fixed-income sleeve are split internally.
# These are fixed baseline proportions (not optimized) that must each sum to 1.
EQUITY_SLEEVE_SPLIT: dict[str, float] = {
    "US_STOCKS": 0.60,
    "INTL_DEV": 0.22,
    "EM": 0.10,
    "REITS": 0.08,
}
FIXED_INCOME_SLEEVE_SPLIT: dict[str, float] = {
    "US_BONDS": 0.75,
    "TIPS": 0.25,
}

# Fallback correlation matrix (order matches ASSET_KEYS). Illustrative,
# long-run, and checked to be positive semi-definite.
FALLBACK_CORRELATION: list[list[float]] = [
    # US_STK  INTL   EM      BONDS   TIPS   REITS
    [1.00, 0.85, 0.75, -0.10, -0.05, 0.65],
    [0.85, 1.00, 0.80, -0.05, 0.00, 0.60],
    [0.75, 0.80, 1.00, -0.05, 0.00, 0.55],
    [-0.10, -0.05, -0.05, 1.00, 0.65, 0.10],
    [-0.05, 0.00, 0.00, 0.65, 1.00, 0.15],
    [0.65, 0.60, 0.55, 0.10, 0.15, 1.00],
]
