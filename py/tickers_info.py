"""
Ticker Info Builder — fetches metadata for a list of tickers and writes a CSV.

Input:  text file with one ticker per line (Yahoo format: .TO / .V / .CN / ...)
Output: CSV file with columns:
    ticker, exchange, company_name, aliases, sector,
    average_volume, market_cap, last_price, spread_estimate

spread_estimate: ask - bid when both are available and non-zero;
                 falls back to 1 % of last_price as a conservative proxy.

Run:
    python py/tickers_info.py --input data/can_tickers --out out/can_tickers_info.csv

Run from IDE:
    from tickers_info import TickerInfoConfig, run_ticker_info_builder
    run_ticker_info_builder(TickerInfoConfig(...))
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TickerInfoConfig:
    input_path: str = "data/can_tickers"
    out_file_path: str = "out/can_tickers_info.csv"
    batch_size: int = 80
    sleep_seconds: float = 1.0
    # Fallback spread when bid/ask unavailable: fraction of last price
    fallback_spread_pct: float = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# I/O HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def read_tickers(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip().upper() for ln in f.readlines()]
    tickers = [t for t in lines if t and not t.startswith("#")]
    seen, out = set(), []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


# ─────────────────────────────────────────────────────────────────────────────
# FIELD EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def _alias(ticker: str) -> str:
    """Strip exchange suffix to get a plain ticker alias (e.g. RY.TO → RY)."""
    return ticker.split(".")[0]


def _safe_str(info: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        v = info.get(k)
        if v and str(v).strip():
            return str(v).strip()
    return default


def _safe_float(info: dict, *keys: str, default: float = float("nan")) -> float:
    for k in keys:
        v = info.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def _spread_estimate(info: dict, last_price: float, fallback_pct: float) -> float:
    """
    Best estimate of bid-ask spread in price units.

    Priority:
      1. ask - bid  (when both are present, positive, and ask > bid)
      2. fallback_pct × last_price
    """
    bid = _safe_float(info, "bid")
    ask = _safe_float(info, "ask")

    if (
            not np.isnan(bid) and not np.isnan(ask)
            and bid > 0 and ask > 0
            and ask > bid
    ):
        return round(ask - bid, 6)

    if not np.isnan(last_price) and last_price > 0:
        return round(last_price * fallback_pct, 6)

    return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-TICKER FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ticker_info(ticker: str, fallback_spread_pct: float) -> dict:
    """
    Fetch yfinance .info for one ticker and map to output schema.
    Returns a dict with all CSV columns populated (NaN / empty string on failure).
    """
    row = {
        "ticker": ticker,
        "exchange": "",
        "company_name": "",
        "aliases": _alias(ticker),
        "sector": "",
        "average_volume": float("nan"),
        "market_cap": float("nan"),
        "last_price": float("nan"),
        "spread_estimate": float("nan"),
    }

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        row["exchange"] = _safe_str(info, "exchange", "market")
        row["company_name"] = _safe_str(info, "longName", "shortName", "displayName")
        row["sector"] = _safe_str(info, "sector", "industry")

        row["average_volume"] = _safe_float(
            info,
            "averageVolume",
            "averageDailyVolume10Day",
            "averageDailyVolume3Month",
        )
        row["market_cap"] = _safe_float(info, "marketCap")

        last_price = _safe_float(
            info,
            "currentPrice",
            "regularMarketPrice",
            "previousClose",
            "navPrice",
        )
        row["last_price"] = last_price
        row["spread_estimate"] = _spread_estimate(info, last_price, fallback_spread_pct)

    except Exception as e:
        print(f"  Warning: {ticker} — info fetch failed ({e})")

    return row


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_ticker_info_builder(cfg: TickerInfoConfig) -> pd.DataFrame:
    """
    Fetches metadata for every ticker in cfg.input_path and writes cfg.out_file_path.

    Returns the resulting DataFrame.
    """
    tickers = read_tickers(cfg.input_path)
    print(f"Loaded {len(tickers)} tickers from {cfg.input_path}")

    rows: List[dict] = []
    batches = chunked(tickers, cfg.batch_size)

    for i, batch in enumerate(batches, 1):
        print(f"Processing batch {i}/{len(batches)} ({len(batch)} tickers)...")
        for ticker in batch:
            row = fetch_ticker_info(ticker, cfg.fallback_spread_pct)
            rows.append(row)
        time.sleep(cfg.sleep_seconds)

    df = pd.DataFrame(rows, columns=[
        "ticker",
        "exchange",
        "company_name",
        "aliases",
        "sector",
        "average_volume",
        "market_cap",
        "last_price",
        "spread_estimate",
    ])

    # ── Write output ─────────────────────────────────────────────────────────
    out_path = Path(cfg.out_file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    filled = df["company_name"].ne("").sum()
    print(f"\nWrote {out_path}  ({len(df)} rows, {filled} with company name resolved)")
    print("\nSample output:")
    print(df.head(5).to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch ticker metadata and write to CSV"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file — one ticker per line",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/can_tickers_info.csv",
        help="Path to output CSV file (default: out/can_tickers_info.csv)",
    )
    parser.add_argument(
        "--fallback-spread-pct",
        type=float,
        default=0.01,
        help="Fallback spread as a fraction of last price when bid/ask unavailable "
             "(default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=80,
        help="Tickers per processing batch (default: 80)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between batches (default: 1.0)",
    )

    args = parser.parse_args()

    config = TickerInfoConfig(
        input_path=args.input,
        out_file_path=args.out,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep,
        fallback_spread_pct=args.fallback_spread_pct,
    )

    run_ticker_info_builder(config)
