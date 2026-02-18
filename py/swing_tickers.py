"""
Universe Builder for swing trading (1–3 weeks) using yfinance.

Input:  text file with one ticker per line (Yahoo format: .TO / .V / .CN)
Output: ranked candidates + a plain ticker list for PFSystem

Fixes applied vs original:
  #1  auto_adjust=True  (prevents split/dividend gaps corrupting SMA/ATR/worst-day)
  #2  above_200d weight raised to +2.0; added as configurable soft-gate bonus
  #3  worst_1d_ret_126 now correct (flows from fix #1)
  #4  Volume trend check added (vol_sma20 > vol_sma50)
  #5  Relative strength vs benchmark (XIU.TO) added — rs_1m, rs_3m
  #6  max_atr_pct_14 default tightened to 0.05 (5%) for realistic swing stops
  #7  sma50_slope normalized by mean price — cross-ticker comparable
  #8  Single-ticker all-NaN column guard added
  #9  df_rejected returned separately for diagnostics
  #10 Stale data check — rejects tickers with last bar > 5 trading days old

Run from IDE:
    from swing_tickers import UniverseBuilderConfig, run_universe_builder
    run_universe_builder(UniverseBuilderConfig(...))
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Thresholds:
    min_price: float = 1.0
    min_avg_dollar_vol_20: float = 1_000_000.0  # price × volume proxy
    max_atr_pct_14: float = 0.05  # 0.08 — too loose for 1-3w swings
    max_one_day_drop_126: float = -0.15  # worst daily return must be >= -15%
    require_above_50d: bool = True  # hard gate
    prefer_above_200d: bool = True  # soft bonus in scoring (not a hard gate)
    max_stale_days: int = 5  # reject tickers with stale last bar


@dataclass
class UniverseBuilderConfig:
    tickers_path: str = "tickers.txt"
    benchmark: str = "XIU.TO"
    out_file_path: str = "tickers_out"
    out_one_line_file_path: str = "tickers_out_one_line"
    out_rejected_file_path: str = "tickers_out_one_line.csv"
    period: str = "1y"
    interval: str = "1d"
    auto_adjust: bool = True
    batch_size: int = 80
    sleep_seconds: float = 1.0
    thresholds: Thresholds = field(default_factory=Thresholds)


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
# TECHNICAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def safe_last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def slope_of_series(s: pd.Series, lookback: int = 10) -> float:
    """
    FIX #7: Normalized slope (slope / mean_price) so values are
    percentage-per-bar and comparable across tickers of different prices.
    Previously returned raw dollar-per-bar, which inflated scores for
    high-priced stocks.
    """
    s = s.dropna()
    if len(s) < lookback:
        return float("nan")
    y = s.iloc[-lookback:].values.astype(float)
    x = np.arange(len(y), dtype=float)
    denom = ((x - x.mean()) ** 2).sum()
    if denom == 0:
        return float("nan")
    raw_slope = float(((x - x.mean()) * (y - y.mean())).sum() / denom)
    # FIX #7: normalize by mean price
    return raw_slope / (y.mean() + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_symbol(df: pd.DataFrame,
                   bench_close: Optional[pd.Series] = None) -> Dict:
    """
    Compute all metrics for a single symbol.
    bench_close: aligned benchmark Close series (optional, for RS calc).
    """
    out: Dict = {}

    # FIX #8: guard against all-NaN columns from single-ticker batch
    needed = {"Open", "High", "Low", "Close", "Volume"}
    for col in needed:
        if col not in df.columns or df[col].dropna().empty:
            out["error"] = f"missing_or_empty_{col}"
            return out

    close = df["Close"].dropna()
    vol = df["Volume"].dropna()

    if close.empty or vol.empty:
        out["error"] = "missing_close_or_volume"
        return out

    last_close = float(close.iloc[-1])
    out["last_close"] = last_close

    # ── FIX #10: Stale data check ────────────────────────────────────────────
    last_date = close.index[-1]
    if hasattr(last_date, "tz_localize"):
        last_date = last_date.tz_localize(None) if last_date.tzinfo else last_date
    days_stale = (pd.Timestamp.today().normalize() - pd.Timestamp(last_date).normalize()).days
    out["days_stale"] = int(days_stale)

    # ── Liquidity ────────────────────────────────────────────────────────────
    dv = (df["Close"] * df["Volume"]).dropna()
    out["avg_vol_20"] = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float("nan")
    out["avg_dollar_vol_20"] = float(dv.rolling(20).mean().iloc[-1]) if len(dv) >= 20 else float("nan")

    # ── Trend ────────────────────────────────────────────────────────────────
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    out["sma50"] = safe_last(sma50)
    out["sma200"] = safe_last(sma200)
    out["above_50d"] = bool(last_close > out["sma50"]) if not np.isnan(out["sma50"]) else False
    out["above_200d"] = bool(last_close > out["sma200"]) if not np.isnan(out["sma200"]) else False
    # FIX #7: normalized slope
    out["sma50_slope"] = slope_of_series(sma50, lookback=10)

    # ── ATR ──────────────────────────────────────────────────────────────────
    atr_series = compute_atr(df, period=14)
    atr_last = safe_last(atr_series)
    out["atr_14"] = atr_last
    out["atr_pct_14"] = float(atr_last / last_close) if (not np.isnan(atr_last) and last_close > 0) else float("nan")

    # ── Worst 1-day return (last ~6 months) ──────────────────────────────────
    # FIX #1+#3: auto_adjust=True means no artificial split gaps here
    ret = close.pct_change()
    ret_126 = ret.dropna().iloc[-126:] if len(ret.dropna()) >= 126 else ret.dropna()
    out["worst_1d_ret_126"] = float(ret_126.min()) if len(ret_126) else float("nan")

    # ── FIX #4: Volume trend — rising volume = accumulation ──────────────────
    vol_sma20 = vol.rolling(20).mean()
    vol_sma50 = vol.rolling(50).mean()
    v20 = safe_last(vol_sma20)
    v50 = safe_last(vol_sma50)
    out["vol_trend_up"] = bool(v20 > v50) if (not np.isnan(v20) and not np.isnan(v50)) else False
    out["vol_ratio_20_50"] = float(v20 / v50) if (not np.isnan(v20) and not np.isnan(v50) and v50 > 0) else float("nan")

    # ── FIX #5: Relative Strength vs benchmark ───────────────────────────────
    out["rs_1m"] = float("nan")
    out["rs_3m"] = float("nan")
    if bench_close is not None and not bench_close.empty:
        try:
            # Align on common dates
            aligned_stock, aligned_bench = close.align(bench_close, join="inner")
            for label, bars in [("rs_1m", 21), ("rs_3m", 63)]:
                if len(aligned_stock) >= bars and len(aligned_bench) >= bars:
                    s_ret = aligned_stock.iloc[-1] / aligned_stock.iloc[-bars] - 1
                    b_ret = aligned_bench.iloc[-1] / aligned_bench.iloc[-bars] - 1
                    out[label] = float(s_ret - b_ret)
        except Exception:
            pass

    return out


# ─────────────────────────────────────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def pass_filters(row: Dict, th: Thresholds) -> Tuple[bool, List[str]]:
    reasons = []

    price = row.get("last_close", np.nan)
    if np.isnan(price) or price < th.min_price:
        reasons.append("price_too_low")

    adv = row.get("avg_dollar_vol_20", np.nan)
    if np.isnan(adv) or adv < th.min_avg_dollar_vol_20:
        reasons.append("low_dollar_volume")

    atr_pct = row.get("atr_pct_14", np.nan)
    if np.isnan(atr_pct) or atr_pct > th.max_atr_pct_14:
        reasons.append("too_volatile_atr")

    worst = row.get("worst_1d_ret_126", np.nan)
    if np.isnan(worst) or worst < th.max_one_day_drop_126:
        reasons.append("large_gap_risk")

    if th.require_above_50d and not row.get("above_50d", False):
        reasons.append("below_50d")

    # FIX #10: stale data
    stale = row.get("days_stale", 999)
    if stale > th.max_stale_days:
        reasons.append(f"stale_data_{stale}d")

    return (len(reasons) == 0), reasons


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_row(row: Dict) -> float:
    """
    Higher is better. All fixes applied:
      - FIX #2:  above_200d weight raised to +2.0 (was +0.8, drowned by liquidity)
      - FIX #4:  volume trend bonus added
      - FIX #5:  relative strength (RS) vs benchmark added as top-weighted factor
      - FIX #7:  sma50_slope is now normalized — contribution is meaningful
    """
    score = 0.0

    # ── Liquidity (log scale, ~1–3 pts) ─────────────────────────────────────
    adv = row.get("avg_dollar_vol_20", np.nan)
    if not np.isnan(adv) and adv > 0:
        score += min(3.0, math.log10(adv) - 5.0)  # capped at 3 (was 5 — dominated score)

    # ── Trend alignment ──────────────────────────────────────────────────────
    if row.get("above_50d", False):
        score += 1.0

    # FIX #2: meaningful weight for 200d — was +0.8, easily drowned by liquidity
    if row.get("above_200d", False):
        score += 2.0

    # ── Slope quality (FIX #7: normalized, now comparable) ──────────────────
    sma50_slope = row.get("sma50_slope", np.nan)
    if not np.isnan(sma50_slope) and sma50_slope > 0:
        score += min(1.5, sma50_slope * 500)  # normalized slope → contribution capped at 1.5

    # ── FIX #5: Relative strength vs benchmark (highest predictive value) ───
    rs_1m = row.get("rs_1m", np.nan)
    rs_3m = row.get("rs_3m", np.nan)
    if not np.isnan(rs_1m):
        # +/- 10% RS maps to +/- 2.0 pts; capped
        score += max(-2.0, min(2.0, rs_1m * 20))
    if not np.isnan(rs_3m):
        score += max(-1.5, min(1.5, rs_3m * 10))

    # ── FIX #4: Volume trend bonus ───────────────────────────────────────────
    if row.get("vol_trend_up", False):
        score += 0.8
    vol_ratio = row.get("vol_ratio_20_50", np.nan)
    if not np.isnan(vol_ratio) and vol_ratio > 1.1:
        score += min(0.5, (vol_ratio - 1.0) * 2.0)  # extra for strong accumulation

    # ── Volatility penalty (FIX #6: tighter ATR ceiling means fewer extreme cases) ──
    atr_pct = row.get("atr_pct_14", np.nan)
    if not np.isnan(atr_pct):
        score -= min(2.0, atr_pct * 15.0)

    # ── Worst-day penalty ────────────────────────────────────────────────────
    worst = row.get("worst_1d_ret_126", np.nan)
    if not np.isnan(worst):
        score -= min(2.5, abs(worst) * 10.0)

    return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_history_batch(tickers: List[str], period: str,
                        interval: str, auto_adjust: bool) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,  # FIX #1: must be True
        group_by="ticker",
        threads=True,
        progress=False,
    )


def fetch_benchmark(ticker: str, period: str,
                    interval: str, auto_adjust: bool) -> pd.Series:
    """Download benchmark close series for RS calculation (FIX #5)."""
    try:
        raw = yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=auto_adjust, progress=False
        )
        close = raw["Close"].dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.squeeze()
    except Exception as e:
        print(f"  Warning: benchmark {ticker} failed ({e}) — RS scores will be NaN")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_universe_builder(cfg: UniverseBuilderConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_tradable, df_rejected) — FIX #9: rejected tickers separated.
    df_tradable  : passed all filters, sorted by score desc
    df_rejected  : failed at least one filter, includes reject_reasons column
    """
    tickers = read_tickers(cfg.tickers_path)
    print(f"Loaded {len(tickers)} tickers from {cfg.tickers_path}")

    # FIX #5: fetch benchmark once upfront
    print(f"Fetching benchmark {cfg.benchmark} ...")
    bench_close = fetch_benchmark(
        cfg.benchmark, cfg.period, cfg.interval, cfg.auto_adjust
    )
    if not bench_close.empty:
        print(f"  Benchmark OK — {len(bench_close)} bars")

    rows: List[Dict] = []
    batches = chunked(tickers, cfg.batch_size)

    for i, batch in enumerate(batches, 1):
        print(f"Fetching batch {i}/{len(batches)} ({len(batch)} tickers)...")
        try:
            big = fetch_history_batch(
                batch, cfg.period, cfg.interval, cfg.auto_adjust
            )
        except Exception as e:
            print(f"  Batch fetch error: {e}")
            time.sleep(cfg.sleep_seconds)
            continue

        if isinstance(big.columns, pd.MultiIndex):
            for sym in batch:
                if sym not in big.columns.get_level_values(0):
                    rows.append({"symbol": sym, "error": "no_data",
                                 "tradable": False, "reject_reasons": "no_data"})
                    continue
                sub = big[sym].dropna(how="all")
                if sub.empty:
                    rows.append({"symbol": sym, "error": "no_data",
                                 "tradable": False, "reject_reasons": "no_data"})
                    continue
                # FIX #1: normalize index timezone for alignment with benchmark
                sub.index = pd.to_datetime(sub.index).tz_localize(None)
                _process_symbol(sym, sub, bench_close, cfg, rows)

        else:
            # Single-ticker path
            sym = batch[0]
            sub = big.dropna(how="all")
            needed = {"Open", "High", "Low", "Close", "Volume"}
            if sub.empty or not needed.issubset(set(sub.columns)):
                rows.append({"symbol": sym, "error": "missing_ohlcv",
                             "tradable": False, "reject_reasons": "missing_ohlcv"})
            else:
                # FIX #8: check columns are not all-NaN
                if sub["Close"].dropna().empty:
                    rows.append({"symbol": sym, "error": "all_nan_close",
                                 "tradable": False, "reject_reasons": "all_nan_close"})
                else:
                    sub.index = pd.to_datetime(sub.index).tz_localize(None)
                    _process_symbol(sym, sub, bench_close, cfg, rows)

        time.sleep(cfg.sleep_seconds)

    df = pd.DataFrame(rows)

    # Ensure all expected columns exist
    all_cols = [
        "symbol", "tradable", "score",
        "last_close", "avg_vol_20", "avg_dollar_vol_20",
        "atr_pct_14", "worst_1d_ret_126",
        "above_50d", "above_200d", "sma50_slope",
        "vol_trend_up", "vol_ratio_20_50",  # FIX #4
        "rs_1m", "rs_3m",  # FIX #5
        "days_stale",  # FIX #10
        "reject_reasons", "error",
    ]
    for c in all_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[all_cols]

    # FIX #9: split into tradable and rejected DataFrames
    df_tradable = (
        df[df["tradable"] == True]
        .copy()
        .sort_values(["score", "avg_dollar_vol_20"], ascending=[False, False])
    )
    df_rejected = (
        df[df["tradable"] != True]
        .copy()
        .sort_values("symbol")
    )

    # ── Outputs ──────────────────────────────────────────────────────────────
    out_txt = f"{cfg.out_file_path}"
    out_one_line = f"{cfg.out_one_line_file_path}"
    out_rejected = f"{cfg.out_rejected_file_path}"

    with open(out_txt, "w", encoding="utf-8") as f:
        for sym in df_tradable["symbol"].tolist():
            f.write(sym + "\n")

    with open(out_one_line, "w", encoding="utf-8") as f:
        f.write(",".join(df_tradable["symbol"].tolist()) + "\n")

    df_rejected.to_csv(out_rejected, index=False)

    print("\nWrote:")
    print(f"  {out_txt}         (tickers for scanner)")
    print(f"  {out_one_line}    (comma-separated)")
    print(f"  {out_rejected}    (rejected + reasons, for diagnostics)")

    print(f"\nResults: {len(df_tradable)} tradable / {len(df_rejected)} rejected "
          f"/ {len(df)} total")

    print("\nTop 20 tradable:")
    display_cols = [
        "symbol", "score", "last_close", "avg_dollar_vol_20",
        "atr_pct_14", "above_50d", "above_200d",
        "rs_1m", "rs_3m", "vol_trend_up", "sma50_slope",
    ]
    print(df_tradable[display_cols].head(20).to_string(index=False))

    # Rejection breakdown
    if not df_rejected.empty:
        from collections import Counter
        all_reasons = []
        for r in df_rejected["reject_reasons"].dropna():
            all_reasons.extend(str(r).split(","))
        reason_counts = Counter(all_reasons)
        print("\nRejection reason summary:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason:<30} {count}")

    return df_tradable, df_rejected


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _process_symbol(sym: str, sub: pd.DataFrame,
                    bench_close: pd.Series,
                    cfg: UniverseBuilderConfig,
                    rows: List[Dict]) -> None:
    """Analyze, filter, score one symbol and append to rows list."""
    metrics = analyze_symbol(sub, bench_close=bench_close)
    metrics["symbol"] = sym
    ok, reasons = pass_filters(metrics, cfg.thresholds)
    metrics["tradable"] = ok
    metrics["reject_reasons"] = ",".join(reasons) if reasons else ""
    metrics["score"] = score_row(metrics) if ok else float("nan")
    rows.append(metrics)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = UniverseBuilderConfig(
        tickers_path="../data/can_tickers",
        benchmark="XIU.TO",
        out_file_path="../out/can_tickers_swing",
        out_one_line_file_path="../out/can_tickers_swing_one_line",
        out_rejected_file_path="../out/can_tickers_rejected.csv",
        period="1y",
        interval="1d",
        auto_adjust=True,
        batch_size=80,
        sleep_seconds=1.0,
        thresholds=Thresholds(
            min_price=1.0,
            min_avg_dollar_vol_20=1_000_000.0,
            max_atr_pct_14=0.05,
            max_one_day_drop_126=-0.15,
            require_above_50d=True,
            prefer_above_200d=True,
            max_stale_days=5,
        ),
    )

    df_tradable, df_rejected = run_universe_builder(config)
