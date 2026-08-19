"""
Tests for swing_universe.py.

Covers every public function, all filter branches, scoring components, output
formatting, and the full pipeline with mocked network calls. No live network
calls are made.

Run from repo root:
    pytest py/test_swing_universe.py -v

Findings from first run (4 failures, now fixed in tests):
  1. ATR gap tests: gap day falls out of the rolling-14 window after 16+ bars
     of steady price, so the last bar's ATR returns to H-L spread (2.0).
     Fix: keep the series short enough that the gap day stays inside the window.
  2. worst_1d_ret_126 is NOT guaranteed negative: for a monotonically rising
     series every pct_change is positive, so min is also positive.
  3. pass_filters produces 6 reasons (not 5) when all conditions are violated:
     price, volume, ATR, worst-day, SMA50, and stale-data are independent.
"""

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from swing_universe import (
    Thresholds,
    UniverseBuilderConfig,
    _process_symbol,
    analyze_symbol,
    atr_band_bonus,
    chunked,
    compute_atr,
    fetch_benchmark,
    fetch_history_batch,
    is_excluded_instrument,
    pass_filters,
    read_tickers,
    run_universe_builder,
    safe_last,
    score_row,
    slope_of_series,
)


# ─── Synthetic data factories ─────────────────────────────────────────────────


def _ohlcv(
    n: int = 252,
    price: float = 20.0,
    volume: int = 100_000,
    atr_pct: float = 0.02,
    drift: float = 0.001,
    end_date=None,
    tz=None,
) -> pd.DataFrame:
    """
    Produce a tz-naive (or tz-aware) OHLCV DataFrame whose default parameters
    pass every filter in Thresholds():
      - price $20, 100k shares → $2M dollar vol  (> $1M floor)
      - ATR ≈ 2%               (< 5% ceiling)
      - drift 0.1%/bar         → above SMA50 after warmup, worst-day ≈ 0%
      - last bar = today        → days_stale = 0
    """
    if end_date is None:
        end_date = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end_date, periods=n)
    if tz is not None:
        dates = dates.tz_localize(tz)
    closes = np.array([price * (1 + drift) ** i for i in range(n)], dtype=float)
    spread = closes * atr_pct / 2
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes + spread,
            "Low": closes - spread,
            "Close": closes,
            "Volume": np.full(n, float(volume)),
        },
        index=dates,
    )


def _bench(n: int = 252, price: float = 30.0, drift: float = 0.0005) -> pd.Series:
    """Synthetic tz-naive benchmark Close series."""
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    closes = np.array([price * (1 + drift) ** i for i in range(n)], dtype=float)
    return pd.Series(closes, index=dates)


def _multiindex_df(tickers, n=252, price=20.0, volume=100_000, atr_pct=0.02, drift=0.001):
    """yfinance-style MultiIndex DataFrame (level-0 = ticker, level-1 = field)."""
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    data = {}
    for t in tickers:
        closes = np.array([price * (1 + drift) ** i for i in range(n)], dtype=float)
        spread = closes * atr_pct / 2
        data.update(
            {
                (t, "Open"): closes,
                (t, "High"): closes + spread,
                (t, "Low"): closes - spread,
                (t, "Close"): closes,
                (t, "Volume"): np.full(n, float(volume)),
            }
        )
    return pd.DataFrame(data, index=dates)


def _cfg(tmp_path, ticker_lines, **overrides):
    f = tmp_path / "tickers.txt"
    f.write_text(ticker_lines)
    defaults = dict(
        tickers_path=str(f),
        benchmark="XIU.TO",
        out_file_path=str(tmp_path / "universe"),
        out_one_line_file_path=str(tmp_path / "one_line"),
        out_rejected_file_path=str(tmp_path / "rejected.csv"),
        period="1y",
        interval="1d",
        auto_adjust=True,
        batch_size=10,
        sleep_seconds=0.0,
        thresholds=Thresholds(),
    )
    defaults.update(overrides)
    return UniverseBuilderConfig(**defaults)


# ─── read_tickers ─────────────────────────────────────────────────────────────


def test_read_tickers_basic(tmp_path):
    (tmp_path / "t.txt").write_text("RY.TO\nTD.TO\nBNS.TO\n")
    assert read_tickers(str(tmp_path / "t.txt")) == ["RY.TO", "TD.TO", "BNS.TO"]


def test_read_tickers_dedup_preserves_order(tmp_path):
    (tmp_path / "t.txt").write_text("RY.TO\nTD.TO\nRY.TO\nBNS.TO\nTD.TO\n")
    assert read_tickers(str(tmp_path / "t.txt")) == ["RY.TO", "TD.TO", "BNS.TO"]


def test_read_tickers_strips_blanks_and_comments(tmp_path):
    (tmp_path / "t.txt").write_text("# header\n\nRY.TO\n\n# skip\nTD.TO\n")
    assert read_tickers(str(tmp_path / "t.txt")) == ["RY.TO", "TD.TO"]


def test_read_tickers_uppercases(tmp_path):
    (tmp_path / "t.txt").write_text("ry.to\ntd.to\n")
    assert read_tickers(str(tmp_path / "t.txt")) == ["RY.TO", "TD.TO"]


def test_read_tickers_strips_whitespace(tmp_path):
    (tmp_path / "t.txt").write_text("  RY.TO  \n\tTD.TO\t\n")
    assert read_tickers(str(tmp_path / "t.txt")) == ["RY.TO", "TD.TO"]


def test_read_tickers_strips_utf8_bom(tmp_path):
    """A BOM must not become part of the first ticker."""
    (tmp_path / "t.txt").write_bytes("﻿RY.TO\nTD.TO\n".encode("utf-8"))
    assert read_tickers(str(tmp_path / "t.txt")) == ["RY.TO", "TD.TO"]


def test_read_tickers_handles_crlf_line_endings(tmp_path):
    (tmp_path / "t.txt").write_bytes(b"RY.TO\r\nTD.TO\r\n")
    assert read_tickers(str(tmp_path / "t.txt")) == ["RY.TO", "TD.TO"]


def test_read_tickers_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_tickers(str(tmp_path / "does_not_exist.txt"))


def test_read_tickers_empty_file_returns_empty_list(tmp_path):
    (tmp_path / "t.txt").write_text("")
    assert read_tickers(str(tmp_path / "t.txt")) == []


# ─── chunked ─────────────────────────────────────────────────────────────────


def test_chunked_empty():
    assert chunked([], 10) == []


def test_chunked_smaller_than_size():
    assert chunked(["A", "B", "C"], 10) == [["A", "B", "C"]]


def test_chunked_exact_size():
    assert chunked(["A", "B"], 2) == [["A", "B"]]


def test_chunked_splits_evenly():
    assert chunked(list("ABCDEF"), 2) == [["A", "B"], ["C", "D"], ["E", "F"]]


def test_chunked_last_chunk_smaller():
    result = chunked(list("ABCDE"), 2)
    assert result == [["A", "B"], ["C", "D"], ["E"]]


def test_chunked_preserves_all_items():
    items = list("ABCDEFGHIJ")
    flat = [x for chunk in chunked(items, 3) for x in chunk]
    assert flat == items


def test_chunked_size_zero_raises():
    with pytest.raises(ValueError):
        chunked(["A"], 0)


# ─── compute_atr ─────────────────────────────────────────────────────────────


def test_compute_atr_flat_price_equals_spread():
    """With no gap risk, ATR = simple H-L spread."""
    df = pd.DataFrame(
        {"High": [21.0] * 30, "Low": [19.0] * 30, "Close": [20.0] * 30}
    )
    atr = compute_atr(df, period=14)
    assert not np.isnan(atr.iloc[-1])
    assert pytest.approx(float(atr.iloc[-1]), abs=1e-9) == 2.0


def test_compute_atr_all_nan_before_period():
    """Fewer bars than period → entire series is NaN."""
    df = pd.DataFrame(
        {"High": [21.0] * 10, "Low": [19.0] * 10, "Close": [20.0] * 10}
    )
    assert compute_atr(df, period=14).dropna().empty


def test_compute_atr_captures_gap_up():
    """
    True Range on a gap-up day must exceed H-L spread.
    Gap is at index 2; series is 16 bars so the last rolling-14 window (bars
    2-15) still includes the gap day, keeping ATR elevated above 2.0.
    """
    closes = [20.0, 20.0] + [30.0] * 14
    highs  = [21.0, 21.0] + [31.0] * 14
    lows   = [19.0, 19.0] + [29.0] * 14
    df = pd.DataFrame({"High": highs, "Low": lows, "Close": closes})
    atr = compute_atr(df, period=14)
    # Last window covers bars 2–15; TR[2]=11 so ATR = (11 + 13*2)/14 ≈ 2.64
    assert float(atr.dropna().iloc[-1]) > 2.0


def test_compute_atr_captures_gap_down():
    """Gap-down: |Low - prev_Close| dominates H-L on the gap day."""
    closes = [20.0, 20.0] + [10.0] * 14
    highs  = [21.0, 21.0] + [11.0] * 14
    lows   = [19.0, 19.0] + [ 9.0] * 14
    df = pd.DataFrame({"High": highs, "Low": lows, "Close": closes})
    atr = compute_atr(df, period=14)
    assert float(atr.dropna().iloc[-1]) > 2.0


# ─── safe_last ───────────────────────────────────────────────────────────────


def test_safe_last_normal():
    assert safe_last(pd.Series([1.0, 2.0, 3.0])) == 3.0


def test_safe_last_all_nan():
    assert math.isnan(safe_last(pd.Series([float("nan"), float("nan")])))


def test_safe_last_empty():
    assert math.isnan(safe_last(pd.Series([], dtype=float)))


def test_safe_last_trailing_nan_returns_last_valid():
    # dropna strips trailing NaN, so the last non-NaN value is returned
    s = pd.Series([1.0, 5.0, float("nan"), float("nan")])
    assert safe_last(s) == 5.0


def test_safe_last_single_element():
    assert safe_last(pd.Series([42.0])) == 42.0


# ─── slope_of_series ─────────────────────────────────────────────────────────


def test_slope_uptrend_positive():
    s = pd.Series(range(20), dtype=float)
    assert slope_of_series(s, lookback=10) > 0


def test_slope_downtrend_negative():
    s = pd.Series([20.0 - i for i in range(20)])
    assert slope_of_series(s, lookback=10) < 0


def test_slope_flat_near_zero():
    s = pd.Series([10.0] * 20)
    assert pytest.approx(slope_of_series(s, lookback=10), abs=1e-9) == 0.0


def test_slope_insufficient_data_returns_nan():
    assert math.isnan(slope_of_series(pd.Series([1.0, 2.0, 3.0]), lookback=10))


def test_slope_exactly_at_lookback_not_nan():
    s = pd.Series(range(10), dtype=float)
    assert not math.isnan(slope_of_series(s, lookback=10))


def test_slope_lookback_1_returns_nan():
    """lookback=1 → zero x-variance → NaN, not a division error."""
    assert math.isnan(slope_of_series(pd.Series([1.0, 2.0, 3.0]), lookback=1))


def test_slope_normalization_higher_price_smaller_value():
    """Same raw $/bar slope must produce a smaller normalized slope for a more expensive stock."""
    low_price = pd.Series([10.0 + i for i in range(20)])
    high_price = pd.Series([100.0 + i for i in range(20)])
    assert slope_of_series(low_price, 10) > slope_of_series(high_price, 10)


# ─── analyze_symbol ──────────────────────────────────────────────────────────


def test_analyze_missing_volume_column_sets_error():
    df = pd.DataFrame({"Open": [1.0], "High": [1.1], "Low": [0.9], "Close": [1.0]})
    result = analyze_symbol(df)
    assert "error" in result
    assert "Volume" in result["error"]


def test_analyze_missing_high_column_sets_error():
    df = pd.DataFrame(
        {"Open": [1.0], "Low": [0.9], "Close": [1.0], "Volume": [100_000.0]}
    )
    result = analyze_symbol(df)
    assert "error" in result
    assert "High" in result["error"]


def test_analyze_all_nan_volume_sets_error():
    """Valid Close but all-NaN Volume must hit the error path, not compute."""
    df = _ohlcv(n=60)
    df["Volume"] = float("nan")
    result = analyze_symbol(df)
    assert "error" in result
    assert "Volume" in result["error"]


def test_analyze_all_nan_close_sets_error():
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [1.1],
            "Low": [0.9],
            "Close": [float("nan")],
            "Volume": [100_000.0],
        }
    )
    result = analyze_symbol(df)
    assert "error" in result


def test_analyze_healthy_returns_required_keys():
    result = analyze_symbol(_ohlcv())
    for key in [
        "last_close", "avg_dollar_vol_20", "above_50d", "above_200d",
        "atr_pct_14", "worst_1d_ret_126", "vol_trend_up", "rs_1m", "rs_3m",
        "days_stale", "sma50_slope",
    ]:
        assert key in result, f"missing key: {key}"
    assert "error" not in result


def test_analyze_last_close_correct():
    df = _ohlcv(n=252, price=50.0, drift=0.0)
    assert pytest.approx(analyze_symbol(df)["last_close"], rel=1e-6) == 50.0


def test_analyze_avg_dollar_vol_correct():
    # price=20, volume=100k → $2M/day; with zero drift close stays at 20
    df = _ohlcv(n=252, price=20.0, volume=100_000, drift=0.0)
    result = analyze_symbol(df)
    assert pytest.approx(result["avg_dollar_vol_20"], rel=0.01) == 20.0 * 100_000


def test_analyze_avg_dollar_vol_nan_with_fewer_than_20_bars():
    result = analyze_symbol(_ohlcv(n=15))
    assert math.isnan(result["avg_dollar_vol_20"])


def test_analyze_above_50d_true_in_uptrend():
    result = analyze_symbol(_ohlcv(n=252, drift=0.002))
    assert result["above_50d"] is True


def test_analyze_above_50d_false_in_downtrend():
    result = analyze_symbol(_ohlcv(n=252, drift=-0.003))
    assert result["above_50d"] is False


def test_analyze_above_50d_false_when_fewer_than_50_bars():
    result = analyze_symbol(_ohlcv(n=30, drift=0.01))
    assert result["above_50d"] is False


def test_analyze_above_200d_false_when_fewer_than_200_bars():
    result = analyze_symbol(_ohlcv(n=100, drift=0.005))
    assert result["above_200d"] is False


# Fixed weekday anchor for staleness tests: bdate_range never rolls a weekday
# end-date back, so expected days_stale values are exact regardless of which
# day the suite runs on. The production clock is pinned via swing_universe._today.
_STALE_ANCHOR = pd.Timestamp("2026-07-22")  # a Wednesday


@patch("swing_universe._today", return_value=_STALE_ANCHOR)
def test_analyze_days_stale_zero_for_fresh_data(_mock_today):
    df = _ohlcv(n=252, end_date=_STALE_ANCHOR)
    assert analyze_symbol(df)["days_stale"] == 0


@patch("swing_universe._today", return_value=_STALE_ANCHOR)
def test_analyze_days_stale_correct_for_old_data(_mock_today):
    stale_end = _STALE_ANCHOR - pd.Timedelta(days=14)  # also a Wednesday
    result = analyze_symbol(_ohlcv(n=252, end_date=stale_end))
    assert result["days_stale"] == 14


@patch("swing_universe._today", return_value=_STALE_ANCHOR)
def test_analyze_days_stale_tz_aware_index_stripped(_mock_today):
    """tz-aware DatetimeIndex must not raise — timezone is stripped internally."""
    stale_end = _STALE_ANCHOR - pd.Timedelta(days=7)
    df = _ohlcv(n=252, end_date=stale_end, tz="UTC")
    result = analyze_symbol(df)
    assert result["days_stale"] == 7


def test_analyze_atr_pct_close_to_spread():
    """atr_pct_14 ≈ H-L spread / price for a flat-price series."""
    df = _ohlcv(n=60, price=20.0, atr_pct=0.02, drift=0.0)
    result = analyze_symbol(df)
    assert pytest.approx(result["atr_pct_14"], abs=0.003) == 0.02


def test_analyze_atr_nan_with_fewer_than_14_bars():
    result = analyze_symbol(_ohlcv(n=10))
    assert math.isnan(result["atr_pct_14"])


def test_analyze_atr_ignores_halted_all_nan_ohlc_days():
    """FIX #15: halted days (all-NaN OHLC, Volume 0) survive dropna(how='all')
    in the pipeline. They must not leave ATR NaN or stale — the halted bars
    are dropped and ATR reflects the most recent real bars."""
    n = 60
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    closes = np.full(n, 20.0)
    # bars 0-39: H-L = 2.0; bars 40+: H-L = 6.0 (so stale vs current differ)
    half_spread = np.where(np.arange(n) >= 40, 3.0, 1.0)
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes + half_spread,
            "Low": closes - half_spread,
            "Close": closes,
            "Volume": np.full(n, 100_000.0),
        },
        index=dates,
    )
    # halt on bars 50-52
    df.iloc[50:53, df.columns.get_indexer(["Open", "High", "Low", "Close"])] = float("nan")
    df.iloc[50:53, df.columns.get_indexer(["Volume"])] = 0.0

    result = analyze_symbol(df)
    # All 14 bars of the cleaned last window sit in the wide zone → ATR = 6.0.
    # The old safe_last walk-back returned a pre-halt window mixing narrow
    # bars (≈4.86); NaN would mean the halt days poisoned the rolling window.
    assert result["atr_pct_14"] == pytest.approx(6.0 / 20.0, rel=0.01)


def test_analyze_atr_nan_when_recent_bars_lack_range_data():
    """FIX #15: if the last bar has a Close but no High/Low, ATR is NaN (→
    atr_unavailable rejection) instead of silently reporting an older window's
    ATR as current."""
    df = _ohlcv(n=60)
    df.iloc[-1, df.columns.get_indexer(["High", "Low"])] = float("nan")
    result = analyze_symbol(df)
    assert math.isnan(result["atr_pct_14"])


def test_analyze_zero_price_atr_pct_nan():
    """last_close == 0 must not divide — atr_pct_14 stays NaN."""
    df = _ohlcv(n=60, drift=0.0)
    df["Close"] = 0.0
    df["Open"] = 0.0
    df["High"] = 0.1
    df["Low"] = 0.0
    result = analyze_symbol(df)
    assert math.isnan(result["atr_pct_14"])


@patch("swing_universe._today", return_value=_STALE_ANCHOR)
def test_analyze_days_stale_negative_for_future_dated_bar(_mock_today):
    """A future-dated last bar yields negative staleness (and passes the
    stale filter) — documents current behavior."""
    future_end = _STALE_ANCHOR + pd.Timedelta(days=7)
    result = analyze_symbol(_ohlcv(n=252, end_date=future_end))
    assert result["days_stale"] == -7
    ok, _ = pass_filters(_row(days_stale=result["days_stale"]), TH)
    assert ok


def test_analyze_worst_1d_ret_uses_full_history_below_126_bars():
    """With fewer than 126 bars the worst-day window falls back to the full
    return series instead of returning NaN."""
    result = analyze_symbol(_ohlcv(n=63, drift=-0.002))
    assert result["worst_1d_ret_126"] < 0.0


def test_analyze_worst_1d_ret_zero_for_flat_series():
    """Flat price → all pct_change = 0 → worst_1d_ret_126 = 0.0."""
    result = analyze_symbol(_ohlcv(n=252, drift=0.0))
    assert result["worst_1d_ret_126"] == pytest.approx(0.0, abs=1e-9)


def test_analyze_worst_1d_ret_negative_for_downtrend():
    """Monotonically falling price → every daily return is negative."""
    result = analyze_symbol(_ohlcv(n=252, drift=-0.002))
    assert result["worst_1d_ret_126"] < 0.0


def test_analyze_vol_trend_up_true():
    """Last 20 bars at high volume, older bars at low volume → vol_sma20 > vol_sma50."""
    # 232 bars at 50k, last 20 bars at 500k
    # vol_sma20 = 500k; vol_sma50 = (30*50k + 20*500k)/50 = 230k → True
    closes = np.full(252, 20.0)
    volumes = np.array([50_000.0] * 232 + [500_000.0] * 20)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252)
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )
    assert analyze_symbol(df)["vol_trend_up"] is True


def test_analyze_vol_trend_up_false_when_flat_volume():
    """Uniform volume → vol_sma20 == vol_sma50 → vol_trend_up False (strict >)."""
    result = analyze_symbol(_ohlcv(n=252, volume=100_000))
    assert result["vol_trend_up"] is False


def test_analyze_rs_nan_without_benchmark():
    result = analyze_symbol(_ohlcv())
    assert math.isnan(result["rs_1m"])
    assert math.isnan(result["rs_3m"])


def test_analyze_rs_nan_with_empty_benchmark():
    result = analyze_symbol(_ohlcv(), bench_close=pd.Series(dtype=float))
    assert math.isnan(result["rs_1m"])
    assert math.isnan(result["rs_3m"])


def test_analyze_rs_positive_when_outperforming():
    """Stock up 10% over the period, bench flat → rs_1m > 0."""
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=63)
    stock_c = pd.Series(np.linspace(100.0, 110.0, 63), index=dates)
    bench_c = pd.Series(np.full(63, 30.0), index=dates)
    df = pd.DataFrame(
        {
            "Open": stock_c,
            "High": stock_c * 1.005,
            "Low": stock_c * 0.995,
            "Close": stock_c,
            "Volume": np.full(63, 100_000.0),
        },
        index=dates,
    )
    assert analyze_symbol(df, bench_close=bench_c)["rs_1m"] > 0


def test_analyze_rs_negative_when_underperforming():
    """Stock flat, bench up 10% → rs_1m < 0."""
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=63)
    stock_c = pd.Series(np.full(63, 20.0), index=dates)
    bench_c = pd.Series(np.linspace(30.0, 33.0, 63), index=dates)
    df = pd.DataFrame(
        {
            "Open": stock_c,
            "High": stock_c * 1.005,
            "Low": stock_c * 0.995,
            "Close": stock_c,
            "Volume": np.full(63, 100_000.0),
        },
        index=dates,
    )
    assert analyze_symbol(df, bench_close=bench_c)["rs_1m"] < 0


def test_analyze_rs_nan_when_no_date_overlap():
    """Disjoint date ranges → zero common bars → both RS fields NaN."""
    dates_stock = pd.bdate_range(start="2022-01-01", periods=10)
    dates_bench = pd.bdate_range(start="2024-01-01", periods=100)
    stock_c = pd.Series(np.full(10, 20.0), index=dates_stock)
    bench_c = pd.Series(np.full(100, 30.0), index=dates_bench)
    df = pd.DataFrame(
        {
            "Open": stock_c,
            "High": stock_c * 1.01,
            "Low": stock_c * 0.99,
            "Close": stock_c,
            "Volume": np.full(10, 100_000.0),
        },
        index=dates_stock,
    )
    result = analyze_symbol(df, bench_close=bench_c)
    assert math.isnan(result["rs_1m"])
    assert math.isnan(result["rs_3m"])


def test_analyze_rs_3m_computed_at_exactly_63_bars():
    """Boundary: exactly 63 common bars is enough for rs_3m."""
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=63)
    closes = pd.Series(np.full(63, 20.0), index=dates)
    bench = pd.Series(np.full(63, 30.0), index=dates)
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.005,
            "Low": closes * 0.995,
            "Close": closes,
            "Volume": np.full(63, 100_000.0),
        },
        index=dates,
    )
    result = analyze_symbol(df, bench_close=bench)
    assert not math.isnan(result["rs_3m"])


def test_analyze_rs_nan_when_benchmark_much_staler_than_stock():
    """FIX #16: a benchmark whose data ends well before the stock's last bar
    must not produce an RS presented as current — it degrades to NaN."""
    df = _ohlcv(n=252)
    bench = pd.Series(np.full(252 - 15, 30.0), index=df.index[:-15])
    result = analyze_symbol(df, bench_close=bench)
    assert math.isnan(result["rs_1m"])
    assert math.isnan(result["rs_3m"])


def test_analyze_rs_endpoint_lag_boundary():
    """FIX #16 boundary: a lag of up to 3 stock bars is tolerated, 4 is not."""
    df = _ohlcv(n=252)
    at_limit = analyze_symbol(
        df, bench_close=pd.Series(np.full(252 - 3, 30.0), index=df.index[:-3])
    )
    over_limit = analyze_symbol(
        df, bench_close=pd.Series(np.full(252 - 4, 30.0), index=df.index[:-4])
    )
    assert not math.isnan(at_limit["rs_1m"])
    assert math.isnan(over_limit["rs_1m"])


def test_analyze_rs_nan_with_tz_aware_benchmark():
    """A tz-aware benchmark against a tz-naive stock index can't align —
    RS silently degrades to NaN (documents current behavior)."""
    bench = _bench()
    bench.index = bench.index.tz_localize("UTC")
    result = analyze_symbol(_ohlcv(), bench_close=bench)
    assert math.isnan(result["rs_1m"])
    assert math.isnan(result["rs_3m"])


def test_analyze_rs_1m_computed_but_3m_nan_with_only_21_bars():
    """Exactly 21 common bars → rs_1m computed, rs_3m NaN (needs 63)."""
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=21)
    closes = pd.Series(np.full(21, 20.0), index=dates)
    bench = pd.Series(np.full(21, 30.0), index=dates)
    df = pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.005,
            "Low": closes * 0.995,
            "Close": closes,
            "Volume": np.full(21, 100_000.0),
        },
        index=dates,
    )
    result = analyze_symbol(df, bench_close=bench)
    assert not math.isnan(result["rs_1m"])
    assert math.isnan(result["rs_3m"])


# ─── pass_filters ────────────────────────────────────────────────────────────

TH = Thresholds()  # default thresholds


def _row(**overrides):
    """Valid row that passes all default thresholds."""
    base = {
        "last_close": 20.0,
        "avg_dollar_vol_20": 2_000_000.0,
        "atr_pct_14": 0.02,
        "worst_1d_ret_126": -0.05,
        "above_50d": True,
        "days_stale": 0,
    }
    base.update(overrides)
    return base


def test_pass_all():
    ok, reasons = pass_filters(_row(), TH)
    assert ok is True
    assert reasons == []


def test_pass_price_nan_rejected():
    ok, reasons = pass_filters(_row(last_close=float("nan")), TH)
    assert not ok and "price_too_low" in reasons


def test_pass_price_below_min_rejected():
    ok, reasons = pass_filters(_row(last_close=0.99), TH)
    assert not ok and "price_too_low" in reasons


def test_pass_price_at_min_passes():
    # strict < so price == min_price passes
    ok, _ = pass_filters(_row(last_close=1.0), TH)
    assert ok


def test_pass_adv_nan_rejected():
    ok, reasons = pass_filters(_row(avg_dollar_vol_20=float("nan")), TH)
    assert not ok and "low_dollar_volume" in reasons


def test_pass_adv_below_min_rejected():
    ok, reasons = pass_filters(_row(avg_dollar_vol_20=999_999.0), TH)
    assert not ok and "low_dollar_volume" in reasons


def test_pass_adv_at_min_passes():
    ok, _ = pass_filters(_row(avg_dollar_vol_20=1_000_000.0), TH)
    assert ok


def test_pass_atr_nan_rejected():
    ok, reasons = pass_filters(_row(atr_pct_14=float("nan")), TH)
    assert not ok and "atr_unavailable" in reasons


def test_pass_atr_above_max_rejected():
    ok, reasons = pass_filters(_row(atr_pct_14=0.051), TH)
    assert not ok and "too_volatile_atr" in reasons


def test_pass_atr_at_max_passes():
    # strict > so atr == max_atr passes
    ok, _ = pass_filters(_row(atr_pct_14=0.05), TH)
    assert ok


def test_pass_atr_below_min_rejected():
    # FIX #11: volatility floor — 1% ATR can't deliver a 1-3 week swing
    ok, reasons = pass_filters(_row(atr_pct_14=0.01), TH)
    assert not ok and "too_quiet_atr" in reasons


def test_pass_atr_at_min_passes():
    # strict < so atr == min_atr passes
    ok, _ = pass_filters(_row(atr_pct_14=0.015), TH)
    assert ok


def test_pass_worst_nan_rejected():
    ok, reasons = pass_filters(_row(worst_1d_ret_126=float("nan")), TH)
    assert not ok and "large_gap_risk" in reasons


def test_pass_worst_below_threshold_rejected():
    ok, reasons = pass_filters(_row(worst_1d_ret_126=-0.16), TH)
    assert not ok and "large_gap_risk" in reasons


def test_pass_worst_at_threshold_passes():
    # strict < so -0.15 == -0.15 passes
    ok, _ = pass_filters(_row(worst_1d_ret_126=-0.15), TH)
    assert ok


def test_pass_below_50d_rejected_by_default():
    ok, reasons = pass_filters(_row(above_50d=False), TH)
    assert not ok and "below_50d" in reasons


def test_pass_below_50d_ignored_when_not_required():
    th = Thresholds(require_above_50d=False)
    ok, _ = pass_filters(_row(above_50d=False), th)
    assert ok


def test_pass_stale_at_max_passes():
    # strict > so stale == max_stale_days passes
    ok, _ = pass_filters(_row(days_stale=5), TH)
    assert ok


def test_pass_stale_one_over_max_rejected():
    ok, reasons = pass_filters(_row(days_stale=6), TH)
    assert not ok
    assert any("stale_data" in r and "6d" in r for r in reasons)


def test_pass_stale_missing_key_defaults_to_999_and_fails():
    row = _row()
    del row["days_stale"]
    ok, reasons = pass_filters(row, TH)
    assert not ok and any("stale_data" in r for r in reasons)


def test_pass_multiple_reasons_accumulated():
    """All three metric failures are collected independently."""
    row = _row(last_close=0.5, avg_dollar_vol_20=1.0, atr_pct_14=0.99)
    ok, reasons = pass_filters(row, TH)
    assert not ok
    assert "price_too_low" in reasons
    assert "low_dollar_volume" in reasons
    assert "too_volatile_atr" in reasons


def test_pass_each_reject_reason_independent():
    """A ticker can accumulate all 6 reject reasons simultaneously."""
    row = {
        "last_close": 0.0,
        "avg_dollar_vol_20": 0.0,
        "atr_pct_14": 1.0,
        "worst_1d_ret_126": -0.99,
        "above_50d": False,
        "days_stale": 999,
    }
    ok, reasons = pass_filters(row, TH)
    assert not ok
    # price_too_low, low_dollar_volume, too_volatile_atr,
    # large_gap_risk, below_50d, stale_data_999d
    assert len(reasons) == 6


# ─── is_excluded_instrument (FIX #14) ────────────────────────────────────────


@pytest.mark.parametrize("symbol", [
    "PMZ-UN.TO",   # REIT / trust unit
    "AP-UN.TO",
    "BTB-UN.TO",
    "HHL-U.TO",    # USD-denominated listing
    "ABC-WT.TO",   # warrant
    "DEF-RT.V",    # right
    "GHI-DB.TO",   # debenture
    "RY-PA.TO",    # preferred series
    "ENB-PC.TO",
    # FIX #17: lettered multi-series variants — real tickers from
    # data/can_tickers_full that leaked through as "common" pre-fix.
    "AD-DBB.TO",   # convertible debenture, series B
    "FC-DBM.TO",   # convertible debenture, series M
    "GPH-WTA.V",   # warrant, series A
    "IAU-WTU.TO",  # warrant, series U
    "TD-PFA.TO",   # rate-reset preferred, Series F sub-series A
    "BN-PFM.TO",   # preferred, Series F sub-series M
    "ENB-PFV.TO",  # preferred, Series F sub-series V
])
def test_excluded_instrument_true(symbol):
    assert is_excluded_instrument(symbol) is True


@pytest.mark.parametrize("symbol", [
    "RY.TO",       # plain common
    "BBD-B.TO",    # share class B — must be kept
    "GIB-A.TO",    # share class A — must be kept
    "POW.TO",
    "UNI.TO",      # 'UN' inside the root, not a suffix
    "DBM.TO",
])
def test_excluded_instrument_false(symbol):
    assert is_excluded_instrument(symbol) is False


def test_excluded_instrument_expects_uppercase():
    """Contract: symbols are matched uppercase (read_tickers uppercases);
    lowercase input is NOT recognized."""
    assert is_excluded_instrument("pmz-un.to") is False


def test_excluded_instrument_works_without_exchange_suffix():
    assert is_excluded_instrument("PMZ-UN") is True
    assert is_excluded_instrument("RY") is False


def test_excluded_instrument_single_letter_class_p_kept():
    """A hypothetical class-P common share is kept — only two-letter
    preferred series (-PA..-PZ) are excluded."""
    assert is_excluded_instrument("X-P.TO") is False


def test_excluded_instrument_preferred_series_letter_beyond_two_kept():
    """FIX #17 boundary: -P plus 1-2 letters (series A..Z, series F-A..F-V)
    is excluded; a 3-letter tail after -P is outside the known convention
    and is kept rather than guessed at."""
    assert is_excluded_instrument("X-PFAX.TO") is False


@pytest.mark.parametrize("symbol", [
    "AGMR-WTB.TO", "RECO-WTB.V", "RECO-WTC.V", "STCK-WTB.TO", "STCK-WTC.TO",
    "ODV-WTV.V", "MAK-WTU.TO",              # warrants, more series letters
    "AD-DBC.TO", "AFN-DBK.TO", "DIV-DBB.TO", "FSZ-DBC.TO", "PBH-DBK.TO",
    "SVI-DBD.TO", "VITL-DBH.TO", "VITL-DBI.TO",  # debentures, more series letters
    "PPL-PFA.TO", "PPL-PFE.TO", "PWF-PFA.TO",
    "TD-PFI.TO", "TD-PFJ.TO",               # preferreds, more Series-F letters
])
def test_excluded_instrument_true_lettered_series_regression(symbol):
    """FIX #17 regression set: real tickers pulled from
    data/can_tickers_full that were confirmed leaking through pre-fix."""
    assert is_excluded_instrument(symbol) is True


# ─── score_row ───────────────────────────────────────────────────────────────


def test_score_empty_row_zero():
    assert score_row({}) == pytest.approx(0.0)


def test_score_liquidity_1m():
    # log10(1e6) - 5 = 1.0
    assert score_row({"avg_dollar_vol_20": 1_000_000.0}) == pytest.approx(1.0)


def test_score_liquidity_10m_hits_cap():
    # log10(1e7) - 5 = 2.0, exactly at the cap
    assert score_row({"avg_dollar_vol_20": 10_000_000.0}) == pytest.approx(2.0)


def test_score_liquidity_capped_at_2():
    # $1B → log10(1e9)-5=4.0, capped at 2.0 so mega-caps can't drown momentum
    assert score_row({"avg_dollar_vol_20": 1_000_000_000.0}) == pytest.approx(2.0)


def test_score_liquidity_nan_zero_contribution():
    assert score_row({"avg_dollar_vol_20": float("nan")}) == pytest.approx(0.0)


def test_score_above_50d_adds_1():
    assert score_row({"above_50d": True}) - score_row({}) == pytest.approx(1.0)


def test_score_above_200d_adds_2():
    assert score_row({"above_200d": True}) - score_row({}) == pytest.approx(2.0)


def test_score_above_200d_default_matches_prefer_true():
    """Default prefer_above_200d=True preserves pre-FIX#18 behavior."""
    row = {"above_200d": True}
    assert score_row(row) == score_row(row, prefer_above_200d=True)


def test_score_above_200d_bonus_disabled_when_not_preferred():
    """FIX #18: prefer_above_200d=False must actually suppress the bonus —
    previously Thresholds.prefer_above_200d was read nowhere and the +2.0
    applied unconditionally regardless of the flag."""
    row = {"above_200d": True}
    assert score_row(row, prefer_above_200d=False) == pytest.approx(0.0)
    assert score_row(row, prefer_above_200d=False) != score_row(row, prefer_above_200d=True)


def test_score_above_200d_false_unaffected_by_prefer_flag():
    row = {"above_200d": False}
    assert score_row(row, prefer_above_200d=False) == score_row(row, prefer_above_200d=True)


def test_score_negative_slope_no_contribution():
    # Only adds when slope > 0
    assert score_row({"sma50_slope": -0.01}) == pytest.approx(0.0)


def test_score_positive_slope_proportional():
    # 0.001 * 500 = 0.5
    assert score_row({"sma50_slope": 0.001}) == pytest.approx(0.5)


def test_score_positive_slope_capped_at_1_5():
    # 0.01 * 500 = 5.0, capped at 1.5
    assert score_row({"sma50_slope": 0.01}) == pytest.approx(1.5)


def test_score_rs1m_positive_proportional():
    # 0.05 * 25 = 1.25
    assert score_row({"rs_1m": 0.05}) == pytest.approx(1.25)


def test_score_rs1m_capped_positive():
    # 0.15 * 25 = 3.75, capped at 2.5
    assert score_row({"rs_1m": 0.15}) == pytest.approx(2.5)


def test_score_rs1m_capped_negative():
    # -0.15 * 25 = -3.75, floored at -2.5
    assert score_row({"rs_1m": -0.15}) == pytest.approx(-2.5)


def test_score_rs3m_positive_proportional():
    # 0.10 * 12 = 1.2
    assert score_row({"rs_3m": 0.10}) == pytest.approx(1.2)


def test_score_rs3m_capped_positive():
    # 0.20 * 12 = 2.4, capped at 2.0
    assert score_row({"rs_3m": 0.20}) == pytest.approx(2.0)


def test_score_rs3m_capped_negative():
    # -0.20 * 12 = -2.4, floored at -2.0
    assert score_row({"rs_3m": -0.20}) == pytest.approx(-2.0)


def test_score_rs_outweighs_liquidity():
    """FIX: strong RS (±4.5 pts total) must outweigh max liquidity (2.0 pts)."""
    liquid_laggard = score_row(
        {"avg_dollar_vol_20": 1_000_000_000.0, "rs_1m": -0.10, "rs_3m": -0.20}
    )
    thin_leader = score_row(
        {"avg_dollar_vol_20": 1_000_000.0, "rs_1m": 0.10, "rs_3m": 0.20}
    )
    assert thin_leader > liquid_laggard


def test_score_vol_trend_up_adds_0_8():
    assert score_row({"vol_trend_up": True}) == pytest.approx(0.8)


def test_score_vol_ratio_below_1_1_no_extra():
    # Threshold is ratio > 1.1
    assert score_row({"vol_ratio_20_50": 1.05}) == pytest.approx(0.0)


def test_score_vol_ratio_above_1_1_adds_extra():
    # (1.15 - 1.0) * 2.0 = 0.3
    assert score_row({"vol_ratio_20_50": 1.15}) == pytest.approx(0.3)


def test_score_vol_ratio_capped_at_0_5():
    # (5.0 - 1.0) * 2.0 = 8.0, capped at 0.5
    assert score_row({"vol_ratio_20_50": 5.0}) == pytest.approx(0.5)


# ─── atr_band_bonus (FIX #12) ────────────────────────────────────────────────


def test_atr_band_nan_zero():
    assert atr_band_bonus(float("nan")) == pytest.approx(0.0)


def test_atr_band_below_ramp_zero():
    assert atr_band_bonus(0.010) == pytest.approx(0.0)


def test_atr_band_at_ramp_lo_zero():
    assert atr_band_bonus(0.015) == pytest.approx(0.0)


def test_atr_band_mid_ramp_up():
    # halfway between 0.015 and 0.025 → half of peak 1.5
    assert atr_band_bonus(0.020) == pytest.approx(0.75)


def test_atr_band_sweet_spot_full_peak():
    assert atr_band_bonus(0.025) == pytest.approx(1.5)
    assert atr_band_bonus(0.030) == pytest.approx(1.5)
    assert atr_band_bonus(0.035) == pytest.approx(1.5)


def test_atr_band_ramp_down():
    # (0.05 - 0.04) / (0.05 - 0.035) = 2/3 of peak 1.5
    assert atr_band_bonus(0.040) == pytest.approx(1.0)


def test_atr_band_at_ramp_hi_zero():
    assert atr_band_bonus(0.05) == pytest.approx(0.0)


def test_atr_band_above_ramp_hi_zero():
    assert atr_band_bonus(0.20) == pytest.approx(0.0)


def test_score_atr_sweet_spot_rewarded():
    """FIX #12: ATR in the sweet spot must add points via score_row."""
    assert score_row({"atr_pct_14": 0.03}) == pytest.approx(1.5)


def test_score_atr_swing_range_beats_quiet():
    """A 3% ATR mover must outscore a 0.8% ATR sleeper — the old penalty
    ranked them the other way around."""
    assert score_row({"atr_pct_14": 0.03}) > score_row({"atr_pct_14": 0.008})


def test_score_worst_day_no_score_effect():
    """FIX #13: worst-day is a hard filter only — it must not move the score."""
    assert score_row({"worst_1d_ret_126": -0.10}) == pytest.approx(0.0)
    assert score_row({"worst_1d_ret_126": -0.30}) == pytest.approx(0.0)


def test_score_higher_rs_higher_score():
    assert score_row({"rs_1m": 0.05}) > score_row({"rs_1m": -0.05})


def test_score_higher_liquidity_higher_score():
    assert score_row({"avg_dollar_vol_20": 10_000_000.0}) > score_row(
        {"avg_dollar_vol_20": 1_000_000.0}
    )


def test_score_full_row_integration():
    """Pin the total for one realistic passing row so component interactions
    can't drift silently."""
    row = {
        "avg_dollar_vol_20": 5_000_000.0,   # log10(5e6) - 5
        "above_50d": True,                  # +1.0
        "above_200d": True,                 # +2.0
        "sma50_slope": 0.002,               # min(1.5, 0.002*500) = 1.0
        "rs_1m": 0.04,                      # 0.04*25 = 1.0
        "rs_3m": 0.05,                      # 0.05*12 = 0.6
        "vol_trend_up": True,               # +0.8
        "vol_ratio_20_50": 1.2,             # (1.2-1.0)*2.0 = 0.4
        "atr_pct_14": 0.03,                 # sweet spot = +1.5
        "worst_1d_ret_126": -0.08,          # no score effect
    }
    expected = (math.log10(5e6) - 5) + 1.0 + 2.0 + 1.0 + 1.0 + 0.6 + 0.8 + 0.4 + 1.5
    assert score_row(row) == pytest.approx(expected)


def test_score_sub_100k_adv_contributes_negative():
    """Documents current behavior: score_row assumes filtered input — an ADV
    below $100k (impossible for a passing row) yields a negative liquidity
    term rather than being clamped at zero."""
    assert score_row({"avg_dollar_vol_20": 10_000.0}) == pytest.approx(-1.0)


# ─── fetch_history_batch / auto_adjust plumbing (FIX #1 regression guard) ────


@patch("swing_universe.yf.download")
def test_fetch_history_batch_passes_auto_adjust_to_yfinance(mock_dl):
    """auto_adjust=True is the load-bearing data-correctness setting (FIX #1):
    without it, splits/dividends corrupt every SMA/ATR/worst-day figure. Pin
    the kwargs actually sent to yf.download."""
    mock_dl.return_value = pd.DataFrame()
    fetch_history_batch(["RY.TO", "TD.TO"], "1y", "1d", True)

    kwargs = mock_dl.call_args.kwargs
    assert kwargs["auto_adjust"] is True
    assert kwargs["tickers"] == ["RY.TO", "TD.TO"]
    assert kwargs["period"] == "1y"
    assert kwargs["interval"] == "1d"
    assert kwargs["group_by"] == "ticker"
    assert kwargs["progress"] is False


@patch("swing_universe.yf.download")
def test_fetch_benchmark_passes_auto_adjust_to_yfinance(mock_dl):
    dates = pd.bdate_range(end="2024-01-10", periods=50)
    mock_dl.return_value = pd.DataFrame({"Close": np.full(50, 30.0)}, index=dates)
    fetch_benchmark("XIU.TO", "1y", "1d", True)

    assert mock_dl.call_args.args == ("XIU.TO",)
    kwargs = mock_dl.call_args.kwargs
    assert kwargs["auto_adjust"] is True
    assert kwargs["period"] == "1y"
    assert kwargs["interval"] == "1d"


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_forwards_auto_adjust_to_both_fetchers(mock_batch, mock_bench, _sleep, tmp_path):
    """run_universe_builder must forward cfg.auto_adjust into both fetch
    helpers — a config regression here would silently corrupt all metrics."""
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(["RY.TO"])

    run_universe_builder(_cfg(tmp_path, "RY.TO\n", auto_adjust=True))

    # both helpers take (…, period, interval, auto_adjust) positionally
    assert mock_bench.call_args.args[3] is True
    assert mock_batch.call_args.args[3] is True


# ─── fetch_benchmark ─────────────────────────────────────────────────────────


@patch("swing_universe.yf.download")
def test_fetch_benchmark_returns_tz_naive_series(mock_dl):
    dates = pd.bdate_range(end="2024-01-10", periods=50).tz_localize("UTC")
    mock_dl.return_value = pd.DataFrame({"Close": np.full(50, 30.0)}, index=dates)
    result = fetch_benchmark("XIU.TO", "1y", "1d", True)
    assert not result.empty
    assert result.index.tz is None


@patch("swing_universe.yf.download")
def test_fetch_benchmark_multiindex_columns_squeezed_to_series(mock_dl):
    """Newer yfinance returns MultiIndex (field, ticker) columns even for a
    single ticker — raw['Close'] is then a one-column DataFrame that must be
    squeezed into a Series."""
    dates = pd.bdate_range(end="2024-01-10", periods=50)
    mock_dl.return_value = pd.DataFrame(
        {("Close", "XIU.TO"): np.full(50, 30.0),
         ("Volume", "XIU.TO"): np.full(50, 1e6)},
        index=dates,
    )
    result = fetch_benchmark("XIU.TO", "1y", "1d", True)
    assert isinstance(result, pd.Series)
    assert len(result) == 50
    assert result.iloc[-1] == 30.0


@patch("swing_universe.yf.download")
def test_fetch_benchmark_empty_download_returns_empty_series(mock_dl):
    mock_dl.return_value = pd.DataFrame(
        {"Close": pd.Series(dtype=float)},
        index=pd.DatetimeIndex([]),
    )
    result = fetch_benchmark("XIU.TO", "1y", "1d", True)
    assert isinstance(result, pd.Series)
    assert result.empty


@patch("swing_universe.yf.download")
def test_fetch_benchmark_exception_returns_empty_series(mock_dl):
    mock_dl.side_effect = RuntimeError("timeout")
    result = fetch_benchmark("XIU.TO", "1y", "1d", True)
    assert isinstance(result, pd.Series)
    assert result.empty


# ─── run_universe_builder integration ────────────────────────────────────────


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_two_passing_tickers(mock_batch, mock_bench, _sleep, tmp_path):
    tickers = ["RY.TO", "TD.TO"]
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(tickers)

    df_t, df_r = run_universe_builder(_cfg(tmp_path, "RY.TO\nTD.TO\n"))
    assert len(df_t) == 2
    assert len(df_r) == 0


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_excludes_non_common_before_download(mock_batch, mock_bench, _sleep, tmp_path):
    """FIX #14: -UN units are rejected up front and never sent to yfinance."""
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(["RY.TO"])

    cfg = _cfg(tmp_path, "RY.TO\nPMZ-UN.TO\n")
    df_t, df_r = run_universe_builder(cfg)

    assert "PMZ-UN.TO" in set(df_r["symbol"])
    rej = df_r[df_r["symbol"] == "PMZ-UN.TO"].iloc[0]
    assert rej["reject_reasons"] == "excluded_instrument"
    # Only the common share was fetched
    fetched = mock_batch.call_args[0][0]
    assert fetched == ["RY.TO"]
    assert "PMZ-UN.TO" not in set(df_t["symbol"])


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_exclusion_can_be_disabled(mock_batch, mock_bench, _sleep, tmp_path):
    """With exclude_non_common=False, unit tickers go through the normal path."""
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(["PMZ-UN.TO"])

    cfg = _cfg(tmp_path, "PMZ-UN.TO\n",
               thresholds=Thresholds(exclude_non_common=False))
    df_t, df_r = run_universe_builder(cfg)

    fetched = mock_batch.call_args[0][0]
    assert fetched == ["PMZ-UN.TO"]
    assert len(df_t) == 1  # passes all metric filters like any other ticker


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_universe_file_one_per_line(mock_batch, mock_bench, _sleep, tmp_path):
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(["RY.TO", "TD.TO"])
    cfg = _cfg(tmp_path, "RY.TO\nTD.TO\n")
    run_universe_builder(cfg)

    lines = Path(cfg.out_file_path).read_text().strip().splitlines()
    assert set(lines) == {"RY.TO", "TD.TO"}
    # Each line must be exactly one ticker with no extra commas or spaces
    for ln in lines:
        assert "," not in ln and ln == ln.strip()


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_one_line_file_comma_separated(mock_batch, mock_bench, _sleep, tmp_path):
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(["RY.TO", "TD.TO"])
    cfg = _cfg(tmp_path, "RY.TO\nTD.TO\n")
    run_universe_builder(cfg)

    content = Path(cfg.out_one_line_file_path).read_text().strip()
    parts = content.split(",")
    assert set(parts) == {"RY.TO", "TD.TO"}
    # Must not have a trailing comma (empty last element)
    assert parts[-1] != ""


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_rejected_csv_exists(mock_batch, mock_bench, _sleep, tmp_path):
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(["RY.TO"])
    cfg = _cfg(tmp_path, "RY.TO\n", thresholds=Thresholds(min_price=9999.0))
    run_universe_builder(cfg)
    assert Path(cfg.out_rejected_file_path).exists()


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_tradable_sorted_by_score_desc(mock_batch, mock_bench, _sleep, tmp_path):
    tickers = ["RY.TO", "TD.TO", "BNS.TO"]
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(tickers)
    df_t, _ = run_universe_builder(_cfg(tmp_path, "\n".join(tickers) + "\n"))

    if len(df_t) > 1:
        scores = df_t["score"].tolist()
        assert scores == sorted(scores, reverse=True)


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_rejected_score_is_nan(mock_batch, mock_bench, _sleep, tmp_path):
    """Rejected tickers must have NaN score — only tradable tickers get scored."""
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(["RY.TO"])
    cfg = _cfg(tmp_path, "RY.TO\n", thresholds=Thresholds(min_price=9999.0))
    _, df_r = run_universe_builder(cfg)

    assert len(df_r) == 1
    assert math.isnan(df_r.iloc[0]["score"])


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_low_volume_lands_in_rejected_with_reason(mock_batch, mock_bench, _sleep, tmp_path):
    """A ticker below the dollar-volume floor appears in df_rejected with the right reason."""
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(["RY.TO"], volume=1)  # $20 * 1 = $20/day

    _, df_r = run_universe_builder(_cfg(tmp_path, "RY.TO\n"))
    assert len(df_r) == 1
    assert "low_dollar_volume" in df_r.iloc[0]["reject_reasons"]


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_ticker_absent_from_batch_is_rejected_no_data(mock_batch, mock_bench, _sleep, tmp_path):
    """If a ticker is not in the MultiIndex columns, it gets reject_reasons='no_data'."""
    mock_bench.return_value = _bench()
    # Return only TD.TO, not RY.TO
    mock_batch.return_value = _multiindex_df(["TD.TO"])

    cfg = _cfg(tmp_path, "RY.TO\nTD.TO\n")
    df_t, df_r = run_universe_builder(cfg)

    rejected_syms = set(df_r["symbol"].tolist())
    assert "RY.TO" in rejected_syms
    ry = df_r[df_r["symbol"] == "RY.TO"].iloc[0]
    assert ry["reject_reasons"] == "no_data"


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_batch_exception_tickers_land_in_rejected(mock_batch, mock_bench, _sleep, tmp_path):
    """A network-level batch exception is caught and every ticker of the failed
    batch appears in df_rejected — it must not silently vanish."""
    mock_bench.return_value = _bench()
    mock_batch.side_effect = RuntimeError("rate limit hit")

    cfg = _cfg(tmp_path, "RY.TO\nTD.TO\n")
    df_t, df_r = run_universe_builder(cfg)
    assert len(df_t) == 0
    assert set(df_r["symbol"]) == {"RY.TO", "TD.TO"}
    assert (df_r["reject_reasons"] == "batch_fetch_failed").all()


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_every_input_ticker_accounted_for(mock_batch, mock_bench, _sleep, tmp_path):
    """Invariant: every input ticker ends up in exactly one of df_tradable /
    df_rejected, across the excluded, failed-batch, and normal paths."""
    mock_bench.return_value = _bench()
    # First batch (size 2) raises; second batch returns data for BNS.TO only
    mock_batch.side_effect = [
        RuntimeError("rate limit hit"),
        _multiindex_df(["BNS.TO"]),
    ]

    cfg = _cfg(tmp_path, "RY.TO\nTD.TO\nBNS.TO\nCM.TO\nPMZ-UN.TO\n", batch_size=2)
    df_t, df_r = run_universe_builder(cfg)

    all_syms = sorted(df_t["symbol"].tolist() + df_r["symbol"].tolist())
    assert all_syms == ["BNS.TO", "CM.TO", "PMZ-UN.TO", "RY.TO", "TD.TO"]
    assert len(df_t) + len(df_r) == 5


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_creates_missing_output_dirs(mock_batch, mock_bench, _sleep, tmp_path):
    """All three output files must create their parent directories, not just
    the universe file."""
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(["RY.TO"])

    cfg = _cfg(
        tmp_path, "RY.TO\n",
        out_file_path=str(tmp_path / "a/universe"),
        out_one_line_file_path=str(tmp_path / "b/nested/one_line"),
        out_rejected_file_path=str(tmp_path / "c/nested/rejected.csv"),
    )
    run_universe_builder(cfg)
    assert Path(cfg.out_file_path).exists()
    assert Path(cfg.out_one_line_file_path).exists()
    assert Path(cfg.out_rejected_file_path).exists()


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_flat_df_multi_ticker_batch_rejected_as_ambiguous(mock_batch, mock_bench, _sleep, tmp_path):
    """Flat (unlabeled) columns for a multi-ticker batch: the data can't be
    attributed to a symbol, so the whole batch is rejected — batch[0] must NOT
    be credited with prices that may belong to another ticker."""
    mock_bench.return_value = _bench()
    mock_batch.return_value = _ohlcv(n=252)  # flat frame, batch of 2

    cfg = _cfg(tmp_path, "DEAD.TO\nRY.TO\n", batch_size=2)
    df_t, df_r = run_universe_builder(cfg)

    assert len(df_t) == 0
    assert set(df_r["symbol"]) == {"DEAD.TO", "RY.TO"}
    assert (df_r["reject_reasons"] == "ambiguous_flat_data").all()


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_empty_download_rejects_whole_batch_no_data(mock_batch, mock_bench, _sleep, tmp_path):
    """A completely empty DataFrame (total download failure) marks every
    ticker of the batch as no_data."""
    mock_bench.return_value = _bench()
    mock_batch.return_value = pd.DataFrame()

    df_t, df_r = run_universe_builder(_cfg(tmp_path, "RY.TO\nTD.TO\n"))
    assert len(df_t) == 0
    assert (df_r["reject_reasons"] == "no_data").all()


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_all_tickers_excluded_completes(mock_batch, mock_bench, _sleep, tmp_path):
    """Everything excluded up front → zero batches, empty universe files,
    pipeline still completes and writes all outputs."""
    mock_bench.return_value = _bench()

    cfg = _cfg(tmp_path, "PMZ-UN.TO\nRY-PA.TO\n")
    df_t, df_r = run_universe_builder(cfg)

    mock_batch.assert_not_called()
    assert len(df_t) == 0 and len(df_r) == 2
    assert Path(cfg.out_file_path).read_text() == ""
    # one-line file degrades to a single newline for an empty universe
    assert Path(cfg.out_one_line_file_path).read_text() == "\n"
    assert Path(cfg.out_rejected_file_path).exists()


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_empty_input_file_completes(mock_batch, mock_bench, _sleep, tmp_path):
    """Zero input tickers → empty outputs, no crash."""
    mock_bench.return_value = _bench()
    df_t, df_r = run_universe_builder(_cfg(tmp_path, ""))
    assert len(df_t) == 0 and len(df_r) == 0
    mock_batch.assert_not_called()


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_multiindex_all_nan_close_ticker_rejected(mock_batch, mock_bench, _sleep, tmp_path):
    """MultiIndex batch where one ticker has all-NaN Close but valid Volume:
    the analyzer error path must reject it while the healthy ticker passes."""
    mock_bench.return_value = _bench()
    big = _multiindex_df(["RY.TO", "BAD.TO"])
    big[("BAD.TO", "Close")] = float("nan")
    mock_batch.return_value = big

    df_t, df_r = run_universe_builder(_cfg(tmp_path, "RY.TO\nBAD.TO\n"))
    assert "RY.TO" in set(df_t["symbol"])
    bad = df_r[df_r["symbol"] == "BAD.TO"].iloc[0]
    assert bad["tradable"] == False
    assert "Close" in str(bad["error"])


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_flat_df_single_ticker_path(mock_batch, mock_bench, _sleep, tmp_path):
    """Flat (non-MultiIndex) DataFrame — single-ticker yfinance response — is handled."""
    mock_bench.return_value = _bench()
    mock_batch.return_value = _ohlcv(n=252)  # flat, no MultiIndex

    cfg = _cfg(tmp_path, "RY.TO\n", batch_size=1)
    df_t, df_r = run_universe_builder(cfg)
    assert len(df_t) + len(df_r) == 1


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_flat_df_missing_close_is_rejected(mock_batch, mock_bench, _sleep, tmp_path):
    mock_bench.return_value = _bench()
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=10)
    df = pd.DataFrame(
        {"Open": [1.0] * 10, "High": [1.1] * 10, "Low": [0.9] * 10, "Volume": [1e5] * 10},
        index=dates,
    )
    mock_batch.return_value = df

    _, df_r = run_universe_builder(_cfg(tmp_path, "RY.TO\n", batch_size=1))
    assert len(df_r) == 1
    assert "missing_ohlcv" in str(df_r.iloc[0]["reject_reasons"])


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_flat_df_all_nan_close_is_rejected(mock_batch, mock_bench, _sleep, tmp_path):
    mock_bench.return_value = _bench()
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=10)
    df = pd.DataFrame(
        {
            "Open": [1.0] * 10,
            "High": [1.1] * 10,
            "Low": [0.9] * 10,
            "Close": [float("nan")] * 10,
            "Volume": [1e5] * 10,
        },
        index=dates,
    )
    mock_batch.return_value = df

    _, df_r = run_universe_builder(_cfg(tmp_path, "RY.TO\n", batch_size=1))
    assert len(df_r) == 1
    assert "all_nan_close" in str(df_r.iloc[0]["reject_reasons"])


@patch("swing_universe.time.sleep")
@patch("swing_universe.fetch_benchmark")
@patch("swing_universe.fetch_history_batch")
def test_pipeline_output_df_has_all_required_columns(mock_batch, mock_bench, _sleep, tmp_path):
    mock_bench.return_value = _bench()
    mock_batch.return_value = _multiindex_df(["RY.TO"])
    df_t, df_r = run_universe_builder(_cfg(tmp_path, "RY.TO\n"))

    required = [
        "symbol", "tradable", "score", "last_close", "avg_vol_20",
        "avg_dollar_vol_20", "atr_pct_14", "worst_1d_ret_126",
        "above_50d", "above_200d", "sma50_slope",
        "vol_trend_up", "vol_ratio_20_50", "rs_1m", "rs_3m",
        "days_stale", "reject_reasons", "error",
    ]
    for col in required:
        assert col in df_t.columns or col in df_r.columns, f"missing column: {col}"


# ─── _process_symbol unit tests ──────────────────────────────────────────────


def test_process_symbol_passing_ticker():
    rows = []
    cfg = UniverseBuilderConfig(thresholds=Thresholds())
    _process_symbol("RY.TO", _ohlcv(n=252), _bench(), cfg, rows)
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "RY.TO"
    assert r["tradable"] is True
    assert r["reject_reasons"] == ""
    assert not math.isnan(r["score"])


def test_process_symbol_honors_prefer_above_200d_false():
    """FIX #18: cfg.thresholds.prefer_above_200d must reach score_row via
    _process_symbol — a passing, above-200d ticker scores 2.0 pts lower with
    the flag off than with it on (all else equal)."""
    df = _ohlcv(n=252)
    bench = _bench()

    rows_on = []
    cfg_on = UniverseBuilderConfig(thresholds=Thresholds(prefer_above_200d=True))
    _process_symbol("RY.TO", df, bench, cfg_on, rows_on)

    rows_off = []
    cfg_off = UniverseBuilderConfig(thresholds=Thresholds(prefer_above_200d=False))
    _process_symbol("RY.TO", df, bench, cfg_off, rows_off)

    r_on, r_off = rows_on[0], rows_off[0]
    assert r_on["tradable"] is True and r_off["tradable"] is True
    assert r_on["above_200d"] is True  # sanity: the fixture is above its 200d SMA
    assert r_on["score"] - r_off["score"] == pytest.approx(2.0)


def test_process_symbol_short_history_rejected_atr_unavailable():
    """<14 bars → ATR is NaN → rejected via atr_unavailable (among others),
    exercising the NaN-ATR chain end-to-end."""
    rows = []
    cfg = UniverseBuilderConfig(thresholds=Thresholds())
    _process_symbol("NEW.TO", _ohlcv(n=10), _bench(), cfg, rows)
    r = rows[0]
    assert r["tradable"] is False
    assert "atr_unavailable" in r["reject_reasons"].split(",")


def test_process_symbol_failing_ticker_score_is_nan():
    rows = []
    cfg = UniverseBuilderConfig(thresholds=Thresholds(min_price=9999.0))
    _process_symbol("FAIL.TO", _ohlcv(n=252), pd.Series(dtype=float), cfg, rows)
    r = rows[0]
    assert r["tradable"] is False
    assert "price_too_low" in r["reject_reasons"]
    assert math.isnan(r["score"])


def test_process_symbol_reject_reasons_comma_separated():
    """Multiple failures must be joined with commas, not spaces or semicolons."""
    rows = []
    cfg = UniverseBuilderConfig(
        thresholds=Thresholds(min_price=9999.0, min_avg_dollar_vol_20=1e12)
    )
    _process_symbol("FAIL.TO", _ohlcv(n=252), pd.Series(dtype=float), cfg, rows)
    reasons = rows[0]["reject_reasons"]
    assert "," in reasons
    parts = reasons.split(",")
    assert "price_too_low" in parts
    assert "low_dollar_volume" in parts
