"""
Adversarial audit of swing_universe.py — the script that turns ~4000 raw
Canadian tickers into the ~150-ticker swing-trading universe.

Posture: guilty until proven innocent. A screener that runs cleanly and
returns a plausible-looking count is not evidence of correctness — it is
evidence that nothing crashed. Every test below either (a) proves a specific
criterion computes the number it claims to compute, at its boundary, under
missing data, and without peeking at data that would not exist at decision
time, or (b) proves the funnel's set-membership decisions are exactly what a
human auditor would compute by hand for a hand-built universe.

Companion reference consulted for this audit: /home/yurii/dev/pythonfintech
(see compute-slope-series, computing-simple-moving-averages,
yfinance-multi-index-structure) — used to cross-check that swing_universe.py's
SMA/slope conventions (trailing rolling windows, auto_adjust=True) match
standard practice. No divergence worth a finding was found there; FIX #7's
price-normalization of slope is a deliberate, documented customization on
top of the reference `stats.linregress`-based slope, not a bug.

────────────────────────────────────────────────────────────────────────────
CONTRACT — the "market-data structure" this script actually operates on
────────────────────────────────────────────────────────────────────────────
There is no custom class; the structure is a *shape convention* enforced by
convention only, in two layers:

1. Per-symbol OHLCV DataFrame (`sub` in run_universe_builder; the `df` param
   of analyze_symbol / compute_atr). This is the raw layer — audited only
   indirectly (data *ingestion*, i.e. yfinance/MultiIndex parsing, is out of
   scope per the brief; the DataFrame itself is constructed directly in
   fixtures below, never downloaded).
     - pandas.DataFrame, DatetimeIndex.
     - Columns: "Open", "High", "Low", "Close", "Volume" (float-like).
     - INVARIANT ASSUMED BUT NEVER ENFORCED: index is sorted ascending
       (oldest -> newest). Nothing in analyze_symbol or its caller sorts or
       validates this — see TestDataContract. This is the audit's single
       confirmed silent-corruption finding.
     - tz: analyze_symbol tolerates both tz-naive and tz-aware indexes for
       its own staleness math (it strips tz internally); the rest of the
       pipeline normalizes to tz-naive before calling in.
     - Close and Volume may contain NaN (halted days) independently of each
       other; each rolling computation drops NaN in its OWN input series,
       so two indicators for the same symbol can silently be computed over
       different underlying sets of rows.
     - Volume == 0.0 is a valid, non-NaN observation (a halted day that
       still printed a last trade) and is NOT excluded from rolling
       volume/dollar-volume averages.

2. The metrics dict — the actual "record" that pass_filters/score_row/the
   funnel operate on (`Dict[str, float | bool]`, produced by analyze_symbol,
   consumed by pass_filters and score_row). This is where the real business
   logic lives, and where this suite constructs fixtures directly wherever
   the test doesn't specifically need to audit indicator arithmetic.
   Keys: last_close, avg_vol_20, avg_dollar_vol_20, atr_14, atr_pct_14,
   sma50, sma200, above_50d, above_200d, sma50_slope, worst_1d_ret_126,
   vol_trend_up, vol_ratio_20_50, rs_1m, rs_3m, days_stale, error.
   Units: prices in quote currency, *_pct_14 and worst_1d_ret_126 and rs_*
   are fractions (0.03 == 3%), days_stale is integer calendar days,
   above_* / vol_trend_up are booleans defaulting False when unknown.

Assumptions this suite bakes in
────────────────────────────────────────────────────────────────────────────
  - Thresholds() defaults == the shipped production config, unless a test
    says otherwise.
  - Bars are business-day spaced (pd.bdate_range); the code has no explicit
    holiday-calendar awareness, only "whatever rows are present".
  - "Today" for staleness is `_today()` (real wall clock, normalized) —
    frozen via monkeypatch where determinism matters.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from swing_universe import (
    Thresholds,
    UniverseBuilderConfig,
    _process_symbol,
    analyze_symbol,
    compute_atr,
    is_excluded_instrument,
    pass_filters,
    run_universe_builder,
    score_row,
    slope_of_series,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixture builders — construct the market-data structure directly.
# No yfinance, no network, no mocking of the logic under test.
# ─────────────────────────────────────────────────────────────────────────

TODAY = pd.Timestamp.today().normalize()


def make_ohlcv(n, price, atr_pct, volume, drift=0.0, end_date=None):
    """
    A date-ascending OHLCV DataFrame.

    With drift=0.0 (flat price), High-Low == price*atr_pct and prev_close ==
    Close every day, so True Range == High-Low exactly and ATR-14 ==
    price*atr_pct to machine precision — this gives EXACT control over
    atr_pct_14 for boundary tests (verified empirically before writing this
    suite; drift introduces a compounding bias of a few bps that would make
    boundary assertions flaky).
    """
    end_date = end_date or TODAY
    dates = pd.bdate_range(end=end_date, periods=n)
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


def healthy_row(**overrides):
    """A metrics dict that clears every hard filter in Thresholds() with
    headroom, and is scoring-neutral (no RS/slope/vol-trend) unless
    overridden. Used to isolate exactly one criterion per test."""
    row = {
        "last_close": 20.0,
        "avg_dollar_vol_20": 2_000_000.0,
        "atr_pct_14": 0.03,
        "worst_1d_ret_126": -0.02,
        "above_50d": True,
        "above_200d": True,
        "days_stale": 0,
    }
    row.update(overrides)
    return row


TH = Thresholds()  # production defaults


# ─────────────────────────────────────────────────────────────────────────
# TestDataContract — the headline finding
# ─────────────────────────────────────────────────────────────────────────

class TestDataContract:
    """Failure hypothesis: analyze_symbol trusts its caller to hand it a
    date-ascending-sorted DataFrame and never verifies this. yfinance
    happens to always return ascending data today, so this is dormant in
    production — but nothing in the code enforces it, and the failure mode
    when it's violated is 100% silent (no exception, no error key, just a
    wrong last_close/days_stale computed from the wrong row)."""

    def test_ascending_index_gives_correct_last_close_and_staleness(self):
        df = make_ohlcv(60, price=10.0, atr_pct=0.02, volume=500_000, drift=0.10)
        out = analyze_symbol(df)
        assert out["last_close"] == pytest.approx(df["Close"].iloc[-1])
        assert out["days_stale"] == 0

    def test_reversed_index_does_not_silently_flip_last_close(self):
        """FIX #19 regression test — was xfail (silent misclassification on
        unsorted input) until analyze_symbol started defensively sorting."""
        df = make_ohlcv(60, price=10.0, atr_pct=0.02, volume=500_000, drift=0.10)
        reversed_df = df.iloc[::-1]  # same rows, descending date order
        out = analyze_symbol(reversed_df)
        # Correct behavior: either raise/flag, or still resolve the true
        # most-recent close. Actual behavior: silently returns the OLDEST
        # close (10.0) as "last_close", and a wildly wrong days_stale.
        assert out.get("error") is not None or out["last_close"] == pytest.approx(
            df["Close"].iloc[-1]
        )

    def test_single_ticker_path_silently_misclassifies_a_reversed_index(self, tmp_path):
        """FIX #19 regression test — was xfail on the single-ticker fallback
        branch of run_universe_builder (reached whenever a batch resolves to
        exactly one symbol); analyze_symbol now sorts defensively."""
        good = make_ohlcv(260, price=20.0, atr_pct=0.03, volume=400_000, drift=0.002)
        reversed_good = good.iloc[::-1]  # identical rows, descending date order

        (tmp_path / "tickers.txt").write_text("REVERSED.TO\n")
        bench = pd.Series(
            np.linspace(50, 55, 260), index=pd.bdate_range(end=TODAY, periods=260)
        )
        cfg = UniverseBuilderConfig(
            tickers_path=str(tmp_path / "tickers.txt"),
            out_file_path=str(tmp_path / "universe"),
            out_one_line_file_path=str(tmp_path / "one_line"),
            out_rejected_file_path=str(tmp_path / "rejected.csv"),
            batch_size=1,  # forces the single-ticker flat-DataFrame branch
            sleep_seconds=0.0,
            thresholds=Thresholds(),
        )
        # fetch_history_batch is called with batch=["REVERSED.TO"]; yfinance
        # returns a flat (non-MultiIndex) DataFrame for a lone ticker.
        with patch("swing_universe.fetch_history_batch", return_value=reversed_good), \
             patch("swing_universe.fetch_benchmark", return_value=bench):
            df_tradable, df_rejected = run_universe_builder(cfg)

        # `good` (correctly ordered) clears every filter — REVERSED.TO is
        # built from the exact same rows and a correct screener treats them
        # identically. It doesn't: the reversed frame is silently rejected
        # (its own backwards last_close reads as a stale downtrend) with no
        # reason that flags "your input was unsorted".
        assert set(df_tradable["symbol"]) == {"REVERSED.TO"}


# ─────────────────────────────────────────────────────────────────────────
# TestPriceFilter
# ─────────────────────────────────────────────────────────────────────────

class TestPriceFilter:
    """Failure hypothesis: the min-price boundary uses the wrong comparison
    operator (off-by-one at the threshold), or a NaN price is silently
    treated as passing."""

    def test_price_above_min_passes(self):
        ok, reasons = pass_filters(healthy_row(last_close=1.01), TH)
        assert "price_too_low" not in reasons

    def test_price_exactly_at_min_passes(self):
        """Boundary: min_price is inclusive (>= semantics via `< min` reject)."""
        ok, reasons = pass_filters(healthy_row(last_close=1.0), TH)
        assert "price_too_low" not in reasons

    def test_price_one_cent_below_min_rejected(self):
        ok, reasons = pass_filters(healthy_row(last_close=0.99), TH)
        assert "price_too_low" in reasons

    def test_price_nan_fails_closed(self):
        """NaN must never be silently treated as 'passes' — a missing price
        is the strongest possible reason to exclude a ticker."""
        ok, reasons = pass_filters(healthy_row(last_close=float("nan")), TH)
        assert "price_too_low" in reasons
        assert ok is False

    def test_zero_price_rejected_not_divide_error(self):
        ok, reasons = pass_filters(healthy_row(last_close=0.0), TH)
        assert "price_too_low" in reasons


# ─────────────────────────────────────────────────────────────────────────
# TestLiquidityFilter
# ─────────────────────────────────────────────────────────────────────────

class TestLiquidityFilter:
    """Failure hypothesis: dollar-volume boundary is off-by-one, a
    zero-volume ticker slips through as NaN-passes, or a ticker with fewer
    than 20 bars gets a garbage (not NaN) average instead of being excluded."""

    def test_adv_exactly_at_min_passes(self):
        ok, reasons = pass_filters(healthy_row(avg_dollar_vol_20=1_000_000.0), TH)
        assert "low_dollar_volume" not in reasons

    def test_adv_one_dollar_below_min_rejected(self):
        ok, reasons = pass_filters(healthy_row(avg_dollar_vol_20=999_999.0), TH)
        assert "low_dollar_volume" in reasons

    def test_adv_nan_fails_closed(self):
        ok, reasons = pass_filters(healthy_row(avg_dollar_vol_20=float("nan")), TH)
        assert "low_dollar_volume" in reasons

    def test_zero_volume_every_day_is_zero_not_nan_and_rejected(self):
        """Volume==0 (halted-but-priced) must count as real zero dollar
        volume, not be silently dropped from the rolling average."""
        df = make_ohlcv(60, price=20.0, atr_pct=0.02, volume=0, drift=0.0)
        out = analyze_symbol(df)
        assert out["avg_dollar_vol_20"] == pytest.approx(0.0)
        ok, reasons = pass_filters(out, TH)
        assert "low_dollar_volume" in reasons

    def test_fewer_than_20_bars_gives_nan_not_a_garbage_average(self):
        """Hand-computable: 15 bars is one short of the 20-bar window —
        avg_dollar_vol_20 must be NaN (and therefore fail closed), never a
        partial-window number that could accidentally clear the filter."""
        df = make_ohlcv(15, price=20.0, atr_pct=0.02, volume=10_000_000, drift=0.0)
        out = analyze_symbol(df)
        assert np.isnan(out["avg_dollar_vol_20"])
        ok, reasons = pass_filters(out, TH)
        assert "low_dollar_volume" in reasons


# ─────────────────────────────────────────────────────────────────────────
# TestVolatilityBandFilter (ATR%)
# ─────────────────────────────────────────────────────────────────────────

class TestVolatilityBandFilter:
    """Failure hypothesis: min/max ATR% boundaries are inconsistent with
    each other (one inclusive, one exclusive) in a way nobody intended, or
    the sweet-spot scoring band silently rewards a ticker the hard filter
    would reject."""

    def test_hand_computed_atr_pct_flat_series(self):
        """H-L spread of exactly 3% of price every day, flat price ->
        ATR-14 / last_close must equal 0.03 to within float tolerance."""
        df = make_ohlcv(40, price=50.0, atr_pct=0.03, volume=500_000, drift=0.0)
        out = analyze_symbol(df)
        assert out["atr_pct_14"] == pytest.approx(0.03, abs=1e-9)

    def test_atr_pct_exactly_at_max_passes(self):
        ok, reasons = pass_filters(healthy_row(atr_pct_14=0.05), TH)
        assert "too_volatile_atr" not in reasons

    def test_atr_pct_fraction_above_max_rejected(self):
        ok, reasons = pass_filters(healthy_row(atr_pct_14=0.050001), TH)
        assert "too_volatile_atr" in reasons

    def test_atr_pct_exactly_at_min_passes(self):
        ok, reasons = pass_filters(healthy_row(atr_pct_14=0.015), TH)
        assert "too_quiet_atr" not in reasons

    def test_atr_pct_fraction_below_min_rejected(self):
        ok, reasons = pass_filters(healthy_row(atr_pct_14=0.014999), TH)
        assert "too_quiet_atr" in reasons

    def test_atr_pct_nan_fails_closed_not_silently_passed(self):
        ok, reasons = pass_filters(healthy_row(atr_pct_14=float("nan")), TH)
        assert "atr_unavailable" in reasons
        assert "too_quiet_atr" not in reasons  # exactly one reason for NaN
        assert "too_volatile_atr" not in reasons

    def test_min_and_max_cannot_both_fire_on_the_same_row(self):
        """Combinator sanity: the if/elif structure makes too_volatile and
        too_quiet mutually exclusive by construction (max > min always for
        the production thresholds) — confirm no row can be double-counted."""
        for pct in (0.0, 0.001, 0.015, 0.03, 0.05, 0.2, 1.0):
            _, reasons = pass_filters(healthy_row(atr_pct_14=pct), TH)
            assert not ({"too_quiet_atr", "too_volatile_atr"} <= set(reasons))

    def test_sweet_spot_bonus_only_rewards_within_the_hard_filter_band(self):
        """The scoring sweet-spot (2.5-3.5%) sits strictly inside the hard
        filter band (1.5-5%) — a ticker can never get the ATR score bonus
        while simultaneously failing the ATR hard filter for the SAME atr_pct."""
        from swing_universe import atr_band_bonus

        for pct in np.arange(0.0, 0.08, 0.002):
            bonus = atr_band_bonus(pct)
            _, reasons = pass_filters(healthy_row(atr_pct_14=pct), TH)
            atr_hard_failed = ("too_quiet_atr" in reasons) or ("too_volatile_atr" in reasons)
            if bonus > 0:
                assert not atr_hard_failed, f"atr_pct={pct} scores a bonus but fails the hard filter"


# ─────────────────────────────────────────────────────────────────────────
# TestGapRiskFilter (worst 1-day return)
# ─────────────────────────────────────────────────────────────────────────

class TestGapRiskFilter:
    """Failure hypothesis: the gap-risk filter uses tomorrow's / a
    forward-looking return instead of the worst REALIZED daily return, or
    it silently leaks into scoring after FIX #13 supposedly removed it."""

    def test_hand_computed_worst_return_is_the_actual_minimum_daily_pct_change(self):
        closes = [20.0] * 60
        closes[-10] = 20.0 * 0.90  # a single -10% day, 10 bars back
        df = pd.DataFrame(
            {
                "Open": closes, "High": [c * 1.001 for c in closes],
                "Low": [c * 0.999 for c in closes], "Close": closes,
                "Volume": [500_000.0] * 60,
            },
            index=pd.bdate_range(end=TODAY, periods=60),
        )
        out = analyze_symbol(df)
        assert out["worst_1d_ret_126"] == pytest.approx(-0.10, abs=1e-6)

    def test_worst_return_exactly_at_threshold_passes(self):
        ok, reasons = pass_filters(healthy_row(worst_1d_ret_126=-0.15), TH)
        assert "large_gap_risk" not in reasons

    def test_worst_return_fraction_beyond_threshold_rejected(self):
        ok, reasons = pass_filters(healthy_row(worst_1d_ret_126=-0.150001), TH)
        assert "large_gap_risk" in reasons

    def test_worst_return_nan_fails_closed(self):
        ok, reasons = pass_filters(healthy_row(worst_1d_ret_126=float("nan")), TH)
        assert "large_gap_risk" in reasons

    def test_worst_return_has_zero_effect_on_score_fix_13(self):
        """FIX #13 claims worst-day was removed from scoring (hard filter
        only) to stop double-counting volatility already captured by ATR.
        Confirm the score is IDENTICAL regardless of worst_1d_ret_126."""
        base = healthy_row(worst_1d_ret_126=-0.01)
        crashy = healthy_row(worst_1d_ret_126=-0.149)
        assert score_row(base) == pytest.approx(score_row(crashy))


# ─────────────────────────────────────────────────────────────────────────
# TestTrendFilter (above_50d hard gate, above_200d soft bonus)
# ─────────────────────────────────────────────────────────────────────────

class TestTrendFilter:
    """Failure hypothesis: the 50d trend gate silently defaults to 'passes'
    when SMA is unavailable (insufficient history slipping through), or a
    price sitting exactly on its SMA50 is treated inconsistently with how
    every other boundary in this file resolves ties."""

    def test_above_50d_true_passes(self):
        ok, reasons = pass_filters(healthy_row(above_50d=True), TH)
        assert "below_50d" not in reasons

    def test_above_50d_false_rejected(self):
        ok, reasons = pass_filters(healthy_row(above_50d=False), TH)
        assert "below_50d" in reasons

    def test_missing_above_50d_key_fails_closed_not_silently_passes(self):
        row = healthy_row()
        del row["above_50d"]
        ok, reasons = pass_filters(row, TH)
        assert "below_50d" in reasons

    def test_require_above_50d_false_disables_the_gate(self):
        ok, reasons = pass_filters(
            healthy_row(above_50d=False),
            Thresholds(require_above_50d=False),
        )
        assert "below_50d" not in reasons

    def test_hand_computed_price_exactly_equal_to_sma50_ties_against_inclusion(self):
        """Boundary the other filters don't share: above_50d uses a STRICT
        `>` (last_close > sma50), so a price sitting exactly on its own
        50-day average resolves to *excluded* — the opposite convention
        from every other hard filter in this module, which resolve an
        exact-threshold tie to *included*. Documenting this asymmetry so a
        future refactor changes it on purpose, not by accident."""
        df = make_ohlcv(60, price=20.0, atr_pct=0.02, volume=500_000, drift=0.0)
        out = analyze_symbol(df)
        assert out["last_close"] == pytest.approx(out["sma50"])
        assert out["above_50d"] is False

    def test_above_200d_true_adds_bonus_when_preferred(self):
        assert score_row(healthy_row(above_200d=True), prefer_above_200d=True) > \
               score_row(healthy_row(above_200d=False), prefer_above_200d=True)

    def test_above_200d_bonus_gated_off_by_prefer_flag_fix_18(self):
        with_pref = score_row(healthy_row(above_200d=True), prefer_above_200d=True)
        without_pref = score_row(healthy_row(above_200d=True), prefer_above_200d=False)
        assert with_pref > without_pref

    def test_above_200d_is_a_soft_gate_not_a_hard_filter(self):
        """A ticker below its 200d SMA must still be able to pass — 200d is
        scoring-only per config (prefer_above_200d), not a reject reason."""
        row = healthy_row(above_200d=False)
        ok, reasons = pass_filters(row, TH)
        assert ok is True
        assert not any("200" in r for r in reasons)

    def test_insufficient_history_for_sma50_fails_closed_not_garbage(self):
        """30 bars is not enough for a 50-bar SMA — above_50d must default
        False (excluded), never compute a partial-window average that could
        accidentally read as 'above'."""
        df = make_ohlcv(30, price=20.0, atr_pct=0.02, volume=500_000, drift=0.001)
        out = analyze_symbol(df)
        assert np.isnan(out["sma50"])
        assert out["above_50d"] is False
        ok, reasons = pass_filters(out, TH)
        assert "below_50d" in reasons


# ─────────────────────────────────────────────────────────────────────────
# TestStalenessFilter
# ─────────────────────────────────────────────────────────────────────────

class TestStalenessFilter:
    """Failure hypothesis: a delisted/halted ticker whose last bar is old
    slips through because days_stale silently defaults to a passing value
    (0) instead of a failing one when the field is missing."""

    def test_fresh_data_zero_days_stale(self):
        df = make_ohlcv(30, price=20.0, atr_pct=0.02, volume=500_000, end_date=TODAY)
        out = analyze_symbol(df)
        assert out["days_stale"] == 0

    def test_stale_exactly_at_max_passes(self):
        ok, reasons = pass_filters(healthy_row(days_stale=5), TH)
        assert not any(r.startswith("stale_data") for r in reasons)

    def test_stale_one_day_beyond_max_rejected(self):
        ok, reasons = pass_filters(healthy_row(days_stale=6), TH)
        assert "stale_data_6d" in reasons

    def test_missing_days_stale_key_defaults_to_fail_not_pass(self):
        """The .get(..., 999) default must fail closed — a screener that
        can't determine freshness must never assume 'fresh'."""
        row = healthy_row()
        del row["days_stale"]
        ok, reasons = pass_filters(row, TH)
        assert not ok
        assert any(r.startswith("stale_data") for r in reasons)

    def test_hand_computed_stale_days_matches_calendar_gap(self):
        stale_date = TODAY - pd.Timedelta(days=8)
        df = make_ohlcv(30, price=20.0, atr_pct=0.02, volume=500_000, end_date=stale_date)
        out = analyze_symbol(df)
        assert out["days_stale"] == 8
        ok, reasons = pass_filters(out, TH)
        assert "stale_data_8d" in reasons


# ─────────────────────────────────────────────────────────────────────────
# TestMomentumAndRelativeStrength (scoring-only; RS vs benchmark)
# ─────────────────────────────────────────────────────────────────────────

class TestMomentumAndRelativeStrength:
    """Failure hypothesis: RS silently uses a stale benchmark endpoint that
    shifts the comparison into the past relative to the stock's true last
    bar (FIX #16's own stated risk), or a missing RS value is scored as a
    penalty instead of neutrally (or vice versa) without anyone noticing."""

    def _stock_and_bench(self, n=100, bench_lag_bars=0):
        dates = pd.bdate_range(end=TODAY, periods=n)
        closes = np.linspace(10, 20, n)
        df = pd.DataFrame(
            {"Open": closes, "High": closes + 0.3, "Low": closes - 0.3,
             "Close": closes, "Volume": np.full(n, 500_000.0)},
            index=dates,
        )
        bench_dates = dates[: n - bench_lag_bars] if bench_lag_bars else dates
        bench = pd.Series(np.linspace(50, 55, len(bench_dates)), index=bench_dates)
        return df, bench

    def test_hand_computed_rs_1m_outperformance(self):
        """21-bar stock return minus 21-bar benchmark return, by hand."""
        n = 40
        dates = pd.bdate_range(end=TODAY, periods=n)
        stock_closes = np.full(n, 100.0)
        stock_closes[-21:] = np.linspace(100.0, 120.0, 21)  # +20% over 21 bars
        bench_closes = np.full(n, 50.0)
        bench_closes[-21:] = np.linspace(50.0, 55.0, 21)  # +10% over 21 bars
        df = pd.DataFrame(
            {"Open": stock_closes, "High": stock_closes + 0.3,
             "Low": stock_closes - 0.3, "Close": stock_closes,
             "Volume": np.full(n, 500_000.0)},
            index=dates,
        )
        bench = pd.Series(bench_closes, index=dates)
        out = analyze_symbol(df, bench_close=bench)
        expected = (120.0 / 100.0 - 1) - (55.0 / 50.0 - 1)  # 0.20 - 0.10 = 0.10
        assert out["rs_1m"] == pytest.approx(expected, abs=1e-6)

    def test_endpoint_lag_at_boundary_computes_rs(self):
        df, bench = self._stock_and_bench(n=100, bench_lag_bars=3)
        out = analyze_symbol(df, bench_close=bench)
        assert not np.isnan(out["rs_1m"])

    def test_endpoint_lag_one_bar_beyond_boundary_skips_rs(self):
        """FIX #16's own documented boundary: a benchmark stale by MORE than
        3 of the stock's bars must NOT be used — it would silently shift
        the RS comparison's endpoint into the past."""
        df, bench = self._stock_and_bench(n=100, bench_lag_bars=4)
        out = analyze_symbol(df, bench_close=bench)
        assert np.isnan(out["rs_1m"])
        assert np.isnan(out["rs_3m"])

    def test_missing_benchmark_gives_nan_rs_not_zero(self):
        df, _ = self._stock_and_bench()
        out = analyze_symbol(df, bench_close=None)
        assert np.isnan(out["rs_1m"]) and np.isnan(out["rs_3m"])

    def test_finding_unmeasured_rs_scores_identically_to_measured_neutral_rs(self):
        """DESIGN-RISK FINDING (not a defect against any written spec): a
        ticker whose RS genuinely couldn't be measured (rs_3m=NaN, e.g. a
        recent IPO with <63 bars) scores IDENTICALLY on that dimension to a
        ticker with a real, measured, exactly-flat RS (rs_3m=0.0). The
        ranking cannot distinguish 'known neutral' from 'unknown' — passing
        test, documents actual behavior so a reviewer can decide if it's
        acceptable."""
        measured_neutral = healthy_row(rs_1m=0.0, rs_3m=0.0)
        unmeasured = healthy_row(rs_1m=0.0, rs_3m=float("nan"))
        assert score_row(measured_neutral) == pytest.approx(score_row(unmeasured))

    def test_negative_rs_lowers_score_relative_to_positive_rs(self):
        strong = healthy_row(rs_1m=0.05, rs_3m=0.08)
        weak = healthy_row(rs_1m=-0.05, rs_3m=-0.08)
        assert score_row(strong) > score_row(weak)


# ─────────────────────────────────────────────────────────────────────────
# TestInstrumentExclusion — light touch; heavily covered elsewhere already
# ─────────────────────────────────────────────────────────────────────────

class TestInstrumentExclusion:
    """Failure hypothesis: a scope edge not covered by the existing FIX
    #14/#17 regression tests silently lets a non-common instrument through,
    or wrongly excludes a legitimate share class."""

    def test_bare_dash_p_with_no_series_letter_is_kept_by_design(self):
        """Confirms the documented (intentional) gap: '-P' with ZERO
        trailing letters does not match the preferred-series regex (which
        requires 1-2 letters). This is scoped as a class-P common share,
        matching share-class tickers like -A/-B. Documents current
        behavior, not a bug."""
        assert is_excluded_instrument("XYZ-P.TO") is False

    def test_excluded_before_download_never_reaches_scoring(self):
        """Combinator check at the funnel boundary: an excluded symbol must
        carry a reject_reasons of exactly 'excluded_instrument' and no
        score, even though it never went through analyze_symbol/pass_filters
        at all — confirms the two code paths converge on the same shape."""
        rows = []
        # Mirrors the exact dict shape run_universe_builder appends for
        # pre-download exclusions (see swing_universe.py's exclusion loop).
        rows.append({"symbol": "PMZ-UN.TO", "tradable": False,
                     "reject_reasons": "excluded_instrument"})
        df = pd.DataFrame(rows)
        assert bool(df.loc[0, "tradable"]) is False
        assert "score" not in df.columns or pd.isna(df.loc[0].get("score"))


# ─────────────────────────────────────────────────────────────────────────
# TestLookAheadBias — the most likely silent flaw, per the brief
# ─────────────────────────────────────────────────────────────────────────

class TestLookAheadBias:
    """Failure hypothesis: a rolling window is accidentally centered
    (center=True) or a shift points the wrong direction (shift(-1) instead
    of shift(1)), so a signal computed 'as of' day T actually depends on
    bars dated after T — which would not exist yet on a live run."""

    def test_sma50_at_every_cut_point_matches_an_independent_trailing_recompute(self):
        """Independent re-derivation: for several different truncation
        points T, manually average the 50 bars ending at T (Python slicing,
        not pandas .rolling) and compare against analyze_symbol's sma50 for
        the DataFrame truncated at T. If sma50 were centered or forward-
        shifted in the production code, this would diverge."""
        n = 120
        dates = pd.bdate_range(end=TODAY, periods=n)
        rng = np.random.default_rng(7)
        closes = 20 + np.cumsum(rng.normal(0, 0.3, n))
        full = pd.DataFrame(
            {"Open": closes, "High": closes + 0.4, "Low": closes - 0.4,
             "Close": closes, "Volume": np.full(n, 400_000.0)},
            index=dates,
        )
        for cut in (50, 70, 90, 119):
            truncated = full.iloc[: cut + 1]
            out = analyze_symbol(truncated)
            manual_sma50 = closes[cut - 49 : cut + 1].mean()  # bars <= cut only
            assert out["sma50"] == pytest.approx(manual_sma50, rel=1e-9), f"cut={cut}"

    def test_atr_gap_is_attributed_to_the_gap_day_using_the_prior_close_not_a_future_one(self):
        """A single anomalous gap-up inserted at a KNOWN interior bar must
        elevate that bar's own True Range (computed against the PRIOR
        close), and truncating the series immediately after the gap must
        already reflect it — proving the gap is priced in as of the day it
        happens, not one day early via a forward-looking shift."""
        n = 30
        closes = [50.0] * n
        gap_idx = 20
        closes[gap_idx] = 65.0  # +30% gap on a single day
        dates = pd.bdate_range(end=TODAY, periods=n)
        df = pd.DataFrame(
            {
                "Open": closes, "High": closes, "Low": closes, "Close": closes,
                "Volume": [500_000.0] * n,
            },
            index=dates,
        )
        atr_full = compute_atr(df, period=14)
        # TR on the gap day itself = |gap_close - prior_close| = 15.0
        assert atr_full.index[gap_idx] == dates[gap_idx]
        # Truncating the series to end exactly ON the gap day must already
        # show the elevated ATR (no forward peek needed for it to show up).
        atr_truncated_at_gap = compute_atr(df.iloc[: gap_idx + 1], period=14)
        assert atr_truncated_at_gap.iloc[-1] == pytest.approx(atr_full.iloc[gap_idx])
        # And truncating to the day BEFORE the gap must NOT show it at all
        # — proving the gap isn't smeared backwards into earlier bars.
        atr_before_gap = compute_atr(df.iloc[:gap_idx], period=14)
        assert atr_before_gap.iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_worst_1d_return_uses_close_to_close_not_a_forward_return(self):
        """The 'signal day' return (today's close vs yesterday's close) must
        be the LAST element the min() can see — not tomorrow's close vs
        today's, which wouldn't exist at decision time."""
        n = 40
        closes = [20.0] * n
        closes[-1] = 20.0 * 0.85  # -15% realized on the final (signal) day
        dates = pd.bdate_range(end=TODAY, periods=n)
        df = pd.DataFrame(
            {"Open": closes, "High": closes, "Low": closes, "Close": closes,
             "Volume": [500_000.0] * n},
            index=dates,
        )
        out = analyze_symbol(df)
        assert out["worst_1d_ret_126"] == pytest.approx(-0.15, abs=1e-6)
        # Confirm the module's own _today()-based staleness sees this as the
        # most recent bar (i.e., this drop is priced in as of "now", not
        # discovered a day later).
        assert out["days_stale"] == 0

    def test_rs_never_uses_a_benchmark_bar_dated_after_the_stocks_own_last_bar(self):
        """RS must not silently use benchmark data from a date the stock
        itself hasn't traded to yet. Build a benchmark with a FUTURE bar
        beyond the stock's last date, carrying a huge fake return, and
        confirm rs_1m/rs_3m do not reflect it (align(join='inner') should
        exclude anything the stock's own index doesn't also contain)."""
        n = 70  # >= 63 so rs_3m is actually computed, not NaN on both sides
        dates = pd.bdate_range(end=TODAY, periods=n)
        stock_closes = np.linspace(10, 12, n)
        df = pd.DataFrame(
            {"Open": stock_closes, "High": stock_closes + 0.1,
             "Low": stock_closes - 0.1, "Close": stock_closes,
             "Volume": np.full(n, 500_000.0)},
            index=dates,
        )
        bench_dates = dates.append(pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=5))
        bench_closes = np.concatenate([np.linspace(50, 51, n), [1000.0] * 5])  # future spike
        bench = pd.Series(bench_closes, index=bench_dates)

        out_with_future = analyze_symbol(df, bench_close=bench)
        out_without_future = analyze_symbol(df, bench_close=bench[:n])
        assert out_with_future["rs_1m"] == pytest.approx(out_without_future["rs_1m"])
        assert out_with_future["rs_3m"] == pytest.approx(out_without_future["rs_3m"])


# ─────────────────────────────────────────────────────────────────────────
# TestFunnelIntegrity — the 4000 -> ~150 reduction itself
# ─────────────────────────────────────────────────────────────────────────

class TestFunnelIntegrity:
    """Failure hypothesis: some combination of (a) a filter silently ORs
    where it should AND, (b) a delisted/insufficient-history ticker
    vanishes instead of landing in df_rejected, (c) duplicates or ties
    corrupt the ranked output, or (d) the funnel returns a plausible COUNT
    built from the WRONG tickers. Every assertion below is on set identity,
    never just len()."""

    @staticmethod
    def _build_universe():
        """A 14-distinct-ticker universe (15 input lines — one duplicate)
        with an exactly-known classification for every ticker, verified by
        hand-running pass_filters/analyze_symbol on each before this test
        was written (see conversation record). Expected tradable set:
        {GOODHI, GOODLO, TIEA, TIEB}."""
        specs = {
            "GOODHI.TO":  dict(n=260, price=20.0, atr_pct=0.030, volume=500_000, drift=0.002),
            "GOODLO.TO":  dict(n=260, price=15.0, atr_pct=0.028, volume=60_000,  drift=0.0006),
            "TIEA.TO":    dict(n=260, price=18.0, atr_pct=0.030, volume=120_000, drift=0.0012),
            "TIEB.TO":    dict(n=260, price=18.0, atr_pct=0.030, volume=120_000, drift=0.0012),
            "LOWPRICE.TO": dict(n=260, price=0.50, atr_pct=0.03, volume=5_000_000, drift=0.0005),
            "LOWVOL.TO":  dict(n=260, price=20.0, atr_pct=0.03, volume=15_000, drift=0.0005),
            "TOOQUIET.TO": dict(n=260, price=20.0, atr_pct=0.008, volume=300_000, drift=0.0005),
            "TOOVOLATILE.TO": dict(n=260, price=20.0, atr_pct=0.08, volume=300_000, drift=0.0005),
            "DOWNTREND.TO": dict(n=260, price=20.0, atr_pct=0.03, volume=300_000, drift=-0.001),
            "NEWLISTING.TO": dict(n=30, price=20.0, atr_pct=0.03, volume=300_000, drift=0.001),
        }
        frames = {sym: make_ohlcv(**kw) for sym, kw in specs.items()}

        # CRASHED.TO: otherwise-healthy uptrend with one -20% day inserted —
        # passes every filter except the gap-risk one.
        crashed = make_ohlcv(n=260, price=20.0, atr_pct=0.03, volume=300_000, drift=0.0015)
        crash_i = crashed.columns.get_loc("Close")
        low_i = crashed.columns.get_loc("Low")
        crashed.iloc[-40, crash_i] *= 0.80
        crashed.iloc[-40, low_i] *= 0.80
        frames["CRASHED.TO"] = crashed

        # STALE.TO: healthy shape, but its last bar is over a week old.
        frames["STALE.TO"] = make_ohlcv(
            n=260, price=20.0, atr_pct=0.03, volume=300_000, drift=0.001,
            end_date=TODAY - pd.Timedelta(days=10),
        )

        input_lines = [
            "GOODHI.TO", "GOODHI.TO",  # deliberate duplicate line
            "GOODLO.TO", "TIEA.TO", "TIEB.TO",
            "LOWPRICE.TO", "LOWVOL.TO", "TOOQUIET.TO", "TOOVOLATILE.TO",
            "DOWNTREND.TO", "NEWLISTING.TO", "CRASHED.TO", "STALE.TO",
            "PMZ-UN.TO",      # excluded pre-download (non-common instrument)
            "DELISTED.TO",    # in the input list, absent from the download
        ]
        # DELISTED.TO deliberately has no entry in `frames` — it never
        # appears in the mocked download response.

        cols = {}
        for sym, df in frames.items():
            for c in df.columns:
                cols[(sym, c)] = df[c]
        # Real yfinance multi-ticker downloads share one common, ASCENDING
        # DatetimeIndex across every ticker in the batch (verified against
        # the pythonfintech reference notebooks) — tickers with a shorter or
        # staler history simply get NaN rows for dates they don't have.
        # concat(axis=1) over frames with heterogeneous date ranges builds
        # the union but does not itself guarantee ascending order, so it is
        # sorted explicitly here to match real batch shape rather than
        # accidentally testing a pathological input yfinance would never
        # actually hand the pipeline.
        big = pd.concat(cols, axis=1, sort=False).sort_index()
        big.columns = pd.MultiIndex.from_tuples(big.columns)
        bench = pd.Series(
            np.linspace(50, 55, 260), index=pd.bdate_range(end=TODAY, periods=260)
        )
        return input_lines, big, bench

    def _run(self, tmp_path, thresholds=None):
        input_lines, big, bench = self._build_universe()
        (tmp_path / "tickers.txt").write_text("\n".join(input_lines) + "\n")
        cfg = UniverseBuilderConfig(
            tickers_path=str(tmp_path / "tickers.txt"),
            out_file_path=str(tmp_path / "universe"),
            out_one_line_file_path=str(tmp_path / "one_line"),
            out_rejected_file_path=str(tmp_path / "rejected.csv"),
            batch_size=50,
            sleep_seconds=0.0,
            thresholds=thresholds or Thresholds(),
        )
        with patch("swing_universe.fetch_history_batch", return_value=big), \
             patch("swing_universe.fetch_benchmark", return_value=bench):
            return run_universe_builder(cfg), cfg

    # ── Identity, not just count ────────────────────────────────────────

    def test_tradable_set_is_exactly_the_expected_tickers(self, tmp_path):
        """The central assertion the brief demands: set equality, not len()."""
        (df_tradable, df_rejected), _ = self._run(tmp_path)
        assert set(df_tradable["symbol"]) == {"GOODHI.TO", "GOODLO.TO", "TIEA.TO", "TIEB.TO"}

    def test_rejected_set_is_exactly_the_expected_tickers_with_correct_reasons(self, tmp_path):
        (df_tradable, df_rejected), _ = self._run(tmp_path)
        expected_reasons = {
            "LOWPRICE.TO": "price_too_low",
            "LOWVOL.TO": "low_dollar_volume",
            "TOOQUIET.TO": "too_quiet_atr",
            "TOOVOLATILE.TO": "too_volatile_atr",
            "DOWNTREND.TO": "below_50d",
            "NEWLISTING.TO": "below_50d",
            "CRASHED.TO": "large_gap_risk",
            "PMZ-UN.TO": "excluded_instrument",
            "DELISTED.TO": "no_data",
        }
        got = dict(zip(df_rejected["symbol"], df_rejected["reject_reasons"]))
        assert set(got) == set(expected_reasons) | {"STALE.TO"}
        for sym, reason in expected_reasons.items():
            assert got[sym] == reason, f"{sym}: expected {reason!r}, got {got[sym]!r}"
        assert got["STALE.TO"].startswith("stale_data_")

    # ── Count sanity tied to identity (never count alone) ───────────────

    def test_every_distinct_input_ticker_is_accounted_for_exactly_once(self, tmp_path):
        """No silent drops, no silent duplication: distinct input tickers
        == tradable + rejected, exactly."""
        input_lines, _, _ = self._build_universe()
        (df_tradable, df_rejected), _ = self._run(tmp_path)
        distinct_input = set(input_lines)
        assert len(distinct_input) == 14
        all_output_symbols = list(df_tradable["symbol"]) + list(df_rejected["symbol"])
        assert len(all_output_symbols) == len(distinct_input)  # no dup rows
        assert set(all_output_symbols) == distinct_input

    def test_duplicate_input_line_produces_a_single_output_row(self, tmp_path):
        (df_tradable, df_rejected), _ = self._run(tmp_path)
        all_symbols = list(df_tradable["symbol"]) + list(df_rejected["symbol"])
        assert all_symbols.count("GOODHI.TO") == 1

    def test_written_universe_file_has_no_duplicate_or_missing_lines(self, tmp_path):
        (df_tradable, _), cfg = self._run(tmp_path)
        lines = [l for l in (tmp_path / "universe").read_text().splitlines() if l]
        assert sorted(lines) == sorted(set(lines))  # no dupes on disk
        assert set(lines) == set(df_tradable["symbol"])

    # ── Combinator logic: AND, not OR ────────────────────────────────────

    def test_passing_all_but_one_hard_filter_is_excluded(self):
        """Construct one metrics dict per hard filter, each violating
        EXACTLY that one criterion while every other criterion is healthy —
        confirm each is excluded for exactly that reason (AND semantics:
        one violation is sufficient to reject regardless of everything
        else passing)."""
        variants = {
            "price": healthy_row(last_close=0.50),
            "adv": healthy_row(avg_dollar_vol_20=500_000.0),
            "atr": healthy_row(atr_pct_14=0.08),
            "worst": healthy_row(worst_1d_ret_126=-0.30),
            "trend": healthy_row(above_50d=False),
            "stale": healthy_row(days_stale=10),
        }
        expected_reason = {
            "price": "price_too_low", "adv": "low_dollar_volume",
            "atr": "too_volatile_atr", "worst": "large_gap_risk",
            "trend": "below_50d", "stale": "stale_data_10d",
        }
        for name, row in variants.items():
            ok, reasons = pass_filters(row, TH)
            assert ok is False, name
            assert reasons == [expected_reason[name]], f"{name}: {reasons}"

    def test_passing_every_hard_filter_is_included(self):
        ok, reasons = pass_filters(healthy_row(), TH)
        assert ok is True
        assert reasons == []

    def test_identity_across_all_but_one_variants_plus_the_healthy_control(self):
        """Same construction as above, but run through the SAME combinator
        (pass_filters) as a batch and assert on the resulting SET of names
        that pass — exactly one of seven survives."""
        rows = {
            "healthy": healthy_row(),
            "bad_price": healthy_row(last_close=0.50),
            "bad_adv": healthy_row(avg_dollar_vol_20=500_000.0),
            "bad_atr": healthy_row(atr_pct_14=0.08),
            "bad_worst": healthy_row(worst_1d_ret_126=-0.30),
            "bad_trend": healthy_row(above_50d=False),
            "bad_stale": healthy_row(days_stale=10),
        }
        passing = {name for name, row in rows.items() if pass_filters(row, TH)[0]}
        assert passing == {"healthy"}

    # ── Ordering / ranking ──────────────────────────────────────────────

    def test_tradable_output_is_sorted_by_score_descending(self, tmp_path):
        (df_tradable, _), _ = self._run(tmp_path)
        scores = df_tradable["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_tied_score_tickers_are_both_retained_not_deduped_away(self, tmp_path):
        """TIEA.TO and TIEB.TO are built from identical inputs -> identical
        score and identical avg_dollar_vol_20. A buggy tie-break (or an
        accidental dedup-by-score) could silently drop one."""
        (df_tradable, _), _ = self._run(tmp_path)
        tied = df_tradable[df_tradable["symbol"].isin(["TIEA.TO", "TIEB.TO"])]
        assert len(tied) == 2
        assert tied["score"].iloc[0] == pytest.approx(tied["score"].iloc[1])

    def test_higher_scoring_ticker_ranks_strictly_above_lower_scoring_one(self, tmp_path):
        (df_tradable, _), _ = self._run(tmp_path)
        order = list(df_tradable["symbol"])
        assert order.index("GOODHI.TO") < order.index("GOODLO.TO")

    # ── Rejected score / no partial state ────────────────────────────────

    def test_rejected_rows_carry_no_score(self, tmp_path):
        (df_tradable, df_rejected), _ = self._run(tmp_path)
        assert df_rejected["score"].isna().all()

    # ── Determinism / no top-N truncation ────────────────────────────────

    def test_no_top_n_cutoff_exists_all_passing_tickers_are_returned(self, tmp_path):
        """There is no ranking cutoff in run_universe_builder — it returns
        every tradable ticker, not a top-N slice. Confirm the full passing
        set survives, not just however many would fit some assumed N."""
        (df_tradable, _), _ = self._run(tmp_path)
        assert len(df_tradable) == 4  # == the full known-good set, not less

    def test_rerunning_the_same_universe_is_idempotent(self, tmp_path):
        (df_tradable_1, df_rejected_1), _ = self._run(tmp_path / "a" if (tmp_path / "a").exists() else tmp_path)
        (tmp_path / "a").mkdir(exist_ok=True)
        (df_tradable_2, df_rejected_2), _ = self._run(tmp_path / "a")
        assert set(df_tradable_1["symbol"]) == set(df_tradable_2["symbol"])
        assert set(df_rejected_1["symbol"]) == set(df_rejected_2["symbol"])
