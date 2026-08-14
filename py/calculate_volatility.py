import pandas as pd
import yfinance as yf


def calculate_squeeze_and_atr(ticker_symbol, verbose=True):
    # 1. Pull 1 year of daily data
    df = yf.download(ticker_symbol, period="1y", interval="1d", auto_adjust=True)

    # 🔥 FIX: Flatten multi-level columns if they exist from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ---- 2. Bollinger Bands Calculation ----
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["StdDev"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA20"] + (2 * df["StdDev"])
    df["BB_Lower"] = df["SMA20"] - (2 * df["StdDev"])

    # ---- 3. ATR Calculation ----
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (low_shift := df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = df["TR"].rolling(window=14).mean()

    # ---- 4. Keltner Channel ----
    df["KC_Upper"] = df["SMA20"] + (1.5 * df["ATR14"])
    df["KC_Lower"] = df["SMA20"] - (1.5 * df["ATR14"])

    # ---- 5. Identify Squeeze State ----
    df["Squeeze_On"] = (df["BB_Upper"] < df["KC_Upper"]) & (
            df["BB_Lower"] > df["KC_Lower"]
    )

    # Grab the latest day's data row as native Python scalars
    latest = df.iloc[-1]
    current_price = float(latest["Close"])
    current_atr = float(latest["ATR14"])
    is_squeezing = bool(latest["Squeeze_On"])

    # Calculate optimal Stop Loss using 2x ATR
    suggested_stop = current_price - (2 * current_atr)

    # ---- ATR% (ATR normalized by price) so volatility reads consistently ----
    # across price levels/time. Conventional bands: <3% low, 3-6% mid, >6% high.
    atr_pct = (current_atr / current_price) * 100 if current_price else float("nan")
    if atr_pct < 3:
        atr_level = "🟢 LOW"
    elif atr_pct < 6:
        atr_level = "🟡 MID"
    else:
        atr_level = "🔴 HIGH"

    if verbose:
        print("\n" + "=" * 55)
        print(f"🎯 SYSTEM CHECK: {ticker_symbol} @ ${current_price:.2f}")
        print("=" * 55)
        print(f"1. Current 14-Day ATR:      ${current_atr:.2f}  ({atr_pct:.2f}% of price)")
        print(f"   -> Volatility Level:       {atr_level}  (<3% low, 3-6% mid, >6% high)")
        print(f"   -> Noise-Free Stop Loss:   ${suggested_stop:.2f} (Entry - 2x ATR)")
        print("-" * 55)

        if is_squeezing:
            print(
                "2. BOLLINGER SQUEEZE:      🟢 ACTIVE SQUEEZE ON!\n"
                "   -> Meaning: Bollinger Bands have pinched inside the ATR channel.\n"
                "   -> Action: Volatility is dangerously low. Prepare for a massive breakout."
            )
        else:
            print(
                "2. BOLLINGER SQUEEZE:      ⚪ NO SQUEEZE\n"
                "   -> Meaning: Price action is still loose or wildly expanding.\n"
                "   -> Action: Wait for the bands to tighten before executing a swing entry."
            )
        print("=" * 55)

    return {
        "ticker": ticker_symbol,
        "price": current_price,
        "atr": current_atr,
        "atr_pct": atr_pct,
        "atr_level": atr_level,
        "is_squeezing": is_squeezing,
    }


def main():
    with open("../data/can_tickers_swing_universe") as f:
        tickers = [line.strip() for line in f if line.strip()]

    hits = []
    for ticker in tickers:
        try:
            result = calculate_squeeze_and_atr(ticker, verbose=False)
        except Exception as e:
            print(f"⚠️  Skipping {ticker}: {e}")
            continue

        if result["atr_level"] == "🟢 LOW" and result["is_squeezing"]:
            hits.append(result)

    print("\n" + "#" * 55)
    print("LOW volatility + SQUEEZE ON candidates:")
    for hit in hits:
        print(f"  {hit['ticker']:<10} ${hit['price']:.2f}  ATR% {hit['atr_pct']:.2f}%")
    if not hits:
        print("(none found)")
    print("#" * 55)


if __name__ == "__main__":
    # main()
    calculate_squeeze_and_atr("RTX", True)
