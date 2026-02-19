# Financing (Python utilities)

A small “entry-point” repo that hosts my Python finance utilities and shared helpers used across my other projects.

Right now, the main utility here is **Swing Tickers (Universe Builder)**: it takes a raw universe of Canadian tickers and filters/ranks the ones that look suitable for **1–3 week swing trading**, using daily data from Yahoo Finance via `yfinance`.

> **Disclaimer**: This is not financial advice. It’s a screening/ranking tool, not a trading system.

---

## Repo layout

```
.
├─ data/
│  └─ can_tickers                 # one ticker per line (Yahoo format: .TO/.V/.CN/...)
├─ py/
│  └─ swing_tickers.py            # universe builder / screener
├─ requirements.txt
└─ README.md
```

---

## Requirements

- Python 3.10+ recommended
- Packages:
  - `yfinance`
  - `pandas`
  - `numpy`

Install:

```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/macOS)
# .venv\Scripts\activate    # (Windows PowerShell)

pip install -r requirements.txt
```

---

## Swing Tickers (Universe Builder)

### What it does

Given a text file of tickers (one per line), it downloads daily OHLCV history and computes a set of liquidity + volatility + trend/RS checks. It then outputs:

1. **Tradable tickers** (newline-separated) — good for feeding into other scanners/tools
2. **Tradable tickers (comma-separated)** — one-line format (handy for quick copy/paste)
3. **Rejected tickers CSV** — includes *rejection reasons* for diagnostics

### Input format

- One ticker per line
- Yahoo symbols (for Canada commonly: `.TO`, `.V`, `.CN`, sometimes `.NE`, etc.)
- Example:

```
BCE.TO
ENB.TO
CVE.TO
SHOP.TO
```

Your repo already includes a universe file at:

- `data/can_tickers`

---

## How to run

### Option A — run as-is (matches your current `__main__` config)

The script’s `__main__` block uses relative paths like `../data/can_tickers`, so run it **from inside `py/`**.

```bash
cd py
mkdir -p ../out
python swing_tickers.py
```

It will write to:

- `out/can_tickers_swing` (newline-separated tickers)
- `out/can_tickers_swing_one_line` (comma-separated tickers)
- `out/can_tickers_rejected.csv` (rejected tickers + reasons)

It will also print:
- counts (tradable/rejected/total)
- **Top 20 tradable** with key metrics
- a rejection reason breakdown

### Option B — run from code (IDE / other scripts)

```python
from swing_tickers import UniverseBuilderConfig, Thresholds, run_universe_builder

cfg = UniverseBuilderConfig(
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

df_tradable, df_rejected = run_universe_builder(cfg)
print(df_tradable.head())
```

---

## Filtering & scoring logic (high-level)

The screener is designed to find names that are:
- **liquid enough** to trade (proxy: price × volume)
- not **too volatile** for practical swing stops (ATR% constraint)
- not obviously “falling off a cliff” (worst 1-day drop over ~6 months)
- preferably aligned with trend (above key moving averages)
- showing some **relative strength** vs a benchmark (default: `XIU.TO`)
- avoiding stale/illiquid tickers (last bar not too old)

Key thresholds (defaults in `Thresholds`):

- `min_price`  
  Minimum last close (default `1.0`)

- `min_avg_dollar_vol_20`  
  20-day average dollar volume proxy (default `1,000,000`)

- `max_atr_pct_14`  
  14-day ATR normalized by price (default `0.05` = 5%)

- `max_one_day_drop_126`  
  Worst daily return over ~126 trading days (default `-0.15`)

- `require_above_50d`  
  Hard gate: require above SMA50 (default `True`)

- `prefer_above_200d`  
  Soft preference: above SMA200 boosts score (default `True`)

- `max_stale_days`  
  Reject if last bar is older than N trading days (default `5`)

Also included:
- **Volume trend** check (`vol_sma20 > vol_sma50`)
- **Relative strength** vs benchmark (`rs_1m`, `rs_3m`)
- SMA50 slope normalized by mean price (cross-ticker comparable)

---

## Output files

After a run, you’ll have:

- **`out/can_tickers_swing`**
  - one ticker per line (easy to load)

- **`out/can_tickers_swing_one_line`**
  - single line, comma-separated tickers

- **`out/can_tickers_rejected.csv`**
  - rejected symbols + one or more `reject_reasons`

Tip: when tuning thresholds, the rejected CSV is the fastest way to see *why* names are failing.

---

## Notes on data quality

- The script uses `auto_adjust=True` by default to reduce issues from splits/dividends when computing SMA/ATR/returns.
- Yahoo data can be missing or inconsistent for some tickers; the code guards against common failure modes and tracks rejections.

---

## Related projects

This repo is intended to be a shared hub for finance-related tooling.

- Stage Radar: `https://github.com/ChernyshovYuriy/stage-radar`

- Point & Figure System: `https://github.com/ChernyshovYuriy/pfsystem`

- TSX Canadian Stock Screener: `https://github.com/ChernyshovYuriy/stock-scanner`

---

## License

- MIT
