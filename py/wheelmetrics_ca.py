#!/usr/bin/env python3
"""
wheelmetrics_ca.py — Canadian adaptation of the WheelMetrics Pro Screener schema.

Emits the SAME 19-column CSV as the US export, computed for TSX (.TO) and
TSX Venture (.V) tickers via yfinance.

Key design decision, and the reason this is not a straight port:
the US screener's grades behave like ABSOLUTE thresholds tuned to a universe
containing 45% ROIC semis and 85% YoY revenue growth. Applied unchanged to the
TSX -- which is ~55% financials, energy and materials -- they return an empty
set. So VS/GS/DS here are CROSS-SECTIONAL percentile ranks computed within the
Canadian universe (optionally within sector). A 5 means top decile *in Canada*,
not top decile against NVDA.

Usage:
    python wheelmetrics_ca.py --universe tsx_universe.txt --out screener_ca.csv
    python wheelmetrics_ca.py --universe tsx_universe.txt --optionable mx_optionable.txt
"""

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field, asdict

import pandas as pd
import numpy as np
import yfinance as yf

# ---------------------------------------------------------------------------
# 1. OUTPUT SCHEMA -- byte-identical column order to the US export
# ---------------------------------------------------------------------------

COLUMNS = [
    "Symbol", "StockGrade", "VS", "GS", "DS", "PEUpside%",
    "RevenueGrowth5Y", "EPSGrowth5Y", "FCFToNetIncome", "ROIC",
    "ProfitMargin", "NetDebtToEBITDA", "DebtToEquity", "DivYield",
    "PayoutRatio", "Div5YRAvg", "PEGForward", "RevenueGrowthNY",
    "10YPctChange",
]

# ---------------------------------------------------------------------------
# 2. CANADA-SPECIFIC UNIVERSE GATES
# ---------------------------------------------------------------------------

# Sectors where NetDebtToEBITDA and DebtToEquity are not meaningful.
# Note the US file already shows this: IFS (a Peruvian bank) has a BLANK
# NetDebtToEBITDA. Leverage ratios are suppressed for financials there too.
LEVERAGE_EXEMPT_SECTORS = {"Financial Services", "Financials", "Real Estate"}

# Payout ratio on EPS is garbage for these -- they distribute out of FFO/AFFO,
# distributable cash, or (split-share corps) capital. CTRE in the US file shows
# 90.6% payout and still scores DS=3, which means the source screener is
# already soft-pedalling payout for REITs.
CASHFLOW_PAYOUT_SECTORS = {"Real Estate"}

MIN_PRICE_CAD = 2.00        # sub-$2 names are not wheelable
MIN_AVG_DOLLAR_VOL = 500_000  # CAD/day, 3-month average
MIN_MARKET_CAP = 250_000_000  # CAD

# ---------------------------------------------------------------------------
# 2b. QUALIFICATION GATE -- inferred from the US export
# ---------------------------------------------------------------------------
# The US CSV is not a graded universe; it is the SURVIVOR set of a hard filter
# (~13 rows out of thousands). So every row must satisfy every criterion, which
# means the loosest survivor bounds how strict each criterion can be.
#
# Two of these are near-certain because a survivor sits EXACTLY on a round
# number -- IFS at DebtToEquity 1.00, HPE at ProfitMargin 4.00. Landing exactly
# on a round value in continuous data is what a threshold boundary looks like.
# The rest are inferred from survivors clustering just inside a round bound.
#
# Caveat: 13 rows cannot distinguish "the filter is D/E <= 1.0" from "no D/E
# filter, and the survivors happened to be low-debt". Treat as calibrated
# starting points and tune against your own results.
#
# (column, op, threshold). Nulls PASS -- 4 US survivors have a blank
# RevenueGrowth5Y, so the source filter tolerates missing data rather than
# dropping the row.
QUALIFY = [
    ("DebtToEquity",    "<=", 1.0),   # exact boundary touch (IFS = 1.00)
    ("ProfitMargin",    ">=", 4.0),   # exact boundary touch (HPE = 4.00)
    ("NetDebtToEBITDA", "<=", 3.0),   # max survivor 2.90
    ("RevenueGrowthNY", ">=", 10.0),  # min survivor 10.60
    ("RevenueGrowth5Y", ">=", 5.0),   # min survivor 5.10
    ("ROIC",            ">",  0.0),   # min survivor 3.90
    ("10YPctChange",    ">",  0.0),   # min survivor 72.16
]


def apply_qualification(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Hard filter. Nulls pass. Reports what each criterion actually costs."""
    keep = pd.Series(True, index=df.index)
    for col, op, thr in QUALIFY:
        s = df[col]
        ok = (s <= thr) if op == "<=" else (s >= thr) if op == ">=" else (s > thr)
        ok = ok | s.isna()
        if verbose:
            print(f"  {col:<18} {op:>2} {thr:<7} drops "
                  f"{int((keep & ~ok).sum()):>4} more  "
                  f"({int(keep.sum())} -> {int((keep & ok).sum())})")
        keep &= ok
    return df[keep]


@dataclass
class Row:
    Symbol: str = ""
    StockGrade: str = ""
    VS: object = None
    GS: object = None
    DS: object = None
    PEUpsidePct: object = None
    RevenueGrowth5Y: object = None
    EPSGrowth5Y: object = None
    FCFToNetIncome: object = None
    ROIC: object = None
    ProfitMargin: object = None
    NetDebtToEBITDA: object = None
    DebtToEquity: object = None
    DivYield: object = None
    PayoutRatio: object = None
    Div5YRAvg: object = None
    PEGForward: object = None
    RevenueGrowthNY: object = None
    TenYPctChange: object = None
    # not exported -- used for gating and diagnostics
    sector: str = ""
    price: object = None
    mcap: object = None
    dollar_vol: object = None


# ---------------------------------------------------------------------------
# 3. METRIC EXTRACTION
# ---------------------------------------------------------------------------

def _safe(d, key, default=None):
    v = d.get(key, default)
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _pct(x, nd=2):
    return None if x is None else round(float(x) * 100, nd)


def _cagr(series):
    """CAGR in % from an ordered pandas Series (oldest -> newest)."""
    s = series.dropna()
    if len(s) < 2:
        return None
    first, last, n = s.iloc[0], s.iloc[-1], len(s) - 1
    if first is None or first <= 0 or last <= 0:
        return None
    return round(((last / first) ** (1 / n) - 1) * 100, 2)


def fetch_metrics(ticker: str) -> Row:
    """Pull one ticker. Returns a Row with raw metrics (unscored)."""
    r = Row(Symbol=ticker)
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        print(f"  ! {ticker}: {e}", file=sys.stderr)
        return r

    r.sector = _safe(info, "sector", "") or ""
    r.price = _safe(info, "currentPrice") or _safe(info, "regularMarketPrice")
    r.mcap = _safe(info, "marketCap")
    avg_vol = _safe(info, "averageVolume3Month") or _safe(info, "averageVolume")
    if r.price and avg_vol:
        r.dollar_vol = r.price * avg_vol

    # --- PEUpside% -------------------------------------------------------
    # Reconstruction, NOT a recovered formula. Defined here as the implied
    # re-rating if the trailing multiple converges to the forward multiple:
    #     (trailingPE / forwardPE - 1) * 100
    # This is the definition most consistent with the US file (MU: high
    # trailing, collapsing forward -> +186%; TWLO: forward above trailing
    # -> -33.5%). Swap it for a 5Y-median-PE reversion if you prefer.
    tpe, fpe = _safe(info, "trailingPE"), _safe(info, "forwardPE")
    if tpe and fpe and fpe > 0 and tpe > 0:
        r.PEUpsidePct = round((tpe / fpe - 1) * 100, 2)

    # --- Growth ----------------------------------------------------------
    try:
        fin = t.financials
        if fin is not None and not fin.empty:
            fin = fin.iloc[:, ::-1]  # oldest -> newest
            if "Total Revenue" in fin.index:
                r.RevenueGrowth5Y = _cagr(fin.loc["Total Revenue"])
            if "Diluted EPS" in fin.index:
                r.EPSGrowth5Y = _cagr(fin.loc["Diluted EPS"])
            elif "Net Income" in fin.index:
                r.EPSGrowth5Y = _cagr(fin.loc["Net Income"])
    except Exception:
        pass

    r.RevenueGrowthNY = _pct(_safe(info, "revenueGrowth"))

    # --- Quality ---------------------------------------------------------
    ni = _safe(info, "netIncomeToCommon")
    fcf = _safe(info, "freeCashflow")
    if ni and fcf and ni != 0:
        r.FCFToNetIncome = round(fcf / ni * 100, 2)

    r.ProfitMargin = _pct(_safe(info, "profitMargins"))

    # ROIC = NOPAT / (total debt + equity - cash). yfinance has no ROIC field;
    # returnOnAssets is a poor stand-in for capital-heavy Canadian names.
    ebit = _safe(info, "ebitda")
    try:
        bs = t.balance_sheet
        if bs is not None and not bs.empty and ni:
            col = bs.iloc[:, 0]
            debt = float(col.get("Total Debt", 0) or 0)
            eq = float(col.get("Stockholders Equity", 0) or 0)
            cash = float(col.get("Cash And Cash Equivalents", 0) or 0)
            invested = debt + eq - cash
            if invested > 0:
                r.ROIC = round(ni / invested * 100, 2)
    except Exception:
        pass
    if r.ROIC is None:
        r.ROIC = _pct(_safe(info, "returnOnAssets"))

    # --- Leverage (suppressed for financials, mirroring the US file) ------
    if r.sector not in LEVERAGE_EXEMPT_SECTORS:
        td, tc = _safe(info, "totalDebt"), _safe(info, "totalCash")
        if td is not None and tc is not None and ebit and ebit > 0:
            r.NetDebtToEBITDA = round((td - tc) / ebit, 2)
        d2e = _safe(info, "debtToEquity")
        if d2e is not None:
            r.DebtToEquity = round(d2e / 100, 2)  # yfinance reports as %

    # --- Dividends -------------------------------------------------------
    # yfinance changed dividendYield's units between releases (0.042 vs 4.2),
    # so do NOT disambiguate with a `> 1` test -- that silently turns a real
    # 0.8% yield into 80%. Compute it from rate/price, which is unambiguous,
    # and fall back to the info field only if rate is missing.
    rate = _safe(info, "dividendRate")
    if rate and r.price:
        r.DivYield = round(rate / r.price * 100, 2)
    else:
        dy = _safe(info, "dividendYield")
        if dy:
            tay = _safe(info, "trailingAnnualDividendYield")  # always a fraction
            r.DivYield = round(dy * 100 if (tay and abs(dy - tay) < 1e-6)
                               else dy, 2)

    r.PayoutRatio = _pct(_safe(info, "payoutRatio"))
    d5 = _safe(info, "fiveYearAvgDividendYield")
    r.Div5YRAvg = round(float(d5), 2) if d5 is not None else None

    # Keep the dividend block internally consistent: a non-payer must not
    # carry a stale payout ratio or 5Y average, or the row contradicts DS=0.
    if not r.DivYield or r.DivYield <= 0:
        r.DivYield = r.PayoutRatio = r.Div5YRAvg = None

    # For REITs, EPS payout is meaningless -- recompute against operating cash
    # flow so a 150%-of-EPS distribution isn't scored as a red flag.
    # Gated on DivYield: this must not resurrect a payout ratio on a name the
    # block above just blanked as a non-payer.
    if r.sector in CASHFLOW_PAYOUT_SECTORS and r.DivYield:
        try:
            cf = t.cashflow
            ocf = float(cf.loc["Operating Cash Flow"].iloc[0])
            paid = abs(float(cf.loc["Cash Dividends Paid"].iloc[0]))
            if ocf > 0:
                r.PayoutRatio = round(paid / ocf * 100, 2)
        except Exception:
            pass

    # --- PEG & long-term price change ------------------------------------
    peg = _safe(info, "trailingPegRatio")
    r.PEGForward = round(float(peg), 2) if peg is not None else None

    try:
        h = t.history(period="10y", interval="1mo")["Close"].dropna()
        if len(h) > 24:
            r.TenYPctChange = round((h.iloc[-1] / h.iloc[0] - 1) * 100, 2)
    except Exception:
        pass

    return r


# ---------------------------------------------------------------------------
# 4. SCORING -- cross-sectional, 0-5
# ---------------------------------------------------------------------------

def pct_score(s: pd.Series, higher_better=True) -> pd.Series:
    """Percentile rank -> integer 0-5. NaN stays NaN."""
    r = s.rank(pct=True, ascending=higher_better)
    return (r * 5).round().clip(0, 5)


def score_universe(df: pd.DataFrame, by_sector=False) -> pd.DataFrame:
    """
    VS = Value    : PEUpside%, PEGForward (inverted), FCFToNetIncome
    GS = Growth   : RevenueGrowth5Y, EPSGrowth5Y, RevenueGrowthNY, 10YPctChange
    DS = Dividend : DivYield, PayoutRatio (inverted), Div5YRAvg

    DS=0 when there is no meaningful dividend. That rule is RECOVERED from the
    US file, not invented: all 9 rows with DS=0 have DivYield blank or 0.1,
    and all 4 rows with DS>0 pay a real dividend. Perfect 13/13 correspondence.
    """
    def _score(g):
        g = g.copy()
        vs = pd.concat([
            pct_score(g["PEUpside%"]),
            pct_score(g["PEGForward"], higher_better=False),
            pct_score(g["FCFToNetIncome"]),
        ], axis=1).mean(axis=1, skipna=True)

        gs = pd.concat([
            pct_score(g["RevenueGrowth5Y"]),
            pct_score(g["EPSGrowth5Y"]),
            pct_score(g["RevenueGrowthNY"]),
            pct_score(g["10YPctChange"]),
        ], axis=1).mean(axis=1, skipna=True)

        ds = pd.concat([
            pct_score(g["DivYield"]),
            pct_score(g["PayoutRatio"], higher_better=False),
            pct_score(g["Div5YRAvg"]),
        ], axis=1).mean(axis=1, skipna=True)
        ds = ds.where(g["DivYield"].fillna(0) > 0.5, 0)

        g["VS"] = vs.round().clip(0, 5)
        g["GS"] = gs.round().clip(0, 5)
        g["DS"] = ds.round().clip(0, 5)
        return g

    if by_sector and "sector" in df.columns:
        return df.groupby("sector", group_keys=False).apply(_score)
    return _score(df)


def assign_stars(df: pd.DataFrame) -> pd.DataFrame:
    """
    StockGrade (1-3 stars).

    The US StockGrade is NOT reproducible from the exported columns -- proven,
    not assumed: six rows share (VS=4, GS=4, DS=0) yet split across 2 and 3
    stars, and no single exported column threshold-separates the tiers. It
    depends on an input that was not exported.

    So this is a NEW rule, not a reconstruction. It is a quality-and-
    tradeability composite, which is what a wheel screen actually needs:
    you only sell puts on names you'd accept assignment on.
    """
    q = pd.concat([
        pct_score(df["ROIC"]),
        pct_score(df["ProfitMargin"]),
        pct_score(df["FCFToNetIncome"]),
        pct_score(df["NetDebtToEBITDA"], higher_better=False),
    ], axis=1).mean(axis=1, skipna=True)

    composite = 0.5 * q + 0.3 * df["GS"].fillna(0) + 0.2 * df["VS"].fillna(0)
    df["_composite"] = composite.round(2)

    # Fixed cuts on a 0-5 composite, NOT qcut terciles. qcut would hand 3 stars
    # to exactly one third of the universe by construction -- so a name's grade
    # would change because its PEERS moved, not because it did, and grades
    # would not be comparable across weekly runs. Fixed cuts mean a weak
    # universe correctly produces few 3-star names.
    df["StockGrade"] = pd.cut(
        composite, bins=[-0.01, 2.5, 3.5, 5.01], labels=["★", "★★", "★★★"]
    ).astype(str)
    return df


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True,
                    help="text file, one ticker per line, .TO / .V suffixed")
    ap.add_argument("--optionable", default=None,
                    help="optional file of MX-listed optionable tickers; "
                         "STRONGLY recommended for wheel use")
    ap.add_argument("--out", default="screener_ca.csv")
    ap.add_argument("--by-sector", action="store_true",
                    help="rank within sector instead of across the whole TSX")
    ap.add_argument("--no-qualify", action="store_true",
                    help="skip the hard qualification gate and grade everything")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    tickers = [t.strip() for t in open(args.universe) if t.strip()]

    if args.optionable:
        allowed = {t.strip() for t in open(args.optionable) if t.strip()}
        before = len(tickers)
        tickers = [t for t in tickers if t in allowed]
        print(f"Optionable gate: {before} -> {len(tickers)} tickers")

    rows = []
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t}", file=sys.stderr)
        rows.append(fetch_metrics(t))
        time.sleep(args.sleep)

    df = pd.DataFrame([asdict(r) for r in rows])
    df = df.rename(columns={"PEUpsidePct": "PEUpside%",
                            "TenYPctChange": "10YPctChange"})

    # Liquidity / size gates
    n0 = len(df)
    df = df[(df.price.fillna(0) >= MIN_PRICE_CAD) &
            (df.mcap.fillna(0) >= MIN_MARKET_CAP) &
            (df.dollar_vol.fillna(0) >= MIN_AVG_DOLLAR_VOL)]
    print(f"Liquidity gate: {n0} -> {len(df)}")

    if df.empty:
        print("Empty after gating. Loosen MIN_* constants.", file=sys.stderr)
        return

    # Score across the FULL universe first, then filter. Ranking only the
    # survivors would compress the percentiles into a narrow, already-good set
    # and make VS/GS/DS nearly meaningless. This way a 5 still means top decile
    # of the whole market.
    df = score_universe(df, by_sector=args.by_sector)
    df = assign_stars(df)

    if not args.no_qualify:
        print(f"Qualification gate ({len(df)} in):")
        df = apply_qualification(df)
        print(f"  -> {len(df)} qualify")
        if df.empty:
            print("Nothing qualifies. Loosen QUALIFY.", file=sys.stderr)
            return

    out = df[COLUMNS].copy()
    for c in ["VS", "GS", "DS"]:
        out[c] = out[c].astype("Int64")
    out = out.sort_values(["StockGrade", "VS", "GS"], ascending=False)
    out.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
