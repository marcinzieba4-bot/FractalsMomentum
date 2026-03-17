"""
Fractal Long/Short Pair Backtest — Top 100 S&P 500 Companies
=============================================================
For each stock:
  - Find its GICS sector → map to Select Sector SPDR ETF
  - Build ratio = stock / sector_ETF
  - Run fractal signal detection on the ratio (n=65)
  - BUY  = long stock / short ETF  (ratio trending up)
  - SELL = short stock / long ETF  (ratio trending down)
  - Forward PnL = ratio return (flipped for SELL)

Output: per-signal table, PnL by holding period, PnL by sector,
        cumulative equity curve sorted by signal date.
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
from datetime import datetime, timedelta

from fractals import fetch_yahoo, calculate_fractals
from signals  import detect_signals, FORWARD_WEEKS, LOW_THRESH, HIGH_THRESH

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Universe: top ~100 SPX constituents by market cap + sector ETF mapping
# ---------------------------------------------------------------------------

SECTOR_ETF = {
    "Information Technology": "XLK",
    "Health Care":            "XLV",
    "Financials":             "XLF",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Industrials":            "XLI",
    "Consumer Staples":       "XLP",
    "Energy":                 "XLE",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Materials":              "XLB",
}

# (ticker, GICS sector)
UNIVERSE = [
    # Information Technology — XLK
    ("NVDA",  "Information Technology"),
    ("AAPL",  "Information Technology"),
    ("MSFT",  "Information Technology"),
    ("AVGO",  "Information Technology"),
    ("ORCL",  "Information Technology"),
    ("CRM",   "Information Technology"),
    ("AMD",   "Information Technology"),
    ("NOW",   "Information Technology"),
    ("INTU",  "Information Technology"),
    ("IBM",   "Information Technology"),
    ("TXN",   "Information Technology"),
    ("QCOM",  "Information Technology"),
    ("AMAT",  "Information Technology"),
    ("ADBE",  "Information Technology"),
    ("KLAC",  "Information Technology"),
    ("LRCX",  "Information Technology"),
    ("PANW",  "Information Technology"),
    ("SNPS",  "Information Technology"),
    ("CDNS",  "Information Technology"),
    ("CSCO",  "Information Technology"),
    # Communication Services — XLC
    ("GOOGL", "Communication Services"),
    ("META",  "Communication Services"),
    ("NFLX",  "Communication Services"),
    ("DIS",   "Communication Services"),
    ("TMUS",  "Communication Services"),
    ("T",     "Communication Services"),
    ("VZ",    "Communication Services"),
    ("CMCSA", "Communication Services"),
    # Consumer Discretionary — XLY
    ("AMZN",  "Consumer Discretionary"),
    ("TSLA",  "Consumer Discretionary"),
    ("HD",    "Consumer Discretionary"),
    ("MCD",   "Consumer Discretionary"),
    ("NKE",   "Consumer Discretionary"),
    ("LOW",   "Consumer Discretionary"),
    ("SBUX",  "Consumer Discretionary"),
    ("TJX",   "Consumer Discretionary"),
    ("BKNG",  "Consumer Discretionary"),
    ("CMG",   "Consumer Discretionary"),
    ("ROST",  "Consumer Discretionary"),
    ("RCL",   "Consumer Discretionary"),
    ("ABNB",  "Consumer Discretionary"),
    # Consumer Staples — XLP
    ("WMT",   "Consumer Staples"),
    ("COST",  "Consumer Staples"),
    ("PG",    "Consumer Staples"),
    ("KO",    "Consumer Staples"),
    ("PEP",   "Consumer Staples"),
    ("PM",    "Consumer Staples"),
    ("MO",    "Consumer Staples"),
    ("MDLZ",  "Consumer Staples"),
    # Health Care — XLV
    ("LLY",   "Health Care"),
    ("UNH",   "Health Care"),
    ("JNJ",   "Health Care"),
    ("ABBV",  "Health Care"),
    ("MRK",   "Health Care"),
    ("TMO",   "Health Care"),
    ("ABT",   "Health Care"),
    ("DHR",   "Health Care"),
    ("BMY",   "Health Care"),
    ("AMGN",  "Health Care"),
    ("GILD",  "Health Care"),
    ("ISRG",  "Health Care"),
    ("VRTX",  "Health Care"),
    ("REGN",  "Health Care"),
    ("SYK",   "Health Care"),
    ("CI",    "Health Care"),
    ("HUM",   "Health Care"),
    # Financials — XLF
    ("BRK-B", "Financials"),
    ("JPM",   "Financials"),
    ("V",     "Financials"),
    ("MA",    "Financials"),
    ("BAC",   "Financials"),
    ("WFC",   "Financials"),
    ("GS",    "Financials"),
    ("MS",    "Financials"),
    ("AXP",   "Financials"),
    ("BLK",   "Financials"),
    ("SPGI",  "Financials"),
    ("MCO",   "Financials"),
    ("ICE",   "Financials"),
    ("CME",   "Financials"),
    ("PGR",   "Financials"),
    # Industrials — XLI
    ("GE",    "Industrials"),
    ("HON",   "Industrials"),
    ("UPS",   "Industrials"),
    ("CAT",   "Industrials"),
    ("DE",    "Industrials"),
    ("LMT",   "Industrials"),
    ("RTX",   "Industrials"),
    ("NOC",   "Industrials"),
    ("ETN",   "Industrials"),
    ("GD",    "Industrials"),
    # Energy — XLE
    ("XOM",   "Energy"),
    ("CVX",   "Energy"),
    ("COP",   "Energy"),
    ("EOG",   "Energy"),
    ("SLB",   "Energy"),
    ("OXY",   "Energy"),
    # Materials — XLB
    ("LIN",   "Materials"),
    ("APD",   "Materials"),
    ("ECL",   "Materials"),
    ("SHW",   "Materials"),
    ("FCX",   "Materials"),
    ("NEM",   "Materials"),
    # Utilities — XLU
    ("NEE",   "Utilities"),
    ("DUK",   "Utilities"),
    ("SO",    "Utilities"),
    ("AEP",   "Utilities"),
    # Real Estate — XLRE
    ("PLD",   "Real Estate"),
    ("AMT",   "Real Estate"),
    ("EQIX",  "Real Estate"),
    ("SPG",   "Real Estate"),
]

N    = 65
DAYS = 1500   # ~4 years of history


# ---------------------------------------------------------------------------
# Cached data fetcher (avoids re-downloading same ticker)
# ---------------------------------------------------------------------------

_cache: dict[str, pd.Series] = {}

def get_close(ticker: str, retries: int = 4) -> pd.Series | None:
    if ticker in _cache:
        return _cache[ticker]
    wait = 2
    for attempt in range(retries):
        try:
            df = fetch_yahoo(ticker, period_days=DAYS)
            s  = df["close"].rename(ticker)
            _cache[ticker] = s
            return s
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
                wait *= 2
            else:
                print(f"    [WARN] Could not fetch {ticker}: {e}")
                return None


# ---------------------------------------------------------------------------
# Forward returns on ratio (pair-aware: SELL flips sign)
# ---------------------------------------------------------------------------

def ratio_forward_returns(
    signals: pd.DataFrame,
    ratio_closes: pd.Series,
    forward_weeks: list = FORWARD_WEEKS,
) -> pd.DataFrame:
    out = signals.copy()
    trd = ratio_closes.index
    for w in forward_weeks:
        td  = w * 5
        col = f"ret_{w}w"
        vals = []
        for sig_date in out.index:
            pos = trd.searchsorted(sig_date)
            tp  = pos + td
            if tp < len(trd):
                r = (ratio_closes.iloc[tp] / ratio_closes.iloc[pos] - 1) * 100
                if out.loc[sig_date, "signal"] == "SELL":
                    r = -r
                vals.append(r)
            else:
                vals.append(np.nan)
        out[col] = vals
    return out


# ---------------------------------------------------------------------------
# Run single stock
# ---------------------------------------------------------------------------

def run_stock(ticker: str, sector: str) -> pd.DataFrame | None:
    etf = SECTOR_ETF[sector]
    s_close = get_close(ticker)
    e_close = get_close(etf)
    if s_close is None or e_close is None:
        return None

    # Align on common dates
    df = pd.concat([s_close, e_close], axis=1).dropna()
    if len(df) < N * 3:
        return None

    ratio = (df[ticker] / df[etf]).rename("close").to_frame()
    ratio["stock_close"] = df[ticker]
    ratio["etf_close"]   = df[etf]

    try:
        fractals = calculate_fractals(ratio, n=N)
    except Exception:
        return None

    signals = detect_signals(fractals)
    if signals.empty:
        return None

    signals = ratio_forward_returns(signals, fractals["close"])
    signals.insert(0, "ticker",  ticker)
    signals.insert(1, "sector",  sector)
    signals.insert(2, "etf",     etf)
    return signals


# ---------------------------------------------------------------------------
# Aggregate PnL helpers
# ---------------------------------------------------------------------------

def pnl_summary(all_signals: pd.DataFrame) -> None:
    ret_cols = [c for c in all_signals.columns if c.startswith("ret_")]
    completed = all_signals.dropna(subset=ret_cols)

    print("\n" + "=" * 75)
    print("  OVERALL PnL SUMMARY  (equal $1 per signal, ratio-neutral return)")
    print("=" * 75)
    print(f"  Total signals   : {len(all_signals)}")
    print(f"  BUY             : {(all_signals['signal']=='BUY').sum()}")
    print(f"  SELL            : {(all_signals['signal']=='SELL').sum()}")
    print(f"  Completed trades: {len(completed)}")
    print()

    summary = pd.DataFrame({
        "n":      completed[ret_cols].count(),
        "mean%":  completed[ret_cols].mean(),
        "median%":completed[ret_cols].median(),
        "win%":   (completed[ret_cols] > 0).mean() * 100,
        "min%":   completed[ret_cols].min(),
        "max%":   completed[ret_cols].max(),
        "total%": completed[ret_cols].sum(),
    })
    summary.index = [c.replace("ret_","").replace("w"," weeks") for c in summary.index]
    print(summary.to_string(float_format=lambda x: f"{x:+.1f}"))

    # By direction
    for direction in ["BUY", "SELL"]:
        sub = completed[completed["signal"] == direction][ret_cols]
        if sub.empty:
            continue
        print(f"\n  {direction} only  (n={len(sub)})")
        print("-" * 55)
        s2 = pd.DataFrame({
            "mean%":  sub.mean(),
            "win%":   (sub > 0).mean() * 100,
            "total%": sub.sum(),
        })
        s2.index = summary.index
        print(s2.to_string(float_format=lambda x: f"{x:+.1f}"))

    print("=" * 75)


def pnl_by_sector(all_signals: pd.DataFrame) -> None:
    ret_cols = [c for c in all_signals.columns if c.startswith("ret_")]
    # Use 4-week return as representative
    col = "ret_4w" if "ret_4w" in ret_cols else ret_cols[0]
    completed = all_signals.dropna(subset=[col])

    print(f"\n  PnL by sector  (holding = {col.replace('ret_','').replace('w',' weeks')})")
    print("-" * 65)
    grp = completed.groupby("sector")[col].agg(
        n="count", mean="mean", win=lambda x: (x > 0).mean() * 100, total="sum"
    ).sort_values("total", ascending=False)
    grp.columns = ["n", "mean%", "win%", "total%"]
    print(grp.to_string(float_format=lambda x: f"{x:+.1f}"))
    print()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_results(all_signals: pd.DataFrame, close_weeks: int = 4) -> None:
    ret_cols = [c for c in all_signals.columns if c.startswith("ret_")]
    close_col = f"ret_{close_weeks}w"
    if close_col not in ret_cols:
        close_col = ret_cols[0]

    completed = all_signals.dropna(subset=ret_cols).copy()
    completed = completed.sort_index()

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Fractal L/S Pair Backtest — Top 100 SPX  |  n={N}  |  "
        f"D<{LOW_THRESH}→D>{HIGH_THRESH}→D<{LOW_THRESH} within 4w",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # ── 1. Mean return by holding period ──
    ax1 = fig.add_subplot(gs[0, 0])
    means  = completed[ret_cols].mean()
    labels = [c.replace("ret_","").replace("w","w") for c in ret_cols]
    colors = ["limegreen" if v >= 0 else "tomato" for v in means]
    ax1.bar(labels, means, color=colors, edgecolor="grey", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Mean return by holding period (%)")
    ax1.set_ylabel("%")
    ax1.grid(True, alpha=0.3, axis="y")

    # ── 2. Win rate by holding period ──
    ax2 = fig.add_subplot(gs[0, 1])
    wins = (completed[ret_cols] > 0).mean() * 100
    ax2.bar(labels, wins, color="steelblue", edgecolor="grey", linewidth=0.5)
    ax2.axhline(50, color="red", linestyle="--", linewidth=0.8, label="50%")
    ax2.set_title("Win rate by holding period (%)")
    ax2.set_ylabel("%")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # ── 3. Cumulative PnL over time (sorted by signal date) ──
    ax3 = fig.add_subplot(gs[1, :])
    for col_lbl in ret_cols:
        sub = completed[col_lbl].dropna().sort_index()
        cum = sub.cumsum()
        lbl = col_lbl.replace("ret_","").replace("w"," weeks")
        ax3.plot(cum.index, cum.values, linewidth=1.2, label=lbl, marker=".", markersize=3)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title("Cumulative PnL over time (sum of ratio returns, equal $1/signal)")
    ax3.set_ylabel("Cumulative %")
    ax3.legend(fontsize=8, ncol=5, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # ── 4. Return distribution (histogram, chosen close period) ──
    ax4 = fig.add_subplot(gs[2, 0])
    buys  = completed[completed["signal"] == "BUY"][close_col].dropna()
    sells = completed[completed["signal"] == "SELL"][close_col].dropna()
    bins  = np.linspace(
        completed[close_col].quantile(0.02),
        completed[close_col].quantile(0.98), 40
    )
    ax4.hist(buys,  bins=bins, alpha=0.6, color="limegreen", label="BUY",  edgecolor="white")
    ax4.hist(sells, bins=bins, alpha=0.6, color="tomato",    label="SELL", edgecolor="white")
    ax4.axvline(0, color="black", linewidth=0.9)
    ax4.set_title(f"Return distribution  ({close_col.replace('ret_','').replace('w',' weeks')})")
    ax4.set_xlabel("%")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── 5. Total PnL by sector (chosen close period) ──
    ax5 = fig.add_subplot(gs[2, 1])
    sector_pnl = (
        completed.dropna(subset=[close_col])
        .groupby("sector")[close_col]
        .agg(total="sum", n="count")
        .sort_values("total")
    )
    colors5 = ["limegreen" if v >= 0 else "tomato" for v in sector_pnl["total"]]
    bars = ax5.barh(sector_pnl.index, sector_pnl["total"],
                    color=colors5, edgecolor="grey", linewidth=0.5)
    for bar, (_, row) in zip(bars, sector_pnl.iterrows()):
        ax5.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"n={int(row['n'])}", va="center", fontsize=7)
    ax5.axvline(0, color="black", linewidth=0.8)
    ax5.set_title(f"Total PnL by sector  ({close_col.replace('ret_','').replace('w',' weeks')})")
    ax5.set_xlabel("Sum of returns (%)")
    ax5.grid(True, alpha=0.3, axis="x")

    out = "backtest_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = []
    unique_etfs = sorted(set(SECTOR_ETF.values()))

    print(f"Pre-fetching {len(unique_etfs)} sector ETFs …")
    for etf in unique_etfs:
        get_close(etf)
        time.sleep(0.3)

    print(f"\nRunning {len(UNIVERSE)} stocks …")
    errors, no_signal = [], []

    for i, (ticker, sector) in enumerate(UNIVERSE, 1):
        etf = SECTOR_ETF[sector]
        print(f"  [{i:3d}/{len(UNIVERSE)}] {ticker:6s} / {etf:4s} …", end=" ", flush=True)
        time.sleep(0.2)   # gentle rate limiting

        result = run_stock(ticker, sector)
        if result is None:
            print("no signal")
            no_signal.append(ticker)
        else:
            print(f"{len(result)} signal(s)")
            all_results.append(result)

    if not all_results:
        print("\nNo signals found across the universe.")
        exit()

    all_signals = pd.concat(all_results).sort_index()
    ret_cols = [c for c in all_signals.columns if c.startswith("ret_")]

    # ── Print full signal table ──
    print("\n" + "=" * 120)
    print("  ALL SIGNALS")
    print("=" * 120)
    display = all_signals[["ticker", "sector", "signal", "R_at_first_dip",
                            "D_at_signal", "close_at_signal"] + ret_cols].copy()
    display["R_at_first_dip"]  = display["R_at_first_dip"].map("{:+.3f}".format)
    display["D_at_signal"]     = display["D_at_signal"].map("{:.4f}".format)
    display["close_at_signal"] = display["close_at_signal"].map("{:.4f}".format)
    for c in ret_cols:
        display[c] = display[c].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "n/a")
    display.index = display.index.date
    display.index.name = "signal_date"
    print(display.to_string())

    # ── Summary tables ──
    pnl_summary(all_signals)
    pnl_by_sector(all_signals)

    # ── Charts ──
    plot_results(all_signals, close_weeks=4)

    # ── Save ──
    all_signals.to_csv("backtest_signals.csv")
    print(f"All signals saved → backtest_signals.csv")
