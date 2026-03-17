"""
Long/Short Pair Fractal Analysis
=================================
Computes a relative price ratio (stock / sector ETF), then runs the full
fractal signal pipeline on that ratio series.

  BUY  signal → stock outperforming sector (go long stock, short ETF)
  SELL signal → stock underperforming sector (go short stock, long ETF)

Pairs defined:
  RCL  / XLY  (Consumer Discretionary)
  AAPL / XLK  (Technology)

Forward returns are measured on the RATIO (= pure relative performance,
independent of broad market direction).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from fractals import fetch_yahoo, calculate_fractals
from signals  import (detect_signals, add_forward_returns,
                      print_signal_table, FORWARD_WEEKS,
                      LOW_THRESH, HIGH_THRESH, MAX_WEEKS_BETWEEN)


def _near_miss_report(df: pd.DataFrame, ticker: str, benchmark: str) -> None:
    """When no signal fires, show how far each recovery window came from firing."""
    from signals import LOW_THRESH, HIGH_THRESH, MAX_WEEKS_BETWEEN
    D = df["D"]
    MAX_DAYS = MAX_WEEKS_BETWEEN * 7
    state = "IDLE"
    first_dip_date = recovery_date = None
    near_misses = []

    for date, d in D.items():
        if state == "IDLE":
            if d < LOW_THRESH:
                state = "FIRST_DIP"; first_dip_date = date
        elif state == "FIRST_DIP":
            if d >= HIGH_THRESH:
                state = "RECOVERED"; recovery_date = date
        elif state == "RECOVERED":
            days = (date - recovery_date).days
            if days > MAX_DAYS:
                near_misses.append({
                    "first_dip":  first_dip_date.date(),
                    "recovery":   recovery_date.date(),
                    "expired":    date.date(),
                    "days_open":  days,
                    "D_at_expiry": round(d, 4),
                })
                state = ("FIRST_DIP" if d < LOW_THRESH else "IDLE")
                first_dip_date = (date if d < LOW_THRESH else None)
                recovery_date = None
            elif d < LOW_THRESH:
                near_misses.append({
                    "first_dip":  first_dip_date.date(),
                    "recovery":   recovery_date.date(),
                    "expired":    date.date(),
                    "days_open":  days,
                    "D_at_expiry": round(d, 4),
                    "NOTE": "FIRED — should not appear here",
                })
                state = "IDLE"; first_dip_date = recovery_date = None

    print(f"  No signals found for {ticker}/{benchmark}.")
    if near_misses:
        nm = pd.DataFrame(near_misses)
        print(f"  Near-miss windows (expired without 2nd dip, n={len(nm)}):")
        print(nm.to_string(index=False))
    else:
        print(f"  D never dipped below {LOW_THRESH} in this history.")
    print()


PAIRS = [
    ("RCL",  "XLY"),
    ("AAPL", "XLK"),
]
N    = 65
DAYS = 1500


# ---------------------------------------------------------------------------
# Build ratio series
# ---------------------------------------------------------------------------

def build_ratio(ticker: str, benchmark: str, period_days: int = DAYS) -> pd.DataFrame:
    """
    Download both legs, align on common trading dates, compute ratio = stock/benchmark.
    Returns a DataFrame with columns: close (ratio), stock_close, bench_close.
    """
    print(f"  Downloading {ticker} and {benchmark} …")
    s = fetch_yahoo(ticker,    period_days=period_days)["close"].rename("stock")
    b = fetch_yahoo(benchmark, period_days=period_days)["close"].rename("bench")

    df = pd.concat([s, b], axis=1).dropna()
    df["close"] = df["stock"] / df["bench"]   # ratio — this is what fractals run on
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Forward returns on the ratio
# ---------------------------------------------------------------------------

def add_ratio_forward_returns(
    signals: pd.DataFrame,
    ratio_df: pd.DataFrame,
    forward_weeks: list = FORWARD_WEEKS,
) -> pd.DataFrame:
    """
    Compute forward returns on the ratio series (relative performance).
    Overwrites any ret_Xw columns from add_forward_returns.
    For SELL signals the ratio return is negated (short stock / long ETF).
    """
    closes = ratio_df["close"]
    trading_dates = closes.index
    result = signals.copy()

    for w in forward_weeks:
        target_days = w * 5
        col = f"ret_{w}w"
        rets = []
        for sig_date in result.index:
            pos = trading_dates.searchsorted(sig_date)
            target_pos = pos + target_days
            if target_pos < len(trading_dates):
                r = (closes.iloc[target_pos] / closes.iloc[pos] - 1) * 100
                # flip sign for SELL (short the ratio)
                if result.loc[sig_date, "signal"] == "SELL":
                    r = -r
                rets.append(r)
            else:
                rets.append(np.nan)
        result[col] = rets

    return result


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_pair_signals(
    ratio_df: pd.DataFrame,
    fractals: pd.DataFrame,
    signals: pd.DataFrame,
    ticker: str,
    benchmark: str,
    n: int = N,
) -> None:
    pair_label = f"{ticker}/{benchmark}"
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"{pair_label} — Fractal Signals on Ratio  "
        f"(D<{LOW_THRESH} → D>{HIGH_THRESH} → D<{LOW_THRESH} within {MAX_WEEKS_BETWEEN}w)\n"
        f"▲ BUY = long {ticker} / short {benchmark}   "
        f"▼ SELL = short {ticker} / long {benchmark}",
        fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 1, hspace=0.12, height_ratios=[1.8, 1, 1.2])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ── Panel 1: ratio ──
    ax1.plot(fractals.index, fractals["close"], color="steelblue",
             linewidth=1.1, label=f"{pair_label} ratio")
    if not signals.empty:
        buys  = signals[signals["signal"] == "BUY"]
        sells = signals[signals["signal"] == "SELL"]
        if not buys.empty:
            ax1.scatter(buys.index, buys["close_at_signal"],
                        marker="^", color="limegreen", s=130, zorder=5,
                        label=f"BUY ({ticker}↑ {benchmark}↓)",
                        edgecolors="darkgreen", linewidths=0.8)
        if not sells.empty:
            ax1.scatter(sells.index, sells["close_at_signal"],
                        marker="v", color="tomato", s=130, zorder=5,
                        label=f"SELL ({ticker}↓ {benchmark}↑)",
                        edgecolors="darkred", linewidths=0.8)
    ax1.set_ylabel(f"{pair_label} ratio")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: individual prices (normalised to 100) ──
    base = ratio_df.index[0]
    norm_s = ratio_df["stock"] / ratio_df["stock"].iloc[0] * 100
    norm_b = ratio_df["bench"] / ratio_df["bench"].iloc[0] * 100
    # align to fractals index
    norm_s = norm_s.reindex(fractals.index)
    norm_b = norm_b.reindex(fractals.index)
    ax2.plot(fractals.index, norm_s, color="steelblue",  linewidth=0.9, label=ticker)
    ax2.plot(fractals.index, norm_b, color="sandybrown", linewidth=0.9, label=benchmark)
    ax2.set_ylabel("Indexed (100)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: fractal D ──
    ax3.plot(fractals.index, fractals["D"], color="darkorange",
             linewidth=1.0, label=f"D (n={n})")
    ax3.axhline(LOW_THRESH,  color="limegreen", linestyle="--",
                linewidth=1.0, label=f"{LOW_THRESH}")
    ax3.axhline(HIGH_THRESH, color="tomato",    linestyle="--",
                linewidth=1.0, label=f"{HIGH_THRESH}")
    ax3.axhline(1.5, color="grey", linestyle=":", linewidth=0.7)
    if not signals.empty:
        buys  = signals[signals["signal"] == "BUY"]
        sells = signals[signals["signal"] == "SELL"]
        if not buys.empty:
            ax3.scatter(buys.index, buys["D_at_signal"],
                        marker="^", color="limegreen", s=100, zorder=5,
                        edgecolors="darkgreen", linewidths=0.8)
        if not sells.empty:
            ax3.scatter(sells.index, sells["D_at_signal"],
                        marker="v", color="tomato", s=100, zorder=5,
                        edgecolors="darkred", linewidths=0.8)
    ax3.set_ylabel("D")
    ax3.set_xlabel("Date")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=35)

    out = f"{ticker}_{benchmark}_signals.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Chart saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_signals = {}

    for ticker, benchmark in PAIRS:
        pair_label = f"{ticker}/{benchmark}"
        print(f"\n{'='*60}")
        print(f"  Pair: {pair_label}")
        print(f"{'='*60}")

        ratio_df = build_ratio(ticker, benchmark)
        print(f"  Common trading days: {len(ratio_df)}  "
              f"({ratio_df.index[0].date()} → {ratio_df.index[-1].date()})")

        fractals = calculate_fractals(ratio_df, n=N)
        print(f"  Rows after fractal trim: {len(fractals)}")

        signals = detect_signals(fractals)

        if signals.empty:
            # Show why: how close did the pattern get?
            _near_miss_report(fractals, ticker, benchmark)
            continue

        signals = add_ratio_forward_returns(signals, fractals)
        print_signal_table(signals)

        plot_pair_signals(ratio_df, fractals, signals, ticker, benchmark, n=N)

        out_csv = f"{ticker}_{benchmark}_signals.csv"
        signals.to_csv(out_csv)
        print(f"  Data saved → {out_csv}")

        all_signals[pair_label] = signals

    # Combined summary across both pairs
    if all_signals:
        print(f"\n{'='*60}")
        print("  COMBINED SUMMARY — all pairs, all signals")
        print(f"{'='*60}")
        combined = pd.concat(all_signals.values())
        ret_cols = [f"ret_{w}w" for w in FORWARD_WEEKS]
        completed = combined[ret_cols].dropna()
        if not completed.empty:
            summary = pd.DataFrame({
                "n":      completed.count(),
                "mean":   completed.mean(),
                "median": completed.median(),
                "win%":   (completed > 0).mean() * 100,
                "min":    completed.min(),
                "max":    completed.max(),
            })
            summary.index = [c.replace("ret_", "").replace("w", " weeks")
                             for c in summary.index]
            print(summary.to_string(float_format=lambda x: f"{x:+.1f}"))
        print()
