"""
Plot detailed pair-trade fractal charts for the last N signals.
Each chart has 3 panels:
  1. Stock vs ETF price (indexed to 100 at start of visible window)
  2. Ratio (stock / ETF) with first-dip / recovery / signal annotations
  3. Fractal D with thresholds and all three key dates marked
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from fractals import fetch_yahoo, calculate_fractals
from signals  import detect_signals, LOW_THRESH, HIGH_THRESH

N         = 65
DAYS      = 1500
LAST_N    = 6          # how many recent signals to plot
LOOKBACK  = 180        # days of context to show around each signal

# ---------------------------------------------------------------------------

def load_pair(ticker: str, etf: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (ratio_fractals_df, raw_df_with_stock_and_etf)."""
    s = fetch_yahoo(ticker, DAYS)["close"].rename("stock")
    e = fetch_yahoo(etf,    DAYS)["close"].rename("etf")
    raw = pd.concat([s, e], axis=1).dropna()
    raw["ratio"] = raw["stock"] / raw["etf"]
    ratio_df = raw["ratio"].rename("close").to_frame()
    fractals = calculate_fractals(ratio_df, n=N)
    fractals["stock"] = raw["stock"]
    fractals["etf"]   = raw["etf"]
    return fractals


def plot_trade(sig_row: pd.Series, sig_date: pd.Timestamp, save_path: str) -> None:
    ticker     = sig_row["ticker"]
    etf        = sig_row["etf"]
    direction  = sig_row["signal"]
    first_dip  = pd.Timestamp(sig_row["first_dip_date"])
    recovery   = pd.Timestamp(sig_row["recovery_date"])
    r_val      = sig_row["R_at_first_dip"]
    d_val      = sig_row["D_at_signal"]

    print(f"  Fetching {ticker}/{etf} …")
    fractals = load_pair(ticker, etf)

    # Window: from LOOKBACK days before first_dip to LOOKBACK/2 after signal
    win_start = first_dip - pd.Timedelta(days=LOOKBACK)
    win_end   = sig_date  + pd.Timedelta(days=max(60, LOOKBACK // 2))
    win = fractals.loc[win_start:win_end].copy()
    if win.empty:
        print(f"    [WARN] No data in window for {ticker}")
        return

    # Index prices to 100 at start of window
    win["stock_idx"] = win["stock"] / win["stock"].iloc[0] * 100
    win["etf_idx"]   = win["etf"]   / win["etf"].iloc[0]   * 100

    color_buy  = "#27ae60"
    color_sell = "#e74c3c"
    sig_color  = color_buy if direction == "BUY" else color_sell

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.8, 1.4, 1.4], "hspace": 0.08})

    ret_cols = {w: f"ret_{w}w" for w in [2, 4, 6, 8, 12]}
    ret_strs = "  |  ".join(
        f"{w}w: {sig_row[c]:+.1f}%"
        if pd.notna(sig_row.get(c)) else f"{w}w: n/a"
        for w, c in ret_cols.items()
    )
    title_dir  = f"{'▲ BUY' if direction=='BUY' else '▼ SELL'}  {ticker}/{etf}"
    title_main = (
        f"{title_dir}   |   Signal: {sig_date.date()}   |   "
        f"R={r_val:+.3f}   D={d_val:.4f}\n"
        f"First dip: {first_dip.date()}   Recovery: {recovery.date()}   "
        f"Signal: {sig_date.date()}\n{ret_strs}"
    )
    fig.suptitle(title_main, fontsize=10, fontweight="bold",
                 color=sig_color, y=0.98)

    # ── Panel 1: individual prices indexed ──
    ax1 = axes[0]
    ax1.plot(win.index, win["stock_idx"], color="steelblue",  lw=1.3, label=ticker)
    ax1.plot(win.index, win["etf_idx"],   color="sandybrown", lw=1.3, label=etf, alpha=0.85)
    ax1.set_ylabel("Price (indexed 100)", fontsize=9)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.25)
    _shade_and_mark(ax1, first_dip, recovery, sig_date, direction, win)

    # ── Panel 2: ratio ──
    ax2 = axes[1]
    ax2.plot(win.index, win["close"], color="mediumpurple", lw=1.3,
             label=f"{ticker}/{etf} ratio")
    ax2.set_ylabel("Ratio", fontsize=9)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.25)
    _shade_and_mark(ax2, first_dip, recovery, sig_date, direction, win)

    # ── Panel 3: fractal D ──
    ax3 = axes[2]
    ax3.plot(win.index, win["D"], color="darkorange", lw=1.1, label=f"D (n={N})")
    ax3.axhline(LOW_THRESH,  color="limegreen", ls="--", lw=1.0, label=f"Low={LOW_THRESH}")
    ax3.axhline(HIGH_THRESH, color="tomato",    ls="--", lw=1.0, label=f"Recovery={HIGH_THRESH}")
    ax3.axhline(1.5, color="grey", ls=":",  lw=0.7, label="1.5 random walk")
    ax3.set_ylabel("Fractal D", fontsize=9)
    ax3.set_xlabel("Date", fontsize=9)
    ax3.legend(fontsize=8, loc="upper right", ncol=2)
    ax3.grid(True, alpha=0.25)
    _shade_and_mark(ax3, first_dip, recovery, sig_date, direction, win,
                    annotate_d=win["D"])

    # Date formatter
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate(rotation=30)

    # Legend for vertical lines
    legend_elements = [
        Line2D([0], [0], color="royalblue",  ls="--", lw=1.5, label="First dip (D<1.30)"),
        Line2D([0], [0], color="goldenrod",  ls="--", lw=1.5, label="Recovery (D>1.37)"),
        Line2D([0], [0], color=sig_color,    ls="-",  lw=2.0,
               label=f"Signal ({'BUY ▲' if direction=='BUY' else 'SELL ▼'})"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=3, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))
    plt.subplots_adjust(bottom=0.08)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {save_path}")


def _shade_and_mark(ax, first_dip, recovery, sig_date, direction, win, annotate_d=None):
    """Draw vertical lines + shaded window on a given axis."""
    sig_color = "#27ae60" if direction == "BUY" else "#e74c3c"
    ymin, ymax = ax.get_ylim()

    # Shade the pattern window (first_dip → signal)
    ax.axvspan(first_dip, sig_date, alpha=0.06, color="gold", zorder=0)

    # Vertical lines
    ax.axvline(first_dip, color="royalblue", ls="--", lw=1.4, alpha=0.8)
    ax.axvline(recovery,  color="goldenrod", ls="--", lw=1.4, alpha=0.8)
    ax.axvline(sig_date,  color=sig_color,   ls="-",  lw=2.0, alpha=0.9)

    # Annotate D values on the D panel only
    if annotate_d is not None:
        for vdate, label, color in [
            (first_dip, "1st dip", "royalblue"),
            (recovery,  "recov.",  "goldenrod"),
            (sig_date,  "signal",  sig_color),
        ]:
            # find nearest index
            idx = annotate_d.index.searchsorted(vdate)
            idx = min(idx, len(annotate_d) - 1)
            actual_date = annotate_d.index[idx]
            if actual_date in annotate_d.index:
                dval = annotate_d.iloc[idx]
                ax.annotate(
                    f"{label}\nD={dval:.3f}",
                    xy=(actual_date, dval),
                    xytext=(8, 12), textcoords="offset points",
                    fontsize=7, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    signals = pd.read_csv("backtest_signals.csv", index_col=0, parse_dates=True)
    signals["first_dip_date"] = pd.to_datetime(signals["first_dip_date"])
    signals["recovery_date"]  = pd.to_datetime(signals["recovery_date"])

    last_trades = signals.tail(LAST_N)
    print(f"Plotting last {LAST_N} trades:\n")
    print(last_trades[["ticker", "etf", "signal", "R_at_first_dip",
                        "D_at_signal", "first_dip_date", "recovery_date"]].to_string())
    print()

    for sig_date, row in last_trades.iterrows():
        fname = f"trade_{sig_date.date()}_{row['ticker']}_{row['etf']}.png"
        plot_trade(row, sig_date, fname)

    print("\nDone.")
