"""
Fractal Buy Signal Detection + Forward Return Analysis
=======================================================
Buy signal logic:
  1. D drops below LOW_THRESH  (< 1.30)  — first dip
  2. D recovers above HIGH_THRESH (> 1.37) — recovery
  3. D drops below LOW_THRESH again (< 1.30) — second dip
     within MAX_WEEKS_BETWEEN weeks of the first dip

Signal date = day the second dip is confirmed (D < 1.30 after recovery).
Forward returns measured at close 2, 4, 6, 8, 12 weeks later.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from fractals import fetch_yahoo, calculate_fractals

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
LOW_THRESH         = 1.30   # D threshold for "low" reading
HIGH_THRESH        = 1.37   # D must exceed this between the two dips
MAX_WEEKS_BETWEEN  = 4      # max weeks allowed between first and second dip
FORWARD_WEEKS      = [2, 4, 6, 8, 12]


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def detect_signals(
    df: pd.DataFrame,
    low_thresh: float = LOW_THRESH,
    high_thresh: float = HIGH_THRESH,
    max_weeks: int = MAX_WEEKS_BETWEEN,
) -> pd.DataFrame:
    """
    Scan fractals DataFrame for buy signals.

    State machine per row:
        IDLE        → watching for first dip below low_thresh
        FIRST_DIP   → D went below low_thresh; now wait for recovery above high_thresh
        RECOVERED   → D recovered above high_thresh; now wait for second dip (within max_weeks)

    Returns a DataFrame of signal dates with columns:
        signal_date, close_at_signal,
        first_dip_date, recovery_date,
        D_at_signal
    """
    max_days = max_weeks * 7   # calendar-day window (generous for trading days)

    signals = []
    state = "IDLE"
    first_dip_date = None
    recovery_date  = None

    D = df["D"]

    for date, d_val in D.items():
        if state == "IDLE":
            if d_val < low_thresh:
                state = "FIRST_DIP"
                first_dip_date = date

        elif state == "FIRST_DIP":
            if d_val >= high_thresh:
                state = "RECOVERED"
                recovery_date = date
            # if D never recovers that's fine — stay in FIRST_DIP
            # but reset if we've been here too long without recovery
            # (no cap on recovery wait — only the second dip must arrive within max_weeks of first)

        elif state == "RECOVERED":
            days_since_first = (date - first_dip_date).days
            if days_since_first > max_days:
                # window expired — if D is still low restart from here
                if d_val < low_thresh:
                    state = "FIRST_DIP"
                    first_dip_date = date
                    recovery_date  = None
                else:
                    state = "IDLE"
                    first_dip_date = None
                    recovery_date  = None
            elif d_val < low_thresh:
                # SIGNAL FIRED
                signals.append({
                    "signal_date":    date,
                    "close_at_signal": df.loc[date, "close"],
                    "D_at_signal":    d_val,
                    "first_dip_date": first_dip_date,
                    "recovery_date":  recovery_date,
                })
                # reset — don't allow overlapping signals, restart from IDLE
                state = "IDLE"
                first_dip_date = None
                recovery_date  = None

    return pd.DataFrame(signals).set_index("signal_date") if signals else pd.DataFrame()


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def add_forward_returns(
    signals: pd.DataFrame,
    price_df: pd.DataFrame,
    forward_weeks: list = FORWARD_WEEKS,
) -> pd.DataFrame:
    """
    For each signal date compute the return at close N weeks later.
    Uses actual trading calendar (nth available close after signal).
    """
    closes = price_df["close"]
    trading_dates = closes.index

    result = signals.copy()

    for w in forward_weeks:
        target_days = w * 5   # approx trading days
        col = f"ret_{w}w"
        rets = []
        for sig_date in result.index:
            # find position of signal date in trading calendar
            pos = trading_dates.searchsorted(sig_date)
            target_pos = pos + target_days
            if target_pos < len(trading_dates):
                future_close = closes.iloc[target_pos]
                sig_close    = result.loc[sig_date, "close_at_signal"]
                rets.append((future_close / sig_close - 1) * 100)
            else:
                rets.append(np.nan)   # insufficient history (most recent signal)
        result[col] = rets

    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_signal_table(signals: pd.DataFrame, forward_weeks: list = FORWARD_WEEKS) -> None:
    ret_cols = [f"ret_{w}w" for w in forward_weeks]
    display_cols = ["first_dip_date", "recovery_date", "D_at_signal",
                    "close_at_signal"] + ret_cols

    print("\n" + "=" * 95)
    print("  FRACTAL BUY SIGNALS  (D<1.30 → D>1.37 → D<1.30 within 4 weeks)")
    print("=" * 95)

    df = signals[display_cols].copy()
    df["first_dip_date"] = df["first_dip_date"].dt.date
    df["recovery_date"]  = df["recovery_date"].dt.date
    df["D_at_signal"]    = df["D_at_signal"].map("{:.4f}".format)
    df["close_at_signal"] = df["close_at_signal"].map("${:.2f}".format)

    for col in ret_cols:
        df[col] = df[col].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "  n/a")

    df.index = df.index.date
    df.index.name = "signal_date"
    print(df.to_string())
    print()

    # Summary stats for completed signals
    completed = signals[ret_cols].dropna()
    if len(completed):
        print(f"  Summary  ({len(completed)} completed signals)")
        print("-" * 60)
        summary = pd.DataFrame({
            "mean":    completed.mean(),
            "median":  completed.median(),
            "win%":    (completed > 0).mean() * 100,
            "min":     completed.min(),
            "max":     completed.max(),
        })
        summary.index = [c.replace("ret_", "").replace("w", " weeks") for c in summary.index]
        print(summary.to_string(float_format=lambda x: f"{x:+.1f}"))
    print("=" * 95 + "\n")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_signals(
    fractals: pd.DataFrame,
    signals: pd.DataFrame,
    ticker: str = "",
    n: int = 65,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5], "hspace": 0.08},
    )
    fig.suptitle(
        f"{ticker} – Fractal Buy Signals  "
        f"(D<{LOW_THRESH} → D>{HIGH_THRESH} → D<{LOW_THRESH} within {MAX_WEEKS_BETWEEN}w)",
        fontsize=13, fontweight="bold",
    )

    # ── Panel 1: Price with signal markers ──
    ax1.plot(fractals.index, fractals["close"], color="steelblue",
             linewidth=1.1, label="Adj Close")
    if not signals.empty:
        ax1.scatter(
            signals.index, signals["close_at_signal"],
            marker="^", color="limegreen", s=120, zorder=5,
            label="Buy signal", edgecolors="darkgreen", linewidths=0.8,
        )
    ax1.set_ylabel("Adj Close ($)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Fractal D with thresholds ──
    ax2.plot(fractals.index, fractals["D"], color="darkorange",
             linewidth=1.0, label=f"D (n={n})")
    ax2.axhline(LOW_THRESH,  color="limegreen", linestyle="--",
                linewidth=1.0, label=f"Low={LOW_THRESH}")
    ax2.axhline(HIGH_THRESH, color="tomato",    linestyle="--",
                linewidth=1.0, label=f"Recovery={HIGH_THRESH}")
    ax2.axhline(1.5, color="grey", linestyle=":", linewidth=0.7, label="Random walk 1.5")

    if not signals.empty:
        ax2.scatter(
            signals.index, signals["D_at_signal"],
            marker="^", color="limegreen", s=100, zorder=5,
            edgecolors="darkgreen", linewidths=0.8,
        )

    ax2.set_ylabel("Fractal Dimension D")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=35)

    out_file = f"{ticker}_signals.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {out_file}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TICKER = "TSLA"
    N      = 65
    DAYS   = 1500

    print(f"Downloading {TICKER} …")
    raw = fetch_yahoo(TICKER, period_days=DAYS)

    print(f"Calculating fractals (n={N}) …")
    fractals = calculate_fractals(raw, n=N)

    print("Detecting buy signals …")
    signals = detect_signals(fractals)

    if signals.empty:
        print("No signals found in the available history.")
    else:
        signals = add_forward_returns(signals, fractals)
        print_signal_table(signals)
        plot_signals(fractals, signals, ticker=TICKER, n=N)

        out_path = f"{TICKER}_signals.csv"
        signals.to_csv(out_path)
        print(f"Signals saved → {out_path}")
