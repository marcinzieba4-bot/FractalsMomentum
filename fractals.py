"""
Fractal Dimension Analysis Framework
=====================================
Implements the fractal dimension (D) calculation based on the ratio of
total path length to net displacement over rolling windows — analogous to
the Hurst exponent / R/S analysis approach.

R port by user → Python implementation.

Formula (per window of length k):
    r_t   = log(close_t / close_{t-1})           # daily log return
    R_k   = log(close_t / close_{t-k})           # k-period log return
    N_k   = sum(|r|, k periods) / (|R_k| / k)   # normalised path length
    D_k   = log(N_k) / log(k)                    # fractal dimension
"""

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_yahoo(ticker: str, period_days: int = 1000) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance (no external library needed).

    Returns a DataFrame with columns: open, high, low, close, volume
    indexed by date.
    """
    end = datetime.today()
    start = end - timedelta(days=period_days)

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{ticker}"
        f"?period1={int(start.timestamp())}"
        f"&period2={int(end.timestamp())}"
        "&interval=1d"
        "&events=history"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()

    data = r.json()
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quotes = result["indicators"]["quote"][0]
    adjclose = result["indicators"]["adjclose"][0]["adjclose"]

    df = pd.DataFrame(
        {
            "open":   quotes["open"],
            "high":   quotes["high"],
            "low":    quotes["low"],
            "close":  adjclose,          # adjusted close — mirrors R convention
            "volume": quotes["volume"],
        },
        index=pd.to_datetime(timestamps, unit="s").normalize(),
    )
    df.index.name = "date"
    df = df.dropna(subset=["close"]).sort_index()
    return df


# ---------------------------------------------------------------------------
# Core fractal calculation
# ---------------------------------------------------------------------------

def calculate_fractals(df: pd.DataFrame, n: int = 65) -> pd.DataFrame:
    """
    Calculate fractal dimension columns, mirroring the R code exactly.

    New columns added
    -----------------
    r      : daily log return
    R      : n-period log return
    R_2    : (2n)-period log return
    N      : normalised path length over n periods
    N_2    : normalised path length over 2n periods
    D      : fractal dimension (n-period window)
    D_2    : fractal dimension (2n-period window)

    Parameters
    ----------
    df : DataFrame with a 'close' column, sorted by date ascending
    n  : lookback period (default 65, ~quarterly)

    Returns
    -------
    DataFrame trimmed to rows where all fractal columns are valid,
    starting from row (n+1) in 0-based indexing — matches R's
    Fractals[(n+1):length(Fractals$date_), ...]
    """
    out = df.copy()

    # --- log returns ---
    out["r"]   = np.log(out["close"] / out["close"].shift(1))
    out["R"]   = np.log(out["close"] / out["close"].shift(n))
    out["R_2"] = np.log(out["close"] / out["close"].shift(n * 2))

    # --- rolling sum of |r| ---
    abs_r = out["r"].abs()
    roll_n   = abs_r.rolling(window=n,     min_periods=n).sum()
    roll_2n  = abs_r.rolling(window=n * 2, min_periods=n * 2).sum()

    # --- normalised path length N ---
    out["N"]   = roll_n   / (out["R"].abs()   / n)
    out["N_2"] = roll_2n  / (out["R_2"].abs() / (n * 2))

    # --- fractal dimension D ---
    out["D"]   = np.log(out["N"])   / np.log(n)
    out["D_2"] = np.log(out["N_2"]) / np.log(n * 2)

    # --- trim: drop leading NaNs, then skip first n rows (R: [(n+1):...]) ---
    out = out.dropna(subset=["D", "D_2"])
    out = out.iloc[n:]          # matches R's (n+1)-based 1-indexed slice

    return out


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_fractals(df: pd.DataFrame, ticker: str = "", n: int = 65) -> None:
    """
    Four-panel chart:
      1. Adjusted close price
      2. Fractal dimension D  (n-period)
      3. Fractal dimension D_2 (2n-period)
      4. Overlay of D and D_2
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"{ticker} – Fractal Dimension Analysis  (n={n})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(4, 1, hspace=0.45)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # Panel 1: price
    ax1.plot(df.index, df["close"], color="steelblue", linewidth=1.2)
    ax1.set_ylabel("Adj Close")
    ax1.set_title("Price")
    ax1.grid(True, alpha=0.3)

    # Panel 2: D (n-period)
    ax2.plot(df.index, df["D"], color="darkorange", linewidth=1.0, label=f"D (n={n})")
    ax2.axhline(1.5, color="red",   linestyle="--", linewidth=0.8, label="D=1.5 (random walk)")
    ax2.axhline(1.0, color="green", linestyle="--", linewidth=0.8, label="D=1.0 (trend)")
    ax2.set_ylabel("D")
    ax2.set_title(f"Fractal Dimension D  (window = {n})")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Panel 3: D_2 (2n-period)
    ax3.plot(df.index, df["D_2"], color="purple", linewidth=1.0, label=f"D_2 (n={n*2})")
    ax3.axhline(1.5, color="red",   linestyle="--", linewidth=0.8, label="D=1.5 (random walk)")
    ax3.axhline(1.0, color="green", linestyle="--", linewidth=0.8, label="D=1.0 (trend)")
    ax3.set_ylabel("D_2")
    ax3.set_title(f"Fractal Dimension D_2  (window = {n * 2})")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Panel 4: overlay
    ax4.plot(df.index, df["D"],   color="darkorange", linewidth=1.0, label=f"D  (n={n})",     alpha=0.85)
    ax4.plot(df.index, df["D_2"], color="purple",     linewidth=1.0, label=f"D_2 (n={n*2})", alpha=0.85)
    ax4.axhline(1.5, color="red",   linestyle="--", linewidth=0.8)
    ax4.axhline(1.0, color="green", linestyle="--", linewidth=0.8)
    ax4.set_ylabel("D")
    ax4.set_title("D vs D_2 overlay")
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.savefig(f"{ticker}_fractals.png", dpi=150, bbox_inches="tight")
    print(f"Chart saved → {ticker}_fractals.png")
    plt.show()


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def fractal_summary(df: pd.DataFrame, n: int = 65) -> None:
    """Print a brief statistical summary of the fractal dimension columns."""
    cols = ["D", "D_2"]
    print("\n" + "=" * 55)
    print(f"  Fractal Dimension Summary   (n={n}, 2n={n*2})")
    print("=" * 55)
    stats = df[cols].describe().T
    stats.columns = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    print(stats.to_string())
    print()

    last = df[cols].iloc[-1]
    print(f"  Latest values  ({df.index[-1].date()})")
    print(f"    D   = {last['D']:.4f}")
    print(f"    D_2 = {last['D_2']:.4f}")

    def interpret(d):
        if d < 1.2:
            return "strong trend  (low fractal dimension)"
        elif d < 1.4:
            return "mild trend"
        elif d < 1.6:
            return "near-random walk"
        else:
            return "noisy / mean-reverting"

    print(f"\n  D   interpretation : {interpret(last['D'])}")
    print(f"  D_2 interpretation : {interpret(last['D_2'])}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TICKER = "TSLA"
    N      = 65          # ~quarterly lookback
    DAYS   = 1500        # history to download (~4 years)

    print(f"Downloading {TICKER} daily prices …")
    raw = fetch_yahoo(TICKER, period_days=DAYS)
    print(f"  Rows fetched : {len(raw)}  ({raw.index[0].date()} → {raw.index[-1].date()})")

    print(f"Calculating fractals (n={N}) …")
    fractals = calculate_fractals(raw, n=N)
    print(f"  Rows after trimming : {len(fractals)}")

    fractal_summary(fractals, n=N)

    print("Plotting …")
    plot_fractals(fractals, ticker=TICKER, n=N)

    # Expose the result as CSV for downstream use
    out_path = f"{TICKER}_fractals.csv"
    fractals.to_csv(out_path)
    print(f"Data saved → {out_path}")
