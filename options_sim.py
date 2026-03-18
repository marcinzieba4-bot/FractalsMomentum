"""
Options Simulation on Fractal Signals
======================================
Usage:  python options_sim.py AAPL
        python options_sim.py NVDA

For each fractal signal on the ticker (run on stock / XLK ratio):
  - BUY  signal → buy ATM CALL
  - SELL signal → buy ATM PUT

Strike  : nearest standard increment to stock price AT signal date
Expiry  : signal_date + HOLD_WEEKS (default 4w), shifted to nearest Friday
Pricing : Black-Scholes using 30-day realised vol at signal date
          risk-free rate hardcoded at 5% annualised

Prices shown are ACTUAL market prices (split-adjusted by reversing Yahoo
cumulative adjustment factor, so AAPL 2017 shows ~$150, not $38).
"""

import sys
import math
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from scipy.stats import norm

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from fractals import calculate_fractals
from signals import detect_signals, FORWARD_WEEKS, LOW_THRESH, HIGH_THRESH

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N          = 65
DAYS       = 4000       # ~11 years of history
HOLD_WEEKS = 4          # option holding period
RISK_FREE  = 0.05       # annual risk-free rate
VOL_WINDOW = 30         # days to compute realised vol

SECTOR_ETF = {
    "XLK": ["NVDA","AAPL","MSFT","AVGO","ORCL","CRM","AMD","NOW","INTU",
             "IBM","TXN","QCOM","AMAT","ADBE","KLAC","LRCX","PANW","SNPS"],
    "XLY": ["RCL","TSLA","AMZN","HD","MCD","NKE","BKNG","CMG","ROST","ABNB"],
    "XLV": ["LLY","UNH","JNJ","ABBV","MRK","ABT","BMY","GILD","SYK","CI","HUM"],
    "XLF": ["JPM","V","MA","BAC","GS","MS","AXP","BLK","PGR"],
    "XLC": ["GOOGL","META","NFLX","DIS","TMUS","T","VZ","CMCSA"],
    "XLI": ["GE","HON","UPS","CAT","LMT","RTX","NOC","ETN","GD"],
    "XLP": ["WMT","COST","PG","KO","PEP","PM","MO","MDLZ"],
    "XLE": ["XOM","CVX","COP","EOG","SLB","OXY"],
    "XLB": ["LIN","APD","ECL","SHW","NEM"],
    "XLU": ["NEE","DUK","SO","AEP"],
    "XLRE": ["PLD","AMT","EQIX","SPG"],
}

def get_etf(ticker: str) -> str:
    for etf, members in SECTOR_ETF.items():
        if ticker in members:
            return etf
    return "SPY"  # fallback


# ---------------------------------------------------------------------------
# Data fetch — returns BOTH adjusted (for fractals) and actual prices
# ---------------------------------------------------------------------------

def fetch_prices(ticker: str, period_days: int = DAYS) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      close_adj   — dividend+split adjusted (used for fractal D calculation)
      close_actual— split-adjusted only (actual market price at the time)
                    i.e. reflects real strike-relevant price

    Yahoo's quote[0]['close'] is split-adjusted but NOT dividend-adjusted,
    which gives the actual traded price series (correct for options strikes).
    Yahoo's adjclose further strips dividends — used for return calculations.
    """
    end   = datetime.today()
    start = end - timedelta(days=period_days)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={int(start.timestamp())}&period2={int(end.timestamp())}"
        "&interval=1d&events=history"
    )
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    data   = r.json()["chart"]["result"][0]
    ts     = data["timestamp"]
    q      = data["indicators"]["quote"][0]
    adj    = data["indicators"]["adjclose"][0]["adjclose"]
    dates  = pd.to_datetime(ts, unit="s").normalize()
    df = pd.DataFrame(
        {"close_actual": q["close"], "close_adj": adj},
        index=dates,
    )
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------------------------------------------------------------------------
# Standard option strike increments
# ---------------------------------------------------------------------------

def nearest_strike(price: float) -> float:
    """Round to nearest standard strike increment."""
    if price < 25:
        inc = 0.5
    elif price < 50:
        inc = 1.0
    elif price < 100:
        inc = 2.5
    elif price < 200:
        inc = 5.0
    elif price < 500:
        inc = 10.0
    else:
        inc = 20.0
    return round(round(price / inc) * inc, 2)


def next_friday(date: pd.Timestamp) -> pd.Timestamp:
    """Return date itself if Friday, else advance to next Friday."""
    days_ahead = (4 - date.weekday()) % 7  # 4 = Friday
    if days_ahead == 0:
        days_ahead = 7
    return date + pd.Timedelta(days=days_ahead)


def expiry_date(signal_date: pd.Timestamp, hold_weeks: int = HOLD_WEEKS) -> pd.Timestamp:
    """Nearest Friday on or after signal_date + hold_weeks."""
    target = signal_date + pd.Timedelta(weeks=hold_weeks)
    return next_friday(target)


# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------

def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """Black-Scholes European option price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def realised_vol(prices: pd.Series, date: pd.Timestamp,
                 window: int = VOL_WINDOW) -> float:
    """30-day annualised realised vol ending at `date`."""
    pos = prices.index.searchsorted(date)
    start = max(0, pos - window)
    sub = prices.iloc[start : pos + 1]
    if len(sub) < 5:
        return 0.25  # fallback
    log_rets = np.log(sub / sub.shift(1)).dropna()
    return float(log_rets.std() * math.sqrt(252))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(ticker: str, direct: bool = False) -> None:
    """
    direct=False : fractal signals on stock/ETF ratio  (pair mode, default)
    direct=True  : fractal signals on stock price only (more signals)
    """
    etf = get_etf(ticker)
    mode_label = "stock direct" if direct else f"ratio {ticker}/{etf}"
    print(f"\nTicker : {ticker}  |  Mode : {mode_label}")
    print(f"Fetching {DAYS}-day history …")

    stock_df = fetch_prices(ticker)
    etf_df   = fetch_prices(etf)

    # Align on common dates
    df = pd.concat(
        [stock_df["close_adj"].rename("s_adj"),
         stock_df["close_actual"].rename("s_actual"),
         etf_df["close_adj"].rename("e_adj")],
        axis=1,
    ).dropna()

    if direct:
        # Run fractals on actual stock price
        frac_input = df["s_actual"].rename("close").to_frame()
    else:
        # Run fractals on adjusted ratio (consistent with backtest)
        frac_input = (df["s_adj"] / df["e_adj"]).rename("close").to_frame()

    fractals = calculate_fractals(frac_input, n=N)
    fractals["s_actual"] = df["s_actual"].reindex(fractals.index)

    signals = detect_signals(fractals)
    if signals.empty:
        print("No signals found.")
        return

    print(f"  Found {len(signals)} signal(s)  "
          f"({signals.index[0].date()} → {signals.index[-1].date()})")

    # Attach actual stock price at signal
    signals["stock_price"] = df["s_actual"].reindex(signals.index)

    # Forward returns (on ratio in pair mode, on stock price in direct mode)
    closes = fractals["close"]
    for w in FORWARD_WEEKS:
        col, vals = f"ret_{w}w", []
        for sd in signals.index:
            pos = closes.index.searchsorted(sd)
            tp  = pos + w * 5
            if tp < len(closes):
                r = (closes.iloc[tp] / closes.iloc[pos] - 1) * 100
                if signals.loc[sd, "signal"] == "SELL":
                    r = -r
                vals.append(r)
            else:
                vals.append(float("nan"))
        signals[col] = vals

    # Options pricing
    rows = []
    for sig_date, row in signals.iterrows():
        S      = row["stock_price"]
        K      = nearest_strike(S)
        exp    = expiry_date(sig_date, HOLD_WEEKS)
        T      = (exp - sig_date).days / 365.0
        sigma  = realised_vol(df["s_actual"], sig_date)
        otype  = "call" if row["signal"] == "BUY" else "put"

        price_entry = bs_price(S, K, T, RISK_FREE, sigma, otype)

        # Price at expiry: use actual stock price on/near expiry date
        exp_pos = df["s_actual"].index.searchsorted(exp)
        exp_pos = min(exp_pos, len(df) - 1)
        S_exp   = df["s_actual"].iloc[exp_pos]
        price_exit = bs_price(
            S_exp, K, 0.0, RISK_FREE, sigma, otype
        )  # at expiry T→0 so intrinsic only

        pnl_pct = (price_exit / price_entry - 1) * 100 if price_entry > 0 else float("nan")

        rows.append({
            "signal_date":  sig_date.date(),
            "signal":       row["signal"],
            "stock_price":  round(S, 2),
            "strike":       K,
            "option_type":  otype.upper(),
            "expiry":       exp.date(),
            "IV_used":      round(sigma * 100, 1),
            "entry_price":  round(price_entry, 2),
            "exit_price":   round(price_exit, 2),
            "pnl_%":        round(pnl_pct, 1) if not math.isnan(pnl_pct) else None,
            "ret_4w_ratio": row.get("ret_4w"),
        })

    out = pd.DataFrame(rows).set_index("signal_date")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(f"\n{'='*110}")
    print(f"  {ticker} — Options Simulation  "
          f"(ATM {HOLD_WEEKS}w, BS pricing, σ={VOL_WINDOW}d realised)")
    print(f"{'='*110}")
    print(out.to_string())

    closed = out.dropna(subset=["pnl_%"])
    if not closed.empty:
        print(f"\n  Closed trades: {len(closed)}")
        print(f"  Win rate     : {(closed['pnl_%']>0).mean()*100:.0f}%")
        print(f"  Avg PnL/trade: {closed['pnl_%'].mean():+.1f}%")
        print(f"  Total PnL    : {closed['pnl_%'].sum():+.1f}%")

    csv_path = f"{ticker}_options_sim.csv"
    out.to_csv(csv_path)
    print(f"\n  Saved → {csv_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args   = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags  = [a for a in sys.argv[1:] if a.startswith("--")]
    ticker = args[0].upper() if args else "AAPL"
    direct = "--direct" in flags
    run(ticker, direct=direct)
