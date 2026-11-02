"""tiny-indicators: a few technical indicators in plain pandas/numpy (no TA-Lib).

All functions take pandas Series/DataFrames and return Series aligned to the input index.
"""
import numpy as np
import pandas as pd

__all__ = ["sma", "ema", "rma", "rsi", "atr", "bollinger", "macd", "donchian", "supertrend", "session_vwap"]


def sma(x, n):
    """Simple moving average."""
    return pd.Series(x).rolling(n).mean()


def ema(x, n):
    """Exponential moving average (span = n)."""
    return pd.Series(x).ewm(span=n, adjust=False).mean()


def rma(x, n):
    """Wilder's smoothing (what RSI/ATR use): EMA with alpha = 1/n."""
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean()


def rsi(close, n=14):
    """Relative Strength Index (Wilder). Flat series -> NaN."""
    close = pd.Series(close)
    d = close.diff()
    up, dn = rma(d.clip(lower=0), n), rma((-d).clip(lower=0), n)
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.where(~((dn == 0) & (up > 0)), 100.0)   # only gains -> 100


def atr(high, low, close, n=14):
    """Average True Range (Wilder)."""
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def bollinger(close, n=20, k=2.0):
    """Bollinger Bands -> DataFrame[mid, upper, lower] (population-free: uses sample std like pandas)."""
    close = pd.Series(close)
    mid, sd = close.rolling(n).mean(), close.rolling(n).std()
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd})


def macd(close, fast=12, slow=26, signal=9):
    """MACD -> DataFrame[macd, signal, histogram]."""
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "histogram": line - sig})


def donchian(high, low, n=20):
    """Donchian channel -> DataFrame[upper, lower]."""
    return pd.DataFrame({"upper": pd.Series(high).rolling(n).max(), "lower": pd.Series(low).rolling(n).min()})


def supertrend(high, low, close, factor=3.0, period=10):
    """Supertrend -> DataFrame[line, direction]; direction is +1 (uptrend) or -1 (downtrend)."""
    h, l, c = (np.asarray(pd.Series(v), dtype=float) for v in (high, low, close))
    a = atr(h, l, c, period).values
    hl2 = (h + l) / 2
    ub, lb = hl2 + factor * a, hl2 - factor * a
    n = len(c)
    fub, flb, line, d = ub.copy(), lb.copy(), np.full(n, np.nan), np.ones(n)
    for i in range(1, n):
        if np.isnan(a[i - 1]):
            continue
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        if d[i - 1] == -1 and c[i] > fub[i - 1]:
            d[i] = 1
        elif d[i - 1] == 1 and c[i] < flb[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
        line[i] = flb[i] if d[i] == 1 else fub[i]
    idx = pd.Series(close).index
    return pd.DataFrame({"line": line, "direction": d}, index=idx)


def session_vwap(high, low, close, volume, session):
    """Volume-weighted average price that resets whenever `session` (any label array) changes group."""
    tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    v = pd.Series(volume).replace(0, np.nan).fillna(1)
    g = pd.Series(np.asarray(session), index=tp.index)
    return (tp * v).groupby(g).cumsum() / v.groupby(g).cumsum()
