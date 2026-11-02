# tiny-indicators

A handful of technical indicators in plain pandas/numpy — no TA-Lib, no compiled dependencies. Small enough to read in five minutes.

SMA · EMA · Wilder RSI · ATR · Bollinger Bands · MACD · Donchian channel · Supertrend · session-anchored VWAP.

```python
import pandas as pd
from tiny_indicators import rsi, supertrend, bollinger

df = pd.read_csv("bars.csv")            # columns: open, high, low, close, volume
df["rsi"] = rsi(df.close, 14)
st = supertrend(df.high, df.low, df.close, factor=3.0, period=10)
df["trend"] = st.direction             # +1 uptrend / -1 downtrend
bands = bollinger(df.close, 20, 2)     # DataFrame: mid, upper, lower
```

## Run the tests
```bash
python -m unittest
```

## Notes
- Extracted from a charting app I built for myself; written to be readable first, fast second.
- Educational code, not financial advice, and not a trading system.
- MIT licensed.
