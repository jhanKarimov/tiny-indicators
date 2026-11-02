import unittest

import numpy as np
import pandas as pd

from tiny_indicators import atr, bollinger, ema, macd, rsi, sma, supertrend, session_vwap


class IndicatorTests(unittest.TestCase):
    def test_sma_matches_manual_mean(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        self.assertAlmostEqual(sma(s, 3).iloc[-1], 4.0)
        self.assertTrue(np.isnan(sma(s, 3).iloc[1]))

    def test_ema_of_constant_is_constant(self):
        self.assertTrue(np.allclose(ema(pd.Series([5.0] * 20), 9), 5.0))

    def test_rsi_all_gains_is_100(self):
        self.assertAlmostEqual(rsi(pd.Series(np.arange(1, 40, dtype=float))).iloc[-1], 100.0)

    def test_rsi_stays_in_range(self):
        rng = np.random.default_rng(1)
        r = rsi(pd.Series(100 + rng.normal(0, 1, 300).cumsum())).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())

    def test_atr_of_constant_range_equals_range(self):
        h, l, c = pd.Series([11.0] * 30), pd.Series([9.0] * 30), pd.Series([10.0] * 30)
        self.assertAlmostEqual(atr(h, l, c, 14).iloc[-1], 2.0, places=6)

    def test_bollinger_bands_bracket_the_mean(self):
        rng = np.random.default_rng(2)
        b = bollinger(pd.Series(50 + rng.normal(0, 2, 100))).dropna()
        self.assertTrue((b.upper >= b.mid).all() and (b.mid >= b.lower).all())

    def test_macd_hist_is_macd_minus_signal(self):
        m = macd(pd.Series(np.linspace(1, 50, 120) + np.sin(np.arange(120))))
        self.assertTrue(np.allclose(m["histogram"], m["macd"] - m["signal"]))

    def test_supertrend_follows_a_clean_trend(self):
        up = pd.Series(np.linspace(100, 200, 120))
        st = supertrend(up + 1, up - 1, up)
        self.assertEqual(st.direction.iloc[-1], 1)
        down = pd.Series(np.linspace(200, 100, 120))
        st = supertrend(down + 1, down - 1, down)
        self.assertEqual(st.direction.iloc[-1], -1)

    def test_vwap_resets_each_session(self):
        px = pd.Series([10.0, 10.0, 20.0, 20.0])
        v = session_vwap(px, px, px, pd.Series([1, 1, 1, 1]), ["a", "a", "b", "b"])
        self.assertEqual(list(v.round(6)), [10.0, 10.0, 20.0, 20.0])


if __name__ == "__main__":
    unittest.main()
