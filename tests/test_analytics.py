import math
import pytest
import pandas as pd
from src.analytics import candles_to_dataframe, compute_indicators, summarize_interval


def _make_candles(n: int = 60) -> list:
    """Generate synthetic OHLCV candles using a sine wave for price."""
    import time as _time
    base_ts = int(_time.time()) - n * 60
    candles = []
    for i in range(n):
        ts = base_ts + i * 60
        price = 100.0 + 10.0 * math.sin(i * 0.2)
        candles.append([str(ts), str(price), str(price), str(price + 1.0), str(price - 1.0), "1000.0", str(price * 1000)])
    return candles


def test_candles_to_dataframe_basic():
    candles = _make_candles(30)
    df = candles_to_dataframe(candles)
    assert set(["time", "open", "close", "high", "low", "volume", "turnover", "timestamp"]).issubset(df.columns)
    assert len(df) == 30


def test_candles_to_dataframe_empty():
    with pytest.raises(ValueError):
        candles_to_dataframe([])


def test_compute_indicators_columns():
    df = candles_to_dataframe(_make_candles(60))
    out = compute_indicators(df)
    for col in ["rsi", "macd", "macd_signal", "macd_hist", "atr", "bb_upper", "bb_lower", "bb_mid", "vwap", "ema_fast", "ema_slow"]:
        assert col in out.columns, f"Missing column: {col}"


def test_rsi_in_range():
    df = candles_to_dataframe(_make_candles(60))
    out = compute_indicators(df)
    rsi_values = out["rsi"].dropna()
    assert len(rsi_values) > 0
    assert (rsi_values >= 0).all()
    assert (rsi_values <= 100).all()


def test_summarize_interval_keys():
    df = candles_to_dataframe(_make_candles(60))
    result = summarize_interval(df, "15min")
    for key in ["interval", "close", "rsi", "trend_bias", "commentary", "atr", "bb_upper", "bb_lower", "vwap"]:
        assert key in result, f"Missing key: {key}"
    assert result["interval"] == "15min"
    assert result["trend_bias"] in ("bullish", "bearish", "neutral-to-bullish", "neutral-to-bearish")
