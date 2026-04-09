import math
import pytest
import pandas as pd
from src.analytics import candles_to_dataframe, compute_indicators, summarize_interval, summarize_multi_timeframe


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
    assert result["trend_bias"] in ("bullish", "bearish", "neutral-to-bullish", "neutral-to-bearish", "neutral")


def test_summarize_multi_timeframe_primary_drives_bias():
    """1h bullish + 15m bearish should still yield bullish (1h is primary)."""
    snapshots = [
        {"interval": "1hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "15min", "trend_bias": "bearish", "volatility": "normal"},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert result["overall_bias"] == "bullish"
    assert result["strength"] == "weak"
    assert "Wait" not in result["entry_hint"]


def test_summarize_multi_timeframe_both_agree():
    """When both intervals agree, strength should be strong."""
    snapshots = [
        {"interval": "1hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "15min", "trend_bias": "bullish", "volatility": "normal"},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert result["overall_bias"] == "bullish"
    assert result["strength"] == "strong"


def test_summarize_multi_timeframe_never_says_wait():
    """entry_hint should never contain 'Wait' for any combination."""
    combos = [
        ("bullish", "bearish"), ("bearish", "bullish"),
        ("neutral", "neutral"), ("bullish", "neutral"),
        ("bearish", "neutral"), ("neutral", "bullish"),
        ("neutral-to-bullish", "bearish"),
    ]
    for primary_bias, secondary_bias in combos:
        snapshots = [
            {"interval": "1hour", "trend_bias": primary_bias, "volatility": "normal"},
            {"interval": "15min", "trend_bias": secondary_bias, "volatility": "normal"},
        ]
        result = summarize_multi_timeframe(snapshots)
        assert "Wait" not in result["entry_hint"], f"Got 'Wait' for {primary_bias}/{secondary_bias}"


def test_summarize_multi_timeframe_has_strength_field():
    snapshots = [
        {"interval": "1hour", "trend_bias": "bearish", "volatility": "elevated"},
        {"interval": "15min", "trend_bias": "neutral", "volatility": "normal"},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert "strength" in result
    assert result["strength"] in ("strong", "moderate", "weak")
