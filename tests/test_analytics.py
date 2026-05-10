import math
import pytest
import pandas as pd
from src.analytics import candles_to_dataframe, compute_indicators, summarize_interval, summarize_multi_timeframe, classify_regime


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


# --- Regime detection tests ---


def test_compute_indicators_has_adx():
    df = candles_to_dataframe(_make_candles(60))
    out = compute_indicators(df)
    assert "adx" in out.columns
    assert "plus_di" in out.columns
    assert "minus_di" in out.columns
    adx_values = out["adx"].dropna()
    assert len(adx_values) > 0
    assert (adx_values >= 0).all()
    assert (adx_values <= 100).all()


def test_compute_indicators_has_bbw():
    df = candles_to_dataframe(_make_candles(60))
    out = compute_indicators(df)
    assert "bbw" in out.columns
    bbw_values = out["bbw"].dropna()
    assert len(bbw_values) > 0
    assert (bbw_values > 0).all()


def test_compute_indicators_has_stochastic():
    df = candles_to_dataframe(_make_candles(60))
    out = compute_indicators(df)
    assert "stoch_k" in out.columns
    assert "stoch_d" in out.columns
    stoch_values = out["stoch_k"].dropna()
    assert len(stoch_values) > 0
    assert (stoch_values >= 0).all()
    assert (stoch_values <= 100).all()


def test_classify_regime_ranging():
    result = classify_regime(adx=15.0, bbw=3.5, atr_pct=1.5)
    assert result["regime"] == "ranging"
    assert result["confidence"] > 0.5


def test_classify_regime_trending():
    result = classify_regime(adx=30.0, bbw=5.0, atr_pct=2.5)
    assert result["regime"] == "trending"
    assert result["confidence"] > 0.5


def test_classify_regime_strong_trend():
    result = classify_regime(adx=40.0, bbw=7.0, atr_pct=3.0)
    assert result["regime"] == "trending"
    assert result["confidence"] > 0.7


def test_classify_regime_squeeze():
    result = classify_regime(adx=12.0, bbw=1.2, atr_pct=0.8)
    assert result["regime"] == "squeeze"


def test_classify_regime_unknown_on_missing_data():
    result = classify_regime(adx=None, bbw=None, atr_pct=None)
    assert result["regime"] == "unknown"


def test_classify_regime_ambiguous():
    result = classify_regime(adx=23.0, bbw=4.0, atr_pct=2.0)
    assert result["regime"] in ("trending", "ranging")
    assert result["confidence"] <= 0.6


def test_summarize_interval_has_regime():
    df = candles_to_dataframe(_make_candles(60))
    result = summarize_interval(df, "15min")
    assert "market_regime" in result
    assert "adx" in result
    assert "bbw" in result
    assert "stoch_k" in result
    assert "stoch_d" in result
    assert result["market_regime"]["regime"] in ("trending", "ranging", "squeeze", "unknown")


def test_summarize_multi_timeframe_has_regime():
    snapshots = [
        {"interval": "1hour", "trend_bias": "neutral", "volatility": "normal",
         "market_regime": {"regime": "ranging", "confidence": 0.8, "details": "test"}},
        {"interval": "15min", "trend_bias": "neutral", "volatility": "normal",
         "market_regime": {"regime": "ranging", "confidence": 0.7, "details": "test"}},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert "market_regime" in result
    assert result["market_regime"] == "ranging"
    assert result["regime_confidence"] > 0.5


def test_summarize_multi_timeframe_squeeze_takes_priority():
    snapshots = [
        {"interval": "1hour", "trend_bias": "bullish", "volatility": "normal",
         "market_regime": {"regime": "ranging", "confidence": 0.7, "details": "test"}},
        {"interval": "15min", "trend_bias": "neutral", "volatility": "normal",
         "market_regime": {"regime": "squeeze", "confidence": 0.8, "details": "test"}},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert result["market_regime"] == "squeeze"


def test_summarize_multi_timeframe_ranging_entry_hint():
    snapshots = [
        {"interval": "1hour", "trend_bias": "neutral", "volatility": "normal",
         "market_regime": {"regime": "ranging", "confidence": 0.8, "details": "test"}},
        {"interval": "15min", "trend_bias": "neutral", "volatility": "normal",
         "market_regime": {"regime": "ranging", "confidence": 0.7, "details": "test"}},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert "RANGE REGIME" in result["entry_hint"]
    assert "mean-reversion" in result["entry_hint"].lower()


def _make_trending_candles(n: int = 60) -> list:
    """Generate candles with a clear uptrend."""
    import time as _time
    base_ts = int(_time.time()) - n * 60
    candles = []
    for i in range(n):
        ts = base_ts + i * 60
        price = 100.0 + i * 0.5
        candles.append([str(ts), str(price), str(price), str(price + 0.3), str(price - 0.3), "1000.0", str(price * 1000)])
    return candles


def test_trending_candles_produce_high_adx():
    df = candles_to_dataframe(_make_trending_candles(60))
    result = summarize_interval(df, "15min")
    assert result.get("adx") is not None
    assert result["adx"] > 20, f"Trending candles ADX too low: {result['adx']}"


def test_sine_wave_commentary_mentions_range():
    df = candles_to_dataframe(_make_candles(60))
    result = summarize_interval(df, "15min")
    regime = result["market_regime"]["regime"]
    if regime == "ranging":
        assert "RANGE DETECTED" in result["commentary"]
