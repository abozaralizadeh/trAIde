import math
import pytest
import pandas as pd
from src.analytics import (
    candles_to_dataframe,
    classify_regime,
    compute_indicators,
    exclude_open_candles,
    summarize_interval,
    summarize_multi_timeframe,
    taker_flow_summary,
    validate_candle_data,
)


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


def test_exclude_open_candles_uses_interval_close_time():
    candles = [
        [str(ts), "100", "100", "101", "99", "10", "1000"]
        for ts in (0, 60, 120, 180)
    ]
    df = candles_to_dataframe(candles)

    closed = exclude_open_candles(df, "1min", as_of=180)

    assert closed["time"].tolist() == [0, 60, 120]
    assert len(candles_to_dataframe(candles, "1min", as_of=180, closed_only=True)) == 3


def test_validate_candle_data_reports_gaps_staleness_and_bad_ohlc():
    candles = [
        ["0", "100", "100", "101", "99", "10", "1000"],
        ["120", "100", "101", "99", "98", "10", "1000"],
    ]
    report = validate_candle_data(candles_to_dataframe(candles), "1min", as_of=400)

    assert report["valid"] is False
    assert report["gap_count"] == 1
    assert report["invalid_ohlc_rows"] == [1]
    assert any("stale" in error for error in report["errors"])


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


@pytest.mark.parametrize(
    ("closes", "expected"),
    [
        ([float(i) for i in range(1, 31)], 100.0),
        ([float(i) for i in range(30, 0, -1)], 0.0),
        ([100.0] * 30, 50.0),
    ],
)
def test_rsi_handles_monotonic_and_flat_prices(closes, expected):
    df = pd.DataFrame({
        "close": closes,
        "high": [price + 1.0 for price in closes],
        "low": [price - 1.0 for price in closes],
        "volume": [1.0] * len(closes),
    })
    rsi = compute_indicators(df)["rsi"].iloc[1:]
    assert rsi.notna().all()
    assert rsi.tolist() == pytest.approx([expected] * len(rsi))


def test_summarize_interval_keys():
    df = candles_to_dataframe(_make_candles(60))
    result = summarize_interval(df, "15min")
    for key in ["interval", "close", "rsi", "trend_bias", "commentary", "atr", "bb_upper", "bb_lower", "vwap"]:
        assert key in result, f"Missing key: {key}"
    assert result["interval"] == "15min"
    assert result["trend_bias"] in ("bullish", "bearish", "neutral-to-bullish", "neutral-to-bearish", "neutral")


def test_summarize_multi_timeframe_conflicting_yields_neutral():
    """1h bullish + 15m bearish yields neutral under weighted scoring."""
    snapshots = [
        {"interval": "1hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "15min", "trend_bias": "bearish", "volatility": "normal"},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert result["overall_bias"] == "neutral"
    assert result["timeframe_conflict"] is True


def test_summarize_multi_timeframe_higher_tf_drives_bias():
    """4h+1h bullish overrides 15m bearish with weighted scoring."""
    snapshots = [
        {"interval": "4hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "1hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "15min", "trend_bias": "bearish", "volatility": "normal"},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert result["overall_bias"] == "bullish"
    assert result["strength"] == "moderate"
    assert result["timeframe_conflict"] is True


def test_summarize_multi_timeframe_daily_gate_vetoes_counter_trend():
    """1D bearish vetoes intraday bullish bias to neutral."""
    snapshots = [
        {"interval": "1day", "trend_bias": "bearish", "volatility": "normal"},
        {"interval": "4hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "1hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "15min", "trend_bias": "bullish", "volatility": "normal"},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert result["overall_bias"] == "neutral"
    assert result["daily_gate_applied"] is True
    assert result["daily_bias"] == "bearish"
    assert "DAILY GATE" in result["entry_hint"]


def test_summarize_multi_timeframe_daily_confirms_boosts_strength():
    """1D bullish + intraday bullish boosts strength from moderate to strong."""
    snapshots = [
        {"interval": "1day", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "4hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "1hour", "trend_bias": "neutral", "volatility": "normal"},
        {"interval": "15min", "trend_bias": "bullish", "volatility": "normal"},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert result["overall_bias"] == "bullish"
    assert result["daily_gate_applied"] is False
    assert result["strength"] == "strong"
    assert "Daily trend confirms" in result["entry_hint"]


def test_exhausted_bearish_daily_surfaces_confirmed_trend_short_exception():
    snapshots = [
        {"interval": "1day", "trend_bias": "bearish", "rsi": 20, "adx": 30, "volatility": "normal"},
        {"interval": "4hour", "trend_bias": "bearish", "volatility": "normal"},
        {"interval": "1hour", "trend_bias": "bearish", "volatility": "normal"},
        {"interval": "15min", "trend_bias": "bearish", "volatility": "normal"},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert result["daily_exhausted"] is True
    assert result["daily_bias_raw"] == "bearish"
    assert "continuation SHORT is eligible" in result["entry_hint"]


def test_weak_daily_adx_neutralizes_gate_without_marking_exhaustion():
    snapshots = [
        {"interval": "1day", "trend_bias": "bearish", "rsi": 50, "adx": 17.9, "volatility": "normal"},
        {"interval": "4hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "1hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "15min", "trend_bias": "bullish", "volatility": "normal"},
    ]

    result = summarize_multi_timeframe(snapshots)

    assert result["overall_bias"] == "bullish"
    assert result["daily_bias_raw"] == "bearish"
    assert result["daily_bias"] == "neutral"
    assert result["daily_trend_weak"] is True
    assert result["daily_exhausted"] is False
    assert result["daily_gate_applied"] is False
    assert "DAILY TREND WEAK" in result["entry_hint"]
    assert "DAILY EXHAUSTED" not in result["entry_hint"]


def test_summarize_multi_timeframe_daily_neutral_no_effect():
    """1D neutral does not gate or boost."""
    snapshots = [
        {"interval": "1day", "trend_bias": "neutral", "volatility": "normal"},
        {"interval": "4hour", "trend_bias": "bullish", "volatility": "normal"},
        {"interval": "1hour", "trend_bias": "bullish", "volatility": "normal"},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert result["overall_bias"] == "bullish"
    assert result["daily_gate_applied"] is False
    assert result["daily_bias"] == "neutral"


def test_summarize_multi_timeframe_no_daily_backward_compatible():
    """Without a 1D snapshot, behavior is unchanged."""
    snapshots = [
        {"interval": "4hour", "trend_bias": "bearish", "volatility": "normal"},
        {"interval": "1hour", "trend_bias": "bearish", "volatility": "normal"},
    ]
    result = summarize_multi_timeframe(snapshots)
    assert result["overall_bias"] == "bearish"
    assert result["daily_bias"] == "neutral"
    assert result["daily_gate_applied"] is False
    assert result["daily_interval"] is None


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


class TestTakerFlowSummary:
    """The tape is the only place the aggressor is visible — klines carry no taker split.

    Everything downstream (the per-poll EWMA, the stamp on a direction call, the with/against
    spread) is a ratio computed here, so a bug in this reducer is invisible until it has quietly
    mislabelled weeks of evidence.
    """

    @staticmethod
    def _tape(rows):
        """Rows as KuCoin returns them: newest first, size in CONTRACTS, ts in nanoseconds."""
        return [
            {"sequence": seq, "side": side, "size": size, "price": "1.0", "ts": ts * 1_000_000_000}
            for seq, side, size, ts in rows
        ]

    def test_volume_and_count_shares_can_disagree(self):
        """One whale buy against many small sells — the divergence is the point of reporting both."""
        out = taker_flow_summary(self._tape([
            (5, "buy", "900", 1_700_000_005),
            (4, "sell", "25", 1_700_000_004),
            (3, "sell", "25", 1_700_000_003),
            (2, "sell", "25", 1_700_000_002),
            (1, "sell", "25", 1_700_000_001),
        ]))
        assert out["buyShare"] == 0.9          # volume says buyers
        assert out["buyTradeShare"] == 0.2     # count says sellers
        assert out["trades"] == 5
        assert out["spanSec"] == 4.0           # nanosecond ts normalized to seconds
        assert out["lastCursor"] == 5

    def test_only_trades_past_the_cursor_count_as_new_information(self):
        """Consecutive 100-trade windows overlap heavily at a 60s poll; re-counting them would
        manufacture a sample size the tape never delivered."""
        tape = self._tape([
            (4, "buy", "10", 1_700_000_004),
            (3, "buy", "10", 1_700_000_003),
            (2, "sell", "10", 1_700_000_002),
            (1, "sell", "10", 1_700_000_001),
        ])
        out = taker_flow_summary(tape, since_cursor=2)
        assert out["buyShare"] == 0.5          # whole window is balanced
        assert out["newTrades"] == 2
        assert out["newBuyShare"] == 1.0       # but everything NEW was a buy
        assert out["gapped"] is False

    def test_a_fully_rolled_window_is_flagged_as_gapped(self):
        out = taker_flow_summary(
            self._tape([(9, "buy", "1", 1_700_000_009), (8, "sell", "1", 1_700_000_008)]),
            since_cursor=2,
        )
        assert out["newTrades"] == 2 and out["gapped"] is True

    def test_no_cursor_means_no_new_slice_claimed(self):
        out = taker_flow_summary(self._tape([(1, "buy", "1", 1_700_000_001)]))
        assert out["newTrades"] == 0 and out["newBuyShare"] is None and out["gapped"] is False

    def test_malformed_rows_are_skipped_not_fatal(self):
        out = taker_flow_summary([
            {"sequence": 3, "side": "buy", "size": "10", "ts": 1_700_000_003_000_000_000},
            {"sequence": 2, "side": "", "size": "10", "ts": 1_700_000_002_000_000_000},
            {"sequence": 1, "side": "sell", "size": "not-a-number", "ts": 1},
            {"sequence": 0, "side": "sell", "size": "0", "ts": 1},
            "garbage",
            None,
        ])
        assert out["trades"] == 1 and out["buyShare"] == 1.0

    def test_empty_tape_reads_as_no_data_not_as_balanced_flow(self):
        """None, not 0.5 and not 0.0 — a silent zero would later be smoothed in as real evidence."""
        for empty in ([], None):
            out = taker_flow_summary(empty)
            assert out["buyShare"] is None and out["buyTradeShare"] is None
            assert out["trades"] == 0 and out["lastCursor"] is None and out["spanSec"] is None

    def test_second_level_timestamps_are_left_alone(self):
        """Not every venue field is nanoseconds; a plain epoch must not be divided into 1970."""
        out = taker_flow_summary([
            {"sequence": 2, "side": "buy", "size": "1", "ts": 1_700_000_060},
            {"sequence": 1, "side": "buy", "size": "1", "ts": 1_700_000_000},
        ])
        assert out["spanSec"] == 60.0 and out["newestTs"] == 1_700_000_060

    def test_falls_back_to_ts_when_the_tape_has_no_sequence(self):
        out = taker_flow_summary([
            {"side": "buy", "size": "1", "ts": 1_700_000_002},
            {"side": "sell", "size": "1", "ts": 1_700_000_001},
        ], since_cursor=1_700_000_001)
        assert out["lastCursor"] == 1_700_000_002
        assert out["newTrades"] == 1 and out["newBuyShare"] == 1.0
