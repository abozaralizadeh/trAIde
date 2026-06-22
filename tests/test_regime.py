"""Tests for regime-aware entry guards (src/regime.py) and the position-extremes reset (A)."""

from types import SimpleNamespace

from src.config import RegimeConfig
from src.regime import (
    is_hostile_regime,
    effective_min_confidence,
    regime_size_factor,
    allow_trend_aligned_short,
    block_alt_long_in_btc_downtrend,
    concentration_scale,
)
from src.memory import MemoryStore


def _cfg(**overrides) -> RegimeConfig:
    base = dict(
        throttle_enabled=True,
        caution_min_confidence=0.75,
        caution_size_factor=0.6,
        trend_shorts_enabled=True,
        trend_short_min_confidence=0.78,
        trend_short_require_15m=True,
    )
    base.update(overrides)
    return RegimeConfig(**base)


# ── Correlation gate: block alt longs while BTC daily is bearish ────────────────


def test_alt_long_blocked_when_btc_bearish():
    cfg = _cfg()
    assert block_alt_long_in_btc_downtrend(symbol="RE-USDT", side="buy", btc_daily_bias="bearish", cfg=cfg) is True
    assert block_alt_long_in_btc_downtrend(symbol="XION-USDT", side="long", btc_daily_bias="bearish", cfg=cfg) is True


def test_alt_long_allowed_when_btc_not_bearish():
    cfg = _cfg()
    assert block_alt_long_in_btc_downtrend(symbol="RE-USDT", side="buy", btc_daily_bias="bullish", cfg=cfg) is False
    assert block_alt_long_in_btc_downtrend(symbol="RE-USDT", side="buy", btc_daily_bias="neutral", cfg=cfg) is False


def test_alt_short_never_blocked():
    cfg = _cfg()
    assert block_alt_long_in_btc_downtrend(symbol="RE-USDT", side="sell", btc_daily_bias="bearish", cfg=cfg) is False


def test_majors_exempt_from_alt_gate():
    cfg = _cfg()
    # BTC/ETH have their own per-symbol daily gate; the alt gate must not touch them.
    assert block_alt_long_in_btc_downtrend(symbol="BTC-USDT", side="buy", btc_daily_bias="bearish", cfg=cfg) is False
    assert block_alt_long_in_btc_downtrend(symbol="ETH-USDT", side="buy", btc_daily_bias="bearish", cfg=cfg) is False


def test_alt_gate_respects_disable_flag():
    cfg = _cfg(alt_long_block_enabled=False)
    assert block_alt_long_in_btc_downtrend(symbol="RE-USDT", side="buy", btc_daily_bias="bearish", cfg=cfg) is False


def test_alt_gate_custom_majors():
    cfg = _cfg(alt_majors=("BTC", "ETH", "SOL"))
    assert block_alt_long_in_btc_downtrend(symbol="SOL-USDT", side="buy", btc_daily_bias="bearish", cfg=cfg) is False
    assert block_alt_long_in_btc_downtrend(symbol="RE-USDT", side="buy", btc_daily_bias="bearish", cfg=cfg) is True


# ── Concentration cap ───────────────────────────────────────────────────────────


def test_concentration_scale_shrinks_oversized_position():
    # RE-USDT: $52 notional on a $70 account, cap 50% -> scale to $35 (0.5*70/52).
    scale = concentration_scale(52.0, 70.0, 0.5)
    assert abs(52.0 * scale - 35.0) < 1e-6


def test_concentration_scale_noop_within_cap():
    assert concentration_scale(20.0, 70.0, 0.5) == 1.0


def test_concentration_scale_disabled_or_unknown_equity():
    assert concentration_scale(52.0, 70.0, 0.0) == 1.0   # cap disabled
    assert concentration_scale(52.0, 0.0, 0.5) == 1.0    # equity unknown
    assert concentration_scale(0.0, 70.0, 0.5) == 1.0    # no notional


# ── B: hostile regime detection + throttle ──────────────────────────────────────


def test_hostile_regime_detection():
    assert is_hostile_regime("bearish", False) is True
    assert is_hostile_regime("bullish", True) is True      # exhausted (overbought) is also hostile
    assert is_hostile_regime("bearish", True) is True
    assert is_hostile_regime("bullish", False) is False    # healthy uptrend
    assert is_hostile_regime("neutral", False) is False


def test_confidence_floor_raised_in_hostile_regime():
    # base 0.65 -> raised to 0.75 when bearish/exhausted
    assert effective_min_confidence(0.65, "bearish", False, _cfg()) == 0.75
    assert effective_min_confidence(0.65, "neutral", False, _cfg()) == 0.65   # unchanged
    # never lowers an already-high base
    assert effective_min_confidence(0.80, "bearish", False, _cfg()) == 0.80


def test_confidence_floor_respects_disabled():
    assert effective_min_confidence(0.65, "bearish", True, _cfg(throttle_enabled=False)) == 0.65


def test_size_factor_shrinks_in_hostile_regime():
    assert regime_size_factor("bearish", False, _cfg()) == 0.6
    assert regime_size_factor("bullish", False, _cfg()) == 1.0
    assert regime_size_factor("bearish", False, _cfg(throttle_enabled=False)) == 1.0


# ── D: trend-aligned shorts ─────────────────────────────────────────────────────


def test_trend_short_allowed_when_confirmed():
    # The exact blocked case from the logs: exhausted-bearish daily, agent wants to short,
    # 1h+15m confirm the downtrend, high conviction -> permit past anti-FOMO.
    assert allow_trend_aligned_short(
        daily_exhausted=True, daily_bias_raw="bearish", side="sell",
        bias_1h="bearish", bias_15m="bearish", confidence=0.80, cfg=_cfg(),
    ) is True


def test_trend_short_blocked_low_confidence():
    assert allow_trend_aligned_short(
        daily_exhausted=True, daily_bias_raw="bearish", side="sell",
        bias_1h="bearish", bias_15m="bearish", confidence=0.70, cfg=_cfg(),
    ) is False


def test_trend_short_blocked_without_1h_confirmation():
    assert allow_trend_aligned_short(
        daily_exhausted=True, daily_bias_raw="bearish", side="sell",
        bias_1h="neutral", bias_15m="bearish", confidence=0.85, cfg=_cfg(),
    ) is False


def test_trend_short_requires_15m_by_default():
    assert allow_trend_aligned_short(
        daily_exhausted=True, daily_bias_raw="bearish", side="sell",
        bias_1h="bearish", bias_15m="bullish", confidence=0.85, cfg=_cfg(),
    ) is False
    # ...but can be relaxed
    assert allow_trend_aligned_short(
        daily_exhausted=True, daily_bias_raw="bearish", side="sell",
        bias_1h="bearish", bias_15m="bullish", confidence=0.85, cfg=_cfg(trend_short_require_15m=False),
    ) is True


def test_trend_short_only_for_sells_and_bearish_and_exhausted():
    base = dict(bias_1h="bearish", bias_15m="bearish", confidence=0.85, cfg=_cfg())
    # a buy is never a "trend-aligned short"
    assert allow_trend_aligned_short(daily_exhausted=True, daily_bias_raw="bearish", side="buy", **base) is False
    # bullish-exhausted (overbought) short is NOT trend-aligned -> stays blocked
    assert allow_trend_aligned_short(daily_exhausted=True, daily_bias_raw="bullish", side="sell", **base) is False
    # not exhausted -> this path doesn't apply (handled by the normal daily gate)
    assert allow_trend_aligned_short(daily_exhausted=False, daily_bias_raw="bearish", side="sell", **base) is False


def test_trend_short_respects_disabled():
    assert allow_trend_aligned_short(
        daily_exhausted=True, daily_bias_raw="bearish", side="sell",
        bias_1h="bearish", bias_15m="bearish", confidence=0.90, cfg=_cfg(trend_shorts_enabled=False),
    ) is False


# ── A: position_extremes reset on close ─────────────────────────────────────────


def test_extremes_reset_when_position_closes(tmp_path):
    """Feeding live positions makes a symbol's peak/trough reset once it closes, instead of
    lingering (the bug where ETH's openTs stuck for 2 days across many closes)."""
    mem = MemoryStore(str(tmp_path / "mem.json"), retention_days=7)
    mem.update_position_extremes({"ETH-USDT": {"netSize": 1.0, "unrealizedPnl": 5.0}})
    mem.update_position_extremes({"ETH-USDT": {"netSize": 1.0, "unrealizedPnl": 9.0}})
    assert mem.get_position_extremes("ETH-USDT").get("peakPnl") == 9.0
    # Position closed -> absent from the live map -> extreme pruned/reset.
    mem.update_position_extremes({})
    assert mem.get_position_extremes("ETH-USDT") == {}
    # A fresh position starts a brand-new peak (not the stale 9.0).
    mem.update_position_extremes({"ETH-USDT": {"netSize": 1.0, "unrealizedPnl": 1.0}})
    assert mem.get_position_extremes("ETH-USDT").get("peakPnl") == 1.0


def test_live_extremes_map_from_snapshot():
    """_live_extremes_map maps live futures positions, skips zero-qty, normalizes symbols."""
    from src.main import _live_extremes_map  # lazy: pulls heavy deps only when this test runs
    snap = SimpleNamespace(futures_positions=[
        {"symbol": "ETHUSDTM", "currentQty": 5, "unrealisedPnl": 3.2},
        {"symbol": "XBTUSDTM", "currentQty": -2, "unrealisedPnl": -1.0},
        {"symbol": "SOLUSDTM", "currentQty": 0, "unrealisedPnl": 0.0},  # flat -> skipped
    ])
    out = _live_extremes_map(snap)
    assert set(out.keys()) == {"ETH-USDT", "BTC-USDT"}
    assert out["ETH-USDT"] == {"netSize": 5.0, "unrealizedPnl": 3.2}
    assert out["BTC-USDT"]["netSize"] == -2.0
