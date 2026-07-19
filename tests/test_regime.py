"""Tests for regime-aware entry guards (src/regime.py) and the position-extremes reset (A)."""

from types import SimpleNamespace

import pytest

from src.config import RegimeConfig
from src.regime import (
    is_hostile_regime,
    effective_min_confidence,
    regime_size_factor,
    allow_reversal_long,
    allow_reversal_short,
    allow_trend_aligned_short,
    add_on_guard_reason,
    block_alt_long_in_btc_downtrend,
    bracket_risk_scale,
    concentration_scale,
    conviction_size_factor,
    is_relative_strength_alt_long,
    oi_price_signal,
    net_reward_risk_ratio,
    resolve_gate_deadlock,
    reward_risk_ratio,
    risk_capped_contracts,
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


def test_oi_price_signal_all_real_quadrants_and_unknown():
    assert oi_price_signal("up", "up")[0] == "strong_trend"
    assert oi_price_signal("up", "down")[0] == "short_covering"
    assert oi_price_signal("down", "up")[0] == "aggressive_shorts"
    assert oi_price_signal("down", "down")[0] == "long_capitulation"
    assert oi_price_signal("up", "flat")[0] == "neutral"
    assert oi_price_signal("down", None)[0] == "neutral"


# ── Reversal long: catch a confirmed turn against a lagging bearish daily ─────────


def test_reversal_long_allowed_when_ltfs_confirm():
    # The missed-bounce case: bearish daily, but 1h+15m have turned bullish and confidence is high.
    assert allow_reversal_long(
        daily_bias="bearish", side="buy", bias_1h="bullish", bias_15m="bullish",
        confidence=0.82, cfg=_cfg(),
    ) is True


def test_reversal_long_blocked_without_full_ltf_confirmation():
    # 15m not yet bullish -> not a confirmed turn
    assert allow_reversal_long(
        daily_bias="bearish", side="buy", bias_1h="bullish", bias_15m="neutral",
        confidence=0.90, cfg=_cfg(),
    ) is False
    # 1h still bearish -> no turn
    assert allow_reversal_long(
        daily_bias="bearish", side="buy", bias_1h="bearish", bias_15m="bullish",
        confidence=0.90, cfg=_cfg(),
    ) is False


def test_reversal_long_blocked_low_confidence():
    assert allow_reversal_long(
        daily_bias="bearish", side="buy", bias_1h="bullish", bias_15m="bullish",
        confidence=0.70, cfg=_cfg(),
    ) is False


def test_reversal_long_only_for_buys_against_bearish_daily():
    base = dict(bias_1h="bullish", bias_15m="bullish", confidence=0.9, cfg=_cfg())
    # a sell is not a reversal long
    assert allow_reversal_long(daily_bias="bearish", side="sell", **base) is False
    # bullish daily is not the case this overrides (longs already allowed there)
    assert allow_reversal_long(daily_bias="bullish", side="buy", **base) is False


def test_reversal_long_15m_relaxable_and_disableable():
    assert allow_reversal_long(
        daily_bias="bearish", side="buy", bias_1h="bullish", bias_15m="neutral",
        confidence=0.85, cfg=_cfg(reversal_long_require_15m=False),
    ) is True
    assert allow_reversal_long(
        daily_bias="bearish", side="buy", bias_1h="bullish", bias_15m="bullish",
        confidence=0.95, cfg=_cfg(reversal_longs_enabled=False),
    ) is False


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


def test_relative_strength_leader_can_override_bearish_btc_at_reduced_risk():
    cfg = _cfg()
    context = dict(
        symbol="ZEC-USDT", side="buy", btc_daily_bias="bearish",
        local_daily_bias="bullish", bias_4h="bullish", bias_1h="bullish", bias_15m="bullish",
        strength="strong", daily_exhausted=False, confidence=0.84, cfg=cfg,
    )
    assert is_relative_strength_alt_long(**context) is True
    assert block_alt_long_in_btc_downtrend(**context) is False


def test_relative_strength_exception_stays_blocked_when_incomplete_or_low_confidence():
    base = dict(
        symbol="ALT-USDT", side="buy", btc_daily_bias="bearish",
        local_daily_bias="bullish", bias_4h="bullish", bias_1h="bullish", bias_15m="bullish",
        strength="strong", daily_exhausted=False, confidence=0.84, cfg=_cfg(),
    )
    assert block_alt_long_in_btc_downtrend(**{**base, "bias_15m": "neutral"}) is True
    assert block_alt_long_in_btc_downtrend(**{**base, "confidence": 0.80}) is True
    assert block_alt_long_in_btc_downtrend(**{**base, "daily_exhausted": True}) is True


# ── Concentration cap ───────────────────────────────────────────────────────────


def test_concentration_scale_shrinks_oversized_position():
    # RE-USDT: $52 notional on a $70 account, cap 50% -> scale to $35 (0.5*70/52).
    scale = concentration_scale(52.0, 70.0, 0.5)
    assert abs(52.0 * scale - 35.0) < 1e-6


def test_concentration_scale_noop_within_cap():
    assert concentration_scale(20.0, 70.0, 0.5) == 1.0


def test_concentration_scale_uses_projected_existing_symbol_exposure():
    assert concentration_scale(20.0, 100.0, 0.5, existing_notional_usd=40.0) == 0.5
    assert concentration_scale(20.0, 100.0, 0.5, existing_notional_usd=50.0) == 0.0


def test_risk_contract_cap_uses_remaining_lifecycle_budget():
    # $1 budget, each contract risks $0.10. Existing leg already consumes $0.60 -> only 4 remain.
    assert risk_capped_contracts(20, 1, 0.01, 100, 90, 100, 0.01, existing_risk_usd=0.60) == 4
    assert risk_capped_contracts(20, 1, 0.01, 100, 90, 100, 0.01, existing_risk_usd=1.0) == 0


def test_add_on_requires_locked_risk_and_monotonic_stop():
    assert "breakeven" in add_on_guard_reason(
        current_qty=10, new_side="buy", avg_entry=100, protective_stop=99,
        proposed_stop=99, fee_buffer_fraction=0,
    )
    assert "loosen" in add_on_guard_reason(
        current_qty=10, new_side="buy", avg_entry=100, protective_stop=101,
        proposed_stop=100.5, fee_buffer_fraction=0,
    )
    assert add_on_guard_reason(
        current_qty=10, new_side="buy", avg_entry=100, protective_stop=101,
        proposed_stop=101, fee_buffer_fraction=0,
    ) is None
    assert "Opposite" in add_on_guard_reason(
        current_qty=10, new_side="sell", avg_entry=100, protective_stop=101,
        proposed_stop=101, fee_buffer_fraction=0,
    )


def test_concentration_scale_disabled_or_unknown_equity():
    assert concentration_scale(52.0, 70.0, 0.0) == 1.0   # cap disabled
    assert concentration_scale(52.0, 0.0, 0.5) == 1.0    # equity unknown
    assert concentration_scale(0.0, 70.0, 0.5) == 1.0    # no notional


def test_bracket_risk_scale_caps_stop_defined_loss():
    # $1,000 notional with a 10% stop risks $100; a $50 budget scales it to half size.
    assert bracket_risk_scale(entry=100, stop_loss=90, notional_usd=1000,
                              equity_usd=1000, risk_fraction=0.05) == 0.5
    # A position already inside budget is unchanged; this helper never sizes up.
    assert bracket_risk_scale(entry=100, stop_loss=98, notional_usd=1000,
                              equity_usd=1000, risk_fraction=0.05) == 1.0


def test_bracket_risk_scale_fails_open_on_unusable_values():
    assert bracket_risk_scale(entry=0, stop_loss=90, notional_usd=1000,
                              equity_usd=1000, risk_fraction=0.05) == 1.0
    assert bracket_risk_scale(entry=100, stop_loss=None, notional_usd=1000,
                              equity_usd=1000, risk_fraction=0.05) == 1.0


# ── Reward:risk ratio (futures bracket) ─────────────────────────────────────────


def test_rr_long_basic():
    # entry 100, TP 110 (+10), SL 95 (-5) -> RR 2.0
    assert reward_risk_ratio("buy", 100, 110, 95) == 2.0
    assert reward_risk_ratio("long", 100, 110, 95) == 2.0


def test_rr_short_basic():
    # short entry 100, TP 90 (+10 reward), SL 105 (-5 risk) -> RR 2.0
    assert reward_risk_ratio("sell", 100, 90, 105) == 2.0
    assert reward_risk_ratio("short", 100, 90, 105) == 2.0


def test_rr_inverted_bracket_is_below_one():
    # the observed failure: short with TP close (+2) and SL wide (-6) -> RR 0.33
    rr = reward_risk_ratio("sell", 100, 98, 106)
    assert abs(rr - (2.0 / 6.0)) < 1e-9
    assert rr < 1.0


def test_rr_none_when_tp_wrong_side():
    # long with TP below entry -> reward negative -> None (caller rejects)
    assert reward_risk_ratio("buy", 100, 95, 90) is None
    # short with TP above entry -> None
    assert reward_risk_ratio("sell", 100, 110, 120) is None


def test_rr_none_when_sl_wrong_side_or_zero_risk():
    # long with SL above entry -> risk negative -> None
    assert reward_risk_ratio("buy", 100, 110, 105) is None
    # SL at entry -> zero risk -> None (no divide-by-zero)
    assert reward_risk_ratio("buy", 100, 110, 100) is None


def test_rr_none_on_bad_input_or_side():
    assert reward_risk_ratio("buy", None, 110, 95) is None
    assert reward_risk_ratio("hold", 100, 110, 95) is None
    assert reward_risk_ratio("", 100, 110, 95) is None


def test_net_rr_accounts_for_costs_on_winner_and_loser():
    # Gross is 2R, but 0.16% modeled friction on each leg lowers reward and increases loss.
    gross = reward_risk_ratio("buy", 100, 102, 99)
    net = net_reward_risk_ratio(
        "buy", 100, 102, 99, fee_rate=0.0006, slippage_rate=0.001,
    )
    assert gross == 2.0
    assert net == pytest.approx((2 - (100 + 102) * 0.0016) / (1 + (100 + 99) * 0.0016))
    assert net < gross


def test_net_rr_rejects_cost_consumed_target_and_invalid_bracket():
    assert net_reward_risk_ratio(
        "buy", 100, 100.1, 99, fee_rate=0.0006, slippage_rate=0.001,
    ) == 0.0
    assert net_reward_risk_ratio("buy", 100, 99, 98, fee_rate=0.0006) is None


# ── Conviction sizing: scale size by how far confidence clears the floor ─────────


def test_conviction_full_size_at_or_above_full_confidence():
    cfg = _cfg()
    assert conviction_size_factor(0.85, 0.65, cfg) == 1.0
    assert conviction_size_factor(0.92, 0.65, cfg) == 1.0


def test_conviction_min_size_at_floor():
    cfg = _cfg()  # full=0.85, min_factor=0.5
    assert conviction_size_factor(0.65, 0.65, cfg) == 0.5
    # below the floor still clamps to the min factor (defensive — caller already rejected these)
    assert conviction_size_factor(0.60, 0.65, cfg) == 0.5


def test_conviction_ramps_linearly_between_floor_and_full():
    cfg = _cfg()  # floor 0.65 -> full 0.85 maps to 0.5 -> 1.0
    # midpoint confidence 0.75 -> midpoint factor 0.75
    assert abs(conviction_size_factor(0.75, 0.65, cfg) - 0.75) < 1e-9
    # the band tracks the *passed* floor, not a fixed value: floor 0.75 -> 0.85, conf 0.80 -> 0.75
    assert abs(conviction_size_factor(0.80, 0.75, cfg) - 0.75) < 1e-9


def test_conviction_respects_disable_flag():
    cfg = _cfg(conviction_sizing_enabled=False)
    assert conviction_size_factor(0.65, 0.65, cfg) == 1.0


def test_conviction_fails_open_on_unknown_confidence():
    cfg = _cfg()
    assert conviction_size_factor(None, 0.65, cfg) == 1.0


def test_conviction_noop_when_floor_above_full():
    # Misconfig / hostile floor above full: an admitted trade (conf >= floor) gets full size.
    cfg = _cfg(conviction_full_confidence=0.70)
    assert conviction_size_factor(0.80, 0.75, cfg) == 1.0


def test_conviction_custom_min_factor():
    cfg = _cfg(conviction_min_size_factor=0.25)
    assert conviction_size_factor(0.65, 0.65, cfg) == 0.25
    assert abs(conviction_size_factor(0.75, 0.65, cfg) - 0.625) < 1e-9  # midpoint of 0.25..1.0


# ── Deadlock break: allow a daily-aligned entry past the 1h counter-bounce gate ──


def test_deadlock_break_short_in_bearish_daily():
    # The exact stall: bearish daily blocks longs, 1h-bullish bounce blocks the short. With the
    # bounce stalling (15m bearish) and high conviction, take the trend-aligned short.
    assert resolve_gate_deadlock(
        daily_bias="bearish", daily_exhausted=False, side="sell",
        bias_1h="bullish", bias_15m="bearish", confidence=0.80, cfg=_cfg(),
    ) is True
    # 15m neutral (bounce no longer pushing up) is also enough.
    assert resolve_gate_deadlock(
        daily_bias="bearish", daily_exhausted=False, side="sell",
        bias_1h="bullish", bias_15m="neutral", confidence=0.80, cfg=_cfg(),
    ) is True


def test_deadlock_break_long_in_bullish_daily_symmetric():
    assert resolve_gate_deadlock(
        daily_bias="bullish", daily_exhausted=False, side="buy",
        bias_1h="bearish", bias_15m="bullish", confidence=0.80, cfg=_cfg(),
    ) is True


def test_deadlock_break_blocked_when_15m_still_counter():
    # 15m still confirms the 1h bounce -> the bounce is alive, do NOT knife-catch it.
    assert resolve_gate_deadlock(
        daily_bias="bearish", daily_exhausted=False, side="sell",
        bias_1h="bullish", bias_15m="bullish", confidence=0.85, cfg=_cfg(),
    ) is False


def test_deadlock_break_blocked_low_confidence():
    assert resolve_gate_deadlock(
        daily_bias="bearish", daily_exhausted=False, side="sell",
        bias_1h="bullish", bias_15m="bearish", confidence=0.70, cfg=_cfg(),
    ) is False


def test_deadlock_break_not_for_counter_trend_side():
    # A long in a bearish daily is counter-trend (the daily gate kills it earlier); never resolved here.
    assert resolve_gate_deadlock(
        daily_bias="bearish", daily_exhausted=False, side="buy",
        bias_1h="bullish", bias_15m="bearish", confidence=0.90, cfg=_cfg(),
    ) is False


def test_deadlock_break_requires_opposing_1h():
    # If 1h already agrees with the daily-aligned side there's no 1h block to resolve -> False.
    assert resolve_gate_deadlock(
        daily_bias="bearish", daily_exhausted=False, side="sell",
        bias_1h="bearish", bias_15m="bearish", confidence=0.85, cfg=_cfg(),
    ) is False


def test_deadlock_break_only_clean_trend():
    base = dict(side="sell", bias_1h="bullish", bias_15m="bearish", confidence=0.85, cfg=_cfg())
    # exhausted daily is the allow_trend_aligned_short path, not this one
    assert resolve_gate_deadlock(daily_bias="bearish", daily_exhausted=True, **base) is False
    # neutral daily -> no deadlock (a direction is open), leave the 1h gate intact
    assert resolve_gate_deadlock(daily_bias="neutral", daily_exhausted=False, **base) is False


def test_deadlock_break_respects_disable_flag():
    assert resolve_gate_deadlock(
        daily_bias="bearish", daily_exhausted=False, side="sell",
        bias_1h="bullish", bias_15m="bearish", confidence=0.90, cfg=_cfg(deadlock_break_enabled=False),
    ) is False


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
    assert out["ETH-USDT"] == {
        "netSize": 5.0,
        "unrealizedPnl": 3.2,
        "positionOpenTime": None,
        "positionSide": "long",
    }
    assert out["BTC-USDT"]["netSize"] == -2.0


# ── Reversal short: catch a confirmed roll-over against a lagging bullish daily ───


def test_reversal_short_allowed_when_ltfs_confirm():
    # The Jul 7-8 2026 pattern: bullish daily, but 1h+15m turned bearish (pullback) + high confidence.
    assert allow_reversal_short(
        daily_bias="bullish", side="sell", bias_1h="bearish", bias_15m="bearish",
        confidence=0.82, cfg=_cfg(),
    ) is True


def test_reversal_short_blocked_without_full_ltf_confirmation():
    assert allow_reversal_short(
        daily_bias="bullish", side="sell", bias_1h="bearish", bias_15m="neutral",
        confidence=0.90, cfg=_cfg(),
    ) is False
    assert allow_reversal_short(
        daily_bias="bullish", side="sell", bias_1h="bullish", bias_15m="bearish",
        confidence=0.90, cfg=_cfg(),
    ) is False


def test_reversal_short_blocked_low_confidence():
    assert allow_reversal_short(
        daily_bias="bullish", side="sell", bias_1h="bearish", bias_15m="bearish",
        confidence=0.70, cfg=_cfg(),
    ) is False


def test_reversal_short_only_for_sells_against_bullish_daily():
    base = dict(bias_1h="bearish", bias_15m="bearish", confidence=0.9, cfg=_cfg())
    # a buy is not a reversal short
    assert allow_reversal_short(daily_bias="bullish", side="buy", **base) is False
    # bearish daily is not the case this overrides (shorts already allowed there)
    assert allow_reversal_short(daily_bias="bearish", side="sell", **base) is False


def test_reversal_short_15m_relaxable_and_disableable():
    assert allow_reversal_short(
        daily_bias="bullish", side="sell", bias_1h="bearish", bias_15m="neutral",
        confidence=0.85, cfg=_cfg(reversal_short_require_15m=False),
    ) is True
    assert allow_reversal_short(
        daily_bias="bullish", side="sell", bias_1h="bearish", bias_15m="bearish",
        confidence=0.95, cfg=_cfg(reversal_shorts_enabled=False),
    ) is False
