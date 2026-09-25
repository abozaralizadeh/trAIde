"""Unit tests for the code-driven profit guards (src/protection.py).

Exchange-facing behavior is exercised with local fakes only; no network calls are made.
"""

import time
from types import SimpleNamespace

import pytest

from src.config import ProfitProtectionConfig
from src.protection import (
    ProtectionManager,
    decide_protection,
    should_block_chase,
    should_close_for_unrealized_loss,
)


def _cfg(**overrides) -> ProfitProtectionConfig:
    base = dict(
        enabled=True,
        dry_run=False,
        # Mechanics tests below pin the arm thresholds at 1R so they exercise the ratchet math at a
        # fixed, explicit boundary (peaks are written as multiples of this). The LIVE default is 0.5R
        # (see ProfitProtectionConfig / test_trail_default_arm_is_half_r) — the stop floor doubled since
        # 1R was calibrated, so 1R now sits ~3x ATR and never arms; 0.5R restores the validated distance.
        breakeven_trigger_r=1.0,
        trail_arm_r=1.0,
        breakeven_fee_pct=0.0015,
        giveback_pct=0.35,
        min_favorable_excursion_pct=0.005,
        no_chase_enabled=True,
        post_win_cooldown_minutes=45.0,
        no_chase_buffer_pct=0.001,
    )
    base.update(overrides)
    return ProfitProtectionConfig(**base)


def test_trail_default_arm_is_half_r():
    """The LIVE profit-lock arms at 0.5R, not 1R (Aug 11 2026 geometry correction).

    The July 1m-path replay endorsed a 1R arm, but 1R is a distance in *stop units*, and the stop
    floor has since roughly doubled (median live stop 3.0x ATR vs 1.4x in July). So 1R now sits at
    ~3x ATR — a move the tape almost never makes (median favourable excursion 0.30R; only 12% of the
    last 58 closes reached +1R), leaving the trail permanently dormant while winners round-tripped to
    full stops. 0.5R restores the ~1.4x-ATR arming distance the replay actually validated. This test
    fails loudly if the default is ever reverted to 1R without re-checking the live stop geometry.
    """
    from src.config import load_config

    cfg = load_config().profit_protection
    assert cfg.trail_arm_r == 0.5
    assert cfg.breakeven_trigger_r == 0.5


# ── decide_protection: early invalidation (P1c) ─────────────────────────────────


def test_early_cut_fires_when_never_green_and_almost_at_stop():
    # Long entry 100, stop 95 (risk 5). Never green (peak 0.1), now -4.5 px = 90% to stop (>= 0.85), 25min.
    # The cut only front-runs an almost-certain stop, not normal pre-breakout heat.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=95.5, sl_price=95.0, peak_fe=0.1,
                          cfg=_cfg(), opened_min_ago=25.0)
    assert d["action"] == "close" and "early invalidation" in d["reason"]


def test_early_cut_does_not_fire_inside_winners_mae_band():
    # The NEAR lesson: never green, 70% to stop (0.70R MAE) — this is INSIDE the band winners breathe
    # (~0.57R here), so cutting it stops out trades that are still right. Must NOT fire at 0.85 threshold.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=96.5, sl_price=95.0, peak_fe=0.1,
                          cfg=_cfg(), opened_min_ago=100.0)
    assert d["action"] == "none"


def test_early_cut_skipped_within_grace():
    # Same failing trade but only 10min old (< 20min grace) → not cut yet.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=95.5, sl_price=95.0, peak_fe=0.1,
                          cfg=_cfg(), opened_min_ago=10.0)
    assert d["action"] == "none"


def test_early_cut_skipped_if_trade_went_green():
    # Peak reached +0.5 (> 0.3% of 100) → it "worked", so early-cut must NOT fire even if now underwater.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=95.5, sl_price=95.0, peak_fe=0.5,
                          cfg=_cfg(), opened_min_ago=25.0)
    assert d["action"] == "none"


def test_early_cut_skipped_if_not_yet_failing():
    # Never green, but only -1 px = 20% to the stop (< 0.85 threshold) → give it room, no cut.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=99.0, sl_price=95.0, peak_fe=0.05,
                          cfg=_cfg(), opened_min_ago=25.0)
    assert d["action"] == "none"


def test_early_cut_short_symmetry():
    # Short entry 100, stop 105 (risk 5). Never green, now 104.3 = 86% to stop (>= 0.85), 25min → cut.
    d = decide_protection(side_long=False, avg_entry=100.0, mark=104.3, sl_price=105.0, peak_fe=0.1,
                          cfg=_cfg(), opened_min_ago=25.0)
    assert d["action"] == "close" and "early invalidation" in d["reason"]


def test_early_cut_disabled_or_no_age():
    # opened_min_ago=None (caller didn't track) → skipped; and disabled flag → skipped.
    assert decide_protection(side_long=True, avg_entry=100.0, mark=96.5, sl_price=95.0, peak_fe=0.1,
                             cfg=_cfg(), opened_min_ago=None)["action"] == "none"
    assert decide_protection(side_long=True, avg_entry=100.0, mark=96.5, sl_price=95.0, peak_fe=0.1,
                             cfg=_cfg(early_cut_enabled=False), opened_min_ago=25.0)["action"] == "none"


# ── decide_protection: breakeven ratchet ────────────────────────────────────────


def test_no_action_before_one_r():
    # Long up only +0.4R (below the 1R trail arm), stop still below: nothing to do yet.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=102.0, sl_price=95.0, peak_fe=2.0, cfg=_cfg())
    assert d["action"] == "none"


def test_move_to_breakeven_at_one_r():
    # peak favourable excursion == initial risk (5 px = 1R) with the give-back (trail-disabled) path:
    # the P1b breakeven ratchet moves the stop to fee-adjusted breakeven.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=5.0, cfg=_gb())
    assert d["action"] == "move_breakeven"
    assert d["stopPrice"] == 100.0 * 1.0015  # above entry by the fee buffer


def test_no_breakeven_when_stop_already_protective():
    # Stop already above entry (in profit) → no further ratchet needed.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=106.0, sl_price=101.0, peak_fe=6.0, cfg=_cfg())
    assert d["action"] == "none"


def test_short_breakeven_symmetry():
    d = decide_protection(side_long=False, avg_entry=100.0, mark=96.0, sl_price=105.0, peak_fe=5.0, cfg=_gb())
    assert d["action"] == "move_breakeven"
    assert d["stopPrice"] == 100.0 * (1 - 0.0015)  # below entry for a short


def test_manager_breakeven_buffer_covers_configured_roundtrip_cost():
    mgr = ProtectionManager(
        _cfg(breakeven_fee_pct=0.0015), None, breakeven_cost_pct=0.0032,
    )
    assert mgr.cfg.breakeven_fee_pct == pytest.approx(0.0032)


def test_manager_does_not_apply_exchange_age_without_persisted_peak():
    mgr = ProtectionManager(_cfg(), None)
    snapshot = SimpleNamespace(
        futures_enabled=True,
        futures_account={"accountEquity": 1000},
        futures_positions=[{
            "symbol": "ETHUSDTM", "currentQty": 1, "avgEntryPrice": 100,
            "markPrice": 96.5, "unrealisedPnl": -3.5,
            "openingTimestamp": int((time.time() - 3600) * 1000),
        }],
        futures_stop_orders=[{
            "symbol": "ETHUSDTM", "side": "sell", "stop": "down",
            "stopPrice": 95, "reduceOnly": True,
        }],
        total_usdt=1000,
    )
    # First post-restart observation has no trustworthy historical MFE, so early-cut must wait.
    assert mgr.run(snapshot) == []


# ── decide_protection: give-back cap ─────────────────────────────────────────────


# The give-back tests exercise the LEGACY path (trail_enabled=False); trailing is the new default and
# supersedes the give-back close (see the trailing-ratchet tests below).
def _gb(**overrides) -> ProfitProtectionConfig:
    overrides.setdefault("trail_enabled", False)
    return _cfg(**overrides)


def test_giveback_close_after_real_run():
    # Chop (sub-runner): ran to +8 px = 1.6R (risk 5), gave back to +4 (50% > 35%) → close to lock the gain.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=8.0, cfg=_gb())
    assert d["action"] == "close"


def test_trend_runner_holds_through_normal_giveback():
    """A revealed trend winner (peak run >= trend_runner_r) tolerates a deeper pullback so it can keep
    running — the ZEC lesson. Risk 5; peak 15 px = 3R (runner). Gave back to +8 (~47% < 55% trend cap)
    → do NOT close; instead lock breakeven and let it run."""
    d = decide_protection(side_long=True, avg_entry=100.0, mark=108.0, sl_price=95.0, peak_fe=15.0, cfg=_gb())
    assert d["action"] == "move_breakeven"


def test_trend_runner_closes_on_deep_giveback():
    # Same 3R runner, but now gave back to +6 (60% > 55% trend cap) → close to lock the bulk of the gain.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=106.0, sl_price=95.0, peak_fe=15.0, cfg=_gb())
    assert d["action"] == "close"


def test_trend_adaptive_off_keeps_tight_giveback():
    # With trend adaptivity disabled, a 3R run that gives back >35% closes (legacy mean-reversion behavior).
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=15.0,
                          cfg=_gb(trend_adaptive_enabled=False))
    assert d["action"] == "close"


def test_no_giveback_when_peak_too_small():
    # Peak of 0.2 px is below the 0.3% min FE on a 100-priced asset → ignore noise.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=99.9, sl_price=95.0, peak_fe=0.2, cfg=_gb())
    assert d["action"] == "none"


def test_giveback_disabled_when_pct_zero():
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=10.0, cfg=_gb(giveback_pct=0.0))
    # With give-back off it should fall through to breakeven (peak 10 >= 1R of 5).
    assert d["action"] == "move_breakeven"


# ── decide_protection: give-back arming at 1R (giveback_arm_r) ───────────────────


def test_giveback_not_armed_below_one_r_run():
    """Sub-1R wobble is the original SL's job — the cap must not book fee-scale scratch wins.
    Risk = 5 px; peak ran only 2 px (0.4R, but > the 0.5% pct floor) then retraced fully."""
    d = decide_protection(side_long=True, avg_entry=100.0, mark=100.1, sl_price=95.0, peak_fe=2.0, cfg=_gb())
    assert d["action"] == "none"


def test_giveback_armed_after_one_r_run():
    # Risk = 5 px; ran to 6 px (1.2R) then gave back >35% → close.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=102.0, sl_price=95.0, peak_fe=6.0, cfg=_gb())
    assert d["action"] == "close"


def test_giveback_pct_arming_when_no_stop_known():
    # Without a live stop the 1R arming can't be computed → falls back to pct arming (old behavior).
    d = decide_protection(side_long=True, avg_entry=100.0, mark=100.5, sl_price=None, peak_fe=2.0, cfg=_gb())
    assert d["action"] == "close"


# ── decide_protection: trailing ratchet (default — lets winners run to TP instead of capping) ──


def test_trail_ratchets_stop_up_and_lets_winner_run():
    # Risk 5 (anchored via risk_override, since the live stop is already at breakeven); peak +10 (2R),
    # price still near peak (mark 109). Lock = max(33% of peak=3.3, peak−1R=5) = 5 → stop +5 (105),
    # NOT a close, so the trade keeps running toward its TP instead of being capped.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=109.0, sl_price=100.15, peak_fe=10.0,
                          cfg=_cfg(), risk_override=5.0)
    assert d["action"] == "move_breakeven"
    assert abs(d["stopPrice"] - 105.0) < 1e-6   # max(0.33*10, 10 - 1.0*5) = 5 above entry


def test_trail_does_not_arm_before_one_r():
    """A sub-1R 'profit' is inside the noise the stop was drawn around — the trail must ignore it.

    This is the Jul 27 2026 correction. Arming at 0.5R and locking half the peak meant the ratchet
    engaged on noise-scale excursions (the live sample's median favourable excursion was 0.27R) and
    booked a fraction of it. Live: 27 trades, 37% win rate, avg win +0.38R vs avg loss -0.60R, net
    -6.4R. Replaying those entries on real 1m paths isolates the exit rule — 63% winners but a 0.35 /
    -1.05 payoff, net -4.5R. Peak +4 (0.8R) must now leave the trade alone.
    """
    d = decide_protection(side_long=True, avg_entry=100.0, mark=103.5, sl_price=95.0, peak_fe=4.0, cfg=_cfg())
    assert d["action"] == "none"


def test_trail_arms_exactly_at_one_r():
    # Peak +5 (exactly 1R) arms the trail; lock = max(0.33*5=1.65, 5-1.0*5=0) = 1.65 above entry.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=5.0, cfg=_cfg())
    assert d["action"] == "move_breakeven"
    assert abs(d["stopPrice"] - 101.65) < 1e-6


def test_trail_does_not_arm_below_arm_threshold():
    # Peak +0.4R (2.0 px) is below the 1R arm → the trail stays dormant, the original SL still owns it.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=102.0, sl_price=95.0, peak_fe=2.0, cfg=_cfg())
    assert d["action"] == "none"


def test_trail_closes_when_price_retraces_past_trail():
    # Ran to +15 (3R, peak 115) then fell to 108. Trail stop = 110; mark 108 <= 110 → breached → close,
    # locking ~+2R (far better than riding back to breakeven).
    d = decide_protection(side_long=True, avg_entry=100.0, mark=108.0, sl_price=100.15, peak_fe=15.0,
                          cfg=_cfg(), risk_override=5.0)
    assert d["action"] == "close" and "trailing stop hit" in d["reason"]


def test_trail_short_symmetry():
    # Short: entry 100, risk 5, peak +10 (price fell to 90). Lock = max(0.33*10=3.3, 10-1.0*5=5) = 5
    # → stop to 95.0 (+1R locked); mark 91.
    d = decide_protection(side_long=False, avg_entry=100.0, mark=91.0, sl_price=99.85, peak_fe=10.0,
                          cfg=_cfg(), risk_override=5.0)
    assert d["action"] == "move_breakeven"
    assert abs(d["stopPrice"] - 95.0) < 1e-6   # entry - max(0.33*10, 10 - 1.0*5)


def test_trail_no_churn_on_tiny_advance():
    # Stop already at 105.5; new trail level 106.25 is only +0.75 (< 0.25R=1.25 min step) → no re-place.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=109.0, sl_price=105.5, peak_fe=10.0,
                          cfg=_cfg(), risk_override=5.0)
    assert d["action"] == "none"


def test_trail_survives_stop_at_breakeven_via_risk_override():
    # The bug this guards: once the stop sits at/above entry the live sl yields no positive risk, which
    # used to disable every R-rule on a winner. With risk_override the trail keeps advancing.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=112.0, sl_price=100.15, peak_fe=12.0,
                          cfg=_cfg(), risk_override=5.0)
    assert d["action"] == "move_breakeven" and abs(d["stopPrice"] - 107.0) < 1e-6  # max(0.33*12, 12-1.0*5)


def test_giveback_arm_r_zero_reverts_to_pct_arming():
    # arm_r=0 disables risk-based arming even with a stop present (legacy pct behavior).
    d = decide_protection(side_long=True, avg_entry=100.0, mark=100.1, sl_price=95.0, peak_fe=2.0, cfg=_cfg(giveback_arm_r=0.0))
    assert d["action"] == "close"


# ── Regression: the actual ETH incident (2026-06-07/08) ──────────────────────────


def test_eth_runner_gives_back_is_closed():
    """The DCA'd ETH long (avg ~1622) ran to ~1711 (+89 px) then reversed.
    With the guard, a retrace past 50% of that run locks profit instead of round-tripping.
    """
    d = decide_protection(side_long=True, avg_entry=1622.0, mark=1665.0, sl_price=1655.0, peak_fe=89.0, cfg=_cfg())
    assert d["action"] == "close"  # would have locked ~+43 px instead of stopping out red


def test_top_entry_with_no_run_takes_no_action():
    """The fresh long opened at the 1711.76 top never ran up; the ratchet correctly
    does nothing (this trade is the no-chase guard's job, not profit-lock's)."""
    d = decide_protection(side_long=True, avg_entry=1711.76, mark=1705.0, sl_price=1658.5, peak_fe=0.0, cfg=_cfg())
    assert d["action"] == "none"


# ── should_block_chase (P2) ─────────────────────────────────────────────────────


def test_block_relong_at_higher_price_after_win():
    # Closed a long at 1709; trying to re-buy at 1711.76 (the exact failure) → blocked.
    assert should_block_chase(close_type="CLOSE_LONG", exit_price=1709.0, new_side="buy", new_price=1711.76, buffer_pct=0.001) is True


def test_allow_relong_at_better_price():
    # Re-buying meaningfully below the exit (a real pullback) is allowed.
    assert should_block_chase(close_type="CLOSE_LONG", exit_price=1709.0, new_side="buy", new_price=1690.0, buffer_pct=0.001) is False


def test_block_reshort_at_lower_price_after_win():
    assert should_block_chase(close_type="CLOSE_SHORT", exit_price=100.0, new_side="sell", new_price=99.95, buffer_pct=0.001) is True


def test_no_block_on_opposite_direction():
    # Closed a long, now going short → not chasing; allowed.
    assert should_block_chase(close_type="CLOSE_LONG", exit_price=1709.0, new_side="sell", new_price=1711.0, buffer_pct=0.001) is False


def test_no_block_on_missing_prices():
    assert should_block_chase(close_type="CLOSE_LONG", exit_price=0.0, new_side="buy", new_price=1711.0, buffer_pct=0.001) is False


# ── Emergency bracket: never leave a filled position naked ───────────────────────


def _mgr(dry_run=True):
    cfg = _cfg(dry_run=dry_run)
    return ProtectionManager(cfg, kucoin_futures=None, notifier=None, emergency_sl_pct=0.02, min_rr=1.5)


def test_emergency_bracket_long_levels():
    # Long entry 100, 2% SL / 1.5R TP → SL 98, TP 103. Dry-run so no order is placed.
    mgr = _mgr(dry_run=True)
    rec = mgr._ensure_emergency_bracket("ETHUSDTM", {"currentQty": 5, "realLeverage": 3}, side_long=True, avg_entry=100.0)
    assert rec["dryRun"] is True
    assert abs(rec["stopLoss"] - 98.0) < 1e-6
    assert abs(rec["takeProfit"] - 103.0) < 1e-6


def test_emergency_bracket_short_levels():
    # Short entry 100 → SL above at 102, TP below at 97.
    mgr = _mgr(dry_run=True)
    rec = mgr._ensure_emergency_bracket("ETHUSDTM", {"currentQty": -5, "realLeverage": 3}, side_long=False, avg_entry=100.0)
    assert abs(rec["stopLoss"] - 102.0) < 1e-6
    assert abs(rec["takeProfit"] - 97.0) < 1e-6


def test_emergency_bracket_debounce_grace():
    # First poll seeing a naked position: it is NOT bracketed yet (grace lets an attached bracket appear).
    mgr = ProtectionManager(_cfg(dry_run=True), kucoin_futures=object(), notifier=None, emergency_sl_pct=0.02, min_rr=1.5)
    snap = type("S", (), {"futures_enabled": True,
                          "futures_positions": [{"symbol": "ETHUSDTM", "currentQty": 5, "avgEntryPrice": 100.0, "markPrice": 100.0, "realLeverage": 3}],
                          "futures_stop_orders": []})()
    actions = mgr.run(snap)
    assert not any(a.get("action") == "emergency_bracket" for a in actions)
    assert "ETHUSDTM" in mgr._naked_since  # armed for next poll


@pytest.mark.parametrize(
    ("failed_direction", "failed_leg", "placed_leg"),
    [
        ("down", "stopLoss", "takeProfit"),
        ("up", "takeProfit", "stopLoss"),
    ],
)
def test_emergency_bracket_tries_both_legs_and_reports_partial_failure(
    failed_direction, failed_leg, placed_leg,
):
    calls = []

    class Client:
        def place_order(self, req):
            calls.append(req.stop)
            if req.stop == failed_direction:
                raise RuntimeError(f"{req.stop} rejected")
            return SimpleNamespace(orderId=f"{req.stop}-order")

    mgr = ProtectionManager(
        _cfg(dry_run=False), Client(), notifier=None, emergency_sl_pct=0.02, min_rr=1.5,
    )
    mgr._tick_cache["ETHUSDTM"] = 0.0

    rec = mgr._ensure_emergency_bracket(
        "ETHUSDTM", {"currentQty": 5, "realLeverage": 3}, side_long=True, avg_entry=100.0,
    )

    assert calls == ["down", "up"]
    assert rec["result"] == "partial"
    assert rec["legs"][failed_leg]["placed"] is False
    assert rec["legs"][placed_leg]["placed"] is True
    assert failed_leg in rec["errors"]


def test_failed_emergency_sl_retries_without_new_grace_or_duplicate_tp():
    calls = []

    class Client:
        def place_order(self, req):
            calls.append(req.stop)
            if req.stop == "down":
                raise RuntimeError("SL rejected")
            return SimpleNamespace(orderId="tp-order")

    mgr = ProtectionManager(
        _cfg(dry_run=False), Client(), notifier=None, emergency_sl_pct=0.02, min_rr=1.5,
    )
    mgr._tick_cache["ETHUSDTM"] = 0.0
    first_seen = time.time() - mgr._emergency_grace_sec - 1
    mgr._naked_since["ETHUSDTM"] = first_seen
    snap = SimpleNamespace(
        futures_enabled=True,
        futures_positions=[{
            "symbol": "ETHUSDTM", "currentQty": 5, "avgEntryPrice": 100.0,
            "markPrice": 100.0, "realLeverage": 3, "openingTimestamp": 1000,
        }],
        futures_stop_orders=[],
    )

    first = mgr.run(snap)
    assert first[0]["legs"]["stopLoss"]["placed"] is False
    assert first[0]["legs"]["takeProfit"]["placed"] is True
    assert mgr._naked_since["ETHUSDTM"] == first_seen
    assert calls == ["down", "up"]

    second = mgr.run(snap)
    assert second[0]["legs"]["stopLoss"]["placed"] is False
    assert second[0]["legs"]["takeProfit"]["existing"] is True
    assert calls == ["down", "up", "down"]
    assert mgr._naked_since["ETHUSDTM"] == first_seen


# ── Atomic stop replacement ─────────────────────────────────────────────────────


def test_breakeven_stop_is_confirmed_before_old_stop_is_cancelled():
    events = []

    class Client:
        def place_order(self, req):
            events.append(("place", req.stopPrice))
            return SimpleNamespace(orderId="new-stop")

        def cancel_order(self, order_id, symbol=None):
            events.append(("cancel", order_id))
            return {}

    mgr = ProtectionManager(_cfg(dry_run=False), Client())
    mgr._tick_cache["ETHUSDTM"] = 0.0
    result = mgr._move_stop_to_breakeven(
        "ETHUSDTM",
        {"currentQty": 5, "realLeverage": 3},
        side_long=True,
        stops=[
            {
                "symbol": "ETHUSDTM", "side": "sell", "stop": "down", "id": "old-stop",
                "reduceOnly": True,
            },
            {"symbol": "ETHUSDTM", "stop": "up", "id": "take-profit"},
            {"symbol": "ETHUSDTM", "side": "sell", "stop": "down", "id": "entry-stop"},
            {
                "symbol": "ETHUSDTM", "side": "buy", "stop": "down", "id": "wrong-side",
                "reduceOnly": True,
            },
        ],
        be_price=100.15,
    )

    assert events == [("place", "100.15"), ("cancel", "old-stop")]
    assert result["orderId"] == "new-stop"
    assert result["cancelled"] == ["old-stop"]


def test_unconfirmed_breakeven_stop_keeps_old_stop_live():
    cancelled = []

    class Client:
        def place_order(self, req):
            return SimpleNamespace(orderId="")

        def cancel_order(self, order_id, symbol=None):
            cancelled.append(order_id)

    mgr = ProtectionManager(_cfg(dry_run=False), Client())
    mgr._tick_cache["ETHUSDTM"] = 0.0

    with pytest.raises(RuntimeError, match="not confirmed"):
        mgr._move_stop_to_breakeven(
            "ETHUSDTM",
            {"currentQty": 5, "realLeverage": 3},
            side_long=True,
            stops=[{
                "symbol": "ETHUSDTM", "side": "sell", "stop": "down", "id": "old-stop",
                "closeOrder": "true",
            }],
            be_price=100.15,
        )

    assert cancelled == []


def test_current_stop_only_trusts_reduce_only_exit_orders_with_correct_side():
    mgr = ProtectionManager(_cfg(), kucoin_futures=None)
    stops = [
        {"symbol": "ETHUSDTM", "side": "sell", "stop": "down", "stopPrice": "80"},
        {
            "symbol": "ETHUSDTM", "side": "buy", "stop": "down", "stopPrice": "85",
            "reduceOnly": True,
        },
        {
            "symbol": "ETHUSDTM", "side": "sell", "stop": "up", "stopPrice": "90",
            "reduceOnly": True,
        },
        {
            "symbol": "ETHUSDTM", "side": "sell", "stop": "down", "stopPrice": "95",
            "closeOrder": "true",
        },
    ]

    assert mgr._current_stop_price("ETHUSDTM", True, stops) == 95.0


@pytest.mark.parametrize(
    ("tick", "price", "expected"),
    [(0.25, 100.13, 100.25), (0.0025, 1.0037, 1.0025), (5.0, 102.6, 105.0)],
)
def test_tick_rounding_supports_arbitrary_increments(tick, price, expected):
    mgr = ProtectionManager(_cfg(), kucoin_futures=None)
    mgr._tick_cache["ETHUSDTM"] = tick
    assert mgr._round_to_tick("ETHUSDTM", price) == expected


@pytest.mark.parametrize(
    "changed",
    [
        {"openingTimestamp": 2000},
        {"currentQty": 6},
        {"avgEntryPrice": 101.0, "markPrice": 102.0},
    ],
)
def test_excursion_lifecycle_resets_on_open_time_quantity_or_average_change(changed):
    mgr = ProtectionManager(
        _cfg(breakeven_trigger_r=100.0, giveback_pct=0.0, early_cut_enabled=False, trail_enabled=False),
        kucoin_futures=object(),
    )
    stop = {
        "symbol": "ETHUSDTM", "side": "sell", "stop": "down", "stopPrice": "90",
        "reduceOnly": True,
    }
    initial = {
        "symbol": "ETHUSDTM", "currentQty": 5, "avgEntryPrice": 100.0,
        "markPrice": 105.0, "openingTimestamp": 1000,
    }
    snap = SimpleNamespace(
        futures_enabled=True, futures_positions=[initial], futures_stop_orders=[stop],
    )
    assert mgr.run(snap) == []
    assert mgr._peak_fe["ETHUSDTM"] == 5.0
    mgr._open_since["ETHUSDTM"] = 1.0

    updated = dict(initial)
    updated.update(changed)
    updated.setdefault("markPrice", 101.0)
    if "markPrice" not in changed:
        updated["markPrice"] = 101.0
    snap.futures_positions = [updated]

    assert mgr.run(snap) == []
    expected_fe = float(updated["markPrice"]) - float(updated["avgEntryPrice"])
    assert mgr._peak_fe["ETHUSDTM"] == expected_fe
    assert mgr._open_since["ETHUSDTM"] > 1.0


# ── Hard unrealized-loss cap ────────────────────────────────────────────────────


def test_unrealized_loss_cap_is_strict_and_safe_on_missing_inputs():
    assert should_close_for_unrealized_loss(
        unrealized_pnl=-10.01, equity=1000, max_loss_equity_fraction=0.01,
    ) is True
    assert should_close_for_unrealized_loss(
        unrealized_pnl=-10.0, equity=1000, max_loss_equity_fraction=0.01,
    ) is False
    assert should_close_for_unrealized_loss(
        unrealized_pnl=-100, equity=0, max_loss_equity_fraction=0.01,
    ) is False
    assert should_close_for_unrealized_loss(
        unrealized_pnl=-100, equity=1000, max_loss_equity_fraction=0,
    ) is False


def test_manager_closes_when_live_unrealized_loss_exceeds_equity_budget():
    mgr = ProtectionManager(
        _cfg(enabled=False, dry_run=True),
        kucoin_futures=object(),
        max_loss_equity_fraction=0.01,
    )
    snap = type(
        "S",
        (),
        {
            "futures_enabled": True,
            "total_usdt": 1000.0,
            "futures_account": {},
            "futures_positions": [
                {"symbol": "ETHUSDTM", "currentQty": 5, "unrealisedPnl": -10.01},
            ],
            "futures_stop_orders": [],
        },
    )()

    actions = mgr.run(snap)

    assert len(actions) == 1
    assert actions[0]["action"] == "close"
    assert actions[0]["dryRun"] is True
    assert "hard unrealized-loss cap" in actions[0]["reason"]


# --- funding-carry thesis hold -------------------------------------------------------------------

def test_next_funding_settlement_is_on_the_8h_utc_grid():
  from src.regime import next_funding_settlement
  # 2026-09-01 21:47 UTC -> next settlement is 2026-09-02 00:00 UTC
  import datetime as _dt
  now = _dt.datetime(2026, 9, 1, 21, 47, tzinfo=_dt.UTC).timestamp()
  nxt = next_funding_settlement(now)
  assert _dt.datetime.fromtimestamp(nxt, _dt.UTC) == _dt.datetime(2026, 9, 2, 0, 0, tzinfo=_dt.UTC)
  # exactly on a boundary rolls to the NEXT one, never returns "now"
  assert next_funding_settlement(nxt) > nxt


def test_carry_hold_deadline_only_applies_to_declared_carry_trades():
  from src.regime import carry_hold_deadline
  import datetime as _dt
  fill = _dt.datetime(2026, 9, 1, 21, 47, tzinfo=_dt.UTC).timestamp()
  now = fill + 600
  assert carry_hold_deadline({"setupFamily": "funding_carry", "fillTs": fill}, now) is not None
  for other in ("continuation", "fade_extreme", "range_edge", "breakout", None):
    assert carry_hold_deadline({"setupFamily": other, "fillTs": fill}, now) is None
  assert carry_hold_deadline(None, now) is None


def test_carry_hold_deadline_expires_after_the_settlement_it_was_opened_for():
  """The deadline anchors on the FILL, so a carried trade becomes normally managed afterwards
  instead of rolling its hold forward to the next cycle forever."""
  from src.regime import carry_hold_deadline
  import datetime as _dt
  fill = _dt.datetime(2026, 9, 1, 21, 47, tzinfo=_dt.UTC).timestamp()
  settle = _dt.datetime(2026, 9, 2, 0, 0, tzinfo=_dt.UTC).timestamp()
  ctx = {"setupFamily": "funding_carry", "fillTs": fill}
  assert carry_hold_deadline(ctx, settle - 60) == settle
  assert carry_hold_deadline(ctx, settle + 1) is None


def test_hold_window_suppresses_early_profit_taking_but_not_the_stop():
  """The 0G-USDT case: a carry trade peaked at ~1R and was ratcheted+shaken out 142min in, collecting
  the transfer only by luck. Inside the hold window the code must leave the model's bracket alone."""
  cfg = _cfg(breakeven_trigger_r=0.5, trail_arm_r=0.5)  # the LIVE arm; see test_trail_default_arm_is_half_r
  entry, risk = 100.0, 10.0
  peak = 0.97 * risk
  base = dict(side_long=True, avg_entry=entry, sl_price=entry - risk, peak_fe=peak,
              cfg=cfg, opened_min_ago=120.0, risk_override=risk)
  # Without a hold, protection ratchets the stop up (which is what gave back the 0G peak).
  assert decide_protection(mark=entry + peak, **base)["action"] == "move_breakeven"
  # With the settlement still ahead, it stands pat.
  held = decide_protection(mark=entry + peak, hold_until_ts=1_000.0, now_ts=0.0, **base)
  assert held["action"] == "none"
  assert "carry hold" in held["reason"]
  assert held["holdRemainingMin"] == pytest.approx(1000.0 / 60.0, abs=0.05)
  # Past the settlement the normal rules resume.
  assert decide_protection(mark=entry + peak, hold_until_ts=1_000.0, now_ts=1_001.0, **base)["action"] == "move_breakeven"


def test_hold_window_does_not_widen_or_remove_the_exchange_stop():
  """Suppressing early exits is only safe because the loss cap is untouched: the decision must never
  ask to move a stop while holding, so the model's original 1R remains live on the exchange."""
  cfg = _cfg(breakeven_trigger_r=0.5, trail_arm_r=0.5)
  entry, risk = 100.0, 10.0
  for mark in (entry - 0.9 * risk, entry, entry + 2.5 * risk):
    d = decide_protection(side_long=True, avg_entry=entry, mark=mark, sl_price=entry - risk,
                          peak_fe=max(0.0, mark - entry), cfg=cfg, opened_min_ago=200.0,
                          risk_override=risk, hold_until_ts=1_000.0, now_ts=0.0)
    assert d["action"] == "none"
    assert "stop_price" not in d and "new_sl" not in d


# --- noise-band trail ----------------------------------------------------------------------------

def test_trail_rides_one_noise_band_behind_the_peak():
  """The 35R give-back: across 81 closes the book reached +31R of favourable excursion and realised
  -4.5R. With trail_distance_r=1.0 the `peak - trail_r*risk` branch is negative until peak > 1.5R —
  which happened 0 times in 81 trades — so every winner banked a flat 33% of its peak instead."""
  cfg = _cfg(breakeven_trigger_r=0.5, trail_arm_r=0.5, trail_enabled=True,
             trail_distance_r=1.0, trail_lock_frac=0.33)
  entry, risk = 100.0, 10.0
  peak = 0.97 * risk                      # the 0G-USDT peak
  base = dict(side_long=True, avg_entry=entry, mark=entry + peak, sl_price=entry - risk,
              peak_fe=peak, cfg=cfg, opened_min_ago=60.0, risk_override=risk)

  flat = decide_protection(**base)
  locked_flat = (flat["stopPrice"] - entry) / risk
  assert flat["action"] == "move_breakeven"
  assert locked_flat == pytest.approx(0.33 * 0.97, abs=0.01)   # the flat-33% branch wins

  # stopAtrMult 2.5 -> the entry stop sits 2.5 ATR out, so one noise band is 0.4R.
  banded = decide_protection(noise_band_r=1.0 / 2.5, **base)
  locked_band = (banded["stopPrice"] - entry) / risk
  assert banded["action"] == "move_breakeven"
  assert locked_band == pytest.approx(0.97 - 0.4, abs=0.01)    # peak minus one noise band
  assert locked_band > locked_flat                              # strictly more of the run kept


def test_noise_band_never_reaches_inside_the_chop():
  """The band is a SAFETY FLOOR on give-back, not merely a narrowing of the configured trail.

  A measured band fully determines the give-back — the trail rides exactly one band behind the peak,
  wider than the config on noisy symbols and tighter on quiet ones. The invariant either way: the
  stop never sits closer to the peak than one noise band. That is exactly the stop-inside-noise geometry this bot was already burned by."""
  entry, risk = 100.0, 10.0
  cfg = _cfg(breakeven_trigger_r=0.5, trail_arm_r=0.5, trail_enabled=True,
             trail_distance_r=0.3, trail_lock_frac=0.0)
  base = dict(side_long=True, avg_entry=entry, sl_price=entry - risk, cfg=cfg,
              opened_min_ago=60.0, risk_override=risk)
  # Config wants a 0.3R trail but the symbol's noise band is 0.8R — the band must win.
  wide = decide_protection(mark=entry + 1.5 * risk, peak_fe=1.5 * risk, noise_band_r=0.8, **base)
  assert (wide["stopPrice"] - entry) / risk == pytest.approx(1.5 - 0.8, abs=0.01)
  # A quieter symbol gets a tighter trail than the config, for the same reason.
  tight = decide_protection(mark=entry + 1.5 * risk, peak_fe=1.5 * risk, noise_band_r=0.1, **base)
  assert (tight["stopPrice"] - entry) / risk == pytest.approx(1.5 - 0.1, abs=0.01)
  # Wherever the TRAIL is the rule that fires, its lock stays at least one band below the peak.
  # (A run shorter than one band cannot arm the trail at all; the separate fee-breakeven ratchet may
  # still move the stop to entry there, which is a level the price actually traded, not a retracement.)
  for atr_mult in (1.5, 2.5, 4.0):
    band = 1.0 / atr_mult
    for peak_r in (0.6, 1.0, 2.0):
      d = decide_protection(mark=entry + peak_r * risk, peak_fe=peak_r * risk,
                            noise_band_r=band, **dict(base, cfg=_cfg(
                              breakeven_trigger_r=0.5, trail_arm_r=0.5, trail_enabled=True,
                              trail_distance_r=1.0, trail_lock_frac=0.33)))
      if d.get("action") != "move_breakeven" or "trailing" not in d.get("reason", ""):
        continue
      locked_r = (d["stopPrice"] - entry) / risk
      assert locked_r <= peak_r - band + 1e-9, (atr_mult, peak_r, locked_r)
      assert peak_r >= band, (atr_mult, peak_r)


def test_trail_without_a_noise_band_is_unchanged():
  """Positions whose entry predates stopAtrMult being recorded must behave exactly as before."""
  cfg = _cfg(breakeven_trigger_r=0.5, trail_arm_r=0.5, trail_enabled=True,
             trail_distance_r=1.0, trail_lock_frac=0.33)
  entry, risk = 100.0, 10.0
  base = dict(side_long=True, avg_entry=entry, mark=entry + 8.0, sl_price=entry - risk,
              peak_fe=8.0, cfg=cfg, opened_min_ago=60.0, risk_override=risk)
  for bad in (None, 0.0, -1.0, float("nan"), "x"):
    assert decide_protection(noise_band_r=bad, **base)["stopPrice"] == pytest.approx(
      decide_protection(**base)["stopPrice"])



# ── restart safety: the manager's in-memory anchors are re-seeded from the recorded trade ─────────

def _restart_snap(*, stop_price, mark=108.0):
  """A long from 100 whose live stop sits at `stop_price`. After a breakeven ratchet that stop is
  ABOVE entry, which is exactly the case the live capture cannot handle on a fresh process.
  ETHUSDTM's real contract multiplier is 0.01 ETH, so USD PnL = move x contracts x 0.01 (the fixture
  used to imply multiplier 1, which hid that pnl/contracts is not a price)."""
  return SimpleNamespace(
    futures_enabled=True,
    futures_account={"accountEquity": 1000},
    futures_positions=[{
      "symbol": "ETHUSDTM", "currentQty": 5, "avgEntryPrice": 100.0,
      "markPrice": mark, "unrealisedPnl": (mark - 100.0) * 5 * 0.01,
      "openingTimestamp": 1000,
    }],
    futures_stop_orders=[{
      "symbol": "ETHUSDTM", "side": "sell", "stop": "down", "stopPrice": stop_price, "reduceOnly": True,
    }],
    total_usdt=1000,
  )


def _restart_mgr(ctx):
  cfg = _cfg(breakeven_trigger_r=0.5, trail_arm_r=0.5, trail_enabled=True, dry_run=True)
  # run() returns [] with no client at all; dry_run means the stub is never actually called.
  return ProtectionManager(cfg, SimpleNamespace(), trade_context_lookup=lambda fsym, pos: dict(ctx))


def test_after_a_restart_a_breakeven_winner_regains_its_1r_anchor_from_the_recorded_stop():
  """Live: a fresh process sees stop 100.15 > entry 100, so entry-stop <= 0 and the anchor is never
  captured -> risk_override None -> trail/breakeven/early-cut silently inert for the position's whole
  remaining life. The recorded entry stop (entry 100, stop 90 => 10.0) restores it."""
  inert = _restart_mgr({})
  inert.run(_restart_snap(stop_price=100.15))
  assert "ETHUSDTM" not in inert._init_risk, "sanity: without seeding the anchor is lost"

  seeded = _restart_mgr({"initRiskPx": 10.0, "noiseBandR": 0.4})
  actions = seeded.run(_restart_snap(stop_price=100.15))
  assert seeded._init_risk["ETHUSDTM"] == pytest.approx(10.0)
  # ...and the R-based trail is ALIVE again: mark 108 = +0.8R peak -> it ratchets the stop up.
  assert any(a.get("action") == "move_breakeven" for a in actions), actions


def test_live_capture_wins_over_the_recorded_stop_when_a_real_stop_is_visible():
  """On a normal (non-restart) poll the stop is still below entry; the live distance is the truth
  about what is actually placed and must not be overridden by the recorded plan."""
  mgr = _restart_mgr({"initRiskPx": 10.0})
  mgr.run(_restart_snap(stop_price=92.0, mark=101.0))     # live risk = 8
  assert mgr._init_risk["ETHUSDTM"] == pytest.approx(8.0)


def test_recorded_peak_can_only_raise_the_in_memory_peak():
  """A restart resets the peak to the current mark, which would let the trail re-arm from scratch.
  The recorded peak (pnl / qty) restores what this lifecycle already reached — and never lowers it."""
  higher = _restart_mgr({"initRiskPx": 10.0, "peakFePx": 12.0})
  higher.run(_restart_snap(stop_price=100.15, mark=108.0))          # live peak = 8
  assert higher._peak_fe["ETHUSDTM"] == pytest.approx(12.0)
  lower = _restart_mgr({"initRiskPx": 10.0, "peakFePx": 5.0})
  lower.run(_restart_snap(stop_price=100.15, mark=108.0))
  assert lower._peak_fe["ETHUSDTM"] == pytest.approx(8.0)


def test_junk_context_values_never_seed_anything():
  for junk in ({"initRiskPx": 0}, {"initRiskPx": -3}, {"initRiskPx": "x"}, {"peakFePx": "nan"}, None):
    mgr = ProtectionManager(_cfg(trail_enabled=True, dry_run=True), SimpleNamespace(),
                            trade_context_lookup=lambda f, p, j=junk: j)
    mgr.run(_restart_snap(stop_price=100.15))
    assert "ETHUSDTM" not in mgr._init_risk


# --- restart-safe trail peak through the REAL writer, lookup and manager (2026-09-25) -------------
#
# The Sep 22 seed never fired: the lookup keyed on KuCoin's raw positionSide ('BOTH') while the writer
# stored 'long'/'short', and its value was peakPnl/|contracts| = price x contract MULTIPLIER. Its only
# test grepped source strings, which both bugs passed. Everything below executes the production
# path — main._live_extremes_map -> MemoryStore.update_position_extremes (writer), then
# main._make_trade_context_lookup -> position_context.trade_context (reader) inside
# ProtectionManager.run — on exchange-shaped payloads, in the loop's own order (writer, then manager).

_FSYM = "HUSDTM"
_OPENED = 1790252000000


def _book_snap(*, qty, entry, mark, stop, mult=10.0, opened=_OPENED, side_field="BOTH"):
  """One live position as KuCoin returns it: positionSide 'BOTH' (one-way mode), USD PnL scaled by the
  contract multiplier, and its reduce-only loss-side stop."""
  is_long = qty > 0
  return SimpleNamespace(
    futures_enabled=True,
    futures_account={"accountEquity": 1000},
    total_usdt=1000,
    futures_positions=[{
      "symbol": _FSYM, "currentQty": qty, "avgEntryPrice": entry, "markPrice": mark,
      "unrealisedPnl": (mark - entry) * qty * mult, "openingTimestamp": opened,
      "positionSide": side_field,
    }],
    futures_stop_orders=[{
      "symbol": _FSYM, "side": "sell" if is_long else "buy", "stop": "down" if is_long else "up",
      "stopPrice": stop, "reduceOnly": True,
    }],
  )


def _record_entry(store, *, side="long", entry=1.0, stop=0.98, opened=_OPENED, atr_mult=2.0):
  """The filled entry behind the position, so the lookup can restore its 1R anchor after a restart."""
  store.record_trade(
    "H-USDT", "buy" if side == "long" else "sell", 5.0, price=entry, venue="futures",
    entry_context={"positionSide": side, "entryPrice": entry, "stopLossPrice": stop,
                   "stopAtrMult": atr_mult, "setupFamily": "continuation"},
  )
  data = store._read()
  data["trades"][-1]["fillTs"] = opened // 1000
  store._write(data)


def _live_lookup_mgr(store):
  from src.main import _make_trade_context_lookup
  cfg = _cfg(breakeven_trigger_r=0.5, trail_arm_r=0.5, trail_enabled=True, dry_run=True)
  return ProtectionManager(cfg, SimpleNamespace(), trade_context_lookup=_make_trade_context_lookup(store, None))


def _drive(monkeypatch, store, steps, *, restart_at=None):
  """Run the loop's order per poll (writer, then manager) over ``steps`` = [(qty, entry, mark)], moving
  the exchange stop exactly as the manager decides. A restart replaces the manager (empty in-process
  state) and keeps the memory file, as a real deploy does. Returns (decisions, final manager)."""
  import src.protection as prot
  from src.main import _live_extremes_map
  captured = []
  real = prot.decide_protection

  def _capture(**kwargs):
    out = real(**kwargs)
    captured.append(out)
    return out

  monkeypatch.setattr(prot, "decide_protection", _capture)
  mgr = _live_lookup_mgr(store)
  stop = 0.98
  decisions = []
  for i, (qty, entry, mark) in enumerate(steps):
    if restart_at is not None and i == restart_at:
      mgr = _live_lookup_mgr(store)
    snap = _book_snap(qty=qty, entry=entry, mark=mark, stop=stop)
    store.update_position_extremes(_live_extremes_map(snap))
    captured.clear()
    mgr.run(snap)
    d = dict(captured[-1]) if captured else {"action": "skipped"}
    decisions.append((d["action"], d.get("stopPrice"), d.get("reason")))
    if d["action"] == "move_breakeven":
      stop = float(d["stopPrice"])
    if d["action"] == "close":
      break
  return decisions, mgr


# A long from 1.000, stop 0.980 (1R = 0.020), three contracts of multiplier 10. It ratchets, peaks at
# +1.1R (1.022) with the last +0.2R of lock still unplaced (under the 0.25R min step), then retraces to
# 1.011 — through the lock the uninterrupted manager computes from the true peak.
_PATH = [(3, 1.0, m) for m in (1.000, 1.006, 1.012, 1.018, 1.022, 1.016, 1.021, 1.011)]
_RESTART = 5   # the process restarts after the 1.022 peak


def test_restart_plus_seed_gives_the_same_decisions_as_an_uninterrupted_manager(tmp_path, monkeypatch):
  """The criterion is 'a restart must be invisible'. Without a working seed, the fresh process
  measures its peak from 1.016 and keeps the position open at 1.011 that the uninterrupted manager
  closes; with it, every decision — action, stop and reason text (which prints the peak) — matches."""
  from src.memory import MemoryStore
  straight_store = MemoryStore(str(tmp_path / "straight.json"))
  _record_entry(straight_store)
  straight, _ = _drive(monkeypatch, straight_store, _PATH)
  restarted_store = MemoryStore(str(tmp_path / "restarted.json"))
  _record_entry(restarted_store)
  restarted, mgr = _drive(monkeypatch, restarted_store, _PATH, restart_at=_RESTART)
  assert straight[-1][0] == "close" and "trailing stop hit" in straight[-1][2], straight
  assert restarted == straight
  assert mgr._peak_fe[_FSYM] == pytest.approx(0.022)


def test_a_restart_on_a_multiplier_10_contract_seeds_exactly_the_true_price_peak(tmp_path, monkeypatch):
  """positionSide 'BOTH' still matches, and the seed is the PRICE excursion (0.022), not the old
  peakPnl/contracts (= 0.022 x multiplier 10 = 0.22), which would have market-closed the winner."""
  from src.main import _live_extremes_map
  from src.memory import MemoryStore
  store = MemoryStore(str(tmp_path / "m.json"))
  _record_entry(store)
  _drive(monkeypatch, store, _PATH[:_RESTART])
  ext = store.get_position_extremes("H-USDT")
  assert ext["peakFePx"] == pytest.approx(0.022)
  assert ext["peakPnl"] / 3 == pytest.approx(0.22), "sanity: the old USD/contracts conversion is 10x"
  fresh = _live_lookup_mgr(store)
  snap = _book_snap(qty=3, entry=1.0, mark=1.016, stop=1.008)
  store.update_position_extremes(_live_extremes_map(snap))
  fresh.run(snap)
  assert fresh._peak_fe[_FSYM] == pytest.approx(0.022)


def test_ordinary_polls_never_inflate_the_peak(tmp_path, monkeypatch):
  """With no restart, the in-process peak is exactly the running max of (mark - entry) — on every
  poll, on a multiplier-10 contract — whatever the recorder holds."""
  from src.memory import MemoryStore
  store = MemoryStore(str(tmp_path / "m.json"))
  _record_entry(store)
  from src.main import _live_extremes_map
  mgr = _live_lookup_mgr(store)
  running = float("-inf")
  for _, _, mark in _PATH[:-1]:
    snap = _book_snap(qty=3, entry=1.0, mark=mark, stop=0.98)
    store.update_position_extremes(_live_extremes_map(snap))
    mgr.run(snap)
    running = max(running, mark - 1.0)
    assert mgr._peak_fe[_FSYM] == pytest.approx(running)


def test_the_seed_applies_only_on_the_first_sighting_of_a_lifecycle():
  """Seeding every poll is the channel through which a unit or key bug becomes a market close. A
  lookup that starts offering a (bogus) larger peak on a LATER poll must be ignored."""
  offers = iter([{}, {"peakFePx": 50.0}, {"peakFePx": 50.0}])
  cfg = _cfg(breakeven_trigger_r=0.5, trail_arm_r=0.5, trail_enabled=True, dry_run=True)
  mgr = ProtectionManager(cfg, SimpleNamespace(), trade_context_lookup=lambda f, p: next(offers))
  for _ in range(3):
    actions = mgr.run(_restart_snap(stop_price=92.0, mark=101.0))
    assert not any(a.get("action") == "close" for a in actions), actions
  assert mgr._peak_fe["ETHUSDTM"] == pytest.approx(1.0)


def test_a_partial_reduction_does_not_seed_above_the_post_reset_peak(tmp_path, monkeypatch):
  """WIF 2026-09-23 closed in three chunks. The lifecycleKey (openTime:side) survives a reduction, so
  a peak keyed only on it — or the USD peak / the SMALLER current size — hands the reset manager a peak
  it deliberately threw away (a 0.53R true peak re-seeded as 1.06R closed at +0.38R in replay)."""
  from src.memory import MemoryStore
  store = MemoryStore(str(tmp_path / "m.json"))
  _record_entry(store)
  steps = [(2, 1.0, 1.000), (2, 1.0, 1.018), (1, 1.0, 1.010), (1, 1.0, 1.012)]
  _, live = _drive(monkeypatch, store, steps)
  assert live._peak_fe[_FSYM] == pytest.approx(0.012)          # in-process: reset at the reduction
  assert store.get_position_extremes("H-USDT")["peakFePx"] == pytest.approx(0.012)
  _, restarted = _drive(monkeypatch, store, steps[-1:])
  assert restarted._peak_fe[_FSYM] == pytest.approx(0.012), "must not re-seed the pre-reduction 0.018"


def test_an_add_on_does_not_re_seed_the_old_peak(tmp_path, monkeypatch):
  """An add-on moves avgEntry; ProtectionManager resets because 'keeping the prior peak against a
  changed average entry can immediately produce a false give-back'. The recorder must reset with it."""
  from src.memory import MemoryStore
  store = MemoryStore(str(tmp_path / "m.json"))
  _record_entry(store)
  steps = [(1, 1.0, 1.000), (1, 1.0, 1.018), (2, 1.009, 1.018), (2, 1.009, 1.016)]
  _, live = _drive(monkeypatch, store, steps)
  assert live._peak_fe[_FSYM] == pytest.approx(0.009)
  _, restarted = _drive(monkeypatch, store, [(2, 1.009, 1.015)])
  assert restarted._peak_fe[_FSYM] == pytest.approx(0.009)


def test_a_flip_does_not_carry_the_old_sides_peak(tmp_path):
  """Even with the SAME openingTimestamp (the adversarial case), a long's peak must never seed the
  short that replaces it: side is part of the key, derived from the sign of qty."""
  from src.memory import MemoryStore
  from src.main import _live_extremes_map
  store = MemoryStore(str(tmp_path / "m.json"))
  mgr = _live_lookup_mgr(store)
  for mark in (1.000, 1.018):
    snap = _book_snap(qty=2, entry=1.0, mark=mark, stop=0.98)
    store.update_position_extremes(_live_extremes_map(snap))
    mgr.run(snap)
  flip = SimpleNamespace(**{**vars(_book_snap(qty=-2, entry=1.015, mark=1.013, stop=1.035)),
                            "futures_stop_orders": [{"symbol": _FSYM, "side": "buy", "stop": "up",
                                                     "stopPrice": 1.035, "reduceOnly": True}]})
  store.update_position_extremes(_live_extremes_map(flip))
  mgr.run(flip)
  assert mgr._peak_fe[_FSYM] == pytest.approx(0.002)
  fresh = _live_lookup_mgr(store)
  fresh.run(flip)
  assert fresh._peak_fe[_FSYM] == pytest.approx(0.002)


def test_reader_and_writer_build_the_key_with_the_same_helper():
  """One helper, both sides, so the two keys cannot drift apart again (the Sep 22 reader built its own
  f-string from a different field than the writer)."""
  import inspect
  import src.memory as memory_mod
  import src.position_context as pc
  writer = inspect.getsource(memory_mod.MemoryStore._update_peak_fe)
  reader = inspect.getsource(pc.trade_context)
  assert "peak_fe_key(" in writer and "peak_fe_key(" in reader
  assert 'get("positionSide")' not in reader, "the reader must derive side from currentQty, never positionSide"
  assert 'get("peakPnl")' not in reader, "no USD-to-price conversion may come back"


@pytest.mark.parametrize("pos", [
  {"openingTimestamp": _OPENED, "currentQty": 3, "avgEntryPrice": 1.0},
  {"openingTime": "1790252000000", "currentQty": "-2", "avgEntryPrice": "0.06008"},
  {"openTime": 1790252000.5, "createdAt": 5, "currentQty": 11, "avgEntryPrice": 0.06},
  {"createdAt": "2026-09-23T11:52:00Z", "currentQty": -1, "avgEntryPrice": 2.5},
  {"currentQty": 4, "avgEntryPrice": 3.0},
])
def test_peak_key_is_the_same_identity_protection_resets_on(pos):
  """Same identity as ProtectionManager._position_signature: equal keys iff equal signatures."""
  from src.memory import peak_fe_key, position_open_time
  qty = float(pos["currentQty"])
  sig = ProtectionManager._position_signature(pos, qty > 0, qty, float(pos["avgEntryPrice"]))
  key = peak_fe_key(position_open_time(pos), pos["currentQty"], pos["avgEntryPrice"])
  assert key == f"{sig[0]}|{'long' if sig[1] else 'short'}|{abs(sig[2])!r}|{sig[3]!r}"
  for changed in ({"currentQty": qty * 2}, {"avgEntryPrice": float(pos["avgEntryPrice"]) * 1.01},
                  {"currentQty": -qty}):
    other = {**pos, **changed}
    oq = float(other["currentQty"])
    assert ProtectionManager._position_signature(other, oq > 0, oq, float(other["avgEntryPrice"])) != sig
    assert peak_fe_key(position_open_time(other), other["currentQty"], other["avgEntryPrice"]) != key


# ── replay_protection_stack: the benchmark an agent close is scored against (2026-09-25) ────────────
# exitDiscipline used to score the model's early closes against the BARE bracket, which the system never
# leaves a position on. On the 5 attributed closes at the time: -6.23R vs the bracket, ~-3.5R vs the live
# stack; all of the gap was DASH and INJ, whose trail would have exited near +0.3..0.5R long before the TP.

from src.protection import replay_protection_stack  # noqa: E402


# ── T5: the READER's own peakFeKey check (the Sep 22 fix), exercised without the writer ────────────────

_LONG_POS = {"symbol": _FSYM, "currentQty": 2, "avgEntryPrice": 1.0, "markPrice": 1.018,
             "openingTimestamp": _OPENED, "positionSide": "BOTH"}


@pytest.mark.parametrize("changed", [
    {"currentQty": -2, "avgEntryPrice": 1.015},   # flip, same openingTimestamp
    {"currentQty": 3, "avgEntryPrice": 1.009},    # add-on
    {"currentQty": 1},                            # partial reduction
    {"openingTimestamp": _OPENED + 60_000},       # close / reopen
])
def test_reader_never_returns_a_peak_recorded_under_another_key(tmp_path, changed):
    """If the extremes write fails on a flip/add-on poll (main only logs a WARNING), the reader alone
    must refuse the old lifecycle's peak, or the fresh manager seeds it and trail-closes."""
    from src.memory import MemoryStore
    from src.main import _live_extremes_map
    from src.position_context import trade_context
    store = MemoryStore(str(tmp_path / "m.json"))
    for mark in (1.000, 1.018):
        store.update_position_extremes(_live_extremes_map(_book_snap(qty=2, entry=1.0, mark=mark, stop=0.98)))
    assert trade_context(store, _FSYM, _LONG_POS, 0.0)["peakFePx"] == pytest.approx(0.018)   # exact key
    assert "peakFePx" not in trade_context(store, _FSYM, {**_LONG_POS, **changed}, 0.0)

_RT = 1_790_000_040          # a minute-aligned fill time (seconds)


def _rbar(i, o, h, l, c):
    """A raw FUTURES kline row for minute ``i`` after the fill: [ts_ms, open, HIGH, LOW, close, vol, turnover]."""
    return [(_RT + 60 * i) * 1000, o, h, l, c, 10.0, 1.0]


# Long 100, stop 98 (1R = 2), target 106 (+3R). Runs to +2R, pulls back through the trail, THEN tags TP.
_ARM_THEN_PULLBACK = [
    _rbar(0, 100.0, 100.5, 99.8, 100.4),
    _rbar(1, 100.4, 102.2, 100.3, 102.0),     # +1R close: the trail arms (band 0.5R) -> stop 101
    _rbar(2, 102.0, 104.2, 101.9, 104.0),     # +2R close: stop ratchets to 103
    _rbar(3, 104.0, 104.1, 102.5, 102.8),     # retrace through 103: the trail's stop fills
    _rbar(4, 102.8, 105.0, 102.6, 104.9),
    _rbar(5, 104.9, 106.5, 104.8, 106.2),     # the bracket's TP, reached only after the pullback
]


def _replay_path(bars, *, end_min=480, **kw):
    args = dict(side_long=True, entry=100.0, stop=98.0, take_profit=106.0, cfg=_cfg(),
                init_risk=2.0, fill_ts=_RT, end_ts=_RT + end_min * 60, noise_band_r=0.5)
    args.update(kw)
    return replay_protection_stack(bars, **args)


def test_replay_scores_a_trail_exit_below_the_brackets_target():
    out = _replay_path(_ARM_THEN_PULLBACK)
    assert out["resolvedBy"] == "trail_stop"
    assert out["stackR"] == pytest.approx(1.5)                 # stopped at 103, not the +3R target
    assert out["stackR"] < 3.0
    assert out["resolvedTs"] == _RT + 4 * 60
    assert out["preCloseExitSuppressed"] is False


def test_replay_a_straight_path_scores_the_target():
    bars = [_rbar(i, 100.0 + i, 101.0 + i, 100.0 + i, 101.0 + i) for i in range(7)]   # never retraces
    out = _replay_path(bars)
    assert out["resolvedBy"] == "take_profit"
    assert out["stackR"] == pytest.approx(3.0)


def test_replay_a_stop_path_scores_minus_one():
    bars = [_rbar(0, 100.0, 100.1, 99.5, 99.6), _rbar(1, 99.6, 99.7, 98.6, 98.7),
            _rbar(2, 98.7, 98.8, 97.9, 98.1)]
    out = _replay_path(bars)
    assert out["resolvedBy"] == "stop"
    assert out["stackR"] == pytest.approx(-1.0)


def test_replay_carry_hold_keeps_the_trail_off_until_the_settlement():
    held = _replay_path(_ARM_THEN_PULLBACK, hold_until_ts=_RT + 10 * 60)
    assert held["resolvedBy"] == "take_profit"                 # no ratchet while held: rides to TP
    assert held["stackR"] == pytest.approx(3.0)
    lifted = _replay_path(_ARM_THEN_PULLBACK, hold_until_ts=_RT + 60)   # hold over after the first bar
    assert lifted["resolvedBy"] == "trail_stop"


def test_replay_refuses_a_bar_whose_high_is_below_its_close():
    """KuCoin FUTURES rows are [ts, o, HIGH, LOW, close]; SPOT is [ts, o, close, high, low]. Mixing them
    up once reversed a finding, so a row that cannot be futures-ordered must abort the replay."""
    with pytest.raises(ValueError):
        _replay_path([_rbar(0, 100.0, 100.5, 99.0, 101.0)])    # high 100.5 < close 101
    with pytest.raises(ValueError):
        _replay_path([[_RT * 1000, 100.0, 100.4, 100.6, 99.9]])    # the spot layout read as futures


def test_replay_takes_the_stop_when_both_legs_sit_in_one_bar():
    out = _replay_path([_rbar(0, 100.0, 106.5, 97.5, 101.0)])
    assert out["resolvedBy"] == "stop" and out["stackR"] == pytest.approx(-1.0)


def test_replay_marks_to_market_at_the_horizon():
    bars = [_rbar(i, 100.2, 100.3, 100.1, 100.2) for i in range(30)]
    out = _replay_path(bars, end_min=20)
    assert out["resolvedBy"] == "expired"
    assert out["stackR"] == pytest.approx(0.1)
    assert out["resolvedTs"] == _RT + 20 * 60                  # bars past the horizon are ignored


def test_replay_never_exits_before_the_known_close_but_keeps_the_state():
    """The model closed at ``open_until_ts``: the live stack demonstrably had not exited by then, so a
    replay exit before it is a replay error (1m last price vs the mark) — skipped, with the ratcheted
    stop carried into the counterfactual."""
    bars = _ARM_THEN_PULLBACK[:4] + [_rbar(4, 102.8, 103.5, 102.6, 103.2)] + _ARM_THEN_PULLBACK[5:]
    out = _replay_path(bars, open_until_ts=_RT + 250)
    assert out["preCloseExitSuppressed"] is True
    assert out["resolvedBy"] == "trail_stop"
    assert out["stackR"] == pytest.approx(1.4)                 # opened through the 103 stop at 102.8
    with pytest.raises(ValueError):
        _replay_path([], end_min=10)                           # no bars: the caller retries later


def test_replay_short_symmetry():
    bars = [_rbar(0, 100.0, 100.2, 99.6, 99.6), _rbar(1, 99.6, 99.7, 97.9, 98.0),
            _rbar(2, 98.0, 98.1, 95.9, 96.0), _rbar(3, 96.0, 97.5, 95.9, 97.2)]
    out = _replay_path(bars, side_long=False, stop=102.0, take_profit=94.0)
    assert out["resolvedBy"] == "trail_stop"
    assert out["stackR"] == pytest.approx(1.5)                 # short from 100, trail stop at 97
    # T7: the same path whose last bar OPENS through the 97 trail stop fills at the worse open.
    gapped = bars[:3] + [_rbar(3, 97.4, 97.6, 97.1, 97.2)]
    out = _replay_path(gapped, side_long=False, stop=102.0, take_profit=94.0)
    assert out["resolvedBy"] == "trail_stop" and out["stackR"] == pytest.approx(1.3)


def test_replay_a_short_gapping_through_its_stop_fills_at_the_open():
    """T7: only the long side of 'a bar that opens through the stop fills at the open' was covered; a
    short credited the stop price instead of the worse open would flatter the stack benchmark."""
    out = _replay_path([_rbar(0, 102.6, 103.0, 102.4, 102.8)], side_long=False, stop=102.0, take_profit=94.0)
    assert out["resolvedBy"] == "stop"
    assert out["stackR"] == pytest.approx(-1.3)                # filled at 102.6, not the 102 stop


def test_replay_releases_the_carry_hold_at_the_settlement_bars_close():
    """T7: the hold is lifted on the bar whose CLOSE is the settlement (decide_protection holds only while
    time remains), so the trail ratchets on that bar; one second later it is still held on it."""
    bars = [_rbar(0, 100.0, 104.2, 99.9, 104.0), _rbar(1, 104.0, 104.1, 102.5, 102.8),
            _rbar(2, 102.8, 102.9, 102.7, 102.8)]
    at_close = _replay_path(bars, hold_until_ts=_RT + 60)
    assert at_close["resolvedBy"] == "trail_stop" and at_close["stackR"] == pytest.approx(1.5)
    later = _replay_path(bars, hold_until_ts=_RT + 61)
    assert later["resolvedBy"] == "trail_close" and later["stackR"] == pytest.approx(1.4)
