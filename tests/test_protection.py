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
        breakeven_trigger_r=1.0,
        breakeven_fee_pct=0.0015,
        giveback_pct=0.35,
        min_favorable_excursion_pct=0.005,
        no_chase_enabled=True,
        post_win_cooldown_minutes=45.0,
        no_chase_buffer_pct=0.001,
    )
    base.update(overrides)
    return ProfitProtectionConfig(**base)


# ── decide_protection: early invalidation (P1c) ─────────────────────────────────


def test_early_cut_fires_when_never_green_and_failing():
    # Long entry 100, stop 95 (risk 5). Never went green (peak 0.1), now -3.5 px = 70% to stop, 25min in.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=96.5, sl_price=95.0, peak_fe=0.1,
                          cfg=_cfg(), opened_min_ago=25.0)
    assert d["action"] == "close" and "early invalidation" in d["reason"]


def test_early_cut_skipped_within_grace():
    # Same failing trade but only 10min old (< 20min grace) → not cut yet.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=96.5, sl_price=95.0, peak_fe=0.1,
                          cfg=_cfg(), opened_min_ago=10.0)
    assert d["action"] == "none"


def test_early_cut_skipped_if_trade_went_green():
    # Peak reached +0.5 (> 0.3% of 100) → it "worked", so early-cut must NOT fire even if now underwater.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=96.5, sl_price=95.0, peak_fe=0.5,
                          cfg=_cfg(), opened_min_ago=25.0)
    assert d["action"] == "none"


def test_early_cut_skipped_if_not_yet_failing():
    # Never green, but only -1 px = 20% to the stop (< 60% threshold) → give it room, no cut.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=99.0, sl_price=95.0, peak_fe=0.05,
                          cfg=_cfg(), opened_min_ago=25.0)
    assert d["action"] == "none"


def test_early_cut_short_symmetry():
    # Short entry 100, stop 105 (risk 5). Never green, now 103.5 = 70% to stop, 25min → cut.
    d = decide_protection(side_long=False, avg_entry=100.0, mark=103.5, sl_price=105.0, peak_fe=0.1,
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
    # Long up only +0.5R, stop still below: nothing to do yet.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=102.5, sl_price=95.0, peak_fe=2.5, cfg=_cfg())
    assert d["action"] == "none"


def test_move_to_breakeven_at_one_r():
    # peak favourable excursion == initial risk (5 px) → ratchet stop to fee-adjusted breakeven.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=5.0, cfg=_cfg())
    assert d["action"] == "move_breakeven"
    assert d["stopPrice"] == 100.0 * 1.0015  # above entry by the fee buffer


def test_no_breakeven_when_stop_already_protective():
    # Stop already above entry (in profit) → no further ratchet needed.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=106.0, sl_price=101.0, peak_fe=6.0, cfg=_cfg())
    assert d["action"] == "none"


def test_short_breakeven_symmetry():
    d = decide_protection(side_long=False, avg_entry=100.0, mark=96.0, sl_price=105.0, peak_fe=5.0, cfg=_cfg())
    assert d["action"] == "move_breakeven"
    assert d["stopPrice"] == 100.0 * (1 - 0.0015)  # below entry for a short


# ── decide_protection: give-back cap ─────────────────────────────────────────────


def test_giveback_close_after_real_run():
    # Ran to +10 px, gave back to +4 px (>50% of peak) → close to lock the gain.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=10.0, cfg=_cfg())
    assert d["action"] == "close"


def test_no_giveback_when_peak_too_small():
    # Peak of 0.2 px is below the 0.3% min FE on a 100-priced asset → ignore noise.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=99.9, sl_price=95.0, peak_fe=0.2, cfg=_cfg())
    assert d["action"] == "none"


def test_giveback_disabled_when_pct_zero():
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=10.0, cfg=_cfg(giveback_pct=0.0))
    # With give-back off it should fall through to breakeven (peak 10 >= 1R of 5).
    assert d["action"] == "move_breakeven"


# ── decide_protection: give-back arming at 1R (giveback_arm_r) ───────────────────


def test_giveback_not_armed_below_one_r_run():
    """Sub-1R wobble is the original SL's job — the cap must not book fee-scale scratch wins.
    Risk = 5 px; peak ran only 2 px (0.4R, but > the 0.5% pct floor) then retraced fully."""
    d = decide_protection(side_long=True, avg_entry=100.0, mark=100.1, sl_price=95.0, peak_fe=2.0, cfg=_cfg())
    assert d["action"] == "none"


def test_giveback_armed_after_one_r_run():
    # Risk = 5 px; ran to 6 px (1.2R) then gave back >35% → close.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=102.0, sl_price=95.0, peak_fe=6.0, cfg=_cfg())
    assert d["action"] == "close"


def test_giveback_pct_arming_when_no_stop_known():
    # Without a live stop the 1R arming can't be computed → falls back to pct arming (old behavior).
    d = decide_protection(side_long=True, avg_entry=100.0, mark=100.5, sl_price=None, peak_fe=2.0, cfg=_cfg())
    assert d["action"] == "close"


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
        _cfg(breakeven_trigger_r=100.0, giveback_pct=0.0, early_cut_enabled=False),
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
