from src.tools import (
    entry_cancel_guard_reason,
    normalize_futures_side,
    round_entry_contracts_down,
    round_price_to_tick,
)


def test_futures_side_aliases_are_normalized_for_kucoin():
    assert normalize_futures_side("long") == "buy"
    assert normalize_futures_side("BUY") == "buy"
    assert normalize_futures_side("short") == "sell"
    assert normalize_futures_side("sell") == "sell"
    assert normalize_futures_side("hold") is None


def test_price_rounding_uses_contract_tick_without_float_noise():
    assert round_price_to_tick(3.5856, 0.001) == 3.586
    assert round_price_to_tick(100.13, 0.25) == 100.25


def test_entry_lot_rounding_never_erases_adaptive_size_cap():
    assert round_entry_contracts_down(5.9, 2) == 4
    assert round_entry_contracts_down(1.9, 1) == 1
    assert round_entry_contracts_down(0.9, 1) == 0


def test_tagged_entry_cannot_be_churned_before_lease_without_objective_flip():
    base_ts = 1_700_000_000
    order = {
        "clientOid": "traide-entry-abc", "side": "buy",
        "createdAt": base_ts * 1000,
    }
    reason = entry_cancel_guard_reason(
        order, {"intraday_bias_1h": "bullish", "daily_bias": "neutral"},
        lease_minutes=30, now=base_ts + 10 * 60,
    )
    assert "validity lease" in reason
    assert entry_cancel_guard_reason(
        order, {"intraday_bias_1h": "bearish", "daily_bias": "neutral"},
        lease_minutes=30, now=base_ts + 10 * 60,
    ) is None
    assert entry_cancel_guard_reason(
        order, {}, lease_minutes=30, now=base_ts + 31 * 60,
    ) is None


def test_cancel_guard_ignores_manual_orders():
    assert entry_cancel_guard_reason(
        {"clientOid": "manual", "createdAt": 1}, {}, lease_minutes=30, now=2,
    ) is None
