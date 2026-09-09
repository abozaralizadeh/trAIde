from src.regime import risk_capped_contracts
from src.tools import (
    entry_cancel_guard_reason,
    entry_contracts_at_least_one_lot,
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


class TestEntrySizingStopsAtTheExchangeFloor:
    """A size factor may shrink a bet to one lot; it may not shrink it out of existence."""

    def test_a_budget_below_one_lot_is_floored_rather_than_zeroed(self):
        assert entry_contracts_at_least_one_lot(0.9, 1) == (1, True)
        assert entry_contracts_at_least_one_lot(0.0, 1) == (1, True)
        assert entry_contracts_at_least_one_lot(1.9, 2) == (2, True)

    def test_a_budget_that_already_buys_a_lot_still_rounds_down(self):
        assert entry_contracts_at_least_one_lot(5.9, 2) == (4, False)
        assert entry_contracts_at_least_one_lot(1.9, 1) == (1, False)
        assert entry_contracts_at_least_one_lot(2.0, 2) == (2, False)

    def test_unreadable_input_still_yields_a_tradeable_lot(self):
        assert entry_contracts_at_least_one_lot("nonsense", 1) == (1, True)
        assert entry_contracts_at_least_one_lot(-5, 1) == (1, True)

    def test_the_hard_risk_cap_is_what_actually_decides(self):
        """The floored lot is handed to the risk cap, which is the guard that owns survival."""
        # PUMP-USDT, 2026-09-08: soft budget bought $2.83 of a $4.60 lot. One lot risks 2.83% of
        # $4.60 = $0.130 against a $0.504 budget, so it must survive the cap that matters.
        contracts, floored = entry_contracts_at_least_one_lot(0.61, 1)
        assert (contracts, floored) == (1, True)
        assert risk_capped_contracts(
            contracts, 1, 4.60, 1.0, 1.0 - 0.0283, 67.14, 0.0075,
        ) == 1

    def test_a_lot_that_genuinely_exceeds_the_risk_budget_is_still_refused(self):
        """Flooring to one lot must not become a way to smuggle oversized risk through."""
        contracts, floored = entry_contracts_at_least_one_lot(0.1, 1)
        assert (contracts, floored) == (1, True)
        # Same lot, but a stop 60% away puts $2.76 of risk against the same $0.504 budget.
        assert risk_capped_contracts(
            contracts, 1, 4.60, 1.0, 0.40, 67.14, 0.0075,
        ) == 0


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
