import time
import pytest
from src.memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(str(tmp_path / "memory.json"), retention_days=7)


def test_handoff_decisions_survive_decline_flood(store):
    """Handoffs live in their own retention bucket, so a flood of declines (far exceeding
    MAX_DECISIONS) cannot evict them — without the fix the two oldest entries (the handoffs)
    would be dropped by the 50-slot null-decision cap."""
    # Log the handoffs FIRST (oldest), then bury them under a decline flood.
    store.log_decision("ALL", "handoff_to_research", 0.0, "Trading Agent -> Research Agent")
    store.log_decision("ALL", "handoff_to_trading", 0.0, "Research Agent -> Trading Agent")
    for i in range(120):
        store.log_decision("BTC-USDT", "decline", 0.5, f"no setup {i}")
    stored_actions = [d.get("action") for d in store._read().get("decisions", [])]
    assert "handoff_to_research" in stored_actions
    assert "handoff_to_trading" in stored_actions
    # The high-volume declines are still capped (≈MAX_DECISIONS; +1 because log_decision prunes
    # then appends one), so memory stays bounded instead of growing to the 120 we logged.
    assert stored_actions.count("decline") <= 51


def test_set_and_get_coins(store):
    store.set_coins(["BTC-USDT", "ETH-USDT"], reason="test")
    coins = store.get_coins()
    assert "BTC-USDT" in coins
    assert "ETH-USDT" in coins


def test_add_coin(store):
    store.set_coins(["BTC-USDT"], reason="init")
    store.add_coin("ETH-USDT", reason="added")
    coins = store.get_coins()
    assert "ETH-USDT" in coins


def test_remove_coin(store):
    store.set_coins(["BTC-USDT", "ETH-USDT"], reason="init")
    store.remove_coin("ETH-USDT", reason="delisted", exit_plan="do not re-add")
    coins = store.get_coins()
    assert "ETH-USDT" not in coins
    assert "BTC-USDT" in coins


def test_has_coins(store):
    assert not store.has_coins()
    store.set_coins(["BTC-USDT"], reason="test")
    assert store.has_coins()


def test_record_and_count_trades(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=True, price=50000.0, size=0.002)
    assert store.trades_today("BTC-USDT") == 1
    store.record_trade("BTC-USDT", "sell", 100.0, paper=True, price=51000.0, size=0.002)
    assert store.trades_today("BTC-USDT") == 2
    assert store.trades_today("ETH-USDT") == 0


def test_update_limits_no_drawdown(store):
    limits = store.update_limits(1000.0, scope="total")
    assert limits["drawdownPct"] == 0.0
    assert "kill" not in limits


def test_update_limits_tracks_drawdown(store):
    store.update_limits(1000.0, scope="total")
    # Simulate a 10% loss — should track it, no kill switch
    limits = store.update_limits(900.0, scope="total")
    assert limits["drawdownPct"] >= 9.9
    assert "kill" not in limits


def test_reset_limits(store):
    store.update_limits(1000.0, scope="total")
    store.update_limits(900.0, scope="total")
    limits = store.reset_limits(900.0, scope="total")
    assert limits["drawdownPct"] == 0.0
    assert "kill" not in limits


def test_pruning_drops_old_entries(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"), retention_days=1)
    store.record_trade("BTC-USDT", "buy", 100.0, paper=True, price=50000.0, size=0.002)
    # Manually age the trade entry
    import json
    path = tmp_path / "memory.json"
    data = json.loads(path.read_text())
    data["trades"][0]["ts"] = int(time.time()) - 2 * 86400  # 2 days ago
    path.write_text(json.dumps(data))
    # Reset cache so next read goes to disk
    store._cache = None
    assert store.trades_today("BTC-USDT") == 0


def test_in_memory_cache_avoids_extra_disk_reads(tmp_path, monkeypatch):
    store = MemoryStore(str(tmp_path / "memory.json"), retention_days=7)
    store.set_coins(["BTC-USDT"], reason="init")

    write_count = 0
    original_write = store._write

    def counting_write(data):
        nonlocal write_count
        write_count += 1
        original_write(data)

    monkeypatch.setattr(store, "_write", counting_write)

    # Multiple reads should NOT trigger writes
    _ = store.get_coins()
    _ = store.get_coins()
    _ = store.get_coins()
    assert write_count == 0


def test_cross_instance_notes_survive_update_limits(tmp_path):
    """Supervisor writes notes via one MemoryStore; main loop's update_limits must not erase them."""
    path = str(tmp_path / "memory.json")
    main_loop = MemoryStore(path, retention_days=7)
    supervisor = MemoryStore(path, retention_days=7)

    # Main loop populates cache via update_limits
    main_loop.update_limits(1000.0, scope="total")

    # Supervisor writes a temporary note (different instance, same file)
    supervisor.add_temporary_note("reduce position sizes by 50%")

    # Main loop does another update_limits (this used to overwrite the note)
    main_loop.update_limits(990.0, scope="total")

    # A fresh reader (like run_trading_agent creates) must see the note
    agent = MemoryStore(path, retention_days=7)
    notes = agent.consume_temporary_notes()
    assert len(notes) == 1
    assert "reduce position sizes" in notes[0]["content"]


def test_same_path_instances_share_process_lock(tmp_path):
    path = str(tmp_path / "memory.json")
    assert MemoryStore(path)._lock is MemoryStore(path)._lock


def test_cross_instance_permanent_notes_survive(tmp_path):
    """Permanent notes written by supervisor must survive main loop writes."""
    path = str(tmp_path / "memory.json")
    main_loop = MemoryStore(path, retention_days=7)
    supervisor = MemoryStore(path, retention_days=7)

    main_loop.set_coins(["BTC-USDT"], reason="init")
    supervisor.add_permanent_note("always check BTC dominance")
    main_loop.update_limits(500.0, scope="total")

    agent = MemoryStore(path, retention_days=7)
    notes = agent.get_permanent_notes()
    assert len(notes) == 1
    assert "BTC dominance" in notes[0]["content"]


def test_permanent_notes_exempt_from_retention_prune(tmp_path):
    """Permanent notes must persist past retention_days — they are designed to live forever."""
    import json
    path = tmp_path / "memory.json"
    now = int(time.time())
    very_old_ts = now - 365 * 86400  # 1 year ago, far beyond any retention window
    payload = {
      "plans": [], "triggers": [], "coins": [], "trades": [], "limits": {},
      "sentiments": [], "decisions": [], "fees": [],
      "supervisor_notes_temporary": [
        {"content": "old temporary note", "ts": very_old_ts, "author": "Supervisor"},
      ],
      "supervisor_notes_permanent": [
        {"content": "always check BTC dominance", "ts": very_old_ts, "author": "Supervisor"},
      ],
    }
    path.write_text(json.dumps(payload))
    store = MemoryStore(str(path), retention_days=7)
    perm = store.get_permanent_notes()
    assert len(perm) == 1, "permanent note older than retention_days was incorrectly pruned"
    assert "BTC dominance" in perm[0]["content"]
    # Temporary note from a year ago should be pruned — sanity check the asymmetry
    notes = store.list_all_notes()
    assert notes["temporary"] == [], "old temporary note should have been pruned"


def test_performance_summary_empty(store):
    summary = store.performance_summary()
    assert summary["totalTrades"] == 0


def test_performance_summary_with_decisions(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=True, price=50000.0, size=0.002)
    store.record_trade("BTC-USDT", "sell", 100.0, paper=True, price=51000.0, size=0.002)
    store.record_trade("ETH-USDT", "buy", 50.0, paper=True, price=3000.0, size=0.016)
    store.record_trade("ETH-USDT", "sell", 50.0, paper=True, price=2900.0, size=0.016)
    # Log decisions with PnL
    store.log_decision("BTC-USDT", "spot_sell", 0.7, "take profit", pnl=2.0)
    store.log_decision("ETH-USDT", "spot_sell", 0.6, "stop loss", pnl=-1.5)
    summary = store.performance_summary()
    assert summary["totalTrades"] == 4
    assert summary["closedWithPnl"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["winRate"] == 0.5
    assert summary["totalRealizedPnl"] == 0.5
    assert summary["avgWin"] == 2.0
    assert summary["avgLoss"] == -1.5
    # Venue breakdown should exist
    assert "spot" in summary
    assert summary["spot"]["totalTrades"] == 4
    assert summary["spot"]["closedWithPnl"] == 2
    assert "futures" in summary
    assert summary["futures"]["totalTrades"] == 0


def test_hold_close_only_is_not_miscounted_as_realized(store):
    store.record_trade("XRP-USDT", "sell", 25.0, paper=False, price=1.0, size=25, venue="futures")
    store.log_decision("XRP-USDT", "hold-close-only", 0.99, "circuit breaker hold", pnl=0.0)
    store.log_decision("XRP-USDT", "futures_buy_triggered", 0.0, "real close", pnl=-0.25)
    summary = store.performance_summary()
    assert summary["closedWithPnl"] == 1
    assert summary["losses"] == 1


def test_close_metadata_survives_restart_for_no_chase(tmp_path):
    path = str(tmp_path / "memory.json")
    mem = MemoryStore(path, retention_days=7)
    mem.log_decision("ETH-USDT", "futures_sell_triggered", 0.0, "tp", pnl=1.0,
                     exit_price=2500.0, close_type="CLOSE_LONG")
    fresh = MemoryStore(path, retention_days=7)
    close = fresh.realized_closes()[0]
    assert close["exitPrice"] == 2500.0
    assert close["closeType"] == "CLOSE_LONG"


def test_position_lifecycle_metadata_survives_restart(tmp_path):
    path = str(tmp_path / "memory.json")
    mem = MemoryStore(path, retention_days=7)
    mem.log_decision(
      "ETH-USDT", "futures_sell_triggered", 0.0, "tp", pnl=1.0,
      position_id="position-123", position_open_time=1_700_000_000_000,
      position_side="long",
    )
    row = MemoryStore(path, retention_days=7).realized_closes()[0]
    assert row["positionId"] == "position-123"
    assert row["positionOpenTime"] == 1_700_000_000_000
    assert row["positionSide"] == "long"
    assert row["positionLifecycleVersion"] == 1


def test_exchange_close_supersedes_recent_local_pnl_estimate(store):
    store.record_trade("ZEC-USDT", "buy", 50.0, paper=False, price=500, size=0.1, venue="futures")
    store.log_decision("ZEC-USDT", "futures_sell", 0.9, "estimated close", pnl=0.56)
    store.log_decision("ZEC-USDT", "futures_sell_triggered", 0.0, "exchange cumulative close", pnl=1.44)
    summary = store.performance_summary()
    assert summary["closedWithPnl"] == 1
    assert summary["totalRealizedPnl"] == 1.44


def test_exchange_close_supersedes_only_the_same_position_lifecycle():
    decisions = [
      {"symbol": "ZEC-USDT", "action": "futures_sell", "pnl": 0.56, "ts": 100,
       "positionId": "old-position", "positionOpenTime": 1_700_000_000_000, "positionSide": "long"},
      {"symbol": "ZEC-USDT", "action": "futures_sell_triggered", "pnl": 1.44, "ts": 120,
       "positionId": "old-position", "positionOpenTime": 1_700_000_000_000, "positionSide": "long"},
    ]
    rows = MemoryStore._authoritative_realized_rows(decisions)
    assert [(row["action"], row["pnl"]) for row in rows] == [("futures_sell_triggered", 1.44)]


def test_new_same_symbol_lifecycle_is_not_dropped_inside_close_window():
    decisions = [
      {"symbol": "ZEC-USDT", "action": "futures_sell", "pnl": 0.56, "ts": 100,
       "positionId": "old-position", "positionOpenTime": 1_700_000_000_000, "positionSide": "long"},
      {"symbol": "ZEC-USDT", "action": "futures_sell_triggered", "pnl": 0.75, "ts": 500,
       "positionId": "new-position", "positionOpenTime": 1_700_001_000_000, "positionSide": "long"},
    ]
    rows = MemoryStore._authoritative_realized_rows(decisions)
    assert [(row["action"], row["pnl"]) for row in rows] == [
      ("futures_sell", 0.56),
      ("futures_sell_triggered", 0.75),
    ]


def test_new_lifecycle_row_without_exchange_ids_fails_safe():
    decisions = [
      {"symbol": "ZEC-USDT", "action": "futures_sell", "pnl": 0.56, "ts": 100,
       "positionSide": "long", "positionLifecycleVersion": 1},
      {"symbol": "ZEC-USDT", "action": "futures_sell_triggered", "pnl": 0.75, "ts": 120,
       "positionId": "new-position", "positionSide": "long", "positionLifecycleVersion": 1},
    ]
    assert MemoryStore._authoritative_realized_rows(decisions) == decisions


def test_two_equal_pnl_closes_with_distinct_position_ids_are_both_kept():
    closes = [
      {"symbol": "ETH-USDT", "action": "futures_sell_triggered", "closeType": "CLOSE_LONG",
       "pnl": 1.0, "ts": 100, "positionId": "position-a"},
      {"symbol": "ETH-USDT", "action": "futures_sell_triggered", "closeType": "CLOSE_LONG",
       "pnl": 1.0, "ts": 200, "positionId": "position-b"},
    ]
    assert MemoryStore._dedupe_realized(closes) == closes


def test_record_trade_venue_futures(store):
    entry = store.record_trade("BTC-USDT", "buy", 500.0, paper=False, price=100000.0, size=0.005, venue="futures")
    assert entry["venue"] == "futures"


def test_record_trade_venue_defaults_to_spot(store):
    entry = store.record_trade("BTC-USDT", "buy", 100.0, paper=True, price=50000.0, size=0.002)
    assert entry["venue"] == "spot"


def test_unfilled_submission_does_not_consume_trade_cap_or_cooldown(store, monkeypatch):
    monkeypatch.setattr("src.memory.time.time", lambda: 1_800_000_000)
    pending = store.record_trade(
        "ZEC-USDT", "buy", 25.0, venue="futures", filled=False,
        order_id="pending-1", client_oid="traide-entry-pending",
    )
    assert store.trades_today("ZEC-USDT") == 0
    assert store.last_trade_time("ZEC-USDT") is None
    store.mark_order_filled(
        pending["orderId"], pending["clientOid"],
        fill_ts=1_800_000_060_000, fill_price=100.5, fill_size=2,
    )
    assert store.trades_today("ZEC-USDT") == 1
    assert store.last_trade_time("ZEC-USDT") == 1_800_000_060


def test_entry_context_survives_restart_and_attributes_realized_r(tmp_path, monkeypatch):
    now = 1_800_000_000
    monkeypatch.setattr("src.memory.time.time", lambda: now)
    path = str(tmp_path / "memory.json")
    mem = MemoryStore(path, retention_days=90)
    context = {
        "policyVersion": "test-v1", "positionSide": "long",
        "plannedMaxLossUsd": 2.0, "plannedNetRr": 1.7,
    }
    mem.record_trade(
        "ZEC-USDT", "buy", 50.0, venue="futures", filled=False,
        order_id="entry-order", client_oid="traide-entry-ctx", entry_context=context,
    )
    mem.mark_order_filled("entry-order", "traide-entry-ctx", fill_ts=(now + 60) * 1000, fill_price=100)
    reloaded = MemoryStore(path, retention_days=90)
    matched = reloaded.entry_context_for_position("ZEC-USDT", (now + 60) * 1000, "long")
    assert matched["policyVersion"] == "test-v1"
    assert matched["fillPrice"] == 100.0
    close = reloaded.log_decision(
        "ZEC-USDT", "futures_sell_triggered", 0.0, "close", pnl=1.0,
        position_open_time=(now + 60) * 1000, position_side="long", entry_price=100,
        entry_context=matched,
    )
    assert close["realizedR"] == 0.5
    persisted = MemoryStore(path, retention_days=90).realized_closes()[-1]
    assert persisted["entryPrice"] == 100.0
    assert persisted["entryContext"]["policyVersion"] == "test-v1"
    assert persisted["realizedR"] == 0.5


def test_intraday_triggers_expire_autonomously(store, monkeypatch):
    now = 1_800_000_000
    monkeypatch.setattr("src.memory.time.time", lambda: now)
    trigger = store.save_trigger(
        "SOL-USDT", "buy", "breakout", target_price=100,
        condition="above", expires_minutes=60,
    )
    assert trigger["expiresAt"] == now + 3600
    assert trigger["triggerId"]
    assert len(store.latest_triggers()) == 1
    assert store.consume_trigger(trigger) is True
    assert store.latest_triggers() == []
    store.save_trigger(
        "SOL-USDT", "buy", "breakout", target_price=100,
        condition="above", expires_minutes=60,
    )
    monkeypatch.setattr("src.memory.time.time", lambda: now + 3601)
    assert store.latest_triggers() == []


def test_old_records_without_venue_default_to_spot(tmp_path):
    import json
    path = tmp_path / "memory.json"
    old_trade = {"symbol": "BTC-USDT", "side": "buy", "notionalUsd": 100.0, "price": 50000.0, "size": 0.002, "paper": False, "ts": int(time.time()), "day": int(time.time() // 86400)}
    path.write_text(json.dumps({"trades": [old_trade], "decisions": [], "plans": [], "triggers": [], "coins": [], "limits": {}, "sentiments": [], "fees": [], "supervisor_notes_temporary": [], "supervisor_notes_permanent": []}))
    store = MemoryStore(str(path), retention_days=7)
    summary = store.performance_summary()
    assert summary["spot"]["totalTrades"] == 1
    assert summary["futures"]["totalTrades"] == 0


def test_performance_summary_splits_spot_futures(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=False, price=50000.0, size=0.002)
    store.record_trade("BTC-USDT", "sell", 100.0, paper=False, price=51000.0, size=0.002)
    store.record_trade("ETH-USDT", "buy", 200.0, paper=False, price=3000.0, size=0.066, venue="futures")
    store.record_trade("ETH-USDT", "sell", 200.0, paper=False, price=3100.0, size=0.066, venue="futures")
    store.log_decision("BTC-USDT", "spot_sell", 0.8, "take profit", pnl=2.0, paper=False)
    store.log_decision("ETH-USDT", "futures_sell", 0.7, "close long", pnl=5.0, paper=False)
    summary = store.performance_summary()
    assert summary["totalTrades"] == 4
    assert summary["spot"]["totalTrades"] == 2
    assert summary["spot"]["closedWithPnl"] == 1
    assert summary["spot"]["totalRealizedPnl"] == 2.0
    assert summary["futures"]["totalTrades"] == 2
    assert summary["futures"]["closedWithPnl"] == 1
    assert summary["futures"]["totalRealizedPnl"] == 5.0


def test_performance_summary_splits_paper_live(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=True, price=50000.0, size=0.002)
    store.record_trade("BTC-USDT", "sell", 100.0, paper=True, price=51000.0, size=0.002)
    store.record_trade("BTC-USDT", "buy", 100.0, paper=False, price=50000.0, size=0.002)
    store.record_trade("BTC-USDT", "sell", 100.0, paper=False, price=52000.0, size=0.002)
    store.log_decision("BTC-USDT", "spot_sell", 0.7, "paper tp", pnl=1.0, paper=True)
    store.log_decision("BTC-USDT", "spot_sell", 0.8, "live tp", pnl=3.0, paper=False)
    summary = store.performance_summary()
    assert summary["spot"]["paper"]["closedWithPnl"] == 1
    assert summary["spot"]["paper"]["totalRealizedPnl"] == 1.0
    assert summary["spot"]["live"]["closedWithPnl"] == 1
    assert summary["spot"]["live"]["totalRealizedPnl"] == 3.0


def test_positions_venue_filter(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=False, price=50000.0, size=0.002)
    store.record_trade("BTC-USDT", "buy", 500.0, paper=False, price=100000.0, size=0.005, venue="futures")
    all_pos = store.positions()
    assert all_pos["BTC-USDT"]["netSize"] == pytest.approx(0.007)
    spot_pos = store.positions(venue="spot")
    assert spot_pos["BTC-USDT"]["netSize"] == pytest.approx(0.002)
    futures_pos = store.positions(venue="futures")
    assert futures_pos["BTC-USDT"]["netSize"] == pytest.approx(0.005)


# --- Position extremes (peak/trough PnL) tests ---


def test_update_position_extremes_tracks_peak_and_trough(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=False, price=50000.0, size=0.002)
    # Simulate rising price
    pos1 = store.positions(prices={"BTC-USDT": 51000.0})
    store.update_position_extremes(pos1)
    ext = store.get_position_extremes("BTC-USDT")
    assert ext["peakPnl"] == pytest.approx(2.0)
    assert ext["troughPnl"] == pytest.approx(2.0)
    # Simulate price drop
    pos2 = store.positions(prices={"BTC-USDT": 49000.0})
    store.update_position_extremes(pos2)
    ext = store.get_position_extremes("BTC-USDT")
    assert ext["peakPnl"] == pytest.approx(2.0)  # peak unchanged
    assert ext["troughPnl"] == pytest.approx(-2.0)  # new trough
    # Simulate new high
    pos3 = store.positions(prices={"BTC-USDT": 53000.0})
    store.update_position_extremes(pos3)
    ext = store.get_position_extremes("BTC-USDT")
    assert ext["peakPnl"] == pytest.approx(6.0)  # new peak
    assert ext["troughPnl"] == pytest.approx(-2.0)  # trough unchanged


def test_position_extremes_reset_when_exchange_lifecycle_changes(store):
    store.update_position_extremes({
      "ETH-USDT": {"netSize": 1, "unrealizedPnl": 5.0, "positionOpenTime": 1000, "positionSide": "long"},
    })
    store.update_position_extremes({
      "ETH-USDT": {"netSize": 1, "unrealizedPnl": -1.0, "positionOpenTime": 2000, "positionSide": "long"},
    })
    ext = store.get_position_extremes("ETH-USDT")
    assert ext["peakPnl"] == -1.0 and ext["troughPnl"] == -1.0
    assert ext["positionOpenTime"] == 2000


def _fe_row(qty, entry, mark, *, mult=10.0, opened=1790252000000):
    """One row as main._live_extremes_map builds it from a KuCoin position."""
    return {"netSize": qty, "unrealizedPnl": (mark - entry) * qty * mult,
            "positionOpenTime": opened, "positionSide": "long" if qty > 0 else "short",
            "lifecycleOpenTime": opened, "markPrice": mark, "avgEntryPrice": entry}


class TestPriceSpacePeakForTheTrail:
    """peakFePx: the restart peak the trail uses, in PRICE units, keyed on ProtectionManager's own
    (openTime, side, |qty|, avgEntry) identity. peakPnl/lifecycleKey keep their meaning for closes/MFE."""

    def test_price_peak_ignores_the_contract_multiplier(self, store):
        store.update_position_extremes({"H-USDT": _fe_row(-11, 0.06008, 0.05950)})
        store.update_position_extremes({"H-USDT": _fe_row(-11, 0.06008, 0.05905)})
        store.update_position_extremes({"H-USDT": _fe_row(-11, 0.06008, 0.05980)})
        ext = store.get_position_extremes("H-USDT")
        assert ext["peakFePx"] == pytest.approx(0.00103)            # short: entry - mark
        assert ext["peakPnl"] / 11 == pytest.approx(0.0103)        # USD/contracts = 10x the price

    def test_add_on_resets_the_price_peak_but_not_the_lifecycle_usd_peak(self, store):
        store.update_position_extremes({"H-USDT": _fe_row(1, 1.0, 1.018)})
        store.update_position_extremes({"H-USDT": _fe_row(2, 1.009, 1.015)})
        ext = store.get_position_extremes("H-USDT")
        assert ext["peakFePx"] == pytest.approx(0.006)             # new baseline vs the new avgEntry
        assert ext["peakPnl"] == pytest.approx(0.18)               # lifecycle MFE unchanged: max(0.18, 0.12)
        assert ext["lifecycleKey"] == "1790252000000:long"

    def test_partial_reduction_resets_the_price_peak(self, store):
        store.update_position_extremes({"H-USDT": _fe_row(2, 1.0, 1.018)})
        store.update_position_extremes({"H-USDT": _fe_row(1, 1.0, 1.010)})
        assert store.get_position_extremes("H-USDT")["peakFePx"] == pytest.approx(0.010)

    def test_rows_without_mark_or_entry_keep_only_the_usd_fields(self, store):
        store.update_position_extremes({"ETH-USDT": {"netSize": 1.0, "unrealizedPnl": 5.0}})
        ext = store.get_position_extremes("ETH-USDT")
        assert ext["peakPnl"] == 5.0 and "peakFePx" not in ext and "peakFeKey" not in ext

    def test_peak_fe_key_side_comes_from_the_sign_of_qty(self):
        from src.memory import peak_fe_key
        assert peak_fe_key(1790252000000, 3, 1.0) == "1790252000000|long|3.0|1.0"
        assert peak_fe_key("1790252000000", "-3", "1.0") == "1790252000000|short|3.0|1.0"
        assert peak_fe_key(None, 3, 1.0) == "None|long|3.0|1.0"
        for bad in ((1, 0, 1.0), (1, 3, 0), (1, "x", 1.0), (1, 3, float("nan"))):
            assert peak_fe_key(*bad) is None


def test_extremes_cleared_when_position_closes(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=False, price=50000.0, size=0.002)
    pos = store.positions(prices={"BTC-USDT": 51000.0})
    store.update_position_extremes(pos)
    assert store.get_position_extremes("BTC-USDT")
    # Close the position
    store.record_trade("BTC-USDT", "sell", 100.0, paper=False, price=51000.0, size=0.002)
    pos_empty = store.positions(prices={"BTC-USDT": 51000.0})
    store.update_position_extremes(pos_empty)
    assert store.get_position_extremes("BTC-USDT") == {}


def test_positions_include_peak_trough(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=False, price=50000.0, size=0.002)
    pos = store.positions(prices={"BTC-USDT": 52000.0})
    store.update_position_extremes(pos)
    pos = store.positions(prices={"BTC-USDT": 49000.0})
    store.update_position_extremes(pos)
    pos = store.positions(prices={"BTC-USDT": 50500.0})
    assert pos["BTC-USDT"]["peakPnl"] == pytest.approx(4.0)
    assert pos["BTC-USDT"]["troughPnl"] == pytest.approx(-2.0)


def test_log_decision_auto_attaches_extremes(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=False, price=50000.0, size=0.002)
    pos = store.positions(prices={"BTC-USDT": 53000.0})
    store.update_position_extremes(pos)
    pos = store.positions(prices={"BTC-USDT": 48000.0})
    store.update_position_extremes(pos)
    # Log a sell decision — should auto-attach peak/trough
    decision = store.log_decision("BTC-USDT", "spot_sell", 0.8, "take profit", pnl=1.0)
    assert decision["peakPnl"] == pytest.approx(6.0)
    assert decision["troughPnl"] == pytest.approx(-4.0)


def test_performance_summary_missed_profit(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=True, price=50000.0, size=0.002)
    store.record_trade("BTC-USDT", "sell", 100.0, paper=True, price=50500.0, size=0.002)
    # Log a decision where peak was much higher than final PnL
    store.log_decision("BTC-USDT", "spot_sell", 0.7, "take profit", pnl=1.0, peak_pnl=5.0, trough_pnl=-0.5)
    summary = store.performance_summary()
    assert summary["missedProfitCount"] == 1
    assert summary["totalMissedProfit"] == pytest.approx(4.0)  # peak 5.0 - actual 1.0
    assert summary["avgMissedProfit"] == pytest.approx(4.0)


def test_performance_summary_no_missed_profit_when_peak_equals_pnl(store):
    store.record_trade("BTC-USDT", "buy", 100.0, paper=True, price=50000.0, size=0.002)
    store.record_trade("BTC-USDT", "sell", 100.0, paper=True, price=51000.0, size=0.002)
    store.log_decision("BTC-USDT", "spot_sell", 0.8, "perfect exit", pnl=2.0, peak_pnl=2.0, trough_pnl=-0.1)
    summary = store.performance_summary()
    assert "missedProfitCount" not in summary


def test_agent_event_inbox_persists_until_acknowledged(tmp_path):
    path = str(tmp_path / "memory.json")
    first = MemoryStore(path, retention_days=7)
    assert first.queue_agent_event("futures_fills", "futures:fill-1", {"id": "fill-1"}) is True
    assert first.queue_agent_event("futures_fills", "futures:fill-1", {"id": "fill-1"}) is False
    assert first.queue_agent_event(
        "auto_triggers", "auto:trigger-1", {"observedPrice": 101.0},
    ) is True

    restarted = MemoryStore(path, retention_days=7)
    assert [event["id"] for event in restarted.get_pending_agent_events()] == [
        "futures:fill-1", "auto:trigger-1",
    ]
    assert len(restarted.acknowledge_agent_events(["futures:fill-1", "auto:trigger-1"])) == 2
    assert restarted.get_pending_agent_events() == []


def test_entry_expired_event_queues_and_survives_restart(tmp_path):
    # entry_expired must pass BOTH whitelists (queue_agent_event source + the read-time sanitizer) so
    # the agent can see its own unfilled limits die and stop re-placing a never-filling pullback limit.
    path = str(tmp_path / "memory.json")
    first = MemoryStore(path, retention_days=7)
    assert first.queue_agent_event(
        "entry_expired", "ONDOUSDTM:469075", {"symbol": "ONDO-USDT", "side": "buy", "price": 0.3968},
    ) is True
    restarted = MemoryStore(path, retention_days=7)
    events = restarted.get_pending_agent_events()
    expiries = [e for e in events if e.get("kind") == "entry_expired"]
    assert len(expiries) == 1 and expiries[0]["payload"]["symbol"] == "ONDO-USDT"


def test_agent_scheduler_persists_restart_cadence_and_price_noise(tmp_path):
    path = str(tmp_path / "memory.json")
    first = MemoryStore(path, retention_days=7)
    first.save_agent_scheduler({
        "lastRunTs": 1234.5,
        "unproductiveRuns": 4,
        "reviewedPrices": {"btcusdt": 50_000, "bad": -1},
        "priceObservations": {
            "btcusdt": {
                "lastPrice": 50_100,
                "noiseEwmaPct": 0.2,
                "samples": 12,
                "updated": 1234,
            },
            "invalid": {"lastPrice": 0},
        },
    })

    restarted = MemoryStore(path, retention_days=7)
    state = restarted.get_agent_scheduler()
    assert state["lastRunTs"] == pytest.approx(1234.5)
    assert state["unproductiveRuns"] == 4
    assert state["reviewedPrices"] == {"BTC-USDT": 50_000.0}
    assert state["priceObservations"]["BTC-USDT"] == {
        "lastPrice": 50_100.0,
        "noiseEwmaPct": 0.2,
        "samples": 12,
        "updated": 1234,
    }
    assert "INVALID" not in state["priceObservations"]


def test_agent_scheduler_persists_taker_flow_across_a_restart(tmp_path):
    """The sanitizer is the ONLY writer of the scheduler shape, so a field it does not whitelist is
    silently dropped on the next save — the flow level would then reset on every restart and the
    EWMA would never build a history."""
    path = str(tmp_path / "memory.json")
    MemoryStore(path, retention_days=7).save_agent_scheduler({
        "flowObservations": {
            "btcusdt": {
                "buyShare": 0.6123, "buyTradeShare": 0.55, "newBuyShare": 0.7,
                "trades": 100, "newTrades": 8, "spanSec": 182.4, "gapped": False,
                "buyShareEwma": 0.58, "samples": 12, "updated": 1234, "lastCursor": 998,
            },
            # A reading with no usable share carries no information — the husk must not be kept.
            "ethusdt": {"trades": 100, "spanSec": 60.0},
            "junk": "not-a-dict",
        },
    })

    state = MemoryStore(path, retention_days=7).get_agent_scheduler()
    assert state["flowObservations"]["BTC-USDT"] == {
        "buyShare": 0.6123, "buyTradeShare": 0.55, "newBuyShare": 0.7,
        "trades": 100, "newTrades": 8, "spanSec": 182.4, "gapped": False,
        "buyShareEwma": 0.58, "samples": 12, "updated": 1234, "lastCursor": 998,
    }
    assert list(state["flowObservations"]) == ["BTC-USDT"]


def test_out_of_range_flow_shares_are_dropped_rather_than_clamped(tmp_path):
    """A clamped 1.4 would persist as a perfectly plausible 1.0 and bias every statistic built on
    it; a missing field is visible as missing evidence."""
    path = str(tmp_path / "memory.json")
    MemoryStore(path, retention_days=7).save_agent_scheduler({
        "flowObservations": {"BTC-USDT": {"buyShare": 0.6, "buyTradeShare": 1.4, "newBuyShare": -0.2}},
    })
    row = MemoryStore(path, retention_days=7).get_agent_scheduler()["flowObservations"]["BTC-USDT"]
    assert row["buyShare"] == 0.6
    assert "buyTradeShare" not in row and "newBuyShare" not in row


def test_a_direction_call_carries_the_tape_reading_that_was_current_when_it_was_made(tmp_path):
    """Without the stamp there is nothing to score flow against later — the reading has to be
    captured at the call, since by settle time the tape is hours gone."""
    store = MemoryStore(str(tmp_path / "memory.json"), retention_days=7)
    store.record_signal_probe(
        "XRP-USDT", "buy", 100.0, "continuation",
        taker_flow={"buyShare": 0.72, "trades": 100, "spanSec": 180.0, "gapped": False,
                    "ageSec": 45, "secret": "dropped"},
    )
    ctx = store.signal_probes(limit=10)[0]["entryContext"]
    assert ctx["takerFlow"] == {"buyShare": 0.72, "trades": 100, "newTrades": 0,
                                "spanSec": 180.0, "ageSec": 45.0, "gapped": False}

    # A call made with no reading available records the absence, rather than a neutral-looking stub.
    store.record_signal_probe("ADA-USDT", "sell", 100.0, "continuation", taker_flow=None)
    unstamped = next(p for p in store.signal_probes(limit=10) if p["symbol"] == "ADA-USDT")
    assert unstamped["entryContext"]["takerFlow"] is None


def test_automatic_quarantine_has_adaptive_expiring_retry_window(store, monkeypatch):
    now = 2_000_000_000
    monkeypatch.setattr("src.memory.time.time", lambda: now)
    store.remove_coin(
        "BANK-USDT",
        reason="Automatic risk quarantine: daily ATR 12.00% exceeds 9.00% hard limit",
        exit_plan="retry later",
    )
    store.remove_coin(
        "LAB-USDT",
        reason="Automatic risk quarantine: daily ATR 1100.00% exceeds 9.00% hard limit",
        exit_plan="retry later",
    )
    store.remove_coin("OLD-USDT", reason="stale", exit_plan="not a quarantine")

    quarantined = {item["symbol"]: item for item in store.get_quarantined_coins(now=now)}
    assert 20 <= quarantined["BANK-USDT"]["remainingHours"] <= 22
    assert quarantined["LAB-USDT"]["remainingHours"] == pytest.approx(168.0)
    assert "OLD-USDT" not in quarantined
    assert store.get_quarantined_coins(now=now + 8 * 86400) == []


def test_pending_limit_record_does_not_create_phantom_position(tmp_path):
    mem = MemoryStore(str(tmp_path / "memory.json"), retention_days=7)
    mem.record_trade(
        "ETH-USDT", "buy", 20.0, price=2000.0, size=0.01,
        venue="futures", filled=False, track_position=False, order_id="order-1",
        client_oid="traide-entry-limit-1",
    )
    # A submitted-but-unfilled limit must not consume the filled-trade cap/cooldown.
    assert mem.trades_today("ETH-USDT") == 0
    assert mem.positions(venue="futures") == {}
    summary = mem.performance_summary()
    assert summary["totalTrades"] == 0
    assert summary["orderSubmissions"] == 1
    assert summary["limitOrdersSubmitted"] == 1
    assert summary["limitOrdersFilled"] == 0
    assert summary["limitFillRate"] == 0.0
    assert mem.mark_order_filled("order-1") is True
    assert mem.trades_today("ETH-USDT") == 1
    filled_summary = mem.performance_summary()
    assert filled_summary["totalTrades"] == 1
    assert filled_summary["limitOrdersFilled"] == 1
    assert filled_summary["limitFillRate"] == 1.0
    assert mem.positions(venue="futures") == {}


def test_limit_fill_moves_trade_accounting_to_execution_day(tmp_path, monkeypatch):
    mem = MemoryStore(str(tmp_path / "memory.json"), retention_days=7)
    before_midnight = 100 * 86400 - 30
    after_midnight = 100 * 86400 + 30
    monkeypatch.setattr("src.memory.time.time", lambda: before_midnight)
    mem.record_trade(
        "ETH-USDT", "buy", 20.0, price=2000.0, size=0.01,
        venue="futures", filled=False, track_position=False, order_id="rollover-order",
        client_oid="traide-entry-rollover",
    )
    assert mem.mark_order_filled("rollover-order", fill_ts=after_midnight)
    monkeypatch.setattr("src.memory.time.time", lambda: after_midnight)
    assert mem.trades_today("ETH-USDT") == 1


def test_market_reduce_only_close_does_not_affect_limit_fill_stats(tmp_path):
    mem = MemoryStore(str(tmp_path / "memory.json"), retention_days=7)
    mem.record_trade(
        "ETH-USDT", "buy", 20.0, price=2000.0, size=0.01,
        venue="futures", filled=False, track_position=False,
        order_id="limit-order-1", client_oid="traide-entry-limit-1",
    )
    before = mem.performance_summary()
    assert before["limitOrdersSubmitted"] == 1
    assert before["limitOrdersFilled"] == 0
    assert before["limitFillRate"] == 0.0

    # Market/reduce-only closes receive exchange order IDs, but never the traide-entry tag.
    mem.record_trade(
        "ETH-USDT", "sell", 20.0, price=1990.0, size=0.01,
        venue="futures", filled=True, track_position=False,
        order_id="market-close-1", client_oid="ethusdtm-close-deadbeef",
    )
    after = mem.performance_summary()
    assert after["limitOrdersSubmitted"] == before["limitOrdersSubmitted"]
    assert after["limitOrdersFilled"] == before["limitOrdersFilled"]
    assert after["limitFillRate"] == before["limitFillRate"]


def test_realized_closes_survive_the_retention_cutoff(tmp_path):
    """Learning data must age out by being SUPERSEDED, never by the clock.

    Measured live 2026-08-06: as the trade rate fell, the 7-day window emptied until only 8 realized
    closes remained, all recent losses. The edge controller then reported an 11% win rate and a 6-loss
    streak, halved position size, and the agent stood aside in 356 of 358 runs — producing no new
    closes, so the window could only get staler and bleaker. A quiet spell must not be self-reinforcing.
    """
    store = MemoryStore(str(tmp_path / "memory.json"), retention_days=1)
    store.log_decision("BTC-USDT", "futures_sell_triggered", 0.9, "tp", pnl=1.25)
    store.log_decision("ETH-USDT", "decline", 0.4, "no setup")

    import json
    path = tmp_path / "memory.json"
    data = json.loads(path.read_text())
    old = int(time.time()) - 30 * 86400          # a month old: far past the 1-day cutoff
    for d in data["decisions"]:
        d["ts"] = old
    path.write_text(json.dumps(data))
    store._cache = None

    kept = store.realized_closes(limit=100)
    assert len(kept) == 1 and kept[0]["pnl"] == 1.25   # the closed trade survives
    # ...while the ephemeral decline (pnl=None) is still pruned by age.
    store._cache = None
    all_decisions = store._read().get("decisions", [])
    assert all(d.get("pnl") is not None for d in all_decisions)


def test_filled_orders_survive_retention_so_slippage_stays_calibrated(tmp_path):
    # measured_slippage_pct needs (planned price, achieved fill price) pairs; time-pruning them would
    # silently drop the estimator back to its stale prior during a quiet spell.
    store = MemoryStore(str(tmp_path / "memory.json"), retention_days=1)
    store.record_trade("BTC-USDT", "buy", 100.0, paper=False, price=50000.0, size=0.002, filled=True)
    store.record_trade("ETH-USDT", "buy", 100.0, paper=False, price=3000.0, size=0.03, filled=False)

    import json
    path = tmp_path / "memory.json"
    data = json.loads(path.read_text())
    for t in data["trades"]:
        t["ts"] = int(time.time()) - 30 * 86400
        if t["symbol"] == "BTC-USDT":
            t["fillPrice"] = 50005.0          # only a real fill is a usable slippage sample
    path.write_text(json.dumps(data))
    store._cache = None

    fills = store.recent_fills(limit=100)
    assert len(fills) == 1 and fills[0]["fillPrice"] == 50005.0


def test_signal_probes_survive_retention_even_when_unfilled(tmp_path):
    """Unfilled plans carry the UNBIASED half of the signal sample and must not age out.

    A resting limit fills preferentially when the move goes against it, so filled orders are
    adverse-selected. Six of the first nine live probes were on unfilled plans; pruning those by the
    clock would quietly bias signal edge toward exactly the contaminated subset the measurement exists
    to avoid. Rows without a price stamp stay ephemeral.
    """
    store = MemoryStore(str(tmp_path / "memory.json"), retention_days=1)
    store.record_trade("BTC-USDT", "buy", 100.0, paper=False, price=50000.0, size=0.002, filled=False,
                       entry_context={"positionSide": "long", "marketPriceAtSignal": 50010.0})
    store.record_trade("ETH-USDT", "buy", 100.0, paper=False, price=3000.0, size=0.03, filled=False)

    import json
    path = tmp_path / "memory.json"
    data = json.loads(path.read_text())
    for t in data["trades"]:
        t["ts"] = int(time.time()) - 30 * 86400
    path.write_text(json.dumps(data))
    store._cache = None

    probes = store.signal_probes(limit=50)
    assert len(probes) == 1 and probes[0]["symbol"] == "BTC-USDT"
    # the unstamped row carries no learning value, so it is still pruned
    assert len(store._read().get("trades", [])) == 1


def test_post_close_review_rows_are_not_counted_as_trades(tmp_path):
    """A review is commentary written AFTER a close and copies its pnl — booking it double-counts.

    Second occurrence of this bug class (the first was "hold-close-only"). On 2026-08-08
    `close_reviewed` / `close_reviewed_hold` slipped through `startswith("close_")`: the dashboard
    showed every closed position twice (once real, once an empty shell), win rate read 36.7% instead
    of 40.0%, and the loss streak read 3 instead of 2 — with CB_MAX_CONSECUTIVE_LOSSES=3 that phantom
    row is the difference between tripping a 120-minute halt and not.
    """
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.log_decision("ADA-USDT", "futures_sell_triggered", 0.0, "TP/SL triggered", pnl=-0.222,
                       close_type="CLOSE_LONG", exit_price=0.19648)
    store.log_decision("ADA-USDT", "close_reviewed", 0.0, "reviewed the close", pnl=-0.222)

    closes = store.realized_closes(limit=50)
    assert len(closes) == 1
    assert closes[0]["action"] == "futures_sell_triggered"
    assert all("review" not in str(c.get("action", "")).lower() for c in closes)


def test_evidence_free_duplicate_pnl_is_dropped_even_under_a_new_action_name(tmp_path):
    """Defence in depth: this bug class has recurred twice under different names.

    A row carrying NO execution evidence that merely repeats a real close's exact pnl on the same
    symbol nearby is a duplicate, whatever it is called.
    """
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.log_decision("XRP-USDT", "futures_buy_triggered", 0.0, "TP", pnl=0.334,
                       close_type="CLOSE_SHORT", exit_price=1.02965)
    store.log_decision("XRP-USDT", "close_position", 0.0, "some future label", pnl=0.334)

    closes = store.realized_closes(limit=50)
    assert len(closes) == 1 and closes[0]["closeType"] == "CLOSE_SHORT"


def test_a_genuine_second_close_on_the_same_symbol_is_still_kept(tmp_path):
    """The dedup must not swallow a real re-entry that happens to be on the same symbol."""
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.log_decision("ADA-USDT", "futures_sell_triggered", 0.0, "TP", pnl=-0.222,
                       close_type="CLOSE_LONG", exit_price=0.19648)
    store.log_decision("ADA-USDT", "futures_sell_triggered", 0.0, "TP", pnl=0.410,
                       close_type="CLOSE_LONG", exit_price=0.21100)
    assert len(store.realized_closes(limit=50)) == 2


def test_direction_calls_are_recorded_even_when_no_order_is_placed(tmp_path):
    """The live deadlock of 2026-08-10, in one test.

    Continuation measured "no edge" -> its risk was cut to the floor -> the resulting $7.49 notional
    fell under the $10.24 contract minimum -> the order was rejected -> no probe was recorded -> with
    no new probes the family could never earn back the evidence that would restore its size. Recording
    at the point of the CALL breaks the circularity: risk can fall as low as the measurement warrants
    without ever starving the measurement.
    """
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.record_signal_probe("XRP-USDT", "sell", 1.0235, "continuation")
    probes = store.signal_probes(limit=50)
    assert len(probes) == 1
    ctx = probes[0]["entryContext"]
    assert ctx["positionSide"] == "short"          # side normalised for scoring
    assert ctx["marketPriceAtSignal"] == 1.0235
    assert ctx["setupFamily"] == "continuation"


def _age_probe(tmp_path, seconds: int, store) -> None:
    """Push the stored probe back in time, as the live loop's clock would."""
    import json
    p = tmp_path / "memory.json"
    d = json.loads(p.read_text())
    d["signal_probes"][0]["ts"] -= seconds
    p.write_text(json.dumps(d))
    store._cache = None


def test_recorded_calls_settle_and_score_like_any_other_probe(tmp_path):
    from src.edge import signal_edge_stats
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.record_signal_probe("XRP-USDT", "sell", 100.0, "continuation")
    # a short that then fell 1% is a correct call
    _age_probe(tmp_path, 61 * 60, store)         # just past the 60m horizon
    # 3 settles: the 60m price, plus 5m/15m written off as missed (their moment is long gone).
    assert store.settle_signal_probes({"XRP-USDT": 99.0}) == 3

    probe = store.signal_probes(limit=50)[0]["entryContext"]["signalProbe"]
    assert probe["m60"] == 99.0
    assert probe["m5"] is None and probe["m15"] is None
    assert "m240" not in probe                   # not due yet, so not touched

    stats = signal_edge_stats(store.signal_probes(limit=50), cost_pct=0.001, min_samples=1)
    assert stats["by_family"]["continuation"]["mean_pct"] == pytest.approx(1.0)


def test_a_long_elapsed_horizon_is_recorded_missed_not_back_stamped(tmp_path):
    """The corruption that adding the 5m horizon would otherwise have caused.

    The settle check used to be "has the horizon passed?", which reads as "stamp today's price on
    every probe old enough". Harmless while the loop runs every 60s, wrong after downtime — and on
    the day a NEW horizon is introduced it would have stamped all 400 retained probes at once,
    labelling a multi-day return as a five-minute one. A missed measurement must record as missed.
    """
    from src.edge import signal_edge_stats
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.record_signal_probe("XRP-USDT", "buy", 100.0, "continuation")
    _age_probe(tmp_path, 5 * 86400, store)       # five days of downtime

    assert store.settle_signal_probes({"XRP-USDT": 400.0}) == 4
    probe = store.signal_probes(limit=50)[0]["entryContext"]["signalProbe"]
    assert probe == {"m5": None, "m15": None, "m60": None, "m240": None}

    # A 300% "5-minute return" must not reach the statistics.
    stats = signal_edge_stats(store.signal_probes(limit=50), cost_pct=0.001, min_samples=1)
    assert stats["by_horizon"] == {}
    assert stats["verdict"] == "insufficient data"

    # And the write-off is final: re-settling must not resurrect the horizon at a newer price.
    assert store.settle_signal_probes({"XRP-USDT": 500.0}) == 0


def test_probes_are_retained_by_count_not_by_clock(tmp_path):
    import json, time as _t
    store = MemoryStore(str(tmp_path / "memory.json"), retention_days=1)
    store.record_signal_probe("ADA-USDT", "buy", 0.20, "fade_extreme")
    p = tmp_path / "memory.json"
    d = json.loads(p.read_text())
    d["signal_probes"][0]["ts"] = int(_t.time()) - 60 * 86400
    p.write_text(json.dumps(d))
    store._cache = None
    assert len(store.signal_probes(limit=50)) == 1


def test_probe_recording_ignores_unusable_calls(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.record_signal_probe("X-USDT", "hold", 100.0)      # not a direction
    store.record_signal_probe("X-USDT", "buy", 0)           # no price
    store.record_signal_probe("X-USDT", "buy", "abc")       # unparseable
    assert store.signal_probes(limit=50) == []


def test_local_close_estimate_is_superseded_by_the_exchange_report(tmp_path):
    """Third occurrence of this bug class, third action name.

    The bot logs its own close estimate immediately, then KuCoin reports the authoritative figure
    seconds later with a slightly DIFFERENT pnl. On 2026-08-30 DASH-USDT logged `close_short` at
    -0.0465 and `futures_buy_triggered` at -0.0412 twenty-five seconds later, and both were booked.
    The earlier guards missed it: the first keys on the action name `futures_*`, and the exact-pnl
    fallback requires identical values, which an estimate never is.
    """
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.log_decision("DASH-USDT", "close_short", 0.0, "closed at market", pnl=-0.0465)
    store.log_decision("DASH-USDT", "futures_buy_triggered", 0.0, "TP/SL triggered", pnl=-0.0412,
                       close_type="CLOSE_SHORT", exit_price=42.99)

    closes = store.realized_closes(limit=50)
    assert len(closes) == 1
    assert closes[0]["action"] == "futures_buy_triggered"
    assert closes[0]["pnl"] == pytest.approx(-0.0412)   # the exchange's figure wins


def test_estimate_suppression_needs_the_same_direction_and_magnitude(tmp_path):
    """It must not swallow a genuinely different trade that merely happens to be nearby."""
    store = MemoryStore(str(tmp_path / "memory.json"))
    # opposite sign -> a different trade, keep both
    store.log_decision("DASH-USDT", "close_short", 0.0, "est", pnl=+0.20)
    store.log_decision("DASH-USDT", "futures_buy_triggered", 0.0, "trig", pnl=-0.04,
                       close_type="CLOSE_SHORT", exit_price=42.99)
    assert len(store.realized_closes(limit=50)) == 2

    other = MemoryStore(str(tmp_path / "memory2.json"))
    # same sign but an order of magnitude apart -> not the same close
    other.log_decision("SUI-USDT", "close_long", 0.0, "est", pnl=-1.50)
    other.log_decision("SUI-USDT", "futures_sell_triggered", 0.0, "trig", pnl=-0.04,
                       close_type="CLOSE_LONG", exit_price=1.23)
    assert len(other.realized_closes(limit=50)) == 2


def test_an_anonymous_narrative_close_cannot_book_a_second_trade(tmp_path):
    """The dashboard's phantom bar: NEAR-USDT, 2026-09-01.

    The bracket fired at +0.00335 and five seconds later the agent logged its own reduce-only
    close at +0.00666 — the same close, described twice, 99% apart because the agent's figure was
    never reconciled with anything. The 50% estimate band let it through, so one trade was booked
    twice: it drew a third bar on the outcome chart with no matching card in "recently closed"
    (cards need entry/exit prices, which a narrative echo has never had), and it corrupted realized
    PnL and every rolling stat downstream.
    """
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.log_decision("NEAR-USDT", "futures_sell_triggered", 0.0, "TP hit", pnl=+0.00335,
                       close_type="CLOSE_LONG", exit_price=2.481)
    store.log_decision("NEAR-USDT", "close_long", 0.0, "closed the runner", pnl=+0.00666)

    closes = store.realized_closes(limit=50)
    assert len(closes) == 1
    assert closes[0]["action"] == "futures_sell_triggered"
    assert closes[0]["pnl"] == pytest.approx(0.00335)


def test_the_widened_band_only_applies_to_rows_that_name_no_position(tmp_path):
    """Provenance beats guesswork: a row that says which position it closed is a trade, not an echo."""
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.log_decision("NEAR-USDT", "futures_sell_triggered", 0.0, "TP hit", pnl=+0.00335,
                       close_type="CLOSE_LONG", exit_price=2.481, position_id="pos-1")
    store.log_decision("NEAR-USDT", "close_long", 0.0, "second scalp", pnl=+0.00666,
                       position_id="pos-2")
    assert len(store.realized_closes(limit=50)) == 2


def test_a_triggered_close_is_never_treated_as_someone_elses_estimate(tmp_path):
    """Old rows predate closeType/realizedR, so they look 'evidence-free' — they must still count."""
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.log_decision("XRP-USDT", "futures_buy_triggered", 0.0, "old-style row", pnl=+0.0757)
    assert len(store.realized_closes(limit=50)) == 1


class TestLatestLimitsReadOnly:
  """A venue that failed to read is UNKNOWN, not zero.

  `total_usdt` is a sum across spot + futures + financial, and each term silently contributes 0.0
  when its fetch fails — a KuCoin 504 on the futures overview drops the whole futures balance out of
  "total equity" without raising. That number drives the drawdown circuit breaker, the daily baseline
  and the COMPOUNDING equity index, so a partial snapshot either trips the breaker on a phantom loss
  or books the recovery as a phantom gain that can never be undone. Live evidence: a +19.46% index
  step across an outage during which the user made no deposits or withdrawals at all — only internal
  spot<->futures transfers, which cancel in that sum by construction.
  """

  def test_returns_the_last_recorded_row_without_writing(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.update_limits(100.0, scope="total")
    before = m.latest_limits("total")
    assert before["currentUsdt"] == pytest.approx(100.0)
    # Reading must not mutate: a second read is identical, and no new baseline is established.
    assert m.latest_limits("total") == before
    assert m.latest_limits("total")["baselineUsdt"] == before["baselineUsdt"]

  def test_unknown_scope_and_empty_store_return_empty(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    assert m.latest_limits("total") == {}
    assert m.latest_limits("nonesuch") == {}
    m.update_limits(50.0, scope="futures")
    assert m.latest_limits("spot") == {}
    assert m.latest_limits("futures")["currentUsdt"] == pytest.approx(50.0)

  def test_the_caller_cannot_corrupt_stored_state(self, tmp_path):
    """It hands back a copy — a caller mutating it must not rewrite the account's history."""
    m = MemoryStore(str(tmp_path / "m.json"))
    m.update_limits(100.0, scope="total")
    got = m.latest_limits("total")
    got["currentUsdt"] = 999999.0
    assert m.latest_limits("total")["currentUsdt"] == pytest.approx(100.0)

  def test_a_partial_snapshot_would_otherwise_look_like_a_total_loss(self, tmp_path):
    """Documents the shape of the bug this exists to prevent: recording a poll whose futures read
    failed reports a ~100% drawdown on an account that never lost anything."""
    m = MemoryStore(str(tmp_path / "m.json"))
    m.update_limits(67.14, scope="total")           # healthy poll: spot 0 + futures 67.14
    partial = m.update_limits(0.0, scope="total")   # futures overview 504s -> the sum is just spot
    assert partial["drawdownPct"] > 99.0, "a dropped venue reads as a near-total loss"
    # Which is exactly why the poll loop reuses latest_limits instead of recording that.


class TestPerFamilyProbeRetention:
  """A loud family must not evict a quiet one's verdict.

  Measured 2026-09-14. A stood-aside playbook still records a probe on every direction call — that is
  deliberate, it is how the family can earn its way back. But it means the BENCHED family is usually
  the loudest: `continuation`, benched for weeks, held 283 of 479 retained probes (59%). Under one
  global ring buffer it evicted `fade_extreme`'s history, whose decimated sample fell from 27 to 17 —
  under the stand-aside's `min_samples=20`. Its verdict flipped from "no edge" to "insufficient data",
  which RELEASES a family to full size, and it immediately took two trades and lost both.

  Losing the evidence for a verdict must never be equivalent to never having had it.
  """

  @staticmethod
  def _probe(family, ts):
    return {"symbol": "X-USDT", "ts": ts,
            "entryContext": {"positionSide": "long", "marketPriceAtSignal": 100.0,
                             "setupFamily": family, "signalProbe": {}}}

  def test_a_loud_family_cannot_evict_a_quiet_one(self):
    from src.memory import _trim_probes_per_family, MAX_PROBES_PER_FAMILY
    loud = [self._probe("continuation", 1000 + i) for i in range(MAX_PROBES_PER_FAMILY * 3)]
    quiet = [self._probe("fade_extreme", 1 + i) for i in range(25)]
    kept = _trim_probes_per_family(quiet + loud)
    families = [r["entryContext"]["setupFamily"] for r in kept]
    assert families.count("fade_extreme") == 25, "the quiet family keeps ALL of its evidence"
    assert families.count("continuation") == MAX_PROBES_PER_FAMILY

  def test_each_family_keeps_its_newest_rows_in_order(self):
    from src.memory import _trim_probes_per_family, MAX_PROBES_PER_FAMILY
    rows = [self._probe("continuation", i) for i in range(MAX_PROBES_PER_FAMILY + 40)]
    kept = _trim_probes_per_family(rows)
    assert len(kept) == MAX_PROBES_PER_FAMILY
    ts = [r["ts"] for r in kept]
    assert ts == sorted(ts), "chronological order must survive trimming"
    assert ts[-1] == MAX_PROBES_PER_FAMILY + 39, "the NEWEST rows are the ones kept"

  def test_untagged_probes_are_bucketed_not_dropped(self):
    from src.memory import _trim_probes_per_family
    rows = [{"symbol": "X", "ts": i, "entryContext": {"positionSide": "long"}} for i in range(5)]
    rows += [{"symbol": "X", "ts": 100 + i} for i in range(3)]        # no entryContext at all
    kept = _trim_probes_per_family(rows)
    assert len(kept) == 8

  def test_junk_rows_are_dropped_without_raising(self):
    from src.memory import _trim_probes_per_family
    assert _trim_probes_per_family(None) == []
    assert _trim_probes_per_family([]) == []
    assert _trim_probes_per_family(["junk", None, 7]) == []

  def test_retention_survives_the_prune_sweep(self, tmp_path):
    """End to end: the store must not collapse back to one shared budget on the next write."""
    from src.memory import MAX_PROBES_PER_FAMILY
    m = MemoryStore(str(tmp_path / "m.json"))
    for i in range(MAX_PROBES_PER_FAMILY + 60):
      m.record_signal_probe("X-USDT", "buy", 100.0, setup_family="continuation")
    for i in range(30):
      m.record_signal_probe("Y-USDT", "sell", 100.0, setup_family="fade_extreme")
    m._write(m._prune(m._read()))
    rows = m._read()["signal_probes"]
    fams = [r["entryContext"]["setupFamily"] for r in rows]
    assert fams.count("fade_extreme") == 30, "quiet family intact after pruning"
    assert fams.count("continuation") == MAX_PROBES_PER_FAMILY


class TestSignalProbesReadAll:
  def test_limit_zero_returns_everything_retained(self, tmp_path):
    """The entry gates must not re-truncate what retention deliberately kept — that second cut is
    how the `fade_extreme` verdict was silently un-learned."""
    m = MemoryStore(str(tmp_path / "m.json"))
    # Distinct symbols: signal_probes dedupes on (symbol, ts), and a loop records within one second.
    for i in range(40):
      m.record_signal_probe(f"S{i}-USDT", "buy", 100.0, setup_family="continuation")
    assert len(m.signal_probes(limit=0)) == 40
    assert len(m.signal_probes(limit=10)) == 10
    assert len(m.signal_probes()) == 40        # default 200 > stored


# ── Settling on the market that fills (futures mark + funding), never the spot ticker ─────────────

def _shift_probe_clock(store, seconds: int, bucket: str = "signal_probes") -> None:
  """Age EVERY row in ``bucket`` by ``seconds``, as the live loop's clock would."""
  data = store._read()
  for row in data.get(bucket) or []:
    row["ts"] -= seconds
  store._write(data)


class TestProbeSettlementMarket:
  """ONE-USDT, 2026-09-20: spot ~0.0050 while the ONEUSDTM mark sat ~0.0038. A probe based on the
  futures mark and settled on spot recorded a +25% '5-minute return' on a +2.5% contract move, and
  eight of them lifted funding_carry from stood-aside to full size in a day."""

  def test_a_futures_based_probe_settles_on_the_futures_mark_and_a_spot_one_on_spot(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_signal_probe("ONE-USDT", "buy", 0.0039, "funding_carry", price_source="futures_mark")
    m.record_signal_probe("TWO-USDT", "buy", 0.0049, "funding_carry", price_source="spot")
    _shift_probe_clock(m, 6 * 60)            # just past the 5m horizon
    futures = {"ONE-USDT": 0.0038, "TWO-USDT": 0.0038}
    spot = {"ONE-USDT": 0.0050, "TWO-USDT": 0.0050}
    assert m.settle_signal_probes(futures, spot_prices=spot) == 2
    by_sym = {r["symbol"]: r["entryContext"] for r in m.signal_probes(limit=0)}
    assert by_sym["ONE-USDT"]["priceSource"] == "futures_mark"
    assert by_sym["ONE-USDT"]["signalProbe"]["m5"] == 0.0038      # NOT the 0.0050 spot ticker
    assert by_sym["TWO-USDT"]["signalProbe"]["m5"] == 0.0050      # its base was spot, so is its settle

  def test_legacy_rows_without_a_source_stamp_count_as_futures(self, tmp_path):
    """The base has been the futures mark since 2026-07-19; every retained probe is newer."""
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_signal_probe("ONE-USDT", "buy", 0.0039, "funding_carry")
    assert "priceSource" not in m.signal_probes(limit=0)[0]["entryContext"]
    _shift_probe_clock(m, 6 * 60)
    assert m.settle_signal_probes({"ONE-USDT": 0.0038}, spot_prices={"ONE-USDT": 0.0050}) == 1
    assert m.signal_probes(limit=0)[0]["entryContext"]["signalProbe"]["m5"] == 0.0038

  def test_no_futures_mark_means_wait_then_unmeasured_never_a_spot_fallback(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_signal_probe("ONE-USDT", "buy", 0.0039, "funding_carry", price_source="futures_mark")
    _shift_probe_clock(m, 6 * 60)
    # Only spot is known this poll: the futures-based probe waits rather than borrowing it.
    assert m.settle_signal_probes({}, spot_prices={"ONE-USDT": 0.0050}) == 0
    assert "m5" not in m.signal_probes(limit=0)[0]["entryContext"]["signalProbe"]
    # Past the 5m tolerance with still no mark: recorded as missed, never back-stamped.
    _shift_probe_clock(m, 5 * 60)
    assert m.settle_signal_probes({}, spot_prices={"ONE-USDT": 0.0050}) == 1
    assert m.signal_probes(limit=0)[0]["entryContext"]["signalProbe"]["m5"] is None

  def test_funding_credit_is_stamped_beside_the_price_for_the_horizon_window(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_signal_probe("ONE-USDT", "buy", 0.0039, "funding_carry", price_source="futures_mark")
    _shift_probe_clock(m, 61 * 60)
    ts0 = m.signal_probes(limit=0)[0]["ts"]
    calls = []

    def funding(symbol, side, t0, t1):
      calls.append((symbol, side, t0, t1))
      return 0.0025

    assert m.settle_signal_probes({"ONE-USDT": 0.0040}, funding_received=funding) == 3
    probe = m.signal_probes(limit=0)[0]["entryContext"]["signalProbe"]
    assert probe["m60"] == 0.0040 and probe["f60"] == 0.0025
    assert probe["m5"] is None and "f5" not in probe          # a written-off horizon earns no credit
    assert calls == [("ONE-USDT", "long", ts0, calls[0][3])]
    assert calls[0][3] >= ts0 + 60 * 60                        # window ends at the observation

  def test_a_failing_funding_lookup_never_costs_the_price_stamp(self, tmp_path, caplog):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_signal_probe("ONE-USDT", "sell", 0.0039, "funding_carry", price_source="futures_mark")
    _shift_probe_clock(m, 6 * 60)

    def broken(*_a):
      raise RuntimeError("funding endpoint down")

    with caplog.at_level("WARNING"):
      assert m.settle_signal_probes({"ONE-USDT": 0.0038}, funding_received=broken) == 1
    probe = m.signal_probes(limit=0)[0]["entryContext"]["signalProbe"]
    # The price still stamps; the credit is recorded UNKNOWN (None), never absent-as-zero (W3).
    assert probe["m5"] == 0.0038 and "f5" in probe and probe["f5"] is None
    assert isinstance(probe["t5"], int)                          # when the price was observed
    assert any("PROBE FUNDING" in r.message for r in caplog.records)

  def test_an_unknown_credit_is_backfilled_for_its_own_window_on_a_later_poll(self, tmp_path):
    """W3 (2026-09-25 review): one transient failure used to lose the carry credit for good — the
    horizon was never due again and absent scored as zero (-0.8% at 240m on a 1h carry at -0.2%)."""
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_signal_probe("ONE-USDT", "sell", 0.0039, "funding_carry", price_source="futures_mark")
    _shift_probe_clock(m, 6 * 60)

    def broken(*_a):
      raise RuntimeError("503 transient")

    assert m.settle_signal_probes({"ONE-USDT": 0.0038}, funding_received=broken) == 1
    data = m._read()
    row = data["signal_probes"][0]
    observed = row["ts"] + 5 * 60 + 30                          # the price was read 30s after the horizon
    row["entryContext"]["signalProbe"]["t5"] = observed
    m._write(data)
    calls = []

    def healthy(symbol, side, t0, t1):
      calls.append((symbol, side, t0, t1))
      return 0.002

    assert m.settle_signal_probes({"ONE-USDT": 0.0038}, funding_received=healthy) == 1
    probe = m.signal_probes(limit=0)[0]["entryContext"]["signalProbe"]
    assert probe["f5"] == pytest.approx(0.002) and "t5" not in probe
    assert calls == [("ONE-USDT", "short", row["ts"], observed)]  # the horizon's window, never 'now'
    assert m.settle_signal_probes({"ONE-USDT": 0.0038}, funding_received=healthy) == 0   # done
    assert len(calls) == 1

  def test_past_its_window_an_unknown_credit_stays_unknown_and_carry_scoring_skips_it(self, tmp_path):
    from src.edge import _probe_observations, signal_edge_stats
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_signal_probe("ONE-USDT", "sell", 0.0039, "funding_carry", price_source="futures_mark")
    m.record_signal_probe("TWO-USDT", "sell", 1.0, "continuation", price_source="futures_mark")
    _shift_probe_clock(m, 6 * 60)

    def broken(*_a):
      raise RuntimeError("503 transient")

    assert m.settle_signal_probes({"ONE-USDT": 0.0038, "TWO-USDT": 0.99}, funding_received=broken) == 2
    _shift_probe_clock(m, 10 * 60)                               # past the 5m backfill window
    calls = []
    assert m.settle_signal_probes({}, funding_received=lambda *a: calls.append(a) or 0.002,
                                  horizons_min=(5,)) == 0
    assert calls == []                                           # a dead symbol is not asked forever
    rows = m.signal_probes(limit=0)
    assert all(r["entryContext"]["signalProbe"]["f5"] is None for r in rows)
    got = {r["symbol"]: signed for r, _c, _h, signed in _probe_observations(rows, (5,))}
    assert "ONE-USDT" not in got                                 # carry: unknown is not zero
    assert got["TWO-USDT"] == pytest.approx(0.01)                # non-carry: absent/None = zero, as before
    assert signal_edge_stats(rows, horizons_min=(5,))["unknownFundingCredit"] == 1

  def test_one_malformed_row_does_not_starve_the_rest(self, tmp_path, caplog):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_signal_probe("ONE-USDT", "buy", 0.0039, "funding_carry", price_source="futures_mark")
    m.record_signal_probe("TWO-USDT", "buy", 1.0, "continuation", price_source="futures_mark")
    _shift_probe_clock(m, 6 * 60)
    data = m._read()
    data["signal_probes"][0]["ts"] = "not-a-time"
    m._write(data)
    with caplog.at_level("WARNING"):
      assert m.settle_signal_probes({"ONE-USDT": 0.0038, "TWO-USDT": 1.01}) == 1
    assert any("SIGNAL PROBE settle failed" in r.message for r in caplog.records)

  def test_symbols_due_lists_only_what_needs_a_futures_mark_now(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_signal_probe("DUE-USDT", "buy", 1.0, "continuation", price_source="futures_mark")
    m.record_signal_probe("SPOT-USDT", "buy", 1.0, "continuation", price_source="spot")
    m.record_signal_probe("NEW-USDT", "buy", 1.0, "continuation", price_source="futures_mark")
    data = m._read()
    for row in data["signal_probes"]:
      if row["symbol"] in ("DUE-USDT", "SPOT-USDT"):
        row["ts"] -= 6 * 60                  # 5m horizon due now
    m._write(data)
    m.record_exit_probe("EXIT-USDT", "long", 100.0, 90.0, 130.0, 101.0, realized_r=0.1)
    m.record_exit_probe("OLD-USDT", "long", 100.0, 90.0, 130.0, 101.0, realized_r=0.1)
    data = m._read()
    data["exit_probes"][1]["ts"] -= 30 * 3600  # far past its measurable life
    m._write(data)
    assert m.symbols_due_for_settlement() == {"DUE-USDT", "EXIT-USDT"}


class TestFundingReceivedFromHistory:
  """KuCoin: a positive rate means longs pay shorts. A long RECEIVES -rate, a short +rate."""

  T = 1_790_000_000                                             # a real epoch (seconds)
  HIST = [
    {"fundingRate": 0.001, "timepoint": (T + 100) * 1000},        # inside, in ms (KuCoin's unit)
    {"fundingRate": -0.0004, "timepoint": T + 3700},              # inside, in seconds
    {"fundingRate": -0.0004, "timepoint": (T + 3700) * 1000},     # the same settlement again (paged)
    {"fundingRate": 0.5, "timepoint": (T - 1000) * 1000},         # before the window
    {"fundingRate": 0.5, "timepoint": (T + 20_000) * 1000},       # after the window
    {"fundingRate": "junk", "timepoint": (T + 200) * 1000},
  ]

  def test_sign_follows_the_side_that_is_paid(self):
    from src.memory import funding_received_from_history as fr
    t0, t1 = self.T, self.T + 10_000
    assert fr(self.HIST, "long", t0, t1) == pytest.approx(-(0.001 - 0.0004))
    assert fr(self.HIST, "short", t0, t1) == pytest.approx(0.001 - 0.0004)
    assert fr(self.HIST, "buy", t0, t1) == pytest.approx(-0.0006)

  def test_window_is_open_at_the_start_and_closed_at_the_end(self):
    from src.memory import funding_received_from_history as fr
    hist = [{"fundingRate": 0.001, "timepoint": self.T * 1000}]
    assert fr(hist, "short", self.T, self.T + 100) == 0.0         # opened AT the settlement: not paid
    assert fr(hist, "short", self.T - 1000, self.T) == pytest.approx(0.001)

  def test_unusable_input_is_unknown_not_zero(self):
    from src.memory import funding_received_from_history as fr
    assert fr(self.HIST, "sideways", 0, 1) is None
    assert fr(None, "long", 0, 1) is None
    assert fr([], "long", 0, 1) == 0.0


class TestExitProbeStaleness:
  """XMR, 2026-09-12: an exit probe was resolved as a STOP seven days later, when its ticker reappeared.
  A bracket outcome is only an outcome inside the probe's measurable life."""

  def _probe(self, tmp_path, age_sec):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_exit_probe("XMR-USDT", "long", 100.0, 90.0, 130.0, 101.0, realized_r=0.1)
    data = m._read()
    data["exit_probes"][0]["ts"] -= age_sec
    m._write(data)
    return m

  def test_no_price_past_expiry_plus_tolerance_is_unmeasured(self, tmp_path):
    m = self._probe(tmp_path, 10 * 3600)     # 8h life + 96min tolerance < 10h
    assert m.settle_exit_probes({}) == 1
    out = m.exit_probes()[0]["outcome"]
    assert out["resolved"] == "unmeasured" and out["bracketR"] is None

  def test_a_late_price_does_not_resolve_it_as_stop_or_target(self, tmp_path):
    m = self._probe(tmp_path, 7 * 86400)
    assert m.settle_exit_probes({"XMR-USDT": 80.0}) == 1       # far through the stop, a week late
    out = m.exit_probes()[0]["outcome"]
    assert out["resolved"] == "unmeasured" and out["bracketR"] is None
    assert m.settle_exit_probes({"XMR-USDT": 200.0}) == 0      # final: never re-scored

  def test_inside_its_life_it_still_resolves_normally(self, tmp_path):
    m = self._probe(tmp_path, 9 * 3600)      # past expiry, inside the tolerance
    assert m.settle_exit_probes({"XMR-USDT": 105.0}) == 1
    assert m.exit_probes()[0]["outcome"]["resolved"] == "expired"


def test_a_direction_call_names_the_model_and_its_stated_confidence(tmp_path):
    """Seven absolute confidence thresholds shape entries and a model swap re-scales the number; none
    of it was measurable because no probe said which model spoke or how sure it claimed to be."""
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.record_signal_probe("SPX-USDT", "sell", 1.0, "continuation",
                              model="gpt-6-luna", confidence=0.78, min_confidence=0.75)
    ctx = store.signal_probes(limit=0)[0]["entryContext"]
    assert (ctx["model"], ctx["confidence"], ctx["minConfidence"]) == ("gpt-6-luna", 0.78, 0.75)

    # An unusable value is dropped, never stored as a fake number; an old-style call stamps nothing.
    store.record_signal_probe("ADA-USDT", "buy", 1.0, "continuation",
                              model="  ", confidence="high", min_confidence=float("nan"))
    store.record_signal_probe("DOT-USDT", "buy", 1.0, "continuation")
    for sym in ("ADA-USDT", "DOT-USDT"):
        row = next(p for p in store.signal_probes(limit=0) if p["symbol"] == sym)["entryContext"]
        assert not {"model", "confidence", "minConfidence"} & set(row)


# ── Execution-map stamps and order-lease extremes on signal probes (2026-09-25) ────────────────────────

def test_a_direction_call_carries_its_planned_execution(tmp_path):
    """What edge.execution_map's counterfactual and crossBand need: the ATR15, the planned bracket, the
    lease, and the net RR if crossed at the live price. Recorded only; unusable values are dropped."""
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.record_signal_probe("SPX-USDT", "sell", 1.0, "continuation", atr15_pct=1.8, planned_entry=1.01,
                              planned_stop=1.03, planned_tp=0.95, crossed_net_rr=1.37, lease_min=15)
    ctx = store.signal_probes(limit=0)[0]["entryContext"]
    assert (ctx["atr15Pct"], ctx["plannedEntry"], ctx["plannedStop"], ctx["plannedTp"]) == (1.8, 1.01, 1.03, 0.95)
    assert (ctx["crossedNetRr"], ctx["leaseMin"]) == (1.37, 15.0)
    store.record_signal_probe("ADA-USDT", "buy", 1.0, "continuation", atr15_pct=-1, planned_stop=float("nan"),
                              crossed_net_rr=0.0, lease_min=None)
    ada = next(p for p in store.signal_probes(limit=0) if p["symbol"] == "ADA-USDT")["entryContext"]
    assert ada["crossedNetRr"] == 0.0                   # costs ate the reward: a real value, kept
    assert not {"atr15Pct", "plannedStop", "leaseMin"} & set(ada)


class _LeaseSpy:
    def __init__(self, result=(0.98, 1.02), exc=None):
        self.result, self.exc, self.calls = result, exc, []

    def __call__(self, symbol, t0, t1):
        self.calls.append((symbol, t0, t1))
        if self.exc:
            raise self.exc
        return self.result


class TestProbeLeaseExtremes:
    """Every call gets the low/high its contract traded over the order lease, so the execution map can
    say whether a limit at ANY depth would have filled — not only at the depth the model used."""

    def _probe(self, tmp_path, **kw):
        m = MemoryStore(str(tmp_path / "m.json"))
        m.record_signal_probe("SPX-USDT", "sell", 1.0, "continuation", atr15_pct=1.0, lease_min=15, **kw)
        return m

    def test_stamped_once_after_the_lease_and_its_last_bar(self, tmp_path):
        m = self._probe(tmp_path)
        spy = _LeaseSpy()
        _shift_probe_clock(m, 15 * 60 + 30)        # lease over, but its last 1m bar is still open
        m.settle_signal_probes({}, lease_extremes=spy)
        assert spy.calls == []
        _shift_probe_clock(m, 60)
        m.settle_signal_probes({}, lease_extremes=spy)
        ts = m.signal_probes(limit=0)[0]["ts"]
        assert spy.calls == [("SPX-USDT", float(ts), float(ts) + 900.0)]
        ctx = m.signal_probes(limit=0)[0]["entryContext"]
        assert (ctx["leaseLow"], ctx["leaseHigh"]) == (0.98, 1.02)
        m.settle_signal_probes({}, lease_extremes=spy)
        assert len(spy.calls) == 1                  # stamped once

    def test_unavailable_bars_retry_then_are_written_off_and_never_back_filled(self, tmp_path):
        m = self._probe(tmp_path)
        _shift_probe_clock(m, 17 * 60)
        m.settle_signal_probes({}, lease_extremes=_LeaseSpy(result=None))
        assert "leaseLow" not in m.signal_probes(limit=0)[0]["entryContext"]     # still waiting
        _shift_probe_clock(m, 30 * 60)              # past lease + one bar + the tolerance
        late = _LeaseSpy()
        m.settle_signal_probes({}, lease_extremes=late)
        ctx = m.signal_probes(limit=0)[0]["entryContext"]
        assert late.calls == [] and ctx["leaseLow"] is None and ctx["leaseHigh"] is None
        m.settle_signal_probes({}, lease_extremes=_LeaseSpy())
        assert m.signal_probes(limit=0)[0]["entryContext"]["leaseLow"] is None      # never back-filled

    def test_a_raising_lookup_warns_and_never_costs_the_price_stamps(self, tmp_path, caplog):
        m = self._probe(tmp_path)
        _shift_probe_clock(m, 16 * 60)
        with caplog.at_level("WARNING"):
            m.settle_signal_probes({"SPX-USDT": 0.99}, lease_extremes=_LeaseSpy(exc=RuntimeError("klines down")))
        ctx = m.signal_probes(limit=0)[0]["entryContext"]
        assert ctx["signalProbe"]["m15"] == 0.99 and "leaseLow" not in ctx
        assert any("PROBE LEASE" in r.message and r.levelname == "WARNING" for r in caplog.records)

    def test_only_probes_that_stamped_a_lease_qualify(self, tmp_path):
        m = MemoryStore(str(tmp_path / "m.json"))
        m.record_signal_probe("SPX-USDT", "sell", 1.0, "continuation")              # recorded before the stamp
        m.record_trade("SPX-USDT", "sell", 10.0, price=1.01, venue="futures", filled=False,
                       entry_context={"marketPriceAtSignal": 1.0, "positionSide": "short", "leaseMin": 15})
        _shift_probe_clock(m, 40 * 60)
        _shift_probe_clock(m, 17 * 60, bucket="trades")        # the trade row WOULD be due, if it qualified
        spy = _LeaseSpy()
        m.settle_signal_probes({}, lease_extremes=spy)
        assert spy.calls == []
        assert "leaseLow" not in m.signal_probes(limit=0)[0]["entryContext"]
        assert all("leaseLow" not in (t.get("entryContext") or {}) for t in m._read()["trades"])

    def test_without_a_lookup_nothing_is_written_off(self, tmp_path):
        """Callers that do not inject the lookup (older call sites, tests) must not burn the window."""
        m = self._probe(tmp_path)
        _shift_probe_clock(m, 40 * 60)
        m.settle_signal_probes({})
        assert "leaseLow" not in m.signal_probes(limit=0)[0]["entryContext"]


# ── Market state on probes, and data-quality failures for the screener (2026-09-25) ──────────────────

from src.memory import MAX_ANALYSIS_FAILURES, sanitize_market_state  # noqa: E402

_STATE = {"asOf": 1_790_300_000, "universe": 68, "breadth24": 0.07, "basketMedian24h": -6.2,
          "btc24h": -1.4, "btc72h": 2.1, "btcDailyAdx": 18.5, "btcDailyBias": "Bullish", "extra": "junk"}


class TestMarketStateSanitizer:
  def test_keeps_the_whitelisted_numbers(self):
    out = sanitize_market_state(_STATE)
    assert out == {"asOf": 1_790_300_000, "universe": 68, "breadth24": 0.07, "basketMedian24h": -6.2,
                   "btc24h": -1.4, "btc72h": 2.1, "btcDailyAdx": 18.5, "btcDailyBias": "bullish"}

  def test_out_of_domain_values_are_dropped_not_clamped(self):
    out = sanitize_market_state({"breadth24": 1.4, "btcDailyAdx": -3, "btc24h": float("nan"),
                                 "btcDailyBias": "sideways", "basketMedian24h": "x", "btc72h": 1.0})
    assert out == {"btc72h": 1.0}

  def test_a_block_with_no_reading_is_none(self):
    assert sanitize_market_state({"asOf": 5, "universe": 3}) is None
    assert sanitize_market_state(None) is None and sanitize_market_state("x") is None


class TestProbesCarryTheMarketState:
  def test_signal_probe_stamps_the_sanitized_block(self, tmp_path):
    m = MemoryStore(str(tmp_path / "p.json"))
    m.record_signal_probe("SPX-USDT", "sell", 1.0, "continuation", market_state=_STATE)
    ctx = m.signal_probes(limit=0)[0]["entryContext"]
    assert ctx["marketState"] == sanitize_market_state(_STATE)

  def test_signal_probe_without_a_reading_has_no_key(self, tmp_path):
    m = MemoryStore(str(tmp_path / "p.json"))
    m.record_signal_probe("SPX-USDT", "sell", 1.0, "continuation")
    assert "marketState" not in m.signal_probes(limit=0)[0]["entryContext"]

  def test_exit_probe_stamps_entry_and_exit_states(self, tmp_path):
    m = MemoryStore(str(tmp_path / "x.json"))
    at_exit = dict(_STATE, breadth24=0.66)
    m.record_exit_probe("ONE-USDT", "long", 100.0, 90.0, 130.0, 101.0, realized_r=0.1,
                        closed_by="protection", market_state=_STATE, market_state_at_exit=at_exit)
    row = m.exit_probes(limit=10)[0]
    assert row["marketState"]["breadth24"] == 0.07 and row["marketStateAtExit"]["breadth24"] == 0.66
    m.record_exit_probe("TWO-USDT", "long", 100.0, 90.0, 130.0, 101.0, realized_r=0.1)
    assert m.exit_probes(limit=10)[-1]["marketState"] is None


class TestAnalysisFailures:
  def test_record_read_and_expire_on_its_own_retry_time(self, tmp_path):
    m = MemoryStore(str(tmp_path / "af.json"))
    now = time.time()
    m.record_analysis_failure("TAKE-USDT", reason="1hour: 6 candle gap(s)", retry_after=now + 7200)
    got = m.analysis_failures(now=now)["TAKE-USDT"]
    assert got["reason"] == "1hour: 6 candle gap(s)" and got["remainingHours"] == pytest.approx(2.0, abs=0.1)
    assert m.analysis_failures(now=now + 7201) == {}

  def test_a_retry_already_past_is_not_stored(self, tmp_path):
    m = MemoryStore(str(tmp_path / "af.json"))
    m.record_analysis_failure("TAKE-USDT", reason="x", retry_after=time.time() - 1)
    m.record_analysis_failure("TAKE-USDT", reason="x", retry_after="soon")
    assert m.analysis_failures() == {}

  def test_clear_and_one_row_per_symbol(self, tmp_path):
    m = MemoryStore(str(tmp_path / "af.json"))
    now = time.time()
    m.record_analysis_failure("TAKEUSDTM", reason="old", retry_after=now + 100)
    m.record_analysis_failure("TAKE-USDT", reason="new", retry_after=now + 200)
    assert list(m.analysis_failures()) == ["TAKE-USDT"] and m.analysis_failures()["TAKE-USDT"]["reason"] == "new"
    assert m.clear_analysis_failure("TAKE-USDT") is True
    assert m.analysis_failures() == {} and m.clear_analysis_failure("TAKE-USDT") is False

  def test_it_survives_the_models_remove_coin(self, tmp_path):
    """Kept apart from the coins list, which remove_coin overwrites and which is capped at 50."""
    m = MemoryStore(str(tmp_path / "af.json"))
    m.record_analysis_failure("TAKE-USDT", reason="gaps", retry_after=time.time() + 3600)
    m.remove_coin("TAKE-USDT", reason="model pruned it", exit_plan="none")
    assert "TAKE-USDT" in m.analysis_failures()

  def test_the_store_is_capped(self, tmp_path):
    m = MemoryStore(str(tmp_path / "af.json"))
    now = time.time()
    for i in range(MAX_ANALYSIS_FAILURES + 5):
      m.record_analysis_failure(f"C{i}-USDT", reason="x", retry_after=now + 3600, now=now + i)
    stored = m.analysis_failures(now=now)
    assert len(stored) == MAX_ANALYSIS_FAILURES and "C0-USDT" not in stored


# ── Gate probes: what each directional gate blocks, kept OUT of the family verdicts (2026-09-25) ─────


def _age_gate_rows(store, seconds: int) -> None:
  """Age every gate-probe row by ``seconds``, as the live loop's clock would."""
  data = store._read()
  for row in data.get("gate_probes") or []:
    row["ts"] -= seconds
  store._write(data)


class TestGateProbeStore:
  """The directional gates returned before the signal probe, so a refused call left no evidence (8 hard
  refusals on 09-22..24, 0 probes). Refusals and model-independent gate-state readings now record into
  their own bucket — which must never leak into signal_probes() and so into any family verdict."""

  def test_signal_probes_never_include_gate_probes(self, tmp_path):
    m = MemoryStore(str(tmp_path / "g.json"))
    m.record_signal_probe("ONE-USDT", "buy", 1.0, "continuation", price_source="futures_mark")
    assert m.record_gate_probe("TWO-USDT", "buy", 2.0, "h1_align", setup_family="continuation",
                               price_source="futures_mark") is True
    assert m.record_gate_state_probes("THREE-USDT", 3.0, {"long": ["anti_fomo"], "short": []},
                                      price_source="futures_mark") == 2
    assert [r["symbol"] for r in m.signal_probes(limit=0)] == ["ONE-USDT"]
    kinds = sorted((r["symbol"], r["entryContext"]["gateProbe"]) for r in m.gate_probes())
    assert kinds == [("THREE-USDT", "state"), ("THREE-USDT", "state"), ("TWO-USDT", "refusal")]

  def test_a_refusal_row_carries_the_gate_the_call_and_the_market(self, tmp_path):
    m = MemoryStore(str(tmp_path / "g.json"))
    m.record_gate_probe("SPX-USDT", "sell", 1.02, "confidence_floor", setup_family="Continuation",
                        price_source="futures_mark", model="gpt-6-luna", confidence=0.72,
                        regime={"daily_bias": "bullish", "daily_exhausted": True, "intraday_bias_1h": "bearish",
                                "analyzed_at": 123.0, "intraday_vwap": 1.0})
    ctx = m.gate_probes()[0]["entryContext"]
    assert ctx["gateProbe"] == "refusal" and ctx["gate"] == "confidence_floor" and ctx["gates"] == ["confidence_floor"]
    assert ctx["positionSide"] == "short" and ctx["marketPriceAtSignal"] == 1.02
    assert ctx["setupFamily"] == "continuation" and ctx["model"] == "gpt-6-luna" and ctx["confidence"] == 0.72
    # Only the bias/flag fields travel: no prices, no timestamps from the gate state.
    assert ctx["regime"] == {"daily_bias": "bullish", "daily_exhausted": True, "intraday_bias_1h": "bearish"}

  def test_a_structural_or_unknown_gate_is_not_recorded(self, tmp_path):
    m = MemoryStore(str(tmp_path / "g.json"))
    assert m.record_gate_probe("SPX-USDT", "buy", 1.0, "pending_entry") is False
    assert m.record_gate_probe("SPX-USDT", "buy", 1.0, "made_up") is False
    assert m.record_gate_probe("SPX-USDT", "sideways", 1.0, "h1_align") is False
    assert m.record_gate_probe("SPX-USDT", "buy", float("nan"), "h1_align") is False
    assert m.gate_probes() == []

  def test_a_loud_gate_cannot_evict_a_quiet_ones_refusals(self, tmp_path, monkeypatch):
    import src.memory as memory_mod
    monkeypatch.setattr(memory_mod, "MAX_GATE_PROBES_PER_GATE", 3)
    m = MemoryStore(str(tmp_path / "g.json"))
    m.record_gate_probe("Q-USDT", "buy", 1.0, "correlation")
    for i in range(10):
      m.record_gate_probe(f"L{i}-USDT", "buy", 1.0, "anti_fomo")
    by_gate = {}
    for r in m.gate_probes():
      by_gate.setdefault(r["entryContext"]["gate"], []).append(r["symbol"])
    assert by_gate["correlation"] == ["Q-USDT"]
    assert by_gate["anti_fomo"] == ["L7-USDT", "L8-USDT", "L9-USDT"]      # newest kept, in order

  def test_state_rows_are_one_per_symbol_side_per_widest_horizon(self, tmp_path):
    from src.memory import GATE_STATE_WINDOW_MIN
    m = MemoryStore(str(tmp_path / "g.json"))
    now = time.time()
    assert m.record_gate_state_probes("A-USDT", 1.0, {"long": [], "short": ["h1_align"]}, now=now) == 2
    # Another analysis minutes later: both sides already have a row inside the window.
    assert m.record_gate_state_probes("A-USDT", 1.0, {"long": ["anti_fomo"], "short": []}, now=now + 600) == 0
    # A different symbol is its own key.
    assert m.record_gate_state_probes("B-USDT", 1.0, {"long": [], "short": []}, now=now + 600) == 2
    # Once the widest horizon has passed, the next reading is a new, non-overlapping row.
    later = now + GATE_STATE_WINDOW_MIN * 60 + 1
    assert m.record_gate_state_probes("A-USDT", 1.0, {"long": [], "short": []}, now=later) == 2
    assert GATE_STATE_WINDOW_MIN == 240

  def test_gate_rows_settle_on_the_futures_mark_with_a_credit_signed_per_side(self, tmp_path):
    """A state reading writes a long and a short row at the SAME instant. Their funding credits have
    opposite signs, so a credit map keyed without the side would hand one row the other's credit."""
    m = MemoryStore(str(tmp_path / "g.json"))
    m.record_gate_state_probes("ONE-USDT", 0.0040, {"long": [], "short": ["anti_fomo"]},
                               price_source="futures_mark")
    _age_gate_rows(m, 61 * 60)

    def funding(symbol, side, t0, t1):
      return 0.002 if side == "long" else -0.002

    # Futures map settles it; the spot map is ignored for a futures-based row.
    m.settle_signal_probes({"ONE-USDT": 0.0041}, spot_prices={"ONE-USDT": 0.0050}, funding_received=funding)
    by_side = {r["entryContext"]["positionSide"]: r["entryContext"]["signalProbe"] for r in m.gate_probes()}
    assert by_side["long"]["m60"] == 0.0041 and by_side["short"]["m60"] == 0.0041
    assert by_side["long"]["f60"] == 0.002 and by_side["short"]["f60"] == -0.002
    assert m.signal_probes(limit=0) == []

  def test_symbols_due_includes_gate_rows(self, tmp_path):
    m = MemoryStore(str(tmp_path / "g.json"))
    m.record_gate_probe("DUE-USDT", "buy", 1.0, "bench", price_source="futures_mark")
    _age_gate_rows(m, 6 * 60)
    assert m.symbols_due_for_settlement() == {"DUE-USDT"}

  def test_fully_settled_state_rows_fold_into_day_cells_once(self, tmp_path):
    from src.edge import gate_state_cells
    m = MemoryStore(str(tmp_path / "g.json"))
    m.record_gate_state_probes("A-USDT", 100.0, {"long": ["h1_align"], "short": []}, price_source="futures_mark")
    m.record_gate_probe("R-USDT", "buy", 1.0, "h1_align", price_source="futures_mark")
    data = m._read()
    for row in data["gate_probes"]:
      row["entryContext"]["signalProbe"] = {"m5": 101.0, "m15": None, "m60": 102.0}   # m240 pending
    m._write(data)
    assert m.fold_settled_gate_states(gate_state_cells) == 0            # not fully settled: nothing folds
    data = m._read()
    for row in data["gate_probes"]:
      row["entryContext"]["signalProbe"]["m240"] = 98.0
    m._write(data)
    assert m.fold_settled_gate_states(gate_state_cells) == 2            # both state sides; never the refusal
    left = m.gate_probes()
    assert [r["entryContext"]["gateProbe"] for r in left] == ["refusal"]
    days = m.gate_state_days()
    (day, cell), = days.items()
    assert cell["long"]["60"][0] == 1 and cell["long"]["60"][1] == pytest.approx(0.02)
    assert cell["long"]["60"][2] == {"h1_align": [1, pytest.approx(0.02)]}
    assert cell["short"]["240"][1] == pytest.approx(0.02) and cell["short"]["240"][2] == {}
    assert "15" not in cell["long"]                                      # a missed horizon adds nothing
    # Stored compactly (one string per day — the file is written with indent=2).
    assert isinstance(m._read()["gate_state_days"][day], str)
    assert m.fold_settled_gate_states(gate_state_cells) == 0            # counted exactly once

  def test_folding_adds_to_an_existing_day(self, tmp_path):
    from src.edge import gate_state_cells
    m = MemoryStore(str(tmp_path / "g.json"))
    now = time.time()
    for k, sym in enumerate(("A-USDT", "B-USDT")):
      m.record_gate_state_probes(sym, 100.0, {"long": ["anti_fomo"]}, now=now + k)
      data = m._read()
      for row in data["gate_probes"]:
        row["entryContext"]["signalProbe"] = {"m5": None, "m15": None, "m60": None, "m240": 101.0 + k}
      m._write(data)
      assert m.fold_settled_gate_states(gate_state_cells) == 1
    (cell,) = m.gate_state_days().values()
    assert cell["long"]["240"][0] == 2 and cell["long"]["240"][1] == pytest.approx(0.01 + 0.02)
    assert cell["long"]["240"][2]["anti_fomo"][0] == 2

  def test_a_failing_fold_function_keeps_the_rows(self, tmp_path, caplog):
    m = MemoryStore(str(tmp_path / "g.json"))
    m.record_gate_state_probes("A-USDT", 100.0, {"long": []})
    data = m._read()
    data["gate_probes"][0]["entryContext"]["signalProbe"] = {"m5": 1, "m15": 1, "m60": 1, "m240": 1}
    m._write(data)

    def boom(_rows):
      raise RuntimeError("scorer exploded")

    with caplog.at_level("WARNING"):
      assert m.fold_settled_gate_states(boom) == 0
    assert len(m.gate_probes()) == 1 and m.gate_state_days() == {}
    assert any("GATE STATE FOLD failed" in r.message for r in caplog.records)

  def test_day_cells_are_kept_by_count(self, tmp_path, monkeypatch):
    import json
    import src.memory as memory_mod
    monkeypatch.setattr(memory_mod, "MAX_GATE_STATE_DAYS", 2)
    m = MemoryStore(str(tmp_path / "g.json"))
    data = m._read()
    data["gate_state_days"] = {str(d): json.dumps({"long": {"240": [1, 0.01, {}]}}) for d in (20001, 20003, 20002)}
    data["gate_state_days"]["junk"] = "{}"
    data["gate_state_days"]["20004"] = "not json"
    m._write(data)
    m._prune(data)
    assert sorted(data["gate_state_days"]) == ["20002", "20003"]

  def test_an_older_store_is_not_rewritten_just_to_add_gate_keys(self, tmp_path):
    import json
    m = MemoryStore(str(tmp_path / "g.json"))
    m.record_signal_probe("ONE-USDT", "buy", 1.0, "continuation")
    raw = json.loads((tmp_path / "g.json").read_text())
    assert "gate_probes" not in raw and "gate_state_days" not in raw


def test_a_direction_call_records_which_gates_it_met_and_the_hatch_that_admitted_it(tmp_path):
  m = MemoryStore(str(tmp_path / "g.json"))
  m.record_signal_probe("A-USDT", "buy", 1.0, "breakout",
                        gates_passed=[{"gate": "h1_align", "hatch": "declared"}, {"gate": "made_up", "hatch": "x"}, "junk"])
  m.record_signal_probe("B-USDT", "buy", 1.0, "continuation", gates_passed=[])
  m.record_signal_probe("C-USDT", "buy", 1.0, "continuation")
  by_sym = {r["symbol"]: r["entryContext"] for r in m.signal_probes(limit=0)}
  assert by_sym["A-USDT"]["gatesPassed"] == [{"gate": "h1_align", "hatch": "declared"}]
  assert by_sym["B-USDT"]["gatesPassed"] == []          # faced none: a real baseline row
  assert "gatesPassed" not in by_sym["C-USDT"]          # not stamped: never counted as 'faced none'


class TestSignalProbeRepeatGap:
  """record_signal_probe(min_gap_sec=...) skips a repeat of the same (symbol, side, family) call made
  less than the gap ago — it could never become a new de-overlapped observation, but it would cost a
  retention slot (2026-09-26)."""

  def _store(self, tmp_path):
    from src.memory import MemoryStore
    return MemoryStore(str(tmp_path / "mem.json"))

  def test_a_repeat_inside_the_gap_is_skipped_and_reported(self, tmp_path, monkeypatch):
    import src.memory as mem_mod
    m = self._store(tmp_path)
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(mem_mod.time, "time", lambda: clock["t"])
    assert m.record_signal_probe("SOL-USDT", "buy", 120.8, "continuation", min_gap_sec=3600) is True
    clock["t"] += 900
    assert m.record_signal_probe("SOL-USDT", "buy", 120.9, "continuation", min_gap_sec=3600) is False
    clock["t"] += 2700                                           # exactly one gap after the first
    assert m.record_signal_probe("SOL-USDT", "buy", 121.0, "continuation", min_gap_sec=3600) is True
    assert [p["entryContext"]["marketPriceAtSignal"] for p in m.signal_probes(limit=0)] == [120.8, 121.0]

  def test_other_symbols_sides_and_families_are_not_repeats(self, tmp_path, monkeypatch):
    import src.memory as mem_mod
    m = self._store(tmp_path)
    monkeypatch.setattr(mem_mod.time, "time", lambda: 2_000_000.0)
    assert m.record_signal_probe("SOL-USDT", "buy", 120.8, "continuation", min_gap_sec=3600)
    assert m.record_signal_probe("SOL-USDT", "sell", 120.8, "continuation", min_gap_sec=3600)
    assert m.record_signal_probe("SOL-USDT", "buy", 120.8, "breakout", min_gap_sec=3600)
    assert m.record_signal_probe("ETH-USDT", "buy", 2700.0, "continuation", min_gap_sec=3600)
    # Count what is STORED (what retention sees); the reader merges rows sharing (symbol, ts).
    assert len(m._read()["signal_probes"]) == 4

  def test_zero_gap_records_every_call_as_before(self, tmp_path, monkeypatch):
    import src.memory as mem_mod
    m = self._store(tmp_path)
    monkeypatch.setattr(mem_mod.time, "time", lambda: 3_000_000.0)
    for _ in range(3):
      assert m.record_signal_probe("SOL-USDT", "buy", 120.8, "continuation") is True
    assert len(m._read()["signal_probes"]) == 3
