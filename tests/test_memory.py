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


def test_a_triggered_close_is_never_treated_as_someone_elses_estimate(tmp_path):
    """Old rows predate closeType/realizedR, so they look 'evidence-free' — they must still count."""
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.log_decision("XRP-USDT", "futures_buy_triggered", 0.0, "old-style row", pnl=+0.0757)
    assert len(store.realized_closes(limit=50)) == 1
