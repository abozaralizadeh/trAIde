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

    restarted = MemoryStore(path, retention_days=7)
    assert [event["id"] for event in restarted.get_pending_agent_events()] == ["futures:fill-1"]
    assert len(restarted.acknowledge_agent_events(["futures:fill-1"])) == 1
    assert restarted.get_pending_agent_events() == []


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
    assert mem.trades_today("ETH-USDT") == 1
    assert mem.positions(venue="futures") == {}
    summary = mem.performance_summary()
    assert summary["totalTrades"] == 0
    assert summary["orderSubmissions"] == 1
    assert summary["limitOrdersSubmitted"] == 1
    assert summary["limitOrdersFilled"] == 0
    assert summary["limitFillRate"] == 0.0
    assert mem.mark_order_filled("order-1") is True
    filled_summary = mem.performance_summary()
    assert filled_summary["totalTrades"] == 1
    assert filled_summary["limitOrdersFilled"] == 1
    assert filled_summary["limitFillRate"] == 1.0
    assert mem.positions(venue="futures") == {}


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
