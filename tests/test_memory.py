import time
import pytest
from src.memory import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(str(tmp_path / "memory.json"), retention_days=7)


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
