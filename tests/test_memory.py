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
    limits = store.update_limits(1000.0, 8.0, scope="total")
    assert limits["drawdownPct"] == 0.0
    assert limits["kill"] is False


def test_update_limits_kill_switch(store):
    store.update_limits(1000.0, 8.0, scope="total")
    # Simulate a 10% loss
    limits = store.update_limits(900.0, 8.0, scope="total")
    assert limits["drawdownPct"] >= 8.0
    assert limits["kill"] is True


def test_reset_limits(store):
    store.update_limits(1000.0, 8.0, scope="total")
    store.update_limits(900.0, 8.0, scope="total")
    limits = store.reset_limits(900.0, scope="total")
    assert limits["kill"] is False
    assert limits["drawdownPct"] == 0.0


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
