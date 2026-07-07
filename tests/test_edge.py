"""Tests for the adaptive edge controller (src/edge.py) and the close-dedup data hygiene."""

from src.config import EdgeConfig
from src.edge import adaptive_min_rr, edge_stats, loss_streak_size_factor, symbol_adaptive_rr, symbol_bench_until
from src.memory import MemoryStore


def _cfg(**overrides) -> EdgeConfig:
    return EdgeConfig(**overrides)


def _close(sym, pnl, ts, close_type="CLOSE_SHORT"):
    return {"symbol": sym, "pnl": pnl, "ts": ts, "closeType": close_type}


# ── edge_stats ───────────────────────────────────────────────────────────────────


def test_edge_stats_basic():
    closes = [
        _close("ETH-USDT", 0.10, 100),
        _close("ETH-USDT", -0.50, 200),
        _close("SOL-USDT", 0.20, 300),
        _close("SOL-USDT", -0.40, 400),
    ]
    s = edge_stats(closes, lookback=30)
    assert s["n"] == 4 and s["wins"] == 2 and s["losses"] == 2
    assert s["win_rate"] == 0.5
    assert abs(s["net"] - (-0.60)) < 1e-9
    assert s["expectancy"] < 0
    assert s["per_symbol"]["ETH-USDT"]["losses"] == 1


def test_edge_stats_loss_streak_counts_from_latest():
    closes = [_close("ETH-USDT", 0.1, 1), _close("ETH-USDT", -0.2, 2), _close("ETH-USDT", -0.3, 3)]
    assert edge_stats(closes, 30)["loss_streak"] == 2
    # a win at the end resets the streak
    closes.append(_close("ETH-USDT", 0.05, 4))
    assert edge_stats(closes, 30)["loss_streak"] == 0


def test_edge_stats_respects_lookback_window():
    closes = [_close("ETH-USDT", -1.0, i) for i in range(10)] + [_close("ETH-USDT", 0.1, 100 + i) for i in range(5)]
    s = edge_stats(closes, lookback=5)
    assert s["n"] == 5 and s["losses"] == 0 and s["net"] > 0


def test_edge_stats_empty():
    s = edge_stats([], 30)
    assert s["n"] == 0 and s["loss_streak"] == 0 and s["per_symbol"] == {}


# ── adaptive_min_rr ──────────────────────────────────────────────────────────────


def test_rr_floor_raised_when_expectancy_negative():
    closes = [_close("ETH-USDT", 0.1, i) for i in range(6)] + [_close("ETH-USDT", -2.0, 10 + i) for i in range(2)]
    stats = edge_stats(closes, 30)
    assert stats["expectancy"] < 0 and stats["n"] == 8
    assert adaptive_min_rr(stats, 1.5, _cfg(), now=12) == 2.0  # now near the closes → fresh


def test_rr_floor_static_when_positive_or_insufficient_data():
    win_closes = [_close("ETH-USDT", 0.3, i) for i in range(10)]
    assert adaptive_min_rr(edge_stats(win_closes, 30), 1.5, _cfg(), now=12) == 1.5
    few = [_close("ETH-USDT", -1.0, i) for i in range(3)]  # n=3 < min_trades=8
    assert adaptive_min_rr(edge_stats(few, 30), 1.5, _cfg(), now=12) == 1.5


def test_rr_floor_decays_when_losses_are_stale():
    # The doom loop: negative expectancy freezes trading, no new closes arrive, floor would stay
    # raised forever. Once the last close is older than rr_stale_hours, revert to base so it can retry.
    base_ts = 1_000_000
    losing = [_close("ETH-USDT", -0.5, base_ts + i) for i in range(10)]
    stats = edge_stats(losing, 30)
    assert stats["expectancy"] < 0 and stats["last_close_ts"] == base_ts + 9
    now_fresh = base_ts + 9 + 3600            # 1h later — still fresh
    now_stale = base_ts + 9 + 30 * 3600       # 30h later — stale
    assert adaptive_min_rr(stats, 1.5, _cfg(), now=now_fresh) == 2.0   # fresh losses → raised
    assert adaptive_min_rr(stats, 1.5, _cfg(), now=now_stale) == 1.5   # stale → decays to base


def test_rr_floor_capped_and_disableable():
    losing = [_close("ETH-USDT", -1.0, i) for i in range(10)]
    stats = edge_stats(losing, 30)
    assert adaptive_min_rr(stats, 2.4, _cfg(), now=12) == 2.5  # capped at rr_cap
    assert adaptive_min_rr(stats, 1.5, _cfg(enabled=False), now=12) == 1.5
    assert adaptive_min_rr(stats, 0.0, _cfg(), now=12) == 0.0  # base 0 = feature off, stays off


# ── per-symbol adaptive RR (don't punish a fresh symbol for another's losses) ─────


def test_symbol_rr_raised_only_for_the_losing_symbol():
    # ETH bleeding, ADA fresh winner — ETH must clear a higher bar, ADA stays at base.
    closes = ([_close("ETH-USDT", -0.6, i) for i in range(4)]
              + [_close("ETH-USDT", 0.1, 100 + i) for i in range(2)]   # net still negative
              + [_close("ADA-USDT", 0.2, 200 + i) for i in range(2)])
    stats = edge_stats(closes, 30)
    assert symbol_adaptive_rr("ETH-USDT", stats, 1.5, _cfg()) == 2.0   # net-negative symbol → raised
    assert symbol_adaptive_rr("ADA-USDT", stats, 1.5, _cfg()) == 1.5   # winning symbol → base
    assert symbol_adaptive_rr("XRP-USDT", stats, 1.5, _cfg()) == 1.5   # no history → base


def test_symbol_rr_needs_min_trades():
    # A single bad close shouldn't raise the floor on noise (symbol_rr_min_trades=2).
    stats = edge_stats([_close("ADA-USDT", -0.5, 1)], 30)
    assert symbol_adaptive_rr("ADA-USDT", stats, 1.5, _cfg()) == 1.5


def test_symbol_rr_disabled_or_base_zero():
    stats = edge_stats([_close("ETH-USDT", -0.5, i) for i in range(4)], 30)
    assert symbol_adaptive_rr("ETH-USDT", stats, 1.5, _cfg(enabled=False)) == 1.5
    assert symbol_adaptive_rr("ETH-USDT", stats, 0.0, _cfg()) == 0.0


# ── symbol bench ─────────────────────────────────────────────────────────────────


def test_symbol_benched_after_repeated_losses_scales_with_severity():
    # The observed ETH pattern: wins interleaved but 3 losses in the last 5, net negative.
    closes = [
        _close("ETH-USDT", 0.099, 1000),
        _close("ETH-USDT", -0.403, 2000),
        _close("ETH-USDT", 0.126, 3000),
        _close("ETH-USDT", -0.501, 4000),
        _close("ETH-USDT", -0.694, 5000),
    ]
    # 3 losses → cooldown scales 12h × min(3, max_mult=4) = 36h
    until = symbol_bench_until(closes, _cfg(bench_cooldown_hours=12, bench_cooldown_max_mult=4))
    assert until == 5000 + 12 * 3 * 3600


def test_symbol_bench_severity_capped():
    # 5 losses but max_mult caps the multiplier at 4 → 48h, not 60h
    closes = [_close("ETH-USDT", -0.5, i * 1000) for i in range(1, 6)]
    until = symbol_bench_until(closes, _cfg(bench_lookback=5, bench_cooldown_hours=12, bench_cooldown_max_mult=4))
    assert until == 5000 + 12 * 4 * 3600


def test_symbol_not_benched_when_net_positive_or_few_losses():
    # 3 losses but big win -> net positive: not benched
    closes = [
        _close("SOL-USDT", 5.0, 1000),
        _close("SOL-USDT", -0.5, 2000),
        _close("SOL-USDT", -0.5, 3000),
        _close("SOL-USDT", -0.5, 4000),
    ]
    assert symbol_bench_until(closes, _cfg()) == 0
    # only 2 losses: not benched
    closes2 = [_close("SOL-USDT", -0.5, 1000), _close("SOL-USDT", -0.5, 2000), _close("SOL-USDT", 0.1, 3000)]
    assert symbol_bench_until(closes2, _cfg()) == 0


def test_symbol_bench_uses_only_recent_lookback():
    # ancient losses beyond the lookback don't bench a now-winning symbol
    closes = [_close("ETH-USDT", -1.0, i) for i in range(5)] + [_close("ETH-USDT", 0.2, 100 + i) for i in range(5)]
    assert symbol_bench_until(closes, _cfg(bench_lookback=5)) == 0


def test_symbol_bench_disabled():
    closes = [_close("ETH-USDT", -1.0, i) for i in range(5)]
    assert symbol_bench_until(closes, _cfg(enabled=False)) == 0


# ── loss-streak size factor ──────────────────────────────────────────────────────


def test_loss_streak_factor():
    cfg = _cfg()
    assert loss_streak_size_factor(0, cfg) == 1.0
    assert loss_streak_size_factor(1, cfg) == 1.0
    assert loss_streak_size_factor(2, cfg) == 0.5
    assert loss_streak_size_factor(5, cfg) == 0.5
    assert loss_streak_size_factor(5, _cfg(enabled=False)) == 1.0


# ── MemoryStore: realized_closes dedup + persistent seen IDs ─────────────────────


def test_realized_closes_dedupes_restart_double_record(tmp_path):
    mem = MemoryStore(str(tmp_path / "mem.json"), retention_days=7)
    # simulate the observed bug: same close recorded at 13:48 and again at 14:00 after a restart
    mem.log_decision("ETH-USDT", "futures_buy_triggered", 0.0, "TP/SL triggered (CLOSE_SHORT, ROE -2.11%)",
                     pnl=-0.5007272, close_type="CLOSE_SHORT")
    mem.log_decision("ETH-USDT", "futures_buy_triggered", 0.0, "TP/SL triggered (CLOSE_SHORT, ROE -2.11%)",
                     pnl=-0.5007272, close_type="CLOSE_SHORT")
    mem.log_decision("ETH-USDT", "hold_short", 0.5, "hold", pnl=-0.1)  # snapshot: excluded (not triggered)
    closes = mem.realized_closes()
    assert len(closes) == 1
    assert closes[0]["pnl"] == -0.5007272


def test_realized_closes_keeps_distinct_pnls(tmp_path):
    mem = MemoryStore(str(tmp_path / "mem.json"), retention_days=7)
    mem.log_decision("ETH-USDT", "futures_buy_triggered", 0.0, "x", pnl=-0.50, close_type="CLOSE_SHORT")
    mem.log_decision("ETH-USDT", "futures_buy_triggered", 0.0, "x", pnl=-0.51, close_type="CLOSE_SHORT")
    assert len(mem.realized_closes()) == 2


def test_seen_close_ids_persist(tmp_path):
    path = str(tmp_path / "mem.json")
    mem = MemoryStore(path, retention_days=7)
    mem.record_seen_close_id("pos-123")
    mem.record_seen_close_id("pos-123")  # idempotent
    mem.record_seen_close_id("pos-456")
    # a fresh instance (= restart) still sees them
    mem2 = MemoryStore(path, retention_days=7)
    assert set(mem2.get_seen_close_ids()) == {"pos-123", "pos-456"}
