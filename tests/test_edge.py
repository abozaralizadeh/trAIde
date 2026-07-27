"""Tests for the adaptive edge controller (src/edge.py) and the close-dedup data hygiene."""

import pytest

from src.config import EdgeConfig
from src.edge import (
    adaptive_min_rr,
    adaptive_stop_atr_mult,
    edge_stats,
    measured_slippage_pct,
    entry_quality_stats,
    expectancy_size_factor,
    loss_streak_size_factor,
    symbol_adaptive_rr,
    symbol_bench_until,
)
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


def test_edge_stats_and_sizing_separate_profitable_shorts_from_losing_longs():
    closes = [
        _close("ETH-USDT", -0.5, i, close_type="CLOSE_LONG") for i in range(1, 7)
    ] + [
        _close("BTC-USDT", 0.4, 100 + i, close_type="CLOSE_SHORT") for i in range(1, 7)
    ]
    stats = edge_stats(closes, 30)
    assert stats["per_direction"]["long"]["net"] == -3.0
    assert stats["per_direction"]["short"]["net"] == 2.4
    cfg = _cfg(direction_min_trades=5, negative_expectancy_size_factor=0.5)
    assert expectancy_size_factor(stats, cfg, direction="long") == 0.5
    assert expectancy_size_factor(stats, cfg, direction="short") == 1.0


def test_expectancy_sizing_waits_for_evidence_and_never_sizes_up():
    stats = edge_stats([_close("ETH-USDT", -1.0, 1, close_type="CLOSE_LONG")], 30)
    cfg = _cfg(direction_min_trades=5, negative_expectancy_size_factor=0.25)
    assert expectancy_size_factor(stats, cfg, direction="long") == 1.0


def test_expectancy_sizing_prefers_realized_r_over_dollar_notional():
    closes = [
        _close("ETH-USDT", 1.0, 1, close_type="CLOSE_LONG") | {"realizedR": 1.0},
        _close("ETH-USDT", -10.0, 2, close_type="CLOSE_LONG") | {"realizedR": -0.5},
    ]
    stats = edge_stats(closes, 30)
    cfg = _cfg(direction_min_trades=2, negative_expectancy_size_factor=0.25)
    # Dollar PnL is -9, but normalized expectancy is +0.25R: size must not depend on notional.
    assert stats["per_direction"]["long"]["r_net"] == 0.5
    assert expectancy_size_factor(stats, cfg, direction="long") == 1.0

    inverse = [
        _close("BTC-USDT", 10.0, 1, close_type="CLOSE_SHORT") | {"realizedR": 0.25},
        _close("BTC-USDT", -1.0, 2, close_type="CLOSE_SHORT") | {"realizedR": -1.0},
    ]
    inverse_stats = edge_stats(inverse, 30)
    assert expectancy_size_factor(inverse_stats, cfg, direction="short") == 0.25


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
    assert symbol_adaptive_rr("ETH-USDT", stats, 1.5, _cfg(), now=202) == 2.0  # net-negative symbol → raised
    assert symbol_adaptive_rr("ADA-USDT", stats, 1.5, _cfg()) == 1.5   # winning symbol → base
    assert symbol_adaptive_rr("XRP-USDT", stats, 1.5, _cfg()) == 1.5   # no history → base


def test_symbol_rr_needs_min_trades():
    # A single bad close shouldn't raise the floor on noise (symbol_rr_min_trades=2).
    stats = edge_stats([_close("ADA-USDT", -0.5, 1)], 30)
    assert symbol_adaptive_rr("ADA-USDT", stats, 1.5, _cfg()) == 1.5


def test_symbol_rr_decays_when_that_symbols_outcomes_are_stale():
    base_ts = 1_000_000
    closes = [
        _close("ETH-USDT", -0.5, base_ts),
        _close("ETH-USDT", -0.5, base_ts + 1),
        # A fresh close on another symbol must not make ETH's own losses fresh.
        _close("ADA-USDT", 0.5, base_ts + 30 * 3600),
    ]
    stats = edge_stats(closes, 30)
    assert stats["per_symbol"]["ETH-USDT"]["last_close_ts"] == base_ts + 1
    assert symbol_adaptive_rr("ETH-USDT", stats, 1.5, _cfg(), now=base_ts + 3600) == 2.0
    assert symbol_adaptive_rr("ETH-USDT", stats, 1.5, _cfg(), now=base_ts + 30 * 3600) == 1.5


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


def test_realized_closes_includes_explicit_close_without_exchange_duplicate(tmp_path):
    mem = MemoryStore(str(tmp_path / "mem.json"), retention_days=7)
    mem.log_decision("XRP-USDT", "futures_close", 0.8, "manual risk close", pnl=-0.01)
    assert [row["action"] for row in mem.realized_closes()] == ["futures_close"]


def test_hold_pnl_rows_do_not_evict_real_close_outcomes(tmp_path):
    mem = MemoryStore(str(tmp_path / "mem.json"), retention_days=7)
    mem.log_decision("ETH-USDT", "futures_sell_triggered", 0.0, "close", pnl=1.0)
    for i in range(240):
        mem.log_decision("ETH-USDT", "hold", 0.7, f"snapshot {i}", pnl=-0.1)
    actions = [row["action"] for row in mem._read()["decisions"]]
    assert "futures_sell_triggered" in actions
    assert actions.count("hold") <= 51


def test_seen_close_ids_persist(tmp_path):
    path = str(tmp_path / "mem.json")
    mem = MemoryStore(path, retention_days=7)
    mem.record_seen_close_id("pos-123")
    mem.record_seen_close_id("pos-123")  # idempotent
    mem.record_seen_close_id("pos-456")
    # a fresh instance (= restart) still sees them
    mem2 = MemoryStore(path, retention_days=7)
    assert set(mem2.get_seen_close_ids()) == {"pos-123", "pos-456"}


def test_seen_fill_ids_persist(tmp_path):
    path = str(tmp_path / "mem.json")
    mem = MemoryStore(path, retention_days=7)
    mem.record_seen_fill_id("fill-123")
    mem.record_seen_fill_id("fill-123")
    assert MemoryStore(path, retention_days=7).get_seen_fill_ids() == ["fill-123"]


def test_open_interest_trend_uses_aged_observation(tmp_path):
    mem = MemoryStore(str(tmp_path / "mem.json"), retention_days=7)
    assert mem.observe_open_interest("ETH-USDT", 1000, price=100, now=1000)["trend"] is None
    assert mem.observe_open_interest("ETH-USDT", 1100, price=101, now=1100)["trend"] is None
    observed = mem.observe_open_interest("ETH-USDT", 1100, price=102, now=1300)
    assert observed["trend"] == "up"
    assert observed["changePct"] == 10.0
    assert observed["priceTrend"] == "up"
    assert observed["priceChangePct"] == 2.0


# ── entry_quality_stats: post-trade entry-timing feedback (decision-support, not a gate) ──


def _qclose(symbol, pnl, planned_risk, trough, peak, ext=None, realized_r=None, ts=0):
    ctx = {"plannedMaxLossUsd": planned_risk}
    if ext is not None:
        ctx["entryExtensionAtr"] = ext
    d = {"symbol": symbol, "pnl": pnl, "troughPnl": trough, "peakPnl": peak, "entryContext": ctx, "ts": ts}
    if realized_r is not None:
        d["realizedR"] = realized_r
    return d


def test_entry_quality_flags_chased_entries():
    # Two entries that each dipped ~0.8R against the fill before working → high MAE, "better entry" flagged.
    closes = [
        _qclose("ONDO-USDT", pnl=0.5, planned_risk=1.0, trough=-0.8, peak=1.2, ext=6.0, realized_r=0.5, ts=1),
        _qclose("SOL-USDT", pnl=0.4, planned_risk=1.0, trough=-0.9, peak=1.0, ext=3.0, realized_r=0.4, ts=2),
    ]
    s = entry_quality_stats(closes, lookback=30)
    assert s["n"] == 2
    assert s["avg_mae_r"] == pytest.approx(0.85)
    assert s["better_entry_rate"] == pytest.approx(1.0)      # both dipped >= 0.5R
    assert s["avg_entry_extension_atr"] == pytest.approx(4.5)
    assert s["worst_entry"]["symbol"] == "SOL-USDT"          # deepest adverse excursion


def test_entry_quality_clean_entries_have_low_mae():
    # Entries that barely went against the fill (well-timed) → low MAE, nothing flagged.
    closes = [
        _qclose("ADA-USDT", pnl=1.0, planned_risk=1.0, trough=-0.1, peak=1.5, ext=0.5, ts=1),
        _qclose("ADA-USDT", pnl=1.2, planned_risk=1.0, trough=0.0, peak=1.6, ext=-0.2, ts=2),
    ]
    s = entry_quality_stats(closes, lookback=30)
    assert s["avg_mae_r"] == pytest.approx(0.05)
    assert s["better_entry_rate"] == pytest.approx(0.0)


def test_entry_quality_skips_rows_without_risk_or_trough():
    # No planned risk or no trough → not usable; empty sample returns {n: 0}.
    assert entry_quality_stats([{"symbol": "X", "pnl": 1.0}], lookback=30) == {"n": 0}
    assert entry_quality_stats([], lookback=30) == {"n": 0}


def test_entry_quality_extension_optional():
    # Missing entryExtensionAtr on all rows → avg is None but MAE stats still compute.
    closes = [_qclose("X-USDT", pnl=0.5, planned_risk=1.0, trough=-0.3, peak=0.8, ts=1)]
    s = entry_quality_stats(closes, lookback=30)
    assert s["n"] == 1 and s["avg_entry_extension_atr"] is None and s["avg_mae_r"] == pytest.approx(0.3)


# ── Self-calibrating friction: measure slippage instead of assuming it ──────────


def _fill(planned, filled_at, filled=True):
    return {"price": planned, "fillPrice": filled_at, "filled": filled}


def test_measured_slippage_replaces_an_overstated_prior():
    """The live case: config assumed 0.10%/side while real fills deviated ~0.01%.

    That 12x overstatement round-trips to 0.32% of notional and, at the account's median
    risk/notional of 1.3%, charges every setup a phantom 0.18R — which forced ~2.7R gross targets
    that the tape never reached. With enough fills the estimate must collapse to what was measured.
    """
    fills = [_fill(100.0, 100.0 * (1 + 0.0001)) for _ in range(20)]
    out = measured_slippage_pct(fills, prior=0.001)
    assert out["source"] == "measured"
    assert out["value"] == pytest.approx(0.0001, rel=0.05)
    assert out["value"] < 0.001


def test_measured_slippage_keeps_prior_until_the_sample_is_real():
    out = measured_slippage_pct([_fill(100.0, 100.0)] * 3, prior=0.001, min_samples=8)
    assert out["source"] == "prior" and out["value"] == 0.001


def test_measured_slippage_uses_a_conservative_percentile_not_the_mean():
    # 8 tight fills (0.02%) and 2 bad ones (0.5%). The mean (0.116%) averages the tail away; the
    # upper percentile keeps it, so the estimate stays conservative rather than optimistic.
    fills = [_fill(100.0, 100.02) for _ in range(8)] + [_fill(100.0, 100.5) for _ in range(2)]
    out = measured_slippage_pct(fills, prior=0.01, min_samples=8, percentile=0.9)
    assert out["mean"] == pytest.approx(0.00116, rel=1e-3)
    assert out["value"] == pytest.approx(0.005, rel=1e-3)
    assert out["value"] > out["mean"]


def test_measured_slippage_can_rise_but_is_capped():
    # Adapts in BOTH directions — if execution degrades the estimate rises — but a data glitch
    # can't drive friction to an absurd value.
    fills = [_fill(100.0, 101.0) for _ in range(20)]     # 1% deviation, way past the prior
    out = measured_slippage_pct(fills, prior=0.001, cap_mult=3.0)
    assert out["value"] == pytest.approx(0.003)
    assert out["capped"] is True


def test_measured_slippage_ignores_unfilled_and_malformed_rows():
    fills = [_fill(100.0, 100.02, filled=False), {"price": None, "fillPrice": 1, "filled": True},
             {"price": 0, "fillPrice": 0, "filled": True}]
    assert measured_slippage_pct(fills, prior=0.001)["source"] == "prior"


def test_measured_slippage_never_claims_zero_friction():
    out = measured_slippage_pct([_fill(100.0, 100.0) for _ in range(20)], prior=0.001, floor=0.0001)
    assert out["value"] == pytest.approx(0.0001)


# ── Self-tuning stop noise floor from realized MAE ──────────────────────────────


def _mae_close(mae_r, realized_r, ts=1000, risk=1.0):
    """A close whose winners' adverse heat drives the floor (MAE = |troughPnl| / planned risk)."""
    return {
        "symbol": "X-USDT", "ts": ts, "pnl": realized_r * risk, "realizedR": realized_r,
        "troughPnl": -mae_r * risk, "peakPnl": max(0.0, realized_r) * risk,
        "entryContext": {"plannedMaxLossUsd": risk},
    }


def test_stop_floor_widens_when_winners_eat_heavy_adverse_heat():
    # Winners routinely dipping 0.8R before working means the stop sits inside the working range.
    closes = [_mae_close(0.8, 1.5, ts=i) for i in range(12)]
    out = adaptive_stop_atr_mult(closes, base_mult=2.5, min_samples=10, step=0.5)
    assert out["value"] == pytest.approx(3.0)
    assert out["source"] == "measured"


def test_stop_floor_never_tightens_on_low_winner_heat():
    """Low heat among winners is survivorship, not slack — the learner must NOT read it as permission.

    On the live account winners averaged 0.17R of adverse heat precisely *because* the 1.4x-ATR stop
    had already eliminated everything that breathed. A symmetric rule would have tightened the floor
    and made the original failure worse, so adaptation is widen-only and low heat holds the floor.
    """
    closes = [_mae_close(0.1, 1.5, ts=i) for i in range(12)]
    out = adaptive_stop_atr_mult(closes, base_mult=2.5, min_samples=10, step=0.5)
    assert out["value"] == pytest.approx(2.5)
    assert out["source"] == "base"


def test_stop_floor_holds_in_the_healthy_band():
    closes = [_mae_close(0.45, 1.5, ts=i) for i in range(12)]
    assert adaptive_stop_atr_mult(closes, base_mult=2.5, min_samples=10)["value"] == pytest.approx(2.5)


def test_stop_floor_ignores_losers_when_measuring_heat():
    # Losers end at ~1R of adverse excursion by construction; only WINNERS carry information about how
    # much room a working trade needs, so a pile of full stop-outs must not itself widen the floor.
    closes = [_mae_close(1.0, -1.0, ts=i) for i in range(30)] + [_mae_close(0.1, 1.5, ts=100 + i) for i in range(12)]
    out = adaptive_stop_atr_mult(closes, base_mult=2.5, min_samples=10, step=0.5)
    assert out["value"] == pytest.approx(2.5)     # driven by the winners' 0.1R, not the losers' 1.0R


def test_stop_floor_falls_back_to_base_without_a_sample():
    assert adaptive_stop_atr_mult([], base_mult=2.5)["value"] == pytest.approx(2.5)
    assert adaptive_stop_atr_mult([_mae_close(0.9, 1.0)], base_mult=2.5, min_samples=10)["value"] == pytest.approx(2.5)


def test_stop_floor_disabled_when_base_is_zero():
    assert adaptive_stop_atr_mult([_mae_close(0.9, 1.0, ts=i) for i in range(12)], base_mult=0.0)["source"] == "disabled"


def test_stop_floor_moves_by_at_most_one_step_and_respects_max():
    closes = [_mae_close(2.0, 1.5, ts=i) for i in range(12)]
    out = adaptive_stop_atr_mult(closes, base_mult=3.8, min_samples=10, step=0.5, max_mult=4.0)
    assert out["value"] == pytest.approx(4.0)     # clamped, not 4.3


def test_entry_quality_reports_target_reachability():
    """The check the bot was missing: is the planned target inside the distribution price delivers?

    Live sample: median MFE 0.27R with brackets planned at 2.3-2.7R gross, and 0% of trades ever
    reached 2R — so every take-profit was unreachable by construction and the ratio was held up by
    dragging the stop inward instead. Surfacing the buckets lets the model plan against the real tape.
    """
    closes = (
        [_mae_close(0.2, 0.3, ts=i) | {"peakPnl": 0.3} for i in range(6)]      # peaked +0.3R
        + [_mae_close(0.2, 1.2, ts=10 + i) | {"peakPnl": 1.2} for i in range(4)]  # peaked +1.2R
    )
    q = entry_quality_stats(closes, lookback=30)
    assert q["n"] == 10
    assert q["mfe_reached_rate"]["0.5R"] == pytest.approx(0.4)
    assert q["mfe_reached_rate"]["1R"] == pytest.approx(0.4)
    assert q["mfe_reached_rate"]["2R"] == pytest.approx(0.0)
    assert q["median_mfe_r"] == pytest.approx(0.3)   # 6 of 10 peaked at 0.3R


def test_entry_quality_reachability_empty_without_mfe_data():
    q = entry_quality_stats([{"symbol": "X", "ts": 1, "troughPnl": -0.5,
                              "entryContext": {"plannedMaxLossUsd": 1.0}}], lookback=30)
    assert q["mfe_reached_rate"] == {} and q["median_mfe_r"] is None
