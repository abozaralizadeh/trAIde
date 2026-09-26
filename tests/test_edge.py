"""Tests for the adaptive edge controller (src/edge.py) and the close-dedup data hygiene."""

import time

import pytest

from src.config import EdgeConfig
from src.edge import (
    adaptive_min_rr,
    adaptive_stop_atr_mult,
    edge_stats,
    measured_slippage_pct,
    signal_edge_stats,
    family_size_factor,
    family_stand_aside,
    family_explore_factor,
    open_families,
    infer_setup_family,
    entry_quality_stats,
    expectancy_size_factor,
    loss_streak_size_factor,
    symbol_adaptive_rr,
    symbol_bench_until,
    taker_flow_edge_stats,
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


# ── Signal edge: does the direction call predict, independent of exits? ─────────


_PROBE_SEQ = [0]


def _probe(side, base, fwd_1h, fwd_4h=None):
    """One INDEPENDENT observation: probes are spaced past the widest horizon so decimation keeps
    them all. Overlapping-sample collapsing is exercised separately by the _ts_probe tests below."""
    ctx = {"positionSide": side, "marketPriceAtSignal": base, "signalProbe": {"m60": fwd_1h}}
    if fwd_4h is not None:
        ctx["signalProbe"]["m240"] = fwd_4h
    _PROBE_SEQ[0] += 1
    return {"symbol": "X-USDT", "ts": 1_000_000 + _PROBE_SEQ[0] * 240 * 60, "entryContext": ctx}


def test_signal_edge_detects_a_real_edge():
    # Longs that reliably go up 1% in an hour, well past a 0.10% round-trip cost.
    probes = [_probe("long", 100.0, 101.0) for _ in range(25)]
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["verdict"] == "edge"
    assert out["by_horizon"]["60m"]["mean_pct"] == pytest.approx(1.0)
    assert out["by_horizon"]["60m"]["hit_rate"] == pytest.approx(1.0)


def test_signal_edge_calls_a_coin_flip_no_edge():
    """The live finding: 96 signals, ~0% forward return, i.e. nothing for exits to protect."""
    probes = ([_probe("long", 100.0, 100.05) for _ in range(13)]
              + [_probe("long", 100.0, 99.95) for _ in range(12)])
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["verdict"] == "no edge"
    assert abs(out["by_horizon"]["60m"]["mean_pct"]) < 0.05


def test_signal_edge_scores_shorts_by_direction():
    # A short is right when price FALLS; the sign must follow the traded direction.
    probes = [_probe("short", 100.0, 99.0) for _ in range(25)]
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["by_horizon"]["60m"]["mean_pct"] == pytest.approx(1.0)
    assert out["verdict"] == "edge"


def test_signal_edge_requires_a_real_sample_before_judging():
    probes = [_probe("long", 100.0, 101.0) for _ in range(5)]
    assert signal_edge_stats(probes, min_samples=20)["verdict"] == "insufficient data"
    assert signal_edge_stats([])["verdict"] == "insufficient data"


def test_signal_edge_needs_the_cost_hurdle_cleared_not_merely_positive():
    # +0.05% per trade is positive but does not pay a 0.10% round trip: that is not a tradeable edge.
    probes = [_probe("long", 100.0, 100.05) for _ in range(25)]
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["by_horizon"]["60m"]["mean_pct"] > 0
    assert out["by_horizon"]["60m"]["net_of_cost_pct"] < 0
    assert out["verdict"] == "no edge"


def test_signal_edge_ignores_probes_without_a_market_price_stamp():
    # Measuring from the LIMIT price scores the resting discount as prediction. Rows lacking the
    # market-price stamp must be excluded rather than silently measured from the wrong base.
    bad = [{"symbol": "X-USDT", "entryContext": {"positionSide": "long", "signalProbe": {"m60": 101.0}}}]
    assert signal_edge_stats(bad)["n"] == 0


# ── Per-family scoring: let capital follow whichever playbook actually pays ─────


def _fam_probe(side, base, fwd, family):
    """Independent observation per call — see _probe."""
    _PROBE_SEQ[0] += 1
    return {"symbol": f"{family.upper()}-USDT", "ts": 1_000_000 + _PROBE_SEQ[0] * 240 * 60,
            "entryContext": {"positionSide": side, "marketPriceAtSignal": base,
                             "setupFamily": family, "signalProbe": {"m60": fwd}}}


def test_families_are_scored_independently():
    probes = ([_fam_probe("short", 100.0, 100.05, "continuation") for _ in range(25)]
              + [_fam_probe("long", 100.0, 101.0, "fade_extreme") for _ in range(25)])
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["by_family"]["continuation"]["verdict"] == "no edge"
    assert out["by_family"]["fade_extreme"]["verdict"] == "edge"


def test_risk_follows_the_family_that_pays():
    probes = ([_fam_probe("short", 100.0, 100.05, "continuation") for _ in range(25)]
              + [_fam_probe("long", 100.0, 101.0, "fade_extreme") for _ in range(25)])
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
    # Deeply negative (-0.15% net against a 0.10% hurdle = 1.5x shortfall) collapses to the floor.
    assert family_size_factor(out, "continuation") == pytest.approx(0.25)
    assert family_size_factor(out, "fade_extreme") == pytest.approx(1.0)


def test_an_untested_family_keeps_full_risk_so_it_can_earn_its_evidence():
    # "insufficient data" is not a bad family, it is an unmeasured one. Shrinking it would prevent it
    # from ever gathering the sample that judges it.
    probes = [_fam_probe("long", 100.0, 101.0, "breakout") for _ in range(5)]
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["by_family"]["breakout"]["verdict"] == "insufficient data"
    assert family_size_factor(out, "breakout") == pytest.approx(1.0)
    assert family_size_factor(out, "never_seen") == pytest.approx(1.0)


def test_family_sizing_never_enlarges_risk():
    probes = [_fam_probe("long", 100.0, 105.0, "fade_extreme") for _ in range(25)]
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert family_size_factor(out, "fade_extreme") <= 1.0


def test_family_inferred_when_the_model_did_not_declare_one():
    aligned = {"positionSide": "short", "regime": {"intraday_bias_4h": "bearish", "intraday_bias_1h": "bearish"}}
    against = {"positionSide": "long", "regime": {"intraday_bias_4h": "bearish", "intraday_bias_1h": "bearish"}}
    assert infer_setup_family(aligned) == "continuation"
    assert infer_setup_family(against) == "fade_extreme"
    assert infer_setup_family({"setupFamily": "breakout"}) == "breakout"   # declaration always wins
    assert infer_setup_family({}) == "other"


def test_family_penalty_is_proportional_to_the_measured_shortfall():
    """No tuned constant: the cut scales with how far the family misses the cost it must clear.

    A family that is marginally short of paying its costs should not be treated like one that loses
    two round-trips per signal. Live on 2026-08-10 continuation measured -0.29% net against a 0.166%
    hurdle — a 1.75x shortfall — and collapsed to the floor.
    """
    # -0.02% net against a 0.10% hurdle = 0.2x shortfall -> keep 80% of risk.
    marginal = {"cost_pct": 0.001,
                "by_family": {"continuation": {"n": 30, "verdict": "no edge", "net_of_cost_pct": -0.02}}}
    assert family_size_factor(marginal, "continuation") == pytest.approx(0.8)
    # -0.29% against the same hurdle = 2.9x shortfall -> floored.
    severe = {"cost_pct": 0.001,
              "by_family": {"continuation": {"n": 30, "verdict": "no edge", "net_of_cost_pct": -0.29}}}
    assert family_size_factor(severe, "continuation") == pytest.approx(0.25)


def test_family_penalty_never_starves_a_family_into_a_doom_loop():
    """min_factor is deliberately non-zero.

    Size is stop-defined, so driving it to nil pushes notional under the exchange contract minimum,
    the order is rejected, no probe is recorded, and the family can never produce the evidence that
    would let it recover — the same doom loop the memory-retention fix had to undo.
    """
    awful = {"cost_pct": 0.001,
             "by_family": {"continuation": {"n": 30, "verdict": "no edge", "net_of_cost_pct": -99.0}}}
    assert family_size_factor(awful, "continuation") == pytest.approx(0.25)
    assert family_size_factor(awful, "continuation", min_factor=0.1) == pytest.approx(0.1)


def test_family_penalty_falls_back_to_the_floor_without_a_usable_hurdle():
    no_cost = {"by_family": {"continuation": {"n": 30, "verdict": "no edge", "net_of_cost_pct": -0.2}}}
    assert family_size_factor(no_cost, "continuation") == pytest.approx(0.25)


# ── Stand aside: a proven-no-edge playbook gets zero stake, i.e. is skipped ──────


def test_stand_aside_on_a_proven_no_edge_family():
    # A real sample whose direction calls don't clear cost has non-positive expectancy: Kelly-zero,
    # so decline the trade rather than stake floor-size fee-dust on it.
    out = {"cost_pct": 0.001,
           "by_family": {"continuation": {"n": 30, "verdict": "no edge", "net_of_cost_pct": -0.29}}}
    assert family_stand_aside(out, "continuation") is True


def test_stand_aside_leaves_an_unproven_family_alone():
    # "insufficient data" is unmeasured, not bad — trading it is how it earns the evidence that judges it.
    out = {"cost_pct": 0.001,
           "by_family": {"breakout": {"n": 5, "verdict": "insufficient data", "net_of_cost_pct": -0.5}}}
    assert family_stand_aside(out, "breakout") is False
    # Below the sample floor even with a "no edge" label -> not yet actionable.
    thin = {"by_family": {"continuation": {"n": 3, "verdict": "no edge", "net_of_cost_pct": -0.2}}}
    assert family_stand_aside(thin, "continuation", min_samples=20) is False


def test_stand_aside_leaves_a_paying_family_alone():
    out = {"cost_pct": 0.001,
           "by_family": {"fade_extreme": {"n": 30, "verdict": "edge", "net_of_cost_pct": 0.4}}}
    assert family_stand_aside(out, "fade_extreme") is False
    # An unmeasured family (no row at all) is never skipped.
    assert family_stand_aside(out, "never_seen") is False
    assert family_stand_aside({}, "continuation") is False


class TestOpenFamilies:
  """Where a refused setup can actually go, read off the live scoreboard.

  The stand-aside used to hand back a hardcoded suggestion ("take a genuine fade_extreme"). On
  2026-09-07 that pointed straight at a family sitting at n=38, net -1.12%, itself stood aside — so
  the model was told to do the one thing guaranteed to be refused again, and re-proposed
  continuation 21 times in six hours while range_edge sat unblocked at a measured +0.72%.
  """

  _SCOREBOARD = {"cost_pct": 0.001, "by_family": {
    "continuation": {"n": 134, "verdict": "no edge", "net_of_cost_pct": -0.356, "stderr_pct": 0.1},
    "fade_extreme": {"n": 38, "verdict": "no edge", "net_of_cost_pct": -1.122, "stderr_pct": 0.2},
    "range_edge": {"n": 22, "verdict": "edge", "net_of_cost_pct": 0.724, "stderr_pct": 0.2},
    "funding_carry": {"n": 10, "verdict": "insufficient data", "net_of_cost_pct": 0.817},
    "breakout": {"n": 7, "verdict": "insufficient data", "net_of_cost_pct": -0.081},
  }}

  def test_a_blocked_family_is_never_offered_as_the_way_out(self):
    out = open_families(self._SCOREBOARD)
    named = {r["family"] for r in out["paying"] + out["unproven"]}
    assert not any(family_stand_aside(self._SCOREBOARD, fam) for fam in named)
    assert "continuation" not in named and "fade_extreme" not in named

  def test_proven_and_unproven_are_reported_separately(self):
    """Conflating them would sell an n=7 coin flip as an edge — the distinction is the whole point
    of measuring, and the model needs to size differently against each."""
    out = open_families(self._SCOREBOARD)
    assert [r["family"] for r in out["paying"]] == ["range_edge"]
    assert [r["family"] for r in out["unproven"]] == ["funding_carry", "breakout"]  # by sample size
    assert out["paying"][0]["n"] == 22 and out["paying"][0]["netOfCostPct"] == 0.724

  def test_paying_families_are_ranked_best_first(self):
    board = {"by_family": {
      "range_edge": {"n": 22, "verdict": "edge", "net_of_cost_pct": 0.3},
      "breakout": {"n": 25, "verdict": "edge", "net_of_cost_pct": 1.1},
    }}
    assert [r["family"] for r in open_families(board)["paying"]] == ["breakout", "range_edge"]

  def test_it_says_nothing_rather_than_inventing_somewhere_to_go(self):
    """With every family blocked the honest answer is "stand down" — a suggestion pulled out of
    nowhere is how a veto turns into a nudge toward a worse trade."""
    all_bad = {"by_family": {
      "continuation": {"n": 30, "verdict": "no edge", "net_of_cost_pct": -0.5, "stderr_pct": 0.1},
      "fade_extreme": {"n": 30, "verdict": "no edge", "net_of_cost_pct": -0.4, "stderr_pct": 0.1},
    }}
    assert open_families(all_bad) == {"paying": [], "unproven": []}
    assert open_families({}) == {"paying": [], "unproven": []}
    assert open_families(None) == {"paying": [], "unproven": []}

  def test_a_family_with_no_evidence_at_all_is_not_advertised(self):
    # An n=0 row is not "open to trade", it is unmeasured — offering it reads as a recommendation.
    board = {"by_family": {"macro_event": {"n": 0, "verdict": "insufficient data"}}}
    assert open_families(board)["unproven"] == []


def test_stand_aside_and_the_probe_pipeline_agree_on_the_verdict():
    # End-to-end: the same probe stream that reads "no edge" via signal_edge_stats also trips stand-aside,
    # so the skip fires on exactly the families the scoreboard condemns — no separate threshold to drift.
    probes = ([_fam_probe("short", 100.0, 100.05, "continuation") for _ in range(25)]
              + [_fam_probe("long", 100.0, 101.0, "fade_extreme") for _ in range(25)])
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert family_stand_aside(out, "continuation") is True
    assert family_stand_aside(out, "fade_extreme") is False


def test_explore_factor_shrinks_an_unproven_family():
    # A family with fewer than min_samples probes has no verdict yet — it trades at explore-size, not
    # full risk, because its evidence records from the market price at call time regardless of our size.
    out = {"by_family": {"breakout": {"n": 5, "verdict": "insufficient data", "net_of_cost_pct": -0.1}}}
    assert family_explore_factor(out, "breakout", explore_factor=0.4) == pytest.approx(0.4)
    # A family never seen at all is also unproven → explore-size.
    assert family_explore_factor({}, "range_edge", explore_factor=0.4) == pytest.approx(0.4)
    assert family_explore_factor({"by_family": {}}, "range_edge", explore_factor=0.25) == pytest.approx(0.25)


def test_explore_factor_lifts_to_full_once_a_family_is_scored():
    # The instant a family crosses min_samples it is scored, so explore stops throttling it and hands
    # sizing back to family_size_factor / family_stand_aside (whichever the verdict warrants).
    scored = {"by_family": {"continuation": {"n": 30, "verdict": "no edge", "net_of_cost_pct": -0.29}}}
    assert family_explore_factor(scored, "continuation") == 1.0
    paying = {"by_family": {"fade_extreme": {"n": 44, "verdict": "edge", "net_of_cost_pct": 0.4}}}
    assert family_explore_factor(paying, "fade_extreme") == 1.0


def test_explore_and_measured_factors_compose_by_the_worst():
    # The two family factors combine by the WORSE of the two (as tools.py does), never their product —
    # so an unproven family is explore-sized (0.4) while family_size_factor still reads its no-op 1.0,
    # and a proven no-edge family collapses via family_size_factor while explore reads its no-op 1.0.
    unproven = {"by_family": {"breakout": {"n": 5, "verdict": "insufficient data"}}}
    combined = min(family_size_factor(unproven, "breakout"),
                   family_explore_factor(unproven, "breakout", explore_factor=0.4))
    assert combined == pytest.approx(0.4)
    proven = {"cost_pct": 0.001,
              "by_family": {"continuation": {"n": 30, "verdict": "no edge", "net_of_cost_pct": -0.29}}}
    combined2 = min(family_size_factor(proven, "continuation"),
                    family_explore_factor(proven, "continuation", explore_factor=0.4))
    assert combined2 < 0.4  # the measured shortfall, not the explore floor, is what binds here


def _ts_probe(sym, ts, side, base, fwd, family="continuation"):
    return {"symbol": sym, "ts": ts,
            "entryContext": {"positionSide": side, "marketPriceAtSignal": base,
                             "setupFamily": family, "signalProbe": {"m60": fwd}}}


def test_overlapping_probes_on_one_symbol_count_once_per_window():
    """Thirty probes on one symbol inside an hour are ~one observation, not thirty.

    Probes are recorded minutes apart, so their forward windows overlap almost entirely. Counting them
    independently inflates the sample and the verdict with it. On 2026-08-11 that produced a FALSE
    POSITIVE on live data: 240m read +0.224% (t=+3.66, n=131) and the verdict flipped to "edge", but
    one-per-symbol-per-window gave +0.068% (t=+0.51, n=31) — below the cost hurdle, i.e. nothing.
    Since this verdict governs how much capital each family gets, an inflated sample can size the bot
    UP on noise, which is the most expensive mistake this module could make.
    """
    # 30 probes 60s apart on ONE symbol, all inside a single 60m window.
    probes = [_ts_probe("XRP-USDT", 1_000_000 + i * 60, "long", 100.0, 102.0) for i in range(30)]
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=1)
    assert out["by_horizon"]["60m"]["n"] == 1, "overlapping probes must collapse to one observation"


def test_probes_spaced_beyond_the_window_all_count():
    probes = [_ts_probe("XRP-USDT", 1_000_000 + i * 3600, "long", 100.0, 102.0) for i in range(5)]
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=1)
    assert out["by_horizon"]["60m"]["n"] == 5


def test_different_symbols_in_the_same_window_are_independent():
    # Two symbols moving at the same time really are two observations.
    probes = [_ts_probe("XRP-USDT", 1_000_000, "long", 100.0, 102.0),
              _ts_probe("ADA-USDT", 1_000_010, "long", 100.0, 102.0)]
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=1)
    assert out["by_horizon"]["60m"]["n"] == 2


def test_decimation_cannot_manufacture_an_edge_verdict_from_repetition():
    """A single lucky move, sampled 50 times, must not clear the cost hurdle."""
    probes = [_ts_probe("XRP-USDT", 1_000_000 + i * 30, "long", 100.0, 105.0) for i in range(50)]
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["by_horizon"]["60m"]["n"] == 1
    assert out["verdict"] == "insufficient data"   # one observation is not evidence


# --- stand-aside release bar: stateless t = net/SE >= 1 (NOT hysteresis — nothing is remembered) ----

def _fam_edge(n, net_pct, se_pct, verdict="no edge", cost_pct=0.0014):
  return {"cost_pct": cost_pct,
          "by_family": {"continuation": {"n": n, "net_of_cost_pct": net_pct,
                                         "stderr_pct": se_pct, "verdict": verdict}}}


def test_stand_aside_does_not_release_on_a_within_noise_blip():
  """2026-09-04: continuation sat at net -0.03% with an SE of ~0.30% over 130 samples. The verdict is
  a sign test on a noisy mean, so it flipped between polls on the same evidence — and a flip to "edge"
  restored FULL size (family x1.00) to the playbook with the longest adverse record. A WIF long went
  in on that one poll and lost a full 1R. Staking needs the net to clear its own standard error
  (t >= 1, recomputed every run) — a stateless bar, not the hysteresis this docstring once claimed."""
  from src.edge import family_stand_aside
  # The exact shape that let WIF through: verdict flipped positive, but net is far inside one SE.
  assert family_stand_aside(_fam_edge(130, +0.02, 0.30, verdict="edge"), "continuation") is True
  # A genuine, decisive improvement releases it.
  assert family_stand_aside(_fam_edge(130, +0.45, 0.30, verdict="edge"), "continuation") is False
  # A settled "no edge" still stands aside regardless of the band.
  assert family_stand_aside(_fam_edge(130, -0.28, 0.30), "continuation") is True


def test_the_band_tightens_as_evidence_accumulates():
  """The gate must not become permanent: SE shrinks with n, so a family that really starts paying
  escapes on its own. Same mean, more evidence -> released."""
  from src.edge import family_stand_aside
  assert family_stand_aside(_fam_edge(25, +0.20, 0.60, verdict="edge"), "continuation") is True
  assert family_stand_aside(_fam_edge(400, +0.20, 0.08, verdict="edge"), "continuation") is False


def test_unproven_and_missing_families_are_untouched():
  from src.edge import family_stand_aside
  # Below min_samples nothing stands aside — an unproven playbook must be free to gather evidence.
  assert family_stand_aside(_fam_edge(5, -2.0, 0.1), "continuation") is False
  assert family_stand_aside({"by_family": {}}, "continuation") is False
  assert family_stand_aside({}, "continuation") is False
  # A row with no dispersion recorded (older payload) must not start blocking on missing data.
  assert family_stand_aside(
    {"by_family": {"continuation": {"n": 100, "net_of_cost_pct": +0.02, "verdict": "edge"}}},
    "continuation") is False


def test_stderr_is_reported_per_family():
  """Callers cannot tell a real shortfall from a wobble without the sample's own dispersion."""
  from src.edge import signal_edge_stats
  import time as _t
  now = int(_t.time()) - 10 * 3600
  probes = []
  for i in range(30):
    probes.append({
      "symbol": f"S{i}-USDT", "ts": now + i * 7200,
      "entryContext": {"positionSide": "long", "marketPriceAtSignal": 100.0,
                       "setupFamily": "continuation", "signalProbe": {"m60": 100.0 + (i % 3) - 1}},
    })
  s = signal_edge_stats(probes, cost_pct=0.0014)
  row = s["by_family"]["continuation"]
  assert row["n"] == 30
  assert row["stderr_pct"] > 0
  # Mean of a symmetric +/-1 spread is ~0, so the SE must dominate the net -> stand aside.
  from src.edge import family_stand_aside
  assert family_stand_aside(s, "continuation") is True


# ── Taker flow: does the aggressor balance at signal time separate the good calls? ──
#
# Measurement only. The prior is that it does NOT at our horizons — order-flow imbalance is largely
# contemporaneous and decays inside a minute, and KuCoin is not where these prices are set — so these
# tests care most about the statistic REFUSING to claim an edge it cannot support.


_FLOW_SEQ = [0]


def _flow_probe(side, base, fwd, buy_share, *, horizon="m60"):
    """One independent flow-stamped observation, spaced past the widest horizon (see _probe)."""
    _FLOW_SEQ[0] += 1
    ctx = {"positionSide": side, "marketPriceAtSignal": base, "signalProbe": {horizon: fwd}}
    if buy_share is not None:
        ctx["takerFlow"] = {"buyShare": buy_share, "trades": 100, "spanSec": 180.0}
    return {"symbol": "F-USDT", "ts": 2_000_000 + _FLOW_SEQ[0] * 240 * 60, "entryContext": ctx}


def test_flow_that_separates_winners_but_not_costs_is_informative_not_tradable():
    """The expected outcome: a real spread that a 0.10% round trip still eats. Reporting this as
    'tradable' is exactly the mistake that turns a microstructure paper into a losing strategy."""
    probes = ([_flow_probe("long", 100.0, 100.05, 0.80) for _ in range(25)]
              + [_flow_probe("long", 100.0, 99.95, 0.20) for _ in range(25)])
    out = taker_flow_edge_stats(probes, cost_pct=0.001, min_samples=20)
    row = out["by_horizon"]["60m"]
    assert row["spread_pct"] == pytest.approx(0.1)      # with-flow beat against-flow by 10bps
    assert row["verdict"] == "informative"              # ...which does not pay a 10bps round trip
    assert out["verdict"] == "informative"


def test_flow_is_called_tradable_only_when_the_with_group_clears_cost_on_its_own():
    probes = ([_flow_probe("long", 100.0, 100.5, 0.80) for _ in range(25)]
              + [_flow_probe("long", 100.0, 99.5, 0.20) for _ in range(25)])
    out = taker_flow_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["by_horizon"]["60m"]["verdict"] == "tradable"
    assert out["verdict"] == "tradable"


def test_a_house_wide_drift_is_not_mistaken_for_a_flow_edge():
    """Every call made +0.5% regardless of the tape. A raw with-flow mean would look excellent; the
    spread form cancels the book's own directional bias and correctly reports nothing."""
    probes = ([_flow_probe("long", 100.0, 100.5, 0.80) for _ in range(25)]
              + [_flow_probe("long", 100.0, 100.5, 0.20) for _ in range(25)])
    out = taker_flow_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["by_horizon"]["60m"]["with"]["mean_pct"] == pytest.approx(0.5)
    assert out["by_horizon"]["60m"]["spread_pct"] == pytest.approx(0.0)
    assert out["verdict"] == "no information"


def test_a_spread_inside_its_own_standard_error_is_not_information():
    probes = []
    for i in range(25):
        probes.append(_flow_probe("long", 100.0, 100.0 + (i % 5) - 2, 0.80))
        probes.append(_flow_probe("long", 100.0, 100.0 + (i % 5) - 2.02, 0.20))
    out = taker_flow_edge_stats(probes, cost_pct=0.001, min_samples=20)
    row = out["by_horizon"]["60m"]
    assert row["spread_pct"] < row["spread_stderr_pct"]
    assert row["verdict"] == "no information"


def test_agreement_is_measured_against_the_traded_direction():
    """Sellers hitting the bid are WITH a short. Scoring the tape without the position's side would
    file every short taken into selling pressure as a trade against the flow."""
    probes = ([_flow_probe("short", 100.0, 99.5, 0.20) for _ in range(25)]     # sell flow + short
              + [_flow_probe("short", 100.0, 100.5, 0.80) for _ in range(25)])  # buy flow + short
    out = taker_flow_edge_stats(probes, cost_pct=0.001, min_samples=20)
    row = out["by_horizon"]["60m"]
    assert row["with"]["n"] == 25 and row["against"]["n"] == 25
    assert row["with"]["mean_pct"] == pytest.approx(0.5)    # short + price fell = a win
    assert row["verdict"] == "tradable"


def test_a_balanced_tape_lands_in_neutral_rather_than_padding_a_side():
    """0.52 is not a buy imbalance. Without a dead band, coin-flip readings would fill both groups
    with noise and dilute whatever signal the decisive readings carry."""
    probes = ([_flow_probe("long", 100.0, 101.0, 0.52) for _ in range(25)]
              + [_flow_probe("long", 100.0, 99.0, 0.48) for _ in range(25)])
    out = taker_flow_edge_stats(probes, cost_pct=0.001, min_samples=20, neutral_band=0.05)
    row = out["by_horizon"]["60m"]
    assert row["neutral"]["n"] == 50
    assert "with" not in row and "against" not in row
    assert row["verdict"] == "insufficient data"


def test_coverage_reports_how_much_of_the_book_the_verdict_speaks_for():
    """A strong spread over a tenth of the calls is a claim about a tenth of the calls."""
    probes = ([_flow_probe("long", 100.0, 100.5, 0.80) for _ in range(25)]
              + [_flow_probe("long", 100.0, 99.5, 0.20) for _ in range(25)]
              + [_flow_probe("long", 100.0, 101.0, None) for _ in range(50)])
    out = taker_flow_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["coverage"] == pytest.approx(0.5)
    assert out["by_horizon"]["60m"]["with"]["n"] == 25      # unstamped rows never enter a bucket


def test_unusable_readings_are_excluded_rather_than_scored_as_neutral():
    for flow in ({"buyShare": None}, {"buyShare": 1.4}, {"buyShare": -0.1}, {}, "not-a-dict"):
        probe = _flow_probe("long", 100.0, 101.0, None)
        probe["entryContext"]["takerFlow"] = flow
        out = taker_flow_edge_stats([probe] * 25, cost_pct=0.001, min_samples=1)
        assert out["n"] == 0 and out["verdict"] == "insufficient data"
        assert out["coverage"] == 0.0


def test_both_groups_must_have_a_real_sample_before_a_verdict():
    probes = ([_flow_probe("long", 100.0, 100.5, 0.80) for _ in range(25)]
              + [_flow_probe("long", 100.0, 99.5, 0.20) for _ in range(3)])
    out = taker_flow_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["by_horizon"]["60m"]["verdict"] == "insufficient data"
    assert out["verdict"] == "insufficient data"
    assert taker_flow_edge_stats([])["verdict"] == "insufficient data"
    assert taker_flow_edge_stats([])["coverage"] is None


def test_short_horizons_are_scored_alongside_the_established_ones():
    """The 5m and 15m windows exist precisely because that is where flow information is documented
    to live; a statistic that only looked at 60m could not see the effect it is testing for."""
    probes = ([_flow_probe("long", 100.0, 100.5, 0.80, horizon="m5") for _ in range(25)]
              + [_flow_probe("long", 100.0, 99.5, 0.20, horizon="m5") for _ in range(25)])
    out = taker_flow_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["by_horizon"]["5m"]["verdict"] == "tradable"


# --- per-family scoring horizons -----------------------------------------------------------------

def _hclose(family, hold_min, ts=10_000_000):
  return {"ts": ts, "entryContext": {"setupFamily": family, "fillTs": ts - hold_min * 60}}


def test_each_family_is_scored_at_its_own_holding_period():
  """Live 2026-09-18: continuation is held a median 162m, fade_extreme 12m, funding_carry 116m.
  Snapped on a LOG scale (ratios matter for time): 162m -> 240m, 12m -> 15m, 116m -> 60m."""
  from src.edge import family_scoring_horizons
  closes = ([_hclose("continuation", 162)] * 8 + [_hclose("fade_extreme", 12)] * 8
            + [_hclose("funding_carry", 116)] * 8)
  h = family_scoring_horizons(closes)
  assert h == {"continuation": 240, "fade_extreme": 15, "funding_carry": 60}


def test_a_family_without_enough_trades_falls_back_to_the_default():
  """breakout had 2 realized closes — no trustworthy median, so it must not get a horizon at all."""
  from src.edge import family_scoring_horizons
  h = family_scoring_horizons([_hclose("breakout", 180)] * 2 + [_hclose("continuation", 162)] * 8)
  assert "breakout" not in h and h["continuation"] == 240


def test_junk_closes_are_ignored_without_raising():
  from src.edge import family_scoring_horizons
  assert family_scoring_horizons(None) == {}
  assert family_scoring_horizons(["x", {}, {"ts": "bad", "entryContext": {}}]) == {}
  assert family_scoring_horizons([_hclose("continuation", -5)] * 8) == {}   # negative hold: invalid


def _hprobe(family, ts, m15, m60, m240, side="long"):
  return {"symbol": f"{family[:3].upper()}{ts}-USDT", "ts": ts,
          "entryContext": {"positionSide": side, "marketPriceAtSignal": 100.0, "setupFamily": family,
                           "signalProbe": {"m15": m15, "m60": m60, "m240": m240}}}


def test_the_live_failure_a_trend_playbook_benched_on_the_wrong_horizon():
  """The exact shape of 2026-09-18: at 60m a trending entry sits mid-pullback (≈ flat), at its real
  240m holding period it is clearly profitable. Scored at 60m it stands aside; at 240m it trades."""
  from src.edge import signal_edge_stats, family_stand_aside
  probes = [_hprobe("continuation", 1_000_000 + i * 90_000, 100.0, 100.0 + ((-1) ** i) * 0.05,
                   101.5) for i in range(30)]
  at60 = signal_edge_stats(probes, cost_pct=0.0014)
  at_own = signal_edge_stats(probes, cost_pct=0.0014, family_horizons={"continuation": 240})
  assert family_stand_aside(at60, "continuation") is True
  assert family_stand_aside(at_own, "continuation") is False


def test_a_short_hold_family_is_not_released_by_a_long_horizon():
  """Why the fix is per-family, not a blanket 240m: fade_extreme is held ~12m. At 240m its calls
  look fine (the move eventually comes); at its own 15m horizon they lose. It must stay benched."""
  from src.edge import signal_edge_stats, family_stand_aside
  probes = [_hprobe("fade_extreme", 1_000_000 + i * 90_000, 99.6, 100.0, 101.0) for i in range(30)]
  blanket = signal_edge_stats(probes, cost_pct=0.0014, family_horizon_min=240)
  own = signal_edge_stats(probes, cost_pct=0.0014, family_horizons={"fade_extreme": 15})
  assert family_stand_aside(blanket, "fade_extreme") is False, "a blanket 240m would wrongly release it"
  assert family_stand_aside(own, "fade_extreme") is True


def test_a_broken_store_degrades_to_the_default_horizon_not_a_blank_verdict():
  """Hold-time derivation is a refinement: if it fails, the edge report must still be produced."""
  from src.edge import safe_family_horizons
  class _Broken:
    def realized_closes(self, **k): raise RuntimeError("disk gone")
  assert safe_family_horizons(_Broken()) == {}


def test_horizons_are_snapped_on_a_log_scale_not_a_linear_one():
  """For a holding period what matters is the RATIO. A 35-minute hold is 2.3x a 15m horizon but only
  1.7x short of a 60m one, so 60m is the proportionally nearer match — a linear snap would pick 15m
  and score the family at less than half its real holding period. (For the live values 162/12/116m
  both scales happen to agree, which is exactly why this needs its own test.)"""
  from src.edge import family_scoring_horizons
  assert family_scoring_horizons([_hclose("x", 35)] * 8) == {"x": 60}
  assert family_scoring_horizons([_hclose("x", 130)] * 8) == {"x": 240}   # linear would say 60


def test_the_horizon_follows_the_market_not_the_average_of_every_regime():
  """Holding time drifts with conditions: funding_carry went 90m in chop -> 171m in the Sep 2026 rally,
  which moves it from the 60m horizon to 240m. The horizon must follow the RECENT regime, not be
  outvoted by however many old closes came before the market changed."""
  from src.edge import family_scoring_horizons
  old = [_hclose("funding_carry", 90, ts=1_000_000 + i) for i in range(40)]       # long chop history
  new = [_hclose("funding_carry", 171, ts=5_000_000 + i) for i in range(20)]      # regime changed
  assert family_scoring_horizons(old) == {"funding_carry": 60}
  assert family_scoring_horizons(old + new) == {"funding_carry": 240}, \
    "40 stale chop closes must not outvote the 20 that describe the market now"


def test_recency_is_by_close_time_not_by_list_order():
  """The store is not guaranteed chronological; recency must come from the timestamps."""
  from src.edge import family_scoring_horizons
  old = [_hclose("x", 90, ts=1_000_000 + i) for i in range(40)]
  new = [_hclose("x", 171, ts=5_000_000 + i) for i in range(20)]
  assert family_scoring_horizons(new + old) == {"x": 240}


# --- graduation ramps with evidence instead of jumping --------------------------------------------

def _grow(n, net, se):
  return {"by_family": {"x": {"n": n, "net_of_cost_pct": net, "stderr_pct": se}}}


def test_graduation_is_continuous_not_a_cliff():
  """The live failure: funding_carry reached n=24 at t=1.18 (net +3.78%, SE 3.20%) and its risk jumped
  0.40x -> 1.00x in one step; the first full-size trade was a -1.04R short into a vertical spike at 2.2x
  the size of the same trade an hour before. Just past graduation, size must sit at the explore floor."""
  from src.edge import family_explore_factor
  below = family_explore_factor(_grow(19, 3.78, 3.20), "x")
  just_over = family_explore_factor(_grow(20, 1.01, 1.00), "x")
  assert below == pytest.approx(0.4)
  assert just_over == pytest.approx(0.4, abs=0.01), "no step at n=20 when the edge only just clears"
  assert family_explore_factor(_grow(24, 3.78, 3.20), "x") == pytest.approx(0.509, abs=0.01)


def test_size_grows_with_the_strength_of_the_evidence():
  from src.edge import family_explore_factor
  ts = [1.0, 1.25, 1.5, 1.75, 2.0]
  sizes = [family_explore_factor(_grow(30, t, 1.0), "x") for t in ts]
  assert sizes == sorted(sizes), "more proof must never mean less size"
  assert sizes[0] == pytest.approx(0.4) and sizes[-1] == pytest.approx(1.0)
  assert family_explore_factor(_grow(30, 9.0, 1.0), "x") == pytest.approx(1.0)   # capped at full


def test_the_ramp_follows_the_market_because_it_reads_live_evidence():
  """No regime constant: the same family shrinks as its measured edge weakens and grows as it
  strengthens, purely from its own net and SE, which are recomputed every run from live probes."""
  from src.edge import family_explore_factor
  strong = family_explore_factor(_grow(40, 2.4, 1.0), "x")
  weakening = family_explore_factor(_grow(40, 1.3, 1.0), "x")
  assert strong > weakening


def test_a_row_without_dispersion_keeps_the_previous_full_size_behaviour():
  from src.edge import family_explore_factor
  assert family_explore_factor({"by_family": {"x": {"n": 30, "net_of_cost_pct": 1.0}}}, "x") == 1.0


# ── Funding credit: a carry call is scored on price AND the transfer it was paid ─────────


def _funded_probe(side, base, fwd_1h, credit=None, *, buy_share=None):
    """Independent observation (see _probe) carrying an optional f60 funding credit."""
    _PROBE_SEQ[0] += 1
    ctx = {"positionSide": side, "marketPriceAtSignal": base, "setupFamily": "funding_carry",
           "signalProbe": {"m60": fwd_1h}}
    if credit is not None:
        ctx["signalProbe"]["f60"] = credit
    if buy_share is not None:
        ctx["takerFlow"] = {"buyShare": buy_share, "trades": 100, "spanSec": 180.0}
    return {"symbol": "C-USDT", "ts": 5_000_000 + _PROBE_SEQ[0] * 240 * 60, "entryContext": ctx}


def test_funding_credit_is_added_to_the_signed_return():
    """Flat price, 0.20% of funding received: the carry call made 0.20%, not zero."""
    paid = [_funded_probe("short", 100.0, 100.0, 0.002) for _ in range(25)]
    out = signal_edge_stats(paid, cost_pct=0.001, min_samples=20)
    assert out["by_horizon"]["60m"]["mean_pct"] == pytest.approx(0.2)
    assert out["by_family"]["funding_carry"]["net_of_cost_pct"] == pytest.approx(0.1)
    # The credit is already signed for the probe's side — a long that PAID funding is charged.
    charged = [_funded_probe("long", 100.0, 101.0, -0.003) for _ in range(25)]
    assert signal_edge_stats(charged, cost_pct=0.001, min_samples=20)["by_horizon"]["60m"]["mean_pct"] \
        == pytest.approx(0.7)


def test_a_probe_without_a_funding_stamp_scores_price_only():
    legacy = [_funded_probe("long", 100.0, 101.0) for _ in range(25)]
    garbage = [_funded_probe("long", 100.0, 101.0, "n/a") for _ in range(25)]
    for probes in (legacy, garbage):
        out = signal_edge_stats(probes, cost_pct=0.001, min_samples=20)
        assert out["by_horizon"]["60m"]["mean_pct"] == pytest.approx(1.0)


def test_taker_flow_statistic_sees_the_same_funded_return():
    """One shared observation stream: the flow spread must not silently score price-only."""
    probes = ([_funded_probe("long", 100.0, 100.0, 0.004, buy_share=0.80) for _ in range(25)]
              + [_funded_probe("long", 100.0, 100.0, 0.0, buy_share=0.20) for _ in range(25)])
    out = taker_flow_edge_stats(probes, cost_pct=0.001, min_samples=20)
    assert out["by_horizon"]["60m"]["with"]["mean_pct"] == pytest.approx(0.4)
    assert out["by_horizon"]["60m"]["against"]["mean_pct"] == pytest.approx(0.0)


# ── Truthful stand-aside + the family split by SIDE (2026-09-25) ──────────────────────────────────────

from src.edge import (  # noqa: E402  (grouped with the tests that use them)
  annotate_family_stakes,
  confidence_edge_for_prompt,
  confidence_edge_stats,
  describe_stake_row,
  family_evidence_row,
  family_stake,
  family_stake_status,
)


class TestFamilyStakeStatus:
  """Why a family is at zero stake — 'no edge' and 'unproven' are different facts."""

  def test_a_positive_net_inside_its_own_se_is_unproven_with_its_t(self):
    st = family_stake_status(_fam_edge(52, +0.47, 0.60, verdict="edge"), "continuation")
    assert st["standAside"] is True
    assert st["reason"] == "unproven: net inside its own SE (t<1)"
    assert st["tStat"] == pytest.approx(0.78)
    assert (st["n"], st["netPct"], st["sePct"], st["verdict"]) == (52, 0.47, 0.60, "edge")

  def test_a_net_at_or_below_zero_is_no_edge(self):
    assert family_stake_status(_fam_edge(52, -0.29, 0.30), "continuation")["reason"] == "no edge"
    # Even an inconsistent payload (verdict 'edge' but net <= 0) is called what it is.
    st = family_stake_status(_fam_edge(52, 0.0, 0.30, verdict="edge"), "continuation")
    assert st["standAside"] is True and st["reason"] == "no edge"

  def test_a_staked_family_has_no_reason(self):
    st = family_stake_status(_fam_edge(52, +0.90, 0.30, verdict="edge"), "continuation")
    assert st["standAside"] is False and st["reason"] is None and st["tStat"] == pytest.approx(3.0)
    thin = family_stake_status(_fam_edge(5, -2.0, 0.1), "continuation")
    assert thin["standAside"] is False and thin["reason"] is None

  @pytest.mark.parametrize("n,net,se,verdict", [
    (130, 0.02, 0.30, "edge"), (130, 0.45, 0.30, "edge"), (130, -0.28, 0.30, "no edge"),
    (25, 0.20, 0.60, "edge"), (400, 0.20, 0.08, "edge"), (5, -2.0, 0.1, "no edge"),
    (100, 0.02, None, "edge"), (100, 0.02, 0.0, "edge"),
  ])
  def test_the_boolean_is_exactly_the_status(self, n, net, se, verdict):
    """family_stand_aside is a thin wrapper: one rule, no second copy to drift."""
    board = _fam_edge(n, net, se, verdict=verdict)
    assert family_stand_aside(board, "continuation") is family_stake_status(board, "continuation")["standAside"]


def _side_probe(sym, ts, side, move, family="continuation"):
  """One observation at the 60m horizon; `move` is the fractional move in the TRADED direction."""
  fwd = 100.0 * (1 + move) if side == "long" else 100.0 * (1 - move)
  return {"symbol": sym, "ts": ts, "entryContext": {
    "positionSide": side, "marketPriceAtSignal": 100.0, "setupFamily": family, "signalProbe": {"m60": fwd}}}


class TestFamilySplitBySide:
  """The Kelly stake is a bet on a SIDE of a playbook. 2026-09-24: continuation longs n=40 +1.24%,
  shorts n=9 -0.18% ± 1.05% — the pooled 'edge' was carried by the longs and sized the shorts."""

  def _probes(self):
    out = []
    for i in range(30):   # longs: +1% +/- 0.5%
      out.append(_side_probe(f"L{i}-USDT", 1_000_000 + i * 7200, "long", 0.01 + (0.005 if i % 2 else -0.005)))
    for i in range(9):    # shorts: a loser
      out.append(_side_probe(f"S{i}-USDT", 1_000_000 + i * 7200, "short", -0.004 + (0.01 if i % 2 else -0.01)))
    return out

  def test_side_rows_split_the_pooled_row_and_leave_it_untouched(self):
    probes = self._probes()
    out = signal_edge_stats(probes, cost_pct=0.0014, min_samples=20)
    pooled = out["by_family"]["continuation"]
    sides = out["by_family_side"]["continuation"]
    assert sides["long"]["n"] == 30 and sides["short"]["n"] == 9
    assert sides["long"]["n"] + sides["short"]["n"] == pooled["n"]
    assert sides["long"]["verdict"] == "edge" and sides["short"]["verdict"] == "insufficient data"
    assert sides["short"]["mean_pct"] < 0 < sides["long"]["mean_pct"]
    # The pooled row is exactly what it was before the split existed (plus t_stat).
    longs_only = signal_edge_stats(probes[:30], cost_pct=0.0014, min_samples=20)["by_family"]["continuation"]
    assert longs_only["n"] == 30
    mean = (sides["long"]["mean_pct"] * 30 + sides["short"]["mean_pct"] * 9) / 39
    assert pooled["mean_pct"] == pytest.approx(mean, abs=1e-3)
    assert pooled["t_stat"] == pytest.approx(pooled["net_of_cost_pct"] / pooled["stderr_pct"], rel=1e-2)

  def test_the_split_is_taken_after_the_de_overlap(self):
    """Same symbol, same window, opposite sides: one observation, not one per side."""
    probes = [_side_probe("X-USDT", 1_000_000, "long", 0.01), _side_probe("X-USDT", 1_000_060, "short", 0.01)]
    out = signal_edge_stats(probes, cost_pct=0.001, min_samples=1)
    assert out["by_family"]["continuation"]["n"] == 1
    assert set(out["by_family_side"]["continuation"]) == {"long"}

  def test_a_side_with_its_own_sample_is_judged_on_it(self):
    board = {"cost_pct": 0.0014, "by_family": {"continuation": _row_e(49, 0.47, 0.60)},
             "by_family_side": {"continuation": {"long": _row_e(40, 1.10, 0.40), "short": _row_e(9, -0.32, 1.05)}}}
    # Pooled (side=None): exactly today's behaviour — stood aside at t=0.78.
    assert family_stand_aside(board, "continuation") is True
    # Longs, judged on their own n=40 row (t=2.75): staked at full measured size.
    assert family_stand_aside(board, "continuation", side="buy") is False
    assert family_explore_factor(board, "continuation", side="long") == pytest.approx(1.0)
    assert family_stake_status(board, "continuation", side="long")["judgedOn"] == "continuation:long"
    # Shorts, thin on their own: the pooled row judges the stand-aside...
    st = family_stake_status(board, "continuation", side="sell")
    assert st["standAside"] is True and st["sideThin"] is True and st["sideN"] == 9
    assert st["judgedOn"] == "continuation"

  def test_a_thin_side_never_inherits_the_pooled_upsizing(self):
    """2026-09-24: SPX continuation shorts were sized 0.52-0.79 on the longs' record."""
    board = {"cost_pct": 0.0014, "by_family": {"continuation": _row_e(49, 1.30, 0.80)},  # pooled t=1.63
             "by_family_side": {"continuation": {"long": _row_e(40, 1.60, 0.50), "short": _row_e(9, -0.18, 1.05)}}}
    assert family_explore_factor(board, "continuation", explore_factor=0.4) == pytest.approx(0.4 + 0.6 * 0.625)
    assert family_explore_factor(board, "continuation", explore_factor=0.4, side="short") == pytest.approx(0.4)
    assert family_stand_aside(board, "continuation", side="short") is False   # pooled clears t>=1
    assert family_stake(board, "continuation", explore_factor=0.4, side="short") == pytest.approx(0.4)

  def test_a_losing_side_with_its_own_sample_stands_aside_under_a_paying_pool(self):
    board = {"cost_pct": 0.0014, "by_family": {"continuation": _row_e(70, 0.80, 0.30)},
             "by_family_side": {"continuation": {"long": _row_e(45, 1.50, 0.35), "short": _row_e(25, -0.40, 0.30)}}}
    assert family_stand_aside(board, "continuation") is False
    st = family_stake_status(board, "continuation", side="short")
    assert st["standAside"] is True and st["reason"] == "no edge" and st["judgedOn"] == "continuation:short"
    # ...and its measured no-edge shrink reads its own row too.
    assert family_size_factor(board, "continuation", side="short") < 1.0
    assert family_size_factor(board, "continuation") == 1.0

  def test_side_none_and_old_payloads_behave_exactly_as_before(self):
    old = {"by_family": {"continuation": _row_e(30, 0.9, 0.3)}}          # no by_family_side at all
    assert family_explore_factor(old, "continuation") == family_explore_factor(old, "continuation", side=None)
    # A side asked of an old payload is thin -> pooled stand-aside, explore floor.
    assert family_stand_aside(old, "continuation", side="long") is False
    assert family_explore_factor(old, "continuation", side="long", explore_factor=0.4) == pytest.approx(0.4)

  def test_the_row_that_judged_is_named(self):
    board = {"by_family": {"continuation": _row_e(49, 0.47, 0.60)},
             "by_family_side": {"continuation": {"long": _row_e(40, 1.1, 0.4), "short": _row_e(9, -0.3, 1.0)}}}
    thin = describe_stake_row(family_stake_status(board, "continuation", side="short"))
    assert thin == "continuation:short n=9 is unproven, judged on pooled continuation (n=49)"
    own = describe_stake_row(family_stake_status(board, "continuation", side="long"))
    assert own == "judged on its own continuation:long record (n=40)"
    assert describe_stake_row(family_stake_status(board, "continuation")) == "judged on pooled continuation (n=49)"
    assert family_evidence_row(board, "continuation", "hold")["judgedOn"] == "continuation"

  def test_open_families_names_a_side_that_is_open_on_its_own_record(self):
    board = {"by_family": {"continuation": _row_e(49, 0.47, 0.60), "range_edge": _row_e(25, 0.4, 0.2)},
             "by_family_side": {"continuation": {"long": _row_e(40, 1.1, 0.4), "short": _row_e(9, -0.3, 1.0)}}}
    paying = open_families(board)["paying"]
    assert {"family": "continuation", "side": "long", "n": 40, "netOfCostPct": 1.1} in paying
    assert not any(r["family"] == "continuation" and r.get("side") != "long" for r in paying)


def _row_e(n, net, se):
  return {"n": n, "net_of_cost_pct": net, "stderr_pct": se, "t_stat": net / se,
          "verdict": "insufficient data" if n < 20 else ("edge" if net > 0 else "no edge")}


class TestScoreboardCarriesTheStake:
  """The model reads by_family BEFORE proposing; it must see the zero stake there, not in a refusal."""

  def test_rows_are_marked_with_the_order_paths_own_answer(self):
    board = {"cost_pct": 0.0014,
             "by_family": {"continuation": _row_e(49, 0.47, 0.60), "range_edge": _row_e(25, 0.8, 0.2),
                           "breakout": _row_e(7, 0.3, 0.5)},
             "by_family_side": {"continuation": {"long": _row_e(40, 1.1, 0.4), "short": _row_e(9, -0.3, 1.0)},
                                "range_edge": {"long": _row_e(25, 0.8, 0.2)}}}
    out = annotate_family_stakes(board, explore_factor=0.4)
    cont = out["by_family"]["continuation"]
    # Pooled t<1, but the longs are open on their own record: the family is NOT closed (2026-09-25
    # review: the pooled mark told the model to skip longs code would take at full size).
    assert cont["standAside"] is False and cont["stakeReason"] is None
    assert cont["standAsideBySide"] == {"long": False, "short": True}
    assert cont["stakeBySide"] == {"long": pytest.approx(1.0), "short": 0.0}
    assert "stake" not in cont                                   # no scalar that no entry receives
    assert cont["verdict"] == "edge"                              # the verdict string is untouched
    assert out["by_family"]["range_edge"]["stakeBySide"] == {"long": pytest.approx(1.0), "short": pytest.approx(0.4)}
    assert out["by_family"]["breakout"]["stakeBySide"] == {"long": pytest.approx(0.4), "short": pytest.approx(0.4)}
    longs = out["by_family_side"]["continuation"]["long"]
    shorts = out["by_family_side"]["continuation"]["short"]
    assert longs["standAside"] is False and longs["judgedOn"] == "own" and longs["stake"] == pytest.approx(1.0)
    assert shorts["standAside"] is True and shorts["judgedOn"] == "pooled" and shorts["stake"] == 0.0
    # Every per-side stake is exactly the order path's answer for an entry on that side.
    for fam, row in out["by_family"].items():
      for s in ("long", "short"):
        assert row["stakeBySide"][s] == pytest.approx(family_stake(board, fam, explore_factor=0.4, side=s), abs=1e-3)
        assert row["standAsideBySide"][s] is family_stake_status(board, fam, side=s)["standAside"]

  def test_a_pooled_open_family_still_shows_its_refused_side(self):
    """The funding_carry shape: pooled t~1.9 reads staked, but the shorts (own n>=20, t<1) are refused
    by code — the pooled row must say so, or the model re-proposes the shorts into a refusal."""
    import random
    from src.edge import _scored_row
    rng = random.Random(7)
    longs = [0.012 + rng.gauss(0, 0.02) for _ in range(23)]
    shorts = [0.003 + rng.gauss(0, 0.03) for _ in range(21)]
    cost = 0.0022
    side_rows = {"long": _scored_row(longs, cost, 20), "short": _scored_row(shorts, cost, 20)}
    pooled = _scored_row(longs + shorts, cost, 20)
    board = {"cost_pct": cost, "by_family": {"funding_carry": pooled}, "by_family_side": {"funding_carry": side_rows}}
    assert family_stake_status(board, "funding_carry")["standAside"] is False          # pooled: open
    short_st = family_stake_status(board, "funding_carry", side="short")
    assert short_st["standAside"] is True and short_st["judgedOn"] == "funding_carry:short"
    row = annotate_family_stakes(board)["by_family"]["funding_carry"]
    assert row["standAside"] is False                                   # the longs are still open
    assert row["standAsideBySide"]["short"] is True and row["stakeBySide"]["short"] == 0.0
    assert row["stakeBySide"]["long"] == pytest.approx(family_stake(board, "funding_carry", side="long"), abs=1e-3)
    assert row["stakeBySide"]["long"] > 0

  def test_a_family_is_closed_only_when_both_sides_are(self):
    board = {"cost_pct": 0.0014, "by_family": {"fade_extreme": _row_e(40, -0.3, 0.2)},
             "by_family_side": {"fade_extreme": {"long": _row_e(22, -0.4, 0.3), "short": _row_e(18, -0.2, 0.3)}}}
    row = annotate_family_stakes(board)["by_family"]["fade_extreme"]
    assert row["standAside"] is True and row["stakeReason"] == "no edge"
    assert row["stakeBySide"] == {"long": 0.0, "short": 0.0}

  def test_with_the_stand_aside_off_nothing_is_zeroed(self):
    """cfg.edge.stand_aside_no_edge_family=False: the order path only takes the worse factor."""
    board = {"cost_pct": 0.0014,
             "by_family": {"continuation": _row_e(49, 0.47, 0.60)},
             "by_family_side": {"continuation": {"long": _row_e(40, 1.1, 0.4), "short": _row_e(9, -0.3, 1.0)}}}
    out = annotate_family_stakes(board, explore_factor=0.4, stand_aside_enabled=False)
    cont = out["by_family"]["continuation"]
    assert cont["standAside"] is False and cont["standAsideBySide"] == {"long": False, "short": False}
    assert cont["stakeBySide"]["short"] == pytest.approx(0.4)             # thin side: explore cap, not 0
    shorts = out["by_family_side"]["continuation"]["short"]
    assert shorts["standAside"] is False and shorts["stakeReason"] is None and shorts["stake"] == pytest.approx(0.4)
    assert family_stake(board, "continuation", side="short", stand_aside_enabled=False) == pytest.approx(0.4)
    assert family_stake(board, "continuation", side="short") == 0.0

  def test_the_shared_edge_state_is_never_mutated(self):
    board = {"by_family": {"continuation": _row_e(49, 0.47, 0.60)}}
    annotate_family_stakes(board)
    assert "standAside" not in board["by_family"]["continuation"]
    assert annotate_family_stakes({"verdict": "insufficient data"}) == {"verdict": "insufficient data"}
    assert annotate_family_stakes(None) == {}


def _conf_probe(model, day, i, conf, move, family="continuation", min_conf=0.65):
  """A probe-era row: model + confidence + the floor it had to clear (``min_conf=None`` = legacy shape)."""
  ts = 1_790_000_000 - (1_790_000_000 % 86400) + day * 86400 + i * 3700
  ctx = {"positionSide": "long", "marketPriceAtSignal": 100.0, "setupFamily": family,
         "signalProbe": {"m60": 100.0 * (1 + move)}}
  if model is not None:
    ctx["model"] = model
  if conf is not None:
    ctx["confidence"] = conf
  if min_conf is not None:
    ctx["minConfidence"] = min_conf
  return {"symbol": f"C{day}-{i}-USDT", "ts": ts, "entryContext": ctx}


class TestConfidenceEdgeStats:
  """Report-only: does each model's stated confidence rank its own calls?"""

  def test_a_regime_level_shift_is_not_mistaken_for_information(self):
    """The gpt-5.6 artefact: calm days carry low confidence AND flat returns, a rally carries high
    confidence AND big returns, so pooled rank correlation looks strong — while within any one day
    confidence orders nothing (here it is exactly inverted within each day)."""
    probes = []
    for day in range(4):
      calm = day < 2
      for i in range(8):
        conf = (0.60 if calm else 0.85) + i * 0.005
        move = (0.000 if calm else 0.02) - i * 0.0005      # higher confidence, WORSE within the day
        probes.append(_conf_probe("gpt-5.6", day, i, conf, move))
    out = confidence_edge_stats(probes, cost_pct=0.0014)["by_model"]["gpt-5.6"]
    assert out["pooled"]["rho"] > 0.5
    assert out["withinDay"]["rho"] < 0
    assert out["verdict"] != "informative"

  def test_a_real_within_day_ordering_is_informative(self):
    probes = [_conf_probe("gpt-6-luna", day, i, 0.6 + i * 0.03, -0.01 + i * 0.003 + (0.0005 if day % 2 else 0))
              for day in range(4) for i in range(8)]
    out = confidence_edge_stats(probes, cost_pct=0.0014)["by_model"]["gpt-6-luna"]
    assert out["withinDay"]["rho"] == pytest.approx(1.0) or out["withinDay"]["rho"] > 0.9
    assert out["verdict"] == "informative"
    assert out["n"] == 32
    assert out["confidence"]["p50"] == pytest.approx(0.69, abs=0.03)

  def test_models_are_kept_apart_and_unstamped_rows_excluded(self):
    probes = ([_conf_probe("a", 0, i, 0.7, 0.01) for i in range(5)]
              + [_conf_probe("b", 1, i, 0.8, 0.01) for i in range(3)]
              + [_conf_probe(None, 2, i, 0.8, 0.01) for i in range(4)]       # legacy: no model
              + [_conf_probe("a", 3, i, None, 0.01) for i in range(4)])      # no confidence
    out = confidence_edge_stats(probes, cost_pct=0.0014)
    assert set(out["by_model"]) == {"a", "b"}
    assert out["by_model"]["a"]["n"] == 5 and out["by_model"]["b"]["n"] == 3
    assert out["by_model"]["a"]["verdict"] == "insufficient data"

  def test_legacy_placed_trade_rows_are_excluded_even_with_a_model_and_confidence(self, tmp_path):
    """W4 (2026-09-25 review): HEAD's futures entryContext already stamped model + confidence, and
    signal_probes() unions placed trades in — so the post-gate placed subset scored as 'the model's
    calls' (the 09-24 gpt-5.6-luna n=29 'inverted' row came entirely from it). The probe-era stamp
    (minConfidence) is what separates every call from the placed subset."""
    from src.memory import MemoryStore
    store = MemoryStore(str(tmp_path / "m.json"))
    data = store._read()
    legacy = [_conf_probe("gpt-5.6-luna", d, i, 0.6 + i * 0.03, 0.02 - i * 0.004, min_conf=None)
              for d in range(3) for i in range(10)]
    data["trades"] = legacy                                            # HEAD-shaped placed trades
    data["signal_probes"] = [_conf_probe("gpt-6-luna", 5, i, 0.7, 0.01) for i in range(4)]
    store._write(data)
    rows = store.signal_probes(limit=0)
    assert sum(1 for r in rows if r["entryContext"].get("model") == "gpt-5.6-luna") == 30   # they reach it
    out = confidence_edge_stats(rows, cost_pct=0.0014)
    assert "gpt-5.6-luna" not in out["by_model"]
    assert out["by_model"]["gpt-6-luna"]["n"] == 4

  def test_it_scores_each_call_at_its_familys_own_horizon(self):
    p = _conf_probe("a", 0, 0, 0.7, 0.01, family="fade_extreme")
    p["entryContext"]["signalProbe"] = {"m15": 101.0, "m60": 90.0}
    out = confidence_edge_stats([p], cost_pct=0.0, family_horizons={"fade_extreme": 15})
    assert out["by_model"]["a"]["meanNetPct"] == pytest.approx(1.0)

  def test_it_never_raises_and_the_prompt_sees_only_real_samples(self):
    assert confidence_edge_stats(None)["by_model"] == {}
    assert confidence_edge_stats([{"junk": 1}, None, "x"])["by_model"] == {}
    stats = {"min_samples": 20, "by_model": {"a": {"n": 25}, "b": {"n": 19}, "c": {"n": 40}}}
    assert set(confidence_edge_for_prompt(stats)) == {"a", "c"}
    # The prompt shows the RUNNING model its own record only — another model's number is not its own.
    assert set(confidence_edge_for_prompt(stats, model="a")) == {"a"}
    assert confidence_edge_for_prompt(stats, model="b") == {}
    assert confidence_edge_for_prompt(None) == {}


class TestSupervisorSeesTheScoreboard:
  """The owner asks the Supervisor 'why did it refuse continuation?' — it must see the same stake."""

  def test_it_reports_side_rows_stakes_and_confidence_from_the_store(self, tmp_path):
    from types import SimpleNamespace
    from src.memory import MemoryStore
    from src.supervisor import edge_scoreboard
    store = MemoryStore(str(tmp_path / "m.json"))
    data = store._read()
    data["signal_probes"] = [_conf_probe("gpt-6-luna", d, i, 0.7, 0.02) for d in range(3) for i in range(8)]
    store._write(data)
    cfg = SimpleNamespace(trading=SimpleNamespace(estimated_slippage_pct=0.001, slippage_autotune_min_samples=8),
                          edge=SimpleNamespace(explore_unproven_family_factor=0.4))
    out = edge_scoreboard(store, cfg)
    row = out["signalEdge"]["by_family"]["continuation"]
    assert row["n"] == 24 and "stakeBySide" in row and "standAside" in row and "stake" not in row
    assert out["signalEdge"]["by_family_side"]["continuation"]["long"]["judgedOn"] == "own"
    assert out["confidenceEdge"]["by_model"]["gpt-6-luna"]["n"] == 24

  def test_it_honours_the_stand_aside_flag_and_the_recorded_taker_fee(self, tmp_path, monkeypatch):
    """The Supervisor quotes 'the stake the order path applies now': with the stand-aside switched off
    the order path zeroes nothing, and the cost is the one shared basis (edge.probe_cost_pct)."""
    from types import SimpleNamespace
    from src import supervisor
    from src.memory import MemoryStore
    store = MemoryStore(str(tmp_path / "m.json"))
    data = store._read()
    # Pooled net positive but inside its SE: stood aside with the flag on.
    data["signal_probes"] = [_conf_probe("m", d, i, 0.7, (0.03 if i % 2 else -0.025)) for d in range(3) for i in range(8)]
    store._write(data)
    tr = SimpleNamespace(estimated_slippage_pct=0.001, slippage_autotune_min_samples=8)
    on = supervisor.edge_scoreboard(store, SimpleNamespace(trading=tr, edge=SimpleNamespace(
      explore_unproven_family_factor=0.4, stand_aside_no_edge_family=True)))
    off = supervisor.edge_scoreboard(store, SimpleNamespace(trading=tr, edge=SimpleNamespace(
      explore_unproven_family_factor=0.4, stand_aside_no_edge_family=False)))
    assert on["signalEdge"]["by_family"]["continuation"]["standAside"] is True
    assert off["signalEdge"]["by_family"]["continuation"]["standAside"] is False
    assert off["signalEdge"]["by_family_side"]["continuation"]["long"]["stake"] > 0
    seen = []
    monkeypatch.setattr(supervisor, "probe_cost_pct", lambda m, c: seen.append(1) or 0.0022)
    supervisor.edge_scoreboard(store, SimpleNamespace(trading=tr, edge=SimpleNamespace(
      explore_unproven_family_factor=0.4)))
    assert seen, "the Supervisor must charge the shared cost basis, not its own copy"

  def test_the_cost_basis_reads_the_recorded_taker_fee(self, tmp_path):
    from types import SimpleNamespace
    from src.edge import probe_cost_pct, probe_taker_fee
    from src.memory import MemoryStore
    store = MemoryStore(str(tmp_path / "m.json"))
    cfg = SimpleNamespace(trading=SimpleNamespace(estimated_slippage_pct=0.001, slippage_autotune_min_samples=8))
    assert probe_taker_fee(store) == pytest.approx(0.0006)          # nothing recorded: KuCoin default
    base = probe_cost_pct(store, cfg)
    data = store._read()
    data["fees"] = [{"futures_taker": 0.0005, "ts": int(time.time())}]
    store._write(data)
    assert probe_taker_fee(store) == pytest.approx(0.0005)
    assert probe_cost_pct(store, cfg) == pytest.approx(base - 2 * 0.0001)
    assert probe_taker_fee(None) == pytest.approx(0.0006)

  def test_the_owner_only_splits_carry_a_do_not_relay_guard(self):
    """by_market_state and other models' confidence rows are one regime's evidence the trading agent is
    deliberately never shown; a Supervisor note would carry them into its prompt as a directive."""
    import inspect
    from src import supervisor
    src = inspect.getsource(supervisor.run_supervisor_agent)
    tool = src[src.index("async def get_edge_scoreboard"):src.index("async def get_gate_scoreboard")]
    assert "by_market_state" in tool and "unless the owner explicitly asks" in tool
    import re
    caps = src[src.index("get_edge_scoreboard):"):src.index("- Read the gate scoreboard")]
    caps = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', "\"" + caps))
    assert "by_market_state" in caps and "unless the owner explicitly asks" in caps

  def test_it_never_raises_and_is_registered_as_a_tool(self):
    import inspect
    from src import supervisor
    assert "error" in supervisor.edge_scoreboard(None, None)
    src = inspect.getsource(supervisor.run_supervisor_agent)
    assert "return edge_scoreboard(memory, cfg)" in src
    tools_list = src[src.index("tools=["):src.index("model=model,\n  )")]
    assert "get_edge_scoreboard," in tools_list


# ── EXECUTION MAP: fill rate and fill quality by resting distance (2026-09-25) ─────────────────────────
# Helper names are deliberately distinct (_xm_*): a shadowed test helper silently broke tests twice.

from src.edge import (  # noqa: E402
  EXECUTION_DISTANCE_BUCKETS,
  distance_bucket,
  execution_map,
  passive_distance_atr,
)


def _xm_trade(side, market, limit, *, atr_pct=2.0, filled=False, ts=1_790_000_000, fill_ts=None,
              symbol="SPX-USDT", oid=None):
  """A limit-entry trade row as tools records it (clientOid tagged, entryContext stamped)."""
  row = {
    "symbol": symbol, "side": side, "price": limit, "filled": filled, "ts": ts,
    "clientOid": f"traide-entry-{oid or ts}", "orderId": oid,
    "entryContext": {
      "positionSide": "long" if side == "buy" else "short", "entryPrice": limit,
      "marketPriceAtSignal": market, "regime": {"intraday_atr_pct": atr_pct},
    },
  }
  if fill_ts is not None:
    row["fillTs"] = fill_ts
  return row


def _xm_close(side, market, limit, realized_r, *, atr_pct=2.0, oid=None, ts=1_790_000_000):
  """A realized close carrying the entry's own context (as memory.log_decision stores it)."""
  return {
    "symbol": "SPX-USDT", "ts": ts, "realizedR": realized_r, "pnl": realized_r,
    "entryContext": {
      "positionSide": side, "entryPrice": limit, "marketPriceAtSignal": market,
      "regime": {"intraday_atr_pct": atr_pct}, "entryOrderId": oid,
    },
  }


def _xm_probe(sym, ts, side, base, settle, *, atr_pct=1.0, low=None, high=None, stop=None, crossed=None,
              family="continuation", horizon=60):
  ctx = {"positionSide": side, "marketPriceAtSignal": base, "setupFamily": family,
         "signalProbe": {f"m{horizon}": settle}, "atr15Pct": atr_pct, "plannedEntry": base}
  if low is not None:
    ctx["leaseLow"], ctx["leaseHigh"] = low, high
  if stop is not None:
    ctx["plannedStop"] = stop
  if crossed is not None:
    ctx["crossedNetRr"] = crossed
  return {"symbol": sym, "ts": ts, "entryContext": ctx}


class TestExecutionMapDistance:
  def test_buys_and_sells_mirror_and_the_atr_is_in_percent(self):
    # 1% below a 100 market with a 2% ATR15 = half an ATR of passive distance, for either side's mirror.
    assert passive_distance_atr("buy", 99.0, 100.0, 2.0) == pytest.approx(0.5)
    assert passive_distance_atr("sell", 101.0, 100.0, 2.0) == pytest.approx(0.5)
    assert passive_distance_atr("long", 99.0, 100.0, 2.0) == pytest.approx(0.5)
    # Crossing is negative for either side.
    assert passive_distance_atr("buy", 100.5, 100.0, 2.0) < 0
    assert passive_distance_atr("sell", 99.5, 100.0, 2.0) < 0
    for bad in ((None, 99, 100, 2), ("buy", 0, 100, 2), ("buy", 99, 100, 0), ("buy", 99, 100, float("nan"))):
      assert passive_distance_atr(*bad) is None

  def test_marketable_is_its_own_bucket_and_edges_are_half_open(self):
    assert [b[0] for b in EXECUTION_DISTANCE_BUCKETS] == ["marketable", "<0.25", "0.25-0.5", "0.5-1", "1-2", ">=2"]
    assert distance_bucket(0.0) == "marketable" and distance_bucket(-0.3) == "marketable"
    assert distance_bucket(0.01) == "<0.25" and distance_bucket(0.2499) == "<0.25"
    assert distance_bucket(0.25) == "0.25-0.5" and distance_bucket(0.5) == "0.5-1"
    assert distance_bucket(1.0) == "1-2" and distance_bucket(2.0) == ">=2" and distance_bucket(9.0) == ">=2"
    assert distance_bucket(None) is None and distance_bucket(float("nan")) is None


class TestExecutionMapBuckets:
  def test_fill_rate_fills_and_insufficient_below_min_n(self):
    t0 = 1_790_000_000
    # 12 buys resting 0.75 ATR away (0.5-1): 3 filled after 2, 4 and 12 minutes.
    trades = [_xm_trade("buy", 100.0, 98.5, ts=t0 + i, filled=i < 3, fill_ts=(t0 + i + [120, 240, 720][i]) if i < 3 else None,
                        symbol=f"S{i % 4}-USDT") for i in range(12)]
    # 5 sells resting 1.5 ATR away (1-2): too few to call a rate.
    trades += [_xm_trade("sell", 100.0, 103.0, ts=t0 + 100 + i, filled=i == 0, fill_ts=t0 + 100 + 60) for i in range(5)]
    # 10 closes entered at 0.75 ATR, 4 at 1.5 ATR (sells), 2 marketable buys.
    closes = [_xm_close("long", 100.0, 98.5, r, oid=f"a{i}") for i, r in enumerate([1, -1, 0.5, -0.5, 1, -1, 0.2, 0.2, -0.4, 0.0])]
    closes += [_xm_close("short", 100.0, 103.0, -1.0, oid=f"b{i}") for i in range(4)]
    closes += [_xm_close("long", 100.0, 100.2, 0.3, oid=f"c{i}") for i in range(2)]
    xm = execution_map(trades, closes, min_n=10)
    mid = xm["byDistance"]["0.5-1"]
    assert (mid["placed"], mid["filled"], mid["fillRate"], mid["ideas"]) == (12, 3, 0.25, 4)
    assert mid["medianMinToFill"] == pytest.approx(4.0)
    assert mid["fills"]["n"] == 10 and mid["fills"]["meanR"] == pytest.approx(0.0)
    assert mid["fills"]["seR"] > 0                                   # SE shown next to every mean
    assert mid["fillAdjustedR"] == pytest.approx(0.0)
    far = xm["byDistance"]["1-2"]
    assert far["placed"] == 5 and far["filled"] == 1 and far["fillRate"] == "insufficient"
    assert far["fills"] == {"n": 4, "meanR": "insufficient"}         # a mean below min_n never shows
    assert far["fillAdjustedR"] is None
    assert xm["byDistance"]["marketable"]["fills"]["n"] == 2
    # A small min_n shows the means — the guard is the only thing hiding them.
    assert execution_map(trades, closes, min_n=2)["byDistance"]["1-2"]["fills"]["meanR"] == pytest.approx(-1.0)

  def test_partial_closes_of_one_entry_are_one_observation(self):
    closes = [_xm_close("long", 100.0, 99.0, 0.4, oid="X"), _xm_close("long", 100.0, 99.0, 0.6, oid="X"),
              _xm_close("long", 100.0, 99.0, -1.0)]
    fills = execution_map([], closes, min_n=1)["byDistance"]["0.5-1"]["fills"]
    assert fills["n"] == 2 and fills["meanR"] == pytest.approx(0.0)   # (0.4 + 0.6) and -1.0

  def test_rows_without_stamps_are_counted_as_unbucketed_so_totals_reconcile(self):
    # The legacy row carries no `filled` key at all: like limitFillRate, only `filled is True` is a fill.
    trades = [_xm_trade("buy", 100.0, 99.0, filled=True, fill_ts=1_790_000_060),
              {"symbol": "X-USDT", "side": "buy", "clientOid": "traide-entry-legacy", "ts": 1}]
    closes = [{"symbol": "X-USDT", "realizedR": 1.0, "entryContext": {}}]
    xm = execution_map(trades, closes)
    assert xm["totals"] == {"placed": 2, "filled": 1, "fillRate": 0.5}
    assert xm["unbucketed"] == {"placed": 1, "filled": 0, "closes": 1}
    assert sum(b["placed"] for b in xm["byDistance"].values()) + xm["unbucketed"]["placed"] == 2

  def test_repeated_placements_of_one_idea_are_visible(self):
    trades = [_xm_trade("sell", 100.0, 102.0, ts=1_790_000_000 + i * 900, symbol="H-USDT") for i in range(12)]
    xm = execution_map(trades, [])
    assert xm["mostReplaced"][0] == {"symbol": "H-USDT", "side": "short", "placed": 12, "filled": 0}
    assert xm["byDistance"]["1-2"]["ideas"] == 1                     # twelve re-placements, one idea

  def test_never_raises_on_garbage(self):
    out = execution_map([None, "x", {"filled": True}], [None, {"realizedR": "nan?"}], probes=[None, {"ts": "x"}])
    assert out["totals"]["placed"] == 1 and "byDistance" in out


class TestExecutionMapMatchesLimitFillRate:
  def test_totals_equal_performance_summary_on_the_same_store(self, tmp_path):
    store = MemoryStore(str(tmp_path / "m.json"))
    ctx = {"positionSide": "long", "entryPrice": 99.0, "marketPriceAtSignal": 100.0, "regime": {"intraday_atr_pct": 2.0}}
    for i in range(5):
      store.record_trade("SPX-USDT", "buy", 10.0, price=99.0, venue="futures", filled=False, track_position=False,
                         order_id=f"o{i}", client_oid=f"traide-entry-{i}", entry_context=ctx)
    store.record_trade("SPX-USDT", "buy", 10.0, price=99.0, venue="futures", filled=False, track_position=False,
                       order_id="legacy", client_oid="traide-entry-legacy")
    store.record_trade("SPX-USDT", "buy", 10.0, venue="futures", filled=True, order_id="manual")  # not a bot limit
    store.mark_order_filled("o1", fill_ts=int(__import__("time").time()) + 30)
    store.mark_order_filled("o3")
    store.mark_order_filled("legacy")
    summary = store.performance_summary()
    xm = execution_map(store.limit_entry_records(), [])
    assert xm["totals"]["placed"] == summary["limitOrdersSubmitted"] == 6
    assert xm["totals"]["filled"] == summary["limitOrdersFilled"] == 3
    assert xm["totals"]["fillRate"] == summary["limitFillRate"]
    assert xm["byDistance"]["0.5-1"]["placed"] + xm["unbucketed"]["placed"] == 6


class TestExecutionMapCounterfactual:
  T = 1_790_000_000

  def test_a_depth_fills_on_the_adverse_move_for_longs_and_for_shorts(self):
    # ATR15 = 1% of 100. The long dipped to 99.4 (0.6 ATR against it); the short spiked to 100.3 (0.3 ATR).
    probes = [_xm_probe("A-USDT", self.T, "long", 100.0, 101.0, low=99.4, high=101.5, stop=98.0),
              _xm_probe("B-USDT", self.T, "short", 100.0, 99.0, low=98.8, high=100.3, stop=102.0)]
    cf = execution_map([], [], probes=probes, cost_pct=0.001, min_n=1)["counterfactual"]
    assert cf["calls"] == 2
    d = cf["byDepth"]
    assert (d["cross"]["filled"], d["0.25"]["filled"], d["0.5"]["filled"], d["1"]["filled"]) == (2, 2, 1, 0)
    # The long's 0.5-ATR fill is at 99.5: +1.5/99.5 gross, less cost, in % and in its own R (2% stop).
    net = (101.0 - 99.5) / 99.5 - 0.001
    assert d["0.5"]["netPct"]["meanPct"] == pytest.approx(round(net * 100, 3))
    assert d["0.5"]["netR"]["meanR"] == pytest.approx(round(net / 0.02, 3))
    # The cross is the plain call-price return for both sides.
    cross = [(101.0 - 100.0) / 100.0 - 0.001, (100.0 - 99.0) / 100.0 - 0.001]
    assert d["cross"]["netPct"]["meanPct"] == pytest.approx(round(sum(cross) / 2 * 100, 3))
    assert d["0.5"]["fillRate"] == 0.5 and d["0.5"]["fillAdjustedR"] == pytest.approx(0.5 * d["0.5"]["netR"]["meanR"], abs=1e-3)

  def test_it_uses_each_familys_own_horizon_and_the_de_overlap(self):
    probes = [_xm_probe("A-USDT", self.T, "long", 100.0, 101.0, low=99.0, high=101.0, horizon=240),
              _xm_probe("A-USDT", self.T + 600, "long", 100.0, 101.0, low=99.0, high=101.0, horizon=240)]
    xm = execution_map([], [], probes=probes, family_horizons={"continuation": 240}, min_n=1)
    assert xm["counterfactual"]["calls"] == 1                        # ten minutes apart = one observation
    assert execution_map([], [], probes=probes, min_n=1)["counterfactual"]["calls"] == 0   # scored at 60m: none

  def test_probes_without_lease_extremes_are_left_out_and_the_section_stays_compact(self):
    probes = [_xm_probe("A-USDT", self.T, "long", 100.0, 101.0)]           # no leaseLow/High yet
    xm = execution_map([], [], probes=probes)
    assert xm["counterfactual"]["calls"] == 0 and "byDepth" not in xm["counterfactual"]
    assert "gross" in xm["counterfactual"]["basis"] and "before trade management" in xm["counterfactual"]["basis"]
    assert "counterfactual" not in execution_map([], [])            # no probes, no section


class TestExecutionMapCrossBand:
  T = 1_790_000_000

  def test_calls_are_banded_by_crossed_net_rr_and_split_by_side(self):
    probes = [
      _xm_probe("A-USDT", self.T, "long", 100.0, 101.0, stop=99.0, crossed=0.8),
      _xm_probe("B-USDT", self.T, "short", 100.0, 99.0, stop=101.0, crossed=1.2),
      _xm_probe("C-USDT", self.T, "long", 100.0, 99.5, stop=99.0, crossed=1.7),
      _xm_probe("D-USDT", self.T, "short", 100.0, 100.5, stop=101.0, crossed=0.0),
    ]
    cb = execution_map([], [], probes=probes, cost_pct=0.0, min_n=1, rr_floor=1.5)["crossBand"]
    assert cb["floor"] == 1.5 and cb["calls"] == 4
    bands = cb["bands"]
    assert list(bands) == ["<1.0", "1.0-1.5", ">=1.5"]
    assert bands["<1.0"]["long"]["meanPct"] == pytest.approx(1.0) and bands["<1.0"]["long"]["meanR"] == pytest.approx(1.0)
    assert bands["<1.0"]["short"]["meanPct"] == pytest.approx(-0.5)   # a 0.0 crossed RR is real, not missing
    assert bands["1.0-1.5"]["short"]["meanPct"] == pytest.approx(1.0) and bands["1.0-1.5"]["long"]["n"] == 0
    assert bands[">=1.5"]["long"]["meanR"] == pytest.approx(-0.5)

  def test_the_upper_band_follows_the_configured_floor(self):
    probes = [_xm_probe("A-USDT", self.T, "long", 100.0, 101.0, stop=99.0, crossed=1.7)]
    cb = execution_map([], [], probes=probes, min_n=1, rr_floor=2.0)["crossBand"]
    assert list(cb["bands"]) == ["<1.0", "1.0-2", ">=2"] and cb["bands"]["1.0-2"]["long"]["n"] == 1


# ── Signal edge split by market state: report-only, off by default (2026-09-25) ─────────────────────


def _ms_signal_probe(i, breadth, move, side="long"):
  """Independent observations (spaced past the widest horizon) carrying a breadth24 stamp."""
  ctx = {"positionSide": side, "marketPriceAtSignal": 100.0, "setupFamily": "continuation",
         "signalProbe": {"m60": 100.0 * (1 + move)}}
  if breadth is not None:
    ctx["marketState"] = {"breadth24": breadth}
  return {"symbol": "X-USDT", "ts": 3_000_000 + i * 240 * 60, "entryContext": ctx}


class TestSignalEdgeByMarketState:
  def _probes(self):
    rows = []
    for i in range(24):                     # low breadth: calls lose
      rows.append(_ms_signal_probe(i, 0.05 + i * 0.001, -0.004))
    for i in range(24, 48):                 # mid
      rows.append(_ms_signal_probe(i, 0.45 + i * 0.001, 0.0))
    for i in range(48, 72):                 # high breadth: calls pay
      rows.append(_ms_signal_probe(i, 0.90 + (i - 48) * 0.001, 0.01))
    rows.append(_ms_signal_probe(72, None, 0.01))     # legacy row with no stamp
    return rows

  def test_absent_unless_asked(self):
    out = signal_edge_stats(self._probes(), cost_pct=0.001, min_samples=20)
    assert "by_market_state" not in out

  def test_buckets_are_the_calls_own_terciles_and_scored_like_a_family(self):
    out = signal_edge_stats(self._probes(), cost_pct=0.001, min_samples=20, market_state_split=True)
    ms = out["by_market_state"]
    assert ms["untagged"] == 1 and ms["cuts"][0] < 0.45 <= ms["cuts"][1] < 0.90
    b = ms["buckets"]
    assert b["low"]["n"] == 24 and b["low"]["verdict"] == "no edge"
    assert b["high"]["n"] == 24 and b["high"]["verdict"] == "edge"
    assert b["high"]["mean_pct"] == pytest.approx(1.0)
    # The split does not change the pooled verdicts the stake path reads.
    plain = signal_edge_stats(self._probes(), cost_pct=0.001, min_samples=20)
    assert out["by_family"] == plain["by_family"]

  def test_a_thin_bucket_reads_insufficient_data_without_numbers(self):
    out = signal_edge_stats(self._probes()[:30], cost_pct=0.001, min_samples=20, market_state_split=True)
    for bucket in out["by_market_state"]["buckets"].values():
      assert bucket["n"] < 20 and bucket == {"n": bucket["n"], "verdict": "insufficient data"}

  def test_the_supervisor_view_carries_it(self, tmp_path):
    from types import SimpleNamespace
    from src.memory import MemoryStore
    from src.supervisor import edge_scoreboard
    store = MemoryStore(str(tmp_path / "m.json"))
    data = store._read()
    data["signal_probes"] = self._probes()
    store._write(data)
    cfg = SimpleNamespace(trading=SimpleNamespace(estimated_slippage_pct=0.001, slippage_autotune_min_samples=8),
                          edge=SimpleNamespace(explore_unproven_family_factor=0.4))
    out = edge_scoreboard(store, cfg)
    assert out["signalEdge"]["by_market_state"]["buckets"]["high"]["n"] == 24


# ── Gate scoreboard: what each directional gate blocks vs what it allows (2026-09-25) ─────────────────


def _gp_row(sym, ts, side, base, stamps, *, kind="state", gates=(), gate=None, family=None):
  """One gate-probe row in the store's shape (memory._gate_probe_row)."""
  ctx = {"positionSide": side, "marketPriceAtSignal": base, "gateProbe": kind, "gates": list(gates),
         "signalProbe": dict(stamps), "setupFamily": family}
  if gate:
    ctx["gate"] = gate
  return {"symbol": sym, "ts": ts, "entryContext": ctx}


def _board_for(days_spec, *, horizon="240", gate="h1_align", side="long"):
  """state_days from {day: (blocked_n, blocked_mean, allowed_n, allowed_mean)} at one horizon."""
  days = {}
  for day, (bn, bm, an, am) in days_spec.items():
    cell = [bn + an, bn * bm + an * am, {gate: [bn, bn * bm]} if bn else {}]
    days[str(day)] = {side: {horizon: cell}}
  return days


class TestGateStateCells:
  def test_both_sides_of_one_reading_count_and_gates_are_attributed(self):
    """A reading writes a long AND a short row at one instant; the shared (symbol, horizon) de-overlap
    would drop the second, so the fold de-overlaps per side."""
    from src.edge import gate_state_cells
    ts = 20_000 * 86400 + 3600
    rows = [_gp_row("A-USDT", ts, "long", 100.0, {"m240": 101.0}, gates=["h1_align", "tf_conflict"]),
            _gp_row("A-USDT", ts, "short", 100.0, {"m240": 101.0})]
    cells = gate_state_cells(rows)
    day = cells["20000"]
    assert day["long"]["240"][0] == 1 and day["long"]["240"][1] == pytest.approx(0.01)
    assert day["long"]["240"][2] == {"h1_align": [1, pytest.approx(0.01)], "tf_conflict": [1, pytest.approx(0.01)]}
    assert day["short"]["240"] == [1, pytest.approx(-0.01), {}]

  def test_overlapping_rows_on_one_side_count_once_and_funding_is_added(self):
    from src.edge import gate_state_cells
    ts = 20_000 * 86400
    rows = [_gp_row("A-USDT", ts, "long", 100.0, {"m60": 101.0, "f60": 0.003}),
            _gp_row("A-USDT", ts + 1800, "long", 100.0, {"m60": 150.0}),        # inside the first's 60m window
            _gp_row("R-USDT", ts, "long", 100.0, {"m60": 150.0}, kind="refusal", gate="bench")]
    cell = gate_state_cells(rows)["20000"]["long"]["60"]
    assert cell[0] == 1 and cell[1] == pytest.approx(0.013)


class TestGateScoreboard:
  def test_a_consistent_gap_over_enough_days_earns_a_verdict(self):
    from src.edge import gate_scoreboard
    spec = {1: (4, -0.010, 6, 0.010), 2: (4, -0.012, 6, 0.010), 3: (4, -0.008, 6, 0.010),
            4: (4, -0.011, 6, 0.010), 5: (4, -0.009, 6, 0.010)}
    board = gate_scoreboard([], cost_pct=0.0014, family_horizons={"continuation": 240},
                            state_days=_board_for(spec))
    row = board["gates"]["h1_align"]["state"]["long"]
    assert row["matchedN"] == 20 and row["matchedDays"] == 5
    assert row["diffPct"] == pytest.approx(-2.0, abs=1e-6)
    assert row["t"] < -2 and row["verdict"] == "blocked underperform"
    assert row["blocked"]["n"] == 20 and row["blocked"]["meanNetPct"] == pytest.approx(-1.0 - 0.14, abs=1e-6)
    assert row["baseline"]["n"] == 30 and row["baseline"]["meanNetPct"] == pytest.approx(1.0 - 0.14, abs=1e-6)
    assert board["costPct"] == pytest.approx(0.14) and board["stateHorizonMin"] == 240

  def test_no_verdict_below_min_samples_however_large_the_gap(self):
    from src.edge import gate_scoreboard
    spec = {1: (4, -0.010, 6, 0.010), 2: (4, -0.012, 6, 0.010), 3: (4, -0.008, 6, 0.010),
            4: (4, -0.011, 6, 0.010)}
    row = gate_scoreboard([], family_horizons={"continuation": 240},
                          state_days=_board_for(spec))["gates"]["h1_align"]["state"]["long"]
    assert row["matchedN"] == 16 and row["diffPct"] == pytest.approx(-2.025, abs=1e-6)
    assert row["verdict"] == "insufficient data" and row["t"] is None

  def test_few_days_need_the_student_t_bar_not_the_normal_one(self):
    """The evidence unit is a DAY: on three days a |t| of ~3 is noise (cut-off ~4.4, df=2), while the
    same per-day effect over ten days clears its bar (~2.3). On pure noise the normal bar called 20% of
    3-day boards significant."""
    from src.edge import gate_scoreboard
    wobble = (-0.004, -0.018, -0.009)
    three = {d: (8, w, 8, 0.0) for d, w in enumerate(wobble)}
    row = gate_scoreboard([], family_horizons={"continuation": 240},
                          state_days=_board_for(three))["gates"]["h1_align"]["state"]["long"]
    assert -4.37 < row["t"] < -2 and row["tCritical"] == pytest.approx(4.37, abs=0.01)
    assert row["verdict"] == "not demonstrated"
    ten = {d: (8, wobble[d % 3], 8, 0.0) for d in range(10)}
    row = gate_scoreboard([], family_horizons={"continuation": 240},
                          state_days=_board_for(ten))["gates"]["h1_align"]["state"]["long"]
    assert row["tCritical"] == pytest.approx(2.32, abs=0.01) and row["verdict"] == "blocked underperform"

  def test_two_days_never_earn_a_verdict(self):
    from src.edge import gate_scoreboard
    row = gate_scoreboard([], family_horizons={"continuation": 240},
                          state_days=_board_for({1: (12, -0.02, 8, 0.0), 2: (12, -0.021, 8, 0.0)}))
    row = row["gates"]["h1_align"]["state"]["long"]
    assert row["matchedN"] == 24 and row["matchedDays"] == 2
    assert row["verdict"] == "insufficient data" and row["t"] is None

  def test_a_regime_shift_between_days_is_never_read_as_the_gates_effect(self):
    """Blocked only on the bad days, allowed only on the good ones: the absolute means differ by 4%,
    but no day holds both, so there is no concurrent comparison and no differential at all."""
    from src.edge import gate_scoreboard
    spec = {d: (8, -0.02, 0, 0.0) for d in (1, 2, 3)}
    spec.update({d: (0, 0.0, 8, 0.02) for d in (4, 5, 6)})
    row = gate_scoreboard([], family_horizons={"continuation": 240},
                          state_days=_board_for(spec))["gates"]["h1_align"]["state"]["long"]
    assert row["blocked"]["meanNetPct"] < 0 < row["baseline"]["meanNetPct"]
    assert row["matchedN"] == 0 and row["diffPct"] is None and row["verdict"] == "insufficient data"

  def test_state_is_scored_at_continuations_own_horizon(self):
    from src.edge import gate_scoreboard
    days = _board_for({1: (2, -0.01, 2, 0.01)}, horizon="60")
    days["1"]["long"]["240"] = [4, 0.0, {"h1_align": [2, 0.08]}]
    at_60 = gate_scoreboard([], family_horizons={"continuation": 60}, state_days=days)
    assert at_60["stateHorizonMin"] == 60
    assert at_60["gates"]["h1_align"]["state"]["long"]["diffPct"] == pytest.approx(-2.0, abs=1e-6)
    at_240 = gate_scoreboard([], family_horizons={"continuation": 240}, state_days=days)
    assert at_240["gates"]["h1_align"]["state"]["long"]["diffPct"] == pytest.approx(8.0, abs=1e-6)

  def test_refusals_are_deoverlapped_per_gate_symbol_side_and_scored_against_admitted_calls(self):
    from src.edge import gate_scoreboard
    ts = 20_000 * 86400
    refusals = [_gp_row("A-USDT", ts + i * 300, "long", 100.0, {"m60": 99.0}, kind="refusal",
                        gates=["anti_fomo"], gate="anti_fomo", family="continuation") for i in range(3)]
    admitted = [{"symbol": "B-USDT", "ts": ts, "entryContext": {
      "positionSide": "long", "marketPriceAtSignal": 100.0, "setupFamily": "continuation",
      "signalProbe": {"m60": 101.0}}}]
    board = gate_scoreboard(refusals, family_horizons={"continuation": 60}, admitted_probes=admitted)
    row = board["gates"]["anti_fomo"]["refused"]["long"]
    assert row["blocked"]["n"] == 1                      # three retries inside one window = one call
    assert row["baseline"]["n"] == 1 and row["matchedN"] == 1
    assert row["diffPct"] == pytest.approx(-2.0, abs=1e-6)
    assert row["verdict"] == "insufficient data"
    assert board["refusalRows"] == 3 and "state" not in board["gates"]["anti_fomo"]

  def test_junk_never_raises_and_empty_says_so(self):
    from src.edge import gate_scoreboard, gate_scoreboard_log_line
    board = gate_scoreboard([None, {"entryContext": "x"}, 5], state_days={"x": "junk", "2": {"long": {"240": "?"}}})
    assert board["gates"] == {}
    assert "nothing scored yet" in gate_scoreboard_log_line(board)
    assert gate_scoreboard_log_line({"error": "disk"}) == "unavailable (disk)"
    assert gate_scoreboard_log_line(None) == "no scoreboard"

  def test_the_log_line_names_each_gate_side_and_verdict(self):
    from src.edge import gate_scoreboard, gate_scoreboard_log_line
    spec = {d: (4, -0.010 - 0.001 * (d % 3), 6, 0.010) for d in range(1, 6)}
    line = gate_scoreboard_log_line(gate_scoreboard([], family_horizons={"continuation": 240},
                                                    state_days=_board_for(spec)))
    assert "state@240m over 5d" in line
    assert "h1_align/long state n=20/5d diff" in line and "blocked underperform" in line

  def test_from_a_store_end_to_end(self, tmp_path):
    """Recorded -> settled -> folded -> scored, through the real store."""
    from src.edge import gate_scoreboard_from_store, gate_state_cells
    m = MemoryStore(str(tmp_path / "g.json"))
    m.record_gate_state_probes("A-USDT", 100.0, {"long": ["h1_align"], "short": []}, price_source="futures_mark")
    m.record_gate_probe("B-USDT", "buy", 100.0, "h1_align", setup_family="continuation")
    data = m._read()
    for row in data["gate_probes"]:
      row["entryContext"]["signalProbe"] = {"m5": 100.0, "m15": 100.0, "m60": 99.0, "m240": 99.0}
    m._write(data)
    assert m.fold_settled_gate_states(gate_state_cells) == 2
    board = gate_scoreboard_from_store(m, cost_pct=0.001)
    assert board["stateDays"] == 1 and board["pendingStateRows"] == 0 and board["refusalRows"] == 1
    assert board["gates"]["h1_align"]["state"]["long"]["blocked"]["n"] == 1
    assert board["gates"]["h1_align"]["refused"]["long"]["blocked"]["n"] == 1
    # The admitted-call baseline is the signal probes; the gate rows themselves never become one.
    assert m.signal_probes(limit=0) == []


class TestSupervisorGateScoreboard:
  def test_the_report_reads_the_store_and_never_raises(self, tmp_path):
    from types import SimpleNamespace
    from src.memory import MemoryStore
    from src.supervisor import gate_scoreboard_report
    store = MemoryStore(str(tmp_path / "m.json"))
    store.record_gate_probe("B-USDT", "buy", 100.0, "tf_conflict", setup_family="continuation")
    cfg = SimpleNamespace(trading=SimpleNamespace(estimated_slippage_pct=0.001, slippage_autotune_min_samples=8))
    out = gate_scoreboard_report(store, cfg)
    assert out["refusalRows"] == 1 and out["costPct"] == pytest.approx(0.32)
    assert "error" in gate_scoreboard_report(None, cfg)

  def test_it_is_a_registered_tool_and_the_instructions_keep_it_from_the_trading_agent(self):
    import inspect
    from src import supervisor
    src = inspect.getsource(supervisor.run_supervisor_agent)
    assert "return gate_scoreboard_report(memory, cfg)" in src
    tools_list = src[src.index("tools=["):src.index("model=model,\n  )")]
    assert "get_gate_scoreboard," in tools_list
    assert "do not pass a gate verdict" in src and "to the trading agent" in src


class TestHatchedCallsAreScoredAgainstCallsThatNeverMetTheGate:
  def test_hatched_vs_clear_with_legacy_rows_in_neither(self):
    from src.edge import gate_scoreboard
    ts = 20_000 * 86400

    def _adm(sym, move, passed, fam="breakout"):
      ctx = {"positionSide": "long", "marketPriceAtSignal": 100.0, "setupFamily": fam,
             "signalProbe": {"m60": 100.0 * (1 + move)}}
      if passed is not None:
        ctx["gatesPassed"] = passed
      return {"symbol": sym, "ts": ts, "entryContext": ctx}

    admitted = ([_adm(f"H{i}-USDT", -0.01, [{"gate": "h1_align", "hatch": "declared"}]) for i in range(3)]
                + [_adm("F-USDT", -0.01, [{"gate": "h1_align", "hatch": "fade"}], fam="fade_extreme")]
                + [_adm(f"C{i}-USDT", 0.01, []) for i in range(2)]
                + [_adm("L-USDT", 0.05, None)])
    board = gate_scoreboard([], family_horizons={"breakout": 60, "fade_extreme": 60}, admitted_probes=admitted)
    row = board["gates"]["h1_align"]["hatched"]["long"]
    assert row["blocked"]["n"] == 4 and row["baseline"]["n"] == 2          # the legacy row is in neither
    assert row["diffPct"] == pytest.approx(-2.0, abs=1e-6)
    assert row["byHatch"] == [{"hatch": "declared", "calls": 3}, {"hatch": "fade", "calls": 1}]
    assert "short" not in board["gates"]["h1_align"]["hatched"]


# --- holding-time weighted family scoring (2026-09-26) --------------------------------------------
# Live 2026-09-25: continuation's median hold straddled 120m (the log midpoint of 60m and 240m). One
# fast +0.39R FET winner moved it 125.5 -> 118.5, the snapped horizon 240m -> 60m, and continuation
# longs from +1.1% to -0.15% net: zero stake for the rest of an alt rally, which no new close could
# undo because a benched family makes none. Weights make the verdict continuous in the holds.

def test_horizon_weights_are_the_share_of_recent_holds_nearest_each_horizon():
  from src.edge import family_horizon_weights
  holds = [58, 63, 75, 140, 262, 576]
  closes = [_hclose("continuation", m, ts=10_000_000 + i) for i, m in enumerate(holds)]
  assert family_horizon_weights(closes) == {"continuation": {60: 0.5, 240: 0.5}}


def test_horizon_weights_need_the_same_evidence_as_the_snapped_horizon():
  from src.edge import family_horizon_weights
  closes = [_hclose("breakout", 180)] * 2 + [_hclose("continuation", 162)] * 8
  w = family_horizon_weights(closes)
  assert "breakout" not in w and w["continuation"] == {240: 1.0}
  assert family_horizon_weights(None) == {}
  assert family_horizon_weights(["x", {}, {"ts": "bad", "entryContext": {}}]) == {}


def _straddle_closes(n_short):
  """20 recent continuation holds: ``n_short`` at 63m (-> 60m), the rest at 141m (-> 240m)."""
  holds = [63] * n_short + [141] * (20 - n_short)
  return [_hclose("continuation", m, ts=10_000_000 + i) for i, m in enumerate(holds)]


def test_one_close_across_the_log_midpoint_no_longer_flips_the_stake():
  """The live failure. Calls that sit flat at 60m and pay at 240m: under the snapped median, one more
  short hold flips the horizon and benches the family; under the holding-time mix the same close moves
  the score by one twentieth of the 60m/240m gap and the stake survives."""
  from src.edge import (family_horizon_weights, family_scoring_horizons, family_stand_aside,
                        signal_edge_stats)
  probes = [_hprobe("continuation", 1_000_000 + i * 90_000, 100.0, 100.0 + ((-1) ** i) * 0.05, 101.5)
            for i in range(30)]
  before, after = _straddle_closes(9), _straddle_closes(10)
  # The old rule, documented: the median crosses 120m and the verdict jumps.
  assert family_scoring_horizons(before)["continuation"] == 240
  assert family_scoring_horizons(after)["continuation"] == 60
  snapped_before = signal_edge_stats(probes, cost_pct=0.0014, family_horizons=family_scoring_horizons(before))
  snapped_after = signal_edge_stats(probes, cost_pct=0.0014, family_horizons=family_scoring_horizons(after))
  assert family_stand_aside(snapped_before, "continuation") is False
  assert family_stand_aside(snapped_after, "continuation") is True
  # The new rule: both sides of the midpoint score the calls over their real holding mix.
  mixed_before = signal_edge_stats(probes, cost_pct=0.0014, family_horizons=family_scoring_horizons(before),
                                   family_horizon_weights=family_horizon_weights(before))
  mixed_after = signal_edge_stats(probes, cost_pct=0.0014, family_horizons=family_scoring_horizons(after),
                                  family_horizon_weights=family_horizon_weights(after))
  assert family_stand_aside(mixed_before, "continuation") is False
  assert family_stand_aside(mixed_after, "continuation") is False
  gap = abs(mixed_before["by_family"]["continuation"]["mean_pct"]
            - mixed_after["by_family"]["continuation"]["mean_pct"])
  assert gap <= 1.5 / 20 + 1e-6          # one close = one twentieth of the 60m->240m difference (1.5%)
  assert mixed_after["by_family"]["continuation"]["horizonWeights"] == {"60m": 0.5, "240m": 0.5}


def test_a_probe_is_scored_on_the_mix_only_once_every_weighted_horizon_settled():
  """A young probe must not be scored on its short leg alone while its long leg is still pending."""
  from src.edge import signal_edge_stats
  probes = [_hprobe("continuation", 1_000_000 + i * 90_000, 100.0, 100.2, None) for i in range(25)]
  mixed = signal_edge_stats(probes, cost_pct=0.0014,
                            family_horizon_weights={"continuation": {60: 0.5, 240: 0.5}})
  assert "continuation" not in (mixed.get("by_family") or {})
  only60 = signal_edge_stats(probes, cost_pct=0.0014, family_horizon_weights={"continuation": {60: 1.0}})
  assert only60["by_family"]["continuation"]["n"] == 25


def test_all_weight_on_one_horizon_reproduces_that_horizons_row_exactly():
  """Backward compatibility by construction: a family held only at 240m scores exactly as the snapped
  240m row did — same n, mean, SE, t and verdict — pooled and per side."""
  from src.edge import signal_edge_stats
  probes = [_hprobe("continuation", 1_000_000 + i * 90_000, 100.0, 100.1 + 0.02 * i, 100.6 + 0.07 * i,
                    side=("long" if i % 3 else "short")) for i in range(30)]
  snapped = signal_edge_stats(probes, cost_pct=0.0014, family_horizons={"continuation": 240})
  mixed = signal_edge_stats(probes, cost_pct=0.0014, family_horizons={"continuation": 240},
                            family_horizon_weights={"continuation": {240: 1.0}})
  for key in ("n", "mean_pct", "hit_rate", "stderr_pct", "net_of_cost_pct", "t_stat", "verdict"):
    assert mixed["by_family"]["continuation"][key] == snapped["by_family"]["continuation"][key], key
    for side in ("long", "short"):
      assert (mixed["by_family_side"]["continuation"][side][key]
              == snapped["by_family_side"]["continuation"][side][key]), (side, key)


def test_mixing_keeps_each_horizons_sample_and_never_overstates_certainty():
  """Two calls on one symbol 90 minutes apart are two observations at 60m and one at 240m; the mix
  keeps both legs' own samples (n = the thinner leg, not a re-decimated set that would push a side
  under min_samples) and its SE is the weighted SUM of the legs' SEs — never tighter than the legs."""
  from src.edge import signal_edge_stats

  def _same_sym(ts, m60, m240):
    row = _hprobe("continuation", ts, 100.0, m60, m240)
    row["symbol"] = "SUI-USDT"
    return row

  probes = [_same_sym(1_000_000 + k * 5_400, 100.3 + 0.1 * k, 101.0 - 0.2 * k) for k in range(2)]
  probes += [_hprobe("continuation", 9_000_000 + i * 90_000, 100.0, 99.8 + 0.03 * i, 100.9 - 0.05 * i)
             for i in range(8)]
  at60 = signal_edge_stats(probes, cost_pct=0.0014, min_samples=1, family_horizons={"continuation": 60})
  at240 = signal_edge_stats(probes, cost_pct=0.0014, min_samples=1, family_horizons={"continuation": 240})
  mixed = signal_edge_stats(probes, cost_pct=0.0014, min_samples=1,
                            family_horizon_weights={"continuation": {60: 0.5, 240: 0.5}})
  r60, r240, rm = (x["by_family"]["continuation"] for x in (at60, at240, mixed))
  assert rm["nByHorizon"] == {"60m": r60["n"], "240m": r240["n"]}
  assert rm["n"] == min(r60["n"], r240["n"])
  assert abs(rm["mean_pct"] - 0.5 * (r60["mean_pct"] + r240["mean_pct"])) < 1e-3
  assert abs(rm["stderr_pct"] - 0.5 * (r60["stderr_pct"] + r240["stderr_pct"])) < 1e-3


def test_families_without_weights_are_scored_exactly_as_before():
  from src.edge import signal_edge_stats
  probes = ([_hprobe("continuation", 1_000_000 + i * 90_000, 100.0, 100.1, 101.0) for i in range(25)]
            + [_hprobe("fade_extreme", 5_000_000 + i * 90_000, 100.4, 100.2, 99.0) for i in range(25)])
  plain = signal_edge_stats(probes, cost_pct=0.0014, family_horizons={"fade_extreme": 15})
  mixed = signal_edge_stats(probes, cost_pct=0.0014, family_horizons={"fade_extreme": 15},
                            family_horizon_weights={"continuation": {60: 0.5, 240: 0.5}})
  assert mixed["by_family"]["fade_extreme"] == plain["by_family"]["fade_extreme"]
  assert mixed["by_family_side"]["fade_extreme"] == plain["by_family_side"]["fade_extreme"]
  assert mixed["by_horizon"] == plain["by_horizon"]            # the reported horizons are untouched


def test_every_verdict_path_scores_families_on_the_holding_time_mix():
  """Wiring, not the pure function: the agent (whose state the ORDER PATH reads), the dashboard and
  the Supervisor must all pass the weights, or the stake shown and the stake applied can disagree."""
  import inspect
  import src.agent as agent_mod
  import src.dashboard_publisher as dash_mod
  import src.supervisor as sup_mod
  agent_src = inspect.getsource(agent_mod.run_trading_agent)
  assert "_fam_weights = safe_family_horizon_weights(memory)" in agent_src
  assert "family_horizon_weights=_fam_weights" in agent_src
  assert "family_horizon_weights=safe_family_horizon_weights(memory)" in inspect.getsource(dash_mod)
  assert "family_horizon_weights=safe_family_horizon_weights(memory)" in inspect.getsource(sup_mod)
