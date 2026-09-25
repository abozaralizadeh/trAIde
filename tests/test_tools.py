from types import SimpleNamespace

import pytest

from src.regime import risk_capped_contracts
from src.tools import (
    entry_cancel_guard_reason,
    entry_funding_stamp,
    entry_contracts_at_least_one_lot,
    live_entry_price_sourced,
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


class TestLiveEntryPriceCarriesItsMarket:
    """The probe base must name its market, so settlement can read the same one (ONE-USDT, 2026-09-20:
    a futures base settled on the spot ticker scored a ~30% perp/spot basis as a directional return)."""

    class _Futures:
        def __init__(self, value=None, fail=False):
            self.value, self.fail = value, fail

        def get_mark_price(self, symbol):
            if self.fail:
                raise RuntimeError("mark endpoint down")
            return {"symbol": symbol, "value": self.value}

    class _Spot:
        def __init__(self, price=None, fail=False):
            self.price, self.fail = price, fail

        def get_ticker(self, symbol):
            if self.fail:
                raise RuntimeError("ticker down")
            return type("T", (), {"price": self.price})()

    def test_futures_mark_first(self):
        got = live_entry_price_sourced(self._Spot(0.0050), self._Futures(0.0038), "ONE-USDT")
        assert got == (0.0038, "futures_mark")

    def test_spot_only_as_a_labelled_fallback(self):
        got = live_entry_price_sourced(self._Spot(0.0050), self._Futures(fail=True), "ONE-USDT")
        assert got == (0.0050, "spot")
        assert live_entry_price_sourced(self._Spot(0.0050), None, "ONE-USDT") == (0.0050, "spot")

    def test_nothing_available(self):
        assert live_entry_price_sourced(self._Spot(fail=True), self._Futures(fail=True), "ONE-USDT") == (None, None)

    def test_the_entry_path_stamps_the_source_on_the_probe_and_the_trade(self):
        """Wiring, not just the helper: the limit-entry tool must hand the source to BOTH records that
        settle_signal_probes walks (the standalone probe and the placed trade's entryContext)."""
        import inspect
        from src import tools
        src = inspect.getsource(tools.build_tools)
        assert "current_price, _price_source = live_entry_price_sourced(kucoin, kucoin_futures, spot_symbol)" in src
        probe_call = src[src.index("memory.record_signal_probe("):]
        assert "price_source=_price_source" in probe_call[:probe_call.index(")\n")]
        assert '"priceSource": _price_source' in src


class TestEntryStampsTheContractsFundingClock:
    """Analysis and order placement are separate tool calls, so the clock read at analysis must travel
    via the gate state onto entryContext['funding'] — the carry hold's fallback and the replays' grid."""

    def test_stamp_carries_rate_interval_and_next_settlement_in_seconds(self):
        gate = {"funding_rate": -0.00196, "funding_interval_sec": 3600.0, "funding_next_ts": 1790272800.0}
        assert entry_funding_stamp(gate) == {
            "rate": -0.00196, "intervalSec": 3600.0, "nextSettlementTs": 1790272800.0,
        }

    def test_unknown_is_none_never_zero(self):
        """A 0 interval or epoch would be read as a clock; None falls back to the 8h grid."""
        assert entry_funding_stamp({}) == {"rate": None, "intervalSec": None, "nextSettlementTs": None}
        assert entry_funding_stamp(None)["intervalSec"] is None
        assert entry_funding_stamp({"funding_interval_sec": "x"})["intervalSec"] is None

    def test_analysis_keeps_the_clock_and_the_entry_stamps_it(self):
        import inspect
        from src import tools
        src = inspect.getsource(tools.build_tools)
        assert "_funding_clock = funding_clock_from_rate(fr)" in src
        assert '_daily_gate_state[symbol]["funding_interval_sec"]' in src
        assert '_daily_gate_state[symbol]["funding_next_ts"]' in src
        assert 'interval_hours=futures_data["fundingIntervalHours"]' in src
        assert '"funding": entry_funding_stamp(_g_fl)' in src


# ── The futures limit entry, driven end to end on fakes (paper mode: no exchange write) ──────────────


class _EntryFutures:
  """Just enough of KucoinFuturesClient for place_futures_limit_order to reach its paper placement."""

  def __init__(self, mark=1.0, equity=100.0):
    self.mark, self.equity = mark, equity

  def get_account_overview(self):
    return {"accountEquity": self.equity}

  def list_positions(self):
    return []

  def list_stop_orders(self, status="active", symbol=None):
    return []

  def list_orders(self, status="active"):
    return []

  def get_mark_price(self, symbol):
    return {"symbol": symbol, "value": self.mark}

  def get_position(self, symbol):
    return {"symbol": symbol, "currentQty": 0}


def _limit_entry_tools(tmp_path, signal_edge=None, *, model="gpt-test", gate_extra=None, edge_extra=None,
                       ctx_extra=None):
  """build_tools over fakes, with every config knob the path reads pinned (the VM's .env must not
  decide whether this test can reach placement)."""
  import time as _time
  from src.config import load_config
  from src.memory import MemoryStore
  from src.tools import build_tools

  cfg = load_config()
  cfg.azure.deployment = model
  t = cfg.trading
  t.sentiment_filter_enabled = False
  t.min_futures_listing_age_days = 0
  t.min_confidence = 0.65
  t.min_trade_interval_minutes = 0
  t.post_loss_cooldown_minutes = 0
  t.max_trades_per_symbol_per_day = 50
  t.min_entry_deviation_pct = 0.001
  t.min_futures_rr = 1.0
  t.min_net_profit_usd = 0.0
  t.min_profit_roi_pct = 0.0
  t.max_24h_volatility_pct = 0.0
  t.max_position_usd = 1000.0
  t.max_position_equity_pct = 0.0
  t.risk_targeted_sizing = True
  t.stop_atr_floor_mult = 0.0
  cfg.profit_protection.no_chase_enabled = False
  cfg.edge.stand_aside_no_edge_family = True
  cfg.edge.explore_unproven_family_factor = 0.4
  cfg.regime.macro_events_enabled = False
  memory = MemoryStore(str(tmp_path / "mem.json"))
  snapshot = SimpleNamespace(
    tickers={}, balances=[], paper_trading=True, max_position_usd=1000.0, min_confidence=0.65,
    max_leverage=5.0, futures_enabled=True, total_usdt=100.0, futures_positions=[],
    futures_account={"accountEquity": 100.0}, futures_stop_orders=[], futures_pending_orders=[],
    spot_pending_orders=[], spot_stop_orders=[], trading_restricted=False, restriction_reason="",
  )
  gate = {
    "daily_bias": "neutral", "daily_bias_raw": "neutral", "daily_exhausted": False,
    "intraday_bias_15m": "neutral", "intraday_bias_1h": "neutral", "intraday_bias_4h": "neutral",
    "timeframe_conflict": False, "analyzed_at": _time.time(), "data_quality_ok": True,
    "strength": "weak", "market_regime": "ranging",
  }
  gate.update(gate_extra or {})
  edge_state = {"bench": {}, "size_factor": 1.0, "signal_edge": signal_edge or {}}
  edge_state.update(edge_extra or {})
  ctx = SimpleNamespace(
    cfg=cfg, kucoin=None, kucoin_futures=_EntryFutures(), memory=memory, snapshot=snapshot,
    allowed_symbols={"SPX-USDT"}, balances_by_currency={}, fees={"futures_taker": 0.0006},
    _daily_gate_state={"SPX-USDT": gate}, _futures_margin_mode="cross",
    _apply_cross_leverage=lambda *a, **k: None, _btc_daily_bias=lambda *a, **k: "neutral",
    _edge_state=lambda: edge_state, _fee_adjusted_breakeven=lambda *a, **k: 0.0,
    _get_contract_spec=lambda fsym: {
      "multiplier": 1.0, "lotSize": 1, "tickSize": 0.0001, "maxLeverage": 20,
      "takerFeeRate": 0.0006, "firstOpenDate": 1_600_000_000_000,
    },
    _repair_allowed_symbol=lambda s, *a, **k: None, _spot_position_info=lambda *a, **k: {},
    _spot_position_size=lambda *a, **k: 0.0, _stop_distance_ok=lambda *a, **k: True,
    safety_state=None, entry_token=None,
  )
  for key, value in (ctx_extra or {}).items():
    setattr(ctx, key, value)
  return build_tools(ctx), memory


def _place(tools, **overrides):
  import asyncio
  import json
  from agents.tool_context import ToolContext
  args = {"symbol": "SPX-USDT", "side": "sell", "notional_usd": 20.0, "entry_price": 1.01,
          "confidence": 0.8, "take_profit_price": 0.95, "stop_loss_price": 1.03,
          "setup_family": "continuation"}
  args.update(overrides)
  raw = json.dumps(args)
  tool = tools.place_futures_limit_order
  return asyncio.run(tool.on_invoke_tool(
    ToolContext(context=None, tool_name=tool.name, tool_call_id="t1", tool_arguments=raw), raw,
  ))


def _row(n, net, se, verdict=None):
  return {"n": n, "net_of_cost_pct": net, "stderr_pct": se, "t_stat": net / se,
          "verdict": verdict or ("insufficient data" if n < 20 else ("edge" if net > 0 else "no edge"))}


# 2026-09-24 in miniature: continuation's pooled t sits just under the release bar, its longs carry a
# real edge on their own n=40 record, and its shorts are thin (n=9) and negative.
_CONTINUATION_0924 = {"cost_pct": 0.0014, "by_family": {
  "continuation": _row(49, 0.47, 0.60),
  "range_edge": _row(25, 0.40, 0.20),
}, "by_family_side": {"continuation": {"long": _row(40, 1.10, 0.40), "short": _row(9, -0.32, 1.05)}}}


class TestFuturesLimitEntryEndToEnd:
  """Order-path behaviour on fakes. Paper mode stops short of the exchange but runs every gate, the
  probe, the stand-aside, the sizing and the entryContext exactly as live does."""

  def test_a_placed_entry_records_how_it_was_sized(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path)
    out = _place(tools)
    assert out.get("paper") is True and out.get("pendingLimitEntry") is True, out
    ctx = out["tradeRecord"]["entryContext"]
    sizing = ctx["sizing"]
    assert set(sizing["contractsByStage"]) == {"raw", "lotFloor", "riskCap", "heatCap", "concentrationCap"}
    assert sizing["contractsByStage"]["concentrationCap"] == out["contracts"]
    assert sizing["equityUsd"] == pytest.approx(100.0)
    assert sizing["riskFracActual"] == pytest.approx(ctx["plannedMaxLossUsd"] / 100.0, rel=1e-4)
    assert sizing["familyExplore"] == pytest.approx(0.4)   # unproven family -> explore size
    assert sizing["familyMeasured"] == pytest.approx(1.0)
    assert "qualityFloorClamped" in sizing and "soft" in sizing
    assert ctx["minConfidence"] == pytest.approx(0.65)

  def test_the_probe_and_the_trade_record_the_futures_mark_as_their_source(self, tmp_path):
    """T2: behaviour, not source text. A relabel to 'spot' on this path would settle futures-based
    probes on the spot ticker again (the ONE-USDT basis-as-edge bias) with every memory test green."""
    tools, memory = _limit_entry_tools(tmp_path)
    out = _place(tools)
    assert out.get("paper") is True, out
    assert memory.signal_probes(limit=0)[0]["entryContext"]["priceSource"] == "futures_mark"
    assert out["tradeRecord"]["entryContext"]["priceSource"] == "futures_mark"

  def test_a_spot_fallback_is_labelled_spot_on_both_records(self, tmp_path):
    class _NoMark(_EntryFutures):
      def get_mark_price(self, symbol):
        raise RuntimeError("mark endpoint down")

    class _Spot:
      def get_ticker(self, symbol):
        return SimpleNamespace(price=1.0)

    tools, memory = _limit_entry_tools(tmp_path, ctx_extra={"kucoin_futures": _NoMark(), "kucoin": _Spot()})
    out = _place(tools)
    assert out.get("paper") is True, out
    assert memory.signal_probes(limit=0)[0]["entryContext"]["priceSource"] == "spot"
    assert out["tradeRecord"]["entryContext"]["priceSource"] == "spot"

  def test_the_probe_names_the_model_and_the_confidence(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path, model="gpt-6-luna")
    _place(tools, confidence=0.77)
    probe = memory.signal_probes(limit=0)[0]["entryContext"]
    assert probe["model"] == "gpt-6-luna"
    assert probe["confidence"] == pytest.approx(0.77)
    assert probe["minConfidence"] == pytest.approx(0.65)

  def test_a_broken_sizing_breakdown_never_blocks_the_order(self, tmp_path, monkeypatch, caplog):
    """Telemetry in the order path must be total (2026-09-04: an entry-path NameError froze every
    edge verdict for 2.6 days). A failure inside the breakdown logs WARNING; the order still goes."""
    import logging
    from src import tools as tools_mod

    def _boom(**_kw):
      raise RuntimeError("sizing telemetry exploded")

    monkeypatch.setattr(tools_mod, "_sizing_breakdown_unsafe", _boom)
    tools, memory = _limit_entry_tools(tmp_path)
    with caplog.at_level(logging.WARNING):
      out = _place(tools)
    assert out.get("paper") is True and out.get("pendingLimitEntry") is True, out
    assert out["tradeRecord"]["entryContext"]["sizing"] is None
    assert any("SIZING TELEMETRY LOST" in r.getMessage() for r in caplog.records)

  def test_a_thin_side_is_judged_on_the_pooled_family_and_told_which_row_did_it(self, tmp_path):
    """Continuation shorts (n=9) inherit the pooled stand-aside (t=0.78<1), and the refusal is the
    truthful one: positive net inside its own noise, with its t — never 'NO EDGE … coin-flip'."""
    tools, memory = _limit_entry_tools(tmp_path, _CONTINUATION_0924)
    out = _place(tools, side="sell")
    assert out.get("rejected") is True, out
    assert "t=0.78" in out["reason"]
    assert "NO EDGE" not in out["reason"] and "coin-flip" not in out["reason"]
    assert "continuation:short n=9 is unproven, judged on pooled continuation" in out["reason"]
    assert "t≥1" in out["hint"]
    # ...and the redirect says the longs are open on their own record, not "nothing is open".
    assert {"family": "continuation", "side": "long", "n": 40, "netOfCostPct": 1.10} in out["openFamilies"]["paying"]
    # The refused call is still evidence.
    assert len(memory.signal_probes(limit=0)) == 1

  def test_a_refused_proposal_is_the_evidence_that_can_re_open_a_stood_aside_family(self, tmp_path):
    """PHIL-1 (2026-09-25): the prompt now tells the model a refused proposal is still scored and is the
    ONLY way a stood-aside family re-opens — that claim must stay true. A stood-aside family (pooled
    n>=20, t<1, both sides judged pooled) gains one probe per refused call and places nothing."""
    board = {"cost_pct": 0.0014, "by_family": {"continuation": _row(30, 0.45, 0.50)}}
    tools, memory = _limit_entry_tools(tmp_path, board)
    assert memory.signal_probes(limit=0) == []
    out = _place(tools, side="sell")
    assert out.get("rejected") is True and "Stand aside" in out["reason"], out
    assert "tradeRecord" not in out and not out.get("paper")
    probes = memory.signal_probes(limit=0)
    assert len(probes) == 1 and probes[0]["entryContext"]["setupFamily"] == "continuation"
    import inspect
    from src import tools as tools_mod
    src = inspect.getsource(tools_mod.build_tools)
    assert src.index("memory.record_signal_probe(") < src.index(
      'if cfg.edge.stand_aside_no_edge_family and _stake_fl["standAside"]:')

  def test_the_side_with_its_own_record_trades_on_it(self, tmp_path):
    """Longs (n=40, t=2.75) are judged on their own row: not stood aside, full measured size — the
    same bet the pooled row (t=0.78) would have refused."""
    tools, memory = _limit_entry_tools(tmp_path, _CONTINUATION_0924)
    out = _place(tools, side="buy", entry_price=0.99, take_profit_price=1.05, stop_loss_price=0.97)
    assert out.get("paper") is True, out
    sizing = out["tradeRecord"]["entryContext"]["sizing"]
    assert sizing["familyExplore"] == pytest.approx(1.0)
    assert "judged on its own continuation:long record" in sizing["familyJudgedOn"]


class TestStandAsideMessageTellsTheTruth:
  """The refusal branches on WHY the stake is zero. 2026-09-24: nine refusals at net +0.41..+0.54%
  (t≈0.7-0.9) said 'NO EDGE … coin-flip minus fees', and the model repeated it in its own decline."""

  @staticmethod
  def _status(net, se, verdict, n=52, **extra):
    from src.edge import family_stake_status
    board = {"by_family": {"continuation": {"n": n, "net_of_cost_pct": net, "stderr_pct": se,
                                             "verdict": verdict}}}
    board.update(extra)
    return family_stake_status(board, "continuation")

  def test_a_positive_net_inside_its_noise_is_called_exactly_that(self):
    from src.tools import stand_aside_message
    msg = stand_aside_message(self._status(0.47, 0.60, "edge"), {"paying": [], "unproven": []})
    assert "t=0.78" in msg["reason"] and "SE 0.600%" in msg["reason"] and "+0.470%" in msg["reason"]
    assert "NO EDGE" not in msg["reason"] and "coin-flip" not in msg["reason"]
    assert "don't clear their round-trip cost" not in msg["reason"]
    assert "t=0.78" in msg["log"] and "SE=0.600%" in msg["log"] and "verdict=edge" in msg["log"]
    assert "once net exceeds its own SE (t≥1)" in msg["hint"]
    assert "beats cost" not in msg["hint"]

  def test_a_real_no_edge_keeps_the_no_edge_wording(self):
    from src.tools import stand_aside_message
    msg = stand_aside_message(self._status(-0.29, 0.10, "no edge"), {"paying": [], "unproven": []})
    assert "NO EDGE" in msg["reason"] and "coin-flip minus fees" in msg["reason"]
    assert "-0.290%" in msg["reason"]

  def test_the_hint_names_where_to_go_and_asks_for_honest_labels(self):
    from src.tools import stand_aside_message
    fams = {"paying": [{"family": "range_edge", "n": 25, "netOfCostPct": 0.4},
                       {"family": "continuation", "side": "long", "n": 40, "netOfCostPct": 1.1}],
            "unproven": [{"family": "breakout", "n": 7, "netOfCostPct": None}]}
    hint = stand_aside_message(self._status(0.47, 0.60, "edge"), fams)["hint"]
    assert "range_edge (n=25, +0.400%)" in hint
    assert "continuation long side only (n=40, +1.100%)" in hint
    assert "breakout (n=7)" in hint
    assert "Label honestly" in hint
    nowhere = stand_aside_message(self._status(0.47, 0.60, "edge"), {})["hint"]
    assert "standing down is a valid answer" in nowhere

  def test_a_refused_side_is_not_told_its_whole_playbook_is_refused(self):
    """2026-09-25 review: a refused continuation SHORT read 'Currently paying: continuation long side
    only … Re-proposing the same playbook will be refused again' — a hint that contradicts itself."""
    from src.edge import family_stake_status, open_families
    from src.tools import stand_aside_message
    board = {"cost_pct": 0.0014,
             "by_family": {"continuation": {"n": 60, "net_of_cost_pct": 0.22, "stderr_pct": 0.25, "verdict": "edge"}},
             "by_family_side": {"continuation": {
               "long": {"n": 40, "net_of_cost_pct": 0.60, "stderr_pct": 0.30, "verdict": "edge"},
               "short": {"n": 20, "net_of_cost_pct": -0.55, "stderr_pct": 0.40, "verdict": "no edge"}}}}
    st = family_stake_status(board, "continuation", side="short")
    assert st["standAside"] is True
    hint = stand_aside_message(st, open_families(board))["hint"]
    assert "continuation long side only" in hint
    assert "Re-proposing the same playbook will be refused again" not in hint
    assert "this side of the playbook (continuation shorts)" in hint
    pooled = stand_aside_message(self._status(0.47, 0.60, "edge"), {})["hint"]
    assert "Re-proposing the same playbook will be refused again" in pooled

  def test_the_order_path_takes_its_words_from_the_stake_status(self):
    """Wiring: the refusal and the log come from stand_aside_message over family_stake_status for the
    ORDER's side — not from a second, hand-rolled copy of the rule."""
    import inspect
    from src import tools
    src = inspect.getsource(tools.build_tools)
    assert '_stake_fl = family_stake_status(_signal_edge_fl, _family_fl or "other", side=_side_fl)' in src
    assert 'if cfg.edge.stand_aside_no_edge_family and _stake_fl["standAside"]:' in src
    assert "_refusal = stand_aside_message(_stake_fl, _open)" in src
    assert "measures NO EDGE (n=%s" not in src          # the old always-'NO EDGE' log line is gone
    for fn in ("family_size_factor(_signal_edge_fl", "family_explore_factor(\n      _signal_edge_fl"):
      assert fn in src
    assert src.count("side=_side_fl") >= 3


class TestSizingBreakdown:
  _KW = dict(
    equity_usd=73.1, eff_risk_frac=0.0075, vol_scale=1.0, soft=[1.0, 0.62, 0.9], quality=0.62,
    quality_floor=0.5, family_measured=1.0, family_explore=0.4, family_judged_on="judged on pooled x (n=3)",
    atr_scale=0.4, risk_scale=1.0, conc_scale=1.0,
    contract_stages={"raw": 3.6, "lotFloor": 3, "riskCap": 3, "heatCap": 2, "concentrationCap": 2},
    lot_floored=False, planned_max_loss_usd=0.21, gross_risk_usd=0.18,
  )

  def test_every_stage_and_factor_is_kept(self):
    from src.tools import sizing_breakdown
    out = sizing_breakdown(**self._KW)
    assert list(out["contractsByStage"]) == ["raw", "lotFloor", "riskCap", "heatCap", "concentrationCap"]
    assert out["contractsByStage"]["heatCap"] == 2        # the cap that bound is visible
    assert out["familyMeasured"] == 1.0 and out["familyExplore"] == 0.4   # both legs, not just the min
    assert out["qualityRaw"] == pytest.approx(0.62) and out["qualityFloorClamped"] is False
    assert out["riskFracTarget"] == pytest.approx(0.0075 * 0.4)
    assert out["riskFracActual"] == pytest.approx(0.21 / 73.1, rel=1e-4)
    assert out["grossRiskFracActual"] == pytest.approx(0.18 / 73.1, rel=1e-4)
    assert out["equityUsd"] == pytest.approx(73.1)

  def test_the_quality_floor_lift_is_flagged(self):
    from src.tools import sizing_breakdown
    kw = dict(self._KW, soft=[0.3, 1.0], quality=0.5)
    out = sizing_breakdown(**kw)
    assert out["qualityRaw"] == pytest.approx(0.3) and out["qualityFloorClamped"] is True

  @pytest.mark.parametrize("broken", [
    {"equity_usd": None}, {"equity_usd": 0.0}, {"atr_scale": float("nan")},
    {"contract_stages": {"raw": 3.6, "lotFloor": 3}},
  ])
  def test_a_missing_input_is_none_and_a_warning_never_an_exception(self, broken, caplog):
    import logging
    from src.tools import sizing_breakdown
    with caplog.at_level(logging.WARNING):
      assert sizing_breakdown(**dict(self._KW, **broken)) is None
    assert any("SIZING TELEMETRY LOST" in r.getMessage() for r in caplog.records)

  def test_the_entry_context_carries_it(self):
    import inspect
    from src import tools
    src = inspect.getsource(tools.build_tools)
    assert '"sizing": sizing_breakdown(' in src
    for stage in ("riskCap", "heatCap", "concentrationCap"):
      assert f'_contract_stages_fl["{stage}"] = contracts' in src


class TestEffectiveMinConfidenceIsShown:
  """The prompt said 0.65 while code enforced 0.75 in a hostile daily regime."""

  def test_hostile_regime_reports_the_raised_floor(self):
    from src.config import load_config
    from src.tools import entry_gate_summary
    cfg = load_config().regime
    cfg.throttle_enabled, cfg.caution_min_confidence = True, 0.75
    hostile = entry_gate_summary(0.65, "bearish", False, cfg)
    assert hostile["minConfidence"] == pytest.approx(0.75) and hostile["raisedByRegime"] is True
    assert "hostile" in hostile["note"]
    assert entry_gate_summary(0.65, "bullish", True, cfg)["minConfidence"] == pytest.approx(0.75)  # exhausted
    calm = entry_gate_summary(0.65, "neutral", False, cfg)
    assert calm["minConfidence"] == pytest.approx(0.65) and calm["raisedByRegime"] is False
    assert entry_gate_summary(None, "bearish", False, cfg)["minConfidence"] is None   # never raises

  def test_analysis_reports_the_floor_the_entry_path_enforces(self):
    import inspect
    from src import tools
    src = inspect.getsource(tools.build_tools)
    assert 'summary["entryGate"] = entry_gate_summary(' in src
    call = src[src.index('summary["entryGate"] = entry_gate_summary('):][:300]
    assert "cfg.trading.min_confidence" in call
    assert '_daily_gate_state[symbol]["daily_bias"]' in call
    assert '_daily_gate_state[symbol]["daily_exhausted"]' in call


# ── The execution map on the order path: probe stamps and the placement note (2026-09-25) ─────────────

_XM_MAP_0924 = {"byDistance": {
  "0.5-1": {"placed": 22, "filled": 9, "fillRate": 0.409, "ideas": 13, "medianMinToFill": 9.6,
            "fills": {"n": 33, "meanR": -0.156, "seR": 0.106}, "fillAdjustedR": -0.064},
  "marketable": {"placed": 4, "filled": 4, "fillRate": "insufficient", "ideas": 3, "medianMinToFill": 0.0,
                 "fills": {"n": 3, "meanR": "insufficient"}, "fillAdjustedR": None},
}}


class TestExecutionMapOnTheOrderPath:
  """The probe records every call's planned execution (so the map can score every depth), and the
  placement response places the new limit in the bot's own record. Both are telemetry: total."""

  def test_the_probe_carries_the_crossed_net_rr_the_atr_and_the_lease(self, tmp_path):
    from src.config import load_config
    from src.regime import net_reward_risk_ratio
    tools, memory = _limit_entry_tools(tmp_path, gate_extra={"intraday_atr_pct": 1.25})
    out = _place(tools)                             # sell 1.01 vs live 1.00, TP 0.95, SL 1.03
    assert out.get("paper") is True, out
    ctx = memory.signal_probes(limit=0)[0]["entryContext"]
    slip = load_config().trading.estimated_slippage_pct
    want = net_reward_risk_ratio("sell", 1.0, 0.95, 1.03, fee_rate=0.0006, slippage_rate=slip)
    assert ctx["crossedNetRr"] == pytest.approx(want)            # the SAME bracket priced at the live price
    assert ctx["atr15Pct"] == 1.25 and ctx["plannedEntry"] == pytest.approx(1.01)
    assert (ctx["plannedStop"], ctx["plannedTp"]) == (1.03, 0.95)
    assert ctx["leaseMin"] == pytest.approx(load_config().trading.entry_limit_expiry_minutes)

  def test_a_refused_call_still_records_its_execution_stamps(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path, _CONTINUATION_0924, gate_extra={"intraday_atr_pct": 1.25})
    assert _place(tools, side="sell").get("rejected") is True    # stood aside
    assert "crossedNetRr" in memory.signal_probes(limit=0)[0]["entryContext"]

  def test_the_placement_note_names_the_bucket_from_the_live_map(self, tmp_path):
    tools, _memory = _limit_entry_tools(tmp_path, gate_extra={"intraday_atr_pct": 1.25},
                                        edge_extra={"execution_map": _XM_MAP_0924})
    note = _place(tools)["note"]
    assert "this limit rests 0.80 ATR15 from price (0.5-1)" in note
    assert "your last 22 limits there filled 41% within the lease (9/22)" in note
    assert "fills there averaged -0.16R (n=33, SE 0.11)" in note
    assert "Crossed at the live price this bracket would net RR" in note

  def test_a_broken_stamp_never_blocks_the_probe_or_the_order(self, tmp_path, monkeypatch, caplog):
    """Only the STAMP's pricing (at the live 1.00) explodes; the RR gate (at the 1.01 limit) still runs."""
    import logging
    from src import tools as tools_mod
    real = tools_mod.net_reward_risk_ratio

    def _flaky(side, entry, *a, **k):
      if abs(float(entry) - 1.0) < 1e-12:
        raise RuntimeError("stamp exploded")
      return real(side, entry, *a, **k)

    monkeypatch.setattr(tools_mod, "net_reward_risk_ratio", _flaky)
    tools, memory = _limit_entry_tools(tmp_path, gate_extra={"intraday_atr_pct": 1.25},
                                       edge_extra={"execution_map": _XM_MAP_0924})
    with caplog.at_level(logging.WARNING):
      out = _place(tools)
    assert out.get("paper") is True and out.get("pendingLimitEntry") is True, out
    assert any("EXECUTION STAMP LOST" in r.getMessage() for r in caplog.records)
    ctx = memory.signal_probes(limit=0)[0]["entryContext"]           # the probe was still recorded...
    assert "crossedNetRr" not in ctx and "atr15Pct" not in ctx        # ...without the stamp
    assert "(0.5-1)" in out["note"] and "Crossed at the live price" not in out["note"]

  def test_a_broken_note_alone_leaves_the_order_and_its_note(self, tmp_path, monkeypatch, caplog):
    import logging
    from src import tools as tools_mod

    def _boom(*_a, **_k):
      raise RuntimeError("note exploded")

    monkeypatch.setattr(tools_mod, "execution_bucket_note", _boom)
    tools, _memory = _limit_entry_tools(tmp_path, gate_extra={"intraday_atr_pct": 1.25},
                                        edge_extra={"execution_map": _XM_MAP_0924})
    with caplog.at_level(logging.WARNING):
      out = _place(tools)
    assert out.get("paper") is True and out["note"].startswith("Futures limit sell placed"), out
    assert "Execution map" not in out["note"]
    assert any("EXECUTION MAP NOTE lost" in r.getMessage() for r in caplog.records)


class TestExecutionBucketNote:
  def test_marketable_and_thin_buckets_say_so_instead_of_a_number(self):
    from src.tools import execution_bucket_note
    text = execution_bucket_note(_XM_MAP_0924, "buy", 1.002, 1.0, 1.25, crossed_net_rr=1.4, rr_floor=1.5)
    assert "crosses the live price (marketable)" in text
    assert "filled 4/4 (too few to call a rate)" in text and "n=3, too few to average" in text
    assert "Crossed at the live price" not in text                 # it IS the cross
    empty = execution_bucket_note({}, "sell", 1.015, 1.0, 1.25)
    assert "(1-2)" in empty and "no recent limits of yours at this distance" in empty

  def test_never_raises(self, caplog):
    from src.tools import execution_bucket_note, probe_execution_stamp
    assert execution_bucket_note({"byDistance": {"0.5-1": {"placed": "x"}}}, "buy", 0.99, 1.0, 1.25) == ""
    assert execution_bucket_note(None, None, None, None, None) == ""
    assert probe_execution_stamp("buy", None, 1.0, 0.9, 1.2, fee_rate=0.0006, slippage_rate=0.0,
                                 atr_pct=1.0, lease_min=15) == {}


# ── Market state rides on every entry and probe (2026-09-25) ──────────────────────────────────────────

_MARKET = {"asOf": 1_790_300_000, "universe": 68, "breadth24": 0.07, "basketMedian24h": -6.2,
           "btc24h": -1.4, "btc72h": 2.1, "btcDailyAdx": 18.5, "btcDailyBias": "bullish"}


class TestEntriesCarryTheMarketState:
  def test_the_entry_and_its_probe_carry_the_loops_reading(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path, ctx_extra={"market_state": lambda: dict(_MARKET)})
    out = _place(tools)
    assert out.get("paper") is True, out
    assert out["tradeRecord"]["entryContext"]["marketState"] == _MARKET
    probe = memory.signal_probes(limit=0)[0]["entryContext"]
    assert probe["marketState"] == _MARKET

  def test_no_reading_records_none_and_the_order_still_goes(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path, ctx_extra={"market_state": lambda: None})
    out = _place(tools)
    assert out.get("paper") is True and out["tradeRecord"]["entryContext"]["marketState"] is None
    assert "marketState" not in memory.signal_probes(limit=0)[0]["entryContext"]

  def test_a_broken_reader_is_a_warning_never_a_refusal(self, tmp_path, caplog):
    import logging

    def _boom():
      raise RuntimeError("cache exploded")

    tools, memory = _limit_entry_tools(tmp_path, ctx_extra={"market_state": _boom})
    with caplog.at_level(logging.WARNING):
      out = _place(tools)
    assert out.get("paper") is True, out
    assert out["tradeRecord"]["entryContext"]["marketState"] is None
    assert any("MARKET STATE" in r.getMessage() for r in caplog.records)
    assert len(memory.signal_probes(limit=0)) == 1


# ── Stop listing and the data-tool resolver speak the model's symbol format (2026-09-25) ─────────────


class _StrictFutures:
  """KuCoin futures fake that, like the live API, rejects any non-contract symbol with 100003."""

  def __init__(self, stops=(), contracts=("DASHUSDTM", "WIFUSDTM", "KCSUSDTM", "SPXUSDTM")):
    self.stops = [dict(s) for s in stops]
    self.contracts = [{"symbol": c} for c in contracts]
    self.stop_calls, self.funding_calls = [], []

  def list_stop_orders(self, status="active", symbol=None):
    self.stop_calls.append({"status": status, "symbol": symbol})
    if symbol is not None and not str(symbol).endswith("USDTM"):
      raise RuntimeError("Kucoin Futures API error: 100003 Contract parameter invalid")
    return [s for s in self.stops if symbol is None or s["symbol"] == symbol]

  def list_active_contracts(self):
    return self.contracts

  def get_funding_rate(self, fsym):
    self.funding_calls.append(fsym)
    if not fsym.endswith("USDTM"):
      raise RuntimeError("Kucoin Futures API error: 100003 Contract parameter invalid")
    return {"symbol": fsym, "value": 0.0001}


def _data_tools(tmp_path, futures, *, allowed=("DASH-USDT", "SPX-USDT")):
  from src.config import load_config
  from src.memory import MemoryStore
  from src.tools import build_tools

  cfg = load_config()
  cfg.kucoin_futures.enabled = True
  snapshot = SimpleNamespace(tickers={}, coins=[], futures_positions=[], futures_pending_orders=[],
                             paper_trading=True, trading_restricted=False, restriction_reason="")
  ctx = SimpleNamespace(
    cfg=cfg, kucoin=None, kucoin_futures=futures, memory=MemoryStore(str(tmp_path / "d.json")),
    snapshot=snapshot, allowed_symbols=set(allowed), balances_by_currency={},
    fees={"futures_taker": 0.0006}, _daily_gate_state={}, _futures_margin_mode="cross",
    _apply_cross_leverage=lambda *a, **k: None, _btc_daily_bias=lambda: "neutral",
    _edge_state=lambda: {}, _fee_adjusted_breakeven=lambda *a, **k: 0.0,
    _get_contract_spec=lambda fsym: None, _repair_allowed_symbol=lambda s: None,
    _spot_position_info=lambda *a, **k: None, _spot_position_size=lambda *a, **k: 0.0,
    _stop_distance_ok=lambda *a, **k: (True, None), safety_state=None, entry_token=None,
  )
  return build_tools(ctx)


def _invoke_tool(tool, **args):
  import asyncio
  import json
  from agents.tool_context import ToolContext
  raw = json.dumps(args)
  return asyncio.run(tool.on_invoke_tool(
    ToolContext(context=None, tool_name=tool.name, tool_call_id="t1", tool_arguments=raw), raw))


_STOPS = [
  {"id": "s1", "symbol": "DASHUSDTM", "stop": "down"},
  {"id": "s2", "symbol": "DASHUSDTM", "stop": "up"},
  {"id": "s3", "symbol": "WIFUSDTM", "stop": "down"},
  {"id": "s4", "symbol": "KCSUSDTM", "stop": "down"},
]


class TestStopListingSymbolFormat:
  def test_spot_format_lists_only_that_contracts_stops_from_an_unfiltered_call(self, tmp_path):
    futures = _StrictFutures(_STOPS)
    out = _invoke_tool(_data_tools(tmp_path, futures).list_futures_stop_orders, symbol="DASH-USDT")
    assert "error" not in out, out
    assert [o["id"] for o in out["orders"]] == ["s1", "s2"]
    assert out["symbol"] == "DASHUSDTM" and out["requested"] == "DASH-USDT"
    # The call site: KuCoin was asked WITHOUT a symbol (the filter is ours), exactly once.
    assert futures.stop_calls == [{"status": "active", "symbol": None}]

  def test_a_coin_outside_the_universe_still_lists_via_the_contract_list(self, tmp_path):
    futures = _StrictFutures(_STOPS)
    out = _invoke_tool(_data_tools(tmp_path, futures).list_futures_stop_orders, symbol="KCS-USDT")
    assert [o["id"] for o in out["orders"]] == ["s4"] and out["symbol"] == "KCSUSDTM"

  def test_contract_format_and_no_symbol_still_work(self, tmp_path):
    futures = _StrictFutures(_STOPS)
    tools = _data_tools(tmp_path, futures)
    assert [o["id"] for o in _invoke_tool(tools.list_futures_stop_orders, symbol="WIFUSDTM")["orders"]] == ["s3"]
    assert len(_invoke_tool(tools.list_futures_stop_orders)["orders"]) == 4

  def test_an_unmappable_symbol_errors_without_calling_the_exchange(self, tmp_path):
    futures = _StrictFutures(_STOPS)
    out = _invoke_tool(_data_tools(tmp_path, futures).list_futures_stop_orders, symbol="NOPE-USDT")
    assert "error" in out and out["requested"] == "NOPE-USDT"
    assert futures.stop_calls == []


class TestDataToolResolverFallback:
  def test_a_spot_listed_coin_outside_the_universe_resolves_for_data_tools(self, tmp_path):
    futures = _StrictFutures()
    out = _invoke_tool(_data_tools(tmp_path, futures).fetch_funding_rate, symbol="KCS-USDT")
    assert out.get("symbol") == "KCSUSDTM" and "error" not in out, out
    assert futures.funding_calls == ["KCSUSDTM"]

  def test_the_order_path_still_refuses_it(self, tmp_path):
    tools = _data_tools(tmp_path, _StrictFutures())
    out = _invoke_tool(tools.place_futures_limit_order, symbol="KCS-USDT", side="buy", notional_usd=10.0,
                       entry_price=1.0, confidence=0.9, take_profit_price=1.1, stop_loss_price=0.95,
                       setup_family="continuation")
    assert out.get("error") == "Unsupported symbol" and out.get("requested") == "KCS-USDT"

  def test_a_symbol_with_no_live_contract_is_still_unknown(self, tmp_path):
    futures = _StrictFutures()
    out = _invoke_tool(_data_tools(tmp_path, futures).fetch_funding_rate, symbol="NOPE-USDT")
    assert out.get("error") == "Unknown symbol 'NOPE-USDT'" and futures.funding_calls == []


# ── analyze_market_context remembers its data-quality refusals (2026-09-25) ──────────────────────────


class _CandleFutures:
  """Futures fake serving synthetic, fresh klines ([ts_ms, open, HIGH, LOW, close, vol, turnover]).

  ``drop`` maps a granularity (minutes) to bar offsets (counted back from the newest closed bar) to
  leave out, which makes a real gap."""

  def __init__(self, drop=None):
    self.drop = drop or {}

  def get_candles(self, symbol, granularity=1, start_at=None, end_at=None):
    import math as _m
    step = granularity * 60
    end_s = int(end_at / 1000)
    first = (int(start_at / 1000) // step) * step
    newest_closed = (end_s // step - 1) * step
    rows = []
    t = first
    while t <= newest_closed:
      back = (newest_closed - t) // step
      if back not in self.drop.get(granularity, ()):
        c = 100.0 + _m.sin(t / 7200.0)
        o = c - 0.05
        rows.append([t * 1000, o, max(o, c) + 0.2, min(o, c) - 0.2, c, 1000.0, 100000.0])
      t += step
    return rows

  def get_funding_rate(self, fsym):
    raise RuntimeError("not needed")

  def get_contract_detail(self, fsym):
    raise RuntimeError("not needed")

  def get_mark_price(self, fsym):
    raise RuntimeError("not needed")


def _analysis_tools(tmp_path, futures, memory=None, *, ctx_extra=None, gate=None):
  """build_tools over the analysis fakes. ``gate`` (optional) is the run's shared _daily_gate_state dict,
  so a test can read what analysis wrote into it; ``ctx_extra`` overrides any ctx attribute."""
  from src.config import load_config
  from src.memory import MemoryStore
  from src.tools import build_tools

  cfg = load_config()
  cfg.kucoin_futures.enabled = True
  cfg.trading.max_atr_pct_for_entry = 50.0
  cfg.regime.alt_long_block_enabled = True
  cfg.regime.alt_majors = ("BTC", "ETH")
  memory = memory or MemoryStore(str(tmp_path / "a.json"))
  snapshot = SimpleNamespace(tickers={}, coins=[], futures_positions=[], futures_pending_orders=[],
                             paper_trading=True, trading_restricted=False, restriction_reason="")
  ctx = SimpleNamespace(
    cfg=cfg, kucoin=None, kucoin_futures=futures, memory=memory, snapshot=snapshot,
    allowed_symbols={"TAKE-USDT"}, balances_by_currency={}, fees={"futures_taker": 0.0006},
    _daily_gate_state=gate if gate is not None else {}, _futures_margin_mode="cross",
    _apply_cross_leverage=lambda *a, **k: None,
    _btc_daily_bias=lambda: "neutral", _edge_state=lambda: {}, _fee_adjusted_breakeven=lambda *a, **k: 0.0,
    _get_contract_spec=lambda fsym: None, _repair_allowed_symbol=lambda s: None,
    _spot_position_info=lambda *a, **k: None, _spot_position_size=lambda *a, **k: 0.0,
    _stop_distance_ok=lambda *a, **k: (True, None), safety_state=None, entry_token=None,
  )
  for key, value in (ctx_extra or {}).items():
    setattr(ctx, key, value)
  return build_tools(ctx), memory


class _FundedCandleFutures(_CandleFutures):
  """The analysis fake with a live funding payload (a 1h contract), so the funding-clock block runs."""

  def __init__(self, *a, **k):
    import time as _t
    super().__init__(*a, **k)
    self.next_ms = (int(_t.time()) // 3600 + 1) * 3600 * 1000

  def get_funding_rate(self, fsym):
    return {"symbol": fsym, "value": -0.00196, "predictedValue": -0.0015, "granularity": 3600000,
            "fundingTime": self.next_ms}


class TestAnalysisRecordsTheFundingClock:
  """T3: the analysis funding block sits inside try/except-pass and every fake raised on it, so field
  swaps and unit slips (fundingIntervalHours x1000, next/interval swapped) survived the whole suite —
  and that stamp is the carry hold's fallback clock and the only clock the carry replays use."""

  def test_the_clock_reaches_the_output_the_gate_state_and_the_entry_stamp(self, tmp_path):
    from src.tools import entry_funding_stamp
    gate: dict = {}
    futures = _FundedCandleFutures()
    tools, _ = _analysis_tools(tmp_path, futures, gate=gate)
    out = _invoke_tool(tools.analyze_market_context, symbol="TAKE-USDT")
    assert out["dataQuality"]["ok"] is True, out
    assert out["futures"]["fundingRate"] == pytest.approx(-0.00196)
    assert out["futures"]["fundingIntervalHours"] == pytest.approx(1.0)
    assert out["futures"]["fundingSetup"] is not None
    g = gate["TAKE-USDT"]
    assert g["funding_interval_sec"] == pytest.approx(3600.0)
    assert g["funding_next_ts"] == pytest.approx(futures.next_ms / 1000)
    assert entry_funding_stamp(g) == {"rate": pytest.approx(-0.00196), "intervalSec": pytest.approx(3600.0),
                                      "nextSettlementTs": pytest.approx(futures.next_ms / 1000)}


class TestAnalysisFailuresAreRemembered:
  def test_a_candle_gap_is_recorded_with_a_retry_time_from_its_own_evidence(self, tmp_path):
    import time as _t
    # 1h bars 10..12 back are missing: the newest gap sits after the bar 13 hours back.
    tools, memory = _analysis_tools(tmp_path, _CandleFutures(drop={60: (10, 11, 12)}))
    before = _t.time()
    out = _invoke_tool(tools.analyze_market_context, symbol="TAKE-USDT")
    assert "candle gap" in out.get("error", ""), out
    newest_closed = (int(before) // 3600 - 1) * 3600
    gap_bar = newest_closed - 13 * 3600
    expected = gap_bar + 50 * 3600          # it ages out once that bar leaves the 50-bar window
    assert abs(out["retryAfter"] - expected) <= 3600
    failure = memory.analysis_failures()["TAKE-USDT"]
    assert "1hour" in failure["reason"] and "candle gap" in failure["reason"]
    assert failure["retryAfter"] == out["retryAfter"]

  def test_a_clean_analysis_clears_the_old_failure(self, tmp_path):
    import time as _t
    tools, memory = _analysis_tools(tmp_path, _CandleFutures())
    memory.record_analysis_failure("TAKE-USDT", reason="1hour: 6 candle gap(s)", retry_after=_t.time() + 9000)
    out = _invoke_tool(tools.analyze_market_context, symbol="TAKE-USDT")
    assert out.get("dataQuality", {}).get("ok") is True, out
    assert "TAKE-USDT" not in memory.analysis_failures()

  def test_a_broken_store_never_breaks_the_analysis(self, tmp_path, caplog):
    import logging

    class _NoWrite:
      def __getattr__(self, name):
        if name in ("record_analysis_failure", "clear_analysis_failure"):
          def _boom(*a, **k):
            raise RuntimeError("disk full")
          return _boom
        raise AttributeError(name)

    tools, _ = _analysis_tools(tmp_path, _CandleFutures(drop={60: (10,)}), memory=_NoWrite())
    with caplog.at_level(logging.WARNING):
      out = _invoke_tool(tools.analyze_market_context, symbol="TAKE-USDT")
    assert "candle gap" in out.get("error", "")
    assert any("ANALYSIS FAILURE not recorded" in r.getMessage() for r in caplog.records)


# ── Gate scoreboard telemetry: every refusal says which gate, and the scored ones leave a probe ──────


def _limit_impl_prefix_node():
  """The AST of _place_futures_limit_order_impl and the line of its signal-probe call."""
  import ast
  from pathlib import Path
  src_text = Path(__file__).resolve().parents[1].joinpath("src", "tools.py").read_text()
  tree = ast.parse(src_text)
  impl = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_place_futures_limit_order_impl")
  probe_line = min(
    n.lineno for n in ast.walk(impl)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    and n.func.attr == "record_signal_probe"
  )
  return impl, probe_line


class TestEveryPreProbeRefusalNamesItsGate:
  """The directional gates return BEFORE the signal probe, so a refusal left no evidence at all
  (09-22..24: 8 hard refusals, 0 probes; the opposing-daily branch did not even log). Every such return
  now carries a 'gate' code — scored (a probe is recorded) or structural (nothing is) — asserted on the
  AST, so a gate added tomorrow is covered by construction rather than by a log marker someone must
  remember to add."""

  def test_every_rejected_return_before_the_probe_carries_a_known_gate(self):
    import ast
    from src.memory import SCORED_GATES, STRUCTURAL_REFUSALS
    impl, probe_line = _limit_impl_prefix_node()
    seen = []
    for node in ast.walk(impl):
      if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Dict) and node.lineno < probe_line):
        continue
      keys = {k.value: v for k, v in zip(node.value.keys, node.value.values) if isinstance(k, ast.Constant)}
      rejected = keys.get("rejected")
      if not (isinstance(rejected, ast.Constant) and rejected.value is True):
        continue
      gate = keys.get("gate")
      assert isinstance(gate, ast.Constant), f"refusal at tools.py:{node.lineno} carries no 'gate' code"
      assert gate.value in set(SCORED_GATES) | set(STRUCTURAL_REFUSALS), (node.lineno, gate.value)
      seen.append(gate.value)
    assert len(seen) >= 25
    # Every gate the brief names is actually wired to a refusal (and so can be scored).
    for code in ("anti_fomo", "daily_opposing", "h1_align", "tf_conflict", "correlation", "move_24h",
                 "bench", "vol_limit", "confidence_floor"):
      assert code in seen, code
    assert not set(SCORED_GATES) & set(STRUCTURAL_REFUSALS)

  def test_the_silent_branches_now_log(self):
    import inspect
    from src import tools
    src = inspect.getsource(tools.build_tools)
    assert 'logger.warning("DAILY GATE BLOCK: futures limit' in src
    assert 'logger.warning("24H MOVE BLOCK: futures limit' in src

  def test_the_tool_is_the_recording_wrapper_around_the_body(self):
    """Wiring: the model-facing tool calls the body and then the recorder, and returns the body's
    result object itself."""
    import inspect
    from src import tools
    src = inspect.getsource(tools.build_tools)
    wrapper = src[src.index("  async def place_futures_limit_order("):]
    wrapper = wrapper[:wrapper.index("\n  async def ", 10)]
    assert "result = await _place_futures_limit_order_impl(" in wrapper
    assert "_record_gate_refusal(result, symbol, side, setup_family, confidence)" in wrapper
    assert wrapper.rstrip().endswith("return result")
    assert "Place a futures limit entry order at a technically derived target price." in wrapper


def _gate_tools(tmp_path, monkeypatch, *, move_cap=0.0, regime=None, **kw):
  """The limit-entry harness with the few extra knobs these tests pin (24h cap, the alt-long veto, and
  any ``regime`` config attribute)."""
  import src.config as config_mod
  captured = []
  real = config_mod.load_config

  def _capture():
    cfg = real()
    captured.append(cfg)
    return cfg

  monkeypatch.setattr(config_mod, "load_config", _capture)
  tools, memory = _limit_entry_tools(tmp_path, **kw)
  cfg = captured[-1]
  cfg.trading.max_24h_volatility_pct = move_cap
  cfg.regime.alt_long_block_enabled = True
  cfg.regime.alt_majors = ("BTC", "ETH")
  for key, value in (regime or {}).items():
    setattr(cfg.regime, key, value)
  return tools, memory


_LONG_ORDER = dict(side="buy", entry_price=0.99, take_profit_price=1.05, stop_loss_price=0.97, confidence=0.70)


class TestGateRefusalsLeaveAProbe:
  def test_a_gated_refusal_records_exactly_one_gate_probe_and_no_signal_probe(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path, gate_extra={"intraday_bias_1h": "bearish"})
    out = _place(tools, **_LONG_ORDER)
    assert out == {"rejected": True, "gate": "h1_align", "reason": "1h trend is bearish — buy entry blocked",
                   "hint": out["hint"]}
    rows = memory.gate_probes()
    assert len(rows) == 1
    ctx = rows[0]["entryContext"]
    assert rows[0]["symbol"] == "SPX-USDT" and ctx["gateProbe"] == "refusal" and ctx["gate"] == "h1_align"
    assert ctx["positionSide"] == "long" and ctx["marketPriceAtSignal"] == 1.0
    assert ctx["priceSource"] == "futures_mark" and ctx["setupFamily"] == "continuation"
    assert ctx["model"] == "gpt-test" and ctx["confidence"] == pytest.approx(0.70)
    assert ctx["regime"]["intraday_bias_1h"] == "bearish"
    assert memory.signal_probes(limit=0) == []          # never into the family verdicts

  def test_the_opposing_daily_branch_logs_and_records(self, tmp_path, caplog):
    import logging
    tools, memory = _limit_entry_tools(tmp_path, gate_extra={"daily_bias": "bearish", "daily_bias_raw": "bearish"})
    with caplog.at_level(logging.WARNING):
      out = _place(tools, **_LONG_ORDER)
    assert out["gate"] == "daily_opposing" and out["rejected"] is True
    assert any("DAILY GATE BLOCK: futures limit buy SPX-USDT" in r.getMessage() for r in caplog.records)
    assert [r["entryContext"]["gate"] for r in memory.gate_probes()] == ["daily_opposing"]

  def test_a_structural_refusal_records_nothing_and_fetches_nothing(self, tmp_path, monkeypatch):
    from src import tools as tools_mod
    fetches = []
    real = tools_mod.live_entry_price_sourced

    def _spy(*a, **k):
      fetches.append(a[-1])
      return real(*a, **k)

    monkeypatch.setattr(tools_mod, "live_entry_price_sourced", _spy)
    tools, memory = _limit_entry_tools(tmp_path)
    out = _place(tools, take_profit_price=None)
    assert out["gate"] == "bracket_missing"
    assert memory.gate_probes() == [] and fetches == []     # not even a price lookup

  def test_an_admitted_call_records_no_gate_probe(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path)
    out = _place(tools)
    assert out.get("paper") is True, out
    assert memory.gate_probes() == [] and len(memory.signal_probes(limit=0)) == 1

  @pytest.mark.parametrize("breakage", ["store", "price"])
  def test_a_telemetry_failure_returns_the_refusal_unchanged(self, tmp_path, monkeypatch, caplog, breakage):
    """An entry-path NameError once froze every verdict for 2.6 days. A broken recorder must cost the
    probe (WARNING) and nothing else — the model gets the very same refusal."""
    import logging
    from src import tools as tools_mod
    clean_tools, _ = _limit_entry_tools(tmp_path / "clean", gate_extra={"intraday_bias_1h": "bearish"})
    expected = _place(clean_tools, **_LONG_ORDER)
    tools, memory = _limit_entry_tools(tmp_path / "broken", gate_extra={"intraday_bias_1h": "bearish"})

    def _boom(*_a, **_k):
      raise RuntimeError("telemetry exploded")

    if breakage == "store":
      monkeypatch.setattr(memory, "record_gate_probe", _boom)
    else:
      monkeypatch.setattr(tools_mod, "live_entry_price_sourced", _boom)
    with caplog.at_level(logging.WARNING):
      out = _place(tools, **_LONG_ORDER)
    assert out == expected
    assert any("GATE PROBE LOST" in r.getMessage() for r in caplog.records)
    assert memory.gate_probes() == []


class TestGateStatePredicateMatchesTheOrderPath:
  """directional_gates_against is a SECOND statement of the gate conditions (it has to be: a state
  reading has no call to run the gates on). Pinned to the real order path — the first gate it names is
  the gate a plain call (no hatch family, confidence under every hatch bar) is refused with."""

  @pytest.mark.parametrize("gate_extra, side", [
    ({"daily_bias": "bullish", "daily_bias_raw": "bullish", "daily_exhausted": True}, "buy"),
    # Exhausted daily, the side AGAINST it: neither anti_fomo (not a continuation) nor daily_opposing
    # (that branch only runs on a non-exhausted daily) — the order path lets it through.
    ({"daily_bias": "bullish", "daily_bias_raw": "bullish", "daily_exhausted": True}, "sell"),
    ({"daily_bias": "neutral", "daily_bias_raw": "bearish", "daily_exhausted": True}, "sell"),
    ({"daily_bias": "bearish", "daily_bias_raw": "bearish"}, "buy"),
    ({"daily_bias": "bullish", "daily_bias_raw": "bullish"}, "sell"),
    ({"intraday_bias_1h": "bearish"}, "buy"),
    ({"intraday_bias_1h": "bullish"}, "sell"),
    ({"timeframe_conflict": True, "intraday_bias_15m": "bearish"}, "buy"),
    ({"timeframe_conflict": True, "intraday_bias_15m": "bullish"}, "buy"),
    ({"daily_bias": "bearish", "daily_bias_raw": "bearish", "intraday_bias_1h": "bearish"}, "buy"),
    ({"daily_bias": "bullish", "daily_bias_raw": "bullish", "daily_exhausted": True,
      "intraday_bias_1h": "bearish", "timeframe_conflict": True, "intraday_bias_15m": "bearish"}, "buy"),
    ({}, "buy"),
    ({}, "sell"),
  ])
  def test_first_named_gate_is_the_refusing_gate(self, tmp_path, monkeypatch, gate_extra, side):
    from src.memory import DIRECTIONAL_GATES
    from src.tools import directional_gates_against
    tools, _ = _gate_tools(tmp_path, monkeypatch, gate_extra=gate_extra)
    order = dict(_LONG_ORDER) if side == "buy" else dict(
      side="sell", entry_price=1.01, take_profit_price=0.95, stop_loss_price=1.03, confidence=0.70)
    out = _place(tools, **order)
    base = {"daily_bias": "neutral", "daily_bias_raw": "neutral", "daily_exhausted": False,
            "intraday_bias_15m": "neutral", "intraday_bias_1h": "neutral", "timeframe_conflict": False}
    base.update(gate_extra)
    predicted = directional_gates_against(base, side)
    if predicted:
      assert out.get("gate") == predicted[0], (predicted, out)
    else:
      assert out.get("gate") not in DIRECTIONAL_GATES, out

  def test_the_symbol_level_gates(self, tmp_path, monkeypatch):
    import time as _t
    from src.tools import directional_gates_against
    tools, _ = _gate_tools(tmp_path, monkeypatch, move_cap=25.0, gate_extra={"price_change_24h_pct": 40.0})
    assert _place(tools, **_LONG_ORDER)["gate"] == "move_24h"
    assert directional_gates_against({}, "sell", move_24h_extreme=True) == ["move_24h"]
    tools, _ = _gate_tools(tmp_path / "b", monkeypatch, edge_extra={"bench": {"SPX-USDT": _t.time() + 3600}})
    assert _place(tools, **_LONG_ORDER)["gate"] == "bench"
    assert directional_gates_against({}, "buy", benched=True) == ["bench"]
    tools, _ = _gate_tools(tmp_path / "c", monkeypatch, ctx_extra={"_btc_daily_bias": lambda *a, **k: "bearish"})
    assert _place(tools, **_LONG_ORDER)["gate"] == "correlation"
    assert directional_gates_against({}, "buy", correlation_blocks=True) == ["correlation"]
    assert directional_gates_against({}, "sell", correlation_blocks=True) == []   # the veto is on alt longs
    assert directional_gates_against({"intraday_bias_1h": "bearish"}, "hold") == []

  def test_the_state_reading_uses_the_order_paths_own_closures(self):
    import inspect
    from src import tools
    src = inspect.getsource(tools.build_tools)
    fn = src[src.index("  def _gates_against_now("):src.index("  def _record_gate_state(")]
    # The correlation reading PEEKS at the run's BTC bias and applies the order path's own predicate; it
    # must never call the caching `_btc_daily_bias()` (S1, 2026-09-25 review).
    assert "_btc_bias_peek()" in fn and "block_alt_long_in_btc_downtrend(" in fn
    code = "\n".join(l for l in fn.splitlines() if not l.strip().startswith("#"))
    assert "_btc_daily_bias()" not in code and "_alt_long_is_blocked(" not in code
    assert "_extreme_24h_move(symbol) is not None" in fn
    assert '_edge_state().get("bench")' in fn
    assert "directional_gates_against(" in fn


class _MarkedCandleFutures(_CandleFutures):
  """The analysis fake with a live futures mark, so a clean analysis records its gate state."""

  def get_mark_price(self, fsym):
    return {"symbol": fsym, "value": 100.5, "indexPrice": 100.4}


class TestAnalysisRecordsTheGateState:
  def test_one_row_per_side_matching_the_predicate_then_deduped(self, tmp_path):
    from src.tools import directional_gates_against
    tools, memory = _analysis_tools(tmp_path, _MarkedCandleFutures())
    out = _invoke_tool(tools.analyze_market_context, symbol="TAKE-USDT")
    assert out.get("dataQuality", {}).get("ok") is True, out
    rows = memory.gate_probes()
    assert sorted(r["entryContext"]["positionSide"] for r in rows) == ["long", "short"]
    for row in rows:
      ctx = row["entryContext"]
      assert ctx["gateProbe"] == "state" and ctx["marketPriceAtSignal"] == 100.5
      assert ctx["priceSource"] == "futures_mark"
      side = "buy" if ctx["positionSide"] == "long" else "sell"
      assert ctx["gates"] == directional_gates_against(ctx["regime"], side)
    assert memory.signal_probes(limit=0) == []
    _invoke_tool(tools.analyze_market_context, symbol="TAKE-USDT")
    assert len(memory.gate_probes()) == 2                 # same window: not recorded twice

  def test_the_telemetry_never_fills_the_btc_bias_the_live_gate_decides_on(self, tmp_path):
    """S1 (2026-09-25 review): `_btc_daily_bias` caches its FIRST answer for the run — before BTC is
    analysed, a raw forming-bar spot fetch. The gate-state telemetry called it at the end of every
    analysis, so analysing an alt before BTC froze the raw bias into the cache and the LIVE correlation
    veto then refused alt longs the closed-bar gate would allow. The telemetry must only peek."""
    cache: dict = {}
    calls = []

    def _deciding_bias():                     # the agent closure's semantics: fetch once, then cached
      calls.append(1)
      cache.setdefault("v", "bearish")
      return cache["v"]

    tools, memory = _analysis_tools(tmp_path, _MarkedCandleFutures(), ctx_extra={
      "_btc_daily_bias": _deciding_bias, "_btc_daily_bias_peek": lambda: cache.get("v")})
    out = _invoke_tool(tools.analyze_market_context, symbol="TAKE-USDT")
    assert out.get("dataQuality", {}).get("ok") is True, out
    assert calls == [] and cache == {}               # nothing decided on the live gate's behalf
    longs = [r for r in memory.gate_probes() if r["entryContext"]["positionSide"] == "long"]
    assert len(longs) == 1 and "correlation" not in longs[0]["entryContext"]["gates"]   # unknown -> not blocked

  def test_a_known_bearish_btc_is_recorded_as_the_correlation_gate(self, tmp_path):
    tools, memory = _analysis_tools(tmp_path, _MarkedCandleFutures(), ctx_extra={
      "_btc_daily_bias": lambda: (_ for _ in ()).throw(AssertionError("telemetry must not decide")),
      "_btc_daily_bias_peek": lambda: "bearish"})
    out = _invoke_tool(tools.analyze_market_context, symbol="TAKE-USDT")
    assert out.get("dataQuality", {}).get("ok") is True, out
    by_side = {r["entryContext"]["positionSide"]: r["entryContext"]["gates"] for r in memory.gate_probes()}
    assert "correlation" in by_side["long"] and "correlation" not in by_side["short"]

  def test_the_agent_peek_is_cache_only_and_wired(self):
    import inspect
    from src import agent as _agent_mod
    src = inspect.getsource(_agent_mod.run_trading_agent)
    peek = src[src.index("  def _btc_daily_bias_peek()"):src.index("  _edge_cache: Dict[str, Any] = {}")]
    code = "\n".join(l for l in peek.splitlines() if not l.strip().startswith(("#", '"')))
    assert "get_candles" not in code and '_btc_bias_cache["v"] =' not in code
    assert "_btc_daily_bias_peek=_btc_daily_bias_peek," in src

  def test_no_mark_no_row_and_a_warning(self, tmp_path, caplog):
    import logging
    tools, memory = _analysis_tools(tmp_path, _CandleFutures())
    with caplog.at_level(logging.WARNING):
      out = _invoke_tool(tools.analyze_market_context, symbol="TAKE-USDT")
    assert out.get("dataQuality", {}).get("ok") is True
    assert memory.gate_probes() == []
    assert any("GATE STATE PROBE skipped" in r.getMessage() for r in caplog.records)

  def test_a_broken_store_never_breaks_the_analysis(self, tmp_path, monkeypatch, caplog):
    import logging
    tools, memory = _analysis_tools(tmp_path, _MarkedCandleFutures())

    def _boom(*_a, **_k):
      raise RuntimeError("disk full")

    monkeypatch.setattr(memory, "record_gate_state_probes", _boom)
    with caplog.at_level(logging.WARNING):
      out = _invoke_tool(tools.analyze_market_context, symbol="TAKE-USDT")
    assert out.get("dataQuality", {}).get("ok") is True and "futures" in out
    assert any("GATE STATE PROBE LOST" in r.getMessage() for r in caplog.records)


def test_the_gate_scoreboard_never_reaches_the_trading_prompt():
  """Report-only by design: a 'gate X's refusals pay' line in the model's state invites relabelling a
  call past the declared-setup hatches. Dashboard, Supervisor and the log get it; the agent never does."""
  from pathlib import Path
  root = Path(__file__).resolve().parents[1] / "src"
  agent_src = (root / "agent.py").read_text()
  for token in ("gate_scoreboard", "gateScoreboard", "gate_probes", "gate_state_days"):
    assert token not in agent_src, token


class TestHatchAdmissionsAreStamped:
  def test_a_declared_playbook_past_the_1h_gate_is_stamped_on_its_probe(self, tmp_path, monkeypatch):
    tools, memory = _gate_tools(
      tmp_path, monkeypatch, gate_extra={"intraday_bias_1h": "bearish"},
      regime={"declared_setups_enabled": True, "declarable_setup_families": ("breakout", "range_edge")},
    )
    out = _place(tools, setup_family="breakout", **_LONG_ORDER)
    assert out.get("gate") != "h1_align", out
    probes = memory.signal_probes(limit=0)
    assert probes[0]["entryContext"]["gatesPassed"] == [{"gate": "h1_align", "hatch": "declared"}]
    assert memory.gate_probes() == []

  def test_a_plain_admission_is_stamped_as_meeting_no_gate(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path)
    _place(tools)
    assert memory.signal_probes(limit=0)[0]["entryContext"]["gatesPassed"] == []

  def test_every_hatch_branch_appends_its_stamp(self):
    import inspect
    from src import tools
    src = inspect.getsource(tools.build_tools)
    body = src[src.index("  async def _place_futures_limit_order_impl("):src.index("memory.record_signal_probe(")]
    markers = [i for i in range(len(body)) if body.startswith("ALLOWED: futures limit", i)
               or body.startswith("DEADLOCK BREAK: futures limit", i)
               or body.startswith("RELATIVE-STRENGTH LONG: futures limit", i)]
    assert len(markers) >= 10
    for i in markers:
      assert "_gates_passed_fl.append(" in body[i:i + 500], body[i - 80:i + 80]
    probe_call = src[src.index("memory.record_signal_probe("):]
    assert "gates_passed=_gates_passed_fl" in probe_call[:probe_call.index("\n      )\n")]
