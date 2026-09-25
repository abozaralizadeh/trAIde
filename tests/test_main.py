import pytest

from types import SimpleNamespace

from src.main import (
  _adaptive_agent_cooldown,
  _adaptive_price_trigger_threshold,
  _agent_made_a_move,
  _close_event_id,
  _crossed_auto_triggers,
  _expired_bot_entry_orders,
  _fetch_futures,
  _fetch_recent_fills,
  _fill_event_id,
  _flow_symbols,
  _futures_position_fingerprint,
  _futures_settlement_marks,
  _FundingClock,
  _idle_hunt_due,
  _make_trade_context_lookup,
  _next_flow_observation,
  _next_price_noise_ewma,
  _productivity_adjusted_flat_cooldown,
  _prune_flow_observations,
  _rebase_reviewed_price_triggers,
  _PollFundingCredit,
  _settle_probes_on_futures,
)
from src.position_context import carry_refresh_targets


def test_futures_position_fingerprint_ignores_pnl_noise_but_detects_lifecycle_changes():
  first = [{"symbol": "XBTUSDTM", "currentQty": "2", "avgEntryPrice": "100", "unrealisedPnl": 1}]
  marked = [{"symbol": "XBTUSDTM", "currentQty": "2", "avgEntryPrice": "100", "unrealisedPnl": 9}]
  partial = [{"symbol": "XBTUSDTM", "currentQty": "1", "avgEntryPrice": "100", "unrealisedPnl": 9}]
  assert _futures_position_fingerprint(first) == _futures_position_fingerprint(marked)
  assert _futures_position_fingerprint(first) != _futures_position_fingerprint(partial)
  assert _futures_position_fingerprint([]) == ()


def test_explicit_auto_triggers_fire_only_on_the_requested_side():
  triggers = [
    {"symbol": "SOL-USDT", "condition": "above", "targetPrice": 100},
    {"symbol": "ZEC-USDT", "condition": "below", "targetPrice": 500},
    {"symbol": "BTC-USDT", "condition": None, "targetPrice": 1},
  ]
  crossed = _crossed_auto_triggers(
    triggers,
    {"SOL-USDT": 101, "ZEC-USDT": 501, "BTC-USDT": 99_000},
  )
  assert [(item[0]["symbol"], item[1]) for item in crossed] == [("SOL-USDT", 101.0)]


class TestAdaptiveAgentCooldown:
  def _cooldown(self, moves=None, *, active=False, events=0):
    return _adaptive_agent_cooldown(
      flat_cooldown_sec=3600,
      active_cooldown_sec=300,
      book_active=active,
      new_events_count=events,
      trigger_move_pcts=moves or [],
      price_trigger_pct=0.5,
    )

  def test_quiet_flat_market_keeps_hourly_ceiling(self):
    assert self._cooldown() == 3600

  def test_trigger_magnitude_shortens_flat_cadence(self):
    assert self._cooldown([0.5]) == pytest.approx(1800)
    assert self._cooldown([1.0]) == pytest.approx(720)
    assert self._cooldown([1.5]) == pytest.approx(360)

  def test_breadth_and_large_moves_converge_on_active_floor(self):
    assert self._cooldown([1.5, 1.5]) == 300

  def test_active_book_or_new_event_uses_active_cadence(self):
    assert self._cooldown(active=True) == 300
    assert self._cooldown(events=1) == 300

  def test_flat_backoff_is_disabled_by_default(self):
    assert _productivity_adjusted_flat_cooldown(600, 0) == 600
    assert _productivity_adjusted_flat_cooldown(600, 1) == 600
    assert _productivity_adjusted_flat_cooldown(600, 100) == 600

  def test_opt_in_flat_backoff_uses_power_of_two_up_to_configured_cap(self):
    assert _productivity_adjusted_flat_cooldown(600, 0, 4.0) == 600
    assert _productivity_adjusted_flat_cooldown(600, 1, 4.0) == 1200
    assert _productivity_adjusted_flat_cooldown(600, 2, 4.0) == 2400
    assert _productivity_adjusted_flat_cooldown(600, 20, 4.0) == 2400
    assert _productivity_adjusted_flat_cooldown(600, 3, 1.5) == 900

  def test_pending_atomic_entry_suppresses_idle_only_hunt(self):
    assert _idle_hunt_due(10, 10, pending_orders=False) is True
    assert _idle_hunt_due(100, 10, pending_orders=True) is False


class TestAdaptivePriceTrigger:
  def test_configured_threshold_is_floor_during_warmup_and_quiet_tape(self):
    assert _adaptive_price_trigger_threshold(0.5, 0.8, 2) == 0.5
    assert _adaptive_price_trigger_threshold(0.5, 0.05, 20) == 0.5

  def test_observed_noise_raises_threshold_but_is_bounded(self):
    # Default ceiling is 2× the base trigger (safety-biased): 0.3×4=1.2 is clamped to 0.5×2=1.0.
    assert _adaptive_price_trigger_threshold(0.5, 0.3, 20) == pytest.approx(1.0)
    assert _adaptive_price_trigger_threshold(0.5, 5.0, 20) == 1.0
    # A wider ceiling can be opted into for more token savings.
    assert _adaptive_price_trigger_threshold(0.5, 0.3, 20, max_multiplier=4.0) == pytest.approx(1.2)
    assert _adaptive_price_trigger_threshold(0.5, 5.0, 20, max_multiplier=4.0) == 2.0

  def test_noise_ewma_decays_and_single_shock_is_winsorized(self):
    seeded = _next_price_noise_ewma(0.0, 0.5, 0.5, 0)
    assert seeded == pytest.approx(0.5)
    assert _next_price_noise_ewma(seeded, 0.0, 0.5, 1) == pytest.approx(0.4)
    # A 20% print is winsorized to the ceiling (base × 2) before it can lift learned noise.
    assert _next_price_noise_ewma(0.5, 20.0, 0.5, 10) == pytest.approx(0.6)

  def test_successful_review_rebases_only_symbols_in_its_snapshot(self):
    moves = {"BTC-USDT": 2.0, "NEW-USDT": 1.5}
    discrete = {"initial:BTC-USDT", "initial:NEW-USDT", "supervisor_note"}
    _rebase_reviewed_price_triggers(moves, discrete, {"BTC-USDT"})
    assert moves == {"NEW-USDT": 1.5}
    assert discrete == {"initial:NEW-USDT", "supervisor_note"}


def test_fill_event_id_prefers_trade_id_and_separates_venues():
  fill = {"tradeId": "abc", "orderId": "order"}
  assert _fill_event_id(fill, "spot") == "spot:abc"
  assert _fill_event_id(fill, "futures") == "futures:abc"


def test_fill_event_id_fallback_distinguishes_partial_fills():
  first = {"orderId": "o1", "createdAt": 100, "symbol": "ETHUSDTM", "side": "buy", "price": 100, "size": 1}
  second = dict(first, price=101)
  assert _fill_event_id(first, "futures") != _fill_event_id(second, "futures")
  assert _fill_event_id({}, "futures").startswith("futures:payload:")


def test_close_event_fallback_includes_symbol_and_close_lifecycle():
  first = {"symbol": "ETHUSDTM", "openTime": 100, "closeTime": 200, "type": "CLOSE_LONG"}
  assert _close_event_id(first) != _close_event_id(dict(first, symbol="BTCUSDTM"))
  assert _close_event_id(first) != _close_event_id(dict(first, closeTime=201))


def test_futures_snapshot_fails_closed_when_one_endpoint_fails():
  class Client:
    def get_account_overview(self): return {"accountEquity": 100}
    def list_positions(self): raise RuntimeError("positions unavailable")
    def list_stop_orders(self, **kwargs): return []

  cfg = SimpleNamespace(kucoin_futures=SimpleNamespace(enabled=True))
  with pytest.raises(RuntimeError, match="Incomplete futures snapshot"):
    _fetch_futures(cfg, Client())


def test_futures_snapshot_returns_complete_truth():
  class Client:
    def get_account_overview(self): return {"accountEquity": 100}
    def list_positions(self): return [{"symbol": "ETHUSDTM"}]
    def list_stop_orders(self, **kwargs): return [{"symbol": "ETHUSDTM"}]

  cfg = SimpleNamespace(kucoin_futures=SimpleNamespace(enabled=True))
  overview, positions, stops = _fetch_futures(cfg, Client())
  assert overview["accountEquity"] == 100 and positions and stops


def test_recent_fill_backfill_paginates_until_short_page():
  now_ms = 2_000_000_000_000

  class Spot:
    def _timestamp_ms(self): return now_ms
    def get_fills(self, page=1, page_size=50):
      if page == 1:
        return [{"tradeId": f"s-{i}", "createdAt": now_ms - 1000 - i} for i in range(50)]
      if page == 2:
        return [{"tradeId": "s-50", "createdAt": now_ms - 2000}]
      return []

  class Futures:
    def get_fills(self, page=1, page_size=50): return []
    def get_position_history(self, page=1, page_size=50): return []

  result = _fetch_recent_fills(Spot(), Futures(), lookback_minutes=30)
  assert len(result["spot_fills"]) == 51
  assert result["_errors"] == []


class TestAgentMadeAMove:
  def test_live_order_counts_as_move(self):
    assert _agent_made_a_move({"tool_results": [{"orderId": "x", "orderRequest": {"side": "buy"}}]})

  def test_paper_order_counts_as_move(self):
    assert _agent_made_a_move({"tool_results": [{"paper": True, "orderRequest": {"side": "buy"}}]})

  def test_decline_is_not_a_move(self):
    assert not _agent_made_a_move({"tool_results": [{"skipped": True, "reason": "low confidence"}]})

  def test_rejected_order_is_not_a_move(self):
    # A rejected order carries an orderRequest but never executed — not a move.
    assert not _agent_made_a_move({"tool_results": [{"rejected": True, "orderRequest": {"side": "buy"}}]})

  def test_error_is_not_a_move(self):
    assert not _agent_made_a_move({"tool_results": [{"error": "boom", "orderRequest": {}}]})

  def test_transfer_only_is_not_a_move(self):
    assert not _agent_made_a_move({"tool_results": [{"transfer": {"orderId": "t"}, "amount": 5}]})

  def test_cancellation_is_not_a_productive_trade(self):
    assert not _agent_made_a_move({
      "tool_results": [{"cancelled": {"cancelledOrderIds": ["x"]}, "orderId": "x"}],
    })

  def test_empty_is_not_a_move(self):
    assert not _agent_made_a_move({"tool_results": []})
    assert not _agent_made_a_move({})

  def test_mixed_declines_then_one_order(self):
    res = {"tool_results": [
      {"skipped": True, "reason": "x"},
      {"rejected": True, "reason": "gate"},
      {"orderId": "ok", "orderRequest": {"side": "sell"}},
    ]}
    assert _agent_made_a_move(res)


class TestExpiredBotEntryOrders:
  def test_only_expired_tagged_entries_are_selected(self):
    now = 2_000_000_000
    orders = [
      {"id": "old", "clientOid": "traide-entry-old", "createdAt": (now - 31 * 60) * 1000},
      {"id": "new", "clientOid": "traide-entry-new", "createdAt": (now - 5 * 60) * 1000},
      {"id": "manual", "clientOid": "manual-order", "createdAt": (now - 60 * 60) * 1000},
      {"id": "protect", "clientOid": "traide-entry-protect", "createdAt": (now - 60 * 60) * 1000,
       "reduceOnly": True},
    ]
    assert [o["id"] for o in _expired_bot_entry_orders(orders, 30, now=now)] == ["old"]

  def test_expiry_can_be_disabled_and_handles_seconds(self):
    now = 2_000_000_000
    order = {"id": "old", "clientOid": "traide-entry-old", "createdAt": now - 3600}
    assert _expired_bot_entry_orders([order], 0, now=now) == []
    assert _expired_bot_entry_orders([order], 30, now=now) == [order]


class TestTakerFlowSampling:
  """Folding the tape into per-symbol state, once per poll.

  The subtle part is that consecutive 100-trade windows overlap heavily at a 60s poll, so the
  running level has to be built from the NEW slice — smoothing the window share would be smoothing
  the same trades repeatedly and would report a confidence the tape never provided.
  """

  @staticmethod
  def _summary(**over):
    base = {"trades": 100, "newTrades": 0, "spanSec": 180.0, "buyShare": None,
            "buyTradeShare": None, "newBuyShare": None, "lastCursor": 100, "gapped": False}
    base.update(over)
    return base

  def test_first_reading_seeds_the_level_from_the_window(self):
    out = _next_flow_observation(None, self._summary(buyShare=0.7), 1_700_000_000)
    assert out["buyShareEwma"] == 0.7 and out["samples"] == 1
    assert out["updated"] == 1_700_000_000

  def test_the_level_tracks_newly_arrived_trades_not_the_overlapping_window(self):
    prior = {"buyShareEwma": 0.5, "samples": 4}
    out = _next_flow_observation(prior, self._summary(buyShare=0.55, newBuyShare=1.0), 1)
    # 0.7*0.5 + 0.3*1.0 — the new slice, not the 0.55 window that mostly repeats old trades.
    assert out["buyShareEwma"] == pytest.approx(0.65)
    assert out["samples"] == 5

  def test_a_quiet_symbol_falls_back_to_the_window_share(self):
    """Nothing new since the last poll is still a reading — the window is what the tape says now."""
    prior = {"buyShareEwma": 0.5, "samples": 4}
    out = _next_flow_observation(prior, self._summary(buyShare=0.9, newBuyShare=None), 1)
    assert out["buyShareEwma"] == pytest.approx(0.62)   # 0.7*0.5 + 0.3*0.9

  def test_an_unmeasurable_poll_carries_the_level_rather_than_resetting_to_neutral(self):
    """A failed or empty read must not look like balanced flow — that is a claim, not a gap."""
    prior = {"buyShareEwma": 0.8, "samples": 6}
    out = _next_flow_observation(prior, self._summary(), 1)
    assert out["buyShareEwma"] == 0.8 and out["samples"] == 6   # unchanged, not 0.5 and not counted

  def test_corrupt_prior_state_does_not_poison_the_series(self):
    out = _next_flow_observation({"buyShareEwma": "x", "samples": "y"}, self._summary(buyShare=0.4), 1)
    assert out["buyShareEwma"] == 0.4 and out["samples"] == 1

  def test_held_positions_are_sampled_before_the_watchlist(self):
    """One REST call per symbol per poll is cheap but capped; the cap must trim the watchlist tail,
    never the trade we are actually carrying."""
    snapshot = SimpleNamespace(
      futures_positions=[{"symbol": "XBTUSDTM", "currentQty": "-3"}],
      tickers={"AAA-USDT": None, "BBB-USDT": None, "CCC-USDT": None},
    )
    assert _flow_symbols(snapshot, 2) == ["BTC-USDT", "AAA-USDT"]

  def test_flat_positions_and_junk_rows_are_ignored(self):
    snapshot = SimpleNamespace(
      futures_positions=[{"symbol": "XBTUSDTM", "currentQty": "0"},
                         {"symbol": "ETHUSDTM", "currentQty": "bad"},
                         "not-a-dict"],
      tickers={"AAA-USDT": None},
    )
    assert _flow_symbols(snapshot, 5) == ["AAA-USDT"]

  def test_sampling_can_be_turned_off_by_the_cap(self):
    snapshot = SimpleNamespace(futures_positions=[], tickers={"AAA-USDT": None})
    assert _flow_symbols(snapshot, 0) == []
    assert _flow_symbols(SimpleNamespace(futures_positions=None, tickers=None), 5) == []

  def test_the_watchlist_tail_is_ranked_by_the_move_waking_the_agent(self):
    """Alphabetical order sounds neutral and is not.

    With a 50-coin universe and a cap of 12, ranking by name recorded the tape for AAVE…FARTCOIN
    for three days while every actual direction call went to TAO, INJ, WLD and XLM — so the
    experiment covered none of the trades it exists to score. Rank by the excursion that is about
    to wake the model instead, so a symbol is sampled from the moment it becomes interesting.
    """
    snapshot = SimpleNamespace(
      futures_positions=[],
      tickers={"AAA-USDT": None, "TAO-USDT": None, "WLD-USDT": None, "ZZZ-USDT": None},
    )
    moves = {"WLD-USDT": 1.4, "TAO-USDT": 3.1}
    assert _flow_symbols(snapshot, 3, moves) == ["TAO-USDT", "WLD-USDT", "AAA-USDT"]

  def test_a_held_position_outranks_even_the_biggest_move(self):
    snapshot = SimpleNamespace(
      futures_positions=[{"symbol": "XBTUSDTM", "currentQty": "-3"}],
      tickers={"AAA-USDT": None, "TAO-USDT": None},
    )
    assert _flow_symbols(snapshot, 1, {"TAO-USDT": 9.9}) == ["BTC-USDT"]

  def test_untriggered_symbols_keep_a_stable_order(self):
    """Ties must not reshuffle between polls, or the cap would rotate symbols in and out at random
    and no symbol would accumulate a continuous record."""
    snapshot = SimpleNamespace(
      futures_positions=[], tickers={"CCC-USDT": None, "AAA-USDT": None, "BBB-USDT": None},
    )
    assert _flow_symbols(snapshot, 3, {}) == ["AAA-USDT", "BBB-USDT", "CCC-USDT"]
    assert _flow_symbols(snapshot, 3, {"QQQ-USDT": 5.0}) == ["AAA-USDT", "BBB-USDT", "CCC-USDT"]


class TestFlowObservationPruning:
  """A tape reading has a shelf life; the persisted state has to enforce it.

  Symbols rotate through the sampling cap, and a symbol that rotated out simply stops being
  refreshed — nothing marked it dead. Live state was found holding 23 symbols of which 11 were up to
  63 HOURS stale, sitting in exactly the dict the entry path reads a "current" reading from.
  """

  @staticmethod
  def _obs(updated):
    return {"buyShare": 0.6, "updated": updated}

  def test_readings_older_than_the_bound_are_forgotten(self):
    now = 1_700_000_000
    observations = {
      "FRESH-USDT": self._obs(now - 60),         # last poll
      "EDGE-USDT": self._obs(now - 600),         # exactly the bound: 10 polls x 60s
      "STALE-USDT": self._obs(now - 63 * 3600),  # the 63-hour reading found in live state
    }
    assert _prune_flow_observations(observations, 60, now) == 1
    assert sorted(observations) == ["EDGE-USDT", "FRESH-USDT"]

  def test_the_bound_scales_with_the_poll_interval(self):
    """Self-tuning: a slower loop must not start throwing away readings it has not had time to
    refresh, and nobody should have to hand-edit a seconds constant when the interval changes."""
    now = 1_700_000_000
    at_10_min = {"A-USDT": self._obs(now - 3000)}
    assert _prune_flow_observations(dict(at_10_min), 60, now) == 1     # 3000s > 10 x 60s
    assert _prune_flow_observations(dict(at_10_min), 300, now) == 0    # 3000s = 10 x 300s

  def test_unusable_rows_are_dropped_rather_than_kept_forever(self):
    now = 1_700_000_000
    observations = {"OK-USDT": self._obs(now), "JUNK-USDT": "not-a-dict",
                    "NOTS-USDT": {"buyShare": 0.5}, "BAD-USDT": {"updated": "soon"}}
    assert _prune_flow_observations(observations, 60, now) == 3
    assert list(observations) == ["OK-USDT"]

  def test_pruning_mutates_the_caller_dict_so_the_persisted_state_actually_shrinks(self):
    """The poll loop shares this dict with the closure that writes scheduler state. Returning a new
    dict would look identical in a unit test and silently persist the unpruned copy."""
    now = 1_700_000_000
    observations = {"STALE-USDT": self._obs(now - 99_999)}
    persisted_view = observations               # what _persist_agent_scheduler closes over
    _prune_flow_observations(observations, 60, now)
    assert persisted_view == {}


class _SettleFuturesClient:
  """Public futures endpoints only: marks and funding history, with call capture."""

  def __init__(self, marks=None, funding=None, *, raise_marks=False, raise_funding=False):
    self.marks = marks or {}
    self.funding = funding or []
    self.raise_marks = raise_marks
    self.raise_funding = raise_funding
    self.mark_calls: list[str] = []
    self.funding_calls: list[tuple] = []

  def get_mark_price(self, symbol):
    self.mark_calls.append(symbol)
    if self.raise_marks:
      raise RuntimeError("mark endpoint down")
    return {"symbol": symbol, "value": self.marks.get(symbol)}

  def get_funding_rate_history(self, symbol, start_at=None, end_at=None):
    self.funding_calls.append((symbol, start_at, end_at))
    if self.raise_funding:
      raise RuntimeError("funding endpoint down")
    return list(self.funding)


class _SettleCaptureMemory:
  def __init__(self, due=(), *, raise_signal=False):
    self.due = set(due)
    self.raise_signal = raise_signal
    self.signal_calls: list[tuple] = []
    self.exit_calls: list[tuple] = []

  def symbols_due_for_settlement(self, now=None):
    return set(self.due)

  def settle_signal_probes(self, prices, **kwargs):
    self.signal_calls.append((dict(prices), kwargs))
    if self.raise_signal:
      raise RuntimeError("corrupt row")
    return 0

  def settle_exit_probes(self, prices, **kwargs):
    self.exit_calls.append((dict(prices), kwargs))
    return 0


class TestProbeSettlementOnFutures:
  """ONE-USDT, 2026-09-20: spot ~0.0050, ONEUSDTM mark ~0.0038. The loop handed its SPOT ticker map to
  both settle steps while the probe base was the futures mark, so the basis was scored as edge and
  funding_carry read t=1.90 ('paying') where the contract said ~0.9."""

  def test_signal_and_exit_probes_receive_futures_marks_not_spot(self):
    memory = _SettleCaptureMemory(due={"ONE-USDT", "KCS-USDT"})
    client = _SettleFuturesClient(marks={"ONEUSDTM": "0.0038"})
    snapshot = SimpleNamespace(futures_positions=[{"symbol": "KCSUSDTM", "currentQty": 3, "markPrice": "10.5"}])
    spot = {"ONE-USDT": 0.0050, "KCS-USDT": 11.0}
    marks = _settle_probes_on_futures(memory, client, snapshot, spot, now=1_790_000_000)
    assert marks == {"ONE-USDT": 0.0038, "KCS-USDT": 10.5}
    (sig_prices, sig_kwargs), = memory.signal_calls
    (exit_prices, _), = memory.exit_calls
    assert sig_prices == exit_prices == {"ONE-USDT": 0.0038, "KCS-USDT": 10.5}
    # Spot only rides along for the rare spot-based probe; it is never the settlement map.
    assert sig_kwargs["spot_prices"] is spot
    assert callable(sig_kwargs["funding_received"])
    # A held position's own mark is reused; only the other symbol costs a call.
    assert client.mark_calls == ["ONEUSDTM"]

  def test_end_to_end_a_real_store_settles_at_the_futures_mark(self, tmp_path):
    from src.memory import MemoryStore
    store = MemoryStore(str(tmp_path / "m.json"))
    store.record_signal_probe("ONE-USDT", "buy", 0.0039, "funding_carry", price_source="futures_mark")
    data = store._read()
    data["signal_probes"][0]["ts"] -= 6 * 60
    store._write(data)
    client = _SettleFuturesClient(marks={"ONEUSDTM": 0.0038})
    _settle_probes_on_futures(store, client, SimpleNamespace(futures_positions=[]), {"ONE-USDT": 0.0050})
    probe = store.signal_probes(limit=0)[0]["entryContext"]["signalProbe"]
    assert probe["m5"] == 0.0038
    assert probe["f5"] == 0.0                     # no settlement inside five minutes: credited as zero
    assert client.mark_calls == ["ONEUSDTM"]

  def test_end_to_end_a_probe_with_no_mark_waits_instead_of_taking_spot(self, tmp_path):
    from src.memory import MemoryStore
    store = MemoryStore(str(tmp_path / "m.json"))
    store.record_signal_probe("TWO-USDT", "buy", 0.0039, "funding_carry", price_source="futures_mark")
    data = store._read()
    data["signal_probes"][0]["ts"] -= 6 * 60
    store._write(data)
    client = _SettleFuturesClient(marks={})        # the mark endpoint has nothing for TWOUSDTM
    _settle_probes_on_futures(store, client, SimpleNamespace(futures_positions=[]), {"TWO-USDT": 0.0050})
    assert "m5" not in store.signal_probes(limit=0)[0]["entryContext"]["signalProbe"]

  def test_the_poll_loop_no_longer_settles_on_the_spot_map(self):
    import inspect
    import src.main as main_mod
    src = inspect.getsource(main_mod.trading_loop)
    assert "settle_signal_probes(live_prices" not in src
    assert "settle_exit_probes(live_prices" not in src
    call = src.index("_settle_probes_on_futures(memory, _measure_client, snapshot, live_prices,")
    # The auto-trigger scan keeps the spot map, and runs first, so a settle failure cannot touch it...
    assert src.index("_crossed_auto_triggers(memory.latest_triggers(), live_prices)") < call
    # ...and the settle's few network calls come AFTER the profit-lock: measurement never delays survival.
    assert src.index("protection_actions = protection.run(snapshot)") < call

  def test_a_raising_futures_client_logs_warning_and_never_raises(self, caplog):
    memory = _SettleCaptureMemory(due={"ONE-USDT"})
    client = _SettleFuturesClient(raise_marks=True, raise_funding=True)
    with caplog.at_level("WARNING"):
      marks = _settle_probes_on_futures(memory, client, SimpleNamespace(futures_positions=[]), {"ONE-USDT": 0.005})
    assert marks == {}                            # no spot fallback
    assert any("PROBE SETTLE" in r.message and r.levelname == "WARNING" for r in caplog.records)
    # Both settle steps still ran (so tolerance write-offs still happen), and nothing propagated.
    assert memory.signal_calls and memory.exit_calls
    # T1: the map actually HANDED to settle is empty too — a `{**spot, **marks}` fallback at the call
    # site would stamp the futures-based probe at spot (the +25% ONE phantom) with CI green.
    assert memory.signal_calls[0][0] == {} and memory.exit_calls[0][0] == {}

  def test_a_symbol_without_a_mark_is_never_backfilled_from_spot(self):
    """T1: only KCS has a mark (from the held position); ONE's mark endpoint returns nothing. The settle
    maps must hold KCS alone — never ONE at its spot price."""
    memory = _SettleCaptureMemory(due={"ONE-USDT", "KCS-USDT"})
    client = _SettleFuturesClient(marks={})
    snapshot = SimpleNamespace(futures_positions=[{"symbol": "KCSUSDTM", "currentQty": 3, "markPrice": "10.5"}])
    _settle_probes_on_futures(memory, client, snapshot, {"ONE-USDT": 0.0050, "KCS-USDT": 11.0})
    assert memory.signal_calls[0][0] == memory.exit_calls[0][0] == {"KCS-USDT": 10.5}

  def test_a_failing_signal_settle_does_not_cost_the_exit_probes_their_poll(self, caplog):
    memory = _SettleCaptureMemory(raise_signal=True)
    with caplog.at_level("WARNING"):
      _settle_probes_on_futures(memory, None, SimpleNamespace(futures_positions=[]), {})
    assert memory.exit_calls
    assert any("SIGNAL PROBE settle failed" in r.message for r in caplog.records)

  def test_no_futures_client_means_no_marks_rather_than_spot(self):
    marks = _futures_settlement_marks({"ONE-USDT"}, None, SimpleNamespace(futures_positions=[]))
    assert marks == {}


class _CutoffMemory(_SettleCaptureMemory):
  """_SettleCaptureMemory that also answers settlement_cutoffs, like the real store."""

  def __init__(self, cutoffs, **kw):
    super().__init__(due=set(cutoffs), **kw)
    self.cutoffs = dict(cutoffs)

  def settlement_cutoffs(self, now=None):
    return dict(self.cutoffs)


class _SlowMarks(_SettleFuturesClient):
  """Each mark call advances a fake clock by ``cost`` seconds (a slow/hanging endpoint), then fails."""

  def __init__(self, clock, cost, **kw):
    super().__init__(**kw)
    self.clock, self.cost = clock, cost

  def get_mark_price(self, symbol):
    self.clock[0] += self.cost
    return super().get_mark_price(symbol)


class TestMeasurementIsBoundedOnTheSurvivalThread:
  """S2 (2026-09-25 review): settlement made serial, uncapped exchange calls in the poll thread with no
  deadline and no memory of failures — 20 due symbols x 15s against a hanging endpoint, every poll."""

  T = 1_790_000_000

  def test_a_hanging_endpoint_costs_at_most_the_poll_budget(self):
    from src.main import _MeasurementBudget
    clock = [0.0]
    client = _SlowMarks(clock, 5.0, raise_marks=True)
    memory = _CutoffMemory({f"S{i:02d}-USDT": self.T + 3000 for i in range(20)})
    budget = _MeasurementBudget(15.0, clock=lambda: clock[0])
    _settle_probes_on_futures(memory, client, SimpleNamespace(futures_positions=[]), {}, now=self.T,
                              budget=budget)
    assert len(client.mark_calls) == 3              # 0s, 5s, 10s — then the budget is spent
    assert budget.deferred == 17
    assert memory.signal_calls and memory.exit_calls   # the settle steps still ran (write-offs happen)

  def test_the_most_urgent_rows_are_asked_first(self):
    from src.main import _MeasurementBudget
    clock = [0.0]
    client = _SlowMarks(clock, 5.0, marks={"AAAUSDTM": 1.0, "ZZZUSDTM": 2.0, "MMMUSDTM": 3.0})
    # ZZZ has a 5m horizon about to be written off; AAA only an 8h exit probe; MMM a 240m horizon.
    memory = _CutoffMemory({"AAA-USDT": self.T + 30_000, "ZZZ-USDT": self.T + 60, "MMM-USDT": self.T + 2000})
    budget = _MeasurementBudget(10.0, clock=lambda: clock[0])
    marks = _settle_probes_on_futures(memory, client, SimpleNamespace(futures_positions=[]), {}, now=self.T,
                                      budget=budget)
    assert client.mark_calls == ["ZZZUSDTM", "MMMUSDTM"]
    assert marks == {"ZZZ-USDT": 2.0, "MMM-USDT": 3.0}

  def test_a_failing_symbol_is_not_re_asked_before_its_backoff(self):
    from src.main import _MeasureBackoff
    client = _SettleFuturesClient(raise_marks=True)
    memory = _CutoffMemory({"ONE-USDT": self.T + 9 * 3600})       # an unresolved 8h exit probe
    backoff = _MeasureBackoff(60.0)
    asked_at = []
    for poll in range(30):                                       # 30 polls, 60s apart
      now = self.T + 60 * poll
      before = len(client.mark_calls)
      _settle_probes_on_futures(memory, client, SimpleNamespace(futures_positions=[]), {}, now=now,
                                backoff=backoff)
      if len(client.mark_calls) > before:
        asked_at.append(poll)
    assert asked_at == [0, 1, 3, 7, 15]                          # 60s, 120s, 240s, 480s backoff
    client.raise_marks = False
    client.marks = {"ONEUSDTM": 0.004}
    _settle_probes_on_futures(memory, client, SimpleNamespace(futures_positions=[]), {}, now=self.T + 60 * 31,
                              backoff=backoff)
    assert memory.signal_calls[-1][0] == {"ONE-USDT": 0.004}
    assert backoff.ready(("mark", "ONEUSDTM"), self.T + 60 * 32)  # success clears it

  def test_the_backoff_never_outlasts_the_rows_remaining_tolerance(self):
    from src.main import _MeasureBackoff
    b = _MeasureBackoff(60.0)
    for _ in range(10):
      delay = b.failed("k", self.T, cap_sec=300.0)
    assert delay == 300.0
    assert b.failed("j", self.T, cap_sec=5.0) == 60.0            # never below one poll

  def test_a_stack_replay_that_runs_out_of_budget_resumes_next_poll_not_in_15_minutes(self, tmp_path):
    from src.main import _MeasurementBudget
    from tests.test_protection import _cfg as _pp_cfg
    store, _ts0, fill = _agent_probe_row(tmp_path)
    clock = [0.0]

    class _SlowKlines(_KlineFake):
      def get_candles(self, *a, **k):
        clock[0] += 4.0
        return super().get_candles(*a, **k)

    client = _SlowKlines(_flat_closes(fill, 30 + 8 * 60 + 5))
    attempts = {}
    budget = _MeasurementBudget(6.0, clock=lambda: clock[0])
    assert _score_exit_probe_stacks(store, client, _pp_cfg(), attempts=attempts, budget=budget) == 0
    assert attempts == {} and "stack" not in store.exit_probes()[0]    # not a failed attempt
    assert _score_exit_probe_stacks(store, client, _pp_cfg(), attempts=attempts) == 1

  def test_measurement_reads_use_a_short_timeout_and_orders_keep_theirs(self, monkeypatch):
    import src.kucoin as kucoin_mod
    from src.main import _MEASURE_TIMEOUT_SEC, _measurement_client
    seen = []

    class _Resp:
      ok = True
      status_code = 200
      text = ""

      def json(self):
        return {"code": "200000", "data": {"value": 1.0}}

    monkeypatch.setattr(kucoin_mod.requests, "request", lambda *a, **k: seen.append(k["timeout"]) or _Resp())
    cfg = SimpleNamespace(kucoin=SimpleNamespace(api_key="", secret="", passphrase=""),
                          kucoin_futures=SimpleNamespace(base_url="https://example.invalid"))
    client = kucoin_mod.KucoinFuturesClient(cfg)
    fast = _measurement_client(client)
    fast.get_mark_price("ONEUSDTM")
    client.get_mark_price("ONEUSDTM")
    assert seen == [_MEASURE_TIMEOUT_SEC, 15.0]
    fake = _SettleFuturesClient()
    assert _measurement_client(fake) is fake and _measurement_client(None) is None

  def test_the_loop_bounds_every_measurement_call_after_survival(self):
    import inspect
    import src.main as main_mod
    src = inspect.getsource(main_mod.trading_loop)
    run = src.index("protection_actions = protection.run(snapshot)")
    budget = src.index("_measure_budget = _MeasurementBudget(_MEASURE_BUDGET_FRACTION")
    assert run < budget
    settle = src[src.index("_settle_probes_on_futures(memory, _measure_client"):][:200]
    assert "budget=_measure_budget" in settle and "backoff=_measure_backoff" in settle
    stacks = src[src.index("_score_exit_probe_stacks(memory, _measure_client"):][:200]
    assert "budget=_measure_budget" in stacks
    assert "if not _budget_spent(_measure_budget):\n        _market_state.refresh()" in src
    assert "_measure_client = _measurement_client(kucoin_futures)" in src
    assert "_FundingClock(_measurement_client(kucoin_futures)" in inspect.getsource(main_mod._make_trade_context_lookup)


class TestPollFundingCredit:
  T = 1_790_000_000

  def test_sign_and_memoized_once_per_symbol_per_poll(self):
    history = [{"symbol": "ONEUSDTM", "fundingRate": -0.002, "timepoint": (self.T + 600) * 1000}]
    client = _SettleFuturesClient(funding=history)
    credit = _PollFundingCredit(client, now=self.T + 3600)
    assert credit("ONE-USDT", "long", self.T, self.T + 3600) == pytest.approx(0.002)    # longs are paid
    assert credit("ONE-USDT", "short", self.T, self.T + 3600) == pytest.approx(-0.002)  # shorts pay
    assert len(client.funding_calls) == 1
    # An EARLIER window than the one fetched forces one refetch, never a silently short history.
    credit("ONE-USDT", "long", self.T - 7200, self.T + 3600)
    assert len(client.funding_calls) == 2

  def test_a_non_list_history_is_unknown_not_zero(self, caplog):
    """W3: a {"dataList": ...}-style or error payload was cached as [] and stamped a silent 0.0 credit."""
    class _OddPayload(_SettleFuturesClient):
      def get_funding_rate_history(self, symbol, start_at=None, end_at=None):
        self.funding_calls.append((symbol, start_at, end_at))
        return {"code": "429", "msg": "too many requests"}

    credit = _PollFundingCredit(_OddPayload(), now=self.T)
    with caplog.at_level("WARNING"):
      assert credit("ONE-USDT", "long", self.T - 600, self.T) is None
    assert any("PROBE FUNDING" in r.message for r in caplog.records)

  def test_a_failing_symbol_is_unknown_and_not_hammered(self, caplog):
    client = _SettleFuturesClient(raise_funding=True)
    credit = _PollFundingCredit(client, now=self.T)
    with caplog.at_level("WARNING"):
      assert credit("ONE-USDT", "long", self.T - 600, self.T) is None
      assert credit("ONE-USDT", "long", self.T - 600, self.T) is None
    assert len(client.funding_calls) == 1
    assert any("PROBE FUNDING" in r.message for r in caplog.records)


# ── Carry hold on the contract's own funding clock: the LIVE lookup wiring (2026-09-25) ──────────


def _utc_ts(day, hh, mm=0):
  import datetime as _dt
  return _dt.datetime(2026, 9, day, hh, mm, tzinfo=_dt.UTC).timestamp()


class _FundingClockClient:
  """get_funding_rate (shaped like /api/v1/funding-rate/{sym}/current) and get_funding_rate_history
  (settlement timepoints in ms), with call capture."""

  def __init__(self, *, granularity_ms=3_600_000, funding_time_s=None, fail=False, history=(),
               history_fail=False):
    self.granularity_ms = granularity_ms
    self.funding_time_s = funding_time_s
    self.fail = fail
    self.history = list(history)          # settlement epoch seconds
    self.history_fail = history_fail
    self.calls: list[str] = []
    self.history_calls: list[tuple] = []

  def get_funding_rate(self, symbol):
    self.calls.append(symbol)
    if self.fail:
      raise RuntimeError("funding endpoint down")
    return {"symbol": f".{symbol}FPI8H", "granularity": self.granularity_ms, "value": -0.00196,
            "fundingTime": int(self.funding_time_s * 1000)}

  def get_funding_rate_history(self, symbol, start_at=None, end_at=None):
    self.history_calls.append((symbol, start_at, end_at))
    if self.fail or self.history_fail:
      raise RuntimeError("funding history endpoint down")
    return [{"symbol": symbol, "fundingRate": -0.00196, "timepoint": int(t * 1000)}
            for t in self.history if start_at <= t * 1000 <= end_at]


def _carry_store(tmp_path, fill_ts, *, family="funding_carry", funding=None):
  """A real MemoryStore holding the filled entry behind a ONE-USDT long, filled at ``fill_ts``."""
  from src.memory import MemoryStore
  store = MemoryStore(str(tmp_path / "carry.json"))
  ctx = {"positionSide": "long", "setupFamily": family, "entryPrice": 0.0039,
         "stopLossPrice": 0.0037, "stopAtrMult": 2.0}
  if funding is not None:
    ctx["funding"] = funding
  store.record_trade("ONE-USDT", "buy", 5.0, price=0.0039, venue="futures", entry_context=ctx)
  data = store._read()
  data["trades"][-1]["fillTs"] = int(fill_ts)
  store._write(data)
  return store


def _one_pos(fill_ts):
  # Exchange-shaped: KuCoin reports positionSide 'BOTH' in one-way mode.
  return {"symbol": "ONEUSDTM", "currentQty": 10, "positionSide": "BOTH",
          "openingTimestamp": int(fill_ts * 1000), "avgEntryPrice": 0.0039, "markPrice": 0.0040}


class TestCarryHoldUsesTheLiveFundingClock:
  """ONE-USDT 2026-09-21 00:24, a 1h contract: the hold ran to 08:00 on the 8h grid with every
  protection rule off, and the trade stopped at -1.29R at 02:23. The lookup the live manager receives
  must hand the carry hold the contract's own clock — read from the cache the poll loop refreshes."""

  FILL = _utc_ts(21, 0, 24)

  def test_one_hour_contract_holds_to_the_next_hour(self, tmp_path):
    client = _FundingClockClient(funding_time_s=_utc_ts(21, 1))
    store = _carry_store(tmp_path, self.FILL)
    lookup = _make_trade_context_lookup(store, client, time_fn=lambda: self.FILL + 60)
    assert lookup.funding_clock.refresh(carry_refresh_targets(store, [_one_pos(self.FILL)])) >= 1
    assert client.calls == ["ONEUSDTM"]
    ctx = lookup("ONEUSDTM", _one_pos(self.FILL))
    assert ctx["holdUntilTs"] == _utc_ts(21, 1)                # the 8h grid said 08:00
    assert client.calls == ["ONEUSDTM"]                        # the lookup itself made no call
    # The rest of the context is unchanged by the move out of main.
    assert ctx["noiseBandR"] == pytest.approx(0.5)
    assert ctx["initRiskPx"] == pytest.approx(0.0002)

  def test_a_raising_client_falls_back_to_the_8h_grid_and_warns_once(self, tmp_path, caplog):
    client = _FundingClockClient(fail=True)
    store = _carry_store(tmp_path, self.FILL)
    lookup = _make_trade_context_lookup(store, client, time_fn=lambda: self.FILL + 60)
    with caplog.at_level("WARNING"):
      lookup.funding_clock.refresh(carry_refresh_targets(store, [_one_pos(self.FILL)]))
      first = lookup("ONEUSDTM", _one_pos(self.FILL))
      second = lookup("ONEUSDTM", _one_pos(self.FILL))
    assert first["holdUntilTs"] == second["holdUntilTs"] == _utc_ts(21, 8)   # never "no hold"
    assert sum("CARRY CLOCK" in r.message for r in caplog.records) == 1

  def test_a_raising_client_uses_the_clock_stamped_on_the_entry_first(self, tmp_path):
    store = _carry_store(tmp_path, self.FILL, funding={"rate": -0.00196, "intervalSec": 3600.0,
                                                       "nextSettlementTs": _utc_ts(20, 23)})
    lookup = _make_trade_context_lookup(store, _FundingClockClient(fail=True),
                                        time_fn=lambda: self.FILL + 60)
    lookup.funding_clock.refresh(carry_refresh_targets(store, [_one_pos(self.FILL)]))
    assert lookup("ONEUSDTM", _one_pos(self.FILL))["holdUntilTs"] == _utc_ts(21, 1)

  def test_non_carry_positions_are_never_refresh_targets(self, tmp_path):
    client = _FundingClockClient(funding_time_s=_utc_ts(21, 1))
    store = _carry_store(tmp_path, self.FILL, family="continuation")
    lookup = _make_trade_context_lookup(store, client, time_fn=lambda: self.FILL + 60)
    assert carry_refresh_targets(store, [_one_pos(self.FILL)]) == {}
    lookup.funding_clock.refresh(carry_refresh_targets(store, [_one_pos(self.FILL)]))
    assert lookup("ONEUSDTM", _one_pos(self.FILL))["holdUntilTs"] is None
    assert client.calls == []
    assert carry_refresh_targets(store, [None, {"symbol": "X", "currentQty": "junk"}]) == {}

  def test_an_interval_changed_after_the_fill_is_judged_on_the_exchange_history(self, tmp_path):
    """W2 at the call site: fill 09:30 stamped 8h (next 16:00); at 11:00 KuCoin moves the contract to 1h
    (live next 12:00). The live grid walked back to the fill invented a 10:00 settlement and lifted the
    hold at 11:05. With the history (nothing paid yet) the hold runs to the real 12:00 payment."""
    fill = _utc_ts(21, 9, 30)
    now = [_utc_ts(21, 11, 5)]
    store = _carry_store(tmp_path, fill, funding={"rate": -0.002, "intervalSec": 8 * 3600.0,
                                                  "nextSettlementTs": _utc_ts(21, 16)})
    client = _FundingClockClient(funding_time_s=_utc_ts(21, 12), granularity_ms=3_600_000,
                                 history=[_utc_ts(21, 8)])      # before the fill: not ours
    lookup = _make_trade_context_lookup(store, client, time_fn=lambda: now[0])
    lookup.funding_clock.refresh(carry_refresh_targets(store, [_one_pos(fill)]))
    assert lookup("ONEUSDTM", _one_pos(fill))["holdUntilTs"] == _utc_ts(21, 12)
    assert lookup.funding_clock.settlement_since("ONEUSDTM", fill) == (None, now[0])
    # The 12:00 payment happens; the next refresh reads it from the history and the hold is over.
    now[0] = _utc_ts(21, 12, 1)
    client.history.append(_utc_ts(21, 12))
    client.funding_time_s = _utc_ts(21, 13)
    lookup.funding_clock.refresh(carry_refresh_targets(store, [_one_pos(fill)]))
    assert lookup("ONEUSDTM", _one_pos(fill))["holdUntilTs"] is None

  def test_a_lengthened_interval_never_re_engages_a_collected_carry(self, tmp_path):
    """The reverse: 4h stamp (next 12:00) -> 8h grid (next 16:00) after the 12:00 payment."""
    fill = _utc_ts(21, 9, 30)
    store = _carry_store(tmp_path, fill, funding={"rate": -0.002, "intervalSec": 4 * 3600.0,
                                                  "nextSettlementTs": _utc_ts(21, 12)})
    client = _FundingClockClient(funding_time_s=_utc_ts(21, 16), granularity_ms=8 * 3_600_000,
                                 history=[_utc_ts(21, 12)])
    lookup = _make_trade_context_lookup(store, client, time_fn=lambda: _utc_ts(21, 12, 1))
    lookup.funding_clock.refresh(carry_refresh_targets(store, [_one_pos(fill)]))
    assert lookup("ONEUSDTM", _one_pos(fill))["holdUntilTs"] is None

  def test_protection_under_the_lock_never_calls_the_exchange_for_the_clock(self, tmp_path):
    """S3: ProtectionManager.run holds order_lock; a hanging funding endpoint used to cost the full
    request timeout per carry position per poll there. The lookup is cache-only now."""
    from src.protection import ProtectionManager
    from tests.test_protection import _cfg as _pp_cfg

    class _Hanging(_FundingClockClient):
      def get_funding_rate(self, symbol):
        self.calls.append(symbol)
        raise TimeoutError("read timed out (15s)")

    client = _Hanging()
    store = _carry_store(tmp_path, self.FILL)
    lookup = _make_trade_context_lookup(store, client, time_fn=lambda: self.FILL + 60)
    mgr = ProtectionManager(_pp_cfg(), None, trade_context_lookup=lookup)
    snap = type("S", (), {"futures_enabled": True, "futures_positions": [_one_pos(self.FILL)],
                          "futures_stop_orders": [], "futures_account": {}, "total_usdt": 0.0})()
    for _ in range(3):
      mgr.run(snap)
    assert client.calls == [] and client.history_calls == []

  def test_the_trading_loop_hands_protection_this_exact_lookup(self):
    """Wiring, not just the helper: the loop builds the lookup from the factory, refreshes its clock
    OUTSIDE order_lock before the protection pass, and the lookup hands the clock and the settlement
    history to carry_hold_deadline."""
    import inspect
    import src.main as main_mod
    import src.position_context as pc
    loop_src = inspect.getsource(main_mod.trading_loop)
    assert "_trade_context = _make_trade_context_lookup(memory, kucoin_futures)" in loop_src
    assert "trade_context_lookup=_trade_context" in loop_src
    assert "funding_clock=clock" in inspect.getsource(main_mod._make_trade_context_lookup)
    refresh = loop_src.index(
      "_trade_context.funding_clock.refresh(carry_refresh_targets(memory, snapshot.futures_positions))")
    run = loop_src.index("protection_actions = protection.run(snapshot)")
    assert refresh < run
    between = loop_src[refresh:run]
    assert between.count("with safety.order_lock:") == 1       # only the one that wraps protection.run
    assert "with safety.order_lock:" not in loop_src[loop_src.rindex("\n", 0, refresh):refresh]
    tc_src = inspect.getsource(pc.trade_context)
    assert "next_settlement_ts=next_ts, interval_sec=interval" in tc_src
    assert "first_paid_ts=first_paid, unpaid_as_of_ts=unpaid_asof" in tc_src
    call_src = inspect.getsource(main_mod._FundingClock.__call__)
    assert "get_funding_rate" not in call_src and "_client" not in call_src


class TestFundingClockCache:
  T = _utc_ts(21, 0, 24)

  def test_refetched_only_when_the_settlement_passes_or_after_fifteen_minutes(self):
    now = [self.T]
    client = _FundingClockClient(funding_time_s=_utc_ts(21, 1))
    clock = _FundingClock(client, time_fn=lambda: now[0])
    assert clock("ONEUSDTM") is None                           # nothing refreshed yet: cache only
    clock.refresh(["ONEUSDTM"])
    assert clock("ONEUSDTM") == (_utc_ts(21, 1), 3600.0)
    now[0] = self.T + 10 * 60
    clock.refresh(["ONEUSDTM"])
    assert len(client.calls) == 1                              # fresh and settlement still ahead
    now[0] = self.T + 16 * 60
    clock.refresh(["ONEUSDTM"])
    assert len(client.calls) == 2                              # older than 15 min
    client.funding_time_s = _utc_ts(21, 2)
    now[0] = _utc_ts(21, 1)
    clock.refresh(["ONEUSDTM"])
    assert clock("ONEUSDTM") == (_utc_ts(21, 2), 3600.0)       # the cached settlement has passed
    assert len(client.calls) == 3
    for _ in range(5):
      clock("ONEUSDTM")
    assert len(client.calls) == 3                              # lookups never fetch

  def test_a_failure_after_a_success_keeps_the_last_clock(self, caplog):
    now = [self.T]
    client = _FundingClockClient(funding_time_s=_utc_ts(21, 1))
    clock = _FundingClock(client, time_fn=lambda: now[0])
    clock.refresh(["ONEUSDTM"])
    client.fail = True
    now[0] = self.T + 20 * 60
    with caplog.at_level("WARNING"):
      clock.refresh(["ONEUSDTM"])
      assert clock("ONEUSDTM") == (_utc_ts(21, 1), 3600.0)
    assert any("last cached clock" in r.message for r in caplog.records)

  def test_a_failing_endpoint_is_asked_once_per_retry_window_not_once_per_poll(self, caplog):
    """S3 backoff: 5 polls 60s apart against a dead endpoint cost ONE clock call (and one history call),
    and a single warning — not five timeouts."""
    now = [self.T]
    client = _FundingClockClient(fail=True)
    clock = _FundingClock(client, time_fn=lambda: now[0])
    with caplog.at_level("WARNING"):
      for _ in range(5):
        clock.refresh({"ONEUSDTM": self.T - 600})
        now[0] += 60
    assert len(client.calls) == 1 and len(client.history_calls) == 1
    assert sum("CARRY CLOCK" in r.message for r in caplog.records) == 1
    now[0] = self.T + _FundingClock.RETRY_SEC + 1
    clock.refresh({"ONEUSDTM": self.T - 600})
    assert len(client.calls) == 2                              # retried after the window

  def test_the_history_is_read_until_the_first_payment_then_never_again(self):
    fill = self.T
    now = [fill + 60]
    client = _FundingClockClient(funding_time_s=_utc_ts(21, 1))
    clock = _FundingClock(client, time_fn=lambda: now[0])
    clock.refresh({"ONEUSDTM": fill})
    assert clock.settlement_since("ONEUSDTM", fill) == (None, fill + 60)
    now[0] = fill + 5 * 60
    clock.refresh({"ONEUSDTM": fill})
    assert len(client.history_calls) == 1                      # no settlement since the last read
    now[0] = _utc_ts(21, 1, 1)                                 # 01:00 passed (history lags: not listed)
    client.funding_time_s = _utc_ts(21, 2)
    clock.refresh({"ONEUSDTM": fill})
    assert clock.settlement_since("ONEUSDTM", fill) == (_utc_ts(21, 1), None)   # the clock's promise
    calls = len(client.history_calls)
    now[0] = _utc_ts(21, 3)
    clock.refresh({"ONEUSDTM": fill})
    assert len(client.history_calls) == calls                  # known: never asked again
    clock.refresh({})                                           # position gone: its record is dropped
    assert clock.settlement_since("ONEUSDTM", fill) == (None, None)

  def test_no_client_or_junk_payload_is_unknown_never_an_exception(self):
    none = _FundingClock(None)
    none.refresh(["ONEUSDTM"])
    assert none("ONEUSDTM") is None
    junk = _FundingClock(_FundingClockClient(funding_time_s=_utc_ts(21, 1), granularity_ms=0))
    junk.refresh({"ONEUSDTM": 1.0, "": 2.0, "BAD": "x"})
    assert junk("ONEUSDTM") is None
    assert junk.refresh(None) == 0 and junk.settlement_since("ONEUSDTM", "x") == (None, None)


# ── exitDiscipline's benchmark: the live exit stack replayed after an agent close (2026-09-25) ────────

import src.main as _main_mod  # noqa: E402
from src.main import _score_exit_probe_stacks  # noqa: E402


class _KlineFake:
  """get_candles only: 1m FUTURES rows [ts_ms, o, HIGH, LOW, close, vol, turnover] from ``closes``."""

  def __init__(self, closes=None, *, fail=False):
    self.closes = dict(closes or {})
    self.fail = fail
    self.calls = []

  def get_candles(self, fsym, granularity=1, start_at=None, end_at=None):
    self.calls.append((fsym, granularity, start_at, end_at))
    if self.fail:
      raise RuntimeError("kline endpoint down")
    return [[t * 1000, c, c, c, c, 1.0, 1.0] for t, c in sorted(self.closes.items())
            if start_at // 1000 <= t <= end_at // 1000]


def _agent_probe_row(tmp_path, *, closed_ago_h=9.0, symbol="DASH-USDT", name="xp.json"):
  """A real store holding one AGENT-closed exit probe with its replay inputs, closed ``closed_ago_h`` ago."""
  import time as _t
  from src.memory import MemoryStore
  store = MemoryStore(str(tmp_path / name))
  store.record_exit_probe(symbol, "long", 100.0, 98.0, 106.0, 100.2, realized_r=0.1, closed_by="agent",
                          setup_family="funding_carry", fill_ts=1.0, init_risk_px=2.0, noise_band_r=0.5)
  data = store._read()
  row = data["exit_probes"][-1]
  row["ts"] = int(_t.time() - closed_ago_h * 3600)
  row["fillTs"] = float(row["ts"] - 1800)
  store._write(data)
  return store, row["ts"], row["fillTs"]


def _flat_closes(fill_ts, minutes, price=100.2):
  base = int(fill_ts) // 60 * 60 + 60
  return {base + 60 * i: price for i in range(minutes)}


def test_the_stack_scorer_uses_protections_effective_cfg_and_the_probes_original_risk(tmp_path, monkeypatch):
  from src.protection import ProtectionManager, replay_protection_stack
  from tests.test_protection import _cfg as _pp_cfg
  store, ts0, fill = _agent_probe_row(tmp_path)
  client = _KlineFake(_flat_closes(fill, 30 + 8 * 60 + 5))
  pm = ProtectionManager(_pp_cfg(), None, breakeven_cost_pct=0.0032)
  captured = {}

  def _spy(bars, **kw):
    captured.update(kw)
    return replay_protection_stack(bars, **kw)

  monkeypatch.setattr(_main_mod, "replay_protection_stack", _spy)
  assert _score_exit_probe_stacks(store, client, pm.cfg) == 1
  assert captured["cfg"] is pm.cfg                       # the EFFECTIVE cfg, not the raw config
  assert captured["cfg"].breakeven_fee_pct == pytest.approx(0.0032)
  assert captured["init_risk"] == pytest.approx(2.0)
  assert captured["fill_ts"] == pytest.approx(fill)
  assert captured["end_ts"] == pytest.approx(ts0 + 8 * 3600)   # the bracket probe's own horizon
  assert captured["open_until_ts"] == ts0 and captured["noise_band_r"] == pytest.approx(0.5)
  assert all(c[0] == "DASHUSDTM" and c[1] == 1 for c in client.calls)
  stack = store.exit_probes()[0]["stack"]
  assert stack["resolvedBy"] == "expired" and stack["stackR"] == pytest.approx(0.1)
  assert stack["source"] == "live_1m_replay"
  assert _score_exit_probe_stacks(store, client, pm.cfg) == 0       # scored once


def test_a_raising_kline_client_leaves_stackR_unset_and_never_raises(tmp_path, caplog):
  from tests.test_protection import _cfg as _pp_cfg
  store, _ts0, _fill = _agent_probe_row(tmp_path)
  attempts = {}
  with caplog.at_level("WARNING"):
    assert _score_exit_probe_stacks(store, _KlineFake(fail=True), _pp_cfg(), attempts=attempts) == 0
  assert "stack" not in store.exit_probes()[0]
  assert any("EXIT STACK" in r.message for r in caplog.records)
  client = _KlineFake(fail=True)
  _score_exit_probe_stacks(store, client, _pp_cfg(), attempts=attempts)
  assert client.calls == []                              # retried at most every 15 min, not every poll


def test_a_replay_that_never_succeeds_is_recorded_unavailable_after_a_day(tmp_path):
  from tests.test_protection import _cfg as _pp_cfg
  store, _ts0, _fill = _agent_probe_row(tmp_path, closed_ago_h=8 + 25)
  assert _score_exit_probe_stacks(store, _KlineFake(fail=True), _pp_cfg()) == 0
  stack = store.exit_probes()[0]["stack"]
  assert stack["stackR"] is None and stack["resolvedBy"] == "unavailable"


def test_the_scorer_waits_for_the_horizon_and_caps_its_work_per_poll(tmp_path):
  from tests.test_protection import _cfg as _pp_cfg
  young, _ts, fill = _agent_probe_row(tmp_path, closed_ago_h=2.0, name="young.json")
  client = _KlineFake(_flat_closes(fill, 600))
  assert _score_exit_probe_stacks(young, client, _pp_cfg()) == 0 and client.calls == []
  store, _ts0, fill = _agent_probe_row(tmp_path, name="many.json")
  for sym in ("INJ-USDT", "KCS-USDT"):
    store.record_exit_probe(sym, "long", 100.0, 98.0, 106.0, 100.2, realized_r=0.1, closed_by="agent",
                            fill_ts=fill, init_risk_px=2.0)
  data = store._read()
  for row in data["exit_probes"]:
    row["ts"], row["fillTs"] = data["exit_probes"][0]["ts"], fill
  store._write(data)
  client = _KlineFake(_flat_closes(fill, 30 + 8 * 60 + 5))
  assert _score_exit_probe_stacks(store, client, _pp_cfg()) == 2           # at most 2 per poll
  assert _score_exit_probe_stacks(store, client, _pp_cfg()) == 1
  assert _score_exit_probe_stacks(store, None, _pp_cfg()) == 0             # no client: nothing, no raise


def test_bars_that_stop_short_of_the_horizon_are_not_stored_as_the_stack(tmp_path, caplog):
  """T4: a kline gap / short page must never be scored as the 8h stack — the row is written once, forever,
  and a replay on truncated bars marks to market at the last bar and reads like a normal 'expired'."""
  from tests.test_protection import _cfg as _pp_cfg
  store, _ts0, fill = _agent_probe_row(tmp_path)
  client = _KlineFake(_flat_closes(fill, 2 * 60))            # bars only for the first 2h after the fill
  with caplog.at_level("WARNING"):
    assert _score_exit_probe_stacks(store, client, _pp_cfg()) == 0
  assert "stack" not in store.exit_probes()[0]
  assert any("incomplete window" in r.message for r in caplog.records)
  old, _ts, fill = _agent_probe_row(tmp_path, closed_ago_h=8 + 25, name="old.json")
  assert _score_exit_probe_stacks(old, _KlineFake(_flat_closes(fill, 2 * 60)), _pp_cfg()) == 0
  assert old.exit_probes()[0]["stack"]["resolvedBy"] == "unavailable"


def test_the_scorer_spends_its_budget_on_agent_rows_oldest_first(tmp_path):
  """T6: only agent closes are benchmarked (protection closes stay on the bracket), and the 2-per-poll
  budget goes to the oldest first — a protection row, which now also carries replay inputs, must never
  take a slot or a kline call."""
  from tests.test_protection import _cfg as _pp_cfg
  store, ts0, fill = _agent_probe_row(tmp_path, name="order.json")
  store.record_exit_probe("SOL-USDT", "long", 100.0, 98.0, 106.0, 100.2, realized_r=0.1,
                          closed_by="protection", fill_ts=fill, init_risk_px=2.0)
  for sym in ("INJ-USDT", "KCS-USDT"):
    store.record_exit_probe(sym, "long", 100.0, 98.0, 106.0, 100.2, realized_r=0.1, closed_by="agent",
                            fill_ts=fill, init_risk_px=2.0)
  data = store._read()
  off = {"DASH-USDT": 0, "SOL-USDT": -120, "INJ-USDT": 60, "KCS-USDT": 120}
  for row in data["exit_probes"]:
    row["ts"], row["fillTs"] = ts0 + off[row["symbol"]], fill
  # Newest first on disk, so a scorer that merely took rows in stored order would pick KCS first.
  data["exit_probes"] = sorted(data["exit_probes"], key=lambda r: -r["ts"])
  store._write(data)
  client = _KlineFake(_flat_closes(fill, 30 + 8 * 60 + 10))
  assert _score_exit_probe_stacks(store, client, _pp_cfg()) == 2
  assert {r["symbol"] for r in store.exit_probes() if r.get("stack")} == {"DASH-USDT", "INJ-USDT"}
  assert _score_exit_probe_stacks(store, client, _pp_cfg()) == 1
  assert _score_exit_probe_stacks(store, client, _pp_cfg()) == 0
  assert "stack" not in next(r for r in store.exit_probes() if r["symbol"] == "SOL-USDT")
  assert all(c[0] != "SOLUSDTM" for c in client.calls)


def test_the_loop_scores_stacks_with_protections_cfg_after_settling_and_records_the_inputs():
  """Wiring, not just helpers: the loop passes protection.cfg (effective) after the settle step, the
  exit-probe record site passes the replay inputs and bias tags, and the agent gets the live clock."""
  import inspect
  loop_src = inspect.getsource(_main_mod.trading_loop)
  settle = loop_src.index("_settle_probes_on_futures(memory, _measure_client, snapshot, live_prices,")
  score = loop_src.index("_score_exit_probe_stacks(memory, _measure_client, protection.cfg, attempts=_stack_attempts,")
  assert settle < score
  assert loop_src.index("protection_actions = protection.run(snapshot)") < settle
  assert "**exit_probe_inputs(_ctx, position_side)" in loop_src
  assert "funding_clock=_trade_context.funding_clock" in loop_src


def test_exit_probe_inputs_are_the_trades_own_data():
  import datetime as _dt
  from src.position_context import exit_probe_inputs
  fill = _dt.datetime(2026, 9, 24, 10, 18, tzinfo=_dt.UTC).timestamp()
  ctx = {"setupFamily": "funding_carry", "fillTs": fill, "fillPrice": 13.8, "stopLossPrice": 13.62,
         "stopAtrMult": 2.5, "funding": {"rate": -0.0005, "intervalSec": 14400.0,
                                         "nextSettlementTs": _dt.datetime(2026, 9, 24, 12, tzinfo=_dt.UTC).timestamp()},
         "regime": {"intraday_bias_15m": "bearish", "intraday_bias_1h": "bearish",
                    "intraday_bias_4h": "bullish", "daily_bias": "bullish"}}
  out = exit_probe_inputs(ctx, "long")
  assert out["fill_ts"] == fill and out["init_risk_px"] == pytest.approx(0.18)
  assert out["noise_band_r"] == pytest.approx(0.4)
  assert out["hold_until_ts"] == _dt.datetime(2026, 9, 24, 12, tzinfo=_dt.UTC).timestamp()  # 4h clock, not 16:00
  assert out["counter_at_entry"] is True and out["htf_aligned"] is True
  assert out["entry_bias"] == {"15m": "bearish", "1h": "bearish", "4h": "bullish", "1D": "bullish"}
  short = exit_probe_inputs(ctx, "short")
  assert short["counter_at_entry"] is False and short["htf_aligned"] is False
  assert exit_probe_inputs(None, "long") == {}
  bare = exit_probe_inputs({"fillTs": fill, "entryPrice": 1.0, "stopLossPrice": 0.9}, "long")
  assert bare["hold_until_ts"] is None and bare["counter_at_entry"] is None and bare["entry_bias"] is None
  # BOTH 15m and 1h must oppose (one is not the entry premise), BOTH 4h and 1D must agree.
  from src.position_context import bias_tags
  assert bias_tags({"15m": "bearish", "1h": "bullish", "4h": "bullish", "1D": "neutral"}, "long") == (False, False)
  assert bias_tags({"15m": "neutral", "1h": "bearish", "4h": "neutral", "1D": "bullish"}, "long") == (False, False)
  assert bias_tags({"15m": "bearish"}, "long") == (None, None)            # untagged, never a "no"
  assert bias_tags({"15m": "bearish", "1h": "bearish"}, "sideways") == (None, None)


def test_the_funding_clock_keeps_the_rate_it_read_without_another_call():
  clock = _FundingClock(_FundingClockClient(funding_time_s=_utc_ts(21, 1)), time_fn=lambda: _utc_ts(21, 0, 24))
  assert clock.rate("ONEUSDTM") is None
  clock.refresh(["ONEUSDTM"])
  assert clock.rate("ONEUSDTM") == pytest.approx(-0.00196)
  clock("ONEUSDTM")
  assert len(clock._client.calls) == 1


# ── Order-lease extremes for the execution map's counterfactual (2026-09-25) ──────────────────────────


class _LeaseBars:
  """get_candles (+ marks/funding, so it can stand in for the whole futures client in a settle)."""

  def __init__(self, bars=None, *, fail=False, spot_order=False):
    # bars: {ts_s: (open, high, low, close)}
    self.bars = dict(bars or {})
    self.fail, self.spot_order = fail, spot_order
    self.calls = []

  def get_candles(self, fsym, granularity=1, start_at=None, end_at=None):
    self.calls.append((fsym, granularity, start_at, end_at))
    if self.fail:
      raise RuntimeError("kline endpoint down")
    out = []
    for t, (o, h, l, c) in sorted(self.bars.items()):
      if start_at // 1000 <= t <= end_at // 1000:
        # FUTURES rows are [ts, open, HIGH, LOW, close]; spot_order simulates the misread trap.
        out.append([t * 1000, o, c, h, l, 1.0, 1.0] if self.spot_order else [t * 1000, o, h, l, c, 1.0, 1.0])
    return out

  def get_mark_price(self, symbol):
    return {"symbol": symbol, "value": 1.0}

  def get_funding_rate_history(self, symbol, start_at=None, end_at=None):
    return []


def _lease_bars(t0, n=20, *, spike_at=None):
  """1m bars from the minute of ``t0``: flat 1.00 with a 0.95 low (and 1.04 high) at ``spike_at``."""
  base = int(t0) // 60 * 60
  bars = {}
  for i in range(n):
    t = base + 60 * i
    low, high = (0.95, 1.04) if t == spike_at else (0.995, 1.005)
    bars[t] = (1.0, high, low, 1.0)
  return bars


class TestPollLeaseExtremes:
  T = 1_790_000_030          # 30s into a minute

  def test_the_window_is_the_lease_own_minutes_and_one_fetch_serves_the_symbol(self):
    from src.main import _PollLeaseExtremes
    first_inside = (self.T // 60 + 1) * 60
    # A spike in the call's own minute (before the call) must NOT count; one inside the lease must.
    client = _LeaseBars(_lease_bars(self.T, spike_at=self.T // 60 * 60))
    lease = _PollLeaseExtremes(client)
    assert lease("SPX-USDT", self.T, self.T + 900) == (0.995, 1.005)
    client.bars = _lease_bars(self.T, spike_at=first_inside + 600)
    lease = _PollLeaseExtremes(client)
    assert lease("SPX-USDT", self.T, self.T + 900) == (0.95, 1.04)
    assert lease("SPX-USDT", self.T + 60, self.T + 600) == (0.995, 1.005)   # served from the same fetch
    fetches = [c for c in client.calls if c[0] == "SPXUSDTM"]
    assert len(fetches) == 2 and all(c[1] == 1 for c in fetches)           # one per poll-instance, 1m bars

  def test_misread_columns_and_a_dead_endpoint_give_none_with_a_warning_and_no_hammering(self, caplog):
    from src.main import _PollLeaseExtremes
    with caplog.at_level("WARNING"):
      assert _PollLeaseExtremes(_LeaseBars(_lease_bars(self.T), spot_order=True))("SPX-USDT", self.T, self.T + 900) is None
      dead = _LeaseBars(fail=True)
      lease = _PollLeaseExtremes(dead)
      assert lease("SPX-USDT", self.T, self.T + 900) is None
      assert lease("SPX-USDT", self.T, self.T + 900) is None
    assert len(dead.calls) == 1
    assert sum("PROBE LEASE" in r.message and r.levelname == "WARNING" for r in caplog.records) == 2
    assert _PollLeaseExtremes(None)("SPX-USDT", self.T, self.T + 900) is None

  def test_the_settle_step_passes_the_lease_lookup(self):
    from src.main import _PollLeaseExtremes
    memory = _SettleCaptureMemory()
    _settle_probes_on_futures(memory, _SettleFuturesClient(), SimpleNamespace(futures_positions=[]), {})
    (_prices, kwargs), = memory.signal_calls
    assert isinstance(kwargs["lease_extremes"], _PollLeaseExtremes)
    memory = _SettleCaptureMemory()
    _settle_probes_on_futures(memory, None, SimpleNamespace(futures_positions=[]), {})
    assert memory.signal_calls[0][1]["lease_extremes"] is None

  def test_end_to_end_a_real_store_gets_its_lease_low_and_high(self, tmp_path):
    import time as _t
    from src.memory import MemoryStore
    store = MemoryStore(str(tmp_path / "m.json"))
    store.record_signal_probe("SPX-USDT", "buy", 1.0, "continuation", atr15_pct=1.0, lease_min=15)
    data = store._read()
    data["signal_probes"][0]["ts"] = int(_t.time()) - 17 * 60
    store._write(data)
    ts = data["signal_probes"][0]["ts"]
    client = _LeaseBars(_lease_bars(ts, n=30, spike_at=(ts // 60 + 3) * 60))
    _settle_probes_on_futures(store, client, SimpleNamespace(futures_positions=[]), {})
    ctx = store.signal_probes(limit=0)[0]["entryContext"]
    assert (ctx["leaseLow"], ctx["leaseHigh"]) == (0.95, 1.04)


# ── Market state: refreshed hourly by the loop, read for free everywhere else (2026-09-25) ────────────

from src.main import _MarketStateClock  # noqa: E402


class _MSFutures:
  """list_active_contracts + XBTUSDTM klines in FUTURES column order ([ts_ms, o, HIGH, LOW, c, v, t])."""

  def __init__(self, *, fail_contracts=False, fail_candles=False, spot_order=False):
    self.fail_contracts, self.fail_candles, self.spot_order = fail_contracts, fail_candles, spot_order
    self.contract_calls, self.candle_calls = 0, []

  def list_active_contracts(self):
    self.contract_calls += 1
    if self.fail_contracts:
      raise RuntimeError("contracts endpoint down")
    first = 1_600_000_000_000
    return [{"symbol": s, "priceChgPct": c, "turnoverOf24h": 5e7, "status": "Open", "firstOpenDate": first,
             "assetClass": "CRYPTO"}
            for s, c in (("XBTUSDTM", -0.01), ("ETHUSDTM", 0.02), ("SOLUSDTM", -0.03), ("WIFUSDTM", 0.04))]

  def get_candles(self, symbol, granularity=1, start_at=None, end_at=None):
    self.candle_calls.append((symbol, granularity))
    if self.fail_candles:
      raise RuntimeError("klines down")
    step = granularity * 60
    rows = []
    t = (start_at // 1000) // step * step
    i = 0
    while t + step <= end_at // 1000:
      c = 100.0 + i
      o, h, l = c - 0.5, c + 1.0, c - 1.0
      # A spot-ordered row ([ts, open, close, high, low]) puts the close where the high belongs.
      rows.append([t * 1000, o, c, h, l, 1, 1] if self.spot_order else [t * 1000, o, h, l, c, 1, 1])
      t += step
      i += 1
    return rows


class TestMarketStateClock:
  def _clock(self, client, now):
    return _MarketStateClock(client, min_turnover=5e6, min_age_days=7, time_fn=lambda: now[0])

  def test_refreshes_at_most_hourly_and_reads_are_cache_only(self):
    now = [2_000_000_000.0]
    client = _MSFutures()
    clock = self._clock(client, now)
    assert clock.current() is None                  # nothing until the loop refreshes
    assert clock.refresh() is True
    state = clock.current()
    assert state["breadth24"] == 0.5 and state["universe"] == 4 and state["btc24h"] == -1.0
    assert "btc72h" in state and "btcDailyAdx" in state and state["btcDailyBias"] in ("bullish", "bearish", "neutral")
    calls = (client.contract_calls, len(client.candle_calls))
    now[0] += 1800
    assert clock.refresh() is False and clock.current() == state
    assert (client.contract_calls, len(client.candle_calls)) == calls   # no network inside the hour
    now[0] += 1801
    assert clock.refresh() is True and client.contract_calls == 2

  def test_a_failed_refresh_keeps_the_last_reading_and_backs_off(self, caplog):
    import logging
    now = [2_000_000_000.0]
    client = _MSFutures()
    clock = self._clock(client, now)
    clock.refresh()
    client.fail_contracts = True
    now[0] += 3601
    with caplog.at_level(logging.WARNING):
      assert clock.refresh() is False
      now[0] += 60
      assert clock.refresh() is False                # backing off: not even attempted
    assert client.contract_calls == 2
    assert sum("MARKET STATE" in r.getMessage() for r in caplog.records) == 1   # once per streak
    assert clock.current() is not None               # the last reading, still under two periods old
    now[0] += 2 * 3600
    assert clock.current() is None                   # too old to stamp as current

  def test_a_misordered_kline_drops_the_btc_fields_not_the_breadth(self):
    clock = self._clock(_MSFutures(spot_order=True), [2_000_000_000.0])
    assert clock.refresh() is True
    state = clock.current()
    assert state["breadth24"] == 0.5 and "btc72h" not in state and "btcDailyAdx" not in state

  def test_never_raises_without_a_client(self):
    clock = _MarketStateClock(None, min_turnover=5e6, min_age_days=7)
    assert clock.refresh() is False and clock.current() is None

  def test_the_loop_refreshes_after_survival_and_hands_the_reader_to_the_agent(self):
    import inspect
    import src.main as main_mod
    src = inspect.getsource(main_mod.trading_loop)
    assert "_market_state = _MarketStateClock(" in src
    refresh = src.index("_market_state.refresh()")
    assert src.index("protection_actions = protection.run(snapshot)") < refresh
    assert src.index("_score_exit_probe_stacks(memory, _measure_client, protection.cfg") < refresh
    assert refresh < src.index("agent_task = asyncio.create_task(_run_in_daemon_thread(")
    assert "market_state=_market_state.current,\n      ))" in src


# ── Gate scoreboard: settled state rows fold in the poll loop; one report-only line an hour ─────────


class TestGateScoreboardInThePollLoop:
  def test_the_settle_step_folds_fully_settled_state_rows(self, tmp_path):
    from src.memory import MemoryStore
    store = MemoryStore(str(tmp_path / "g.json"))
    store.record_gate_state_probes("A-USDT", 100.0, {"long": ["h1_align"], "short": []}, price_source="futures_mark")
    data = store._read()
    for row in data["gate_probes"]:
      row["entryContext"]["signalProbe"] = {"m5": 100.0, "m15": 100.0, "m60": 101.0, "m240": 102.0}
    store._write(data)
    _settle_probes_on_futures(store, None, SimpleNamespace(futures_positions=[]), {})
    assert store.gate_probes() == []
    (cell,) = store.gate_state_days().values()
    assert cell["long"]["240"][2]["h1_align"][0] == 1

  def test_a_failing_fold_is_a_warning_and_settlement_still_ran(self, caplog):
    memory = _SettleCaptureMemory()
    with caplog.at_level("WARNING"):
      _settle_probes_on_futures(memory, None, SimpleNamespace(futures_positions=[]), {})
    assert memory.signal_calls and memory.exit_calls
    assert any("GATE STATE FOLD failed" in r.message for r in caplog.records)

  def test_the_line_is_logged_at_most_hourly_and_never_raises(self, tmp_path, caplog):
    from src.main import _log_gate_scoreboard
    from src.memory import MemoryStore
    store = MemoryStore(str(tmp_path / "g.json"))
    cfg = SimpleNamespace(trading=SimpleNamespace(estimated_slippage_pct=0.001, slippage_autotune_min_samples=8))
    last: dict = {}
    with caplog.at_level("INFO"):
      assert _log_gate_scoreboard(store, cfg, last, now=10_000.0) is True
      assert _log_gate_scoreboard(store, cfg, last, now=10_000.0 + 1800) is False
      assert _log_gate_scoreboard(store, cfg, last, now=10_000.0 + 3600) is True
    lines = [r.message for r in caplog.records if "GATE SCOREBOARD (report-only)" in r.message]
    assert len(lines) == 2 and "nothing scored yet" in lines[0]

    class _Broken:
      def __getattr__(self, name):
        raise RuntimeError("store gone")

    caplog.clear()
    with caplog.at_level("INFO"):
      assert _log_gate_scoreboard(_Broken(), cfg, {}, now=10_000.0) is True   # no raise
    assert any("GATE SCOREBOARD (report-only): unavailable" in r.message for r in caplog.records)

  def test_the_loop_logs_it_after_survival_and_settlement(self):
    import inspect
    import src.main as main_mod
    src = inspect.getsource(main_mod.trading_loop)
    log = src.index("_log_gate_scoreboard(memory, cfg, _gate_log_last)")
    assert src.index("protection_actions = protection.run(snapshot)") < log
    assert src.index("_settle_probes_on_futures(memory, _measure_client, snapshot, live_prices,") < log
    assert "memory.fold_settled_gate_states(gate_state_cells)" in inspect.getsource(main_mod._settle_probes_on_futures)
