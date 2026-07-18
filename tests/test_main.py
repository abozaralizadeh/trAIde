import pytest

from types import SimpleNamespace

from src.main import (
  _adaptive_agent_cooldown,
  _adaptive_price_trigger_threshold,
  _agent_made_a_move,
  _close_event_id,
  _expired_bot_entry_orders,
  _fetch_futures,
  _fetch_recent_fills,
  _fill_event_id,
  _next_price_noise_ewma,
  _productivity_adjusted_flat_cooldown,
  _rebase_reviewed_price_triggers,
)


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
