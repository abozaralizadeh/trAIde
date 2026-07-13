from src.main import _agent_made_a_move, _expired_bot_entry_orders


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
