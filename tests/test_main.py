from src.main import _agent_made_a_move


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
