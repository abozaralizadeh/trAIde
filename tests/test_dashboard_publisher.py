from types import SimpleNamespace

from src.dashboard_publisher import DashboardPublisher


def _publisher(disclosure: str = "normalized") -> DashboardPublisher:
  cfg = SimpleNamespace(dashboard=SimpleNamespace(disclosure=disclosure))
  return DashboardPublisher(cfg)


class TestSanitizeDecisionHandoffMarking:
  def test_handoff_to_research_is_marked(self):
    pub = _publisher()
    out = pub._sanitize_decision({
      "symbol": "ALL", "action": "handoff_to_research", "confidence": 0.0,
      "reason": "Trading Agent → Research Agent", "ts": 100, "day": 1,
    })
    assert out["isHandoff"] is True
    assert out["agent"] == "research"
    assert out["handoffTo"] == "research"
    assert out["action"] == "handoff_to_research"

  def test_handoff_to_trading_is_marked(self):
    pub = _publisher()
    out = pub._sanitize_decision({
      "symbol": "ALL", "action": "handoff_to_trading", "confidence": 0.0,
      "reason": "Research Agent → Trading Agent", "ts": 101, "day": 1,
    })
    assert out["isHandoff"] is True
    assert out["agent"] == "trading"
    assert out["handoffTo"] == "trading"

  def test_regular_decision_attributed_to_trading(self):
    pub = _publisher()
    out = pub._sanitize_decision({
      "symbol": "BTC-USDT", "action": "spot_buy_limit", "confidence": 0.7,
      "reason": "bullish", "ts": 102, "day": 1,
    })
    assert out["agent"] == "trading"
    assert "isHandoff" not in out

  def test_handoff_has_no_win_or_pnl(self):
    pub = _publisher(disclosure="absolute")
    out = pub._sanitize_decision({
      "symbol": "ALL", "action": "handoff_to_research", "confidence": 0.0,
      "reason": "x", "ts": 103, "day": 1, "pnl": None,
    })
    assert "win" not in out and "pnl" not in out


class TestSanitizeCoins:
  def test_active_coins_come_first_then_by_recency(self):
    pub = _publisher()
    out = pub._sanitize_coins([
      {"symbol": "OLD-USDT", "status": "removed", "reason": "stale", "ts": 50},
      {"symbol": "ETH-USDT", "status": "active", "reason": "liquid major", "ts": 100},
      {"symbol": "SOL-USDT", "status": "active", "reason": "trend", "ts": 200},
    ])
    assert [c["symbol"] for c in out] == ["SOL-USDT", "ETH-USDT", "OLD-USDT"]
    assert out[0]["status"] == "active" and out[-1]["status"] == "removed"

  def test_coins_are_public_safe_fields_only(self):
    pub = _publisher()
    out = pub._sanitize_coins([
      {"symbol": "BTC-USDT", "status": "active", "reason": "x" * 900, "exitPlan": "secret", "ts": 1},
    ])
    assert set(out[0].keys()) == {"symbol", "status", "reason", "ts"}
    assert len(out[0]["reason"]) == 500  # truncated

  def test_ignores_malformed_entries(self):
    pub = _publisher()
    out = pub._sanitize_coins([{"status": "active"}, "nope", {"symbol": "XRP-USDT", "status": "active"}])
    assert [c["symbol"] for c in out] == ["XRP-USDT"]
