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
