"""Tests for run-item extraction: handoffs, per-agent attribution, and research activity.

Exercises src.agent._collect_run_artifacts / _summarize_research with REAL Agents-SDK item
instances so the logic that feeds Telegram (issue 1) and the dashboard (issue 4) is regression-
guarded without needing to spin up a full Runner.run (Azure + KuCoin).
"""
from types import SimpleNamespace

from agents import Agent
from agents.items import HandoffOutputItem, ToolCallItem, ToolCallOutputItem

from src.agent import _collect_run_artifacts, _summarize_research

_TRADE = Agent(name="Trading Agent", instructions="x")
_RESEARCH = Agent(name="Research Agent", instructions="y")


def _call(name: str, cid: str) -> SimpleNamespace:
  return SimpleNamespace(name=name, call_id=cid, type="function_call")


def _out(cid: str) -> dict:
  return {"type": "function_call_output", "call_id": cid, "output": "ok"}


class TestSummarizeResearch:
  def test_log_research_note(self):
    out = {"title": "Research: AVAX breakout", "summary": "fresh listing, high vol", "author": "Research Agent"}
    assert _summarize_research("log_research", out) == "Research: AVAX breakout — fresh listing, high vol"

  def test_add_coin(self):
    out = {"added": {"symbol": "AVAX-USDT", "status": "active", "reason": "breakout"}, "coins": ["AVAX-USDT"]}
    assert _summarize_research("add_coin", out) == "added coin AVAX-USDT — breakout"

  def test_remove_coin(self):
    out = {"removed": {"symbol": "DOGE-USDT", "status": "removed", "reason": "no catalyst"}}
    assert _summarize_research("remove_coin", out) == "removed coin DOGE-USDT — no catalyst"

  def test_remove_source_uses_title(self):
    out = {"removed": {"title": "Removed Source: SketchyBlog", "summary": "low quality"}}
    assert _summarize_research("remove_source", out) == "Removed Source: SketchyBlog"

  def test_sentiment(self):
    out = {"sentiment": {"symbol": "BTC-USDT", "score": 0.7, "rationale": "ETF inflows"}}
    assert _summarize_research("log_sentiment", out) == "sentiment BTC-USDT=0.7"

  def test_noisy_market_data_suppressed(self):
    assert _summarize_research("analyze_market_context", {"summary": {"weighted_score": 0.2}}) is None

  def test_error_output_suppressed(self):
    assert _summarize_research("add_coin", {"error": "not found"}) is None


class TestCollectRunArtifacts:
  def test_full_round_trip(self):
    items = [
      ToolCallItem(agent=_TRADE, raw_item=_call("decline_trade", "c1")),
      ToolCallOutputItem(agent=_TRADE, raw_item=_out("c1"), output={"skipped": True, "reason": "no edge"}),
      HandoffOutputItem(agent=_TRADE, raw_item=_out("h1"), source_agent=_TRADE, target_agent=_RESEARCH),
      ToolCallItem(agent=_RESEARCH, raw_item=_call("log_research", "c2")),
      ToolCallOutputItem(agent=_RESEARCH, raw_item=_out("c2"),
                         output={"title": "Research: AVAX", "summary": "fresh", "author": "Research Agent"}),
      ToolCallItem(agent=_RESEARCH, raw_item=_call("add_coin", "c3")),
      ToolCallOutputItem(agent=_RESEARCH, raw_item=_out("c3"),
                         output={"added": {"symbol": "AVAX-USDT", "status": "active", "reason": "breakout"}}),
      HandoffOutputItem(agent=_RESEARCH, raw_item=_out("h2"), source_agent=_RESEARCH, target_agent=_TRADE),
    ]
    art = _collect_run_artifacts(items)

    assert art["handoffs"] == [
      {"from": "Trading Agent", "to": "Research Agent"},
      {"from": "Research Agent", "to": "Trading Agent"},
    ]
    assert art["research"] == ["Research: AVAX — fresh", "added coin AVAX-USDT — breakout"]
    assert sorted(art["agents_used"]) == ["Research Agent", "Trading Agent"]
    # All three ToolCallOutputItems are collected regardless of producing agent (decline + 2 research).
    assert len(art["tool_outputs"]) == 3

  def test_trading_tool_outputs_not_summarized_as_research(self):
    # An order placed by the Trading Agent must not appear in research activity.
    items = [
      ToolCallItem(agent=_TRADE, raw_item=_call("place_market_order", "c1")),
      ToolCallOutputItem(agent=_TRADE, raw_item=_out("c1"),
                         output={"orderId": "z9", "orderRequest": {"side": "buy"}}),
    ]
    art = _collect_run_artifacts(items)
    assert art["research"] == []
    assert art["handoffs"] == []
    assert art["tool_outputs"] == [{"orderId": "z9", "orderRequest": {"side": "buy"}}]

  def test_empty(self):
    art = _collect_run_artifacts([])
    assert art == {"tool_outputs": [], "handoffs": [], "research": [], "agents_used": set()}
