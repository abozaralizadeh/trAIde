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


# ── entryThesis on each open futures position + STEP 1b "fresh" = changed since entry (2026-09-25) ──
# STEP 1b said "fresh 15m AND 1h biases both oppose the position" -> confirmed reversal, and the model's
# position view was raw exchange data. 27 of 31 historical fires were already true at the fill: DASH and
# KCS were funding_carry longs entered with 15m/1h bearish and later closed on that same condition.

import datetime as _dt  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import pytest  # noqa: E402

import src.agent as _agent_mod  # noqa: E402
from src.agent import TradingSnapshot, _attach_entry_theses, _format_snapshot  # noqa: E402

_FILL = _dt.datetime(2026, 9, 24, 10, 18, tzinfo=_dt.timezone.utc).timestamp()


def _thesis_store(tmp_path, *, family="funding_carry", regime=True):
  """A real MemoryStore holding the filled KCS-shaped carry LONG, entered with 15m and 1h BEARISH."""
  from src.memory import MemoryStore
  store = MemoryStore(str(tmp_path / "thesis.json"))
  ctx = {"positionSide": "long", "setupFamily": family, "entryPrice": 13.80, "stopLossPrice": 13.62,
         "takeProfitPrice": 14.14, "stopAtrMult": 2.5, "confidence": 0.62,
         "funding": {"rate": -0.0005, "intervalSec": 14400.0,
                     "nextSettlementTs": _dt.datetime(2026, 9, 24, 12, tzinfo=_dt.timezone.utc).timestamp()}}
  if regime:
    ctx["regime"] = {"intraday_bias_15m": "bearish", "intraday_bias_1h": "bearish",
                     "intraday_bias_4h": "bullish", "daily_bias": "bullish", "intraday_atr_pct": 0.52}
  store.record_trade("KCS-USDT", "buy", 5.0, price=13.80, venue="futures", entry_context=ctx)
  data = store._read()
  data["trades"][-1]["fillTs"] = int(_FILL)
  data["trades"][-1]["fillPrice"] = 13.80
  store._write(data)
  return store


def _thesis_state(mark=13.89):
  pos = {"symbol": "KCSUSDTM", "currentQty": 3, "positionSide": "BOTH", "avgEntryPrice": 13.80,
         "markPrice": mark, "openingTimestamp": int(_FILL * 1000)}
  snap = TradingSnapshot(coins=["KCS-USDT"], tickers={}, balances=[], paper_trading=False,
                         max_position_usd=50.0, min_confidence=0.5, max_leverage=5.0, futures_enabled=True,
                         futures_positions=[pos])
  return json.loads(_format_snapshot(snap, {}))


class _Clock:
  """The live funding clock's interface: (next_settlement_ts, interval_sec) and a cached rate."""

  def __init__(self, next_ts, interval, rate):
    self.value = (next_ts, interval)
    self._rate = rate

  def __call__(self, fsym):
    return self.value

  def rate(self, fsym):
    return self._rate


class TestEntryThesis:
  NOW = _FILL + 30 * 60

  def test_each_position_carries_the_entrys_own_record(self, tmp_path):
    state = _thesis_state()
    clock = _Clock(_dt.datetime(2026, 9, 24, 12, tzinfo=_dt.timezone.utc).timestamp(), 14400.0, 0.0001)
    assert _attach_entry_theses(state, _thesis_store(tmp_path), self.NOW, funding_clock=clock) == 1
    th = state["futuresPositions"][0]["entryThesis"]
    assert th["entryBias"] == {"15m": "bearish", "1h": "bearish", "4h": "bullish", "1D": "bullish"}
    assert th["setupFamily"] == "funding_carry" and th["entryConfidence"] == pytest.approx(0.62)
    assert th["fillPrice"] == pytest.approx(13.80) and th["plannedStop"] == pytest.approx(13.62)
    assert th["plannedTp"] == pytest.approx(14.14)
    assert th["currentR"] == pytest.approx(0.5)                   # (13.89 - 13.80) / 0.18
    assert th["noiseBandR"] == pytest.approx(0.4) and th["entryAtr15mPct"] == pytest.approx(0.52)
    assert th["holdUntil"] == "2026-09-24T12:00:00Z" and th["carryHoldActive"] is True
    assert th["minutesToSettlement"] == pytest.approx(72.0)
    assert th["intervalHours"] == pytest.approx(4.0)
    assert th["fundingRateAtEntry"] == pytest.approx(-0.0005)
    assert th["fundingRateNow"] == pytest.approx(0.0001)         # flipped to longs-pay since entry
    assert th["carryPerSettlementR"] == pytest.approx(-0.0001 / (0.18 / 13.80), abs=1e-4)

  def test_without_the_live_clock_the_stamped_clock_still_gives_the_hold(self, tmp_path):
    state = _thesis_state()
    _attach_entry_theses(state, _thesis_store(tmp_path), self.NOW)
    th = state["futuresPositions"][0]["entryThesis"]
    assert th["holdUntil"] == "2026-09-24T12:00:00Z" and th["intervalHours"] == pytest.approx(4.0)
    assert th["fundingRateNow"] is None
    assert th["carryPerSettlementR"] == pytest.approx(0.0005 / (0.18 / 13.80), abs=1e-4)

  def test_non_carry_positions_get_no_carry_fields(self, tmp_path):
    state = _thesis_state()
    _attach_entry_theses(state, _thesis_store(tmp_path, family="fade_extreme"), self.NOW)
    th = state["futuresPositions"][0]["entryThesis"]
    assert th["setupFamily"] == "fade_extreme" and "holdUntil" not in th and "fundingRateNow" not in th

  def test_missing_context_means_no_block_and_no_exception(self, tmp_path):
    from src.memory import MemoryStore
    state = _thesis_state()
    assert _attach_entry_theses(state, MemoryStore(str(tmp_path / "empty.json")), self.NOW) == 0
    assert "entryThesis" not in state["futuresPositions"][0]

    class _Broken:
      def entry_context_for_position(self, *a, **k):
        raise RuntimeError("store unreadable")

    state = _thesis_state()
    assert _attach_entry_theses(state, _Broken(), self.NOW) == 0
    assert "entryThesis" not in state["futuresPositions"][0]
    assert _attach_entry_theses({"futuresPositions": [None, "junk"]}, _Broken(), self.NOW) == 0

  def test_run_trading_agent_attaches_the_thesis_to_the_payload_it_sends(self):
    """Wiring, not just the builder: the payload the model receives is the one enriched."""
    src = inspect.getsource(_agent_mod.run_trading_agent)
    attach = src.index("_attach_entry_theses(user_state_obj, memory, time.time(), funding_clock=funding_clock)")
    assert src.index('user_state_obj = json.loads(user_state)') < attach
    assert attach < src.index("input_payload = json.dumps(user_state_obj)")
    assert inspect.signature(_agent_mod.run_trading_agent).parameters["funding_clock"].default is None


def _prompt_section(start: str, end: str) -> str:
  """The model-facing text between two headings of the trading prompt, string literals joined."""
  src = inspect.getsource(_agent_mod.run_trading_agent)
  body = src[src.index(start):src.index(end)]
  body = "\n".join(line for line in body.splitlines() if not line.strip().startswith("#"))
  return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', body))


class TestStep1bMeansChangedSinceEntry:
  def test_fresh_is_defined_against_the_entry_thesis(self):
    text = _prompt_section("## STEP 1b", "## STEP 1c")
    assert "fresh 15m AND 1h biases both oppose" not in text
    assert "entryThesis" in text and "since entry" in text.lower()
    assert "entryBias" in text
    assert "not grandfathered past thesis failure" in text        # the rule itself is kept
    assert "already present at entry" in text and "not new evidence" in text
    assert "exitDiscipline" in text

  def test_it_neither_presumes_holding_nor_argues_from_forfeited_carry(self):
    """Verdicts: the carry transfer is ~0.005-0.1R per settlement and a hold nudge is contradicted by
    G/XMR (closing helped). The text must stay neutral."""
    text = _prompt_section("## STEP 1b", "## STEP 1c").lower()
    for banned in ("forfeit", "gives up the carry", "give up the carry", "only invalidation",
                   "stop is the invalidation", "hold to settlement", "-2.06", "31 fires"):
      assert banned not in text

  def test_the_31_fire_measurement_lives_in_a_code_comment_not_the_prompt(self):
    src = inspect.getsource(_agent_mod.run_trading_agent)
    block = src[src.index("## STEP 1b"):src.index("## STEP 1c")]
    comments = "\n".join(l for l in block.splitlines() if l.strip().startswith("#"))
    assert "31 fires" in comments and "-2.06R" in comments

  def test_the_exit_legend_names_the_system_not_the_bare_bracket(self):
    src = inspect.getsource(_agent_mod.run_trading_agent)
    legend = src[src.index("EARLY CLOSES ARE MEASURED"):src.index("ACT ON THE SCOREBOARD")]
    assert "stackR" in legend and "trailing" in legend and "carry hold" in legend
    assert "legacyBracketScored" in legend and "byCounterAtEntry" in legend
    assert "against what the \"\n        \"bracket would have returned" not in legend


def _joined_literals(start: str, end: str) -> str:
  src = inspect.getsource(_agent_mod.run_trading_agent)
  body = src[src.index(start):src.index(end)]
  body = "\n".join(line for line in body.splitlines() if not line.strip().startswith("#"))
  return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', body))


class TestScoreboardTellsTheTruthBeforeTheModelProposes:
  """2026-09-24: by_family read 'edge' while the stand-aside held continuation at zero stake, so the
  model re-proposed it nine times and learned the truth only from nine refusals."""

  def test_edge_report_rows_carry_the_order_paths_stake(self):
    src = inspect.getsource(_agent_mod.run_trading_agent)
    block = src[src.index('"signalEdge": annotate_family_stakes('):][:260]
    assert '_edge_now.get("signal_edge", {"verdict": "insufficient data"})' in block
    assert "explore_factor=cfg.edge.explore_unproven_family_factor" in block
    # The flag the order path checks before zeroing a stake (tools.py) must reach the marks too.
    assert "stand_aside_enabled=cfg.edge.stand_aside_no_edge_family" in block

  def test_the_confidence_report_reaches_the_prompt_only_with_a_real_sample(self):
    src = inspect.getsource(_agent_mod.run_trading_agent)
    assert 'state["confidence_edge"] = confidence_edge_stats(' in src
    gate = src.index('_conf_rows = confidence_edge_for_prompt(_edge_now.get("confidence_edge"), model=cfg.azure.deployment)')
    assert src.index('user_state_obj["edgeReport"]["confidenceEdge"]') > gate

  def test_the_stale_sizing_claims_are_gone_and_the_stake_is_explained(self):
    text = _joined_literals('"ACT ON THE SCOREBOARD', "DIRECTIONAL HONESTY")
    assert "in proportion to the shortfall" not in text
    assert "keeps full risk" not in text
    assert "standAside is at zero stake" in text
    assert "reduced explore size" in text
    assert "'unproven'" in text and "t below 1" in text          # the two causes are distinguished
    assert "by_family_side" in text and "judgedOn" in text
    # Stakes are per side; a family is closed only when BOTH sides are (2026-09-25 review).
    assert "stakeBySide" in text and "standAsideBySide" in text and "BOTH sides" in text
    assert "one side's record never sizes the other" not in text   # false: a thin side is judged pooled
    # A refused proposal is the only evidence that can re-open a stood-aside family — the prompt must
    # not tell the model to stop making them (probe starvation freezes the stand-aside).
    assert "do not spend a turn proposing it" not in text
    assert "re-opens by itself" not in text
    assert "ONLY way a stood-aside family" in text
    assert "stop feeding it" not in _joined_literals('"SETUP FAMILIES', "DIRECTIONAL HONESTY")
    carry = _joined_literals('"FUNDING CARRY', "SCHEDULED MACRO EVENTS")
    assert "Every technical playbook you have tried so far measures no edge" not in carry

  def test_no_frozen_early_close_claim_is_left_in_the_prompt(self):
    """2026-09-25: the rewritten EARLY CLOSES paragraph kept an old 'your closes slightly BEAT the
    brackets' parenthetical — a one-sample direction claim about a benchmark no longer used, while the
    live stack measurement read the other way. History lives in a code comment, never in the prompt."""
    text = _joined_literals('"EARLY CLOSES ARE MEASURED', '"ACT ON THE SCOREBOARD')
    assert "exitDiscipline" in text and "stackR" in text
    for stale in ("BEAT", "brackets alone", "withdrawn", "data-parsing"):
      assert stale not in text

  def test_the_prompt_points_at_the_enforced_confidence_floor(self):
    src = inspect.getsource(_agent_mod.run_trading_agent)
    line = src[src.index("- Only place a trade if your confidence >="):][:400]
    assert "summary.entryGate.minConfidence" in line


class TestEntryDistanceFollowsTheLiveExecutionMap:
  """2026-09-25: the prompt taught a fixed '0.5-1.5 x ATR' band (the 8-33% fill zone, measured) and the
  Jul-30 replay figures, whose RR-filter and TTL sub-claims did not reproduce on a new 124-limit sample.
  The model now reads its own live fill rate and realized R by resting distance instead."""

  def _prompt(self):
    return _joined_literals('"**How to pick entry_price', '"## OPTIMAL ENTRY PLANNING')

  def test_the_unreproduced_figures_and_the_fixed_band_are_gone(self):
    src = inspect.getsource(_agent_mod.run_trading_agent)
    text = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))  # model-facing text only
    for banned in ("13.05R", "+0.388R", "-0.37R", "0.5–1.5 × ATR", "Farther than 1.5 ATR rarely fills",
                   "Try the cross and let that gate answer", "82 expired limits", "fills ~18% of the time"):
      assert banned not in text, banned

  def test_no_older_line_ties_crossing_to_conviction_or_calls_it_preferable(self):
    """PHIL-4 (2026-09-25): the new rule is 'crossing follows executionMap, no confidence bar', but four
    untouched lines still said cross on high conviction / crossing beats missing the move. One rule."""
    src = inspect.getsource(_agent_mod.run_trading_agent)
    text = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', "\n".join(
      l for l in src.splitlines() if not l.strip().startswith("#"))))
    for stale in ("high-conviction continuation you may take a MARKETABLE entry",
                  "marketable entry is preferable to missing the move",
                  "marketable if conviction is high",
                  "bracketed marketable CONTINUATION entry now"):
      assert stale not in text, stale
    assert "not by how confident you feel" in text
    assert "resting or marketable per executionMap" in text
    from src import tools as _tools_mod
    tsrc = inspect.getsource(_tools_mod.build_tools)
    doc = tsrc[tsrc.index("async def place_futures_limit_order("):]
    doc = doc[:doc.index("result = await _place_futures_limit_order_impl(")]
    assert "Rejects if entry_price is too close to current price" not in doc   # false at the defaults
    assert "edgeReport.executionMap" in doc

  def test_it_points_at_the_map_and_states_the_mechanism_without_figures(self):
    text = self._prompt()
    assert "edgeReport.executionMap" in text and "fillAdjustedR" in text
    assert "fill rate x realized R of fills" in text and "never planned RR" in text
    assert "fee guard, not a quality filter" in text
    assert "Marketable is its own bucket" in text                 # crossing is not sold as the answer
    assert "EVERY playbook" in text                              # the every-playbook guidance survives
    assert not re.search(r"[+-]\d+\.\d+R", text)                 # no historical R figure in the prompt
    note = _joined_literals('"ENTRY-QUALITY LEARNING', '"TARGET REACHABILITY')
    assert "fillAdjustedR" in note                                # deeper rests are checked against the map

  def test_the_edge_state_builds_the_map_from_the_fill_rate_records_and_the_report_carries_it(self):
    src = inspect.getsource(_agent_mod.run_trading_agent)
    build = src[src.index('state["execution_map"] = execution_map('):][:400]
    assert "memory.limit_entry_records()" in build and "memory.realized_closes(limit=MAX_CLOSED_TRADES)" in build
    assert "probes=_probes" in build and "rr_floor=cfg.trading.min_futures_rr" in build
    assert '"executionMap": _edge_now.get("execution_map"' in src
    # Its own try: a map failure is a WARNING, never the loss of the signal edge beside it.
    assert 'logger.warning("EXECUTION MAP unavailable this run' in src


# ── Decision feed: a failure is never printed as a live order (2026-09-25) ──────────────────────────

from src.agent import (  # noqa: E402
  _exit_discipline_for_prompt,
  _market_state_for_prompt,
  _summarize_tool_output,
)


class TestDecisionFeedLabels:
  def test_a_lease_rejection_carrying_an_order_id_prints_as_rejected(self):
    line = _summarize_tool_output({"rejected": True, "reason": "entry lease active", "orderId": "491868",
                                   "symbol": "WIF-USDT"})
    assert line.startswith("rejected: entry lease active") and "live order" not in line

  def test_an_exchange_error_carrying_an_order_id_prints_as_error(self):
    line = _summarize_tool_output({"error": "order cannot be canceled", "orderId": "123", "symbol": "FOLKS-USDT"})
    assert line.startswith("error: order cannot be canceled") and "live order" not in line

  def test_an_empty_cancel_response_is_neutral_not_live_and_not_failed(self):
    # KuCoin can return null data on a SUCCESSFUL cancel, so an empty response is not a failure.
    for empty in (None, {}, False):
      line = _summarize_tool_output({"cancelled": empty, "orderId": "777", "symbol": "WIF-USDT"})
      assert line == "cancel sent: 777 (exchange returned no confirmation)"
    assert _summarize_tool_output({"cancelled": {"cancelledOrderIds": ["777"]}, "orderId": "777"}) == \
      "cancelled order: 777"
    assert _summarize_tool_output({"paper": True, "cancelled": {"orderId": "p1"}}) == "cancelled order: p1"

  def test_a_real_order_still_prints_as_live(self):
    line = _summarize_tool_output({"orderId": "999", "side": "sell", "symbol": "SPX-USDT"})
    assert line.startswith("live order: sell SPX-USDT (orderId=999)")

  def test_the_run_uses_the_module_helper(self):
    src = inspect.getsource(_agent_mod.run_trading_agent)
    assert "summary = _summarize_tool_output(out)" in src
    assert "def _summarize(" not in src


# ── Market state: the model sees the current reading as plain numbers only (2026-09-25) ─────────────


class TestMarketStateForTheModel:
  STATE = {"asOf": 1_790_300_000, "universe": 68, "breadth24": 0.07, "basketMedian24h": -6.2,
           "btc24h": -1.4, "btc72h": 2.1, "btcDailyAdx": 18.5, "btcDailyBias": "bullish"}

  def test_plain_numbers_and_a_legend_with_no_record_and_no_rule(self):
    block = _market_state_for_prompt(self.STATE, 1_790_300_000 + 1800)
    assert {k: block[k] for k in ("breadth24", "universe", "basketMedian24h", "btc24h", "btc72h",
                                  "btcDailyAdx", "btcDailyBias")} == {k: v for k, v in self.STATE.items()
                                                                      if k != "asOf"}
    assert block["ageMin"] == 30
    note = block["note"].lower()
    # No per-bucket R line, no win rate, no sizing — the reading only.
    for banned in ("r/trade", "win", "size", "stake", "expectancy", "+0.", "-0."):
      assert banned not in note
    assert set(block) - {"note", "ageMin"} <= set(self.STATE)

  def test_nothing_to_show_is_none(self):
    assert _market_state_for_prompt(None, 0) is None
    assert _market_state_for_prompt({"asOf": 5}, 10) is None

  def test_exit_discipline_reaches_the_model_without_the_market_state_split(self):
    stats = {"verdict": "neutral", "trailByRegime": {"a": 1}, "trailByMarketState": {"buckets": {}}}
    shown = _exit_discipline_for_prompt(stats)
    assert "trailByMarketState" not in shown and shown["trailByRegime"] == {"a": 1}
    assert "trailByMarketState" in stats          # the input is not mutated

  def test_the_run_wires_the_reader_into_the_tools_and_the_payload(self):
    src = inspect.getsource(_agent_mod.run_trading_agent)
    assert inspect.signature(_agent_mod.run_trading_agent).parameters["market_state"].default is None
    assert "market_state=market_state,\n  ))" in src           # build_tools ctx (entry + probe stamps)
    shown = src.index('user_state_obj["marketState"] = _ms_block')
    assert src.index("user_state_obj = json.loads(user_state)") < shown < src.index(
      "input_payload = json.dumps(user_state_obj)")
    assert '"exitDiscipline": _exit_discipline_for_prompt(' in src
    # The signal-edge copy the model gets never carries the per-state split either.
    edge = inspect.getsource(_agent_mod)
    assert "market_state_split=True" not in edge

  def test_the_research_prompt_points_at_the_perp_book_and_the_exclusions(self):
    src = inspect.getsource(_agent_mod.run_trading_agent)
    text = src[src.index("## COIN-LIST CURATION"):src.index("- Do NOT stay anchored")]
    assert "fetch_futures_orderbook" in text and "excluded" in text
    assert "lastAnalysisFailure" in text and "quarantined" in text
