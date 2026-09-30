"""Jev dual run (src/jev.py): a second trader on the same order path, judged on its own record.

Covers the contract end to end on fakes (no network): config, the per-trader evidence split in memory, the
code-built bracket, the System One wire format through the official SDK on a mock transport, the pass's
live/shadow routing, ownership on the real order path, and the report's disclosure policy.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from src import jev
from src.memory import MemoryStore
from src.regime import net_reward_risk_ratio
from tests.test_tools import _invoke_tool, _limit_entry_tools, _place


# ── fixtures ─────────────────────────────────────────────────────────────────────────────────────────
def _analysis(sym="SOL-USDT", *, price=100.0, atr=0.5, vwap=99.0, bb_mid=99.5, h1_lower=98.0, h1_upper=102.0,
              funding_setup=None, ok=True):
  snaps = [
    {"interval": "15min", "close": price, "ema_fast": 100.1, "ema_slow": 99.8, "rsi": 58.0, "adx": 24.0,
     "macd_hist": 0.01, "atr": atr, "atr_pct": atr / price * 100, "bb_upper": 101.0, "bb_lower": 99.0,
     "bb_mid": bb_mid, "bbw": 2.0, "market_regime": {"regime": "trending"}, "trend_bias": "bullish"},
    {"interval": "1hour", "close": price, "bb_upper": h1_upper, "bb_lower": h1_lower, "rsi": 55.0,
     "market_regime": {"regime": "trending"}, "trend_bias": "bullish"},
    {"interval": "4hour", "close": price, "trend_bias": "neutral"},
    {"interval": "1day", "close": price, "trend_bias": "bearish"},
  ]
  return {
    "symbol": sym, "snapshots": snaps, "dataQuality": {"ok": ok, "errors": {}},
    "summary": {
      "overall_bias": "bullish", "strength": "moderate", "weighted_score": 0.31, "daily_bias": "bearish",
      "daily_exhausted": False, "timeframe_conflict": True, "market_regime": "trending",
      "entryMap": {"price": price, "vwap15m": vwap, "bbMid15m": bb_mid, "atr15m": atr, "rsi15m": 58.0,
                   "extensionAtrLong": 2.0, "extensionAtrShort": -2.0, "fadeSetup": None},
      "entryGate": {"minConfidence": 0.75},
    },
    "futures": {"markPrice": price, "fundingRate": 0.0001, "fundingIntervalHours": 8.0,
                "fundingSetup": funding_setup, "oiTrend": "up", "basisPct": 0.01, "priceChgPct24h": 1.2},
  }


def _cfg(mode="shadow", **jev_over):
  jcfg = SimpleNamespace(mode=mode, model="jev-latest", max_open_positions=1, max_entries_per_day=6,
                         max_symbols_per_run=8, timeout_sec=5.0, risk_scale=1.0)
  for k, v in jev_over.items():
    setattr(jcfg, k, v)
  return SimpleNamespace(
    jev=jcfg,
    trading=SimpleNamespace(min_futures_rr=1.5, max_entry_leverage=3.0, estimated_slippage_pct=0.001, stop_atr_floor_mult=2.5,
                            slippage_autotune_min_samples=8),
  )


def _choice(label, probs):
  return SimpleNamespace(choice=label, confidence=probs.get(label), probabilities=probs)


class _Client:
  """System One stand-in: one scripted direction per symbol."""

  def __init__(self, directions, raise_for=()):
    self.directions, self.raise_for, self.calls, self.closed = directions, set(raise_for), [], False

  async def aclose(self):
    self.closed = True

  async def system_one(self, state, questions):
    self.calls.append((state, questions))
    sym = state["symbol"]
    if sym in self.raise_for:
      raise RuntimeError("boom")
    label, p = self.directions.get(sym, ("stand_aside", 0.8))
    rest = (1.0 - p) / 2
    probs = {k: (p if k == label else rest) for k in ("long", "short", "stand_aside")}
    return SimpleNamespace(
      model="jev-1.4.2", usage=SimpleNamespace(input_tokens=900, output_tokens=0),
      choices={
        "direction": _choice(label, probs),
        "setup_family": _choice("continuation", {"continuation": 0.6, "breakout": 0.4}),
        "entry": _choice("at_market", {"at_market": 0.7, "pullback": 0.3}),
        "target": _choice("near", {"near": 0.6, "extended": 0.4}),
      },
    )


class _Tools:
  def __init__(self, analyses, *, owners=None, held=(), results=None):
    self.analyses, self.owners, self.held = analyses, owners or {}, list(held)
    self.results, self.orders, self.analyzed = results or {}, [], []

  def latest_analyses(self, max_age_sec=600.0):
    return dict(self.analyses)

  def trader_book(self, name):
    return list(self.held) if name == "jev" else []

  async def analyze_for(self, symbol):
    self.analyzed.append(symbol)
    return _analysis(symbol)

  def owner_of(self, symbol):
    return self.owners.get(symbol)

  async def place_futures_limit_order_for(self, trader, dry_run=False, **kwargs):
    self.orders.append({"trader": trader, "dry_run": dry_run, **kwargs})
    if kwargs["symbol"] in self.results:
      return self.results[kwargs["symbol"]]
    return {"shadow": True} if dry_run else {"paper": True, "pendingLimitEntry": True}


def _run(cfg, tools, memory, client, **kw):
  return asyncio.run(jev.run_jev_pass(
    cfg, tools, memory, universe=kw.pop("universe", ["BTC-USDT", "ETH-USDT", "SOL-USDT"]), equity_usd=75.0,
    noise_mult=2.5, cost_rate=0.0016, client=client, now=kw.pop("now", 1_790_000_000.0), **kw,
  ))


# ── config ───────────────────────────────────────────────────────────────────────────────────────────
class TestConfig:
  def test_defaults_are_off_and_the_loader_agrees(self, monkeypatch):
    from src.config import JevConfig, load_config
    for key in ("JEV_MODE", "JEV_MODEL", "JEV_MAX_OPEN_POSITIONS", "JEV_RISK_SCALE"):
      monkeypatch.delenv(key, raising=False)
    cfg = load_config()
    d = JevConfig()
    assert cfg.jev.mode == d.mode == "off"
    assert (cfg.jev.model, cfg.jev.max_open_positions, cfg.jev.max_entries_per_day,
            cfg.jev.max_symbols_per_run, cfg.jev.timeout_sec, cfg.jev.risk_scale) == \
           (d.model, d.max_open_positions, d.max_entries_per_day, d.max_symbols_per_run, d.timeout_sec, d.risk_scale)

  def test_env_overrides(self, monkeypatch):
    from src.config import load_config
    monkeypatch.setenv("JEV_MODE", " Live ")
    monkeypatch.setenv("JEV_MODEL", "jev-1.13.0")
    monkeypatch.setenv("JEV_RISK_SCALE", "0.5")
    cfg = load_config()
    assert (cfg.jev.mode, cfg.jev.model, cfg.jev.risk_scale) == ("live", "jev-1.13.0", 0.5)

  @pytest.mark.parametrize("key,value", [("JEV_MODE", "yolo"), ("JEV_RISK_SCALE", "1.5"),
                                         ("JEV_RISK_SCALE", "0"), ("JEV_MAX_SYMBOLS_PER_RUN", "0"),
                                         ("JEV_TIMEOUT_SEC", "0")])
  def test_invalid_values_refuse_to_start(self, monkeypatch, key, value):
    from src.config import load_config
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
      load_config()


# ── memory: each trader's evidence is its own ────────────────────────────────────────────────────────
class TestMemorySeparation:
  def test_probes_split_by_trader_and_llm_is_the_default(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_signal_probe("SOL-USDT", "buy", 100.0, "continuation")
    m.record_signal_probe("ETH-USDT", "sell", 2000.0, "continuation", trader="jev")
    assert [p["symbol"] for p in m.signal_probes(limit=0)] == ["SOL-USDT"]
    assert [p["symbol"] for p in m.signal_probes(limit=0, trader="jev")] == ["ETH-USDT"]
    assert len(m.signal_probes(limit=0, trader="all")) == 2
    assert m.signal_probes(limit=0, trader="jev")[0]["entryContext"]["trader"] == "jev"
    assert "trader" not in m.signal_probes(limit=0)[0]["entryContext"]   # LLM rows stay unstamped

  def test_a_jev_call_is_not_a_repeat_of_the_llms(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    assert m.record_signal_probe("SOL-USDT", "buy", 100.0, "continuation", min_gap_sec=3600) is not False
    assert m.record_signal_probe("SOL-USDT", "buy", 100.0, "continuation", min_gap_sec=3600, trader="jev") is not False
    assert m.record_signal_probe("SOL-USDT", "buy", 100.0, "continuation", min_gap_sec=3600, trader="jev") is False

  def test_retention_is_per_trader(self, tmp_path):
    from src.memory import MAX_PROBES_PER_FAMILY
    m = MemoryStore(str(tmp_path / "m.json"))
    for i in range(3):
      m.record_signal_probe(f"L{i}-USDT", "buy", 1.0, "continuation")
    for i in range(MAX_PROBES_PER_FAMILY + 5):
      m.record_signal_probe(f"J{i}-USDT", "buy", 1.0, "continuation", trader="jev")
    assert len(m.signal_probes(limit=0)) == 3          # Jev's flood evicted none of the LLM's rows
    assert len(m.signal_probes(limit=0, trader="jev")) == MAX_PROBES_PER_FAMILY

  def test_gate_scoreboard_reads_the_llm_by_default(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_gate_probe("SOL-USDT", "buy", 100.0, "confidence_floor", setup_family="continuation")
    m.record_gate_probe("ETH-USDT", "buy", 100.0, "confidence_floor", setup_family="continuation", trader="jev")
    assert [r["symbol"] for r in m.gate_probes()] == ["SOL-USDT"]
    assert len(m.gate_probes(trader="all")) == 2

  def test_jev_decisions_are_capped_and_plain_json(self, tmp_path):
    from src.memory import MAX_JEV_DECISIONS
    m = MemoryStore(str(tmp_path / "m.json"))
    for i in range(MAX_JEV_DECISIONS + 3):
      m.record_jev_decision({"symbol": "SOL-USDT", "i": i, "odd": {1, 2} if i == 0 else None})
    rows = m.jev_decisions(limit=0)
    assert len(rows) == MAX_JEV_DECISIONS and rows[-1]["i"] == MAX_JEV_DECISIONS + 2
    assert len(m.jev_decisions(limit=5)) == 5

  def test_owner_lookups(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_trade("SOL-USDT", "sell", 20.0, paper=True, price=100.0, size=0.2, venue="futures", filled=True,
                   track_position=False, client_oid="c-jev",
                   entry_context={"trader": "jev", "positionSide": "short"})
    m.record_trade("ETH-USDT", "buy", 20.0, paper=True, price=2000.0, size=0.01, venue="futures", filled=True,
                   track_position=False, client_oid="c-llm", entry_context={"positionSide": "long"})
    assert m.trader_for_order(None, "c-jev") == "jev"
    assert m.trader_for_order(None, "c-llm") == "llm"
    assert m.trader_for_order(None, "nope") is None
    now_ms = int(time.time() * 1000)
    assert m.trader_for_position("SOL-USDT", now_ms, "short") == "jev"
    assert m.trader_for_position("ETH-USDT", now_ms, "long") == "llm"


# ── state + bracket ──────────────────────────────────────────────────────────────────────────────────
class TestState:
  def test_market_facts_only(self):
    state = jev.jev_state("SOL-USDT", _analysis())
    text = json.dumps(state).lower()
    for banned in ("equity", "balance", "notional", "stake", "verdict", "minconfidence", "entrygate"):
      assert banned not in text
    assert set(state["timeframes"]) == {"15min", "1hour", "4hour", "1day"}
    assert state["timeframes"]["15min"]["bollingerPctB"] == pytest.approx(0.5)

  def test_unusable_analysis_is_none(self):
    assert jev.jev_state("X", {"error": "No candles"}) is None
    assert jev.jev_state("X", _analysis(ok=False)) is None
    assert jev.jev_state("X", _analysis(atr=0.0)) is None

  def test_funding_carry_is_offered_only_when_it_exists(self):
    plain = jev.jev_questions(jev.jev_state("S", _analysis()), near_r=1.65, extended_r=3.3)
    assert "funding_carry" not in plain["setup_family"]["criteria"]
    carry = jev.jev_state("S", _analysis(funding_setup={"side": "sell", "reason": "funding +0.2%"}))
    assert carry["futures"]["fundingCarry"]["paidSide"] == "short"
    assert "funding_carry" in jev.jev_questions(carry, near_r=1.65, extended_r=3.3)["setup_family"]["criteria"]

  def test_long_and_short_are_worded_as_mirrors(self):
    crit = jev.jev_questions(jev.jev_state("S", _analysis()), near_r=1.65, extended_r=3.3)["direction"]["criteria"]
    swapped = crit["long"].replace("LONG", "X").replace("rise", "Y").replace("fall", "rise").replace("Y", "fall")
    assert swapped.replace("X", "SHORT") == crit["short"]


class TestBracket:
  KW = {"noise_mult": 2.5, "rr_floor": 1.5, "cost_rate": 0.0016}

  def test_long_structural_stop_and_near_target_net_r(self):
    b = jev.build_bracket(_analysis(), "long", **self.KW)
    assert b["entry"] == 100.0 and b["stop"] == pytest.approx(98.0)       # 1h BB lower, wider than 1.25 floor
    net = net_reward_risk_ratio("buy", b["entry"], b["takeProfit"], b["stop"], fee_rate=0.0016)
    assert net == pytest.approx(1.5 * jev.NEAR_TARGET_CUSHION, rel=1e-6)
    assert b["stopAtr"] == pytest.approx(4.0)

  def test_short_mirrors(self):
    b = jev.build_bracket(_analysis(), "short", target_kind="extended", **self.KW)
    assert b["stop"] == pytest.approx(102.0) and b["takeProfit"] < b["entry"]
    net = net_reward_risk_ratio("sell", b["entry"], b["takeProfit"], b["stop"], fee_rate=0.0016)
    assert net == pytest.approx(1.5 * jev.NEAR_TARGET_CUSHION * jev.EXTENDED_TARGET_MULT, rel=1e-6)

  def test_stop_never_inside_the_noise_floor_and_never_past_twice_it(self):
    tight = jev.build_bracket(_analysis(h1_lower=99.9), "long", **self.KW)
    assert tight["stop"] == pytest.approx(100.0 - 2.5 * 0.5)
    far = jev.build_bracket(_analysis(h1_lower=80.0), "long", **self.KW)
    assert far["stop"] == pytest.approx(100.0 - 2 * 2.5 * 0.5)

  def test_pullback_uses_the_nearest_anchor_on_the_entry_side_else_market(self):
    b = jev.build_bracket(_analysis(), "long", entry_kind="pullback", **self.KW)
    assert (b["entry"], b["entryKind"]) == (99.5, "pullback")
    s = jev.build_bracket(_analysis(), "short", entry_kind="pullback", **self.KW)   # anchors are below price
    assert (s["entry"], s["entryKind"]) == (100.0, "at_market")

  def test_no_bracket_without_levels(self):
    assert jev.build_bracket(_analysis(atr=0.0), "long", **self.KW) is None
    assert jev.build_bracket(_analysis(), "flat", **self.KW) is None


# ── the wire contract, through the official SDK on a mock transport ───────────────────────────────────
class TestWireFormat:
  def test_request_and_response_round_trip(self):
    httpx2 = pytest.importorskip("httpx2")
    sdk = pytest.importorskip("typesafe_sdk")
    seen = {}

    def handler(request):
      seen["url"] = str(request.url)
      seen["auth"] = request.headers.get("authorization")
      seen["body"] = json.loads(request.content)
      return httpx2.Response(200, json={
        "model": "jev-1.4.2", "usage": {"input_tokens": 812, "output_tokens": 0},
        "answers": {
          "direction": {"type": "choice", "choice": "short", "confidence": 0.7,
                        "probabilities": {"long": 0.1, "short": 0.7, "stand_aside": 0.2}},
          "setup_family": {"type": "choice", "choice": "fade_extreme", "confidence": 0.5,
                           "probabilities": {"continuation": 0.2, "fade_extreme": 0.5, "breakout": 0.2, "range_edge": 0.1}},
          "entry": {"type": "choice", "choice": "pullback", "confidence": 0.6, "probabilities": {"at_market": 0.4, "pullback": 0.6}},
          "target": {"type": "choice", "choice": "extended", "confidence": 0.55, "probabilities": {"near": 0.45, "extended": 0.55}},
        },
      })

    state = jev.jev_state("SOL-USDT", _analysis())
    questions = jev.jev_questions(state, near_r=1.65, extended_r=3.3)

    async def go():
      async with sdk.AsyncTypeSafeClient(api_key="test-key", model="jev-latest",
                                         transport=httpx2.MockTransport(handler)) as client:
        return await client.system_one(state=state, questions=questions)

    resp = asyncio.run(go())
    assert seen["url"].endswith("/v1/systemone")
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"]["model"] == "jev-latest" and seen["body"]["state"]["symbol"] == "SOL-USDT"
    assert seen["body"]["questions"]["direction"]["type"] == "choice"
    assert set(seen["body"]["questions"]["direction"]["criteria"]) == {"long", "short", "stand_aside"}
    parsed = jev.parse_answers(resp, families=list(questions["setup_family"]["criteria"]))
    assert parsed["direction"] == "short" and parsed["confidence"] == pytest.approx(0.7)
    assert (parsed["setupFamily"], parsed["entryKind"], parsed["targetKind"]) == ("fade_extreme", "pullback", "extended")
    assert parsed["model"] == "jev-1.4.2" and parsed["inputTokens"] == 812

  def test_a_label_we_did_not_offer_falls_back_to_the_most_probable_offered_one(self):
    resp = SimpleNamespace(model="m", usage=None, choices={
      "direction": _choice("long", {"long": 0.6, "short": 0.2, "stand_aside": 0.2}),
      "setup_family": _choice("funding_carry", {"funding_carry": 0.7, "breakout": 0.2, "continuation": 0.1}),
    })
    parsed = jev.parse_answers(resp, families=["continuation", "breakout"])
    assert parsed["setupFamily"] == "breakout"
    assert (parsed["entryKind"], parsed["targetKind"]) == ("at_market", "near")


# ── the pass ─────────────────────────────────────────────────────────────────────────────────────────
class TestPass:
  def test_off_does_nothing(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    tools = _Tools({"SOL-USDT": _analysis()})
    out = _run(_cfg("off"), tools, m, _Client({}))
    assert out["asked"] == 0 and not tools.orders and not m.jev_decisions(limit=0)

  def test_no_key_idles_without_raising(self, tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    m = MemoryStore(str(tmp_path / "m.json"))
    out = asyncio.run(jev.run_jev_pass(_cfg("live"), _Tools({}), m, universe=[], equity_usd=75.0,
                                       noise_mult=2.5, cost_rate=0.0016))
    assert out["skipped"] == "TYPESAFE_API_KEY not set"

  def test_shadow_runs_every_call_through_the_gates_without_an_order(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    tools = _Tools({"SOL-USDT": _analysis("SOL-USDT")})
    client = _Client({"SOL-USDT": ("long", 0.7), "BTC-USDT": ("short", 0.66), "ETH-USDT": ("stand_aside", 0.9)})
    out = _run(_cfg("shadow"), tools, m, client)
    assert out["asked"] == 3 and sorted(tools.analyzed) == ["BTC-USDT", "ETH-USDT"]   # cached SOL reused
    assert all(o["dry_run"] for o in tools.orders) and len(tools.orders) == 2
    sol = next(o for o in tools.orders if o["symbol"] == "SOL-USDT")
    assert sol["side"] == "buy" and sol["confidence"] == pytest.approx(0.7) and sol["trader"]["name"] == "jev"
    assert sol["trader"]["model"] == "jev-1.4.2"                      # the resolved version, not the alias
    assert sol["stop_loss_price"] < sol["entry_price"] < sol["take_profit_price"]
    rows = {r["symbol"]: r for r in m.jev_decisions(limit=0)}
    assert rows["ETH-USDT"]["outcome"] == "stand_aside"
    assert rows["SOL-USDT"]["outcome"] == "shadow" and rows["SOL-USDT"]["mode"] == "shadow"

  def test_live_gives_the_slot_to_the_most_confident_call_and_shadows_the_rest(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    tools = _Tools({}, owners={"ETH-USDT": "llm"})
    client = _Client({"SOL-USDT": ("long", 0.66), "BTC-USDT": ("short", 0.81), "ETH-USDT": ("long", 0.9)})
    _run(_cfg("live", max_open_positions=1), tools, m, client)
    by = {o["symbol"]: o for o in tools.orders}
    assert by["ETH-USDT"]["dry_run"] is True                          # the LLM holds it: shadow, still recorded
    assert by["BTC-USDT"]["dry_run"] is False                         # strongest free call takes the slot
    assert by["SOL-USDT"]["dry_run"] is True                          # slot used
    rows = {r["symbol"]: r for r in m.jev_decisions(limit=0)}
    assert rows["BTC-USDT"]["outcome"] == "placed" and rows["ETH-USDT"]["heldBy"] == "llm"

  def test_a_refused_live_entry_keeps_the_slot(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    tools = _Tools({}, results={"BTC-USDT": {"rejected": True, "gate": "confidence_floor", "reason": "x"}})
    client = _Client({"SOL-USDT": ("long", 0.7), "BTC-USDT": ("short", 0.8)})
    _run(_cfg("live"), tools, m, client)
    by = {o["symbol"]: o for o in tools.orders}
    assert by["BTC-USDT"]["dry_run"] is False and by["SOL-USDT"]["dry_run"] is False
    rows = {r["symbol"]: r for r in m.jev_decisions(limit=0)}
    assert (rows["BTC-USDT"]["outcome"], rows["BTC-USDT"]["detail"]) == ("refused", "confidence_floor")

  def test_caps_already_used_turn_live_into_shadow(self, tmp_path):
    now = 1_790_000_000.0
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_jev_decision({"symbol": "X-USDT", "outcome": "placed", "ts": int(now) - 60})
    tools = _Tools({})
    _run(_cfg("live", max_entries_per_day=1), tools, m, _Client({"SOL-USDT": ("long", 0.9)}), now=now)
    assert [o["dry_run"] for o in tools.orders] == [True]
    tools2 = _Tools({}, held=["BTC-USDT"])
    _run(_cfg("live", max_open_positions=1), tools2, m, _Client({"SOL-USDT": ("long", 0.9)}))
    assert [o["dry_run"] for o in tools2.orders] == [True]
    assert all(o["symbol"] != "BTC-USDT" for o in tools2.orders)      # Jev's own lifecycle is not re-called

  def test_an_api_error_is_a_row_not_an_exception(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    tools = _Tools({})
    out = _run(_cfg("shadow"), tools, m, _Client({"SOL-USDT": ("long", 0.7)}, raise_for={"BTC-USDT"}))
    rows = {r["symbol"]: r for r in out["rows"]}
    assert rows["BTC-USDT"]["outcome"] == "error" and "RuntimeError" in rows["BTC-USDT"]["detail"]
    assert rows["SOL-USDT"]["outcome"] == "shadow"
    assert "errors 1" in jev.describe_pass(out)


class TestAgentHook:
  """agent._run_jev_dual_pass: after the LLM's run, in its own event loop, never raising into the loop."""

  def _call(self, cfg, tools, memory, authorized=True):
    from src.agent import _run_jev_dual_pass
    snapshot = SimpleNamespace(total_usdt=75.0)
    return _run_jev_dual_pass(cfg, tools, memory, snapshot, {"SOL-USDT"}, {"futures_taker": 0.0006},
                              lambda: {"slippage_pct": 0.0005, "stop_atr_floor_mult": 2.5}, authorized=authorized)

  def test_runs_with_its_own_client_and_closes_it(self, tmp_path, monkeypatch):
    client = _Client({"SOL-USDT": ("long", 0.7)})
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(jev, "_build_client", lambda cfg: client)
    m = MemoryStore(str(tmp_path / "m.json"))
    tools = _Tools({})
    out = self._call(_cfg("shadow"), tools, m)
    assert out["asked"] == 1 and out["calls"] == 1 and "rows" not in out
    assert client.closed and tools.orders[0]["dry_run"] is True

  def test_off_or_unauthorized_is_skipped(self, tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "_build_client", lambda cfg: pytest.fail("must not build a client"))
    m = MemoryStore(str(tmp_path / "m.json"))
    assert self._call(_cfg("off"), _Tools({}), m) is None
    assert self._call(_cfg("live"), _Tools({}), m, authorized=False) is None

  def test_a_broken_pass_never_raises(self, tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setattr(jev, "_build_client", lambda cfg: _Client({}))

    class _Broken(_Tools):
      def latest_analyses(self, max_age_sec=600.0):
        raise RuntimeError("cache gone")

    out = self._call(_cfg("shadow"), _Broken({}), MemoryStore(str(tmp_path / "m.json")))
    assert out is not None and "RuntimeError" in out["error"]


# ── ownership + trader identity on the REAL order path (paper mode, fakes) ────────────────────────────
def _jev_place(tools, dry_run=False, trader=None, **over):
  args = {"symbol": "SPX-USDT", "side": "sell", "notional_usd": 20.0, "entry_price": 1.01,
          "confidence": 0.8, "take_profit_price": 0.95, "stop_loss_price": 1.03, "setup_family": "continuation"}
  args.update(over)
  return asyncio.run(tools.place_futures_limit_order_for(
    trader or {"name": "jev", "model": "jev-1.4.2", "sizeScale": 1.0}, dry_run=dry_run, **args))


def _held_position(memory, snapshot, trader):
  ctx = {"positionSide": "short"}
  if trader:
    ctx["trader"] = trader
  memory.record_trade("SPX-USDT", "sell", 20.0, paper=True, price=1.0, size=20, venue="futures", filled=True,
                      track_position=False, client_oid="held-1", entry_context=ctx)
  snapshot.futures_positions = [{"symbol": "SPXUSDTM", "currentQty": -20, "openingTimestamp": int(time.time() * 1000)}]


class TestOrderPath:
  def test_a_jev_entry_is_stamped_and_scored_on_its_own_record(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path)
    out = _jev_place(tools)
    assert out.get("paper") is True, out
    ctx = out["tradeRecord"]["entryContext"]
    assert (ctx["trader"], ctx["model"]) == ("jev", "jev-1.4.2")
    assert "build" not in ctx or not ctx["build"] or "prompt" not in ctx["build"]
    assert memory.signal_probes(limit=0) == []
    assert len(memory.signal_probes(limit=0, trader="jev")) == 1
    assert tools.owner_of("SPX-USDT") == "jev" and tools.trader_book("jev") == ["SPX-USDT"]

  def test_size_scale_only_shrinks(self, tmp_path):
    full, _ = _limit_entry_tools(tmp_path / "a")
    half, _ = _limit_entry_tools(tmp_path / "b")
    a = _jev_place(full)
    b = _jev_place(half, trader={"name": "jev", "model": "m", "sizeScale": 0.5})
    assert 0 < b["contracts"] < a["contracts"]

  def test_dry_run_records_the_call_and_places_nothing(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path)
    out = _jev_place(tools, dry_run=True)
    assert out["shadow"] is True and out["probeRecorded"] is True, out
    assert len(memory.signal_probes(limit=0, trader="jev")) == 1
    assert tools.trader_book("jev") == [] and tools.owner_of("SPX-USDT") is None
    assert not (memory._read().get("trades") or [])

  def test_the_llm_cannot_touch_a_jev_position(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path)
    ctx_snapshot = _snapshot_of(tools)
    _held_position(memory, ctx_snapshot, "jev")
    out = _place(tools)
    assert (out.get("gate"), out.get("owner")) == ("trader_conflict", "jev"), out
    assert memory.gate_probes(trader="all") == []                    # structural: records nothing
    for tool, args in (
      (tools.place_futures_market_order, {"symbol": "SPX-USDT", "side": "buy", "notional_usd": 10.0, "reduce_only": True}),
      (tools.place_futures_stop_order, {"symbol": "SPX-USDT", "side": "buy", "leverage": 1.0, "stop_price": 1.1}),
      (tools.set_futures_position_protection, {"symbol": "SPX-USDT", "stop_loss_price": 1.2}),
    ):
      res = _invoke_tool(tool, **args)
      assert isinstance(res, dict) and res.get("gate") == "trader_conflict", (tool.name, res)
    res = _invoke_tool(tools.cancel_futures_order, order_id="held-1")
    assert res.get("gate") == "trader_conflict", res
    # A bracket leg is not a trade record: found by id among the live stop orders, judged by its symbol.
    _snapshot_of(tools).futures_stop_orders = [{"id": "sl-leg-9", "symbol": "SPXUSDTM", "stop": "up"}]
    res = _invoke_tool(tools.cancel_futures_order, order_id="sl-leg-9")
    assert res.get("gate") == "trader_conflict", res

  def test_jev_cannot_touch_an_llm_position_but_can_still_shadow_the_call(self, tmp_path):
    tools, memory = _limit_entry_tools(tmp_path)
    _held_position(memory, _snapshot_of(tools), None)
    out = _jev_place(tools)
    assert (out.get("gate"), out.get("owner")) == ("trader_conflict", "llm"), out
    shadow = _jev_place(tools, dry_run=True)
    assert shadow.get("shadow") is True, shadow


def _snapshot_of(tools):
  """The snapshot object build_tools closed over (every tool shares it)."""
  fn = tools.trader_book
  for cell in fn.__closure__ or ():
    val = cell.cell_contents
    if hasattr(val, "futures_positions") and hasattr(val, "paper_trading"):
      return val
  raise AssertionError("snapshot not found in closure")


# ── the LLM's view + the report ──────────────────────────────────────────────────────────────────────
class TestReportingAndView:
  def test_only_other_traders_rows_are_marked_for_the_llm(self):
    from src.agent import _attach_dual_run_owners
    state = {"futuresPositions": [{"symbol": "SOLUSDTM"}, {"symbol": "ETHUSDTM"}],
             "pendingLimitOrders": {"futures": [{"symbol": "SOLUSDTM"}]}, "stops": {"futures": [{"symbol": "BTCUSDTM"}]}}
    owners = {"SOLUSDTM": "jev", "ETHUSDTM": "llm"}
    assert _attach_dual_run_owners(state, owners.get) == 2
    assert state["futuresPositions"][0]["managedBy"] == "jev"
    assert "managedBy" not in state["futuresPositions"][1] and "managedBy" not in state["stops"]["futures"][0]

  def test_scrub_detail_keeps_gate_codes_and_drops_money(self):
    assert jev.scrub_detail("confidence_floor") == "confidence_floor"
    out = jev.scrub_detail("Below contract minimum notional 12.34 with equity $75.10")
    assert "12.34" not in out and "75.10" not in out
    assert jev.scrub_detail(None) is None

  def test_report_splits_traders_over_the_same_window_without_money(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    start = int(time.time()) - 600
    m.record_jev_decision({"symbol": "SOL-USDT", "direction": "long", "confidence": 0.7, "outcome": "placed",
                           "probabilities": {"long": 0.7, "short": 0.1, "stand_aside": 0.2}, "latencyMs": 180,
                           "detail": "notional 12.5 USDT", "ts": start})
    m.record_jev_decision({"symbol": "BTC-USDT", "direction": "stand_aside", "outcome": "stand_aside", "ts": start + 1})
    m.record_signal_probe("SOL-USDT", "buy", 100.0, "continuation")
    m.record_signal_probe("SOL-USDT", "buy", 100.0, "continuation", trader="jev")
    rep = jev.dual_run_report(m, _cfg("live"), cost_pct=0.0016)
    assert rep["mode"] == "live" and rep["since"] == start
    assert rep["traders"]["llm"]["verdict"] and rep["traders"]["jev"]["verdict"]
    assert rep["outcomes"] == {"placed": 1, "stand_aside": 1} and rep["medianLatencyMs"] == 180
    assert rep["agreement"] == {"agree": 1, "disagree": 0}
    assert rep["recent"][-1]["detail"] and "12.5" not in rep["recent"][-1]["detail"]
    assert "equity" not in json.dumps(rep).lower()

  def test_report_is_just_the_mode_when_it_never_ran(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    assert jev.dual_run_report(m, _cfg("off"), cost_pct=0.0016) == {"mode": "off"}

  def test_dashboard_tags_only_jev_rows(self):
    from src.dashboard_publisher import DashboardPublisher
    assert DashboardPublisher._row_trader({"trader": "jev"}) == "jev"
    assert DashboardPublisher._row_trader({"entryContext": {"trader": "jev"}}) == "jev"
    assert DashboardPublisher._row_trader({"trader": "llm"}) is None
    assert DashboardPublisher._row_trader({}) is None


# ── observability: logs, status, LangSmith, the enriched report (Sep 30 pm) ──────────────────────────
class _LsClient:
  """LangSmith stand-in: records every run RunTree posts and every feedback."""

  def __init__(self):
    self.runs, self.feedback = [], []

  def create_run(self, **kw):
    self.runs.append(kw)

  def create_feedback(self, run_id, key, score=None, comment=None, **kw):
    self.feedback.append((str(run_id), key, score))


def _ls_cfg(mode="shadow", rate=0.1):
  cfg = _cfg(mode)
  cfg.langsmith = SimpleNamespace(enabled=True, tracing=True, api_key="k", api_url=None, project="p", sample_rate=rate)
  return cfg


class TestObservability:
  def test_startup_line_says_what_will_happen(self, monkeypatch):
    assert "off" in jev.startup_line(_cfg("off")) and "JEV_MODE=shadow" in jev.startup_line(_cfg("off"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    line = jev.startup_line(_cfg("shadow"))
    assert "mode=shadow" in line and "key=MISSING" in line
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert "key=set" in jev.startup_line(_ls_cfg("live")) and "langsmith=on" in jev.startup_line(_ls_cfg("live"))

  def test_one_line_per_symbol(self):
    row = {"symbol": "SOL-USDT", "direction": "long", "confidence": 0.71,
           "probabilities": {"long": 0.71, "short": 0.1, "stand_aside": 0.19}, "setupFamily": "continuation",
           "bracket": {"entryKind": "at_market", "targetKind": "near", "targetNetR": 1.65, "entry": 100.0,
                       "stop": 98.0, "takeProfit": 103.4}, "outcome": "shadow", "recorded": True,
           "stake": "explore 0.40"}
    line = jev.describe_row(row)
    assert line.startswith("JEV SOL-USDT: LONG 0.71 (L 0.71 / S 0.10 / stand 0.19)")
    assert "entry 100 stop 98 tp 103.4 → shadow [scored] stake explore 0.40" in line
    assert "stand" in jev.describe_row({"symbol": "X", "direction": "stand_aside", "confidence": 0.8, "probabilities": {}})
    assert jev.describe_row({"symbol": "X", "outcome": "error", "detail": "boom"}) == "JEV X: error — boom"

  def test_an_idle_pass_leaves_a_status_the_dashboard_can_show(self, tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    m = MemoryStore(str(tmp_path / "m.json"))
    asyncio.run(jev.run_jev_pass(_cfg("shadow"), _Tools({}), m, universe=[], equity_usd=1.0,
                                 noise_mult=2.5, cost_rate=0.0016, now=1_790_000_000.0))
    st = m.jev_status()
    assert (st["state"], st["reason"], st["asked"]) == ("idle", "TYPESAFE_API_KEY not set", 0)
    pub = jev.public_status(st)
    assert set(pub) == {"ts", "state", "reason", "asked", "outcomes", "medianLatencyMs", "resolvedModel", "traced"}
    assert pub["state"] == "idle" and pub["reason"] == "TYPESAFE_API_KEY not set"

  def test_a_pass_records_status_and_what_the_order_path_said(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    tools = _Tools({}, results={"SOL-USDT": {"shadow": True, "probeRecorded": False, "repeat": True,
                                             "stake": "explore 0.40 (n=3 < 20)"}})
    _run(_cfg("shadow"), tools, m, _Client({"SOL-USDT": ("long", 0.7), "BTC-USDT": ("short", 0.66)}))
    rows = {r["symbol"]: r for r in m.jev_decisions(limit=0)}
    assert rows["SOL-USDT"]["repeat"] is True and rows["SOL-USDT"]["recorded"] is False
    assert rows["SOL-USDT"]["stake"].startswith("explore 0.40")
    st = m.jev_status()
    assert st["state"] == "ok" and st["asked"] == 3 and st["outcomes"] == {"shadow": 2, "stand_aside": 1}
    assert st["resolvedModel"] == "jev-1.4.2" and st["inputTokens"] == 2700

  def test_langsmith_trace_posts_the_whole_pass_and_ids_ride_on_the_rows(self, tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "_ls_traced_once", False)
    ls = _LsClient()
    tracer = jev.langsmith_tracer(_ls_cfg(), client=ls, rng=lambda: 0.99)
    m = MemoryStore(str(tmp_path / "m.json"))
    out = _run(_cfg("shadow"), _Tools({}), m, _Client({"SOL-USDT": ("long", 0.7)}), tracer=tracer)
    assert out["traced"] is True
    names = [r["name"] for r in ls.runs]
    assert names[0] == "Jev Dual Run (shadow)" and "Jev SOL-USDT" in names and len(names) == 4
    child = next(r for r in ls.runs if r["name"] == "Jev SOL-USDT")
    assert child["run_type"] == "llm" and child["inputs"]["state"]["symbol"] == "SOL-USDT"
    assert child["outputs"]["direction"] == "long" and child["outputs"]["usage_metadata"]["input_tokens"] == 900
    rows = {r["symbol"]: r for r in m.jev_decisions(limit=0)}
    assert rows["SOL-USDT"]["lsRunId"] == str(child["id"])
    # A routine pass after the first is sampled (0.99 >= 0.1: not posted)...
    ls.runs.clear()
    out = _run(_cfg("shadow"), _Tools({}), m, _Client({"SOL-USDT": ("long", 0.7)}), tracer=tracer)
    assert out.get("traced") is False and ls.runs == []
    # ...but a pass with an error always goes.
    out = _run(_cfg("shadow"), _Tools({}), m, _Client({}, raise_for={"BTC-USDT"}), tracer=tracer)
    assert out["traced"] is True and ls.runs

  def test_no_langsmith_config_no_tracer(self):
    assert jev.langsmith_tracer(_cfg()) is None and jev.langsmith_scorer(_cfg()) is None

  def test_settled_calls_get_the_markets_answer_as_feedback_once(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    t0 = int(time.time()) - 6 * 3600
    m.record_signal_probe("SOL-USDT", "buy", 100.0, "continuation", trader="jev")
    d = m._read()
    d["signal_probes"][0]["ts"] = t0
    d["signal_probes"][0]["entryContext"]["signalProbe"] = {"m15": 100.5, "m60": 99.0, "m240": 102.0}
    m._write(d)
    m.record_jev_decision({"ts": t0, "symbol": "SOL-USDT", "direction": "long", "outcome": "shadow",
                           "recorded": True, "lsRunId": "run-1"})
    m.record_jev_decision({"ts": t0, "symbol": "BTC-USDT", "direction": "stand_aside", "outcome": "stand_aside",
                           "lsRunId": "run-2"})
    m.record_jev_decision({"ts": int(time.time()) - 600, "symbol": "ETH-USDT", "direction": "long",
                           "outcome": "shadow", "recorded": True, "lsRunId": "run-3"})   # not due yet
    ls = _LsClient()
    score = jev.langsmith_scorer(_ls_cfg(), client=ls)
    assert score(m) == 2
    fb = {(rid, key): val for rid, key, val in ls.feedback}
    assert fb[("run-1", "fwd_15m")] == pytest.approx(0.5) and fb[("run-1", "fwd_60m")] == pytest.approx(-1.0)
    assert fb[("run-1", "fwd_240m")] == pytest.approx(2.0) and fb[("run-1", "right_way_60m")] == 0.0
    assert not any(rid == "run-2" for rid, _, _ in ls.feedback)          # a stand-aside has no direction
    assert score(m) == 0                                                 # each call scored once

  def test_report_rows_carry_the_llms_call_and_the_forward_return(self, tmp_path):
    m = MemoryStore(str(tmp_path / "m.json"))
    now = int(time.time())
    m.record_signal_probe("SOL-USDT", "buy", 100.0, "continuation", trader="jev")
    m.record_signal_probe("SOL-USDT", "sell", 100.0, "fade_extreme", confidence=0.74)
    d = m._read()
    for p in d["signal_probes"]:
      p["ts"] = now - 7200
      p["entryContext"]["signalProbe"] = {"m15": 101.0, "m60": 100.5}
    m._write(d)
    m.record_jev_decision({"ts": now - 7200, "symbol": "SOL-USDT", "direction": "long", "confidence": 0.7,
                           "outcome": "shadow", "recorded": True, "stake": "explore 0.40",
                           "bracket": {"entryKind": "at_market", "targetKind": "near", "targetNetR": 1.65, "stopAtr": 2.5}})
    m.set_jev_status({"ts": now - 60, "state": "ok", "asked": 1, "outcomes": {"shadow": 1}, "resolvedModel": "jev-1.4.2"})
    rep = jev.dual_run_report(m, _cfg("shadow"), cost_pct=0.0016)
    row = rep["recent"][0]
    assert row["llm"] == {"side": "short", "confidence": 0.74, "setupFamily": "fade_extreme"}
    assert row["fwdPct"] == {"15m": pytest.approx(1.0), "60m": pytest.approx(0.5), "240m": None}
    assert (row["recorded"], row["stake"], row["stopAtr"], row["targetKind"]) == (True, "explore 0.40", 2.5, "near")
    assert rep["status"]["state"] == "ok" and rep["status"]["resolvedModel"] == "jev-1.4.2"
    assert set(rep["traders"]["jev"]["byHorizon"]) >= {"15m", "60m"}
    line = jev.describe_comparison(rep)
    assert line.startswith("JEV vs LLM since") and "same side as the LLM 0/1" in line
