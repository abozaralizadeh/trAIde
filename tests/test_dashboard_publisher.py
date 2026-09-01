import pytest
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


class TestTriggerFreshness:
  def test_stale_triggers_dropped_recent_kept(self):
    import time as _t
    from src.dashboard_publisher import _TRIGGER_FRESHNESS_SEC
    pub = _publisher()
    now = _t.time()
    out = pub._sanitize_triggers([
      {"symbol": "SOL-USDT", "direction": "buy", "ts": now - _TRIGGER_FRESHNESS_SEC - 3600},  # stale
      {"symbol": "ETH-USDT", "direction": "sell", "ts": now - 600},                            # fresh
      {"symbol": "NOPE-USDT", "direction": "buy"},                                             # no ts
    ])
    syms = {t["symbol"] for t in out}
    assert syms == {"ETH-USDT"}


class TestClosedLifecycles:
  def test_lifecycle_fields_and_order(self):
    pub = _publisher()
    rows = pub._closed_position_lifecycles(_FakeMem([
      {"action": "futures_buy_triggered", "symbol": "ZEC-USDT", "pnl": -3.7, "ts": 1_784_003_600,
       "closeType": "CLOSE_LONG", "positionOpenTime": 1_784_000_000_000, "exitPrice": 560.0,
       "entryPrice": 590.0, "reason": "TP/SL triggered (CLOSE_LONG, ROE -5.56%)",
       "realizedR": -1.0, "troughPnl": -3.7, "peakPnl": 0.4,
       "entryContext": {"plannedMaxLossUsd": 3.7, "entryExtensionAtr": 3.2}},
      {"action": "futures_sell_triggered", "symbol": "XRP-USDT", "pnl": 0.3, "ts": 1_784_010_000,
       "closeType": "CLOSE_SHORT", "exitPrice": 1.05, "reason": "TP/SL triggered (CLOSE_SHORT, ROE 2.0%)"},
      {"action": "hold_short", "symbol": "ETH-USDT", "pnl": -0.1, "ts": 1_784_011_000},  # not a realized close
    ]), limit=3)
    assert [r["symbol"] for r in rows] == ["XRP-USDT", "ZEC-USDT"]  # newest first, hold excluded
    zec = rows[1]
    assert zec["side"] == "long" and zec["win"] is False and zec["roePct"] == -5.56
    assert zec["openTs"] == 1_784_000_000 and zec["closeTs"] == 1_784_003_600  # ms normalized to seconds
    assert zec["entryPrice"] == 590.0 and zec["exitPrice"] == 560.0
    # Entry/exit-quality feedback (unitless R + ATR; no dollars): ZEC ran fully against the entry.
    assert zec["realizedR"] == -1.0 and zec["maeR"] == 1.0 and zec["mfeR"] == round(0.4 / 3.7, 2)
    assert zec["entryExtensionAtr"] == 3.2 and zec["betterEntryAvailable"] is True
    # XRP has no entryContext → feedback fields degrade to None/False without error.
    assert rows[0]["realizedR"] is None and rows[0]["maeR"] is None and rows[0]["betterEntryAvailable"] is False


class _FakeMem:
  def __init__(self, decisions):
    self._decisions = decisions
  def latest_items(self, kind, limit=5):
    return {"items": list(self._decisions)}
  def realized_closes(self, limit=100, symbol=None):
    from src.memory import MemoryStore
    rows = [
      d for d in self._decisions
      if isinstance(d, dict) and d.get("pnl") is not None
      and MemoryStore._is_realized_close(str(d.get("action") or ""))
    ]
    rows.sort(key=lambda d: d.get("ts") or 0)
    return rows[-max(1, int(limit)):]


class TestPendingOrders:
  def test_pending_orders_public_safe_and_normalized(self):
    pub = _publisher()
    snap = SimpleNamespace(
      spot_pending_orders=[
        {"symbol": "ETH-USDT", "side": "buy", "type": "limit", "price": "1700", "size": "0.5", "createdAt": 1_784_000_000_000},
      ],
      futures_pending_orders=[
        {"symbol": "XBTUSDTM", "side": "sell", "type": "limit", "price": "62000", "size": "3",
         "clientOid": "traide-entry-abc", "createdAt": 1_784_000_100_000},
        {"symbol": "SOLUSDTM", "side": "buy", "type": "limit", "price": "78", "reduceOnly": True, "createdAt": 1_784_000_050_000},
      ],
    )
    out = pub._sanitize_pending_orders(snap)
    # newest first
    assert [o["symbol"] for o in out] == ["BTC-USDT", "SOL-USDT", "ETH-USDT"]
    btc = out[0]
    assert btc["side"] == "sell" and btc["venue"] == "futures" and btc["kind"] == "entry" and btc["botEntry"] is True
    assert btc["price"] == 62000 and btc["ts"] == 1_784_000_100  # ms->s
    # no size/quantity ever leaks
    assert all("size" not in o and "quantity" not in o for o in out)
    # reduce-only flagged
    assert next(o for o in out if o["symbol"] == "SOL-USDT")["kind"] == "reduce"

  def test_pending_orders_empty(self):
    pub = _publisher()
    assert pub._sanitize_pending_orders(SimpleNamespace(spot_pending_orders=[], futures_pending_orders=[])) == []


class TestStrategyEdgePanel:
  """The dashboard should show WHY the bot is winning or losing, not just that it is.

  Outcomes (win rate, PnL) conflate the direction call with fill quality and exit management, so they
  cannot answer whether the strategy has an edge at all. strategyEdge measures the signal alone and
  reports which playbook is currently paying its costs.
  """

  @staticmethod
  def _memory(probes, fills=()):
    return SimpleNamespace(
      signal_probes=lambda limit=200: list(probes),
      recent_fills=lambda limit=100: list(fills),
    )

  @staticmethod
  def _cfg():
    return SimpleNamespace(trading=SimpleNamespace(
      estimated_slippage_pct=0.001, slippage_autotune_min_samples=8,
    ))

  _seq = [0]

  @classmethod
  def _probe(cls, side, base, fwd, family):
    """One INDEPENDENT observation — spaced past the widest horizon and keyed per family, since
    signal_edge_stats collapses probes that overlap on the same symbol (see test_edge.py)."""
    cls._seq[0] += 1
    return {"symbol": f"{family.upper()}-USDT", "ts": 1_000_000 + cls._seq[0] * 240 * 60,
            "entryContext": {"positionSide": side, "marketPriceAtSignal": base,
                             "setupFamily": family, "signalProbe": {"m60": fwd}}}

  def test_reports_per_family_verdicts_and_risk_factors(self):
    pub = _publisher()
    probes = ([self._probe("short", 100.0, 100.05, "continuation") for _ in range(25)]
              + [self._probe("long", 100.0, 102.0, "fade_extreme") for _ in range(25)])
    out = pub._build_strategy_edge(self._memory(probes), self._cfg())
    assert out["byFamily"]["continuation"]["verdict"] == "no edge"
    assert out["byFamily"]["fade_extreme"]["verdict"] == "edge"
    # ...and the multiplier that explains where capital is going.
    assert out["familyRiskFactor"]["continuation"] == 0.25
    assert out["familyRiskFactor"]["fade_extreme"] == 1.0

  def test_publishes_no_money_figures_under_normalized_disclosure(self):
    """Percentages, counts and verdicts only — nothing here can leak balance or position size."""
    pub = _publisher("normalized")
    probes = [self._probe("long", 100.0, 101.0, "continuation") for _ in range(25)]
    out = pub._build_strategy_edge(self._memory(probes), self._cfg())
    banned = {"equity", "balance", "notional", "usd", "size", "accountid"}
    def _keys(obj, acc):
      if isinstance(obj, dict):
        for k, v in obj.items():
          acc.add(str(k).lower()); _keys(v, acc)
      elif isinstance(obj, list):
        for v in obj:
          _keys(v, acc)
      return acc
    keys = _keys(out, set())
    assert not any(b in k for k in keys for b in banned), keys

  def test_degrades_quietly_when_there_is_nothing_to_measure(self):
    pub = _publisher()
    out = pub._build_strategy_edge(self._memory([]), self._cfg())
    assert out["verdict"] == "insufficient data" and out["n"] == 0

  def test_never_raises_into_the_publish_loop(self):
    pub = _publisher()
    broken = SimpleNamespace(
      signal_probes=lambda limit=200: (_ for _ in ()).throw(RuntimeError("boom")),
      recent_fills=lambda limit=100: [],
    )
    out = pub._build_strategy_edge(broken, self._cfg())
    assert out["verdict"] == "insufficient data"


class TestSetupFamilyOnPositions:
  """Aggregate family scores answer 'which playbook pays'; the rows answer 'which trades were those'.

  Without a family on each row you can see that continuation is losing but cannot identify the trades
  behind the number, and a fade would be invisible until it aggregated into a bucket.
  """

  @staticmethod
  def _memory(fills=(), probes=()):
    return SimpleNamespace(
      recent_fills=lambda limit=200: list(fills),
      signal_probes=lambda limit=200: list(probes),
    )

  @staticmethod
  def _row(sym, side, family, oid=None):
    return {"symbol": sym, "clientOid": oid,
            "entryContext": {"positionSide": side, "setupFamily": family,
                             "marketPriceAtSignal": 100.0}}

  def test_index_resolves_by_client_oid_and_by_symbol_side(self):
    pub = _publisher()
    mem = self._memory(fills=[self._row("XRP-USDT", "short", "fade_extreme", oid="traide-entry-abc")])
    idx = pub._family_index(mem)
    assert pub._family_for(idx, "XRP-USDT", "sell", "traide-entry-abc") == "fade_extreme"   # exact
    assert pub._family_for(idx, "XRP-USDT", "short") == "fade_extreme"                      # fallback
    assert pub._family_for(idx, "DOGE-USDT", "long") is None                                # unknown

  def test_client_oid_wins_over_the_symbol_fallback(self):
    # Two entries on the same symbol/side: the exact order id must not be shadowed by the newer one.
    pub = _publisher()
    mem = self._memory(fills=[
      self._row("ADA-USDT", "long", "fade_extreme", oid="traide-entry-1"),
      self._row("ADA-USDT", "long", "continuation", oid="traide-entry-2"),
    ])
    idx = pub._family_index(mem)
    assert pub._family_for(idx, "ADA-USDT", "buy", "traide-entry-1") == "fade_extreme"
    assert pub._family_for(idx, "ADA-USDT", "long") == "continuation"   # most recent wins by symbol

  def test_index_falls_back_to_inference_when_undeclared(self):
    pub = _publisher()
    undeclared = {"symbol": "SOL-USDT", "clientOid": "traide-entry-x",
                  "entryContext": {"positionSide": "long", "marketPriceAtSignal": 100.0,
                                   "regime": {"intraday_bias_4h": "bearish", "intraday_bias_1h": "bearish"}}}
    idx = pub._family_index(self._memory(fills=[undeclared]))
    assert pub._family_for(idx, "SOL-USDT", "long", "traide-entry-x") == "fade_extreme"

  def test_family_lookup_is_safe_on_missing_index_and_bad_input(self):
    pub = _publisher()
    assert pub._family_for(None, "XRP-USDT", "long") is None
    assert pub._family_for({}, None, None, None) is None

  def test_index_never_raises_when_memory_misbehaves(self):
    pub = _publisher()
    broken = SimpleNamespace(
      recent_fills=lambda limit=200: (_ for _ in ()).throw(RuntimeError("boom")),
      signal_probes=lambda limit=200: [],
    )
    assert pub._family_index(broken) == {"byOid": {}, "bySymbolSide": {}}


class TestEquityIndexSanity:
  """The published index is a DAILY CHAIN over a durable, never-rewritten Azure series.

  indexClose_today = prevDayClose * (1 + intradayReturn). That makes one bad point permanent: every
  later day multiplies it forward. On 2026-08-31, after a two-week outage, the live dashboard showed
  an indexed return of +72,546,760% — index 72,546,860 against a base of 100, a 725,468x blow-up.
  """

  @staticmethod
  def _pub(prev_close):
    pub = _publisher()
    pub.cfg = SimpleNamespace(disclosure="normalized", index_base=100.0)

    class _Table:
      def query_entities(self, **kw):
        return [{"RowKey": "00020695", "indexClose": prev_close}]

    pub._table_client = _Table()
    return pub

  @staticmethod
  def _mem():
    return SimpleNamespace(latest_items=lambda *a, **k: {"items": []})

  def test_chain_guard_reanchors_a_corrupt_previous_close(self):
    # The exact value seen on the live dashboard.
    assert self._pub(72546860.79)._prev_day_close(20696) == 100.0

  def test_chain_guard_leaves_a_healthy_series_alone(self):
    assert self._pub(118.4)._prev_day_close(20696) == 118.4
    # boundaries of the sane band are still accepted
    assert self._pub(0.1)._prev_day_close(20696) == 0.1
    assert self._pub(100000.0)._prev_day_close(20696) == 100000.0

  def test_step_guard_holds_the_index_flat_on_a_partial_balance_snapshot(self):
    """A near-zero daily baseline (spot only, futures 504'd) fabricates a five-figure return."""
    bad = {"total": {"baselineUsdt": 0.001, "currentUsdt": 67.44, "drawdownPct": 0.0}}
    out = self._pub(118.4)._compute_today_equity(self._mem(), bad, 20696)
    assert out["indexClose"] == pytest.approx(118.4)      # carried, not compounded

  def test_step_guard_lets_a_real_day_through(self):
    good = {"total": {"baselineUsdt": 67.27, "currentUsdt": 67.44, "drawdownPct": 0.0}}
    out = self._pub(118.4)._compute_today_equity(self._mem(), good, 20696)
    assert out["indexClose"] == pytest.approx(118.4 * (1 + (67.44 - 67.27) / 67.27))

  def test_step_guard_still_allows_a_large_but_believable_move(self):
    # -30% in a day is a catastrophe, not a data error — it must be published honestly.
    rough = {"total": {"baselineUsdt": 100.0, "currentUsdt": 70.0, "drawdownPct": 30.0}}
    out = self._pub(118.4)._compute_today_equity(self._mem(), rough, 20696)
    assert out["indexClose"] == pytest.approx(118.4 * 0.7)


  def test_corrupt_history_is_hidden_from_the_published_curve(self):
    """Today's value healing is not enough — the durable table still holds the bad rows.

    Azure history is never rewritten here, so without filtering the read the chart keeps rendering
    the 725,468x spike even once the chain guard has re-anchored the present.
    """
    pub = self._pub(118.4)

    class _Table:
      def query_entities(self, **kw):
        return [
          {"RowKey": "00020690", "indexClose": 101.2, "drawdownPct": 0.1},
          {"RowKey": "00020695", "indexClose": 72546860.79, "drawdownPct": 0.0},   # corrupt
          {"RowKey": "00020696", "indexClose": 118.4, "drawdownPct": 0.2},
        ]

    pub._table_client = _Table()
    days = [p["day"] for p in pub._read_equity_series()]
    assert days == [20690, 20696], "the corrupt point must not reach the chart"


class TestClosedPositionsRenderability:
  """A closed position needs a side and a price to draw as a trade.

  Seen live on 2026-09-01: NEAR appeared TWICE in "Recently closed" — once complete
  (RANGE EDGE / SHORT / entry 1.99400 -> exit 1.99100 / +0.02R) and once as an empty card with no
  side, no prices and no family. The MemoryStore dedup catches upstream duplicates by shape, but this
  bug class has now surfaced under four different action names, so the presentation layer refuses
  un-renderable rows outright rather than waiting to learn the fifth.
  """

  @staticmethod
  def _mem(rows):
    return SimpleNamespace(realized_closes=lambda limit=100, symbol=None: list(rows))

  REAL = {
    "symbol": "NEAR-USDT", "ts": 1000, "pnl": 0.0006, "closeType": "CLOSE_SHORT",
    "entryPrice": 1.994, "exitPrice": 1.991, "realizedR": 0.02,
    "reason": "TP/SL triggered (CLOSE_SHORT, ROE 0.06%)",
    "entryContext": {"setupFamily": "range_edge", "plannedMaxLossUsd": 0.03},
  }

  @pytest.mark.parametrize("shell", [
    {"symbol": "NEAR-USDT", "ts": 1001, "pnl": 0.0006, "exitPrice": 1.991},              # price only
    {"symbol": "NEAR-USDT", "ts": 1002, "pnl": 0.0006, "realizedR": 0.02},               # R only
    {"symbol": "NEAR-USDT", "ts": 1003, "pnl": 0.0006, "closeType": "CLOSE_SHORT"},      # side only
    {"symbol": "NEAR-USDT", "ts": 1004, "pnl": 0.0006, "action": "futures_buy_triggered"},
  ])
  def test_a_fragment_never_becomes_a_second_card(self, shell):
    rows = _publisher()._closed_position_lifecycles(self._mem([self.REAL, shell]))
    assert len(rows) == 1
    assert rows[0]["entryPrice"] == pytest.approx(1.994)

  def test_the_genuine_trade_still_publishes_in_full(self):
    rows = _publisher()._closed_position_lifecycles(self._mem([self.REAL]))
    assert len(rows) == 1
    r = rows[0]
    assert r["side"] == "short" and r["setupFamily"] == "range_edge"
    assert r["entryPrice"] == pytest.approx(1.994) and r["exitPrice"] == pytest.approx(1.991)
    assert r["realizedR"] == pytest.approx(0.02) and r["roePct"] == pytest.approx(0.06)

  def test_an_older_row_missing_only_realized_r_is_still_shown(self):
    """Rows predating realizedR/setupFamily must not be swept up — they render fine."""
    old = {"symbol": "NEAR-USDT", "ts": 900, "pnl": -0.032, "closeType": "CLOSE_LONG",
           "entryPrice": 2.10, "exitPrice": 2.060167}
    rows = _publisher()._closed_position_lifecycles(self._mem([old]))
    assert len(rows) == 1 and rows[0]["side"] == "long"
