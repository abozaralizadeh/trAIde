import time
from types import SimpleNamespace

import pytest

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


class TestClosedTradeBarChartAgreesWithTheCards:
  """Two panels counting the same trades must reach the same number.

  Reported live on 2026-09-01: "recently closed" showed one win and one loss while the outcome bar
  chart showed one win and TWO losses. The chart filtered the raw decisions feed itself and so was
  the only closed-trade surface that skipped the estimate/echo dedupe every other panel goes
  through, and NEAR-USDT had been reported twice — the bracket, then the agent narrating the same
  close seconds later at a different figure.
  """

  @staticmethod
  def _mem(rows):
    return SimpleNamespace(latest_items=lambda kind, limit=50: {"items": list(rows)})

  def test_a_narrated_echo_does_not_draw_its_own_bar(self):
    rows = [
      {"symbol": "NEAR-USDT", "action": "close_long", "ts": 1005, "pnl": +0.00666,
       "reason": "closed the runner"},
      {"symbol": "NEAR-USDT", "action": "futures_sell_triggered", "ts": 1000, "pnl": +0.00335,
       "closeType": "CLOSE_LONG", "exitPrice": 2.481, "reason": "TP/SL triggered (ROE 0.34%)"},
    ]
    out = _publisher()._closed_trades(self._mem(rows))
    assert len(out) == 1
    assert out[0]["action"] == "futures_sell_triggered"

  def test_genuinely_separate_closes_all_keep_their_bars_newest_first(self):
    rows = [
      {"symbol": "XRP-USDT", "action": "futures_sell_triggered", "ts": 3000, "pnl": -0.41,
       "closeType": "CLOSE_LONG", "exitPrice": 2.9},
      {"symbol": "NEAR-USDT", "action": "futures_buy_triggered", "ts": 2000, "pnl": +0.12,
       "closeType": "CLOSE_SHORT", "exitPrice": 2.4},
    ]
    out = _publisher()._closed_trades(self._mem(rows))
    assert [d["symbol"] for d in out] == ["XRP-USDT", "NEAR-USDT"]

  def test_rows_that_are_not_closes_are_still_excluded(self):
    rows = [
      {"symbol": "ADA-USDT", "action": "decline", "ts": 4000, "reason": "no edge"},
      {"symbol": "ADA-USDT", "action": "futures_buy", "ts": 4100, "reason": "entry placed"},
    ]
    assert _publisher()._closed_trades(self._mem(rows)) == []


class TestMacroEventsPanel:
  """The calendar and blackout state published to the dashboard.

  Everything here is public information — release names, times and which window the bot is in — so it
  is safe in every disclosure mode. The state a reader most needs is the one that is easy to get wrong:
  a stale calendar means NO blackout is in force, which must be visible rather than inferred.
  """

  @staticmethod
  def _cfg():
    return SimpleNamespace(
      dashboard=SimpleNamespace(disclosure="normalized"),
      regime=SimpleNamespace(macro_events_enabled=True,
                             macro_event_before_min=60.0, macro_event_after_min=60.0),
    )

  @staticmethod
  def _store(tmp_path):
    from src.memory import MemoryStore
    return MemoryStore(str(tmp_path / "m.json"))

  def test_empty_calendar_publishes_no_blackout(self, tmp_path):
    cfg = self._cfg()
    out = DashboardPublisher(cfg)._build_macro_events(self._store(tmp_path), cfg)
    assert out["upcoming"] == []
    assert out["entriesBlockedNow"] is False
    assert out["activeWindow"] is None
    assert out["calendarAgeHours"] is None      # never fetched — readers must see that

  def test_imminent_release_publishes_the_active_window(self, tmp_path):
    import time
    cfg, m = self._cfg(), self._store(tmp_path)
    m.record_macro_events([
      {"name": "US CPI (Aug)", "ts": time.time() + 22 * 60, "impact": "high"},
      {"name": "FOMC statement", "ts": time.time() + 3 * 86400, "impact": "high"},
    ])
    out = DashboardPublisher(cfg)._build_macro_events(m, cfg)
    assert out["entriesBlockedNow"] is True
    assert out["activeWindow"]["phase"] == "before"
    assert out["activeWindow"]["name"] == "US CPI (Aug)"
    assert [e["name"] for e in out["upcoming"]] == ["US CPI (Aug)", "FOMC statement"]
    assert out["upcoming"][0]["inMinutes"] == pytest.approx(22, abs=1)

  def test_disabled_publishes_the_calendar_but_no_block(self, tmp_path):
    import time
    cfg, m = self._cfg(), self._store(tmp_path)
    cfg.regime.macro_events_enabled = False
    m.record_macro_events([{"name": "US CPI", "ts": time.time() + 10 * 60, "impact": "high"}])
    out = DashboardPublisher(cfg)._build_macro_events(m, cfg)
    assert out["enabled"] is False
    assert out["entriesBlockedNow"] is False     # the guard is off...
    assert len(out["upcoming"]) == 1             # ...but the calendar is still informative

  def test_publishes_no_account_data(self, tmp_path):
    import json, time
    cfg, m = self._cfg(), self._store(tmp_path)
    m.record_macro_events([{"name": "US CPI", "ts": time.time() + 600, "impact": "high"}])
    blob = json.dumps(DashboardPublisher(cfg)._build_macro_events(m, cfg)).lower()
    for banned in ("usdt", "equity", "balance", "notional", "accountid", "$"):
      assert banned not in blob


class TestExitDisciplinePanel:
  def test_r_multiples_only_never_dollars(self, tmp_path):
    """R is a ratio to the trade's own risk, so this is safe under `normalized` disclosure."""
    import json, time
    from src.memory import MemoryStore
    cfg = SimpleNamespace(dashboard=SimpleNamespace(disclosure="normalized"))
    m = MemoryStore(str(tmp_path / "m.json"))
    m.record_exit_probe("AAVE-USDT", "short", 132.35, 135.25, 127.10, 131.94, realized_r=0.14,
                        setup_family="fade_extreme")
    data = m._read()
    data["exit_probes"][0]["outcome"] = {"resolved": "take_profit", "bracketR": 1.76,
                                         "resolvedTs": int(time.time())}
    m._write(data)
    out = DashboardPublisher(cfg)._build_exit_discipline(m)
    assert out["n"] == 1
    assert out["takenR"] == pytest.approx(0.14)
    assert out["bracketR"] == pytest.approx(1.76)
    assert out["deltaR"] == pytest.approx(-1.62)
    blob = json.dumps(out).lower()
    for banned in ("usd", "equity", "balance", "notional", "$"):
      assert banned not in blob

  def test_survives_a_store_with_no_probes(self, tmp_path):
    from src.memory import MemoryStore
    cfg = SimpleNamespace(dashboard=SimpleNamespace(disclosure="normalized"))
    out = DashboardPublisher(cfg)._build_exit_discipline(MemoryStore(str(tmp_path / "m.json")))
    assert out == {"verdict": "insufficient data", "n": 0, "takenR": 0, "bracketR": 0,
                   "deltaR": 0, "deltaRPerTrade": None, "beatBracket": 0, "byFamily": {}}


class TestTakerFlowPanel:
  """The dashboard's window onto the flow experiment while it runs.

  Two halves that answer different questions: `live` is what the tape is doing right now (genuinely
  new information on the panel — until now it only ever showed closed candles at 15m and above), and
  `byHorizon` is whether that reading has ever predicted anything at our horizons.
  """

  @staticmethod
  def _memory(probes=(), flow=None):
    return SimpleNamespace(
      signal_probes=lambda limit=200: list(probes),
      recent_fills=lambda limit=100: [],
      get_agent_scheduler=lambda: {"flowObservations": dict(flow or {})},
    )

  @staticmethod
  def _cfg(enabled=True):
    return SimpleNamespace(
      trading=SimpleNamespace(estimated_slippage_pct=0.001, slippage_autotune_min_samples=8,
                              poll_interval_sec=60),
      edge=SimpleNamespace(taker_flow_enabled=enabled),
    )

  _seq = [0]

  @classmethod
  def _probe(cls, side, fwd, buy_share):
    """One independent flow-stamped observation, spaced past the widest horizon (see test_edge.py)."""
    cls._seq[0] += 1
    ctx = {"positionSide": side, "marketPriceAtSignal": 100.0, "signalProbe": {"m60": fwd}}
    if buy_share is not None:
      ctx["takerFlow"] = {"buyShare": buy_share}
    return {"symbol": "F-USDT", "ts": 3_000_000 + cls._seq[0] * 240 * 60, "entryContext": ctx}

  def test_publishes_the_with_against_spread_and_its_verdict(self):
    pub = _publisher()
    probes = ([self._probe("long", 100.5, 0.8) for _ in range(25)]
              + [self._probe("long", 99.5, 0.2) for _ in range(25)])
    out = pub._build_taker_flow(self._memory(probes), self._cfg())
    assert out["enabled"] is True
    assert out["byHorizon"]["60m"]["spread_pct"] == pytest.approx(1.0)
    assert out["byHorizon"]["60m"]["verdict"] == "tradable"
    assert out["coverage"] == 1.0

  def test_live_readings_carry_an_age_so_a_stalled_sampler_is_visible(self):
    """Without an age, a sampler that stopped an hour ago looks exactly like a calm market."""
    import time as _t
    pub = _publisher()
    now = int(_t.time())
    out = pub._build_taker_flow(self._memory(flow={
      "BTC-USDT": {"buyShare": 0.61, "buyShareEwma": 0.58, "buyTradeShare": 0.55,
                   "trades": 100, "spanSec": 182.4, "samples": 12, "updated": now - 90},
      # No usable share: publishing the husk would show as balanced flow rather than as no data.
      "ETH-USDT": {"trades": 100, "updated": now},
    }), self._cfg())
    assert out["live"]["BTC-USDT"]["buyShare"] == 0.61
    assert out["live"]["BTC-USDT"]["buyShareEwma"] == 0.58
    assert 85 <= out["live"]["BTC-USDT"]["ageSec"] <= 120
    assert "ETH-USDT" not in out["live"]

  def test_a_reading_past_its_shelf_life_is_dropped_rather_than_shown_as_live(self):
    """A panel labelled "live" must not carry a reading from days ago.

    Symbols rotate through the sampling cap and a rotated-out symbol simply stops being refreshed;
    live state was found holding readings 63 HOURS old. An old reading is not a quiet tape, and at
    a glance a stale buyShare of 0.8 is indistinguishable from a real one.
    """
    import time as _t
    pub = _publisher()
    now = int(_t.time())
    out = pub._build_taker_flow(self._memory(flow={
      "BTC-USDT": {"buyShare": 0.61, "updated": now - 120},          # inside 10 polls x 60s
      "OLD-USDT": {"buyShare": 0.80, "updated": now - 63 * 3600},    # the reading found live
      "NOTS-USDT": {"buyShare": 0.55},                               # no timestamp: unverifiable
    }), self._cfg())
    assert list(out["live"]) == ["BTC-USDT"]

  def test_the_panel_says_when_collection_is_switched_off(self):
    pub = _publisher()
    out = pub._build_taker_flow(self._memory(), self._cfg(enabled=False))
    assert out["enabled"] is False and out["verdict"] == "insufficient data"

  def test_publishes_no_money_figures_under_normalized_disclosure(self):
    """Shares, counts, ages and percentages only — nothing here can leak balance or position size."""
    pub = _publisher("normalized")
    probes = ([self._probe("long", 100.5, 0.8) for _ in range(25)]
              + [self._probe("long", 99.5, 0.2) for _ in range(25)])
    out = pub._build_taker_flow(
      self._memory(probes, flow={"BTC-USDT": {"buyShare": 0.6, "updated": int(time.time())}}),
      self._cfg())
    banned = {"equity", "balance", "notional", "usd", "size", "accountid", "qty", "leverage"}
    def _keys(obj, acc):
      if isinstance(obj, dict):
        for k, v in obj.items():
          # Symbol keys ("BTC-USDT") are names, not disclosures — everything else must be clean.
          if "-" not in str(k):
            acc.add(str(k).lower())
          _keys(v, acc)
      elif isinstance(obj, list):
        for v in obj:
          _keys(v, acc)
      return acc
    keys = _keys(out, set())
    assert not any(b in k for k in keys for b in banned), keys

  def test_a_broken_half_never_takes_down_the_other_or_the_publish_loop(self):
    pub = _publisher()
    broken_stats = SimpleNamespace(
      signal_probes=lambda limit=200: (_ for _ in ()).throw(RuntimeError("boom")),
      recent_fills=lambda limit=100: [],
      get_agent_scheduler=lambda: {
        "flowObservations": {"BTC-USDT": {"buyShare": 0.6, "updated": int(time.time())}}},
    )
    out = pub._build_taker_flow(broken_stats, self._cfg())
    assert out["verdict"] == "insufficient data"
    assert out["live"]["BTC-USDT"]["buyShare"] == 0.6      # the live half still published

    broken_live = SimpleNamespace(
      signal_probes=lambda limit=200: [],
      recent_fills=lambda limit=100: [],
      get_agent_scheduler=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert pub._build_taker_flow(broken_live, self._cfg())["live"] == {}


class TestEquityChainGapGuard:
  """A gap in the equity chain must not be booked as one day's return.

  Live, 2026-08-15 -> 08-31: the bot was down 16 days. On restart it compared today's equity against
  a baseline from whenever it stopped, so the whole gap — including a capital change the bot had no
  part in — landed as a single daily return. The index stepped 84.17 -> 100.55, a fabricated +19.46%
  that sailed through the 50% step guard, and because the index COMPOUNDS it inflated every later
  point: the curve now reads ~flat since June on an account that is actually down ~16.5%.
  """

  @staticmethod
  def _pub(prev_close, prev_day, today, baseline, current):
    cfg = SimpleNamespace(dashboard=SimpleNamespace(disclosure="normalized", index_base=100.0))
    pub = DashboardPublisher(cfg)
    pub._prev_day_point = lambda _t: (prev_close, prev_day)
    limits = {"total": {"baselineUsdt": baseline, "currentUsdt": current, "drawdownPct": 0.0}}
    return pub._compute_today_equity(SimpleNamespace(latest_items=lambda *a, **k: {"items": []}),
                                     limits, today)

  def test_the_live_16_day_outage_no_longer_fabricates_a_return(self):
    # baseline 56.59 -> current 67.60 is the +19.46% the outage produced.
    out = self._pub(84.172806, 20680, 20696, 56.59, 67.60)
    assert out["indexClose"] == pytest.approx(84.172806, abs=1e-6), \
      "a 16-day gap must hold the index flat, not compound the gap as one day"

  def test_a_normal_consecutive_day_still_compounds(self):
    """The guard must not freeze the curve in ordinary operation."""
    out = self._pub(100.0, 20704, 20705, 100.0, 101.0)
    assert out["indexClose"] == pytest.approx(101.0, abs=1e-6)

  def test_a_single_missed_day_is_tolerated_as_a_trading_day(self):
    """One missing publish is a hiccup, not an outage — day-1 is still 'yesterday' enough to compound
    (the 50% step guard remains the backstop for an implausible move)."""
    out = self._pub(100.0, 20704, 20705, 100.0, 102.0)
    assert out["indexClose"] == pytest.approx(102.0, abs=1e-6)

  def test_a_gap_with_no_move_is_untouched(self):
    out = self._pub(84.0, 20680, 20696, 67.0, 67.0)
    assert out["indexClose"] == pytest.approx(84.0, abs=1e-6)

  def test_the_first_ever_point_has_no_chain_to_break(self):
    """prev_day None = nothing stored yet; the guard must not swallow the opening day."""
    out = self._pub(100.0, None, 20611, 100.0, 101.3)
    assert out["indexClose"] == pytest.approx(101.3, abs=1e-6)

  def test_the_step_guard_still_catches_an_implausible_same_day_move(self):
    """Unrelated backstop, still armed: a >50% one-day move is a bad snapshot even without a gap."""
    out = self._pub(100.0, 20704, 20705, 10.0, 100.0)
    assert out["indexClose"] == pytest.approx(100.0, abs=1e-6)


class TestCorruptCloseSkipsToLastGood:
  """Corruption must be SKIPPED, not made a reason to forget the history behind it.

  Live sequence that destroyed the published curve:
      day 20680 = 84.172806   (last good close, the account really was down ~15.8%)
      day 20693-20695 ~ 7.27e7 (corrupt, written before the step guard existed)
      day 20696 = 100.552734   <- the old heal re-anchored to base 100
  Re-anchoring healed the arithmetic but discarded four months of real performance: the curve then
  read roughly break-even since June on an account down ~16.5%. The last sane close is still sitting
  in the table one row back, so resume from it.
  """

  @staticmethod
  def _pub(rows):
    cfg = SimpleNamespace(dashboard=SimpleNamespace(disclosure="normalized", index_base=100.0))
    pub = DashboardPublisher(cfg)
    pub._table_client = SimpleNamespace(
      query_entities=lambda **k: [{"RowKey": f"{d:08d}", "indexClose": v} for d, v in rows])
    return pub

  def test_resumes_from_the_last_sane_close_instead_of_base(self):
    rows = [(20680, 84.172806), (20693, 72714000.4928),
            (20694, 72738827.545142), (20695, 72358594.525117)]
    close, day = self._pub(rows)._prev_day_point(20696)
    assert close == pytest.approx(84.172806), "must resume the real curve, not reset to base 100"
    assert day == 20680

  def test_a_clean_series_is_unaffected(self):
    close, day = self._pub([(20704, 99.5), (20705, 99.7)])._prev_day_point(20706)
    assert close == pytest.approx(99.7) and day == 20705

  def test_falls_back_to_base_only_when_no_sane_close_exists(self):
    """If every stored row is corrupt there is nothing to resume from — base is the honest anchor."""
    close, day = self._pub([(20693, 7.2e7), (20694, 7.3e7)])._prev_day_point(20695)
    assert close == pytest.approx(100.0) and day is None

  def test_an_empty_table_anchors_at_base(self):
    close, day = self._pub([])._prev_day_point(20611)
    assert close == pytest.approx(100.0) and day is None

  def test_the_gap_guard_still_sees_the_last_good_day_not_the_corrupt_one(self):
    """The two guards compose: the day returned must be the SANE row's day, so the chain-gap check
    measures the real gap (20680 -> 20696 = 16 days) rather than a 1-day hop off a corrupt row."""
    rows = [(20680, 84.172806), (20695, 72358594.525117)]
    _, day = self._pub(rows)._prev_day_point(20696)
    assert day == 20680
