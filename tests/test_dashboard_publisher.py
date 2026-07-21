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
