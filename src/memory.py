from __future__ import annotations

import copy
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import normalize_symbol as _normalize_symbol

logger = logging.getLogger(__name__)

# Hard caps to keep the memory file small and readable (agent only needs recent history).
MAX_PLANS = 3
MAX_TRIGGERS = 5
MAX_COINS = 50
MAX_TRADES = 100
MAX_SENTIMENTS = 10
MAX_DECISIONS = 50       # entry/decline decisions (pnl=None)
MAX_CLOSED_TRADES = 200  # closed-trade outcomes (pnl != None) — kept separately
MAX_FEES = 3
MAX_TEMPORARY_NOTES = 20
MAX_PERMANENT_NOTES = 10


class MemoryStore:
  """Lightweight JSON-backed store for agent plans/notes."""

  def __init__(self, path: str, retention_days: int = 7) -> None:
    self.path = Path(path)
    self._lock = threading.Lock()
    self.retention_days = retention_days
    self._cache: Dict[str, Any] | None = None
    self._cache_mtime: float | None = None
    # Sanitize on init.
    self._read()

  def _read(self) -> Dict[str, Any]:
    _empty: Dict[str, Any] = {"plans": [], "triggers": [], "coins": [], "trades": [], "limits": {}, "sentiments": [], "decisions": [], "fees": [], "supervisor_notes_temporary": [], "supervisor_notes_permanent": [], "position_extremes": {}}
    if self._cache is not None:
      try:
        disk_mtime = self.path.stat().st_mtime
        if disk_mtime != self._cache_mtime:
          self._cache = None
      except OSError:
        pass
    if self._cache is not None:
      return copy.deepcopy(self._cache)
    if not self.path.exists():
      return _empty
    try:
      raw = json.loads(self.path.read_text())
      data = copy.deepcopy(raw)
      if isinstance(data, dict):
        data.setdefault("plans", [])
        data.setdefault("triggers", [])
        data.setdefault("coins", [])
        data.setdefault("trades", [])
        data.setdefault("limits", {})
        data.setdefault("sentiments", [])
        data.setdefault("decisions", [])
        data.setdefault("fees", [])
        data.setdefault("supervisor_notes_temporary", [])
        data.setdefault("supervisor_notes_permanent", [])
        data.setdefault("position_extremes", {})
        # prune invalid entries while keeping timestamp
        data["plans"] = [
          p
          for p in data.get("plans", [])
          if isinstance(p, dict) and p.get("title") and p.get("summary") and isinstance(p.get("actions"), list)
        ]
        data["triggers"] = [
          {
            "symbol": _normalize_symbol(t.get("symbol", "")),
            "direction": t.get("direction"),
            "rationale": t.get("rationale"),
            "targetPrice": t.get("targetPrice"),
            "stopPrice": t.get("stopPrice"),
            "ts": t.get("ts"),
          }
          for t in data.get("triggers", [])
          if isinstance(t, dict) and t.get("symbol") and t.get("direction")
        ]
        data["coins"] = [
          {
            "symbol": _normalize_symbol(c.get("symbol", "")),
            "status": c.get("status", "active"),
            "reason": c.get("reason"),
            "exitPlan": c.get("exitPlan"),
            "ts": c.get("ts"),
          }
          for c in data.get("coins", [])
          if isinstance(c, dict) and c.get("symbol")
        ]
        data["trades"] = [
          {
            "symbol": _normalize_symbol(t.get("symbol", "")),
            "side": t.get("side"),
            "notionalUsd": t.get("notionalUsd"),
            "price": t.get("price"),
            "size": t.get("size"),
            "paper": t.get("paper", False),
            "venue": t.get("venue", "spot"),
            "ts": t.get("ts") or int(time.time()),
            "day": t.get("day"),
          }
          for t in data.get("trades", [])
          if isinstance(t, dict) and t.get("symbol") and t.get("ts")
        ]
        data["sentiments"] = [
          {
            "symbol": _normalize_symbol(s.get("symbol", "")),
            "score": s.get("score"),
            "rationale": s.get("rationale", ""),
            "source": s.get("source", ""),
            "ts": s.get("ts") or int(time.time()),
            "day": s.get("day"),
          }
          for s in data.get("sentiments", [])
          if isinstance(s, dict) and s.get("symbol") and s.get("ts") is not None
        ]
        data["decisions"] = [
          {
            "symbol": _normalize_symbol(d.get("symbol", "")),
            "action": d.get("action"),
            "confidence": d.get("confidence"),
            "reason": d.get("reason", ""),
            "pnl": d.get("pnl"),
            "paper": d.get("paper", False),
            "ts": d.get("ts") or int(time.time()),
            "day": d.get("day"),
            "peakPnl": d.get("peakPnl"),
            "troughPnl": d.get("troughPnl"),
          }
          for d in data.get("decisions", [])
          if isinstance(d, dict) and d.get("symbol") and d.get("ts") is not None
        ]
        limits = data.get("limits") or {}
        # Normalize legacy limits (single dict) into per-scope dict keyed by "total".
        if isinstance(limits, dict) and any(
          k in limits for k in ("baselineUsdt", "currentUsdt", "drawdownPct", "kill")
        ) and "total" not in limits:
          limits = {
            "total": {
              "day": limits.get("day"),
              "baselineUsdt": limits.get("baselineUsdt"),
              "currentUsdt": limits.get("currentUsdt"),
              "drawdownPct": limits.get("drawdownPct"),
              "kill": limits.get("kill", False),
              "reason": limits.get("reason", ""),
              "updated": limits.get("updated"),
            }
          }
        if not isinstance(limits, dict):
          limits = {}
        data["limits"] = limits
        data = self._prune(data)
        if data != raw:
          self._write(data)
        else:
          self._cache = copy.deepcopy(data)
          try:
            self._cache_mtime = self.path.stat().st_mtime
          except OSError:
            self._cache_mtime = None
        return data
    except Exception:
      return {"plans": [], "triggers": [], "coins": [], "trades": [], "limits": {}, "sentiments": [], "decisions": [], "fees": [], "supervisor_notes_temporary": [], "supervisor_notes_permanent": []}

  def _prune(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop entries older than retention_days."""
    now = int(time.time())
    cutoff = now - self.retention_days * 86400
    data["plans"] = [p for p in data.get("plans", []) if (p.get("ts") or now) >= cutoff]
    data["triggers"] = [t for t in data.get("triggers", []) if (t.get("ts") or now) >= cutoff]
    data["coins"] = [c for c in data.get("coins", []) if (c.get("ts") or now) >= cutoff]
    data["trades"] = [t for t in data.get("trades", []) if (t.get("ts") or now) >= cutoff]
    data.setdefault("limits", {})
    if isinstance(data["limits"], dict):
      for scope, lim in list(data["limits"].items()):
        if not isinstance(lim, dict):
          data["limits"].pop(scope, None)
          continue
        if (lim.get("updated") or now) < cutoff:
          data["limits"].pop(scope, None)
    data["sentiments"] = [s for s in data.get("sentiments", []) if (s.get("ts") or now) >= cutoff]
    data["decisions"] = [d for d in data.get("decisions", []) if (d.get("ts") or now) >= cutoff]
    data["fees"] = [f for f in data.get("fees", []) if (f.get("ts") or now) >= cutoff]
    data["supervisor_notes_temporary"] = [n for n in data.get("supervisor_notes_temporary", []) if (n.get("ts") or now) >= cutoff]
    # Permanent notes are exempt from the retention-days cutoff by design — they persist
    # until manually deleted. Only the count cap (MAX_PERMANENT_NOTES) applies below.
    data.setdefault("supervisor_notes_permanent", [])

    def _cap_list(key: str, max_items: int) -> None:
      items = data.get(key) or []
      if len(items) > max_items:
        # keep most recent by ts if available, else last items
        try:
          items = sorted(items, key=lambda x: x.get("ts", 0))[-max_items:]
        except Exception:
          items = items[-max_items:]
      data[key] = items

    _cap_list("plans", MAX_PLANS)
    _cap_list("triggers", MAX_TRIGGERS)
    _cap_list("coins", MAX_COINS)
    _cap_list("trades", MAX_TRADES)
    _cap_list("sentiments", MAX_SENTIMENTS)
    # Two-tier cap: trade outcomes (pnl != None) get a larger cap so losses are never
    # evicted by high-volume entry/decline decisions (pnl=None).
    pnl_decisions = [d for d in (data.get("decisions") or []) if d.get("pnl") is not None]
    null_decisions = [d for d in (data.get("decisions") or []) if d.get("pnl") is None]
    if len(pnl_decisions) > MAX_CLOSED_TRADES:
      try:
        pnl_decisions = sorted(pnl_decisions, key=lambda x: x.get("ts", 0))[-MAX_CLOSED_TRADES:]
      except Exception:
        pnl_decisions = pnl_decisions[-MAX_CLOSED_TRADES:]
    if len(null_decisions) > MAX_DECISIONS:
      try:
        null_decisions = sorted(null_decisions, key=lambda x: x.get("ts", 0))[-MAX_DECISIONS:]
      except Exception:
        null_decisions = null_decisions[-MAX_DECISIONS:]
    data["decisions"] = sorted(pnl_decisions + null_decisions, key=lambda x: x.get("ts", 0))
    _cap_list("fees", MAX_FEES)
    _cap_list("supervisor_notes_temporary", MAX_TEMPORARY_NOTES)
    _cap_list("supervisor_notes_permanent", MAX_PERMANENT_NOTES)

    return data

  def _write(self, data: Dict[str, Any]) -> None:
    self._cache = copy.deepcopy(data)
    self.path.write_text(json.dumps(data, indent=2))
    try:
      self._cache_mtime = self.path.stat().st_mtime
    except OSError:
      self._cache_mtime = None

  def save_plan(self, title: str, summary: str, actions: list[str], author: str | None = None) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      entry = {
        "title": title,
        "summary": summary,
        "actions": actions,
        "author": author,
        "ts": int(time.time()),
      }
      data.setdefault("plans", [])
      data["plans"].append(entry)
      self._write(data)
      return entry

  def latest_plan(self) -> Optional[Dict[str, Any]]:
    with self._lock:
      data = self._prune(self._read())
      plans = data.get("plans") or []
      return plans[-1] if plans else None

  def latest_items(self, kind: str, limit: int = 5) -> Dict[str, Any]:
    """Fetch the latest N entries for a given kind (plans/sentiments/decisions/trades/triggers/coins/fees)."""
    alias_map = {
      "plan": "plans",
      "plans": "plans",
      "research": "plans",
      "note": "plans",
      "notes": "plans",
      "sentiment": "sentiments",
      "sentiments": "sentiments",
      "decision": "decisions",
      "decisions": "decisions",
      "trade": "trades",
      "trades": "trades",
      "trigger": "triggers",
      "triggers": "triggers",
      "coin": "coins",
      "coins": "coins",
      "fee": "fees",
      "fees": "fees",
    }
    key = alias_map.get((kind or "").strip().lower())
    if not key:
      return {"error": f"unsupported kind '{kind}'", "allowed": sorted(set(alias_map.keys()))}
    try:
      lim = int(limit)
    except (TypeError, ValueError):
      lim = 5
    lim = min(max(lim, 1), 50)

    with self._lock:
      data = self._prune(self._read())
      items = data.get(key) or []
      if not isinstance(items, list):
        return {"error": f"kind '{kind}' unavailable"}
      latest = sorted(items, key=lambda x: x.get("ts", 0))[-lim:]
    return {"kind": key, "requested": lim, "items": list(reversed(latest))}

  def clear_plans(self) -> Dict[str, Any]:
    with self._lock:
      existing = self._read()
      data = {
        "plans": [],
        "triggers": [],
        "coins": existing.get("coins", []),
        "trades": existing.get("trades", []),
        "limits": existing.get("limits", {}),
        "sentiments": existing.get("sentiments", []),
        "decisions": existing.get("decisions", []),
      }
      self._write(self._prune(data))
      return self._prune(data)

  def save_trigger(
    self,
    symbol: str,
    direction: str,
    rationale: str,
    target_price: Optional[float] = None,
    stop_price: Optional[float] = None,
  ) -> Dict[str, Any]:
    with self._lock:
      data = self._read()
      entry = {
        "symbol": _normalize_symbol(symbol),
        "direction": direction,
        "rationale": rationale,
        "targetPrice": target_price,
        "stopPrice": stop_price,
        "ts": int(time.time()),
      }
      data.setdefault("triggers", [])
      data["triggers"].append(entry)
      self._write(data)
      return entry

  def latest_triggers(self) -> list[Dict[str, Any]]:
    with self._lock:
      data = self._prune(self._read())
      return data.get("triggers", []) or []

  def get_coins(self, default: list[str] | None = None) -> list[str]:
    with self._lock:
      data = self._prune(self._read())
      coins = data.get("coins", []) or []
      if coins:
        return [c["symbol"] for c in coins if c.get("status", "active") == "active"]
      return default or []

  def has_coins(self) -> bool:
    with self._lock:
      data = self._prune(self._read())
      coins = data.get("coins", []) or []
      return bool(coins)

  def set_coins(self, coins: list[str], reason: str = "update") -> list[Dict[str, Any]]:
    with self._lock:
      entries = []
      now = int(time.time())
      for sym in coins:
        entries.append({"symbol": _normalize_symbol(sym), "status": "active", "reason": reason, "ts": now})
      data = self._read()
      data["coins"] = entries
      self._write(data)
      return entries

  def add_coin(self, symbol: str, reason: str) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      coins = data.get("coins", [])
      now = int(time.time())
      symbol_up = _normalize_symbol(symbol)
      # avoid duplicates; replace existing with latest reason
      coins = [c for c in coins if c.get("symbol") != symbol_up]
      coins.append({"symbol": symbol_up, "status": "active", "reason": reason, "ts": now})
      data["coins"] = coins
      self._write(data)
      return {"symbol": symbol_up, "status": "active", "reason": reason, "ts": now}

  def remove_coin(self, symbol: str, reason: str, exit_plan: str) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      coins = data.get("coins", [])
      symbol_up = _normalize_symbol(symbol)
      now = int(time.time())
      coins = [c for c in coins if c.get("symbol") != symbol_up]
      entry = {
        "symbol": symbol_up,
        "status": "removed",
        "reason": reason,
        "exitPlan": exit_plan,
        "ts": now,
      }
      coins.append(entry)
      data["coins"] = coins
      self._write(data)
      return entry

  def trades_today(self, symbol: str) -> int:
    with self._lock:
      data = self._prune(self._read())
      day_key = int(time.time() // 86400)
      return len(
        [
          t
          for t in data.get("trades", []) or []
          if t.get("symbol") == _normalize_symbol(symbol) and (t.get("day") or 0) == day_key
        ]
      )

  def record_trade(
    self,
    symbol: str,
    side: str,
    notional_usd: float,
    paper: bool = False,
    price: float | None = None,
    size: float | None = None,
    venue: str = "spot",
  ) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      day_key = int(now // 86400)
      entry = {
        "symbol": _normalize_symbol(symbol),
        "side": side,
        "notionalUsd": notional_usd,
        "price": price,
        "size": size,
        "paper": paper,
        "venue": venue,
        "ts": now,
        "day": day_key,
      }
      data.setdefault("trades", [])
      data["trades"].append(entry)
      self._write(data)
      return entry

  def log_sentiment(self, symbol: str, score: float, rationale: str, source: str = "") -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      day_key = int(now // 86400)
      entry = {
        "symbol": _normalize_symbol(symbol),
        "score": float(score),
        "rationale": rationale,
        "source": source,
        "ts": now,
        "day": day_key,
      }
      data.setdefault("sentiments", [])
      data["sentiments"].append(entry)
      self._write(data)
      return entry

  def latest_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
    with self._lock:
      data = self._prune(self._read())
      sym = _normalize_symbol(symbol)
      sentiments = [s for s in data.get("sentiments", []) if s.get("symbol") == sym]
      return sentiments[-1] if sentiments else None

  def log_decision(
    self,
    symbol: str,
    action: str,
    confidence: float,
    reason: str,
    pnl: Optional[float] = None,
    paper: bool = False,
    peak_pnl: Optional[float] = None,
    trough_pnl: Optional[float] = None,
  ) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      day_key = int(now // 86400)
      sym = _normalize_symbol(symbol)
      if pnl is not None and (peak_pnl is None or trough_pnl is None):
        ext = data.get("position_extremes", {}).get(sym, {})
        if ext:
          if peak_pnl is None:
            peak_pnl = ext.get("peakPnl")
          if trough_pnl is None:
            trough_pnl = ext.get("troughPnl")
      entry: Dict[str, Any] = {
        "symbol": sym,
        "action": action,
        "confidence": float(confidence),
        "reason": reason,
        "pnl": pnl,
        "paper": paper,
        "ts": now,
        "day": day_key,
      }
      if peak_pnl is not None:
        entry["peakPnl"] = round(float(peak_pnl), 4)
      if trough_pnl is not None:
        entry["troughPnl"] = round(float(trough_pnl), 4)
      data.setdefault("decisions", [])
      data["decisions"].append(entry)
      self._write(data)
      return entry

  def trades_today_total(self) -> int:
    with self._lock:
      data = self._prune(self._read())
      day_key = int(time.time() // 86400)
      return len([t for t in data.get("trades", []) or [] if (t.get("day") or 0) == day_key])

  def update_limits(self, current_usdt: float, scope: str = "total") -> Dict[str, Any]:
    """Track daily drawdown percentage for informational context. No kill switch."""
    with self._lock:
      data = self._read()
      now = int(time.time())
      day_key = int(now // 86400)
      limits_all = data.get("limits") if isinstance(data.get("limits"), dict) else {}
      limits = limits_all.get(scope) or {}
      if limits.get("day") != day_key:
        limits = {
          "day": day_key,
          "baselineUsdt": float(current_usdt or 0.0),
          "currentUsdt": float(current_usdt or 0.0),
          "drawdownPct": 0.0,
          "updated": now,
        }
      baseline = limits.get("baselineUsdt") or float(current_usdt or 0.0)
      if baseline <= 0:
        baseline = float(current_usdt or 0.0)
      drawdown_pct = 0.0
      if baseline > 0:
        drawdown_pct = max(0.0, (baseline - float(current_usdt or 0.0)) / baseline * 100)
      limits.update(
        {
          "day": day_key,
          "baselineUsdt": baseline,
          "currentUsdt": float(current_usdt or 0.0),
          "drawdownPct": drawdown_pct,
          "updated": now,
        }
      )
      limits_all[scope] = limits
      data["limits"] = limits_all
      self._write(data)
      return limits

  def positions(self, prices: Dict[str, float] | None = None, venue: str | None = None) -> Dict[str, Any]:
    """Derive positions (avg entry, unrealized/realized PnL) from recorded trades. Filter by venue ('spot'/'futures') when set."""
    with self._lock:
      data = self._prune(self._read())
      trades = sorted(data.get("trades", []), key=lambda t: t.get("ts", 0))
    if venue:
      trades = [t for t in trades if t.get("venue", "spot") == venue]

    positions: Dict[str, Dict[str, float]] = {}

    def _avg(cost: float, qty: float) -> float:
      return cost / qty if qty else 0.0

    for t in trades:
      sym = t.get("symbol")
      side = (t.get("side") or "").lower()
      try:
        qty = float(t.get("size") or 0)
        price = float(t.get("price") or 0)
      except Exception:
        continue
      if not sym or qty <= 0 or price <= 0 or side not in {"buy", "sell"}:
        continue
      pos = positions.setdefault(sym, {"netSize": 0.0, "cost": 0.0, "realizedPnl": 0.0, "venue": t.get("venue", "spot"), "lastTs": t.get("ts", 0)})
      net = pos["netSize"]
      cost = pos["cost"]
      realized = pos["realizedPnl"]
      ts = t.get("ts", 0)
      if side == "buy":
        if net < 0:
          close_amt = min(qty, -net)
          avg_entry = _avg(cost, net) if net != 0 else 0.0
          realized += (avg_entry - price) * close_amt
          net += close_amt
          cost += avg_entry * close_amt
          qty -= close_amt
        if qty > 0:
          net += qty
          cost += price * qty
      else:  # sell
        if net > 0:
          close_amt = min(qty, net)
          avg_entry = _avg(cost, net) if net != 0 else 0.0
          realized += (price - avg_entry) * close_amt
          net -= close_amt
          cost -= avg_entry * close_amt
          qty -= close_amt
        if qty > 0:
          net -= qty
          cost -= price * qty
      pos["netSize"] = net
      pos["cost"] = cost
      pos["realizedPnl"] = realized
      pos["lastTs"] = max(pos.get("lastTs", 0), ts)

    # Attach derived values, unrealized PnL, and peak/trough extremes when prices are supplied.
    prices = prices or {}
    with self._lock:
      extremes = self._read().get("position_extremes", {})
    for sym, pos in positions.items():
      net = pos.get("netSize", 0.0)
      cost = pos.get("cost", 0.0)
      cur_price = float(prices.get(sym) or 0.0)
      avg_entry = _avg(cost, net) if net else None
      unrealized = None
      if net and cur_price > 0:
        unrealized = (cur_price - avg_entry) * net
      pos["avgEntry"] = avg_entry
      pos["unrealizedPnl"] = unrealized
      pos["currentPrice"] = cur_price or None
      ext = extremes.get(sym, {})
      if ext:
        pos["peakPnl"] = ext.get("peakPnl")
        pos["troughPnl"] = ext.get("troughPnl")
    return positions

  def update_position_extremes(self, positions: Dict[str, Dict[str, Any]]) -> None:
    """Update peak/trough unrealized PnL for open positions. Call each poll round."""
    with self._lock:
      data = self._read()
      extremes = data.get("position_extremes", {})
      now = int(time.time())
      active_symbols: set[str] = set()
      for sym, pos in positions.items():
        net = pos.get("netSize") or 0
        upnl = pos.get("unrealizedPnl")
        if not net or upnl is None:
          continue
        active_symbols.add(sym)
        ext = extremes.get(sym)
        if ext is None:
          extremes[sym] = {"peakPnl": upnl, "troughPnl": upnl, "peakTs": now, "troughTs": now, "openTs": now}
        else:
          if upnl > (ext.get("peakPnl") or float("-inf")):
            ext["peakPnl"] = upnl
            ext["peakTs"] = now
          if upnl < (ext.get("troughPnl") or float("inf")):
            ext["troughPnl"] = upnl
            ext["troughTs"] = now
      for sym in list(extremes.keys()):
        if sym not in active_symbols:
          del extremes[sym]
      data["position_extremes"] = extremes
      self._write(data)

  def get_position_extremes(self, symbol: str | None = None) -> Dict[str, Any]:
    """Get peak/trough PnL extremes for open positions (or a specific symbol)."""
    with self._lock:
      data = self._read()
      extremes = data.get("position_extremes", {})
    if symbol:
      return extremes.get(_normalize_symbol(symbol), {})
    return extremes

  @staticmethod
  def _is_realized_close(action: str) -> bool:
    """Return True if the action represents a realized trade close, not a hold/manage snapshot."""
    a = action.lower()
    return (
      "triggered" in a
      or "cut" in a
      or "close" in a
      or "reduce" in a
      or a in ("futures_sell", "futures_buy", "spot_sell")
    )

  @staticmethod
  def _pnl_stats(decisions: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute win/loss stats from a list of decision dicts (realized closes only)."""
    realized: list[float] = []
    missed_profits: list[float] = []
    unnecessary_losses: list[float] = []
    for d in decisions:
      if not MemoryStore._is_realized_close(d.get("action") or ""):
        continue
      pnl = d.get("pnl")
      if pnl is not None:
        try:
          pnl_f = float(pnl)
        except (TypeError, ValueError):
          continue
        realized.append(pnl_f)
        peak = d.get("peakPnl")
        trough = d.get("troughPnl")
        if peak is not None:
          try:
            peak_f = float(peak)
          except (TypeError, ValueError):
            peak_f = None
        else:
          peak_f = None
        if peak_f is not None and peak_f > pnl_f and peak_f > 0:
          missed_profits.append(round(peak_f - pnl_f, 4))
        if trough is not None and pnl_f <= 0:
          try:
            trough_f = float(trough)
          except (TypeError, ValueError):
            trough_f = None
          if trough_f is not None and trough_f < pnl_f:
            unnecessary_losses.append(round(pnl_f - trough_f, 4))
    wins = [p for p in realized if p > 0]
    losses = [p for p in realized if p < 0]
    total = sum(realized)
    wr = len(wins) / len(realized) if realized else 0.0
    stats: Dict[str, Any] = {
      "closedWithPnl": len(realized),
      "wins": len(wins),
      "losses": len(losses),
      "winRate": round(wr, 3),
      "totalRealizedPnl": round(total, 4),
      "avgWin": round(sum(wins) / len(wins), 4) if wins else 0.0,
      "avgLoss": round(sum(losses) / len(losses), 4) if losses else 0.0,
      "bestTrade": round(max(wins), 4) if wins else 0.0,
      "worstTrade": round(min(losses), 4) if losses else 0.0,
    }
    if missed_profits:
      stats["missedProfitCount"] = len(missed_profits)
      stats["totalMissedProfit"] = round(sum(missed_profits), 4)
      stats["avgMissedProfit"] = round(sum(missed_profits) / len(missed_profits), 4)
    if unnecessary_losses:
      stats["unnecessaryLossCount"] = len(unnecessary_losses)
      stats["totalUnnecessaryLoss"] = round(sum(unnecessary_losses), 4)
    return stats

  def performance_summary(self) -> Dict[str, Any]:
    """Compute win/loss stats from recorded decisions, split by venue and paper/live."""
    with self._lock:
      data = self._prune(self._read())
      trades = data.get("trades", [])
      decisions = data.get("decisions", [])

    if not trades:
      return {"totalTrades": 0, "message": "No trade history yet."}

    overall = self._pnl_stats(decisions)
    overall["totalTrades"] = len(trades)

    spot_decisions = [d for d in decisions if (d.get("action") or "").startswith("spot_")]
    futures_decisions = [d for d in decisions if (d.get("action") or "").startswith("futures_")]
    spot_trades = [t for t in trades if t.get("venue", "spot") == "spot"]
    futures_trades = [t for t in trades if t.get("venue", "spot") == "futures"]

    def _venue_block(venue_decisions: list, venue_trades: list) -> Dict[str, Any]:
      block = self._pnl_stats(venue_decisions)
      block["totalTrades"] = len(venue_trades)
      live_d = [d for d in venue_decisions if not d.get("paper", False)]
      paper_d = [d for d in venue_decisions if d.get("paper", False)]
      live_t = [t for t in venue_trades if not t.get("paper", False)]
      paper_t = [t for t in venue_trades if t.get("paper", False)]
      live_stats = self._pnl_stats(live_d)
      live_stats["totalTrades"] = len(live_t)
      paper_stats = self._pnl_stats(paper_d)
      paper_stats["totalTrades"] = len(paper_t)
      block["live"] = live_stats
      block["paper"] = paper_stats
      return block

    overall["spot"] = _venue_block(spot_decisions, spot_trades)
    overall["futures"] = _venue_block(futures_decisions, futures_trades)
    return overall

  def reset_limits(self, current_usdt: float, scope: str = "total") -> Dict[str, Any]:
    """Reset daily drawdown baseline to current_usdt."""
    with self._lock:
      data = self._read()
      now = int(time.time())
      day_key = int(now // 86400)
      limits_all = data.get("limits") if isinstance(data.get("limits"), dict) else {}
      limits = {
        "day": day_key,
        "baselineUsdt": float(current_usdt or 0.0),
        "currentUsdt": float(current_usdt or 0.0),
        "drawdownPct": 0.0,
        "updated": now,
      }
      limits_all[scope] = limits
      data["limits"] = limits_all
      self._write(data)
      return limits

  def save_fee_info(self, spot_taker: float, spot_maker: float, futures_taker: float | None = None, futures_maker: float | None = None) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      entry = {
        "spot_taker": float(spot_taker),
        "spot_maker": float(spot_maker),
        "futures_taker": float(futures_taker) if futures_taker is not None else None,
        "futures_maker": float(futures_maker) if futures_maker is not None else None,
        "ts": now,
      }
      data.setdefault("fees", [])
      data["fees"].append(entry)
      self._write(data)
      return entry

  def latest_fees(self) -> Optional[Dict[str, Any]]:
    with self._lock:
      data = self._prune(self._read())
      fees = data.get("fees") or []
      return fees[-1] if fees else None

  def add_temporary_note(self, content: str, author: str = "Supervisor") -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      entry = {"content": content, "author": author, "ts": int(time.time())}
      data.setdefault("supervisor_notes_temporary", [])
      data["supervisor_notes_temporary"].append(entry)
      self._write(data)
      return entry

  def add_permanent_note(self, content: str, author: str = "Supervisor") -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      entry = {"content": content, "author": author, "ts": int(time.time())}
      data.setdefault("supervisor_notes_permanent", [])
      data["supervisor_notes_permanent"].append(entry)
      self._write(data)
      return entry

  def consume_temporary_notes(self) -> list[Dict[str, Any]]:
    """Return all temporary notes and delete them atomically."""
    with self._lock:
      data = self._read()
      notes = list(data.get("supervisor_notes_temporary") or [])
      if notes:
        data["supervisor_notes_temporary"] = []
        self._write(data)
      return notes

  def get_permanent_notes(self) -> list[Dict[str, Any]]:
    with self._lock:
      data = self._read()
      return list(data.get("supervisor_notes_permanent") or [])

  def list_all_notes(self) -> Dict[str, Any]:
    with self._lock:
      data = self._read()
      return {
        "temporary": list(data.get("supervisor_notes_temporary") or []),
        "permanent": list(data.get("supervisor_notes_permanent") or []),
      }

  def delete_permanent_note(self, index: int) -> Dict[str, Any]:
    with self._lock:
      data = self._read()
      notes = data.get("supervisor_notes_permanent") or []
      if index < 0 or index >= len(notes):
        return {"error": f"Invalid index {index}; {len(notes)} permanent notes exist"}
      removed = notes.pop(index)
      data["supervisor_notes_permanent"] = notes
      self._write(data)
      return {"deleted": removed}

  def kelly_fraction(self, venue: str | None = None, lookback: int = 50) -> float:
    """Compute quarter-Kelly fraction from recent trade performance.
    Returns a sizing fraction between 0.01 and 0.25."""
    with self._lock:
      data = self._prune(self._read())
      decisions = data.get("decisions", [])
    if venue:
      prefix = f"{venue}_"
      decisions = [d for d in decisions if (d.get("action") or "").startswith(prefix)]
    decisions = sorted(decisions, key=lambda d: d.get("ts", 0))[-lookback:]
    realized = []
    for d in decisions:
      pnl = d.get("pnl")
      if pnl is not None:
        try:
          realized.append(float(pnl))
        except (TypeError, ValueError):
          continue
    if len(realized) < 10:
      return 0.05
    wins = [p for p in realized if p > 0]
    losses = [p for p in realized if p < 0]
    if not wins or not losses:
      return 0.05
    win_rate = len(wins) / len(realized)
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    if avg_loss == 0:
      return 0.25
    reward_risk = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / reward_risk
    return max(0.01, min(0.25, kelly * 0.25))

  def consecutive_losses(self, venue: str | None = None) -> int:
    """Count current streak of consecutive losing CLOSED trades (most recent first).

    Only counts decisions that represent realized closes (TP/SL triggered or
    explicit close orders). Excludes 'manage', 'hold', 'decline' decisions
    whose pnl field reflects unrealized state, not a realized loss.
    """
    with self._lock:
      data = self._prune(self._read())
      decisions = data.get("decisions", [])
    if venue:
      prefix = f"{venue}_"
      decisions = [d for d in decisions if (d.get("action") or "").startswith(prefix)]
    decisions = sorted(decisions, key=lambda d: d.get("ts", 0), reverse=True)
    streak = 0
    for d in decisions:
      action = (d.get("action") or "").lower()
      # Only count realized closes — skip manage/hold/decline regardless of pnl
      if not MemoryStore._is_realized_close(action):
        continue
      pnl = d.get("pnl")
      if pnl is None:
        continue
      try:
        if float(pnl) < 0:
          streak += 1
        else:
          break
      except (TypeError, ValueError):
        continue
    return streak

  def last_trade_time(self, symbol: str) -> int | None:
    """Return the timestamp of the most recent trade (any side) for a symbol, or None."""
    sym = _normalize_symbol(symbol)
    with self._lock:
      data = self._prune(self._read())
      trades = data.get("trades", [])
    for t in sorted(trades, key=lambda t: t.get("ts", 0), reverse=True):
      if t.get("symbol") == sym:
        return t.get("ts")
    return None

  def last_loss_time(self, symbol: str) -> int | None:
    """Return the timestamp of the most recent realized losing close for a symbol, or None.

    Only counts realized closes (triggered TP/SL or explicit close orders),
    not 'manage'/'hold' decisions whose pnl reflects unrealized state.
    """
    sym = _normalize_symbol(symbol)
    with self._lock:
      data = self._prune(self._read())
      decisions = data.get("decisions", [])
    for d in sorted(decisions, key=lambda d: d.get("ts", 0), reverse=True):
      if d.get("symbol") != sym:
        continue
      action = (d.get("action") or "").lower()
      if not MemoryStore._is_realized_close(action):
        continue
      pnl = d.get("pnl")
      if pnl is None:
        continue
      try:
        if float(pnl) < 0:
          return d.get("ts")
        else:
          return None
      except (TypeError, ValueError):
        continue
    return None

  def portfolio_heat(self, stop_distances: Dict[str, float], total_equity: float) -> float:
    """Compute portfolio heat = sum of capital at risk / total equity * 100.
    stop_distances: {symbol: usd_amount_at_risk}"""
    if total_equity <= 0:
      return 0.0
    total_risk = sum(abs(v) for v in stop_distances.values())
    return round(total_risk / total_equity * 100, 2)
