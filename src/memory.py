from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Hard caps to keep the memory file small and readable (agent only needs recent history).
MAX_PLANS = 3
MAX_TRIGGERS = 5
MAX_COINS = 50
MAX_TRADES = 10
MAX_SENTIMENTS = 10
MAX_DECISIONS = 10
MAX_FEES = 3


class MemoryStore:
  """Lightweight JSON-backed store for agent plans/notes."""

  def __init__(self, path: str, retention_days: int = 7) -> None:
    self.path = Path(path)
    self._lock = threading.Lock()
    self.retention_days = retention_days
    # Sanitize on init.
    self._read()

  def _read(self) -> Dict[str, Any]:
    if not self.path.exists():
      return {"plans": [], "triggers": [], "coins": [], "trades": [], "limits": {}, "sentiments": [], "decisions": [], "fees": []}
    try:
      data = json.loads(self.path.read_text())
      if isinstance(data, dict):
        data.setdefault("plans", [])
        data.setdefault("triggers", [])
        data.setdefault("coins", [])
        data.setdefault("trades", [])
        data.setdefault("limits", {})
        data.setdefault("sentiments", [])
        data.setdefault("decisions", [])
        data.setdefault("fees", [])
        # prune invalid entries while keeping timestamp
        data["plans"] = [
          p
          for p in data.get("plans", [])
          if isinstance(p, dict) and p.get("title") and p.get("summary") and isinstance(p.get("actions"), list)
        ]
        data["triggers"] = [
          t
          for t in data.get("triggers", [])
          if isinstance(t, dict) and t.get("symbol") and t.get("direction")
        ]
        data["coins"] = [
          {
            "symbol": c.get("symbol", "").upper(),
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
            "symbol": t.get("symbol", "").upper(),
            "side": t.get("side"),
            "notionalUsd": t.get("notionalUsd"),
            "price": t.get("price"),
            "size": t.get("size"),
            "paper": t.get("paper", False),
            "ts": t.get("ts") or int(time.time()),
            "day": t.get("day"),
          }
          for t in data.get("trades", [])
          if isinstance(t, dict) and t.get("symbol") and t.get("ts")
        ]
        data["sentiments"] = [
          {
            "symbol": s.get("symbol", "").upper(),
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
            "symbol": d.get("symbol", "").upper(),
            "action": d.get("action"),
            "confidence": d.get("confidence"),
            "reason": d.get("reason", ""),
            "pnl": d.get("pnl"),
            "paper": d.get("paper", False),
            "ts": d.get("ts") or int(time.time()),
            "day": d.get("day"),
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
        return self._prune(data)
    except Exception:
      return {"plans": [], "triggers": [], "coins": [], "trades": [], "limits": {}}
    return {"plans": [], "triggers": [], "coins": [], "trades": [], "limits": {}, "fees": []}

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
    _cap_list("decisions", MAX_DECISIONS)
    _cap_list("fees", MAX_FEES)

    return data

  def _write(self, data: Dict[str, Any]) -> None:
    self.path.write_text(json.dumps(data, indent=2))

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
        "symbol": symbol,
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
        entries.append({"symbol": sym.upper(), "status": "active", "reason": reason, "ts": now})
      data = self._read()
      data["coins"] = entries
      self._write(data)
      return entries

  def add_coin(self, symbol: str, reason: str) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      coins = data.get("coins", [])
      now = int(time.time())
      symbol_up = symbol.upper()
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
      symbol_up = symbol.upper()
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
          if t.get("symbol") == symbol.upper() and (t.get("day") or 0) == day_key
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
  ) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      day_key = int(now // 86400)
      entry = {
        "symbol": symbol.upper(),
        "side": side,
        "notionalUsd": notional_usd,
        "price": price,
        "size": size,
        "paper": paper,
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
        "symbol": symbol.upper(),
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
      sym = symbol.upper()
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
  ) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      day_key = int(now // 86400)
      entry = {
        "symbol": symbol.upper(),
        "action": action,
        "confidence": float(confidence),
        "reason": reason,
        "pnl": pnl,
        "paper": paper,
        "ts": now,
        "day": day_key,
      }
      data.setdefault("decisions", [])
      data["decisions"].append(entry)
      self._write(data)
      return entry

  def trades_today_total(self) -> int:
    with self._lock:
      data = self._prune(self._read())
      day_key = int(time.time() // 86400)
      return len([t for t in data.get("trades", []) or [] if (t.get("day") or 0) == day_key])

  def update_limits(self, current_usdt: float, drawdown_limit_pct: float, scope: str = "total") -> Dict[str, Any]:
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
          "kill": False,
          "reason": "",
          "updated": now,
        }
      baseline = limits.get("baselineUsdt") or float(current_usdt or 0.0)
      if baseline <= 0:
        baseline = float(current_usdt or 0.0)
      drawdown_pct = 0.0
      if baseline > 0:
        drawdown_pct = max(0.0, (baseline - float(current_usdt or 0.0)) / baseline * 100)

      if not limits.get("kill") and drawdown_limit_pct > 0 and drawdown_pct >= drawdown_limit_pct:
        limits["kill"] = True
        limits["reason"] = f"Daily drawdown {drawdown_pct:.2f}% >= limit {drawdown_limit_pct}%"

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

  def positions(self, prices: Dict[str, float] | None = None) -> Dict[str, Any]:
    """Derive positions (avg entry, unrealized/realized PnL) from recorded trades."""
    with self._lock:
      data = self._prune(self._read())
      trades = sorted(data.get("trades", []), key=lambda t: t.get("ts", 0))

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
      pos = positions.setdefault(sym, {"netSize": 0.0, "cost": 0.0, "realizedPnl": 0.0, "lastTs": t.get("ts", 0)})
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

    # Attach derived values and unrealized PnL when prices are supplied.
    prices = prices or {}
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
    return positions

  def kill_active(self, scope: str = "total") -> bool:
    with self._lock:
      data = self._read()
      limits_all = data.get("limits") or {}
      limits = limits_all.get(scope) or {}
      return bool(limits.get("kill"))

  def reset_limits(self, current_usdt: float, scope: str = "total") -> Dict[str, Any]:
    """Reset daily limits baseline to current_usdt and clear kill flag."""
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
        "kill": False,
        "reason": "manual reset",
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
