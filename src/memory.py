from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


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
      return {"plans": [], "triggers": [], "coins": []}
    try:
      data = json.loads(self.path.read_text())
      if isinstance(data, dict):
        data.setdefault("plans", [])
        data.setdefault("triggers", [])
        data.setdefault("coins", [])
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
        return self._prune(data)
    except Exception:
      return {"plans": [], "triggers": [], "coins": []}
    return {"plans": [], "triggers": [], "coins": []}

  def _prune(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop entries older than retention_days."""
    now = int(time.time())
    cutoff = now - self.retention_days * 86400
    data["plans"] = [p for p in data.get("plans", []) if (p.get("ts") or now) >= cutoff]
    data["triggers"] = [t for t in data.get("triggers", []) if (t.get("ts") or now) >= cutoff]
    data["coins"] = [c for c in data.get("coins", []) if (c.get("ts") or now) >= cutoff]
    return data

  def _write(self, data: Dict[str, Any]) -> None:
    self.path.write_text(json.dumps(data, indent=2))

  def save_plan(self, title: str, summary: str, actions: list[str]) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      entry = {
        "title": title,
        "summary": summary,
        "actions": actions,
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

  def clear_plans(self) -> Dict[str, Any]:
    with self._lock:
      data = {"plans": [], "triggers": [], "coins": self._read().get("coins", [])}
      self._write(data)
      return data

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
