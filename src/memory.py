from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class MemoryStore:
  """Lightweight JSON-backed store for agent plans/notes."""

  def __init__(self, path: str) -> None:
    self.path = Path(path)
    self._lock = threading.Lock()

  def _read(self) -> Dict[str, Any]:
    if not self.path.exists():
      return {"plans": [], "triggers": []}
    try:
      data = json.loads(self.path.read_text())
      if isinstance(data, dict):
        data.setdefault("plans", [])
        data.setdefault("triggers", [])
        return data
    except Exception:
      return {"plans": [], "triggers": []}
    return {"plans": [], "triggers": []}

  def _write(self, data: Dict[str, Any]) -> None:
    self.path.write_text(json.dumps(data, indent=2))

  def save_plan(self, title: str, summary: str, actions: list[str]) -> Dict[str, Any]:
    with self._lock:
      data = self._read()
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
      data = self._read()
      plans = data.get("plans") or []
      return plans[-1] if plans else None

  def clear_plans(self) -> Dict[str, Any]:
    with self._lock:
      data = {"plans": [], "triggers": []}
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
      data = self._read()
      return data.get("triggers", []) or []
