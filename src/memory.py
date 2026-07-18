from __future__ import annotations

import copy
import json
import logging
import os
import re
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
MAX_HANDOFF_DECISIONS = 30  # agent handoff markers (pnl=None) — kept in their own bucket so the
                            # high-volume decline/hold snapshots don't evict them from the feed
MAX_CLOSED_TRADES = 200  # closed-trade outcomes (pnl != None) — kept separately
MAX_FEES = 3
MAX_TEMPORARY_NOTES = 20
MAX_PERMANENT_NOTES = 10
MAX_PENDING_AGENT_EVENTS = 200
MAX_AGENT_SCHEDULER_SYMBOLS = 100
_ATR_QUARANTINE_RE = re.compile(
  r"daily ATR\s+([0-9]+(?:\.[0-9]+)?)%\s+exceeds\s+([0-9]+(?:\.[0-9]+)?)%",
  re.IGNORECASE,
)


def _sanitize_agent_scheduler(value: Any) -> Dict[str, Any]:
  """Normalize the small restart-safe state used to throttle discretionary model calls."""
  raw = value if isinstance(value, dict) else {}

  try:
    last_run_ts = max(0.0, float(raw.get("lastRunTs") or 0.0))
  except (TypeError, ValueError):
    last_run_ts = 0.0
  try:
    unproductive_runs = min(100, max(0, int(raw.get("unproductiveRuns") or 0)))
  except (TypeError, ValueError):
    unproductive_runs = 0

  reviewed_prices: Dict[str, float] = {}
  for symbol, price in (raw.get("reviewedPrices") or {}).items() if isinstance(raw.get("reviewedPrices"), dict) else []:
    try:
      normalized = _normalize_symbol(str(symbol))
      numeric = float(price)
    except (TypeError, ValueError):
      continue
    if normalized and numeric > 0:
      reviewed_prices[normalized] = numeric

  observations: Dict[str, Dict[str, Any]] = {}
  raw_observations = raw.get("priceObservations") or {}
  if isinstance(raw_observations, dict):
    for symbol, observation in raw_observations.items():
      if not isinstance(observation, dict):
        continue
      try:
        normalized = _normalize_symbol(str(symbol))
        last_price = float(observation.get("lastPrice") or 0.0)
        noise = max(0.0, float(observation.get("noiseEwmaPct") or 0.0))
        samples = max(0, int(observation.get("samples") or 0))
        updated = max(0, int(observation.get("updated") or 0))
      except (TypeError, ValueError):
        continue
      if normalized and last_price > 0:
        observations[normalized] = {
          "lastPrice": last_price,
          "noiseEwmaPct": noise,
          "samples": samples,
          "updated": updated,
        }

  # Keep the newest bounded set if research rotated through many temporary candidates.
  observations = dict(
    sorted(observations.items(), key=lambda item: item[1].get("updated", 0))[-MAX_AGENT_SCHEDULER_SYMBOLS:]
  )
  if len(reviewed_prices) > MAX_AGENT_SCHEDULER_SYMBOLS:
    reviewed_prices = {
      symbol: price
      for symbol, price in reviewed_prices.items()
      if symbol in observations
    }
    reviewed_prices = dict(list(reviewed_prices.items())[-MAX_AGENT_SCHEDULER_SYMBOLS:])

  return {
    "lastRunTs": last_run_ts,
    "unproductiveRuns": unproductive_runs,
    "reviewedPrices": reviewed_prices,
    "priceObservations": observations,
  }


def _adaptive_quarantine_seconds(reason: str) -> int:
  """Back off unsafe candidates in proportion to how far their volatility exceeded the gate."""
  text = str(reason or "")
  if "automatic risk quarantine" not in text.lower():
    return 0
  match = _ATR_QUARANTINE_RE.search(text)
  if not match:
    return 24 * 3600
  observed = float(match.group(1))
  limit = max(1e-9, float(match.group(2)))
  excess_ratio = max(1.0, observed / limit)
  # A marginal breach rests for roughly half a day; severe/data-scale discontinuities rest up to
  # one week. The retry time adapts to the evidence and expires automatically without maintenance.
  hours = min(7 * 24.0, max(12.0, 12.0 * excess_ratio * excess_ratio))
  return int(hours * 3600)


class MemoryStore:
  """Lightweight JSON-backed store for agent plans/notes."""

  _path_locks_guard = threading.Lock()
  _path_locks: Dict[str, threading.Lock] = {}

  def __init__(self, path: str, retention_days: int = 7) -> None:
    self.path = Path(path)
    lock_key = str(self.path.expanduser().resolve())
    with self._path_locks_guard:
      self._lock = self._path_locks.setdefault(lock_key, threading.Lock())
    self.retention_days = retention_days
    self._cache: Dict[str, Any] | None = None
    self._cache_mtime: float | None = None
    # Sanitize on init.
    self._read()

  def _read(self) -> Dict[str, Any]:
    _empty: Dict[str, Any] = {"plans": [], "triggers": [], "coins": [], "trades": [], "limits": {}, "sentiments": [], "decisions": [], "fees": [], "supervisor_notes_temporary": [], "supervisor_notes_permanent": [], "position_extremes": {}, "seen_close_ids": [], "seen_fill_ids": [], "open_interest_observations": {}, "pending_agent_events": [], "agent_scheduler": _sanitize_agent_scheduler({})}
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
        data.setdefault("seen_close_ids", [])
        data.setdefault("seen_fill_ids", [])
        data.setdefault("open_interest_observations", {})
        data.setdefault("pending_agent_events", [])
        data["agent_scheduler"] = _sanitize_agent_scheduler(data.get("agent_scheduler"))
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
            "filled": t.get("filled", True),
            "trackPosition": t.get("trackPosition", True),
            "orderId": t.get("orderId"),
            "clientOid": t.get("clientOid"),
            "ts": t.get("ts") or int(time.time()),
            "day": t.get("day"),
          }
          for t in data.get("trades", [])
          if isinstance(t, dict) and t.get("symbol") and t.get("ts")
        ]
        data["pending_agent_events"] = [
          {
            "id": str(event.get("id") or ""),
            "kind": str(event.get("kind") or ""),
            "payload": event.get("payload"),
            "ts": int(event.get("ts") or time.time()),
          }
          for event in data.get("pending_agent_events", [])
          if isinstance(event, dict)
          and event.get("id")
          and event.get("kind") in {"spot_fills", "futures_fills", "closed_positions"}
          and isinstance(event.get("payload"), dict)
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
            # These fields power the no-chase guard and close deduplication.  Dropping them while
            # sanitizing on startup silently disabled both protections after every process restart.
            "exitPrice": d.get("exitPrice"),
            "closeType": d.get("closeType"),
            "positionId": d.get("positionId"),
            "positionOpenTime": d.get("positionOpenTime"),
            "positionSide": d.get("positionSide"),
            "positionLifecycleVersion": d.get("positionLifecycleVersion"),
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
      return _empty
    except Exception:
      return _empty

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
    data["pending_agent_events"] = [
      event for event in data.get("pending_agent_events", [])
      if isinstance(event, dict) and (event.get("ts") or now) >= cutoff
    ][-MAX_PENDING_AGENT_EVENTS:]
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
    # Two-tier cap: only actual close actions get the larger outcome bucket. The model often logs
    # unrealized PnL on hold/decline rows; treating those as closes crowded real outcomes out of
    # memory and made diagnostics misleading.
    def _is_handoff(d: Dict[str, Any]) -> bool:
      return str(d.get("action") or "").startswith("handoff")

    pnl_decisions = [
      d for d in (data.get("decisions") or [])
      if d.get("pnl") is not None and self._is_realized_close(str(d.get("action") or ""))
    ]
    null_all = [d for d in (data.get("decisions") or []) if d not in pnl_decisions]
    # Handoff markers get their own bucket so the high-volume decline/hold snapshots (capped at
    # MAX_DECISIONS) can't evict them — the dashboard surfaces handoffs from the recent feed.
    handoff_decisions = [d for d in null_all if _is_handoff(d)]
    null_decisions = [d for d in null_all if not _is_handoff(d)]
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
    if len(handoff_decisions) > MAX_HANDOFF_DECISIONS:
      try:
        handoff_decisions = sorted(handoff_decisions, key=lambda x: x.get("ts", 0))[-MAX_HANDOFF_DECISIONS:]
      except Exception:
        handoff_decisions = handoff_decisions[-MAX_HANDOFF_DECISIONS:]
    data["decisions"] = sorted(pnl_decisions + null_decisions + handoff_decisions, key=lambda x: x.get("ts", 0))
    _cap_list("fees", MAX_FEES)
    _cap_list("supervisor_notes_temporary", MAX_TEMPORARY_NOTES)
    _cap_list("supervisor_notes_permanent", MAX_PERMANENT_NOTES)

    return data

  def _write(self, data: Dict[str, Any]) -> None:
    payload = json.dumps(data, indent=2)
    temp_path = self.path.with_name(self.path.name + ".tmp")
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open("w", encoding="utf-8") as handle:
      handle.write(payload)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temp_path, self.path)
    self._cache = copy.deepcopy(data)
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
      data = self._read()
      data["plans"] = []
      data["triggers"] = []
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

  def get_quarantined_coins(self, now: int | None = None) -> list[Dict[str, Any]]:
    """Return automatic risk quarantines whose adaptive retry window has not expired."""
    current = int(time.time() if now is None else now)
    with self._lock:
      data = self._prune(self._read())
      quarantined: list[Dict[str, Any]] = []
      for coin in data.get("coins", []) or []:
        if coin.get("status") != "removed":
          continue
        reason = str(coin.get("reason") or "")
        duration = _adaptive_quarantine_seconds(reason)
        if duration <= 0:
          continue
        removed_at = int(coin.get("ts") or 0)
        retry_after = removed_at + duration
        if retry_after <= current:
          continue
        quarantined.append({
          "symbol": coin.get("symbol"),
          "reason": reason,
          "retryAfter": retry_after,
          "remainingHours": round((retry_after - current) / 3600.0, 1),
        })
      return sorted(quarantined, key=lambda item: item["retryAfter"])

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
    filled: bool = True,
    track_position: bool = True,
    order_id: str | None = None,
    client_oid: str | None = None,
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
        "filled": bool(filled),
        "trackPosition": bool(track_position),
        "orderId": str(order_id) if order_id not in (None, "") else None,
        "clientOid": str(client_oid) if client_oid not in (None, "") else None,
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
    exit_price: Optional[float] = None,
    close_type: Optional[str] = None,
    position_id: Optional[str] = None,
    position_open_time: Optional[int | float | str] = None,
    position_side: Optional[str] = None,
    entry_price: Optional[float] = None,
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
      if exit_price is not None:
        try:
          entry["exitPrice"] = float(exit_price)
        except (TypeError, ValueError):
          pass
      if entry_price is not None:
        try:
          entry["entryPrice"] = float(entry_price)
        except (TypeError, ValueError):
          pass
      if close_type:
        entry["closeType"] = str(close_type)
      if position_id not in (None, ""):
        entry["positionId"] = str(position_id)
      if position_open_time not in (None, ""):
        try:
          entry["positionOpenTime"] = int(float(position_open_time))
        except (TypeError, ValueError):
          pass
      if position_side and str(position_side).lower() in {"long", "short"}:
        entry["positionSide"] = str(position_side).lower()
      if any(value not in (None, "") for value in (position_id, position_open_time, position_side)):
        # Marks rows written by lifecycle-aware code. If an exchange omits the identifiers, fail
        # safe by keeping both PnL rows instead of falling back to an unsafe symbol/time merge.
        entry["positionLifecycleVersion"] = 1
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
      if t.get("filled") is False or t.get("trackPosition") is False:
        continue
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
        lifecycle_key = f"{pos.get('positionOpenTime') or ''}:{pos.get('positionSide') or ('long' if net > 0 else 'short')}"
        if ext is None or ext.get("lifecycleKey") != lifecycle_key:
          extremes[sym] = {
            "peakPnl": upnl, "troughPnl": upnl, "peakTs": now, "troughTs": now,
            "openTs": now, "lifecycleKey": lifecycle_key,
            "positionOpenTime": pos.get("positionOpenTime"),
            "positionSide": pos.get("positionSide") or ("long" if net > 0 else "short"),
          }
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
  def _dedupe_realized(closes: list[Dict[str, Any]], window_sec: int = 1800) -> list[Dict[str, Any]]:
    """Drop re-recorded duplicates, preferring lifecycle identity when it is available.

    The main loop's seen-ID set used to live only in process memory, so a restart within the
    30-min fill lookback re-recorded the same close (observed: ETH -0.5007 at 13:48 and again
    at 14:00 across a restart). IDs are persisted now; this read-time filter also heals data
    recorded before the fix so rolling stats aren't skewed by phantom losses.
    """
    kept: list[Dict[str, Any]] = []
    last_seen: Dict[tuple, int] = {}
    for c in sorted(closes, key=lambda d: d.get("ts") or 0):
      pnl = c.get("pnl")
      position_id = str(c.get("positionId") or "").strip()
      position_open_time = MemoryStore._position_open_time_ms(c)
      if position_id:
        key = (c.get("symbol"), "position-id", position_id)
      elif position_open_time is not None:
        key = (c.get("symbol"), "position-open", position_open_time)
      elif c.get("positionLifecycleVersion"):
        kept.append(c)
        continue
      else:
        try:
          key = (c.get("symbol"), "legacy", c.get("closeType") or "", round(float(pnl), 6))
        except (TypeError, ValueError):
          kept.append(c)
          continue
      ts = int(c.get("ts") or 0)
      prev = last_seen.get(key)
      if prev is not None and 0 <= ts - prev <= window_sec:
        continue
      last_seen[key] = ts
      kept.append(c)
    return kept

  @staticmethod
  def _position_open_time_ms(row: Dict[str, Any]) -> Optional[int]:
    value = row.get("positionOpenTime")
    if value in (None, ""):
      return None
    try:
      timestamp = int(float(value))
    except (TypeError, ValueError):
      return None
    return timestamp if timestamp > 1_000_000_000_000 else timestamp * 1000

  @staticmethod
  def _position_side(row: Dict[str, Any]) -> Optional[str]:
    explicit = str(row.get("positionSide") or "").lower()
    if explicit in {"long", "short"}:
      return explicit
    close_type = str(row.get("closeType") or "").upper()
    if "LONG" in close_type:
      return "long"
    if "SHORT" in close_type:
      return "short"
    action = str(row.get("action") or "").lower()
    if action.startswith("futures_sell"):
      return "long"
    if action.startswith("futures_buy"):
      return "short"
    return None

  @staticmethod
  def _same_position_lifecycle(
    local: Dict[str, Any],
    authoritative: Dict[str, Any],
    *,
    window_sec: int,
  ) -> bool:
    """Match closes from one position without merging a rapid same-symbol re-entry.

    Current records use position ID/open time. Only legacy local rows lacking both identifiers
    use a shorter direction-aware time match so historical restart duplicates remain healed.
    """
    local_id = str(local.get("positionId") or "").strip()
    auth_id = str(authoritative.get("positionId") or "").strip()
    local_open = MemoryStore._position_open_time_ms(local)
    auth_open = MemoryStore._position_open_time_ms(authoritative)

    comparable = False
    if local_id and auth_id:
      comparable = True
      if local_id == auth_id:
        return True
    if local_open is not None and auth_open is not None:
      comparable = True
      if abs(local_open - auth_open) <= 1000:
        return True
    if comparable or local_id or local_open is not None or local.get("positionLifecycleVersion"):
      return False

    local_side = MemoryStore._position_side(local)
    auth_side = MemoryStore._position_side(authoritative)
    if not local_side or local_side != auth_side:
      return False
    delta = int(authoritative.get("ts") or 0) - int(local.get("ts") or 0)
    legacy_window_sec = min(max(0, int(window_sec)), 600)
    return 0 <= delta <= legacy_window_sec

  def realized_closes(self, limit: int = 100, symbol: str | None = None) -> list[Dict[str, Any]]:
    """Recent realized closes, exchange-triggered or explicit, deduped oldest→newest.

    The strict realized set (exchange-confirmed TP/SL closes) the adaptive edge controller
    computes its rolling stats from — hold/manage snapshots are excluded.
    """
    with self._lock:
      data = self._read()
    sym = _normalize_symbol(symbol) if symbol else None
    rows = [
      d for d in (data.get("decisions") or [])
      if isinstance(d, dict)
      and self._is_realized_close(str(d.get("action") or ""))
      and d.get("pnl") is not None
      and (sym is None or d.get("symbol") == sym)
    ]
    rows = self._authoritative_realized_rows(rows)
    return rows[-max(1, int(limit)):]

  def get_seen_close_ids(self) -> list[str]:
    """Exchange close-position IDs already recorded as decisions (persists across restarts)."""
    with self._lock:
      data = self._read()
    return [str(x) for x in (data.get("seen_close_ids") or [])]

  def record_seen_close_id(self, close_id: str, cap: int = 200) -> None:
    """Persist a recorded close ID so a restart can't double-record the same close."""
    cid = str(close_id or "").strip()
    if not cid:
      return
    with self._lock:
      data = self._read()
      ids = [str(x) for x in (data.get("seen_close_ids") or [])]
      if cid not in ids:
        ids.append(cid)
        data["seen_close_ids"] = ids[-cap:]
        self._write(data)

  def get_seen_fill_ids(self) -> list[str]:
    """Exchange fill IDs already delivered to the agent (persists across polls/restarts)."""
    with self._lock:
      data = self._read()
    return [str(x) for x in (data.get("seen_fill_ids") or [])]

  def mark_order_filled(self, order_id: Any = None, client_oid: Any = None) -> bool:
    """Promote a submitted-order record to executed without using it for position reconstruction."""
    refs = {
      str(value) for value in (order_id, client_oid)
      if value not in (None, "")
    }
    if not refs:
      return False
    with self._lock:
      data = self._prune(self._read())
      trades = data.get("trades", []) or []
      for trade in reversed(trades):
        trade_refs = {
          str(value) for value in (trade.get("orderId"), trade.get("clientOid"))
          if value not in (None, "")
        }
        if refs & trade_refs:
          if trade.get("filled") is not True:
            trade["filled"] = True
            self._write(data)
          return True
    return False

  def record_seen_fill_id(self, fill_id: str, cap: int = 1000) -> None:
    fid = str(fill_id or "").strip()
    if not fid:
      return
    with self._lock:
      data = self._read()
      ids = [str(x) for x in (data.get("seen_fill_ids") or [])]
      if fid not in ids:
        ids.append(fid)
        data["seen_fill_ids"] = ids[-cap:]
        self._write(data)

  def queue_agent_event(self, kind: str, event_id: str, payload: Dict[str, Any]) -> bool:
    """Persist a fill/close until a successful model run acknowledges that exact event."""
    if kind not in {"spot_fills", "futures_fills", "closed_positions"}:
      return False
    eid = str(event_id or "").strip()
    if not eid or not isinstance(payload, dict):
      return False
    with self._lock:
      data = self._prune(self._read())
      pending = data.setdefault("pending_agent_events", [])
      if any(str(event.get("id") or "") == eid for event in pending if isinstance(event, dict)):
        return False
      pending.append({"id": eid, "kind": kind, "payload": copy.deepcopy(payload), "ts": int(time.time())})
      data["pending_agent_events"] = pending[-MAX_PENDING_AGENT_EVENTS:]
      self._write(data)
      return True

  def get_pending_agent_events(self) -> list[Dict[str, Any]]:
    with self._lock:
      data = self._prune(self._read())
      return copy.deepcopy(data.get("pending_agent_events", []) or [])

  def acknowledge_agent_events(self, event_ids: list[str]) -> list[Dict[str, Any]]:
    ids = {str(event_id) for event_id in event_ids if str(event_id or "").strip()}
    if not ids:
      return []
    with self._lock:
      data = self._prune(self._read())
      pending = data.get("pending_agent_events", []) or []
      acknowledged = [event for event in pending if str(event.get("id") or "") in ids]
      data["pending_agent_events"] = [
        event for event in pending if str(event.get("id") or "") not in ids
      ]
      self._write(data)
      return copy.deepcopy(acknowledged)

  def get_agent_scheduler(self) -> Dict[str, Any]:
    """Return persisted model-cadence and adaptive price-noise state."""
    with self._lock:
      data = self._prune(self._read())
      return copy.deepcopy(_sanitize_agent_scheduler(data.get("agent_scheduler")))

  def save_agent_scheduler(self, state: Dict[str, Any]) -> Dict[str, Any]:
    """Atomically persist scheduler state without exposing arbitrary memory keys."""
    normalized = _sanitize_agent_scheduler(state)
    with self._lock:
      data = self._prune(self._read())
      if data.get("agent_scheduler") != normalized:
        data["agent_scheduler"] = normalized
        self._write(data)
      return copy.deepcopy(normalized)

  def observe_open_interest(
    self,
    symbol: str,
    value: float,
    *,
    price: Optional[float] = None,
    now: Optional[int] = None,
    min_age_sec: int = 300,
    change_threshold: float = 0.005,
    price_change_threshold: float = 0.001,
  ) -> Dict[str, Any]:
    """Compare timestamp-aligned OI and price observations; never infer either from volume."""
    ts = int(now if now is not None else time.time())
    sym = _normalize_symbol(symbol)
    current = float(value)
    current_price = float(price) if price is not None else None
    if current_price is not None and current_price <= 0:
      current_price = None
    if current <= 0:
      return {"trend": None, "changePct": None, "priceTrend": None, "priceChangePct": None, "ageSec": None}
    with self._lock:
      data = self._read()
      observations = data.setdefault("open_interest_observations", {})
      previous = observations.get(sym) if isinstance(observations, dict) else None
      if not isinstance(previous, dict) or not previous.get("value") or not previous.get("ts"):
        observations[sym] = {"value": current, "price": current_price, "ts": ts}
        data["open_interest_observations"] = observations
        self._write(data)
        return {"trend": None, "changePct": None, "priceTrend": None, "priceChangePct": None, "ageSec": None}

      prior = float(previous["value"])
      age = max(0, ts - int(previous["ts"]))
      change = (current - prior) / prior if prior > 0 else 0.0
      trend = None
      prior_price = float(previous.get("price")) if previous.get("price") not in (None, "") else None
      price_change = (
        (current_price - prior_price) / prior_price
        if current_price is not None and prior_price is not None and prior_price > 0
        else None
      )
      price_trend = None
      if age >= max(1, int(min_age_sec)):
        threshold = max(0.0, float(change_threshold))
        trend = "up" if change >= threshold else "down" if change <= -threshold else "flat"
        if price_change is not None:
          price_threshold = max(0.0, float(price_change_threshold))
          price_trend = "up" if price_change >= price_threshold else "down" if price_change <= -price_threshold else "flat"
        observations[sym] = {"value": current, "price": current_price, "ts": ts}
        data["open_interest_observations"] = observations
        self._write(data)
      return {
        "trend": trend,
        "changePct": change * 100.0,
        "priceTrend": price_trend,
        "priceChangePct": price_change * 100.0 if price_change is not None else None,
        "ageSec": age,
      }

  @staticmethod
  def _is_realized_close(action: str) -> bool:
    """Return True if the action represents a realized trade close, not a hold/manage snapshot."""
    a = action.lower()
    if "triggered" in a:
      return True
    # Exact execution actions only.  The old substring test counted labels such as
    # "hold-close-only" as a closed trade and polluted win rate, streak and Kelly calculations.
    if a in {
      "futures_close", "spot_close", "futures_sell", "futures_buy", "spot_sell",
      "cut_loss", "early_cut", "profit_lock_close", "reduce_position",
    }:
      return True
    return a.startswith("close_") or a.endswith("_closed")

  @staticmethod
  def _pnl_stats(decisions: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute win/loss stats from a list of decision dicts (realized closes only)."""
    decisions = MemoryStore._authoritative_realized_rows(decisions)
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
    breakeven = [p for p in realized if p == 0]
    total = sum(realized)
    # Win rate is over decided closes only — break-even (pnl == 0) trades are neither wins nor
    # losses, so counting them in the denominator understates the rate. closedWithPnl keeps its
    # original meaning (every realized close, break-evens included) for the trading loop.
    decided = len(wins) + len(losses)
    wr = len(wins) / decided if decided else 0.0
    stats: Dict[str, Any] = {
      "closedWithPnl": len(realized),
      "wins": len(wins),
      "losses": len(losses),
      "breakeven": len(breakeven),
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

  @staticmethod
  def _authoritative_realized_rows(decisions: list[Dict[str, Any]], window_sec: int = 1800) -> list[Dict[str, Any]]:
    """Return realized rows without double-counting pre-close estimates and exchange final PnL.

    A market close is logged immediately with an estimated PnL.  KuCoin then reports the closed
    position with authoritative cumulative PnL (including earlier partial reductions).  Counting
    both overstated the ZEC lifecycle by +0.5606 USDT. Position identity/open time now determines
    whether a local close belongs to the exchange result, avoiding collisions after rapid re-entry.
    """
    rows = [
      d for d in decisions
      if isinstance(d, dict)
      and MemoryStore._is_realized_close(str(d.get("action") or ""))
      and d.get("pnl") is not None
    ]
    rows.sort(key=lambda d: d.get("ts") or 0)
    triggered = [d for d in rows if "triggered" in str(d.get("action") or "").lower()]
    kept: list[Dict[str, Any]] = []
    for row in rows:
      action = str(row.get("action") or "").lower()
      if action.startswith("futures_") and "triggered" not in action:
        ts = int(row.get("ts") or 0)
        if any(
          later.get("symbol") == row.get("symbol")
          and 0 <= int(later.get("ts") or 0) - ts <= window_sec
          and MemoryStore._same_position_lifecycle(row, later, window_sec=window_sec)
          for later in triggered
        ):
          continue
      kept.append(row)
    return MemoryStore._dedupe_realized(kept, window_sec=window_sec)

  def performance_summary(self) -> Dict[str, Any]:
    """Compute win/loss stats from recorded decisions, split by venue and paper/live."""
    with self._lock:
      data = self._prune(self._read())
      all_trade_records = data.get("trades", [])
      decisions = data.get("decisions", [])
    trades = [trade for trade in all_trade_records if trade.get("filled") is not False]
    submissions = [trade for trade in all_trade_records if trade.get("filled") is False]

    def _limit_execution_stats(records: list[Dict[str, Any]]) -> Dict[str, Any]:
      limit_records = [
        trade for trade in records
        if str(trade.get("clientOid") or "").startswith("traide-entry-")
      ]
      filled_limits = [trade for trade in limit_records if trade.get("filled") is True]
      return {
        "limitOrdersSubmitted": len(limit_records),
        "limitOrdersFilled": len(filled_limits),
        "limitFillRate": round(len(filled_limits) / len(limit_records), 3) if limit_records else None,
      }

    execution_stats = _limit_execution_stats(all_trade_records)

    if not trades:
      return {
        "totalTrades": 0,
        "orderSubmissions": len(submissions),
        **execution_stats,
        "message": "No executed trade history yet.",
      }

    overall = self._pnl_stats(decisions)
    overall["totalTrades"] = len(trades)
    overall["orderSubmissions"] = len(submissions)
    overall.update(execution_stats)

    spot_decisions = [d for d in decisions if (d.get("action") or "").startswith("spot_")]
    futures_decisions = [d for d in decisions if (d.get("action") or "").startswith("futures_")]
    spot_trades = [t for t in trades if t.get("venue", "spot") == "spot"]
    futures_trades = [t for t in trades if t.get("venue", "spot") == "futures"]

    def _venue_block(venue_decisions: list, venue_trades: list, venue_submissions: list) -> Dict[str, Any]:
      block = self._pnl_stats(venue_decisions)
      block["totalTrades"] = len(venue_trades)
      block["orderSubmissions"] = len(venue_submissions)
      venue = venue_trades + venue_submissions
      block.update(_limit_execution_stats(venue))
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

    spot_submissions = [t for t in submissions if t.get("venue", "spot") == "spot"]
    futures_submissions = [t for t in submissions if t.get("venue", "spot") == "futures"]
    overall["spot"] = _venue_block(spot_decisions, spot_trades, spot_submissions)
    overall["futures"] = _venue_block(futures_decisions, futures_trades, futures_submissions)
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
    decisions = MemoryStore._authoritative_realized_rows(decisions)
    decisions = sorted(decisions, key=lambda d: d.get("ts", 0))[-lookback:]
    realized = []
    for d in decisions:
      if not MemoryStore._is_realized_close(str(d.get("action") or "")):
        continue
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
    decisions = MemoryStore._authoritative_realized_rows(decisions)
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
    decisions = MemoryStore._authoritative_realized_rows(decisions)
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

  def recent_win_close(self, symbol: str, within_minutes: float) -> Optional[Dict[str, Any]]:
    """Most recent realized *winning* close for a symbol within the window, or None.

    Powers the no-chase guard: after taking profit, re-entering the same direction at a
    worse price is blocked for a cooldown. Returns {ts, pnl, exitPrice, closeType}.
    """
    sym = _normalize_symbol(symbol)
    cutoff = int(time.time()) - int(max(0.0, within_minutes) * 60)
    with self._lock:
      data = self._prune(self._read())
      decisions = data.get("decisions", [])
    decisions = MemoryStore._authoritative_realized_rows(decisions)
    for d in sorted(decisions, key=lambda d: d.get("ts", 0), reverse=True):
      ts = d.get("ts") or 0
      if ts < cutoff:
        break  # sorted newest-first: nothing else is within the window
      if d.get("symbol") != sym:
        continue
      if not MemoryStore._is_realized_close((d.get("action") or "").lower()):
        continue
      pnl = d.get("pnl")
      if pnl is None:
        continue
      try:
        if float(pnl) <= 0:
          continue
      except (TypeError, ValueError):
        continue
      return {"ts": ts, "pnl": float(pnl), "exitPrice": d.get("exitPrice"), "closeType": d.get("closeType")}
    return None

  def portfolio_heat(self, stop_distances: Dict[str, float], total_equity: float) -> float:
    """Compute portfolio heat = sum of capital at risk / total equity * 100.
    stop_distances: {symbol: usd_amount_at_risk}"""
    if total_equity <= 0:
      return 0.0
    total_risk = sum(abs(v) for v in stop_distances.values())
    return round(total_risk / total_equity * 100, 2)
