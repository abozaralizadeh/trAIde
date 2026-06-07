"""Publish a sanitized, public-safe view of the agents' activity to Azure Blob + Table.

This is the PRODUCER half of the public spectator dashboard. A read-only consumer lives in the
SandBox repo and renders whatever this writes. Hard guarantees:

  * Never raises into the trading loop (every public entry point is wrapped in try/except).
  * No-ops cleanly when disabled or unconfigured.
  * Throttles to `publish_interval_sec` even if called every poll.
  * Idempotent: deterministic RowKeys + upsert, so repeated publishes never duplicate.
  * Privacy by whitelist: only safe fields are ever serialized. Account IDs, balances, total
    equity, position $ size (notionalUsd/netSize/cost) and API keys are NEVER published. In the
    default "normalized" disclosure mode, NO dollar figure leaves the process at all — money is
    expressed only as an indexed return curve (starting at `index_base`) and percentages.

Azure is the durable system of record: the local MemoryStore is pruned to ~14 days, but these
rows are written before pruning and never deleted here, so daily/weekly/monthly/all-time history
accumulates indefinitely.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from .memory import MemoryStore

logger = logging.getLogger(__name__)

try:
  from azure.data.tables import TableServiceClient, UpdateMode
  from azure.storage.blob import BlobServiceClient, ContentSettings
  _AZURE_AVAILABLE = True
except Exception:  # azure SDK not installed
  _AZURE_AVAILABLE = False

# Azure Table caps a single string property at 64 KiB / 32 K chars; stay comfortably under.
MAX_TABLE_PROPERTY_CHARS = 30000

PK_EQUITY = "equity"      # durable daily index series  (RowKey = {day:08d})
PK_DECISION = "decision"  # append-only decision feed   (RowKey = {ts:010d}-{symbol})
PK_TRADE = "trade"        # append-only closed outcomes (RowKey = {day:08d}-{ts}-{symbol}-{action})
PK_META = "meta"          # singleton state             (RowKey = "state")

_BAD_KEY_CHARS = re.compile(r"[\\/#?]")


def _safe_key(value: Any) -> str:
  """Strip characters Azure forbids in PartitionKey/RowKey."""
  return _BAD_KEY_CHARS.sub("_", str(value or ""))


def _f(value: Any) -> Optional[float]:
  try:
    return float(value) if value is not None else None
  except (TypeError, ValueError):
    return None


def _round(value: Any, ndigits: int = 6) -> Optional[float]:
  v = _f(value)
  return round(v, ndigits) if v is not None else None


def _month_key(day: int) -> str:
  t = time.gmtime(int(day) * 86400)
  return f"{t.tm_year:04d}-{t.tm_mon:02d}"


class DashboardPublisher:
  """Reads MemoryStore, sanitizes, and accumulates a public-safe projection to Azure."""

  def __init__(self, cfg) -> None:
    self.cfg = cfg.dashboard
    self._last_publish_ts: float = 0.0
    self._table_client = None
    self._container_client = None
    self._init_failed = False

  # ----- lifecycle ---------------------------------------------------------

  def disabled_reason(self) -> Optional[str]:
    """Human-readable reason the publisher won't run, or None when it is good to go."""
    c = self.cfg
    if not c.enabled:
      return "DASHBOARD_PUBLISH_ENABLED is not 'true'"
    if not _AZURE_AVAILABLE:
      return ("azure SDK not importable — run `pip install -r requirements.txt` "
              "(needs azure-data-tables + azure-storage-blob) in the deployed venv, then restart")
    if not c.connection_string:
      return "the `connection_string` env var is empty (set it in the deployed .env / service env)"
    if not c.table_name:
      return "TRAIDE_TABLE_NAME is empty"
    if not c.container_name:
      return "TRAIDE_BLOB_NAME is empty"
    return None

  @property
  def enabled(self) -> bool:
    return self.disabled_reason() is None

  def _ensure_clients(self) -> bool:
    """Lazily create the table + container clients (auto-create both). Never raises."""
    if not self.enabled or self._init_failed:
      return False
    if self._table_client is not None and self._container_client is not None:
      return True
    try:
      tsc = TableServiceClient.from_connection_string(conn_str=self.cfg.connection_string)
      try:
        tsc.create_table(table_name=self.cfg.table_name)
      except Exception:
        pass  # already exists
      self._table_client = tsc.get_table_client(self.cfg.table_name)

      bsc = BlobServiceClient.from_connection_string(self.cfg.connection_string)
      cc = bsc.get_container_client(self.cfg.container_name)
      try:
        cc.create_container()  # private by default — read server-side only
      except Exception:
        pass
      self._container_client = cc
      return True
    except Exception as exc:
      logger.warning("Dashboard Azure init failed, disabling publisher for this process: %s", exc)
      self._init_failed = True
      return False

  def publish(self, memory: MemoryStore, last_prices: Dict[str, float], cfg) -> None:
    """Top-level entry point called from the trading loop. NEVER raises; logs warnings only."""
    try:
      if not self.enabled:
        return
      now = time.time()
      if now - self._last_publish_ts < self.cfg.publish_interval_sec:
        return
      if not self._ensure_clients():
        return

      payload = self._build_payload(memory, last_prices or {}, cfg)
      self._write_tables(payload)
      series = self._read_equity_series()
      payload["equityPoints"] = series[-90:]
      self._write_blobs(payload, series)
      self._last_publish_ts = now
      logger.info(
        "Dashboard published: %d positions, %d feed, %d equity points (disclosure=%s)",
        len(payload.get("positions", [])), len(payload.get("feed", [])),
        len(series), self.cfg.disclosure,
      )
    except Exception as exc:
      logger.warning("Dashboard publish failed (continuing trading loop): %s", exc)

  # ----- payload build -----------------------------------------------------

  def _build_payload(self, memory: MemoryStore, last_prices: Dict[str, float], cfg) -> Dict[str, Any]:
    perf = memory.performance_summary()
    positions = memory.positions(last_prices)
    decisions = memory.latest_items("decisions", limit=self.cfg.feed_limit).get("items", [])
    closed = self._closed_trades(memory)
    notes_all = memory.list_all_notes()
    perm = memory.get_permanent_notes()
    triggers = memory.latest_triggers()
    sentiments = memory.latest_items("sentiments", limit=10).get("items", [])
    plans = memory.latest_items("plans", limit=5).get("items", [])
    raw_limits = self._read_limits(memory)
    today = int(time.time() // 86400)

    pos_list = [sp for sp in (self._sanitize_position(s, p) for s, p in positions.items()) if sp]
    dd_total = round(_f((raw_limits.get("total") or {}).get("drawdownPct")) or 0.0, 3)

    return {
      "generatedTs": int(time.time()),
      "schema": 1,
      "paperTrading": bool(cfg.trading.paper_trading),
      "disclosure": self.cfg.disclosure,
      "kpis": self._sanitize_perf(perf),
      "drawdownPct": dd_total,
      "openPositions": len(pos_list),
      "positions": pos_list,
      "feed": [self._sanitize_decision(d) for d in decisions],
      "trades": [self._sanitize_decision(d) for d in closed],
      "equityToday": self._compute_today_equity(memory, raw_limits, today),
      "notes": {
        "permanent": self._sanitize_notes(perm),
        "temporary": self._sanitize_notes(notes_all.get("temporary", [])),
      },
      "research": {
        "plans": self._sanitize_plans(plans),
        "triggers": self._sanitize_triggers(triggers),
        "sentiments": self._sanitize_sentiments(sentiments),
      },
    }

  # ----- sanitizers (whitelist only) --------------------------------------

  def _allow_usd(self) -> bool:
    return self.cfg.disclosure in ("absolute", "both")

  def _sanitize_perf(self, perf: Dict[str, Any]) -> Dict[str, Any]:
    allow = self._allow_usd()

    def block(b: Any) -> Dict[str, Any]:
      if not isinstance(b, dict):
        return {}
      out = {k: b.get(k) for k in ("closedWithPnl", "wins", "losses", "winRate", "totalTrades") if k in b}
      if allow:
        for k in ("totalRealizedPnl", "avgWin", "avgLoss", "bestTrade", "worstTrade"):
          if k in b:
            out[k] = b.get(k)
      return out

    top = block(perf)
    for venue in ("spot", "futures"):
      vb = perf.get(venue) or {}
      top[venue] = {**block(vb), "live": block(vb.get("live", {})), "paper": block(vb.get("paper", {}))}
    return top

  @staticmethod
  def _return_pct(avg_entry: Any, current_price: Any, net: float) -> Optional[float]:
    ae, cp = _f(avg_entry), _f(current_price)
    if not ae or not cp or not net:
      return None
    sign = 1.0 if net > 0 else -1.0
    return round((cp / ae - 1.0) * sign * 100.0, 4)

  @staticmethod
  def _pnl_to_pct(pnl: Any, cost: Any) -> Optional[float]:
    """Express a $ PnL as a % of the (undisclosed) entry notional — privacy-safe."""
    p, c = _f(pnl), _f(cost)
    if p is None or not c:
      return None
    return round(p / abs(c) * 100.0, 4)

  def _sanitize_position(self, symbol: str, p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    net = _f(p.get("netSize")) or 0.0
    if not net:
      return None
    cost = p.get("cost")
    out: Dict[str, Any] = {
      "symbol": symbol,
      "side": "long" if net > 0 else "short",
      "venue": p.get("venue", "spot"),
      "avgEntry": _round(p.get("avgEntry")),
      "currentPrice": _round(p.get("currentPrice")),
      "returnPct": self._return_pct(p.get("avgEntry"), p.get("currentPrice"), net),
      "peakPnlPct": self._pnl_to_pct(p.get("peakPnl"), cost),
      "troughPnlPct": self._pnl_to_pct(p.get("troughPnl"), cost),
      "lastTs": p.get("lastTs"),
    }
    if self._allow_usd():
      out["unrealizedPnl"] = _round(p.get("unrealizedPnl"))
      out["realizedPnl"] = _round(p.get("realizedPnl"))
    return out

  def _sanitize_decision(self, d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
      "symbol": d.get("symbol"),
      "action": d.get("action"),
      "confidence": _round(d.get("confidence"), 4),
      "reason": (d.get("reason") or "")[:500],
      "paper": bool(d.get("paper")),
      "ts": d.get("ts"),
      "day": d.get("day"),
    }
    pnl = d.get("pnl")
    if pnl is not None:
      # The WIN/LOSS outcome is safe to reveal (win rate is already public); the $ amount is not.
      out["win"] = bool(_f(pnl) and _f(pnl) > 0)
      if self._allow_usd():
        out["pnl"] = _round(pnl, 4)
    return out

  def _sanitize_notes(self, notes: Any) -> List[Dict[str, Any]]:
    return [
      {"content": (n.get("content") or "")[:1000], "author": n.get("author"), "ts": n.get("ts")}
      for n in (notes or []) if isinstance(n, dict)
    ]

  def _sanitize_plans(self, plans: Any) -> List[Dict[str, Any]]:
    return [
      {
        "title": (p.get("title") or "")[:200],
        "summary": (p.get("summary") or "")[:1000],
        "actions": [str(a)[:200] for a in (p.get("actions") or [])][:10],
        "author": p.get("author"),
        "ts": p.get("ts"),
      }
      for p in (plans or []) if isinstance(p, dict)
    ]

  def _sanitize_triggers(self, triggers: Any) -> List[Dict[str, Any]]:
    return [
      {
        "symbol": t.get("symbol"),
        "direction": t.get("direction"),
        "rationale": (t.get("rationale") or "")[:500],
        "targetPrice": _round(t.get("targetPrice")),
        "stopPrice": _round(t.get("stopPrice")),
        "ts": t.get("ts"),
      }
      for t in (triggers or []) if isinstance(t, dict)
    ]

  def _sanitize_sentiments(self, sentiments: Any) -> List[Dict[str, Any]]:
    return [
      {
        "symbol": s.get("symbol"),
        "score": _round(s.get("score"), 4),
        "rationale": (s.get("rationale") or "")[:500],
        "source": (s.get("source") or "")[:120],
        "ts": s.get("ts"),
      }
      for s in (sentiments or []) if isinstance(s, dict)
    ]

  # ----- equity index ------------------------------------------------------

  def _read_limits(self, memory: MemoryStore) -> Dict[str, Any]:
    try:
      with memory._lock:  # same access pattern the circuit breaker in main.py uses
        return dict(memory._read().get("limits", {}) or {})
    except Exception:
      return {}

  def _closed_trades(self, memory: MemoryStore, limit: int = 100) -> List[Dict[str, Any]]:
    items = memory.latest_items("decisions", limit=50).get("items", [])
    closed = [
      d for d in items
      if d.get("pnl") is not None and MemoryStore._is_realized_close(d.get("action") or "")
    ]
    return closed[:limit]

  def _prev_day_close(self, today: int) -> float:
    """The indexClose of the most recent finalized day before `today` (durable, from Azure)."""
    try:
      rk = f"{int(today):08d}"
      rows = self._table_client.query_entities(
        query_filter=f"PartitionKey eq '{PK_EQUITY}' and RowKey lt '{rk}'",
        results_per_page=1000,
      )
      best_day: Optional[int] = None
      best_close: Optional[float] = None
      for r in rows:
        try:
          d = int(r["RowKey"])
          c = _f(r.get("indexClose"))
        except Exception:
          continue
        if c is None:
          continue
        if best_day is None or d > best_day:
          best_day, best_close = d, c
      return best_close if best_close is not None else float(self.cfg.index_base)
    except Exception:
      return float(self.cfg.index_base)

  def _compute_today_equity(self, memory: MemoryStore, raw_limits: Dict[str, Any], today: int) -> Dict[str, Any]:
    """Build today's index point from intraday return ONLY (no absolute $ ever published).

    baseline_today == equity at the close of yesterday, so the index compounds:
      indexClose_today = prevDayClose * (1 + intradayReturn)
    Absolute USDT is read here purely to form the ratio, then divided out.
    """
    total = raw_limits.get("total") or {}
    baseline = _f(total.get("baselineUsdt"))
    current = _f(total.get("currentUsdt"))
    dd = _f(total.get("drawdownPct")) or 0.0
    prev_close = self._prev_day_close(today)

    intraday_return = 0.0
    if baseline and baseline > 0 and current and current > 0:
      intraday_return = (current - baseline) / baseline

    index_close = prev_close * (1.0 + intraday_return)
    out: Dict[str, Any] = {"day": today, "indexClose": round(index_close, 6), "drawdownPct": round(dd, 3)}
    if self._allow_usd():
      out["dayRealizedPnl"] = round(self._today_realized_pnl(memory, today), 4)
    return out

  def _today_realized_pnl(self, memory: MemoryStore, today: int) -> float:
    total = 0.0
    for d in memory.latest_items("decisions", limit=50).get("items", []):
      if (d.get("day") or 0) != today:
        continue
      pnl = d.get("pnl")
      if pnl is not None and MemoryStore._is_realized_close(d.get("action") or ""):
        total += _f(pnl) or 0.0
    return total

  def _read_equity_series(self) -> List[Dict[str, Any]]:
    try:
      rows = self._table_client.query_entities(
        query_filter=f"PartitionKey eq '{PK_EQUITY}'",
        results_per_page=1000,
      )
      series: List[Dict[str, Any]] = []
      for r in rows:
        try:
          day = int(r["RowKey"])
        except Exception:
          continue
        point = {"day": day, "indexClose": _f(r.get("indexClose")), "drawdownPct": _f(r.get("drawdownPct"))}
        if "dayRealizedPnl" in r:
          point["dayRealizedPnl"] = _f(r.get("dayRealizedPnl"))
        series.append(point)
      series.sort(key=lambda p: p["day"])
      return series
    except Exception:
      return []

  # ----- writers -----------------------------------------------------------

  def _write_tables(self, payload: Dict[str, Any]) -> None:
    tc = self._table_client

    et = payload.get("equityToday") or {}
    if et.get("indexClose") is not None:
      entity = {
        "PartitionKey": PK_EQUITY,
        "RowKey": f"{int(et['day']):08d}",
        "indexClose": float(et["indexClose"]),
        "drawdownPct": float(et.get("drawdownPct") or 0.0),
      }
      if "dayRealizedPnl" in et:
        entity["dayRealizedPnl"] = float(et["dayRealizedPnl"])
      tc.upsert_entity(entity=entity, mode=UpdateMode.MERGE)

    for d in payload.get("feed", []):
      ts, sym = d.get("ts"), d.get("symbol")
      if ts is None or not sym:
        continue
      rk = f"{int(ts):010d}-{_safe_key(sym)}"
      tc.upsert_entity(
        entity={"PartitionKey": PK_DECISION, "RowKey": rk, "ts": int(ts),
                "data": json.dumps(d)[:MAX_TABLE_PROPERTY_CHARS]},
        mode=UpdateMode.MERGE,
      )

    for d in payload.get("trades", []):
      ts, sym = d.get("ts"), d.get("symbol")
      if ts is None or not sym:
        continue
      day = int(d.get("day") or (int(ts) // 86400))
      rk = f"{day:08d}-{int(ts)}-{_safe_key(sym)}-{_safe_key(d.get('action'))}"
      tc.upsert_entity(
        entity={"PartitionKey": PK_TRADE, "RowKey": rk, "ts": int(ts),
                "data": json.dumps(d)[:MAX_TABLE_PROPERTY_CHARS]},
        mode=UpdateMode.MERGE,
      )

    tc.upsert_entity(
      entity={
        "PartitionKey": PK_META,
        "RowKey": "state",
        "generatedTs": payload["generatedTs"],
        "schema": 1,
        "disclosure": payload["disclosure"],
        "indexAnchor": float(self.cfg.index_base),
      },
      mode=UpdateMode.REPLACE,
    )

  def _write_blobs(self, payload: Dict[str, Any], series: List[Dict[str, Any]]) -> None:
    cc = self._container_client
    cs = ContentSettings(content_type="application/json")
    cc.get_blob_client("live.json").upload_blob(
      json.dumps(payload).encode("utf-8"), overwrite=True, content_settings=cs
    )
    for name, obj in self._build_rollups(series, payload).items():
      cc.get_blob_client(f"rollups/{name}.json").upload_blob(
        json.dumps(obj).encode("utf-8"), overwrite=True, content_settings=cs
      )

  @staticmethod
  def _bucket_last(series: List[Dict[str, Any]], key) -> List[Dict[str, Any]]:
    """Keep the last (highest-day) point per bucket. `series` must be ascending by day."""
    buckets: Dict[Any, Dict[str, Any]] = {}
    for p in series:
      buckets[key(p)] = p
    return sorted(buckets.values(), key=lambda p: p["day"])

  def _build_rollups(self, series: List[Dict[str, Any]], payload: Dict[str, Any]) -> Dict[str, Any]:
    kpis = payload.get("kpis", {})
    gen = payload["generatedTs"]
    return {
      "daily": {"points": series[-60:], "kpis": kpis, "generatedTs": gen},
      "weekly": {"points": self._bucket_last(series, lambda p: p["day"] // 7), "kpis": kpis, "generatedTs": gen},
      "monthly": {"points": self._bucket_last(series, lambda p: _month_key(p["day"])), "kpis": kpis, "generatedTs": gen},
      "alltime": {"points": series, "kpis": kpis, "generatedTs": gen},
    }
