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
from .utils import normalize_symbol as _normalize_symbol

logger = logging.getLogger(__name__)

# A trigger the agent hasn't refreshed within this window is stale intent, not something it's still
# "watching" — dropping it keeps the dashboard's Watching panel from fixating on an old level (a
# single Jul-10 SOL trigger kept showing for a week once retention was extended to 90 days).
_TRIGGER_FRESHNESS_SEC = 12 * 3600
# How many most-recent closed positions to publish with full open→close lifecycle detail.
_CLOSED_DETAIL_LIMIT = 3

try:
  from azure.data.tables import TableServiceClient, UpdateMode
  from azure.storage.blob import BlobServiceClient, ContentSettings
  _AZURE_AVAILABLE = True
except Exception:  # azure SDK not installed
  _AZURE_AVAILABLE = False

# The Azure SDK logs every HTTP request/response (status, headers, body markers) at INFO via its
# http_logging_policy, which floods the trading logs. Force the whole azure.* hierarchy — and the
# HTTP policy specifically — to WARNING so these never appear (real azure errors still surface).
for _azure_logger in (
  "azure",
  "azure.core.pipeline.policies.http_logging_policy",
  "azure.storage",
  "azure.data.tables",
  "azure.identity",
  "azure.core",
):
  logging.getLogger(_azure_logger).setLevel(logging.WARNING)

# Azure Table caps a single string property at 64 KiB / 32 K chars; stay comfortably under.
MAX_TABLE_PROPERTY_CHARS = 30000

PK_EQUITY = "equity"      # durable daily index series  (RowKey = {day:08d})
PK_DECISION = "decision"  # append-only decision feed   (RowKey = {ts:010d}-{symbol})
PK_TRADE = "trade"        # append-only closed outcomes (RowKey = {day:08d}-{ts}-{symbol}-{action})
PK_PLAN = "plan"          # durable research-plan log    (RowKey = {ts:010d}-{title})
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


def _normalize_ts_sec(value: Any) -> Optional[int]:
  """Coerce an epoch that may be in seconds, ms, or ns to whole seconds. None if unparseable."""
  v = _f(value)
  if v is None or v <= 0:
    return None
  while v > 1e12:  # ms (1e12) or ns (1e18) -> seconds
    v /= 1000.0
  return int(v)


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
      logger.debug("Dashboard Azure init failed, disabling publisher for this process: %s", exc)
      self._init_failed = True
      return False

  def publish(self, memory: MemoryStore, snapshot, last_prices: Dict[str, float], cfg) -> None:
    """Top-level entry point called from the trading loop. NEVER raises; silent at INFO level
    (failures log at DEBUG only, so steady-state logs stay clean).

    `snapshot` is the current TradingSnapshot — open positions are taken from it (the exchange's
    live truth) rather than from MemoryStore, which only synthesizes positions from recorded
    trades and would keep showing a position after a TP/SL trigger closes it on the exchange."""
    try:
      if not self.enabled:
        return
      now = time.time()
      if now - self._last_publish_ts < self.cfg.publish_interval_sec:
        return
      if not self._ensure_clients():
        return

      payload = self._build_payload(memory, snapshot, last_prices or {}, cfg)
      self._write_tables(payload)
      series = self._read_equity_series()
      # The equity table is the durable record, but a concurrent publisher (or read-after-write
      # skew) can leave today's row stale relative to the equityToday we just computed in this
      # payload. equityToday is the authoritative latest point, so splice it over today's slot to
      # keep the blob's equityPoints + rollups internally consistent with the headline drawdownPct.
      series = self._merge_today_point(series, payload.get("equityToday"))
      payload["equityPoints"] = series[-90:]
      self._write_blobs(payload, series)
      self._last_publish_ts = now
    except Exception as exc:
      logger.debug("Dashboard publish failed (continuing trading loop): %s", exc)

  # ----- payload build -----------------------------------------------------

  def _build_payload(self, memory: MemoryStore, snapshot, last_prices: Dict[str, float], cfg) -> Dict[str, Any]:
    perf = memory.performance_summary()
    decisions = memory.latest_items("decisions", limit=self.cfg.feed_limit).get("items", [])
    closed = self._closed_trades(memory)
    notes_all = memory.list_all_notes()
    perm = memory.get_permanent_notes()
    triggers = memory.latest_triggers()
    sentiments = memory.latest_items("sentiments", limit=10).get("items", [])
    plans = memory.latest_items("plans", limit=5).get("items", [])
    coins = memory.latest_items("coins", limit=30).get("items", [])
    raw_limits = self._read_limits(memory)
    today = int(time.time() // 86400)

    # Open positions come from the live exchange snapshot (truth), not from MemoryStore.
    pos_list = self._build_positions(snapshot, memory)
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
      "pendingOrders": self._sanitize_pending_orders(snapshot),
      "closedPositions": self._closed_position_lifecycles(memory),
      "coins": self._sanitize_coins(coins),
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
        "handoffs": [
          self._sanitize_decision(d) for d in decisions
          if (d.get("action") or "").startswith("handoff")
        ],
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
      out = {k: b.get(k) for k in ("closedWithPnl", "wins", "losses", "breakeven", "winRate", "totalTrades") if k in b}
      if allow:
        for k in ("totalRealizedPnl", "avgWin", "avgLoss", "bestTrade", "worstTrade"):
          if k in b:
            out[k] = b.get(k)
      return out

    top = block(perf)
    for venue in ("spot", "futures"):
      vb = perf.get(venue) or {}
      top[venue] = {**block(vb), "live": block(vb.get("live", {})), "paper": block(vb.get("paper", {}))}

    # Reconcile the public headline counts with the spot+futures breakdown shown on the page. The
    # raw overall classifies closes by action prefix, so a venue-less advisory close (e.g.
    # "close/stand-aside") lands in the overall but in no venue block — leaving wins/closedWithPnl
    # that don't sum to spot+futures. Recompute the headline from the venues so the dashboard is
    # internally consistent. totalTrades stays the true overall count.
    sw = (top["spot"].get("wins") or 0) + (top["futures"].get("wins") or 0)
    sl = (top["spot"].get("losses") or 0) + (top["futures"].get("losses") or 0)
    sb = (top["spot"].get("breakeven") or 0) + (top["futures"].get("breakeven") or 0)
    top["wins"], top["losses"], top["breakeven"] = sw, sl, sb
    top["closedWithPnl"] = sw + sl + sb
    top["winRate"] = round(sw / (sw + sl), 3) if (sw + sl) else 0.0
    return top

  @staticmethod
  def _return_pct(avg_entry: Any, current_price: Any, net: float) -> Optional[float]:
    ae, cp = _f(avg_entry), _f(current_price)
    if not ae or not cp or not net:
      return None
    sign = 1.0 if net > 0 else -1.0
    return round((cp / ae - 1.0) * sign * 100.0, 4)

  @staticmethod
  def _futures_to_display(fsym: str) -> str:
    """Map a KuCoin futures contract symbol (e.g. XBTUSDTM) to a display symbol (BTC-USDT)."""
    s = (fsym or "").upper()
    if not s:
      return ""
    if s.endswith("M"):
      s = s[:-1]
    if s.endswith("USDT"):
      base, quote = s[:-4], "USDT"
    elif s.endswith("USDC"):
      base, quote = s[:-4], "USDC"
    elif s.endswith("USD"):
      base, quote = s[:-3], "USD"
    else:
      base, quote = s, "USDT"
    if base == "XBT":
      base = "BTC"
    return f"{base}-{quote}" if base else (fsym or "")

  def _build_positions(self, snapshot, memory: MemoryStore) -> list:
    """Open positions from the LIVE exchange snapshot (truth) — not MemoryStore, which only
    synthesizes positions from recorded trades and lingers after a TP/SL trigger closes one.
    Privacy-safe: symbol, side, venue, entry/mark price and return % only — never size,
    quantity, or notional."""
    out: list = []
    if snapshot is None:
      return out

    # --- Futures: signed currentQty straight from the exchange ---
    for p in (getattr(snapshot, "futures_positions", None) or []):
      qty = _f(p.get("currentQty")) or 0.0
      if not qty:
        continue  # closed positions report currentQty == 0
      entry = _f(p.get("avgEntryPrice"))
      mark = _f(p.get("markPrice"))
      disp = self._futures_to_display(p.get("symbol") or "")
      side = "long" if qty > 0 else "short"
      tp, sl = self._bracket_for(getattr(snapshot, "futures_stop_orders", None), disp, side, mark)
      rec: Dict[str, Any] = {
        "symbol": disp,
        "side": side,
        "venue": "futures",
        "avgEntry": _round(entry),
        "currentPrice": _round(mark),
        "returnPct": self._return_pct(entry, mark, qty),
        "tpPrice": _round(tp),
        "slPrice": _round(sl),
        "peakPnlPct": None,
        "troughPnlPct": None,
        "lastTs": int(time.time()),
      }
      if self._allow_usd():
        upnl = p.get("unrealisedPnl")
        if upnl is None:
          upnl = p.get("unrealizedPnl")
        rec["unrealizedPnl"] = _round(upnl)
      out.append(rec)

    # --- Spot: actual coin balances above a dust threshold, priced via tickers ---
    try:
      spot_mem = memory.positions(venue="spot")
    except Exception:
      spot_mem = {}
    tickers = getattr(snapshot, "tickers", None) or {}
    seen: set = set()
    for acct in (getattr(snapshot, "spot_accounts", None) or []):
      cur = (getattr(acct, "currency", "") or "").upper()
      if not cur or cur in ("USDT", "USDC", "USD", "DAI"):
        continue
      sym = f"{cur}-USDT"
      if sym in seen:
        continue
      try:
        bal = float(getattr(acct, "balance", 0) or 0)
      except (TypeError, ValueError):
        bal = 0.0
      if bal <= 0:
        continue
      tk = tickers.get(sym)
      price = _f(getattr(tk, "price", None)) if tk is not None else None
      if not price or bal * price < 1.0:
        continue  # skip dust (< $1) — value used only to filter, never published
      seen.add(sym)
      entry = _f((spot_mem.get(sym) or {}).get("avgEntry"))
      tp, sl = self._bracket_for(getattr(snapshot, "spot_stop_orders", None), sym, "long", price)
      out.append({
        "symbol": sym,
        "side": "long",
        "venue": "spot",
        "avgEntry": _round(entry),
        "currentPrice": _round(price),
        "returnPct": self._return_pct(entry, price, 1.0),
        "tpPrice": _round(tp),
        "slPrice": _round(sl),
        "peakPnlPct": None,
        "troughPnlPct": None,
        "lastTs": int(time.time()),
      })

    return out

  def _sanitize_pending_orders(self, snapshot) -> List[Dict[str, Any]]:
    """Resting (unfilled) limit/entry orders waiting to trigger — public-safe: symbol, side, type,
    limit price, venue, entry-vs-reduce, and age only. Never size/quantity (privacy). Entries the bot
    placed are tagged (`traide-entry-`) so the dashboard can show what it's waiting to open."""
    out: List[Dict[str, Any]] = []
    if snapshot is None:
      return out
    groups = (
      ("spot", getattr(snapshot, "spot_pending_orders", None) or []),
      ("futures", getattr(snapshot, "futures_pending_orders", None) or []),
    )
    for venue, orders in groups:
      for o in orders:
        if not isinstance(o, dict):
          continue
        raw_sym = str(o.get("symbol") or "")
        disp = self._futures_to_display(raw_sym) if venue == "futures" else _normalize_symbol(raw_sym)
        if not disp:
          continue
        reduce_only = bool(o.get("reduceOnly")) or bool(o.get("closeOrder"))
        out.append({
          "symbol": disp,
          "side": (str(o.get("side") or "")).lower(),
          "type": (str(o.get("type") or "limit")).lower(),
          "venue": venue,
          "price": _round(o.get("price")),
          "kind": "reduce" if reduce_only else "entry",
          "botEntry": str(o.get("clientOid") or "").startswith("traide-entry-"),
          "ts": _normalize_ts_sec(o.get("createdAt") or o.get("orderTime") or o.get("ts")),
        })
    out.sort(key=lambda r: r.get("ts") or 0, reverse=True)
    return out

  def _bracket_for(self, stop_orders, display_symbol, side, mark):
    """Pick the nearest take-profit and stop-loss trigger prices for one position from its active
    reduce-only stop orders on the exchange. Returns (tp_price, sl_price); either may be None.

    KuCoin futures stops carry a `stop` direction ('up'/'down'); for a long, an up-trigger is the
    take-profit and a down-trigger the stop-loss (reversed for a short). When the direction is
    absent (e.g. spot), fall back to classifying by where the trigger sits relative to mark. Only
    prices are exposed — never size — so this stays within the normalized disclosure policy."""
    m = _f(mark)
    tps: List[float] = []
    sls: List[float] = []
    for s in (stop_orders or []):
      if not isinstance(s, dict):
        continue
      raw = s.get("symbol") or ""
      # Futures stops use a contract symbol (XBTUSDTM -> BTC-USDT); spot stops already use the
      # display form (SOL-USDT). Accept either so both venues match their position.
      if raw != display_symbol and self._futures_to_display(raw) != display_symbol:
        continue
      sp = _f(s.get("stopPrice"))
      if sp is None or sp <= 0:
        continue
      direction = str(s.get("stop") or "").lower()
      if direction in ("up", "down"):
        is_tp = (direction == "up") if side == "long" else (direction == "down")
      elif m:
        above = sp >= m
        is_tp = above if side == "long" else (not above)
      else:
        continue
      (tps if is_tp else sls).append(sp)

    def nearest(vals: List[float]) -> Optional[float]:
      if not vals:
        return None
      return min(vals, key=lambda v: abs(v - m)) if m else vals[0]

    return nearest(tps), nearest(sls)

  def _sanitize_decision(self, d: Dict[str, Any]) -> Dict[str, Any]:
    action = d.get("action") or ""
    out: Dict[str, Any] = {
      "symbol": d.get("symbol"),
      "action": action,
      "confidence": _round(d.get("confidence"), 4),
      "reason": (d.get("reason") or "")[:500],
      "paper": bool(d.get("paper")),
      "ts": d.get("ts"),
      "day": d.get("day"),
    }
    # Attribute each feed item to an agent and flag handoffs so the dashboard can show when the
    # baton passed to the Research Agent instead of rendering only Trading Agent decisions.
    if action.startswith("handoff"):
      out["isHandoff"] = True
      out["agent"] = "research" if "research" in action else "trading"
      out["handoffTo"] = "research" if "research" in action else "trading"
    else:
      out["agent"] = "trading"
    pnl = d.get("pnl")
    if pnl is not None:
      # The WIN/LOSS outcome is safe to reveal (win rate is already public); the $ amount is not.
      out["win"] = bool(_f(pnl) and _f(pnl) > 0)
      if self._allow_usd():
        out["pnl"] = _round(pnl, 4)
    # Closed futures positions report a return-on-equity % in the reason string (e.g.
    # "TP/SL triggered (CLOSE_LONG, ROE -1.81%)"). ROE is a percentage — safe under the normalized
    # policy — and gives the trade-outcome chart a real magnitude instead of the always-0 confidence.
    if d.get("closeType"):
      out["closeType"] = str(d.get("closeType"))
    m = re.search(r"ROE\s*(-?\d+(?:\.\d+)?)\s*%", d.get("reason") or "")
    if m:
      out["roePct"] = round(float(m.group(1)), 4)
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
    now = time.time()
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
      # Only surface triggers the agent set/refreshed recently — an old level it moved on from is
      # not something it is still watching.
      and t.get("ts") and (now - float(t.get("ts") or 0)) <= _TRIGGER_FRESHNESS_SEC
    ]

  def _sanitize_coins(self, coins: Any) -> List[Dict[str, Any]]:
    """The curated coin universe (public-safe: symbols + status + rationale, no money).

    Active coins first, then most-recently-touched, so the dashboard can show what the bot is
    currently watching and why. Removed coins are kept (with their reason) for recent context.
    """
    rows = [
      {
        "symbol": c.get("symbol"),
        "status": c.get("status", "active"),
        "reason": (c.get("reason") or "")[:500],
        "ts": c.get("ts"),
      }
      for c in (coins or []) if isinstance(c, dict) and c.get("symbol")
    ]
    rows.sort(key=lambda r: (0 if r["status"] == "active" else 1, -(r["ts"] or 0)))
    return rows

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

  def _closed_position_lifecycles(self, memory: MemoryStore, limit: int = _CLOSED_DETAIL_LIMIT) -> List[Dict[str, Any]]:
    """The most recent closed positions with open→close timing + entry/exit prices (public-safe).

    Lets the dashboard chart each finished trade against candles and mark where it opened, where it
    closed, and — visible from the candles between those points — whether a better exit was missed.
    Prices are already disclosed by the live position charts; no size/balance/identity is exposed.
    """
    items = memory.latest_items("decisions", limit=80).get("items", [])
    rows: List[Dict[str, Any]] = []
    for d in items:
      if d.get("pnl") is None or not MemoryStore._is_realized_close(d.get("action") or ""):
        continue
      close_ts = d.get("ts")
      if not close_ts:
        continue
      open_ts = _normalize_ts_sec(d.get("positionOpenTime"))
      close_type = str(d.get("closeType") or "")
      side = "long" if "LONG" in close_type.upper() else ("short" if "SHORT" in close_type.upper() else (d.get("positionSide") or ""))
      m = re.search(r"ROE\s*(-?\d+(?:\.\d+)?)\s*%", d.get("reason") or "")
      rows.append({
        "symbol": d.get("symbol"),
        "side": side,
        "openTs": open_ts,
        "closeTs": int(close_ts),
        "entryPrice": _round(d.get("entryPrice")),
        "exitPrice": _round(d.get("exitPrice")),
        "win": bool(_f(d.get("pnl")) and _f(d.get("pnl")) > 0),
        "roePct": round(float(m.group(1)), 4) if m else None,
        "closeType": close_type or None,
      })
    rows.sort(key=lambda r: r["closeTs"], reverse=True)
    return rows[:limit]

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

  @staticmethod
  def _merge_today_point(series: List[Dict[str, Any]], today_point: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return `series` (ascending by day) with today's slot replaced by the authoritative
    `today_point`. The durable table can lag the freshly-computed equityToday under concurrent
    publishes; this guarantees equityPoints[-1] == equityToday so the blob never contradicts the
    headline drawdown. No-op when today_point is missing or has no indexClose."""
    if not today_point or today_point.get("indexClose") is None or today_point.get("day") is None:
      return series
    day = int(today_point["day"])
    point: Dict[str, Any] = {
      "day": day,
      "indexClose": _f(today_point.get("indexClose")),
      "drawdownPct": _f(today_point.get("drawdownPct")) or 0.0,
    }
    if "dayRealizedPnl" in today_point:
      point["dayRealizedPnl"] = _f(today_point.get("dayRealizedPnl"))
    merged = [p for p in series if p.get("day") != day]
    merged.append(point)
    merged.sort(key=lambda p: p["day"])
    return merged

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

    # Research plans: the local store hard-caps at MAX_PLANS (3), so without durable accumulation
    # the dashboard could only ever show the last three. Write each published plan to its own
    # partition (idempotent by ts+title) before the cap evicts it — the same way equity/trades
    # survive the local prune — so the dashboard can show several days of research history.
    for p in (payload.get("research") or {}).get("plans", []):
      ts = p.get("ts")
      if ts is None:
        continue
      day = int(int(ts) // 86400)
      rk = f"{int(ts):010d}-{_safe_key((p.get('title') or '')[:24])}"
      tc.upsert_entity(
        entity={"PartitionKey": PK_PLAN, "RowKey": rk, "ts": int(ts), "day": day,
                "data": json.dumps(p)[:MAX_TABLE_PROPERTY_CHARS]},
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
