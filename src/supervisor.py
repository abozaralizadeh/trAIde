from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests
from agents import Agent, OpenAIResponsesModel, Runner
from agents.tool import WebSearchTool, function_tool
from openai import AsyncAzureOpenAI

from .analytics import (
  INTERVAL_SECONDS,
  candles_to_dataframe,
  compute_indicators,
  summarize_interval,
  summarize_multi_timeframe,
)
from .agent import (
  _aggregate_account_totals,
  _base_currency,
  _build_compact_account_state,
  _to_float,
  _to_futures_symbol,
  _truncate_to_increment,
)
from .config import AppConfig
from .conversation_memory import ConversationMemory
from .kucoin import (
  KucoinClient,
  KucoinFuturesClient,
  KucoinFuturesOrderRequest,
  KucoinOrderRequest,
)
from .memory import MemoryStore
from .utils import normalize_symbol

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent


def _build_openai_client(cfg: AppConfig) -> AsyncAzureOpenAI:
  if cfg.apim.subscription_key:
    return AsyncAzureOpenAI(
      api_key=cfg.apim.subscription_key,
      api_version=cfg.apim.api_version,
      azure_endpoint=cfg.apim.endpoint,
      azure_deployment=cfg.apim.deployment,
    )
  return AsyncAzureOpenAI(
    api_key=cfg.azure.api_key,
    api_version=cfg.azure.api_version,
    azure_endpoint=cfg.azure.endpoint,
    azure_deployment=cfg.azure.deployment,
  )


def run_supervisor_agent(
  cfg: AppConfig,
  message_text: str,
  openai_client: AsyncAzureOpenAI | None = None,
) -> str:
  if openai_client is None:
    openai_client = _build_openai_client(cfg)
  model = OpenAIResponsesModel(
    model=cfg.azure.deployment,
    openai_client=openai_client,
  )
  memory = MemoryStore(cfg.memory_file, retention_days=cfg.retention_days)
  log_file_path = Path(cfg.supervisor.log_file)

  kucoin: KucoinClient | None = None
  kucoin_futures: KucoinFuturesClient | None = None
  try:
    kucoin = KucoinClient(cfg)
    if cfg.kucoin_futures.enabled:
      kucoin_futures = KucoinFuturesClient(cfg)
  except Exception as exc:
    logger.warning("Supervisor: KuCoin client init failed: %s", exc)

  _contract_cache: Dict[str, Dict[str, Any]] = {}

  def _get_contract_spec(futures_symbol: str) -> Dict[str, Any] | None:
    if not futures_symbol:
      return None
    if futures_symbol in _contract_cache:
      return _contract_cache[futures_symbol]
    if not kucoin_futures:
      return None
    url = f"{kucoin_futures.base_url}/api/v1/contracts/{futures_symbol}"
    try:
      resp = requests.get(url, timeout=10)
      if not resp.ok:
        return None
      payload = resp.json()
      if payload.get("code") not in ("200000", "200"):
        return None
      data = payload.get("data")
      if isinstance(data, dict):
        _contract_cache[futures_symbol] = data
        return data
    except Exception as exc:
      logger.warning("Unable to fetch futures contract %s: %s", futures_symbol, exc)
    return None

  # ── Logging / Source / Config tools ───────────────────────────────

  @function_tool
  async def read_logs(lines: int = 100) -> str:
    """Read the last N lines from the application log file."""
    lines = min(max(1, lines), 1000)
    if not log_file_path.exists():
      return "Log file not found."
    try:
      with open(log_file_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        chunk = min(size, lines * 500)
        f.seek(max(0, size - chunk))
        data = f.read().decode("utf-8", errors="replace")
      result = data.splitlines()[-lines:]
      return "\n".join(result)
    except Exception as exc:
      return f"Error reading logs: {exc}"

  @function_tool
  async def search_logs(query: str, max_results: int = 20) -> str:
    """Search for lines matching a query string in the log file."""
    max_results = min(max(1, max_results), 100)
    if not query:
      return "Query required."
    if not log_file_path.exists():
      return "Log file not found."
    try:
      matches: list[str] = []
      with open(log_file_path, "r", errors="replace") as f:
        for line in f:
          if query.lower() in line.lower():
            matches.append(line.rstrip())
            if len(matches) >= max_results:
              break
      if not matches:
        return f"No matches for '{query}'."
      return "\n".join(matches)
    except Exception as exc:
      return f"Error searching logs: {exc}"

  @function_tool
  async def read_source_file(path: str) -> str:
    """Read a source file from the project. Path relative to src/ (e.g. 'agent.py')."""
    if not path:
      return "Path required."
    target = (SRC_DIR / path).resolve()
    if not str(target).startswith(str(SRC_DIR)):
      return "Access denied: path must be within src/."
    if not target.exists():
      return f"File not found: {path}"
    if not target.is_file():
      return f"Not a file: {path}"
    try:
      content = target.read_text(errors="replace")
      if len(content) > 50000:
        content = content[:50000] + "\n... (truncated)"
      return content
    except Exception as exc:
      return f"Error reading file: {exc}"

  @function_tool
  async def list_source_files() -> List[str]:
    """List all Python files in the src/ directory."""
    return sorted(str(p.relative_to(SRC_DIR)) for p in SRC_DIR.rglob("*.py") if "__pycache__" not in str(p))

  @function_tool
  async def get_config_summary() -> Dict[str, Any]:
    """Get non-secret configuration values."""
    return {
      "coins": cfg.trading.coins,
      "flexible_coins_enabled": cfg.trading.flexible_coins_enabled,
      "paper_trading": cfg.trading.paper_trading,
      "max_position_usd": cfg.trading.max_position_usd,
      "risk_per_trade_pct": cfg.trading.risk_per_trade_pct,
      "min_confidence": cfg.trading.min_confidence,
      "sentiment_filter_enabled": cfg.trading.sentiment_filter_enabled,
      "sentiment_min_score": cfg.trading.sentiment_min_score,
      "poll_interval_sec": cfg.trading.poll_interval_sec,
      "price_change_trigger_pct": cfg.trading.price_change_trigger_pct,
      "max_idle_polls": cfg.trading.max_idle_polls,
      "max_leverage": cfg.trading.max_leverage,
      "max_trades_per_symbol_per_day": cfg.trading.max_trades_per_symbol_per_day,
      "min_net_profit_usd": cfg.trading.min_net_profit_usd,
      "min_profit_roi_pct": cfg.trading.min_profit_roi_pct,
      "estimated_slippage_pct": cfg.trading.estimated_slippage_pct,
      "futures_enabled": cfg.kucoin_futures.enabled,
      "telegram_enabled": cfg.telegram.enabled,
      "supervisor_enabled": cfg.supervisor.enabled,
      "memory_file": cfg.memory_file,
      "retention_days": cfg.retention_days,
      "agent_max_turns": cfg.agent_max_turns,
    }

  # ── Memory tools ──────────────────────────────────────────────────

  @function_tool
  async def read_memory() -> Dict[str, Any]:
    """Read the full agent memory file (.agent_memory.json)."""
    try:
      path = Path(cfg.memory_file)
      if not path.exists():
        return {"error": "Memory file not found"}
      return json.loads(path.read_text())
    except Exception as exc:
      return {"error": str(exc)}

  @function_tool
  async def get_performance_summary() -> Dict[str, Any]:
    """Get trading performance summary (win rate, PnL, trade counts)."""
    return memory.performance_summary()

  @function_tool
  async def get_positions() -> Dict[str, Any]:
    """Get current tracked positions with unrealized PnL."""
    return memory.positions()

  @function_tool
  async def get_recent_decisions(limit: int = 10) -> Dict[str, Any]:
    """Get recent trading decisions from memory."""
    return memory.latest_items("decisions", min(max(1, limit), 50))

  @function_tool
  async def get_recent_trades(limit: int = 10) -> Dict[str, Any]:
    """Get recent trades from memory."""
    return memory.latest_items("trades", min(max(1, limit), 50))

  @function_tool
  async def latest_plan() -> Dict[str, Any]:
    """Get the latest stored trading plan."""
    return {"latest_plan": memory.latest_plan()}

  @function_tool
  async def latest_items(kind: str, limit: int = 5) -> Dict[str, Any]:
    """Fetch latest N memory entries (plans/research/sentiments/decisions/trades/triggers/coins/fees)."""
    if not kind:
      return {"error": "kind required"}
    return memory.latest_items(kind, limit)

  @function_tool
  async def save_trade_plan(title: str, summary: str, actions: List[str]) -> Dict[str, Any]:
    """Persist a trading plan (title, summary, actions) for recall."""
    return memory.save_plan(title=title, summary=summary, actions=actions, author="Supervisor Agent")

  @function_tool
  async def clear_plans() -> Dict[str, Any]:
    """Clear all stored plans and triggers."""
    return memory.clear_plans()

  @function_tool
  async def log_sentiment(symbol: str, score: float, rationale: str, source: str = "") -> Dict[str, Any]:
    """Store a sentiment score (0-1) with rationale and source for gating trades."""
    try:
      score_val = float(score)
    except (TypeError, ValueError):
      return {"error": "Invalid score"}
    if not symbol:
      return {"error": "symbol required"}
    return {"sentiment": memory.log_sentiment(normalize_symbol(symbol), score_val, rationale, source)}

  @function_tool
  async def log_decision(
    symbol: str,
    action: str,
    confidence: float,
    reason: str,
    pnl: float | None = None,
    paper: bool = False,
  ) -> Dict[str, Any]:
    """Record decision/confidence for calibration."""
    if not symbol:
      return {"error": "symbol required"}
    try:
      conf = float(confidence)
    except (TypeError, ValueError):
      return {"error": "Invalid confidence"}
    return {"decision": memory.log_decision(normalize_symbol(symbol), action, conf, reason, pnl=pnl, paper=paper)}

  @function_tool
  async def log_research(topic: str, summary: str, actions: List[str]) -> Dict[str, Any]:
    """Record a research note/strategy idea (persists in memory)."""
    if not topic or not summary:
      return {"error": "topic and summary required"}
    return memory.save_plan(title=f"Research: {topic}", summary=summary, actions=actions, author="Supervisor Agent")

  @function_tool
  async def set_auto_trigger(
    symbol: str,
    direction: str,
    rationale: str,
    target_price: float | None = None,
    stop_price: float | None = None,
  ) -> Dict[str, Any]:
    """Store an auto-buy/sell trigger idea (persists to disk for follow-up by future runs)."""
    return memory.save_trigger(
      symbol=normalize_symbol(symbol),
      direction=direction,
      rationale=rationale,
      target_price=target_price,
      stop_price=stop_price,
    )

  @function_tool
  async def list_triggers() -> Dict[str, Any]:
    """List stored auto triggers (buy/sell ideas) for follow-up."""
    return {"triggers": memory.latest_triggers()}

  @function_tool
  async def list_coins() -> Dict[str, Any]:
    """List the current active coin universe."""
    return {"coins": memory.get_coins(default=cfg.trading.coins)}

  @function_tool
  async def add_coin(symbol: str, reason: str) -> Dict[str, Any]:
    """Add a coin to the active universe (requires reason)."""
    if not symbol:
      return {"error": "symbol required"}
    entry = memory.add_coin(normalize_symbol(symbol), reason)
    return {"added": entry, "coins": memory.get_coins(default=cfg.trading.coins)}

  @function_tool
  async def remove_coin(symbol: str, reason: str, exit_plan: str) -> Dict[str, Any]:
    """Remove a coin from the active universe with an exit plan noted."""
    if not symbol:
      return {"error": "symbol required"}
    if not exit_plan:
      return {"error": "exit_plan required to remove coin"}
    entry = memory.remove_coin(normalize_symbol(symbol), reason, exit_plan)
    return {"removed": entry, "coins": memory.get_coins(default=cfg.trading.coins)}

  @function_tool
  async def add_source(name: str, url: str, reason: str) -> Dict[str, Any]:
    """Add a data/research source to memory."""
    if not name or not url:
      return {"error": "name and url required"}
    return {"source": memory.save_plan(title=f"Source: {name}", summary=url, actions=[reason or ""], author="Supervisor Agent")}

  @function_tool
  async def remove_source(name: str, reason: str) -> Dict[str, Any]:
    """Mark a data/research source as removed."""
    if not name or not reason:
      return {"error": "name and reason required"}
    return {"removed": memory.save_plan(title=f"Removed Source: {name}", summary=reason, actions=[], author="Supervisor Agent")}

  # ── Note tools ────────────────────────────────────────────────────

  @function_tool
  async def write_temporary_note(content: str) -> Dict[str, Any]:
    """Write a temporary note for the trading agent. It will be read once on the next run, then deleted."""
    if not content:
      return {"error": "Content required"}
    entry = memory.add_temporary_note(content)
    return {"status": "saved", "note": entry}

  @function_tool
  async def write_permanent_note(content: str) -> Dict[str, Any]:
    """Write a permanent note that becomes part of the trading agent's system prompt until deleted."""
    if not content:
      return {"error": "Content required"}
    entry = memory.add_permanent_note(content)
    return {"status": "saved", "note": entry}

  @function_tool
  async def list_notes() -> Dict[str, Any]:
    """List all supervisor notes (both temporary and permanent)."""
    return memory.list_all_notes()

  @function_tool
  async def delete_permanent_note(index: int) -> Dict[str, Any]:
    """Delete a permanent note by its index (0-based). Use list_notes first to see indices."""
    return memory.delete_permanent_note(index)

  # ── Account / Balance tools ───────────────────────────────────────

  @function_tool
  async def get_account_snapshot() -> Dict[str, Any]:
    """Fetch live KuCoin account balances, futures overview, and open stop orders."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    result: Dict[str, Any] = {}
    try:
      accounts = kucoin.get_trade_accounts()
      balances: Dict[str, float] = {}
      for acct in accounts:
        balances[acct.currency] = balances.get(acct.currency, 0.0) + float(acct.available or 0)
      result["spot_balances"] = balances
    except Exception as exc:
      result["spot_error"] = str(exc)
    if kucoin_futures:
      try:
        overview = kucoin_futures.get_account_overview()
        result["futures_overview"] = overview
      except Exception as exc:
        result["futures_error"] = str(exc)
      try:
        positions = kucoin_futures.list_positions() or []
        result["futures_positions"] = positions
      except Exception as exc:
        result["futures_positions_error"] = str(exc)
    try:
      stops = kucoin.list_stop_orders(status="active")
      result["spot_stop_orders"] = stops
    except Exception as exc:
      result["spot_stops_error"] = str(exc)
    return result

  @function_tool
  async def fetch_account_state() -> Dict[str, Any]:
    """Get up-to-date balances for all spot, funding, financial, and futures accounts."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    try:
      spot_accounts_fresh = kucoin.get_trade_accounts()
      main_accounts_fresh = kucoin.get_accounts("main")
      fin_accounts_fresh = kucoin.get_financial_accounts()
    except Exception as exc:
      return {"error": f"account_fetch_failed: {exc}"}
    futures_overview = None
    futures_error = None
    if cfg.kucoin_futures.enabled and kucoin_futures:
      try:
        futures_overview = kucoin_futures.get_account_overview()
      except Exception as exc:
        futures_error = f"futures_fetch_failed: {exc}"
    return _build_compact_account_state(
      spot_totals=_aggregate_account_totals(spot_accounts_fresh),
      funding_totals=_aggregate_account_totals(main_accounts_fresh),
      financial_totals=_aggregate_account_totals(fin_accounts_fresh),
      futures_overview=futures_overview,
      futures_enabled=bool(cfg.kucoin_futures.enabled),
      futures_error=futures_error,
    )

  @function_tool
  async def transfer_funds(
    direction: str,
    currency: str = "USDT",
    amount: float = 0.0,
  ) -> Dict[str, Any]:
    """Transfer funds between spot (trade), futures (contract), and financial (earn/pool). Directions: spot_to_futures, futures_to_spot, financial_to_spot, spot_to_financial, financial_to_futures, futures_to_financial."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    dir_norm = (direction or "").lower()
    allowed_dirs = {
      "spot_to_futures": ("trade", "contract"),
      "futures_to_spot": ("contract", "trade"),
      "financial_to_spot": ("financial", "trade"),
      "spot_to_financial": ("trade", "financial"),
      "financial_to_futures": ("financial", "contract"),
      "futures_to_financial": ("contract", "financial"),
    }
    if dir_norm not in allowed_dirs:
      return {"error": "Invalid direction", "allowed": sorted(allowed_dirs)}
    try:
      amt = float(amount or 0)
    except (TypeError, ValueError):
      return {"error": "Invalid amount"}
    if amt <= 0:
      return {"error": "Amount must be positive"}
    from_acct, to_acct = allowed_dirs[dir_norm]
    try:
      result = kucoin.transfer_funds(currency=currency.upper(), amount=amt, from_account=from_acct, to_account=to_acct)
      return {"transferred": result, "direction": dir_norm, "currency": currency.upper(), "amount": amt}
    except Exception as exc:
      return {"error": str(exc), "direction": dir_norm}

  @function_tool
  async def refresh_fee_rates() -> Dict[str, Any]:
    """Fetch latest fee rates (spot base fee)."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    try:
      base_fee = kucoin.get_base_fee()
      spot_taker = float(base_fee.get("takerFeeRate") or 0.001)
      spot_maker = float(base_fee.get("makerFeeRate") or 0.001)
    except Exception as exc:
      return {"error": f"base_fee_failed: {exc}"}
    entry = memory.save_fee_info(spot_taker=spot_taker, spot_maker=spot_maker)
    return {"fee": entry}

  # ── Market data tools ─────────────────────────────────────────────

  @function_tool
  async def fetch_recent_candles(
    symbol: str,
    interval: str = "1min",
    lookback_minutes: int = 120,
  ) -> Dict[str, Any]:
    """Fetch recent candles for symbol. Interval: 1min, 5min, 15min, 1hour. Caps to 500 rows."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    symbol = normalize_symbol(symbol)
    interval_seconds = {"1min": 60, "5min": 300, "15min": 900, "1hour": 3600}
    if interval not in interval_seconds:
      return {"error": "Invalid interval", "allowed": list(interval_seconds.keys())}
    lookback_min = max(1, min(int(lookback_minutes or 0), 720))
    end_at = int(time.time())
    max_points = 500
    interval_sec = interval_seconds[interval]
    points = min(max_points, max(1, int(lookback_min * 60 / interval_sec)))
    start_at = end_at - points * interval_sec
    candles = kucoin.get_candles(symbol, interval=interval, start_at=start_at, end_at=end_at)
    return {"symbol": symbol, "interval": interval, "startAt": start_at, "endAt": end_at, "points": candles[:max_points], "rows": len(candles)}

  @function_tool
  async def fetch_orderbook(symbol: str, depth: int = 20) -> Dict[str, Any]:
    """Fetch level2 orderbook snapshot (depth 20 or 100)."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    symbol = normalize_symbol(symbol)
    depth_safe = 20 if depth <= 20 else 100
    ob = kucoin.get_orderbook_levels(symbol, depth=depth_safe)
    ob["bids"] = ob.get("bids", [])[:depth_safe]
    ob["asks"] = ob.get("asks", [])[:depth_safe]
    return {"symbol": symbol, "depth": depth_safe, "orderbook": ob}

  @function_tool
  async def analyze_market_context(
    symbol: str,
    fast_interval: str = "15min",
    slow_interval: str = "1hour",
    lookback_minutes: int = 360,
  ) -> Dict[str, Any]:
    """Compute EMA/RSI/MACD/ATR/Bollinger/VWAP across two intervals and summarize bias."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    symbol = normalize_symbol(symbol)
    interval_order: list[str] = []
    for iv in [fast_interval, slow_interval]:
      if iv not in INTERVAL_SECONDS:
        return {"error": "Invalid interval", "allowed": list(INTERVAL_SECONDS.keys())}
      if iv not in interval_order:
        interval_order.append(iv)
    snapshots: list[Dict[str, Any]] = []
    end_at = int(time.time())
    for iv in interval_order:
      interval_sec = INTERVAL_SECONDS[iv]
      lookback_min = max(120, min(int(lookback_minutes or 0), 720))
      points = min(500, max(50, int(lookback_min * 60 / interval_sec)))
      start_at = end_at - points * interval_sec
      candles = kucoin.get_candles(symbol, interval=iv, start_at=start_at, end_at=end_at)
      if not candles:
        return {"error": "No candles returned", "interval": iv}
      try:
        df = candles_to_dataframe(candles)
        snapshot = summarize_interval(df, iv)
        snapshot["rows"] = len(df)
        snapshots.append(snapshot)
      except Exception as exc:
        return {"error": str(exc), "interval": iv}
    summary = summarize_multi_timeframe(snapshots)
    return {"symbol": symbol, "snapshots": snapshots, "summary": summary}

  @function_tool
  async def plan_spot_position(
    symbol: str,
    risk_pct: float | None = None,
    atr_multiple: float = 1.5,
    target_rr: float = 2.0,
    entry_price: float | None = None,
  ) -> Dict[str, Any]:
    """Size a spot trade using risk-per-trade % with ATR-based stop/target."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    symbol = normalize_symbol(symbol)
    spot_accounts = kucoin.get_trade_accounts()
    balance_usdt = sum(float(b.available or 0) for b in spot_accounts if b.currency == "USDT")
    if balance_usdt <= 0:
      return {"error": "No USDT balance available"}
    risk_fraction = risk_pct if risk_pct is not None else cfg.trading.risk_per_trade_pct
    risk_fraction = max(0.0, float(risk_fraction))
    risk_dollars = balance_usdt * risk_fraction
    if risk_dollars <= 0:
      return {"error": "Risk dollars computed to zero", "riskPct": risk_fraction}
    fee_rate = 0.001
    stored_fees = memory.latest_fees() or {}
    if stored_fees.get("spot_taker"):
      fee_rate = float(stored_fees["spot_taker"])
    spendable = max(0.0, balance_usdt * 0.90)
    if entry_price:
      price = float(entry_price)
    else:
      ticker = kucoin.get_ticker(symbol)
      price = float(ticker.price)
    lookback_min = 180
    end_at = int(time.time())
    interval = "15min"
    interval_sec = INTERVAL_SECONDS[interval]
    points_count = min(500, max(50, int(lookback_min * 60 / interval_sec)))
    start_at = end_at - points_count * interval_sec
    candles = kucoin.get_candles(symbol, interval=interval, start_at=start_at, end_at=end_at)
    if not candles:
      return {"error": "No candles returned for ATR sizing", "interval": interval}
    df = candles_to_dataframe(candles)
    enriched = compute_indicators(df)
    atr_raw = enriched["atr"].iloc[-1]
    try:
      atr_val = float(atr_raw)
    except Exception:
      atr_val = None
    if atr_val != atr_val:
      atr_val = None
    stop_distance = atr_val * atr_multiple if atr_val else price * 0.005
    if stop_distance <= 0:
      return {"error": "Stop distance invalid", "atr": atr_val}
    raw_size = risk_dollars / stop_distance
    notional_unclipped = raw_size * price
    cap_notional = min(cfg.trading.max_position_usd, spendable)
    notional = min(notional_unclipped, cap_notional)
    size = notional / price if price else 0
    stop_price = max(0.0, price - stop_distance)
    target_price = price + stop_distance * target_rr
    warnings: list[str] = []
    if atr_val and atr_val / price * 100 >= 5:
      warnings.append("Volatility elevated (ATR% >=5); consider smaller size or skip.")
    if notional_unclipped > cap_notional:
      warnings.append("Size clipped by maxPositionUsd or 10% reserve.")
    return {
      "symbol": symbol, "price": price, "riskPct": risk_fraction, "riskDollars": risk_dollars,
      "atr": atr_val, "atrMultiple": atr_multiple, "stopDistance": stop_distance,
      "stopPrice": stop_price, "targetPrice": target_price, "targetRR": target_rr,
      "size": size, "notional": notional, "feeRate": fee_rate,
      "notionalWithFee": notional * (1 + fee_rate), "spendable": spendable,
      "warnings": warnings,
    }

  @function_tool
  async def fetch_kucoin_news(limit: int = 10) -> Dict[str, Any]:
    """Fetch latest Kucoin news RSS (lang=en). Returns list of {title, link, pubDate}."""
    url = "https://www.kucoin.com/rss/news?lang=en"
    try:
      resp = requests.get(url, timeout=10)
      resp.raise_for_status()
    except Exception as exc:
      return {"error": f"fetch_failed: {exc}"}
    try:
      import xml.etree.ElementTree as ET
      root = ET.fromstring(resp.content)
      items: List[Dict[str, Any]] = []
      for item in root.findall(".//item")[: max(1, min(int(limit or 0), 20))]:
        title_el = item.find("title")
        link_el = item.find("link")
        date_el = item.find("pubDate")
        items.append({
          "title": (title_el.text or "").strip() if title_el is not None else "",
          "link": (link_el.text or "").strip() if link_el is not None else "",
          "pubDate": (date_el.text or "").strip() if date_el is not None else "",
        })
      return {"items": items}
    except Exception as exc:
      return {"error": f"parse_failed: {exc}"}

  # ── Spot order tools ──────────────────────────────────────────────

  @function_tool
  async def place_market_order(
    symbol: str,
    side: str,
    funds: float,
    rationale: str = "",
  ) -> Dict[str, Any]:
    """Place a spot market order. For buys: uses funds (USDT). For sells: sells equivalent base asset amount. Supervisor bypass — no PnL gates."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    symbol = normalize_symbol(symbol)
    try:
      funds_val = float(funds or 0)
    except (TypeError, ValueError):
      return {"error": "Invalid funds"}
    if funds_val <= 0 or not symbol:
      return {"error": "Invalid funds or symbol"}
    side_norm = (side or "").lower()
    if side_norm not in ("buy", "sell"):
      return {"error": "Side must be 'buy' or 'sell'"}
    ticker = kucoin.get_ticker(symbol)
    price = float(ticker.price)
    if price <= 0:
      return {"error": "Invalid price"}
    sym_info = {}
    try:
      sym_info = kucoin.get_symbol_info(symbol)
    except Exception:
      pass
    base_increment = sym_info.get("baseIncrement", "0.00000001")
    if side_norm == "buy":
      order_req = KucoinOrderRequest(symbol=symbol, side="buy", type="market", funds=f"{funds_val:.2f}", clientOid=str(uuid.uuid4()))
    else:
      size_est = funds_val / price
      # Clamp to available balance
      accounts = kucoin.get_trade_accounts()
      base_cur = _base_currency(symbol)
      available = sum(float(b.available or 0) for b in accounts if b.currency == base_cur)
      if available <= 0:
        return {"rejected": True, "reason": "No available balance to sell"}
      if size_est > available:
        size_est = available
      # Cancel existing sell-side stops to release holds
      try:
        active_stops = kucoin.list_stop_orders(status="active", symbol=symbol) or []
        for order in active_stops:
          if not isinstance(order, dict):
            continue
          if str(order.get("side") or "").lower() != "sell":
            continue
          oid = str(order.get("id") or order.get("orderId") or "").strip()
          coid = str(order.get("clientOid") or "").strip() or None
          if oid or coid:
            try:
              kucoin.cancel_stop_order(order_id=oid or None, client_oid=coid)
            except Exception:
              pass
      except Exception:
        pass
      # Re-fetch available after cancelling stops
      accounts = kucoin.get_trade_accounts()
      available = sum(float(b.available or 0) for b in accounts if b.currency == base_cur)
      if size_est > available:
        size_est = available
      truncated = _truncate_to_increment(size_est, base_increment)
      order_req = KucoinOrderRequest(symbol=symbol, side="sell", type="market", size=truncated, clientOid=str(uuid.uuid4()))
    if cfg.trading.paper_trading:
      record = memory.record_trade(symbol, side_norm, funds_val, paper=True, price=price)
      return {"paper": True, "orderRequest": order_req.__dict__, "tradeRecord": record}
    try:
      res = kucoin.place_order(order_req).__dict__
      record = memory.record_trade(symbol, side_norm, funds_val, paper=False, price=price)
      res["tradeRecord"] = record
      res["rationale"] = rationale
      return res
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__}

  @function_tool
  async def place_spot_stop_order(
    symbol: str,
    side: str,
    stop_price: float,
    stop_price_type: str = "MP",
    order_type: str = "market",
    size: float | None = None,
    funds: float | None = None,
    limit_price: float | None = None,
  ) -> Dict[str, Any]:
    """Place a spot stop order (stop-loss or take-profit). stop_price_type: TP (trigger when price rises) or MP (falls)."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    symbol = normalize_symbol(symbol)
    side_norm = (side or "").lower()
    if side_norm not in ("buy", "sell"):
      return {"error": "Side must be 'buy' or 'sell'"}
    sym_info = {}
    try:
      sym_info = kucoin.get_symbol_info(symbol)
    except Exception:
      pass
    base_increment = sym_info.get("baseIncrement", "0.00000001")
    order = KucoinOrderRequest(
      symbol=symbol,
      side=side_norm,
      type=order_type if order_type in ("market", "limit") else "market",
      size=_truncate_to_increment(float(size), base_increment) if size else None,
      funds=f"{float(funds):.2f}" if funds and not size else None,
      price=f"{float(limit_price)}" if limit_price and order_type == "limit" else None,
      stopPrice=f"{float(stop_price)}",
      stopPriceType=stop_price_type.upper() if stop_price_type else "MP",
      clientOid=str(uuid.uuid4()),
    )
    if cfg.trading.paper_trading:
      return {"paper": True, "orderRequest": order.__dict__}
    try:
      res = kucoin.place_stop_order(order).__dict__
      return {"placed": res, "orderRequest": order.__dict__}
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order.__dict__}

  @function_tool
  async def cancel_spot_stop_order(order_id: str | None = None, client_oid: str | None = None) -> Dict[str, Any]:
    """Cancel a spot stop order by orderId or clientOid."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    if cfg.trading.paper_trading:
      return {"paper": True, "cancelled": {"orderId": order_id, "clientOid": client_oid}}
    try:
      res = kucoin.cancel_stop_order(order_id=order_id, client_oid=client_oid)
      return {"cancelled": res, "orderId": order_id, "clientOid": client_oid}
    except Exception as exc:
      return {"error": str(exc), "orderId": order_id, "clientOid": client_oid}

  @function_tool
  async def list_spot_stop_orders(status: str = "active", symbol: str | None = None) -> Dict[str, Any]:
    """List spot stop orders; status usually 'active' or 'done'."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    try:
      orders = kucoin.list_stop_orders(status=status, symbol=normalize_symbol(symbol) if symbol else None)
      return {"orders": orders, "status": status, "symbol": symbol}
    except Exception as exc:
      return {"error": str(exc), "status": status, "symbol": symbol}

  @function_tool
  async def set_spot_position_protection(
    symbol: str,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    cancel_existing: bool = True,
  ) -> Dict[str, Any]:
    """Add or replace TP/SL for an existing open spot position using sell stop orders."""
    if not kucoin:
      return {"error": "KuCoin client not available"}
    symbol = normalize_symbol(symbol)
    tp_val = _to_float(take_profit_price)
    sl_val = _to_float(stop_loss_price)
    if not tp_val and not sl_val:
      return {"error": "At least one of take_profit_price or stop_loss_price is required"}
    # Get position size from live balance
    accounts = kucoin.get_trade_accounts()
    base_cur = _base_currency(symbol)
    position_size = sum(float(b.available or 0) for b in accounts if b.currency == base_cur)
    if position_size <= 0:
      return {"error": "No open spot position", "symbol": symbol}
    sym_info = {}
    try:
      sym_info = kucoin.get_symbol_info(symbol)
    except Exception:
      pass
    base_increment = sym_info.get("baseIncrement", "0.00000001")
    cancelled: list[Dict[str, Any]] = []
    if cancel_existing:
      try:
        active_orders = kucoin.list_stop_orders(status="active", symbol=symbol) or []
        for order in active_orders:
          if not isinstance(order, dict):
            continue
          if str(order.get("side") or "").lower() != "sell":
            continue
          oid = str(order.get("id") or order.get("orderId") or "").strip()
          coid = str(order.get("clientOid") or "").strip() or None
          if oid or coid:
            try:
              res = kucoin.cancel_stop_order(order_id=oid or None, client_oid=coid)
              cancelled.append({"orderId": oid, "response": res})
            except Exception as exc:
              cancelled.append({"orderId": oid, "error": str(exc)})
      except Exception as exc:
        cancelled.append({"stage": "list", "error": str(exc)})
    bracket: Dict[str, Any] = {}
    size_str = _truncate_to_increment(position_size, base_increment)
    if sl_val:
      order = KucoinOrderRequest(
        symbol=symbol, side="sell", type="market",
        size=size_str, stopPrice=f"{sl_val}", stopPriceType="MP",
        clientOid=str(uuid.uuid4()),
      )
      try:
        res = kucoin.place_stop_order(order).__dict__
        bracket["stopLoss"] = {"placed": res}
      except Exception as exc:
        bracket["stopLoss"] = {"error": str(exc)}
    if tp_val:
      order = KucoinOrderRequest(
        symbol=symbol, side="sell", type="limit",
        size=size_str, price=f"{tp_val}", stopPrice=f"{tp_val}", stopPriceType="TP",
        clientOid=str(uuid.uuid4()),
      )
      try:
        res = kucoin.place_stop_order(order).__dict__
        bracket["takeProfit"] = {"placed": res}
      except Exception as exc:
        bracket["takeProfit"] = {"error": str(exc)}
    return {"symbol": symbol, "positionSize": position_size, "bracket": bracket, "cancelled": cancelled}

  # ── Futures order tools ───────────────────────────────────────────

  @function_tool
  async def place_futures_market_order(
    symbol: str,
    side: str,
    notional_usd: float,
    leverage: float = 1.0,
    reduce_only: bool = False,
    rationale: str = "",
  ) -> Dict[str, Any]:
    """Place a futures market order. Supervisor bypass — no PnL gates. Computes contract size from notional."""
    if not kucoin or not kucoin_futures:
      return {"error": "Futures client not available"}
    symbol = normalize_symbol(symbol)
    futures_symbol = _to_futures_symbol(symbol)
    if not futures_symbol:
      return {"error": "Invalid symbol for futures", "symbol": symbol}
    side_norm = (side or "").lower()
    if side_norm not in ("buy", "sell"):
      return {"error": "Side must be 'buy' or 'sell'"}
    try:
      notional = float(notional_usd or 0)
      lev = max(1.0, min(float(leverage or 1), cfg.trading.max_leverage))
    except (TypeError, ValueError):
      return {"error": "Invalid notional or leverage"}
    if notional <= 0:
      return {"error": "Notional must be positive"}
    ticker = kucoin.get_ticker(symbol)
    price = float(ticker.price)
    if price <= 0:
      return {"error": "Invalid price"}
    contract_info = _get_contract_spec(futures_symbol)
    if not contract_info:
      return {"error": f"Contract spec unavailable for {futures_symbol}"}
    multiplier = _to_float(contract_info.get("multiplier"))
    lot_size = int(contract_info.get("lotSize") or 1)
    if multiplier <= 0:
      return {"error": "Invalid contract multiplier"}
    base_size = notional / price
    contracts_raw = base_size / multiplier
    lot = max(1, lot_size)
    contracts = int(math.ceil(contracts_raw / lot) * lot)
    order_req = KucoinFuturesOrderRequest(
      symbol=futures_symbol, side=side_norm, type="market",
      leverage=str(int(lev)), size=str(contracts),
      clientOid=str(uuid.uuid4()), reduceOnly=reduce_only,
    )
    if cfg.trading.paper_trading:
      record = memory.record_trade(symbol, side_norm, notional, paper=True, price=price, size=contracts * multiplier)
      return {"paper": True, "orderRequest": order_req.__dict__, "tradeRecord": record}
    try:
      kucoin_futures.set_leverage(futures_symbol, lev)
    except Exception:
      pass
    try:
      res = kucoin_futures.place_order(order_req).__dict__
      record = memory.record_trade(symbol, side_norm, notional, paper=False, price=price, size=contracts * multiplier)
      res["tradeRecord"] = record
      res["rationale"] = rationale
      res["futuresSymbol"] = futures_symbol
      res["contracts"] = contracts
      res["appliedLeverage"] = lev
      return res
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__}

  @function_tool
  async def place_futures_stop_order(
    symbol: str,
    side: str,
    leverage: float,
    size: float,
    stop_price: float,
    stop: str = "down",
    stop_price_type: str = "MP",
    reduce_only: bool = False,
    order_type: str = "market",
    limit_price: float | None = None,
  ) -> Dict[str, Any]:
    """Place a futures stop/TP/SL order. stop: 'up' or 'down'. stop_price_type: TP/MP/IP."""
    if not kucoin_futures:
      return {"error": "Futures client not available"}
    symbol = normalize_symbol(symbol)
    futures_symbol = _to_futures_symbol(symbol)
    if not futures_symbol:
      return {"error": "Invalid symbol for futures"}
    contract_info = _get_contract_spec(futures_symbol)
    if not contract_info:
      return {"error": f"Contract spec unavailable for {futures_symbol}"}
    multiplier = _to_float(contract_info.get("multiplier"))
    lot_size = int(contract_info.get("lotSize") or 1)
    if multiplier <= 0:
      return {"error": "Invalid contract multiplier"}
    contracts = int(math.ceil((float(size) / multiplier) / max(1, lot_size)) * max(1, lot_size))
    order_req = KucoinFuturesOrderRequest(
      symbol=futures_symbol, side=(side or "sell").lower(), type=order_type,
      leverage=str(int(max(1, float(leverage)))), size=str(contracts),
      price=f"{float(limit_price)}" if limit_price and order_type == "limit" else None,
      stop=stop if stop in ("up", "down") else "down",
      stopPriceType=stop_price_type.upper() if stop_price_type else "MP",
      clientOid=str(uuid.uuid4()),
      reduceOnly=reduce_only, closeOrder=reduce_only,
    )
    order_req.stopPrice = f"{float(stop_price)}"
    if cfg.trading.paper_trading:
      return {"paper": True, "orderRequest": order_req.__dict__}
    try:
      res = kucoin_futures.place_order(order_req).__dict__
      return {"placed": res, "orderRequest": order_req.__dict__}
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__}

  @function_tool
  async def cancel_futures_order(order_id: str, symbol: str | None = None) -> Dict[str, Any]:
    """Cancel a futures order (stop or regular) by orderId."""
    if not kucoin_futures:
      return {"error": "Futures client not available"}
    if cfg.trading.paper_trading:
      return {"paper": True, "cancelled": {"orderId": order_id, "symbol": symbol}}
    try:
      futures_sym = _to_futures_symbol(normalize_symbol(symbol)) if symbol else None
      res = kucoin_futures.cancel_order(order_id, symbol=futures_sym)
      return {"cancelled": res, "orderId": order_id}
    except Exception as exc:
      return {"error": str(exc), "orderId": order_id}

  @function_tool
  async def list_futures_stop_orders(status: str = "active", symbol: str | None = None) -> Dict[str, Any]:
    """List futures stop orders; status: active/done."""
    if not kucoin_futures:
      return {"error": "Futures client not available"}
    try:
      futures_sym = _to_futures_symbol(normalize_symbol(symbol)) if symbol else None
      orders = kucoin_futures.list_stop_orders(status=status, symbol=futures_sym)
      return {"orders": orders, "status": status, "symbol": symbol}
    except Exception as exc:
      return {"error": str(exc)}

  @function_tool
  async def list_futures_positions(status: str | None = None) -> Dict[str, Any]:
    """List current futures positions (live fetch)."""
    if not kucoin_futures:
      return {"error": "Futures client not available"}
    try:
      positions = kucoin_futures.list_positions(status=status)
      return {"positions": positions or []}
    except Exception as exc:
      return {"error": str(exc)}

  @function_tool
  async def set_futures_position_protection(
    symbol: str,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    cancel_existing: bool = True,
  ) -> Dict[str, Any]:
    """Add or replace TP/SL for an existing open futures position using reduce-only close orders."""
    if not kucoin_futures:
      return {"error": "Futures client not available"}
    symbol = normalize_symbol(symbol)
    futures_symbol = _to_futures_symbol(symbol)
    if not futures_symbol:
      return {"error": "Invalid symbol for futures"}
    tp_val = _to_float(take_profit_price)
    sl_val = _to_float(stop_loss_price)
    if not tp_val and not sl_val:
      return {"error": "At least one of take_profit_price or stop_loss_price is required"}
    try:
      position = kucoin_futures.get_position(futures_symbol)
    except Exception as exc:
      return {"error": f"Position lookup failed: {exc}"}
    if not isinstance(position, dict):
      return {"error": "Unexpected position payload"}
    current_qty = _to_float(position.get("currentQty"))
    if abs(current_qty) <= 0:
      return {"error": "No open futures position", "symbol": symbol}
    contract_info = _get_contract_spec(futures_symbol)
    if not contract_info:
      return {"error": f"Contract spec unavailable for {futures_symbol}"}
    multiplier = _to_float(contract_info.get("multiplier"))
    if multiplier <= 0:
      return {"error": "Invalid contract multiplier"}
    base_size = abs(current_qty) * multiplier
    leverage_val = _to_float(position.get("realLeverage")) or _to_float(position.get("leverage")) or cfg.trading.max_leverage
    if leverage_val <= 0:
      leverage_val = cfg.trading.max_leverage or 1.0
    exit_side = "sell" if current_qty > 0 else "buy"
    tp_stop = "up" if current_qty > 0 else "down"
    sl_stop = "down" if current_qty > 0 else "up"
    cancelled: list[Dict[str, Any]] = []
    if cancel_existing:
      try:
        active_orders = kucoin_futures.list_stop_orders(status="active", symbol=futures_symbol) or []
        for order in active_orders:
          if not isinstance(order, dict):
            continue
          if str(order.get("side") or "").lower() != exit_side:
            continue
          oid = str(order.get("id") or order.get("orderId") or "").strip()
          if oid:
            try:
              res = kucoin_futures.cancel_order(oid, symbol=futures_symbol)
              cancelled.append({"orderId": oid, "response": res})
            except Exception as exc:
              cancelled.append({"orderId": oid, "error": str(exc)})
      except Exception as exc:
        cancelled.append({"stage": "list", "error": str(exc)})
    lot_size = int(contract_info.get("lotSize") or 1)
    contracts = int(math.ceil((base_size / multiplier) / max(1, lot_size)) * max(1, lot_size))
    bracket: Dict[str, Any] = {}
    if sl_val:
      sl_req = KucoinFuturesOrderRequest(
        symbol=futures_symbol, side=exit_side, type="market",
        leverage=str(int(leverage_val)), size=str(contracts),
        stop=sl_stop, stopPriceType="MP",
        clientOid=f"{futures_symbol.lower()}-sl-{uuid.uuid4().hex[:12]}",
        reduceOnly=True, closeOrder=True,
      )
      sl_req.stopPrice = f"{sl_val}"
      try:
        res = kucoin_futures.place_order(sl_req).__dict__
        bracket["stopLoss"] = {"placed": res}
      except Exception as exc:
        bracket["stopLoss"] = {"error": str(exc)}
    if tp_val:
      tp_req = KucoinFuturesOrderRequest(
        symbol=futures_symbol, side=exit_side, type="market",
        leverage=str(int(leverage_val)), size=str(contracts),
        stop=tp_stop, stopPriceType="MP",
        clientOid=f"{futures_symbol.lower()}-tp-{uuid.uuid4().hex[:12]}",
        reduceOnly=True, closeOrder=True,
      )
      tp_req.stopPrice = f"{tp_val}"
      try:
        res = kucoin_futures.place_order(tp_req).__dict__
        bracket["takeProfit"] = {"placed": res}
      except Exception as exc:
        bracket["takeProfit"] = {"error": str(exc)}
    return {"symbol": symbol, "futuresSymbol": futures_symbol, "currentQty": current_qty, "bracket": bracket, "cancelled": cancelled}

  # ── Instructions ──────────────────────────────────────────────────

  instructions = (
    "You are the Supervisor Agent for trAIde, an automated crypto trading bot. "
    "You help the bot owner monitor, understand, and influence the trading bot's behavior via Telegram.\n\n"

    "## Capabilities:\n"
    "- Read and search application logs\n"
    "- Read agent memory (trades, decisions, plans, positions, performance)\n"
    "- Read source code files\n"
    "- View non-secret configuration\n"
    "- Fetch live KuCoin account balances and positions\n"
    "- Write notes to the trading agent:\n"
    "  - **Temporary notes**: read once by the trading agent on its next run, then auto-deleted\n"
    "  - **Permanent notes**: added to the trading agent's system prompt until manually deleted\n\n"

    "## Guidelines:\n"
    "- Keep responses concise — Telegram messages are limited to 4096 characters.\n"
    "- When asked about status, positions, or performance, always use the appropriate tools to fetch live data.\n"
    "- You cannot place trades directly. Use notes to influence the trading agent's behavior.\n"
    "- If asked to analyze something, use the available tools to gather data first, then provide your analysis.\n"
    "- You have conversation memory. Use prior context when relevant, but always fetch fresh data via tools.\n\n"

    "## Writing notes:\n"
    "- Relay the owner's instructions verbatim. Do not rephrase commands into suggestions.\n"
    "- Temporary notes: top-priority directive for the next run.\n"
    "- Permanent notes: ongoing rules in the trading agent's system prompt.\n"
  )

  conv_memory = ConversationMemory(
    file_path=Path(cfg.memory_file).with_name(".supervisor_conversation.json"),
    recent_count=3,
    max_context_message_chars=500,
  )

  context = conv_memory.get_context()
  if context:
    instructions += "\n## Conversation History:\n" + context + "\n"

  supervisor_agent = Agent(
    name="Supervisor Agent",
    instructions=instructions,
    tools=[
      WebSearchTool(search_context_size="high"),
      # Logging / source / config
      read_logs,
      search_logs,
      read_source_file,
      list_source_files,
      get_config_summary,
      # Memory / state
      read_memory,
      get_performance_summary,
      get_positions,
      get_recent_decisions,
      get_recent_trades,
      latest_plan,
      latest_items,
      save_trade_plan,
      clear_plans,
      log_sentiment,
      log_decision,
      log_research,
      set_auto_trigger,
      list_triggers,
      list_coins,
      add_coin,
      remove_coin,
      add_source,
      remove_source,
      # Notes
      write_temporary_note,
      write_permanent_note,
      list_notes,
      delete_permanent_note,
      # Account / balance
      get_account_snapshot,
      fetch_account_state,
      transfer_funds,
      refresh_fee_rates,
      # Market data
      fetch_recent_candles,
      fetch_orderbook,
      analyze_market_context,
      plan_spot_position,
      fetch_kucoin_news,
      # Spot orders
      place_market_order,
      place_spot_stop_order,
      cancel_spot_stop_order,
      list_spot_stop_orders,
      set_spot_position_protection,
      # Futures orders
      place_futures_market_order,
      place_futures_stop_order,
      cancel_futures_order,
      list_futures_stop_orders,
      list_futures_positions,
      set_futures_position_protection,
    ],
    model=model,
  )

  result = asyncio.run(Runner.run(supervisor_agent, message_text, max_turns=30))
  response = str(result.final_output)

  conv_memory.add_exchange(message_text, response)

  conv_memory.compact_with_llm(model, context="a crypto trading bot owner and a supervisor assistant")

  return response
