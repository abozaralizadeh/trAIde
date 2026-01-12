from __future__ import annotations

import contextlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from agents import (
  Agent,
  Runner,
  OpenAIResponsesModel,
  add_trace_processor,
  set_tracing_export_api_key,
  gen_trace_id,
)
from agents.items import ToolCallOutputItem
from agents.tool import WebSearchTool, function_tool
from agents.tracing.processors import BatchTraceProcessor, ConsoleSpanExporter
from agents.tracing.setup import get_trace_provider
from openai import AsyncAzureOpenAI

_TRACING_INITIALIZED = False

from .analytics import (
  INTERVAL_SECONDS,
  candles_to_dataframe,
  compute_indicators,
  summarize_interval,
  summarize_multi_timeframe,
)
from .config import AppConfig
from .kucoin import (
  KucoinAccount,
  KucoinClient,
  KucoinFuturesClient,
  KucoinFuturesOrderRequest,
  KucoinOrderRequest,
  KucoinTicker,
)
from .memory import MemoryStore


class _RedactingConsoleExporter(ConsoleSpanExporter):
  """Redact bulky outputs (candles/orderbooks) before console export."""

  def export(self, spans):
    for s in spans:
      sd = getattr(s, "span_data", None)
      if getattr(sd, "type", "") == "function" and getattr(sd, "name", "") in (
        "fetch_recent_candles",
        "fetch_orderbook",
      ):
        if hasattr(sd, "output"):
          sd.output = "(redacted: large payload)"
    return super().export(spans)


def _remove_trace_processor(proc: Any) -> None:
  """Best-effort removal of a tracing processor from the provider."""
  try:
    provider = get_trace_provider()
    # Access the internal processor list safely
    if hasattr(provider, "_multi_processor"):
        target_list = getattr(provider._multi_processor, "_processors", None)
        if target_list is not None and proc in target_list:
            target_list.remove(proc)
            return
            
    # Fallback for different SDK versions
    current_processors = getattr(provider, "_processors", [])
    if proc in current_processors:
        current_processors.remove(proc)
        
  except Exception as exc:
    print("Warning: failed to remove trace processor:", exc)


def setup_tracing(cfg: AppConfig) -> None:
  """Register tracing processors once at startup."""
  global _TRACING_INITIALIZED
  if _TRACING_INITIALIZED:
    return
  try:
    if not cfg.tracing_enabled:
      return
    if cfg.console_tracing:
      add_trace_processor(BatchTraceProcessor(exporter=_RedactingConsoleExporter()))
    if cfg.openai_trace_api_key:
      set_tracing_export_api_key(cfg.openai_trace_api_key)
      print("OpenAI tracing enabled with provided OPENAI_TRACE_API_KEY.")
  except Exception as exc:
    print("Tracing setup failed:", exc)
  else:
    _TRACING_INITIALIZED = True


@dataclass
class TradingSnapshot:
  coins: List[str]
  tickers: Dict[str, KucoinTicker]
  balances: List[KucoinAccount]
  paper_trading: bool
  max_position_usd: float
  min_confidence: float
  max_leverage: float
  futures_enabled: bool
  risk_off: bool = False
  drawdown_pct: float = 0.0
  total_usdt: float = 0.0


def _build_openai_client(cfg: AppConfig) -> AsyncAzureOpenAI:
  # Prefer APIM when subscription key is provided; else use direct Azure OpenAI.
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


def _format_snapshot(snapshot: TradingSnapshot, balances_by_currency: Dict[str, float]) -> str:
  user_content = {
    "coins": snapshot.coins,
    "tickers": {k: vars(v) for k, v in snapshot.tickers.items()},
    "balances": balances_by_currency,
    "paperTrading": snapshot.paper_trading,
    "maxPositionUsd": snapshot.max_position_usd,
    "minConfidence": snapshot.min_confidence,
    "maxLeverage": snapshot.max_leverage,
    "futuresEnabled": snapshot.futures_enabled,
    "riskOff": snapshot.risk_off,
    "drawdownPct": snapshot.drawdown_pct,
    "totalUsdt": snapshot.total_usdt,
    "guidance": "If you place an order, prefer market orders sized in USDT funds.",
  }
  return json.dumps(user_content)


async def run_trading_agent(
  cfg: AppConfig,
  snapshot: TradingSnapshot,
  kucoin: KucoinClient,
  kucoin_futures: KucoinFuturesClient | None = None,
  openai_client: AsyncAzureOpenAI | None = None,
) -> dict[str, Any]:
  # Azure OpenAI async client configured for Agents SDK.
  if openai_client is None:
    openai_client = _build_openai_client(cfg)
  model = OpenAIResponsesModel(
    model=cfg.azure.deployment,
    openai_client=openai_client,
  )
  balances_by_currency: Dict[str, float] = {}
  for bal in snapshot.balances:
    balances_by_currency[bal.currency] = balances_by_currency.get(bal.currency, 0.0) + float(
      bal.available or 0
    )
  unique_trace_id = gen_trace_id()

  allowed_symbols = set(snapshot.tickers.keys())
  memory = MemoryStore(cfg.memory_file)

  # --- LANGSMITH SETUP (Cleaned) ---
  run_name = f"Trading Loop {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
  ls_processor = None
  
  if cfg.langsmith.enabled and cfg.langsmith.tracing and cfg.langsmith.api_key:
    try:
      from langsmith import Client as LangsmithClient
      from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor
      # Removed: get_run_tree_context (creates conflict)

      ls_client = LangsmithClient(api_key=cfg.langsmith.api_key, api_url=cfg.langsmith.api_url or None)
      
      ls_processor = OpenAIAgentsTracingProcessor(
        client=ls_client,
        project_name=cfg.langsmith.project or None,
        tags=["trAIde", "openai-agents"],
        name=run_name, # Name the trace here directly
      )
      add_trace_processor(ls_processor)
    except ImportError:
      print("LangSmith tracing modules unavailable; skipping per-run LangSmith.")
    except Exception as exc:
      print("LangSmith setup failed:", exc)

  @function_tool
  async def place_market_order(
    symbol: str,
    side: str,
    funds: float,
    confidence: float | None = None,
    rationale: str | None = None,
  ) -> Dict[str, Any]:
    # ... (No changes to tool logic) ...
    try:
      funds_val = float(funds or 0)
    except (TypeError, ValueError):
      funds_val = 0.0
    if funds_val <= 0 or not symbol:
      return {"error": "Invalid funds or symbol"}
    if funds_val > snapshot.max_position_usd:
      return {"rejected": True, "reason": "Exceeds maxPositionUsd"}
    trades_today = memory.trades_today(symbol)
    if trades_today >= cfg.trading.max_trades_per_symbol_per_day:
      return {
        "rejected": True,
        "reason": "Daily trade cap reached",
        "tradesToday": trades_today,
        "limit": cfg.trading.max_trades_per_symbol_per_day,
      }
    if cfg.trading.sentiment_filter_enabled:
      latest_sent = memory.latest_sentiment(symbol)
      day_key = int(time.time() // 86400)
      if not latest_sent or latest_sent.get("day") != day_key:
        return {"rejected": True, "reason": "Sentiment missing for today", "minScore": cfg.trading.sentiment_min_score}
      if latest_sent.get("score", 0) < cfg.trading.sentiment_min_score:
        return {
          "rejected": True,
          "reason": "Sentiment below threshold",
          "score": latest_sent.get("score"),
          "minScore": cfg.trading.sentiment_min_score,
        }
    fresh_balances = kucoin.get_trade_accounts()
    fresh_by_currency: Dict[str, float] = {}
    for bal in fresh_balances:
      fresh_by_currency[bal.currency] = fresh_by_currency.get(bal.currency, 0.0) + float(bal.available or 0)

    spot_usdt = fresh_by_currency.get("USDT", 0.0)
    futures_available = 0.0
    if cfg.kucoin_futures.enabled and kucoin_futures:
      try:
        overview = kucoin_futures.get_account_overview()
        futures_available = float(overview.get("availableBalance") or 0.0)
      except Exception as exc:
        print("Warning: futures overview unavailable:", exc)

    total_available = spot_usdt + futures_available
    reserve_total = total_available * 0.10
    max_spend = max(0.0, total_available - reserve_total)
    if funds_val > max_spend:
      return {
        "rejected": True,
        "reason": "Exceeds spendable USDT after 10% reserve",
        "maxSpend": max_spend,
        "spotAvailable": spot_usdt,
        "futuresAvailable": futures_available,
      }

    spot_reserve = spot_usdt * 0.10
    spot_spendable = max(0.0, spot_usdt - spot_reserve)
    transfer_used: dict[str, Any] | None = None
    if funds_val > spot_spendable and futures_available > 0 and not snapshot.paper_trading and kucoin_futures:
      futures_reserve = futures_available * 0.10
      futures_spendable = max(0.0, futures_available - futures_reserve)
      need = funds_val - spot_spendable
      transfer_amt = min(need, futures_spendable)
      if transfer_amt > 0:
        try:
          transfer_used = kucoin.transfer_funds(
            currency="USDT",
            amount=transfer_amt,
            from_account="contract",
            to_account="trade",
          )
          spot_usdt += transfer_amt
          spot_spendable = max(0.0, spot_usdt - spot_usdt * 0.10)
        except Exception as exc:
          print("Warning: futures->spot transfer failed:", exc)

    order_req = KucoinOrderRequest(
      symbol=symbol,
      side="buy" if side == "buy" else "sell",
      type="market",
      funds=f"{funds_val:.2f}",
      clientOid=str(uuid.uuid4()),
    )
    if snapshot.paper_trading:
      record = memory.record_trade(symbol, side, funds_val, paper=True)
      decision = None
      if confidence is not None:
        decision = memory.log_decision(
          symbol,
          f"spot_{side}",
          float(confidence),
          rationale or "paper trade",
          pnl=None,
          paper=True,
        )
      return {"paper": True, "orderRequest": order_req.__dict__, "tradeRecord": record, "decisionLog": decision}
    try:
      res = kucoin.place_order(order_req).__dict__
      record = memory.record_trade(symbol, side, funds_val, paper=False)
      res["tradeRecord"] = record
      if confidence is not None:
        res["decisionLog"] = memory.log_decision(
          symbol,
          f"spot_{side}",
          float(confidence),
          rationale or "live trade",
          pnl=None,
          paper=False,
        )
      return res
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__, "transfer": transfer_used}

  # ... (Rest of tools: decline_trade, fetch_recent_candles, etc. - No Changes needed) ...
  @function_tool
  async def decline_trade(reason: str, confidence: float) -> Dict[str, Any]:
    memory.log_decision("ALL", "decline", confidence, reason, paper=True)
    return {"skipped": True, "reason": reason, "confidence": confidence}

  @function_tool
  async def fetch_recent_candles(symbol: str, interval: str = "1min", lookback_minutes: int = 120) -> Dict[str, Any]:
    # ... Implementation identical to your file ...
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
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
    return {
      "symbol": symbol,
      "interval": interval,
      "startAt": start_at,
      "endAt": end_at,
      "points": candles[:max_points],
      "rows": len(candles),
    }

  @function_tool
  async def fetch_orderbook(symbol: str, depth: int = 20) -> Dict[str, Any]:
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
    depth_safe = 20 if depth <= 20 else 100
    ob = kucoin.get_orderbook_levels(symbol, depth=depth_safe)
    ob["bids"] = ob.get("bids", [])[:depth_safe]
    ob["asks"] = ob.get("asks", [])[:depth_safe]
    return {"symbol": symbol, "depth": depth_safe, "orderbook": ob}

  @function_tool
  async def analyze_market_context(symbol: str, fast_interval: str = "15min", slow_interval: str = "1hour", lookback_minutes: int = 360) -> Dict[str, Any]:
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
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
  async def plan_spot_position(symbol: str, risk_pct: float | None = None, atr_multiple: float = 1.5, target_rr: float = 2.0, entry_price: float | None = None) -> Dict[str, Any]:
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
    balance_usdt = balances_by_currency.get("USDT", 0.0)
    if balance_usdt <= 0:
      return {"error": "No USDT balance available"}
    risk_fraction = risk_pct if risk_pct is not None else cfg.trading.risk_per_trade_pct
    risk_fraction = max(0.0, float(risk_fraction))
    risk_dollars = balance_usdt * risk_fraction
    if risk_dollars <= 0:
      return {"error": "Risk dollars computed to zero", "riskPct": risk_fraction}
    spendable = max(0.0, balance_usdt * 0.90)
    price = float(entry_price) if entry_price else float(snapshot.tickers[symbol].price)
    lookback_min = 180
    end_at = int(time.time())
    interval = "15min"
    interval_sec = INTERVAL_SECONDS[interval]
    points = min(500, max(50, int(lookback_min * 60 / interval_sec)))
    start_at = end_at - points * interval_sec
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
    cap_notional = min(snapshot.max_position_usd, spendable)
    notional = min(notional_unclipped, cap_notional)
    size = notional / price if price else 0
    stop_price = max(0.0, price - stop_distance)
    target_price = price + stop_distance * target_rr
    trades_today = memory.trades_today(symbol)
    warnings: list[str] = []
    if atr_val and atr_val / price * 100 >= 5:
      warnings.append("Volatility elevated (ATR% >=5); consider smaller size or skip.")
    if notional_unclipped > cap_notional:
      warnings.append("Size clipped by maxPositionUsd or 10% reserve.")
    if trades_today >= cfg.trading.max_trades_per_symbol_per_day:
      warnings.append("Trade cap reached; do not enter without manual override.")
    return {
      "symbol": symbol,
      "price": price,
      "riskPct": risk_fraction,
      "riskDollars": risk_dollars,
      "spendableAfterReserve": spendable,
      "atr": atr_val,
      "atrMultiple": atr_multiple,
      "stopDistance": stop_distance,
      "stopPrice": stop_price,
      "targetPrice": target_price,
      "rr": target_rr,
      "size": size,
      "notionalUsd": notional,
      "rawNotionalUsd": notional_unclipped,
      "tradesToday": trades_today,
      "maxTradesPerDay": cfg.trading.max_trades_per_symbol_per_day,
      "warnings": warnings,
    }

  @function_tool
  async def place_futures_market_order(symbol: str, side: str, notional_usd: float, leverage: float = 1.0, size_override: float | None = None, confidence: float | None = None, rationale: str | None = None) -> Dict[str, Any]:
    # ... Implementation identical to your file ...
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"paper": True, "reason": "Futures disabled in config"}
    try:
      notional = float(notional_usd or 0)
      lev = float(leverage or 0)
    except (TypeError, ValueError):
      return {"error": "Invalid notional or leverage"}
    if notional <= 0 or lev <= 0:
      return {"error": "Invalid notional or leverage"}
    if lev > snapshot.max_leverage:
      return {"rejected": True, "reason": "Exceeds max_leverage", "max_leverage": snapshot.max_leverage}
    if notional > snapshot.max_position_usd * snapshot.max_leverage:
      return {"rejected": True, "reason": "Exceeds max notional cap", "cap": snapshot.max_position_usd * snapshot.max_leverage}
    trades_today = memory.trades_today(symbol)
    if trades_today >= cfg.trading.max_trades_per_symbol_per_day:
      return {
        "rejected": True,
        "reason": "Daily trade cap reached",
        "tradesToday": trades_today,
        "limit": cfg.trading.max_trades_per_symbol_per_day,
      }
    if cfg.trading.sentiment_filter_enabled:
      latest_sent = memory.latest_sentiment(symbol)
      day_key = int(time.time() // 86400)
      if not latest_sent or latest_sent.get("day") != day_key:
        return {"rejected": True, "reason": "Sentiment missing for today", "minScore": cfg.trading.sentiment_min_score}
      if latest_sent.get("score", 0) < cfg.trading.sentiment_min_score:
        return {
          "rejected": True,
          "reason": "Sentiment below threshold",
          "score": latest_sent.get("score"),
          "minScore": cfg.trading.sentiment_min_score,
        }
    price = float(snapshot.tickers[symbol].price)
    est_size = notional / price if price else 0
    size = size_override if size_override is not None else est_size
    if size <= 0:
      return {"error": "Computed size is zero"}
    order_req = KucoinFuturesOrderRequest(
      symbol=symbol,
      side="buy" if side == "buy" else "sell",
      type="market",
      leverage=f"{lev}",
      size=f"{size}",
      clientOid=str(uuid.uuid4()),
    )
    if snapshot.paper_trading:
      record = memory.record_trade(symbol, side, notional, paper=True)
      decision = None
      if confidence is not None:
        decision = memory.log_decision(
          symbol,
          f"futures_{side}",
          float(confidence),
          rationale or "paper trade",
          pnl=None,
          paper=True,
        )
      return {
        "paper": True,
        "orderRequest": order_req.__dict__,
        "tradeRecord": record,
        "decisionLog": decision,
        "rationale": rationale,
      }
    res = kucoin_futures.place_order(order_req).__dict__
    record = memory.record_trade(symbol, side, notional, paper=False)
    res["tradeRecord"] = record
    if confidence is not None:
      res["decisionLog"] = memory.log_decision(
        symbol,
        f"futures_{side}",
        float(confidence),
        rationale or "live trade",
        pnl=None,
        paper=False,
      )
    res["rationale"] = rationale
    return res

  @function_tool
  async def transfer_funds(direction: str, currency: str = "USDT", amount: float = 0.0) -> Dict[str, Any]:
    # ... Implementation identical to your file ...
    dir_norm = (direction or "").lower()
    if dir_norm not in {"spot_to_futures", "futures_to_spot"}:
      return {"error": "Invalid direction", "allowed": ["spot_to_futures", "futures_to_spot"]}
    try:
      amt = float(amount or 0)
    except (TypeError, ValueError):
      return {"error": "Invalid amount"}
    if amt <= 0:
      return {"error": "Amount must be positive"}
    fresh_balances = kucoin.get_trade_accounts()
    balance_map: Dict[str, float] = {}
    for bal in fresh_balances:
      balance_map[bal.currency] = balance_map.get(bal.currency, 0.0) + float(bal.available or 0)
    from_acct = "trade" if dir_norm == "spot_to_futures" else "contract"
    to_acct = "contract" if dir_norm == "spot_to_futures" else "trade"
    available = balance_map.get(currency.upper(), 0.0)
    if amt > available:
      return {"rejected": True, "reason": "Insufficient balance", "available": available}
    if snapshot.paper_trading:
      return {
        "paper": True,
        "direction": dir_norm,
        "currency": currency.upper(),
        "amount": amt,
        "from": from_acct,
        "to": to_acct,
      }
    try:
      res = kucoin.transfer_funds(
        currency=currency.upper(),
        amount=amt,
        from_account=from_acct,
        to_account=to_acct,
      )
      return {"transfer": res, "direction": dir_norm, "currency": currency.upper(), "amount": amt}
    except Exception as exc:
      return {"error": str(exc), "direction": dir_norm, "currency": currency.upper(), "amount": amt}

  # ... (Rest of tools: save_trade_plan, latest_plan, etc. - No Changes needed) ...
  @function_tool
  async def save_trade_plan(title: str, summary: str, actions: List[str]) -> Dict[str, Any]:
    return memory.save_plan(title=title, summary=summary, actions=actions, author="Trading Agent")

  @function_tool
  async def latest_plan() -> Dict[str, Any]:
    return {"latest_plan": memory.latest_plan()}

  @function_tool
  async def clear_plans() -> Dict[str, Any]:
    return memory.clear_plans()

  @function_tool
  async def set_auto_trigger(symbol: str, direction: str, rationale: str, target_price: float | None = None, stop_price: float | None = None) -> Dict[str, Any]:
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
    return memory.save_trigger(
      symbol=symbol,
      direction=direction,
      rationale=rationale,
      target_price=target_price,
      stop_price=stop_price,
    )

  @function_tool
  async def list_triggers() -> Dict[str, Any]:
    return {"triggers": memory.latest_triggers()}

  @function_tool
  async def list_coins() -> Dict[str, Any]:
    return {"coins": memory.get_coins(default=list(allowed_symbols))}

  @function_tool
  async def add_coin(symbol: str, reason: str) -> Dict[str, Any]:
    if not symbol:
      return {"error": "symbol required"}
    entry = memory.add_coin(symbol, reason)
    return {"added": entry, "coins": memory.get_coins(default=list(allowed_symbols))}

  @function_tool
  async def remove_coin(symbol: str, reason: str, exit_plan: str) -> Dict[str, Any]:
    if not symbol:
      return {"error": "symbol required"}
    if not exit_plan:
      return {"error": "exit_plan required to remove coin"}
    entry = memory.remove_coin(symbol, reason, exit_plan)
    return {"removed": entry, "coins": memory.get_coins(default=list(allowed_symbols))}

  @function_tool
  async def log_sentiment(symbol: str, score: float, rationale: str, source: str = "") -> Dict[str, Any]:
    try:
      score_val = float(score)
    except (TypeError, ValueError):
      return {"error": "Invalid score"}
    if not symbol:
      return {"error": "symbol required"}
    entry = memory.log_sentiment(symbol, score_val, rationale, source)
    return {"sentiment": entry}

  @function_tool
  async def log_decision(symbol: str, action: str, confidence: float, reason: str, pnl: float | None = None, paper: bool = False) -> Dict[str, Any]:
    if not symbol:
      return {"error": "symbol required"}
    try:
      conf = float(confidence)
    except (TypeError, ValueError):
      return {"error": "Invalid confidence"}
    entry = memory.log_decision(symbol, action, conf, reason, pnl=pnl, paper=paper)
    return {"decision": entry}

  @function_tool
  async def fetch_kucoin_news(limit: int = 10) -> Dict[str, Any]:
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
        items.append(
          {
            "title": (title_el.text or "").strip() if title_el is not None else "",
            "link": (link_el.text or "").strip() if link_el is not None else "",
            "pubDate": (date_el.text or "").strip() if date_el is not None else "",
          }
        )
      return {"items": items}
    except Exception as exc:
      return {"error": f"parse_failed: {exc}"}

  @function_tool
  async def log_research(topic: str, summary: str, actions: List[str]) -> Dict[str, Any]:
    if not topic or not summary:
      return {"error": "topic and summary required"}
    return memory.save_plan(title=f"Research: {topic}", summary=summary, actions=actions, author="Research Agent")

  @function_tool
  async def add_source(name: str, url: str, reason: str) -> Dict[str, Any]:
    if not name or not url:
      return {"error": "name and url required"}
    entry = memory.save_plan(title=f"Source: {name}", summary=url, actions=[reason or ""], author="Research Agent")
    return {"source": entry}

  @function_tool
  async def remove_source(name: str, reason: str) -> Dict[str, Any]:
    if not name or not reason:
      return {"error": "name and reason required"}
    entry = memory.save_plan(title=f"Removed Source: {name}", summary=reason, actions=[], author="Research Agent")
    return {"removed": entry}

  instructions = (
    "You are a disciplined quantitative crypto trader.\n"
    # ... (Instructions identical to your file) ...
    "Priorities: maximize risk-adjusted profit, minimize drawdown, avoid over-trading.\n"
    "- First, run one or more web_search calls on the symbols/market to gather fresh sentiment, news, and catalysts.\n"
    "- Use fetch_recent_candles to pull 60-120 minutes of 1m/5m/15m data for BTC and ETH when missing intraday context.\n"
    "- Use analyze_market_context (15m + 1h default) to get EMA/RSI/MACD/ATR/Bollinger/VWAP; only trade when both intervals align and ATR% is reasonable (<5% if conviction is low).\n"
    "- If analyze_market_context shows mixed bias or elevated volatility without a high-conviction catalyst, prefer decline_trade.\n"
    "- After web_search and fetch_kucoin_news, assign a sentiment score 0-1; if sentiment_filter_enabled and score < sentiment_min_score, do NOT buy. Log via log_sentiment.\n"
    "- Before placing a spot trade, call plan_spot_position to size with risk_per_trade_pct and ATR-based stop/target; reject/skip if size is clipped or volatility is high.\n"
    "- Use fetch_orderbook to inspect depth/imbalances (top 20/100 levels) when you need microstructure context.\n"
    "- Choose mode per idea: spot (place_market_order) vs futures (place_futures_market_order) within leverage<=max_leverage.\n"
    "- Use transfer_funds when you need to rebalance USDT between spot(trade) and futures(contract) before/after a plan.\n"
    "- When idle/riskOff and no clean setups on current coins, handoff to the Research Agent to scout other high-confidence coins or better data sources; only adopt new coins/sources after clear evidence and explicit decision. Use research outputs (log_research, backtests) before committing capital.\n"
    "- Avoid putting all eggs in one basket: keep USDT split across spot and futures where practical so both venues remain tradable; rebalance with transfer_funds instead of concentrating all capital in one account.\n"
    "- Curate the coin universe with list_coins/add_coin/remove_coin (requires reason and exit plan before removal); persist choices in memory.\n"
    "- Keep memory of current plan via save_trade_plan/latest_plan and update when conditions change; log auto triggers via set_auto_trigger.\n"
    "- Focus on intraday/day-trading setups, not long holds. Prefer short holding periods.\n"
    "- Consider leverage only when conviction is high and risk is controlled; default to low/no leverage.\n"
    "- If riskOff=true in the snapshot (set by daily drawdown guard when drawdownPct > MAX_DAILY_DRAWDOWN_PCT), avoid new speculative entries. You may:\n"
    "  - Close/trim existing exposure if it reduces risk.\n"
    "  - Hedge (including futures shorts) or transfer funds between spot/futures to reduce net risk.\n"
    "  - Set or adjust protective triggers. Avoid adding net long risk unless explicitly justified as a hedge.\n"
    "- When riskOff and drawdownPct is high, analyze causes and propose steps to recover safely (e.g., hedges, rebalance, pause trades, tighten stops). If rules block entries, log a plan to exit riskOff when conditions improve.\n"
    f"- Consider only the provided symbols and market snapshot.\n"
    f"- Do NOT exceed maxPositionUsd={snapshot.max_position_usd} USDT per trade.\n"
    f"- Max trades per symbol per day: {cfg.trading.max_trades_per_symbol_per_day}. If reached, decline new trades.\n"
    f"- Futures leverage must stay <= {snapshot.max_leverage}x. Keep sizing realistic; if unsure, prefer spot.\n"
    f"- Only place a trade if your confidence >= {snapshot.min_confidence}; otherwise decline.\n"
    f"- Sentiment filter enabled: {cfg.trading.sentiment_filter_enabled}. Min score: {cfg.trading.sentiment_min_score}.\n"
    "- Keep at least 10% of USDT balance untouched for safety.\n"
    "- Log every decision with confidence using log_decision for calibration; include reason and whether paper/live.\n"
    "- Be explicit about your reasoning in the final narrative.\n"
    f"- PAPER_TRADING={snapshot.paper_trading}. When true, just simulate orders via the tool."
  )

  research_agent = Agent(
    name="Research Agent",
    instructions=(
      "You are a research scout for alternative crypto opportunities.\n"
      "- Goal: find high-confidence setups on coins beyond the current universe.\n"
      "- Use web_search and KuCoin news to identify catalysts, liquidity, and momentum on other symbols.\n"
      "- When idle, also discover new high-quality data/news sources; add via add_source(name, url, reason) and remove low-value ones via remove_source(name, reason).\n"
      "- NEVER place orders or change the coin list yourself; instead propose candidates with evidence.\n"
      "- Log findings via log_research (topic, summary, actions) and recommend adds for the main agent to decide.\n"
      "- Prioritize liquid, tradable pairs; avoid illiquid/obscure tokens. Return concise recommendations with confidence and why they beat current options."
    ),
    tools=[
      WebSearchTool(search_context_size="high"),
      fetch_recent_candles,
      analyze_market_context,
      fetch_orderbook,
      fetch_kucoin_news,
      log_research,
      add_source,
      remove_source,
      log_sentiment,
      log_decision,
    ],
    model=model,
  )

  trading_agent = Agent(
    name="Trading Agent",
    instructions=instructions,
    tools=[
      WebSearchTool(search_context_size="high"),
      fetch_recent_candles,
      fetch_orderbook,
      analyze_market_context,
      plan_spot_position,
      transfer_funds,
      place_futures_market_order,
      place_market_order,
      save_trade_plan,
      latest_plan,
      set_auto_trigger,
      list_triggers,
      list_coins,
      add_coin,
      remove_coin,
      clear_plans,
      decline_trade,
      fetch_kucoin_news,
      log_research,
      add_source,
      remove_source,
      log_sentiment,
      log_decision,
    ],
    handoffs=[research_agent],
    model=model,
  )

  input_payload = _format_snapshot(snapshot, balances_by_currency)

  # --- EXECUTION ---
  try:
    provider = get_trace_provider()
    # Manually create the trace with the specific ID so the Processor picks it up as the root
    tr = provider.create_trace(run_name, trace_id=unique_trace_id)
    tr.start(mark_as_current=True)
    try:
      result = await Runner.run(trading_agent, input_payload)
    finally:
      try:
        tr.finish(reset_current=True)
      except Exception as exc:
        print("Trace cleanup failed:", exc)
  finally:
    # --- CLEANUP ---
    if ls_processor:
      # CRITICAL: Give the processor a moment to batch export spans before we remove it.
      # If available, use a force flush (not standard in all SDK versions, but sleep is safe).
      if hasattr(ls_processor, "force_flush"):
          try:
            ls_processor.force_flush() 
          except:
            pass
      else:
          time.sleep(1.0) # Wait 1s for export
          
      _remove_trace_processor(ls_processor)

  narrative = str(result.final_output)

  tool_outputs: List[Any] = []
  for item in result.new_items:
    if isinstance(item, ToolCallOutputItem):
      tool_outputs.append(item.output)

  decisions: list[str] = []

  def _summarize(output: Any) -> str | None:
    if not isinstance(output, dict):
      return None
    if output.get("skipped"):
      return f"decline: {output.get('reason','unspecified')} (conf={output.get('confidence')})"
    if output.get("paper") and output.get("orderRequest"):
      req = output.get("orderRequest", {})
      rationale = output.get("rationale") or output.get("decisionLog", {}).get("reason")
      suffix = f" rationale={rationale}" if rationale else ""
      return f"paper order: {req.get('side')} {req.get('symbol')} funds={req.get('funds') or req.get('size')} (pnl=n/a){suffix}"
    if output.get("orderId") or output.get("orderRequest"):
      side = output.get("side") or output.get("orderRequest", {}).get("side")
      sym = output.get("symbol") or output.get("orderRequest", {}).get("symbol")
      rationale = output.get("rationale") or output.get("decisionLog", {}).get("reason")
      suffix = f" rationale={rationale}" if rationale else ""
      return f"live order: {side} {sym} (orderId={output.get('orderId')}) (pnl=n/a){suffix}"
    if output.get("transfer"):
      t = output.get("transfer", {})
      return f"transfer: {output.get('amount')} {output.get('currency')} {output.get('direction')} (id={t.get('orderId') or t.get('applyId')})"
    if output.get("rejected"):
      return f"rejected: {output.get('reason','unspecified')}"
    if output.get("error"):
      return f"error: {output.get('error')}"
    return None

  for out in tool_outputs:
    summary = _summarize(out)
    if summary:
      decisions.append(summary)

  return {"narrative": narrative, "tool_results": tool_outputs, "decisions": decisions}
