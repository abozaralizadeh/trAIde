from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

import requests
from datetime import datetime, timezone
import contextlib

from agents import (
  Agent,
  InputGuardrail,
  GuardrailFunctionOutput,
  Runner,
  OpenAIChatCompletionsModel,
  OpenAIResponsesModel,
  add_trace_processor,
  set_default_openai_client,
  set_tracing_export_api_key,
  gen_trace_id,
  gen_span_id)
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


def setup_tracing(cfg: AppConfig):
  """Register tracing processors once at startup."""
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

def setup_lstracing(cfg: AppConfig):
  """Register tracing processors once at startup."""
  try:
    if cfg.langsmith.enabled and cfg.langsmith.tracing:
      from langsmith import Client as LangsmithClient
      from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor

      ls_client = LangsmithClient(
        api_key=cfg.langsmith.api_key,
        api_url=cfg.langsmith.api_url or None,
      )
      processor = OpenAIAgentsTracingProcessor(
        client=ls_client,
        project_name=cfg.langsmith.project or None,
        tags=["trAIde", "openai-agents"],
        name=f"trAIde-agent",
      )
      add_trace_processor(processor)
      print("LangSmith tracing enabled via OpenAIAgentsTracingProcessor (per-run, OpenAI traces retained)")
    return ls_client
  except Exception as exc:
    print("Tracing setup failed:", exc)

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
  risk_off_spot: bool = False
  risk_off_futures: bool = False
  drawdown_pct: float = 0.0
  drawdown_pct_spot: float = 0.0
  drawdown_pct_futures: float = 0.0
  total_usdt: float = 0.0
  futures_positions: list[dict[str, Any]] = field(default_factory=list)
  spot_accounts: List[KucoinAccount] = field(default_factory=list)
  futures_account: Dict[str, Any] | None = None
  all_accounts: List[KucoinAccount] = field(default_factory=list)
  spot_stop_orders: List[Dict[str, Any]] = field(default_factory=list)
  futures_stop_orders: List[Dict[str, Any]] = field(default_factory=list)
  fees: Dict[str, Any] = field(default_factory=dict)


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

def _to_float(val: Any, default: float = 0.0) -> float:
  try:
    return float(val)
  except (TypeError, ValueError):
    return default


def _maybe_number(val: Any) -> Any:
  if isinstance(val, (int, float)):
    return float(val)
  if isinstance(val, str):
    try:
      return float(val)
    except ValueError:
      return val
  return val


def _serialize_accounts(accounts: List[KucoinAccount]) -> List[Dict[str, Any]]:
  return [
    {
      "id": acct.id,
      "currency": acct.currency,
      "type": acct.type,
      "balance": _to_float(acct.balance),
      "available": _to_float(acct.available),
      "holds": _to_float(acct.holds),
    }
    for acct in accounts
  ]


def _aggregate_account_totals(accounts: List[KucoinAccount]) -> Dict[str, Dict[str, float]]:
  totals: Dict[str, Dict[str, float]] = {}
  for acct in accounts:
    cur = acct.currency
    bucket = totals.setdefault(cur, {"available": 0.0, "holds": 0.0, "balance": 0.0})
    bucket["available"] += _to_float(acct.available)
    bucket["holds"] += _to_float(acct.holds)
    bucket["balance"] += _to_float(acct.balance)
  return totals


def _summarize_futures_account(overview: Dict[str, Any] | None) -> Dict[str, Any]:
  if not overview:
    return {}
  normalized = {k: _maybe_number(v) for k, v in overview.items()}
  currency = str(overview.get("currency", "USDT"))
  normalized["summary"] = {
    "currency": currency,
    "availableBalance": _to_float(overview.get("availableBalance")),
    "marginBalance": _to_float(overview.get("marginBalance")),
    "accountEquity": _to_float(overview.get("accountEquity") or overview.get("marginBalance")),
    "frozenBalance": _to_float(overview.get("frozenBalance")),
    "unrealisedPnl": _to_float(overview.get("unrealisedPnl") or overview.get("unrealisedPNL")),
  }
  return normalized


def _base_currency(symbol: str) -> str:
  """Return the base currency for a trading symbol (e.g., BTC-USDT -> BTC)."""
  if not symbol or "-" not in symbol:
    return symbol or ""
  return symbol.split("-")[0].upper()


def _to_futures_symbol(spot_symbol: str) -> str | None:
  """Map spot symbol (e.g., BTC-USDT) to KuCoin futures contract symbol (e.g., XBTUSDTM)."""
  if not spot_symbol or "-" not in spot_symbol:
    return None
  base, quote = spot_symbol.split("-", 1)
  base = base.upper()
  quote = quote.upper()
  # KuCoin futures uses XBT for BTC.
  if base == "BTC":
    base = "XBT"
  return f"{base}{quote}M"


def _format_snapshot(snapshot: TradingSnapshot, balances_by_currency: Dict[str, float]) -> str:
  spot_accounts = snapshot.spot_accounts or snapshot.balances
  all_accounts = snapshot.all_accounts or spot_accounts
  spot_serialized = _serialize_accounts(spot_accounts)
  all_serialized = _serialize_accounts(all_accounts)
  spot_totals = _aggregate_account_totals(spot_accounts)
  all_totals = _aggregate_account_totals(all_accounts)
  futures_info = _summarize_futures_account(snapshot.futures_account)

  user_content = {
    "coins": snapshot.coins,
    "tickers": {k: vars(v) for k, v in snapshot.tickers.items()},
    "balances": balances_by_currency,
    "accountDetails": {
      "spot": {"accounts": spot_serialized, "byCurrency": spot_totals},
      "futures": futures_info,
      "allAccounts": {"accounts": all_serialized, "byCurrency": all_totals},
    },
    "paperTrading": snapshot.paper_trading,
    "maxPositionUsd": snapshot.max_position_usd,
    "minConfidence": snapshot.min_confidence,
    "maxLeverage": snapshot.max_leverage,
    "futuresEnabled": snapshot.futures_enabled,
    "riskOff": snapshot.risk_off,
    "riskOffSpot": snapshot.risk_off_spot,
    "riskOffFutures": snapshot.risk_off_futures,
    "drawdownPct": snapshot.drawdown_pct,
    "drawdownPctSpot": snapshot.drawdown_pct_spot,
    "drawdownPctFutures": snapshot.drawdown_pct_futures,
    "totalUsdt": snapshot.total_usdt,
    "futuresPositions": snapshot.futures_positions,
    "guidance": "If you place an order, prefer market orders sized in USDT funds.",
    "stops": {
      "spot": snapshot.spot_stop_orders,
      "futures": snapshot.futures_stop_orders,
    },
    "fees": snapshot.fees if hasattr(snapshot, "fees") else {},
  }
  return json.dumps(user_content)

from langsmith import Client as LangsmithClient

async def run_trading_agent(
  cfg: AppConfig,
  snapshot: TradingSnapshot,
  kucoin: KucoinClient,
  kucoin_futures: KucoinFuturesClient | None = None,
  openai_client: AsyncAzureOpenAI | None = None,
  langsmith_client: LangsmithClient | None = None,
) -> dict[str, Any]:
  # Azure OpenAI async client configured for Agents SDK.
  if openai_client is None:
    openai_client = _build_openai_client(cfg)
  model = OpenAIResponsesModel(
    model=cfg.azure.deployment,
    openai_client=openai_client,
  )
  balances_by_currency: Dict[str, float] = {}
  spot_accounts = snapshot.spot_accounts or snapshot.balances
  spot_totals = _aggregate_account_totals(spot_accounts)
  for cur, totals in spot_totals.items():
    balances_by_currency[cur] = balances_by_currency.get(cur, 0.0) + totals.get("available", 0.0)

  if snapshot.futures_account:
    fut_currency = str(snapshot.futures_account.get("currency", "USDT"))
    fut_available = _to_float(snapshot.futures_account.get("availableBalance"))
    if fut_available:
      balances_by_currency[fut_currency] = balances_by_currency.get(fut_currency, 0.0) + fut_available
  unique_trace_id = gen_trace_id()
  unique_span_id = gen_span_id()

  allowed_symbols = set(snapshot.tickers.keys())
  memory = MemoryStore(cfg.memory_file)
  current_prices = {sym: float(t.price) for sym, t in snapshot.tickers.items()}
  positions = memory.positions(current_prices)
  # Reconcile tracked positions with live spot balances to avoid phantom exposure when trades happen outside the agent.
  spot_balances_by_currency = {
    cur.upper(): _to_float(totals.get("balance"))
    for cur, totals in (spot_totals or {}).items()
  }
  reconciled_positions: Dict[str, Dict[str, Any]] = {}
  zero_tol = 1e-9
  for sym, pos in positions.items():
    base = _base_currency(sym)
    actual_qty = spot_balances_by_currency.get(base, 0.0)
    net = _to_float(pos.get("netSize"))
    if actual_qty <= zero_tol:
      # Drop positions when the wallet no longer holds the asset.
      continue
    adj = dict(pos)
    if abs(actual_qty - net) > zero_tol:
      avg_entry = adj.get("avgEntry")
      adj["netSize"] = actual_qty
      if avg_entry is not None:
        adj["cost"] = _to_float(avg_entry) * actual_qty
        cur_price = current_prices.get(sym) or 0.0
        adj["unrealizedPnl"] = (cur_price - _to_float(avg_entry)) * actual_qty if cur_price else None
      else:
        adj["cost"] = None
        adj["unrealizedPnl"] = None
    reconciled_positions[sym] = adj
  # Add any live holdings that are missing from the recorded trade log.
  for sym in snapshot.tickers.keys():
    if sym in reconciled_positions:
      continue
    base = _base_currency(sym)
    actual_qty = spot_balances_by_currency.get(base, 0.0)
    if actual_qty <= zero_tol:
      continue
    cur_price = current_prices.get(sym) or None
    reconciled_positions[sym] = {
      "netSize": actual_qty,
      "cost": None,
      "avgEntry": None,
      "realizedPnl": None,
      "unrealizedPnl": None,
      "lastTs": None,
      "currentPrice": cur_price,
    }
  positions = reconciled_positions
  triggers = memory.latest_triggers()
  # Pull latest stored fees; fallback defaults.
  fee_defaults = {"spot_taker": 0.001, "spot_maker": 0.001, "futures_taker": 0.0006, "futures_maker": 0.0002}
  stored_fees = memory.latest_fees() or {}
  fees = {
    "spot_taker": float(stored_fees.get("spot_taker") or snapshot.fees.get("spot_taker") or fee_defaults["spot_taker"]),
    "spot_maker": float(stored_fees.get("spot_maker") or snapshot.fees.get("spot_maker") or fee_defaults["spot_maker"]),
    "futures_taker": float(stored_fees.get("futures_taker") or snapshot.fees.get("futures_taker") or fee_defaults["futures_taker"]),
    "futures_maker": float(stored_fees.get("futures_maker") or snapshot.fees.get("futures_maker") or fee_defaults["futures_maker"]),
    "ts": stored_fees.get("ts") or None,
  }
  contract_cache: Dict[str, Dict[str, Any]] = {}

  def _get_contract_spec(futures_symbol: str) -> Dict[str, Any] | None:
    """Fetch and cache KuCoin futures contract specs to validate size/mins."""
    if not futures_symbol:
      return None
    if futures_symbol in contract_cache:
      return contract_cache[futures_symbol]
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
        contract_cache[futures_symbol] = data
        return data
    except Exception as exc:
      print(f"Warning: unable to fetch futures contract {futures_symbol}:", exc)
    return None

  def _spot_position_info(symbol: str) -> Dict[str, float] | None:
    """Return current net size and avg entry for a spot position."""
    pos = positions.get(symbol)
    if not pos:
      return None
    try:
      net = float(pos.get("netSize") or 0.0)
    except Exception:
      return None
    if net <= 0:
      return None
    avg_entry = pos.get("avgEntry")
    if avg_entry is None:
      cost = _to_float(pos.get("cost"))
      if net and cost:
        avg_entry = cost / net
    try:
      avg_entry_f = float(avg_entry or 0.0)
    except Exception:
      return None
    if avg_entry_f <= 0:
      return None
    return {"net": net, "avg_entry": avg_entry_f}

  def _fee_adjusted_breakeven(avg_entry: float, fee_rate: float) -> float:
    """Breakeven price that covers taker fees on both entry and exit."""
    fee_rate = max(0.0, float(fee_rate))
    exit_factor = max(1e-9, 1.0 - fee_rate)
    return avg_entry * (1.0 + fee_rate) / exit_factor

  # Create a LangSmith run context per loop to isolate traces.
  langsmith_ctx = contextlib.nullcontext()
  run_name = "Trading Agent Run"
  if cfg.langsmith.enabled and cfg.langsmith.tracing and cfg.langsmith.api_key:
    try:
      from langsmith.run_helpers import tracing_context

      run_name = f"Trading Loop {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}"
      langsmith_ctx = tracing_context(
        project_name=cfg.langsmith.project,
        run_name=run_name,
        run_id=unique_trace_id,
        trace_id=unique_trace_id,
        span_id=unique_span_id,
        tags=["trAIde", "openai-agents"],
        client=langsmith_client,
      )
    except ImportError:
      print("LangSmith tracing_context unavailable; skipping per-run LangSmith context.")
    except Exception as exc:
      print("LangSmith run context init failed:", exc)

  @function_tool
  async def place_market_order(
    symbol: str,
    side: str,
    funds: float,
    confidence: float | None = None,
    rationale: str | None = None,
  ) -> Dict[str, Any]:
    """Place a spot market order on Kucoin using quote funds in USDT. Respects maxPositionUsd and paper_trading flag."""
    try:
      funds_val = float(funds or 0)
    except (TypeError, ValueError):
      funds_val = 0.0
    if funds_val <= 0 or not symbol:
      return {"error": "Invalid funds or symbol"}
    fee_rate = fees.get("spot_taker", 0.001)
    funds_with_fee = funds_val * (1 + fee_rate)
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
    # Refresh balances to reduce stale balance failures.
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

    transfer_used: dict[str, Any] | None = None
    if side != "sell":
      total_available = spot_usdt + futures_available
      reserve_total = total_available * 0.10
      max_spend = max(0.0, total_available - reserve_total)
      if funds_with_fee > max_spend:
        return {
          "rejected": True,
          "reason": "Exceeds spendable USDT after 10% reserve (incl fee)",
          "maxSpend": max_spend,
          "spotAvailable": spot_usdt,
          "futuresAvailable": futures_available,
          "feeRate": fee_rate,
        }

      # If spot alone is insufficient but futures has free balance, auto-transfer the shortfall (respecting futures 10% reserve).
      spot_reserve = spot_usdt * 0.10
      spot_spendable = max(0.0, spot_usdt - spot_reserve)
      if funds_with_fee > spot_spendable and futures_available > 0 and not snapshot.paper_trading and kucoin_futures:
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
    else:
      # No spend limits/transfers are needed for sells; they're bounded by position size below.
      max_spend = float("inf")

    price = float(snapshot.tickers[symbol].price)
    size_est = funds_val / price if price else 0.0

    # Adjust funds down so fee fits in spend cap.
    if funds_with_fee > max_spend and funds_val > 0:
      scale = max_spend / funds_with_fee
      funds_val = max(0.0, funds_val * scale)
      funds_with_fee = funds_val * (1 + fee_rate)
      size_est = funds_val / price if price else 0.0

    order_req = KucoinOrderRequest(
      symbol=symbol,
      side="buy" if side == "buy" else "sell",  # enforce allowed values
      type="market",
      size=None,
      funds=f"{funds_val:.2f}" if side == "buy" else None,
      clientOid=str(uuid.uuid4()),
    )

    if side == "sell":
      pos_info = _spot_position_info(symbol)
      net_size = pos_info["net"] if pos_info else 0.0
      if net_size <= 0:
        return {"rejected": True, "reason": "No spot position available to sell"}
      if size_est <= 0:
        size_est = net_size
        funds_val = size_est * price
      if size_est > net_size:
        size_est = net_size
        funds_val = size_est * price
      breakeven_px = _fee_adjusted_breakeven(pos_info["avg_entry"], fee_rate) if pos_info else None
      expected_proceeds = price * size_est * (1 - fee_rate)
      entry_cost = pos_info["avg_entry"] * size_est * (1 + fee_rate) if pos_info else 0.0
      expected_pnl = expected_proceeds - entry_cost
      rationale_norm = (rationale or "").lower() if rationale else ""
      allow_loss = any(term in rationale_norm for term in ("stop", "cut", "hedge", "risk"))
      if expected_pnl <= 0 and not allow_loss:
        return {
          "rejected": True,
          "reason": "Sell below fee-adjusted breakeven",
          "breakevenPrice": breakeven_px,
          "expectedPnl": expected_pnl,
          "positionSize": net_size,
          "requestedSize": size_est,
          "feeRate": fee_rate,
        }
      order_req.size = f"{size_est:.8f}"
      order_req.funds = None
    else:
      if size_est <= 0:
        return {"error": "Computed size is zero", "price": price}
    if snapshot.paper_trading:
      record = memory.record_trade(symbol, side, funds_val, paper=True, price=price, size=size_est)
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
      record = memory.record_trade(symbol, side, funds_val, paper=False, price=price, size=size_est)
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

  @function_tool
  async def place_spot_stop_order(
    symbol: str,
    side: str,
    stop_price: float,
    stop_price_type: str = "MP",
    order_type: str = "limit",
    size: float | None = None,
    funds: float | None = None,
    limit_price: float | None = None,
    client_oid: str | None = None,
  ) -> Dict[str, Any]:
    """Place a spot stop order (stop-loss or take-profit). stop_price_type: TP(trigger when price rises) or MP(falls)."""
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
    try:
      stop_price_f = float(stop_price or 0)
    except Exception:
      return {"error": "Invalid stop price"}
    if stop_price_f <= 0:
      return {"error": "Stop price must be positive"}
    order_req = KucoinOrderRequest(
      symbol=symbol,
      side="buy" if side == "buy" else "sell",
      type="limit" if order_type == "limit" else "market",
      size=f"{size}" if size is not None else None,
      funds=f"{funds:.2f}" if funds is not None else None,
      price=f"{limit_price:.6f}" if limit_price is not None else None,
      clientOid=client_oid or str(uuid.uuid4()),
      stopPrice=f"{stop_price_f}",
      stopPriceType="TP" if stop_price_type.upper() == "TP" else "MP",
    )
    if snapshot.paper_trading:
      return {"paper": True, "orderRequest": order_req.__dict__}
    try:
      res = kucoin.place_stop_order(order_req).__dict__
      res["orderRequest"] = order_req.__dict__
      return res
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__}

  @function_tool
  async def cancel_spot_stop_order(order_id: str | None = None, client_oid: str | None = None) -> Dict[str, Any]:
    """Cancel a spot stop order by orderId or clientOid."""
    if snapshot.paper_trading:
      return {"paper": True, "cancelled": {"orderId": order_id, "clientOid": client_oid}}
    try:
      res = kucoin.cancel_stop_order(order_id=order_id, client_oid=client_oid)
      return {"cancelled": res, "orderId": order_id, "clientOid": client_oid}
    except Exception as exc:
      return {"error": str(exc), "orderId": order_id, "clientOid": client_oid}

  @function_tool
  async def list_spot_stop_orders(status: str = "active", symbol: str | None = None) -> Dict[str, Any]:
    """List spot stop orders; status usually 'active' or 'done'."""
    try:
      orders = kucoin.list_stop_orders(status=status, symbol=symbol)
      return {"orders": orders, "status": status, "symbol": symbol}
    except Exception as exc:
      return {"error": str(exc), "status": status, "symbol": symbol}

  @function_tool
  async def decline_trade(reason: str, confidence: float) -> Dict[str, Any]:
    """Decline trading due to low confidence or risk."""
    memory.log_decision("ALL", "decline", confidence, reason, paper=True)
    return {"skipped": True, "reason": reason, "confidence": confidence}

  @function_tool
  async def fetch_recent_candles(
    symbol: str,
    interval: str = "1min",
    lookback_minutes: int = 120,
  ) -> Dict[str, Any]:
    """Fetch recent candles for symbol. Interval options: 1min, 5min, 15min, 1hour. Caps to 500 rows."""
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}

    interval_seconds = {"1min": 60, "5min": 300, "15min": 900, "1hour": 3600}
    if interval not in interval_seconds:
      return {"error": "Invalid interval", "allowed": list(interval_seconds.keys())}

    lookback_min = max(1, min(int(lookback_minutes or 0), 720))
    end_at = int(time.time())
    # bound number of points to avoid oversized responses
    max_points = 500
    interval_sec = interval_seconds[interval]
    points = min(max_points, max(1, int(lookback_min * 60 / interval_sec)))
    start_at = end_at - points * interval_sec

    candles = kucoin.get_candles(
      symbol,
      interval=interval,
      start_at=start_at,
      end_at=end_at,
    )

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
    """Fetch level2 orderbook snapshot (depth 20 or 100)."""
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
    depth_safe = 20 if depth <= 20 else 100
    ob = kucoin.get_orderbook_levels(symbol, depth=depth_safe)
    # trim in case API returns more than requested
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
  async def plan_spot_position(
    symbol: str,
    risk_pct: float | None = None,
    atr_multiple: float = 1.5,
    target_rr: float = 2.0,
    entry_price: float | None = None,
  ) -> Dict[str, Any]:
    """Size a spot trade using risk-per-trade % with ATR-based stop/target."""
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

    fee_rate = fees.get("spot_taker", 0.001)
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
    if atr_val != atr_val:  # NaN check
      atr_val = None
    stop_distance = atr_val * atr_multiple if atr_val else price * 0.005  # fallback 0.5%
    if stop_distance <= 0:
      return {"error": "Stop distance invalid", "atr": atr_val}

    raw_size = risk_dollars / stop_distance
    notional_unclipped = raw_size * price
    cap_notional = min(snapshot.max_position_usd, spendable)
    notional = min(notional_unclipped, cap_notional)
    notional_with_fee = notional * (1 + fee_rate)
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
      "notionalUsdWithFee": notional_with_fee,
      "rawNotionalUsd": notional_unclipped,
      "tradesToday": trades_today,
      "maxTradesPerDay": cfg.trading.max_trades_per_symbol_per_day,
      "warnings": warnings,
      "feeRate": fee_rate,
    }

  @function_tool
  async def place_futures_market_order(
    symbol: str,
    side: str,
    notional_usd: float,
    leverage: float = 1.0,
    size_override: float | None = None,
    confidence: float | None = None,
    rationale: str | None = None,
    stop: str | None = None,
    stop_price_type: str | None = None,
    stop_price: float | None = None,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    reduce_only: bool | None = None,
  ) -> Dict[str, Any]:
    """Place a linear futures market order using notional and leverage (supports optional attached TP/SL). Falls back to paper if futures disabled."""
    spot_symbol = symbol
    if symbol not in allowed_symbols:
      # Accept futures contract symbols directly (e.g., XBTUSDTM -> BTC-USDT)
      if "-" not in symbol and symbol.upper().endswith("M"):
        base_quote = symbol[:-1]
        if base_quote.upper().startswith("XBT"):
          base_quote = "BTC" + base_quote[3:]
        if len(base_quote) > 4:
          spot_symbol = base_quote[:-4] + "-" + base_quote[-4:]
      if spot_symbol not in allowed_symbols:
        return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols), "requested": symbol}
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"paper": True, "reason": "Futures disabled in config"}
    try:
      notional_input = float(notional_usd or 0)
      lev_requested = float(leverage or 0)
    except (TypeError, ValueError):
      return {"error": "Invalid notional or leverage"}
    if notional_input <= 0 or lev_requested <= 0:
      return {"error": "Invalid notional or leverage"}
    trades_today = memory.trades_today(spot_symbol)
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

    futures_symbol = _to_futures_symbol(spot_symbol)
    if not futures_symbol:
      return {"error": "Invalid symbol for futures", "symbol": spot_symbol}

    contract = _get_contract_spec(futures_symbol)
    if not contract:
      return {"error": "Futures contract spec unavailable", "futuresSymbol": futures_symbol}

    price = float(snapshot.tickers[spot_symbol].price)
    if price <= 0:
      return {"error": "Invalid or missing price"}

    multiplier = _to_float(contract.get("multiplier"))
    lot_size = int(contract.get("lotSize") or 1)
    max_order_qty = contract.get("maxOrderQty")
    contract_max_leverage = _to_float(contract.get("maxLeverage")) or None
    if multiplier <= 0:
      return {"error": "Invalid contract multiplier", "contract": contract}
    lev = lev_requested
    if contract_max_leverage:
      lev = min(lev, contract_max_leverage)
    if snapshot.max_leverage:
      lev = min(lev, snapshot.max_leverage)
    lev = max(1.0, lev)

    try:
      base_size = float(size_override) if size_override is not None else notional_input / price
    except (TypeError, ValueError):
      return {"error": "Invalid size_override"}
    if base_size <= 0:
      return {"error": "Computed size is zero"}

    contracts_raw = base_size / multiplier
    lot = max(1, lot_size)
    contracts = int(math.ceil(contracts_raw / lot) * lot)
    actual_notional = contracts * multiplier * price
    min_notional = lot * multiplier * price
    if actual_notional < min_notional:
      return {
        "rejected": True,
        "reason": "Below contract minimum notional",
        "minNotionalUsd": min_notional,
        "requestedNotionalUsd": notional_input,
      }
    if max_order_qty and isinstance(max_order_qty, (int, float)) and contracts > max_order_qty:
      return {"rejected": True, "reason": "Exceeds max order size", "maxContracts": max_order_qty, "contracts": contracts}
    notional = actual_notional
    if lev > snapshot.max_leverage:
      return {"rejected": True, "reason": "Exceeds max_leverage", "max_leverage": snapshot.max_leverage}
    if notional > snapshot.max_position_usd * snapshot.max_leverage:
      return {
        "rejected": True,
        "reason": "Exceeds max notional cap",
        "cap": snapshot.max_position_usd * snapshot.max_leverage,
        "notional": notional,
      }
    # Prefer live contract fee if available.
    fee_rate = _to_float(contract.get("takerFeeRate")) or fees.get("futures_taker", 0.0006)
    notional_with_fee = notional * (1 + fee_rate)
    # Ensure required margin is funded; auto-transfer from spot if futures are empty.
    spot_accounts = kucoin.get_accounts()
    spot_balance_map: Dict[str, float] = {}
    for bal in spot_accounts:
      spot_balance_map[bal.currency] = spot_balance_map.get(bal.currency, 0.0) + float(bal.available or 0)

    futures_overview: Dict[str, Any] | None = None
    futures_available = 0.0
    margin_mode: str | None = None
    try:
      position_info = kucoin_futures.get_position(futures_symbol)
      if isinstance(position_info, dict) and "crossMode" in position_info:
        margin_mode = "cross" if position_info.get("crossMode") else "isolated"
    except Exception as exc:
      print("Warning: futures position lookup failed; margin mode unknown:", exc)
    if margin_mode is None:
      margin_mode = "cross"
    if kucoin_futures:
      try:
        futures_overview = kucoin_futures.get_account_overview()
        futures_available = _to_float(futures_overview.get("availableBalance"))
      except Exception as exc:
        print("Warning: futures overview unavailable:", exc)

    margin_needed = 0.0 if reduce_only else notional / lev
    fee_allowance = notional * fee_rate
    buffer = max(1.0, margin_needed * 0.01)
    required_futures_balance = margin_needed + fee_allowance + buffer

    transfer_used: dict[str, Any] | None = None
    if not snapshot.paper_trading and futures_available < required_futures_balance:
      spot_usdt = spot_balance_map.get("USDT", 0.0)
      spot_reserve = spot_usdt * 0.10
      spot_spendable = max(0.0, spot_usdt - spot_reserve)
      need = required_futures_balance - futures_available
      transfer_amt = min(spot_spendable, need)
      if transfer_amt > 0:
        try:
          transfer_used = kucoin.transfer_funds(
            currency="USDT",
            amount=transfer_amt,
            from_account="trade",
            to_account="contract",
          )
          futures_available += transfer_amt
        except Exception as exc:
          print("Warning: spot->futures transfer failed:", exc)
    if required_futures_balance > futures_available and not snapshot.paper_trading:
      return {
        "rejected": True,
        "reason": "Insufficient futures margin after transfer attempt",
        "requiredBalance": required_futures_balance,
        "available": futures_available,
        "spotAvailable": spot_balance_map.get("USDT", 0.0),
        "transferAttempt": transfer_used,
      }

    def _build_order(mode: str | None) -> KucoinFuturesOrderRequest:
      # KuCoin docs show uppercase margin modes; include autoDeposit for isolated.
      mode_norm = None
      auto_deposit = None
      if mode:
        mode_norm = mode.upper()
        if mode_norm == "ISOLATED":
          auto_deposit = False
      # If only stop_loss_price is provided, use it as stopPrice to avoid KuCoin stop-price missing errors.
      effective_stop_price = stop_price if stop_price is not None else stop_loss_price
      effective_stop_type = stop_price_type
      if effective_stop_price is not None and not effective_stop_type:
        effective_stop_type = "MP"
      stop_price_str = f"{effective_stop_price}" if effective_stop_price is not None else None
      tp_str = f"{take_profit_price}" if take_profit_price is not None else None
      sl_str = f"{stop_loss_price}" if stop_loss_price is not None else None
      return KucoinFuturesOrderRequest(
        symbol=futures_symbol,
        side="buy" if side == "buy" else "sell",
        type="market",
        leverage=f"{lev}",
        size=str(contracts),  # Kucoin futures expects integer contract size (contracts)
        clientOid=str(uuid.uuid4()),
        marginMode=mode_norm,
        autoDeposit=auto_deposit,
        stop=stop,
        stopPriceType=effective_stop_type,
        stopPrice=stop_price_str,
        takeProfitPrice=tp_str,
        stopLossPrice=sl_str,
        reduceOnly=reduce_only,
      )

    if snapshot.paper_trading:
      order_req = _build_order(margin_mode)
      record = memory.record_trade(symbol, side, notional, paper=True, price=price, size=contracts * multiplier)
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
        "feeRate": fee_rate,
        "notionalWithFee": notional_with_fee,
        "futuresSymbol": futures_symbol,
        "contracts": contracts,
        "contractSpec": {"multiplier": multiplier, "lotSize": lot_size},
        "appliedLeverage": lev,
      }

    attempts: list[Dict[str, Any]] = []
    order_req = _build_order(margin_mode)
    try:
      kucoin_futures.set_margin_mode(futures_symbol, (margin_mode or "cross"), auto_deposit=(margin_mode or "").upper() == "ISOLATED" and False)
    except Exception as exc:
      print("Warning: set_margin_mode failed (continuing):", exc)
    try:
      kucoin_futures.set_leverage(futures_symbol, lev)
    except Exception as exc:
      print("Warning: set_leverage failed (continuing):", exc)
    try:
      res = kucoin_futures.place_order(order_req).__dict__
      record = memory.record_trade(spot_symbol, side, notional, paper=False, price=price, size=contracts * multiplier)
      res["tradeRecord"] = record
      if confidence is not None:
        res["decisionLog"] = memory.log_decision(
          spot_symbol,
          f"futures_{side}",
          float(confidence),
          rationale or "live trade",
          pnl=None,
          paper=False,
        )
      res["rationale"] = rationale
      res["feeRate"] = fee_rate
      res["notionalWithFee"] = notional_with_fee
      res["futuresSymbol"] = futures_symbol
      res["contracts"] = contracts
      res["contractSpec"] = {"multiplier": multiplier, "lotSize": lot_size}
      res["transferUsed"] = transfer_used
      res["marginModeUsed"] = margin_mode
      res["appliedLeverage"] = lev
      res["previousAttempts"] = attempts
      return res
    except Exception as exc:
      attempts.append({"marginMode": margin_mode, "error": str(exc)})

    # If we reach here, try the opposite mode as a fallback.
    opposite_mode = "isolated" if margin_mode == "cross" else "cross"
    order_req = _build_order(opposite_mode)
    try:
      kucoin_futures.set_margin_mode(futures_symbol, opposite_mode, auto_deposit=opposite_mode.upper() == "ISOLATED" and False)
    except Exception as exc:
      print("Warning: set_margin_mode fallback failed (continuing):", exc)
    try:
      kucoin_futures.set_leverage(futures_symbol, lev)
    except Exception as exc:
      print("Warning: set_leverage fallback failed (continuing):", exc)
    try:
      res = kucoin_futures.place_order(order_req).__dict__
      record = memory.record_trade(spot_symbol, side, notional, paper=False, price=price, size=contracts * multiplier)
      res["tradeRecord"] = record
      if confidence is not None:
        res["decisionLog"] = memory.log_decision(
          spot_symbol,
          f"futures_{side}",
          float(confidence),
          rationale or "live trade",
          pnl=None,
          paper=False,
        )
      res["rationale"] = rationale
      res["feeRate"] = fee_rate
      res["notionalWithFee"] = notional_with_fee
      res["futuresSymbol"] = futures_symbol
      res["contracts"] = contracts
      res["contractSpec"] = {"multiplier": multiplier, "lotSize": lot_size}
      res["transferUsed"] = transfer_used
      res["marginModeUsed"] = opposite_mode
      res["appliedLeverage"] = lev
      res["previousAttempts"] = attempts
      return res
    except Exception as exc:
      attempts.append({"marginMode": opposite_mode, "error": str(exc)})

    return {
      "error": "Futures order failed",
      "attempts": attempts,
      "futuresSymbol": futures_symbol,
      "contracts": contracts,
      "marginModeDetected": margin_mode,
      "transferUsed": transfer_used,
    }

  @function_tool
  async def place_futures_stop_order(
    symbol: str,
    side: str,
    leverage: float,
    size: float | None = None,
    stop_price: float | None = None,
    stop: str | None = None,
    stop_price_type: str | None = None,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    reduce_only: bool | None = None,
    close_order: bool | None = None,
    order_type: str = "limit",
    limit_price: float | None = None,
    client_oid: str | None = None,
  ) -> Dict[str, Any]:
    """Place a futures stop/TP/SL order (works for reduce-only hedges)."""
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    futures_symbol = _to_futures_symbol(symbol)
    if not futures_symbol:
      return {"error": "Invalid symbol for futures", "symbol": symbol}
    try:
      lev = float(leverage or 0)
    except Exception:
      return {"error": "Invalid leverage"}
    if lev <= 0 or lev > snapshot.max_leverage:
      lev = min(max(1.0, lev), snapshot.max_leverage)
    if size is None or size <= 0:
      return {"error": "size required and >0"}
    contract = _get_contract_spec(futures_symbol)
    contract_max_leverage = _to_float(contract.get("maxLeverage")) if contract else None
    if contract_max_leverage:
      lev = min(lev, contract_max_leverage)
    multiplier = _to_float(contract.get("multiplier")) if contract else None
    lot_size = int(contract.get("lotSize") or 1) if contract else 1
    if multiplier is None or multiplier <= 0:
      return {"error": "Invalid contract multiplier for stops", "contract": contract}
    contracts_raw = float(size) / multiplier
    lot = max(1, lot_size)
    contracts = int(math.ceil(contracts_raw / lot) * lot)
    if contracts <= 0:
      return {"error": "Computed contract size <=0", "contracts": contracts, "baseSize": size, "multiplier": multiplier}
    stop_price_str = f"{stop_price}" if stop_price is not None else None
    tp_str = f"{take_profit_price}" if take_profit_price is not None else None
    sl_str = f"{stop_loss_price}" if stop_loss_price is not None else None

    margin_mode: str | None = None
    try:
      position_info = kucoin_futures.get_position(futures_symbol)
      if isinstance(position_info, dict) and "crossMode" in position_info:
        margin_mode = "cross" if position_info.get("crossMode") else "isolated"
    except Exception as exc:
      print("Warning: futures position lookup failed for stop order; margin mode unknown:", exc)

    margin_mode_norm = margin_mode.upper() if margin_mode else None
    auto_deposit = False if margin_mode_norm == "ISOLATED" else None
    try:
      kucoin_futures.set_margin_mode(futures_symbol, margin_mode_norm or "CROSS", auto_deposit=auto_deposit)
    except Exception as exc:
      print("Warning: set_margin_mode failed for stop order (continuing):", exc)
    try:
      kucoin_futures.set_leverage(futures_symbol, lev, cross=margin_mode_norm != "ISOLATED")
    except Exception as exc:
      print("Warning: set_leverage failed for stop order (continuing):", exc)
    order_req = KucoinFuturesOrderRequest(
      symbol=futures_symbol,
      side="buy" if side == "buy" else "sell",
      type="limit" if order_type == "limit" else "market",
      leverage=f"{lev}",
      size=f"{contracts}",
      price=f"{limit_price}" if limit_price is not None else None,
      clientOid=client_oid or str(uuid.uuid4()),
      stop=stop,
      stopPriceType=stop_price_type,
      stopPrice=stop_price_str,
      reduceOnly=reduce_only,
      closeOrder=close_order,
      takeProfitPrice=tp_str,
      stopLossPrice=sl_str,
      marginMode=margin_mode_norm,
      autoDeposit=auto_deposit,
    )
    if snapshot.paper_trading:
      return {"paper": True, "orderRequest": order_req.__dict__}
    try:
      res = kucoin_futures.place_order(order_req).__dict__
      res["orderRequest"] = order_req.__dict__
      return res
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__}

  @function_tool
  async def cancel_futures_order(order_id: str, symbol: str | None = None) -> Dict[str, Any]:
    """Cancel a futures order (stop or regular) by orderId."""
    if snapshot.paper_trading:
      return {"paper": True, "cancelled": {"orderId": order_id, "symbol": symbol}}
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    try:
      res = kucoin_futures.cancel_order(order_id, symbol=symbol)
      return {"cancelled": res, "orderId": order_id, "symbol": symbol}
    except Exception as exc:
      return {"error": str(exc), "orderId": order_id, "symbol": symbol}

  @function_tool
  async def list_futures_stop_orders(status: str = "active", symbol: str | None = None) -> Dict[str, Any]:
    """List futures stop orders; status: active/done."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    try:
      orders = kucoin_futures.list_stop_orders(status=status, symbol=symbol)
      return {"orders": orders, "status": status, "symbol": symbol}
    except Exception as exc:
      return {"error": str(exc), "status": status, "symbol": symbol}

  @function_tool
  async def list_futures_positions(status: str | None = None) -> Dict[str, Any]:
    """List current futures positions (live fetch; falls back to snapshot)."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    try:
      positions = kucoin_futures.list_positions(status=status)
      return {"positions": positions, "status": status}
    except Exception as exc:
      return {"error": str(exc), "status": status, "snapshotPositions": snapshot.futures_positions}

  @function_tool
  async def transfer_funds(
    direction: str,
    currency: str = "USDT",
    amount: float = 0.0,
  ) -> Dict[str, Any]:
    """Transfer funds between spot (trade) and futures (contract). Direction: spot_to_futures or futures_to_spot."""
    dir_norm = (direction or "").lower()
    if dir_norm not in {"spot_to_futures", "futures_to_spot"}:
      return {"error": "Invalid direction", "allowed": ["spot_to_futures", "futures_to_spot"]}
    try:
      amt = float(amount or 0)
    except (TypeError, ValueError):
      return {"error": "Invalid amount"}
    if amt <= 0:
      return {"error": "Amount must be positive"}

    # Pull fresh balances to avoid stale state; include main/trade so deposits are counted.
    spot_accounts = kucoin.get_accounts()
    spot_balance_map: Dict[str, float] = {}
    for bal in spot_accounts:
      spot_balance_map[bal.currency] = spot_balance_map.get(bal.currency, 0.0) + float(bal.available or 0)

    futures_available = 0.0
    futures_overview: Dict[str, Any] | None = None
    if cfg.kucoin_futures.enabled and kucoin_futures:
      try:
        futures_overview = kucoin_futures.get_account_overview()
        futures_available = _to_float(futures_overview.get("availableBalance"))
      except Exception as exc:
        # Record but continue to avoid hard failure.
        futures_overview = {"error": str(exc)}

    from_acct = "trade" if dir_norm == "spot_to_futures" else "contract"
    to_acct = "contract" if dir_norm == "spot_to_futures" else "trade"
    cur = currency.upper()

    if dir_norm == "spot_to_futures":
      available = spot_balance_map.get(cur, 0.0)
      if amt > available:
        return {"rejected": True, "reason": "Insufficient spot balance", "available": available}
    else:
      if not cfg.kucoin_futures.enabled or not kucoin_futures:
        return {"error": "Futures disabled or client unavailable"}
      available = futures_available
      if amt > available:
        return {
          "rejected": True,
          "reason": "Insufficient futures balance",
          "available": available,
          "futuresOverview": futures_overview,
        }

    if snapshot.paper_trading:
      return {
        "paper": True,
        "direction": dir_norm,
        "currency": currency.upper(),
        "amount": amt,
        "from": from_acct,
        "to": to_acct,
        "spotAvailable": spot_balance_map.get(cur, 0.0),
        "futuresAvailable": futures_available,
      }

    try:
      res = kucoin.transfer_funds(
        currency=currency.upper(),
        amount=amt,
        from_account=from_acct,  # type: ignore[arg-type]
        to_account=to_acct,  # type: ignore[arg-type]
      )
      return {
        "transfer": res,
        "direction": dir_norm,
        "currency": currency.upper(),
        "amount": amt,
        "spotAvailable": spot_balance_map.get(cur, 0.0),
        "futuresAvailable": futures_available,
        "futuresOverview": futures_overview,
      }
    except Exception as exc:
      return {"error": str(exc), "direction": dir_norm, "currency": currency.upper(), "amount": amt}

  @function_tool
  async def fetch_account_state() -> Dict[str, Any]:
    """Get up-to-date balances for all spot accounts and futures (including non-trade funds)."""
    try:
      spot_accounts_fresh = kucoin.get_trade_accounts()
      all_accounts_fresh = kucoin.get_accounts()
    except Exception as exc:
      return {"error": f"spot_fetch_failed: {exc}"}

    futures_overview = None
    futures_error = None
    if cfg.kucoin_futures.enabled and kucoin_futures:
      try:
        futures_overview = kucoin_futures.get_account_overview()
      except Exception as exc:
        futures_error = f"futures_fetch_failed: {exc}"

    spot_serialized = _serialize_accounts(spot_accounts_fresh)
    spot_totals = _aggregate_account_totals(spot_accounts_fresh)
    all_serialized = _serialize_accounts(all_accounts_fresh)
    all_totals = _aggregate_account_totals(all_accounts_fresh)
    balances_map = {cur: vals.get("available", 0.0) for cur, vals in all_totals.items()}

    futures_serialized: Dict[str, Any] = {}
    if futures_overview:
      futures_serialized = _summarize_futures_account(futures_overview)
      fut_cur = str(futures_overview.get("currency", "USDT"))
      balances_map[fut_cur] = balances_map.get(fut_cur, 0.0) + _to_float(futures_overview.get("availableBalance"))

    result: Dict[str, Any] = {
      "spot": {"accounts": spot_serialized, "byCurrency": spot_totals},
      "allAccounts": {"accounts": all_serialized, "byCurrency": all_totals},
      "balancesAvailable": balances_map,
      "futuresEnabled": bool(cfg.kucoin_futures.enabled),
    }
    if futures_overview:
      result["futures"] = futures_serialized or {"enabled": True}
    else:
      result["futures"] = {"enabled": bool(cfg.kucoin_futures.enabled)}
    if futures_error:
      result["futuresError"] = futures_error
    return result

  @function_tool
  async def refresh_fee_rates() -> Dict[str, Any]:
    """Fetch latest fee rates (spot base fee). Futures fee not provided by API; keep previous/default."""
    try:
      base_fee = kucoin.get_base_fee()
      spot_taker = float(base_fee.get("takerFeeRate") or 0.001)
      spot_maker = float(base_fee.get("makerFeeRate") or 0.001)
    except Exception as exc:
      return {"error": f"base_fee_failed: {exc}"}
    entry = memory.save_fee_info(
      spot_taker=spot_taker,
      spot_maker=spot_maker,
      futures_taker=fees.get("futures_taker"),
      futures_maker=fees.get("futures_maker"),
    )
    return {"fee": entry}

  @function_tool
  async def save_trade_plan(title: str, summary: str, actions: List[str]) -> Dict[str, Any]:
    """Persist a trading plan (title, summary, actions) for recall."""
    return memory.save_plan(title=title, summary=summary, actions=actions, author="Trading Agent")

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
  async def clear_plans() -> Dict[str, Any]:
    """Clear all stored plans and triggers."""
    return memory.clear_plans()

  @function_tool
  async def set_auto_trigger(
    symbol: str,
    direction: str,
    rationale: str,
    target_price: float | None = None,
    stop_price: float | None = None,
  ) -> Dict[str, Any]:
    """Store an auto-buy/sell trigger idea (persists to disk for follow-up by future runs)."""
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
    """List stored auto triggers (buy/sell ideas) for follow-up."""
    return {"triggers": memory.latest_triggers()}

  @function_tool
  async def list_coins() -> Dict[str, Any]:
    """List the current active coin universe (dynamic if enabled)."""
    return {"coins": memory.get_coins(default=list(allowed_symbols))}

  @function_tool
  async def add_coin(symbol: str, reason: str) -> Dict[str, Any]:
    """Add a coin to the active universe (requires reason)."""
    if not symbol:
      return {"error": "symbol required"}
    entry = memory.add_coin(symbol, reason)
    return {"added": entry, "coins": memory.get_coins(default=list(allowed_symbols))}

  @function_tool
  async def remove_coin(symbol: str, reason: str, exit_plan: str) -> Dict[str, Any]:
    """Remove a coin from the active universe with an exit plan noted."""
    if not symbol:
      return {"error": "symbol required"}
    if not exit_plan:
      return {"error": "exit_plan required to remove coin"}
    entry = memory.remove_coin(symbol, reason, exit_plan)
    return {"removed": entry, "coins": memory.get_coins(default=list(allowed_symbols))}

  @function_tool
  async def log_sentiment(symbol: str, score: float, rationale: str, source: str = "") -> Dict[str, Any]:
    """Store a sentiment score (0-1) with rationale and source for gating trades."""
    try:
      score_val = float(score)
    except (TypeError, ValueError):
      return {"error": "Invalid score"}
    if not symbol:
      return {"error": "symbol required"}
    entry = memory.log_sentiment(symbol, score_val, rationale, source)
    return {"sentiment": entry}

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
    entry = memory.log_decision(symbol, action, conf, reason, pnl=pnl, paper=paper)
    return {"decision": entry}

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
    """Record a research note/strategy idea (persists in memory)."""
    if not topic or not summary:
      return {"error": "topic and summary required"}
    return memory.save_plan(title=f"Research: {topic}", summary=summary, actions=actions, author="Research Agent")

  @function_tool
  async def add_source(name: str, url: str, reason: str) -> Dict[str, Any]:
    """Add a data/research source to memory."""
    if not name or not url:
      return {"error": "name and url required"}
    entry = memory.save_plan(title=f"Source: {name}", summary=url, actions=[reason or ""], author="Research Agent")
    return {"source": entry}

  @function_tool
  async def remove_source(name: str, reason: str) -> Dict[str, Any]:
    """Mark a data/research source as removed."""
    if not name or not reason:
      return {"error": "name and reason required"}
    entry = memory.save_plan(title=f"Removed Source: {name}", summary=reason, actions=[], author="Research Agent")
    return {"removed": entry}

  instructions = (
    "You are a disciplined quantitative crypto trader.\n"
    "Priorities: maximize risk-adjusted profit, minimize drawdown, avoid over-trading.\n"
    "- Act autonomously as the execution owner; do not ask the user what to do. Choose the best venue (spot vs futures) and act or decline yourself.\n"
    "- First, run one or more web_search calls on the symbols/market to gather fresh sentiment, news, and catalysts.\n"
    "- Use fetch_recent_candles to pull 60-120 minutes of 1m/5m/15m data for BTC and ETH when missing intraday context.\n"
    "- Use analyze_market_context (15m + 1h default) to get EMA/RSI/MACD/ATR/Bollinger/VWAP; only trade when both intervals align and ATR% is reasonable (<5% if conviction is low).\n"
    "- If analyze_market_context shows mixed bias or elevated volatility without a high-conviction catalyst, prefer decline_trade.\n"
    "- After web_search and fetch_kucoin_news, assign a sentiment score 0-1; if sentiment_filter_enabled and score < sentiment_min_score, do NOT buy. Log via log_sentiment.\n"
    "- Use the provided positions (avgEntry, unrealized, realized PnL) to manage exits/hedges; if size>0 with profit risk of mean-reversion, consider trims/stops instead of only new entries.\n"
    "- Set and maintain protection: place_spot_stop_order/place_futures_stop_order for stops/TP, cancel/replace them when thesis changes; keep stops in sync with position size. If spot balance+position size is ~0 but a SELL stop exists, cancel it via cancel_spot_stop_order to avoid stale orders.\n"
    "- Keep sizing fee-aware: use latest fees from state; refresh via refresh_fee_rates when stale; ensure spend caps include taker fees.\n"
    "- For spot sells, compute fee-adjusted breakeven (avgEntry*(1+fee)/(1-fee)); do not sell below breakeven unless explicitly cutting risk/stop/hedge. Aim for positive net PnL after fees on exits.\n"
    "- Favor intraday/day-trading profits: when confidence is high and research backs momentum/catalysts, prefer futures with sensible leverage (<= max_leverage) for higher R; size prudently and keep stops tight.\n"
    "- Evaluate every account each run: make a decision for spot balances, then separately for futures balances (if enabled). Consider transfers to free capital instead of skipping because one venue is low on USDT.\n"
    "- Before placing a spot trade, call plan_spot_position to size with risk_per_trade_pct and ATR-based stop/target; reject/skip if size is clipped or volatility is high.\n"
    "- Use fetch_orderbook to inspect depth/imbalances (top 20/100 levels) when you need microstructure context.\n"
    "- Choose mode per idea: spot (place_market_order) vs futures (place_futures_market_order) within leverage<=max_leverage.\n"
    "- Use transfer_funds when you need to rebalance USDT between spot(trade) and futures(contract) before/after a plan.\n"
    "- Always at the end if no active trades, or decide to decline trade, handoff to the Research Agent to update info, scout other high-confidence coins or better data sources; only adopt new coins/sources after clear evidence and explicit decision. Use research outputs (log_research, backtests) before committing capital.\n"
    "- If you lack recent context on a coin, sector, or macro driver, trigger a research handoff to refresh yourself (news, catalysts, new coins) before deciding; do not defer to the user for guidance.\n"
    "- Avoid putting all eggs in one basket: keep USDT split across spot and futures where practical so both venues remain tradable; rebalance with transfer_funds instead of concentrating all capital in one account.\n"
    "- Curate the coin universe with list_coins/add_coin/remove_coin (requires reason and exit plan before removal); persist choices in memory.\n"
    "- Keep memory of current plan via save_trade_plan/latest_plan and update when conditions change; log auto triggers via set_auto_trigger.\n"
    "- Focus on intraday/day-trading setups, not long holds. Prefer short holding periods.\n"
    "- Consider leverage only when conviction is high and risk is controlled; \n"
    "- If riskOff=true in the snapshot (set by daily drawdown guard when drawdownPct > MAX_DAILY_DRAWDOWN_PCT), avoid new speculative entries. You may:\n"
    "  - Close/trim existing exposure if it reduces risk.\n"
    "  - Hedge (including futures shorts) or transfer funds between spot/futures to reduce net risk.\n"
    "  - Set or adjust protective triggers. Avoid adding net long risk unless explicitly justified as a hedge.\n"
    "- When riskOff and drawdownPct is high, analyze causes and propose steps to recover safely (e.g., hedges, rebalance, pause trades, tighten stops). If rules block entries, log a plan to exit riskOff when conditions improve.\n"
    f"- Do NOT exceed maxPositionUsd={snapshot.max_position_usd} USDT per trade.\n"
    f"- Max trades per symbol per day: {cfg.trading.max_trades_per_symbol_per_day}. If reached, decline new trades.\n"
    f"- Futures leverage must stay <= {snapshot.max_leverage}x. Keep sizing realistic; if unsure, prefer spot.\n"
    f"- Only place a trade if your confidence >= {snapshot.min_confidence}; otherwise decline.\n"
    f"- Sentiment filter enabled: {cfg.trading.sentiment_filter_enabled}. Min score: {cfg.trading.sentiment_min_score}.\n"
    "- Keep at least 10% of USDT balance untouched for safety.\n"
    "- Log every decision with confidence using log_decision for calibration; include reason and whether paper/live.\n"
    "- Be explicit about your reasoning in the final narrative.\n"
    f"- PAPER_TRADING={snapshot.paper_trading}. When true, just simulate orders via the tool.\n"
  )

  # Secondary research agent to scout new coins while idle/riskOff.
  research_agent = Agent(
    name="Research Agent",
    instructions=(
      "You are a proactive research scout for alternative crypto opportunities.\n"
      "- Mission: Find high-confidence setups beyond the current universe, plus catalysts the main agent may have missed.\n"
      "- Sources to prioritize when using web_search: CoinDesk/The Block/Cointelegraph for news; exchange blogs (Binance/Coinbase/OKX/Bybit) for listings/delistings/rule changes; X/Twitter lists (founders, analysts, journalists) for real-time signals; macro outlets (Bloomberg/Reuters/FT) when relevant; on-chain sentiment/flows when available.\n"
      "- Tasks: gather news, sentiment, liquidity, catalysts, and narrative strength; highlight listings/delistings, hacks, regulatory moves, funding rounds, and smart-money activity.\n"
      "- Output: concise recommendations with evidence (why this beats current options), liquidity check, and confidence. Prefer liquid, tradable pairs; avoid illiquid/obscure tokens.\n"
      "- When idle, discover high-quality sources; add via add_source(name, url, reason) and remove low-value ones via remove_source(name, reason).\n"
      "- NEVER place orders or change the coin list yourself; propose candidates only.\n"
      "- Log findings via log_research (topic, summary, actions) so the main agent can decide."
      "- always return to the main Trading Agent after your research is done."
    ),
    tools=[
      WebSearchTool(search_context_size="high"),
      fetch_recent_candles,
      analyze_market_context,
      fetch_orderbook,
      fetch_kucoin_news,
      log_research,
      latest_items,
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
      fetch_account_state,
      refresh_fee_rates,
      place_futures_market_order,
      place_market_order,
      place_spot_stop_order,
      cancel_spot_stop_order,
      list_spot_stop_orders,
      place_futures_stop_order,
      cancel_futures_order,
      list_futures_stop_orders,
      list_futures_positions,
      save_trade_plan,
      latest_plan,
      latest_items,
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

  research_agent.handoffs = [trading_agent]

  # Provide snapshot as serialized context input.
  user_state = _format_snapshot(snapshot, balances_by_currency)
  # Enrich snapshot payload with positions (avgEntry/unrealized/realized) so agent can manage exits.
  user_state_obj = json.loads(user_state)
  user_state_obj["positions"] = positions
  user_state_obj["fees"] = fees
  user_state_obj["triggers"] = triggers
  input_payload = json.dumps(user_state_obj)

  # Ensure a fresh trace per agent loop using the official processor setup and a unique trace_id.
  with langsmith_ctx:
    provider = get_trace_provider()
    tr = provider.create_trace(run_name, trace_id=unique_trace_id)
    tr.start(mark_as_current=True)
    try:
      result = await Runner.run(trading_agent, input_payload, max_turns=cfg.agent_max_turns)
    finally:
      try:
        tr.finish(reset_current=True)
      except Exception as exc:
        print("Trace cleanup failed:", exc)
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
