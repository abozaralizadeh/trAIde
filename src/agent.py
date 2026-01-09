from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

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
)
from agents.items import ToolCallOutputItem
from agents.tool import WebSearchTool, function_tool
from agents.tracing.processors import BatchTraceProcessor, ConsoleSpanExporter
from openai import AsyncAzureOpenAI

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
    "guidance": "If you place an order, prefer market orders sized in USDT funds.",
  }
  return json.dumps(user_content)


async def run_trading_agent(
  cfg: AppConfig,
  snapshot: TradingSnapshot,
  kucoin: KucoinClient,
  kucoin_futures: KucoinFuturesClient | None = None,
) -> dict[str, Any]:
  # Azure OpenAI async client configured for Agents SDK.
  openai_client = _build_openai_client(cfg)
  set_default_openai_client(openai_client, use_for_tracing=cfg.tracing_enabled)

  if cfg.tracing_enabled:
    if cfg.console_tracing:
      add_trace_processor(BatchTraceProcessor(exporter=ConsoleSpanExporter()))
    if cfg.openai_trace_api_key:
      set_tracing_export_api_key(cfg.openai_trace_api_key)

  model = OpenAIResponsesModel(
    model=cfg.azure.deployment,
    openai_client=openai_client,
  )

  balances_by_currency: Dict[str, float] = {}
  for bal in snapshot.balances:
    balances_by_currency[bal.currency] = balances_by_currency.get(bal.currency, 0.0) + float(
      bal.available or 0
    )

  allowed_symbols = set(snapshot.tickers.keys())
  memory = MemoryStore(cfg.memory_file)

  @function_tool
  async def place_market_order(symbol: str, side: str, funds: float) -> Dict[str, Any]:
    """Place a spot market order on Kucoin using quote funds in USDT. Respects maxPositionUsd and paper_trading flag."""
    try:
      funds_val = float(funds or 0)
    except (TypeError, ValueError):
      funds_val = 0.0
    if funds_val <= 0 or not symbol:
      return {"error": "Invalid funds or symbol"}
    if funds_val > snapshot.max_position_usd:
      return {"rejected": True, "reason": "Exceeds maxPositionUsd"}
    # Refresh balances to reduce stale balance failures.
    fresh_balances = kucoin.get_trade_accounts()
    fresh_by_currency: Dict[str, float] = {}
    for bal in fresh_balances:
      fresh_by_currency[bal.currency] = fresh_by_currency.get(bal.currency, 0.0) + float(bal.available or 0)

    usdt_balance = fresh_by_currency.get("USDT", 0.0)
    reserve = usdt_balance * 0.10
    max_spend = max(0.0, usdt_balance - reserve)
    if funds_val > max_spend:
      return {"rejected": True, "reason": "Exceeds spendable USDT after 10% reserve", "maxSpend": max_spend}

    order_req = KucoinOrderRequest(
      symbol=symbol,
      side="buy" if side == "buy" else "sell",  # enforce allowed values
      type="market",
      funds=f"{funds_val:.2f}",
      clientOid=str(uuid.uuid4()),
    )
    if snapshot.paper_trading:
      return {"paper": True, "orderRequest": order_req.__dict__}
    try:
      return kucoin.place_order(order_req).__dict__
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__}

  @function_tool
  async def decline_trade(reason: str, confidence: float) -> Dict[str, Any]:
    """Decline trading due to low confidence or risk."""
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
  async def place_futures_market_order(
    symbol: str,
    side: str,
    notional_usd: float,
    leverage: float = 1.0,
    size_override: float | None = None,
  ) -> Dict[str, Any]:
    """Place a linear futures market order using notional and leverage. Falls back to paper if futures disabled."""
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
      return {"paper": True, "orderRequest": order_req.__dict__}
    return kucoin_futures.place_order(order_req).__dict__

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

    # Pull fresh balances to avoid stale state.
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
        from_account=from_acct,  # type: ignore[arg-type]
        to_account=to_acct,  # type: ignore[arg-type]
      )
      return {"transfer": res, "direction": dir_norm, "currency": currency.upper(), "amount": amt}
    except Exception as exc:
      return {"error": str(exc), "direction": dir_norm, "currency": currency.upper(), "amount": amt}

  @function_tool
  async def save_trade_plan(title: str, summary: str, actions: List[str]) -> Dict[str, Any]:
    """Persist a trading plan (title, summary, actions) for recall."""
    return memory.save_plan(title=title, summary=summary, actions=actions)

  @function_tool
  async def latest_plan() -> Dict[str, Any]:
    """Get the latest stored trading plan."""
    return {"latest_plan": memory.latest_plan()}

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

  instructions = (
    "You are a disciplined quantitative crypto trader using Azure OpenAI gpt-5.2.\n"
    "Priorities: maximize risk-adjusted profit, minimize drawdown, avoid over-trading.\n"
    "- First, run one or more web_search calls on the symbols/market to gather fresh sentiment, news, and catalysts.\n"
    "- Use fetch_recent_candles to pull 60-120 minutes of 1m/5m/15m data for BTC and ETH when missing intraday context.\n"
    "- Use fetch_orderbook to inspect depth/imbalances (top 20/100 levels) when you need microstructure context.\n"
    "- Choose mode per idea: spot (place_market_order) vs futures (place_futures_market_order) within leverage<=max_leverage.\n"
    "- Use transfer_funds when you need to rebalance USDT between spot(trade) and futures(contract) before/after a plan.\n"
    "- Curate the coin universe with list_coins/add_coin/remove_coin (requires reason and exit plan before removal); persist choices in memory.\n"
    "- Keep memory of current plan via save_trade_plan/latest_plan and update when conditions change; log auto triggers via set_auto_trigger.\n"
    "- Focus on intraday/day-trading setups, not long holds. Prefer short holding periods.\n"
    "- Consider leverage only when conviction is high and risk is controlled; default to low/no leverage.\n"
    f"- Consider only the provided symbols and market snapshot.\n"
    f"- Do NOT exceed maxPositionUsd={snapshot.max_position_usd} USDT per trade.\n"
    f"- Futures leverage must stay <= {snapshot.max_leverage}x. Keep sizing realistic; if unsure, prefer spot.\n"
    f"- Only place a trade if your confidence >= {snapshot.min_confidence}; otherwise decline.\n"
    "- Keep at least 10% of USDT balance untouched for safety.\n"
    "- Be explicit about your reasoning in the final narrative.\n"
    f"- PAPER_TRADING={snapshot.paper_trading}. When true, just simulate orders via the tool."
  )

  trading_agent = Agent(
    name="Trading Agent",
    instructions=instructions,
    tools=[
      WebSearchTool(search_context_size="high"),
      fetch_recent_candles,
      fetch_orderbook,
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
    ],
    model=model,
  )

  # Provide snapshot as serialized context input.
  input_payload = _format_snapshot(snapshot, balances_by_currency)

  result = await Runner.run(trading_agent, input_payload)
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
      return f"paper order: {req.get('side')} {req.get('symbol')} funds={req.get('funds') or req.get('size')}"
    if output.get("orderId") or output.get("orderRequest"):
      side = output.get("side") or output.get("orderRequest", {}).get("side")
      sym = output.get("symbol") or output.get("orderRequest", {}).get("symbol")
      return f"live order: {side} {sym} (orderId={output.get('orderId')})"
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
