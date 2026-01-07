from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, OpenAIChatCompletionsModel, OpenAIResponsesModel
from agents import set_default_openai_client
from agents.items import ToolCallOutputItem
from agents.tool import WebSearchTool, function_tool
from openai import AsyncAzureOpenAI

from .config import AppConfig
from .kucoin import KucoinAccount, KucoinClient, KucoinOrderRequest, KucoinTicker


@dataclass
class TradingSnapshot:
  tickers: Dict[str, KucoinTicker]
  balances: List[KucoinAccount]
  paper_trading: bool
  max_position_usd: float
  min_confidence: float


def _build_openai_client(cfg: AppConfig) -> AsyncAzureOpenAI:
  return AsyncAzureOpenAI(
    api_key=cfg.azure.api_key,
    api_version=cfg.azure.api_version,
    azure_endpoint=cfg.azure.endpoint,
    azure_deployment=cfg.azure.deployment,
  )


def _format_snapshot(snapshot: TradingSnapshot, balances_by_currency: Dict[str, float]) -> str:
  user_content = {
    "tickers": {k: vars(v) for k, v in snapshot.tickers.items()},
    "balances": balances_by_currency,
    "paperTrading": snapshot.paper_trading,
    "maxPositionUsd": snapshot.max_position_usd,
    "minConfidence": snapshot.min_confidence,
    "guidance": "If you place an order, prefer market orders sized in USDT funds.",
  }
  return json.dumps(user_content)


async def run_trading_agent(
  cfg: AppConfig,
  snapshot: TradingSnapshot,
  kucoin: KucoinClient,
) -> dict[str, Any]:
  # Azure OpenAI async client configured for Agents SDK.
  openai_client = _build_openai_client(cfg)
  set_default_openai_client(openai_client, use_for_tracing=False)

  model = OpenAIResponsesModel(
    model=cfg.azure.deployment,
    openai_client=openai_client,
  )

  balances_by_currency: Dict[str, float] = {}
  for bal in snapshot.balances:
    balances_by_currency[bal.currency] = balances_by_currency.get(bal.currency, 0.0) + float(
      bal.available or 0
    )

  @function_tool
  async def place_market_order(symbol: str, side: str, funds: float) -> Dict[str, Any]:
    """Place a market order on Kucoin using quote funds in USDT. Respects maxPositionUsd and paper_trading flag."""
    try:
      funds_val = float(funds or 0)
    except (TypeError, ValueError):
      funds_val = 0.0
    if funds_val <= 0 or not symbol:
      return {"error": "Invalid funds or symbol"}
    if funds_val > snapshot.max_position_usd:
      return {"rejected": True, "reason": "Exceeds maxPositionUsd"}

    order_req = KucoinOrderRequest(
      symbol=symbol,
      side="buy" if side == "buy" else "sell",  # enforce allowed values
      type="market",
      funds=f"{funds_val:.2f}",
      clientOid=str(uuid.uuid4()),
    )
    if snapshot.paper_trading:
      return {"paper": True, "orderRequest": order_req.__dict__}
    return kucoin.place_order(order_req).__dict__

  @function_tool
  async def decline_trade(reason: str, confidence: float) -> Dict[str, Any]:
    """Decline trading due to low confidence or risk."""
    return {"skipped": True, "reason": reason, "confidence": confidence}

  instructions = (
    "You are a disciplined quantitative crypto trader using Azure OpenAI gpt-5.2.\n"
    "Priorities: maximize risk-adjusted profit, minimize drawdown, avoid over-trading.\n"
    "- First, run a web_search on the symbols/market to gather fresh sentiment, news, and catalysts.\n"
    "- Focus on intraday/day-trading setups, not long holds. Prefer short holding periods.\n"
    "- Consider leverage only when conviction is high and risk is controlled; default to low/no leverage.\n"
    f"- Consider only the provided symbols and market snapshot.\n"
    f"- Do NOT exceed maxPositionUsd={snapshot.max_position_usd} USDT per trade.\n"
    f"- Only place a trade if your confidence >= {snapshot.min_confidence}; otherwise decline.\n"
    "- Keep at least 10% of USDT balance untouched for safety.\n"
    "- Be explicit about your reasoning in the final narrative.\n"
    f"- PAPER_TRADING={snapshot.paper_trading}. When true, just simulate orders via the tool."
  )

  trading_agent = Agent(
    name="Trading Agent",
    instructions=instructions,
    tools=[WebSearchTool(search_context_size="medium"), place_market_order, decline_trade],
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

  return {"narrative": narrative, "tool_results": tool_outputs}
