from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from openai import OpenAI

from .config import AppConfig
from .kucoin import KucoinAccount, KucoinClient, KucoinOrderRequest, KucoinTicker


@dataclass
class TradingSnapshot:
  tickers: Dict[str, KucoinTicker]
  balances: List[KucoinAccount]
  paper_trading: bool
  max_position_usd: float
  min_confidence: float


@dataclass
class ToolResult:
  name: str
  result: Any


def _build_openai_client(cfg: AppConfig) -> OpenAI:
  base_url = cfg.azure.endpoint.rstrip("/")
  deployment = cfg.azure.deployment
  return OpenAI(
    api_key=cfg.azure.api_key,
    base_url=f"{base_url}/openai/deployments/{deployment}",
    default_query={"api-version": cfg.azure.api_version},
  )


def run_trading_agent(
  cfg: AppConfig,
  snapshot: TradingSnapshot,
  kucoin: KucoinClient,
) -> dict[str, Any]:
  client = _build_openai_client(cfg)
  tools = [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web for latest market/news/sentiment on the symbols. Do this before trading.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Search query for tickers/market context."},
          },
          "required": ["query"],
        },
      },
    },
    {
      "type": "function",
      "function": {
        "name": "place_market_order",
        "description": (
          "Execute a market order on Kucoin. Respect maxPositionUsd and available balances. "
          "Use funds for quote size in USDT."
        ),
        "parameters": {
          "type": "object",
          "properties": {
            "symbol": {"type": "string", "description": "Symbol like BTC-USDT"},
            "side": {"type": "string", "enum": ["buy", "sell"]},
            "funds": {
              "type": "number",
              "description": "Quote amount in USDT to spend/receive. Must be <= maxPositionUsd.",
            },
          },
          "required": ["symbol", "side", "funds"],
        },
      },
    },
    {
      "type": "function",
      "function": {
        "name": "decline_trade",
        "description": (
          "Use this when conditions are not favorable or confidence is too low. "
          "Provide rationale to avoid unnecessary risk."
        ),
        "parameters": {
          "type": "object",
          "properties": {
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
          },
          "required": ["reason", "confidence"],
        },
      },
    },
  ]

  balances_by_currency: Dict[str, float] = {}
  for bal in snapshot.balances:
    balances_by_currency[bal.currency] = balances_by_currency.get(bal.currency, 0.0) + float(
      bal.available or 0
    )

  system_message = (
    "You are a disciplined quantitative crypto trader using Azure OpenAI gpt-5.2.\n"
    "Priorities: maximize risk-adjusted profit, minimize drawdown, avoid over-trading.\n"
    "- First, run a deep web search on the symbols/market to gather fresh sentiment, news, catalysts.\n"
    "- Focus on intraday/day-trading setups, not long holds. Prefer short holding periods.\n"
    "- Consider leverage only when conviction is high and risk is controlled; default to low/no leverage.\n"
    f"- Consider only the provided symbols and market snapshot.\n"
    f"- Do NOT exceed maxPositionUsd={snapshot.max_position_usd} USDT per trade.\n"
    f"- Only place a trade if your confidence >= {snapshot.min_confidence}; otherwise decline.\n"
    "- Keep at least 10% of USDT balance untouched for safety.\n"
    "- Be explicit about your reasoning in the final narrative.\n"
    f"- PAPER_TRADING={snapshot.paper_trading}. When true, just simulate orders via the tool."
  )

  user_content = {
    "tickers": {k: vars(v) for k, v in snapshot.tickers.items()},
    "balances": balances_by_currency,
    "paperTrading": snapshot.paper_trading,
    "maxPositionUsd": snapshot.max_position_usd,
    "minConfidence": snapshot.min_confidence,
    "guidance": "If you place an order, prefer market orders sized in USDT funds.",
  }

  messages: List[Dict[str, Any]] = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": [{"type": "text", "text": json.dumps(user_content)}]},
  ]

  tool_results: List[ToolResult] = []
  max_rounds = 3
  round_count = 0

  while round_count < max_rounds:
    round_count += 1
    response = client.chat.completions.create(
      model=cfg.azure.deployment,
      messages=messages,
      tools=tools,
      tool_choice="auto",
      temperature=0.2,
    )

    msg = response.choices[0].message if response.choices else None
    tool_calls = msg.tool_calls if msg else []

    if msg and tool_calls:
      messages.append(msg.model_dump(exclude_unset=True))
    elif msg and not tool_calls:
      return {
        "narrative": msg.content if msg else "No action",
        "tool_results": [tr.__dict__ for tr in tool_results],
      }
    else:
      break

    for call in tool_calls:
      if call.function is None:
        continue
      args = json.loads(call.function.arguments or "{}")

      if call.function.name == "web_search":
        query = args.get("query") or ""
        result = perform_web_search(query)
        tool_results.append(ToolResult(call.function.name, result))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

      elif call.function.name == "place_market_order":
        try:
          funds = float(args.get("funds", 0) or 0)
        except (TypeError, ValueError):
          funds = 0.0
        symbol = str(args.get("symbol", ""))
        side = "buy" if args.get("side") == "buy" else "sell"

        if funds <= 0 or not symbol:
          result = {"error": "Invalid funds or symbol"}
          tool_results.append(ToolResult(call.function.name, result))
          messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
          continue

        if funds > snapshot.max_position_usd:
          result = {"rejected": True, "reason": "Exceeds maxPositionUsd"}
          tool_results.append(ToolResult(call.function.name, result))
          messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
          continue

        order_req = KucoinOrderRequest(
          symbol=symbol,
          side=side,  # type: ignore[arg-type]
          type="market",
          funds=f"{funds:.2f}",
          clientOid=str(uuid.uuid4()),
        )

        result = (
          {"paper": True, "orderRequest": order_req.__dict__}
          if snapshot.paper_trading
          else kucoin.place_order(order_req).__dict__
        )

        tool_results.append(ToolResult(call.function.name, result))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

      elif call.function.name == "decline_trade":
        result = {
          "skipped": True,
          "reason": args.get("reason", "No reason supplied"),
          "confidence": args.get("confidence"),
        }
        tool_results.append(ToolResult(call.function.name, result))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

  return {
    "narrative": "No narrative.",
    "tool_results": [tr.__dict__ for tr in tool_results],
  }


def perform_web_search(query: str) -> Dict[str, Any]:
  if not query:
    return {"error": "Empty query"}
  try:
    resp = requests.get(
      "https://api.duckduckgo.com/",
      params={"q": query, "format": "json", "no_redirect": 1, "no_html": 1},
      timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    related = []
    for item in data.get("RelatedTopics", [])[:5]:
      if isinstance(item, dict) and item.get("Text"):
        related.append({"text": item.get("Text"), "first_url": item.get("FirstURL")})
    return {
      "query": query,
      "abstract": data.get("AbstractText", ""),
      "heading": data.get("Heading", ""),
      "related": related,
    }
  except Exception as exc:
    return {"error": f"search_failed: {exc}", "query": query}
