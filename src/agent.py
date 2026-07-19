from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List

import requests
from datetime import datetime, timezone
import contextlib

logger = logging.getLogger(__name__)

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
from agents.items import HandoffOutputItem, ToolCallItem, ToolCallOutputItem
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
from .edge import expectancy_size_factor, edge_stats, loss_streak_size_factor, symbol_bench_until
from .memory import MemoryStore
from .protection import should_block_chase
from .regime import (
  allow_reversal_long,
  allow_trend_aligned_short,
  block_alt_long_in_btc_downtrend,
  concentration_scale,
  conviction_size_factor,
  effective_min_confidence,
  regime_size_factor,
  resolve_gate_deadlock,
  reward_risk_ratio,
)
from .utils import normalize_symbol as _normalize_symbol


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
      logger.info("OpenAI tracing enabled with provided OPENAI_TRACE_API_KEY.")
  except Exception as exc:
    logger.warning("Tracing setup failed: %s", exc)

def setup_lstracing(cfg: AppConfig):
  """Register tracing processors once at startup."""
  ls_client = None
  try:
    if cfg.langsmith.enabled and cfg.langsmith.tracing:
      from langsmith import Client as LangsmithClient
      from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor

      # Head-sample runs so we stay under the LangSmith monthly unique-trace cap (the processor
      # was previously 429-ing on every run). 0.0–1.0; the client applies sampling to all exports.
      sample_rate = min(1.0, max(0.0, float(getattr(cfg.langsmith, "sample_rate", 1.0))))
      ls_client = LangsmithClient(
        api_key=cfg.langsmith.api_key,
        api_url=cfg.langsmith.api_url or None,
        tracing_sampling_rate=sample_rate,
      )
      logger.info("LangSmith tracing sample rate set to %.2f", sample_rate)
      processor = OpenAIAgentsTracingProcessor(
        client=ls_client,
        project_name=cfg.langsmith.project or None,
        tags=["trAIde", "openai-agents"],
        name="trAIde-agent",
      )
      add_trace_processor(processor)
      logger.info("LangSmith tracing enabled via OpenAIAgentsTracingProcessor (per-run, OpenAI traces retained)")
  except Exception as exc:
    logger.warning("Tracing setup failed: %s", exc)
  return ls_client

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
  spot_pending_orders: List[Dict[str, Any]] = field(default_factory=list)
  futures_pending_orders: List[Dict[str, Any]] = field(default_factory=list)
  financial_accounts: List[KucoinAccount] = field(default_factory=list)
  fees: Dict[str, Any] = field(default_factory=dict)
  trading_restricted: bool = False
  restriction_reason: str = ""


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


def _truncate_to_increment(value: float, increment: str) -> str:
  """Truncate *value* down to the nearest allowed increment (e.g. '0.01' → 2 decimals).

  KuCoin rejects orders whose size doesn't match the symbol's baseIncrement.
  We always truncate (floor) rather than round to avoid exceeding the available balance.
  """
  inc = float(increment)
  if inc <= 0:
    return f"{value:.8f}"
  # Number of decimal places implied by the increment string (e.g. '0.001' → 3).
  if "." in increment:
    decimals = len(increment.rstrip("0").split(".")[-1])
  else:
    decimals = 0
  truncated = math.floor(value / inc) * inc
  return f"{truncated:.{decimals}f}"


def _aggregate_account_totals(accounts: List[KucoinAccount]) -> Dict[str, Dict[str, float]]:
  totals: Dict[str, Dict[str, float]] = {}
  for acct in accounts:
    cur = acct.currency
    bucket = totals.setdefault(cur, {"available": 0.0, "holds": 0.0, "balance": 0.0})
    bucket["available"] += _to_float(acct.available)
    bucket["holds"] += _to_float(acct.holds)
    bucket["balance"] += _to_float(acct.balance)
  return totals


def _has_meaningful_balance(balance: Dict[str, float], threshold: float = 1e-12) -> bool:
  return (
    abs(balance.get("balance", 0.0)) > threshold
    or abs(balance.get("available", 0.0)) > threshold
    or abs(balance.get("holds", 0.0)) > threshold
  )


def _compact_balance_entry(currency: str, balance: Dict[str, float], venue_amount: float | None = None) -> Dict[str, Any]:
  entry: Dict[str, Any] = {
    "currency": currency,
    "total": balance.get("balance", 0.0),
  }
  if abs(balance.get("available", 0.0)) > 0:
    entry["free"] = balance.get("available", 0.0)
  if abs(balance.get("holds", 0.0)) > 0:
    entry["locked"] = balance.get("holds", 0.0)
  if venue_amount is not None and abs(venue_amount) > 0:
    entry["amount"] = venue_amount
  return entry


def _compact_venue_balances(totals: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
  entries: List[Dict[str, Any]] = []
  for currency in sorted(totals):
    balance = totals[currency]
    if not _has_meaningful_balance(balance):
      continue
    entries.append(_compact_balance_entry(currency, balance))
  return entries


def _build_compact_account_state(
  *,
  spot_totals: Dict[str, Dict[str, float]],
  funding_totals: Dict[str, Dict[str, float]],
  financial_totals: Dict[str, Dict[str, float]],
  futures_overview: Dict[str, Any] | None,
  futures_enabled: bool,
  futures_error: str | None = None,
) -> Dict[str, Any]:
  portfolio_index: Dict[str, Dict[str, Any]] = {}

  def merge_venue(venue: str, totals: Dict[str, Dict[str, float]]) -> None:
    for currency, balance in totals.items():
      if not _has_meaningful_balance(balance):
        continue
      entry = portfolio_index.setdefault(
        currency,
        {
          "currency": currency,
          "total": 0.0,
          "free": 0.0,
          "locked": 0.0,
          "venues": {},
        },
      )
      entry["total"] += balance.get("balance", 0.0)
      entry["free"] += balance.get("available", 0.0)
      entry["locked"] += balance.get("holds", 0.0)
      venue_amount = balance.get("balance", 0.0)
      if abs(venue_amount) > 0:
        entry["venues"][venue] = venue_amount

  merge_venue("spot", spot_totals)
  merge_venue("funding", funding_totals)
  merge_venue("financial", financial_totals)

  portfolio = []
  for currency in sorted(portfolio_index):
    entry = portfolio_index[currency]
    compact: Dict[str, Any] = {
      "currency": currency,
      "total": entry["total"],
      "venues": entry["venues"],
    }
    if abs(entry["free"]) > 0:
      compact["free"] = entry["free"]
    if abs(entry["locked"]) > 0:
      compact["locked"] = entry["locked"]
    portfolio.append(compact)

  futures_summary = {"enabled": futures_enabled}
  futures_free = 0.0
  if futures_overview:
    futures_currency = str(futures_overview.get("currency", "USDT")).upper()
    futures_free = _to_float(futures_overview.get("availableBalance"))
    futures_equity = _to_float(futures_overview.get("accountEquity") or futures_overview.get("marginBalance"))
    futures_pnl = _to_float(futures_overview.get("unrealisedPnl") or futures_overview.get("unrealisedPNL"))
    futures_summary = {
      "enabled": True,
      "currency": futures_currency,
      "available": futures_free,
      "equity": futures_equity,
      "unrealizedPnl": futures_pnl,
    }
    if abs(_to_float(futures_overview.get("positionMargin"))) > 0:
      futures_summary["positionMargin"] = _to_float(futures_overview.get("positionMargin"))
    if abs(_to_float(futures_overview.get("orderMargin"))) > 0:
      futures_summary["orderMargin"] = _to_float(futures_overview.get("orderMargin"))
    if abs(_to_float(futures_overview.get("riskRatio"))) > 0:
      futures_summary["riskRatio"] = _to_float(futures_overview.get("riskRatio"))
  if futures_error:
    futures_summary["error"] = futures_error

  total_free_usdt = (
    spot_totals.get("USDT", {}).get("available", 0.0)
    + funding_totals.get("USDT", {}).get("available", 0.0)
    + financial_totals.get("USDT", {}).get("available", 0.0)
    + futures_free
  )

  result: Dict[str, Any] = {
    "summary": {
      "futuresEnabled": futures_enabled,
      "assetCount": len(portfolio),
      "freeUsdt": {
        "spot": spot_totals.get("USDT", {}).get("available", 0.0),
        "funding": funding_totals.get("USDT", {}).get("available", 0.0),
        "financial": financial_totals.get("USDT", {}).get("available", 0.0),
        "futures": futures_free,
        "total": total_free_usdt,
      },
    },
    "portfolio": portfolio,
    "venues": {
      "spot": _compact_venue_balances(spot_totals),
      "funding": _compact_venue_balances(funding_totals),
      "financial": _compact_venue_balances(financial_totals),
      "futures": futures_summary,
    },
    "transferability": {
      "canUseFinancialFunds": True,
      "supportedDirections": [
        "financial_to_spot",
        "financial_to_futures",
        "spot_to_futures",
        "futures_to_spot",
      ],
    },
  }
  return result


def _base_currency(symbol: str) -> str:
  """Return the base currency for a trading symbol (e.g., BTC-USDT -> BTC)."""
  if not symbol or "-" not in symbol:
    return symbol or ""
  return symbol.split("-")[0].upper()


_CITATION_PUA_RE = re.compile("[\ue000-\uf8ff]")  # private-use chars wrapping citation markers
_CITATION_TOKEN_RE = re.compile(
  r"\bcite(?:turn\d+\w*)+\b"                          # 'citeturn0search0' fused tokens
  r"|\bturn\d+(?:search|news|view|fetch|image)\d+\b"  # bare 'turn0search0' fragments
)


def _strip_citation_tokens(text: str) -> str:
  """Remove model citation artifacts ('citeturn0search0'-style tokens + their invisible wrappers).

  The web-search tool wraps citation markers in Unicode private-use characters; logged or sent to
  Telegram they render as literal 'citeturn0search0' noise. Two passes: dropping the invisible
  wrappers first makes the token contiguous so the second pass can remove it whole.
  """
  if not text:
    return text
  return _CITATION_TOKEN_RE.sub("", _CITATION_PUA_RE.sub("", text)).strip()


def _screen_contracts(
  contracts: list,
  *,
  min_turnover: float,
  min_age_days: float,
  now: float,
  sort_by: str = "momentum",
  side: str = "both",
  top_n: int = 15,
  max_abs_change_pct: float | None = None,
) -> Dict[str, Any]:
  """Rank the full active-perp universe into a shortlist of tradable opportunities. Pure/testable.

  Filters to USDT perps that clear the liquidity floor (24h turnover) and the listing-age bar (same
  bar the entry guard enforces — no fresh micro-caps), then ranks by ``sort_by`` and optionally keeps
  only one side. This is the market EYES the research scout was missing: it can only look at coins it
  already names, so it never discovered the mover in a 500-coin field.
  """
  qualified: list[Dict[str, Any]] = []
  scanned = 0
  for c in contracts or []:
    if not isinstance(c, dict):
      continue
    sym = str(c.get("symbol") or "").strip().upper()
    if not sym.endswith("USDTM"):
      continue
    scanned += 1
    if str(c.get("status") or "Open").lower() != "open":
      continue
    turnover = _to_float(c.get("turnoverOf24h")) or 0.0
    if min_turnover > 0 and turnover < min_turnover:
      continue
    age_days = None
    fo = _to_float(c.get("firstOpenDate"))
    if fo:
      fo_s = fo / 1000.0 if fo > 1e12 else fo
      age_days = (now - fo_s) / 86400.0
      if min_age_days > 0 and age_days < min_age_days:
        continue
    chg = _to_float(c.get("priceChgPct"))
    chg_pct = chg * 100 if chg is not None else None
    # Extreme 24h movers are usually already extended and frequently fail the downstream ATR gate.
    # Excluding them here stops the research loop from repeatedly selecting an untradeable top mover
    # (LAB monopolised dozens of July 13 cycles this way) and rotates to the next liquid candidate.
    if max_abs_change_pct and chg_pct is not None and abs(chg_pct) > max_abs_change_pct:
      continue
    qualified.append({
      "symbol": _normalize_symbol(sym),
      "futuresSymbol": sym,
      "chgPct24h": round(chg_pct, 2) if chg_pct is not None else None,
      "turnover24hUsd": round(turnover),
      "funding": _to_float(c.get("fundingFeeRate")),
      "markPrice": _to_float(c.get("markPrice")) or _to_float(c.get("lastTradePrice")),
      "ageDays": round(age_days, 1) if age_days is not None else None,
    })

  s = (side or "both").lower()
  if s == "long":
    qualified = [r for r in qualified if (r["chgPct24h"] or 0) > 0]
  elif s == "short":
    qualified = [r for r in qualified if (r["chgPct24h"] or 0) < 0]

  sb = (sort_by or "momentum").lower()
  if sb == "gainers":
    qualified.sort(key=lambda r: r["chgPct24h"] or 0, reverse=True)
  elif sb == "losers":
    qualified.sort(key=lambda r: r["chgPct24h"] or 0)
  elif sb == "volume":
    qualified.sort(key=lambda r: r["turnover24hUsd"] or 0, reverse=True)
  else:  # momentum = biggest absolute move, liquidity as tiebreak
    qualified.sort(key=lambda r: (abs(r["chgPct24h"] or 0), r["turnover24hUsd"] or 0), reverse=True)

  n = max(1, min(int(top_n or 15), 40))
  return {"scanned": scanned, "qualified": len(qualified), "results": qualified[:n]}


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



def _resolve_allowed_spot_symbol(symbol: str, allowed_symbols: set[str]) -> str | None:
  """Resolve user input to an allowed spot symbol, accepting direct futures contract symbols."""
  spot_symbol = _normalize_symbol(symbol)
  if spot_symbol in allowed_symbols:
    return spot_symbol
  return None


def _format_snapshot(snapshot: TradingSnapshot, balances_by_currency: Dict[str, float]) -> str:
  spot_accounts = snapshot.spot_accounts or snapshot.balances
  financial_accounts = snapshot.financial_accounts
  all_accounts = snapshot.all_accounts or (spot_accounts + financial_accounts)
  spot_totals = _aggregate_account_totals(spot_accounts)
  financial_totals = _aggregate_account_totals(financial_accounts)
  funding_accounts = [acct for acct in all_accounts if getattr(acct, "type", "") == "main"]
  funding_totals = _aggregate_account_totals(funding_accounts)
  compact_accounts = _build_compact_account_state(
    spot_totals=spot_totals,
    funding_totals=funding_totals,
    financial_totals=financial_totals,
    futures_overview=snapshot.futures_account,
    futures_enabled=snapshot.futures_enabled,
  )
  spot_usdt = spot_totals.get("USDT", {}).get("available", 0.0)
  financial_usdt = financial_totals.get("USDT", {}).get("available", 0.0)
  futures_usdt = _to_float(snapshot.futures_account.get("availableBalance")) if snapshot.futures_account else 0.0

  user_content = {
    "coins": snapshot.coins,
    "tickers": {k: vars(v) for k, v in snapshot.tickers.items()},
    "balances": balances_by_currency,
    "availableUsdt": {
      "spot": spot_usdt,
      "financial": financial_usdt,
      "futures": futures_usdt,
      "total": spot_usdt + financial_usdt + futures_usdt,
    },
    "transferability": {
      "financialTransferAvailable": True,
      "supportedDirections": [
        "financial_to_spot",
        "financial_to_futures",
        "spot_to_futures",
        "futures_to_spot",
      ],
      "note": "Financial/Earn funds are transferable out via redeem->transfer; transfers into Earn are not supported by this bot.",
    },
    "accountState": compact_accounts,
    "paperTrading": snapshot.paper_trading,
    "maxPositionUsd": snapshot.max_position_usd,
    "minConfidence": snapshot.min_confidence,
    "maxLeverage": snapshot.max_leverage,
    "futuresEnabled": snapshot.futures_enabled,
    "drawdownPct": snapshot.drawdown_pct,
    "drawdownPctSpot": snapshot.drawdown_pct_spot,
    "drawdownPctFutures": snapshot.drawdown_pct_futures,
    "totalUsdt": snapshot.total_usdt,
    "futuresPositions": snapshot.futures_positions,
    "guidance": "New exposure must use an atomic-bracket futures limit; market orders are close/emergency-only.",
    "stops": {
      "spot": snapshot.spot_stop_orders,
      "futures": snapshot.futures_stop_orders,
    },
    "pendingLimitOrders": {
      "spot": snapshot.spot_pending_orders,
      "futures": snapshot.futures_pending_orders,
    },
    "fees": snapshot.fees if hasattr(snapshot, "fees") else {},
    "tradingRestricted": snapshot.trading_restricted,
    "restrictionReason": snapshot.restriction_reason if snapshot.trading_restricted else None,
  }
  return json.dumps(user_content)


async def _run_agent_with_tracing(
  trading_agent: Agent,
  input_payload: str,
  cfg: AppConfig,
  langsmith_ctx: Any,
  run_name: str,
  unique_trace_id: str,
) -> Any:
  """Execute the agent run inside a tracing context."""
  with langsmith_ctx:
    provider = get_trace_provider()
    tr = provider.create_trace(run_name, trace_id=unique_trace_id)
    tr.start(mark_as_current=True)
    try:
      return await asyncio.wait_for(
        Runner.run(trading_agent, input_payload, max_turns=cfg.agent_max_turns),
        timeout=20 * 60,
      )
    finally:
      try:
        tr.finish(reset_current=True)
      except Exception as exc:
        logger.warning("Trace cleanup failed: %s", exc)


# Tool calls whose outputs are bulky/uninteresting market data — never surfaced as research activity.
_RESEARCH_NOISY_TOOLS = {
  "analyze_market_context", "fetch_recent_candles", "fetch_orderbook", "fetch_futures_candles",
  "fetch_futures_orderbook", "fetch_futures_mark_price", "fetch_funding_rate",
  "fetch_open_interest", "fetch_contract_details", "latest_items", "list_coins",
}


def _summarize_research(tool_name: str, output: Any) -> str | None:
  """One-line summary of a Research Agent tool result for Telegram/dashboard surfacing."""
  if not isinstance(output, dict) or output.get("error"):
    return None
  title = output.get("title")  # log_research returns the plan entry directly
  if title:
    summ = (output.get("summary") or "").strip()
    return (f"{title} — {summ}" if summ else str(title))[:220]
  added = output.get("added")
  if isinstance(added, dict) and added.get("symbol"):
    reason = (added.get("reason") or "").strip()
    return f"added coin {added['symbol']}" + (f" — {reason}" if reason else "")
  removed = output.get("removed")
  if isinstance(removed, dict):
    if removed.get("symbol") and not removed.get("title"):  # remove_coin
      reason = (removed.get("reason") or "").strip()
      return f"removed coin {removed['symbol']}" + (f" — {reason}" if reason else "")
    if removed.get("title"):  # remove_source
      return str(removed["title"])[:220]
  source = output.get("source")
  if isinstance(source, dict) and source.get("title"):
    return str(source["title"])[:220]
  sentiment = output.get("sentiment")
  if isinstance(sentiment, dict) and sentiment.get("symbol"):
    return f"sentiment {sentiment['symbol']}={sentiment.get('score')}"
  if tool_name and tool_name not in _RESEARCH_NOISY_TOOLS:
    return tool_name.replace("_", " ")
  return None


def _collect_run_artifacts(new_items: List[Any]) -> Dict[str, Any]:
  """Extract tool outputs, agent handoffs, per-agent attribution, and Research Agent activity
  from a Runner result's items.

  Single forward pass: a ToolCallItem always precedes its ToolCallOutputItem, so the
  call_id -> tool name map is populated before we need it. Defensive throughout — a SDK shape
  change must never break the trading loop.
  """
  tool_outputs: List[Any] = []
  handoff_events: List[Dict[str, str]] = []
  research_activity: List[str] = []
  agents_used: set[str] = set()
  call_tool_names: Dict[str, str] = {}

  for item in new_items:
    try:
      agent_name = getattr(getattr(item, "agent", None), "name", "") or ""
      if agent_name:
        agents_used.add(agent_name)
      if isinstance(item, ToolCallItem):
        raw = getattr(item, "raw_item", None)
        cid = getattr(raw, "call_id", None) or (raw.get("call_id") if isinstance(raw, dict) else None)
        tname = getattr(raw, "name", None) or (raw.get("name") if isinstance(raw, dict) else None)
        if cid and tname:
          call_tool_names[cid] = tname
      elif isinstance(item, HandoffOutputItem):
        src = getattr(getattr(item, "source_agent", None), "name", "") or "?"
        tgt = getattr(getattr(item, "target_agent", None), "name", "") or "?"
        handoff_events.append({"from": src, "to": tgt})
      elif isinstance(item, ToolCallOutputItem):
        tool_outputs.append(item.output)
        raw = getattr(item, "raw_item", None)
        cid = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
        if agent_name == "Research Agent":
          summary = _summarize_research(call_tool_names.get(cid, ""), item.output)
          if summary:
            research_activity.append(summary)
    except Exception as exc:
      logger.debug("Run-item inspection skipped an item: %s", exc)

  return {
    "tool_outputs": tool_outputs,
    "handoffs": handoff_events,
    "research": research_activity,
    "agents_used": agents_used,
  }


def run_trading_agent(
  cfg: AppConfig,
  snapshot: TradingSnapshot,
  kucoin: KucoinClient,
  kucoin_futures: KucoinFuturesClient | None = None,
  openai_client: AsyncAzureOpenAI | None = None,
  langsmith_client: Any | None = None,
  recent_fills: Dict[str, Any] | None = None,
  force_research: bool = False,
  safety_state: Any | None = None,
  entry_token: str | None = None,
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

  # Aggregate financial (Earn/Pool-X) balances as available funds
  if snapshot.financial_accounts:
    financial_totals = _aggregate_account_totals(snapshot.financial_accounts)
    for cur, totals in financial_totals.items():
      available = totals.get("available", 0.0)
      if available:
        balances_by_currency[cur] = balances_by_currency.get(cur, 0.0) + available
  unique_trace_id = gen_trace_id()
  unique_span_id = gen_span_id()

  allowed_symbols = set(snapshot.tickers.keys())
  # A symbol removed from the research universe must remain manageable until every live futures
  # position and pending order is flat/cancelled. Otherwise the next run cannot close or protect it.
  for row in (snapshot.futures_positions or []) + (snapshot.futures_pending_orders or []):
    if not isinstance(row, dict):
      continue
    live_symbol = _normalize_symbol(str(row.get("symbol") or ""))
    if live_symbol:
      allowed_symbols.add(live_symbol)
  # Use the configured learning horizon here too.  The per-run store previously defaulted to seven
  # days and pruned outcomes that the main loop intended to retain for longer, making adaptive edge
  # controls forget losses simply because the agent ran.
  memory = MemoryStore(cfg.memory_file, retention_days=cfg.retention_days)
  permanent_notes = memory.get_permanent_notes()
  temporary_notes = memory.consume_temporary_notes()
  current_prices = {sym: float(t.price) for sym, t in snapshot.tickers.items()}

  # Shared state: daily gate info per symbol, written by analyze_market_context,
  # read by order functions to hard-block counter-daily-trend entries.
  _daily_gate_state: Dict[str, Dict[str, Any]] = {}

  # BTC daily regime, computed at most once per run (cached). Drives the correlation gate that
  # blocks alt longs while BTC is in a confirmed daily downtrend.
  _btc_bias_cache: Dict[str, str] = {}

  def _btc_daily_bias() -> str:
    """Return BTC's daily bias ('bullish'|'bearish'|'neutral'), cached per run.

    Prefers the value analyze_market_context already computed this run; otherwise fetches BTC 1D
    candles once and derives it. Never raises — defaults to 'neutral' on any failure (so the gate
    fails open and never blocks trading because of a data hiccup)."""
    if "v" in _btc_bias_cache:
      return _btc_bias_cache["v"]
    bias = "neutral"
    gate = _daily_gate_state.get("BTC-USDT")
    if gate and gate.get("daily_bias"):
      bias = str(gate.get("daily_bias"))
    else:
      try:
        end_at = int(time.time())
        start_at = end_at - 60 * 86400
        candles = kucoin.get_candles("BTC-USDT", interval="1day", start_at=start_at, end_at=end_at)
        if candles:
          snap = summarize_interval(candles_to_dataframe(candles), "1day")
          tb = str(snap.get("trend_bias") or "neutral")
          bias = "bearish" if tb == "bearish" else ("bullish" if tb == "bullish" else "neutral")
      except Exception as exc:
        logger.debug("BTC daily bias computation failed (correlation gate fails open): %s", exc)
    _btc_bias_cache["v"] = bias
    return bias

  _edge_cache: Dict[str, Any] = {}

  def _edge_state() -> Dict[str, Any]:
    """Adaptive edge posture for this run, computed once from rolling realized closes.

    Returns {stats, required_rr, size_factor, bench: {symbol: until_ts}}. Never raises —
    on any failure it degrades to the static config behavior (base RR floor, full size,
    nothing benched), so the edge controller can only tighten risk, never break trading.
    """
    if "v" in _edge_cache:
      return _edge_cache["v"]
    state: Dict[str, Any] = {
      "stats": {},
      "required_rr": cfg.trading.min_futures_rr,
      "size_factor": 1.0,
      "direction_size_factor": {},
      "symbol_size_factor": {},
      "bench": {},
    }
    try:
      if cfg.edge.enabled:
        closes = memory.realized_closes(limit=max(cfg.edge.lookback_trades * 2, 50))
        stats = edge_stats(closes, cfg.edge.lookback_trades)
        state["stats"] = stats
        # Keep targets structural. Stretching TP farther because old trades lost made fills/targets
        # less reachable; weak evidence now adapts capital-at-risk instead (below).
        state["required_rr"] = cfg.trading.min_futures_rr
        state["size_factor"] = loss_streak_size_factor(stats.get("loss_streak", 0), cfg.edge)
        state["direction_size_factor"] = {
          direction: expectancy_size_factor(stats, cfg.edge, direction=direction)
          for direction in (stats.get("per_direction") or {})
        }
        state["symbol_size_factor"] = {
          symbol: expectancy_size_factor(stats, cfg.edge, symbol=symbol)
          for symbol in (stats.get("per_symbol") or {})
        }
        bench: Dict[str, int] = {}
        now = int(time.time())
        for sym in {c.get("symbol") for c in closes if c.get("symbol")}:
          until = symbol_bench_until([c for c in closes if c.get("symbol") == sym], cfg.edge)
          if until > now:
            bench[sym] = until
        state["bench"] = bench
        if bench:
          logger.info("ADAPTIVE EDGE: benched symbols: %s", ", ".join(f"{s} ({(u - now) / 3600:.1f}h left)" for s, u in bench.items()))
        if state["size_factor"] < 1.0:
          logger.info("ADAPTIVE EDGE: loss streak %d — entry size factor %.2f", stats.get("loss_streak", 0), state["size_factor"])
        reduced_directions = {
          key: value for key, value in state["direction_size_factor"].items() if value < 1.0
        }
        if reduced_directions:
          logger.info("ADAPTIVE EDGE: negative direction expectancy — risk factors %s", reduced_directions)
    except Exception as exc:
      logger.warning("Adaptive edge computation failed (falling back to static guards): %s", exc)
    _edge_cache["v"] = state
    return state

  # Configured futures margin mode. The bot manages leverage via the CROSS leverage endpoint, so
  # the cross-leverage call is only issued in cross mode — in isolated mode the order's own
  # `leverage` field applies and calling the cross endpoint just errors ("switch to cross margin").
  _futures_margin_mode = (cfg.kucoin_futures.margin_mode or "cross").strip().lower()

  def _apply_cross_leverage(fsym: str, lev: float, mode: str | None, *, context: str = "") -> None:
    """Set cross leverage only when the order's margin mode is cross; no-op (never raises) otherwise.

    KuCoin's leverage endpoint is cross-only, so calling it in isolated mode just errors with
    'switch to cross margin' — and it is unnecessary there because the order carries its own
    `leverage` field. Keyed on the order's actual mode (not config) so fallback paths stay correct."""
    if not kucoin_futures or (mode or "").strip().lower() != "cross":
      return
    try:
      kucoin_futures.set_leverage(fsym, lev)
    except Exception as exc:
      logger.warning("set_leverage failed%s (continuing): %s", f" for {context}" if context else "", exc)

  def _repair_allowed_symbol(requested_symbol: str) -> str | None:
    candidate = _normalize_symbol(requested_symbol)
    if not candidate:
      return None
    if candidate in allowed_symbols:
      return candidate

    active_source = memory.get_coins(default=cfg.trading.coins) if cfg.trading.flexible_coins_enabled else cfg.trading.coins
    normalized_active: list[str] = []
    seen_active: set[str] = set()
    for sym in active_source:
      norm = _normalize_symbol(sym)
      if norm and norm not in seen_active:
        normalized_active.append(norm)
        seen_active.add(norm)

    if candidate not in seen_active:
      return None

    try:
      ticker = kucoin.get_ticker(candidate)
    except Exception as exc:
      logger.warning("Runtime symbol repair failed for %s -> %s: %s", requested_symbol, candidate, exc)
      return None

    snapshot.tickers[candidate] = ticker
    if candidate not in snapshot.coins:
      snapshot.coins.append(candidate)
    allowed_symbols.add(candidate)
    current_prices[candidate] = float(ticker.price)
    return candidate

  positions = memory.positions(current_prices)
  # Reconcile tracked positions with live spot balances to avoid phantom exposure when trades happen outside the agent.
  spot_balances_by_currency = {
    cur.upper(): _to_float(totals.get("available"))
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
  latest_plan_entry = memory.latest_plan()
  research_plans = memory.latest_items("research", limit=5)
  research_notes: list[Dict[str, Any]] = []
  if isinstance(research_plans, dict) and not research_plans.get("error"):
    for item in research_plans.get("items", []):
      if not item:
        continue
      author = item.get("author")
      if author and author != "Research Agent":
        continue
      research_notes.append(item)
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
      logger.warning("Unable to fetch futures contract %s: %s", futures_symbol, exc)
    return None

  def _spot_position_info(symbol: str) -> Dict[str, float | None] | None:
    """Return current net size and avg entry for a spot position. avg_entry may be None for externally-acquired coins."""
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
      avg_entry_f = 0.0
    return {"net": net, "avg_entry": avg_entry_f if avg_entry_f > 0 else None}

  def _spot_position_size(symbol: str) -> float:
    """Return current live spot size for a symbol, even if avg entry is unknown."""
    pos = positions.get(symbol)
    if not pos:
      return 0.0
    try:
      net = float(pos.get("netSize") or 0.0)
    except Exception:
      return 0.0
    return net if net > 0 else 0.0

  def _fee_adjusted_breakeven(avg_entry: float, fee_rate: float) -> float:
    """Breakeven price that covers taker fees on both entry and exit."""
    fee_rate = max(0.0, float(fee_rate))
    exit_factor = max(1e-9, 1.0 - fee_rate)
    return avg_entry * (1.0 + fee_rate) / exit_factor
  
  def _stop_distance_ok(symbol: str, side: str, stop_price: float, ref_price: float, fee_rate: float) -> tuple[bool, str | None]:
    """Validate stop distance to avoid churn; ensures correct side and a minimum % away."""
    if stop_price <= 0 or ref_price <= 0:
      return False, "Invalid price for stop validation"
    min_pct = max(0.003, 3 * fee_rate)  # at least 0.3% or 3x fee to clear costs
    side_norm = (side or "").lower()
    if side_norm == "sell":
      if stop_price >= ref_price:
        return False, "Stop for sell must be below entry/mark"
      if (ref_price - stop_price) / ref_price < min_pct:
        return False, f"Stop too tight (<{min_pct*100:.2f}% from price)"
    elif side_norm == "buy":
      if stop_price <= ref_price:
        return False, "Stop for buy must be above entry/mark"
      if (stop_price - ref_price) / ref_price < min_pct:
        return False, f"Stop too tight (<{min_pct*100:.2f}% from price)"
    return True, None
  
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
      logger.warning("LangSmith tracing_context unavailable; skipping per-run LangSmith context.")
    except Exception as exc:
      logger.warning("LangSmith run context init failed: %s", exc)

  # Tools live in src/tools.py; build them bound to this run's context (lazy import avoids a cycle).
  from .tools import build_tools
  _tools = build_tools(SimpleNamespace(
    cfg=cfg,
    kucoin=kucoin,
    kucoin_futures=kucoin_futures,
    memory=memory,
    snapshot=snapshot,
    allowed_symbols=allowed_symbols,
    balances_by_currency=balances_by_currency,
    fees=fees,
    _daily_gate_state=_daily_gate_state,
    _futures_margin_mode=_futures_margin_mode,
    _apply_cross_leverage=_apply_cross_leverage,
    _btc_daily_bias=_btc_daily_bias,
    _edge_state=_edge_state,
    _fee_adjusted_breakeven=_fee_adjusted_breakeven,
    _get_contract_spec=_get_contract_spec,
    _repair_allowed_symbol=_repair_allowed_symbol,
    _spot_position_info=_spot_position_info,
    _spot_position_size=_spot_position_size,
    _stop_distance_ok=_stop_distance_ok,
    safety_state=safety_state,
    entry_token=entry_token,
  ))
  place_market_order = _tools.place_market_order
  place_limit_order = _tools.place_limit_order
  place_spot_stop_order = _tools.place_spot_stop_order
  cancel_spot_stop_order = _tools.cancel_spot_stop_order
  cancel_spot_limit_order = _tools.cancel_spot_limit_order
  list_spot_stop_orders = _tools.list_spot_stop_orders
  set_spot_position_protection = _tools.set_spot_position_protection
  decline_trade = _tools.decline_trade
  fetch_recent_candles = _tools.fetch_recent_candles
  fetch_orderbook = _tools.fetch_orderbook
  analyze_market_context = _tools.analyze_market_context
  plan_spot_position = _tools.plan_spot_position
  place_futures_market_order = _tools.place_futures_market_order
  place_futures_limit_order = _tools.place_futures_limit_order
  place_futures_stop_order = _tools.place_futures_stop_order
  cancel_futures_order = _tools.cancel_futures_order
  list_futures_stop_orders = _tools.list_futures_stop_orders
  list_futures_positions = _tools.list_futures_positions
  get_recent_fills = _tools.get_recent_fills
  get_closed_positions = _tools.get_closed_positions
  fetch_funding_rate = _tools.fetch_funding_rate
  fetch_open_interest = _tools.fetch_open_interest
  scan_futures_market = _tools.scan_futures_market
  fetch_futures_orderbook = _tools.fetch_futures_orderbook
  fetch_futures_mark_price = _tools.fetch_futures_mark_price
  fetch_futures_candles = _tools.fetch_futures_candles
  fetch_contract_details = _tools.fetch_contract_details
  set_futures_position_protection = _tools.set_futures_position_protection
  transfer_funds = _tools.transfer_funds
  fetch_account_state = _tools.fetch_account_state
  refresh_fee_rates = _tools.refresh_fee_rates
  save_trade_plan = _tools.save_trade_plan
  latest_plan = _tools.latest_plan
  latest_items = _tools.latest_items
  clear_plans = _tools.clear_plans
  set_auto_trigger = _tools.set_auto_trigger
  list_triggers = _tools.list_triggers
  list_coins = _tools.list_coins
  add_coin = _tools.add_coin
  remove_coin = _tools.remove_coin
  log_sentiment = _tools.log_sentiment
  log_decision = _tools.log_decision
  fetch_kucoin_news = _tools.fetch_kucoin_news
  fetch_coindesk_news = _tools.fetch_coindesk_news
  log_research = _tools.log_research
  add_source = _tools.add_source
  remove_source = _tools.remove_source

  instructions = (
    "You are a quantitative intraday crypto trader. Your mission is to grow the account by finding and executing profitable trades "
    "with proper risk management (TP + SL on every position).\n\n"

    "## CRITICAL — Trade only when there is real edge:\n"
    "- Your default is to screen efficiently and place a trade only when fresh, complete data shows a liquid setup "
    "with positive expected value after fees that clears every code-enforced risk and edge gate.\n"
    "- Declining is correct when no setup naturally offers a structure-based invalidation and reachable target at the "
    "required reward:risk, or when data-quality, cooldown, volatility, concentration, or circuit-breaker gates block it.\n"
    "- Never manufacture confidence, loosen protection, increase risk, or force a fill merely to create activity.\n"
    "- You are FULLY AUTONOMOUS. NEVER write 'If you want...', 'Would you like...', or 'Do you want...'. "
    "Just execute. No one is reading your output interactively.\n"
    "- DIVERSIFY: if the primary coin is blocked or has no edge, move to the next screened candidate. "
    "Do NOT spend the entire run on a single symbol.\n"
    "- EXCEPTION (no-setup tape): when the majors (BTC/ETH) are gated in BOTH directions — daily blocks one side, "
    "intraday blocks the other — and no liquid, established coin offers a clean setup, then STANDING ASIDE IS THE "
    "CORRECT ACTION, not a failure. Do NOT reach for low-liquidity, freshly-listed, or high-beta altcoins (especially "
    "longs while BTC's daily is bearish) just to manufacture a trade — that is how the account took its worst loss. "
    "Forcing a low-quality trade in a no-setup tape is worse than declining.\n\n"

    "## STEP 1 — Account audit (always first):\n"
    "- The input already contains a fresh account snapshot, balances, positions, stops, and pending orders. Use it first; "
    "call account/list tools only if a write changed state or the snapshot is ambiguous.\n"
    "- For any open position missing a TP or SL: use set_spot_position_protection or set_futures_position_protection "
    "to add the bracket BEFORE looking for new trades.\n"
    "- Cancel stale stop orders (stop exists but no position) via cancel_spot_stop_order.\n"
    "- Check the 'staleStops' field. Cancel an orphaned stop only when no position exists. For an open position, use "
    "set_spot_position_protection or set_futures_position_protection to place and confirm valid replacement protection; "
    "never cancel the only loss-side stop first. If replacement fails, retain the existing protection and report it.\n"
    "- Check 'pendingLimitOrders' in your input for any limit entries still waiting to fill:\n"
    "  * If a pending order's symbol now appears in open positions (it filled): immediately place bracket TP/SL via "
    "set_spot_position_protection or set_futures_position_protection.\n"
    "  * If a pending order was placed more than ~30 minutes ago and has not filled: cancel it via "
    "cancel_spot_limit_order (spot) or cancel_futures_order (futures) and reassess the setup.\n"
    "  * If a pending order is still within its expiry window: leave it open. Deterministic code expires bot entries; "
    "do not cancel early merely because it has not filled or because you prefer a slightly different price. Cancel early "
    "only when fresh evidence invalidates the thesis (directional 1h/daily flip, broken structure, or invalid bracket).\n\n"

    "## STEP 1b — Review protection on existing positions:\n"
    "- Check the 'positions' field in your input — it lists every coin you hold in spot with netSize, avgEntry, "
    "unrealizedPnl, and currentPrice.\n"
    "- Prefer exchange TP/SL orders for exits. Use a reduce-only market close only for an emergency, hard risk limit, "
    "or clear thesis invalidation — never as a new entry.\n"
    "- If a position has no TP or SL: set them immediately via set_spot_position_protection or set_futures_position_protection.\n"
    "- Protection is monotonic after entry: a replacement SL may preserve or improve protection but must never widen "
    "the existing loss. Place and confirm the new stop before retiring the old one.\n"
    "- Existing exposure is not grandfathered past thesis failure: if fresh 15m AND 1h biases both oppose the position, "
    "treat that as a confirmed intraday reversal even when 4h/1D still lag. Reassess for a reduce-only close or a "
    "structurally tighter stop; never keep or add merely because the daily trend still agrees.\n"
    "- Revise an SL or TP only to a defensible structural level; do not mechanically squeeze a bracket to make a metric pass.\n"
    "- The code-driven protection loop runs every poll and may move a stop to breakeven or close a position at the "
    "hard unrealized-loss equity cap. Never re-open or weaken protection to override that safety action.\n"
    "- For positions with unknown avgEntry: estimate a reasonable SL based on current support/ATR and set protection.\n\n"

    "## STEP 1c — Portfolio health review (unlisted/unknown-entry holdings):\n"
    "- Check 'unlistedHoldings' in your input: these are coins you hold but that were NOT in your active coin list, "
    "often bought externally. They have unknown entry prices.\n"
    "- For EACH unlisted holding:\n"
    "  1. Review recent Research Agent notes and current KuCoin/CoinDesk news for project health and fundamentals; "
    "hand off to the Research Agent if broad-web verification is genuinely needed.\n"
    "  2. Call analyze_market_context (15min + 1hour) to assess current trend and momentum (also fetches 4H + 1D automatically).\n"
    "  3. Decide: HOLD (if fundamentals are strong and trend is recovering) or CLOSE (if project is declining, "
    "no catalyst, or capital is better deployed elsewhere).\n"
    "  4. If HOLD: add the coin to your universe via add_coin, set proper TP/SL via set_spot_position_protection.\n"
    "  5. If CLOSE: sell via place_market_order with rationale containing 'portfolio review' or 'close position', "
    "then cancel any stale stop orders for that symbol.\n"
    "- A position with no recovery catalyst is a strong candidate for closing to free capital.\n"
    "- Do NOT keep positions indefinitely out of hope — evaluate objectively with data.\n\n"

    "## STEP 1d — Review recent events (triggered TP/SL):\n"
    "- Check the 'recentEvents' field in your input. If present, orders were filled or positions were closed since last round.\n"
    "- 'closedPositions': futures positions closed by triggered TP/SL — review the realized PnL, ROE, and close type.\n"
    "- 'spotFills' / 'futuresFills': individual trade executions — check if stop orders triggered.\n"
    "- Use this feedback to adapt: if a SL triggered, assess whether the stop was too tight or the entry was wrong.\n"
    "  If a TP triggered, note the profit and consider whether you left money on the table.\n"
    "- Call get_recent_fills or get_closed_positions for more detail if needed.\n\n"

    "## STEP 2 — Research (required before every entry decision):\n"
    "- Call analyze_market_context for each coin — it fetches 15m + 1h + 4h + 1D timeframes automatically. "
    "An entry requires dataQuality.ok=true and analysis no more than 15 minutes old; if either condition fails, refresh "
    "once and otherwise stand aside.\n"
    "  The 4H timeframe has the highest intraday weight (40%). The 1D acts as a HARD REGIME GATE:\n"
    "  - Counter-daily trades are BLOCKED at the code level. If daily_bias='bearish', buy/long orders are rejected. "
    "If daily_bias='bullish', sell/short orders are rejected. This is NOT optional.\n"
    "  - EXCEPTION — confirmed reversal (BOTH directions): the daily trend lags at turning points. A LONG "
    "against a bearish daily is allowed IF 1h AND 15m have both turned bullish and confidence >= 0.80 "
    "(majors only — alt longs stay blocked by the correlation gate). Symmetrically, a SHORT against a "
    "bullish daily is allowed IF 1h AND 15m have both turned bearish and confidence >= 0.80. Confirmed "
    "turns only — never knife-catch or fade mere strength/weakness without both timeframes agreeing.\n"
    "  - If the 1D confirms your intraday bias, strength is boosted (e.g., moderate -> strong).\n"
    "  - Check 'daily_bias' and 'daily_gate_applied' in the summary. Trade WITH the daily trend, not against it.\n"
    "  When futures are enabled, it also returns a 'futures' field with funding rate, open interest, basis, OI-price signal, and funding divergence.\n"
    "- The summary includes: weighted_score (-1 to +1), daily_bias, daily_gate_applied, timeframe_conflict (bool), and volume_profile (POC, VAH, VAL).\n"
    "- **Volume Profile levels**: POC = Point of Control (highest volume price, acts as magnet). "
    "VAH = Value Area High (resistance). VAL = Value Area Low (support). Use these for entry/exit targeting:\n"
    "  - For mean-reversion: enter near VAL (longs) or VAH (shorts), target POC.\n"
    "  - For trend-following: use POC as pullback entry level; target VAH (bullish) or VAL (bearish).\n"
    "- Call fetch_recent_candles if you need raw price detail to set precise TP/SL levels.\n"
    "- Broad-web discovery belongs to the Research Agent during flat-book research handoffs, where findings are cached "
    "with log_research. On ordinary trading runs, use recent research plus fetch_kucoin_news/fetch_coindesk_news only "
    "when a catalyst is material to the thesis; do not repeat the same news search on every poll.\n"
    "- Use fetch_orderbook when you need microstructure (depth, imbalance) to time entry or set tight stops.\n"
    "- Assign a sentiment score 0–1 from news. If sentiment_filter_enabled and score < sentiment_min_score, skip buys.\n\n"
    "- News evidence must name the same asset/project and include a traceable source. Uncited, stale, or unrelated "
    "articles are neutral and must never be repurposed as a symbol-specific catalyst.\n\n"

    "## STEP 2b — Futures-specific research (when considering futures trades):\n"
    "- analyze_market_context now returns pre-computed futures signals. Read them carefully:\n"
    "  - **oiPriceSignal** compares two timestamped, real open-interest samples: price+OI up='strong_trend', "
    "price up/OI down='short_covering', price down/OI up='aggressive_shorts', and price+OI down='long_capitulation'. "
    "It is 'neutral' until enough observations exist or changes are meaningful. Never infer OI direction from volume "
    "or from the absolute OI level.\n"
    "  - **fundingDivergence**: 'hidden_strength' (bullish despite negative funding), "
    "'hidden_weakness' (bearish despite positive funding), 'aligned', 'neutral'.\n"
    "- Treat OI/funding as secondary confirmation, never as a standalone entry, exit, or add-on instruction. "
    "Short covering can weaken a rally thesis and long capitulation can flag exhaustion, but both still require "
    "price structure, multi-timeframe confirmation, liquidity, and acceptable net reward:risk.\n"
    "- Call fetch_funding_rate for detailed rate history, fetch_open_interest for OI levels.\n"
    "- Call fetch_futures_mark_price to check basis. Large basis = mean reversion risk.\n"
    "- Call fetch_contract_details to check multiplier, maxLeverage, tick size, and fees before sizing.\n\n"

    + (
    "## STEP 2c — Regime-aware strategy (range vs. trend):\n"
    "- analyze_market_context now returns a 'market_regime' field: 'trending', 'ranging', or 'squeeze'.\n"
    "- The summary also includes 'market_regime' with multi-timeframe confirmation.\n"
    "- ADX < 20 + BBW 2-6% = RANGING. ADX > 25 = TRENDING. BBW < 2% + ADX < 20 = SQUEEZE.\n\n"

    "**When market_regime is 'ranging' (confirmed by summary):**\n"
    "- STRATEGY: Mean-reversion. Buy low, sell/short high within the range.\n"
    "- LONG ENTRY: Price near BB lower band AND (RSI < 35 OR Stochastic %K < 20). Target: BB midline.\n"
    "- SHORT ENTRY (futures): Price near BB upper band AND (RSI > 65 OR Stochastic %K > 80). Target: BB midline.\n"
    "- TP: Set at BB midline (the mean). This is a conservative target — do NOT aim for the opposite band.\n"
    "- SL: 1.0 ATR beyond the BB band you entered at. Example: long near BB lower at 80,800 with ATR=200, SL=80,600.\n"
    "- SIZE: 50-70% of normal position size. Range trades are higher frequency, lower conviction.\n"
    "- CRITICAL: If the entry would yield less than 0.3% profit after fees (2x round-trip), skip it. "
    "The range must be wide enough to profit from.\n"
    "- Alternating long/short range trades must be sequential: fully close the current lifecycle with reduce_only before "
    "opening the opposite direction. Never implicitly reverse or stack conflicting same-symbol exposure.\n"
    "- Use reduce_only=True to take profit on a portion when price reaches the midline.\n\n"

    "**When market_regime is 'squeeze':**\n"
    "- A Bollinger Band squeeze means volatility is compressed and a big move is coming.\n"
    "- Do NOT enter new range trades during a squeeze — the breakout will stop you out.\n"
    "- On existing positions, ratchet a stop only when current structure supports the new level; never mechanically "
    "squeeze it into market noise. Reduce size on any new entries.\n"
    "- When the squeeze resolves (BBW expands AND ADX rises above 25), enter in the breakout direction.\n\n"

    "**When summary.squeeze_breakout is set (structured breakout signal):**\n"
    "- summary.squeeze_breakout = 'long' or 'short' fires only on the FRESH transition out of a 1h squeeze "
    "(BBW expanding ≥25% off the floor, ADX>20, price beyond BB band, RSI confirming).\n"
    "- Treat this as higher quality only after volume confirmation; request normal size and let code-enforced risk/volatility limits size it.\n"
    "- REQUIRED CONFIRMATION: volume on the breakout candle ≥ 1.5× the 20-candle average. "
    "If volume is weak, decline_trade — false breakouts ('head fakes') are common.\n"
    "- TP: use the nearest credible futures structural target that clears edgeReport.baseRr after costs; a runner may extend "
    "toward 2-3× stop distance only when structure and volume support it. "
    "Squeezes resolve in extended moves; don't cap the target at the minimum reward:risk — let it run to 2-3×.\n"
    "- Daily exhaustion normally blocks continuation. Narrow exception: an exhausted-BEARISH daily may still take a "
    "continuation SHORT after a rally/retest when 1h AND 15m are bearish and confidence clears the elevated trend-short "
    "bar; submit the complete bracket and let the code gate validate it. Exhausted-bullish continuation longs remain blocked.\n\n"

    "**When market_regime is 'trending':**\n"
    "- Use your standard trend-following strategy (STEP 3 below). The regime confirms your edge.\n"
    "- Do NOT mean-revert against a confirmed trend. Do NOT short rallies or buy dips against the trend.\n"
    "- A trending regime with ADX > 35 strengthens the thesis, but request normal risk size and let the code's "
    "live equity, volatility, concentration, and rolling-edge controls determine final size.\n\n"

    "**Regime transitions (breakout/breakdown):**\n"
    "- If the commentary says 'ADX rising' during a range, a breakout is forming. Review range-trade protection and "
    "ratchet it only to a valid structural level; prepare to flip only after the existing lifecycle is closed.\n"
    "- If the commentary says 'ADX weakening' during a trend, the trend is exhausting. "
    "Take profit on trend trades and prepare for range conditions.\n"
    "- Close range trades when ADX crosses 25. Close trend trades when ADX drops below 18.\n\n"
    if cfg.trading.range_trading_enabled else "") +

    "## STEP 3 — Trade decision:\n"
    "Use the research to decide direction and build a complete trade plan (entry, stop, TP) BEFORE placing the order.\n\n"

    "**Entry tool selection (MANDATORY for all new positions):**\n"
    "- New exposure is FUTURES-ONLY and must use place_futures_limit_order with an exchange-atomic TP/SL bracket. "
    "Spot order tools are management/close-only because the spot entry path cannot attach both exits atomically.\n"
    "- Reserve place_market_order / place_futures_market_order ONLY for closing positions or emergency exits.\n"
    "- Why: market orders execute at current price regardless of chart structure. Limit orders let price "
    "come TO you at a better level — avoiding the worst timing failures (shorting into a dump, buying into a pump).\n"
    "- FEE EDGE: limit (maker) fills are ~3x cheaper than market (taker) fills. Round-trip taker fees + slippage "
    "are ~0.14%, so a TP only ~0.1-0.3% away barely beats costs — those are the pennies-in-front-of-the-steamroller "
    "scalps that bled the account. Only take a setup whose TP clears total round-trip cost by a comfortable margin "
    "(aim for net edge >= 2-3x fees); otherwise skip it.\n\n"

    "**How to pick entry_price (REQUIRED for limit order tools):**\n"
    "Choose a level where MULTIPLE reference points converge (confluence of 2+ levels is far more reliable than any single one):\n"
    "  - SHORT entry — set entry_price ABOVE current price at a resistance zone:\n"
    "    * 15m/1h EMA (price bouncing into a declining EMA from below)\n"
    "    * Upper Bollinger Band (price stretched to +2 SD, RSI > 65)\n"
    "    * Recent swing high or VAH (Value Area High from volume profile)\n"
    "    * VWAP retest from below after a breakdown\n"
    "    * Fibonacci retracement of the recent drop: 38.2%, 50%, or 61.8% bounce levels\n"
    "  - LONG entry — set entry_price BELOW current price at a support zone:\n"
    "    * 15m/1h EMA pullback (price returning to a rising EMA from above)\n"
    "    * Lower Bollinger Band (price stretched to -2 SD, RSI < 35)\n"
    "    * Recent swing low or VAL (Value Area Low)\n"
    "    * VWAP retest from above after a breakout\n"
    "    * Fibonacci retracement of the recent rally: 38.2%, 50%, or 61.8% pullback levels\n"
    "- **Entry distance**: target levels within 0.5–1.5 × ATR of current price. Farther than 1.5 ATR rarely fills in normal conditions.\n"
    "- **Execution adaptation**: read performanceSummary.limitOrdersSubmitted and limitFillRate. After at least 5 recent "
    "limit submissions, if fill rate is below 25%, prefer the NEAREST still-valid structural confluence (roughly 0.2–0.75 "
    "ATR, never inside the code minimum) instead of repeatedly choosing an idealized deep level. This improves executable "
    "edge without permitting market chasing; TP/SL, daily/regime alignment, and RR must remain valid.\n"
    "- **Volume check**: the target level is stronger when it coincides with a prior high-volume node (POC or VAH/VAL from volume profile).\n"
    "- **Rejection confirmation**: for shorts, prefer entry at resistance only when you also see bearish RSI divergence or RSI > 65. "
    "For longs, prefer pullback entries when RSI < 40 or bullish divergence is present.\n"
    "- If current price is already at or through the optimal level, do not chase with a market order; wait for a "
    "valid pullback limit or decline the setup. Market tools remain close/emergency-only.\n"
    "- A fresh breakout impulse is not executable as a passive entry here. Trade only its first confirmed retest/pullback "
    "with an atomic bracket; do not simulate breakout execution by placing a wrong-side/marketable limit.\n"
    "- NEVER set entry_price at or worse than the current price for a new entry "
    "(do NOT short at current price when price just dumped — wait for the bounce to resistance).\n\n"

    "**Entry signal — trade when the 1h trend_bias is clear (bullish or bearish):**\n"
    "- Strong setup (request normal risk size): 1h AND 15m both agree on direction, RSI 40–65, MACD confirms; "
    "a catalyst is supporting evidence when one exists, not a reason to force a news lookup or a trade.\n"
    "- Normal setup (trade reduced size, 50–70% of normal): 1h is clear, 15m mixed, OR 1h mixed but strong news catalyst.\n"
    "- TREND-SHORT EXCEPTION (important): daily_bias_raw='bearish' plus daily_exhausted=true is NOT a blanket short veto. "
    "If both 1h and 15m are bearish, confidence is high, and a non-chasing rally/retest limit has valid RR, attempt the "
    "short; allow_trend_aligned_short enforces the final bar and regime sizing in code.\n"
    + (
    "- Range setup (when market_regime='ranging'): Use mean-reversion entries per STEP 2c above. "
    "Flat/ranging is now tradeable — oscillations within the range are your edge.\n"
    "- Skip (decline): RSI at extreme (>75 or <25) against direction, clearly negative news, "
    "or range too narrow for fees (< 0.3% half-range).\n\n"
    if cfg.trading.range_trading_enabled else
    "- Skip (decline): 1h AND 15m both flat/ranging with no catalyst, RSI at extreme (>75 or <25) against direction, "
    "or clearly negative news.\n\n"
    ) +

    "**Venue policy:**\n"
    "- Use futures for new exposure; use spot tools only to protect or close an existing spot holding.\n"
    "- Read all balances, but never transfer funds or increase leverage merely to force an otherwise unaffordable setup. "
    "If the safe futures size cannot clear exchange minimums and profit-after-cost gates, skip it.\n\n"

    "**Sizing:**\n"
    "- Do not plan new spot positions. For existing spot holdings, protection/close decisions must use live balance and structural levels.\n"
    f"- For futures: propose no more than maxPositionUsd={snapshot.max_position_usd}; code caps projected same-symbol "
    "notional (existing plus new legs) and stop-defined loss across the entire position lifecycle using the remaining "
    "equity-risk and concentration budgets. The safe code-sized amount can be much smaller than the requested amount.\n"
    "- ADD-ONS/PYRAMIDING: add only in the existing position's direction, only after the exchange confirms its stop "
    "at or beyond breakeven, and never loosen that stop. All legs share one lifecycle risk and concentration budget; "
    "never average down or repeatedly add to a losing position. If those conditions are not verified, do not add.\n"
    "- Keep at least 10% USDT reserve across all venues combined.\n\n"

    "**TP/SL rules (mandatory on every new entry):**\n"
    "- Stop-loss: ATR-based (1.5–2.5x ATR from entry) OR 1.0–2.0% if ATR is unavailable. Place the SL where "
    "your thesis is invalidated, then control dollar risk with SIZE — NOT by widening the stop past the target.\n"
    f"- Take-profit: {cfg.trading.min_futures_rr:.1f}R is the minimum POST-COST floor. Read edgeReport.baseRr; "
    "the code HARD-REJECTS an entry whose reward/risk after estimated fees and slippage is below the floor. The nearest realistic structural "
    "target and true invalidation must naturally clear it. If they do not, SKIP — never distort either level to force a fill.\n"
    "- For FUTURES limit entries: ALWAYS pass BOTH take_profit_price and stop_loss_price in the SAME "
    "place_futures_limit_order call. They are attached to the order (KuCoin bracket) and arm the instant "
    "it fills — so decide entry + TP + SL together, up front. Do NOT defer protection to a later run; a "
    "fill between runs would otherwise sit unprotected. (Spot limit entries still bracket on the next run.)\n"
    + (
    f"- FOR RANGE TRADES: TP toward the BB midline (mean), SL beyond the BB band. Still respect the structural "
    f"post-cost floor in edgeReport.baseRr (currently {cfg.trading.min_futures_rr:.1f}R) — if the mean-reversion target is too close to justify the stop, "
    "skip the trade. Never set a range trade TP beyond the opposite BB band.\n\n"
    if cfg.trading.range_trading_enabled else "\n") +

    "## STEP 4 — Log and save:\n"
    "- Log every decision (trade or decline) with confidence via log_decision.\n"
    "- Save/update the trade plan via save_trade_plan.\n"
    "- Log sentiment via log_sentiment.\n\n"

    "## Capital and risk limits:\n"
    f"- Do NOT request more than maxPositionUsd={snapshot.max_position_usd} USDT for a new order. This is only a ceiling, "
    "not permission: projected same-symbol exposure and whole-lifecycle stop risk can impose a lower limit.\n"
    f"- Max trades per symbol per day: {cfg.trading.max_trades_per_symbol_per_day}. If reached, decline new trades for that symbol.\n"
    f"- Futures entry leverage is HARD CAPPED at {cfg.trading.max_entry_leverage}x by the system. Do NOT request higher leverage to meet profit thresholds — if a trade doesn't work at {cfg.trading.max_entry_leverage}x, skip it.\n"
    f"- Only place a trade if your confidence >= {snapshot.min_confidence}.\n"
    "- Keep at least 10% of total USDT untouched as reserve.\n"
    f"- Minimum {cfg.trading.min_trade_interval_minutes:.0f} minutes between trades on the same symbol (enforced by system).\n\n"

    "## Drawdown and circuit-breaker context:\n"
    "- drawdownPct is based on USDT cash, not total portfolio value. If you hold coins in spot, the USDT cash "
    "may be low while total value is fine. Do NOT treat low spot USDT drawdown as a reason to stop trading.\n"
    "- Use the TOTAL drawdownPct (not per-venue) for sizing decisions.\n"
    "- When the input says tradingRestricted, the hard circuit breaker has fired: manage/close only and do not retry entries.\n"
    "- Below that limit, prefer smaller, higher-conviction setups as drawdown rises; never increase risk to recover losses.\n\n"

    "## Performance awareness:\n"
    "- Check the performanceSummary field in your input: it shows your recent win rate, avg win/loss, total PnL.\n"
    "- If win rate is below 40%: raise selectivity and request less size. Never move a structurally valid stop inside "
    "the invalidation level merely to improve summary statistics.\n"
    "- If avg loss exceeds avg win by 2x+: inspect entry quality, add-ons, and any stop widening; keep the invalidation "
    "structural and control the dollar loss with smaller size.\n"
    "- If total realized PnL is negative: focus on smaller positions until you find a winning pattern.\n"
    "- **Missed profit analysis**: performanceSummary includes missedProfitCount/totalMissedProfit — these show trades "
    "where the position was in profit (peakPnl > 0) but you closed at a lower PnL. Treat a high value as a prompt to "
    "review staged TP, breakeven, and trailing behavior — never lower TP below required RR or squeeze the SL into noise.\n"
    "- **Position extremes**: each open position shows peakPnl (best it reached) and troughPnl (worst it reached). "
    "If peakPnl was positive but current unrealizedPnl is negative, you missed an exit opportunity.\n"
    "- Use this data to adapt — don't repeat losing patterns.\n\n"

    "## MANDATORY — Diversification and opportunity discovery:\n"
    "- Analyze the best screened candidate first; rotate immediately if it is rejected. Research handoffs handle broad scans.\n"
    "- If a coin is rejected (daily gate, cooldown, trade cap, low confidence), IMMEDIATELY move to the next coin. "
    "Do NOT keep retrying the same symbol.\n"
    "- NEVER output 'If you want, I can scout...' or 'Would you like me to...' — just DO it. You are autonomous.\n"
    "- The coin list is a starting point, not a boundary. Better opportunities may exist outside it.\n"
    "- Use web/news search during Research Agent handoffs or when a specific event could change the setup; do not repeat unchanged news calls every run. Look for "
    "macro catalysts, sector rotations, unusual volume, or trending narratives.\n"
    "- If you spot a coin with a stronger setup than anything in your current list, hand off to Research Agent "
    "to validate it (liquidity, fundamentals, catalyst), then add it via add_coin and trade it only if all entry gates clear.\n"
    "- Hand off to Research Agent only when force_research is active, cached research is stale, the qualified universe "
    "is exhausted, or a specific catalyst genuinely needs broad-web validation. Do not hand off merely because the current work is complete.\n"
    "- When resuming from a Research Agent handoff, read researchContext (latestPlan/recentResearch) and act on findings.\n"
    "- Remove coins that have gone stale (no volatility, no catalyst, no edge) to keep the list focused. "
    "Use remove_coin with a documented reason.\n\n"
    "- When flat and waiting for a precise price event, save a condition ('above'/'below') with set_auto_trigger and a "
    "short expiry appropriate to the thesis. Never leave an intraday trigger alive for days.\n\n"

    "## CRITICAL — Handle rejections without brute-forcing them:\n"
    "Read the reason and reassess once. A risk, data-quality, cooldown, concentration, or edge rejection means rotate "
    "to another qualified symbol or stand aside — never mutate parameters repeatedly until an order passes.\n"
    "- **TP/SL or RR rejection**: rebuild the complete plan from the true invalidation and nearest reachable structural "
    "target. Do not nudge one bracket leg merely to clear the gate; skip if the natural plan still fails.\n"
    "- **Insufficient balance**: reduce requested futures notional within the same risk plan or skip. Never transfer funds or raise leverage just to make a rejected setup fit.\n"
    "- **Insufficient margin (futures)**: use transfer_funds('spot_to_futures') or sell a spot position first, "
    "or reduce notional. Do NOT increase leverage to compensate.\n"
    "- **Size too small / below minimum**: skip this coin if the safely code-sized order is below the exchange minimum. "
    "Never increase lifecycle risk just to reach a lot-size threshold.\n"
    "- **Daily trade cap**: move to another symbol.\n"
    "- **Profit too low**: improve the resting entry or use a realistic structural target; otherwise skip. Never increase "
    "size, leverage, or risk solely to pass a minimum-profit check.\n"
    "- **Daily gate rejection**: the 1D trend opposes your trade direction. Do NOT retry the same direction — trade WITH the daily trend or move to another symbol.\n"
    "- **Trade interval cooldown**: you traded this symbol too recently. Move to another symbol or wait.\n\n"
    "- **Pending atomic entry**: do not cancel/replace it before its deterministic lease expires merely because a later "
    "model run calls it stale. Early cancellation is allowed only after a completed 1h/daily direction flip objectively "
    "invalidates the thesis; code enforces this.\n\n"
    "**Multi-step execution is expected.** If you want to open a futures trade but funds are in spot:\n"
    "1. Sell an unneeded spot position (place_market_order with rationale 'portfolio review')\n"
    "2. Transfer the freed USDT to futures (transfer_funds)\n"
    "3. Place the futures order\n"
    "This is a normal workflow — do not skip steps 1-2 and accept rejection.\n\n"

    "## Circuit Breaker Awareness:\n"
    + (f"- **TRADING RESTRICTED**: {snapshot.restriction_reason}. You are in CLOSE-ONLY mode. "
       "Do NOT open new positions. You may only: close existing positions, adjust TP/SL, and manage risk. "
       "Focus on protecting capital until the restriction lifts.\n\n"
       if snapshot.trading_restricted else
       "- Trading is unrestricted. Circuit breakers are active and will block entries if daily drawdown "
       f"exceeds {cfg.circuit_breaker.max_daily_drawdown_pct}% or after {cfg.circuit_breaker.max_consecutive_losses} consecutive losses.\n\n") +

    "## Staged Take-Profit:\n"
    + ("- **Partial TP is ENABLED.** When you set position protection (set_spot_position_protection / set_futures_position_protection), "
       "the system automatically splits your TP into 2 tranches:\n"
       "  - Tranche 1 (60%): TP at 60% of the way to your target — locks in base profit early.\n"
       "  - Tranche 2 (40%): TP at your full target — captures the extended move.\n"
       "- Set your TP at the full intended target; the system handles the split.\n"
       "- For range trades, the midline TP is split 60/40 automatically.\n\n"
       if cfg.trading.partial_tp_enabled else "") +

    "## Post-Loss Cooldown:\n"
    + (f"- After a stop-loss on any symbol, a {int(cfg.trading.post_loss_cooldown_minutes)}-minute cooldown is enforced. "
       "You cannot re-enter the same symbol during cooldown. This prevents revenge trading. "
       "Move to a different symbol or wait.\n\n"
       if cfg.trading.post_loss_cooldown_minutes > 0 else "") +

    "## Narrative:\n"
    "- Be explicit: state what research showed, what trade you placed (or declined and why), what TP/SL you set, and your confidence.\n"
    f"- PAPER_TRADING={snapshot.paper_trading}. When true, simulate orders via the tool.\n"
  )

  if force_research:
    # The trading loop detected several consecutive runs with no executed trade. Force a
    # research handoff up front so the Research Agent refreshes the coin universe instead of
    # the Trading Agent declining yet again. Prepended so it is the first thing the model reads.
    instructions = (
      "## 🚨 STUCK STATE — RESEARCH HANDOFF REQUIRED THIS RUN (highest priority):\n"
      "You have completed several consecutive runs WITHOUT placing a single trade. The current coin "
      "universe is not producing actionable setups. BEFORE analyzing or declining anything else this run, "
      "you MUST hand off to the Research Agent and instruct it to OVERHAUL the coin list: remove stale, "
      "illiquid, or no-catalyst symbols and add fresh, liquid, high-opportunity coins with real catalysts. "
      "After the Research Agent hands control back to you, re-audit the refreshed universe and trade only if a candidate "
      "clears every entry and risk gate. If none qualifies, log that result and stand aside; research is mandatory here, "
      "but a trade is not.\n\n"
      + instructions
    )

  if temporary_notes:
    temp_text = "\n".join(f"- {n['content']}" for n in temporary_notes)
    instructions += (
      "\n## URGENT Supervisor Directives (one-time, highest priority):\n"
      "The bot owner sent the following instructions via the Supervisor Agent. "
      "These override any conflicting rules for THIS RUN ONLY. Follow them exactly:\n"
      + temp_text + "\n"
    )

  if permanent_notes:
    notes_text = "\n".join(f"- {n['content']}" for n in permanent_notes)
    instructions += (
      "\n## Supervisor Directives (permanent):\n"
      "The following instructions were set by the bot owner via the Supervisor Agent. "
      "Treat them as additional rules:\n" + notes_text + "\n"
    )

  # Secondary research agent to scout new coins while idle.
  research_agent = Agent(
    name="Research Agent",
    instructions=(
      "You are a proactive research scout for alternative crypto opportunities.\n"
      "- Mission: Find high-confidence setups beyond the current universe, plus catalysts the main agent may have missed.\n"
      "- Sources to prioritize when using web_search: CoinDesk/The Block/Cointelegraph for news; exchange blogs (Binance/Coinbase/OKX/Bybit) for listings/delistings/rule changes; X/Twitter lists (founders, analysts, journalists) for real-time signals; macro outlets (Bloomberg/Reuters/FT) when relevant; on-chain sentiment/flows when available.\n"
      "- Tasks: gather news, sentiment, liquidity, catalysts, and narrative strength; highlight listings/delistings, hacks, regulatory moves, funding rounds, and smart-money activity.\n"
      "- CITATION DISCIPLINE: record the actual source URL and verify it concerns the same asset. Treat uncited, stale, "
      "or cross-asset articles as neutral; never use generic BTC news as evidence for an unrelated altcoin.\n"
      "- Output: concise recommendations with evidence (why this beats current options), liquidity check, and confidence. Prefer liquid, tradable pairs; avoid illiquid/obscure tokens.\n"
      "- When idle, discover high-quality sources; add via add_source(name, url, reason) and remove low-value ones via remove_source(name, reason).\n\n"
      "## COIN-LIST CURATION (your core job — act, don't just suggest):\n"
      "- FIRST call list_coins. Its quarantined array is an adaptive, expiring blocklist derived from failed volatility/data "
      "checks. Exclude those symbols from deep validation and add_coin attempts until retryAfter; do not spend tools proving "
      "the same invalid candidate again.\n"
      "- THEN scan the whole market every run: call scan_futures_market (the ONLY tool that sees all ~500 "
      "perps, not just named ones) to find what is actually moving. The current 2-3 coins are almost never the "
      "only opportunity — even in a bad tape, some liquid coin is trending. Scan 'momentum' and, in a bearish "
      "BTC daily, 'losers'/'short'; then deep-validate the top few with analyze_market_context + fetch_orderbook.\n"
      "- Do NOT stay anchored to BTC/ETH/SOL out of habit. If the scan surfaces a more liquid, cleaner setup "
      "(strong trend, catalyst, good structure) that clears the bars, ADD it for the Trading Agent to evaluate.\n"
      "- You OWN the active coin list. It is a watchlist, not a claim that every name is immediately enterable. Maintain "
      "roughly 4-6 liquid, mature, structurally distinct candidates when the scan provides them, so one temporary timeframe "
      "conflict cannot deadlock execution:\n"
      "  - add_coin(symbol, reason) for LIQUID, ESTABLISHED pairs with a real catalyst or strong technical setup. Verify liquidity with analyze_market_context/fetch_orderbook before adding.\n"
      "  - remove_coin(symbol, reason, exit_plan) for structural problems: lost liquidity, invalid data, repeated losses, or a stale multi-day thesis. Do not remove a sound liquid candidate solely for a temporary 15m/1h disagreement.\n"
      "- HARD QUALITY BARS for any coin you add (they mirror code-level gates — adding a coin that violates them just wastes the Trading Agent's run):\n"
      "  * LIQUIDITY: deep book + meaningful 24h volume. Reject thin micro-caps.\n"
      "  * MATURITY: do NOT add freshly-listed perpetuals (< ~7 days old). New listings are thin and swing 50-100% intraday — exactly how the RE-USDT position blew up.\n"
      "  * CORRELATION: when BTC's daily trend is bearish, do NOT add altcoins as LONG candidates (alts are high-beta to BTC; longing them into a BTC downtrend loses). Surface only SHORT candidates or majors in a bearish BTC regime.\n"
      "- NEVER add a coin just to force a trade. A diversified watchlist may contain names waiting for confirmation; the Trading Agent still stands aside until one clears every gate.\n"
      "- When you are handed off because the Trading Agent is STUCK (declining run after run), prune the dead weight and add new opportunities ONLY if they clear every bar above; if nothing qualifies, say so plainly in your research note and hand back.\n"
      "- You must still NEVER place orders or set TP/SL — that is the Trading Agent's job. You only curate the list and log research.\n\n"
      "- Log findings via log_research (topic, summary, actions) so the main agent can decide.\n"
      "- **Strategy review**: Use latest_items('decisions') to review recent trades. Look for patterns:\n"
      "  - Trades with high peakPnl but low/negative final pnl = review staged TP, breakeven, and trailing behavior; "
      "do not prescribe a mechanically tighter bracket.\n"
      "  - Trades with deep troughPnl = review entry quality, add-ons, and risk sizing; do not blindly tighten a structural stop.\n"
      "  - Repeated losses on the same coin = bad coin choice (remove it via remove_coin).\n"
      "  - Log strategy improvement suggestions via log_research so the Trading Agent can adapt.\n\n"
      "## MANDATORY FINAL STEP — always hand back:\n"
      "- You MUST finish EVERY turn by handing control back to the Trading Agent. Never end your turn without handing off.\n"
      "- Before handing off, log a research note (log_research) summarizing what you found and exactly which coins you "
      "added/removed, and remind the Trading Agent to read it via latest_plan/latest_items and evaluate the refreshed universe.\n"
    ),
    tools=[
      WebSearchTool(search_context_size="high"),
      fetch_recent_candles,
      analyze_market_context,
      fetch_orderbook,
      fetch_funding_rate,
      fetch_open_interest,
      fetch_futures_orderbook,
      fetch_futures_mark_price,
      fetch_futures_candles,
      fetch_contract_details,
      scan_futures_market,
      fetch_kucoin_news,
      fetch_coindesk_news,
      log_research,
      latest_items,
      list_coins,
      add_coin,
      remove_coin,
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
      fetch_recent_candles,
      fetch_orderbook,
      analyze_market_context,
      plan_spot_position,
      transfer_funds,
      fetch_account_state,
      refresh_fee_rates,
      place_limit_order,
      place_futures_limit_order,
      place_futures_market_order,
      place_market_order,
      place_spot_stop_order,
      set_spot_position_protection,
      cancel_spot_stop_order,
      cancel_spot_limit_order,
      list_spot_stop_orders,
      set_futures_position_protection,
      cancel_futures_order,
      list_futures_stop_orders,
      list_futures_positions,
      get_recent_fills,
      get_closed_positions,
      fetch_funding_rate,
      fetch_open_interest,
      fetch_futures_orderbook,
      fetch_futures_mark_price,
      fetch_futures_candles,
      fetch_contract_details,
      scan_futures_market,
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
      fetch_coindesk_news,
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
  user_state_obj["performanceSummary"] = memory.performance_summary()
  # Adaptive edge posture — the CODE-ENFORCED risk stance derived from rolling realized results.
  # Surfaced so the agent proposes trades that will pass the gates instead of burning turns on
  # rejections, and so it explains the posture in its narrative.
  try:
    _edge_now = _edge_state()
    _edge_stats_now = _edge_now.get("stats") or {}
    _per_sym = _edge_stats_now.get("per_symbol", {})
    # Realized reward:risk actually achieved vs the RR intended at entry. A gap is diagnostic of
    # entry/management quality; it is not permission to compress a structural target.
    _aw = float(_edge_stats_now.get("avg_win") or 0.0)
    _al = abs(float(_edge_stats_now.get("avg_loss") or 0.0))
    _realized_rr = round(_aw / _al, 2) if _al > 0 else None
    user_state_obj["edgeReport"] = {
      "rolling": {k: _edge_stats_now.get(k) for k in ("n", "wins", "losses", "win_rate", "avg_win", "avg_loss", "payoff", "profit_factor", "net", "expectancy", "loss_streak")},
      "perSymbol": _per_sym,
      "perDirection": _edge_stats_now.get("per_direction", {}),
      "baseRr": cfg.trading.min_futures_rr,
      "realizedRewardRisk": _realized_rr,
      "entrySizeFactor": _edge_now.get("size_factor"),
      "entrySizeFactorByDirection": _edge_now.get("direction_size_factor", {}),
      "entrySizeFactorBySymbol": _edge_now.get("symbol_size_factor", {}),
      "benchedSymbols": {s: f"{max(0, u - int(time.time())) / 3600:.1f}h left" for s, u in (_edge_now.get("bench") or {}).items()},
      "note": (
        "ENFORCED IN CODE this run: the structural net reward:risk floor is checked after "
        "estimated fees and slippage, benched symbols are rejected, and risk is scaled by loss streak plus "
        "long/short and symbol expectancy factors. "
        "The RR floor stays structural rather than moving targets farther away after losses. The bench and risk "
        "scaling are SYMBOL-SPECIFIC — a symbol you keep losing on gets smaller and rests longer. So DIVERSIFY: "
        "rotate off losing/benched symbols toward the liquid movers scan_futures_market surfaces. "
        "REALITY CHECK: realizedRewardRisk is what your closed trades ACTUALLY achieved (avg win / avg loss), but "
        "a gap from planned RR is diagnostic rather than an instruction to compress every bracket. Select entries where "
        "the nearest reachable structural target (VWAP, POC, prior swing, band mid) and the true invalidation naturally "
        "clear edgeReport.baseRr after costs. Otherwise skip; never lower TP or move SL inside invalidation solely to pass the floor."
      ),
    }
  except Exception as _edge_exc:
    logger.debug("Edge report unavailable: %s", _edge_exc)

  if recent_fills:
    events: Dict[str, Any] = {}
    if recent_fills.get("spot_fills"):
      events["spotFills"] = recent_fills["spot_fills"]
    if recent_fills.get("futures_fills"):
      events["futuresFills"] = recent_fills["futures_fills"]
    if recent_fills.get("closed_positions"):
      events["closedPositions"] = recent_fills["closed_positions"]
    if recent_fills.get("auto_triggers"):
      events["autoTriggers"] = recent_fills["auto_triggers"]
    if events:
      user_state_obj["recentEvents"] = events
      user_state_obj["recentEventsNote"] = (
        "NEW since last round: an order/position lifecycle event or an explicit price trigger fired. "
        "Re-check live structure before acting; a trigger invites evaluation and never bypasses risk gates."
      )

  # Detect stale/ineffective stop orders
  stale_stops = []
  for order in (snapshot.spot_stop_orders or []):
    if not isinstance(order, dict):
      continue
    order_symbol = _normalize_symbol(str(order.get("symbol") or ""))
    stop_px = _to_float(order.get("stopPrice"))
    if not order_symbol or stop_px <= 0:
      continue
    cur_price = current_prices.get(order_symbol)
    if not cur_price or cur_price <= 0:
      continue
    side = str(order.get("side") or "").lower()
    distance_pct = abs(stop_px - cur_price) / cur_price * 100
    is_stale = False
    reason = ""
    if side == "sell":
      if stop_px > cur_price and distance_pct > 10:
        is_stale = True
        reason = f"Stop at {stop_px:.4f} is {distance_pct:.1f}% ABOVE current price {cur_price:.4f} — will never trigger as SL"
      elif stop_px < cur_price and distance_pct > 50:
        is_stale = True
        reason = f"Stop at {stop_px:.4f} is {distance_pct:.1f}% below current price — extremely loose"
    if is_stale:
      stale_stops.append({
        "orderId": order.get("id") or order.get("orderId"),
        "symbol": order_symbol,
        "side": side,
        "stopPrice": stop_px,
        "currentPrice": cur_price,
        "distancePct": round(distance_pct, 1),
        "issue": reason,
      })
  if stale_stops:
    user_state_obj["staleStops"] = stale_stops

  # Flag holdings with unknown entry price for portfolio review
  unlisted_holdings = []
  for sym, pos in positions.items():
    if pos.get("avgEntry") is None and _to_float(pos.get("netSize")) > 0:
      cur_price = pos.get("currentPrice") or current_prices.get(sym)
      net_size = _to_float(pos.get("netSize"))
      value_usd = net_size * cur_price if cur_price else None
      unlisted_holdings.append({
        "symbol": sym,
        "netSize": net_size,
        "currentPrice": cur_price,
        "estimatedValueUsd": round(value_usd, 2) if value_usd else None,
        "note": "Entry price unknown (externally acquired). Evaluate hold vs close.",
      })
  if unlisted_holdings:
    user_state_obj["unlistedHoldings"] = unlisted_holdings

  research_context: Dict[str, Any] = {}
  if latest_plan_entry:
    research_context["latestPlan"] = latest_plan_entry
  if research_notes:
    research_context["recentResearch"] = research_notes
  if research_context:
    user_state_obj["researchContext"] = research_context
  input_payload = json.dumps(user_state_obj)

  # Ensure a fresh trace per agent loop using the official processor setup and a unique trace_id.
  result = asyncio.run(
    _run_agent_with_tracing(
      trading_agent,
      input_payload,
      cfg,
      langsmith_ctx,
      run_name,
      unique_trace_id,
    )
  )
  narrative = _strip_citation_tokens(str(result.final_output))

  # Log token usage and prompt cache stats
  total_input = sum(r.usage.input_tokens for r in result.raw_responses if r.usage)
  total_output = sum(r.usage.output_tokens for r in result.raw_responses if r.usage)
  total_cached = sum(
    (r.usage.input_tokens_details.cached_tokens or 0)
    for r in result.raw_responses
    if r.usage and r.usage.input_tokens_details
  )
  cache_pct = (total_cached / total_input * 100) if total_input > 0 else 0
  logger.info(
    "Agent run tokens: input=%d output=%d cached=%d (%.0f%%) requests=%d",
    total_input, total_output, total_cached, cache_pct, len(result.raw_responses),
  )

  artifacts = _collect_run_artifacts(result.new_items)
  tool_outputs: List[Any] = artifacts["tool_outputs"]
  handoff_events: List[Dict[str, str]] = artifacts["handoffs"]
  research_activity: List[str] = artifacts["research"]
  agents_used: set[str] = artifacts["agents_used"]
  run_still_authorized = True
  if safety_state is not None:
    run_still_authorized = bool(
      safety_state.check(
        entry_token,
        max_age_sec=max(180.0, float(cfg.trading.poll_interval_sec) * 3.0),
      ).get("allowed")
    )

  # Record handoffs in memory so they surface in the dashboard decision feed (marked as
  # handoff_* actions). pnl stays None so they never count toward win/loss stats.
  for h in handoff_events if run_still_authorized else []:
    direction = "research" if h.get("to") == "Research Agent" else "trading"
    try:
      memory.log_decision(
        "ALL", f"handoff_to_{direction}", 0.0,
        f"{h.get('from')} → {h.get('to')}", paper=snapshot.paper_trading,
      )
    except Exception as exc:
      logger.debug("Failed to log handoff decision: %s", exc)

  decisions: list[str] = []

  def _summarize(output: Any) -> str | None:
    if not isinstance(output, dict):
      return None
    if output.get("skipped"):
      return f"decline: {output.get('reason','unspecified')} (conf={output.get('confidence')})"
    if output.get("cancelled"):
      return f"cancelled order: {output.get('orderId') or output.get('symbol') or 'unknown'}"
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
      sym = output.get("symbol") or output.get("requested") or output.get("futuresSymbol")
      return f"rejected: {output.get('reason','unspecified')}" + (f" [{sym}]" if sym else "")
    if output.get("error"):
      sym = output.get("symbol") or output.get("requested") or output.get("futuresSymbol")
      return f"error: {output.get('error')}" + (f" [{sym}]" if sym else "")
    return None

  for out in tool_outputs:
    summary = _summarize(out)
    if summary:
      decisions.append(summary)

  if handoff_events:
    logger.info("Agent handoffs this run: %s", " | ".join(f"{h['from']}→{h['to']}" for h in handoff_events))
  if research_activity:
    logger.info("Research Agent activity: %s", " | ".join(research_activity))

  return {
    "narrative": narrative,
    "tool_results": tool_outputs,
    "decisions": decisions,
    "handoffs": handoff_events,
    "research": research_activity,
    "agentsUsed": sorted(agents_used),
  }
