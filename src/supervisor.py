from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from agents import Agent, OpenAIResponsesModel, Runner
from agents.tool import WebSearchTool, function_tool
from openai import AsyncAzureOpenAI

from .config import AppConfig
from .conversation_memory import ConversationMemory
from .kucoin import KucoinClient, KucoinFuturesClient
from .memory import MemoryStore

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

  # --- Tools ---

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

  @function_tool
  async def get_account_snapshot() -> Dict[str, Any]:
    """Fetch live KuCoin account balances and futures overview."""
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
    "- When writing notes for the trading agent, be specific and actionable.\n"
    "- For temporary notes: phrase as a one-time instruction (e.g., 'On this run, reduce all position sizes by 50%').\n"
    "- For permanent notes: phrase as an ongoing rule (e.g., 'Always check BTC dominance chart before trading altcoins').\n"
    "- You cannot place trades directly. Use notes to influence the trading agent's behavior.\n"
    "- If asked to analyze something, use the available tools to gather data first, then provide your analysis.\n"
    "- You have conversation memory. Use prior context when relevant, but always fetch fresh data via tools.\n"
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
      read_logs,
      search_logs,
      read_memory,
      get_performance_summary,
      get_positions,
      get_recent_decisions,
      get_recent_trades,
      read_source_file,
      list_source_files,
      get_config_summary,
      get_account_snapshot,
      write_temporary_note,
      write_permanent_note,
      list_notes,
      delete_permanent_note,
    ],
    model=model,
  )

  result = asyncio.run(Runner.run(supervisor_agent, message_text, max_turns=30))
  response = str(result.final_output)

  conv_memory.add_exchange(message_text, response)

  if conv_memory.needs_compaction():
    _compact_conversation(conv_memory, openai_client, cfg.azure.deployment)

  return response


def _compact_conversation(
  conv_memory: ConversationMemory,
  openai_client: AsyncAzureOpenAI,
  deployment: str,
) -> None:
  """Summarize older messages using the LLM, keeping only recent exchanges."""

  def summarizer(existing_summary: str, old_messages: list) -> str:
    formatted = "\n".join(
      f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:800]}"
      for m in old_messages
    )
    prompt = (
      "Summarize this conversation between a crypto trading bot owner and a supervisor assistant. "
      "Capture: key questions asked, important answers/findings, any instructions or directives given, "
      "and ongoing topics of interest. Be concise (max 300 words).\n\n"
    )
    if existing_summary:
      prompt += f"Existing summary to incorporate:\n{existing_summary}\n\n"
    prompt += f"New messages to summarize:\n{formatted}"

    async def _call() -> str:
      resp = await openai_client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
      )
      return resp.choices[0].message.content or ""

    return asyncio.run(_call())

  try:
    conv_memory.compact(summarizer)
  except Exception as exc:
    logger.warning("Supervisor conversation compaction failed: %s", exc)
