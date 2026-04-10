from __future__ import annotations

import html
import logging
import queue
import threading
from typing import Any, Dict, List, Optional

import requests

from .config import AppConfig

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_LENGTH = 4096
SEND_TIMEOUT_SEC = 10


class TelegramNotifier:
  """Fire-and-forget Telegram notification sender with background worker thread."""

  def __init__(self, cfg: AppConfig) -> None:
    self.enabled = cfg.telegram.enabled
    self.token = cfg.telegram.bot_token
    self.chat_id = cfg.telegram.chat_id
    self.silent = cfg.telegram.silent
    self._queue: queue.Queue[str] = queue.Queue(maxsize=100)
    self._thread: Optional[threading.Thread] = None

    if self.enabled:
      if not self.token or not self.chat_id:
        logger.warning("Telegram enabled but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing; disabling.")
        self.enabled = False
      else:
        self._start_worker()
        logger.info("Telegram notifier started (chat_id=%s).", self.chat_id)

  def _start_worker(self) -> None:
    self._thread = threading.Thread(target=self._worker, daemon=True, name="telegram-notifier")
    self._thread.start()

  def _worker(self) -> None:
    while True:
      try:
        msg = self._queue.get(timeout=5)
        self._send_raw(msg)
      except queue.Empty:
        continue
      except Exception as exc:
        logger.warning("Telegram worker error: %s", exc)

  def _send_raw(self, text: str) -> None:
    suffix = "\n\n<i>...truncated</i>"
    if len(text) > MAX_MESSAGE_LENGTH:
      text = text[: MAX_MESSAGE_LENGTH - len(suffix)] + suffix
    url = TELEGRAM_API_URL.format(token=self.token)
    payload = {
      "chat_id": self.chat_id,
      "text": text,
      "parse_mode": "HTML",
      "disable_notification": self.silent,
    }
    try:
      resp = requests.post(url, json=payload, timeout=SEND_TIMEOUT_SEC)
      if resp.status_code != 200:
        logger.warning("Telegram API error %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
      logger.warning("Telegram send failed: %s", exc)

  def send(self, text: str) -> None:
    """Enqueue a message for async sending. Drops if queue is full."""
    if not self.enabled:
      return
    try:
      self._queue.put_nowait(text)
    except queue.Full:
      logger.warning("Telegram queue full; dropping message.")

  # ---- Typed notification methods ----

  def notify_startup(self, cfg: AppConfig) -> None:
    mode = "PAPER" if cfg.trading.paper_trading else "LIVE"
    coins = ", ".join(cfg.trading.coins) if cfg.trading.coins else "none"
    futures = "enabled" if cfg.kucoin_futures.enabled else "disabled"
    lines = [
      "<b>trAIde Bot Started</b>",
      f"<b>Mode:</b> {mode}",
      f"<b>Coins:</b> {_esc(coins)}",
      f"<b>Max Position:</b> ${cfg.trading.max_position_usd:.0f}",
      f"<b>Max Leverage:</b> {cfg.trading.max_leverage}x",
      f"<b>Futures:</b> {futures}",
    ]
    self.send("\n".join(lines))

  def notify_agent_run(self, triggers: List[str], result: Dict[str, Any]) -> None:
    """Send summary of an agent run including decisions and order details."""
    decisions: List[str] = result.get("decisions") or []
    tool_results: List[Any] = result.get("tool_results") or []
    narrative: str = result.get("narrative") or ""

    parts: List[str] = ["<b>trAIde Agent Run</b>"]

    # Triggers
    trigger_str = ", ".join(triggers) if triggers else "idle_threshold"
    parts.append(f"<b>Triggers:</b> {_esc(trigger_str)}")

    # Order details from tool_results
    order_msgs = _format_orders(tool_results)
    if order_msgs:
      parts.append("")
      parts.extend(order_msgs)

    # Decisions summary
    if decisions:
      parts.append("")
      parts.append("<b>Decisions:</b>")
      for d in decisions:
        parts.append(f" - {_esc(d)}")

    # Truncated narrative
    if narrative:
      parts.append("")
      parts.append("<b>Narrative:</b>")
      excerpt = narrative[:500]
      if len(narrative) > 500:
        excerpt += "..."
      parts.append(f"<i>{_esc(excerpt)}</i>")

    self.send("\n".join(parts))

  def notify_error(self, error: str, context: str = "") -> None:
    lines = ["<b>trAIde Error</b>"]
    if context:
      lines.append(f"<b>Context:</b> {_esc(context)}")
    lines.append(f"<pre>{_esc(error[:1000])}</pre>")
    self.send("\n".join(lines))


def _esc(text: str) -> str:
  return html.escape(str(text))


def _fmt_price(val: Any) -> str:
  try:
    f = float(val)
    if f >= 1000:
      return f"${f:,.2f}"
    if f >= 1:
      return f"${f:.4f}"
    return f"${f:.6f}"
  except (TypeError, ValueError):
    return str(val)


def _format_orders(tool_results: List[Any]) -> List[str]:
  """Extract and format order notifications from tool_results."""
  msgs: List[str] = []
  for out in tool_results:
    if not isinstance(out, dict):
      continue

    # Live order
    if out.get("orderId") or (out.get("orderRequest") and not out.get("paper") and not out.get("skipped") and not out.get("rejected")):
      msgs.append(_format_single_order(out, paper=False))
      continue

    # Paper order
    if out.get("paper") and out.get("orderRequest"):
      msgs.append(_format_single_order(out, paper=True))
      continue

    # Transfer
    if out.get("transfer"):
      t = out.get("transfer", {})
      msgs.append(
        f"<b>Transfer</b>: {_esc(str(out.get('amount')))} {_esc(str(out.get('currency')))} "
        f"{_esc(str(out.get('direction')))} (id={_esc(str(t.get('orderId') or t.get('applyId')))})"
      )

  return msgs


def _format_single_order(out: Dict[str, Any], paper: bool) -> str:
  req = out.get("orderRequest") or {}
  side = (out.get("side") or req.get("side") or "").upper()
  symbol = out.get("symbol") or req.get("symbol") or ""
  funds = req.get("funds") or ""
  size = req.get("size") or ""
  rationale = out.get("rationale") or ""
  if not rationale:
    dl = out.get("decisionLog") or {}
    rationale = dl.get("reason") or ""
  rr = out.get("rr")

  emoji = "BUY" if side == "BUY" else "SELL" if side == "SELL" else side
  mode = "PAPER" if paper else "LIVE"

  lines = [f"<b>Order {emoji} ({mode})</b>"]
  lines.append(f"<b>Symbol:</b> {_esc(symbol)}")
  lines.append(f"<b>Side:</b> {side}")
  if funds:
    lines.append(f"<b>Funds:</b> ${_esc(str(funds))}")
  if size:
    lines.append(f"<b>Size:</b> {_esc(str(size))}")

  # Price from trade record
  tr = out.get("tradeRecord") or {}
  price = tr.get("price") or req.get("price")
  if price:
    lines.append(f"<b>Price:</b> {_fmt_price(price)}")

  # Bracket (TP/SL)
  bracket = out.get("bracket") or {}
  stop_info = bracket.get("stop") or {}
  tp_info = bracket.get("takeProfit") or {}
  stop_px = stop_info.get("stopPrice") or stop_info.get("stop_price")
  tp_px = tp_info.get("stopPrice") or tp_info.get("stop_price")
  if stop_px:
    lines.append(f"<b>Stop Loss:</b> {_fmt_price(stop_px)}")
  if tp_px:
    lines.append(f"<b>Take Profit:</b> {_fmt_price(tp_px)}")
  if rr is not None:
    try:
      lines.append(f"<b>RR:</b> {float(rr):.2f}")
    except (TypeError, ValueError):
      pass

  # PnL for sells
  if out.get("expectedPnl") is not None:
    try:
      pnl = float(out["expectedPnl"])
      roi = float(out.get("expectedRoi") or 0) * 100
      lines.append(f"<b>Expected PnL:</b> ${pnl:.4f} ({roi:.2f}%)")
    except (TypeError, ValueError):
      pass

  order_id = out.get("orderId")
  if order_id:
    lines.append(f"<b>Order ID:</b> <code>{_esc(str(order_id))}</code>")

  if rationale:
    lines.append(f"<b>Rationale:</b> {_esc(rationale[:200])}")

  return "\n".join(lines)
