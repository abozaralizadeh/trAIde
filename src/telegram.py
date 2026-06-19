from __future__ import annotations

import html
import logging
import queue
import re
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
    url = TELEGRAM_API_URL.format(token=self.token)
    for chunk in _split_message(text):
      payload = {
        "chat_id": self.chat_id,
        "text": chunk,
        "parse_mode": "HTML",
        "disable_notification": self.silent,
      }
      try:
        resp = requests.post(url, json=payload, timeout=SEND_TIMEOUT_SEC)
        # If HTML parsing fails (a stray tag/entity), resend as plain text so the
        # message is still delivered rather than silently dropped.
        if resp.status_code == 400 and "parse" in resp.text.lower():
          logger.warning("Telegram HTML parse failed; resending as plain text: %s", resp.text[:200])
          plain = {"chat_id": self.chat_id, "text": _strip_tags(chunk), "disable_notification": self.silent}
          resp = requests.post(url, json=plain, timeout=SEND_TIMEOUT_SEC)
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
    """Send summary of an agent run including decisions, handoffs, research, and order details."""
    decisions: List[str] = result.get("decisions") or []
    tool_results: List[Any] = result.get("tool_results") or []
    narrative: str = result.get("narrative") or ""
    handoffs: List[Any] = result.get("handoffs") or []
    research: List[Any] = result.get("research") or []
    agents_used: List[Any] = result.get("agentsUsed") or []

    parts: List[str] = ["<b>trAIde Agent Run</b>"]

    # Triggers
    trigger_str = ", ".join(triggers) if triggers else "idle_threshold"
    parts.append(f"<b>Triggers:</b> {_esc(trigger_str)}")
    if len(agents_used) > 1 or any(a not in ("", "Trading Agent") for a in agents_used):
      parts.append(f"<b>Agents:</b> {_esc(', '.join(str(a) for a in agents_used))}")

    # Order details from tool_results
    order_msgs = _format_orders(tool_results)
    if order_msgs:
      parts.append("")
      parts.extend(order_msgs)

    # Handoffs between Trading and Research agents
    if handoffs:
      parts.append("")
      parts.append("<b>Handoffs:</b>")
      for h in handoffs:
        if isinstance(h, dict):
          parts.append(f" \U0001F501 {_esc(h.get('from') or '?')} → {_esc(h.get('to') or '?')}")
        else:
          parts.append(f" \U0001F501 {_esc(str(h))}")

    # Research Agent activity (notes, coin-list changes, sentiment, sources)
    if research:
      parts.append("")
      parts.append("<b>\U0001F52C Research Agent:</b>")
      for r in research:
        parts.append(f" - {_esc(str(r))}")

    # Decisions summary
    if decisions:
      parts.append("")
      parts.append("<b>Decisions:</b>")
      for d in decisions:
        parts.append(f" - {_esc(d)}")

    if narrative:
      parts.append("")
      parts.append("<b>Narrative:</b>")
      # The narrative is the LLM's markdown; convert it so bold/headings/code render
      # instead of showing literal ** and ## characters.
      parts.append(markdown_to_telegram_html(narrative))

    self.send("\n".join(parts))

  def notify_error(self, error: str, context: str = "") -> None:
    lines = ["<b>trAIde Error</b>"]
    if context:
      lines.append(f"<b>Context:</b> {_esc(context)}")
    lines.append(f"<pre>{_esc(error[:1000])}</pre>")
    self.send("\n".join(lines))


def _split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> List[str]:
  """Split a long message into chunks that each fit within *limit* characters.

  Splits at newline boundaries when possible; falls back to hard split.
  """
  if len(text) <= limit:
    return [text]
  chunks: List[str] = []
  while text:
    if len(text) <= limit:
      chunks.append(text)
      break
    cut = text.rfind("\n", 0, limit)
    if cut <= 0:
      cut = limit
    chunks.append(text[:cut])
    text = text[cut:].lstrip("\n")
  return chunks


def _esc(text: str) -> str:
  return html.escape(str(text))


def _strip_tags(text: str) -> str:
  """Remove HTML tags and unescape entities for the plain-text send fallback."""
  no_tags = re.sub(r"<[^>]+>", "", text)
  return html.unescape(no_tags)


# --- Markdown -> Telegram HTML -------------------------------------------------
# The agents emit GitHub-flavored markdown (**bold**, ## headings, `code`, - bullets).
# Telegram's HTML parse_mode does NOT understand markdown, so those symbols render
# literally ("**foo**" instead of bold). This converts the common markdown subset to
# the small set of tags Telegram supports (<b>, <i>, <s>, <code>, <a>). All formatting
# is kept LINE-LOCAL — no tag ever spans a newline — so _split_message can never cut a
# message in the middle of a tag (which would trigger "can't parse entities").
_RE_CODE_FENCE = re.compile(r"```[ \t]*[\w+-]*\n?(.*?)```", re.DOTALL)
_RE_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_RE_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_RE_BOLD = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*")
_RE_BOLD_US = re.compile(r"(?<!\w)__(?!\s)(.+?)(?<!\s)__(?!\w)")
_RE_STRIKE = re.compile(r"~~(?!\s)(.+?)(?<!\s)~~")
_RE_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\*\w])")
_RE_ITALIC_US = re.compile(r"(?<![_\w])_(?!\s)([^_\n]+?)(?<!\s)_(?![_\w])")
_RE_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")
_RE_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_RE_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")


def markdown_to_telegram_html(text: str) -> str:
  """Translate a markdown string into Telegram-safe HTML (parse_mode=HTML)."""
  if not text:
    return ""
  text = str(text).replace("\r\n", "\n").replace("\r", "\n")
  # 1) Escape the three chars Telegram treats specially. Do this FIRST so any markup we
  #    add below is preserved verbatim. We deliberately do not escape quotes.
  text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

  # 2) Stash code spans/blocks so their contents are never touched by emphasis rules.
  stash: List[str] = []

  def _stash(fragment: str) -> str:
    stash.append(fragment)
    return f"\x00{len(stash) - 1}\x00"

  def _fence(m: "re.Match[str]") -> str:
    # Render each line of a fenced block as its own <code> so nothing spans a newline.
    body = m.group(1).strip("\n")
    return "\n".join(_stash(f"<code>{line}</code>") for line in (body.split("\n") or [""]))

  text = _RE_CODE_FENCE.sub(_fence, text)
  text = _RE_INLINE_CODE.sub(lambda m: _stash(f"<code>{m.group(1)}</code>"), text)

  # 3) Links before emphasis so URL punctuation (_ * ~) isn't mangled.
  text = _RE_LINK.sub(lambda m: _stash(f'<a href="{m.group(2).replace(chr(34), "%22")}">{m.group(1)}</a>'), text)

  # 4) Inline emphasis (bold before italic so '**' is consumed first).
  text = _RE_BOLD.sub(r"<b>\1</b>", text)
  text = _RE_BOLD_US.sub(r"<b>\1</b>", text)
  text = _RE_STRIKE.sub(r"<s>\1</s>", text)
  text = _RE_ITALIC.sub(r"<i>\1</i>", text)
  text = _RE_ITALIC_US.sub(r"<i>\1</i>", text)

  # 5) Block level, line by line: headings -> bold line, bullets -> "• ".
  out_lines: List[str] = []
  for line in text.split("\n"):
    heading = _RE_HEADING.match(line)
    if heading:
      out_lines.append(f"<b>{heading.group(1)}</b>")
      continue
    bullet = _RE_BULLET.match(line)
    if bullet:
      out_lines.append(f"{bullet.group(1)}• {bullet.group(2)}")
      continue
    out_lines.append(line)
  text = "\n".join(out_lines)

  # 6) Restore protected code fragments.
  return _RE_PLACEHOLDER.sub(lambda m: stash[int(m.group(1))], text)


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
