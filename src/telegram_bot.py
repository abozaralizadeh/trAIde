from __future__ import annotations

import logging
import time
from typing import Callable

import requests

from .config import AppConfig
from .supervisor import run_supervisor_agent, _build_openai_client

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"
POLL_TIMEOUT = 30
MAX_MESSAGE_LENGTH = 4096


class TelegramBotPoller:
  """Long-poll Telegram getUpdates and dispatch messages to a handler."""

  def __init__(self, token: str, chat_id: str, message_handler: Callable[[str], str]) -> None:
    self._token = token
    self._chat_id = chat_id
    self._handler = message_handler
    self._offset = 0
    self._running = False
    self._base = TELEGRAM_API.format(token=token)

  def start(self) -> None:
    self._running = True
    logger.info("Telegram bot poller started (chat_id=%s).", self._chat_id)
    while self._running:
      updates = self._get_updates()
      for update in updates:
        self._process_update(update)

  def stop(self) -> None:
    self._running = False

  def _get_updates(self) -> list:
    params = {
      "offset": self._offset,
      "timeout": POLL_TIMEOUT,
      "allowed_updates": ["message"],
    }
    try:
      resp = requests.get(f"{self._base}/getUpdates", params=params, timeout=POLL_TIMEOUT + 5)
      data = resp.json()
      if not data.get("ok"):
        logger.warning("Telegram getUpdates not ok: %s", data)
        return []
      return data.get("result", [])
    except Exception as exc:
      logger.warning("Telegram getUpdates failed: %s", exc)
      time.sleep(5)
      return []

  def _process_update(self, update: dict) -> None:
    self._offset = update["update_id"] + 1
    message = update.get("message") or {}
    chat_id = str(message.get("chat", {}).get("id", ""))
    text = (message.get("text") or "").strip()

    if chat_id != self._chat_id:
      return
    if not text:
      return

    logger.info("Supervisor received message: %s", text[:100])
    self._send_typing(chat_id)

    try:
      reply = self._handler(text)
    except Exception as exc:
      logger.error("Supervisor agent error: %s", exc, exc_info=True)
      reply = f"Error processing your message: {exc}"

    self._send_reply(chat_id, reply)

  def _send_typing(self, chat_id: str) -> None:
    try:
      requests.post(f"{self._base}/sendChatAction", json={"chat_id": chat_id, "action": "typing"}, timeout=5)
    except Exception:
      pass

  def _send_reply(self, chat_id: str, text: str) -> None:
    if len(text) > MAX_MESSAGE_LENGTH:
      text = text[: MAX_MESSAGE_LENGTH - 4] + "\n..."
    payload = {"chat_id": chat_id, "text": text}
    try:
      resp = requests.post(f"{self._base}/sendMessage", json=payload, timeout=10)
      if resp.status_code != 200:
        logger.warning("Telegram sendMessage error %d: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
      logger.warning("Telegram send reply failed: %s", exc)


def start_telegram_bot(cfg: AppConfig) -> None:
  """Entry point: create supervisor handler and start polling. Blocking."""
  if not cfg.telegram.bot_token or not cfg.telegram.chat_id:
    logger.error("Supervisor bot requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
    return

  openai_client = _build_openai_client(cfg)

  def handle_message(text: str) -> str:
    return run_supervisor_agent(cfg, text, openai_client=openai_client)

  poller = TelegramBotPoller(
    token=cfg.telegram.bot_token,
    chat_id=cfg.telegram.chat_id,
    message_handler=handle_message,
  )
  poller.start()
