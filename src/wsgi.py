from __future__ import annotations

import asyncio
import logging
import threading
from logging.handlers import RotatingFileHandler
from typing import Iterable

# Configure logging before anything else so all modules emit to stdout/stderr
# (captured by gunicorn and forwarded to journald).
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from .config import load_config
from .main import trading_loop

logger = logging.getLogger(__name__)

cfg = load_config()

if cfg.supervisor.log_file:
  _fh = RotatingFileHandler(
    cfg.supervisor.log_file,
    maxBytes=cfg.supervisor.log_max_bytes,
    backupCount=cfg.supervisor.log_backup_count,
  )
  _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
  logging.getLogger().addHandler(_fh)

_started = False
_lock = threading.Lock()


def _start_background_loop() -> None:
  """Run the trading loop in a dedicated thread event loop."""
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  logger.info("Starting trading loop (background thread via wsgi)...")
  loop.run_until_complete(trading_loop())


def _start_supervisor_bot() -> None:
  """Run the Telegram supervisor bot (blocking long-poll)."""
  from .telegram_bot import start_telegram_bot
  start_telegram_bot(cfg)


# Start immediately at module import so no HTTP hit is required.
with _lock:
  if not _started:
    thread = threading.Thread(target=_start_background_loop, daemon=True)
    thread.start()

    if cfg.supervisor.enabled and cfg.telegram.enabled:
      bot_thread = threading.Thread(target=_start_supervisor_bot, daemon=True, name="supervisor-bot")
      bot_thread.start()
      logger.info("Supervisor Telegram bot started.")

    _started = True


def application(environ, start_response) -> Iterable[bytes]:
  """
  Minimal WSGI entrypoint so gunicorn can supervise the trading loop.
  """
  start_response("200 OK", [("Content-Type", "text/plain")])
  return [b"trAIde trading loop running"]
