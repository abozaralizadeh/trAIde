from __future__ import annotations

import asyncio
import threading
from typing import Iterable

from .main import trading_loop

_started = False
_lock = threading.Lock()


def _start_background_loop() -> None:
  """Run the trading loop in a dedicated thread event loop."""
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(trading_loop())


def application(environ, start_response) -> Iterable[bytes]:
  """
  Minimal WSGI entrypoint so gunicorn can supervise the trading loop.
  Starts a single background thread on first HTTP request/health check.
  """
  global _started
  with _lock:
    if not _started:
      thread = threading.Thread(target=_start_background_loop, daemon=True)
      thread.start()
      _started = True

  start_response("200 OK", [("Content-Type", "text/plain")])
  return [b"trAIde trading loop running"]
