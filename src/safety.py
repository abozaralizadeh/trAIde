from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict


class TradingSafetyState:
  """Thread-safe authority shared by the polling loop and background agent.

  A model run receives a unique token. Entries remain authorized only while the main loop keeps
  producing complete snapshots and that exact run has not timed out or been revoked. Exchange
  mutations share ``order_lock`` so shutdown/protection and agent writes cannot interleave.
  """

  def __init__(self) -> None:
    self.order_lock = threading.RLock()
    self._lock = threading.RLock()
    self._healthy = False
    self._reason = "No complete snapshot yet"
    self._updated = 0.0
    self._total_usdt = 0.0
    self._active_token: str | None = None
    self._shutdown = False

  def refresh(self, total_usdt: float) -> None:
    with self._lock:
      if self._shutdown:
        return
      self._healthy = True
      self._reason = ""
      self._updated = time.time()
      self._total_usdt = max(0.0, float(total_usdt or 0.0))

  def authorize_run(self) -> str | None:
    with self._lock:
      if self._shutdown or not self._healthy:
        return None
      token = uuid.uuid4().hex
      self._active_token = token
      return token

  def finish_run(self, token: str | None) -> None:
    if not token:
      return
    with self._lock:
      if self._active_token == token:
        self._active_token = None

  def revoke_run(self, token: str | None, reason: str) -> None:
    if not token:
      return
    with self.order_lock:
      with self._lock:
        if self._active_token == token:
          self._active_token = None
          self._reason = reason

  def invalidate(self, reason: str, *, revoke_active: bool = True) -> None:
    with self.order_lock:
      with self._lock:
        self._healthy = False
        self._reason = str(reason or "Entry authority invalidated")
        self._updated = time.time()
        if revoke_active:
          self._active_token = None

  def begin_shutdown(self) -> None:
    # Wait for an already-started exchange mutation to finish, then prevent every later entry.
    with self.order_lock:
      with self._lock:
        self._shutdown = True
        self._healthy = False
        self._active_token = None
        self._reason = "Process is shutting down"
        self._updated = time.time()

  def check(self, token: str | None, *, max_age_sec: float) -> Dict[str, Any]:
    with self._lock:
      age = time.time() - self._updated if self._updated else float("inf")
      allowed = (
        bool(token)
        and not self._shutdown
        and self._healthy
        and self._active_token == token
        and age <= max(1.0, float(max_age_sec))
      )
      reason = self._reason
      if not reason and self._active_token != token:
        reason = "This model run no longer has entry authority"
      if not reason and age > max(1.0, float(max_age_sec)):
        reason = f"Latest complete account snapshot is stale ({age:.0f}s old)"
      return {
        "allowed": allowed,
        "reason": reason or "Entry authority unavailable",
        "snapshotAgeSec": age,
        "totalUsdt": self._total_usdt,
        "shutdown": self._shutdown,
      }
