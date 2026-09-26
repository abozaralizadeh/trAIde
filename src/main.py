from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import signal
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, Optional

from agents import set_default_openai_client
from agents.tracing import (get_trace_provider)
from .agent import (
  TradingSnapshot, run_trading_agent, setup_tracing, setup_lstracing, _build_openai_client,
  _to_futures_symbol, SPOT_DUST_VALUE_USD,
)
from .analytics import flow_reading_max_age_sec, market_state, taker_flow_summary
from .config import load_config
from .dashboard_publisher import DashboardPublisher
from .edge import gate_scoreboard_from_store, gate_scoreboard_log_line, gate_state_cells, probe_cost_pct
from .kucoin import KucoinClient, KucoinFuturesClient, KucoinAccount, KucoinTicker
from .memory import (
  EXIT_PROBE_EXPIRE_HOURS,
  MemoryStore,
  _lease_settle_tolerance_sec,
  funding_received_from_history,
  position_open_time,
  sanitize_market_state,
)
from .position_context import carry_refresh_targets, exit_probe_inputs, trade_context
from .protection import ProtectionManager, replay_protection_stack
from .protection import _replay_bar as _validated_futures_bar
from .regime import (
  first_settlement_after, first_settlement_in_history, funding_clock_from_rate, held_position_noise_pct,
  macro_calendar_refresh_reason,
)
from .safety import TradingSafetyState
from .telegram import TelegramNotifier
from .utils import normalize_symbol

logger = logging.getLogger(__name__)

_AGENT_RUN_TIMEOUT_SEC = 20 * 60
_CALENDAR_RETRY_SEC = 6 * 3600   # min gap between calendar-refresh attempts if one fails
_AGENT_SHUTDOWN_GRACE_SEC = 30
_PRICE_NOISE_MULTIPLIER = 4.0
# Ceiling on the adaptive trigger, as a multiple of the base trigger. 2.0 keeps worst-case blindness
# bounded (any move >= 2× the base trigger always earns a fresh model look) — biased toward safety;
# raise via PRICE_TRIGGER_MAX_MULTIPLIER to save more tokens. Overridable per call from config.
_PRICE_TRIGGER_MAX_MULTIPLIER = 2.0
_PRICE_NOISE_ALPHA = 0.20
_PRICE_NOISE_MIN_SAMPLES = 3


class IncompleteFuturesSnapshot(RuntimeError):
  def __init__(self, failures: list[str], overview: dict | None, positions: list, stops: list) -> None:
    super().__init__("Incomplete futures snapshot (" + "; ".join(failures) + ")")
    self.failures = failures
    self.overview = overview
    self.positions = positions
    self.stops = stops


async def _run_in_daemon_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
  """Await blocking work in a daemon thread so a timed-out model cannot hold process shutdown."""
  loop = asyncio.get_running_loop()
  future = loop.create_future()

  def _worker() -> None:
    try:
      outcome = (True, fn(*args, **kwargs))
    except BaseException as exc:  # propagate model/tool failures back to the loop
      outcome = (False, exc)

    def _settle() -> None:
      if future.done():
        return
      if outcome[0]:
        future.set_result(outcome[1])
      else:
        future.set_exception(outcome[1])

    try:
      loop.call_soon_threadsafe(_settle)
    except RuntimeError:
      pass  # event loop already closed; daemon worker has no remaining authority

  threading.Thread(target=_worker, daemon=True, name="trAIde-agent").start()
  return await future


def _adaptive_agent_cooldown(
  *,
  flat_cooldown_sec: float,
  active_cooldown_sec: float,
  book_active: bool,
  new_events_count: int,
  trigger_move_pcts: list[float],
  price_trigger_pct: float,
) -> float:
  """Return a model-call cooldown that falls continuously as a flat-market move gets urgent.

  The configured flat value remains the quiet-market/cost ceiling and the active value remains
  the floor.  A move exactly at the trigger halves the quiet cooldown; larger moves and breadth
  (several symbols moving together) accelerate it further without another setting to tune.
  """
  flat = max(0.0, float(flat_cooldown_sec))
  active = max(0.0, min(flat, float(active_cooldown_sec)))
  if book_active or new_events_count:
    return active
  threshold = float(price_trigger_pct)
  moves = [abs(float(move)) for move in trigger_move_pcts if float(move) > 0]
  if threshold <= 0 or not moves or flat <= 0:
    return flat
  # At 1x threshold: flat/2; at 2x: flat/5; at 3x: flat/10. Multiple triggered
  # symbols add breadth to the urgency score, so a market-wide move is reviewed sooner.
  urgency_sq = (max(moves) / threshold) ** 2 * len(moves)
  return max(active, min(flat, flat / (1.0 + urgency_sq)))


def _productivity_adjusted_flat_cooldown(
  flat_cooldown_sec: float,
  unproductive_runs: int,
  max_multiplier: float = 1.0,
) -> float:
  """Optionally back off no-action deliberation; disabled when max_multiplier is 1."""
  base = max(0.0, float(flat_cooldown_sec))
  if base <= 0:
    return 0.0
  cap = max(1.0, float(max_multiplier))
  exponent = max(0, int(unproductive_runs))
  multiplier = min(cap, 2.0 ** exponent)
  return base * multiplier


def _idle_hunt_due(idle_polls: int, max_idle_polls: int, pending_orders: bool) -> bool:
  """Pending atomic entries wait for fill/expiry; they do not need idle LLM babysitting."""
  return not pending_orders and int(idle_polls) >= max(1, int(max_idle_polls))


def _crossed_auto_triggers(
  stored_triggers: list[dict],
  prices: Dict[str, float],
) -> list[tuple[dict, float]]:
  """Return explicit above/below trigger levels currently crossed by live prices."""
  crossed: list[tuple[dict, float]] = []
  for trigger in stored_triggers or []:
    if not isinstance(trigger, dict):
      continue
    condition = str(trigger.get("condition") or "").lower()
    if condition not in {"above", "below"}:
      continue
    symbol = normalize_symbol(trigger.get("symbol") or "")
    try:
      target = float(trigger.get("targetPrice"))
      current = float(prices.get(symbol))
    except (TypeError, ValueError):
      continue
    if target <= 0 or current <= 0:
      continue
    if (condition == "above" and current >= target) or (condition == "below" and current <= target):
      crossed.append((trigger, current))
  return crossed


# ── Measurement I/O on the survival thread: bounded per poll, backed off across polls (2026-09-25) ────
#
# Probe settlement, the stack replays and the market-state refresh run in the poll-loop thread right
# after the protection pass. Settlement used to read the snapshot tickers (zero calls); on futures marks
# it makes serial public calls per due symbol, and with no cap, no deadline and no memory of failures a
# hanging endpoint cost up to 20 symbols x 15s per poll — delaying the NEXT poll's ratchet, trail and
# naked-position repair (the scratch log already showed ~78s polls against POLL_INTERVAL_SEC=60). Three
# bounds, all derived from the loop's own cadence rather than tuned: a per-poll wall-clock budget (a
# quarter of the poll interval), a per-symbol retry backoff that doubles from one poll interval up to the
# row's remaining settle tolerance, and a short HTTP timeout for these reads only. Deferred rows simply
# wait; the existing tolerance rules write them off honestly.

_MEASURE_BUDGET_FRACTION = 0.25     # share of the poll interval measurement I/O may take per poll
_MEASURE_TIMEOUT_SEC = 5.0          # HTTP timeout for measurement reads (orders/positions keep 15s)


class _MeasurementBudget:
  """A per-poll wall-clock budget for measurement I/O. ``spent()`` is checked before every exchange call;
  ``defer()`` counts a call skipped because the budget ran out (logged once per poll by the loop)."""

  def __init__(self, seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
    self._clock = clock
    self.seconds = max(0.0, float(seconds))
    self._deadline = clock() + self.seconds
    self.deferred = 0

  def spent(self) -> bool:
    return self._clock() >= self._deadline

  def defer(self, n: int = 1) -> None:
    self.deferred += int(n)


def _budget_spent(budget: Optional[_MeasurementBudget]) -> bool:
  """True (and counted) when a budget is given and exhausted; no budget = unbounded (tests, scripts)."""
  if budget is not None and budget.spent():
    budget.defer()
    return True
  return False


class _MeasureBackoff:
  """Loop-owned per-key retry backoff for measurement fetches: ``{key: (retry_after, failures)}``.

  A failing symbol is re-asked after one poll interval, then 2, 4, 8... — capped at the remaining
  settle tolerance of the row that needs it (``cap_sec``), because past that the row is written off
  anyway. Success clears the key. Without it a hanging mark endpoint was re-asked every 60s for 9.6h
  (an unresolved exit probe) or 48 polls (a 240m horizon). Pure bookkeeping; never raises.
  """

  def __init__(self, base_sec: float) -> None:
    self.base = max(1.0, float(base_sec))
    self._state: Dict[Any, tuple[float, int]] = {}

  def ready(self, key: Any, now: float) -> bool:
    got = self._state.get(key)
    return got is None or float(now) >= got[0]

  def failed(self, key: Any, now: float, cap_sec: Optional[float] = None) -> float:
    failures = (self._state.get(key) or (0.0, 0))[1] + 1
    delay = self.base * (2 ** min(failures - 1, 30))
    if cap_sec is not None and math.isfinite(float(cap_sec)):
      delay = min(delay, max(self.base, float(cap_sec)))
    self._state[key] = (float(now) + delay, failures)
    return delay

  def ok(self, key: Any) -> None:
    self._state.pop(key, None)


def _measurement_client(kucoin_futures, timeout_sec: float = _MEASURE_TIMEOUT_SEC):
  """The futures client with a short HTTP timeout for measurement reads (``with_timeout``), or the client
  itself when it cannot make one (fakes, older clients). Never raises."""
  if kucoin_futures is None:
    return None
  make = getattr(kucoin_futures, "with_timeout", None)
  if callable(make):
    try:
      return make(timeout_sec)
    except Exception as exc:
      logger.warning("MEASUREMENT: short-timeout client unavailable (%s) — using the default timeout", exc)
  return kucoin_futures


def _futures_settlement_marks(symbols, kucoin_futures, snapshot, *, budget: Optional[_MeasurementBudget] = None,
                              backoff: Optional[_MeasureBackoff] = None, now: Optional[float] = None
                              ) -> Dict[str, float]:
  """FUTURES mark prices for the symbols whose probes settle this poll — never the spot ticker.

  A held position's own ``markPrice`` (already in the snapshot) is used first; anything else costs one
  small public ``get_mark_price`` call, made once per symbol per poll and only for symbols that are
  actually due, so this is a handful of calls rather than a 1.4 MB ``/contracts/active`` pull every
  minute. A symbol whose mark cannot be read is left OUT — never filled from spot, because a spot
  price against a futures base is exactly the basis error this map exists to remove. Its probes wait
  and the settle tolerance records them as unmeasured if the mark never comes. Failures are isolated
  per symbol and logged at WARNING: a silent settle outage freezes every edge verdict (2026-09).

  ``symbols`` may be a ``{symbol: cutoff_ts}`` map (memory.settlement_cutoffs): symbols are then asked
  in order of their earliest cutoff, so a limited ``budget`` goes to the measurements that would be
  lost first, and a failing symbol's ``backoff`` is capped at its remaining tolerance. Once the budget
  is spent the remaining symbols wait for a later poll.
  """
  now_ts = time.time() if now is None else float(now)
  cutoffs = dict(symbols) if isinstance(symbols, dict) else {s: float("inf") for s in (symbols or ())}
  marks: Dict[str, float] = {}
  held: Dict[str, float] = {}
  for position in getattr(snapshot, "futures_positions", None) or []:
    if not isinstance(position, dict):
      continue
    symbol = normalize_symbol(position.get("symbol") or "")
    try:
      mark = float(position.get("markPrice") or 0)
    except (TypeError, ValueError):
      continue
    if symbol and math.isfinite(mark) and mark > 0:
      held[symbol] = mark
  for symbol in sorted((s for s in cutoffs if s), key=lambda s: (cutoffs[s], s)):
    if symbol in held:
      marks[symbol] = held[symbol]
      continue
    if kucoin_futures is None:
      continue
    futures_symbol = _to_futures_symbol(symbol)
    if not futures_symbol:
      continue
    key = ("mark", futures_symbol)
    if backoff is not None and not backoff.ready(key, now_ts):
      continue
    if _budget_spent(budget):
      continue
    try:
      value = float((kucoin_futures.get_mark_price(futures_symbol) or {}).get("value") or 0)
    except Exception as exc:
      retry = backoff.failed(key, now_ts, cap_sec=cutoffs[symbol] - now_ts) if backoff is not None else None
      logger.warning("PROBE SETTLE: futures mark for %s unavailable (%s) — its due probes wait%s "
                     "and are written off as unmeasured if it never comes", futures_symbol, exc,
                     f" (next try in {retry:.0f}s)" if retry is not None else "")
      continue
    if backoff is not None:
      backoff.ok(key)
    if math.isfinite(value) and value > 0:
      marks[symbol] = value
  return marks


class _PollFundingCredit:
  """``funding_received(symbol, side, t0, t1)`` for one poll: lazy, memoized per symbol, never raises.

  Fetches ``get_funding_rate_history`` for a symbol the first time a probe on it settles this poll and
  reuses it for every other probe on that symbol (re-fetching only if a later probe needs an earlier
  start). A symbol whose fetch fails is not retried inside the same poll. Returns the funding a
  position on ``side`` would have RECEIVED over (t0, t1] as a fraction of notional — longs receive
  ``-rate``, shorts ``+rate`` — or None when it cannot be known (the probe then carries no credit).
  """

  _FAILED = object()

  def __init__(self, kucoin_futures, now: float, *, budget: Optional[_MeasurementBudget] = None,
               backoff: Optional[_MeasureBackoff] = None) -> None:
    self._client = kucoin_futures
    self._now = float(now)
    self._memo: Dict[str, Any] = {}
    self._budget = budget
    self._backoff = backoff

  def __call__(self, symbol: str, side: str, t0: float, t1: float) -> Optional[float]:
    if self._client is None:
      return None
    futures_symbol = _to_futures_symbol(normalize_symbol(symbol or ""))
    if not futures_symbol:
      return None
    start_ms = int(float(t0) * 1000)
    cached = self._memo.get(futures_symbol)
    if cached is self._FAILED:
      return None
    if cached is None or cached[0] > start_ms:
      key = ("funding", futures_symbol)
      if self._backoff is not None and not self._backoff.ready(key, self._now):
        return None       # unknown this poll: memory keeps f{h} None and backfills it later
      if _budget_spent(self._budget):
        return None
      try:
        rows = self._client.get_funding_rate_history(
          futures_symbol, start_at=start_ms, end_at=int((self._now + 60) * 1000),
        )
        # A non-list payload is NOT "no settlements": caching [] here once stamped a silent 0.0.
        if not isinstance(rows, list):
          raise ValueError(f"funding history payload is {type(rows).__name__}, not a list")
      except Exception as exc:
        if self._backoff is not None:
          self._backoff.failed(key, self._now)
        logger.warning("PROBE FUNDING: history for %s unavailable (%s) — the credit stays unknown and is "
                       "backfilled on a later poll", futures_symbol, exc)
        self._memo[futures_symbol] = self._FAILED
        return None
      if self._backoff is not None:
        self._backoff.ok(key)
      cached = (start_ms, rows)
      self._memo[futures_symbol] = cached
    return funding_received_from_history(cached[1], side, t0, t1)


class _PollLeaseExtremes:
  """``lease_extremes(symbol, t0, t1)`` for one poll: the ``(low, high)`` a contract traded over a probe's
  order lease. Never raises.

  Read from 1m FUTURES bars whose OPEN time falls in ``[t0, t1)`` — the lease's own minutes; the partial
  minute before the call is left out, so a low printed just before the call is never counted as a fill.
  Every bar must prove its column order (``protection._replay_bar``: high = row max, low = row min —
  futures rows are ``[ts, open, HIGH, LOW, close]``, and a spot-order misread once reversed a finding).
  One paged fetch per symbol per poll serves every due probe on it; a symbol that fails is not retried
  inside the same poll (WARNING) and simply waits — memory writes it off after its tolerance. Returns
  None when there is no bar inside the window yet.

  Why a probe needs this (2026-09-25): the execution map could only see the depths the model actually
  rested at, so once it stops resting deep, the deep buckets go blind — exactly the regime (chop,
  range-edge fades) where deep limits might pay again. With the lease low/high, `edge.execution_map`
  scores EVERY call at EVERY depth.
  """

  _FAILED = object()

  def __init__(self, kucoin_futures, *, budget: Optional[_MeasurementBudget] = None,
               backoff: Optional[_MeasureBackoff] = None, now: Optional[float] = None) -> None:
    self._client = kucoin_futures
    self._memo: Dict[str, Any] = {}
    self._budget = budget
    self._backoff = backoff
    self._now = time.time() if now is None else float(now)

  def __call__(self, symbol: str, t0: float, t1: float) -> Optional[tuple]:
    if self._client is None:
      return None
    futures_symbol = _to_futures_symbol(normalize_symbol(symbol or ""))
    if not futures_symbol:
      return None
    lo_s, hi_s = float(t0), float(t1)
    cached = self._memo.get(futures_symbol)
    if cached is self._FAILED:
      return None
    if cached is None or cached[0] > lo_s or cached[1] < hi_s:
      start = lo_s if cached is None else min(lo_s, cached[0])
      end = hi_s if cached is None else max(hi_s, cached[1])
      key = ("lease", futures_symbol)
      if self._backoff is not None and not self._backoff.ready(key, self._now):
        return None
      if _budget_spent(self._budget):
        return None
      try:
        bars = [_validated_futures_bar(row) for row in _fetch_futures_1m_bars(
          self._client, futures_symbol, start, end, budget=self._budget)]
      except _BudgetSpent:
        return None       # not a failure: the rest waits for a later poll
      except Exception as exc:
        if self._backoff is not None:
          # Capped at the lease's own retry window: past it the probe's extremes are written off.
          self._backoff.failed(key, self._now, cap_sec=_lease_settle_tolerance_sec((hi_s - lo_s) / 60.0))
        logger.warning("PROBE LEASE: 1m futures bars for %s unavailable (%s) — lease extremes wait",
                       futures_symbol, exc)
        self._memo[futures_symbol] = self._FAILED
        return None
      if self._backoff is not None:
        self._backoff.ok(key)
      cached = (start, end, bars)
      self._memo[futures_symbol] = cached
    inside = [bar for bar in cached[2] if lo_s <= bar[0] < hi_s]
    if not inside:
      return None
    return min(bar[3] for bar in inside), max(bar[2] for bar in inside)


def _settle_probes_on_futures(memory, kucoin_futures, snapshot, spot_prices, now: float | None = None, *,
                              budget: Optional[_MeasurementBudget] = None,
                              backoff: Optional[_MeasureBackoff] = None) -> Dict[str, float]:
  """Settle signal and exit probes on the market that actually fills: the FUTURES MARK, plus funding.

  Until 2026-09-25 the loop passed its SPOT ticker map (``live_prices``) straight into both settle
  steps while the probe base was the futures mark. On extreme-funding coins the perp/spot basis was
  then scored as a directional return: ONE-USDT on 09-20 traded ~0.0050 spot vs ~0.0038 perp, its
  probes recorded +25% at 5m on a +2.5% contract move, and `funding_carry` read t=1.90 ("paying",
  sized 0.94x) where futures settlement gives t~0.90 — a coin flip on the stand-aside line. Every other
  family was within ~0.06% either way, which is how the bias hid.

  ``spot_prices`` still settles the rare probe whose base fell back to spot (``priceSource`` spot) and
  is still what the auto-triggers read — this step only takes over settlement. Total by construction:
  every stage is isolated and logged at WARNING, nothing here can alter a trading decision or raise
  into the poll loop. Returns the futures mark map it settled with (for logging and tests).

  ``budget`` / ``backoff`` (loop-owned) bound the exchange calls this makes on the survival thread: the
  most urgent symbols first, nothing once the per-poll budget is spent, and a failing symbol re-asked
  on a doubling backoff instead of every poll (see ``_MeasurementBudget``).
  """
  now_ts = time.time() if now is None else float(now)
  marks: Dict[str, float] = {}
  try:
    cutoffs = getattr(memory, "settlement_cutoffs", None)
    due = cutoffs(now_ts) if callable(cutoffs) else memory.symbols_due_for_settlement(now_ts)
  except Exception as exc:
    logger.warning("PROBE SETTLE: could not list due probes (%s) — settling with no futures marks", exc)
    due = set()
  try:
    marks = _futures_settlement_marks(due, kucoin_futures, snapshot, budget=budget, backoff=backoff, now=now_ts)
  except Exception as exc:
    logger.warning("PROBE SETTLE: futures mark map failed (%s) — due probes wait this poll", exc)
    marks = {}
  funding = (_PollFundingCredit(kucoin_futures, now_ts, budget=budget, backoff=backoff)
             if kucoin_futures is not None else None)
  # Each probe's order-lease low/high, for the execution map's every-call-every-depth counterfactual.
  lease = (_PollLeaseExtremes(kucoin_futures, budget=budget, backoff=backoff, now=now_ts)
           if kucoin_futures is not None else None)
  # Separate try blocks: a failing signal settle must not also cost the exit probes their poll.
  try:
    memory.settle_signal_probes(marks, spot_prices=spot_prices, funding_received=funding,
                                lease_extremes=lease)
  except Exception as exc:
    logger.warning("SIGNAL PROBE settle failed (%s) — edge verdicts go stale while this repeats", exc)
  try:
    memory.settle_exit_probes(marks)
  except Exception as exc:
    logger.warning("EXIT PROBE settle failed (%s) — exitDiscipline goes stale while this repeats", exc)
  # Gate-state rows whose every horizon is now stamped fold into their day cells (the gate scoreboard's
  # long-horizon store); edge.gate_state_cells owns the return arithmetic and the de-overlap. Own try.
  try:
    memory.fold_settled_gate_states(gate_state_cells)
  except Exception as exc:
    logger.warning("GATE STATE FOLD failed (%s) — settled state rows wait for the next poll", exc)
  return marks


_GATE_SCOREBOARD_LOG_SEC = 3600   # the report-only gate scoreboard is logged at most this often


def _log_gate_scoreboard(memory, cfg, last: Dict[str, float], now: float | None = None) -> bool:
  """At most once an hour, one INFO line: what each directional gate blocks vs what it allows.

  REPORT-ONLY and deliberately NOT in the trading prompt (edge.gate_scoreboard says why). The line is
  what lets the operator see — from the log alone — that gate-state rows are still flowing (a stalled
  probe feed froze every edge verdict for days in Sep 2026) and how far each gate is from a verdict.
  Returns True when it logged. Total: never raises into the loop.
  """
  now_ts = time.time() if now is None else float(now)
  if now_ts - float(last.get("ts") or 0.0) < _GATE_SCOREBOARD_LOG_SEC:
    return False
  last["ts"] = now_ts
  try:
    board = gate_scoreboard_from_store(memory, cost_pct=probe_cost_pct(memory, cfg))
    logger.info("GATE SCOREBOARD (report-only): %s", gate_scoreboard_log_line(board))
  except Exception as exc:
    logger.warning("GATE SCOREBOARD log failed (%s)", exc)
  return True


# Live-exit-stack replay of the model's early closes (exitDiscipline's benchmark), 2026-09-25.
_STACK_MAX_PER_POLL = 2              # replays per poll: a few small kline pages each, after protection
_STACK_KLINE_PAGE_BARS = 200         # KuCoin futures kline page size (1m bars per request)
_STACK_RETRY_SEC = 15 * 60           # a failed replay is retried at most this often
_STACK_GIVE_UP_SEC = 24 * 3600       # ...and recorded as unavailable this long after its horizon
_STACK_MIN_COVERAGE_SEC = 15 * 60    # a replay whose bars stop this far short of the horizon is incomplete


class _BudgetSpent(RuntimeError):
  """The per-poll measurement budget ran out mid-fetch: not a failure, the work resumes next poll."""


def _fetch_futures_1m_bars(kucoin_futures, fsym: str, start_s: float, end_s: float, *,
                           budget: Optional[_MeasurementBudget] = None) -> list:
  """Raw 1m FUTURES kline rows covering [start_s, end_s], paged, de-duplicated by timestamp.

  Rows are returned as the exchange sends them ([ts_ms, open, HIGH, LOW, close, ...]);
  replay_protection_stack validates each one's column order. Raises on any page failure — a partial
  window must never be replayed as if it were complete — and raises ``_BudgetSpent`` when the per-poll
  measurement budget runs out before the next page (a long hold's replay can need tens of pages).
  """
  page = _STACK_KLINE_PAGE_BARS * 60
  by_ts: Dict[float, list] = {}
  t = int(start_s) // 60 * 60
  while t <= end_s:
    if _budget_spent(budget):
      raise _BudgetSpent(f"measurement budget spent before the page at {int(t)}")
    rows = kucoin_futures.get_candles(fsym, granularity=1, start_at=int(t * 1000),
                                      end_at=int(min(end_s, t + page) * 1000))
    for row in rows or []:
      try:
        by_ts[float(row[0])] = row
      except (TypeError, ValueError, IndexError):
        continue
    t += page
  return [by_ts[k] for k in sorted(by_ts)]


def _score_exit_probe_stacks(
  memory,
  kucoin_futures,
  protection_cfg,
  *,
  now: float | None = None,
  attempts: Dict[tuple, float] | None = None,
  max_per_poll: int = _STACK_MAX_PER_POLL,
  expire_hours: float = EXIT_PROBE_EXPIRE_HOURS,
  budget: Optional[_MeasurementBudget] = None,
) -> int:
  """Replay the LIVE exit stack for matured agent-closed exit probes and store ``stackR``. Never raises.

  exitDiscipline scores the model's early closes; its benchmark must be what the SYSTEM would have done
  had the model not closed — the bracket plus breakeven/trail/carry hold — not the bare bracket (on the
  5 attributed closes at the time: -6.23R vs the bracket, ~-3.5R vs the stack, and the sign of that gap
  depends on the regime). For at most ``max_per_poll`` agent probes (oldest first) whose horizon
  (``ts`` + ``expire_hours``, the same one bracketR expires on) has passed and that carry the replay
  inputs but no stack result yet: fetch 1m futures bars from the fill to the horizon, run
  protection.replay_protection_stack with ``protection_cfg`` — pass ProtectionManager's EFFECTIVE cfg
  (``protection.cfg``: breakeven_fee_pct raised to the round-trip cost), not the raw config — and store
  the result with memory.set_exit_probe_stack. Replaying from the FILL also rebuilds any ratchet the
  stop had made before the close, which the bracket probe ignored.

  A failure is logged at WARNING and retried no more than every 15 minutes (``attempts``, owned by the
  loop); a row still failing a day after its horizon is stored as ``unavailable`` and is excluded from
  exitDiscipline (counted as ``stackUnavailable``, never scored on its bracket instead).
  Returns the number of rows stored.
  """
  if kucoin_futures is None or protection_cfg is None:
    return 0
  now_ts = time.time() if now is None else float(now)
  memo = attempts if attempts is not None else {}
  horizon_sec = float(expire_hours) * 3600.0
  stored = 0
  try:
    rows = memory.exit_probes(limit=200)
  except Exception as exc:
    logger.warning("EXIT STACK: could not read exit probes (%s)", exc)
    return 0
  due = []
  for row in rows:
    try:
      if str(row.get("closedBy") or "").lower() != "agent" or row.get("stack"):
        continue
      ts0 = float(row.get("ts"))
      fill_ts = float(row.get("fillTs"))
      init_risk = float(row.get("initRiskPx"))
      if not (fill_ts > 0 and init_risk > 0 and math.isfinite(fill_ts) and math.isfinite(init_risk)):
        continue
      if now_ts < ts0 + horizon_sec:
        continue
      key = (row.get("symbol"), int(ts0))
      if now_ts - memo.get(key, 0.0) < _STACK_RETRY_SEC:
        continue
      due.append((ts0, row))
    except (TypeError, ValueError):
      continue
  for ts0, row in sorted(due, key=lambda item: item[0])[:max(0, int(max_per_poll))]:
    if _budget_spent(budget):
      break                     # the rest waits for a later poll; not a failed attempt
    symbol = row.get("symbol")
    key = (symbol, int(ts0))
    memo[key] = now_ts
    end_ts = ts0 + horizon_sec
    try:
      fsym = _to_futures_symbol(normalize_symbol(symbol or ""))
      if not fsym:
        raise ValueError("no futures contract mapping")
      fill_ts = float(row["fillTs"])
      try:
        bars = _fetch_futures_1m_bars(kucoin_futures, fsym, fill_ts - 60, end_ts, budget=budget)
      except _BudgetSpent:
        memo.pop(key, None)     # ran out of time mid-fetch: retry next poll, not in 15 minutes
        break
      result = replay_protection_stack(
        bars,
        side_long=str(row.get("positionSide") or "").lower() == "long",
        entry=float(row["entryPrice"]),
        stop=float(row["stopPrice"]),
        take_profit=float(row["takeProfitPrice"]),
        cfg=protection_cfg,
        init_risk=float(row["initRiskPx"]),
        fill_ts=fill_ts,
        end_ts=end_ts,
        noise_band_r=row.get("noiseBandR"),
        hold_until_ts=row.get("holdUntilTs"),
        open_until_ts=ts0,
      )
      if result.get("resolvedBy") == "expired" and float(result.get("resolvedTs") or 0) < end_ts - _STACK_MIN_COVERAGE_SEC:
        raise ValueError("bars stop short of the horizon — incomplete window, retrying later")
      if memory.set_exit_probe_stack(
        symbol, ts0, result.get("stackR"), result.get("resolvedBy"), result.get("resolvedTs"),
        "live_1m_replay", pre_close_exit_suppressed=result.get("preCloseExitSuppressed"),
      ):
        stored += 1
        logger.info("EXIT STACK: %s close @ %d scored against the live stack: taken %+.2fR vs stack %+.2fR (%s)",
                    symbol, int(ts0), float(row.get("realizedR") or 0.0), float(result.get("stackR") or 0.0),
                    result.get("resolvedBy"))
    except Exception as exc:
      if now_ts - end_ts > _STACK_GIVE_UP_SEC:
        logger.warning("EXIT STACK: %s close @ %d could not be replayed for a day (%s) — stored as "
                       "unavailable; excluded from exitDiscipline", symbol, int(ts0), exc)
        try:
          memory.set_exit_probe_stack(symbol, ts0, None, "unavailable", None, f"gave_up: {exc}"[:200])
        except Exception:
          pass
      else:
        logger.warning("EXIT STACK: replay for %s close @ %d failed (%s) — stackR unset, retrying later",
                       symbol, int(ts0), exc)
  return stored


def _adaptive_price_trigger_threshold(
  base_trigger_pct: float,
  noise_ewma_pct: float,
  samples: int,
  *,
  noise_multiplier: float = _PRICE_NOISE_MULTIPLIER,
  max_multiplier: float = _PRICE_TRIGGER_MAX_MULTIPLIER,
) -> float:
  """Raise the model trigger above a symbol's ordinary poll-to-poll noise.

  The configured threshold is always the floor. After a few observations, routine volatility must be
  exceeded by ``noise_multiplier`` times its EWMA, capped at ``max_multiplier`` times the configured
  floor so a genuinely structural move can never be suppressed indefinitely (the cap bounds
  worst-case blindness; the default keeps it tight for safety).
  """
  base = max(0.0, float(base_trigger_pct))
  if base <= 0 or int(samples) < _PRICE_NOISE_MIN_SAMPLES:
    return base
  noise_threshold = max(0.0, float(noise_ewma_pct)) * float(noise_multiplier)
  return min(base * max(1.0, float(max_multiplier)), max(base, noise_threshold))


def _next_price_noise_ewma(
  previous_ewma_pct: float,
  observed_move_pct: float,
  base_trigger_pct: float,
  samples: int,
  *,
  max_multiplier: float = _PRICE_TRIGGER_MAX_MULTIPLIER,
) -> float:
  """Update robust price noise without letting one shock disable later model reviews."""
  base = max(0.0, float(base_trigger_pct))
  ceiling = base * max(1.0, float(max_multiplier)) if base > 0 else abs(float(observed_move_pct))
  observed = min(abs(float(observed_move_pct)), ceiling)
  if int(samples) <= 0:
    return observed
  previous = max(0.0, float(previous_ewma_pct))
  return (1.0 - _PRICE_NOISE_ALPHA) * previous + _PRICE_NOISE_ALPHA * observed


_FLOW_EWMA_ALPHA = 0.3  # same responsiveness as the price-noise EWMA above


def _to_epoch(value: Any) -> float:
  """Coerce a persisted timestamp to seconds; anything unreadable is treated as infinitely old."""
  try:
    return float(value or 0)
  except (TypeError, ValueError):
    return 0.0


def _prune_flow_observations(
  observations: Dict[str, Dict[str, Any]],
  poll_interval_sec: float,
  now_ts: float,
) -> int:
  """Drop tape readings that have aged out, in place. Returns how many were forgotten.

  Without this the persisted state grows with every symbol the coin universe ever rotated through,
  and — the real damage — keeps serving a reading that *looks* like a tape observation while
  describing a market from days ago. See analytics.flow_reading_max_age_sec.

  Mutates rather than rebinding on purpose: the caller's dict is shared with the closure that
  persists scheduler state, and a rebinding there would silently persist the unpruned copy if this
  block ever moved into a nested function.
  """
  cutoff = now_ts - flow_reading_max_age_sec(poll_interval_sec)
  stale = [
    symbol for symbol, observation in observations.items()
    if not isinstance(observation, dict) or _to_epoch(observation.get("updated")) < cutoff
  ]
  for symbol in stale:
    observations.pop(symbol, None)
  return len(stale)


def _next_flow_observation(
  previous: Dict[str, Any] | None,
  summary: Dict[str, Any],
  now_ts: int,
) -> Dict[str, Any]:
  """Fold one taker-tape reading into a symbol's running flow state.

  The EWMA tracks the share of the *newly arrived* trades, not the whole window. One 100-trade
  window spans several minutes on a mid-cap perp while the loop polls every 60s, so consecutive
  windows share most of their rows; smoothing the window share would mostly be smoothing the same
  trades over and over, and would report a confidence the sample does not have. When the new slice
  is empty (nothing traded, or no cursor to compare against) the window share seeds the series
  instead, so a quiet symbol still has a usable level rather than a gap.
  """
  prior = previous if isinstance(previous, dict) else {}
  window_share = summary.get("buyShare")
  fresh_share = summary.get("newBuyShare")
  observed = fresh_share if fresh_share is not None else window_share
  try:
    samples = max(0, int(prior.get("samples") or 0))
  except (TypeError, ValueError):
    samples = 0
  try:
    previous_ewma = float(prior.get("buyShareEwma"))
  except (TypeError, ValueError):
    previous_ewma = None

  state = dict(summary)
  state["updated"] = int(now_ts)
  if observed is None:
    # Nothing measurable this poll: carry the level forward rather than resetting it to neutral,
    # which would read as "balanced flow" when it actually means "no data".
    state["buyShareEwma"] = previous_ewma
    state["samples"] = samples
    return state
  observed = float(observed)
  state["buyShareEwma"] = round(
    observed if previous_ewma is None or samples <= 0
    else (1.0 - _FLOW_EWMA_ALPHA) * previous_ewma + _FLOW_EWMA_ALPHA * observed,
    4,
  )
  state["samples"] = min(samples + 1, 1_000_000)
  return state


def _flow_symbols(
  snapshot,
  max_symbols: int,
  pending_moves: Dict[str, float] | None = None,
) -> list[str]:
  """Which symbols to sample the tape for: held positions, then whatever the model is about to look at.

  One public REST call per symbol per poll is cheap but not free, and the coin list can rotate wide
  when flexible coins are on. Positions we are actually carrying are sampled first so the cap can
  never starve an open trade of its record in favour of a watchlist name.

  Everything after that is ranked by the move that is *waking the agent* — `pending_moves` is the
  same per-symbol excursion the price trigger uses, and it is retained until the model reviews it, so
  a symbol stays sampled from the moment it becomes interesting until the call is actually made. The
  first version ranked alphabetically, which sounds neutral and is not: with a 50-coin universe and a
  cap of 12 the tape was permanently recorded for AAVE…FARTCOIN while the model spent 2026-09-05..07
  calling TAO, INJ, WLD and XLM — none of them sampled. A reading that never covers the symbols being
  traded cannot answer the question it is collected to answer.
  """
  held: list[str] = []
  for position in getattr(snapshot, "futures_positions", None) or []:
    if not isinstance(position, dict):
      continue
    try:
      qty = float(position.get("currentQty") or 0)
    except (TypeError, ValueError):
      continue
    symbol = normalize_symbol(position.get("symbol") or "")
    if qty and symbol and symbol not in held:
      held.append(symbol)
  known = [s for s in sorted(getattr(snapshot, "tickers", None) or {}) if s not in held]
  moves = pending_moves or {}
  # Largest pending move first; alphabetical within the untriggered tail keeps the order stable.
  known.sort(key=lambda s: -float(moves.get(s) or 0.0))
  return (held + known)[:max(0, int(max_symbols))]


def _rebase_reviewed_price_triggers(
  pending_moves: Dict[str, float],
  pending_discrete_triggers: set[str],
  reviewed_symbols: set[str],
) -> None:
  """Discard trigger evidence measured against the pre-run anchor for reviewed symbols."""
  for symbol in reviewed_symbols:
    pending_moves.pop(symbol, None)
  stale_initials = {
    trigger
    for trigger in pending_discrete_triggers
    if trigger.startswith("initial:") and trigger.split(":", 1)[1] in reviewed_symbols
  }
  pending_discrete_triggers.difference_update(stale_initials)


def _load_active_coins(cfg, memory: MemoryStore) -> list[str]:
  if not cfg.trading.flexible_coins_enabled:
    return cfg.trading.coins

  if not memory.has_coins():
    memory.set_coins(cfg.trading.coins, reason="seed-from-config")
    return cfg.trading.coins

  coins = memory.get_coins(default=cfg.trading.coins)
  if not coins:
    memory.set_coins(cfg.trading.coins, reason="seed-from-config")
    return cfg.trading.coins
  return coins


_TICKER_FAIL_COUNTS: dict[str, int] = {}
_TICKER_FAIL_THRESHOLD = 3  # consecutive failures before removal


def _fetch_tickers(
  cfg,
  kucoin: KucoinClient,
  coins: list[str],
  memory: MemoryStore,
  preserve_symbols: set[str] | None = None,
) -> tuple[list[str], dict]:
  """Normalize symbols, fetch tickers with retry. Only remove after repeated consecutive failures."""
  normalized: list[str] = []
  seen: set[str] = set()
  for sym in coins:
    norm = normalize_symbol(sym)
    if norm and norm not in seen:
      normalized.append(norm)
      seen.add(norm)

  tickers = {}
  missing: list[str] = []
  for symbol in normalized:
    try:
      tickers[symbol] = kucoin.get_ticker(symbol)
      _TICKER_FAIL_COUNTS.pop(symbol, None)  # reset on success
    except Exception as exc:
      missing.append(symbol)
      _TICKER_FAIL_COUNTS[symbol] = _TICKER_FAIL_COUNTS.get(symbol, 0) + 1
      fail_count = _TICKER_FAIL_COUNTS[symbol]
      logger.warning("Ticker fetch failed for %s (%d/%d): %s", symbol, fail_count, _TICKER_FAIL_THRESHOLD, exc)
      if (
        cfg.trading.flexible_coins_enabled
        and fail_count >= _TICKER_FAIL_THRESHOLD
        and symbol not in (preserve_symbols or set())
      ):
        try:
          removal = memory.remove_coin(
            symbol,
            reason=f"Ticker unavailable {fail_count} consecutive times: {exc}",
            exit_plan="Auto-removed after repeated failures; re-add when symbol is confirmed available.",
          )
          logger.warning("Removed unavailable symbol from active universe after %d failures: %s", fail_count, removal.get("symbol"))
          _TICKER_FAIL_COUNTS.pop(symbol, None)
        except Exception as remove_exc:
          logger.warning("Failed to remove unavailable symbol %s: %s", symbol, remove_exc)

  if not tickers and not (preserve_symbols or set()):
    raise RuntimeError(f"No tickers available; failed symbols: {missing}")

  return list(tickers.keys()), tickers


def _discover_unlisted_holdings(kucoin: KucoinClient, spot_accounts: list, existing_tickers: dict, min_value_usd: float = SPOT_DUST_VALUE_USD) -> tuple[list[str], dict]:
  """Discover spot holdings not in the active coin list by scanning balances."""
  known_bases = {sym.split("-")[0].upper() for sym in existing_tickers if "-" in sym}
  known_bases.add("USDT")
  extra_coins: list[str] = []
  extra_tickers: dict = {}
  for acct in spot_accounts:
    cur = acct.currency.upper()
    if cur in known_bases:
      continue
    bal = float(acct.balance or 0)
    if bal <= 0:
      continue
    symbol = f"{cur}-USDT"
    try:
      ticker = kucoin.get_ticker(symbol)
      if bal * float(ticker.price) < min_value_usd:
        continue
      extra_coins.append(symbol)
      extra_tickers[symbol] = ticker
      known_bases.add(cur)
      logger.info("Discovered unlisted holding: %s (%.4f units, ~$%.2f)", symbol, bal, bal * float(ticker.price))
    except Exception:
      continue
  return extra_coins, extra_tickers


def _fetch_balances(kucoin: KucoinClient) -> tuple[list, list, list]:
  """Fetch spot trade accounts, financial accounts, and all accounts. Returns (spot, financial, all)."""
  spot_accounts = kucoin.get_trade_accounts()
  financial_accounts: list = []
  try:
    financial_accounts = kucoin.get_financial_accounts()
  except Exception as exc:
    logger.warning("Unable to fetch financial accounts: %s", exc)
  all_accounts = kucoin.get_accounts()
  return spot_accounts, financial_accounts, all_accounts


def _fetch_futures(cfg, kucoin_futures: KucoinFuturesClient | None) -> tuple[dict | None, list, list]:
  """Fetch futures account overview, positions, and stop orders. Returns (overview, positions, stops)."""
  if not (cfg.kucoin_futures.enabled and kucoin_futures):
    return None, [], []
  failures: list[str] = []
  overview = None
  positions: list = []
  stops: list = []
  try:
    overview = kucoin_futures.get_account_overview()
  except Exception as exc:
    failures.append(f"overview: {exc}")
  try:
    positions = kucoin_futures.list_positions() or []
  except Exception as exc:
    failures.append(f"positions: {exc}")
  try:
    stops = kucoin_futures.list_stop_orders(status="active") or []
  except Exception as exc:
    failures.append(f"stops: {exc}")
  if failures:
    # Never turn a partial API failure into an apparently flat/unprotected book. The outer loop
    # retries the entire snapshot and blocks all entries until exchange truth is complete again.
    raise IncompleteFuturesSnapshot(failures, overview, positions, stops)
  return overview, positions, stops


def _fill_event_id(entry: dict, venue: str) -> str:
  """Stable fill identity for poll/restart deduplication, including partial fills."""
  for key in ("tradeId", "id", "fillId"):
    value = entry.get(key)
    if value not in (None, ""):
      return f"{venue}:{value}"
  parts = [
    entry.get("orderId"), entry.get("createdAt") or entry.get("tradeCreatedAt"),
    entry.get("symbol"), entry.get("side"), entry.get("price"),
    entry.get("size") or entry.get("filledSize"),
  ]
  if not any(value not in (None, "") for value in parts):
    digest = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return f"{venue}:payload:{digest}"
  return f"{venue}:" + ":".join(str(value or "") for value in parts)


def _close_event_id(entry: dict) -> str:
  """Stable close identity; fallback includes lifecycle fields instead of open time alone."""
  value = entry.get("id") or entry.get("positionId")
  if value not in (None, ""):
    return str(value)
  parts = [
    entry.get("symbol"), entry.get("type") or entry.get("side"),
    entry.get("openTime") or entry.get("openingTimestamp"),
    entry.get("closeTime") or entry.get("updatedAt"),
  ]
  if not any(value not in (None, "") for value in parts):
    digest = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return f"close:payload:{digest}"
  return "close:" + ":".join(str(value or "") for value in parts)


def _fetch_recent_fills(kucoin: KucoinClient, kucoin_futures: KucoinFuturesClient | None, lookback_minutes: int = 10) -> Dict:
  """Fetch fills and closed positions from the last N minutes (using KuCoin server time to avoid clock drift)."""
  now_ms = kucoin._timestamp_ms()
  cutoff_ms = now_ms - lookback_minutes * 60 * 1000
  result: Dict[str, Any] = {"spot_fills": [], "futures_fills": [], "closed_positions": [], "_errors": []}

  def _after_cutoff(entry: dict, *ts_keys: str) -> bool:
    for k in ts_keys:
      v = entry.get(k)
      if v is not None:
        ts = int(v) if int(v) > 1e12 else int(v) * 1000
        if ts >= cutoff_ms:
          return True
    return False

  def _paged(fetch: Callable[..., list], *, page_size: int, ts_keys: tuple[str, ...], label: str) -> list:
    rows_in_window: list = []
    seen_payloads: set[str] = set()
    previous_page_signature = ""
    max_pages = 20
    for page in range(1, max_pages + 1):
      rows = fetch(page=page, page_size=page_size) or []
      signature = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
      if page > 1 and signature == previous_page_signature:
        break  # defensive: endpoint ignored page and repeated page 1
      previous_page_signature = signature
      for row in rows:
        if not isinstance(row, dict) or not _after_cutoff(row, *ts_keys):
          continue
        payload_id = hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()
        if payload_id not in seen_payloads:
          seen_payloads.add(payload_id)
          rows_in_window.append(row)
      if len(rows) < page_size:
        break
      timestamps = []
      for row in rows:
        if not isinstance(row, dict):
          continue
        for key in ts_keys:
          value = row.get(key)
          if value is None:
            continue
          try:
            ts = int(value)
            timestamps.append(ts if ts > 1e12 else ts * 1000)
          except (TypeError, ValueError):
            pass
          break
      if timestamps and min(timestamps) < cutoff_ms:
        break
    else:
      result["_errors"].append(f"{label}: pagination exceeded {max_pages} pages")
    return rows_in_window

  try:
    result["spot_fills"] = _paged(
      kucoin.get_fills, page_size=50,
      ts_keys=("createdAt", "tradeCreatedAt"), label="spot fills",
    )
  except Exception as exc:
    logger.warning("Unable to fetch spot fills: %s", exc)
    result["_errors"].append(f"spot fills: {exc}")

  if kucoin_futures:
    try:
      result["futures_fills"] = _paged(
        kucoin_futures.get_fills, page_size=50,
        ts_keys=("createdAt", "tradeCreatedAt"), label="futures fills",
      )
    except Exception as exc:
      logger.warning("Unable to fetch futures fills: %s", exc)
      result["_errors"].append(f"futures fills: {exc}")

    try:
      result["closed_positions"] = _paged(
        kucoin_futures.get_position_history, page_size=50,
        ts_keys=("closeTime", "updatedAt"), label="closed positions",
      )
    except Exception as exc:
      logger.warning("Unable to fetch closed positions: %s", exc)
      result["_errors"].append(f"closed positions: {exc}")

  return result


def _fetch_fees(kucoin: KucoinClient) -> dict:
  """Fetch base fee rates; returns empty dict on failure."""
  try:
    fee_info = kucoin.get_base_fee()
    if fee_info:
      return {
        "spot_taker": float(fee_info.get("takerFeeRate") or 0.001),
        "spot_maker": float(fee_info.get("makerFeeRate") or 0.001),
      }
  except Exception as exc:
    logger.warning("Unable to fetch base fee: %s", exc)
  return {}


def _live_extremes_map(snapshot) -> Dict[str, dict]:
  """Active-position map from live exchange truth (futures) for peak/trough PnL tracking.

  Driving `update_position_extremes` off live positions makes a symbol's extremes reset when its
  position actually closes — instead of lingering forever on a memory-reconstructed phantom long
  (futures triggered-closes are logged as decisions, not written to the trades ledger, so the
  ledger only ever accumulates buys). Futures-only: the bot trades futures exclusively; add spot
  positions here if spot trading resumes.
  """
  out: Dict[str, dict] = {}
  for p in getattr(snapshot, "futures_positions", None) or []:
    if not isinstance(p, dict):
      continue
    try:
      qty = float(p.get("currentQty") or 0)
    except (TypeError, ValueError):
      continue
    if not qty:
      continue
    sym = normalize_symbol(p.get("symbol") or "")
    if not sym:
      continue
    upnl = p.get("unrealisedPnl")
    if upnl is None:
      upnl = p.get("unrealizedPnl")
    try:
      upnl = float(upnl) if upnl is not None else None
    except (TypeError, ValueError):
      upnl = None
    out[sym] = {
      "netSize": qty,
      "unrealizedPnl": upnl,
      "positionOpenTime": p.get("openingTimestamp") or p.get("openTime"),
      "positionSide": "long" if qty > 0 else "short",
      # For the PRICE-space trail peak (memory.update_position_extremes -> peakFePx), keyed on the
      # same (openTime, side, |qty|, avgEntry) identity ProtectionManager resets on. The open time is
      # read in ProtectionManager's own key order, which `positionOpenTime` above does not use.
      "lifecycleOpenTime": position_open_time(p),
      "markPrice": p.get("markPrice"),
      "avgEntryPrice": p.get("avgEntryPrice"),
    }
  return out


class _FundingClock:
  """The exchange's own funding-settlement clock per contract, for HELD funding_carry positions.

  ``clock(fsym) -> (next_settlement_ts, interval_sec)`` in epoch seconds, or None when unknown. Read
  from ``get_funding_rate``: ``fundingTime`` is the next settlement (ms), ``granularity`` the interval
  (ms). KuCoin changes a contract's interval as funding moves (ONE 8h->1h on 2026-09-17, G 4h->1h on
  09-20), so a clock is refetched once its cached settlement has passed or it is older than
  ``REFRESH_SEC``.

  CACHE-ONLY LOOKUPS, NETWORK ONLY IN ``refresh`` (2026-09-25 review). ``__call__``, ``settlement_since``
  and ``rate`` never touch the network: ProtectionManager.run calls the lookup under ``order_lock`` (and
  on the degraded-snapshot path that exists precisely for a flaky exchange), and a hanging funding
  endpoint used to cost up to the 15s request timeout per held carry position per poll there —
  delaying later positions' hard caps and blocking the agent's order mutations. Only the poll loop
  calls ``refresh``, OUTSIDE the lock and before protection, the same split as ``_MarketStateClock``.

  ``refresh`` also reads the contract's funding HISTORY since each carry's fill until it shows the first
  settlement after it (then never again for that fill), so ``carry_hold_deadline`` can tell whether a
  payment has actually happened instead of walking a changed grid into the past (W2). A settlement the
  previous clock promised at or before now that the history does not list yet (publication lag) is
  taken as paid at that time, so a lag can never re-engage a hold.

  Total by construction: a failed fetch keeps the last cached values, logs WARNING once per failure
  streak, and is not retried for ``RETRY_SEC`` — a hanging endpoint costs at most one timeout per
  symbol per ``RETRY_SEC``, never one per poll. With no clock the hold falls back to the clock stamped
  on the entry, then the 8h grid — never to "no hold".
  """

  REFRESH_SEC = 15 * 60
  RETRY_SEC = 5 * 60

  def __init__(self, kucoin_futures, *, time_fn: Callable[[], float] = time.time) -> None:
    self._client = kucoin_futures
    self._time = time_fn
    self._cache: Dict[str, tuple[float, float, float]] = {}
    self._rates: Dict[str, float] = {}
    self._failed_at: Dict[str, float] = {}
    # (fsym, fill_ts) -> (checked_at, first settlement after the fill or None)
    self._paid: Dict[tuple, tuple[float, Optional[float]]] = {}
    self._hist_failed_at: Dict[tuple, float] = {}
    self._warned: set = set()
    self._lock = threading.Lock()

  # ── network: the poll loop only ──

  def refresh(self, targets: Any) -> int:
    """Refetch due clocks (and, for ``{fsym: fill_ts}`` targets, the settlement history since the fill)
    for the held carry contracts. Returns the number of successful fetches. Never raises."""
    try:
      items = list(targets.items()) if isinstance(targets, dict) else [(t, None) for t in (targets or ())]
    except Exception:
      return 0
    done = 0
    held_keys = set()
    for fsym, fill in items:
      if not fsym:
        continue
      try:
        fill_f = float(fill) if fill is not None else None
      except (TypeError, ValueError):
        fill_f = None
      if fill_f is not None and not (math.isfinite(fill_f) and fill_f > 0):
        fill_f = None
      try:
        now = float(self._time())
        with self._lock:
          prev = self._cache.get(fsym)
        done += int(self._refresh_clock(fsym, now))
        if fill_f is not None:
          held_keys.add((fsym, fill_f))
          done += int(self._refresh_history(fsym, fill_f, now, prev))
      except Exception as exc:  # belt and braces: each step is already total
        logger.warning("CARRY CLOCK: refresh for %s failed (%s)", fsym, exc)
    if isinstance(targets, dict):
      with self._lock:   # forget fills no longer held (bounded state)
        for key in [k for k in self._paid if k not in held_keys]:
          self._paid.pop(key, None)
        for key in [k for k in self._hist_failed_at if k not in held_keys]:
          self._hist_failed_at.pop(key, None)
    return done

  def _warn_once(self, key: Any, message: str, *args: Any) -> None:
    with self._lock:
      first = key not in self._warned
      self._warned.add(key)
    if first:
      logger.warning(message, *args)

  def _refresh_clock(self, fsym: str, now: float) -> bool:
    with self._lock:
      cached = self._cache.get(fsym)
      failed_at = self._failed_at.get(fsym)
    if cached is not None:
      fetched_at, next_ts, _interval = cached
      if now < next_ts and (now - fetched_at) <= self.REFRESH_SEC:
        return False
    if failed_at is not None and now - failed_at < self.RETRY_SEC:
      return False
    try:
      if self._client is None:
        raise RuntimeError("no futures client")
      payload = self._client.get_funding_rate(fsym)
      clock = funding_clock_from_rate(payload)
      if clock is None:
        raise ValueError("payload has no usable fundingTime/granularity")
    except Exception as exc:
      with self._lock:
        self._failed_at[fsym] = now
      self._warn_once(
        fsym, "CARRY CLOCK: funding clock for %s unavailable (%s) — carry hold uses %s; retried every %dmin",
        fsym, exc,
        "the last cached clock" if cached is not None else "the clock stamped at entry, else the 8h grid",
        self.RETRY_SEC // 60,
      )
      return False
    try:
      rate = float((payload or {}).get("value"))
    except (TypeError, ValueError, AttributeError):
      rate = None
    with self._lock:
      self._cache[fsym] = (now, clock[0], clock[1])
      if rate is not None and math.isfinite(rate):
        self._rates[fsym] = rate
      self._failed_at.pop(fsym, None)
      self._warned.discard(fsym)
    return True

  def _refresh_history(self, fsym: str, fill: float, now: float, prev: Optional[tuple]) -> bool:
    key = (fsym, fill)
    with self._lock:
      known = self._paid.get(key)
      failed_at = self._hist_failed_at.get(key)
      clock = self._cache.get(fsym)
    if known is not None and known[1] is not None:
      return False                               # the first payment is on record: permanent
    if failed_at is not None and now - failed_at < self.RETRY_SEC:
      return False
    if known is not None:
      checked_at = known[0]
      settled_since = False
      if clock is not None:
        nxt = first_settlement_after(checked_at, clock[1], clock[2])
        settled_since = nxt is not None and nxt <= now
      if not settled_since and now - checked_at <= self.REFRESH_SEC:
        return False
    try:
      if self._client is None:
        raise RuntimeError("no futures client")
      rows = self._client.get_funding_rate_history(
        fsym, start_at=int(fill * 1000), end_at=int((now + 60) * 1000))
      usable, first = first_settlement_in_history(rows, fill)
      if not usable:
        raise ValueError("history payload is not a list")
    except Exception as exc:
      with self._lock:
        self._hist_failed_at[key] = now
      self._warn_once(("hist", fsym), "CARRY HISTORY: funding history for %s unavailable (%s) — the carry "
                      "hold judges the first payment from the clocks; retried every %dmin",
                      fsym, exc, self.RETRY_SEC // 60)
      return False
    if first is None and prev is not None and fill < prev[1] <= now:
      # The clock we held promised a settlement that has passed; the history just has not listed it.
      first = prev[1]
    with self._lock:
      self._paid[key] = (now, first)
      self._hist_failed_at.pop(key, None)
      self._warned.discard(("hist", fsym))
    return True

  # ── cache only: safe under order_lock ──

  def __call__(self, fsym: str) -> Optional[tuple[float, float]]:
    """The last clock ``refresh`` read for ``fsym``, or None. Cache only — never a network call."""
    with self._lock:
      cached = self._cache.get(fsym)
    return (cached[1], cached[2]) if cached is not None else None

  def settlement_since(self, fsym: str, fill_ts: Any) -> tuple[Optional[float], Optional[float]]:
    """``(first_paid_ts, unpaid_as_of_ts)`` for the carry filled at ``fill_ts`` — see
    regime.carry_hold_deadline. ``(None, None)`` when the history is unknown. Cache only."""
    try:
      key = (fsym, float(fill_ts))
    except (TypeError, ValueError):
      return None, None
    with self._lock:
      known = self._paid.get(key)
    if known is None:
      return None, None
    checked_at, first = known
    return (first, None) if first is not None else (None, checked_at)

  def rate(self, fsym: str) -> Optional[float]:
    """The per-settlement funding rate read with the last successful clock fetch, or None.

    Cache only — never a network call — so the agent's entry thesis can show the carry's CURRENT rate
    next to the one stamped at entry for free (the clock is refreshed for held carry positions anyway).
    """
    with self._lock:
      return self._rates.get(fsym)


def _futures_candles_for_analytics(rows: Any) -> list:
  """FUTURES kline rows ([ts_ms, open, HIGH, LOW, close, vol, turnover]) -> analytics column order
  ([ts_s, open, close, high, low, vol, turnover]). Every row must prove its column order first
  (protection._replay_bar: high is the row max, low the row min) or ValueError — a replay once read
  futures candles in spot order and reversed a finding."""
  out = []
  for row in rows or []:
    ts, o, h, l, c = _validated_futures_bar(row)
    vol = row[5] if len(row) > 5 else 0
    turnover = row[6] if len(row) > 6 else 0
    out.append([ts, o, c, h, l, vol, turnover])
  return out


class _MarketStateClock:
  """The market-state block (analytics.market_state), refreshed at most hourly BY THE POLL LOOP.

  ``refresh()`` is the only method that touches the network — about three public calls an hour
  (/contracts/active, then XBTUSDTM 1D and 1h klines) — and only the poll loop calls it, after the
  profit-lock and probe settlement. ``current()`` is cache-only, so the order path, the agent run and
  the exit-probe recorder read it for free and nothing trading-critical ever waits on it.

  Total by construction: a failed refresh keeps the last reading, logs WARNING once per failure streak,
  and is retried after ``RETRY_SEC``; a BTC-candle failure only drops the BTC fields. A reading older
  than two refresh periods is withheld (``current`` -> None) — a stale state stamped as current would
  be worse than a missing one, which the splits count as untagged.
  """

  REFRESH_SEC = 3600
  RETRY_SEC = 15 * 60

  def __init__(self, kucoin_futures, *, min_turnover: float, min_age_days: float,
               time_fn: Callable[[], float] = time.time) -> None:
    self._client = kucoin_futures
    self._min_turnover = float(min_turnover or 0.0)
    self._min_age_days = float(min_age_days or 0.0)
    self._time = time_fn
    self._value: Optional[Dict[str, Any]] = None
    self._fetched_at = 0.0
    self._last_attempt = 0.0
    self._failing = False
    self._lock = threading.Lock()

  def refresh(self) -> bool:
    """Refresh when due; True when a new reading was stored. Never raises."""
    now = float(self._time())
    with self._lock:
      fresh = self._value is not None and now - self._fetched_at < self.REFRESH_SEC
      backing_off = self._failing and now - self._last_attempt < self.RETRY_SEC
      if fresh or backing_off or self._client is None:
        return False
      self._last_attempt = now
    try:
      contracts = self._client.list_active_contracts()
      if not contracts:
        raise ValueError("empty contract list")
    except Exception as exc:
      with self._lock:
        first = not self._failing
        self._failing = True
      if first:
        logger.warning("MARKET STATE: contract list unavailable (%s) — %s", exc,
                       "keeping the last reading" if self._value is not None else "entries record none")
      return False
    daily = hourly = None
    try:
      daily = _futures_candles_for_analytics(self._client.get_candles(
        "XBTUSDTM", granularity=1440, start_at=int((now - 60 * 86400) * 1000), end_at=int(now * 1000)))
    except Exception as exc:
      logger.warning("MARKET STATE: BTC daily candles unavailable (%s) — btcDailyAdx/bias omitted", exc)
    try:
      hourly = _futures_candles_for_analytics(self._client.get_candles(
        "XBTUSDTM", granularity=60, start_at=int((now - 80 * 3600) * 1000), end_at=int(now * 1000)))
    except Exception as exc:
      logger.warning("MARKET STATE: BTC hourly candles unavailable (%s) — btc72h omitted", exc)
    try:
      state = sanitize_market_state(market_state(
        contracts, now, min_turnover=self._min_turnover, min_age_days=self._min_age_days,
        btc_daily_candles=daily, btc_hourly_candles=hourly,
      ))
    except Exception as exc:
      state = None
      logger.warning("MARKET STATE: computation failed (%s)", exc)
    if not state:
      with self._lock:
        first = not self._failing
        self._failing = True
      if first:
        logger.warning("MARKET STATE: no usable reading from %d contracts — entries record none", len(contracts))
      return False
    with self._lock:
      self._value, self._fetched_at, self._failing = state, now, False
    logger.info("MARKET STATE: breadth24=%s basketMedian24h=%s btc24h=%s btc72h=%s btcDailyAdx=%s (%s)",
                state.get("breadth24"), state.get("basketMedian24h"), state.get("btc24h"),
                state.get("btc72h"), state.get("btcDailyAdx"), state.get("btcDailyBias"))
    return True

  def current(self) -> Optional[Dict[str, Any]]:
    """The latest reading (a copy), or None when there is none or it is older than two refresh periods."""
    now = float(self._time())
    with self._lock:
      if self._value is None or now - self._fetched_at > 2 * self.REFRESH_SEC:
        return None
      return dict(self._value)


def _make_trade_context_lookup(memory, kucoin_futures, *, time_fn: Callable[[], float] = time.time):
  """ProtectionManager's ``trade_context_lookup``: position_context.trade_context on the live clock.

  A thin wrapper so the loop and the tests build the SAME callable — the carry hold must see the
  contract's own funding clock, and a wiring slip here (not in the pure helpers) is exactly the kind
  of bug a helper-only test misses.
  """
  # The clock's refresh runs on the survival thread (before the protection pass), so it reads with the
  # short measurement timeout — never the 15s an order call is allowed.
  clock = _FundingClock(_measurement_client(kucoin_futures), time_fn=time_fn)

  def _trade_context(fsym: str, pos: Any) -> Dict[str, Any]:
    return trade_context(memory, fsym, pos, time_fn(), funding_clock=clock)

  _trade_context.funding_clock = clock  # type: ignore[attr-defined]  # exposed for inspection/tests
  return _trade_context


def _futures_position_fingerprint(positions: list[dict] | None) -> tuple[tuple[str, float, float], ...]:
  """Stable lifecycle fingerprint used to flag narratives built from an older position book."""
  rows: list[tuple[str, float, float]] = []
  for position in positions or []:
    if not isinstance(position, dict):
      continue
    try:
      qty = float(position.get("currentQty") or 0.0)
      entry = float(position.get("avgEntryPrice") or 0.0)
    except (TypeError, ValueError):
      continue
    if not qty:
      continue
    symbol = normalize_symbol(position.get("symbol") or "")
    if symbol:
      rows.append((symbol, round(qty, 12), round(entry, 12)))
  return tuple(sorted(rows))


def _agent_made_a_move(result: Dict) -> bool:
  """True if the agent placed any order this run (entry, close, or protective stop).

  Used to detect the 'stuck declining' state: when the agent makes no move for several
  consecutive runs, the loop forces a Research Agent handoff to refresh the coin universe.
  Pure declines/rejections and bare fund transfers do NOT count as a move.
  """
  for out in result.get("tool_results") or []:
    if not isinstance(out, dict):
      continue
    if out.get("rejected") or out.get("skipped") or out.get("error"):
      continue
    if out.get("cancelled"):
      continue
    if out.get("orderId") or out.get("orderRequest"):
      return True
  return False


def _expired_bot_entry_orders(pending_orders: list[dict], expiry_minutes: float, now: float | None = None) -> list[dict]:
  """Select expired trAIde-created GTC futures entries. Pure/testable.

  KuCoin has no client-side "expire in N minutes" behavior for the submitted GTC orders.  Only
  orders tagged by this bot are eligible, so autonomous cleanup never cancels a manual/user order.
  Protective/reduce-only orders are excluded even if an API happens to return them in this list.
  """
  if expiry_minutes <= 0:
    return []
  ref = float(now if now is not None else time.time())

  def _true(value: Any) -> bool:
    return value is True or str(value or "").strip().lower() in {"1", "true", "yes"}

  expired: list[dict] = []
  for order in pending_orders or []:
    if not isinstance(order, dict):
      continue
    if not str(order.get("clientOid") or "").startswith("traide-entry-"):
      continue
    if _true(order.get("reduceOnly")) or _true(order.get("closeOrder")):
      continue
    raw_ts = order.get("createdAt") or order.get("orderTime") or order.get("created_at")
    try:
      created = float(raw_ts)
    except (TypeError, ValueError):
      continue
    if created > 1e15:  # nanoseconds
      created /= 1e9
    elif created > 1e12:  # milliseconds
      created /= 1e3
    if ref - created >= expiry_minutes * 60:
      expired.append(order)
  return expired


def build_snapshot(cfg, kucoin: KucoinClient, kucoin_futures: KucoinFuturesClient | None, memory: MemoryStore) -> TradingSnapshot:
  raw_coins = _load_active_coins(cfg, memory)
  futures_overview, futures_positions, futures_stops = _fetch_futures(cfg, kucoin_futures)
  live_futures_symbols: set[str] = set()
  for position in futures_positions:
    if not isinstance(position, dict):
      continue
    try:
      qty = float(position.get("currentQty") or 0)
    except (TypeError, ValueError):
      continue
    symbol = normalize_symbol(position.get("symbol") or "")
    if qty and symbol:
      live_futures_symbols.add(symbol)
  coins, tickers = _fetch_tickers(
    cfg, kucoin, raw_coins, memory,
    preserve_symbols=live_futures_symbols,
  )
  for symbol in sorted(live_futures_symbols):
    if symbol in tickers:
      continue
    try:
      tickers[symbol] = kucoin.get_ticker(symbol)
      coins.append(symbol)
    except Exception as exc:
      live_position = next(
        (position for position in futures_positions if normalize_symbol(position.get("symbol") or "") == symbol),
        {},
      )
      mark = float(live_position.get("markPrice") or 0) if isinstance(live_position, dict) else 0.0
      if mark > 0:
        mark_str = str(mark)
        tickers[symbol] = KucoinTicker(
          sequence="", bestAsk=mark_str, size="0", price=mark_str,
          bestBidSize="0", bestBid=mark_str, bestAskSize="0", time=kucoin._timestamp_ms(),
        )
        coins.append(symbol)
        logger.warning("Using futures mark for manageable live symbol %s because spot ticker failed: %s", symbol, exc)
      else:
        logger.warning("Live futures symbol %s has no spot ticker; management remains enabled: %s", symbol, exc)
  spot_accounts, financial_accounts, all_accounts = _fetch_balances(kucoin)

  # Discover spot holdings not in the active coin list (e.g., externally bought coins).
  extra_coins, extra_tickers = _discover_unlisted_holdings(kucoin, spot_accounts, tickers)
  coins.extend(extra_coins)
  tickers.update(extra_tickers)

  spot_stops: list[dict] = []
  try:
    spot_stops = kucoin.list_stop_orders(status="active")
  except Exception as exc:
    logger.warning("Unable to fetch spot stop orders: %s", exc)

  spot_pending: list[dict] = []
  try:
    spot_pending = kucoin.list_orders(status="active")
  except Exception as exc:
    logger.warning("Unable to fetch spot pending orders: %s", exc)

  futures_pending: list[dict] = []
  if cfg.kucoin_futures.enabled and kucoin_futures:
    try:
      futures_pending = kucoin_futures.list_orders(status="active")
    except Exception as exc:
      logger.warning("Unable to fetch futures pending orders: %s", exc)

  fees = _fetch_fees(kucoin)

  balances = list(spot_accounts)
  for fa in financial_accounts:
    balances.append(fa)

  if futures_overview:
    balances.append(
      KucoinAccount(
        id="futures",
        currency=str(futures_overview.get("currency", "USDT")),
        type="contract",
        balance=str(futures_overview.get("accountEquity") or futures_overview.get("marginBalance") or futures_overview.get("availableBalance") or "0"),
        available=str(futures_overview.get("availableBalance") or "0"),
        holds=str(futures_overview.get("frozenBalance") or "0"),
      )
    )

  return TradingSnapshot(
    coins=coins,
    tickers=tickers,
    balances=balances,
    paper_trading=cfg.trading.paper_trading,
    max_position_usd=cfg.trading.max_position_usd,
    min_confidence=cfg.trading.min_confidence,
    max_leverage=cfg.trading.max_leverage,
    futures_enabled=cfg.kucoin_futures.enabled,
    drawdown_pct=0.0,
    drawdown_pct_spot=0.0,
    drawdown_pct_futures=0.0,
    total_usdt=0.0,
    spot_accounts=spot_accounts,
    futures_account=futures_overview or {},
    futures_positions=futures_positions,
    all_accounts=all_accounts,
    spot_stop_orders=spot_stops,
    futures_stop_orders=futures_stops,
    spot_pending_orders=spot_pending,
    futures_pending_orders=futures_pending,
    financial_accounts=financial_accounts,
    fees=fees,
  )


async def trading_loop(
  stop_event: asyncio.Event | None = None,
  safety: TradingSafetyState | None = None,
) -> None:
  cfg = load_config()
  stop_event = stop_event or asyncio.Event()
  safety = safety or TradingSafetyState()
  setup_tracing(cfg)
  ls_client = setup_lstracing(cfg)
  azure_client = _build_openai_client(cfg)
  # Azure client is for inference only; tracing uses platform key via set_tracing_export_api_key in setup_tracing.
  set_default_openai_client(azure_client, use_for_tracing=False)
  kucoin = KucoinClient(cfg)
  kucoin_futures = KucoinFuturesClient(cfg) if cfg.kucoin_futures.enabled else None
  memory = MemoryStore(cfg.memory_file, retention_days=cfg.retention_days)
  scheduler_state = memory.get_agent_scheduler()
  now_ts = time.time()
  try:
    last_agent_run_ts = min(now_ts, max(0.0, float(scheduler_state.get("lastRunTs") or 0.0)))
  except (TypeError, ValueError):
    last_agent_run_ts = 0.0
  unproductive_runs = min(100, max(0, int(scheduler_state.get("unproductiveRuns") or 0)))
  reviewed_prices: Dict[str, float] = dict(scheduler_state.get("reviewedPrices") or {})
  price_observations: Dict[str, Dict[str, Any]] = dict(scheduler_state.get("priceObservations") or {})
  flow_observations: Dict[str, Dict[str, Any]] = dict(scheduler_state.get("flowObservations") or {})
  last_prices: Dict[str, float] = {
    symbol: float(observation["lastPrice"])
    for symbol, observation in price_observations.items()
    if isinstance(observation, dict) and float(observation.get("lastPrice") or 0) > 0
  }

  def _persist_agent_scheduler() -> None:
    memory.save_agent_scheduler({
      "lastRunTs": last_agent_run_ts,
      "unproductiveRuns": unproductive_runs,
      "reviewedPrices": reviewed_prices,
      "priceObservations": price_observations,
      "flowObservations": flow_observations,
    })

  idle_polls = 0
  consecutive_no_trade_runs = 0  # runs where the agent placed no order (drives forced research)
  force_research = False         # when True, next agent run must hand off to Research first
  last_forced_research_ts = 0.0  # wall-clock of the last forced research handoff (cooldown gate)
  # Calendar-only research is decided by CODE and, unlike the whole-market overhaul, is allowed while
  # positions are open: it is one web lookup and one tool call, and it never touches the coin list, so
  # neither harm the flat-only rule guards against (long protection blind spots, add-on temptation)
  # applies. Retried at most every _CALENDAR_RETRY_SEC if a run fails to refresh — roughly four
  # attempts a day, so a model that keeps missing it cannot turn this into a cost leak.
  last_calendar_attempt_ts = 0.0
  agent_task_calendar_refresh = False
  pending_trigger_moves: Dict[str, float] = {}  # strongest move per symbol, retained until reviewed
  pending_agent_triggers: set[str] = set()
  agent_task: asyncio.Task | None = None
  agent_task_triggers: list[str] = []
  agent_task_forced_research = False
  agent_task_token: str | None = None
  agent_task_event_ids: list[str] = []
  agent_task_started_ts = 0.0
  agent_task_timed_out = False
  agent_task_trigger_moves: Dict[str, float] = {}
  agent_task_discrete_triggers: set[str] = set()
  agent_task_prices: Dict[str, float] = {}
  agent_task_position_fingerprint: tuple[tuple[str, float, float], ...] = ()
  last_complete_fill_poll_ts = 0.0
  # Seed from the persisted set: a restart inside the 30-min fill lookback used to re-detect and
  # double-record the same close (an in-memory-only set), corrupting realized-PnL stats.
  logged_closed_position_ids: set[str] = set(memory.get_seen_close_ids())
  logged_fill_ids: set[str] = set(memory.get_seen_fill_ids())
  for pending_event in memory.get_pending_agent_events():
    if pending_event.get("kind") in {"spot_fills", "futures_fills"}:
      logged_fill_ids.add(str(pending_event.get("id") or ""))
  notifier = TelegramNotifier(cfg)
  notifier.notify_startup(cfg)
  dashboard = DashboardPublisher(cfg)
  if dashboard.enabled:
    logger.info("Public dashboard publishing enabled (disclosure=%s, table=%s, container=%s).",
                cfg.dashboard.disclosure, cfg.dashboard.table_name, cfg.dashboard.container_name)
  elif cfg.dashboard.enabled:
    logger.warning("Dashboard publishing requested but DISABLED: %s", dashboard.disabled_reason())

  # Facts about the trade behind each open position (carry hold on the contract's own funding clock,
  # noise band, original risk, recorded peak) — see position_context.trade_context. Built by a module
  # factory so the exact lookup the live manager receives is executable in tests.
  _trade_context = _make_trade_context_lookup(memory, kucoin_futures)
  _stack_attempts: Dict[tuple, float] = {}   # exit-probe stack replays: (symbol, ts) -> last attempt
  _gate_log_last: Dict[str, float] = {}      # hourly GATE SCOREBOARD log line
  # Hourly market-state reading (breadth over the screener's own universe, BTC 24h/72h, BTC daily ADX),
  # stamped on entries and probes so verdicts can be split by market. Refreshed only by this loop.
  # Measurement reads on this thread (probe settlement, stack replays, market state) use a short HTTP
  # timeout, a per-poll wall-clock budget and a cross-poll backoff — see _MeasurementBudget.
  _measure_client = _measurement_client(kucoin_futures)
  _measure_backoff = _MeasureBackoff(float(cfg.trading.poll_interval_sec or 60))
  _market_state = _MarketStateClock(
    _measure_client,
    min_turnover=cfg.trading.screener_min_turnover_usd_24h,
    min_age_days=cfg.trading.min_futures_listing_age_days,
  )

  protection = ProtectionManager(
    cfg.profit_protection, kucoin_futures, notifier=notifier,
    trade_context_lookup=_trade_context,
    emergency_sl_pct=cfg.trading.emergency_sl_pct, min_rr=cfg.trading.min_futures_rr,
    max_loss_equity_fraction=cfg.trading.risk_per_trade_pct,
    breakeven_cost_pct=2.0 * (
      float((memory.latest_fees() or {}).get("futures_taker") or 0.0006)
      + float(cfg.trading.estimated_slippage_pct or 0.0)
    ),
  )
  if cfg.profit_protection.enabled:
    logger.info(
      "Profit-lock enabled (dry_run=%s, breakeven@%.1fR, giveback=%.0f%%, no_chase=%s/%.0fmin).",
      protection.cfg.dry_run, protection.cfg.breakeven_trigger_r,
      protection.cfg.giveback_pct * 100, protection.cfg.no_chase_enabled,
      cfg.profit_protection.post_win_cooldown_minutes,
    )

  logger.info("Starting trading loop...")
  while not stop_event.is_set():
    try:
      snapshot = build_snapshot(cfg, kucoin, kucoin_futures, memory)
    except Exception as exc:
      safety.invalidate(f"Snapshot incomplete: {exc}", revoke_active=True)
      logger.error("Snapshot failed, retrying after delay: %s", exc)
      notifier.notify_error(str(exc), context="Snapshot build")
      # If positions and stops were both fetched, keep the deterministic safety loop alive even
      # though entries stay disabled because some other part of exchange truth is incomplete.
      if isinstance(exc, IncompleteFuturesSnapshot) and kucoin_futures:
        failed_names = {failure.split(":", 1)[0] for failure in exc.failures}
        if "positions" not in failed_names and "stops" not in failed_names:
          try:
            degraded = type("DegradedSnapshot", (), {
              "futures_enabled": True,
              "futures_positions": exc.positions,
              "futures_stop_orders": exc.stops,
              "futures_account": exc.overview or {},
              "total_usdt": 0.0,
            })()
            with safety.order_lock:
              actions = protection.run(degraded)
            if actions:
              logger.warning("Protection actions taken from degraded snapshot: %s", actions)
          except Exception as protection_exc:
            logger.warning("Degraded protection failed: %s", protection_exc)
      try:
        await asyncio.wait_for(stop_event.wait(), timeout=cfg.trading.poll_interval_sec)
      except asyncio.TimeoutError:
        pass
      continue
    if stop_event.is_set():
      break

    # GTC entry orders do not expire just because the tool response contains an expiresAt note.
    # Enforce the configured TTL on bot-tagged orders before the agent sees the book, preventing a
    # stale thesis from filling hours later.  Manual and protective orders are never touched.
    if kucoin_futures and snapshot.futures_pending_orders:
      expired_orders = _expired_bot_entry_orders(
        snapshot.futures_pending_orders,
        cfg.trading.entry_limit_expiry_minutes,
      )
      cancelled_ids: set[str] = set()
      for order in expired_orders:
        oid = str(order.get("id") or order.get("orderId") or "")
        if not oid:
          continue
        try:
          with safety.order_lock:
            kucoin_futures.cancel_order(oid, symbol=order.get("symbol"))
          cancelled_ids.add(oid)
          logger.info("Expired stale futures entry %s (%s) after %.0fmin", oid, order.get("symbol"), cfg.trading.entry_limit_expiry_minutes)
          # Surface the expiry to the agent's next run. Without this, unfilled expiries were only
          # logged, never recorded — so the agent kept re-placing a pullback limit on a running
          # trend, blind to the fact its last attempts never filled (the ONDO churn). Seeing repeated
          # entryExpiries on one symbol is the signal to take a continuation entry or stand down.
          try:
            fsym = str(order.get("symbol") or "")
            memory.queue_agent_event("entry_expired", oid, {
              "symbol": normalize_symbol(fsym),
              "futuresSymbol": fsym,
              "side": order.get("side"),
              "price": order.get("price"),
              "reason": f"unfilled limit entry expired after {cfg.trading.entry_limit_expiry_minutes:.0f}min",
            })
          except Exception:
            pass
        except Exception as exc:
          logger.warning("Unable to expire stale futures entry %s: %s", oid, exc)
      if cancelled_ids:
        # Cancellation can race a partial fill and can alter attached child protection. Refresh
        # authoritative positions/stops before the protection pass; never reason from pre-cancel truth.
        try:
          overview, positions, stops = _fetch_futures(cfg, kucoin_futures)
          snapshot.futures_account = overview or {}
          snapshot.futures_positions = positions
          snapshot.futures_stop_orders = stops
          snapshot.futures_pending_orders = kucoin_futures.list_orders(status="active") or []
        except Exception as exc:
          safety.invalidate(f"Post-cancel futures reconciliation failed: {exc}", revoke_active=True)
          logger.error("Post-cancel futures reconciliation failed; new entries disabled: %s", exc)

    spot_usdt = 0.0
    for acct in snapshot.spot_accounts:
      if acct.currency == "USDT":
        try:
          avail = float(acct.available or 0)
          bal = float(acct.balance or 0)
          spot_usdt += max(avail, bal)
        except Exception:
          continue

    futures_usdt = 0.0
    if cfg.kucoin_futures.enabled and snapshot.futures_account:
      try:
        fut_avail = float(snapshot.futures_account.get("availableBalance") or 0)
        fut_equity = float(snapshot.futures_account.get("accountEquity") or snapshot.futures_account.get("marginBalance") or fut_avail)
        futures_usdt += max(fut_avail, fut_equity)
      except Exception:
        pass

    financial_usdt = 0.0
    for acct in snapshot.financial_accounts:
      if acct.currency == "USDT":
        try:
          avail = float(acct.available or 0)
          bal = float(acct.balance or 0)
          financial_usdt += max(avail, bal)
        except Exception:
          continue

    total_usdt = spot_usdt + futures_usdt + financial_usdt

    # Is this a COMPLETE view of the account? total_usdt is a sum across venues, and each term
    # silently contributes 0.0 when its read fails — a KuCoin 504 on the futures overview drops the
    # entire futures balance out of "total equity" without raising anything. That number then drives
    # three things that must never see a partial account: the drawdown circuit breaker, the daily
    # baseline, and the compounding equity index.
    #
    # Measured live: the durable index holds ~7.27e7 on days 20693-20695 and a fabricated +19.46%
    # step across 2026-08-15 -> 08-31. There were NO deposits or withdrawals in that window — only
    # internal spot<->futures transfers, which cancel in this sum by construction. A partial snapshot
    # is the only thing that moves the total without money entering or leaving.
    #
    # Money in the account is not lost because an HTTP call was, so treat an unreadable venue as
    # "unknown", not as "zero", and decline to record equity from it at all.
    equity_complete = True
    equity_gap: str | None = None
    if cfg.kucoin_futures.enabled and not snapshot.futures_account:
      equity_complete = False
      equity_gap = "futures account overview unavailable"
    elif cfg.kucoin_futures.enabled and futures_usdt <= 0 and spot_usdt <= 0 and financial_usdt <= 0:
      equity_complete = False
      equity_gap = "every venue reported zero — treating as an unread account, not an empty one"
    snapshot.equity_complete = equity_complete

    # Track daily drawdown per venue; the circuit-breaker section below converts the total scope
    # into a hard close-only restriction when its configured limit is breached.
    if equity_complete:
      safety.refresh(total_usdt)
      limits_total = memory.update_limits(total_usdt, scope="total")
      limits_spot = memory.update_limits(spot_usdt, scope="spot")
      limits_futures = memory.update_limits(futures_usdt, scope="futures")
    else:
      # Carry the last known-good limits. Recording this poll would either trip the drawdown breaker
      # on a phantom loss or, on the next successful read, book the recovery as a phantom gain — and
      # the equity index compounds, so that second one is permanent.
      logger.warning(
        "EQUITY SNAPSHOT INCOMPLETE (%s) — spot=%.4f futures=%.4f financial=%.4f. Holding the last "
        "known limits and skipping this poll's equity point; a missing venue is unknown, not zero.",
        equity_gap, spot_usdt, futures_usdt, financial_usdt,
      )
      limits_total = memory.latest_limits("total") or {}
      limits_spot = memory.latest_limits("spot") or {}
      limits_futures = memory.latest_limits("futures") or {}
      total_usdt = float(limits_total.get("currentUsdt") or total_usdt)

    snapshot.total_usdt = total_usdt
    snapshot.drawdown_pct = float(limits_total.get("drawdownPct") or 0.0)
    snapshot.drawdown_pct_spot = float(limits_spot.get("drawdownPct") or 0.0)
    snapshot.drawdown_pct_futures = float(limits_futures.get("drawdownPct") or 0.0)

    if snapshot.drawdown_pct > 0:
      logger.info("Daily drawdown: total=%.2f%% spot=%.2f%% futures=%.2f%%",
                  snapshot.drawdown_pct, snapshot.drawdown_pct_spot, snapshot.drawdown_pct_futures)

    if (
      agent_task is not None
      and not agent_task.done()
      and not agent_task_timed_out
      and agent_task_started_ts > 0
      and time.time() - agent_task_started_ts >= _AGENT_RUN_TIMEOUT_SEC
    ):
      agent_task_timed_out = True
      safety.revoke_run(agent_task_token, "Background agent exceeded its 20-minute authority window")
      logger.error("Background agent exceeded %dsec; all later exchange writes from that run are disabled.", _AGENT_RUN_TIMEOUT_SEC)
      notifier.notify_error("Model run exceeded 20 minutes; entry/order authority revoked.", context="Agent timeout")
      # Cancelling the asyncio wrapper cannot stop a thread stuck in blocking SDK code. Detach it
      # after revoking its unique token so it cannot touch the exchange, retain its inbox events,
      # and allow a fresh single-flight run after the normal active cooldown.
      agent_task.cancel()
      for symbol, move in agent_task_trigger_moves.items():
        pending_trigger_moves[symbol] = max(move, pending_trigger_moves.get(symbol, 0.0))
      pending_agent_triggers.update(agent_task_discrete_triggers)
      last_agent_run_ts = time.time()
      _persist_agent_scheduler()
      agent_task = None
      agent_task_triggers = []
      agent_task_forced_research = False
      agent_task_token = None
      agent_task_event_ids = []
      agent_task_started_ts = 0.0
      agent_task_timed_out = False
      agent_task_trigger_moves = {}
      agent_task_discrete_triggers = set()
      agent_task_prices = {}
      agent_task_position_fingerprint = ()

    # Harvest a completed single-flight model run without ever pausing the snapshot/protection loop.
    if agent_task is not None and agent_task.done():
      last_agent_run_ts = time.time()  # enforce cadence from completion, not from a long run's start
      _persist_agent_scheduler()
      try:
        result = agent_task.result()
        current_position_fingerprint = _futures_position_fingerprint(snapshot.futures_positions)
        if current_position_fingerprint != agent_task_position_fingerprint:
          current_positions = ", ".join(
            f"{symbol} qty={qty:g} entry={entry:g}"
            for symbol, qty, entry in current_position_fingerprint
          ) or "none"
          result = dict(result)
          result["stateChangedDuringRun"] = True
          result["narrative"] = (
            "LIVE RECONCILIATION: the futures position book changed while this model run was in "
            f"progress. Current snapshot positions: {current_positions}. Treat any conflicting "
            "position statement below as historical.\n\n"
            + str(result.get("narrative") or "")
          )
        made_move = _agent_made_a_move(result)
        if agent_task_event_ids or snapshot.futures_positions:
          unproductive_runs = 0
        elif made_move:
          # A submitted order is useful progress but not proof of edge until it fills.
          unproductive_runs = max(0, unproductive_runs - 1)
        else:
          unproductive_runs = min(100, unproductive_runs + 1)
        # Anchor future triggers to the exact snapshot the successful run reviewed. A move during
        # the model call remains visible and can therefore schedule a genuinely new follow-up.
        reviewed_prices.update(agent_task_prices)
        # Polls during the run measured displacement from the previous anchor. Recalculate against
        # this run's snapshot below instead of carrying that stale magnitude into another call.
        _rebase_reviewed_price_triggers(
          pending_trigger_moves,
          pending_agent_triggers,
          set(agent_task_prices),
        )
        _persist_agent_scheduler()
        logger.info("--- Agent Decision Narrative ---\n%s", result.get("narrative", ""))
        if result.get("decisions"):
          logger.info("--- Decisions ---\n%s", "\n".join(f"- {d}" for d in result["decisions"]))
        notifier.notify_agent_run(agent_task_triggers or ["background_run"], result)

        acknowledged = memory.acknowledge_agent_events(agent_task_event_ids)
        for event in acknowledged:
          if event.get("kind") in {"spot_fills", "futures_fills"}:
            memory.record_seen_fill_id(str(event.get("id") or ""))

        threshold = cfg.trading.research_handoff_after_no_trade_runs
        research_cooldown_sec = cfg.trading.research_handoff_cooldown_min * 60
        if agent_task_calendar_refresh:
          last_calendar_attempt_ts = time.time()
          _cal_after = memory.macro_calendar_state()
          _still = macro_calendar_refresh_reason(_cal_after["events"], _cal_after["updated"], time.time())
          if _still:
            logger.warning("MACRO CALENDAR: refresh run finished but calendar is still %s; retrying in %.0fh.",
                           _still, _CALENDAR_RETRY_SEC / 3600)
          else:
            logger.info("MACRO CALENDAR: refreshed (%d events stored).", len(_cal_after["events"]))
        if agent_task_forced_research:
          consecutive_no_trade_runs = 0
          force_research = False
          last_forced_research_ts = time.time()
        elif made_move:
          consecutive_no_trade_runs = 0
        else:
          consecutive_no_trade_runs += 1
        current_book_active = bool(
          snapshot.futures_positions or snapshot.futures_pending_orders or snapshot.spot_pending_orders
        )
        # Whole-market research is deliberately flat-only. Running it while capital is exposed
        # caused 10–25 minute protection blind spots and encouraged unrelated add-ons.
        if threshold > 0 and consecutive_no_trade_runs >= threshold and not current_book_active:
          elapsed = time.time() - last_forced_research_ts
          if elapsed >= research_cooldown_sec:
            force_research = True
            logger.info("No trade for %d flat-book runs — forcing Research Agent handoff next run.", consecutive_no_trade_runs)
      except Exception as exc:
        # Log the TYPE and a traceback, not just str(exc): several exception classes stringify to the
        # empty string, and on 2026-08-08 three of the run failures logged as bare "Agent run failed:"
        # with nothing after it — undiagnosable from the log alone.
        logger.error("Agent run failed: %s: %s", type(exc).__name__, exc or "<no message>", exc_info=True)
        notifier.notify_error(f"{type(exc).__name__}: {exc}".strip(), context="Agent run")
        for symbol, move in agent_task_trigger_moves.items():
          pending_trigger_moves[symbol] = max(move, pending_trigger_moves.get(symbol, 0.0))
        pending_agent_triggers.update(agent_task_discrete_triggers)
      finally:
        safety.finish_run(agent_task_token)
        agent_task = None
        agent_task_triggers = []
        agent_task_forced_research = False
        agent_task_token = None
        agent_task_event_ids = []
        agent_task_started_ts = 0.0
        agent_task_timed_out = False
        agent_task_trigger_moves = {}
        agent_task_discrete_triggers = set()
        agent_task_prices = {}
        agent_task_position_fingerprint = ()

    # A symbol we already hold has its own definition of noise: the band its stop was floored to.
    # Waking the model to re-decide inside that band is what produced a 13-minute median hold on
    # theses that need hours (see regime.held_position_noise_pct). Computed once per poll.
    held_noise_pct: Dict[str, float] = {}
    for _p in getattr(snapshot, "futures_positions", None) or []:
      if not isinstance(_p, dict):
        continue
      try:
        _qty = float(_p.get("currentQty") or 0)
      except (TypeError, ValueError):
        continue
      if not _qty:
        continue
      _sym = normalize_symbol(_p.get("symbol") or "")
      if not _sym:
        continue
      try:
        _ctx = memory.entry_context_for_position(
          _p.get("symbol") or "",
          next((_p.get(k) for k in ("openingTimestamp", "openingTime", "openTime", "createdAt")
                if _p.get(k) not in (None, "")), None),
          "long" if _qty > 0 else "short",
        )
        _band = held_position_noise_pct(_ctx)
      except Exception:
        logger.debug("held-noise lookup failed for %s", _sym, exc_info=True)
        _band = None
      if _band:
        held_noise_pct[_sym] = _band

    triggers: list[str] = []
    for symbol, ticker in snapshot.tickers.items():
      price = float(ticker.price)
      prev = last_prices.get(symbol)
      observation = price_observations.get(symbol) or {}
      noise_ewma = float(observation.get("noiseEwmaPct") or 0.0)
      samples = int(observation.get("samples") or 0)
      reviewed_price = reviewed_prices.get(symbol)
      if reviewed_price is None:
        initial_trigger = f"initial:{symbol}"
        triggers.append(initial_trigger)
        pending_agent_triggers.add(initial_trigger)
      elif reviewed_price > 0:
        # Compare with the last successful model-reviewed state, not the immediately preceding
        # poll. The old per-poll comparison repeatedly called the model on ordinary oscillation.
        state_move_pct = abs(price - reviewed_price) / reviewed_price * 100
        adaptive_threshold = _adaptive_price_trigger_threshold(
          cfg.trading.price_change_trigger_pct,
          noise_ewma,
          samples,
          noise_multiplier=cfg.trading.price_noise_multiplier,
          max_multiplier=cfg.trading.price_trigger_max_multiplier,
        )
        held_band = held_noise_pct.get(symbol)
        if held_band:
          # Never LOWER the threshold — only refuse to treat sub-noise drift on an open trade as news.
          adaptive_threshold = max(adaptive_threshold, held_band)
        if state_move_pct >= adaptive_threshold:
          triggers.append(f"price_move:{symbol}:{state_move_pct:.2f}%")
          pending_trigger_moves[symbol] = max(
            state_move_pct,
            pending_trigger_moves.get(symbol, 0.0),
          )

      if prev is not None and prev > 0:
        poll_move_pct = abs(price - prev) / prev * 100
        noise_ewma = _next_price_noise_ewma(
          noise_ewma,
          poll_move_pct,
          cfg.trading.price_change_trigger_pct,
          samples,
          max_multiplier=cfg.trading.price_trigger_max_multiplier,
        )
        samples = min(samples + 1, 1_000_000)
      price_observations[symbol] = {
        "lastPrice": price,
        "noiseEwmaPct": noise_ewma,
        "samples": samples,
        "updated": int(time.time()),
      }
      last_prices[symbol] = price

    # Continuous taker-flow record. The bot has always analysed CLOSED CANDLES at 15m and above, so
    # it could see what price did but never who was pushing it — KuCoin klines carry no taker split,
    # and this public tape is the only endpoint that does. Sampled every poll into persisted state so
    # a reading exists at the moment of a direction call rather than being fetched inside the hot
    # path. Purely observational: nothing in the trading path reads it yet, by design — see
    # edge.taker_flow_edge_stats for the question it is being collected to answer.
    if cfg.edge.taker_flow_enabled and kucoin_futures is not None:
      for symbol in _flow_symbols(snapshot, cfg.edge.taker_flow_max_symbols, pending_trigger_moves):
        futures_symbol = _to_futures_symbol(symbol)
        if not futures_symbol:
          continue
        try:
          previous = flow_observations.get(symbol) or {}
          summary = taker_flow_summary(
            kucoin_futures.get_trade_history(futures_symbol),
            since_cursor=previous.get("lastCursor"),
          )
          flow_observations[symbol] = _next_flow_observation(previous, summary, int(time.time()))
        except Exception:
          # Market colour is never worth a poll. A failed tape read leaves the last state in place.
          logger.debug("taker-flow sample failed for %s", symbol, exc_info=True)
    # Outside the enabled check on purpose: switching collection off must still let the last
    # readings expire, or they sit in .agent_memory.json for good describing a market from whenever
    # sampling stopped.
    _dropped = _prune_flow_observations(
      flow_observations, cfg.trading.poll_interval_sec, time.time()
    )
    if _dropped:
      logger.debug("taker-flow: forgot %d stale symbol reading(s)", _dropped)

    # SPOT tickers: what the auto-triggers below compare against. NOT what probes settle on.
    live_prices = {
      normalize_symbol(symbol): float(ticker.price)
      for symbol, ticker in snapshot.tickers.items()
    }
    for stored_trigger, observed_price in _crossed_auto_triggers(memory.latest_triggers(), live_prices):
      symbol = normalize_symbol(stored_trigger.get("symbol") or "")
      condition = str(stored_trigger.get("condition") or "").lower()
      target = float(stored_trigger.get("targetPrice"))
      stable_id = str(stored_trigger.get("triggerId") or "").strip() or (
        f"{symbol}:{stored_trigger.get('ts')}:{condition}:{target}"
      )
      payload = {
        "trigger": stored_trigger,
        "observedPrice": observed_price,
        "crossedAt": int(time.time()),
      }
      memory.queue_agent_event("auto_triggers", f"auto:{stable_id}", payload)
      memory.consume_trigger(stored_trigger)
      event_label = f"auto_trigger:{symbol}:{condition}:{target:g}"
      triggers.append(event_label)
      pending_agent_triggers.add(event_label)
      logger.info(
        "Auto-trigger fired for %s: price %.8g is %s %.8g",
        symbol, observed_price, condition, target,
      )
    _persist_agent_scheduler()

    # Snapshot peak/trough BEFORE the update prunes just-closed positions — otherwise a position's
    # MFE/MAE is reset before we record its close, and every close lands with peak/trough = None
    # (which is why we can't tell whether TP targets were realistically reachable).
    pre_close_extremes = memory.get_position_extremes()
    try:
      # Drive extremes off live exchange positions so peak/trough reset when a position closes.
      memory.update_position_extremes(_live_extremes_map(snapshot))
    except Exception as exc:
      logger.warning("Failed to update position extremes: %s", exc)

    # The carry hold's funding clock (and settlement history since each carry's fill) is refreshed
    # HERE, before the protection pass and OUTSIDE order_lock: ProtectionManager's lookup under the lock
    # only reads its cache, so a hanging funding endpoint can never hold the lock or delay another
    # position's hard cap (2026-09-25 review). Held funding_carry positions only; backs off on failure.
    try:
      _trade_context.funding_clock.refresh(carry_refresh_targets(memory, snapshot.futures_positions))
    except Exception as exc:  # refresh is total; belt and braces
      logger.warning("CARRY CLOCK: refresh step failed (%s) — the hold uses the cached/stamped clock", exc)

    # Code-driven profit protection: ratchet stops to breakeven and cap give-back on
    # live futures positions every poll, independent of whether the agent runs. Never raises.
    try:
      with safety.order_lock:
        protection_actions = protection.run(snapshot)
      if protection_actions:
        logger.info("Profit-lock actions taken: %s", protection_actions)
    except Exception as exc:
      logger.warning("Profit-lock run failed: %s", exc)

    # Settle signal-edge probes: stamp the forward price on any entry signal whose measurement horizon
    # has elapsed, and resolve each recorded early close against the bracket it overrode. This is the
    # feedback loop that tells the bot whether its DIRECTION CALLS predict, as opposed to whether its
    # exits were lucky — see memory.settle_signal_probes / edge.signal_edge_stats. It settles on the
    # FUTURES MARK (+ funding), the market the base was read from and the brackets trigger on; the
    # spot `live_prices` map scored the perp/spot basis as edge until 2026-09-25 and now only feeds
    # the auto-triggers above. Placed AFTER the profit-lock on purpose: it makes a few small public
    # calls (a mark per due symbol, funding history), and measurement must never delay survival.
    # Total: never raises, never touches a trading decision, logs failures at WARNING.
    # Every exchange call from here to the market-state refresh shares ONE per-poll budget (a quarter of
    # the poll interval), so a hanging public endpoint can never stretch the loop past the next
    # protection pass; deferred work waits for a later poll and the tolerance rules stay honest.
    _measure_budget = _MeasurementBudget(_MEASURE_BUDGET_FRACTION * float(cfg.trading.poll_interval_sec or 60))
    _settle_probes_on_futures(memory, _measure_client, snapshot, live_prices,
                              budget=_measure_budget, backoff=_measure_backoff)
    # Score matured agent closes against the LIVE exit stack (bracket + trail/breakeven + carry hold),
    # replayed on 1m futures bars with ProtectionManager's EFFECTIVE cfg — exitDiscipline's benchmark.
    # At most 2 per poll, only after a close's 8h horizon; total, never raises, WARNING on failure.
    try:
      _score_exit_probe_stacks(memory, _measure_client, protection.cfg, attempts=_stack_attempts,
                               budget=_measure_budget)
    except Exception as exc:
      logger.warning("EXIT STACK: scoring step failed (%s) — exitDiscipline goes stale while this repeats", exc)
    # Market state: at most one refresh an hour (~3 public calls), after survival and settlement, so it
    # never delays either; the agent run, the order path and the exit probes only read its cache.
    try:
      if not _budget_spent(_measure_budget):
        _market_state.refresh()
    except Exception as exc:  # refresh is total; belt and braces
      logger.warning("MARKET STATE: refresh step failed (%s)", exc)
    if _measure_budget.deferred:
      logger.warning("MEASUREMENT BUDGET: %.0fs spent this poll; %d exchange call(s) deferred to later polls "
                     "(their rows wait, and are written off only past their settle tolerance)",
                     _measure_budget.seconds, _measure_budget.deferred)
    # Report-only gate scoreboard, at most hourly, after survival and settlement. Total.
    _log_gate_scoreboard(memory, cfg, _gate_log_last)

    # Retain throttled moves until the model actually reviews them. Without this queue, a trigger
    # followed by a quiet poll vanished and the adaptive shorter cooldown never got another chance.
    idle_hunt_due = _idle_hunt_due(
      idle_polls,
      cfg.trading.max_idle_polls,
      bool(snapshot.futures_pending_orders or snapshot.spot_pending_orders),
    )
    run_candidate = bool(triggers or pending_trigger_moves or pending_agent_triggers) or idle_hunt_due

    fill_lookback_minutes = 1440 if last_complete_fill_poll_ts <= 0 else max(
      30,
      min(1440, int((time.time() - last_complete_fill_poll_ts) / 60) + 5),
    )
    recent_fills = _fetch_recent_fills(kucoin, kucoin_futures, lookback_minutes=fill_lookback_minutes)
    fill_poll_errors = recent_fills.pop("_errors", [])
    critical_fill_errors = [
      error for error in fill_poll_errors
      if kucoin_futures is None or not str(error).startswith("spot fills:")
    ]
    if critical_fill_errors:
      safety.invalidate("Fill/close history incomplete: " + "; ".join(critical_fill_errors), revoke_active=True)
      logger.error("Entry/order authority disabled because futures event history is incomplete: %s", critical_fill_errors)
    if not fill_poll_errors:
      last_complete_fill_poll_ts = time.time()
    new_spot_fills = []
    for fill in recent_fills["spot_fills"]:
      fill_id = _fill_event_id(fill, "spot")
      if fill_id in logged_fill_ids:
        continue
      logged_fill_ids.add(fill_id)
      memory.mark_order_filled(
        fill.get("orderId"), fill.get("clientOid"),
        fill_ts=fill.get("createdAt") or fill.get("ts") or fill.get("time"),
        fill_price=fill.get("price") or fill.get("dealPrice"),
        fill_size=fill.get("size") or fill.get("filledSize"),
      )
      if memory.queue_agent_event("spot_fills", fill_id, fill):
        new_spot_fills.append(fill)
    new_futures_fills = []
    for fill in recent_fills["futures_fills"]:
      fill_id = _fill_event_id(fill, "futures")
      if fill_id in logged_fill_ids:
        continue
      logged_fill_ids.add(fill_id)
      memory.mark_order_filled(
        fill.get("orderId"), fill.get("clientOid"),
        fill_ts=fill.get("createdAt") or fill.get("ts") or fill.get("time"),
        fill_price=fill.get("price") or fill.get("dealPrice"),
        fill_size=fill.get("size") or fill.get("filledSize"),
      )
      if memory.queue_agent_event("futures_fills", fill_id, fill):
        new_futures_fills.append(fill)
    new_closed_positions = []
    for cp in recent_fills["closed_positions"]:
      close_id = _close_event_id(cp)
      legacy_id = str(cp.get("openTime") or "")
      if close_id in logged_closed_position_ids or (legacy_id and legacy_id in logged_closed_position_ids):
        continue
      memory.queue_agent_event("closed_positions", f"closed:{close_id}", cp)
      new_closed_positions.append(cp)
    recent_fills = {
      "spot_fills": new_spot_fills,
      "futures_fills": new_futures_fills,
      "closed_positions": new_closed_positions,
    }
    new_events_count = len(recent_fills["spot_fills"]) + len(recent_fills["futures_fills"]) + len(recent_fills["closed_positions"])
    pending_events_count = len(memory.get_pending_agent_events())
    if new_events_count:
      logger.info("Detected %d new fill/close events (spot=%d, futures=%d, closed=%d)",
                   new_events_count, len(recent_fills["spot_fills"]), len(recent_fills["futures_fills"]), len(recent_fills["closed_positions"]))

    for cp in recent_fills["closed_positions"]:
      cp_id = _close_event_id(cp)
      if not cp_id or cp_id in logged_closed_position_ids:
        continue
      try:
        sym = normalize_symbol(cp.get("symbol") or "")
        pnl = float(cp.get("pnl") or 0)
        roe = float(cp.get("roe") or 0)
        close_type = cp.get("type") or "unknown"
        side = "sell" if "LONG" in close_type.upper() else "buy"
        position_side = (
          "long" if "LONG" in close_type.upper()
          else "short" if "SHORT" in close_type.upper()
          else None
        )
        # Capture the exit price so the no-chase guard knows where we sold/covered.
        exit_price = None
        for _k in ("closePrice", "avgExitPrice", "settleClosePrice", "markPrice", "lastPrice"):
          _v = cp.get(_k)
          if _v not in (None, "", 0, "0"):
            try:
              exit_price = float(_v)
              break
            except (TypeError, ValueError):
              continue
        if exit_price is None:
          exit_price = last_prices.get(sym)  # fallback: latest poll price ≈ exit
        # Entry price for the closed-position chart's open marker (public-safe price, no size).
        entry_price = None
        for _k in ("avgEntryPrice", "openPrice", "entryPrice", "avgEntry"):
          _v = cp.get(_k)
          if _v not in (None, "", 0, "0"):
            try:
              entry_price = float(_v)
              break
            except (TypeError, ValueError):
              continue
        # MFE/MAE from the pre-reset extremes: how far the trade ran in profit (peak) and underwater
        # (trough) before closing — the data that tells us if TPs are set within realistic reach.
        _ext = pre_close_extremes.get(sym, {}) if isinstance(pre_close_extremes, dict) else {}
        cp_open_time = cp.get("openTime") or cp.get("openingTimestamp")
        if _ext.get("positionOpenTime") not in (None, "") and cp_open_time not in (None, ""):
          try:
            ext_open = int(float(_ext["positionOpenTime"]))
            close_open = int(float(cp_open_time))
            ext_open = ext_open if ext_open > 1_000_000_000_000 else ext_open * 1000
            close_open = close_open if close_open > 1_000_000_000_000 else close_open * 1000
            if abs(ext_open - close_open) > 1000:
              _ext = {}
          except (TypeError, ValueError):
            _ext = {}
        peak_pnl = _ext.get("peakPnl")
        trough_pnl = _ext.get("troughPnl")
        # Include the fee/funding-adjusted terminal result so every close lies inside its recorded
        # lifecycle range; sampled unrealized extrema alone missed terminal slippage and fees.
        peak_pnl = pnl if peak_pnl is None else max(float(peak_pnl), pnl)
        trough_pnl = pnl if trough_pnl is None else min(float(trough_pnl), pnl)
        entry_context = memory.entry_context_for_position(
          sym,
          cp.get("openTime") or cp.get("openingTimestamp"),
          position_side,
        )
        memory.log_decision(
          sym,
          f"futures_{side}_triggered",
          confidence=0.0,
          reason=f"TP/SL triggered ({close_type}, ROE {roe:.2%})",
          pnl=pnl,
          paper=False,
          exit_price=exit_price,
          close_type=close_type,
          position_id=cp.get("id"),
          position_open_time=cp.get("openTime") or cp.get("openingTimestamp"),
          position_side=position_side,
          peak_pnl=peak_pnl,
          trough_pnl=trough_pnl,
          entry_price=entry_price,
          entry_context=entry_context,
        )
        logged_closed_position_ids.add(cp_id)
        memory.record_seen_close_id(cp_id)
        logger.info("Recorded triggered close for %s: PnL=%.4f (%s)", sym, pnl, close_type)
        # An exit that reached NEITHER the stop nor the target was somebody's discretionary call, not
        # the bracket resolving. Score it later against what the bracket would have done — that is the
        # book's largest measured leak and the model currently gets no feedback on it at all.
        try:
          _ctx = entry_context if isinstance(entry_context, dict) else {}
          _e = _ctx.get("fillPrice") or _ctx.get("entryPrice") or entry_price
          _sl, _tp = _ctx.get("stopLossPrice"), _ctx.get("takeProfitPrice")
          if _e and _sl and _tp and exit_price:
            _long = str(position_side).lower() == "long"
            _reached = (
              (float(exit_price) >= float(_tp) or float(exit_price) <= float(_sl)) if _long
              else (float(exit_price) <= float(_tp) or float(exit_price) >= float(_sl))
            )
            if not _reached:
              _risk = abs(float(_e) - float(_sl))
              _taken = (
                ((float(exit_price) - float(_e)) if _long else (float(_e) - float(exit_price))) / _risk
                if _risk > 0 else None
              )
              memory.record_exit_probe(
                sym, position_side, _e, _sl, _tp, exit_price,
                realized_r=_taken, setup_family=_ctx.get("setupFamily"),
                closed_by="agent" if memory.recent_agent_close(sym) else "protection",
                regime=_ctx.get("regime") if isinstance(_ctx.get("regime"), dict) else None,
                # What the live-stack replay needs (fill, original risk, noise band, carry hold) and the
                # entry-bias tags (counterAtEntry / htfAligned) that split the model's exit record.
                **exit_probe_inputs(_ctx, position_side),
                # The market the trade was ENTERED in (splits the trail's record: trailByMarketState)
                # and the loop's current reading at the close. Cache reads only.
                market_state=_ctx.get("marketState") if isinstance(_ctx.get("marketState"), dict) else None,
                market_state_at_exit=_market_state.current(),
              )
        except Exception as exc:
          logger.warning("EXIT PROBE: recording failed for %s (%s) — exitDiscipline misses this close",
                         sym, exc)
      except Exception as exc:
        logger.warning("Failed to record triggered close: %s", exc)

    # --- Circuit Breaker Checks ---
    cb = cfg.circuit_breaker
    restriction_reasons: list[str] = []

    if snapshot.drawdown_pct >= cb.max_daily_drawdown_pct:
      restriction_reasons.append(f"Daily drawdown {snapshot.drawdown_pct:.1f}% >= {cb.max_daily_drawdown_pct}% limit")

    consec_losses = memory.consecutive_losses()
    if consec_losses >= cb.max_consecutive_losses:
      last_loss_ts = None
      with memory._lock:
        decs = memory._read().get("decisions", [])
      decs = memory._authoritative_realized_rows(decs)
      for d in sorted(decs, key=lambda x: x.get("ts", 0), reverse=True):
        pnl = d.get("pnl")
        if pnl is not None:
          try:
            if float(pnl) < 0:
              last_loss_ts = d.get("ts")
              break
          except (TypeError, ValueError):
            continue
      if last_loss_ts:
        elapsed_min = (int(time.time()) - last_loss_ts) / 60
        if elapsed_min < cb.cooldown_minutes:
          restriction_reasons.append(
            f"{consec_losses} consecutive losses (cooldown {int(cb.cooldown_minutes - elapsed_min)}min remaining)"
          )

    if restriction_reasons:
      snapshot.trading_restricted = True
      snapshot.restriction_reason = "; ".join(restriction_reasons)
      logger.warning("CIRCUIT BREAKER: %s", snapshot.restriction_reason)
      notifier.send(f"<b>CIRCUIT BREAKER ACTIVE</b>\n{snapshot.restriction_reason}\nAgent restricted to close-only mode.")

    # The model is the most expensive and least deterministic part of the loop. Code-driven
    # protection and deterministic order expiry still evaluate every poll. Only an actual position
    # uses active model cadence: an exchange-atomic pending bracket is not exposed capital and does
    # not need the model to babysit/cancel it before its deterministic expiry.
    capital_exposed = bool(snapshot.futures_positions)
    pending_orders_active = bool(snapshot.futures_pending_orders or snapshot.spot_pending_orders)
    open_book = capital_exposed or pending_orders_active
    effective_flat_cooldown = _productivity_adjusted_flat_cooldown(
      cfg.trading.flat_agent_cooldown_sec,
      unproductive_runs,
      cfg.trading.flat_backoff_max_multiplier,
    )
    cooldown_sec = _adaptive_agent_cooldown(
      flat_cooldown_sec=effective_flat_cooldown,
      active_cooldown_sec=cfg.trading.active_agent_cooldown_sec,
      book_active=capital_exposed,
      new_events_count=pending_events_count,
      trigger_move_pcts=list(pending_trigger_moves.values()),
      price_trigger_pct=cfg.trading.price_change_trigger_pct,
    )
    run_candidate = run_candidate or bool(pending_events_count)
    cooldown_elapsed = time.time() - last_agent_run_ts
    should_run = (
      agent_task is None
      and run_candidate
      and (last_agent_run_ts <= 0 or cooldown_elapsed >= max(0.0, cooldown_sec))
    )
    if agent_task is None and run_candidate and not should_run and idle_polls % max(1, cfg.trading.max_idle_polls) == 0:
      logger.info(
        "Agent run throttled for another %dsec (capital_exposed=%s pending_orders=%s no_action_runs=%d)",
        int(cooldown_sec - cooldown_elapsed), capital_exposed, pending_orders_active, unproductive_runs,
      )

    if should_run and not stop_event.is_set():
      run_token = safety.authorize_run()
      if not run_token:
        logger.warning("Agent run skipped because live entry/order authority is unavailable.")
        idle_polls += 1
        try:
          await asyncio.wait_for(stop_event.wait(), timeout=cfg.trading.poll_interval_sec)
        except asyncio.TimeoutError:
          pass
        continue
      idle_polls = 0
      last_agent_run_ts = time.time()
      _persist_agent_scheduler()
      agent_triggers = list(pending_agent_triggers)
      agent_triggers.extend(trigger for trigger in triggers if trigger not in agent_triggers)
      agent_triggers.extend(
        f"pending_price_move:{symbol}:{move:.2f}%"
        for symbol, move in sorted(pending_trigger_moves.items())
        if not any(trigger.startswith(f"price_move:{symbol}:") for trigger in agent_triggers)
      )
      if not agent_triggers:
        agent_triggers = ["idle_threshold"]
      agent_task_trigger_moves = dict(pending_trigger_moves)
      agent_task_discrete_triggers = set(pending_agent_triggers)
      agent_task_prices = {
        symbol: float(ticker.price)
        for symbol, ticker in snapshot.tickers.items()
      }
      agent_task_position_fingerprint = _futures_position_fingerprint(snapshot.futures_positions)
      pending_trigger_moves.clear()
      pending_agent_triggers.clear()
      pending_batch = memory.get_pending_agent_events()
      events_for_agent: Dict[str, list] = {}
      for event in pending_batch:
        kind = str(event.get("kind") or "")
        if kind:
          events_for_agent.setdefault(kind, []).append(event.get("payload"))
      force_research_for_run = bool(force_research and not open_book)
      _cal = memory.macro_calendar_state()
      _cal_reason = macro_calendar_refresh_reason(_cal["events"], _cal["updated"], time.time())
      refresh_calendar_for_run = bool(
        cfg.regime.macro_events_enabled and _cal_reason
        and time.time() - last_calendar_attempt_ts >= _CALENDAR_RETRY_SEC
      )
      if refresh_calendar_for_run:
        logger.info("MACRO CALENDAR: %s — this run will hand off to Research for a calendar-only refresh.", _cal_reason)
      agent_task_calendar_refresh = refresh_calendar_for_run
      agent_task_triggers = agent_triggers
      agent_task_forced_research = force_research_for_run
      agent_task_token = run_token
      agent_task_event_ids = [str(event.get("id") or "") for event in pending_batch]
      agent_task_started_ts = time.time()
      agent_task_timed_out = False
      logger.info("Starting background agent. Triggers: %s", agent_triggers)
      agent_task = asyncio.create_task(_run_in_daemon_thread(
        run_trading_agent, cfg, snapshot, kucoin, kucoin_futures, azure_client, ls_client,
        recent_fills=events_for_agent or None,
        force_research=force_research_for_run,
        refresh_calendar=refresh_calendar_for_run,
        calendar_state=_cal,
        safety_state=safety,
        entry_token=run_token,
        # The same lock-protected live clock the carry hold uses, so entryThesis shows the settlement
        # ProtectionManager is actually holding for (cache only after the first read).
        funding_clock=_trade_context.funding_clock,
        # Cache-only reader of the hourly market state: shown to the model as plain numbers and stamped
        # on each entry and probe. Never fetches.
        market_state=_market_state.current,
      ))
    else:
      idle_polls += 1
      logger.info(
        "Agent %s. Idle polls: %d/%d",
        "running in background" if agent_task is not None else "idle/throttled",
        idle_polls,
        cfg.trading.max_idle_polls,
      )

    # Publish a sanitized public-safe snapshot for the read-only spectator dashboard.
    # Self-throttled and never raises, so it is safe to call every poll. Open positions are
    # read from `snapshot` (live exchange truth), not MemoryStore (which lingers after TP/SL).
    dashboard.publish(memory, snapshot, last_prices, cfg)

    try:
      await asyncio.wait_for(stop_event.wait(), timeout=cfg.trading.poll_interval_sec)
    except asyncio.TimeoutError:
      pass

  safety.begin_shutdown()
  if agent_task is not None and not agent_task.done():
    safety.revoke_run(agent_task_token, "Process is shutting down")
    try:
      await asyncio.wait_for(asyncio.shield(agent_task), timeout=_AGENT_SHUTDOWN_GRACE_SEC)
    except (asyncio.TimeoutError, asyncio.CancelledError):
      agent_task.cancel()


def main() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  cfg = load_config()

  if cfg.supervisor.log_file:
    fh = RotatingFileHandler(
      cfg.supervisor.log_file,
      maxBytes=cfg.supervisor.log_max_bytes,
      backupCount=cfg.supervisor.log_backup_count,
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

  if cfg.supervisor.enabled and cfg.telegram.enabled:
    from .telegram_bot import start_telegram_bot
    bot_thread = threading.Thread(target=start_telegram_bot, args=(cfg,), daemon=True, name="supervisor-bot")
    bot_thread.start()
    logger.info("Supervisor Telegram bot started.")

  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  stop_event = asyncio.Event()
  safety = TradingSafetyState()
  def _request_shutdown() -> None:
    safety.begin_shutdown()
    stop_event.set()
  loop.add_signal_handler(signal.SIGTERM, _request_shutdown)
  loop.add_signal_handler(signal.SIGINT, _request_shutdown)
  try:
    loop.run_until_complete(trading_loop(stop_event=stop_event, safety=safety))
  except KeyboardInterrupt:
    logger.info("Shutting down...")
  except Exception as exc:
    logger.error("Fatal error: %s", exc)
    sys.exit(1)
  finally:
    safety.begin_shutdown()
    loop.close()


if __name__ == "__main__":
  main()
