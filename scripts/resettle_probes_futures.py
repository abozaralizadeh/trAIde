"""Re-settle the STORED signal probes on the futures market, once, offline.

Why this exists
---------------
Until 2026-09-25 every signal probe took its base price from the FUTURES mark (`tools._live_entry_price`)
but was settled from the SPOT ticker (`main.py` passed its spot `live_prices` into
`memory.settle_signal_probes`). On a coin whose perp trades away from spot that gap was scored as a
directional return, and it landed almost entirely on `funding_carry` — the playbook that is chosen
exactly when the gap is widest. ONE-USDT on 2026-09-20: spot ~0.0050 vs perp ~0.0038, eight long probes
each recording ~+7.5% at 60m that the contract never made. The live code now settles on the futures
mark (plus funding); this script fixes the rows that were stored before that.

What it does
------------
For every retained signal probe with NO ``priceSource`` stamp (the legacy rows — anything recorded
after the fix carries one and is left alone, which also makes the script safe to re-run):

* re-stamps ``m5 / m15 / m60 / m240`` from the close of the 1m FUTURES kline that ended at the due
  minute, forward-filling up to 3 minutes for a thin contract that did not trade in that minute; with
  no bar in that window the horizon is recorded ``None`` (unmeasured) — never borrowed from spot;
* stamps ``f{h}``, the funding the position would have RECEIVED over (signal, due] from the contract's
  funding history (longs receive -rate, shorts +rate — `memory.funding_received_from_history`);
* stamps ``priceSource = "futures_mark_resettled"`` and ``resettledTs``;
* also fills horizons the live loop MISSED — probes on symbols with no spot ticker in the snapshot
  never settled, and probes older than the 5m/15m horizons (added 2026-09-04) had them written off.
  That is not back-stamping: each is the contract's own close at that horizon's due minute. On the
  2026-09-24 copy this added ~20 observations to `fade_extreme` (still stood aside) and the report
  prints the per-family n before and after so the change is visible before ``--apply``;
* checks EVERY kline bar: ``high`` must be the row maximum and ``low`` the row minimum. KuCoin futures
  klines are ``[ts, open, HIGH, LOW, close, ...]`` while spot is ``[ts, open, close, high, low]``; an
  offline replay that mixed them up once reversed a finding. A violation aborts the whole run.
* marks an exit probe that was resolved AFTER its measurable life (``expire_hours`` + tolerance) as
  ``unmeasured`` — the same rule the live settle now applies (an XMR probe from 09-12 was resolved
  seven days late); the old outcome is kept under ``supersededOutcome``.
* re-resolves every other SPOT-settled exit-probe bracket (a resolved outcome with no ``priceSource``;
  the live settle stamps one since 2026-09-25) on 1m futures bar CLOSES — the poll-like convention the
  live mark settle uses — keeping the old outcome under ``supersededOutcome`` (ONE-USDT 09-21 00:04 was
  stored as a +1.70R take-profit and was stopped on the contract). These feed ``otherExits`` and the
  trail's regime/market-state splits, the evidence the adaptive-trail decision waits on.

Rows are RE-SETTLED, never dropped: deleting the high-basis rows would cut exactly the extreme-funding
sample and could un-learn a verdict. A symbol whose data cannot be fetched is left untouched (still
unstamped) and reported, so a re-run picks it up.

* backfills ``stack`` (``stackR`` / ``resolvedBy`` / ``resolvedTs``) on every AGENT-closed exit probe
  whose 8h horizon has passed and that has none: the live exit stack — exchange bracket + breakeven /
  noise-band trail + carry hold, `protection.replay_protection_stack` with ProtectionManager's
  EFFECTIVE config (breakeven fee raised to the round-trip cost, exactly as `main.py` builds it) — is
  replayed on 1m futures bars from the fill to the probe's horizon. That is the benchmark
  `edge.exit_discipline_stats` now scores the model's closes against; the live loop does the same for
  new closes, so this only covers rows recorded before it existed (the bare-bracket scoreboard read
  -6.23R on the 5 attributed closes where the stack reads about -3.5R). A legacy row lacks the replay
  inputs, so its entry is found in the trades ledger (same symbol and side, filled before the close,
  same original stop) — or, once the 100-row ledger has pruned it (~2.5 days), in the realized close
  record, which carries the same entry context — and the row gains ``fillTs / initRiskPx / noiseBandR /
  holdUntilTs`` and the
  entry-bias tags (``counterAtEntry`` / ``htfAligned``) too. A row whose entry or bars cannot be found
  is left untouched and listed; the scoreboard shows it as bracket-only audit (``legacyBracketScored``)
  and EXCLUDES it from the exitDiscipline verdict, so run this soon after deploy (see below).

How to run it (on the VM) — SOON after deploying the 2026-09-25 change
----------------------------------------------------------------------
The trades ledger holds only ~2.5 days of placements; the close records keep the legacy entries
longer, but the sooner this runs the fewer rows depend on that fallback.

1. STOP THE BOT:   sudo systemctl stop traide          (it rewrites the memory file every poll)
2. Back it up:     cp .agent_memory.json .agent_memory.json.manual.bak
3. Dry run:        .venv/bin/python -m scripts.resettle_probes_futures --memory .agent_memory.json
4. Apply:          .venv/bin/python -m scripts.resettle_probes_futures --memory .agent_memory.json --apply
5. Start it again: sudo systemctl start traide

``--apply`` also writes its own timestamped backup next to the file before touching it, refuses to run
while the file looks live (modified in the last few minutes) unless ``--bot-stopped`` is given, and
refuses to write if the file changed while it was working. Public market-data endpoints only; no keys.
The stack replay reads the bot's profit-protection settings from the VM's own config (``.env``) so
it replays the stack the live loop runs; ``--skip-stack`` leaves exit probes on their bracket.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from src.memory import (  # noqa: E402
  EXIT_PROBE_EXPIRE_HOURS,
  SIGNAL_PROBE_HORIZONS_MIN,
  MemoryStore,
  _probe_settle_tolerance_sec,
  funding_received_from_history,
)
from src.position_context import exit_probe_inputs  # noqa: E402
from src.protection import replay_protection_stack  # noqa: E402
from src.utils import normalize_symbol  # noqa: E402

RESETTLED_SOURCE = "futures_mark_resettled"
STACK_SOURCE = "resettle_script_1m_replay"
STACK_MIN_COVERAGE_SEC = 15 * 60       # bars ending this far short of the horizon = incomplete window
BAR_SEC = 60
CHUNK_BARS = 200                       # KuCoin futures kline page size
CHUNK_SEC = CHUNK_BARS * BAR_SEC
FORWARD_FILL_SEC = 3 * 60              # thin contracts: carry the last close forward at most 3 minutes
FUNDING_CHUNK_SEC = 2 * 86400          # funding history paged in 2-day windows (<= 48 settlements at 1h)
LIVE_FILE_GUARD_SEC = 180              # the loop rewrites the file every poll; younger than this = running
RETRY_PAUSE_SEC = 1.0                  # back-off unit between retries of one API call


class ColumnOrderError(ValueError):
  """A kline row whose high/low are not the row extremes — the column order is not what we assume."""


def futures_symbol(spot_symbol: str) -> Optional[str]:
  """``ONE-USDT`` -> ``ONEUSDTM`` (``BTC`` -> ``XBT``), the same mapping as `agent._to_futures_symbol`."""
  sym = normalize_symbol(spot_symbol or "")
  if "-" not in sym:
    return None
  base, quote = sym.split("-", 1)
  if base == "BTC":
    base = "XBT"
  return f"{base}{quote}M"


def validate_bar(row: Any) -> List[float]:
  """``[ts_sec, open, high, low, close]`` from a raw FUTURES kline row, or raise ColumnOrderError.

  Futures rows are ``[ts_ms, open, HIGH, LOW, close, volume, turnover]``. The high must be the maximum
  and the low the minimum of the four prices on EVERY bar; one violation means the column order is
  wrong and nothing computed from these rows can be trusted.
  """
  try:
    ts = float(row[0])
    o, h, l, c = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
  except (TypeError, ValueError, IndexError) as exc:
    raise ColumnOrderError(f"unreadable kline row {row!r}: {exc}") from exc
  if not all(math.isfinite(v) for v in (ts, o, h, l, c)):
    raise ColumnOrderError(f"non-finite kline row {row!r}")
  if h != max(o, h, l, c) or l != min(o, h, l, c):
    raise ColumnOrderError(
      f"kline row {row!r}: high/low are not the row extremes — futures rows must be "
      "[ts, open, HIGH, LOW, close]; refusing to settle on misread candles"
    )
  if ts > 1e12:
    ts /= 1000.0
  return [ts, o, h, l, c]


def close_at_due(bars: List[List[float]], bar_starts: List[float], due_s: float,
                 max_fill_sec: float = FORWARD_FILL_SEC) -> Optional[float]:
  """Close of the last 1m bar that had CLOSED by ``due_s``, if it closed no more than ``max_fill_sec`` before.

  ``bars`` are validated ``[ts, o, h, l, c]`` rows sorted by start time and ``bar_starts`` their start
  times (for bisect). A bar starting at ``t`` closes at ``t + 60``; the bar ending exactly at the due
  minute is used when it exists, otherwise the most recent earlier one within the fill window — a thin
  contract with no trade in a minute has no bar for it, and its last trade is still its price.
  """
  idx = bisect.bisect_right(bar_starts, due_s - BAR_SEC) - 1
  if idx < 0:
    return None
  bar = bars[idx]
  if due_s - (bar[0] + BAR_SEC) > max_fill_sec:
    return None
  return bar[4]


def _retry(fn: Callable[[], Any], attempts: int = 4) -> Any:
  last: Optional[BaseException] = None
  for i in range(attempts):
    try:
      return fn()
    except ColumnOrderError:
      raise                         # a misread candle is a bug, not a glitch — never retried away
    except Exception as exc:        # network/API error — retried, then surfaced to the caller
      last = exc
      if i + 1 < attempts:
        time.sleep(RETRY_PAUSE_SEC * (i + 1))
  raise last  # type: ignore[misc]


class BarBook:
  """1m futures bars for one contract, fetched in grid-aligned 200-bar chunks and cached."""

  def __init__(self, client: Any, fsym: str, sleep_sec: float = 0.12) -> None:
    self.client = client
    self.fsym = fsym
    self.sleep_sec = sleep_sec
    self._chunks: Dict[int, List[List[float]]] = {}
    self.rows_checked = 0

  def ensure(self, start_s: float, end_s: float) -> None:
    c0 = int(start_s) // CHUNK_SEC * CHUNK_SEC
    while c0 <= end_s:
      if c0 not in self._chunks:
        raw = _retry(lambda c=c0: self.client.get_candles(
          self.fsym, granularity=1, start_at=c * 1000, end_at=(c + CHUNK_SEC) * 1000,
        ))
        self._chunks[c0] = [validate_bar(r) for r in (raw or [])]
        self.rows_checked += len(self._chunks[c0])
        if self.sleep_sec:
          time.sleep(self.sleep_sec)
      c0 += CHUNK_SEC

  def series(self) -> tuple[List[List[float]], List[float]]:
    by_ts: Dict[float, List[float]] = {}
    for rows in self._chunks.values():
      for r in rows:
        by_ts[r[0]] = r
    bars = [by_ts[k] for k in sorted(by_ts)]
    return bars, [b[0] for b in bars]


def fetch_funding_history(client: Any, fsym: str, start_s: float, end_s: float,
                          sleep_sec: float = 0.12) -> List[Dict[str, Any]]:
  """Every funding settlement of ``fsym`` in [start_s, end_s], paged and de-duplicated by timepoint."""
  out: Dict[Any, Dict[str, Any]] = {}
  t = int(start_s)
  while t <= end_s:
    t_end = min(int(end_s), t + FUNDING_CHUNK_SEC)
    rows = _retry(lambda a=t, b=t_end: client.get_funding_rate_history(fsym, start_at=a * 1000, end_at=b * 1000))
    for r in rows or []:
      if isinstance(r, dict):
        key = r.get("timepoint") if r.get("timepoint") is not None else r.get("timePoint")
        out[key] = r
    if sleep_sec:
      time.sleep(sleep_sec)
    t = t_end + 1
  return list(out.values())


def _legacy_signal_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
  """Rows the live code never settled on futures: carry a base price and no ``priceSource`` stamp."""
  rows = []
  for row in list(data.get("trades") or []) + list(data.get("signal_probes") or []):
    ctx = row.get("entryContext") if isinstance(row, dict) else None
    if not isinstance(ctx, dict) or not ctx.get("marketPriceAtSignal"):
      continue
    if ctx.get("priceSource"):
      continue
    if str(ctx.get("positionSide") or "").lower() not in ("long", "short"):
      continue
    rows.append(row)
  return rows


def resettle(
  data: Dict[str, Any],
  client: Any,
  *,
  now: Optional[float] = None,
  horizons_min: Iterable[int] = SIGNAL_PROBE_HORIZONS_MIN,
  expire_hours: float = EXIT_PROBE_EXPIRE_HOURS,
  sleep_sec: float = 0.12,
  log: Callable[[str], None] = print,
) -> Dict[str, Any]:
  """Re-settle ``data`` (a loaded memory file) IN PLACE on 1m futures bars; return a report.

  Raises ColumnOrderError on any misread bar — nothing is half-written in that case because the
  caller only persists after this returns. Network failures are per symbol: that symbol's rows are
  left exactly as they were and listed under ``failedSymbols``.
  """
  now_s = float(time.time() if now is None else now)
  horizons = tuple(int(h) for h in horizons_min)
  max_h = max(horizons)
  report: Dict[str, Any] = {
    "rows": 0, "stamps": 0, "unmeasured": 0, "recovered": 0, "fundingCredited": 0, "failedSymbols": {},
    "barsChecked": 0, "baseGapPct": [], "exitProbesMarkedUnmeasured": 0,
  }

  by_symbol: Dict[str, List[Dict[str, Any]]] = {}
  for row in _legacy_signal_rows(data):
    by_symbol.setdefault(str(row.get("symbol") or ""), []).append(row)

  for symbol in sorted(by_symbol):
    rows = by_symbol[symbol]
    fsym = futures_symbol(symbol)
    if not fsym:
      report["failedSymbols"][symbol] = "no futures contract mapping"
      continue
    ts_list = [int(r.get("ts") or 0) for r in rows]
    book = BarBook(client, fsym, sleep_sec=sleep_sec)
    try:
      for ts0 in ts_list:
        # Signal minute (for the base-gap check) through the widest due minute.
        book.ensure(ts0 - FORWARD_FILL_SEC - BAR_SEC, min(now_s, ts0 + max_h * 60))
    except ColumnOrderError:
      raise
    except Exception as exc:
      report["failedSymbols"][symbol] = f"klines: {exc}"
      log(f"  {symbol}: kline fetch failed ({exc}) — rows left untouched, re-run to retry")
      continue
    report["barsChecked"] += book.rows_checked
    bars, starts = book.series()
    try:
      funding = fetch_funding_history(
        client, fsym, min(ts_list), min(now_s, max(ts_list) + max_h * 60), sleep_sec=sleep_sec,
      )
    except Exception as exc:
      funding = None
      log(f"  {symbol}: funding history unavailable ({exc}) — price re-settled, no funding credit")

    for row in rows:
      ctx = row["entryContext"]
      ts0 = int(row.get("ts") or 0)
      side = str(ctx.get("positionSide")).lower()
      probe = ctx.get("signalProbe") if isinstance(ctx.get("signalProbe"), dict) else {}
      new_probe: Dict[str, Any] = {k: v for k, v in probe.items() if not (
        isinstance(k, str) and k[:1] in ("m", "f") and k[1:].isdigit()
      )}
      for h in horizons:
        due = ts0 + h * 60
        if due > now_s - 2 * BAR_SEC:
          # Not due yet, or its bar has barely closed: leave it for the live settle (futures mark).
          continue
        px = close_at_due(bars, starts, due)
        new_probe[f"m{h}"] = px
        report["stamps"] += 1
        if px is None:
          report["unmeasured"] += 1
          continue
        if probe.get(f"m{h}") is None:
          report["recovered"] += 1     # missed live (None or never stamped), measured now
        if funding is not None:
          credit = funding_received_from_history(funding, side, ts0, due)
          if credit is not None:
            new_probe[f"f{h}"] = credit
            if credit:
              report["fundingCredited"] += 1
      base = float(ctx.get("marketPriceAtSignal"))
      fut_at_signal = close_at_due(bars, starts, ts0)
      if fut_at_signal:
        report["baseGapPct"].append((symbol, ts0, (base / fut_at_signal - 1.0) * 100.0))
      ctx["signalProbe"] = new_probe
      ctx["priceSource"] = RESETTLED_SOURCE
      ctx["resettledTs"] = int(now_s)
      report["rows"] += 1

  report["exitProbesMarkedUnmeasured"] = mark_late_exit_probes(data, expire_hours=expire_hours)
  # ...then the spot-settled bracket outcomes on the rows still resolved (never the ones just marked).
  report["exitBrackets"] = resettle_exit_brackets(
    data, client, now=now_s, expire_hours=expire_hours, sleep_sec=sleep_sec, log=log)
  return report


def effective_stack_cfg(data: Dict[str, Any], app_cfg: Any = None) -> Any:
  """ProtectionManager's EFFECTIVE profit-protection config, built the way `main.trading_loop` builds it.

  The live manager raises ``breakeven_fee_pct`` to twice (futures taker + estimated slippage); the raw
  ``cfg.profit_protection`` is the wrong comparator. ``app_cfg`` defaults to `load_config()` (the VM's
  env); the taker fee is the latest one the bot recorded in this memory file.
  """
  from src.protection import ProtectionManager

  if app_cfg is None:
    from src.config import load_config
    app_cfg = load_config()
  fees = [f for f in (data.get("fees") or []) if isinstance(f, dict)]
  taker = float((fees[-1] if fees else {}).get("futures_taker") or 0.0006)
  cost = 2.0 * (taker + float(app_cfg.trading.estimated_slippage_pct or 0.0))
  return ProtectionManager(app_cfg.profit_protection, None, breakeven_cost_pct=cost).cfg


def _entry_from_close_records(data: Dict[str, Any], sym: Any, side: str, close_ts: float,
                              stop: float) -> Optional[Dict[str, Any]]:
  """The entry context stored on the REALIZED close record (``decisions`` rows with a pnl), or None.

  The trades ledger is capped at 100 placements (~2.5 days at ~40/day), so the entries behind the
  legacy agent closes fall out of it within days of the deploy — DASH and INJ, the two rows that carry
  the whole bracket-vs-stack gap, around 09-26. The close record keeps the same context
  (`main` stores `entry_context_for_position`, the trade's context with ``fillTs`` / ``fillPrice``) and
  lives far longer (MAX_CLOSED_TRADES=200). Matched exactly like the ledger: same symbol and side, the
  same original stop, filled at or before the close; the latest fill wins.
  """
  want = normalize_symbol(str(sym or ""))
  best: Optional[tuple] = None
  for dec in data.get("decisions") or []:
    if not isinstance(dec, dict) or dec.get("pnl") is None:
      continue
    ctx = dec.get("entryContext")
    if not isinstance(ctx, dict) or normalize_symbol(str(dec.get("symbol") or "")) != want:
      continue
    if str(ctx.get("positionSide") or dec.get("positionSide") or "").lower() != side:
      continue
    try:
      fill_ts = float(ctx.get("fillTs"))
      ctx_stop = float(ctx.get("stopLossPrice"))
    except (TypeError, ValueError):
      continue
    if not math.isfinite(fill_ts) or fill_ts > close_ts or not math.isclose(ctx_stop, stop, rel_tol=1e-9, abs_tol=0.0):
      continue
    if best is None or fill_ts > best[0]:
      best = (fill_ts, ctx)
  return copy.deepcopy(best[1]) if best is not None else None


def _entry_for_exit_probe(data: Dict[str, Any], row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
  """The filled entry behind a legacy exit probe, or None when it cannot be pinned down.

  Same symbol and side, filled at or before the close, and the SAME original stop (the probe's
  ``stopPrice`` is the entry's ``stopLossPrice``). The latest such fill wins. The result is the entry's
  context with ``fillTs`` / ``fillPrice`` filled in from the trade row, as
  `MemoryStore.entry_context_for_position` returns it. The trades ledger is searched first; once it has
  pruned the entry, the realized close record's own copy is used (``_entry_from_close_records``).
  """
  sym = row.get("symbol")
  side = str(row.get("positionSide") or "").lower()
  try:
    close_ts = float(row.get("ts"))
    stop = float(row.get("stopPrice"))
  except (TypeError, ValueError):
    return None
  best: Optional[tuple] = None
  for trade in data.get("trades") or []:
    if not isinstance(trade, dict) or trade.get("symbol") != sym or trade.get("filled") is not True:
      continue
    ctx = trade.get("entryContext")
    if not isinstance(ctx, dict):
      continue
    fallback = "long" if str(trade.get("side") or "").lower() == "buy" else "short"
    if str(ctx.get("positionSide") or fallback).lower() != side:
      continue
    try:
      fill_ts = float(trade.get("fillTs") or trade.get("ts"))
      ctx_stop = float(ctx.get("stopLossPrice"))
    except (TypeError, ValueError):
      continue
    if fill_ts > close_ts or not math.isclose(ctx_stop, stop, rel_tol=1e-9, abs_tol=0.0):
      continue
    if best is None or fill_ts > best[0]:
      best = (fill_ts, trade)
  if best is None:
    return _entry_from_close_records(data, sym, side, close_ts, stop)
  trade = best[1]
  out = copy.deepcopy(trade["entryContext"])
  out.setdefault("fillTs", trade.get("fillTs") or trade.get("ts"))
  out.setdefault("fillPrice", trade.get("fillPrice"))
  if out.get("fillTs") is None:
    out["fillTs"] = best[0]
  return out


def backfill_exit_stacks(
  data: Dict[str, Any],
  client: Any,
  stack_cfg: Any,
  *,
  now: Optional[float] = None,
  expire_hours: float = EXIT_PROBE_EXPIRE_HOURS,
  sleep_sec: float = 0.12,
  log: Callable[[str], None] = print,
) -> Dict[str, Any]:
  """Replay the live exit stack for matured AGENT exit probes that have no ``stack`` yet, IN PLACE.

  Raises ColumnOrderError on a misread bar (the whole run aborts, nothing is written). Everything else
  is per row: a row whose entry or bars cannot be found is left untouched and listed under ``failed``.
  """
  now_s = float(time.time() if now is None else now)
  horizon = float(expire_hours) * 3600.0
  report: Dict[str, Any] = {"scored": [], "failed": {}, "barsChecked": 0}
  for row in data.get("exit_probes") or []:
    if not isinstance(row, dict) or str(row.get("closedBy") or "").lower() != "agent" or row.get("stack"):
      continue
    try:
      ts0 = float(row.get("ts"))
    except (TypeError, ValueError):
      continue
    if now_s < ts0 + horizon:
      continue          # not matured: the live loop scores it once its horizon passes
    label = f"{row.get('symbol')} @ {int(ts0)}"
    side = str(row.get("positionSide") or "").lower()
    inputs: Dict[str, Any] = {}
    if not (row.get("fillTs") and row.get("initRiskPx")):
      ctx = _entry_for_exit_probe(data, row)
      if ctx is None:
        report["failed"][label] = "entry not found in the trades ledger or close records"
        continue
      inputs = exit_probe_inputs(ctx, side)
    fill_ts = row.get("fillTs") or inputs.get("fill_ts")
    init_risk = row.get("initRiskPx") or inputs.get("init_risk_px")
    if not fill_ts or not init_risk:
      report["failed"][label] = "no fill time or original risk"
      continue
    noise_band = row.get("noiseBandR") if row.get("noiseBandR") is not None else inputs.get("noise_band_r")
    hold_until = row.get("holdUntilTs") if row.get("holdUntilTs") is not None else inputs.get("hold_until_ts")
    fsym = futures_symbol(str(row.get("symbol") or ""))
    if not fsym:
      report["failed"][label] = "no futures contract mapping"
      continue
    end_ts = ts0 + horizon
    book = BarBook(client, fsym, sleep_sec=sleep_sec)
    try:
      book.ensure(float(fill_ts) - BAR_SEC, end_ts)
    except ColumnOrderError:
      raise
    except Exception as exc:
      report["failed"][label] = f"klines: {exc}"
      log(f"  {label}: kline fetch failed ({exc}) — left unscored, re-run to retry")
      continue
    report["barsChecked"] += book.rows_checked
    bars, _starts = book.series()
    try:
      result = replay_protection_stack(
        bars, side_long=side == "long", entry=float(row["entryPrice"]), stop=float(row["stopPrice"]),
        take_profit=float(row["takeProfitPrice"]), cfg=stack_cfg, init_risk=float(init_risk),
        fill_ts=float(fill_ts), end_ts=end_ts, noise_band_r=noise_band, hold_until_ts=hold_until,
        open_until_ts=ts0,
      )
      if result["resolvedBy"] == "expired" and float(result["resolvedTs"]) < end_ts - STACK_MIN_COVERAGE_SEC:
        raise ValueError("bars stop short of the 8h horizon")
    except (KeyError, TypeError, ValueError) as exc:
      report["failed"][label] = f"replay: {exc}"
      continue
    # Legacy rows gain the inputs the live recorder now stores, so the entry-bias splits cover them too.
    for key, value in (("fillTs", fill_ts), ("initRiskPx", init_risk), ("noiseBandR", noise_band),
                       ("holdUntilTs", hold_until), ("entryBias", inputs.get("entry_bias")),
                       ("counterAtEntry", inputs.get("counter_at_entry")),
                       ("htfAligned", inputs.get("htf_aligned"))):
      if row.get(key) is None:
        row[key] = value
    row["stack"] = {
      "stackR": result["stackR"], "resolvedBy": result["resolvedBy"], "resolvedTs": result["resolvedTs"],
      "source": STACK_SOURCE, "scoredTs": int(now_s),
      "preCloseExitSuppressed": bool(result.get("preCloseExitSuppressed")),
    }
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    report["scored"].append({
      "probe": label, "family": row.get("setupFamily"), "takenR": row.get("realizedR"),
      "bracketR": outcome.get("bracketR"), "stackR": result["stackR"], "resolvedBy": result["resolvedBy"],
    })
  return report


EXIT_BRACKET_SOURCE = "futures_1m_resettled"
_RESOLVED_BRACKETS = ("take_profit", "stop", "expired")


def resolve_bracket_on_closes(bars: List[List[float]], *, side_long: bool, entry: float, stop: float,
                              take_profit: float, start_ts: float, end_ts: float) -> Optional[tuple]:
  """``(resolved, bracketR, resolvedTs)`` for a bracket walked over validated 1m futures bars, or None
  when the bars do not reach the horizon (an incomplete window is never resolved).

  Resolution CONVENTION: 1m bar CLOSES, not highs/lows. The live settle resolves on the futures mark
  once per poll, so closes are the comparable reading — legacy and new rows stay on one convention in
  one scoreboard. (On the 09-24 copy the choice is worth ~7.8R on protection exits: close-only +31.25R
  vs high/low +39.05R; most of the old -16.88R "gap" was intrabar resolution, not spot.) Target is
  checked before stop on each close, exactly as `MemoryStore.settle_exit_probes` does; at ``end_ts``
  the position is marked to market at the last close.
  """
  risk = abs(float(entry) - float(stop))
  if risk <= 0:
    return None
  last_px = last_t = None
  for ts, _o, _h, _l, c in bars:
    t_close = ts + BAR_SEC
    if t_close <= start_ts:
      continue
    if t_close > end_ts:
      break
    last_px, last_t = c, t_close
    hit_tp = c >= take_profit if side_long else c <= take_profit
    hit_sl = c <= stop if side_long else c >= stop
    if hit_tp:
      return "take_profit", abs(take_profit - entry) / risk, t_close
    if hit_sl:
      return "stop", -1.0, t_close
  if last_t is None or last_t < end_ts - STACK_MIN_COVERAGE_SEC:
    return None
  mark = (last_px - entry) / risk if side_long else (entry - last_px) / risk
  return "expired", mark, last_t


def resettle_exit_brackets(
  data: Dict[str, Any],
  client: Any,
  *,
  now: Optional[float] = None,
  expire_hours: float = EXIT_PROBE_EXPIRE_HOURS,
  sleep_sec: float = 0.12,
  log: Callable[[str], None] = print,
) -> Dict[str, Any]:
  """Re-resolve the SPOT-settled bracket outcome of every stored exit probe on 1m FUTURES bars, IN PLACE.

  Until 2026-09-25 exit probes were resolved on the spot ticker, which invented bracket outcomes on a
  wide perp/spot basis (ONE-USDT 09-21 00:04 stored as a +1.70R take-profit was stopped on the contract;
  16:00 stored +1.17R expired was -0.24R) — and those rows feed ``otherExits.protection``,
  ``trailByRegime`` / ``trailByMarketState`` (the adaptive-trail evidence) and the audit bracket. Which
  rows: a resolved outcome (take_profit / stop / expired) with NO ``priceSource`` — the live settle now
  stamps one on every resolution, so a re-run never touches a row twice. Rows whose horizon has not
  passed are left for the live settle. The old outcome is kept under ``supersededOutcome``; a row
  whose bars are missing or stop short of its horizon is left untouched and listed. Raises
  ColumnOrderError on any misread bar (the run aborts; nothing is written).
  """
  now_s = float(time.time() if now is None else now)
  horizon = float(expire_hours) * 3600.0
  report: Dict[str, Any] = {"reResolved": 0, "flipped": [], "failed": {}, "notMatured": 0, "barsChecked": 0}
  by_symbol: Dict[str, List[Dict[str, Any]]] = {}
  for row in data.get("exit_probes") or []:
    if not isinstance(row, dict):
      continue
    outcome = row.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("resolved") not in _RESOLVED_BRACKETS:
      continue
    if outcome.get("priceSource"):
      continue
    try:
      ts0 = float(row.get("ts"))
    except (TypeError, ValueError):
      continue
    if ts0 + horizon > now_s - 2 * BAR_SEC:
      report["notMatured"] += 1
      continue
    by_symbol.setdefault(str(row.get("symbol") or ""), []).append(row)
  for symbol in sorted(by_symbol):
    rows = by_symbol[symbol]
    fsym = futures_symbol(symbol)
    if not fsym:
      for row in rows:
        report["failed"][f"{symbol} @ {row.get('ts')}"] = "no futures contract mapping"
      continue
    book = BarBook(client, fsym, sleep_sec=sleep_sec)
    try:
      for row in rows:
        ts0 = float(row["ts"])
        book.ensure(ts0 - BAR_SEC, ts0 + horizon)
    except ColumnOrderError:
      raise
    except Exception as exc:
      for row in rows:
        report["failed"][f"{symbol} @ {row.get('ts')}"] = f"klines: {exc}"
      log(f"  {symbol}: kline fetch failed ({exc}) — exit brackets left as they were, re-run to retry")
      continue
    report["barsChecked"] += book.rows_checked
    bars, _starts = book.series()
    for row in rows:
      label = f"{symbol} @ {int(float(row['ts']))}"
      try:
        got = resolve_bracket_on_closes(
          bars, side_long=str(row.get("positionSide") or "").lower() == "long",
          entry=float(row["entryPrice"]), stop=float(row["stopPrice"]),
          take_profit=float(row["takeProfitPrice"]), start_ts=float(row["ts"]),
          end_ts=float(row["ts"]) + horizon,
        )
      except (KeyError, TypeError, ValueError) as exc:
        report["failed"][label] = f"row: {exc}"
        continue
      if got is None:
        report["failed"][label] = "bars stop short of the 8h horizon"
        continue
      old = copy.deepcopy(row["outcome"])
      resolved, bracket_r, resolved_ts = got
      row["outcome"] = {
        "resolved": resolved, "bracketR": bracket_r, "resolvedTs": resolved_ts,
        "priceSource": EXIT_BRACKET_SOURCE, "resettledTs": int(now_s), "supersededOutcome": old,
      }
      report["reResolved"] += 1
      if old.get("resolved") != resolved:
        report["flipped"].append({"probe": label, "closedBy": row.get("closedBy"), "from": old.get("resolved"),
                                  "fromR": old.get("bracketR"), "to": resolved, "toR": round(bracket_r, 3)})
  return report


def mark_late_exit_probes(data: Dict[str, Any], expire_hours: float = EXIT_PROBE_EXPIRE_HOURS) -> int:
  """Exit probes resolved after their measurable life become ``unmeasured``, as the live settle now rules."""
  stale_after = float(expire_hours) * 3600 + _probe_settle_tolerance_sec(float(expire_hours) * 60.0)
  changed = 0
  for row in data.get("exit_probes") or []:
    if not isinstance(row, dict):
      continue
    outcome = row.get("outcome")
    if not isinstance(outcome, dict) or not outcome.get("resolved") or outcome.get("resolved") == "unmeasured":
      continue
    try:
      late = float(outcome.get("resolvedTs")) - float(row.get("ts")) > stale_after
    except (TypeError, ValueError):
      continue
    if late:
      row["outcome"] = {
        "resolved": "unmeasured", "bracketR": None, "resolvedTs": outcome.get("resolvedTs"),
        "supersededOutcome": copy.deepcopy(outcome),
      }
      changed += 1
  return changed


def _edge_summary(data: Dict[str, Any], cost_pct: float) -> Dict[str, Any]:
  """Per-family signal edge exactly as the bot computes it, read from a throwaway COPY of ``data``.

  The stand-aside and the explore size are reported PER SIDE, because the order path always judges the
  order's side (a thin side on the pooled row at the explore cap): the pooled flag alone once read
  'open' for funding_carry while code refused its shorts (2026-09-25 review).
  """
  from src.edge import (SIDES, annotate_family_stakes, family_explore_factor, safe_family_horizons,
                        signal_edge_stats)

  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "memory.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    store = MemoryStore(str(path))
    probes = store.signal_probes(limit=0)
    stats = signal_edge_stats(probes, cost_pct=cost_pct, family_horizons=safe_family_horizons(store))
  board = annotate_family_stakes(stats)
  out = {"by_horizon": stats.get("by_horizon") or {}, "families": {}}
  for fam, row in sorted((board.get("by_family") or {}).items()):
    se = float(row.get("stderr_pct") or 0.0)
    out["families"][fam] = {
      "n": row.get("n"),
      "net": row.get("net_of_cost_pct"),
      "t": round(float(row.get("net_of_cost_pct") or 0.0) / se, 2) if se > 0 else None,
      "verdict": row.get("verdict"),
      # Closed only when BOTH sides are refused (edge.annotate_family_stakes / open_families).
      "standAside": bool(row.get("standAside")),
      "standAsideBySide": dict(row.get("standAsideBySide") or {}),
      "stakeBySide": dict(row.get("stakeBySide") or {}),
      "exploreBySide": {s: round(float(family_explore_factor(stats, fam, side=s)), 2) for s in SIDES},
    }
  return out


def _print_comparison(before: Dict[str, Any], after: Dict[str, Any], log: Callable[[str], None]) -> None:
  log("\nPer-family signal edge (net of cost, at each family's own horizon; L/S = the stake an entry on "
      "that side gets, ASIDE = refused):")
  log(f"  {'family':14s} {'before':>44s}   {'after':>44s}")
  for fam in sorted(set(before["families"]) | set(after["families"])):
    def fmt(row: Optional[Dict[str, Any]]) -> str:
      if not row:
        return "-"
      sides = []
      for s, tag in (("long", "L"), ("short", "S")):
        if (row.get("standAsideBySide") or {}).get(s):
          sides.append(f"{tag}:ASIDE")
        else:
          sides.append(f"{tag}:{float((row.get('stakeBySide') or {}).get(s) or 0.0):.2f}")
      return f"n={row['n']:>3} net={row['net']:+.3f}% t={row['t']} {' '.join(sides)}"
    log(f"  {fam:14s} {fmt(before['families'].get(fam)):>44s}   {fmt(after['families'].get(fam)):>44s}")
  log("\nBy horizon (mean %):")
  for key in sorted(set(before["by_horizon"]) | set(after["by_horizon"]), key=lambda k: int(k[:-1])):
    b = (before["by_horizon"].get(key) or {}).get("mean_pct")
    a = (after["by_horizon"].get(key) or {}).get("mean_pct")
    log(f"  {key:>5s}  before={b}  after={a}")


def _public_futures_client() -> Any:
  """The production futures client with NO credentials — only public market-data endpoints are used."""
  from src.kucoin import KucoinFuturesClient

  base_url = os.getenv("KUCOIN_FUTURES_BASE_URL", "https://api-futures.kucoin.com")
  cfg = SimpleNamespace(
    kucoin=SimpleNamespace(api_key="", secret="", passphrase=""),
    kucoin_futures=SimpleNamespace(base_url=base_url),
  )
  return KucoinFuturesClient(cfg)


def _write_atomic(path: Path, data: Dict[str, Any]) -> None:
  tmp = path.with_name(path.name + ".resettle.tmp")
  with tmp.open("w", encoding="utf-8") as handle:
    handle.write(json.dumps(data, indent=2))
    handle.flush()
    os.fsync(handle.fileno())
  os.replace(tmp, path)


def _exit_summary(data: Dict[str, Any]) -> Dict[str, Any]:
  from src.edge import exit_discipline_stats

  rows = [r for r in (data.get("exit_probes") or []) if isinstance(r, dict)]
  return exit_discipline_stats(copy.deepcopy(rows[-200:]))


def _print_exit_comparison(before: Dict[str, Any], after: Dict[str, Any], report: Dict[str, Any],
                           log: Callable[[str], None]) -> None:
  log("\nAgent-closed exit probes scored against the LIVE exit stack (1m futures replay, 8h horizon):")
  for r in report["scored"]:
    def f(v: Any) -> str:
      return f"{float(v):+.2f}R" if isinstance(v, (int, float)) else "  n/a"
    log(f"  {r['probe']:<28s} {str(r['family'] or '-'):<14s} taken {f(r['takenR'])}  bracket "
        f"{f(r['bracketR'])}  stack {f(r['stackR'])} ({r['resolvedBy']})")
  if report["failed"]:
    log(f"  NOT scored (bracket-only audit, excluded from the verdict; re-run to retry): {report['failed']}")
  for label, x in (("before", before), ("after", after)):
    log(f"  exitDiscipline {label}: n={x.get('n')} deltaR={x.get('deltaR')} (benchmark "
        f"{x.get('benchmarkR')}R; stack-scored {x.get('stackScored')}, bracket-only audit "
        f"{x.get('legacyBracketScored')}) verdict={x.get('verdict')}")
    others = {k: v.get("deltaR") for k, v in (x.get("otherExits") or {}).items()}
    trail = {k: (v.get("n"), v.get("deltaR")) for k, v in (x.get("trailByRegime") or {}).items()}
    log(f"    otherExits deltaR {label}: {others}   trailByRegime (n, deltaR) {label}: {trail}")


def main(argv: Optional[List[str]] = None, *, client: Any = None, log: Callable[[str], None] = print,
         stack_cfg: Any = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("--memory", required=True, help="path to .agent_memory.json (bot STOPPED)")
  parser.add_argument("--apply", action="store_true", help="write the re-settled file (default: dry run)")
  parser.add_argument("--bot-stopped", action="store_true",
                      help="skip the 'file was modified in the last few minutes' guard")
  parser.add_argument("--cost-pct", type=float, default=0.0014,
                      help="round-trip cost for the before/after report only (fraction, default 0.0014)")
  parser.add_argument("--sleep", type=float, default=0.12, help="pause between API calls, seconds")
  parser.add_argument("--skip-stack", action="store_true",
                      help="do not backfill stackR on agent-closed exit probes")
  args = parser.parse_args(argv)

  path = Path(args.memory).expanduser().resolve()
  if not path.is_file():
    log(f"no such file: {path}")
    return 2
  stat0 = path.stat()
  age = time.time() - stat0.st_mtime
  if args.apply and age < LIVE_FILE_GUARD_SEC and not args.bot_stopped:
    log(f"{path} was modified {age:.0f}s ago — the bot looks like it is RUNNING. Stop it first "
        "(it rewrites this file every poll), or pass --bot-stopped if you are sure.")
    return 3
  data = json.loads(path.read_text(encoding="utf-8"))
  before = _edge_summary(copy.deepcopy(data), args.cost_pct)

  # Before ANY change, so the exit comparison shows the bracket re-resolution and the stack backfill.
  exit_before = _exit_summary(data)

  log(f"Re-settling legacy signal probes in {path} on 1m FUTURES bars ...")
  client = client or _public_futures_client()
  report = resettle(data, client, sleep_sec=args.sleep, log=log)
  after = _edge_summary(copy.deepcopy(data), args.cost_pct)
  stack_report: Dict[str, Any] = {"scored": [], "failed": {}, "barsChecked": 0}
  if not args.skip_stack and any(
    isinstance(r, dict) and str(r.get("closedBy") or "").lower() == "agent" and not r.get("stack")
    for r in data.get("exit_probes") or []
  ):
    cfg = stack_cfg if stack_cfg is not None else effective_stack_cfg(data)
    log(f"Replaying the live exit stack for agent closes (breakeven_fee_pct {cfg.breakeven_fee_pct:.4f}, "
        f"trail {'on' if cfg.trail_enabled else 'off'}, arm {getattr(cfg, 'trail_arm_r', None)}R) ...")
    stack_report = backfill_exit_stacks(data, client, cfg, sleep_sec=args.sleep, log=log)
  exit_after = _exit_summary(data)

  gaps = sorted(abs(g) for _, _, g in report["baseGapPct"])
  log(f"\nrows re-settled: {report['rows']}   stamps: {report['stamps']}   unmeasured: {report['unmeasured']}"
      f"   recovered (missed live): {report['recovered']}   funding-credited: {report['fundingCredited']}"
      f"   bars checked (high/low): {report['barsChecked']}")
  if gaps:
    log(f"base vs futures close at the signal minute: median |gap| {statistics.median(gaps):.3f}%, "
        f"max {gaps[-1]:.3f}% (a large gap means that row's base was not the futures mark)")
    for sym, ts0, g in sorted(report["baseGapPct"], key=lambda x: -abs(x[2]))[:5]:
      if abs(g) > 1.0:
        log(f"  check: {sym} @ {ts0} base differs from the futures close by {g:+.2f}%")
  if report["failedSymbols"]:
    log(f"symbols NOT re-settled (left untouched, re-run to retry): {report['failedSymbols']}")
  log(f"exit probes resolved after their measurable life -> unmeasured: {report['exitProbesMarkedUnmeasured']}")
  eb = report.get("exitBrackets") or {}
  log(f"spot-settled exit brackets re-resolved on 1m futures closes: {eb.get('reResolved', 0)}"
      f" (outcome flipped on {len(eb.get('flipped') or [])}; not yet matured: {eb.get('notMatured', 0)})")
  for f in eb.get("flipped") or []:
    log(f"  flipped: {f['probe']} ({f['closedBy'] or 'unattributed'}) {f['from']} {f['fromR']} -> "
        f"{f['to']} {f['toR']:+.2f}R")
  if eb.get("failed"):
    log(f"  exit brackets NOT re-resolved (left as they were, re-run to retry): {eb['failed']}")
  _print_comparison(before, after, log)
  _print_exit_comparison(exit_before, exit_after, stack_report, log)

  if not args.apply:
    log("\nDRY RUN — nothing written. Re-run with --apply (bot stopped) to save.")
    return 0
  stat1 = path.stat()
  if (stat1.st_mtime, stat1.st_size) != (stat0.st_mtime, stat0.st_size):
    log("the memory file CHANGED while this ran — the bot is running. Nothing written.")
    return 4
  backup = path.with_name(f"{path.name}.pre-resettle-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.bak")
  shutil.copy2(path, backup)
  _write_atomic(path, data)
  log(f"\nWritten. Backup of the original: {backup}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
