"""Code-driven profit protection for live futures positions.

This module enforces two risk rules *in code*, independent of the LLM agent, so a
trade that has already shown decent profit cannot quietly round-trip into a loss
(the ETH long on 2026-06-07/08 peaked at +8.54 USDT and closed at -3.08):

  P1 — Breakeven ratchet + give-back cap
       * Once a position's favourable excursion reaches ``breakeven_trigger_r`` times
         its initial risk (distance to the protective stop), move the stop-loss to a
         fee-adjusted breakeven so the trade can no longer turn into a loss.
       * If, after a *meaningful* run, price gives back ``giveback_pct`` of the peak
         favourable excursion, close the position at market to lock the remaining gain.

The decision logic (:func:`decide_protection`) is pure and unit-tested. Execution goes
through the public :class:`~src.kucoin.KucoinFuturesClient` methods only, and every
public entry point swallows its own exceptions — this runs in the hot poll loop and
must never take the bot down.

All price comparisons are done in *price space* (favourable excursion = how far mark has
travelled from average entry), which avoids contract-multiplier / PnL-source mismatches.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import ProfitProtectionConfig
from .kucoin import KucoinFuturesClient, KucoinFuturesOrderRequest
from .utils import normalize_symbol

logger = logging.getLogger(__name__)


# ── Pure decision helpers (unit-tested) ─────────────────────────────────────────


def _to_float(value: Any) -> Optional[float]:
  try:
    if value is None:
      return None
    return float(value)
  except (TypeError, ValueError):
    return None


def should_close_for_unrealized_loss(
  *,
  unrealized_pnl: Any,
  equity: Any,
  max_loss_equity_fraction: Any,
) -> bool:
  """Return whether an open position has breached its hard equity-loss budget.

  Invalid/missing inputs and a non-positive fraction disable the guard. The comparison is
  deliberately strict: a loss exactly equal to the budget has not yet *exceeded* it.
  """
  pnl = _to_float(unrealized_pnl)
  account_equity = _to_float(equity)
  fraction = _to_float(max_loss_equity_fraction)
  if pnl is None or account_equity is None or fraction is None:
    return False
  if not all(math.isfinite(value) for value in (pnl, account_equity, fraction)):
    return False
  if pnl >= 0 or account_equity <= 0 or fraction <= 0:
    return False
  return -pnl > account_equity * fraction


def decide_protection(
  *,
  side_long: bool,
  avg_entry: float,
  mark: float,
  sl_price: Optional[float],
  peak_fe: float,
  cfg: ProfitProtectionConfig,
  opened_min_ago: Optional[float] = None,
  risk_override: Optional[float] = None,
  hold_until_ts: Optional[float] = None,
  now_ts: Optional[float] = None,
  noise_band_r: Optional[float] = None,
) -> Dict[str, Any]:
  """Decide what protective action (if any) a position needs. Pure function.

  Args:
    side_long: True for a long position, False for a short.
    avg_entry: average entry price.
    mark: current mark price.
    sl_price: current protective stop price, or None if no stop is live.
    peak_fe: peak favourable excursion in price terms (>=0 once the trade has been green).
    cfg: thresholds.
    opened_min_ago: minutes since the position opened, for the early-invalidation cut (None skips it).
    hold_until_ts: epoch seconds before which this trade's thesis has not yet had its scheduled
      chance to pay (funding carry: the next 8h settlement). While it is in the future, the code's own
      *early profit-taking* is suppressed — the position simply keeps the bracket the model gave it.
      The exchange-side stop is never touched by this, so the loss cap is unchanged.
    now_ts: current epoch seconds, for ``hold_until_ts`` (defaults to wall clock).
    noise_band_r: this instrument's noise band expressed in R — i.e. ``1 / stopAtrMult``, the ATR
      distance the entry's own stop was floored to. When given, the trail rides this far behind the
      peak instead of the configured ``trail_distance_r``, so the give-back is set by the symbol's
      actual volatility rather than a fixed fraction. None keeps the configured behaviour.
    risk_override: the ORIGINAL risk distance (entry − initial stop, in price). Once the stop ratchets
      to breakeven the live ``sl_price`` no longer yields a positive risk, which would silently disable
      every R-based rule (trail/give-back/breakeven) exactly when the trade is winning; the caller
      passes the lifecycle's initial risk here so R stays anchored to the trade's real 1R.

  Returns a dict with ``action`` in {"none", "move_breakeven", "close"} plus context.
  """
  if avg_entry <= 0 or mark <= 0:
    return {"action": "none", "reason": "missing price data"}

  # Thesis-horizon hold. Every rule below is tuned for trades that resolve in minutes, which is right
  # for the technical playbooks but silently guts any trade whose edge accrues on a schedule: a funding
  # carry that is breakeven-ratcheted and shaken out 20 minutes in has paid the round-trip cost and
  # collected exactly none of the transfer it was opened for. Holding here costs nothing extra — the
  # stop the model set is still live on the exchange — so the downside stays capped at the planned 1R
  # while the trade keeps the one thing it needs, which is time.
  if hold_until_ts is not None:
    _now = time.time() if now_ts is None else now_ts
    try:
      _remaining = float(hold_until_ts) - float(_now)
    except (TypeError, ValueError):
      _remaining = 0.0
    if _remaining > 0:
      return {
        "action": "none",
        "reason": (
          f"carry hold — {_remaining / 60.0:.0f}min to the funding settlement this trade was opened "
          "for; keeping the model's original bracket (stop unchanged) instead of taking profit early"
        ),
        "holdUntilTs": float(hold_until_ts),
        "holdRemainingMin": round(_remaining / 60.0, 1),
      }

  fe_now = (mark - avg_entry) if side_long else (avg_entry - mark)
  risk_dist = None
  if risk_override is not None and risk_override > 0:
    risk_dist = float(risk_override)          # stable 1R, survives the stop moving to breakeven
  elif sl_price is not None and sl_price > 0:
    rd = (avg_entry - sl_price) if side_long else (sl_price - avg_entry)
    if rd > 0:
      risk_dist = rd

  # P1c — early invalidation: cut a trade that NEVER went meaningfully green and is now ALMOST at its
  # stop. This is a *time/heat stop*, and MAE research (John Sweeney; QuantifiedStrategies) is explicit:
  # the cut must sit OUTSIDE the adverse-excursion band of your WINNING trades, or it stops you out when
  # you're right. The bot's own winners breathe ~0.57R of MAE, so `early_cut_mae_frac` is 0.85 — the cut
  # only fires when price is nearly at the stop (front-running an almost-certain hit / gap), NOT at the
  # 0.6R that once killed live trend trades (e.g. NEAR, cut mid-coil right before a +4% breakout).
  # Trend setups need room to base; this leaves normal pre-breakout heat alone.
  if (
    cfg.early_cut_enabled
    and opened_min_ago is not None and opened_min_ago >= cfg.early_cut_grace_min
    and risk_dist is not None
  ):
    never_worked = peak_fe < cfg.early_cut_min_favorable_pct * avg_entry
    failing = fe_now <= -cfg.early_cut_mae_frac * risk_dist
    if never_worked and failing:
      return {
        "action": "close",
        "reason": (
          f"early invalidation — never reached +{cfg.early_cut_min_favorable_pct:.1%} in {opened_min_ago:.0f}min "
          f"and now {(-fe_now / risk_dist):.0%} to stop (winners work fast; this one didn't)"
        ),
        "feNow": fe_now,
        "peakFe": peak_fe,
      }

  # Trend-adaptive give-back: a trade whose peak favorable run has reached ``trend_runner_r`` × its
  # own risk is a REVEALED trend winner — its behavior proves the trend, so no external regime feed is
  # needed. For such runners the give-back cap loosens (arms later, tolerates a deeper pullback) so a
  # normal trend breather no longer books a small win and misses the rest of the move — the exact
  # failure that turned a correctly-picked ZEC (+354%/mo) into a net loss across 8 lifecycles. Chop
  # (a trade that never ran far) keeps the tight defaults that protect small gains.
  giveback_pct = cfg.giveback_pct
  giveback_arm_r = cfg.giveback_arm_r
  is_runner = False
  if (
    getattr(cfg, "trend_adaptive_enabled", False)
    and risk_dist is not None
    and cfg.trend_runner_r and cfg.trend_runner_r > 0
    and peak_fe >= cfg.trend_runner_r * risk_dist
  ):
    is_runner = True
    giveback_pct = cfg.trend_giveback_pct
    giveback_arm_r = cfg.trend_giveback_arm_r

  # P1a-trail — trailing ratchet (preferred): once the trade arms (>= breakeven_trigger_r × own risk),
  # ratchet the STOP up to lock (peak - trail_distance_r × risk), never below fee-breakeven and never
  # against the trade. Unlike the give-back market-close it does NOT exit on a shallow retrace, so the
  # trade rides through a normal wobble to its bracket TP or trails a genuine runner — fixing the
  # "every winner capped at ~1.1R gross → ~0.4R net after fees" problem. The give-back close below is
  # kept only for when trailing is disabled (backward-compatible).
  trail_arm_r = float(getattr(cfg, "trail_arm_r", cfg.breakeven_trigger_r))
  _band_r = None
  if noise_band_r is not None:
    try:
      _b = float(noise_band_r)
      if math.isfinite(_b) and _b > 0:
        _band_r = _b
    except (TypeError, ValueError):
      _band_r = None
  # A run shorter than one noise band cannot be locked at all: any stop placed inside it is just
  # handing the symbol's own chop a free exit. So the band raises the arm threshold as well as
  # setting the trail distance — below it, the trade keeps its original stop and is left to develop.
  _arm_r = trail_arm_r if _band_r is None else max(trail_arm_r, _band_r)
  if getattr(cfg, "trail_enabled", False) and risk_dist is not None and peak_fe >= _arm_r * risk_dist:
    trail_r = max(0.0, float(getattr(cfg, "trail_distance_r", 0.75)))
    # Ride one NOISE BAND behind the peak, not a fixed slice of risk. A 1.0R trail is wider than the
    # whole move most of these trades ever make, so the lock_frac branch always won and the trade
    # banked a flat 33% of its peak: across 81 closes the book reached +31R of favourable excursion
    # and realised -4.5R, a 35R give-back that dwarfs the entire net loss. One noise band is the same
    # standard the ENTRY stop is floored to, so this tightens the trail without putting it anywhere
    # the symbol's own chop can reach. Only ever narrows the configured distance, never widens it.
    if _band_r is not None:
      trail_r = _band_r
    lock_frac = max(0.0, min(1.0, float(getattr(cfg, "trail_lock_frac", 0.5))))
    be = avg_entry * (1.0 + cfg.breakeven_fee_pct) if side_long else avg_entry * (1.0 - cfg.breakeven_fee_pct)
    # Lock the GREATER of (a) a fraction of the peak run and (b) a fixed R-trail below the peak. (a)
    # captures the mid-size runs (peak 0.6–1.5R) that used to give everything back before reaching the
    # old 1R arm; (b) captures genuine big runners tightly. Floored at fee-breakeven.
    if _band_r is not None:
      # With a measured band the give-back is fully determined: ride exactly one band behind the peak.
      # The lock_frac branch is a heuristic for when we cannot measure the symbol's noise, and here we
      # can — using it would only ever place the stop somewhere the chop can reach.
      lock_fe = peak_fe - trail_r * risk_dist
    else:
      lock_fe = max(lock_frac * peak_fe, peak_fe - trail_r * risk_dist)
    # Only re-place the stop on a meaningful advance (>= 0.25R), so a slowly-creeping peak doesn't
    # cancel+replace the stop every poll — an internal churn guard, not a strategy parameter.
    min_step = 0.25 * risk_dist
    if side_long:
      new_stop = max(be, avg_entry + lock_fe)
      breached = mark <= new_stop           # price already retraced past the trail → lock it now
      improves = (new_stop - sl_price) > min_step if sl_price is not None else True
    else:
      new_stop = min(be, avg_entry - lock_fe)
      breached = mark >= new_stop
      improves = (sl_price - new_stop) > min_step if sl_price is not None else True
    locked_r = (abs(new_stop - avg_entry) / risk_dist) if risk_dist else 0.0
    at_be = abs(new_stop - be) <= 1e-9
    if breached:
      return {
        "action": "close",
        "reason": (
          f"trailing stop hit — price retraced past the lock from peak +{peak_fe:.4f} px; "
          f"locking {'breakeven' if at_be else f'+{locked_r:.1f}R'}"
        ),
        "peakFe": peak_fe,
        "feNow": fe_now,
      }
    if improves:
      return {
        "action": "move_breakeven",
        "stopPrice": new_stop,
        "reason": (
          f"trailing ratchet — stop to {'breakeven' if at_be else f'+{locked_r:.1f}R locked'} "
          f"(peak +{peak_fe:.4f} px, lock {lock_frac:.0%}/{trail_r:.2f}R); let the winner run to TP or trail the trend"
        ),
        "riskDist": risk_dist,
      }
    return {"action": "none", "reason": "trailing stop already at computed level", "feNow": fe_now}

  # P1a — give-back cap (used only when trailing is disabled): lock the gain once a real run retraces
  # too far. Arming is tied to the trade's OWN risk (giveback_arm_r × stop distance): sub-arm wobble is
  # noise the original SL owns.
  if giveback_pct and giveback_pct > 0:
    min_fe = cfg.min_favorable_excursion_pct * avg_entry
    if giveback_arm_r and giveback_arm_r > 0 and risk_dist is not None:
      min_fe = max(min_fe, giveback_arm_r * risk_dist)
    if peak_fe >= min_fe and fe_now <= (1.0 - giveback_pct) * peak_fe:
      return {
        "action": "close",
        "reason": (
          f"gave back >={giveback_pct:.0%} of peak run "
          f"(peak +{peak_fe:.4f} px, now +{fe_now:.4f} px{', trend-runner' if is_runner else ''})"
        ),
        "peakFe": peak_fe,
        "feNow": fe_now,
      }

  # P1b — breakeven ratchet: once +1R (configurable), the trade can't become a loss.
  if risk_dist is not None:
    if peak_fe >= cfg.breakeven_trigger_r * risk_dist:
      be = avg_entry * (1.0 + cfg.breakeven_fee_pct) if side_long else avg_entry * (1.0 - cfg.breakeven_fee_pct)
      already_protective = (sl_price >= be) if side_long else (sl_price <= be)
      if not already_protective:
        return {
          "action": "move_breakeven",
          "stopPrice": be,
          "reason": (
            f"reached {cfg.breakeven_trigger_r:.1f}R "
            f"(risk {risk_dist:.4f} px, peak +{peak_fe:.4f} px) — lock breakeven"
          ),
          "riskDist": risk_dist,
        }

  return {"action": "none", "reason": "no protective action needed", "feNow": fe_now}


_BAR_SEC = 60.0


def _replay_bar(row: Any) -> tuple[float, float, float, float, float]:
  """``(start_s, open, high, low, close)`` from a 1m FUTURES kline row, or ValueError.

  KuCoin FUTURES rows are ``[ts_ms, open, HIGH, LOW, close, ...]``; SPOT rows are
  ``[ts, open, close, high, low]``. An offline replay that mixed the two up once reversed a finding (it
  checked stops against the close instead of the low and missed most stop-outs), so every bar must
  prove its order: high is the row maximum and low the row minimum, or nothing computed from it is
  trusted.
  """
  try:
    ts = float(row[0])
    o, h, l, c = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
  except (TypeError, ValueError, IndexError) as exc:
    raise ValueError(f"unreadable kline row {row!r}: {exc}") from exc
  if not all(math.isfinite(v) for v in (ts, o, h, l, c)):
    raise ValueError(f"non-finite kline row {row!r}")
  if h != max(o, h, l, c) or l != min(o, h, l, c):
    raise ValueError(
      f"kline row {row!r}: high/low are not the row extremes — futures rows are "
      "[ts, open, HIGH, LOW, close]; refusing to replay misread candles"
    )
  if ts > 1e12:
    ts /= 1000.0
  return ts, o, h, l, c


def _replay_close_label(reason: Any) -> str:
  text = str(reason or "").lower()
  if text.startswith("trailing stop hit"):
    return "trail_close"
  if text.startswith("early invalidation"):
    return "early_cut"
  if text.startswith("gave back"):
    return "giveback_close"
  return "protection_close"


def replay_protection_stack(
  bars: Any,
  *,
  side_long: bool,
  entry: float,
  stop: float,
  take_profit: float,
  cfg: ProfitProtectionConfig,
  init_risk: float,
  fill_ts: float,
  end_ts: float,
  noise_band_r: Optional[float] = None,
  hold_until_ts: Optional[float] = None,
  open_until_ts: Optional[float] = None,
) -> Dict[str, Any]:
  """What the LIVE exit stack would have done with a position the model closed early. Pure function.

  The exit scoreboard (edge.exit_discipline_stats) asks "did the model's close beat leaving the position
  to the system?". Until 2026-09-25 the answer was the bare exchange bracket — original stop and TP, or
  an 8h mark — but the system never leaves a position on its bare bracket: breakeven, the noise-band
  trail and the carry hold (``decide_protection``) manage it every poll. On the 5 attributed agent
  closes at the time, bracket-only read -6.23R and this replay about -3.5R; all of the gap was two
  trades (DASH, INJ) whose trail would have exited near +0.3..0.5R long before the TP the bracket
  credited. The bias is REGIME-SIGNED, not a fixed overstatement: in a trend the trail leaks against
  the bracket, so bracket-only makes closes look costly; in chop a run that armed the trail and
  reversed scores -1R on the bracket but ~0R here, so bracket-only flatters closes. Replaying the real
  stack removes the regime from the benchmark instead of correcting for one sample of it.

  Walks 1m FUTURES ``bars`` ([ts, open, high, low, close, ...], ts in ms or s) whose minute starts at or
  after ``fill_ts`` and before ``end_ts``, exactly as the live manager sees a position — one mark per
  poll — with these rules per bar:

  1. The bar's order is checked (high = row max, low = row min) or ValueError: the column-order trap.
  2. The simulated EXCHANGE legs are checked first against the bar's high/low. When the stop and the
     target are both inside one bar the stop is taken (conservative: a 1m bar cannot say which came
     first). A bar that OPENS through the stop fills at the open, the way a stop-market order would.
  3. The peak is the maximum favourable CLOSE excursion so far (from 0), then ``decide_protection``
     runs on the bar close with ``cfg`` (pass ProtectionManager's EFFECTIVE cfg — its breakeven_fee_pct
     is raised to the round-trip cost), the trade's ORIGINAL risk as ``risk_override``, its own noise
     band and carry hold, the age since fill for the early cut, and ``now_ts`` = the bar's close time
     (the moment its close price exists). ``move_breakeven`` moves the simulated stop; ``close`` exits
     at the bar close.
  4. Still open at ``end_ts``: marked to market at the last bar's close (the same horizon the bracket
     probe expires on, so the two benchmarks are comparable).

  ``open_until_ts``: the position is KNOWN to have been open until then (the model closed it at that
  time). An exit the replay would take before it is not taken — the live stack demonstrably did not
  exit, so the replay has diverged from reality (1m last-trade prices vs the live mark) — but the
  stop/peak state keeps evolving, so the counterfactual starts from the state the stack would be in
  at the close. ``preCloseExitSuppressed`` says when that happened.

  Returns GROSS R on the original risk — the same basis as the probe's ``realizedR`` and ``bracketR``
  — as ``{'stackR', 'resolvedBy', 'resolvedTs', 'preCloseExitSuppressed'}``, ``resolvedBy`` one of
  take_profit / stop / trail_stop / trail_close / early_cut / giveback_close / protection_close /
  expired. Known bias: 1m last-trade prices, while the live manager and the MP stops read the mark —
  median |error| ~0.05R on validation, slightly conservative in sum. Raises ValueError on bad input or
  when no bar falls inside the window (the caller retries later).
  """
  e = float(entry)
  sl = float(stop)
  tp = float(take_profit)
  risk = float(init_risk) if init_risk is not None else abs(e - sl)
  fill = float(fill_ts)
  end = float(end_ts)
  if not all(math.isfinite(v) for v in (e, sl, tp, risk, fill, end)):
    raise ValueError("non-finite replay input")
  if e <= 0 or sl <= 0 or tp <= 0 or risk <= 0 or end <= fill:
    raise ValueError("replay needs a positive entry/stop/target/risk and end_ts after fill_ts")
  held_until = float(open_until_ts) if open_until_ts is not None else None
  if held_until is not None and not math.isfinite(held_until):
    held_until = None
  original_sl = sl
  suppressed = False

  def _r(px: float) -> float:
    return ((px - e) if side_long else (e - px)) / risk

  parsed = sorted((_replay_bar(row) for row in (bars or [])), key=lambda b: b[0])
  peak_fe = 0.0
  last_close: Optional[float] = None
  last_close_ts: Optional[float] = None
  seen: set = set()
  for ts, o, h, l, c in parsed:
    if ts in seen or ts < fill or ts >= end:
      continue
    seen.add(ts)
    t_close = ts + _BAR_SEC
    may_exit = held_until is None or t_close > held_until
    if side_long:
      hit_sl, hit_tp = l <= sl, h >= tp
      stop_px = min(o, sl)
    else:
      hit_sl, hit_tp = h >= sl, l <= tp
      stop_px = max(o, sl)
    if hit_sl or hit_tp:
      if may_exit:
        if hit_sl:  # both inside one bar -> the stop (conservative)
          return {
            "stackR": _r(stop_px), "resolvedBy": "stop" if sl == original_sl else "trail_stop",
            "resolvedTs": t_close, "preCloseExitSuppressed": suppressed,
          }
        return {"stackR": _r(tp), "resolvedBy": "take_profit", "resolvedTs": t_close,
                "preCloseExitSuppressed": suppressed}
      suppressed = True
    fe = (c - e) if side_long else (e - c)
    peak_fe = max(peak_fe, fe)
    decision = decide_protection(
      side_long=side_long, avg_entry=e, mark=c, sl_price=sl, peak_fe=peak_fe, cfg=cfg,
      opened_min_ago=(t_close - fill) / 60.0, risk_override=risk,
      hold_until_ts=hold_until_ts, now_ts=t_close, noise_band_r=noise_band_r,
    )
    action = decision.get("action")
    if action == "move_breakeven":
      new_sl = _to_float(decision.get("stopPrice"))
      if new_sl is not None and math.isfinite(new_sl) and new_sl > 0:
        sl = new_sl
    elif action == "close":
      if may_exit:
        return {"stackR": _r(c), "resolvedBy": _replay_close_label(decision.get("reason")),
                "resolvedTs": t_close, "preCloseExitSuppressed": suppressed}
      suppressed = True
    last_close, last_close_ts = c, t_close
  if last_close is None:
    raise ValueError("no futures bars inside the replay window")
  return {"stackR": _r(last_close), "resolvedBy": "expired", "resolvedTs": min(end, float(last_close_ts)),
          "preCloseExitSuppressed": suppressed}


def should_block_chase(
  *,
  close_type: str,
  exit_price: float,
  new_side: str,
  new_price: float,
  buffer_pct: float,
) -> bool:
  """P2 — block re-entering the *same direction* at a *worse* price than a recent win.

  After taking profit on a long, re-longing at/above the exit price is chasing; after a
  short, re-shorting at/below the exit is chasing. Re-entering at a genuinely better
  price (a real pullback) is allowed.
  """
  if exit_price <= 0 or new_price <= 0:
    return False
  ct = (close_type or "").upper()
  closed_long = "LONG" in ct
  closed_short = "SHORT" in ct
  side = (new_side or "").lower()
  if closed_long and side == "buy":
    # worse = not meaningfully below the exit price
    return new_price >= exit_price * (1.0 - buffer_pct)
  if closed_short and side == "sell":
    return new_price <= exit_price * (1.0 + buffer_pct)
  return False


# ── Live manager (thin execution layer) ─────────────────────────────────────────


class ProtectionManager:
  """Runs the profit guards over live futures positions once per poll.

  Reads positions/stops from the already-built snapshot (no extra read calls) and only
  issues write calls when an action is required. Never raises out of :meth:`run`.
  """

  def __init__(
    self,
    cfg: ProfitProtectionConfig,
    kucoin_futures: Optional[KucoinFuturesClient],
    notifier: Any = None,
    emergency_sl_pct: float = 0.0,
    min_rr: float = 1.5,
    max_loss_equity_fraction: float = 0.0,
    breakeven_cost_pct: float = 0.0,
    trade_context_lookup: Optional[Any] = None,
  ) -> None:
    self.cfg = replace(
      cfg,
      breakeven_fee_pct=max(
        float(cfg.breakeven_fee_pct or 0.0),
        max(0.0, float(breakeven_cost_pct or 0.0)),
      ),
    )
    self.kucoin_futures = kucoin_futures
    self.notifier = notifier
    # Safety net: if an open position has NO loss-side stop (e.g. a limit filled between agent runs
    # and the bracket wasn't attached), place an emergency SL this far from entry and a TP at min_rr×
    # that distance — within one poll (<=60s), not the next agent run. 0 disables. The agent still
    # refines the bracket later; this just guarantees no position is ever left naked.
    self.emergency_sl_pct = float(emergency_sl_pct or 0.0)
    self.min_rr = float(min_rr or 1.5)
    # Independent last-resort loss cap. 0 disables it; callers can wire the account's normal
    # per-trade equity budget here so a missing/gapped stop cannot produce an unbounded loss.
    self.max_loss_equity_fraction = max(0.0, _to_float(max_loss_equity_fraction) or 0.0)
    self._peak_fe: Dict[str, float] = {}      # futures symbol -> peak favourable excursion (px)
    self._init_risk: Dict[str, float] = {}    # futures symbol -> original risk distance (entry−initial stop, px)
    self._peak_side: Dict[str, bool] = {}     # futures symbol -> side_long (to reset on flip)
    self._position_lifecycle: Dict[str, tuple[Any, ...]] = {}
    self._tick_cache: Dict[str, float] = {}   # futures symbol -> tick size
    self._naked_since: Dict[str, float] = {}  # futures symbol -> ts first seen open with no stop
    self._emergency_placed_legs: Dict[str, set[str]] = {}
    self._emergency_grace_sec: float = 90.0   # let an atomic/attached bracket appear before we add one
    self._open_since: Dict[str, float] = {}   # futures symbol -> ts first seen open (for early-cut age)
    # Optional callable(futures_symbol, position) -> {"holdUntilTs", "noiseBandR", "initRiskPx",
    # "peakFePx"} describing the trade behind an open position: when its edge is scheduled to accrue
    # (funding settlement on the contract's own clock), how wide its own noise band is, and its
    # original risk and recorded price-space peak for restart safety (position_context.trade_context).
    # Injected rather than read here so this module keeps no memory/exchange dependency and stays
    # unit-testable.
    self._trade_context_lookup = trade_context_lookup

  # -- public -------------------------------------------------------------------

  def run(self, snapshot: Any) -> List[Dict[str, Any]]:
    """Evaluate and (unless dry-run) apply protective actions. Returns actions taken."""
    actions: List[Dict[str, Any]] = []
    try:
      profit_guards_enabled = bool(self.cfg.enabled)
      if not profit_guards_enabled and self.max_loss_equity_fraction <= 0:
        return actions
      if not self.kucoin_futures or not getattr(snapshot, "futures_enabled", False):
        return actions

      positions = list(getattr(snapshot, "futures_positions", []) or [])
      stops = list(getattr(snapshot, "futures_stop_orders", []) or [])
      equity = self._snapshot_equity(snapshot)
      open_symbols: set[str] = set()

      for pos in positions:
        if not isinstance(pos, dict):
          continue
        qty = _to_float(pos.get("currentQty")) or 0.0
        if abs(qty) <= 0:
          continue
        fsym = str(pos.get("symbol") or "").strip()
        if not fsym:
          continue
        open_symbols.add(fsym)
        side_long = qty > 0

        unrealized_pnl = pos.get("unrealisedPnl")
        if unrealized_pnl is None:
          unrealized_pnl = pos.get("unrealizedPnl")
        if should_close_for_unrealized_loss(
          unrealized_pnl=unrealized_pnl,
          equity=equity,
          max_loss_equity_fraction=self.max_loss_equity_fraction,
        ):
          pnl = _to_float(unrealized_pnl) or 0.0
          budget = equity * self.max_loss_equity_fraction
          decision = {
            "action": "close",
            "reason": (
              f"hard unrealized-loss cap — loss {abs(pnl):.4f} exceeded "
              f"{self.max_loss_equity_fraction:.2%} equity budget {budget:.4f}"
            ),
          }
          result = self._apply(fsym, pos, side_long, stops, decision)
          if result:
            actions.append(result)
          continue

        # A configured hard cap remains active independently, but the ratchet, give-back, early-cut,
        # and emergency-bracket rules retain their existing master switch.
        if not profit_guards_enabled:
          continue

        avg_entry = _to_float(pos.get("avgEntryPrice")) or 0.0
        mark = _to_float(pos.get("markPrice")) or 0.0
        if avg_entry <= 0 or mark <= 0:
          continue
        lifecycle = self._position_signature(pos, side_long, qty, avg_entry)
        # A same-side close/reopen, add-on, or partial reduction is a new excursion baseline. Keeping
        # the prior peak against a changed average entry can immediately produce a false give-back.
        # First sighting of this lifecycle by THIS process: the only poll on which a recorded peak may
        # be applied (see the trade-context block below).
        fresh_lifecycle = self._position_lifecycle.get(fsym) != lifecycle
        if fresh_lifecycle:
          self._position_lifecycle[fsym] = lifecycle
          self._peak_side[fsym] = side_long
          self._peak_fe.pop(fsym, None)
          self._init_risk.pop(fsym, None)
          self._emergency_placed_legs.pop(fsym, None)
          self._open_since[fsym] = time.time()
        # Early-cut depends on both observed age and observed MFE. Trusting an exchange age after a
        # restart, with the MFE reset, could call a mature retracing winner "never worked" and close it
        # immediately, so the age clock starts with this process. Since 2026-09-25 the MFE is restored
        # from the recorded lifecycle peak (below) when its key matches, which makes the shorter age
        # purely conservative: it can delay an early cut, never add one. Restoring the age as well
        # would need the lifecycle's first-seen time persisted next to that peak.
        opened_min_ago = (time.time() - self._open_since.setdefault(fsym, time.time())) / 60.0

        fe_now = (mark - avg_entry) if side_long else (avg_entry - mark)
        peak_fe = max(self._peak_fe.get(fsym, fe_now), fe_now)
        self._peak_fe[fsym] = peak_fe

        sl_price = self._current_stop_price(fsym, side_long, stops)

        # Capture the lifecycle's ORIGINAL risk (entry − initial loss-side stop) the first time a
        # below-entry stop is seen, and keep it. Once the stop later ratchets to breakeven the live
        # distance goes to zero, so without this anchor every R-based rule would silently switch off
        # on winners. Only overwrite while still an actual risk stop (below entry for a long).
        if sl_price is not None and sl_price > 0:
          _rd = (avg_entry - sl_price) if side_long else (sl_price - avg_entry)
          if _rd > 0 and fsym not in self._init_risk:
            self._init_risk[fsym] = _rd

        # Safety net: never leave a filled position naked. If no loss-side stop exists, attach an
        # emergency bracket — but only after a short grace so a just-attached (st-orders) bracket has
        # a poll to appear, avoiding a duplicate stop. This closes the "unprotected between the fill
        # and the next agent run" gap (positions were repeatedly found missing protection).
        if sl_price is None and self.emergency_sl_pct > 0:
          first = self._naked_since.setdefault(fsym, time.time())
          if (time.time() - first) >= self._emergency_grace_sec:
            already_placed = set(self._emergency_placed_legs.get(fsym, set()))
            tp_dir = "up" if side_long else "down"
            if self._has_protective_exit_stop(fsym, side_long, stops, tp_dir):
              already_placed.add("takeProfit")
            res = self._ensure_emergency_bracket(
              fsym, pos, side_long, avg_entry, skip_legs=already_placed,
            )
            if res:
              actions.append(res)
              for key, leg in (res.get("legs") or {}).items():
                # Remember a confirmed TP only to avoid stacking another one while retrying a
                # failed SL. A stop must be visible in the next live snapshot; never trust this
                # local cache as proof that critical loss protection is still active.
                if key == "takeProfit" and isinstance(leg, dict) and leg.get("placed"):
                  self._emergency_placed_legs.setdefault(fsym, set()).add(key)
              stop_leg = (res.get("legs") or {}).get("stopLoss") or {}
              # A failed SL must retry on the next poll without another grace period. TP-only
              # success is useful but does not make the position protected.
              if res.get("dryRun") or stop_leg.get("placed"):
                self._naked_since.pop(fsym, None)
            continue  # emergency placement attempt handled this poll
        else:
          self._naked_since.pop(fsym, None)
          self._emergency_placed_legs.pop(fsym, None)

        hold_until = noise_band = None
        if self._trade_context_lookup is not None:
          try:
            _tc = self._trade_context_lookup(fsym, pos) or {}
            hold_until = _tc.get("holdUntilTs")
            noise_band = _tc.get("noiseBandR")
            # Restart safety. Both dicts below live only in this process. The live capture above
            # only fires when a stop is still BELOW entry, so after a restart a winner already
            # ratcheted to breakeven would never regain its 1R anchor and every R-based rule would go
            # silently inert for it. The recorded entry stop is the same number the live capture
            # would have seen on the trade's first poll; the recorded peak can only RAISE the
            # in-memory peak to a level this trade already reached, which is what the trail would
            # already have locked. Live values win whenever they exist.
            _rec_risk = _to_float(_tc.get("initRiskPx"))
            if fsym not in self._init_risk and _rec_risk and _rec_risk > 0:
              self._init_risk[fsym] = _rec_risk
            # The recorded peak is applied ONLY on the poll this process first sees the lifecycle
            # signature — after a restart, the first sighting; in steady state, the poll of an add-on /
            # reduction / flip, where the recorder has just reset to this same poll's excursion, so it
            # is a no-op. After that this process tracks the same polls the recorder does, so a
            # per-poll max() adds nothing, and it is the channel through which any future unit/key bug
            # would become a market close (the Sep 22 seed divided USD by contracts: 10-520,000x on
            # multiplier>1 contracts, dormant only because its key never matched). The value is
            # price-space and keyed by the writer on this same (openTime, side, |qty|, avgEntry)
            # signature, so it can never re-inject a peak the reset above threw away.
            _rec_peak = _to_float(_tc.get("peakFePx"))
            if (
              fresh_lifecycle and _rec_peak is not None and math.isfinite(_rec_peak)
              and _rec_peak > 0 and _rec_peak > peak_fe
            ):
              self._peak_fe[fsym] = _rec_peak
              peak_fe = _rec_peak
          except Exception:  # a lookup failure must never disable the guards
            logger.debug("trade-context lookup failed for %s", fsym, exc_info=True)
            hold_until = noise_band = None

        decision = decide_protection(
          side_long=side_long,
          avg_entry=avg_entry,
          mark=mark,
          sl_price=sl_price,
          peak_fe=peak_fe,
          cfg=self.cfg,
          opened_min_ago=opened_min_ago,
          risk_override=self._init_risk.get(fsym),
          hold_until_ts=hold_until,
          noise_band_r=noise_band,
        )
        action = decision.get("action")
        if action == "none":
          continue

        result = self._apply(fsym, pos, side_long, stops, decision)
        if result:
          actions.append(result)

      # Drop per-position state for symbols no longer open (fresh peak / naked-timer next lifecycle).
      for sym in list(self._peak_fe.keys()):
        if sym not in open_symbols:
          self._peak_fe.pop(sym, None)
          self._peak_side.pop(sym, None)
          self._position_lifecycle.pop(sym, None)
      for sym in list(self._init_risk.keys()):
        if sym not in open_symbols:
          self._init_risk.pop(sym, None)
      for sym in list(self._naked_since.keys()):
        if sym not in open_symbols:
          self._naked_since.pop(sym, None)
      for sym in list(self._open_since.keys()):
        if sym not in open_symbols:
          self._open_since.pop(sym, None)
      for sym in list(self._emergency_placed_legs.keys()):
        if sym not in open_symbols:
          self._emergency_placed_legs.pop(sym, None)
    except Exception as exc:  # never break the poll loop
      logger.warning("ProtectionManager.run failed (continuing): %s", exc)
    return actions

  # -- internals ----------------------------------------------------------------

  def _snapshot_equity(self, snapshot: Any) -> float:
    """Best available positive account-equity figure for the hard loss cap."""
    total = _to_float(getattr(snapshot, "total_usdt", None))
    if total is not None and math.isfinite(total) and total > 0:
      return total
    account = getattr(snapshot, "futures_account", None)
    if isinstance(account, dict):
      for key in ("accountEquity", "marginBalance", "availableBalance"):
        value = _to_float(account.get(key))
        if value is not None and math.isfinite(value) and value > 0:
          return value
    return 0.0

  @staticmethod
  def _position_signature(
    pos: Dict[str, Any], side_long: bool, qty: float, avg_entry: float,
  ) -> tuple[Any, ...]:
    """Stable identity for excursion/age state, including same-side lifecycle changes."""
    opened = None
    for key in ("openingTimestamp", "openingTime", "openTime", "createdAt"):
      value = pos.get(key)
      if value not in (None, ""):
        try:
          opened = int(float(value))
        except (TypeError, ValueError):
          opened = str(value)
        break
    return (opened, bool(side_long), float(qty), float(avg_entry))

  @staticmethod
  def _order_flag(value: Any) -> bool:
    return value is True or str(value or "").strip().lower() in {"1", "true", "yes"}

  def _is_protective_exit_stop(
    self,
    order: Dict[str, Any],
    fsym: str,
    side_long: bool,
    stop_direction: str,
  ) -> bool:
    """Only trust conditional orders that can actually reduce/close this position."""
    if not isinstance(order, dict):
      return False
    if str(order.get("symbol") or "").strip() != fsym:
      return False
    if str(order.get("stop") or "").strip().lower() != stop_direction:
      return False
    expected_side = "sell" if side_long else "buy"
    if str(order.get("side") or "").strip().lower() != expected_side:
      return False
    return self._order_flag(order.get("reduceOnly")) or self._order_flag(order.get("closeOrder"))

  def _has_protective_exit_stop(
    self,
    fsym: str,
    side_long: bool,
    stops: List[Dict[str, Any]],
    stop_direction: str,
  ) -> bool:
    return any(
      self._is_protective_exit_stop(order, fsym, side_long, stop_direction)
      for order in stops
    )

  def _current_stop_price(self, fsym: str, side_long: bool, stops: List[Dict[str, Any]]) -> Optional[float]:
    """Find the protective stop-loss price for a position from active stop orders.

    The SL is the loss-side trigger: for a long it triggers below (stop=='down');
    for a short it triggers above (stop=='up'). If several exist, take the furthest
    from price (the true risk). TP tranches sit on the opposite side and are ignored.
    """
    loss_dir = "down" if side_long else "up"
    candidates: List[float] = []
    for o in stops:
      if not self._is_protective_exit_stop(o, fsym, side_long, loss_dir):
        continue
      sp = _to_float(o.get("stopPrice"))
      if sp and sp > 0:
        candidates.append(sp)
    if not candidates:
      return None
    return min(candidates) if side_long else max(candidates)

  def _apply(
    self,
    fsym: str,
    pos: Dict[str, Any],
    side_long: bool,
    stops: List[Dict[str, Any]],
    decision: Dict[str, Any],
  ) -> Optional[Dict[str, Any]]:
    spot_symbol = normalize_symbol(fsym)
    action = decision.get("action")
    reason = decision.get("reason", "")
    record = {"symbol": spot_symbol, "futuresSymbol": fsym, "action": action, "reason": reason}

    if self.cfg.dry_run:
      logger.warning("PROFIT-LOCK [DRY-RUN] %s on %s — %s", action, spot_symbol, reason)
      record["dryRun"] = True
      self._notify(f"\U0001f9ea <b>Profit-lock (dry-run)</b>\n{spot_symbol}: would {action} — {reason}")
      return record

    try:
      if action == "move_breakeven":
        record["result"] = self._move_stop_to_breakeven(fsym, pos, side_long, stops, float(decision["stopPrice"]))
        logger.warning("PROFIT-LOCK moved SL to breakeven on %s — %s", spot_symbol, reason)
        self._notify(f"\U0001f6e1 <b>Profit-lock</b>\n{spot_symbol}: stop moved to breakeven\n{reason}")
      elif action == "close":
        record["result"] = self._close_position(fsym, pos, side_long)
        logger.warning("PROFIT-LOCK closed %s — %s", spot_symbol, reason)
        self._notify(f"\U0001f512 <b>Profit-lock</b>\n{spot_symbol}: position closed\n{reason}")
      else:
        return None
    except Exception as exc:
      logger.warning("PROFIT-LOCK action %s failed for %s: %s", action, spot_symbol, exc)
      record["error"] = str(exc)
    return record

  def _ensure_emergency_bracket(
    self,
    fsym: str,
    pos: Dict[str, Any],
    side_long: bool,
    avg_entry: float,
    *,
    skip_legs: Optional[set[str]] = None,
  ) -> Optional[Dict[str, Any]]:
    """Attach a baseline SL (+TP at min_rr) to a position that has no protective stop.

    A last-resort net, not the agent's considered bracket: it caps risk at ``emergency_sl_pct`` from
    entry so an unbracketed fill can't run away, and the agent replaces it with structure-based levels
    on its next run. Reduce-only close stops, same mechanism as the breakeven ratchet.
    """
    spot_symbol = normalize_symbol(fsym)
    sl = avg_entry * (1 - self.emergency_sl_pct) if side_long else avg_entry * (1 + self.emergency_sl_pct)
    tp = avg_entry * (1 + self.emergency_sl_pct * self.min_rr) if side_long else avg_entry * (1 - self.emergency_sl_pct * self.min_rr)
    sl_r = self._round_to_tick(fsym, sl)
    tp_r = self._round_to_tick(fsym, tp)
    reason = f"position had no stop — emergency bracket ({self.emergency_sl_pct:.1%} SL, {self.min_rr:.1f}R TP)"
    record = {"symbol": spot_symbol, "futuresSymbol": fsym, "action": "emergency_bracket", "stopLoss": sl_r, "takeProfit": tp_r, "reason": reason}

    if self.cfg.dry_run:
      logger.warning("PROFIT-LOCK [DRY-RUN] emergency bracket on %s — %s", spot_symbol, reason)
      record["dryRun"] = True
      self._notify(f"\U0001f9ea <b>Profit-lock (dry-run)</b>\n{spot_symbol}: would attach emergency bracket — {reason}")
      return record

    common = self._order_common(pos)
    exit_side = "sell" if side_long else "buy"
    loss_dir = "down" if side_long else "up"
    tp_dir = "up" if side_long else "down"
    leg_results: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    skip_legs = set(skip_legs or set())
    for key, label, sdir, sp in (
      ("stopLoss", "SL", loss_dir, sl_r),
      ("takeProfit", "TP", tp_dir, tp_r),
    ):
      if key in skip_legs:
        leg_results[key] = {"placed": True, "existing": True}
        continue
      try:
        req = KucoinFuturesOrderRequest(
          symbol=fsym, side=exit_side, type="market",
          leverage=common["leverage"], size=common["size"],
          clientOid=f"{fsym.lower()}-emrg{label.lower()}-{uuid.uuid4().hex[:10]}",
          stop=sdir, stopPriceType="MP", stopPrice=f"{sp}",
          reduceOnly=True, closeOrder=True, marginMode=common["marginMode"],
          autoDeposit=False if common["marginMode"] == "ISOLATED" else None,
        )
        resp = self.kucoin_futures.place_order(req)
        order_id = self._confirmed_order_id(resp)
        if not order_id:
          raise RuntimeError(f"{label} placement was not confirmed (missing orderId)")
        leg_results[key] = {"placed": True, "orderId": order_id}
      except Exception as exc:
        errors[key] = str(exc)
        leg_results[key] = {"placed": False, "error": str(exc)}

    record["legs"] = leg_results
    placed = [key for key, result in leg_results.items() if result.get("placed")]
    if not errors:
      logger.warning("PROFIT-LOCK attached EMERGENCY bracket on %s (no stop existed) — SL %s / TP %s", spot_symbol, sl_r, tp_r)
      self._notify(f"\U0001f6e1 <b>Profit-lock</b>\n{spot_symbol}: emergency bracket attached (was unprotected)\nSL {sl_r} · TP {tp_r}")
      record["result"] = "placed"
    else:
      record["errors"] = errors
      record["error"] = "; ".join(f"{key}: {error}" for key, error in errors.items())
      if placed:
        record["result"] = "partial"
        logger.warning(
          "Emergency bracket PARTIAL for %s — placed %s; failed %s",
          spot_symbol, ", ".join(placed), record["error"],
        )
        self._notify(
          f"\u26a0\ufe0f <b>Profit-lock</b>\n{spot_symbol}: emergency bracket only partially attached\n"
          f"Placed: {', '.join(placed)} · Failed: {record['error']}"
        )
      else:
        record["result"] = "failed"
        logger.warning("Emergency bracket failed for %s — %s", spot_symbol, record["error"])
        self._notify(f"\u26a0\ufe0f <b>Profit-lock</b>\n{spot_symbol}: emergency bracket failed\n{record['error']}")
    return record

  @staticmethod
  def _confirmed_order_id(response: Any) -> str:
    """Extract the exchange acknowledgement used before retiring old protection."""
    if isinstance(response, dict):
      value = response.get("orderId")
    else:
      value = getattr(response, "orderId", None)
    return str(value or "").strip()

  def _order_common(self, pos: Dict[str, Any]) -> Dict[str, Any]:
    """Leverage / margin-mode fields shared by the protective orders."""
    lev = _to_float(pos.get("realLeverage")) or _to_float(pos.get("leverage")) or 1.0
    lev = max(1, int(round(lev)))
    cross = pos.get("crossMode")
    margin_mode = "CROSS" if (cross is None or cross) else "ISOLATED"
    qty = abs(_to_float(pos.get("currentQty")) or 0.0)
    size = max(1, int(round(qty)))
    return {"leverage": str(lev), "size": str(size), "marginMode": margin_mode}

  def _move_stop_to_breakeven(
    self,
    fsym: str,
    pos: Dict[str, Any],
    side_long: bool,
    stops: List[Dict[str, Any]],
    be_price: float,
  ) -> Dict[str, Any]:
    # Place and confirm the replacement first. A duplicate protective stop is safer than even a
    # brief naked window, and an unconfirmed placement must leave every old stop untouched.
    loss_dir = "down" if side_long else "up"
    common = self._order_common(pos)
    exit_side = "sell" if side_long else "buy"
    be_rounded = self._round_to_tick(fsym, be_price)
    req = KucoinFuturesOrderRequest(
      symbol=fsym,
      side=exit_side,
      type="market",
      leverage=common["leverage"],
      size=common["size"],
      clientOid=f"{fsym.lower()}-belock-{uuid.uuid4().hex[:12]}",
      stop=loss_dir,
      stopPriceType="MP",
      stopPrice=f"{be_rounded}",
      reduceOnly=True,
      closeOrder=True,
      marginMode=common["marginMode"],
      autoDeposit=False if common["marginMode"] == "ISOLATED" else None,
    )
    resp = self.kucoin_futures.place_order(req)
    new_order_id = self._confirmed_order_id(resp)
    if not new_order_id:
      raise RuntimeError("breakeven stop placement was not confirmed (missing orderId); old stop retained")

    cancelled: List[str] = []
    cancel_errors: List[Dict[str, str]] = []
    for o in stops:
      if not self._is_protective_exit_stop(o, fsym, side_long, loss_dir):
        continue
      oid = str(o.get("id") or o.get("orderId") or "").strip()
      if not oid:
        continue
      try:
        self.kucoin_futures.cancel_order(oid, symbol=fsym)
        cancelled.append(oid)
      except Exception as exc:
        logger.warning("PROFIT-LOCK could not cancel stop %s on %s: %s", oid, fsym, exc)
        cancel_errors.append({"orderId": oid, "error": str(exc)})

    result: Dict[str, Any] = {
      "cancelled": cancelled,
      "newStop": be_rounded,
      "orderId": new_order_id,
    }
    if cancel_errors:
      result["cancelErrors"] = cancel_errors
    return result

  def _close_position(self, fsym: str, pos: Dict[str, Any], side_long: bool) -> Dict[str, Any]:
    common = self._order_common(pos)
    exit_side = "sell" if side_long else "buy"
    req = KucoinFuturesOrderRequest(
      symbol=fsym,
      side=exit_side,
      type="market",
      leverage=common["leverage"],
      size=common["size"],
      clientOid=f"{fsym.lower()}-pglock-{uuid.uuid4().hex[:12]}",
      reduceOnly=True,
      closeOrder=True,
      marginMode=common["marginMode"],
      autoDeposit=False if common["marginMode"] == "ISOLATED" else None,
    )
    resp = self.kucoin_futures.place_order(req)
    return {"closed": True, "orderId": getattr(resp, "orderId", None)}

  def _round_to_tick(self, fsym: str, price: float) -> float:
    tick = self._tick_cache.get(fsym)
    if tick is None:
      tick = 0.0
      try:
        detail = self.kucoin_futures.get_contract_detail(fsym) or {}
        tick = _to_float(detail.get("tickSize")) or 0.0
      except Exception as exc:
        logger.warning("PROFIT-LOCK tickSize lookup failed for %s: %s", fsym, exc)
      self._tick_cache[fsym] = tick
    if not tick or tick <= 0:
      return price
    # Decimal arithmetic supports arbitrary increments such as 0.25 and 0.0025. Inferring decimal
    # places from log10(tick) corrupts those ticks (for example, 100.25 became 100.2).
    try:
      tick_decimal = Decimal(str(tick))
      price_decimal = Decimal(str(price))
      if not tick_decimal.is_finite() or not price_decimal.is_finite() or tick_decimal <= 0:
        return price
      steps = (price_decimal / tick_decimal).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
      return float(steps * tick_decimal)
    except (InvalidOperation, ValueError):
      return price

  def _notify(self, message: str) -> None:
    if not self.notifier:
      return
    try:
      self.notifier.send(message)
    except Exception:
      pass
