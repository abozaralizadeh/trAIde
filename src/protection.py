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
import uuid
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


def decide_protection(
  *,
  side_long: bool,
  avg_entry: float,
  mark: float,
  sl_price: Optional[float],
  peak_fe: float,
  cfg: ProfitProtectionConfig,
) -> Dict[str, Any]:
  """Decide what protective action (if any) a position needs. Pure function.

  Args:
    side_long: True for a long position, False for a short.
    avg_entry: average entry price.
    mark: current mark price.
    sl_price: current protective stop price, or None if no stop is live.
    peak_fe: peak favourable excursion in price terms (>=0 once the trade has been green).
    cfg: thresholds.

  Returns a dict with ``action`` in {"none", "move_breakeven", "close"} plus context.
  """
  if avg_entry <= 0 or mark <= 0:
    return {"action": "none", "reason": "missing price data"}

  fe_now = (mark - avg_entry) if side_long else (avg_entry - mark)

  # P1a — give-back cap: lock the gain once a real run retraces too far.
  if cfg.giveback_pct and cfg.giveback_pct > 0:
    min_fe = cfg.min_favorable_excursion_pct * avg_entry
    if peak_fe >= min_fe and fe_now <= (1.0 - cfg.giveback_pct) * peak_fe:
      return {
        "action": "close",
        "reason": (
          f"gave back >={cfg.giveback_pct:.0%} of peak run "
          f"(peak +{peak_fe:.4f} px, now +{fe_now:.4f} px)"
        ),
        "peakFe": peak_fe,
        "feNow": fe_now,
      }

  # P1b — breakeven ratchet: once +1R (configurable), the trade can't become a loss.
  if sl_price is not None and sl_price > 0:
    risk_dist = (avg_entry - sl_price) if side_long else (sl_price - avg_entry)
    if risk_dist > 0 and peak_fe >= cfg.breakeven_trigger_r * risk_dist:
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
  ) -> None:
    self.cfg = cfg
    self.kucoin_futures = kucoin_futures
    self.notifier = notifier
    self._peak_fe: Dict[str, float] = {}      # futures symbol -> peak favourable excursion (px)
    self._peak_side: Dict[str, bool] = {}     # futures symbol -> side_long (to reset on flip)
    self._tick_cache: Dict[str, float] = {}   # futures symbol -> tick size

  # -- public -------------------------------------------------------------------

  def run(self, snapshot: Any) -> List[Dict[str, Any]]:
    """Evaluate and (unless dry-run) apply protective actions. Returns actions taken."""
    actions: List[Dict[str, Any]] = []
    try:
      if not self.cfg.enabled:
        return actions
      if not self.kucoin_futures or not getattr(snapshot, "futures_enabled", False):
        return actions

      positions = list(getattr(snapshot, "futures_positions", []) or [])
      stops = list(getattr(snapshot, "futures_stop_orders", []) or [])
      open_symbols: set[str] = set()

      for pos in positions:
        if not isinstance(pos, dict):
          continue
        qty = _to_float(pos.get("currentQty")) or 0.0
        if abs(qty) <= 0:
          continue
        fsym = str(pos.get("symbol") or "").strip()
        avg_entry = _to_float(pos.get("avgEntryPrice")) or 0.0
        mark = _to_float(pos.get("markPrice")) or 0.0
        if not fsym or avg_entry <= 0 or mark <= 0:
          continue
        open_symbols.add(fsym)
        side_long = qty > 0

        # Reset peak tracking if the position flipped direction since we last saw it.
        if self._peak_side.get(fsym) != side_long:
          self._peak_side[fsym] = side_long
          self._peak_fe.pop(fsym, None)

        fe_now = (mark - avg_entry) if side_long else (avg_entry - mark)
        peak_fe = max(self._peak_fe.get(fsym, fe_now), fe_now)
        self._peak_fe[fsym] = peak_fe

        sl_price = self._current_stop_price(fsym, side_long, stops)

        decision = decide_protection(
          side_long=side_long,
          avg_entry=avg_entry,
          mark=mark,
          sl_price=sl_price,
          peak_fe=peak_fe,
          cfg=self.cfg,
        )
        action = decision.get("action")
        if action == "none":
          continue

        result = self._apply(fsym, pos, side_long, stops, decision)
        if result:
          actions.append(result)

      # Drop peak state for positions that are no longer open (fresh peak next lifecycle).
      for sym in list(self._peak_fe.keys()):
        if sym not in open_symbols:
          self._peak_fe.pop(sym, None)
          self._peak_side.pop(sym, None)
    except Exception as exc:  # never break the poll loop
      logger.warning("ProtectionManager.run failed (continuing): %s", exc)
    return actions

  # -- internals ----------------------------------------------------------------

  def _current_stop_price(self, fsym: str, side_long: bool, stops: List[Dict[str, Any]]) -> Optional[float]:
    """Find the protective stop-loss price for a position from active stop orders.

    The SL is the loss-side trigger: for a long it triggers below (stop=='down');
    for a short it triggers above (stop=='up'). If several exist, take the furthest
    from price (the true risk). TP tranches sit on the opposite side and are ignored.
    """
    loss_dir = "down" if side_long else "up"
    candidates: List[float] = []
    for o in stops:
      if not isinstance(o, dict):
        continue
      if str(o.get("symbol") or "").strip() != fsym:
        continue
      if str(o.get("stop") or "").lower() != loss_dir:
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
        logger.warning("PROFIT-LOCK closed %s to lock gains — %s", spot_symbol, reason)
        self._notify(f"\U0001f512 <b>Profit-lock</b>\n{spot_symbol}: position closed to lock gains\n{reason}")
      else:
        return None
    except Exception as exc:
      logger.warning("PROFIT-LOCK action %s failed for %s: %s", action, spot_symbol, exc)
      record["error"] = str(exc)
    return record

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
    # 1) cancel the existing loss-side stop(s) so we don't stack protection.
    loss_dir = "down" if side_long else "up"
    cancelled: List[str] = []
    for o in stops:
      if not isinstance(o, dict) or str(o.get("symbol") or "").strip() != fsym:
        continue
      if str(o.get("stop") or "").lower() != loss_dir:
        continue
      oid = str(o.get("id") or o.get("orderId") or "").strip()
      if not oid:
        continue
      try:
        self.kucoin_futures.cancel_order(oid, symbol=fsym)
        cancelled.append(oid)
      except Exception as exc:
        logger.warning("PROFIT-LOCK could not cancel stop %s on %s: %s", oid, fsym, exc)

    # 2) place a fresh reduce-only close stop at breakeven.
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
    return {"cancelled": cancelled, "newStop": be_rounded, "orderId": getattr(resp, "orderId", None)}

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
    # round to the nearest tick, then normalise float noise to the tick's precision
    steps = math.floor(price / tick + 0.5)
    decimals = max(0, -int(math.floor(math.log10(tick)))) if tick < 1 else 0
    return round(steps * tick, decimals)

  def _notify(self, message: str) -> None:
    if not self.notifier:
      return
    try:
      self.notifier.send(message)
    except Exception:
      pass
