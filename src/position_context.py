"""Facts about the trade behind an open futures position, for its exit mechanics.

Moved out of ``main.trading_loop`` (where it was the closure ``_trade_context``) on 2026-09-25 so it can
be executed in tests with exchange-shaped payloads rather than string-matched in source: the Sep 22
restart seed passed its only test (a source grep) while its key could never match in production.
``main`` wraps it as ProtectionManager's ``trade_context_lookup``; later work reuses it to show the
model its own entry thesis per open position.

Keeps ProtectionManager free of any memory/exchange dependency: everything here is looked up and
handed over as plain numbers.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Any, Callable, Dict, Optional, Tuple

from .memory import peak_fe_key, position_open_time
from .regime import carry_hold_deadline, first_settlement_after, next_funding_settlement
from .utils import normalize_symbol

logger = logging.getLogger(__name__)

# fsym -> (next_settlement_ts, interval_sec), epoch SECONDS, or None when unknown. Must never raise.
FundingClock = Callable[[str], Optional[Tuple[float, float]]]


def _is_funding_carry(ctx: Any) -> bool:
  return isinstance(ctx, dict) and str(ctx.get("setupFamily") or "").strip().lower() == "funding_carry"


def trade_context(
  memory: Any,
  fsym: str,
  pos: Any,
  now_ts: float,
  *,
  funding_clock: FundingClock | None = None,
) -> Dict[str, Any]:
  """What an open position's exit mechanics need to know about the trade behind it.

  ``holdUntilTs`` — a declared funding-carry trade is held to the first settlement after its fill, on
  the CONTRACT'S OWN funding clock (``funding_clock``; see regime.carry_hold_deadline for the fallback
  order). None for every other playbook, leaving their management completely unchanged. The clock is
  only consulted for carry positions, so no other position ever costs a network call.
  ``noiseBandR`` — the entry's own ATR stop multiple, inverted into R, so the trail can ride one
  noise band behind the peak instead of a fixed slice of risk.
  ``initRiskPx`` — the trade's ORIGINAL stop distance. ProtectionManager captures it only when it
  sees a stop BELOW entry, so after a restart a winner already ratcheted to breakeven would never
  regain its 1R anchor and every R-based rule would go silently inert for the rest of that position.
  ``peakFePx`` — the peak favourable excursion recorded for THIS exact lifecycle, in price units, so
  a restart does not erase it. Returned only when the persisted ``peakFeKey`` equals the key built
  here from the live payload with the SAME helper the writer uses (memory.peak_fe_key), i.e. the
  identity ProtectionManager resets its own peak on.

  Two bugs removed here on 2026-09-25 (the seed had never fired): the side came from KuCoin's raw
  ``positionSide`` ('BOTH' in one-way mode), so the key never matched the writer's qty-derived
  'long'/'short'; and the peak was ``peakPnl / |contracts|`` — price x contract multiplier, 10x on
  H/WIF/ONDO and 520,000x on PEPE — which would have market-closed multiplier>1 winners on ordinary
  polls the moment anyone fixed only the key. Side now comes from the sign of ``currentQty``, exactly
  as the writer derives it, and no USD-to-price conversion exists any more.

  Never raises: a failure returns whatever was built so far.
  """
  out: Dict[str, Any] = {}
  try:
    if not isinstance(pos, dict):
      return out
    try:
      qty = float(pos.get("currentQty") or 0.0)
    except (TypeError, ValueError):
      qty = 0.0
    if not math.isfinite(qty):
      qty = 0.0
    side = "long" if qty > 0 else ("short" if qty < 0 else None)
    opened = position_open_time(pos)
    ctx = memory.entry_context_for_position(fsym, opened, side)

    next_ts = interval = None
    first_paid = unpaid_asof = None
    if funding_clock is not None and _is_funding_carry(ctx):
      try:
        clock = funding_clock(fsym)
        if clock:
          next_ts, interval = clock
      except Exception as exc:  # the clock is supposed to be total; belt and braces
        logger.warning("CARRY CLOCK: lookup for %s failed (%s) — holding on the stamped/8h clock", fsym, exc)
        next_ts = interval = None
      # Has a payment actually happened since the fill? The exchange's own history, when the loop has
      # read it — the only exact answer once KuCoin changed the interval after the fill (W2).
      since = getattr(funding_clock, "settlement_since", None)
      fill_ts = ctx.get("fillTs") or ctx.get("ts")
      if callable(since) and fill_ts is not None:
        try:
          first_paid, unpaid_asof = since(fsym, fill_ts)
        except Exception as exc:
          logger.warning("CARRY HISTORY: lookup for %s failed (%s) — judging from the clocks", fsym, exc)
          first_paid = unpaid_asof = None
    out["holdUntilTs"] = carry_hold_deadline(
      ctx, now_ts, next_settlement_ts=next_ts, interval_sec=interval,
      first_paid_ts=first_paid, unpaid_as_of_ts=unpaid_asof,
    )

    if isinstance(ctx, dict):
      try:
        atr_mult = float(ctx.get("stopAtrMult") or 0.0)
      except (TypeError, ValueError):
        atr_mult = 0.0
      if atr_mult > 0:
        out["noiseBandR"] = 1.0 / atr_mult
      try:
        _e = float(ctx.get("fillPrice") or ctx.get("entryPrice") or 0.0)
        _sl = float(ctx.get("stopLossPrice") or 0.0)
        if _e > 0 and _sl > 0 and abs(_e - _sl) > 0:
          out["initRiskPx"] = abs(_e - _sl)
      except (TypeError, ValueError):
        pass

    # Recorded peak for THIS exact lifecycle signature only, already in price units.
    try:
      key = peak_fe_key(opened, qty, pos.get("avgEntryPrice"))
      ext = memory.get_position_extremes(normalize_symbol(fsym)) or {}
      if key is not None and ext.get("peakFeKey") == key:
        peak = float(ext.get("peakFePx"))
        if math.isfinite(peak) and peak > 0:
          out["peakFePx"] = peak
    except Exception:
      logger.debug("recorded-peak lookup failed for %s", fsym, exc_info=True)
  except Exception:
    logger.debug("trade-context lookup failed for %s", fsym, exc_info=True)
  return out


def carry_refresh_targets(memory: Any, positions: Any) -> Dict[str, float]:
  """``{fsym: fill_ts}`` for every held position whose entry was a declared funding_carry.

  What the poll loop hands ``main._FundingClock.refresh`` OUTSIDE ``order_lock``, so ProtectionManager's
  lookup (under the lock) only ever reads the clock's cache. Matched exactly as ``trade_context``
  matches the entry (same helpers, same qty-derived side). Never raises; a bad row is skipped.
  """
  out: Dict[str, float] = {}
  for pos in positions or []:
    try:
      if not isinstance(pos, dict):
        continue
      fsym = str(pos.get("symbol") or "").strip()
      qty = float(pos.get("currentQty") or 0.0)
      if not fsym or not math.isfinite(qty) or qty == 0:
        continue
      side = "long" if qty > 0 else "short"
      ctx = memory.entry_context_for_position(fsym, position_open_time(pos), side)
      if not _is_funding_carry(ctx):
        continue
      fill = float(ctx.get("fillTs") or ctx.get("ts") or 0.0)
      if math.isfinite(fill) and fill > 0:
        out[fsym] = fill
    except Exception:
      logger.debug("carry refresh target skipped for %s", pos, exc_info=True)
  return out


# ── The model's view of its own trade (entry thesis) and the exit-probe tags (2026-09-25) ────────────

# (label shown to the model, key in entryContext.regime). Stamped at order placement from the same
# analyze_market_context read the gates used, so it is exactly what the trade was entered into.
_BIAS_KEYS = (("15m", "intraday_bias_15m"), ("1h", "intraday_bias_1h"), ("4h", "intraday_bias_4h"),
              ("1D", "daily_bias"))


def _num(value: Any) -> Optional[float]:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return None
  return out if math.isfinite(out) else None


def _pos_num(value: Any) -> Optional[float]:
  out = _num(value)
  return out if out is not None and out > 0 else None


def entry_bias(ctx: Any) -> Optional[Dict[str, str]]:
  """``{15m, 1h, 4h, 1D}`` trend biases recorded on the entry (``entryContext.regime``), or None."""
  regime = ctx.get("regime") if isinstance(ctx, dict) else None
  if not isinstance(regime, dict):
    return None
  out = {label: str(regime.get(key)).strip().lower() for label, key in _BIAS_KEYS if regime.get(key)}
  return out or None


def bias_tags(bias: Any, side: Any) -> Tuple[Optional[bool], Optional[bool]]:
  """``(counterAtEntry, htfAligned)`` for a position on ``side`` entered into ``bias``.

  counterAtEntry — the entry's 15m AND 1h biases BOTH opposed the side: the trade was opened into an
  intraday counter-trend on purpose (fade_extreme, range_edge and funding_carry usually are).
  htfAligned — the entry's 4h AND 1D biases both agreed with the side. None when a needed bias is
  missing, so an untagged row never masquerades as a "no".
  """
  s = str(side or "").strip().lower()
  if s not in ("long", "short") or not isinstance(bias, dict):
    return None, None
  with_trend = "bullish" if s == "long" else "bearish"
  against = "bearish" if s == "long" else "bullish"
  b15, b1h, b4h, b1d = (str(bias.get(k) or "").lower() for k in ("15m", "1h", "4h", "1D"))
  counter = (b15 == against and b1h == against) if (b15 and b1h) else None
  htf = (b4h == with_trend and b1d == with_trend) if (b4h and b1d) else None
  return counter, htf


def exit_probe_inputs(ctx: Any, side: Any) -> Dict[str, Any]:
  """The keyword arguments ``memory.record_exit_probe`` needs to replay the live exit stack later.

  fill_ts / init_risk_px (|fill - ORIGINAL stop|) / noise_band_r (1/stopAtrMult) / hold_until_ts (the
  carry hold's first settlement after the fill: the clock stamped on the entry, else the 8h grid — the
  same fallback order the live hold uses when the exchange clock is unreadable) plus the entry-bias
  tags. Every value is the trade's own data; missing ones are None. Never raises.
  """
  out: Dict[str, Any] = {}
  try:
    if not isinstance(ctx, dict):
      return out
    fill_ts = _pos_num(ctx.get("fillTs"))
    e = _pos_num(ctx.get("fillPrice")) or _pos_num(ctx.get("entryPrice"))
    sl = _pos_num(ctx.get("stopLossPrice"))
    atr_mult = _pos_num(ctx.get("stopAtrMult"))
    out["fill_ts"] = fill_ts
    out["init_risk_px"] = abs(e - sl) if (e and sl and e != sl) else None
    out["noise_band_r"] = (1.0 / atr_mult) if atr_mult else None
    # Anchored on the FILL: carry_hold_deadline returns the first settlement after it (never None for a
    # carry trade when "now" is the fill itself), which is the hold the replay must reproduce.
    out["hold_until_ts"] = carry_hold_deadline(ctx, fill_ts) if fill_ts else None
    bias = entry_bias(ctx)
    counter, htf = bias_tags(bias, side)
    out["entry_bias"] = bias
    out["counter_at_entry"] = counter
    out["htf_aligned"] = htf
  except Exception:
    logger.debug("exit-probe inputs failed", exc_info=True)
  return out


def _utc_iso(ts: Any) -> Optional[str]:
  t = _num(ts)
  if t is None or t <= 0:
    return None
  return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def entry_thesis(
  memory: Any,
  pos: Any,
  now_ts: float,
  *,
  funding_clock: FundingClock | None = None,
) -> Optional[Dict[str, Any]]:
  """The trade behind an open futures position, as the MODEL should see it — or None.

  Why (2026-09-25): the prompt's STEP 1b said "fresh 15m AND 1h biases both oppose the position" means a
  confirmed reversal, and the model's position view was raw exchange data, so it could not tell a NEW
  flip from the condition it had entered into. 27 of 31 historical fires were true at the fill:
  counter-intraday playbooks (fade_extreme, range_edge, funding_carry) closed on their own entry premise
  at the next audit — DASH and KCS were carry longs entered with 15m/1h already bearish. This block
  gives the model the entry's own record to compare against. It is data only: no gate, no nudge.

  Fields: setupFamily, entryConfidence, entryBias {15m, 1h, 4h, 1D} (from entryContext.regime),
  fillPrice / plannedStop / plannedTp, currentR (signed (mark - fill) / ORIGINAL risk), noiseBandR
  (1/stopAtrMult: one noise band in R) and entryAtr15mPct, heldMin. For funding_carry also:
  holdUntil (UTC ISO of the settlement the code's carry hold waits for; None once it has passed),
  minutesToSettlement (the next one on the contract's clock), intervalHours, fundingRateAtEntry /
  fundingRateNow (per settlement; longs receive -rate) and carryPerSettlementR (what one settlement
  pays this side, in R of this trade's own risk — self-scaling, and it shows how small the transfer is
  against the stop).

  None when the entry cannot be matched (no context → no block, never a guess). Never raises.
  """
  try:
    if not isinstance(pos, dict):
      return None
    fsym = str(pos.get("symbol") or "").strip()
    qty = _num(pos.get("currentQty")) or 0.0
    side = "long" if qty > 0 else ("short" if qty < 0 else None)
    if not fsym or side is None:
      return None
    ctx = memory.entry_context_for_position(fsym, position_open_time(pos), side)
    if not isinstance(ctx, dict):
      return None
    now = float(now_ts)
    tc = trade_context(memory, fsym, pos, now, funding_clock=funding_clock)
    sign = 1.0 if side == "long" else -1.0
    fill = _pos_num(ctx.get("fillPrice")) or _pos_num(ctx.get("entryPrice"))
    init_risk = _pos_num(tc.get("initRiskPx"))
    mark = _pos_num(pos.get("markPrice"))
    regime = ctx.get("regime") if isinstance(ctx.get("regime"), dict) else {}
    family = str(ctx.get("setupFamily") or "").strip().lower() or None
    thesis: Dict[str, Any] = {
      "setupFamily": family,
      "entryConfidence": _num(ctx.get("confidence")),
      "entryBias": entry_bias(ctx),
      "fillPrice": fill,
      "plannedStop": _pos_num(ctx.get("stopLossPrice")),
      "plannedTp": _pos_num(ctx.get("takeProfitPrice")),
      "currentR": round(sign * (mark - fill) / init_risk, 2) if (mark and fill and init_risk) else None,
      "noiseBandR": round(float(tc["noiseBandR"]), 3) if _pos_num(tc.get("noiseBandR")) else None,
      "entryAtr15mPct": _num(regime.get("intraday_atr_pct")),
    }
    fill_ts = _pos_num(ctx.get("fillTs"))
    if fill_ts:
      thesis["heldMin"] = round(max(0.0, now - fill_ts) / 60.0)
    if family == "funding_carry":
      hold = _pos_num(tc.get("holdUntilTs"))
      thesis["holdUntil"] = _utc_iso(hold) if hold else None
      thesis["carryHoldActive"] = bool(hold and hold > now)
      next_ts = interval = None
      if funding_clock is not None:
        try:
          clock = funding_clock(fsym)   # cache only: the poll loop refreshes held carry clocks
          if clock:
            next_ts, interval = clock
        except Exception:
          next_ts = interval = None
      stamp = ctx.get("funding") if isinstance(ctx.get("funding"), dict) else {}
      if next_ts is None or interval is None:
        next_ts, interval = _pos_num(stamp.get("nextSettlementTs")), _pos_num(stamp.get("intervalSec"))
      nxt = first_settlement_after(now, next_ts, interval) if (next_ts and interval) else None
      if nxt is None:
        nxt = next_funding_settlement(now)
      thesis["minutesToSettlement"] = round((nxt - now) / 60.0, 1) if nxt else None
      thesis["intervalHours"] = round(float(interval) / 3600.0, 3) if _pos_num(interval) else None
      rate_now = None
      rate_fn = getattr(funding_clock, "rate", None)
      if callable(rate_fn):
        try:
          rate_now = _num(rate_fn(fsym))
        except Exception:
          rate_now = None
      rate_entry = _num(stamp.get("rate"))
      thesis["fundingRateAtEntry"] = rate_entry
      thesis["fundingRateNow"] = rate_now
      rate = rate_now if rate_now is not None else rate_entry
      if rate is not None and fill and init_risk:
        received = -rate if side == "long" else rate
        thesis["carryPerSettlementR"] = round(received / (init_risk / fill), 4)
    return thesis
  except Exception:
    logger.warning("ENTRY THESIS: could not build for %s — the position is shown without it",
                   pos.get("symbol") if isinstance(pos, dict) else pos, exc_info=True)
    return None
