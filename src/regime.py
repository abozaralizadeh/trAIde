"""Regime-aware entry adjustments, enforced in code alongside the daily gate.

Two pure-function guards (unit-tested in tests/test_regime.py):

  B — Regime throttle: in a *hostile* regime (bearish or RSI-exhausted daily) raise the
      confidence bar and shrink position size, so the bot trades less and more selectively
      instead of churning low-conviction bounce-scalps in a downtrend.

  D — Trend-aligned shorts: the daily gate normally blocks *all* continuation entries when the
      daily is RSI-exhausted (anti-FOMO). In a confirmed downtrend that also blocks the
      trend-aligned short — forcing the bot to only ever long bounces. This re-permits a short
      into an exhausted-bearish daily, but only when 1h (and 15m) confirm the bounce has rolled
      over and confidence clears a higher bar, so it fires on trend continuation rather than on
      shorting a fresh oversold low.

These are deliberately small and side-effect free; the call sites in src/agent.py decide what to
do with the results.
"""

from __future__ import annotations

import math

from .config import RegimeConfig


def is_hostile_regime(daily_bias: str, daily_exhausted: bool) -> bool:
  """A regime where the bot should be more selective: bearish daily or RSI-exhausted (either side)."""
  return bool(daily_exhausted) or (daily_bias == "bearish")


def effective_min_confidence(base_min: float, daily_bias: str, daily_exhausted: bool, cfg: RegimeConfig) -> float:
  """Confidence floor for an entry — raised in a hostile regime, never lowered."""
  if cfg.throttle_enabled and is_hostile_regime(daily_bias, daily_exhausted):
    return max(base_min, cfg.caution_min_confidence)
  return base_min


def regime_size_factor(daily_bias: str, daily_exhausted: bool, cfg: RegimeConfig) -> float:
  """Position-size multiplier (<= 1.0) applied in a hostile regime, else 1.0."""
  if cfg.throttle_enabled and is_hostile_regime(daily_bias, daily_exhausted):
    return cfg.caution_size_factor
  return 1.0


def conviction_size_factor(confidence, min_confidence, cfg: RegimeConfig) -> float:
  """Position-size multiplier (<= 1.0) scaled by how far confidence clears the floor.

  An entry whose confidence barely clears the admission floor is low-conviction and gets
  `cfg.conviction_min_size_factor` of full size; size then ramps linearly to 1.0 as confidence
  approaches `cfg.conviction_full_confidence`. This shrinks exactly the low-conviction full-size
  trades that drove the SOL drawdown (the agent's own "mixed/low-conviction" read should size down,
  not in/out). Fails open to 1.0 when disabled or confidence/floor are unknown.
  """
  if not cfg.conviction_sizing_enabled:
    return 1.0
  try:
    conf = float(confidence)
    floor = float(min_confidence)
    full = float(cfg.conviction_full_confidence)
  except (TypeError, ValueError):
    return 1.0
  min_factor = min(1.0, max(0.0, float(cfg.conviction_min_size_factor)))
  if conf >= full:
    return 1.0
  if conf <= floor or full <= floor:
    return min_factor
  frac = (conf - floor) / (full - floor)
  return min_factor + frac * (1.0 - min_factor)


def combined_size_factor(factors, floor: float = 0.5) -> float:
  """Combine soft position-size multipliers by taking the WORST single signal, not their product.

  The old admission path multiplied regime × relative-strength × conviction × loss-streak ×
  expectancy into one running scale. Each is an independent "be a bit cautious" read of 0.5-0.6,
  so five of them compounded a 1% risk budget down to ~0.03% — every position became fee-dust that
  could not clear round-trip costs even when the trade was right (the account's small wins / big
  losses shape). Taking the minimum applies only the single most cautious signal, floored so a real
  edge is still sized to matter. Hard dollar-risk caps (ATR, risk budget, concentration, heat) are
  applied separately downstream and still shrink from here — this only governs the *soft* stack.

  ``factors`` may contain ``None`` (ignored). Each is clamped to [0, 1]. Empty → 1.0 (no shrink).
  ``floor`` bounds the combined result from below (1.0 disables soft shrink entirely).
  """
  vals = []
  for f in factors or ():
    try:
      if f is None:
        continue
      vals.append(min(1.0, max(0.0, float(f))))
    except (TypeError, ValueError):
      continue
  if not vals:
    return 1.0
  try:
    lo = min(1.0, max(0.0, float(floor)))
  except (TypeError, ValueError):
    lo = 0.0
  return max(lo, min(vals))


def resolve_gate_deadlock(
  *,
  daily_bias: str,
  daily_exhausted: bool,
  side: str,
  bias_1h: str,
  bias_15m: str,
  confidence,
  cfg: RegimeConfig,
) -> bool:
  """True if a daily-aligned entry should pass the 1h-alignment gate to break a both-blocked deadlock.

  The deadlock the audit flagged: in a clean (non-exhausted) daily trend, the daily gate blocks the
  counter-trend direction while the 1h gate — reacting to a counter-trend *bounce* within that daily
  trend — blocks the daily-aligned direction, stranding the agent flat in both directions. This
  re-permits the daily-aligned trade (a short in a bearish daily, a long in a bullish daily) past the
  1h gate, but only when the bounce is stalling (15m no longer confirms the 1h counter-move) and
  confidence clears a raised bar — so it takes the trend-continuation trade rather than knife-catching
  a live bounce. Disjoint from `allow_trend_aligned_short` (which handles the exhausted-daily case).
  """
  if not cfg.deadlock_break_enabled:
    return False
  bias = str(daily_bias or "").strip().lower()
  s = (side or "").lower()
  if daily_exhausted or bias not in ("bullish", "bearish"):
    return False
  daily_aligned = (bias == "bearish" and s == "sell") or (bias == "bullish" and s == "buy")
  if not daily_aligned:
    return False
  one_h = str(bias_1h or "").strip().lower()
  one_h_opposes = (one_h == "bullish" and s == "sell") or (one_h == "bearish" and s == "buy")
  if not one_h_opposes:
    return False
  fifteen = str(bias_15m or "").strip().lower()
  fifteen_still_counter = (fifteen == "bullish" and s == "sell") or (fifteen == "bearish" and s == "buy")
  if fifteen_still_counter:
    return False
  try:
    if float(confidence or 0.0) < cfg.deadlock_min_confidence:
      return False
  except (TypeError, ValueError):
    return False
  return True


def block_alt_long_in_btc_downtrend(
  *,
  symbol: str,
  side: str,
  btc_daily_bias: str,
  cfg: RegimeConfig,
  local_daily_bias: str = "neutral",
  bias_4h: str = "neutral",
  bias_1h: str = "neutral",
  bias_15m: str = "neutral",
  strength: str = "weak",
  daily_exhausted: bool = False,
  confidence=None,
) -> bool:
  """True if a LONG on a non-major altcoin should be blocked because BTC's daily regime is bearish.

  Alts are high-beta to BTC: longing them while the market leader is in a confirmed daily
  downtrend is the setup that blew up on RE-USDT. Majors (cfg.alt_majors) are exempt — they have
  their own per-symbol daily gate. Only fires on a strict bearish BTC daily read.
  """
  if not cfg.alt_long_block_enabled:
    return False
  if (side or "").lower() not in ("buy", "long"):
    return False
  base = (symbol or "").split("-")[0].strip().upper()
  if not base or base in {m.upper() for m in (cfg.alt_majors or ())}:
    return False
  if str(btc_daily_bias or "").strip().lower() != "bearish":
    return False
  # Let a true relative-strength leader through at reduced size (the caller applies the factor).
  # All four local timeframes must agree, the daily cannot be exhausted, and the model must clear
  # a deliberately high bar.  Missing context therefore preserves the conservative block.
  if is_relative_strength_alt_long(
    symbol=symbol,
    side=side,
    btc_daily_bias=btc_daily_bias,
    local_daily_bias=local_daily_bias,
    bias_4h=bias_4h,
    bias_1h=bias_1h,
    bias_15m=bias_15m,
    strength=strength,
    daily_exhausted=daily_exhausted,
    confidence=confidence,
    cfg=cfg,
  ):
    return False
  return True


def is_relative_strength_alt_long(
  *,
  symbol: str,
  side: str,
  btc_daily_bias: str,
  local_daily_bias: str,
  bias_4h: str,
  bias_1h: str,
  bias_15m: str,
  strength: str,
  daily_exhausted: bool,
  confidence,
  cfg: RegimeConfig,
) -> bool:
  """Whether an alt long is strong enough to override a bearish-BTC correlation veto.

  This captures rotating leadership without hardcoding yesterday's winning symbol.  The exception
  is intentionally narrow and callers must still apply the configured reduced-size factor plus all
  ordinary volatility, R:R, fee and concentration gates.
  """
  if not cfg.relative_strength_longs_enabled or daily_exhausted:
    return False
  if (side or "").lower() not in ("buy", "long"):
    return False
  base = (symbol or "").split("-")[0].strip().upper()
  if not base or base in {m.upper() for m in (cfg.alt_majors or ())}:
    return False
  if str(btc_daily_bias or "").strip().lower() != "bearish":
    return False
  aligned = (local_daily_bias, bias_4h, bias_1h, bias_15m)
  if any(str(v or "").strip().lower() != "bullish" for v in aligned):
    return False
  if str(strength or "").strip().lower() != "strong":
    return False
  try:
    return float(confidence or 0.0) >= float(cfg.relative_strength_min_confidence)
  except (TypeError, ValueError):
    return False


def bracket_risk_scale(
  *,
  entry,
  stop_loss,
  notional_usd,
  equity_usd,
  risk_fraction,
) -> float:
  """Scale an entry down so its stop-defined dollar loss stays within the equity risk budget.

  This only shrinks; it never increases a model-requested position.  Unknown/invalid inputs fail
  open because the entry call separately requires a valid bracket.
  """
  try:
    e = float(entry)
    sl = float(stop_loss)
    notional = float(notional_usd)
    equity = float(equity_usd)
    fraction = float(risk_fraction)
  except (TypeError, ValueError):
    return 1.0
  if e <= 0 or sl <= 0 or notional <= 0 or equity <= 0 or fraction <= 0:
    return 1.0
  stop_fraction = abs(e - sl) / e
  if stop_fraction <= 0:
    return 1.0
  planned_risk = notional * stop_fraction
  budget = equity * fraction
  return min(1.0, max(0.0, budget / planned_risk))


def reward_risk_ratio(side: str, entry, take_profit, stop_loss):
  """Reward:risk of an entry bracket — |TP - entry| / |entry - SL| — or None if it can't form.

  Returns None when inputs are non-numeric, the stop distance is non-positive, or the TP/SL sit on
  the wrong side of entry for the direction (a long needs TP above and SL below entry; a short the
  reverse). Callers treat None as a reject: a bracket that can't be measured shouldn't be traded.
  """
  try:
    e = float(entry)
    tp = float(take_profit)
    sl = float(stop_loss)
  except (TypeError, ValueError):
    return None
  s = (side or "").lower()
  if s in ("buy", "long"):
    reward = tp - e
    risk = e - sl
  elif s in ("sell", "short"):
    reward = e - tp
    risk = sl - e
  else:
    return None
  if risk <= 0 or reward <= 0:
    return None
  return reward / risk


def net_reward_risk_ratio(
  side: str,
  entry,
  take_profit,
  stop_loss,
  *,
  fee_rate: float = 0.0,
  slippage_rate: float = 0.0,
):
  """Reward/risk after estimated entry+exit fees and slippage.

  Gross chart distance is not trade expectancy: friction reduces a winner and increases a loser.
  Rates are per side, matching the execution gate's existing conservative cost model. Returns
  ``None`` for an invalid bracket and ``0`` when costs consume all projected reward.
  """
  gross = reward_risk_ratio(side, entry, take_profit, stop_loss)
  if gross is None:
    return None
  try:
    e = float(entry)
    tp = float(take_profit)
    sl = float(stop_loss)
    rate = max(0.0, float(fee_rate)) + max(0.0, float(slippage_rate))
  except (TypeError, ValueError):
    return None
  direction = str(side or "").lower()
  if direction in {"buy", "long"}:
    gross_reward = tp - e
    gross_risk = e - sl
  else:
    gross_reward = e - tp
    gross_risk = sl - e
  net_reward = gross_reward - (e + tp) * rate
  net_risk = gross_risk + (e + sl) * rate
  if net_risk <= 0:
    return None
  return max(0.0, net_reward) / net_risk


def concentration_scale(
  notional_usd: float,
  total_equity_usd: float,
  max_pct: float,
  existing_notional_usd: float = 0.0,
) -> float:
  """Shrink an order so projected same-symbol notional stays within the equity cap.

  Returns 1.0 when the cap is disabled (max_pct<=0), equity is unknown (<=0), or the position is
  already within the cap. Caps the per-position blast radius regardless of leverage.
  """
  try:
    notional = float(notional_usd)
    equity = float(total_equity_usd)
    pct = float(max_pct)
    existing = max(0.0, float(existing_notional_usd))
  except (TypeError, ValueError):
    return 1.0
  if pct <= 0 or equity <= 0 or notional <= 0:
    return 1.0
  cap = pct * equity
  remaining = max(0.0, cap - existing)
  if notional <= remaining:
    return 1.0
  return max(0.0, remaining / notional)


def risk_capped_contracts(
  requested_contracts: int,
  lot_size: int,
  multiplier: float,
  entry_price: float,
  stop_price: float,
  equity_usd: float,
  risk_fraction: float,
  existing_risk_usd: float = 0.0,
) -> int:
  """Cap a new leg by the remaining risk budget for the whole position lifecycle."""
  try:
    requested = int(requested_contracts)
    lot = max(1, int(lot_size))
    risk_per_contract = float(multiplier) * abs(float(entry_price) - float(stop_price))
    remaining = max(0.0, float(equity_usd) * float(risk_fraction) - float(existing_risk_usd))
  except (TypeError, ValueError):
    return int(requested_contracts)
  if requested <= 0 or risk_per_contract <= 0:
    return requested
  maximum = int(math.floor((remaining / risk_per_contract) / lot) * lot)
  return min(requested, maximum)


def add_on_guard_reason(
  *,
  current_qty: float,
  new_side: str,
  avg_entry: float,
  protective_stop: float | None,
  proposed_stop: float,
  fee_buffer_fraction: float = 0.0,
) -> str | None:
  """Reject pyramiding that reverses, re-risks, or loosens an existing lifecycle."""
  qty = float(current_qty)
  side = str(new_side or "").lower()
  same_direction = (qty > 0 and side == "buy") or (qty < 0 and side == "sell")
  if not same_direction:
    return "Opposite-side entry would implicitly reverse an open position; close it explicitly with reduce_only first"
  if protective_stop is None:
    return "Add-on blocked until the existing position has a verified breakeven-or-better stop"
  entry = float(avg_entry)
  stop = float(protective_stop)
  proposed = float(proposed_stop)
  fee = max(0.0, float(fee_buffer_fraction))
  breakeven = entry * (1 + fee) if qty > 0 else entry * (1 - fee)
  locked = stop >= breakeven if qty > 0 else stop <= breakeven
  if not locked:
    return "Add-on blocked until existing lifecycle risk is locked at breakeven"
  loosens = proposed < stop if qty > 0 else proposed > stop
  if loosens:
    return "Add-on stop would loosen existing protection"
  return None


def oi_price_signal(price_direction: str | None, oi_trend: str | None) -> tuple[str, str]:
  """Classify price/OI quadrants only when a real, timestamped OI trend exists."""
  price = str(price_direction or "").lower()
  oi = str(oi_trend or "").lower()
  if price not in {"up", "down"} or oi not in {"up", "down"}:
    return "neutral", "Open-interest change is unavailable or flat; do not use OI as confirmation."
  if price == "up" and oi == "up":
    return "strong_trend", "Rising price + rising OI supports trend continuation."
  if price == "up" and oi == "down":
    return "short_covering", "Rising price + falling OI suggests short covering; do not treat it as fresh long conviction."
  if price == "down" and oi == "up":
    return "aggressive_shorts", "Falling price + rising OI indicates new short positioning and bearish continuation risk."
  return "long_capitulation", "Falling price + falling OI indicates long liquidation/capitulation and possible exhaustion."


def allow_reversal_long(
  *,
  daily_bias: str,
  side: str,
  bias_1h: str,
  bias_15m: str,
  confidence,
  cfg: RegimeConfig,
) -> bool:
  """True if a LONG should be permitted past a bearish daily gate because a reversal is confirmed.

  The daily gate hard-blocks longs while the 1D trend reads bearish — a lagging signal that stays
  bearish through the *bottom* of a move, so the bot structurally cannot catch the reversal (it sat
  out an +11% ETH bounce in the Jul 2-5 2026 chop, blocked from every long). This yields the gate ONLY
  when the lower timeframes have clearly turned up (1h and 15m both bullish) and confidence clears a
  high bar — so it fires on a confirmed turn, not on knife-catching a falling market. The R:R floor
  still applies to whatever it lets through, and non-major alts remain blocked by the correlation gate.
  """
  if not cfg.reversal_longs_enabled:
    return False
  if str(daily_bias or "").strip().lower() != "bearish" or (side or "").lower() not in ("buy", "long"):
    return False
  if str(bias_1h or "").strip().lower() != "bullish":
    return False
  if cfg.reversal_long_require_15m and str(bias_15m or "").strip().lower() != "bullish":
    return False
  try:
    if float(confidence or 0.0) < cfg.reversal_long_min_confidence:
      return False
  except (TypeError, ValueError):
    return False
  return True


def allow_reversal_short(
  *,
  daily_bias: str,
  side: str,
  bias_1h: str,
  bias_15m: str,
  confidence,
  cfg: RegimeConfig,
) -> bool:
  """True if a SHORT should be permitted past a bullish daily gate because a roll-over is confirmed.

  Exact mirror of `allow_reversal_long`. When the market flipped to a bullish daily regime
  (July 2026 recovery), the daily gate blocked every short while intraday had clearly turned down
  (Jul 7-8 pullback: SOL -5%, ETH -2.3% with 'daily is bullish, shorts blocked' repeating in the
  log) — the same lagging-daily straitjacket as before, mirrored. Yields the gate ONLY when 1h and
  15m are both bearish and confidence clears a high bar: a confirmed turn, not fading strength.
  The per-symbol R:R floor, bench, and sizing throttles still apply to whatever passes.
  """
  if not cfg.reversal_shorts_enabled:
    return False
  if str(daily_bias or "").strip().lower() != "bullish" or (side or "").lower() not in ("sell", "short"):
    return False
  if str(bias_1h or "").strip().lower() != "bearish":
    return False
  if cfg.reversal_short_require_15m and str(bias_15m or "").strip().lower() != "bearish":
    return False
  try:
    if float(confidence or 0.0) < cfg.reversal_short_min_confidence:
      return False
  except (TypeError, ValueError):
    return False
  return True


def allow_trend_aligned_short(
  *,
  daily_exhausted: bool,
  daily_bias_raw: str,
  side: str,
  bias_1h: str,
  bias_15m: str,
  confidence,
  cfg: RegimeConfig,
) -> bool:
  """True if a SHORT into an exhausted-bearish daily should be permitted past the anti-FOMO gate.

  Requires the lower timeframes to confirm the downtrend is resuming (1h bearish, and 15m bearish
  unless disabled) plus a higher confidence bar — so it fires on trend continuation, not on
  shorting a fresh oversold bounce that may squeeze.
  """
  if not cfg.trend_shorts_enabled:
    return False
  if not (daily_exhausted and daily_bias_raw == "bearish" and (side or "").lower() == "sell"):
    return False
  if bias_1h != "bearish":
    return False
  if cfg.trend_short_require_15m and bias_15m != "bearish":
    return False
  try:
    if float(confidence or 0.0) < cfg.trend_short_min_confidence:
      return False
  except (TypeError, ValueError):
    return False
  return True
