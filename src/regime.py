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


def concentration_scale(notional_usd: float, total_equity_usd: float, max_pct: float) -> float:
  """Scale factor (<=1.0) that shrinks a position's notional to at most `max_pct` of equity.

  Returns 1.0 when the cap is disabled (max_pct<=0), equity is unknown (<=0), or the position is
  already within the cap. Caps the per-position blast radius regardless of leverage.
  """
  try:
    notional = float(notional_usd)
    equity = float(total_equity_usd)
    pct = float(max_pct)
  except (TypeError, ValueError):
    return 1.0
  if pct <= 0 or equity <= 0 or notional <= 0:
    return 1.0
  cap = pct * equity
  if notional <= cap:
    return 1.0
  return max(0.0, cap / notional)


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
