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
