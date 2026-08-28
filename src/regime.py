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


def overextension_atr(side: str, price, vwap, atr) -> float | None:
  """How far a price sits beyond the intraday VWAP anchor, in ATR units, in a trade's direction.

  Positive = extended in the trade's direction (a long above VWAP / a short below it) → later in the
  move, likelier to pull back before continuing; negative = on the favorable side (a pullback below
  VWAP for a long). This is a *decision-support measurement*, surfaced to the agent so it can plan the
  highest-EV entry (target the pullback/retest, not chase the peak) — deliberately NOT a hard gate:
  entry timing is the model's judgement, bounded by the risk caps, and improves as the model does.
  Returns ``None`` on missing/invalid inputs so callers can simply omit it from context.
  """
  try:
    p = float(price)
    v = float(vwap)
    a = float(atr)
  except (TypeError, ValueError):
    return None
  if a <= 0 or p <= 0 or v <= 0 or not (math.isfinite(p) and math.isfinite(v) and math.isfinite(a)):
    return None
  s = (side or "").lower()
  if s in ("buy", "long"):
    return (p - v) / a
  if s in ("sell", "short"):
    return (v - p) / a
  return None


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


def noise_floored_stop(
  side: str,
  entry,
  stop_loss,
  atr_abs,
  floor_mult: float,
  *,
  max_widen_mult: float = 4.0,
):
  """Push a stop that sits *inside the instrument's noise band* out to the edge of it.

  A stop is only an invalidation level if price reaching it means something. When the stop sits
  closer than one bar of ordinary volatility, it is hit by noise before the thesis can resolve —
  the trade is a coin-flip on microstructure regardless of how good the read was.

  The live account's own record (Jul 2026, 27 closed futures lifecycles) is unambiguous: the median
  stop sat at 1.4x the 15m ATR (~0.7x the 1h ATR — *less than one hourly bar*), median favourable
  excursion was only +0.27R against targets planned at 2.3-2.7R gross, and **not one trade in the
  sample reached its take-profit**. Trades with tighter-than-median stops averaged -0.40R; wider ones
  -0.08R. Replaying those same entries on real 1m paths with this floor applied turns -4.5R into
  +4.9R, and the effect is flat-positive for every floor from 1.5x to 5x ATR — it is the geometry
  that matters, not the constant.

  This is a *survival* guard, not an opportunity gate: it never vetoes a setup and never changes
  which symbol or direction is traded. It only widens the risk leg. Because position size is
  stop-defined, a wider stop automatically buys fewer contracts, so dollar risk per trade is
  unchanged — the trade simply gets room to breathe.

  Args:
    side: "buy"/"long" or "sell"/"short".
    entry: entry price.
    stop_loss: the stop the caller planned.
    atr_abs: ATR in absolute price units on the decision timeframe (15m), or None if unknown.
    floor_mult: minimum stop distance as a multiple of ``atr_abs``. <=0 disables the floor.
    max_widen_mult: never widen the stop by more than this factor. A stop that would need a 4x+
      widening is not a slightly-tight stop, it is a different trade — leave it alone and let the
      RR gate judge it, rather than silently rewriting the caller's thesis.

  Returns ``(stop_price, info)``. ``stop_price`` is the original stop when no widening applies, so
  callers can use the result unconditionally. ``info["applied"]`` says whether it moved.
  """
  info = {"applied": False, "reason": "no floor applied"}
  try:
    e = float(entry)
    sl = float(stop_loss)
    atr = float(atr_abs) if atr_abs is not None else None
    mult = float(floor_mult)
  except (TypeError, ValueError):
    return stop_loss, {"applied": False, "reason": "non-numeric inputs"}
  if atr is None or not math.isfinite(atr) or atr <= 0 or mult <= 0 or e <= 0:
    return stop_loss, {"applied": False, "reason": "no usable ATR"}
  s = (side or "").lower()
  if s in ("buy", "long"):
    planned = e - sl
    long_side = True
  elif s in ("sell", "short"):
    planned = sl - e
    long_side = False
  else:
    return stop_loss, {"applied": False, "reason": "unknown side"}
  if planned <= 0:
    return stop_loss, {"applied": False, "reason": "stop on the wrong side of entry"}
  required = mult * atr
  if planned >= required:
    info["reason"] = "planned stop already outside the noise band"
    info.update(plannedAtrMult=planned / atr, requiredAtrMult=mult)
    return stop_loss, info
  # Cap the rewrite: widen toward the floor but never past ``max_widen_mult`` x the planned stop.
  widened = min(required, planned * max(1.0, float(max_widen_mult)))
  new_stop = e - widened if long_side else e + widened
  return new_stop, {
    "applied": True,
    "reason": (
      f"stop was {planned / atr:.2f}x ATR (inside the noise band); widened to "
      f"{widened / atr:.2f}x ATR so the level is a real invalidation"
    ),
    "plannedStop": sl,
    "newStop": new_stop,
    "plannedAtrMult": planned / atr,
    "requiredAtrMult": mult,
    "widenFactor": widened / planned,
    "capped": widened < required - 1e-12,
  }


def scale_target_to_widened_stop(side: str, entry, take_profit, widen_factor):
  """Keep a bracket's planned R-multiple when the noise floor rescales the risk leg.

  The stop floor was measured live on 2026-08-02 and produced an interaction I did not anticipate:
  widening the risk leg while leaving the target where the model put it **mechanically destroys
  reward:risk**, and the admission gate then rejects the trade. Over the first 12.5 live hours the
  median *gross* RR of rejected setups fell 1.79 -> 1.23 even though the cost-model fix had cut the
  friction drag 0.62 -> 0.28; the two changes cancelled, rejections stayed flat at 0.19/run, and the
  order rate dropped 72%.

  The resolution is that the floor is a statement about the *scale of movement*, not about the thesis.
  If the model's stop sat inside the noise band, its target was drawn on that same too-tight scale —
  so when code rescales risk by ``widen_factor`` the reward leg has to travel with it, leaving the
  intended R-multiple untouched and the floor RR-neutral. Position size still shrinks, so dollar risk
  per trade is unchanged; the trade simply gets proportional room on both legs. This is not "moving
  the target to pass the gate": the R-multiple the model chose is preserved exactly, only the unit of
  R changes. Replaying the recorded lifecycles, holding the target at a constant multiple of the
  *floored* stop scores +4.12R against +4.09R for the unscaled target — outcomes are insensitive to
  target distance over 0.8-2.0R, so this restores throughput without trading away exit quality.

  Returns the (possibly unchanged) take-profit, so callers can apply it unconditionally.
  """
  try:
    e = float(entry)
    tp = float(take_profit)
    factor = float(widen_factor)
  except (TypeError, ValueError):
    return take_profit
  if not math.isfinite(factor) or factor <= 1.0 or e <= 0:
    return take_profit
  s = (side or "").lower()
  if s in ("buy", "long"):
    reward = tp - e
    if reward <= 0:
      return take_profit
    return e + reward * factor
  if s in ("sell", "short"):
    reward = e - tp
    if reward <= 0:
      return take_profit
    return e - reward * factor
  return take_profit


def fade_setup_available(rsi, cfg: RegimeConfig):
  """Flag a live fade-extreme opportunity at analysis time, so the playbook is FINDABLE.

  Unblocking the gates was necessary but not sufficient: over the first 39 measured signals, 38 were
  `continuation` and only one was a fade. The model was not withholding fades — it had no prompt at
  the moment of decision telling it one was on the table. Every piece of guidance the analysis emits
  (entry_hint, the entryMap note, the ATR-extension framing) is written for arriving at a good price
  on a *trend* trade, and the regime label reads "trending" on ~93% of symbols, so the mean-reversion
  hints keyed on "ranging" essentially never fire.

  This says the quiet part out loud: RSI is at an extreme, a fade is permitted here, and it must be
  declared as such to pass the alignment gates. It asserts nothing about whether fading is a good
  idea — `signalEdge.by_family` keeps that score, and risk follows the measurement. It exists so the
  hypothesis can be *tested at all*, which it currently cannot be at one probe per 39.

  Returns ``None`` when there is no extreme, else a dict describing the available direction.
  """
  if not getattr(cfg, "fade_extreme_enabled", False):
    return None
  try:
    value = float(rsi)
  except (TypeError, ValueError):
    return None
  if not math.isfinite(value):
    return None
  oversold = float(getattr(cfg, "fade_extreme_oversold_rsi", 30.0))
  overbought = float(getattr(cfg, "fade_extreme_overbought_rsi", 70.0))
  if value <= oversold:
    side, why = "buy", f"15m RSI {value:.0f} <= {oversold:.0f} (oversold)"
  elif value >= overbought:
    side, why = "sell", f"15m RSI {value:.0f} >= {overbought:.0f} (overbought)"
  else:
    return None
  return {
    "side": side,
    "rsi15m": round(value, 1),
    "reason": why,
    "note": (
      f"FADE SETUP AVAILABLE — {why}. A {side.upper()} here fades the stretch back toward value. "
      "This is a different PLAYBOOK from trend continuation, and the 1h/daily will oppose it by "
      "definition — that is what makes it a fade, not a reason to skip it. To take it you MUST pass "
      "setup_family='fade_extreme' to place_futures_limit_order; without that declaration the "
      "alignment gates reject it. It is scored separately in signalEdge.by_family, so taking it is "
      "how the bot learns whether this playbook pays."
    ),
  }


def allow_fade_extreme(
  *,
  side: str,
  setup_family: str | None,
  rsi: float | None,
  cfg: RegimeConfig,
) -> bool:
  """Permit a declared FADE-EXTREME entry past the trend-alignment gates.

  The bot could only ever express one playbook. Its `market_regime` label read "trending" on 68 of 69
  recorded entries, so the mean-reversion hints never fired; the daily gate blocks counter-daily
  entries; and the 1h-alignment gate blocks any entry the 1h opposes. A fade of an oversold extreme
  has the 1h opposing it *by definition* — that is what makes it a fade — so the architecture made the
  setup unreachable. Meanwhile the family it could express, trend continuation, measured -0.017% gross
  over 3,408 samples on the live universe: flat, and flat does not cover a 0.10% round trip.

  This does not assert that fading works. Measured on the same 50 days it was positive in both halves
  but only t~1.0-1.6 over 135 independent events — suggestive, not established, and mean reversion
  always flatters itself in a range. The point is to let the setup REACH the model so
  `edge.signal_edge_stats` can score it per family and risk can follow whatever actually pays. An
  unmeasurable hypothesis can never be disproved either.

  The extreme test uses the textbook RSI 30/70 bands rather than anything fitted to this sample, and
  requires the extreme to be *against* the entry direction — buying only into oversold, selling only
  into overbought. Everything downstream (stop floor, RR floor, risk budget, family sizing) still
  applies, so this widens what may be proposed, never what may be risked.
  """
  if not getattr(cfg, "fade_extreme_enabled", False):
    return False
  if str(setup_family or "").strip().lower() != "fade_extreme":
    return False
  try:
    value = float(rsi)
  except (TypeError, ValueError):
    return False
  if not math.isfinite(value):
    return False
  s = (side or "").lower()
  if s in ("buy", "long"):
    return value <= float(getattr(cfg, "fade_extreme_oversold_rsi", 30.0))
  if s in ("sell", "short"):
    return value >= float(getattr(cfg, "fade_extreme_overbought_rsi", 70.0))
  return False


def allow_declared_setup(
  *,
  setup_family: str | None,
  cfg: RegimeConfig,
) -> bool:
  """Permit a deliberately-declared alternative playbook past the trend-alignment gates.

  The alignment gates (daily + 1h) only ever admit a *trend-aligned* entry freely, and a trend-aligned
  entry is tagged ``continuation`` by ``edge.infer_setup_family``. So continuation was the only playbook
  that could reach the book at all: it accumulated ~every probe, measured "no edge", and the bot kept
  taking it because it had nothing else it was allowed to express. ``allow_fade_extreme`` opened ONE
  escape hatch, keyed on a genuine RSI extreme. But ``breakout`` and ``range_edge`` had no hatch and are
  never auto-inferred, so a counter-trend breakout or range fade was rejected before it could ever log a
  probe — no placement, no probe, ``samples: 0`` forever. The taxonomy listed five families while the
  gates made three of them structurally unreachable.

  This generalises the fade-extreme carve-out: a family the operator has listed in
  ``declarable_setup_families`` is admitted past the gates *on the model's declaration alone*. That is
  deliberate and philosophy-consistent — WHICH playbook to run is an OPPORTUNITY call (the model's job),
  and a hardcoded trend gate deciding it is exactly the kind of veto this codebase avoids. Survival is
  still fully code-governed, just DOWNSTREAM of the gate rather than at it: the probe is recorded at the
  call, ``family_explore_factor`` sizes an unproven playbook down to exploration-size, ``family_size_factor``
  shrinks it if it measures a shortfall, and ``family_stand_aside`` skips it outright once it settles to
  "no edge" over a real sample. So a declared breakout that turns out not to pay still logs its evidence
  and is then sized to a quarter, and eventually to zero — by MEASUREMENT, not by a pre-trade trend veto.
  This widens what may be PROPOSED, never what may be RISKED.

  No market-condition test is attached on purpose. A regime filter (e.g. "only range_edge in a ranging
  tape") would re-block the family, because the live ``market_regime`` label reads "trending" ~93% of the
  time — that mislabel is precisely why range_edge never fired. The declaration is the trigger; the
  scoreboard is the judge.
  """
  if not getattr(cfg, "declared_setups_enabled", False):
    return False
  fam = str(setup_family or "").strip().lower()
  if not fam:
    return False
  declarable = tuple(str(f).strip().lower() for f in (getattr(cfg, "declarable_setup_families", ()) or ()))
  return fam in declarable


def coherent_risk_fraction(
  configured_fraction,
  max_daily_drawdown_pct,
  max_consecutive_losses,
):
  """The risk-per-trade fraction that is actually consistent with the circuit breakers.

  A per-trade risk budget and a daily drawdown stop are not independent settings: if one trade risks
  R% and the day halts at D%, the bot can only absorb D/R losers before it stops trading. The live
  config asked for 2% per trade against a 3% daily stop — **two losses would end the day**, on an
  account taking 5-9 trades a day. That inconsistency went unnoticed for months only because realized
  risk was never actually 2% (it averaged 0.52%, because sizing merely *capped* at the budget instead
  of targeting it), so the drawdown stop never had a chance to bite.

  Deriving the fraction from limits the operator already set keeps the two coherent with no extra
  knob to maintain: allow ``max_consecutive_losses`` full stop-outs plus one before the daily stop
  trips. With 3% / (3+1) that is 0.75%. Never raises above ``configured_fraction`` — the configured
  value stays a ceiling, so this can only make the posture more survivable.
  """
  try:
    configured = float(configured_fraction)
  except (TypeError, ValueError):
    return {"value": 0.0, "source": "invalid"}
  if configured <= 0:
    return {"value": 0.0, "source": "disabled"}
  try:
    drawdown = float(max_daily_drawdown_pct) / 100.0
    losses = int(max_consecutive_losses)
  except (TypeError, ValueError):
    return {"value": configured, "source": "configured"}
  if drawdown <= 0 or losses <= 0:
    return {"value": configured, "source": "configured"}
  derived = drawdown / (losses + 1)
  if derived >= configured:
    return {"value": configured, "source": "configured", "derived": derived}
  return {
    "value": derived,
    "source": "derived",
    "configured": configured,
    "reason": (
      f"{drawdown:.1%} daily drawdown stop / {losses + 1} tolerated stop-outs = {derived:.2%} per trade "
      f"(configured {configured:.2%} would halt the day after {max(1, int(drawdown / configured))} losses)"
    ),
  }


def risk_targeted_notional(*, entry, stop_loss, equity_usd, risk_fraction):
  """Notional that puts exactly ``risk_fraction`` of equity at risk for this stop distance.

  The counterpart to :func:`bracket_risk_scale`, which only ever *shrinks* a model-requested size.
  Capping alone is not sizing: it leaves the actual bet to whatever notional the model happened to
  name, so the live account's realized risk ranged over **18.9x** (min $0.06, max $1.17 on a ~$68
  account) with no relation to conviction or outcome. The cost of that is not theoretical — across the
  35 recorded lifecycles the winners were systematically the small bets and the losers the large ones,
  so the last 9 trades came in at **+0.50R but -$0.12**. A positive edge only becomes money if every
  trade bets the same fraction of it.

  Returns ``None`` when inputs cannot form a sizing decision, so callers keep their previous behavior.
  """
  try:
    e = float(entry)
    sl = float(stop_loss)
    equity = float(equity_usd)
    fraction = float(risk_fraction)
  except (TypeError, ValueError):
    return None
  if e <= 0 or sl <= 0 or equity <= 0 or fraction <= 0:
    return None
  stop_fraction = abs(e - sl) / e
  if stop_fraction <= 0:
    return None
  return (equity * fraction) / stop_fraction


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
