"""Jev dual run — a second trader deciding from the same analysis, judged on its own record (2026-09-30).

Jev (typesafe.ai, System One API) is a fast, cheap classifier: it reads a state and returns calibrated
probabilities over labelled choices, not free text or tool calls. The dual run asks it, per symbol, the
OPPORTUNITY questions only — which direction (or stand aside), which playbook, rest at a pullback or enter
now, near or extended target — and hands its answer to the SAME order path the LLM agent uses:

* **Survival stays code's.** Every gate, the risk-per-trade cap, the atomic bracket, the circuit breakers,
  ProtectionManager's trail/breakeven and the stand-aside apply unchanged (``tools.place_futures_limit_order_for``).
  The bracket is BUILT BY CODE from the analysis levels (``build_bracket``) — Jev only chooses between
  code-built variants — so a classifier never invents a stop.
* **Its own record.** Every Jev call is stamped ``trader='jev'`` and scored in its own buckets
  (``memory.signal_probes(trader='jev')`` → ``signal_edge_jev``); the LLM's verdicts never see it. An unproven
  trader starts at the explore floor (trial size), and earns or loses stake by the same stateless bar.
* **One owner per lifecycle.** A symbol the LLM holds is not Jev's to trade and vice versa
  (``trader_conflict``); Jev positions are exited by code only (bracket + trail), never by either model.
* **Modes** (``JEV_MODE``): ``off`` (default, nothing runs), ``shadow`` (every call runs the full gate chain
  and is recorded as a probe, no order — ``dry_run``), ``live`` (up to ``JEV_MAX_OPEN_POSITIONS`` real
  trial-size entries; calls past the caps or on another trader's symbol are still recorded as shadow calls,
  so the record does not depend on which symbols happened to be free).

Nothing here raises into the trading loop: every failure becomes a row with ``outcome='error'`` and a log line.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

from .edge import safe_family_horizon_weights, safe_family_horizons, signal_edge_stats
from .memory import DEFAULT_TRADER, KNOWN_TRADERS, SCORED_GATES, STRUCTURAL_REFUSALS
from .regime import net_reward_risk_ratio
from .utils import normalize_symbol

logger = logging.getLogger(__name__)

DIRECTIONS = ("long", "short", "stand_aside")
# The playbooks Jev may declare. macro_event and other are left out: a scheduled release is not in the
# state Jev sees, and 'other' carries no hypothesis to score.
JEV_FAMILIES = ("continuation", "fade_extreme", "breakout", "range_edge", "funding_carry")
# An analysis older than this is re-run before Jev reads it (the order path refuses > 15 min anyway).
ANALYSIS_MAX_AGE_SEC = 600.0
# A pullback anchor this close to price is a market entry anyway; beyond the far bound it rarely fills
# inside one entry lease. Geometry for the menu, not a gate: the other variant is always on the menu.
PULLBACK_MIN_ATR = 0.25
PULLBACK_MAX_ATR = 2.0
# The structural stop may widen the noise floor up to this multiple of it and no further: a wider stop only
# shrinks size (risk is stop-defined), until the contract minimum refuses the order outright.
STRUCTURAL_STOP_MAX_FLOOR_MULT = 2.0
# Target menu, in NET R after round-trip cost: 'near' sits just over the fee-guard floor (the cushion keeps
# tick rounding from pushing it under), 'extended' is twice that. ProtectionManager trails either one.
NEAR_TARGET_CUSHION = 1.1
EXTENDED_TARGET_MULT = 2.0
MAX_CONCURRENT_ASKS = 4

_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args: Any) -> None:
  if key not in _warned:
    _warned.add(key)
    logger.warning(msg, *args)


def _num(value: Any) -> Optional[float]:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return None
  return out if math.isfinite(out) else None


def _r(value: Any, nd: int = 4) -> Optional[float]:
  v = _num(value)
  return round(v, nd) if v is not None else None


def _side_word(side: Any) -> Optional[str]:
  s = str(side or "").strip().lower()
  return {"buy": "long", "long": "long", "sell": "short", "short": "short"}.get(s)


# ── State ────────────────────────────────────────────────────────────────────────────────────────────
def _interval_row(snap: Dict[str, Any]) -> Dict[str, Any]:
  upper, lower, close = _num(snap.get("bb_upper")), _num(snap.get("bb_lower")), _num(snap.get("close"))
  pct_b = None
  if upper is not None and lower is not None and close is not None and upper > lower:
    pct_b = round((close - lower) / (upper - lower), 3)
  regime = snap.get("market_regime")
  ema_f, ema_s = _num(snap.get("ema_fast")), _num(snap.get("ema_slow"))
  return {
    "trend": snap.get("trend_bias"),
    "regime": regime.get("regime") if isinstance(regime, dict) else regime,
    "rsi": _r(snap.get("rsi"), 1),
    "adx": _r(snap.get("adx"), 1),
    "plusDI": _r(snap.get("plus_di"), 1),
    "minusDI": _r(snap.get("minus_di"), 1),
    "macdHist": _r(snap.get("macd_hist"), 8),
    "emaFastAboveSlow": (ema_f > ema_s) if ema_f is not None and ema_s is not None else None,
    "atrPct": _r(snap.get("atr_pct"), 3),
    "bollingerPctB": pct_b,
    "bollingerWidthPct": _r(snap.get("bbw"), 3),
    "stochK": _r(snap.get("stoch_k"), 1),
    "volatility": snap.get("volatility"),
  }


def jev_state(symbol: str, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
  """The market facts Jev decides on, from one ``analyze_market_context`` result — or None if unusable.

  Market data only: no balance, equity, position size, gate text or scoreboard (the survival layer is
  code's, and a classifier that sees its own record would learn to game it). The same numbers the LLM
  reads, reshaped into one flat JSON object per timeframe.
  """
  if not isinstance(analysis, dict) or analysis.get("error"):
    return None
  if not (analysis.get("dataQuality") or {}).get("ok", False):
    return None
  summary = analysis.get("summary") or {}
  emap = summary.get("entryMap") or {}
  if not _num(emap.get("price")) or not _num(emap.get("atr15m")):
    return None
  frames = {}
  for snap in analysis.get("snapshots") or []:
    if isinstance(snap, dict) and snap.get("interval") in ("15min", "1hour", "4hour", "1day"):
      frames[snap["interval"]] = _interval_row(snap)
  fade = emap.get("fadeSetup") if isinstance(emap.get("fadeSetup"), dict) else None
  state: Dict[str, Any] = {
    "symbol": symbol,
    "instrument": "USDT-margined perpetual future",
    "timeframes": frames,
    "summary": {
      "overallBias": summary.get("overall_bias"),
      "strength": summary.get("strength"),
      "weightedScore": _r(summary.get("weighted_score"), 3),
      "dailyBias": summary.get("daily_bias"),
      "dailyExhausted": summary.get("daily_exhausted"),
      "dailyTrendWeak": summary.get("daily_trend_weak"),
      "timeframeConflict": summary.get("timeframe_conflict"),
      "marketRegime": summary.get("market_regime"),
      "squeezeBreakout": summary.get("squeeze_breakout"),
    },
    "entry": {
      "extensionAtrLong": emap.get("extensionAtrLong"),
      "extensionAtrShort": emap.get("extensionAtrShort"),
      "rsi15m": _r(emap.get("rsi15m"), 1),
      "fadeSetup": {"side": _side_word(fade.get("side")), "reason": fade.get("reason")} if fade else None,
    },
  }
  fut = analysis.get("futures") if isinstance(analysis.get("futures"), dict) else {}
  if fut:
    fs = fut.get("fundingSetup") if isinstance(fut.get("fundingSetup"), dict) else None
    rate = _num(fut.get("fundingRate"))
    state["futures"] = {
      "fundingRatePctPerSettlement": round(rate * 100, 5) if rate is not None else None,
      "fundingIntervalHours": fut.get("fundingIntervalHours"),
      "fundingCarry": {"paidSide": _side_word(fs.get("side")), "reason": fs.get("reason")} if fs else None,
      "fundingDivergence": fut.get("fundingDivergence"),
      "openInterestTrend": fut.get("oiTrend"),
      "oiPriceSignal": fut.get("oiPriceSignal"),
      "basisPct": _r(fut.get("basisPct"), 4),
      "priceChange24hPct": _r(fut.get("priceChgPct24h"), 3),
    }
  return state


# ── Bracket (code-built) ─────────────────────────────────────────────────────────────────────────────
def _snapshot(analysis: Dict[str, Any], interval: str) -> Dict[str, Any]:
  for snap in analysis.get("snapshots") or []:
    if isinstance(snap, dict) and snap.get("interval") == interval:
      return snap
  return {}


def pullback_anchor(analysis: Dict[str, Any], side: str) -> Optional[float]:
  """The nearest 15m value anchor (VWAP / BB mid) on the entry side of price, within the fillable band."""
  emap = ((analysis or {}).get("summary") or {}).get("entryMap") or {}
  price, atr = _num(emap.get("price")), _num(emap.get("atr15m"))
  if not price or not atr or atr <= 0:
    return None
  best = None
  for key in ("vwap15m", "bbMid15m"):
    anchor = _num(emap.get(key))
    if anchor is None or anchor <= 0:
      continue
    dist = (price - anchor) if side == "long" else (anchor - price)
    if PULLBACK_MIN_ATR * atr <= dist <= PULLBACK_MAX_ATR * atr and (best is None or dist < best[0]):
      best = (dist, anchor)
  return best[1] if best else None


def _target_gross(side: str, entry: float, stop_dist: float, net_r: float, rate: float) -> Optional[float]:
  """Gross target distance that makes regime.net_reward_risk_ratio == ``net_r`` (its cost model, solved)."""
  if side == "long":
    net_risk = stop_dist + (2 * entry - stop_dist) * rate
    denom = 1.0 - rate
  else:
    net_risk = stop_dist + (2 * entry + stop_dist) * rate
    denom = 1.0 + rate
  if denom <= 0 or net_risk <= 0:
    return None
  return (net_r * net_risk + 2 * entry * rate) / denom


def build_bracket(
  analysis: Dict[str, Any],
  side: str,
  entry_kind: str = "at_market",
  target_kind: str = "near",
  *,
  noise_mult: float,
  rr_floor: float,
  cost_rate: float,
) -> Optional[Dict[str, Any]]:
  """Entry / stop / target for ``side`` from the analysis levels. None when a valid bracket cannot be built.

  * entry — the live mark (``at_market``) or the nearest pullback anchor (``pullback``; falls back to the
    mark when no anchor is in the fillable band, and says so in ``entryKind``).
  * stop — at least the bot's own measured noise floor (``noise_mult`` x 15m ATR, the order path's floor),
    out to the 1h Bollinger band on the far side when that is wider, capped at
    STRUCTURAL_STOP_MAX_FLOOR_MULT x the floor.
  * target — the gross distance that nets ``near`` (fee-guard floor x cushion) or ``extended`` R after
    round-trip cost, on the same cost model as the order path's RR gate.
  """
  if side not in ("long", "short"):
    return None
  emap = ((analysis or {}).get("summary") or {}).get("entryMap") or {}
  fut = (analysis or {}).get("futures") or {}
  atr = _num(emap.get("atr15m"))
  price = _num(fut.get("markPrice")) or _num(emap.get("price"))
  if not price or price <= 0 or not atr or atr <= 0:
    return None
  kind = "at_market"
  entry = price
  if entry_kind == "pullback":
    anchor = pullback_anchor(analysis, side)
    if anchor:
      entry, kind = anchor, "pullback"
  floor = max(0.0, float(noise_mult or 0.0)) * atr
  if floor <= 0:
    floor = atr
  h1 = _snapshot(analysis, "1hour")
  band = _num(h1.get("bb_lower") if side == "long" else h1.get("bb_upper"))
  struct = (entry - band) if (band and side == "long") else ((band - entry) if band else None)
  stop_dist = floor
  if struct is not None and struct > floor:
    stop_dist = min(struct, STRUCTURAL_STOP_MAX_FLOOR_MULT * floor)
  stop = entry - stop_dist if side == "long" else entry + stop_dist
  rate = max(0.0, float(cost_rate or 0.0))
  near_r = max(float(rr_floor or 0.0), 1.0) * NEAR_TARGET_CUSHION
  net_r = near_r * (EXTENDED_TARGET_MULT if target_kind == "extended" else 1.0)
  gross = _target_gross(side, entry, stop_dist, net_r, rate)
  if gross is None or gross <= 0:
    return None
  tp = entry + gross if side == "long" else entry - gross
  if stop <= 0 or tp <= 0:
    return None
  net = net_reward_risk_ratio("buy" if side == "long" else "sell", entry, tp, stop, fee_rate=rate)
  return {
    "entry": entry,
    "stop": stop,
    "takeProfit": tp,
    "entryKind": kind,
    "targetKind": "extended" if target_kind == "extended" else "near",
    "stopAtr": round(stop_dist / atr, 2),
    "targetNetR": round(net_r, 2),
    "netRr": round(net, 3) if net is not None else None,
  }


# ── Questions & answers ──────────────────────────────────────────────────────────────────────────────
def jev_questions(state: Dict[str, Any], *, near_r: float, extended_r: float) -> Dict[str, Dict[str, Any]]:
  """The System One questions for one symbol (raw question dicts — the SDK accepts them as-is).

  Long and short are described in mirrored words; neither is the default. ``funding_carry`` is offered only
  when the state carries a live carry for some side — its mechanism is otherwise absent and the order path
  would refuse the label (regime.verify_declared_setup).
  """
  families = {
    "continuation": "Trading WITH an established trend that is expected to persist (timeframes agree).",
    "fade_extreme": "Fading a stretched move back toward value (an overbought top or an oversold bottom).",
    "breakout": "Entering on the break of a range or level, expecting expansion.",
    "range_edge": "Buying support or selling resistance inside a defined range.",
  }
  if (state.get("futures") or {}).get("fundingCarry"):
    families["funding_carry"] = (
      "Taking the side that funding PAYS at each settlement (futures.fundingCarry.paidSide); held across "
      "at least one settlement, it pays whichever way price moves."
    )
  return {
    "direction": {
      "type": "choice",
      "instructions": (
        "You trade this USDT perpetual future. Which position has the better expected return over the "
        "next 1 to 8 hours, after about 0.15% round-trip trading cost? A stop and a target are attached "
        "at entry either way, and a trailing stop protects any profit."
      ),
      "criteria": {
        "long": "Open a LONG now: price is more likely to rise far enough to pay the costs than to fall to the stop.",
        "short": "Open a SHORT now: price is more likely to fall far enough to pay the costs than to rise to the stop.",
        "stand_aside": "No position: the expected move in either direction is too small or too uncertain to pay the costs.",
      },
    },
    "setup_family": {
      "type": "choice",
      "instructions": "If a position is opened here, which playbook is it?",
      "criteria": families,
    },
    "entry": {
      "type": "choice",
      "instructions": "If a position is opened here, how should it enter?",
      "criteria": {
        "at_market": "Enter now at the live price; it fills immediately at the current level.",
        "pullback": (
          "Rest a limit at the nearest 15m value anchor (VWAP or Bollinger mid) on the entry side; a "
          "better price, but it may not fill before the order expires."
        ),
      },
    },
    "target": {
      "type": "choice",
      "instructions": "If a position is opened here, where should the take-profit sit?",
      "criteria": {
        "near": f"About {near_r:.1f}x the risk after costs: reached more often, smaller win.",
        "extended": f"About {extended_r:.1f}x the risk after costs: a bigger win, reached less often.",
      },
    },
  }


def _answer(resp: Any, name: str) -> Any:
  choices = getattr(resp, "choices", None)
  if isinstance(choices, dict) and name in choices:
    return choices[name]
  answers = getattr(resp, "answers", None)
  if isinstance(answers, dict):
    return answers.get(name)
  return None


def _probs(answer: Any) -> Dict[str, float]:
  raw = getattr(answer, "probabilities", None)
  if raw is None and isinstance(answer, dict):
    raw = answer.get("probabilities")
  out: Dict[str, float] = {}
  for key, val in (raw or {}).items():
    v = _num(val)
    if v is not None:
      out[str(key)] = round(max(0.0, min(1.0, v)), 4)
  return out


def _pick(answer: Any, allowed: tuple[str, ...] | list[str], default: Optional[str]) -> Optional[str]:
  """The chosen label when it is one we offered; else the most probable offered label; else ``default``."""
  label = getattr(answer, "choice", None)
  if label is None and isinstance(answer, dict):
    label = answer.get("choice")
  if label in allowed:
    return str(label)
  probs = {k: v for k, v in _probs(answer).items() if k in allowed}
  if probs:
    return max(probs, key=probs.get)
  return default


def parse_answers(resp: Any, *, families: List[str]) -> Dict[str, Any]:
  """Jev's answers as plain data: direction + its probability (the stated confidence), family, entry, target."""
  d_ans = _answer(resp, "direction")
  probs = _probs(d_ans)
  direction = _pick(d_ans, DIRECTIONS, None)
  confidence = probs.get(direction) if direction else None
  if confidence is None and direction:
    confidence = _num(getattr(d_ans, "confidence", None))
  usage = getattr(resp, "usage", None)
  return {
    "direction": direction,
    "confidence": round(confidence, 4) if confidence is not None else None,
    "probabilities": probs,
    "setupFamily": _pick(_answer(resp, "setup_family"), families, "continuation"),
    "entryKind": _pick(_answer(resp, "entry"), ("at_market", "pullback"), "at_market"),
    "targetKind": _pick(_answer(resp, "target"), ("near", "extended"), "near"),
    "model": str(getattr(resp, "model", "") or "")[:80] or None,
    "inputTokens": getattr(usage, "input_tokens", None) if usage is not None else None,
  }


# ── The pass ─────────────────────────────────────────────────────────────────────────────────────────
def _entries_today(memory: Any, now: float) -> int:
  day_start = int(now // 86400) * 86400
  try:
    rows = memory.jev_decisions(limit=0)
  except Exception:
    return 0
  return sum(1 for r in rows if r.get("outcome") == "placed" and int(r.get("ts") or 0) >= day_start)


def _outcome(result: Any, dry_run: bool) -> tuple[str, Optional[str]]:
  if not isinstance(result, dict):
    return "error", "no result"
  if result.get("rejected"):
    return "refused", str(result.get("gate") or result.get("reason") or "refused")[:160]
  if result.get("error"):
    return "error", str(result.get("error"))[:160]
  if dry_run or result.get("shadow"):
    return "shadow", None
  return "placed", None


def _build_client(cfg: Any) -> Any:
  """AsyncTypeSafeClient for THIS event loop (the pass runs in its own asyncio.run), or None."""
  try:
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
  except ImportError:
    _warn_once("sdk", "JEV: typesafe-sdk is not installed (pip install -r requirements.txt) — the dual run is idle")
    return None
  try:
    return AsyncTypeSafeClient(
      model=cfg.jev.model, timeout=float(cfg.jev.timeout_sec), retry=RetryPolicy(max_retries=1),
    )
  except Exception as exc:  # missing/invalid key is raised here
    _warn_once("client", "JEV: client unavailable (%s) — set TYPESAFE_API_KEY; the dual run is idle", exc)
    return None


async def run_jev_pass(
  cfg: Any,
  tools: Any,
  memory: Any,
  *,
  universe: List[str],
  equity_usd: float,
  noise_mult: float,
  cost_rate: float,
  client: Any = None,
  now: Optional[float] = None,
) -> Dict[str, Any]:
  """One dual-run pass: ask Jev about up to ``max_symbols_per_run`` symbols, then enter / shadow its calls.

  ``tools`` is the run's build_tools namespace (analysis cache, ownership, the order path); ``universe`` the
  tradeable spot symbols; ``equity_usd`` only seeds the requested notional (the order path sizes TO the risk
  budget); ``noise_mult`` and ``cost_rate`` are the run's measured stop floor and per-side cost. Returns a
  summary for the log; every per-symbol answer is also stored with ``memory.record_jev_decision``.
  """
  jcfg = cfg.jev
  now = float(now if now is not None else time.time())
  summary: Dict[str, Any] = {"mode": jcfg.mode, "asked": 0, "rows": []}
  if jcfg.mode not in ("shadow", "live"):
    return summary
  own_client = client is None
  if own_client:
    if not os.getenv("TYPESAFE_API_KEY", "").strip():
      _warn_once("key", "JEV: JEV_MODE=%s but TYPESAFE_API_KEY is not set — the dual run is idle", jcfg.mode)
      summary["skipped"] = "TYPESAFE_API_KEY not set"
      return summary
    client = _build_client(cfg)
    if client is None:
      summary["skipped"] = "client unavailable"
      return summary

  answers: List[Dict[str, Any]] = []
  try:
    # Candidates: what this run already analysed (free), then the rest of the universe in an hourly rotation
    # so every symbol is visited — Jev's record must not depend on which symbols the LLM chose to look at.
    analyses = tools.latest_analyses(max_age_sec=ANALYSIS_MAX_AGE_SEC)
    held = set(tools.trader_book("jev"))
    budget = max(1, int(jcfg.max_symbols_per_run))
    ordered = [s for s in analyses if s in universe and s not in held]
    rest = sorted(s for s in universe if s not in analyses and s not in held)
    if rest:
      k = int(now // 3600) % len(rest)
      rest = rest[k:] + rest[:k]
    candidates = ordered[:budget]
    for sym in rest:
      if len(candidates) >= budget:
        break
      try:
        res = await tools.analyze_for(sym)
      except Exception as exc:
        logger.warning("JEV: analysis failed for %s (%s)", sym, exc)
        continue
      if isinstance(res, dict) and not res.get("error"):
        analyses[sym] = res
        candidates.append(sym)

    near_r = max(float(cfg.trading.min_futures_rr or 0.0), 1.0) * NEAR_TARGET_CUSHION
    extended_r = near_r * EXTENDED_TARGET_MULT
    sem = asyncio.Semaphore(MAX_CONCURRENT_ASKS)

    async def ask(sym: str) -> Dict[str, Any]:
      state = jev_state(sym, analyses.get(sym) or {})
      if state is None:
        return {"symbol": sym, "outcome": "error", "detail": "analysis unusable"}
      questions = jev_questions(state, near_r=near_r, extended_r=extended_r)
      families = list(questions["setup_family"]["criteria"])
      async with sem:
        t0 = time.monotonic()
        try:
          resp = await client.system_one(state=state, questions=questions)
        except Exception as exc:
          return {"symbol": sym, "outcome": "error", "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}
        latency = int((time.monotonic() - t0) * 1000)
      parsed = parse_answers(resp, families=families)
      return {"symbol": sym, "latencyMs": latency, **parsed}

    answers = list(await asyncio.gather(*(ask(s) for s in candidates)))
    summary["asked"] = len(candidates)

    # Act: the most confident directional calls first, so a live slot goes to the strongest call.
    slots = 0
    if jcfg.mode == "live":
      slots = max(0, int(jcfg.max_open_positions) - len(held))
      slots = min(slots, max(0, int(jcfg.max_entries_per_day) - _entries_today(memory, now)))
    lev = float(cfg.trading.max_entry_leverage or 1.0) if cfg.trading.max_entry_leverage > 0 else 1.0
    directional = sorted(
      (a for a in answers if a.get("direction") in ("long", "short")),
      key=lambda a: -(a.get("confidence") or 0.0),
    )
    for row in answers:
      if row.get("direction") == "stand_aside":
        row["outcome"] = "stand_aside"
    for row in directional:
      sym = row["symbol"]
      bracket = build_bracket(
        analyses.get(sym) or {}, row["direction"], row.get("entryKind") or "at_market",
        row.get("targetKind") or "near",
        noise_mult=noise_mult, rr_floor=float(cfg.trading.min_futures_rr or 0.0), cost_rate=cost_rate,
      )
      if bracket is None:
        row.update(outcome="error", detail="no valid bracket from the analysis levels")
        continue
      row["bracket"] = {k: (_r(v, 8) if isinstance(v, float) else v) for k, v in bracket.items()}
      owner = tools.owner_of(sym)
      dry_run = not (jcfg.mode == "live" and slots > 0 and owner is None)
      trader = {"name": "jev", "model": row.get("model") or jcfg.model, "sizeScale": float(jcfg.risk_scale)}
      probs = row.get("probabilities") or {}
      rationale = (
        f"Jev dual run ({row.get('model') or jcfg.model}): P(long)={probs.get('long', 0):.2f} "
        f"P(short)={probs.get('short', 0):.2f} P(stand aside)={probs.get('stand_aside', 0):.2f}; "
        f"{row.get('setupFamily')}, {bracket['entryKind']} entry, {bracket['targetKind']} target "
        f"({bracket['targetNetR']}R net), code-built bracket."
      )
      try:
        result = await tools.place_futures_limit_order_for(
          trader, dry_run=dry_run,
          symbol=sym, side="buy" if row["direction"] == "long" else "sell",
          notional_usd=max(0.0, float(equity_usd or 0.0)), entry_price=bracket["entry"], leverage=lev,
          confidence=row.get("confidence"), rationale=rationale,
          take_profit_price=bracket["takeProfit"], stop_loss_price=bracket["stop"],
          setup_family=row.get("setupFamily"),
        )
      except Exception as exc:
        logger.warning("JEV: order path failed for %s (%s)", sym, exc)
        result = {"error": f"{type(exc).__name__}: {exc}"}
      outcome, detail = _outcome(result, dry_run)
      row["outcome"] = outcome
      if detail:
        row["detail"] = detail
      if dry_run and owner:
        row["heldBy"] = owner
      if outcome == "placed":
        slots -= 1
  except Exception as exc:
    logger.warning("JEV: pass failed (%s)", exc)
    summary["error"] = f"{type(exc).__name__}: {exc}"
  finally:
    if own_client and client is not None:
      try:
        await client.aclose()
      except Exception:
        pass

  for row in answers:
    row.setdefault("outcome", "error")
    row["ts"] = int(now)
    row["mode"] = jcfg.mode
    memory.record_jev_decision(row)
  summary["rows"] = answers
  return summary


def describe_pass(summary: Dict[str, Any]) -> str:
  """One log line for a pass."""
  if summary.get("skipped"):
    return f"JEV ({summary.get('mode')}): idle — {summary['skipped']}"
  rows = summary.get("rows") or []
  parts = []
  for r in rows:
    if r.get("direction") in ("long", "short"):
      conf = r.get("confidence")
      tag = f"{r['symbol']} {r['direction']}" + (f" {conf:.2f}" if isinstance(conf, (int, float)) else "")
      tag += f" [{r.get('outcome')}{': ' + r['detail'] if r.get('detail') else ''}]"
      parts.append(tag)
  stands = sum(1 for r in rows if r.get("outcome") == "stand_aside")
  errors = sum(1 for r in rows if r.get("outcome") == "error")
  line = f"JEV ({summary.get('mode')}): asked {summary.get('asked', 0)}, stand aside {stands}"
  if errors:
    line += f", errors {errors}"
  if parts:
    line += " | " + "; ".join(parts)
  if summary.get("error"):
    line += f" | pass error: {summary['error']}"
  return line


# ── Report (dashboard dualRun panel + Supervisor) ────────────────────────────────────────────────────
# Money words whose trailing figure must not leave the box (a refusal reason can quote a notional).
_MONEY_RE = re.compile(r"(\$|usdt?|notional|equity|balance|margin|risk)(\W{0,3})[-+]?\d[\d.,]*", re.I)
# An LLM call on the same symbol within this window counts as "the same moment" for the agreement tally.
AGREEMENT_WINDOW_SEC = 1800


def row_trader(d: Any) -> Optional[str]:
  """'jev' when a decision / close / probe row came from the dual run's second trader, else None."""
  if not isinstance(d, dict):
    return None
  ctx = d.get("entryContext") if isinstance(d.get("entryContext"), dict) else {}
  for value in (d.get("trader"), ctx.get("trader")):
    t = str(value or "").strip().lower()
    if t and t != DEFAULT_TRADER and t in KNOWN_TRADERS:
      return t
  return None


def scrub_detail(text: Any) -> Optional[str]:
  """A refusal's gate code verbatim, else its reason with any money figure removed, capped at 100 chars."""
  t = str(text or "").strip()
  if not t:
    return None
  if t in SCORED_GATES or t in STRUCTURAL_REFUSALS:
    return t
  return _MONEY_RE.sub(lambda m: m.group(1) + m.group(2) + "…", t)[:100]


def _f(value: Any) -> Optional[float]:
  return _num(value)


def dual_run_report(memory: Any, cfg: Any, *, cost_pct: float) -> Dict[str, Any]:
  """Both dual-run traders, each judged on its own calls over the SAME window (since Jev's first call).

  Signal edge (every call's forward return net of ``cost_pct``, % and t — the probe scoreboard), closes
  (count, wins, R-multiples), Jev's outcome counts and median latency, how often it agreed with the LLM on
  the same symbol at the same time, and its 25 latest answers. Percentages, ratios, counts, labels and gate
  codes only — no balance, size or $ figure. ``{"mode": "off"}`` alone when it never ran. Never raises.
  """
  jcfg = getattr(cfg, "jev", None)
  mode = str(getattr(jcfg, "mode", "off") or "off")
  out: Dict[str, Any] = {"mode": mode}
  try:
    recent = memory.jev_decisions(limit=0) if callable(getattr(memory, "jev_decisions", None)) else []
    if mode == "off" and not recent:
      return out
    out["model"] = getattr(jcfg, "model", None)
    since = min((int(r.get("ts") or 0) for r in recent if r.get("ts")), default=None)
    out["since"] = since
    horizons = safe_family_horizons(memory)
    weights = safe_family_horizon_weights(memory)
    closes = memory.realized_closes(limit=400)
    traders: Dict[str, Any] = {}
    for name in KNOWN_TRADERS:
      probes = [p for p in memory.signal_probes(limit=0, trader=name)
                if since is None or int(p.get("ts") or 0) >= since]
      stats = signal_edge_stats(probes, cost_pct=cost_pct, family_horizons=horizons, family_horizon_weights=weights)
      best = stats.get("best_horizon")
      row = (stats.get("by_horizon") or {}).get(best or "", {}) if best else {}
      mine = [c for c in closes if (row_trader(c) or DEFAULT_TRADER) == name
              and (since is None or int(c.get("ts") or 0) >= since)]
      wins = sum(1 for c in mine if (_f(c.get("pnl")) or 0.0) > 0)
      rs = [r for r in (_f(c.get("realizedR")) for c in mine) if r is not None]
      traders[name] = {
        "calls": int(stats.get("n") or 0),
        "verdict": stats.get("verdict", "insufficient data"),
        "bestHorizon": best,
        "netPct": _r(row.get("net_of_cost_pct"), 4),
        "tStat": _r(row.get("t_stat"), 2),
        "closes": len(mine),
        "wins": wins,
        "winRate": _r(wins / len(mine), 3) if mine else None,
        "avgR": _r(sum(rs) / len(rs), 3) if rs else None,
        "sumR": _r(sum(rs), 2) if rs else None,
      }
    out["traders"] = traders
    counts: Dict[str, int] = {}
    for r in recent:
      key = str(r.get("outcome") or "error")
      counts[key] = counts.get(key, 0) + 1
    out["outcomes"] = counts
    lat = sorted(int(r["latencyMs"]) for r in recent[-100:] if isinstance(r.get("latencyMs"), (int, float)))
    out["medianLatencyMs"] = lat[len(lat) // 2] if lat else None
    # Agreement: of Jev's directional calls with an LLM call on the same symbol within the window, how many
    # picked the same side — whether the second trader is a second opinion or an echo.
    llm_calls = [
      (normalize_symbol(str(p.get("symbol") or "")), int(p.get("ts") or 0),
       str(((p.get("entryContext") or {}).get("positionSide") or "")).lower())
      for p in memory.signal_probes(limit=0, trader=DEFAULT_TRADER)
    ]
    agree = disagree = 0
    for r in recent[-200:]:
      if r.get("direction") not in ("long", "short"):
        continue
      sym, ts = normalize_symbol(str(r.get("symbol") or "")), int(r.get("ts") or 0)
      near = [side for (s2, t2, side) in llm_calls if s2 == sym and abs(t2 - ts) <= AGREEMENT_WINDOW_SEC and side]
      if near:
        if r["direction"] in near:
          agree += 1
        else:
          disagree += 1
    out["agreement"] = {"agree": agree, "disagree": disagree}
    out["recent"] = [
      {
        "ts": r.get("ts"),
        "symbol": r.get("symbol"),
        "direction": r.get("direction"),
        "confidence": _r(r.get("confidence"), 3),
        "probabilities": {k: _r(v, 3) for k, v in (r.get("probabilities") or {}).items() if k in DIRECTIONS},
        "setupFamily": r.get("setupFamily"),
        "entryKind": (r.get("bracket") or {}).get("entryKind") or r.get("entryKind"),
        "targetNetR": (r.get("bracket") or {}).get("targetNetR"),
        "outcome": r.get("outcome"),
        "detail": scrub_detail(r.get("detail")),
        "latencyMs": r.get("latencyMs"),
      }
      for r in reversed(recent[-25:])
    ]
  except Exception as exc:  # report-only: never raises into the loop or the publisher
    logger.warning("dual-run report unavailable: %s", exc)
    out["error"] = f"{type(exc).__name__}"
  return out
