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

from .edge import _signed_probe_return, safe_family_horizon_weights, safe_family_horizons, signal_edge_stats
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


def _sdk_version() -> Optional[str]:
  try:
    from importlib.metadata import version
    return version("typesafe-sdk")
  except Exception:
    return None


def startup_line(cfg: Any) -> str:
  """One line at process start saying exactly what the dual run will do (logged even when it is off)."""
  jcfg = getattr(cfg, "jev", None)
  mode = str(getattr(jcfg, "mode", "off") or "off")
  if mode == "off":
    return "JEV DUAL RUN: off (JEV_MODE=off) — set JEV_MODE=shadow and TYPESAFE_API_KEY in .env, then restart"
  key = "set" if os.getenv("TYPESAFE_API_KEY", "").strip() else "MISSING (idle until TYPESAFE_API_KEY is set)"
  sdk = _sdk_version()
  ls = getattr(cfg, "langsmith", None)
  traced = bool(ls and ls.enabled and ls.tracing and ls.api_key)
  return (
    f"JEV DUAL RUN: mode={mode} model={jcfg.model} key={key} "
    f"sdk={'typesafe-sdk ' + sdk if sdk else 'MISSING (pip install -r requirements.txt)'} "
    f"caps: {jcfg.max_open_positions} open, {jcfg.max_entries_per_day}/day, {jcfg.max_symbols_per_run} symbols/run, "
    f"risk scale {jcfg.risk_scale:g} | langsmith={'on' if traced else 'off'}"
  )


def describe_row(row: Dict[str, Any]) -> str:
  """One log line per symbol Jev was asked about: its answer, the code-built bracket and what happened."""
  sym = row.get("symbol")
  if row.get("direction") is None:
    return f"JEV {sym}: {row.get('outcome') or 'error'} — {row.get('detail') or 'no answer'}"
  p = row.get("probabilities") or {}
  probs = f"(L {p.get('long', 0):.2f} / S {p.get('short', 0):.2f} / stand {p.get('stand_aside', 0):.2f})"
  conf = row.get("confidence")
  head = f"JEV {sym}: {row['direction'].upper().replace('_', ' ')} {conf:.2f} {probs}" if isinstance(conf, (int, float)) \
    else f"JEV {sym}: {row['direction']} {probs}"
  if row.get("direction") == "stand_aside":
    return head + (f" {row['latencyMs']}ms" if row.get("latencyMs") is not None else "")
  b = row.get("bracket") or {}
  plan = f" {row.get('setupFamily')} · {b.get('entryKind') or row.get('entryKind')} · {b.get('targetKind') or row.get('targetKind')}"
  if b:
    plan += f" {b.get('targetNetR')}R | entry {b.get('entry'):.8g} stop {b.get('stop'):.8g} tp {b.get('takeProfit'):.8g}"
  out = f"{row.get('outcome') or 'pending'}"
  if row.get("detail"):
    out += f" ({row['detail']})"
  if row.get("repeat"):
    out += " [repeat — not stored again]"
  elif row.get("recorded"):
    out += " [scored]"
  if row.get("stake"):
    out += f" stake {row['stake']}"
  if row.get("heldBy"):
    out += f" [{row['heldBy']} holds it]"
  return f"{head}{plan} → {out}"


def _status(cfg: Any, summary: Dict[str, Any], rows: List[Dict[str, Any]], now: float) -> Dict[str, Any]:
  """The pass's health record (memory ``jev_status``) — read by the dashboard and the Supervisor."""
  counts: Dict[str, int] = {}
  for r in rows:
    key = str(r.get("outcome") or "error")
    counts[key] = counts.get(key, 0) + 1
  lat = sorted(int(r["latencyMs"]) for r in rows if isinstance(r.get("latencyMs"), (int, float)))
  resolved = next((r.get("model") for r in rows if r.get("model")), None)
  state = "idle" if summary.get("skipped") else ("error" if summary.get("error") else "ok")
  return {
    "ts": int(now),
    "mode": summary.get("mode"),
    "model": getattr(cfg.jev, "model", None),
    "resolvedModel": resolved,
    "state": state,
    "reason": summary.get("skipped") or summary.get("error"),
    "asked": int(summary.get("asked") or 0),
    "outcomes": counts,
    "medianLatencyMs": lat[len(lat) // 2] if lat else None,
    "inputTokens": sum(int(r.get("inputTokens") or 0) for r in rows) or None,
    "traced": bool(summary.get("traced")),
  }


def _save_status(memory: Any, status: Dict[str, Any]) -> None:
  try:
    memory.set_jev_status(status)
  except Exception as exc:
    logger.warning("JEV: status not recorded (%s)", exc)


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
  tracer: Any = None,
) -> Dict[str, Any]:
  """One dual-run pass: ask Jev about up to ``max_symbols_per_run`` symbols, then enter / shadow its calls.

  ``tools`` is the run's build_tools namespace (analysis cache, ownership, the order path); ``universe`` the
  tradeable spot symbols; ``equity_usd`` only seeds the requested notional (the order path sizes TO the risk
  budget); ``noise_mult`` and ``cost_rate`` are the run's measured stop floor and per-side cost. Returns a
  summary for the log; every per-symbol answer is also stored with ``memory.record_jev_decision``, the pass's
  health with ``memory.set_jev_status`` (idle passes too), and each symbol gets one INFO log line.
  ``tracer`` (``langsmith_tracer``) receives the whole pass — states, questions, answers — for LangSmith.
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
    else:
      client = _build_client(cfg)
      if client is None:
        summary["skipped"] = ("typesafe-sdk not installed" if _sdk_version() is None
                              else "client unavailable (check TYPESAFE_API_KEY)")
    if summary.get("skipped"):
      _save_status(memory, _status(cfg, summary, [], now))
      return summary

  answers: List[Dict[str, Any]] = []
  trace_inputs: Dict[str, Dict[str, Any]] = {}
  started_at = time.time()
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
    from_run = len(candidates)
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

    logger.info("JEV (%s) pass: asking %d symbol(s) with %s — %d from this run's analyses, %d analysed now%s",
                jcfg.mode, len(candidates), jcfg.model, from_run, len(candidates) - from_run,
                f"; holding {', '.join(sorted(held))}" if held else "")
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
        started = time.time()
        try:
          resp = await client.system_one(state=state, questions=questions)
        except Exception as exc:
          trace_inputs[sym] = {"state": state, "questions": questions, "start": started, "end": time.time()}
          return {"symbol": sym, "outcome": "error", "detail": f"{type(exc).__name__}: {str(exc)[:120]}"}
        latency = int((time.monotonic() - t0) * 1000)
      trace_inputs[sym] = {"state": state, "questions": questions, "start": started, "end": time.time()}
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
      if isinstance(result, dict):
        # What the order path made of the call: was it stored as evidence (a repeat inside the family's
        # shortest horizon is not), and the stake an entry would get on Jev's own record.
        if result.get("shadow"):
          row["recorded"] = bool(result.get("probeRecorded"))
          row["repeat"] = bool(result.get("repeat"))
          if result.get("stake"):
            row["stake"] = str(result["stake"])[:80]
        elif outcome == "placed":
          row["recorded"] = True
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
    logger.info(describe_row(row))
  if tracer is not None and (answers or summary.get("error")):
    try:
      posted = tracer({
        "mode": jcfg.mode, "model": jcfg.model, "start": started_at, "end": time.time(),
        "rows": answers, "inputs": trace_inputs, "error": summary.get("error"),
      })
      summary["traced"] = bool(posted)
      # Each symbol's LangSmith run id rides on its decision row, so the market's answer can be attached
      # to that run as feedback once the call settles (langsmith_scorer).
      for row in answers:
        rid = posted.get(row.get("symbol")) if isinstance(posted, dict) else None
        if rid:
          row["lsRunId"] = rid
    except Exception as exc:
      logger.warning("JEV: LangSmith trace failed (%s)", exc)
  for row in answers:
    memory.record_jev_decision(row)
  summary["rows"] = answers
  _save_status(memory, _status(cfg, summary, answers, now))
  return summary


# ── LangSmith ────────────────────────────────────────────────────────────────────────────────────────
_LS_CLIENTS: Dict[tuple, Any] = {}
_ls_traced_once = False


def _trace_worthy(record: Dict[str, Any]) -> bool:
  """A pass that did something real: a live order attempt, a refusal of one, or an error."""
  if record.get("error"):
    return True
  for r in record.get("rows") or []:
    if r.get("outcome") in ("placed", "error") or (r.get("outcome") == "refused" and r.get("mode") == "live"):
      return True
  return False


def langsmith_tracer(cfg: Any, *, client: Any = None, rng: Any = None) -> Any:
  """A callable that posts one Jev pass to LangSmith as a trace, or None when LangSmith is off.

  One root run per pass ("Jev Dual Run", tags trAIde/jev/<mode>) with one ``llm`` child per symbol: inputs =
  the exact state and questions sent, outputs = Jev's answers, the code-built bracket and what the order
  path did, plus the model version and input tokens. Its own LangSmith client WITHOUT the agent's head
  sampling, so the policy here decides: the first pass after start, every pass that attempted or was
  refused a live entry or hit an error, and the rest at LANGSMITH_SAMPLE_RATE — the same budget the agent
  runs use (the monthly trace cap is why they are sampled). Returns True when the pass was posted.
  """
  ls = getattr(cfg, "langsmith", None)
  if not (ls and getattr(ls, "enabled", False) and getattr(ls, "tracing", False) and getattr(ls, "api_key", None)):
    return None
  import random as _random
  rand = rng or _random.random
  rate = min(1.0, max(0.0, float(getattr(ls, "sample_rate", 1.0) or 0.0)))

  def _client() -> Any:
    if client is not None:
      return client
    key = (ls.api_key, getattr(ls, "api_url", None))
    if key not in _LS_CLIENTS:
      from langsmith import Client
      _LS_CLIENTS[key] = Client(api_key=ls.api_key, api_url=getattr(ls, "api_url", None) or None)
    return _LS_CLIENTS[key]

  def trace(record: Dict[str, Any]) -> Dict[str, str]:
    global _ls_traced_once
    if _ls_traced_once and not _trace_worthy(record) and rand() >= rate:
      return {}
    from datetime import datetime, timezone
    from langsmith.run_trees import RunTree

    def ts(value: Any) -> Any:
      return datetime.fromtimestamp(float(value), tz=timezone.utc) if value else None

    rows = record.get("rows") or []
    counts: Dict[str, int] = {}
    for r in rows:
      counts[str(r.get("outcome") or "error")] = counts.get(str(r.get("outcome") or "error"), 0) + 1
    root = RunTree(
      name=f"Jev Dual Run ({record.get('mode')})", run_type="chain",
      inputs={"mode": record.get("mode"), "model": record.get("model"), "symbols": [r.get("symbol") for r in rows]},
      tags=["trAIde", "jev", str(record.get("mode"))],
      project_name=getattr(ls, "project", None) or None,
      client=_client(), start_time=ts(record.get("start")),
      extra={"metadata": {"ls_provider": "typesafe", "trader": "jev"}},
    )
    run_ids: Dict[str, str] = {}
    for r in rows:
      sent = (record.get("inputs") or {}).get(r.get("symbol")) or {}
      child = root.create_child(
        name=f"Jev {r.get('symbol')}", run_type="llm",
        inputs={"state": sent.get("state"), "questions": sent.get("questions")},
        start_time=ts(sent.get("start")),
        extra={"metadata": {"ls_provider": "typesafe", "ls_model_name": r.get("model") or record.get("model")}},
        tags=[str(r.get("outcome"))],
      )
      tokens = int(r.get("inputTokens") or 0)
      child.end(
        outputs={
          "direction": r.get("direction"), "confidence": r.get("confidence"),
          "probabilities": r.get("probabilities"), "setupFamily": r.get("setupFamily"),
          "entryKind": r.get("entryKind"), "targetKind": r.get("targetKind"),
          "bracket": r.get("bracket"), "outcome": r.get("outcome"), "detail": r.get("detail"),
          "heldBy": r.get("heldBy"), "latencyMs": r.get("latencyMs"),
          "usage_metadata": {"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
        },
        error=r.get("detail") if r.get("outcome") == "error" else None,
        end_time=ts(sent.get("end")),
      )
      if r.get("symbol"):
        run_ids[str(r["symbol"])] = str(child.id)
    root.end(outputs={"asked": len(rows), "outcomes": counts}, error=record.get("error"), end_time=ts(record.get("end")))
    root.post(exclude_child_runs=False)
    _ls_traced_once = True
    return run_ids or {"_root": str(root.id)}

  return trace


# A call is scored in LangSmith once its longest horizon has settled (or been written off): 240m plus the
# settlement tolerance. Bounded per pass — feedback posts are one HTTP call each, on the agent's thread.
SCORE_AFTER_SEC = (240 + 60) * 60
SCORE_GIVE_UP_SEC = 48 * 3600
MAX_SCORES_PER_PASS = 10


def langsmith_scorer(cfg: Any, *, client: Any = None) -> Any:
  """A callable(memory, now) that attaches the market's answer to traced Jev calls, or None when off.

  For each traced decision whose 240m window has settled: its probe's signed forward return at 15m / 60m /
  240m (%, price + funding — edge's one definition) as LangSmith feedback ``fwd_15m`` / ``fwd_60m`` /
  ``fwd_240m``, plus ``right_way_60m`` (1 if the call pointed the right way at 60m). A stand-aside has no
  direction and is not scored. Each decision is scored once (``lsScored``). Returns how many were scored.
  """
  ls = getattr(cfg, "langsmith", None)
  if not (ls and getattr(ls, "enabled", False) and getattr(ls, "tracing", False) and getattr(ls, "api_key", None)):
    return None

  def _client() -> Any:
    if client is not None:
      return client
    key = (ls.api_key, getattr(ls, "api_url", None))
    if key not in _LS_CLIENTS:
      from langsmith import Client
      _LS_CLIENTS[key] = Client(api_key=ls.api_key, api_url=getattr(ls, "api_url", None) or None)
    return _LS_CLIENTS[key]

  def score(memory: Any, now: Optional[float] = None) -> int:
    now = float(now if now is not None else time.time())
    due = [r for r in memory.jev_decisions(limit=0)
           if r.get("lsRunId") and not r.get("lsScored") and now - int(r.get("ts") or 0) >= SCORE_AFTER_SEC]
    if not due:
      return 0
    index = _probe_index(memory.signal_probes(limit=0, trader="jev"))
    done: List[tuple] = []
    for r in due:
      if len(done) >= MAX_SCORES_PER_PASS:
        break
      # Only a call that reached the probe (shadow, or a placed entry) has a forward return to score; a
      # stand-aside, an error or a refusal before the probe is marked done with no feedback.
      scoreable = r.get("direction") in ("long", "short") and (r.get("recorded") or r.get("outcome") == "placed")
      fwd = _forward_pcts(_nearest_probe(index, r.get("symbol"), r.get("direction"), int(r.get("ts") or 0))) \
        if scoreable else None
      if scoreable and (not fwd or fwd.get("240m") is None) and now - int(r.get("ts") or 0) < SCORE_GIVE_UP_SEC:
        continue                                   # not settled yet: try again next pass
      for key, val in (fwd or {}).items():
        if val is not None:
          _client().create_feedback(r["lsRunId"], key=f"fwd_{key}", score=float(val),
                                    comment=f"{r.get('symbol')} {r.get('direction')}: signed forward return, %")
      if fwd and fwd.get("60m") is not None:
        _client().create_feedback(r["lsRunId"], key="right_way_60m", score=1.0 if fwd["60m"] > 0 else 0.0)
      done.append((int(r.get("ts") or 0), r.get("symbol")))
    if done:
      memory.mark_jev_decisions_scored(done)
    return len(done)

  return score


def describe_comparison(report: Dict[str, Any]) -> Optional[str]:
  """One log line comparing the two traders over the same window (the dashboard panel, in text)."""
  tr = report.get("traders") or {}
  if not tr:
    return None

  def side(name: str) -> str:
    t = tr.get(name) or {}
    h = (t.get("byHorizon") or {}).get("60m") or {}
    net = h.get("netPct")
    hit = h.get("hitRate")
    return (f"{name.upper()} calls {t.get('calls', 0)}, @60m net {net:+.3f}% hit {hit * 100:.0f}% (n={h.get('n', 0)}), "
            f"closes {t.get('closes', 0)} avg {t.get('avgR') if t.get('avgR') is not None else '—'}R"
            if net is not None and hit is not None else
            f"{name.upper()} calls {t.get('calls', 0)} (60m not settled yet), closes {t.get('closes', 0)}")

  ag = report.get("agreement") or {}
  oc = report.get("outcomes") or {}
  since = report.get("since")
  when = time.strftime("%Y-%m-%d %H:%M", time.gmtime(since)) if since else "start"
  return (f"JEV vs LLM since {when} UTC — {side('jev')} | {side('llm')} | Jev outcomes "
          + ", ".join(f"{k} {v}" for k, v in sorted(oc.items()))
          + f" | same side as the LLM {ag.get('agree', 0)}/{ag.get('agree', 0) + ag.get('disagree', 0)}")


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


RECENT_ROWS = 40
# A decision row and its probe are written in the same pass (seconds apart); this bounds the match.
_PROBE_MATCH_SEC = 300
FORWARD_HORIZONS_MIN = (15, 60, 240)


def _probe_index(probes: List[Dict[str, Any]]) -> Dict[str, List[tuple]]:
  """symbol -> [(ts, side, entryContext)] for fast nearest-call lookups."""
  out: Dict[str, List[tuple]] = {}
  for p in probes or []:
    ctx = p.get("entryContext") if isinstance(p, dict) else None
    if not isinstance(ctx, dict):
      continue
    sym = normalize_symbol(str(p.get("symbol") or ""))
    out.setdefault(sym, []).append((int(p.get("ts") or 0), str(ctx.get("positionSide") or "").lower(), ctx))
  return out


def _nearest_call(index: Dict[str, List[tuple]], symbol: Any, ts: int) -> Optional[Dict[str, Any]]:
  rows = [(abs(t - ts), side, ctx) for (t, side, ctx) in index.get(normalize_symbol(str(symbol or "")), [])
          if abs(t - ts) <= AGREEMENT_WINDOW_SEC and side in ("long", "short")]
  if not rows:
    return None
  _, side, ctx = min(rows, key=lambda x: x[0])
  return {"side": side, "confidence": _r(ctx.get("confidence"), 3), "setupFamily": ctx.get("setupFamily")}


def _nearest_probe(index: Dict[str, List[tuple]], symbol: Any, direction: Any, ts: int) -> Optional[Dict[str, Any]]:
  if direction not in ("long", "short"):
    return None
  rows = [(abs(t - ts), ctx) for (t, side, ctx) in index.get(normalize_symbol(str(symbol or "")), [])
          if side == direction and abs(t - ts) <= _PROBE_MATCH_SEC]
  return min(rows, key=lambda x: x[0])[1] if rows else None


def _forward_pcts(ctx: Optional[Dict[str, Any]]) -> Optional[Dict[str, Optional[float]]]:
  """Signed forward return (%) of one call at 15m / 60m / 240m — edge's one definition (price + funding)."""
  if not isinstance(ctx, dict):
    return None
  base = _num(ctx.get("marketPriceAtSignal"))
  probe = ctx.get("signalProbe") if isinstance(ctx.get("signalProbe"), dict) else {}
  side = str(ctx.get("positionSide") or "").lower()
  if not base or base <= 0 or side not in ("long", "short"):
    return None
  out: Dict[str, Optional[float]] = {}
  for h in FORWARD_HORIZONS_MIN:
    try:
      v = _signed_probe_return(ctx, probe, base, side, h)
    except Exception:
      v = None
    out[f"{h}m"] = round(v * 100, 3) if v is not None else None
  return out


def public_status(status: Any) -> Optional[Dict[str, Any]]:
  """The last pass's health, whitelisted: when, idle/ok/error and why, how many asked, outcome counts,
  latency, the resolved model and whether it went to LangSmith. None before the first pass."""
  if not isinstance(status, dict) or not status.get("ts"):
    return None
  return {
    "ts": int(status.get("ts") or 0),
    "state": status.get("state") if status.get("state") in ("ok", "idle", "error") else "error",
    "reason": scrub_detail(status.get("reason")),
    "asked": int(status.get("asked") or 0),
    "outcomes": {str(k): int(v) for k, v in (status.get("outcomes") or {}).items() if isinstance(v, (int, float))},
    "medianLatencyMs": status.get("medianLatencyMs"),
    "resolvedModel": str(status.get("resolvedModel") or "")[:80] or None,
    "traced": bool(status.get("traced")),
  }


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
    out["status"] = public_status(memory.jev_status() if callable(getattr(memory, "jev_status", None)) else {})
    since = min((int(r.get("ts") or 0) for r in recent if r.get("ts")), default=None)
    out["since"] = since
    horizons = safe_family_horizons(memory)
    weights = safe_family_horizon_weights(memory)
    closes = memory.realized_closes(limit=400)
    traders: Dict[str, Any] = {}
    probes_by_trader = {name: memory.signal_probes(limit=0, trader=name) for name in KNOWN_TRADERS}
    for name in KNOWN_TRADERS:
      probes = [p for p in probes_by_trader[name] if since is None or int(p.get("ts") or 0) >= since]
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
        # Every scored horizon, not only the best one: n (de-overlapped), mean and net of cost in %, and
        # how often the call pointed the right way — the like-for-like comparison of the two traders.
        "byHorizon": {
          h: {"n": int(v.get("n") or 0), "meanPct": _r(v.get("mean_pct"), 4),
              "netPct": _r(v.get("net_of_cost_pct"), 4), "hitRate": _r(v.get("hit_rate"), 3)}
          for h, v in (stats.get("by_horizon") or {}).items() if isinstance(v, dict)
        },
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
      for p in probes_by_trader[DEFAULT_TRADER]
    ]
    llm_index = _probe_index(probes_by_trader[DEFAULT_TRADER])
    jev_index = _probe_index(probes_by_trader["jev"])
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
        "targetKind": (r.get("bracket") or {}).get("targetKind") or r.get("targetKind"),
        "targetNetR": (r.get("bracket") or {}).get("targetNetR"),
        "stopAtr": (r.get("bracket") or {}).get("stopAtr"),
        "outcome": r.get("outcome"),
        "detail": scrub_detail(r.get("detail")),
        "recorded": r.get("recorded"),
        "repeat": bool(r.get("repeat")),
        "stake": scrub_detail(r.get("stake")),
        "heldBy": r.get("heldBy"),
        "latencyMs": r.get("latencyMs"),
        # The LLM's own call on the same coin within the agreement window (None = it made none).
        "llm": _nearest_call(llm_index, r.get("symbol"), int(r.get("ts") or 0)),
        # How the market answered this call: its probe's signed forward return (%), once settled.
        "fwdPct": _forward_pcts(_nearest_probe(jev_index, r.get("symbol"), r.get("direction"), int(r.get("ts") or 0))),
      }
      for r in reversed(recent[-RECENT_ROWS:])
    ]
  except Exception as exc:  # report-only: never raises into the loop or the publisher
    logger.warning("dual-run report unavailable: %s", exc)
    out["error"] = f"{type(exc).__name__}"
  return out
