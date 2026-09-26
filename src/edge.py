"""Adaptive edge controller — self-tuning risk from the bot's own realized results.

The Jun–Jul 2026 reviews kept finding the same shape: high win rate, small wins, a few
oversized losses, and a bot that keeps re-taking the losing setup (e.g. re-shorting ETH
four times into the early-July oversold bounce). Hardcoded parameters fix yesterday's
regime; this module instead derives the risk posture from the ROLLING REALIZED OUTCOMES,
so the bot tightens up when it is losing and relaxes back when it is earning — with no
human re-tuning:

  - edge_stats:               rolling win rate / payoff / expectancy / loss streak / per-symbol PnL
  - expectancy_size_factor:  shrink risk for a losing direction or symbol
  - symbol_bench_until:       bench a symbol that keeps losing (auto-lifts after a cooldown)
  - loss_streak_size_factor:  shrink size during a losing streak (anti-martingale)

The older adaptive RR helpers remain for compatibility and offline comparisons, but the live
entry path deliberately keeps targets structural and adapts capital at risk instead.

Everything is a pure function over realized-close dicts ({symbol, pnl, ts, closeType}),
unit-tested in tests/test_edge.py. Call sites live in src/agent.py; config in EdgeConfig.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional

from .config import EdgeConfig


def _f(value: Any) -> float | None:
  try:
    if value is None:
      return None
    return float(value)
  except (TypeError, ValueError):
    return None


def _position_side(close: Dict[str, Any]) -> str | None:
  explicit = str(close.get("positionSide") or "").strip().lower()
  if explicit in {"long", "short"}:
    return explicit
  close_type = str(close.get("closeType") or "").upper()
  if "LONG" in close_type:
    return "long"
  if "SHORT" in close_type:
    return "short"
  action = str(close.get("action") or "").lower()
  if action.startswith("futures_sell"):
    return "long"
  if action.startswith("futures_buy"):
    return "short"
  return None


def edge_stats(closes: List[Dict[str, Any]], lookback: int) -> Dict[str, Any]:
  """Rolling performance stats over the last `lookback` realized closes.

  `closes` are realized-close dicts (pnl required); order does not matter — they are
  sorted by ts here. Returns zeroed stats when there is no usable data.
  """
  usable = [c for c in (closes or []) if _f(c.get("pnl")) is not None]
  usable.sort(key=lambda c: c.get("ts") or 0)
  window = usable[-max(1, int(lookback)):] if usable else []

  pnls = [float(c["pnl"]) for c in window]
  wins = [p for p in pnls if p > 0]
  losses = [p for p in pnls if p < 0]
  gross_win = sum(wins)
  gross_loss = -sum(losses)
  decided = len(wins) + len(losses)

  streak = 0
  for p in reversed(pnls):
    if p < 0:
      streak += 1
    elif p > 0:
      break

  per_symbol: Dict[str, Dict[str, Any]] = {}
  per_direction: Dict[str, Dict[str, Any]] = {}
  for c in window:
    pnl = float(c["pnl"])
    sym = str(c.get("symbol") or "?")
    row = per_symbol.setdefault(
      sym,
      {"n": 0, "net": 0.0, "losses": 0, "r_n": 0, "r_net": 0.0, "last_close_ts": 0},
    )
    row["n"] += 1
    row["net"] = round(row["net"] + pnl, 6)
    row["last_close_ts"] = int(c.get("ts") or 0)
    if pnl < 0:
      row["losses"] += 1
    realized_r = _f(c.get("realizedR"))
    if realized_r is not None and math.isfinite(realized_r):
      row["r_n"] += 1
      row["r_net"] = round(row["r_net"] + realized_r, 6)
    direction = _position_side(c)
    if direction:
      drow = per_direction.setdefault(
        direction,
        {
          "n": 0, "net": 0.0, "wins": 0, "losses": 0,
          "r_n": 0, "r_net": 0.0, "last_close_ts": 0,
        },
      )
      drow["n"] += 1
      drow["net"] = round(drow["net"] + pnl, 6)
      drow["last_close_ts"] = int(c.get("ts") or 0)
      if pnl > 0:
        drow["wins"] += 1
      elif pnl < 0:
        drow["losses"] += 1
      if realized_r is not None and math.isfinite(realized_r):
        drow["r_n"] += 1
        drow["r_net"] = round(drow["r_net"] + realized_r, 6)

  for row in list(per_symbol.values()) + list(per_direction.values()):
    row["r_expectancy"] = round(row["r_net"] / row["r_n"], 5) if row["r_n"] else None
  for row in per_direction.values():
    row["expectancy"] = round(row["net"] / row["n"], 5) if row["n"] else 0.0
    decided_count = row["wins"] + row["losses"]
    row["win_rate"] = round(row["wins"] / decided_count, 3) if decided_count else 0.0

  win_rate = (len(wins) / decided) if decided else 0.0
  avg_win = (gross_win / len(wins)) if wins else 0.0
  avg_loss = (gross_loss / len(losses)) if losses else 0.0
  last_close_ts = int(window[-1].get("ts") or 0) if window else 0
  return {
    "n": len(window),
    "wins": len(wins),
    "losses": len(losses),
    "win_rate": round(win_rate, 3),
    "avg_win": round(avg_win, 4),
    "avg_loss": round(avg_loss, 4),
    "payoff": round(avg_win / avg_loss, 3) if avg_loss > 0 else None,
    "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
    "net": round(sum(pnls), 4),
    "expectancy": round(sum(pnls) / len(pnls), 5) if pnls else 0.0,
    "loss_streak": streak,
    "last_close_ts": last_close_ts,
    "per_symbol": per_symbol,
    "per_direction": per_direction,
  }


def entry_quality_stats(closes: List[Dict[str, Any]], lookback: int) -> Dict[str, Any]:
  """Post-trade ENTRY-QUALITY aggregation over recent closes — decision-support, never a gate.

  For each close, from data already recorded, it derives:
    - mae_r:   max adverse excursion in R = |troughPnl| / planned risk — how far price went AGAINST the
               entry before the trade worked. A high value means a better arrival price was available
               (the pullback), i.e. the entry was early/chased.
    - mfe_r:   max favorable excursion in R = peakPnl / planned risk.
    - entry_extension_atr: how stretched the entry was vs the 15m VWAP at fill (stamped in entryContext).

  These are fed back to the model so it sharpens its own entry timing (rest the limit at the pullback
  when recent entries show high adverse excursion / high extension). It imposes NO restriction: entry
  timing stays the model's judgement and improves as the model improves. Zeroed when no usable sample.
  """
  usable: List[Dict[str, Any]] = []
  for c in closes or []:
    ctx = c.get("entryContext") if isinstance(c.get("entryContext"), dict) else {}
    planned_risk = _f(ctx.get("plannedMaxLossUsd"))
    trough = _f(c.get("troughPnl"))
    if planned_risk is None or planned_risk <= 0 or trough is None:
      continue
    peak = _f(c.get("peakPnl"))
    mae_r = max(0.0, -trough) / planned_risk
    mfe_r = (max(0.0, peak) / planned_risk) if peak is not None else None
    realized_r = _f(c.get("realizedR"))
    ext = _f(ctx.get("entryExtensionAtr"))
    usable.append({
      "ts": c.get("ts") or 0,
      "symbol": c.get("symbol"),
      "mae_r": round(mae_r, 3),
      "mfe_r": round(mfe_r, 3) if mfe_r is not None else None,
      "realized_r": round(realized_r, 3) if realized_r is not None else None,
      "entry_extension_atr": round(ext, 2) if ext is not None else None,
    })
  usable.sort(key=lambda r: r["ts"])
  window = usable[-max(1, int(lookback)):] if usable else []
  if not window:
    return {"n": 0}

  def _avg(vals: List[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None

  mae_vals = [r["mae_r"] for r in window]
  ext_vals = [r["entry_extension_atr"] for r in window if r["entry_extension_atr"] is not None]
  # "Better entry was available" = the trade dipped a meaningful fraction of its risk against the fill
  # before working; a purely descriptive label (not a threshold that blocks anything).
  better_entry = [r for r in window if r["mae_r"] >= 0.5]
  worst = max(window, key=lambda r: r["mae_r"]) if window else None
  # TARGET REACHABILITY — how far price ACTUALLY travelled in your favour, as a share of trades that
  # reached each R milestone. This is the check the bot was missing: over 27 live lifecycles the median
  # favourable excursion was 0.27R while every bracket was planned at 2.3-2.7R gross, so **no trade
  # ever reached its take-profit**. A target beyond the distribution below is not ambitious, it is
  # unreachable — and it drags the stop in tight to keep the RR ratio, which is how a good read still
  # loses. Decision-support only: nothing here vetoes a setup.
  mfe_vals = sorted(r["mfe_r"] for r in window if r["mfe_r"] is not None)
  reached = {}
  if mfe_vals:
    for level in (0.5, 1.0, 1.5, 2.0, 3.0):
      reached[f"{level:g}R"] = round(sum(1 for v in mfe_vals if v >= level) / len(mfe_vals), 3)
  median_mfe = mfe_vals[len(mfe_vals) // 2] if mfe_vals else None
  return {
    "n": len(window),
    "avg_mae_r": _avg(mae_vals),
    "avg_mfe_r": _avg([r["mfe_r"] for r in window]),
    "median_mfe_r": round(median_mfe, 3) if median_mfe is not None else None,
    "mfe_reached_rate": reached,
    "avg_entry_extension_atr": _avg(ext_vals) if ext_vals else None,
    "better_entry_rate": round(len(better_entry) / len(window), 3),
    "worst_entry": {"symbol": worst["symbol"], "mae_r": worst["mae_r"], "entry_extension_atr": worst["entry_extension_atr"]} if worst else None,
  }


def _percentile(values: List[float], pct: float) -> float | None:
  """Nearest-rank percentile. Small, dependency-free, and good enough for <100 samples."""
  vals = sorted(v for v in values if v is not None and math.isfinite(v))
  if not vals:
    return None
  idx = max(0, min(len(vals) - 1, int(math.ceil(pct * len(vals))) - 1))
  return vals[idx]


def measured_slippage_pct(
  fills: List[Dict[str, Any]],
  prior: float,
  *,
  min_samples: int = 8,
  percentile: float = 0.8,
  floor: float = 0.0001,
  cap_mult: float = 3.0,
) -> Dict[str, Any]:
  """Per-side slippage estimated from the bot's OWN fills, instead of a hand-set constant.

  Every RR gate, net-profit check and fee-adjusted breakeven prices friction as
  ``fee_rate + estimated_slippage_pct`` *per side*. A number set once and never revisited silently
  becomes the strategy: the live config assumed 0.10%/side while measured entry slippage was
  **0.008% mean / 0.025% p90** — a ~12x overstatement. Round-trip that is 0.32% of notional against
  a real ~0.08%, and at the account's median risk/notional (1.3%) it charges every setup a phantom
  **0.18R**. To still clear a 1.5 net-RR floor the model had to plan ~2.7R *gross* targets, which at
  the same time pushed stops in tight — and the sample's median favourable excursion was 0.27R, so
  no trade ever reached one. Overstating costs does not make a bot conservative; it makes it plan
  trades that cannot win.

  So: measure it. Uses a high percentile (not the mean) so the estimate stays conservative, needs
  ``min_samples`` fills before it displaces the prior, and is clamped to ``[floor, cap_mult*prior]``
  so neither a data glitch nor a run of perfect fills can drive friction to an absurd value. It
  adapts in BOTH directions — if execution genuinely degrades the estimate rises on its own.

  Args:
    fills: trade records with a planned ``price`` and an achieved ``fillPrice``.
    prior: the configured ``estimated_slippage_pct`` — used when the sample is too thin.
  Returns ``{"value", "source", "n", ...}``; ``value`` is always usable.
  """
  prior_val = _f(prior)
  prior_val = max(0.0, prior_val if prior_val is not None else 0.0)
  devs: List[float] = []
  for t in fills or []:
    if not t.get("filled"):
      continue
    want = _f(t.get("price"))
    got = _f(t.get("fillPrice"))
    if want is None or got is None or want <= 0 or got <= 0:
      continue
    devs.append(abs(got - want) / want)
  if len(devs) < max(1, int(min_samples)):
    return {"value": prior_val, "source": "prior", "n": len(devs), "prior": prior_val}
  measured = _percentile(devs, percentile)
  if measured is None:
    return {"value": prior_val, "source": "prior", "n": len(devs), "prior": prior_val}
  cap = prior_val * max(1.0, float(cap_mult)) if prior_val > 0 else max(float(floor), measured)
  value = min(max(float(measured), float(floor)), cap)
  return {
    "value": value,
    "source": "measured",
    "n": len(devs),
    "prior": prior_val,
    "p80": round(measured, 6),
    "mean": round(sum(devs) / len(devs), 6),
    "capped": value < measured - 1e-12,
  }


SETUP_FAMILIES = ("continuation", "fade_extreme", "breakout", "range_edge", "funding_carry", "macro_event", "other")


def infer_setup_family(entry_context: Dict[str, Any]) -> str:
  """Best-effort family label when the model did not declare one, from data already stamped.

  Only used as a fallback so historical and untagged entries still group somewhere sensible; the
  model's own declaration always wins.
  """
  ctx = entry_context if isinstance(entry_context, dict) else {}
  declared = str(ctx.get("setupFamily") or "").strip().lower()
  if declared in SETUP_FAMILIES:
    return declared
  regime = ctx.get("regime") if isinstance(ctx.get("regime"), dict) else {}
  side = str(ctx.get("positionSide") or "").lower()
  bias_4h = str(regime.get("intraday_bias_4h") or "").lower()
  bias_1h = str(regime.get("intraday_bias_1h") or "").lower()
  want = "bullish" if side == "long" else "bearish"
  if bias_4h == want and bias_1h == want:
    return "continuation"
  if bias_4h and bias_4h != want:
    return "fade_extreme"
  return "other"


SIDES = ("long", "short")


def _norm_side(side: Any) -> str | None:
  """'long'/'short' from any of the spellings the order path and the probes use; None otherwise."""
  s = str(side or "").strip().lower()
  if s in ("buy", "long"):
    return "long"
  if s in ("sell", "short"):
    return "short"
  return None


def family_evidence_row(
  signal_edge: Dict[str, Any],
  family: str,
  side: Any = None,
  *,
  min_samples: int = 20,
) -> Dict[str, Any]:
  """Which measured row judges a bet on ``family`` taken on ``side``.

  The Kelly stake is a bet on a SIDE of a playbook, not on the playbook's label. Pooling the two sides
  lets one side's record size the other: on 2026-09-24 `continuation` read 'edge' pooled, but that
  edge was carried entirely by its longs (n=40, +1.24% at 240m) while its shorts sat at n=9, -0.18%
  ± 1.05% — and three SPX continuation SHORTS were sized 0.52-0.79 on the longs' record. So:

  * ``side=None`` — the pooled family row, exactly as before (dashboard, open_families, older callers).
  * a side whose OWN row has ``n >= min_samples`` — that row alone judges it (stand-aside, the
    measured no-edge shrink and the explore ramp all read it).
  * a side still thin on its own — the POOLED row judges the stand-aside and the no-edge shrink (a
    losing playbook stays losing whichever side is asked), but ``sideThin`` is set so the explore
    factor caps it at the explore floor: an unproven side never inherits the other side's upsizing.

  Symmetric across regimes (in a downtrend shorts earn their own verdict), no new constant — the
  sample floor is the stand-aside's own ``min_samples``. The de-overlap that produced the side rows is
  ``_probe_observations``' per-symbol rule, applied BEFORE the split (see ``signal_edge_stats``).
  """
  fam = str(family or "other").strip().lower()
  board = signal_edge if isinstance(signal_edge, dict) else {}
  by_family = board.get("by_family") if isinstance(board.get("by_family"), dict) else {}
  pooled = by_family.get(fam) if isinstance(by_family.get(fam), dict) else None
  s = _norm_side(side)
  if s is None:
    return {"family": fam, "row": pooled, "judgedOn": fam, "side": None, "sideN": None, "sideThin": False}
  by_side = board.get("by_family_side") if isinstance(board.get("by_family_side"), dict) else {}
  fam_sides = by_side.get(fam) if isinstance(by_side.get(fam), dict) else {}
  side_row = fam_sides.get(s) if isinstance(fam_sides.get(s), dict) else None
  side_n = int(side_row.get("n") or 0) if side_row else 0
  if side_row is not None and side_n >= max(1, int(min_samples)):
    return {"family": fam, "row": side_row, "judgedOn": f"{fam}:{s}", "side": s, "sideN": side_n,
            "sideThin": False}
  return {"family": fam, "row": pooled, "judgedOn": fam, "side": s, "sideN": side_n, "sideThin": True}


def family_stake_status(
  signal_edge: Dict[str, Any],
  family: str,
  *,
  min_samples: int = 20,
  side: Any = None,
) -> Dict[str, Any]:
  """Is this bet at zero stake, and WHY — the stand-aside decision plus the numbers behind it.

  ``reason`` is ``'no edge'`` when the judging row's net is at or below zero (the calls do not clear
  their round trip), ``'unproven: net inside its own SE (t<1)'`` when net is positive but smaller than
  its own standard error, and None when the bet is staked. The two are different facts and the model
  must be told the right one: on 2026-09-24 all nine continuation stand-asides fired with net +0.41%
  to +0.54% (verdict 'edge', t≈0.7-0.9), yet the refusal said 'NO EDGE … coin-flip minus fees' and
  the model repeated 'edge does not clear costs' in its own decline. ``family_stand_aside`` is the
  thin boolean view of this, so there is one rule and no second copy to drift.

  ``judgedOn`` names the row that decided (``'continuation:short'`` for a side judged on its own
  record, ``'continuation'`` for the pooled row); ``sideThin`` says a side was too thin to be judged on
  its own (see ``family_evidence_row``).
  """
  ev = family_evidence_row(signal_edge, family, side, min_samples=min_samples)
  row = ev["row"]
  n = int(row.get("n") or 0) if isinstance(row, dict) else 0
  net = _f(row.get("net_of_cost_pct")) if isinstance(row, dict) else None
  se = _f(row.get("stderr_pct")) if isinstance(row, dict) else None
  t = (net / se) if (net is not None and se is not None and se > 0) else None
  out = {
    "standAside": False, "reason": None,
    "family": ev["family"], "judgedOn": ev["judgedOn"], "side": ev["side"],
    "sideN": ev["sideN"], "sideThin": ev["sideThin"],
    "n": n, "verdict": row.get("verdict") if isinstance(row, dict) else None,
    "netPct": net, "sePct": se, "tStat": round(t, 2) if t is not None else None,
  }
  if not isinstance(row, dict) or n < max(1, int(min_samples)):
    return out
  if row.get("verdict") == "no edge":
    out.update(standAside=True, reason="no edge")
    return out
  # A stateless release bar at t = net/SE >= 1, recomputed from the live probes every run. NOT
  # hysteresis — there is no state, and this comment used to claim otherwise ("entering still only
  # needs the sign"). The rule that actually runs: stake is zero while the net is smaller than its own
  # standard error. t = 1 is the zero point of uncertainty-shrunk Kelly, (1 - 1/t²)+: below it the
  # measured edge is inside its own noise and the growth-optimal stake on it is nil. Because it is
  # stateless it can flip back and forth on noise near the line — live on 2026-09-24 continuation was
  # refused at 09:23, admitted at 09:40, refused 09:54-12:35 and admitted at 13:00/13:07 (t hovering
  # at 0.9-1.02); the 13:07 SPX short that got through lost -0.98R. What bounds the cost of a flip is
  # the size, not a state machine: at the line ``family_explore_factor`` stakes only its explore floor
  # (0 -> 0.4x, never 0 -> 1.0x). Stateless is also restart-safe (no in-process verdict to lose), and
  # stateful hysteresis ("release at t>1, bench only below 0") would have re-opened all nine of that
  # morning's refusals — a loosening that needs its own evidence. The band is the sample's own
  # dispersion, so it tightens as evidence accumulates and a family that genuinely pays escapes.
  if net is None or se is None or se <= 0:
    return out
  if net < se:
    out.update(standAside=True, reason="no edge" if net <= 0 else "unproven: net inside its own SE (t<1)")
  return out


def describe_stake_row(status: Dict[str, Any]) -> str:
  """One phrase naming the row that judged a bet, for the refusal the model reads and the operator log.

  Without it a side judged on the pooled record reads as if its own calls had been measured, which is
  precisely the confusion the side split exists to remove.
  """
  st = status if isinstance(status, dict) else {}
  fam = st.get("family") or "other"
  side = st.get("side")
  if side and st.get("sideThin"):
    return (f"{fam}:{side} n={int(st.get('sideN') or 0)} is unproven, judged on pooled {fam} "
            f"(n={int(st.get('n') or 0)})")
  if side:
    return f"judged on its own {fam}:{side} record (n={int(st.get('n') or 0)})"
  return f"judged on pooled {fam} (n={int(st.get('n') or 0)})"


def family_size_factor(
  signal_edge: Dict[str, Any],
  family: str,
  *,
  min_factor: float = 0.25,
  min_samples: int = 20,
  side: Any = None,
) -> float:
  """Risk multiplier for a setup family, from its OWN measured forward-return edge.

  The point of tagging families is that the bot no longer needs anybody to decide whether it should
  be trend-following or fading — it measures each playbook separately and lets capital follow whatever
  currently pays. Measured on the live universe (50d, 12 symbols, 4h holding, net of a 0.10% round
  trip): the continuation family returned -0.017% gross over 3,408 samples, i.e. flat, and flat does
  not cover costs. Fading extremes was positive in both halves of the period but only t~1.0-1.6 over
  135 independent events — suggestive, not established. Neither of those is a fact to hardcode; both
  are hypotheses this factor keeps score on.

  Never enlarges risk (mirrors ``expectancy_size_factor``): a family that clears the cost hurdle simply
  keeps full configured risk. An unproven family is left at 1.0 so it can gather the evidence that
  judges it — otherwise a new playbook could never earn its way in.

  The penalty is PROPORTIONAL to how badly the family misses, expressed as its shortfall in units of
  the cost hurdle it has to clear — so there is no tuned constant, and a family that is marginally
  short is treated very differently from one that is deeply negative. Live on 2026-08-10 the
  continuation family measured -0.29% net against a 0.166% hurdle: a shortfall of 1.75x, i.e. it loses
  nearly two round-trips of cost on every signal, so it collapses to the floor.

  ``min_factor`` is deliberately NOT zero. Position size is stop-defined, so driving it to nil pushes
  notional under the exchange's contract minimum, the order is rejected, no probe is recorded, and the
  family can never produce the evidence that would let it recover — the same doom loop that the memory
  retention fix had to undo. A quartered position still trades, still generates probes, and still
  recovers on its own if the measurement improves.

  ``side`` (optional) judges the bet on that side's own row when it has a real sample, else on the
  pooled row — see ``family_evidence_row``. ``side=None`` is the pooled behaviour, unchanged.
  """
  row = family_evidence_row(signal_edge, family, side, min_samples=min_samples)["row"]
  if not isinstance(row, dict):
    return 1.0
  if int(row.get("n") or 0) < max(1, int(min_samples)):
    return 1.0
  if row.get("verdict") != "no edge":
    return 1.0
  floor = max(0.0, min(1.0, float(min_factor)))
  net = _f(row.get("net_of_cost_pct"))
  hurdle = _f((signal_edge or {}).get("cost_pct"))
  if net is None or hurdle is None or hurdle <= 0:
    return floor
  shortfall = max(0.0, -net) / (hurdle * 100.0)   # cost_pct is a fraction; net_of_cost_pct is a percent
  return max(floor, min(1.0, 1.0 - shortfall))


def family_stand_aside(
  signal_edge: Dict[str, Any],
  family: str,
  *,
  min_samples: int = 20,
  side: Any = None,
) -> bool:
  """Should the bot DECLINE to execute this setup because its measured edge is not there?

  ``family_size_factor`` shrinks a losing playbook but floors at a quarter-size, because driving a
  stop-defined position to nil once pushed notional under the contract minimum, the order was rejected,
  and — back when probes were recorded only on placed orders — the family then starved of the very
  evidence that would let it recover. That floor was a workaround for a doom loop that no longer exists:
  direction calls are now recorded as probes at call time, before any sizing or RR rejection
  (``memory.record_signal_probe``), so a *skipped* trade still feeds the measurement and the family can
  climb back on its own.

  With the evidence supply decoupled from execution, the floor is free to fall to its
  mathematically-correct value. What actually runs (see ``family_stake_status``, of which this is the
  boolean view) is ONE stateless bar, recomputed every run from the live probes: on a real sample
  (``n >= min_samples``) the stake is zero while the judging row's net-of-cost return is smaller than
  its own standard error, t = net / SE < 1 — which includes every 'no edge' row (net <= 0) and also a
  positive net that is still inside its own noise. t = 1 is the zero point of uncertainty-shrunk Kelly,
  (1 - 1/t²)+: below it the growth-optimal stake is nil, and "zero" for a stop-defined position means
  do not place it. It is not hysteresis — nothing is remembered between runs — so near the line it can
  flip back and forth on noise; ``family_explore_factor`` bounds what a flip costs by staking only its
  explore floor at t = 1 (a 0 -> 0.4x step, never 0 -> 1.0x). This is bet-sizing (survival), not a view
  on which coin or direction is right (opportunity): it fires only on the bot's OWN measurement of its
  OWN direction calls, and reverses automatically the moment that measurement clears one SE.

  An unproven family (``n < min_samples``) is never stood aside — it trades at the explore size while
  it earns its verdict. ``side`` judges the bet on that side's own row once it has a real sample (see
  ``family_evidence_row``); ``side=None`` is the pooled behaviour.
  """
  return bool(family_stake_status(signal_edge, family, min_samples=min_samples, side=side)["standAside"])


def open_families(signal_edge: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
  """Which playbooks are actually available right now, split by how well proven they are.

  Wisdom, not a gate. When a stand-aside refuses a setup the model needs somewhere to go, and the
  answer has to come from the *current* scoreboard: a hardcoded suggestion goes stale silently and
  then points at a family that is itself stood aside, which is exactly what happened live on
  2026-09-07 (the hint said "take a genuine fade_extreme" while fade_extreme sat at n=38,
  net -1.12%, blocked). Nothing here decides anything — it names what the bot's own measurement
  currently says, so the model can redirect instead of re-proposing the family it was just refused.

  ``paying`` = measured edge, best net first. ``unproven`` = not yet judged, so still open to trade
  and still worth evidence. Families that are stood aside on BOTH sides appear in neither. Since a bet
  is judged per side (``family_evidence_row``), a family open on only ONE side is listed with
  ``side`` and that side's judging row — otherwise a refusal of continuation shorts would either hide
  that its longs are open on their own record, or advertise the shorts as open when they are not.
  """
  by_family = (signal_edge or {}).get("by_family") or {}
  paying: List[Dict[str, Any]] = []
  unproven: List[Dict[str, Any]] = []
  for fam in SETUP_FAMILIES:
    row = by_family.get(fam)
    if not isinstance(row, dict):
      continue
    status = {s: family_stake_status(signal_edge, fam, side=s) for s in SIDES}
    open_sides = [s for s in SIDES if not status[s]["standAside"]]
    if not open_sides:
      continue
    if len(open_sides) == len(SIDES):
      entry = {"family": fam, "n": int(row.get("n") or 0), "netOfCostPct": _f(row.get("net_of_cost_pct"))}
      verdict = row.get("verdict")
    else:
      st = status[open_sides[0]]
      entry = {"family": fam, "side": open_sides[0], "n": int(st["n"] or 0), "netOfCostPct": st["netPct"]}
      verdict = st["verdict"]
    if verdict == "edge":
      paying.append(entry)
    elif entry["n"] > 0:
      unproven.append(entry)
  paying.sort(key=lambda e: -(e["netOfCostPct"] or 0.0))
  unproven.sort(key=lambda e: -e["n"])
  return {"paying": paying, "unproven": unproven}


def family_explore_factor(
  signal_edge: Dict[str, Any],
  family: str,
  *,
  explore_factor: float = 0.4,
  min_samples: int = 20,
  full_size_tstat: float = 2.0,
  side: Any = None,
) -> float:
  """Risk multiplier for a setup family that has not yet earned a scored verdict.

  ``family_size_factor`` leaves an unproven family at full risk on the reasoning that a new playbook
  "could never earn its way in" if it were shrunk. That reasoning predates call-time probing and no
  longer holds: ``memory.record_signal_probe`` writes the probe from the MARKET price at signal time,
  before any sizing, so the forward-return evidence that scores a family is entirely independent of the
  notional we put behind it. A family gathers its ``min_samples`` probes at the same rate whether we
  size it at 1.0, 0.4, or skip it — the evidence is size-independent. That severs the old link between
  "explore" and "risk full size": we can measure a new playbook while risking little on it.

  This matters because opening the alignment gates to deliberately-declared playbooks (breakout,
  range_edge — see ``regime.allow_declared_setup``) lets families reach the book that have NO score yet
  and, on a ~$70 account, full-risk exploration of an unproven hypothesis is exactly the overtrading
  that fees punish. So while a family is still earning its verdict, it trades at ``explore_factor`` of
  configured risk. Once it crosses ``min_samples`` the size RAMPS with the strength of the evidence
  rather than jumping to full: from ``explore_factor`` when the edge only just clears the stand-aside's
  release bar (t = net / SE = 1) up to 1.0 at ``full_size_tstat``. Cheap to learn, weight in proportion
  to proof, zero once disproven.

  Why a ramp and not a switch (2026-09-19): this used to return 1.0 the instant n reached 20, whatever
  the evidence said. `funding_carry` graduated at n=24 on net +3.78% with a standard error of 3.20% —
  t = 1.18, barely distinguishable from noise — and its risk jumped 0.40x -> 1.00x in one 2.5x step.
  The first full-size trade after that was a G-USDT short into a +28.6%-in-25-minutes spike, stopped
  for -1.04R at 2.2x the size of the same trade an hour earlier. The size should follow what the
  measurement actually supports. ``full_size_tstat`` is a statement about statistical confidence (two
  standard errors), not about the market, so it does not need re-tuning as conditions change — the
  family's own net and SE, recomputed every run from live probes, do all the adapting.

  Combine by taking the WORSE of this and ``family_size_factor`` — never their product — for the same
  reason the soft stack does: two independent cautions must not compound into fee-dust.

  ``side`` (optional): a side with its own real sample ramps on its OWN row; a side still thin on its
  own is capped at the explore floor, whatever the pooled family has earned — the pooled t is the
  other side's evidence, and an unproven side must not inherit its upsizing (2026-09-24: continuation
  shorts, n=9 at -0.18%, sized 0.52-0.79 on the longs' n=40 at +1.24%). ``side=None`` is unchanged.
  """
  ev = family_evidence_row(signal_edge, family, side, min_samples=min_samples)
  row = ev["row"]
  n = int(row.get("n") or 0) if isinstance(row, dict) else 0
  floor = max(0.0, min(1.0, float(explore_factor)))
  if ev["sideThin"] or n < max(1, int(min_samples)):
    return floor
  net = _f(row.get("net_of_cost_pct"))
  se = _f(row.get("stderr_pct"))
  if net is None or se is None or se <= 0:
    return 1.0   # no dispersion recorded (older payload): keep the previous behaviour
  t = net / se
  span = max(1e-9, float(full_size_tstat) - 1.0)
  proof = max(0.0, min(1.0, (t - 1.0) / span))
  return floor + (1.0 - floor) * proof


def family_stake(
  signal_edge: Dict[str, Any],
  family: str,
  *,
  explore_factor: float = 0.4,
  min_samples: int = 20,
  side: Any = None,
  stand_aside_enabled: bool = True,
) -> float:
  """The family multiplier an entry on ``side`` would actually get: 0 when stood aside, else the worse
  of the measured (``family_size_factor``) and explore (``family_explore_factor``) factors — the same
  composition the order path uses (tools.place_futures_limit_order), which ALWAYS passes the order's
  side. So only a sided call is an entry's stake; ``side=None`` is a pooled VIEW that no order ever
  receives (2026-09-25 review: pooled funding_carry read 0.93 while code refused its shorts at 0).

  ``stand_aside_enabled`` mirrors ``cfg.edge.stand_aside_no_edge_family``: with it off the order path
  never zeroes the stake, it only applies the worse of the two factors.
  """
  if stand_aside_enabled and family_stand_aside(signal_edge, family, min_samples=min_samples, side=side):
    return 0.0
  return min(
    family_size_factor(signal_edge, family, min_samples=min_samples, side=side),
    family_explore_factor(signal_edge, family, explore_factor=explore_factor,
                          min_samples=min_samples, side=side),
  )


def annotate_family_stakes(
  signal_edge: Dict[str, Any],
  *,
  explore_factor: float = 0.4,
  min_samples: int = 20,
  stand_aside_enabled: bool = True,
) -> Dict[str, Any]:
  """A COPY of the scoreboard with each family row (and each side row) marked standAside + stake.

  The model reads ``signalEdge.by_family`` BEFORE it proposes, and until 2026-09-25 that block showed
  only the raw verdict — 'edge' for continuation while the stand-aside held it at zero stake — so on
  09-24 it re-proposed the benched family in nine runs over about three hours and learned the truth
  only from nine refusals. Marking the row with the SAME functions the order path calls means there is
  no second rule to drift. The verdict strings are left untouched: ``family_size_factor``,
  ``open_families`` and the dashboard all read them.

  Every order is judged on its SIDE (the order path always passes it), so a stake only exists per
  side. Side rows carry what an entry on THAT side would get, including when it is judged on the pooled
  row because its own is still thin (``judgedOn: 'pooled'``). A POOLED ``by_family`` row carries no
  scalar stake — no entry ever gets one — but ``stakeBySide`` / ``standAsideBySide`` for both sides
  (computed even when a side has no row of its own, because the order path then judges it on the
  pooled row at the explore cap), and ``standAside`` only when BOTH sides are refused, the same rule
  ``open_families`` uses. 2026-09-25 review: the pooled-only marks told the model funding_carry was
  staked at 0.93 while code refused its shorts at 0, and told it continuation was at zero stake while
  code would take its longs — a veto written by the scoreboard, not by the order path.

  ``stand_aside_enabled`` is ``cfg.edge.stand_aside_no_edge_family``: with it off nothing is marked
  standAside and the stake is the worse of the measured and explore factors, as the order path sizes
  it. Never mutates the input (it is the shared per-run edge state the order path reads). Never raises;
  returns the input unchanged on bad data.
  """
  enabled = bool(stand_aside_enabled)

  def _mark(fam: str, side: Any) -> Dict[str, Any]:
    st = family_stake_status(signal_edge, fam, min_samples=min_samples, side=side)
    aside = enabled and bool(st["standAside"])
    return {
      "standAside": aside,
      "reason": st["reason"] if aside else None,
      "sideThin": bool(st["sideThin"]),
      "stake": round(family_stake(signal_edge, fam, explore_factor=explore_factor, min_samples=min_samples,
                                  side=side, stand_aside_enabled=enabled), 3),
    }

  try:
    board = copy.deepcopy(signal_edge) if isinstance(signal_edge, dict) else {}
    for fam, row in (board.get("by_family") or {}).items():
      if not isinstance(row, dict):
        continue
      marks = {s: _mark(fam, s) for s in SIDES}
      row.pop("stake", None)
      row["standAsideBySide"] = {s: marks[s]["standAside"] for s in SIDES}
      row["stakeBySide"] = {s: marks[s]["stake"] for s in SIDES}
      row["standAside"] = all(marks[s]["standAside"] for s in SIDES)
      reasons = [marks[s]["reason"] for s in SIDES]
      if not row["standAside"]:
        row["stakeReason"] = None
      elif len(set(reasons)) == 1:
        row["stakeReason"] = reasons[0]
      else:
        row["stakeReason"] = "; ".join(f"{s}: {r}" for s, r in zip(SIDES, reasons))
    for fam, sides in (board.get("by_family_side") or {}).items():
      if not isinstance(sides, dict):
        continue
      for side, row in sides.items():
        if not isinstance(row, dict) or _norm_side(side) is None:
          continue
        m = _mark(fam, side)
        row["standAside"] = m["standAside"]
        row["stakeReason"] = m["reason"]
        row["judgedOn"] = "pooled" if m["sideThin"] else "own"
        row["stake"] = m["stake"]
    return board
  except Exception:
    return signal_edge if isinstance(signal_edge, dict) else {}


def _stderr(vals: List[float]) -> float:
  """Standard error of the mean; 0.0 for a sample too small to have one."""
  n = len(vals)
  if n < 2:
    return 0.0
  mean = sum(vals) / n
  var = sum((v - mean) ** 2 for v in vals) / (n - 1)
  return math.sqrt(var / n)


def _carry_credit_unknown(ctx: Dict[str, Any], probe: Dict[str, Any], horizon: Any) -> bool:
  """True when a funding_carry row's funding credit at ``horizon`` is recorded as UNKNOWN (``f{h}``
  present and None — memory.settle_signal_probes writes that on a failed lookup and backfills it).
  Absent (legacy rows) is not unknown: those keep the old absent-as-zero rule."""
  key = f"f{int(horizon)}"
  return (key in probe and probe.get(key) is None
          and str(ctx.get("setupFamily") or "").strip().lower() == "funding_carry")


def unknown_carry_credits(probes: List[Dict[str, Any]], horizons_min: tuple = (5, 15, 60, 240)) -> int:
  """How many priced funding_carry horizons carry an UNKNOWN funding credit (and so are not scored).
  Reported beside the scoreboard so an exchange outage shows up as missing evidence, not as a verdict."""
  n = 0
  for row in probes or []:
    ctx = row.get("entryContext") if isinstance(row, dict) else None
    probe = ctx.get("signalProbe") if isinstance(ctx, dict) else None
    if not isinstance(probe, dict):
      continue
    for horizon in horizons_min:
      px = _f(probe.get(f"m{int(horizon)}"))
      if px is not None and px > 0 and _carry_credit_unknown(ctx, probe, horizon):
        n += 1
  return n


def _probe_observations(
  probes: List[Dict[str, Any]],
  horizons_min: tuple,
  *,
  require: Any = None,
):
  """Yield ``(row, ctx, horizon_min, signed_return)`` for every usable probe observation.

  Shared by every statistic computed off signal probes so the de-overlap rule below lives in exactly
  one place — it is subtle, it is load-bearing, and a second copy of it would drift. The signed return
  is price return in the traded direction PLUS any funding credited to the window (``f{h}``).

  Probes on the same symbol are recorded minutes apart, so their forward windows OVERLAP almost
  entirely — thirty probes on one symbol inside four hours are close to one observation, not thirty.
  Counting them independently inflates the sample and, with it, the verdict. On 2026-08-11 that
  produced a false positive: the 240m horizon read +0.224% (t=+3.66, n=131) and the verdict flipped
  to "edge", but decimating to one probe per symbol per window gave +0.068% (t=+0.51, n=31) — below
  the cost hurdle, i.e. nothing. Since that verdict governs how much capital each family gets, an
  inflated sample can talk the bot into sizing UP on noise, which is the most expensive mistake this
  module could make. Keep one observation per symbol per horizon window.

  ``require`` optionally filters on the entry context (used to restrict to flow-stamped probes).
  Note the filter runs BEFORE the de-overlap, so each statistic decimates its own eligible set rather
  than inheriting gaps from rows it never counted.
  """
  last_seen: Dict[tuple, int] = {}
  ordered = sorted(
    (r for r in (probes or []) if isinstance(r, dict)),
    key=lambda r: int(r.get("ts") or 0),
  )
  for row in ordered:
    ctx = row.get("entryContext")
    if not isinstance(ctx, dict):
      continue
    base = _f(ctx.get("marketPriceAtSignal"))
    side = str(ctx.get("positionSide") or "").lower()
    probe = ctx.get("signalProbe")
    if not base or base <= 0 or side not in {"long", "short"} or not isinstance(probe, dict):
      continue
    if require is not None and not require(ctx):
      continue
    symbol = str(row.get("symbol") or "?")
    ts = int(row.get("ts") or 0)
    for horizon in horizons_min:
      signed = _signed_probe_return(ctx, probe, base, side, horizon)
      if signed is None:
        continue
      key = (symbol, int(horizon))
      if ts - last_seen.get(key, -10**9) < int(horizon) * 60:
        continue
      last_seen[key] = ts
      yield row, ctx, int(horizon), signed


def _signed_probe_return(ctx: Dict[str, Any], probe: Dict[str, Any], base: float, side: str,
                         horizon: int) -> float | None:
  """One probe's return at one horizon, signed by its side, plus funding credited; None if unusable.

  The single definition shared by the per-horizon and the holding-time-weighted observations.
  """
  px = _f(probe.get(f"m{int(horizon)}"))
  if px is None or px <= 0:
    return None
  if _carry_credit_unknown(ctx, probe, horizon):
    # A carry call is a bet on the transfer: scoring it with an UNKNOWN credit as zero biased the
    # family down on every exchange hiccup (2026-09-25 review). Skipped (not counted as zero) while
    # memory backfills it, and for good once its backfill window passed with the credit unknown.
    return None
  ret = (px - base) / base
  signed = ret if side == "long" else -ret
  # Funding the position would have been PAID over the same window (already signed for its side
  # by memory.funding_received_from_history). A carry call is a bet on the transfer as well as on
  # price, so a price-only return understates it. ABSENT (legacy rows) counts as zero, which is what
  # every non-carry window crossing no settlement is anyway; an explicit None (a failed lookup since
  # 2026-09-25) is also zero for non-carry rows, but a carry row with it was skipped above.
  credit = _f(probe.get(f"f{int(horizon)}"))
  if credit is not None and math.isfinite(credit):
    signed += credit
  return signed


def family_scoring_horizons(
  closes,
  *,
  available: tuple = (5, 15, 60, 240),
  min_trades: int = 6,
  recent: int = 20,
) -> Dict[str, int]:
  """The settled probe horizon that best matches how long each family is actually held.

  Derived from the family's own realized trades — the median minutes from fill to close — and snapped
  to the nearest horizon that probes actually settle at, measured on a LOG scale because what matters
  for a holding period is the ratio, not the difference (162m sits nearer 240m than 60m; 116m nearer
  60m than 240m). A family with fewer than ``min_trades`` realized closes has no trustworthy median
  and is omitted, so it falls back to the caller's default horizon. Self-tuning: if a playbook starts
  being held longer or shorter, its verdict follows without anyone editing a constant. Never raises.

  Only each family's most recent ``recent`` closes count, so the horizon follows the market rather than
  averaging over every regime the bot has ever seen. Holding time is mostly a property of the playbook
  (fade ~12m, range ~14m) but it DOES move with conditions: `funding_carry` went from a 90m median in
  chop to 171m in the Sep 2026 rally, which moves it from the 60m horizon to 240m. An all-history
  median would lag a regime change by however many old closes had to be outvoted. The window matches
  the stand-aside's own ``min_samples`` (20), so the horizon a family is judged at comes from the same
  recent sample size that judges it, rather than from a second, unrelated constant.
  """
  opts = sorted({int(h) for h in available if int(h) > 0})
  out: Dict[str, int] = {}
  if not opts:
    return out
  for fam, vals in _recent_family_holds(closes, recent=recent, min_trades=min_trades).items():
    vals = sorted(vals)
    n = len(vals)
    median = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    out[fam] = _nearest_horizon(median, opts)
  return out


def _nearest_horizon(minutes: float, opts: List[int]) -> int:
  """The settled horizon nearest a holding time on a LOG scale (the ratio is what matters)."""
  return min(opts, key=lambda h: abs(math.log(h) - math.log(max(minutes, 1e-9))))


def _recent_family_holds(closes, *, recent: int = 20, min_trades: int = 6) -> Dict[str, List[float]]:
  """Minutes from fill to close of each family's most recent ``recent`` realized closes.

  Families with fewer than ``min_trades`` such closes are omitted (no trustworthy holding time).
  Shared by :func:`family_scoring_horizons` and :func:`family_horizon_weights` so both read the same
  holds under the same rules. Never raises.
  """
  holds: Dict[str, List[tuple]] = {}
  for row in closes or []:
    if not isinstance(row, dict):
      continue
    ctx = row.get("entryContext") if isinstance(row.get("entryContext"), dict) else {}
    try:
      filled = float(ctx.get("fillTs"))
      closed = float(row.get("ts"))
    except (TypeError, ValueError):
      continue
    minutes = (closed - filled) / 60.0
    if not (math.isfinite(minutes) and minutes > 0):
      continue
    fam = str(ctx.get("setupFamily") or "").strip().lower()
    if not fam:
      continue
    holds.setdefault(fam, []).append((closed, minutes))
  out: Dict[str, List[float]] = {}
  for fam, dated in holds.items():
    dated.sort(key=lambda p: p[0])                         # oldest -> newest
    vals = [m for _, m in dated[-max(1, int(recent)):]]    # this family's most recent closes only
    if len(vals) >= max(1, int(min_trades)):
      out[fam] = vals
  return out


def family_horizon_weights(
  closes,
  *,
  available: tuple = (5, 15, 60, 240),
  min_trades: int = 6,
  recent: int = 20,
) -> Dict[str, Dict[int, float]]:
  """How each family's recent trades were actually held, as weights over the settled probe horizons.

  Each of the family's last ``recent`` realized holds votes for its nearest settled horizon (log
  scale, as in :func:`family_scoring_horizons`); a horizon's weight is its share of the votes. The
  family's verdict is then scored on the forward return over the holding times it really uses — a
  weighted MIX of horizons — instead of the single horizon nearest the median.

  Why (2026-09-25): continuation's median hold sat at 118-141 minutes, straddling 120m, the log
  midpoint between the 60m and 240m horizons. Snapping the median made the verdict a step function
  of one close: a +0.39R FET winner banked in 63 minutes moved the median from 125.5 to 118.5, the
  horizon from 240m to 60m, and continuation longs from +1.1% net to -0.15% — zero stake, during a
  broad alt rally, on the same calls (declined continuation longs then ran +0.05% at 60m and +1.44%
  at 240m). And a benched family makes no closes, so the step could not step back: the faster the
  trail banked winners, the more firmly the winning playbook stayed benched. With weights, one close
  moves the mix by one twentieth — the verdict is continuous in the data, with no constant added and
  no regime assumption (a family held mostly at 60m is still scored mostly at 60m).
  """
  opts = sorted({int(h) for h in available if int(h) > 0})
  out: Dict[str, Dict[int, float]] = {}
  if not opts:
    return out
  for fam, vals in _recent_family_holds(closes, recent=recent, min_trades=min_trades).items():
    votes: Dict[int, int] = {}
    for minutes in vals:
      h = _nearest_horizon(minutes, opts)
      votes[h] = votes.get(h, 0) + 1
    total = float(sum(votes.values()))
    out[fam] = {h: votes[h] / total for h in sorted(votes)}
  return out


def safe_family_horizons(memory: Any, **kwargs: Any) -> Dict[str, int]:
  """`family_scoring_horizons` over a store's realized closes, or {} if they cannot be read.

  Hold-time derivation is a refinement of the verdict, not a precondition for having one: if it
  fails, every family falls back to the default horizon rather than the whole edge report going blank
  (which silently disables the stand-aside and every family size factor with it).
  """
  try:
    return family_scoring_horizons(memory.realized_closes(limit=1000), **kwargs)
  except Exception:
    return {}


def safe_family_horizon_weights(memory: Any, **kwargs: Any) -> Dict[str, Dict[int, float]]:
  """`family_horizon_weights` over a store's realized closes, or {} if they cannot be read (the
  verdict then falls back to the snapped horizon, never to a blank report)."""
  try:
    return family_horizon_weights(memory.realized_closes(limit=1000), **kwargs)
  except Exception:
    return {}


MARKET_STATE_BUCKETS = ("low", "mid", "high")


def market_breadth(state: Any) -> float | None:
  """breadth24 from a stamped market-state block (analytics.market_state), or None."""
  if not isinstance(state, dict):
    return None
  val = _f(state.get("breadth24"))
  return val if val is not None and math.isfinite(val) and 0.0 <= val <= 1.0 else None


def market_adx(state: Any) -> float | None:
  """BTC's daily ADX from a stamped market-state block, or None. ADX measures trend STRENGTH (BTC's
  daily ADX sat at 10-21 through the Jul 22-Aug 12 chop); breadth24 measures DIRECTION."""
  if not isinstance(state, dict):
    return None
  val = _f(state.get("btcDailyAdx"))
  return val if val is not None and math.isfinite(val) and val >= 0.0 else None


def breadth_terciles(values: List[float]) -> tuple | None:
  """Rolling tercile cut points (q1, q2) of breadth24 over the rows given, or None under 3 values.

  The buckets are the market's OWN recent distribution, not fixed edges: a 0.35/0.65 cut-off tuned on
  the Sep 2026 rally is exactly the kind of constant the project rules out, and the retained rows are
  the rolling window, so the cuts move with the market by themselves.
  """
  vals = sorted(v for v in values if v is not None and math.isfinite(v))
  if len(vals) < 3:
    return None
  return (_percentile(vals, 1.0 / 3.0), _percentile(vals, 2.0 / 3.0))


def breadth_bucket(value: float | None, cuts: tuple | None) -> str:
  """'low' / 'mid' / 'high' tercile of ``value`` under ``cuts``, or 'untagged'."""
  if value is None or not cuts:
    return "untagged"
  if value <= cuts[0]:
    return "low"
  if value > cuts[1]:
    return "high"
  return "mid"


def _scored_row(vals: List[float], cost_pct: float, min_samples: int) -> Dict[str, Any]:
  """One scoreboard row (a family, or one side of a family) from its de-overlapped signed returns."""
  mean = sum(vals) / len(vals)
  se = _stderr(vals)
  net = mean - cost_pct
  return {
    "n": len(vals),
    "mean_pct": round(mean * 100, 4),
    "hit_rate": round(sum(1 for v in vals if v > 0) / len(vals), 3),
    # Standard error of THIS row's own mean. Without it a caller cannot tell a real shortfall from a
    # rounding wobble, and the stand-aside chatters: on 2026-09-04 continuation sat at net -0.03% with
    # an SE of ~0.30%, flipped verdict between polls, and a WIF long went in at FULL size (family
    # x1.00) on the one poll it read non-negative — then lost a full 1R.
    "stderr_pct": round(se * 100, 4),
    "net_of_cost_pct": round(net * 100, 4),
    # net / SE — the number the stand-aside's release bar (t >= 1) and the explore ramp (t 1 -> 2)
    # actually read. Published so the model sees how far a family is from the line, not just a verdict.
    "t_stat": round(net / se, 3) if se > 0 else None,
    "verdict": ("insufficient data" if len(vals) < max(1, int(min_samples))
                else ("edge" if mean > cost_pct else "no edge")),
  }


def _mixed_row(vals_by_h: Dict[int, List[float]], weights: Dict[int, float], cost_pct: float,
               min_samples: int) -> Dict[str, Any] | None:
  """A scoreboard row over a holding-time MIX of horizons, or None if a weighted horizon has no data.

  Each horizon keeps its own de-overlapped sample (exactly what a single-horizon row would use), and
  the row is their weighted combination: mean and hit rate are the weighted averages, ``n`` is the
  SMALLEST leg (the evidence is only as deep as its thinnest part), and the standard error is the
  weighted SUM of the legs' errors — the exact error if the horizons were perfectly correlated and an
  upper bound otherwise (|cov| <= SE_i x SE_j), so mixing can never make a verdict look surer than
  its legs. With all the weight on one horizon it is that horizon's row, number for number.
  """
  total_w = sum(w for w in weights.values() if w > 0)
  if total_w <= 0:
    return None
  mean = se = hit = 0.0
  n = None
  for h, w in weights.items():
    if w <= 0:
      continue
    vals = vals_by_h.get(h) or []
    if not vals:
      return None
    frac = w / total_w
    mean += frac * (sum(vals) / len(vals))
    se += frac * _stderr(vals)
    hit += frac * (sum(1 for v in vals if v > 0) / len(vals))
    n = len(vals) if n is None else min(n, len(vals))
  net = mean - cost_pct
  return {
    "n": int(n or 0),
    "mean_pct": round(mean * 100, 4),
    "hit_rate": round(hit, 3),
    "stderr_pct": round(se * 100, 4),
    "net_of_cost_pct": round(net * 100, 4),
    "t_stat": round(net / se, 3) if se > 0 else None,
    "verdict": ("insufficient data" if int(n or 0) < max(1, int(min_samples))
                else ("edge" if mean > cost_pct else "no edge")),
    "horizonWeights": {f"{h}m": round(w / total_w, 3) for h, w in sorted(weights.items()) if w > 0},
    "nByHorizon": {f"{h}m": len(vals_by_h.get(h) or []) for h, w in sorted(weights.items()) if w > 0},
  }


def signal_edge_stats(
  probes: List[Dict[str, Any]],
  *,
  cost_pct: float = 0.001,
  horizons_min: tuple = (5, 15, 60, 240),
  verdict_horizons: tuple = (60, 240),
  family_horizon_min: int = 60,
  min_samples: int = 20,
  family_horizons: Dict[str, int] | None = None,
  market_state_split: bool = False,
  family_horizon_weights: Dict[str, Dict[int, float]] | None = None,
) -> Dict[str, Any]:
  """Does the agent's DIRECTION CALL predict? The one question that decides profitability.

  Every other statistic in this module measures an *outcome*, which conflates three separate things:
  whether the direction was right, whether the fill was any good, and whether the exit was managed
  well. That conflation is why six rounds of correct exit/cost/sizing fixes did not stop the bleeding.
  This measures the signal alone: forward return from the MARKET price at signal time, signed by the
  traded direction, so neither the limit-order discount nor the exit logic can flatter it.

  (Measuring from the *limit* price instead inflates the result badly — it scores the discount the
  order was resting at as if it were prediction. Doing that on this account's data showed a spurious
  +1.24%/15m at a 92% hit rate; measured correctly from market price it was -0.007%, i.e. nothing.)

  A signal is only worth trading when its mean forward return clears the round-trip cost. Below that,
  no exit or sizing scheme can produce profit — it can only lose more slowly. ``verdict`` is therefore
  the honest summary: "edge" / "no edge" / "insufficient data".

  ``horizons_min`` is what gets REPORTED; ``verdict_horizons`` and ``family_horizon_min`` are what
  gets ACTED ON, and they are separate on purpose. The 5m and 15m points were added to find out
  whether anything predicts at the short horizons where order-flow signals are supposed to live, and
  a new measurement must not silently move live capital: were the verdict taken over all horizons,
  the noisiest short one would win ``best_horizon`` by chance, and were family scoring left keyed to
  ``horizons_min[0]`` it would have jumped from the 60m point to the 5m point — re-pricing every
  playbook's risk multiplier as a side effect of adding a chart. Report widely, act narrowly.

  ``family_horizons`` narrows that one step further, per playbook: each family is scored at the
  horizon that matches how long it is actually HELD (see :func:`family_scoring_horizons`), falling
  back to ``family_horizon_min`` for any family not in the map. One shared 60m horizon measured every
  playbook against a holding period most of them never use. Live, 2026-09-18: `continuation` is held
  a median 162 minutes, and in a strong rally its entries sit mid-pullback at the 60m mark, so its
  verdict hovered at net -0.006% over ~110 probes and the stand-aside benched it for the entire second
  day of the move — 143 blocked calls that measured +0.17% at 60m and **+1.27% net at 240m with a 76%
  hit rate**, by the same probe method. The fix is per-family on purpose: `fade_extreme` is held a
  median 12 minutes, and a blanket 240m horizon would have RELEASED it (+0.77% at 240m) on evidence
  from a holding period it never uses; at its own 15m horizon it correctly stays benched (-0.30%).

  ``market_state_split`` (report-only, off by default so the agent's copy never carries it) adds
  ``by_market_state``: the same family-horizon observations split by the terciles of the calls' own
  breadth24 stamp (see :func:`breadth_terciles`), each bucket reading 'insufficient data' below
  ``min_samples``. It is there so a later reader can test whether any market state separates the calls
  that paid from the ones that did not — nothing sizes or gates on it.

  ``family_horizon_weights`` (from :func:`family_horizon_weights`) replaces the single snapped horizon
  for every family it covers: that family's ``by_family`` and ``by_family_side`` rows are scored on the
  holding-time MIX of horizons, so the verdict no longer jumps when the median hold crosses a log
  midpoint (2026-09-25: continuation flipped 240m -> 60m on one close and went from +1.1% to -0.15%
  net). Each weighted horizon keeps its own de-overlapped sample — the one a single-horizon row would
  use — and the row mixes them (:func:`_mixed_row`: weighted mean, n = thinnest leg, SE = weighted sum,
  an upper bound), so mixing never shrinks a sample nor overstates certainty, and all weight on one
  horizon reproduces that horizon's row exactly. Such rows carry ``horizonWeights`` and ``nByHorizon``.
  The report-only market-state split reads the mix's heaviest horizon. Families the weights do not
  cover keep ``family_horizons`` exactly as before.
  """
  # `by_horizon` is seeded so the return shape is the same whether or not anything settled — callers
  # (dashboard, agent state) should not have to distinguish "no data" from "key absent".
  out: Dict[str, Any] = {
    "n": 0, "verdict": "insufficient data", "cost_pct": cost_pct, "by_horizon": {},
  }
  # Carry horizons whose funding credit is UNKNOWN (a failed lookup) are not scored — counted here so a
  # thinner carry sample during an exchange outage is visible as missing evidence.
  unknown_credit = unknown_carry_credits(probes, horizons_min)
  if unknown_credit:
    out["unknownFundingCredit"] = unknown_credit
  by_h: Dict[str, List[float]] = {}
  by_fam: Dict[str, List[float]] = {}
  by_fam_side: Dict[str, Dict[str, List[float]]] = {}
  by_state: List[tuple] = []           # (breadth24 or None, signed) at each call's family horizon
  fam_h = {str(k).strip().lower(): int(v) for k, v in (family_horizons or {}).items() if v}
  fam_w = {
    str(k).strip().lower(): {int(h): float(w) for h, w in (v or {}).items() if w and float(w) > 0}
    for k, v in (family_horizon_weights or {}).items()
  }
  fam_w = {k: v for k, v in fam_w.items() if v}

  def _add_family_obs(fam: str, ctx: Dict[str, Any], signed: float) -> None:
    by_fam.setdefault(fam, []).append(signed)
    # The side split is taken AFTER the de-overlap, from the very same observations, so the two
    # sides always sum to the pooled row and the de-overlap rule keeps living in one place.
    side = str(ctx.get("positionSide") or "").lower()
    by_fam_side.setdefault(fam, {}).setdefault(side, []).append(signed)
    if market_state_split:
      by_state.append((market_breadth(ctx.get("marketState")), signed))

  # A family with holding-time weights keeps one de-overlapped sample PER weighted horizon (pooled and
  # per side); its rows are mixed from those samples below (see _mixed_row).
  mix_fam: Dict[str, Dict[int, List[float]]] = {}
  mix_side: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
  for _row, ctx, horizon, signed in _probe_observations(probes, horizons_min):
    by_h.setdefault(f"{horizon}m", []).append(signed)
    # Family scoring uses ONE horizon PER FAMILY so a setup is never counted twice with different
    # holding periods — that horizon being the family's own, not a single one shared by all. A family
    # with holding-time weights is scored on its mix instead.
    fam = infer_setup_family(ctx)
    weights = fam_w.get(fam)
    if weights:
      if horizon in weights:
        side = str(ctx.get("positionSide") or "").lower()
        mix_fam.setdefault(fam, {}).setdefault(horizon, []).append(signed)
        mix_side.setdefault(fam, {}).setdefault(side, {}).setdefault(horizon, []).append(signed)
        # The report-only market-state split reads the mix's heaviest horizon.
        if market_state_split and horizon == max(weights, key=lambda h: (weights[h], h)):
          by_state.append((market_breadth(ctx.get("marketState")), signed))
      continue
    if horizon == int(fam_h.get(fam, family_horizon_min)):
      _add_family_obs(fam, ctx, signed)
  if market_state_split:
    cuts = breadth_terciles([b for b, _ in by_state if b is not None])
    grouped: Dict[str, List[float]] = {k: [] for k in MARKET_STATE_BUCKETS}
    untagged = 0
    for breadth, signed in by_state:
      bucket = breadth_bucket(breadth, cuts)
      if bucket == "untagged":
        untagged += 1
        continue
      grouped[bucket].append(signed)
    out["by_market_state"] = {
      "key": "breadth24 terciles of the calls' own stamps (rolling)",
      "cuts": [round(c, 4) for c in cuts] if cuts else None,
      "untagged": untagged,
      "buckets": {
        k: (_scored_row(v, cost_pct, min_samples) if len(v) >= max(1, int(min_samples))
            else {"n": len(v), "verdict": "insufficient data"})
        for k, v in grouped.items()
      },
    }
  mixed_fam = {fam: _mixed_row(legs, fam_w[fam], cost_pct, min_samples) for fam, legs in mix_fam.items()}
  mixed_fam = {fam: row for fam, row in mixed_fam.items() if row is not None}
  mixed_side = {
    fam: {side: row for side, legs in sorted(sides.items())
          if (row := _mixed_row(legs, fam_w[fam], cost_pct, min_samples)) is not None}
    for fam, sides in mix_side.items()
  }
  if by_fam or mixed_fam:
    out["by_family"] = {fam: _scored_row(vals, cost_pct, min_samples) for fam, vals in by_fam.items()}
    out["by_family"].update(mixed_fam)
    # Each playbook split by SIDE at the family's own horizon. The Kelly stake is a bet on a side, and
    # a pooled row lets one side's record size the other: on 2026-09-24 continuation's 'edge' was
    # carried by its longs (n=40, +1.24%) while its shorts sat at n=9, -0.18% ± 1.05% — and the SPX
    # shorts were sized on the longs' record. The stake path judges a side on its own row once that
    # row has a real sample (see family_evidence_row); the pooled row above is left exactly as it was.
    out["by_family_side"] = {
      fam: {side: _scored_row(vals, cost_pct, min_samples) for side, vals in sorted(sides.items())}
      for fam, sides in by_fam_side.items()
    }
    out["by_family_side"].update({fam: sides for fam, sides in mixed_side.items() if sides})
  if not by_h:
    return out
  scoring = {f"{int(h)}m" for h in verdict_horizons}
  detail = {}
  best = None
  for key, vals in by_h.items():
    mean = sum(vals) / len(vals)
    hit = sum(1 for v in vals if v > 0) / len(vals)
    detail[key] = {
      "n": len(vals),
      "mean_pct": round(mean * 100, 4),
      "hit_rate": round(hit, 3),
      "stderr_pct": round(_stderr(vals) * 100, 4),
      "net_of_cost_pct": round((mean - cost_pct) * 100, 4),
      # Whether this horizon is one the bot acts on, or one it is only watching. Published so a
      # reader of the dashboard is never left guessing which numbers move money.
      "scored": key in scoring,
    }
    if key in scoring and (best is None or mean > best[1]):
      best = (key, mean, len(vals))
  out["n"] = max((d["n"] for k, d in detail.items() if k in scoring), default=0)
  out["by_horizon"] = detail
  out["best_horizon"] = best[0] if best else None
  out["verdict_horizons"] = sorted(scoring, key=lambda k: int(k[:-1]))
  if out["n"] < max(1, int(min_samples)):
    out["verdict"] = "insufficient data"
  elif best and best[1] > cost_pct:
    out["verdict"] = "edge"
  else:
    out["verdict"] = "no edge"
  return out


def _flow_agreement(ctx: Dict[str, Any]) -> float | None:
  """How far the taker tape leaned the way the trade was taken, in share points either side of 0.5.

  Positive means the aggressors were pushing with the position (buyers into a long, sellers into a
  short); negative means the trade was taken into the flow. Returns None when the probe carries no
  usable reading, so unstamped probes are excluded rather than silently scored as neutral.
  """
  flow = ctx.get("takerFlow")
  if not isinstance(flow, dict):
    return None
  share = _f(flow.get("buyShare"))
  if share is None or not (0.0 <= share <= 1.0):
    return None
  bias = share - 0.5
  side = str(ctx.get("positionSide") or "").lower()
  if side == "long":
    return bias
  if side == "short":
    return -bias
  return None


def taker_flow_edge_stats(
  probes: List[Dict[str, Any]],
  *,
  cost_pct: float = 0.001,
  horizons_min: tuple = (5, 15, 60, 240),
  min_samples: int = 20,
  neutral_band: float = 0.05,
) -> Dict[str, Any]:
  """Did the taker tape at signal time separate the direction calls that worked from those that did not?

  MEASUREMENT ONLY — nothing in the trading path reads this. It exists to answer, from this account's
  own data on this venue, a question the literature cannot answer for us. Order-flow imbalance is a
  genuine and well-documented effect (Cont, Kukanov & Stoikov 2014), but it is measured at a
  ten-second bucket and is largely *contemporaneous* — price moves because of the flow — with the
  lagged, forecastable part concentrated inside a minute and decaying fast after. Published work on
  the crypto retail version (CVD, taker buy/sell ratio) is descriptive: no out-of-sample results, no
  hit rates, no ICs. And KuCoin is not where these prices are set, so its tape is one venue's slice
  of a market led elsewhere. All of which means the honest prior is "probably nothing at our horizon",
  and the only way to know is to record the reading at the call and score it forward.

  The statistic is a SPREAD, not a return. Splitting probes into calls taken *with* the flow and
  calls taken *against* it, the difference between the two groups' forward returns is the flow's
  information content, and it is robust to the thing that would otherwise dominate: the bot's overall
  directional bias. A raw "with-flow calls returned +0.1%" says nothing if every call returned +0.1%.

  Two verdicts are reported because they are different questions, and conflating them is how a real
  but unusable effect gets traded:

  * ``informative`` — the spread clears its own standard error. Flow carries signal.
  * ``tradable`` — the with-flow group ALSO clears the round-trip cost. Flow carries enough signal to
    pay for the trade it would trigger. At a ~0.2% round trip against a few basis points of
    short-horizon drift, this is the bar that is expected to fail.

  Never raises; returns "insufficient data" until both groups have a real sample.
  """
  out: Dict[str, Any] = {
    "n": 0, "verdict": "insufficient data", "cost_pct": cost_pct,
    "neutral_band": neutral_band, "by_horizon": {}, "coverage": None,
  }
  band = max(0.0, float(neutral_band))
  buckets: Dict[int, Dict[str, List[float]]] = {}
  for _row, ctx, horizon, signed in _probe_observations(
    probes, horizons_min, require=lambda c: _flow_agreement(c) is not None,
  ):
    agreement = _flow_agreement(ctx)
    bucket = "with" if agreement > band else ("against" if agreement < -band else "neutral")
    buckets.setdefault(horizon, {}).setdefault(bucket, []).append(signed)

  # What fraction of retained probes carry a reading at all. A spread computed over 12% coverage is
  # a statement about 12% of the book, and the reader needs to see that next to the verdict.
  rows = [r for r in (probes or []) if isinstance(r, dict)]
  stamped = sum(
    1 for r in rows
    if isinstance(r.get("entryContext"), dict) and _flow_agreement(r["entryContext"]) is not None
  )
  if rows:
    out["coverage"] = round(stamped / len(rows), 3)

  if not buckets:
    return out

  def _describe(vals: List[float]) -> Dict[str, Any]:
    mean = sum(vals) / len(vals)
    return {
      "n": len(vals),
      "mean_pct": round(mean * 100, 4),
      "hit_rate": round(sum(1 for v in vals if v > 0) / len(vals), 3),
      "stderr_pct": round(_stderr(vals) * 100, 4),
    }

  detail: Dict[str, Any] = {}
  for horizon in sorted(buckets):
    groups = buckets[horizon]
    row: Dict[str, Any] = {name: _describe(vals) for name, vals in groups.items() if vals}
    with_vals, against_vals = groups.get("with") or [], groups.get("against") or []
    verdict = "insufficient data"
    if len(with_vals) >= min_samples and len(against_vals) >= min_samples:
      with_mean = sum(with_vals) / len(with_vals)
      against_mean = sum(against_vals) / len(against_vals)
      spread = with_mean - against_mean
      # Standard error of a DIFFERENCE of two independent means: errors add in quadrature. Using
      # either group's own SE would understate the noise and manufacture significance.
      spread_se = math.sqrt(_stderr(with_vals) ** 2 + _stderr(against_vals) ** 2)
      row["spread_pct"] = round(spread * 100, 4)
      row["spread_stderr_pct"] = round(spread_se * 100, 4)
      if spread <= spread_se:
        verdict = "no information"
      elif with_mean > cost_pct:
        verdict = "tradable"
      else:
        verdict = "informative"
    row["verdict"] = verdict
    detail[f"{horizon}m"] = row

  out["by_horizon"] = detail
  out["n"] = max((sum(g["n"] for g in r.values() if isinstance(g, dict)) for r in detail.values()),
                 default=0)
  ranked = [r for r in detail.values() if r.get("verdict") in ("tradable", "informative")]
  if any(r["verdict"] == "tradable" for r in ranked):
    out["verdict"] = "tradable"
  elif ranked:
    out["verdict"] = "informative"
  elif any(r.get("verdict") == "no information" for r in detail.values()):
    out["verdict"] = "no information"
  return out


def _ranks(vals: List[float]) -> List[float]:
  """1-based ranks with ties sharing their average rank (the Spearman convention)."""
  order = sorted(range(len(vals)), key=lambda i: vals[i])
  ranks = [0.0] * len(vals)
  i = 0
  while i < len(order):
    j = i
    while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
      j += 1
    avg = (i + j) / 2.0 + 1.0
    for k in range(i, j + 1):
      ranks[order[k]] = avg
    i = j + 1
  return ranks


def _stratified_rank_corr(strata: List[List[tuple]]) -> Dict[str, Any]:
  """Rank correlation of (x, y) pairs computed WITHIN each stratum, pooled across strata.

  Ranks are taken inside a stratum and centred on that stratum's own mean rank, so a level shift
  BETWEEN strata (a calm week of low confidence and flat returns next to a rally of high confidence and
  big returns) contributes nothing — only the ordering of calls against each other inside the same
  stratum does. A single stratum is the ordinary Spearman correlation. ``t`` uses n - strata - 1
  degrees of freedom (each stratum spends one on its own mean). Strata with fewer than 3 pairs carry no
  ordering worth the name and are skipped.
  """
  sxy = sxx = syy = 0.0
  n = used = 0
  for pairs in strata:
    if len(pairs) < 3:
      continue
    rx = _ranks([p[0] for p in pairs])
    ry = _ranks([p[1] for p in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    for a, b in zip(rx, ry):
      sxy += (a - mx) * (b - my)
      sxx += (a - mx) ** 2
      syy += (b - my) ** 2
    n += len(pairs)
    used += 1
  if n == 0 or sxx <= 0 or syy <= 0:
    return {"rho": None, "t": None, "n": n, "strata": used}
  rho = max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))
  df = n - used - 1
  t = None
  if df > 0:
    # A perfect ordering has 1 - rho² = 0; floor it so t stays a finite (JSON-safe) large number.
    t = rho * math.sqrt(df / max(1.0 - rho * rho, 1e-9))
  return {"rho": round(rho, 3), "t": round(t, 2) if t is not None else None, "n": n, "strata": used}


def confidence_edge_stats(
  probes: List[Dict[str, Any]],
  *,
  cost_pct: float = 0.001,
  family_horizons: Dict[str, int] | None = None,
  family_horizon_min: int = 60,
  horizons_min: tuple = (5, 15, 60, 240),
  min_samples: int = 20,
  full_size_tstat: float = 2.0,
) -> Dict[str, Any]:
  """Does a model's stated CONFIDENCE rank its own direction calls? Per model. REPORT ONLY.

  Seven absolute confidence thresholds shape entries (floor, hostile-regime floor, deadlock, trend-
  short, reversal, relative-strength, full-conviction size), all implicitly calibrated on one model's
  scale — yet the stated number drifts with the model AND with the tape (median stated confidence on
  closes: 0.82 for gpt-5.4-mini, 0.74 for gpt-5.4, 0.78 for gpt-5.6, 0.76 for gpt-6; gpt-5.6's own
  benign-regime median moved 0.715 -> 0.80 across periods, as large as the between-model gaps, so the
  data cannot yet separate the two). Whether the number carries any
  information was never measurable, because probes did not record the model or the confidence. They do
  from 2026-09-25 (``memory.record_signal_probe``); this reads them. Nothing in the trading path acts on
  it — it is the evidence any future calibration would need, not a calibration.

  Per model, over the de-overlapped observations at each call's OWN family horizon (the same
  ``_probe_observations`` rule and the same horizon map ``signal_edge_stats`` scores families with):

  * ``withinDay`` — the rank correlation of confidence with the net forward return, computed within
    each UTC day and pooled (``_stratified_rank_corr``). This is the verdict basis. A POOLED correlation
    mostly measures regime: on gpt-5.6 it read +0.20 (t=2.25) only because 25 of 43 pre-rally closes
    sat below 0.75 while 83 of 85 rally closes sat above it — a level change between periods, not an
    ordering of calls. Ranking calls against the other calls of the same day removes that.
  * ``pooled`` — the ordinary Spearman correlation, reported so the difference is visible.
  * ``confidence`` — the model's own distribution of stated confidence (p10..p90) over the same
    retained, de-overlapped calls, so a scale shift between models is visible next to the ranking.

  ``verdict``: 'insufficient data' until the model has ``min_samples`` observations sharing days;
  'informative' / 'inverted' when the within-day t clears ``full_size_tstat`` either way; otherwise
  'not demonstrated' — which means exactly that, not 'demonstrated uninformative'. The t still treats
  different coins in the same window as independent, so read it as an upper bound on the evidence.

  Only rows carrying the probe-era stamp — ``model`` + a finite ``confidence`` + a finite
  ``minConfidence`` (from 2026-09-25 on every probe and every new placed trade) — are scored; the rest
  are excluded BEFORE the de-overlap, so each model is decimated on its own calls. Legacy PLACED-trade
  rows (which ``signal_probes()`` unions in) already carried ``model`` and ``confidence`` but never
  ``minConfidence``, and they are excluded on purpose: they are the post-gate subset (the calls that
  also cleared the stand-aside, the net-RR gate, sizing and the profit floor), not every direction
  call, so they would score a selected population as if it were the model's calls. Never raises.
  """
  out: Dict[str, Any] = {"cost_pct": cost_pct, "min_samples": int(min_samples),
                         "verdict_basis": "withinDay", "by_model": {}}
  try:
    fam_h = {str(k).strip().lower(): int(v) for k, v in (family_horizons or {}).items() if v}

    def _stamped(ctx: Dict[str, Any]) -> bool:
      conf = _f(ctx.get("confidence"))
      floor = _f(ctx.get("minConfidence"))
      return (bool(str(ctx.get("model") or "").strip()) and conf is not None and math.isfinite(conf)
              and floor is not None and math.isfinite(floor))

    per_model: Dict[str, List[tuple]] = {}
    for row, ctx, horizon, signed in _probe_observations(probes, horizons_min, require=_stamped):
      if horizon != int(fam_h.get(infer_setup_family(ctx), family_horizon_min)):
        continue
      model = str(ctx.get("model")).strip()[:80]
      per_model.setdefault(model, []).append(
        (int(row.get("ts") or 0), float(_f(ctx.get("confidence"))), signed - cost_pct)
      )
    for model, obs in sorted(per_model.items()):
      confs = [c for _, c, _ in obs]
      nets = [r for _, _, r in obs]
      by_day: Dict[int, List[tuple]] = {}
      for ts, conf, net in obs:
        by_day.setdefault(ts // 86400, []).append((conf, net))
      within = _stratified_rank_corr(list(by_day.values()))
      pooled = _stratified_rank_corr([[(c, r) for _, c, r in obs]])
      verdict = "insufficient data"
      if len(obs) >= max(1, int(min_samples)) and within["n"] >= max(1, int(min_samples)) \
          and within["t"] is not None:
        if within["t"] >= float(full_size_tstat):
          verdict = "informative"
        elif within["t"] <= -float(full_size_tstat):
          verdict = "inverted"
        else:
          verdict = "not demonstrated"
      out["by_model"][model] = {
        "n": len(obs),
        "meanNetPct": round(sum(nets) / len(nets) * 100, 4),
        "confidence": {f"p{q}": _percentile(confs, q / 100.0) for q in (10, 25, 50, 75, 90)},
        "withinDay": within,
        "pooled": pooled,
        "verdict": verdict,
      }
  except Exception:
    return out
  return out


def confidence_edge_for_prompt(
  stats: Any, *, model: str | None = None, min_samples: int | None = None,
) -> Dict[str, Any]:
  """The confidence rows worth showing the model: its OWN row, and only once ``n >= min_samples``.

  The agent's edgeReport carries this only once the model has a real sample — a thin correlation in the
  prompt reads as evidence it is not — and only for the model that is running (``model``): another
  model's record describes another model's number, and on the 09-24 copy the previous model's row read
  'inverted' (gpt-5.6-luna, n=29, 2 day-strata), which the current model must not read as its own.
  (That row predates the 09-25 probe stamp, so it was built entirely from legacy PLACED-trade rows — a
  post-gate subset ``confidence_edge_stats`` now excludes via the ``minConfidence`` requirement.) The
  dashboard and the Supervisor get the full per-model report. ``model=None`` keeps every model. Never
  raises.
  """
  try:
    board = stats if isinstance(stats, dict) else {}
    floor = int(min_samples if min_samples is not None else board.get("min_samples") or 20)
    want = str(model).strip() if model else None
    return {
      name: row for name, row in (board.get("by_model") or {}).items()
      if isinstance(row, dict) and int(row.get("n") or 0) >= max(1, floor)
      and (want is None or str(name).strip() == want)
    }
  except Exception:
    return {}


# ── GATE SCOREBOARD: what each directional gate blocks, against what it allows (2026-09-25) ───────────
#
# REPORT-ONLY. Published to the dashboard, the Supervisor and a periodic log line — never to the trading
# prompt: a line saying "gate X's refusals pay" invites relabelling a call past the declared-setup
# hatches (breakout/range_edge pass the daily and 1h gates on the label), which is how a label became a
# gate bypass once already. Gate behaviour is unchanged; any future relaxation should follow a SUSTAINED
# differential across regimes, never an absolute reading from one window.
#
# Why a differential and not "what did refused calls return". In a rally every refused long scores
# positive (09-22..24: the 1h gate's two refusals +2.17%, the exhaustion gate's one +4.16%), so an
# absolute number argues for loosening a gate whatever it is worth. The offline study that DID separate
# something compared gated vs concurrent ungated calls — and even that did not survive the regime test:
# the exhaustion gate's whole advantage (-0.08% vs +0.11% at 4h, n=1209 vs 2824) came from the Sep 17-24
# rally week; without it, exhausted longs did BETTER (+0.07% vs -0.08%), and day-bootstrap intervals span
# zero for both it and the 1h gate. Honest status on 2026-09-25: value currently unmeasurable. This is
# the instrument that can measure it, slowly: a ~0.17%/call gap at a per-call SD of 1.7-3.2% needs months
# of day-level data, and alts moving together make each day closer to one observation than to fifty.


def gate_state_cells(rows: List[Dict[str, Any]], horizons_min: tuple = (5, 15, 60, 240)) -> Dict[str, Any]:
  """Fold settled gate-STATE rows into ``{day: {side: {horizon: [n, s, {gate: [n, s]}]}}}`` increments.

  ``s`` is the sum of the signed forward returns (fractions, funding credit included) — the SAME
  observations `_probe_observations` yields for any probe, de-overlapped per (symbol, side, horizon):
  the call runs once per side because a state reading writes a long AND a short row at one instant,
  and the shared de-overlap key is (symbol, horizon). ``day`` is the row's UTC day number. The per-gate
  pair repeats the sums over the rows that gate would have refused; the total minus it is the gate's
  allowed complement. Pure; `memory.fold_settled_gate_states` adds these to the stored day cells.
  """
  out: Dict[str, Any] = {}
  for side in SIDES:
    def _want(ctx: Dict[str, Any], _s: str = side) -> bool:
      return ctx.get("gateProbe") == "state" and str(ctx.get("positionSide") or "").lower() == _s

    for row, ctx, horizon, signed in _probe_observations(rows, horizons_min, require=_want):
      day = str(int(row.get("ts") or 0) // 86400)
      cell = out.setdefault(day, {}).setdefault(side, {}).setdefault(str(int(horizon)), [0, 0.0, {}])
      cell[0] += 1
      cell[1] += signed
      for gate in dict.fromkeys(str(g) for g in (ctx.get("gates") or [])):
        pair = cell[2].setdefault(gate, [0, 0.0])
        pair[0] += 1
        pair[1] += signed
  return out


def _day_clustered_stats(cells: Dict[Any, list], cost_pct: float) -> Dict[str, Any] | None:
  """Mean (net of cost) of the observations in ``{day: [n, sum]}``, with an SE clustered by DAY.

  Day clusters because the alts move together: fifty calls on one day share most of their variance,
  so the naive per-call SE overstates the evidence. SE is None below two days.
  """
  used = {d: (int(v[0]), float(v[1])) for d, v in cells.items() if int(v[0]) > 0}
  total_n = sum(n for n, _ in used.values())
  if total_n <= 0:
    return None
  mean = sum(s for _, s in used.values()) / total_n
  days = len(used)
  se = None
  if days >= 2:
    ss = sum((s - n * mean) ** 2 for n, s in used.values())
    se = math.sqrt(days / (days - 1) * ss) / total_n
  return {
    "n": total_n, "days": days,
    "meanNetPct": round((mean - cost_pct) * 100, 4),
    "sePct": round(se * 100, 4) if se is not None else None,
  }


def _matched_differential(blocked: Dict[tuple, list], baseline: Dict[tuple, list]) -> Dict[str, Any] | None:
  """Blocked-minus-baseline mean, matched within each (day, horizon) cell, SE clustered by day.

  Only cells holding BOTH groups count, so a regime difference between the days the gate was active
  and the days it was not can never read as the gate's effect (the absolute numbers carry that bias).
  Each cell's difference is weighted by its blocked count; days are the clusters.
  """
  per_day: Dict[Any, list] = {}
  for key, (nb, sb) in blocked.items():
    na, sa = baseline.get(key, (0, 0.0))
    if int(nb) <= 0 or int(na) <= 0:
      continue
    diff = float(sb) / int(nb) - float(sa) / int(na)
    acc = per_day.setdefault(key[0], [0, 0.0])
    acc[0] += int(nb)
    acc[1] += int(nb) * diff
  weight = sum(w for w, _ in per_day.values())
  if weight <= 0:
    return None
  mean = sum(wd for _, wd in per_day.values()) / weight
  days = len(per_day)
  se = None
  if days >= 2:
    ss = sum((wd - w * mean) ** 2 for w, wd in per_day.values())
    se = math.sqrt(days / (days - 1) * ss) / weight
  return {"diff": mean, "se": se, "n": weight, "days": days}


def _t_critical(z: float, df: int) -> float:
  """Student-t critical value matching the normal cut-off ``z`` at ``df`` degrees of freedom.

  A day-clustered t has about (days - 1) degrees of freedom, not infinitely many: with three days the
  cut-off matching |z| >= 2 is ~4.5, not 2. Simulated on pure noise (2026-09-25, 1,200 boards): the
  normal bar called 20% of 3-day boards and 15% of 5-day boards significant; this one 5% and 8%, ~4% by
  15-30 days (nominal 4.6%). Cornish-Fisher expansion (no scipy): 4.37 vs the exact 4.53 at df = 2,
  within 0.03 from df = 3, exact to two decimals by df = 10.
  """
  nu = float(max(1, int(df)))
  z3, z5, z7 = z ** 3, z ** 5, z ** 7
  return (z + (z3 + z) / (4 * nu) + (5 * z5 + 16 * z3 + 3 * z) / (96 * nu ** 2)
          + (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384 * nu ** 3))


# Fewest day clusters a gate verdict may rest on: below 3 (df < 2) the t cut-off above is not reliable,
# and a two-day comparison is one regime by construction.
_GATE_MIN_DAYS = 3


def _gate_row(blocked: Dict[tuple, list], baseline: Dict[tuple, list], *, cost_pct: float,
              min_samples: int, tstat: float) -> Dict[str, Any]:
  """One gate/side row: blocked and baseline stats plus their matched differential and its verdict."""

  def _by_day(cells: Dict[tuple, list]) -> Dict[Any, list]:
    out: Dict[Any, list] = {}
    for (day, _cell), (n, s) in cells.items():
      acc = out.setdefault(day, [0, 0.0])
      acc[0] += int(n)
      acc[1] += float(s)
    return out

  row: Dict[str, Any] = {
    "blocked": _day_clustered_stats(_by_day(blocked), cost_pct),
    "baseline": _day_clustered_stats(_by_day(baseline), cost_pct),
    "diffPct": None, "diffSePct": None, "t": None, "matchedN": 0, "matchedDays": 0,
    "verdict": "insufficient data",
  }
  diff = _matched_differential(blocked, baseline)
  if diff is None:
    return row
  row["diffPct"] = round(diff["diff"] * 100, 4)
  row["diffSePct"] = round(diff["se"] * 100, 4) if diff["se"] is not None else None
  row["matchedN"], row["matchedDays"] = diff["n"], diff["days"]
  if (diff["se"] is None or diff["se"] <= 0 or diff["n"] < max(1, int(min_samples))
      or diff["days"] < _GATE_MIN_DAYS):
    return row
  t = diff["diff"] / diff["se"]
  crit = _t_critical(float(tstat), diff["days"] - 1)
  row["t"] = round(t, 2)
  row["tCritical"] = round(crit, 2)
  if t <= -crit:
    row["verdict"] = "blocked underperform"
  elif t >= crit:
    row["verdict"] = "blocked outperform"
  else:
    row["verdict"] = "not demonstrated"
  return row


def gate_scoreboard(
  gate_probes: List[Dict[str, Any]],
  *,
  cost_pct: float = 0.001,
  family_horizons: Dict[str, int] | None = None,
  state_days: Dict[str, Any] | None = None,
  admitted_probes: List[Dict[str, Any]] | None = None,
  family_horizon_min: int = 60,
  horizons_min: tuple = (5, 15, 60, 240),
  min_samples: int = 20,
  tstat: float = 2.0,
) -> Dict[str, Any]:
  """Per gate and side: what the gate BLOCKS against what it ALLOWS, net of cost. REPORT ONLY.

  Two instruments, each with its own baseline:

  * ``state`` (the headline) — model-independent gate-state readings (`memory.record_gate_state_probes`),
    folded into day cells (``state_days``). Blocked = the sides the gate would have refused; baseline =
    the same side's readings the gate did NOT refuse, on the same UTC day. Scored at continuation's own
    holding horizon (``family_horizons['continuation']``, self-tuning; 240m in Sep 2026): a state reading
    is no family's call, and the directional gates are rules about the trend bet. This is the footprint
    the model's self-censorship leaves, at ~6x the hard-refusal rate.
  * ``refused`` — calls a gate hard-refused (``gate_probes`` refusal rows), each scored at its declared
    family's horizon, against the calls that got through (``admitted_probes`` = the signal probes) on
    the same side, day and horizon. Selection-biased by construction (these are the calls the model
    made despite the rule), so it is shown, never headlined.
  * ``hatched`` — admitted calls that MET the gate and were let through by a hatch (declared/mechanical
    playbook, fade, reversal, deadlock break, relative strength: the probe's ``gatesPassed`` stamp),
    against admitted calls that never met it; ``byHatch`` counts which hatch. The cheapest signal of
    the three (09-22..24: 28 hatch admissions vs 8 hard refusals), and the one that shows whether a
    hatch is admitting worse calls — e.g. a label walking a trade past a gate.

  The number to read is ``diffPct`` (blocked minus baseline, matched within day x horizon cells, SE
  clustered by day). Negative = the gate refuses calls that did worse than the ones it lets through.
  ``verdict``: 'insufficient data' until ``matchedN >= min_samples`` over at least three days; then
  'blocked underperform' / 'blocked outperform' when |t| clears ``tCritical`` — the Student-t value
  at (days - 1) degrees of freedom matching ``tstat`` (the 2.0 `confidence_edge_stats` uses), because
  the evidence unit is a day, not a call — else 'not demonstrated', which is exactly that, not 'the
  gate does nothing'. De-overlap is `_probe_observations`', per (symbol, side, horizon). Never raises.
  """
  fam_h = {str(k).strip().lower(): int(v) for k, v in (family_horizons or {}).items() if v}
  state_h = int(fam_h.get("continuation", family_horizon_min))
  out: Dict[str, Any] = {
    "costPct": round(float(cost_pct) * 100, 4), "min_samples": int(min_samples), "tstat": float(tstat),
    "stateHorizonMin": state_h, "stateDays": 0, "pendingStateRows": 0, "refusalRows": 0,
    "verdictBasis": "matched blocked-minus-baseline differential (day x horizon cells, day-clustered SE)",
    "gates": {},
  }
  try:
    gates: Dict[str, Any] = {}
    # ── state: folded day cells at continuation's horizon ──
    days = state_days if isinstance(state_days, dict) else {}
    out["stateDays"] = len(days)
    seen: Dict[str, set] = {}
    totals: Dict[str, Dict[str, list]] = {s: {} for s in SIDES}
    for day, sides in days.items():
      for side in SIDES:
        cell = ((sides or {}).get(side) or {}).get(str(state_h))
        if not isinstance(cell, list) or len(cell) < 3 or int(cell[0] or 0) <= 0:
          continue
        totals[side][day] = [int(cell[0]), float(cell[1]), dict(cell[2] or {})]
        for gate in cell[2] or {}:
          seen.setdefault(str(gate), set()).add(side)
    for gate, gate_sides in seen.items():
      for side in sorted(gate_sides):
        blocked: Dict[tuple, list] = {}
        allowed: Dict[tuple, list] = {}
        for day, (n, s, g) in totals[side].items():
          gn, gs = (g.get(gate) or [0, 0.0])[:2]
          if int(gn) > 0:
            blocked[(day, state_h)] = [int(gn), float(gs)]
          if n - int(gn) > 0:
            allowed[(day, state_h)] = [n - int(gn), s - float(gs)]
        gates.setdefault(gate, {}).setdefault("state", {})[side] = _gate_row(
          blocked, allowed, cost_pct=cost_pct, min_samples=min_samples, tstat=tstat)

    # ── refused: raw refusal rows at each call's family horizon, vs admitted calls ──
    rows = [r for r in (gate_probes or []) if isinstance(r, dict)]
    out["pendingStateRows"] = sum(
      1 for r in rows if isinstance(r.get("entryContext"), dict) and r["entryContext"].get("gateProbe") == "state")
    refusals = [r for r in rows if isinstance(r.get("entryContext"), dict)
                and r["entryContext"].get("gateProbe") == "refusal"]
    out["refusalRows"] = len(refusals)

    def _cells(probes: List[Dict[str, Any]], want) -> Dict[tuple, list]:
      cells: Dict[tuple, list] = {}
      for row, ctx, horizon, signed in _probe_observations(probes, horizons_min, require=want):
        if horizon != int(fam_h.get(infer_setup_family(ctx), family_horizon_min)):
          continue
        acc = cells.setdefault((int(row.get("ts") or 0) // 86400, int(horizon)), [0, 0.0])
        acc[0] += 1
        acc[1] += signed
      return cells

    baselines: Dict[str, Dict[tuple, list]] = {}
    for side in SIDES:
      baselines[side] = _cells(admitted_probes or [], lambda c, _s=side: str(c.get("positionSide") or "").lower() == _s)
    refused_gates = sorted({str(r["entryContext"].get("gate")) for r in refusals if r["entryContext"].get("gate")})
    for gate in refused_gates:
      for side in SIDES:
        blocked = _cells(refusals, lambda c, _g=gate, _s=side: (
          c.get("gate") == _g and str(c.get("positionSide") or "").lower() == _s))
        if not blocked:
          continue
        gates.setdefault(gate, {}).setdefault("refused", {})[side] = _gate_row(
          blocked, baselines[side], cost_pct=cost_pct, min_samples=min_samples, tstat=tstat)

    # ── hatched: admitted calls that MET the gate and passed through a hatch, vs admitted calls that
    # never met it (both carrying the gatesPassed stamp, so legacy rows are in neither group) ──
    admitted = [r for r in (admitted_probes or []) if isinstance(r, dict)]

    def _met(ctx: Dict[str, Any], gate: str) -> bool | None:
      passed = ctx.get("gatesPassed")
      if not isinstance(passed, list):
        return None
      return any(isinstance(i, dict) and i.get("gate") == gate for i in passed)

    hatch_gates = sorted({
      str(i.get("gate")) for r in admitted if isinstance(r.get("entryContext"), dict)
      for i in (r["entryContext"].get("gatesPassed") or []) if isinstance(i, dict) and i.get("gate")
    })
    for gate in hatch_gates:
      for side in SIDES:
        def _side_ok(c: Dict[str, Any], _s: str = side) -> bool:
          return str(c.get("positionSide") or "").lower() == _s
        met = _cells(admitted, lambda c, _g=gate: _side_ok(c) and _met(c, _g) is True)
        if not met:
          continue
        clear = _cells(admitted, lambda c, _g=gate: _side_ok(c) and _met(c, _g) is False)
        row = _gate_row(met, clear, cost_pct=cost_pct, min_samples=min_samples, tstat=tstat)
        calls: Dict[str, int] = {}
        for r in admitted:
          ctx = r.get("entryContext") if isinstance(r.get("entryContext"), dict) else {}
          if not _side_ok(ctx):
            continue
          for item in ctx.get("gatesPassed") or []:
            if isinstance(item, dict) and item.get("gate") == gate:
              calls[str(item.get("hatch") or "?")] = calls.get(str(item.get("hatch") or "?"), 0) + 1
        row["byHatch"] = [{"hatch": h, "calls": n} for h, n in sorted(calls.items())]
        gates.setdefault(gate, {}).setdefault("hatched", {})[side] = row
    out["gates"] = dict(sorted(gates.items()))
  except Exception:
    return out
  return out


def probe_taker_fee(memory: Any) -> float:
  """The futures taker fee the probe scoreboards charge: the recorded ``fees.futures_taker`` (what the
  Trading Agent's own cost basis reads, agent.py), else KuCoin's 0.06% default. Never raises."""
  try:
    latest = memory.latest_fees() if callable(getattr(memory, "latest_fees", None)) else None
    fee = _f((latest or {}).get("futures_taker"))
    if fee is not None and math.isfinite(fee) and fee > 0:
      return fee
  except Exception:
    pass
  return 0.0006


def probe_cost_pct(memory: Any, cfg: Any) -> float:
  """Round-trip cost the probe scoreboards charge: 2 x (futures taker + measured slippage per side).

  The same basis the dashboard's strategyEdge, the Supervisor's scoreboard and the Trading Agent's
  edge state use — the taker is the recorded futures fee (``probe_taker_fee``), so no caller carries
  its own copy of the constant. Falls back to the configured slippage prior when fills cannot be read.
  Never raises.
  """
  prior = float(getattr(getattr(cfg, "trading", None), "estimated_slippage_pct", 0.001) or 0.0)
  taker = probe_taker_fee(memory)
  try:
    slip = measured_slippage_pct(
      memory.recent_fills(limit=100), prior,
      min_samples=int(getattr(cfg.trading, "slippage_autotune_min_samples", 8)),
    )
    return 2.0 * (taker + float(slip.get("value") or 0.0))
  except Exception:
    return 2.0 * (taker + prior)


def gate_scoreboard_from_store(memory: Any, *, cost_pct: float, min_samples: int = 20) -> Dict[str, Any]:
  """`gate_scoreboard` over a store: its gate probes, folded state days and signal probes (as the
  admitted baseline), at the families' measured horizons. Returns {"error": ...} instead of raising."""
  try:
    return gate_scoreboard(
      memory.gate_probes(), cost_pct=cost_pct, family_horizons=safe_family_horizons(memory),
      state_days=memory.gate_state_days(), admitted_probes=memory.signal_probes(limit=0),
      min_samples=min_samples,
    )
  except Exception as exc:
    return {"error": str(exc)}


def gate_scoreboard_log_line(board: Any) -> str:
  """One compact line for the periodic log: every gate/side with data, blocked vs baseline. Never raises."""
  try:
    if not isinstance(board, dict):
      return "no scoreboard"
    if board.get("error"):
      return f"unavailable ({board['error']})"
    parts = []
    for gate, sections in (board.get("gates") or {}).items():
      for kind in ("state", "refused", "hatched"):
        for side, row in ((sections or {}).get(kind) or {}).items():
          blocked = row.get("blocked") or {}
          diff = row.get("diffPct")
          se = row.get("diffSePct")
          parts.append(
            f"{gate}/{side} {kind} n={blocked.get('n', 0)}/{blocked.get('days', 0)}d "
            + (f"diff {diff:+.3f}%" if diff is not None else "diff n/a")
            + (f"±{se:.3f}" if se is not None else "")
            + f" matched={row.get('matchedN', 0)} → {row.get('verdict')}"
          )
    head = (f"state@{board.get('stateHorizonMin')}m over {board.get('stateDays', 0)}d "
            f"(+{board.get('pendingStateRows', 0)} settling), {board.get('refusalRows', 0)} refusal rows")
    return head + (": " + "; ".join(parts) if parts else ": nothing scored yet")
  except Exception as exc:
    return f"unavailable ({exc})"


# ── EXECUTION MAP: fill rate and fill quality by resting distance (2026-09-25) ────────────────────────
#
# Why this replaced two static prompt lines. The prompt used to say "target levels within 0.5-1.5 x ATR
# ... farther than 1.5 ATR rarely fills", and quoted a Jul-30 replay of 82 expired limits (+13.05R for
# crossing, -0.37R for waiting longer, +0.388R for "still clears the RR floor after crossing"). Measured
# on 183 placements Sep 20-24 the band itself sat in the unfillable zone — within the 15-minute lease
# 62% of limits filled at <0.25 ATR15, 53% at 0.25-0.5, 33% at 0.5-1, 8% at 1-2 and 0% at >=2 — and
# on a new 124-limit sample the Jul-30 direction held (crossing beat resting) but the RR-filter and TTL
# sub-claims did not reproduce. Worse, both the old and the new replay numbers are one-regime: the
# crossing gain came from rally longs (+0.32R, n=83) while shorts were ~0 (+0.01R, n=41) and the live
# model's crosses averaged -0.03R. A frozen figure in the prompt goes stale; this table re-reads the
# bot's OWN recent placements every run, so it follows the regime and is never a gate.
#
# The bucket edges are UNITS (fractions of one 15m ATR), not tuned thresholds. "marketable" (the limit
# crossed the live price) is its own bucket on purpose: in the Aug-Sep chop it was the WORST category
# (-0.42R over 7), in the Sep 17-23 rally one of the better ones, so a table folding it into "near
# price" would imply an answer the data does not give.
EXECUTION_DISTANCE_BUCKETS: tuple = (
  ("marketable", None, 0.0),
  ("<0.25", 0.0, 0.25),
  ("0.25-0.5", 0.25, 0.5),
  ("0.5-1", 0.5, 1.0),
  ("1-2", 1.0, 2.0),
  (">=2", 2.0, None),
)
# Counterfactual resting depths (ATR15): the bucket EDGES, so each live bucket [a, b) reads against the
# counterfactual at both of its edges. 0 is a cross at the call price — filled by definition.
EXECUTION_CF_DEPTHS_ATR: tuple = (0.0, 0.25, 0.5, 1.0, 2.0)


def passive_distance_atr(side: Any, limit_price: Any, market_price: Any, atr_pct: Any) -> float | None:
  """How far a limit rested from the live price, in 15m ATRs; ``<= 0`` means it crossed (marketable).

  ``d = (market - limit) / market`` for a buy and ``(limit - market) / market`` for a sell — positive when
  the order waits for price to come to it — divided by the 15m ATR as a fraction of price. ``atr_pct``
  is ``regime.intraday_atr_pct``, which is in PERCENT (``analytics``: atr / close * 100), hence the
  /100. None when any input is missing or unusable. Never raises.
  """
  s = _norm_side(side)
  lim, mkt, atr = _f(limit_price), _f(market_price), _f(atr_pct)
  if s is None or lim is None or mkt is None or atr is None:
    return None
  if not all(math.isfinite(v) for v in (lim, mkt, atr)) or lim <= 0 or mkt <= 0 or atr <= 0:
    return None
  frac = (mkt - lim) / mkt if s == "long" else (lim - mkt) / mkt
  return frac / (atr / 100.0)


def distance_bucket(d: Any) -> str | None:
  """The EXECUTION_DISTANCE_BUCKETS label for a passive distance in ATR15, or None if unusable."""
  v = _f(d)
  if v is None or not math.isfinite(v):
    return None
  if v <= 0.0:
    return "marketable"
  for label, _lo, hi in EXECUTION_DISTANCE_BUCKETS[1:]:
    if hi is None or v < hi:
      return label
  return None


def _entry_distance_atr(row: Any) -> float | None:
  """Passive distance (ATR15) of the entry a trade row or a realized close was placed with."""
  if not isinstance(row, dict):
    return None
  ctx = row.get("entryContext") if isinstance(row.get("entryContext"), dict) else {}
  regime = ctx.get("regime") if isinstance(ctx.get("regime"), dict) else {}
  side = ctx.get("positionSide") or row.get("side")
  limit = ctx.get("entryPrice") if ctx.get("entryPrice") is not None else row.get("price")
  return passive_distance_atr(side, limit, ctx.get("marketPriceAtSignal"), regime.get("intraday_atr_pct"))


def _xm_stat(vals: List[float], min_n: int, *, mean_key: str, se_key: str, digits: int = 3) -> Dict[str, Any]:
  """``{n, mean, se}`` — or the mean spelled 'insufficient' below ``min_n``, so noise never reads as signal."""
  n = len(vals)
  if n < max(1, int(min_n)):
    return {"n": n, mean_key: "insufficient"}
  return {"n": n, mean_key: round(sum(vals) / n, digits), se_key: round(_stderr(vals), digits)}


def _span_days(stamps: List[float]) -> float | None:
  stamps = [s for s in stamps if s]
  return round((max(stamps) - min(stamps)) / 86400.0, 1) if len(stamps) >= 2 else None


def execution_map(
  trades: List[Dict[str, Any]],
  closes: List[Dict[str, Any]],
  *,
  min_n: int = 10,
  probes: List[Dict[str, Any]] | None = None,
  cost_pct: float = 0.001,
  family_horizons: Dict[str, int] | None = None,
  family_horizon_min: int = 60,
  horizons_min: tuple = (5, 15, 60, 240),
  rr_floor: float = 1.5,
) -> Dict[str, Any]:
  """Fill rate and fill quality by resting distance, from the bot's OWN placements. Data, never a gate.

  ``trades`` must be the bot's limit-entry records exactly as ``performanceSummary.limitFillRate`` counts
  them (``memory.limit_entry_records()``): every one is a placement and ``filled is True`` is a fill, so
  the two numbers cannot disagree (``totals`` equals that rate). Per EXECUTION_DISTANCE_BUCKETS bucket:

  * ``placed`` / ``filled`` / ``fillRate`` within the lease, ``ideas`` (distinct symbol+side — 12
    re-placements of one H-USDT short are one idea, not twelve) and ``medianMinToFill``;
  * ``fills`` — the realized R of CLOSED trades entered at that distance, read from ``closes``' own
    entryContext (marketPriceAtSignal, entryPrice, regime.intraday_atr_pct), one observation per entry
    order (partial closes of one ``entryOrderId`` are summed). The closes list holds ~200 closes (weeks);
    joining through ``trades`` (~100 placements, days) would leave the far buckets empty — about 6 far
    fills in 183 placements. The two windows differ, and ``window`` says by how much;
  * ``fillAdjustedR`` = fillRate x fills.meanR — what a placement at that distance returned on average,
    an unfilled limit counting zero. Compare THIS across distances, not planned RR.

  Every mean below ``min_n`` samples is spelled 'insufficient' instead of a number.

  ``counterfactual`` (with ``probes``) scores EVERY direction call at every depth in
  EXECUTION_CF_DEPTHS_ATR from its signal probe, so depths the model has stopped using stay measured:
  a limit ``d`` ATR15 away counts as filled when the adverse move inside the lease (``leaseLow`` /
  ``leaseHigh``, 1m futures bars) reached it, and its outcome is the family-horizon settlement against
  that fill price, net of ``cost_pct``, in % and in the call's own R (the planned stop distance). These
  are GROSS forward moves before trade management — on the validated 1m replay (Sep 20-24) probe means
  ran ~0.1R above the managed outcome — and are labelled so. ``crossBand`` buckets the same calls by the
  net RR their bracket would have had if CROSSED at the live price (``crossedNetRr``: <1, 1..floor,
  >=floor), per side, scored the same way. Report-only: it is the zero-stake evidence any future
  sub-floor lane would have to clear, replacing a rally-only replay. Both use ``_probe_observations`` so
  the one-per-symbol-per-window de-overlap applies, each family at its own horizon.

  Never raises: any failure returns what was computed so far.
  """
  out: Dict[str, Any] = {
    "unit": "ATR15",
    "minN": int(min_n),
    "byDistance": {},
    "totals": {"placed": 0, "filled": 0, "fillRate": None},
    "unbucketed": {"placed": 0, "filled": 0, "closes": 0},
    "window": {},
    "note": (
      "Your own recent limit entries by resting distance: |limit - live price| at the call, in 15m ATRs "
      "(marketable = crossed). placed/filled/fillRate use the same records as performanceSummary.limitFillRate "
      "(the most recent placements); fills.meanR is the realized R of closed trades entered at that distance "
      "(a longer window of closes) — see window. fillAdjustedR = fillRate x fills.meanR, an unfilled limit "
      "earning zero, is the number to compare across distances, not planned RR. 'insufficient' means fewer "
      "than minN samples: not evidence either way. counterfactual scores EVERY call at every depth from its "
      "signal probe, so depths you have stopped using stay measured; it is the gross forward move at the "
      "family's own horizon, before trade management, net of round-trip cost, so it reads higher than a "
      "managed trade would. crossBand is report-only: the calls' net RR if crossed at the live price, per "
      "side, scored the same way. The rest is pooled over both sides and whatever regime the window covers: "
      "a one-regime window is weak evidence."
    ),
  }
  try:
    rows = [t for t in (trades or []) if isinstance(t, dict)]
    buckets: Dict[str, Dict[str, Any]] = {
      label: {"placed": 0, "filled": 0, "ideas": set(), "mins": []}
      for label, _lo, _hi in EXECUTION_DISTANCE_BUCKETS
    }
    per_idea: Dict[tuple, List[int]] = {}
    placed_ts: List[float] = []
    for t in rows:
      filled = t.get("filled") is True
      out["totals"]["placed"] += 1
      out["totals"]["filled"] += int(filled)
      ts = _f(t.get("ts")) or 0.0
      placed_ts.append(ts)
      ctx = t.get("entryContext") if isinstance(t.get("entryContext"), dict) else {}
      side = _norm_side(ctx.get("positionSide") or t.get("side"))
      idea = (str(t.get("symbol") or "?"), side or "?")
      tally = per_idea.setdefault(idea, [0, 0])
      tally[0] += 1
      tally[1] += int(filled)
      label = distance_bucket(_entry_distance_atr(t))
      if label is None:
        out["unbucketed"]["placed"] += 1
        out["unbucketed"]["filled"] += int(filled)
        continue
      b = buckets[label]
      b["placed"] += 1
      b["ideas"].add(idea)
      if filled:
        b["filled"] += 1
        fill_ts = _f(t.get("fillTs"))
        if fill_ts is not None and ts and fill_ts >= ts:
          b["mins"].append((fill_ts - ts) / 60.0)
    tot = out["totals"]
    tot["fillRate"] = round(tot["filled"] / tot["placed"], 3) if tot["placed"] else None

    # Realized R per entry order, bucketed on the close's OWN entry context.
    by_entry: Dict[str, Dict[str, Any]] = {}
    close_ts: List[float] = []
    for i, c in enumerate(closes or []):
      if not isinstance(c, dict):
        continue
      r = _f(c.get("realizedR"))
      if r is None or not math.isfinite(r):
        continue
      label = distance_bucket(_entry_distance_atr(c))
      if label is None:
        out["unbucketed"]["closes"] += 1
        continue
      ctx = c.get("entryContext") if isinstance(c.get("entryContext"), dict) else {}
      oid = str(ctx.get("entryOrderId") or ctx.get("entryClientOid") or "").strip()
      key = f"oid:{oid}" if oid else f"row:{i}"
      g = by_entry.setdefault(key, {"label": label, "r": 0.0})
      g["r"] += r
      close_ts.append(_f(c.get("ts")) or 0.0)
    fills_by_label: Dict[str, List[float]] = {}
    for g in by_entry.values():
      fills_by_label.setdefault(g["label"], []).append(g["r"])

    for label, _lo, _hi in EXECUTION_DISTANCE_BUCKETS:
      b = buckets[label]
      placed = b["placed"]
      fill_rate: Any = None
      if placed:
        fill_rate = round(b["filled"] / placed, 3) if placed >= max(1, int(min_n)) else "insufficient"
      mins = sorted(b["mins"])
      median_min = None
      if mins:
        k = len(mins)
        median_min = round(mins[k // 2] if k % 2 else 0.5 * (mins[k // 2 - 1] + mins[k // 2]), 1)
      fills = _xm_stat(fills_by_label.get(label, []), min_n, mean_key="meanR", se_key="seR")
      adj = None
      if isinstance(fill_rate, float) and isinstance(fills.get("meanR"), float):
        adj = round(fill_rate * fills["meanR"], 3)
      out["byDistance"][label] = {
        "placed": placed,
        "filled": b["filled"],
        "fillRate": fill_rate,
        "ideas": len(b["ideas"]),
        "medianMinToFill": median_min,
        "fills": fills,
        "fillAdjustedR": adj,
      }
    # The ideas re-placed most often in the window, with their fills: churn on one never-filling level
    # is invisible in entryExpiries (it only counts since the last run) — one H-USDT short was re-placed
    # 12 times in 9h on Sep 23 at 0.9-2.3 ATR15, for 0 fills.
    repeats = sorted(((n[0], n[1], k) for k, n in per_idea.items() if n[0] >= 2), key=lambda x: -x[0])[:3]
    out["mostReplaced"] = [
      {"symbol": k[0], "side": k[1], "placed": p, "filled": f} for p, f, k in repeats
    ]
    out["window"] = {
      "placements": tot["placed"],
      "placementsSpanDays": _span_days(placed_ts),
      "closesScored": sum(len(v) for v in fills_by_label.values()),
      "closesSpanDays": _span_days(close_ts),
    }
    if probes:
      out["counterfactual"] = _execution_counterfactual(
        probes, min_n=min_n, cost_pct=cost_pct, family_horizons=family_horizons,
        family_horizon_min=family_horizon_min, horizons_min=horizons_min,
      )
      out["crossBand"] = _execution_cross_band(
        probes, min_n=min_n, cost_pct=cost_pct, family_horizons=family_horizons,
        family_horizon_min=family_horizon_min, horizons_min=horizons_min, rr_floor=rr_floor,
      )
  except Exception:
    return out
  return out


def _risk_frac(entry: Any, stop: Any) -> float | None:
  """|entry - stop| / entry — one R as a fraction of price — or None when unusable."""
  e, s = _f(entry), _f(stop)
  if e is None or s is None or not (math.isfinite(e) and math.isfinite(s)) or e <= 0 or s <= 0:
    return None
  r = abs(e - s) / e
  return r if r > 0 else None


def _family_horizon_filter(family_horizons: Dict[str, int] | None, default: int):
  fam_h = {str(k).strip().lower(): int(v) for k, v in (family_horizons or {}).items() if v}
  return lambda ctx, horizon: horizon == int(fam_h.get(infer_setup_family(ctx), default))


def _execution_counterfactual(
  probes: List[Dict[str, Any]],
  *,
  min_n: int,
  cost_pct: float,
  family_horizons: Dict[str, int] | None,
  family_horizon_min: int,
  horizons_min: tuple,
) -> Dict[str, Any]:
  """Every call at every depth: would a limit ``d`` ATR15 away have filled in the lease, and then what.

  For a long the adverse move is ``(base - leaseLow) / base``, for a short ``(leaseHigh - base) / base``,
  in ATR15 (``atr15Pct`` stamped at the call). A depth ``d > 0`` fills when that move reached ``d``; the
  fill price is ``base * (1 -/+ d * atr)`` and the outcome is the family-horizon settlement (plus any
  funding credited to the window) against it, net of ``cost_pct`` (the taker round trip — conservative
  for a resting maker entry). R is the call's own planned stop distance, the same unit at every depth.
  Depth 0 is a cross at the call price.
  """
  at_horizon = _family_horizon_filter(family_horizons, family_horizon_min)

  def _usable(ctx: Dict[str, Any]) -> bool:
    low, high, atr = _f(ctx.get("leaseLow")), _f(ctx.get("leaseHigh")), _f(ctx.get("atr15Pct"))
    return all(v is not None and math.isfinite(v) and v > 0 for v in (low, high, atr))

  acc = {d: {"calls": 0, "filled": 0, "pct": [], "r": []} for d in EXECUTION_CF_DEPTHS_ATR}
  calls = 0
  for _row, ctx, horizon, signed in _probe_observations(probes, horizons_min, require=_usable):
    if not at_horizon(ctx, horizon):
      continue
    probe = ctx.get("signalProbe") or {}
    base = float(_f(ctx.get("marketPriceAtSignal")))
    settle = _f(probe.get(f"m{int(horizon)}"))
    if settle is None or settle <= 0:
      continue
    credit = _f(probe.get(f"f{int(horizon)}"))
    credit = credit if credit is not None and math.isfinite(credit) else 0.0
    long_side = str(ctx.get("positionSide") or "").lower() == "long"
    atr = float(_f(ctx.get("atr15Pct"))) / 100.0
    adverse = ((base - float(_f(ctx.get("leaseLow")))) if long_side
               else (float(_f(ctx.get("leaseHigh"))) - base)) / base
    risk = _risk_frac(ctx.get("plannedEntry"), ctx.get("plannedStop"))
    calls += 1
    for depth, slot in acc.items():
      slot["calls"] += 1
      if depth > 0 and adverse < depth * atr - 1e-12:
        continue
      slot["filled"] += 1
      px = base * (1.0 - depth * atr) if long_side else base * (1.0 + depth * atr)
      if px <= 0:
        continue
      ret = (settle - px) / px if long_side else (px - settle) / px
      net = ret + credit - float(cost_pct)
      slot["pct"].append(net * 100.0)
      if risk is not None:
        slot["r"].append(net / risk)
  by_depth: Dict[str, Any] = {}
  for depth, slot in acc.items():
    key = "cross" if depth == 0 else f"{depth:g}"
    n_calls = slot["calls"]
    rate: Any = None
    if n_calls:
      rate = round(slot["filled"] / n_calls, 3) if n_calls >= max(1, int(min_n)) else "insufficient"
    net_r = _xm_stat(slot["r"], min_n, mean_key="meanR", se_key="seR")
    by_depth[key] = {
      "calls": n_calls,
      "filled": slot["filled"],
      "fillRate": rate,
      "netPct": _xm_stat(slot["pct"], min_n, mean_key="meanPct", se_key="sePct"),
      "netR": net_r,
      "fillAdjustedR": (round(rate * net_r["meanR"], 3)
                        if isinstance(rate, float) and isinstance(net_r.get("meanR"), float) else None),
    }
  out: Dict[str, Any] = {
    "basis": "gross forward move at each family's horizon from signal probes, before trade management; "
             "net of round-trip cost",
    "calls": calls,
  }
  if calls:
    out["byDepth"] = by_depth
  return out


def _execution_cross_band(
  probes: List[Dict[str, Any]],
  *,
  min_n: int,
  cost_pct: float,
  family_horizons: Dict[str, int] | None,
  family_horizon_min: int,
  horizons_min: tuple,
  rr_floor: float,
) -> Dict[str, Any]:
  """Calls bucketed by the net RR their bracket would have had CROSSED at the live price, per side.

  Scored at each family's own horizon from the call price, net of cost, in % and in R from the live
  price to the planned stop. Report-only: no gate reads it. 1.0 is the symmetric-bet line; the upper
  edge is the live RR floor (``rr_floor`` = MIN_FUTURES_RR), so the bands follow the config.
  """
  at_horizon = _family_horizon_filter(family_horizons, family_horizon_min)
  floor = float(rr_floor)
  if floor > 1.0:
    bands = (("<1.0", None, 1.0), (f"1.0-{floor:g}", 1.0, floor), (f">={floor:g}", floor, None))
  else:
    bands = ((f"<{floor:g}", None, floor), (f">={floor:g}", floor, None))

  def _stamped(ctx: Dict[str, Any]) -> bool:
    rr = _f(ctx.get("crossedNetRr"))
    return rr is not None and math.isfinite(rr) and rr >= 0

  acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {
    label: {s: {"pct": [], "r": []} for s in SIDES} for label, _lo, _hi in bands
  }
  for _row, ctx, horizon, signed in _probe_observations(probes, horizons_min, require=_stamped):
    if not at_horizon(ctx, horizon):
      continue
    rr = float(_f(ctx.get("crossedNetRr")))
    side = _norm_side(ctx.get("positionSide"))
    if side is None:
      continue
    label = next(lbl for lbl, lo, hi in bands if (lo is None or rr >= lo) and (hi is None or rr < hi))
    net = signed - float(cost_pct)
    acc[label][side]["pct"].append(net * 100.0)
    risk = _risk_frac(ctx.get("marketPriceAtSignal"), ctx.get("plannedStop"))
    if risk is not None:
      acc[label][side]["r"].append(net / risk)
  out: Dict[str, Any] = {
    "basis": "net RR of the planned bracket if crossed at the live price; outcome = gross forward move "
             "from the call price at the family's horizon, net of cost, before trade management; "
             "report-only",
    "floor": floor,
    "calls": sum(len(v["pct"]) for sides in acc.values() for v in sides.values()),
  }
  if not out["calls"]:
    return out
  out["bands"] = {
    label: {
      side: {**_xm_stat(v["pct"], min_n, mean_key="meanPct", se_key="sePct"),
             **{k: v2 for k, v2 in _xm_stat(v["r"], min_n, mean_key="meanR", se_key="seR").items()
                if k != "n"}}
      for side, v in sides.items()
    }
    for label, sides in acc.items()
  }
  return out


def adaptive_stop_atr_mult(
  closes: List[Dict[str, Any]],
  base_mult: float,
  *,
  lookback: int = 30,
  min_samples: int = 10,
  step: float = 0.5,
  max_mult: float = 4.0,
) -> Dict[str, Any]:
  """Widen (or relax) the noise floor on stop distance from the bot's own MAE record.

  Classic MAE analysis (Sweeney): a stop belongs just *outside* the adverse excursion that your
  WINNING trades routinely survive. If winners habitually dip most of the way to the stop before
  working, the stop is inside the noise and is converting winners into losers at random.

  **The adaptation is deliberately one-directional — it can widen, never tighten.** The two readings
  are not symmetric evidence. Winners surviving deep heat is a *direct* observation that the stop
  nearly killed a trade that then worked. Winners showing little heat is *ambiguous*: it means either
  the stop has room to spare, or the stop already eliminated everything that breathed, leaving a
  survivor pool biased toward trades that went green immediately. Under a too-tight stop those two
  are indistinguishable — and on this account's real data they are actively misleading (winners
  averaged 0.17R of heat precisely *because* the 1.4x-ATR stop truncated the rest, which a symmetric
  rule would have read as permission to tighten further). The consequences are asymmetric too: a stop
  inside the noise destroys the strategy, while a slightly generous one only costs some position size.
  So the learner adds room on evidence and otherwise leaves the configured floor alone; lowering the
  floor stays an explicit operator decision via ``stop_atr_floor_mult``.

  The signal is already recorded per close (``troughPnl`` / ``plannedMaxLossUsd``), so the floor tunes
  itself instead of being a constant someone has to revisit. Moves by at most one ``step`` per
  evaluation so the geometry drifts rather than lurches, and falls back to ``base_mult`` until there
  is a real sample.
  """
  base = _f(base_mult) or 0.0
  if base <= 0:
    return {"value": base, "source": "disabled", "n": 0}
  quality = entry_quality_stats(closes or [], lookback)
  winners = [c for c in (closes or []) if (_f(c.get("realizedR")) or 0.0) > 0]
  wq = entry_quality_stats(winners, lookback)
  n = int(wq.get("n") or 0)
  if n < max(1, int(min_samples)):
    return {"value": base, "source": "base", "n": n, "avg_mae_r_winners": wq.get("avg_mae_r")}
  winner_mae = _f(wq.get("avg_mae_r"))
  if winner_mae is None:
    return {"value": base, "source": "base", "n": n}
  # Winners eating >=0.6R of heat on average means the stop sits inside the working range: add room.
  # Anything less is not trustworthy evidence in the other direction (see the docstring): hold.
  if winner_mae >= 0.6:
    value = min(float(max_mult), base + float(step))
    why = f"winners average {winner_mae:.2f}R of adverse heat — stop sits inside the working range"
  else:
    value = base
    why = (
      f"winners average {winner_mae:.2f}R of adverse heat — holding the configured floor "
      "(low heat under a tight stop is survivorship, not evidence of slack)"
    )
  return {
    "value": value,
    "source": "measured" if value != base else "base",
    "n": n,
    "avg_mae_r_winners": winner_mae,
    "avg_mae_r_all": quality.get("avg_mae_r"),
    "reason": why,
  }


def expectancy_size_factor(
  stats: Dict[str, Any],
  cfg: EdgeConfig,
  *,
  direction: str | None = None,
  symbol: str | None = None,
) -> float:
  """Risk multiplier for a losing evidence bucket; it never increases configured risk.

  Target stretching is a poor response to weak realized results because it can make take-profits
  less reachable. This controller instead keeps structural targets intact and reduces capital at
  risk until the relevant long/short or symbol bucket recovers. Insufficient samples stay at full
  configured risk rather than pretending a tiny sample is conclusive.
  """
  if not cfg.enabled:
    return 1.0
  row: Dict[str, Any] | None
  minimum = max(1, int(cfg.direction_min_trades))
  if symbol:
    row = (stats.get("per_symbol") or {}).get(symbol)
    minimum = max(minimum, int(cfg.symbol_rr_min_trades))
  elif direction:
    row = (stats.get("per_direction") or {}).get(str(direction).lower())
  else:
    row = stats
    minimum = max(minimum, int(cfg.min_trades))
  if not row:
    return 1.0
  # Realized R makes outcomes comparable across notionals. During migration, fall back to legacy
  # dollar PnL until a full attributed sample exists; never mix dollars and R in one estimate.
  r_count = int(row.get("r_n") or 0)
  if r_count >= minimum:
    count = r_count
    outcome = float(row.get("r_net") or 0.0)
  else:
    count = int(row.get("n") or 0)
    outcome = float(row.get("net") or 0.0)
  if count < minimum:
    return 1.0
  if outcome < 0:
    return min(1.0, max(0.0, float(cfg.negative_expectancy_size_factor)))
  return 1.0


def adaptive_min_rr(stats: Dict[str, Any], base_rr: float, cfg: EdgeConfig, now: float | None = None) -> float:
  """Futures reward:risk floor, raised while the rolling expectancy is *recently* negative.

  With too little data (or the controller disabled) this is exactly `base_rr` — the static guard
  keeps working. When the last `lookback` trades are net-losing, demand `rr_step` more reward per
  unit risk (capped at `rr_cap`); it relaxes back to base automatically once realized expectancy
  turns positive.

  Staleness guard (fixes a self-defeating doom loop): if the raised floor freezes trading — no
  qualifying setup can clear it in a choppy tape — then no new closes arrive, so expectancy stays
  negative on the *same old* losses and the floor would stay raised forever, preventing the very
  wins that would lower it. So once the last close is older than `rr_stale_hours`, the negative
  signal is treated as stale and the floor decays back to base to let the bot try again at the
  (still-validated) base R:R.
  """
  if not cfg.enabled or base_rr <= 0:
    return base_rr
  if int(stats.get("n") or 0) < cfg.min_trades:
    return base_rr
  if float(stats.get("expectancy") or 0.0) >= 0:
    return base_rr
  last_ts = int(stats.get("last_close_ts") or 0)
  if cfg.rr_stale_hours > 0 and last_ts > 0:
    ref = now if now is not None else time.time()
    if (ref - last_ts) > cfg.rr_stale_hours * 3600:
      return base_rr
  return min(base_rr + cfg.rr_step, max(cfg.rr_cap, base_rr))


def symbol_adaptive_rr(
  symbol: str,
  stats: Dict[str, Any],
  base_rr: float,
  cfg: EdgeConfig,
  now: float | None = None,
) -> float:
  """Reward:risk floor for ONE symbol, raised only while THAT symbol is net-losing.

  The old global floor punished every symbol for one symbol's losses — with ETH bleeding, even a
  fresh, liquid screener find (ADA won) had to clear RR 2.0 and mostly got rejected, starving the
  diversification that was actually working. This makes the penalty symbol-specific: a symbol whose
  own recent net is negative (over ``symbol_rr_min_trades``+ closes) must clear ``base+rr_step``
  (capped at ``rr_cap``); symbols with no bad history — including every new coin — trade at ``base_rr``.
  So capital rotates toward what's working instead of being frozen out by the worst name.
  """
  if not cfg.enabled or base_rr <= 0:
    return base_rr
  row = (stats.get("per_symbol") or {}).get(symbol)
  if not row or int(row.get("n") or 0) < cfg.symbol_rr_min_trades:
    return base_rr
  if float(row.get("net") or 0.0) < 0:
    last_ts = int(row.get("last_close_ts") or 0)
    if cfg.rr_stale_hours > 0 and last_ts > 0:
      ref = now if now is not None else time.time()
      if (ref - last_ts) > cfg.rr_stale_hours * 3600:
        return base_rr
    return min(base_rr + cfg.rr_step, max(cfg.rr_cap, base_rr))
  return base_rr


def symbol_bench_until(symbol_closes: List[Dict[str, Any]], cfg: EdgeConfig) -> int:
  """Timestamp until which a symbol is benched (0 = not benched). Pure — caller compares to now.

  A symbol earns the bench when, over its last `bench_lookback` realized closes, it has at least
  `bench_min_losses` losses AND a negative net — the "keeps re-taking the same losing trade" pattern
  (ETH whipsawing in the July chop: 9 trades, -1.61, both directions stopped out). The rest scales
  with severity — `bench_cooldown_hours × min(losses, bench_cooldown_max_mult)` — so a symbol that
  keeps bleeding sits out progressively longer (a fixed 12h let ETH straight back to lose again),
  and it still auto-lifts, so no manual un-benching.
  """
  if not cfg.enabled:
    return 0
  usable = [c for c in (symbol_closes or []) if _f(c.get("pnl")) is not None]
  usable.sort(key=lambda c: c.get("ts") or 0)
  recent = usable[-max(1, int(cfg.bench_lookback)):]
  if len(recent) < cfg.bench_min_losses:
    return 0
  pnls = [float(c["pnl"]) for c in recent]
  losses = len([p for p in pnls if p < 0])
  if losses >= cfg.bench_min_losses and sum(pnls) < 0:
    last_ts = int(recent[-1].get("ts") or 0)
    mult = max(1, min(losses, cfg.bench_cooldown_max_mult))
    return last_ts + int(cfg.bench_cooldown_hours * mult * 3600)
  return 0


def loss_streak_size_factor(loss_streak: int, cfg: EdgeConfig) -> float:
  """Size multiplier (<=1.0) during a losing streak; back to 1.0 on the first win.

  A soft stage before the consecutive-loss circuit breaker: at `streak_threshold`
  consecutive realized losses, scale entries by `streak_size_factor` so the drawdown
  digs slower while the bot re-finds its edge.
  """
  if not cfg.enabled:
    return 1.0
  if int(loss_streak or 0) >= cfg.streak_threshold:
    return min(1.0, max(0.0, cfg.streak_size_factor))
  return 1.0


def _finite(value: Any) -> Optional[float]:
  try:
    out = float(value)
  except (TypeError, ValueError):
    return None
  return out if math.isfinite(out) else None


def _stack_inputs_present(row: Dict[str, Any]) -> bool:
  """A probe recorded with what the live-stack replay needs (fill time and original risk)."""
  fill = _finite(row.get("fillTs"))
  risk = _finite(row.get("initRiskPx"))
  return bool(fill and fill > 0 and risk and risk > 0)


def _delta_block(pairs: list) -> Dict[str, Any]:
  return {"n": len(pairs), "deltaR": round(sum(t for t, _ in pairs) - sum(b for _, b in pairs), 3)}


def _tercile_split(rows: list, cuts: tuple | None, key: str) -> Dict[str, Any]:
  """(value, taken, bracket) rows grouped by the tercile of ``value`` under ``cuts``, with n per bucket."""
  grouped: Dict[str, list] = {}
  for value, taken, bracket in rows:
    grouped.setdefault(breadth_bucket(value, cuts), []).append((taken, bracket))
  return {
    "key": key,
    "cuts": [round(c, 4) for c in cuts] if cuts else None,
    "buckets": {k: _delta_block(v) for k, v in sorted(grouped.items())},
  }


def _trail_by_market_state(rows: list, cuts: tuple | None, adx_rows: list | None = None,
                           adx_cuts: tuple | None = None) -> Dict[str, Any]:
  """Protection exits (taken R vs bracket R) grouped by their entry's breadth24 tercile, with n — and,
  NESTED (so the prompt filter that drops this whole key keeps covering it), by the entry's BTC daily
  ADX tercile. Breadth is direction: an August-style high-breadth chop spreads across every breadth
  tercile, so only the ADX split can ever isolate the chop row the adaptive-trail decision waits for."""
  out = _tercile_split(rows, cuts, "entry breadth24 terciles over the retained exit probes (rolling)")
  out["byBtcDailyAdx"] = _tercile_split(
    adx_rows or [], adx_cuts, "entry BTC daily ADX terciles over the retained exit probes (rolling)")
  return out


def exit_discipline_stats(probes, min_samples: int = 8) -> Dict[str, Any]:
  """Score the model's DISCRETIONARY closes against what leaving the position to the system would have done.

  THE BENCHMARK (2026-09-25). An agent close is scored against ``stack.stackR``: a replay of the LIVE
  exit stack — exchange bracket + breakeven/noise-band trail + carry hold, i.e. ``decide_protection``
  with ProtectionManager's effective config — from the fill over the contract's 1m bars
  (protection.replay_protection_stack, run by main.py once the probe's 8h horizon has passed). Until
  then the benchmark was the bare bracket (original stop/TP or an 8h mark), which the system never
  actually leaves a position on. On the 5 attributed agent closes at the time: -6.23R vs the bracket,
  about -3.5R vs the stack; the whole gap was DASH and INJ, whose trail would have exited near
  +0.3..0.5R long before the TP the bracket credited. That bias is REGIME-SIGNED — in a trend the trail
  leaks against the bracket, so bracket-only makes closes look costly; in chop a run that armed the trail
  and then reversed scores -1R on the bracket but ~0R on the stack, so bracket-only flatters closes — so
  the fix is the right comparator, not a correction factor. ``bracketR`` stays on every row for audit
  and ``stackVsBracket`` shows the gap, i.e. what the trail itself changed. Resolution differs: the
  stack is replayed on 1m bars (last price, ~0.05R median error vs the live mark), the bracket probe
  settles at poll resolution (a touch-and-retrace between polls is missed), so read small gaps loosely.

  Row handling for agent closes — ONE comparator in the verdict (2026-09-25 review): only rows with a
  finite ``stackR`` (``stackScored``) are in ``n`` and the verdict. A row recorded WITH replay inputs
  whose replay has not run yet is ``stackPending``; a row whose replay was given up
  (``resolvedBy`` 'unavailable') is ``stackUnavailable``; a row with neither stack nor replay inputs
  (recorded before the replay existed) is bracket-only and shown for audit as ``legacyBracketScored`` /
  ``legacyDeltaR``. None of those three is counted: falling back to the bracket for them blended two
  benchmarks in one verdict — 5 legacy rows + 3 stack rows read 'closes destroy value' where the 3
  stack rows alone were +0.017R/trade — and would bake the bracket's regime-signed bias into the next
  model's first verdict. scripts/resettle_probes_futures.py backfills the legacy rows once.

  ATTRIBUTION. Only rows with ``closedBy == "agent"`` count toward the verdict. Until 2026-09-19 every
  exit that landed between the original stop and target was scored here, and a trailing-stop exit lands
  exactly there — so of 30 "discretionary closes" the scoreboard reported (verdict "closes destroy
  value", -12.41R), 16 were the code's trailing stop (-8.99R) and ONE was the model (+1.54R, it helped).
  Those exits are still scored — under ``otherExits`` — against the BRACKET, because "did the trail beat
  the bracket?" is exactly the question there (against the stack, a trail exit would score ~0 by
  construction and the evidence a regime-adaptive trail needs would vanish). They are split by how the
  bracket resolved (take_profit / stop / expired), because the sign of the trail's gap depends on it.

  SPLITS, shown with n before any verdict: ``byFamily``, ``byCounterAtEntry`` (the entry's 15m AND 1h
  both opposed the side — the model closing on the condition it entered into) and ``byHtfAligned``
  (4h AND 1D both agreed). The only evidence on premise-opposition closes is n=5 in one rally and it
  splits both ways (DASH/KCS lost by closing, G/XMR helped), so the record decides, not a rule.

  Correction (2026-09-19): an earlier version of this note claimed a replay showed the brackets worth
  +3.05R against +0.42R taken. That replay read KuCoin FUTURES candles in SPOT column order, so it
  missed most stop-outs; re-run correctly over the same 17 trades the early closes HELPED, by +0.84R.

  This is deliberately a MEASUREMENT, not a gate. It is symmetric by construction: if discretionary
  closes start beating the benchmark the verdict flips to ``closes add value`` and says so. ``deltaR``
  is positive when the closes HELPED. ``insufficient data`` until ``min_samples`` closes are scored,
  so a couple of lucky exits never reads as a policy.
  """
  taken: list[float] = []
  bench: list[float] = []
  stack_pairs: list = []           # (taken, stackR) for stack-scored rows
  legacy_pairs: list = []          # (taken, bracketR) for bracket-fallback rows
  both: list = []                  # (stackR, bracketR) where both exist: what the trail changed
  bracket_known: list = []         # (taken, bracketR) for every scored row with a bracket result
  by_family: Dict[str, list] = {}
  by_counter: Dict[str, list] = {}
  by_htf: Dict[str, list] = {}
  others: Dict[str, list] = {}
  others_by_res: Dict[str, Dict[str, list]] = {}
  by_regime: Dict[str, list] = {}
  by_state_rows: list = []         # (entry breadth24 or None, taken, bracket) for protection exits
  # Terciles of the ENTRY breadth24 over every retained exit probe (the rolling window), so the trail's
  # buckets follow the market's own recent distribution rather than fixed edges.
  state_cuts = breadth_terciles([
    b for b in (market_breadth(r.get("marketState")) for r in (probes or []) if isinstance(r, dict))
    if b is not None
  ])
  # ...and of the entry's BTC daily ADX the same way — the chop marker breadth is not.
  adx_cuts = breadth_terciles([
    a for a in (market_adx(r.get("marketState")) for r in (probes or []) if isinstance(r, dict))
    if a is not None
  ])
  by_adx_rows: list = []           # (entry BTC daily ADX or None, taken, bracket) for protection exits
  pending = 0
  unavailable = 0
  for row in probes or []:
    if not isinstance(row, dict):
      continue
    t = _finite(row.get("realizedR"))
    if t is None:
      continue
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    b = _finite(outcome.get("bracketR")) if outcome.get("resolved") else None
    who = str(row.get("closedBy") or "").strip().lower()
    if who != "agent":
      # The code's trailing stop / profit-lock (or a legacy row recorded before attribution existed).
      # Real information about the TRAIL, but not a decision the model made — scored on the bracket.
      if b is None:
        continue
      key = who or "unattributed"
      others.setdefault(key, []).append((t, b))
      res = str(outcome.get("resolved") or "?")
      others_by_res.setdefault(key, {}).setdefault(res, []).append((t, b))
      reg = row.get("regime") if isinstance(row.get("regime"), dict) else None
      if reg and who == "protection":
        rkey = f"{reg.get('market_regime') or '?'}/{reg.get('strength') or '?'}"
        by_regime.setdefault(rkey, []).append((t, b))
      if who == "protection":
        by_state_rows.append((market_breadth(row.get("marketState")), t, b))
        by_adx_rows.append((market_adx(row.get("marketState")), t, b))
      continue
    stack = row.get("stack") if isinstance(row.get("stack"), dict) else None
    s = _finite(stack.get("stackR")) if stack else None
    if s is not None:
      benchmark = s
      stack_pairs.append((t, s))
      if b is not None:
        both.append((s, b))
    elif stack is None and _stack_inputs_present(row):
      pending += 1                 # its replay is still to come: never blend benchmarks mid-flight
      continue
    elif stack is not None:
      unavailable += 1             # replay given up: excluded, never scored on the bracket instead
      continue
    else:
      if b is not None:
        legacy_pairs.append((t, b))   # recorded before the replay existed: audit only, not in the verdict
      continue
    taken.append(t)
    bench.append(benchmark)
    if b is not None:
      bracket_known.append((t, b))
    fam = str(row.get("setupFamily") or "other").strip().lower()
    by_family.setdefault(fam, []).append((t, benchmark))
    for tag, bucket in (("counterAtEntry", by_counter), ("htfAligned", by_htf)):
      v = row.get(tag)
      bucket.setdefault("true" if v is True else ("false" if v is False else "untagged"), []).append(
        (t, benchmark))

  n = len(taken)
  delta = sum(taken) - sum(bench)
  out: Dict[str, Any] = {
    "n": n,
    "takenR": round(sum(taken), 3),
    # What leaving each position to the system was worth: the replayed live stack (the only benchmark).
    "benchmarkR": round(sum(bench), 3),
    "stackR": round(sum(s for _, s in stack_pairs), 3),
    # The bare bracket over the scored rows where it resolved — audit only, NOT the benchmark.
    "bracketR": round(sum(b for _, b in bracket_known), 3),
    "deltaR": round(delta, 3),
    "deltaRPerTrade": round(delta / n, 4) if n else None,
    "beatBenchmark": sum(1 for x, y in zip(taken, bench) if x > y),
    "beatBracket": sum(1 for x, y in bracket_known if x > y),
    "stackScored": len(stack_pairs),
    # Bracket-only closes (recorded before the replay existed): AUDIT ONLY, never in n or the verdict.
    "legacyBracketScored": len(legacy_pairs),
    "legacyDeltaR": round(sum(x for x, _ in legacy_pairs) - sum(y for _, y in legacy_pairs), 3),
    "stackPending": pending,
    "stackUnavailable": unavailable,
    # stack - bracket where both exist: what the trail/breakeven/carry hold changed vs the bare bracket.
    "stackVsBracket": {
      "n": len(both), "deltaR": round(sum(x for x, _ in both) - sum(y for _, y in both), 3),
    },
    "byFamily": {
      f: _delta_block(v) for f, v in sorted(by_family.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    },
    "byCounterAtEntry": {k: _delta_block(v) for k, v in sorted(by_counter.items())},
    "byHtfAligned": {k: _delta_block(v) for k, v in sorted(by_htf.items())},
    # Early exits the MODEL did not make. "protection" = the code's trailing stop / profit-lock, scored
    # against the BRACKET: negative deltaR there means the trail left the bracket's target on the table,
    # which is a statement about the trail, not about the model's judgement.
    "otherExits": {
      k: {**_delta_block(v), "byResolution": {
        r: _delta_block(p) for r, p in sorted((others_by_res.get(k) or {}).items())
      }}
      for k, v in sorted(others.items())
    },
    # The trail's record split by the entry's regime. It is expected to be negative in a trend and
    # positive in chop; a regime-adaptive trail is only justified once BOTH rows exist.
    "trailByRegime": {k: _delta_block(v) for k, v in sorted(by_regime.items())},
    # ...and by the MARKET the entry was taken in. The per-symbol regime tag read 'trending/strong' on
    # 164 of 188 closes and on every tagged trail exit (2026-09-24), so trailByRegime's chop row could
    # never fill. breadth24 terciles alone cannot fill it either — breadth is DIRECTION, and an
    # August-style high-breadth chop spreads across all three — so the same exits are also split by the
    # entry's BTC daily ADX tercile (byBtcDailyAdx, nested), the trend-STRENGTH marker that sat at 10-21
    # through the Aug chop. Report only — NOT shown to the model (agent._exit_discipline_for_prompt drops
    # the whole key) and nothing adapts on it yet.
    "trailByMarketState": _trail_by_market_state(by_state_rows, state_cuts, by_adx_rows, adx_cuts),
  }
  excluded = len(legacy_pairs) + unavailable
  excluded_note = (f" {excluded} close(s) with only a bare-bracket result are shown for audit and not counted."
                   if excluded else "")
  if n < max(1, int(min_samples)):
    out["verdict"] = "insufficient data"
    out["note"] = (
      f"{n} of your own early close(s) scored so far; no verdict until {int(min_samples)}. The splits "
      "(byFamily, byCounterAtEntry, byHtfAligned) are shown with their n but are too small to act on. "
      "(Exits made by the code's trailing stop are reported separately under otherExits and are "
      "not counted here — they were not your decision.)" + excluded_note
    )
    return out
  per = out["deltaRPerTrade"] or 0.0
  if per > 0.02:
    out["verdict"] = "closes add value"
  elif per < -0.02:
    out["verdict"] = "closes destroy value"
  else:
    out["verdict"] = "neutral"
  out["note"] = (
    f"Your last {n} discretionary closes returned {out['takenR']:+.2f}R in total; leaving each "
    f"position to the system (its bracket plus the code's breakeven/trailing stop and carry hold) would "
    f"have returned {out['benchmarkR']:+.2f}R ({out['deltaR']:+.2f}R, {per:+.3f}R per trade; "
    f"{out['beatBenchmark']}/{n} of your closes beat it). The entry already stated the thesis and the "
    "invalidation level, so closing before either is "
    "reached is only an improvement when something has genuinely changed since entry — not when price has "
    "merely moved inside the trade's own noise band." + excluded_note
  )
  return out
