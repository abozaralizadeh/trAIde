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

import math
import time
from typing import Any, Dict, List

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


def signal_edge_stats(
  probes: List[Dict[str, Any]],
  *,
  cost_pct: float = 0.001,
  horizons_min: tuple = (60, 240),
  min_samples: int = 20,
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
  """
  out: Dict[str, Any] = {"n": 0, "verdict": "insufficient data", "cost_pct": cost_pct}
  by_h: Dict[str, List[float]] = {}
  for row in probes or []:
    ctx = row.get("entryContext") if isinstance(row, dict) else None
    if not isinstance(ctx, dict):
      continue
    base = _f(ctx.get("marketPriceAtSignal"))
    side = str(ctx.get("positionSide") or "").lower()
    probe = ctx.get("signalProbe")
    if not base or base <= 0 or side not in {"long", "short"} or not isinstance(probe, dict):
      continue
    for horizon in horizons_min:
      px = _f(probe.get(f"m{int(horizon)}"))
      if px is None or px <= 0:
        continue
      ret = (px - base) / base
      by_h.setdefault(f"{int(horizon)}m", []).append(ret if side == "long" else -ret)
  if not by_h:
    return out
  detail = {}
  best = None
  for key, vals in by_h.items():
    mean = sum(vals) / len(vals)
    hit = sum(1 for v in vals if v > 0) / len(vals)
    detail[key] = {
      "n": len(vals),
      "mean_pct": round(mean * 100, 4),
      "hit_rate": round(hit, 3),
      "net_of_cost_pct": round((mean - cost_pct) * 100, 4),
    }
    if best is None or mean > best[1]:
      best = (key, mean, len(vals))
  out["n"] = max(d["n"] for d in detail.values())
  out["by_horizon"] = detail
  out["best_horizon"] = best[0] if best else None
  if out["n"] < max(1, int(min_samples)):
    out["verdict"] = "insufficient data"
  elif best and best[1] > cost_pct:
    out["verdict"] = "edge"
  else:
    out["verdict"] = "no edge"
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
