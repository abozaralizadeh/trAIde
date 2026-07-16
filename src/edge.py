"""Adaptive edge controller — self-tuning risk from the bot's own realized results.

The Jun–Jul 2026 reviews kept finding the same shape: high win rate, small wins, a few
oversized losses, and a bot that keeps re-taking the losing setup (e.g. re-shorting ETH
four times into the early-July oversold bounce). Hardcoded parameters fix yesterday's
regime; this module instead derives the risk posture from the ROLLING REALIZED OUTCOMES,
so the bot tightens up when it is losing and relaxes back when it is earning — with no
human re-tuning:

  - edge_stats:               rolling win rate / payoff / expectancy / loss streak / per-symbol PnL
  - adaptive_min_rr:          raise the futures reward:risk floor while expectancy is negative
  - symbol_bench_until:       bench a symbol that keeps losing (auto-lifts after a cooldown)
  - loss_streak_size_factor:  shrink size during a losing streak (anti-martingale)

Everything is a pure function over realized-close dicts ({symbol, pnl, ts, closeType}),
unit-tested in tests/test_edge.py. Call sites live in src/agent.py; config in EdgeConfig.
"""

from __future__ import annotations

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
  for c in window:
    sym = str(c.get("symbol") or "?")
    row = per_symbol.setdefault(sym, {"n": 0, "net": 0.0, "losses": 0, "last_close_ts": 0})
    row["n"] += 1
    row["net"] = round(row["net"] + float(c["pnl"]), 6)
    row["last_close_ts"] = int(c.get("ts") or 0)
    if float(c["pnl"]) < 0:
      row["losses"] += 1

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
  }


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
