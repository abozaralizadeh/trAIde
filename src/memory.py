from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .utils import normalize_symbol as _normalize_symbol

logger = logging.getLogger(__name__)

# Hard caps to keep the memory file small and readable (agent only needs recent history).
MAX_PLANS = 3
MAX_TRIGGERS = 5
MAX_COINS = 50
MAX_TRADES = 100
MAX_SENTIMENTS = 10
MAX_DECISIONS = 50       # entry/decline decisions (pnl=None)
MAX_HANDOFF_DECISIONS = 30  # agent handoff markers (pnl=None) — kept in their own bucket so the
                            # high-volume decline/hold snapshots don't evict them from the feed
MAX_CLOSED_TRADES = 200  # closed-trade outcomes (pnl != None) — kept separately
MAX_FEES = 3
MAX_TEMPORARY_NOTES = 20
MAX_PERMANENT_NOTES = 10
MAX_PENDING_AGENT_EVENTS = 200
# Direction calls kept for edge measurement. Retained by COUNT, never by clock — see
# `record_signal_probe` for why coupling evidence supply to anything else creates a doom loop.
MAX_SIGNAL_PROBES = 400
# Per-family retention floor. The global cap alone lets a LOUD family evict a QUIET one's history,
# and the loud one is usually the benched one: a stood-aside family still records a probe on every
# direction call (deliberately — that is how it earns its way back), so it keeps consuming the
# evidence budget while never trading. Measured 2026-09-14: `continuation`, benched for weeks, held
# 283 of 479 probes (59%), and truncating to the 400 cap dropped `fade_extreme` from n=27/"no edge"
# to n=17/"insufficient data" — which RELEASED it to full size, and it promptly lost two trades.
# Losing the evidence for a verdict must never be equivalent to never having had it.
MAX_PROBES_PER_FAMILY = 150
MAX_EXIT_PROBES = 200
# How long an exit probe waits for its bracket to resolve before it is marked to market. Shared by the
# settle step and by the poll loop's "which symbols need a price" query, so both agree on a probe's life.
EXIT_PROBE_EXPIRE_HOURS = 8.0
EXIT_PROBE_PRICE_SOURCE = "futures_mark"   # stamped on every live exit-probe resolution since 2026-09-25
MAX_MACRO_EVENTS = 60
# analyze_market_context data-quality refusals kept for the screener, one per symbol. Each expires on
# its own evidence-derived retryAfter; the cap only bounds the file if the bot analyses a whole universe.
MAX_ANALYSIS_FAILURES = 200
MAX_AGENT_SCHEDULER_SYMBOLS = 100
# Forward-return measurement points, in minutes. 60/240 are the horizons the live entry gates score
# against; 5/15 were added 2026-09-04 to test whether anything predicts at the short horizons where
# order-flow signals are supposed to live — the bot had never looked below one hour. Adding them here
# only measures: `edge.signal_edge_stats` keeps its verdict and its per-family sizing anchored to the
# original horizons, so nothing the bot DOES changes until the evidence says it should.
SIGNAL_PROBE_HORIZONS_MIN: tuple[int, ...] = (5, 15, 60, 240)
# ── Gate probes: what each directional gate blocks (2026-09-25) ──────────────────────────────────────
# The directional/timing gates in the futures limit path returned BEFORE the signal probe, so a refused
# call left no evidence and nothing scored any gate live: 09-22..24 had 8 hard directional refusals,
# 0 probes (the opposing-daily branch did not even log), against 23 of 23 probed NET RR / STAND ASIDE
# refusals. The gates' real footprint is the model's self-censorship (97 of 131 declines cited daily
# exhaustion), which only a MODEL-INDEPENDENT reading of the gate state can see. Both land in their
# own ``gate_probes`` bucket, NEVER in ``signal_probes()`` — a refused or never-proposed call must not
# move a family verdict or evict continuation evidence (already at its 150 cap). Report-only.
#
# Gate codes, in the order place_futures_limit_order checks them. ``DIRECTIONAL_GATES`` are the ones a
# gate-STATE row can evaluate without a call (no confidence, no family); ``SCORED_GATES`` adds the
# refusals that still judge THIS call (volatility hard limit, confidence floor, no-chase, the two
# cooldowns). Every other pre-probe refusal carries a ``STRUCTURAL_REFUSALS`` code (malformed request,
# stale analysis, circuit breaker, caps) and records nothing — it says nothing about the call.
DIRECTIONAL_GATES: tuple[str, ...] = (
  "anti_fomo", "daily_opposing", "h1_align", "tf_conflict", "correlation", "move_24h", "bench",
)
SCORED_GATES: tuple[str, ...] = DIRECTIONAL_GATES + (
  "vol_limit", "confidence_floor", "no_chase", "post_loss_cooldown", "trade_interval",
)
STRUCTURAL_REFUSALS: tuple[str, ...] = (
  "pending_entry", "bracket_missing", "bracket_invalid", "atomic_bracket_disabled", "entry_context",
  "live_book", "restricted", "trade_cap", "sentiment", "new_listing",
)
# Hard-refusal rows are kept per gate, mirroring MAX_PROBES_PER_FAMILY: a loud gate (the exhaustion
# gate in a rally) must not evict a quiet one's history. Retained by count, never by clock.
MAX_GATE_PROBES_PER_GATE = MAX_PROBES_PER_FAMILY
# A gate-STATE row per (symbol, side) at most once per this many minutes — the WIDEST settled horizon,
# so no two stored rows of one symbol/side overlap at any horizon they are scored at. That is what lets
# settled state rows be folded into per-day totals (``gate_state_days``) exactly, with no second de-overlap
# rule: `edge._probe_observations` over the rows being folded is the only one.
GATE_STATE_WINDOW_MIN = max(SIGNAL_PROBE_HORIZONS_MIN)
# Why fold at all: state rows arrive at ~6x the hard-refusal rate (every analysed symbol, both sides),
# and resolving a ~0.17%/call gate effect needs MONTHS of day-level data. Raw rows at that rate would
# add ~1 MB a month to a file rewritten every poll, and a per-gate row cap would hold only the last few
# days — a one-regime scoreboard by construction. So a fully settled state row is folded into its UTC
# day's per-side, per-horizon sums (total, and per gate) and dropped. Day cells are kept by COUNT (four
# months), each as ONE compact JSON string: the store is written with indent=2, which would otherwise
# spread a day's ~60 numbers over ~120 lines (~2.5 KB/day instead of ~0.9 KB).
MAX_GATE_STATE_DAYS = 120
# Backstop only: unsettled state rows normally fold ~4.8h after they are recorded.
MAX_GATE_STATE_UNFOLDED = 600


# Bot-placed limit entries carry this clientOid prefix. ONE predicate (`is_limit_entry_record`) defines
# "a limit placement" for both `performanceSummary.limitFillRate` and `edge.execution_map`, so the
# aggregate rate and the per-distance table are computed from the same records and cannot disagree.
LIMIT_ENTRY_CLIENT_OID_PREFIX = "traide-entry-"


def is_limit_entry_record(trade: Any) -> bool:
  """True for a trade row the bot placed as a tagged limit entry (filled or not)."""
  return isinstance(trade, dict) and str(trade.get("clientOid") or "").startswith(LIMIT_ENTRY_CLIENT_OID_PREFIX)


_ATR_QUARANTINE_RE = re.compile(
  r"daily ATR\s+([0-9]+(?:\.[0-9]+)?)%\s+exceeds\s+([0-9]+(?:\.[0-9]+)?)%",
  re.IGNORECASE,
)


def _sanitize_taker_flow(value: Any) -> Optional[Dict[str, Any]]:
  """Whitelist a taker-flow reading down to the fields worth persisting, or None.

  Shares are ratios in [0,1] and are dropped (not clamped) when out of range, so a parsing bug
  shows up as missing evidence rather than as a plausible-looking 0.0 that would quietly bias every
  statistic computed from it. ``ageSec`` is kept because a reading taken three polls ago is weaker
  evidence than one taken this poll, and the analysis needs to be able to tell.
  """
  raw = value if isinstance(value, dict) else None
  if not raw:
    return None
  out: Dict[str, Any] = {}
  for key in ("buyShare", "buyTradeShare", "newBuyShare"):
    try:
      share = float(raw.get(key))
    except (TypeError, ValueError):
      continue
    if 0.0 <= share <= 1.0:
      out[key] = round(share, 4)
  for key in ("trades", "newTrades"):
    try:
      out[key] = max(0, int(raw.get(key) or 0))
    except (TypeError, ValueError):
      continue
  for key in ("spanSec", "ageSec"):
    try:
      out[key] = round(max(0.0, float(raw.get(key))), 1)
    except (TypeError, ValueError):
      continue
  out["gapped"] = bool(raw.get("gapped"))
  # A reading with no buy share carries no information; storing the husk would only make probes look
  # flow-stamped to the analysis when they are not.
  return out if "buyShare" in out else None


_MARKET_STATE_BIASES = ("bullish", "bearish", "neutral")


def sanitize_market_state(value: Any) -> Optional[Dict[str, Any]]:
  """Whitelist an analytics.market_state block down to its raw numbers, or None.

  Shared by every place that stamps it (entryContext, signal probes, exit probes) so all three carry the
  same shape. A value out of its domain is DROPPED, not clamped — a parsing bug should show up as missing
  evidence, never as a plausible number that quietly biases a split (the same rule as taker flow).
  """
  raw = value if isinstance(value, dict) else None
  if not raw:
    return None
  out: Dict[str, Any] = {}

  def _num(key: str) -> Optional[float]:
    try:
      val = float(raw.get(key))
    except (TypeError, ValueError):
      return None
    return val if math.isfinite(val) else None

  as_of = _num("asOf")
  if as_of is not None and as_of > 0:
    out["asOf"] = int(as_of)
  universe = _num("universe")
  if universe is not None and universe >= 0:
    out["universe"] = int(universe)
  breadth = _num("breadth24")
  if breadth is not None and 0.0 <= breadth <= 1.0:
    out["breadth24"] = round(breadth, 4)
  for key in ("basketMedian24h", "btc24h", "btc72h"):
    val = _num(key)
    if val is not None:
      out[key] = round(val, 3)
  adx = _num("btcDailyAdx")
  if adx is not None and adx >= 0:
    out["btcDailyAdx"] = round(adx, 2)
  bias = str(raw.get("btcDailyBias") or "").strip().lower()
  if bias in _MARKET_STATE_BIASES:
    out["btcDailyBias"] = bias
  # A block with no reading at all (only a timestamp) carries no information.
  return out if set(out) - {"asOf", "universe"} else None


def _sanitize_agent_scheduler(value: Any) -> Dict[str, Any]:
  """Normalize the small restart-safe state used to throttle discretionary model calls."""
  raw = value if isinstance(value, dict) else {}

  try:
    last_run_ts = max(0.0, float(raw.get("lastRunTs") or 0.0))
  except (TypeError, ValueError):
    last_run_ts = 0.0
  try:
    unproductive_runs = min(100, max(0, int(raw.get("unproductiveRuns") or 0)))
  except (TypeError, ValueError):
    unproductive_runs = 0

  reviewed_prices: Dict[str, float] = {}
  for symbol, price in (raw.get("reviewedPrices") or {}).items() if isinstance(raw.get("reviewedPrices"), dict) else []:
    try:
      normalized = _normalize_symbol(str(symbol))
      numeric = float(price)
    except (TypeError, ValueError):
      continue
    if normalized and numeric > 0:
      reviewed_prices[normalized] = numeric

  observations: Dict[str, Dict[str, Any]] = {}
  raw_observations = raw.get("priceObservations") or {}
  if isinstance(raw_observations, dict):
    for symbol, observation in raw_observations.items():
      if not isinstance(observation, dict):
        continue
      try:
        normalized = _normalize_symbol(str(symbol))
        last_price = float(observation.get("lastPrice") or 0.0)
        noise = max(0.0, float(observation.get("noiseEwmaPct") or 0.0))
        samples = max(0, int(observation.get("samples") or 0))
        updated = max(0, int(observation.get("updated") or 0))
      except (TypeError, ValueError):
        continue
      if normalized and last_price > 0:
        observations[normalized] = {
          "lastPrice": last_price,
          "noiseEwmaPct": noise,
          "samples": samples,
          "updated": updated,
        }

  # Keep the newest bounded set if research rotated through many temporary candidates.
  observations = dict(
    sorted(observations.items(), key=lambda item: item[1].get("updated", 0))[-MAX_AGENT_SCHEDULER_SYMBOLS:]
  )
  if len(reviewed_prices) > MAX_AGENT_SCHEDULER_SYMBOLS:
    reviewed_prices = {
      symbol: price
      for symbol, price in reviewed_prices.items()
      if symbol in observations
    }
    reviewed_prices = dict(list(reviewed_prices.items())[-MAX_AGENT_SCHEDULER_SYMBOLS:])

  # Continuous taker-flow state, one row per symbol. Whitelisted here for the same reason every other
  # key is: this function is the ONLY writer of the persisted scheduler shape, so a field absent from
  # this dict is silently dropped on the next save and the state resets every restart.
  flow: Dict[str, Dict[str, Any]] = {}
  raw_flow = raw.get("flowObservations") or {}
  if isinstance(raw_flow, dict):
    for symbol, observation in raw_flow.items():
      if not isinstance(observation, dict):
        continue
      try:
        normalized = _normalize_symbol(str(symbol))
      except (TypeError, ValueError):
        continue
      reading = _sanitize_taker_flow(observation)
      if not normalized or not reading:
        continue
      for key, caster in (("buyShareEwma", float), ("samples", int), ("updated", int),
                          ("lastCursor", int)):
        try:
          reading[key] = caster(observation.get(key))
        except (TypeError, ValueError):
          continue
      flow[normalized] = reading
  flow = dict(
    sorted(flow.items(), key=lambda item: item[1].get("updated", 0))[-MAX_AGENT_SCHEDULER_SYMBOLS:]
  )

  return {
    "lastRunTs": last_run_ts,
    "unproductiveRuns": unproductive_runs,
    "reviewedPrices": reviewed_prices,
    "priceObservations": observations,
    "flowObservations": flow,
  }


def _probe_settle_tolerance_sec(horizon_min: float) -> float:
  """How late a forward-return stamp may be before it stops being that horizon's return.

  Proportional to the horizon (20%) with a two-poll floor, so a 5-minute point is not allowed the
  half-hour of slack that a 4-hour point can absorb without changing what it measures. The floor
  gives the loop two chances to catch each horizon at a 60s poll; miss both and the observation is
  dropped rather than recorded wrong.
  """
  return max(120.0, 0.2 * max(0.0, float(horizon_min)) * 60.0)


def _probe_price_source(ctx: Dict[str, Any]) -> str:
  """Which market a signal probe's base price came from — and so which one must settle it.

  A forward return is only a return when both ends are read from the SAME market. The base has come
  from the futures mark since 2026-07-19 (`tools._live_entry_price`), but settlement read the SPOT
  ticker, so on a coin where the perp traded away from spot the gap was scored as prediction. It
  landed almost entirely on `funding_carry`, which is chosen exactly when that gap is widest: ONE-USDT
  on 2026-09-20 had spot ~0.0050 against a futures mark ~0.0038, and eight ONE long probes each
  recorded a ~+7.5% 60m return that the contract never made — enough to lift the family from t=0.97
  (stood aside) to t=2.02 (full size) in a day.

  Rows stamped ``spot`` (the entry path fell back to the spot ticker) settle on spot. EVERYTHING else
  settles on futures, including legacy rows with no stamp, because their base has been the futures
  mark since 07-19 and every retained probe is newer than that.
  """
  src = str(ctx.get("priceSource") or "").strip().lower() if isinstance(ctx, dict) else ""
  return "spot" if src == "spot" else "futures"


def _price_from(book: Any, symbol: Any) -> Optional[float]:
  """A positive finite price for ``symbol`` from a price map (plain numbers or ticker objects), or None."""
  if not book:
    return None
  try:
    px = book.get(symbol)
  except Exception:
    return None
  px = getattr(px, "price", px)
  try:
    px = float(px)
  except (TypeError, ValueError):
    return None
  return px if math.isfinite(px) and px > 0 else None


def funding_received_from_history(history: Any, side: Any, t0: float, t1: float) -> Optional[float]:
  """Funding a position on ``side`` would have RECEIVED over the window (t0, t1], as a fraction of notional.

  KuCoin's convention: a positive ``fundingRate`` means longs pay shorts. So at every settlement whose
  ``timepoint`` falls inside the window a long receives ``-rate`` and a short ``+rate``; the result is
  their sum, signed from the position's side (positive = the position was paid). A position opened at
  the settlement instant does not receive it, hence the open left edge.

  Why a probe needs this at all: `funding_carry` is a bet on the TRANSFER, not only on price. A
  price-only forward return scores a carry call as if the payment never happened, which understates a
  real carry edge by roughly ``rate x horizon / interval`` — small at 60m, but the honest number
  either way, and it must never be what keeps a phantom alive (that was the spot/futures basis).

  Returns 0.0 when no settlement fell inside the window and None when the history is unusable, so a
  caller can tell "nothing was paid" from "cannot know". Duplicate timepoints count once. Never raises.
  """
  s = str(side or "").strip().lower()
  if s in ("long", "buy"):
    sign = -1.0
  elif s in ("short", "sell"):
    sign = 1.0
  else:
    return None
  if not isinstance(history, (list, tuple)):
    return None
  try:
    lo, hi = float(t0), float(t1)
  except (TypeError, ValueError):
    return None
  by_ts: Dict[float, float] = {}
  for item in history:
    if not isinstance(item, dict):
      continue
    try:
      rate = float(item.get("fundingRate"))
      ts = float(item.get("timepoint") if item.get("timepoint") is not None else item.get("timePoint"))
    except (TypeError, ValueError):
      continue
    if not (math.isfinite(rate) and math.isfinite(ts)):
      continue
    if ts > 1e12:          # KuCoin reports milliseconds
      ts /= 1000.0
    if lo < ts <= hi:
      by_ts[ts] = rate
  return sign * sum(by_ts.values())


def _signal_stamps_due(
  row: Any,
  prices: Optional[Dict[str, Any]],
  spot_prices: Optional[Dict[str, Any]],
  horizons_min: tuple,
  now: int,
) -> list:
  """The ``(key, horizon, value)`` stamps a signal-probe row is due this poll, without writing them.

  ``value`` is a price (read from the row's OWN market, see `_probe_price_source`) or None for a
  horizon that is past its settle tolerance and so recorded as missed. A due horizon whose market has
  no price yet is simply absent: it is retried next poll until the tolerance writes it off. Pure, so
  the settle step can plan (and fetch funding) outside the store lock and apply under it.
  """
  if not isinstance(row, dict):
    return []
  ctx = row.get("entryContext")
  if not isinstance(ctx, dict) or not ctx.get("marketPriceAtSignal"):
    return []
  probe = ctx.get("signalProbe") if isinstance(ctx.get("signalProbe"), dict) else {}
  ts0 = int(row.get("ts") or 0)
  book = spot_prices if _probe_price_source(ctx) == "spot" else prices
  px: Any = False    # not looked up yet — most rows have nothing due, so skip the lookup for them
  out = []
  for horizon in horizons_min:
    key = f"m{int(horizon)}"
    due_at = ts0 + int(horizon) * 60
    if key in probe or now < due_at:
      continue
    if now - due_at > _probe_settle_tolerance_sec(horizon):
      out.append((key, int(horizon), None))
      continue
    if px is False:
      px = _price_from(book, row.get("symbol"))
    if px is None:
      continue
    out.append((key, int(horizon), px))
  return out


def _lease_settle_tolerance_sec(lease_min: float) -> float:
  """How long after its lease a probe's lease extremes may still be fetched before they are written off.

  Unlike a forward PRICE (which is only that horizon's return if read on time — see
  `_probe_settle_tolerance_sec`), a lease low/high comes from historical 1m bars and is exact whenever
  it is fetched. The window therefore only bounds retries: one lease length (at least the price-stamp
  tolerance) covers a routine restart or deploy without re-asking for a dead symbol forever.
  """
  return max(_probe_settle_tolerance_sec(lease_min), max(0.0, float(lease_min)) * 60.0)


def _funding_backfill_window_sec(horizon_min: float) -> float:
  """How long after a horizon an UNKNOWN funding credit (``f{h}`` None) may still be backfilled.

  Funding history is exact and permanent, like a lease's 1m bars, so a late fetch is not back-stamping
  the way a late price would be; the window only bounds retries so a dead symbol is not asked forever
  (the same rule as ``_lease_settle_tolerance_sec``).
  """
  return _lease_settle_tolerance_sec(horizon_min)


def _credit_backfills_due(row: Any, now: int, horizons_min: tuple) -> list:
  """``(horizon, t_end)`` for every horizon whose price is stamped but whose funding credit is UNKNOWN
  (``f{h}`` present and None) and still inside ``_funding_backfill_window_sec``. ``t_end`` is the time
  the price was observed (``t{h}``, stamped with the unknown credit), else the horizon itself — never
  ``now``, or a late backfill would credit settlements after the horizon. Pure; never raises."""
  try:
    ctx = row.get("entryContext") if isinstance(row, dict) else None
    probe = ctx.get("signalProbe") if isinstance(ctx, dict) else None
    if not isinstance(probe, dict):
      return []
    ts0 = int(row.get("ts") or 0)
    out = []
    for horizon in horizons_min:
      h = int(horizon)
      fk = f"f{h}"
      if fk not in probe or probe.get(fk) is not None:
        continue
      try:
        px = float(probe.get(f"m{h}"))
      except (TypeError, ValueError):
        continue
      if not (math.isfinite(px) and px > 0):
        continue
      due_at = ts0 + h * 60
      if now - due_at > _funding_backfill_window_sec(h):
        continue
      try:
        t_end = float(probe.get(f"t{h}")) if probe.get(f"t{h}") is not None else float(due_at)
      except (TypeError, ValueError):
        t_end = float(due_at)
      out.append((h, t_end))
    return out
  except Exception:
    return []


def _lease_window(row: Any) -> Optional[tuple]:
  """``(t0, t1)`` — the call's order lease in epoch seconds — for a probe that carries ``leaseMin``."""
  if not isinstance(row, dict):
    return None
  ctx = row.get("entryContext")
  if not isinstance(ctx, dict):
    return None
  try:
    lease_min = float(ctx.get("leaseMin"))
    t0 = float(row.get("ts"))
  except (TypeError, ValueError):
    return None
  if not (math.isfinite(lease_min) and lease_min > 0 and math.isfinite(t0) and t0 > 0):
    return None
  return t0, t0 + lease_min * 60.0


def _lease_stamp_due(row: Any, now: float) -> Optional[str]:
  """``'fetch'`` when a probe's lease extremes are due, ``'missed'`` past the tolerance, else None.

  Due once the lease has elapsed AND the last 1m bar inside it has closed (``t1 + 60``), so the high/low
  is final. Rows already stamped (even with None) are never touched again: a measurement that was
  missed is recorded as missed, never back-filled later. Only rows that stamped ``leaseMin`` at the
  call qualify — probes recorded before 2026-09-25 carry no ATR either and could not be scored.
  """
  window = _lease_window(row)
  if window is None:
    return None
  ctx = row["entryContext"]
  if "leaseLow" in ctx or "leaseHigh" in ctx:
    return None
  due_at = window[1] + 60.0
  if now < due_at:
    return None
  lease_min = (window[1] - window[0]) / 60.0
  if now - due_at > _lease_settle_tolerance_sec(lease_min):
    return "missed"
  return "fetch"


def _adaptive_quarantine_seconds(reason: str) -> int:
  """Back off unsafe candidates in proportion to how far their volatility exceeded the gate."""
  text = str(reason or "")
  if "automatic risk quarantine" not in text.lower():
    return 0
  match = _ATR_QUARANTINE_RE.search(text)
  if not match:
    return 24 * 3600
  observed = float(match.group(1))
  limit = max(1e-9, float(match.group(2)))
  excess_ratio = max(1.0, observed / limit)
  # A marginal breach rests for roughly half a day; severe/data-scale discontinuities rest up to
  # one week. The retry time adapts to the evidence and expires automatically without maintenance.
  hours = min(7 * 24.0, max(12.0, 12.0 * excess_ratio * excess_ratio))
  return int(hours * 3600)


def _probe_family(row: Any) -> str:
  ctx = row.get("entryContext") if isinstance(row, dict) else None
  if not isinstance(ctx, dict):
    return "other"
  return str(ctx.get("setupFamily") or "other").strip().lower() or "other"


def _trim_probes_per_family(probes: Any) -> list:
  """Cap probe retention PER FAMILY, preserving chronological order.

  A single global ring buffer makes families compete for one budget, and the family that generates
  the most probes is not the one that most needs them — a stood-aside playbook still records a probe
  on every direction call (that is how it can earn its way back), so it evicts the history of the
  families still trading. When an adversely-judged family's sample falls under the stand-aside's
  min_samples it stops reading "no edge" and starts reading "insufficient data", which RELEASES it to
  full size: the system silently un-learns a verdict it had already paid to establish.

  Keeping the newest ``MAX_PROBES_PER_FAMILY`` of each family bounds the file just as well (families
  are a small fixed set) while guaranteeing every playbook keeps enough evidence to sustain its own
  verdict. Never raises; non-dict rows are dropped.
  """
  rows = [r for r in (probes or []) if isinstance(r, dict)]
  if not rows:
    return []
  keep_ids: set[int] = set()
  buckets: Dict[str, list] = {}
  for row in rows:
    buckets.setdefault(_probe_family(row), []).append(row)
  for bucket in buckets.values():
    for row in bucket[-MAX_PROBES_PER_FAMILY:]:
      keep_ids.add(id(row))
  return [r for r in rows if id(r) in keep_ids]


def _gate_probe_kind(row: Any) -> Optional[str]:
  ctx = row.get("entryContext") if isinstance(row, dict) else None
  kind = ctx.get("gateProbe") if isinstance(ctx, dict) else None
  return kind if kind in ("state", "refusal") else None


def _trim_gate_probes(rows: Any) -> list:
  """Bound the ``gate_probes`` bucket, preserving chronological order. Never raises.

  Refusal rows keep the newest ``MAX_GATE_PROBES_PER_GATE`` PER GATE (the per-family rule, for the
  same reason). State rows are transient — folded into ``gate_state_days`` once settled — so they get
  one backstop count cap that only bites if folding never runs. Rows of unknown kind are dropped.
  """
  kept = [r for r in (rows or []) if isinstance(r, dict) and _gate_probe_kind(r)]
  keep_ids: set[int] = set()
  buckets: Dict[str, list] = {}
  states: list = []
  for row in kept:
    if _gate_probe_kind(row) == "state":
      states.append(row)
    else:
      buckets.setdefault(str(row["entryContext"].get("gate") or "?"), []).append(row)
  for bucket in buckets.values():
    for row in bucket[-MAX_GATE_PROBES_PER_GATE:]:
      keep_ids.add(id(row))
  for row in states[-MAX_GATE_STATE_UNFOLDED:]:
    keep_ids.add(id(row))
  return [r for r in kept if id(r) in keep_ids]


def _decode_gate_day(cell: Any) -> Optional[Dict[str, Any]]:
  """One stored day cell -> ``{side: {horizon: [n, s, {gate: [n, s]}]}}``, or None when unreadable."""
  try:
    val = json.loads(cell) if isinstance(cell, str) else cell
  except (TypeError, ValueError):
    return None
  return val if isinstance(val, dict) else None


def _encode_gate_day(cell: Dict[str, Any]) -> str:
  return json.dumps(cell, separators=(",", ":"), sort_keys=True)


def _trim_gate_state_days(days: Any) -> Dict[str, Any]:
  """The newest ``MAX_GATE_STATE_DAYS`` readable day cells (keys: UTC day numbers as strings)."""
  if not isinstance(days, dict):
    return {}
  valid = []
  for key, cell in days.items():
    try:
      day = int(key)
    except (TypeError, ValueError):
      continue
    if _decode_gate_day(cell) is not None:
      valid.append((day, str(key), cell if isinstance(cell, str) else _encode_gate_day(cell)))
  valid.sort()
  return {key: cell for _, key, cell in valid[-MAX_GATE_STATE_DAYS:]}


def _gate_regime_stamp(regime: Any) -> Optional[Dict[str, Any]]:
  """The gate-state fields a gate probe keeps: biases and flags only (whitelisted, no prices)."""
  raw = regime if isinstance(regime, dict) else None
  if not raw:
    return None
  out: Dict[str, Any] = {}
  for key in ("daily_bias", "daily_bias_raw", "intraday_bias_4h", "intraday_bias_1h", "intraday_bias_15m"):
    val = str(raw.get(key) or "").strip().lower()
    if val in ("bullish", "bearish", "neutral"):
      out[key] = val
  for key in ("daily_exhausted", "timeframe_conflict"):
    if key in raw:
      out[key] = bool(raw.get(key))
  return out or None


# Keys an exchange position may carry its opening time under, in the order ProtectionManager's own
# `_position_signature` reads them. The restart peak must be keyed on the SAME identity the in-process
# manager resets on, so this order is shared rather than re-derived (tests assert they agree).
POSITION_OPEN_TIME_KEYS = ("openingTimestamp", "openingTime", "openTime", "createdAt")


def position_open_time(pos: Any) -> Any:
  """The raw opening time of a live exchange position (first present key), or None."""
  if not isinstance(pos, dict):
    return None
  for key in POSITION_OPEN_TIME_KEYS:
    value = pos.get(key)
    if value not in (None, ""):
      return value
  return None


def peak_fe_key(open_time: Any, qty: Any, avg_entry: Any) -> Optional[str]:
  """Lifecycle identity of a persisted price-space peak: (openTime, side, |qty|, avgEntry).

  This is exactly the identity ``ProtectionManager._position_signature`` resets its in-process peak
  on: a same-side close/reopen, an add-on, a partial reduction or a flip each start a new excursion
  baseline there, because keeping a prior peak against a changed size or average entry "can
  immediately produce a false give-back". The persisted peak must reset on the same events, or a
  restart would hand back a peak the in-process manager had deliberately thrown away.

  Why a separate key from ``lifecycleKey`` (openTime:side): that one deliberately SURVIVES add-ons and
  partial reductions, because close records and MFE stats want the whole lifecycle's USD peak. The
  peak the trail uses must not. Side comes from the SIGN of qty, never from KuCoin's ``positionSide``
  (BOTH in one-way mode, LONG/SHORT in hedge mode) — reading that raw field is why the Sep 22 restart
  seed never matched a single stored key. Returns a string (JSON-stable), or None when unkeyable.
  """
  try:
    q = float(qty)
    a = float(avg_entry)
  except (TypeError, ValueError):
    return None
  if not (math.isfinite(q) and math.isfinite(a)) or q == 0 or a <= 0:
    return None
  opened: Any = None
  if open_time not in (None, ""):
    try:
      opened = int(float(open_time))
    except (TypeError, ValueError):
      opened = str(open_time)
  side = "long" if q > 0 else "short"
  return f"{opened}|{side}|{abs(q)!r}|{a!r}"


class MemoryStore:
  """Lightweight JSON-backed store for agent plans/notes."""

  _path_locks_guard = threading.Lock()
  _path_locks: Dict[str, threading.Lock] = {}

  def __init__(self, path: str, retention_days: int = 7) -> None:
    self.path = Path(path)
    lock_key = str(self.path.expanduser().resolve())
    with self._path_locks_guard:
      self._lock = self._path_locks.setdefault(lock_key, threading.Lock())
    self.retention_days = retention_days
    self._cache: Dict[str, Any] | None = None
    self._cache_mtime: float | None = None
    # Sanitize on init.
    self._read()

  def _read(self) -> Dict[str, Any]:
    _empty: Dict[str, Any] = {"plans": [], "triggers": [], "coins": [], "trades": [], "limits": {}, "sentiments": [], "decisions": [], "fees": [], "supervisor_notes_temporary": [], "supervisor_notes_permanent": [], "position_extremes": {}, "seen_close_ids": [], "seen_fill_ids": [], "open_interest_observations": {}, "pending_agent_events": [], "signal_probes": [], "agent_scheduler": _sanitize_agent_scheduler({})}
    if self._cache is not None:
      try:
        disk_mtime = self.path.stat().st_mtime
        if disk_mtime != self._cache_mtime:
          self._cache = None
      except OSError:
        pass
    if self._cache is not None:
      return copy.deepcopy(self._cache)
    if not self.path.exists():
      return _empty
    try:
      raw = json.loads(self.path.read_text())
      data = copy.deepcopy(raw)
      if isinstance(data, dict):
        data.setdefault("plans", [])
        data.setdefault("triggers", [])
        data.setdefault("coins", [])
        data.setdefault("trades", [])
        data.setdefault("limits", {})
        data.setdefault("sentiments", [])
        data.setdefault("decisions", [])
        data.setdefault("fees", [])
        data.setdefault("supervisor_notes_temporary", [])
        data.setdefault("supervisor_notes_permanent", [])
        data.setdefault("signal_probes", [])
        data.setdefault("position_extremes", {})
        data.setdefault("seen_close_ids", [])
        data.setdefault("seen_fill_ids", [])
        data.setdefault("open_interest_observations", {})
        data.setdefault("pending_agent_events", [])
        data["agent_scheduler"] = _sanitize_agent_scheduler(data.get("agent_scheduler"))
        # prune invalid entries while keeping timestamp
        data["plans"] = [
          p
          for p in data.get("plans", [])
          if isinstance(p, dict) and p.get("title") and p.get("summary") and isinstance(p.get("actions"), list)
        ]
        data["triggers"] = [
          {
            "triggerId": t.get("triggerId"),
            "symbol": _normalize_symbol(t.get("symbol", "")),
            "direction": t.get("direction"),
            "rationale": t.get("rationale"),
            "targetPrice": t.get("targetPrice"),
            "stopPrice": t.get("stopPrice"),
            "condition": t.get("condition"),
            "expiresAt": t.get("expiresAt"),
            "ts": t.get("ts"),
          }
          for t in data.get("triggers", [])
          if isinstance(t, dict) and t.get("symbol") and t.get("direction")
        ]
        data["coins"] = [
          {
            "symbol": _normalize_symbol(c.get("symbol", "")),
            "status": c.get("status", "active"),
            "reason": c.get("reason"),
            "exitPlan": c.get("exitPlan"),
            "ts": c.get("ts"),
          }
          for c in data.get("coins", [])
          if isinstance(c, dict) and c.get("symbol")
        ]
        data["trades"] = [
          {
            "symbol": _normalize_symbol(t.get("symbol", "")),
            "side": t.get("side"),
            "notionalUsd": t.get("notionalUsd"),
            "price": t.get("price"),
            "size": t.get("size"),
            "paper": t.get("paper", False),
            "venue": t.get("venue", "spot"),
            "filled": t.get("filled", True),
            "trackPosition": t.get("trackPosition", True),
            "orderId": t.get("orderId"),
            "clientOid": t.get("clientOid"),
            "fillTs": t.get("fillTs"),
            "fillPrice": t.get("fillPrice"),
            "fillSize": t.get("fillSize"),
            "entryContext": t.get("entryContext") if isinstance(t.get("entryContext"), dict) else None,
            "ts": t.get("ts") or int(time.time()),
            "day": t.get("day"),
          }
          for t in data.get("trades", [])
          if isinstance(t, dict) and t.get("symbol") and t.get("ts")
        ]
        data["pending_agent_events"] = [
          {
            "id": str(event.get("id") or ""),
            "kind": str(event.get("kind") or ""),
            "payload": event.get("payload"),
            "ts": int(event.get("ts") or time.time()),
          }
          for event in data.get("pending_agent_events", [])
          if isinstance(event, dict)
          and event.get("id")
          and event.get("kind") in {"spot_fills", "futures_fills", "closed_positions", "auto_triggers", "entry_expired"}
          and isinstance(event.get("payload"), dict)
        ]
        data["sentiments"] = [
          {
            "symbol": _normalize_symbol(s.get("symbol", "")),
            "score": s.get("score"),
            "rationale": s.get("rationale", ""),
            "source": s.get("source", ""),
            "ts": s.get("ts") or int(time.time()),
            "day": s.get("day"),
          }
          for s in data.get("sentiments", [])
          if isinstance(s, dict) and s.get("symbol") and s.get("ts") is not None
        ]
        data["decisions"] = [
          {
            "symbol": _normalize_symbol(d.get("symbol", "")),
            "action": d.get("action"),
            "confidence": d.get("confidence"),
            "reason": d.get("reason", ""),
            "pnl": d.get("pnl"),
            "paper": d.get("paper", False),
            "ts": d.get("ts") or int(time.time()),
            "day": d.get("day"),
            "peakPnl": d.get("peakPnl"),
            "troughPnl": d.get("troughPnl"),
            "entryPrice": d.get("entryPrice"),
            "entryContext": d.get("entryContext") if isinstance(d.get("entryContext"), dict) else None,
            "realizedR": d.get("realizedR"),
            # These fields power the no-chase guard and close deduplication.  Dropping them while
            # sanitizing on startup silently disabled both protections after every process restart.
            "exitPrice": d.get("exitPrice"),
            "closeType": d.get("closeType"),
            "positionId": d.get("positionId"),
            "positionOpenTime": d.get("positionOpenTime"),
            "positionSide": d.get("positionSide"),
            "positionLifecycleVersion": d.get("positionLifecycleVersion"),
          }
          for d in data.get("decisions", [])
          if isinstance(d, dict) and d.get("symbol") and d.get("ts") is not None
        ]
        limits = data.get("limits") or {}
        # Normalize legacy limits (single dict) into per-scope dict keyed by "total".
        if isinstance(limits, dict) and any(
          k in limits for k in ("baselineUsdt", "currentUsdt", "drawdownPct", "kill")
        ) and "total" not in limits:
          limits = {
            "total": {
              "day": limits.get("day"),
              "baselineUsdt": limits.get("baselineUsdt"),
              "currentUsdt": limits.get("currentUsdt"),
              "drawdownPct": limits.get("drawdownPct"),
              "kill": limits.get("kill", False),
              "reason": limits.get("reason", ""),
              "updated": limits.get("updated"),
            }
          }
        if not isinstance(limits, dict):
          limits = {}
        data["limits"] = limits
        data = self._prune(data)
        if data != raw:
          self._write(data)
        else:
          self._cache = copy.deepcopy(data)
          try:
            self._cache_mtime = self.path.stat().st_mtime
          except OSError:
            self._cache_mtime = None
        return data
      return _empty
    except Exception:
      return _empty

  def _prune(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Drop entries older than retention_days."""
    now = int(time.time())
    cutoff = now - self.retention_days * 86400
    data["plans"] = [p for p in data.get("plans", []) if (p.get("ts") or now) >= cutoff]
    data["triggers"] = [t for t in data.get("triggers", []) if (t.get("ts") or now) >= cutoff]
    data["coins"] = [c for c in data.get("coins", []) if (c.get("ts") or now) >= cutoff]
    # Two kinds of order row are LEARNING DATA and are retained by COUNT (MAX_TRADES) rather than by
    # wall-clock age — see the closed-trade note below for why time-pruning evidence is harmful:
    #   * a real (planned price, achieved fill price) pair — the sample `edge.measured_slippage_pct`
    #     calibrates execution cost on (condition mirrors `recent_fills` exactly);
    #   * a `marketPriceAtSignal` stamp — the sample `edge.signal_edge_stats` measures directional edge
    #     on. Note this deliberately includes UNFILLED plans, and that is the whole point: 6 of the
    #     first 9 probes were unfilled, and those are the *unbiased* part of the sample. Filled orders
    #     are adverse-selected (a resting limit fills preferentially when the move goes against it), so
    #     pruning the unfilled ones would quietly bias signal edge toward exactly the contaminated
    #     subset the measurement exists to avoid.
    # Everything else stays ephemeral.
    data["trades"] = [
      t for t in data.get("trades", [])
      if (t.get("ts") or now) >= cutoff
      or (
        isinstance(t, dict) and (
          (t.get("filled") and t.get("fillPrice") is not None)
          or (isinstance(t.get("entryContext"), dict) and t["entryContext"].get("marketPriceAtSignal"))
        )
      )
    ]
    data.setdefault("limits", {})
    if isinstance(data["limits"], dict):
      for scope, lim in list(data["limits"].items()):
        if not isinstance(lim, dict):
          data["limits"].pop(scope, None)
          continue
        if (lim.get("updated") or now) < cutoff:
          data["limits"].pop(scope, None)
    data["sentiments"] = [s for s in data.get("sentiments", []) if (s.get("ts") or now) >= cutoff]
    # REALIZED CLOSES (pnl != None) are exempt from the time cutoff — they are retained purely by count
    # (MAX_CLOSED_TRADES, applied below). They are the training data for every adaptive guard:
    # edge_stats, entry_quality_stats, expectancy sizing, the symbol bench, measured slippage and the
    # adaptive stop floor all read them. Pruning them by wall-clock age created a doom loop that was
    # measured live (2026-08-06): as the trade rate fell, the 7-day window emptied until only 8 closes
    # remained, all recent losses. The controller then reported an 11% win rate and a 6-loss streak,
    # halved position size, and the agent stood aside in 356 of 358 runs — which produced no new closes,
    # so the window could only get staler and bleaker. Evidence must age out by being SUPERSEDED, never
    # by the clock, or a quiet spell is self-reinforcing. Declines/entries (pnl None) stay time-pruned.
    data["decisions"] = [
      d for d in data.get("decisions", [])
      if (d.get("ts") or now) >= cutoff or (isinstance(d, dict) and d.get("pnl") is not None)
    ]
    data["fees"] = [f for f in data.get("fees", []) if (f.get("ts") or now) >= cutoff]
    data["supervisor_notes_temporary"] = [n for n in data.get("supervisor_notes_temporary", []) if (n.get("ts") or now) >= cutoff]
    data["pending_agent_events"] = [
      event for event in data.get("pending_agent_events", [])
      if isinstance(event, dict) and (event.get("ts") or now) >= cutoff
    ][-MAX_PENDING_AGENT_EVENTS:]
    # Signal probes are learning data (every direction call, placed or not): retained by COUNT only,
    # never by clock — see record_signal_probe for why tying evidence to anything else deadlocks.
    data["signal_probes"] = _trim_probes_per_family(data.get("signal_probes") or [])
    # Gate probes (report-only, never part of signal_probes()): count-capped per gate, never by clock;
    # the folded state days by count of days. Only touched once a store has them, so an older file is
    # not rewritten just to add empty keys.
    if "gate_probes" in data:
      data["gate_probes"] = _trim_gate_probes(data.get("gate_probes"))
    if "gate_state_days" in data:
      data["gate_state_days"] = _trim_gate_state_days(data.get("gate_state_days"))
    # The macro calendar is forward-looking: drop anything more than a day past, cap the rest. A stale
    # or empty calendar must degrade to "no events known" (ordinary trading), never to a stuck blackout.
    data["macro_events"] = [
      e for e in (data.get("macro_events") or [])
      if isinstance(e, dict) and (e.get("ts") or 0) >= now - 86400
    ][-MAX_MACRO_EVENTS:]
    data["agent_closes"] = [m for m in (data.get("agent_closes") or []) if isinstance(m, dict)][-200:]
    # Data-quality failures expire on their own evidence-derived retryAfter.
    if "analysis_failures" in data:
      _af = data.get("analysis_failures")
      data["analysis_failures"] = {
        k: v for k, v in (_af.items() if isinstance(_af, dict) else [])
        if isinstance(v, dict) and (v.get("retryAfter") or 0) > now
      }
    # Exit probes are learning data too (was the discretionary close better than the bracket?):
    # count-capped, never clock-pruned, for the same reason signal probes are not.
    _xp = data.get("exit_probes") or []
    if len(_xp) > MAX_EXIT_PROBES:
      data["exit_probes"] = _xp[-MAX_EXIT_PROBES:]
    # Permanent notes are exempt from the retention-days cutoff by design — they persist
    # until manually deleted. Only the count cap (MAX_PERMANENT_NOTES) applies below.
    data.setdefault("supervisor_notes_permanent", [])

    def _cap_list(key: str, max_items: int) -> None:
      items = data.get(key) or []
      if len(items) > max_items:
        # keep most recent by ts if available, else last items
        try:
          items = sorted(items, key=lambda x: x.get("ts", 0))[-max_items:]
        except Exception:
          items = items[-max_items:]
      data[key] = items

    _cap_list("plans", MAX_PLANS)
    _cap_list("triggers", MAX_TRIGGERS)
    _cap_list("coins", MAX_COINS)
    _cap_list("trades", MAX_TRADES)
    _cap_list("sentiments", MAX_SENTIMENTS)
    # Two-tier cap: only actual close actions get the larger outcome bucket. The model often logs
    # unrealized PnL on hold/decline rows; treating those as closes crowded real outcomes out of
    # memory and made diagnostics misleading.
    def _is_handoff(d: Dict[str, Any]) -> bool:
      return str(d.get("action") or "").startswith("handoff")

    pnl_decisions = [
      d for d in (data.get("decisions") or [])
      if d.get("pnl") is not None and self._is_realized_close(str(d.get("action") or ""))
    ]
    null_all = [d for d in (data.get("decisions") or []) if d not in pnl_decisions]
    # Handoff markers get their own bucket so the high-volume decline/hold snapshots (capped at
    # MAX_DECISIONS) can't evict them — the dashboard surfaces handoffs from the recent feed.
    handoff_decisions = [d for d in null_all if _is_handoff(d)]
    null_decisions = [d for d in null_all if not _is_handoff(d)]
    if len(pnl_decisions) > MAX_CLOSED_TRADES:
      try:
        pnl_decisions = sorted(pnl_decisions, key=lambda x: x.get("ts", 0))[-MAX_CLOSED_TRADES:]
      except Exception:
        pnl_decisions = pnl_decisions[-MAX_CLOSED_TRADES:]
    if len(null_decisions) > MAX_DECISIONS:
      try:
        null_decisions = sorted(null_decisions, key=lambda x: x.get("ts", 0))[-MAX_DECISIONS:]
      except Exception:
        null_decisions = null_decisions[-MAX_DECISIONS:]
    if len(handoff_decisions) > MAX_HANDOFF_DECISIONS:
      try:
        handoff_decisions = sorted(handoff_decisions, key=lambda x: x.get("ts", 0))[-MAX_HANDOFF_DECISIONS:]
      except Exception:
        handoff_decisions = handoff_decisions[-MAX_HANDOFF_DECISIONS:]
    data["decisions"] = sorted(pnl_decisions + null_decisions + handoff_decisions, key=lambda x: x.get("ts", 0))
    _cap_list("fees", MAX_FEES)
    _cap_list("supervisor_notes_temporary", MAX_TEMPORARY_NOTES)
    _cap_list("supervisor_notes_permanent", MAX_PERMANENT_NOTES)

    return data

  def _write(self, data: Dict[str, Any]) -> None:
    payload = json.dumps(data, indent=2)
    temp_path = self.path.with_name(self.path.name + ".tmp")
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open("w", encoding="utf-8") as handle:
      handle.write(payload)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temp_path, self.path)
    self._cache = copy.deepcopy(data)
    try:
      self._cache_mtime = self.path.stat().st_mtime
    except OSError:
      self._cache_mtime = None

  def save_plan(self, title: str, summary: str, actions: list[str], author: str | None = None) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      entry = {
        "title": title,
        "summary": summary,
        "actions": actions,
        "author": author,
        "ts": int(time.time()),
      }
      data.setdefault("plans", [])
      data["plans"].append(entry)
      self._write(data)
      return entry

  def latest_plan(self) -> Optional[Dict[str, Any]]:
    with self._lock:
      data = self._prune(self._read())
      plans = data.get("plans") or []
      return plans[-1] if plans else None

  def latest_items(self, kind: str, limit: int = 5) -> Dict[str, Any]:
    """Fetch the latest N entries for a given kind (plans/sentiments/decisions/trades/triggers/coins/fees)."""
    alias_map = {
      "plan": "plans",
      "plans": "plans",
      "research": "plans",
      "note": "plans",
      "notes": "plans",
      "sentiment": "sentiments",
      "sentiments": "sentiments",
      "decision": "decisions",
      "decisions": "decisions",
      "trade": "trades",
      "trades": "trades",
      "trigger": "triggers",
      "triggers": "triggers",
      "coin": "coins",
      "coins": "coins",
      "fee": "fees",
      "fees": "fees",
    }
    key = alias_map.get((kind or "").strip().lower())
    if not key:
      return {"error": f"unsupported kind '{kind}'", "allowed": sorted(set(alias_map.keys()))}
    try:
      lim = int(limit)
    except (TypeError, ValueError):
      lim = 5
    lim = min(max(lim, 1), 50)

    with self._lock:
      data = self._prune(self._read())
      items = data.get(key) or []
      if not isinstance(items, list):
        return {"error": f"kind '{kind}' unavailable"}
      latest = sorted(items, key=lambda x: x.get("ts", 0))[-lim:]
    return {"kind": key, "requested": lim, "items": list(reversed(latest))}

  def clear_plans(self) -> Dict[str, Any]:
    with self._lock:
      data = self._read()
      data["plans"] = []
      data["triggers"] = []
      self._write(self._prune(data))
      return self._prune(data)

  def save_trigger(
    self,
    symbol: str,
    direction: str,
    rationale: str,
    target_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    condition: Optional[str] = None,
    expires_minutes: float = 360.0,
  ) -> Dict[str, Any]:
    with self._lock:
      data = self._read()
      now = int(time.time())
      condition_norm = str(condition or "").strip().lower() or None
      if condition_norm not in {None, "above", "below"}:
        raise ValueError("condition must be 'above' or 'below'")
      entry = {
        "triggerId": uuid.uuid4().hex,
        "symbol": _normalize_symbol(symbol),
        "direction": direction,
        "rationale": rationale,
        "targetPrice": target_price,
        "stopPrice": stop_price,
        "condition": condition_norm,
        "expiresAt": now + int(max(1.0, float(expires_minutes)) * 60),
        "ts": now,
      }
      data.setdefault("triggers", [])
      data["triggers"].append(entry)
      self._write(data)
      return entry

  def latest_triggers(self) -> list[Dict[str, Any]]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      current = []
      for trigger in data.get("triggers", []) or []:
        expires = trigger.get("expiresAt")
        if expires is None:
          expires = int(trigger.get("ts") or 0) + 24 * 3600  # bounded legacy migration
        try:
          if int(expires) > now:
            current.append(trigger)
        except (TypeError, ValueError):
          continue
      if len(current) != len(data.get("triggers", []) or []):
        data["triggers"] = current
        self._write(data)
      return current

  def consume_trigger(self, trigger: Dict[str, Any]) -> bool:
    """Remove one fired trigger; a queued agent event preserves delivery across restarts."""
    if not isinstance(trigger, dict):
      return False
    trigger_id = str(trigger.get("triggerId") or "").strip()

    def _matches(candidate: Dict[str, Any]) -> bool:
      if trigger_id:
        return str(candidate.get("triggerId") or "").strip() == trigger_id
      return all(
        candidate.get(key) == trigger.get(key)
        for key in ("symbol", "ts", "condition", "targetPrice")
      )

    with self._lock:
      data = self._read()
      existing = data.get("triggers", []) or []
      remaining = [
        item for item in existing
        if not (isinstance(item, dict) and _matches(item))
      ]
      if len(remaining) == len(existing):
        return False
      data["triggers"] = remaining
      self._write(self._prune(data))
      return True

  def get_coins(self, default: list[str] | None = None) -> list[str]:
    with self._lock:
      data = self._prune(self._read())
      coins = data.get("coins", []) or []
      if coins:
        return [c["symbol"] for c in coins if c.get("status", "active") == "active"]
      return default or []

  def get_quarantined_coins(self, now: int | None = None) -> list[Dict[str, Any]]:
    """Return automatic risk quarantines whose adaptive retry window has not expired."""
    current = int(time.time() if now is None else now)
    with self._lock:
      data = self._prune(self._read())
      quarantined: list[Dict[str, Any]] = []
      for coin in data.get("coins", []) or []:
        if coin.get("status") != "removed":
          continue
        reason = str(coin.get("reason") or "")
        duration = _adaptive_quarantine_seconds(reason)
        if duration <= 0:
          continue
        removed_at = int(coin.get("ts") or 0)
        retry_after = removed_at + duration
        if retry_after <= current:
          continue
        quarantined.append({
          "symbol": coin.get("symbol"),
          "reason": reason,
          "retryAfter": retry_after,
          "remainingHours": round((retry_after - current) / 3600.0, 1),
        })
      return sorted(quarantined, key=lambda item: item["retryAfter"])

  def record_analysis_failure(self, symbol: str, *, reason: str, retry_after: Any, now: Any = None) -> None:
    """Remember that analyze_market_context refused ``symbol`` on data quality, until ``retry_after``.

    Kept in its own key, NOT the coins list: that list is capped at 50 and the model's own remove_coin
    overwrites an entry, which is how an ATR quarantine could be erased. One row per symbol (the latest
    refusal); ``retry_after`` comes from the failure's own evidence (analytics.candle_quality_retry_after),
    never a hand-set TTL. Informational only — nothing is blocked by it; the scan and list_coins show it.
    """
    sym = _normalize_symbol(symbol)
    try:
      retry = float(retry_after)
    except (TypeError, ValueError):
      return
    current = int(time.time() if now is None else float(now))
    if not sym or not math.isfinite(retry) or retry <= current:
      return
    with self._lock:
      data = self._prune(self._read())
      failures = data.get("analysis_failures") if isinstance(data.get("analysis_failures"), dict) else {}
      failures[sym] = {"reason": str(reason or "")[:300], "ts": current, "retryAfter": int(math.ceil(retry))}
      if len(failures) > MAX_ANALYSIS_FAILURES:
        keep = sorted(failures.items(), key=lambda kv: kv[1].get("ts") or 0)[-MAX_ANALYSIS_FAILURES:]
        failures = dict(keep)
      data["analysis_failures"] = failures
      self._write(data)

  def clear_analysis_failure(self, symbol: str) -> bool:
    """Drop ``symbol``'s recorded data-quality failure (a later analysis came back clean)."""
    sym = _normalize_symbol(symbol)
    with self._lock:
      data = self._read()
      failures = data.get("analysis_failures")
      if not isinstance(failures, dict) or sym not in failures:
        return False
      failures.pop(sym, None)
      data["analysis_failures"] = failures
      self._write(data)
      return True

  def analysis_failures(self, now: Any = None) -> Dict[str, Dict[str, Any]]:
    """Active data-quality failures {symbol: {reason, ts, retryAfter, remainingHours}}; expired ones omitted."""
    current = float(time.time() if now is None else now)
    with self._lock:
      data = self._read()
    out: Dict[str, Dict[str, Any]] = {}
    failures = data.get("analysis_failures")
    for sym, row in (failures.items() if isinstance(failures, dict) else []):
      if not isinstance(row, dict):
        continue
      try:
        retry = float(row.get("retryAfter"))
      except (TypeError, ValueError):
        continue
      if retry <= current:
        continue
      out[sym] = {
        "reason": row.get("reason"),
        "ts": row.get("ts"),
        "retryAfter": int(retry),
        "remainingHours": round((retry - current) / 3600.0, 1),
      }
    return out

  def has_coins(self) -> bool:
    with self._lock:
      data = self._prune(self._read())
      coins = data.get("coins", []) or []
      return bool(coins)

  def set_coins(self, coins: list[str], reason: str = "update") -> list[Dict[str, Any]]:
    with self._lock:
      entries = []
      now = int(time.time())
      for sym in coins:
        entries.append({"symbol": _normalize_symbol(sym), "status": "active", "reason": reason, "ts": now})
      data = self._read()
      data["coins"] = entries
      self._write(data)
      return entries

  def add_coin(self, symbol: str, reason: str) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      coins = data.get("coins", [])
      now = int(time.time())
      symbol_up = _normalize_symbol(symbol)
      # avoid duplicates; replace existing with latest reason
      coins = [c for c in coins if c.get("symbol") != symbol_up]
      coins.append({"symbol": symbol_up, "status": "active", "reason": reason, "ts": now})
      data["coins"] = coins
      self._write(data)
      return {"symbol": symbol_up, "status": "active", "reason": reason, "ts": now}

  def remove_coin(self, symbol: str, reason: str, exit_plan: str) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      coins = data.get("coins", [])
      symbol_up = _normalize_symbol(symbol)
      now = int(time.time())
      coins = [c for c in coins if c.get("symbol") != symbol_up]
      entry = {
        "symbol": symbol_up,
        "status": "removed",
        "reason": reason,
        "exitPlan": exit_plan,
        "ts": now,
      }
      coins.append(entry)
      data["coins"] = coins
      self._write(data)
      return entry

  def trades_today(self, symbol: str) -> int:
    with self._lock:
      data = self._prune(self._read())
      day_key = int(time.time() // 86400)
      return len(
        [
          t
          for t in data.get("trades", []) or []
          if t.get("symbol") == _normalize_symbol(symbol)
          and (t.get("day") or 0) == day_key
          and t.get("filled") is not False
        ]
      )

  def record_trade(
    self,
    symbol: str,
    side: str,
    notional_usd: float,
    paper: bool = False,
    price: float | None = None,
    size: float | None = None,
    venue: str = "spot",
    filled: bool = True,
    track_position: bool = True,
    order_id: str | None = None,
    client_oid: str | None = None,
    entry_context: Dict[str, Any] | None = None,
  ) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      day_key = int(now // 86400)
      entry = {
        "symbol": _normalize_symbol(symbol),
        "side": side,
        "notionalUsd": notional_usd,
        "price": price,
        "size": size,
        "paper": paper,
        "venue": venue,
        "filled": bool(filled),
        "trackPosition": bool(track_position),
        "orderId": str(order_id) if order_id not in (None, "") else None,
        "clientOid": str(client_oid) if client_oid not in (None, "") else None,
        "entryContext": copy.deepcopy(entry_context) if isinstance(entry_context, dict) else None,
        "ts": now,
        "day": day_key,
      }
      data.setdefault("trades", [])
      data["trades"].append(entry)
      self._write(data)
      return entry

  def log_sentiment(self, symbol: str, score: float, rationale: str, source: str = "") -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      day_key = int(now // 86400)
      entry = {
        "symbol": _normalize_symbol(symbol),
        "score": float(score),
        "rationale": rationale,
        "source": source,
        "ts": now,
        "day": day_key,
      }
      data.setdefault("sentiments", [])
      data["sentiments"].append(entry)
      self._write(data)
      return entry

  def latest_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
    with self._lock:
      data = self._prune(self._read())
      sym = _normalize_symbol(symbol)
      sentiments = [s for s in data.get("sentiments", []) if s.get("symbol") == sym]
      return sentiments[-1] if sentiments else None

  def log_decision(
    self,
    symbol: str,
    action: str,
    confidence: float,
    reason: str,
    pnl: Optional[float] = None,
    paper: bool = False,
    peak_pnl: Optional[float] = None,
    trough_pnl: Optional[float] = None,
    exit_price: Optional[float] = None,
    close_type: Optional[str] = None,
    position_id: Optional[str] = None,
    position_open_time: Optional[int | float | str] = None,
    position_side: Optional[str] = None,
    entry_price: Optional[float] = None,
    entry_context: Optional[Dict[str, Any]] = None,
  ) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      day_key = int(now // 86400)
      sym = _normalize_symbol(symbol)
      if pnl is not None and (peak_pnl is None or trough_pnl is None):
        ext = data.get("position_extremes", {}).get(sym, {})
        if ext:
          if peak_pnl is None:
            peak_pnl = ext.get("peakPnl")
          if trough_pnl is None:
            trough_pnl = ext.get("troughPnl")
      entry: Dict[str, Any] = {
        "symbol": sym,
        "action": action,
        "confidence": float(confidence),
        "reason": reason,
        "pnl": pnl,
        "paper": paper,
        "ts": now,
        "day": day_key,
      }
      if peak_pnl is not None:
        entry["peakPnl"] = round(float(peak_pnl), 4)
      if trough_pnl is not None:
        entry["troughPnl"] = round(float(trough_pnl), 4)
      if exit_price is not None:
        try:
          entry["exitPrice"] = float(exit_price)
        except (TypeError, ValueError):
          pass
      if entry_price is not None:
        try:
          entry["entryPrice"] = float(entry_price)
        except (TypeError, ValueError):
          pass
      if isinstance(entry_context, dict):
        entry["entryContext"] = copy.deepcopy(entry_context)
        try:
          planned_risk = float(entry_context.get("plannedMaxLossUsd") or 0.0)
          if pnl is not None and planned_risk > 0:
            entry["realizedR"] = round(float(pnl) / planned_risk, 6)
        except (TypeError, ValueError):
          pass
      if close_type:
        entry["closeType"] = str(close_type)
      if position_id not in (None, ""):
        entry["positionId"] = str(position_id)
      if position_open_time not in (None, ""):
        try:
          entry["positionOpenTime"] = int(float(position_open_time))
        except (TypeError, ValueError):
          pass
      if position_side and str(position_side).lower() in {"long", "short"}:
        entry["positionSide"] = str(position_side).lower()
      if any(value not in (None, "") for value in (position_id, position_open_time, position_side)):
        # Marks rows written by lifecycle-aware code. If an exchange omits the identifiers, fail
        # safe by keeping both PnL rows instead of falling back to an unsafe symbol/time merge.
        entry["positionLifecycleVersion"] = 1
      data.setdefault("decisions", [])
      data["decisions"].append(entry)
      self._write(data)
      return entry

  def trades_today_total(self) -> int:
    with self._lock:
      data = self._prune(self._read())
      day_key = int(time.time() // 86400)
      return len([t for t in data.get("trades", []) or [] if (t.get("day") or 0) == day_key])

  def latest_limits(self, scope: str = "total") -> Dict[str, Any]:
    """The last recorded limits for a scope, without writing anything.

    Used when a balance snapshot comes back incomplete: a venue that failed to read is UNKNOWN, not
    zero, so the poll must reuse the last known-good figures rather than record a partial account.
    """
    with self._lock:
      data = self._read()
    limits_all = data.get("limits")
    if not isinstance(limits_all, dict):
      return {}
    row = limits_all.get(str(scope or "total"))
    return copy.deepcopy(row) if isinstance(row, dict) else {}

  def update_limits(self, current_usdt: float, scope: str = "total") -> Dict[str, Any]:
    """Track daily drawdown percentage for informational context. No kill switch."""
    with self._lock:
      data = self._read()
      now = int(time.time())
      day_key = int(now // 86400)
      limits_all = data.get("limits") if isinstance(data.get("limits"), dict) else {}
      limits = limits_all.get(scope) or {}
      if limits.get("day") != day_key:
        # Rolling into a new day anchors the baseline to whatever equity reading arrives first, and a
        # PARTIAL snapshot (spot only, futures API timed out — this account's log is full of KuCoin
        # 504s) would anchor the day to near-zero. Every later reading then reports a five-figure
        # "return", which the dashboard compounds into its durable index and can never undo: the live
        # series reached 725,468x its base exactly this way. A real overnight move cannot be a
        # tenfold change in either direction, so treat one as a bad read and keep the prior baseline.
        prev_baseline = float(limits.get("baselineUsdt") or 0.0)
        opening = float(current_usdt or 0.0)
        implausible = (
          prev_baseline > 0 and opening > 0
          and (opening < prev_baseline / 10.0 or opening > prev_baseline * 10.0)
        )
        if implausible:
          logger.warning(
            "DAILY BASELINE: opening equity %.4g is >10x away from yesterday's %.4g — treating as a "
            "partial/failed balance snapshot and keeping the previous baseline for day %s.",
            opening, prev_baseline, day_key,
          )
        limits = {
          "day": day_key,
          "baselineUsdt": prev_baseline if implausible else opening,
          "currentUsdt": opening,
          "drawdownPct": 0.0,
          "updated": now,
        }
      baseline = limits.get("baselineUsdt") or float(current_usdt or 0.0)
      if baseline <= 0:
        baseline = float(current_usdt or 0.0)
      drawdown_pct = 0.0
      if baseline > 0:
        drawdown_pct = max(0.0, (baseline - float(current_usdt or 0.0)) / baseline * 100)
      limits.update(
        {
          "day": day_key,
          "baselineUsdt": baseline,
          "currentUsdt": float(current_usdt or 0.0),
          "drawdownPct": drawdown_pct,
          "updated": now,
        }
      )
      limits_all[scope] = limits
      data["limits"] = limits_all
      self._write(data)
      return limits

  def positions(self, prices: Dict[str, float] | None = None, venue: str | None = None) -> Dict[str, Any]:
    """Derive positions (avg entry, unrealized/realized PnL) from recorded trades. Filter by venue ('spot'/'futures') when set."""
    with self._lock:
      data = self._prune(self._read())
      trades = sorted(data.get("trades", []), key=lambda t: t.get("ts", 0))
    if venue:
      trades = [t for t in trades if t.get("venue", "spot") == venue]

    positions: Dict[str, Dict[str, float]] = {}

    def _avg(cost: float, qty: float) -> float:
      return cost / qty if qty else 0.0

    for t in trades:
      if t.get("filled") is False or t.get("trackPosition") is False:
        continue
      sym = t.get("symbol")
      side = (t.get("side") or "").lower()
      try:
        qty = float(t.get("size") or 0)
        price = float(t.get("price") or 0)
      except Exception:
        continue
      if not sym or qty <= 0 or price <= 0 or side not in {"buy", "sell"}:
        continue
      pos = positions.setdefault(sym, {"netSize": 0.0, "cost": 0.0, "realizedPnl": 0.0, "venue": t.get("venue", "spot"), "lastTs": t.get("ts", 0)})
      net = pos["netSize"]
      cost = pos["cost"]
      realized = pos["realizedPnl"]
      ts = t.get("ts", 0)
      if side == "buy":
        if net < 0:
          close_amt = min(qty, -net)
          avg_entry = _avg(cost, net) if net != 0 else 0.0
          realized += (avg_entry - price) * close_amt
          net += close_amt
          cost += avg_entry * close_amt
          qty -= close_amt
        if qty > 0:
          net += qty
          cost += price * qty
      else:  # sell
        if net > 0:
          close_amt = min(qty, net)
          avg_entry = _avg(cost, net) if net != 0 else 0.0
          realized += (price - avg_entry) * close_amt
          net -= close_amt
          cost -= avg_entry * close_amt
          qty -= close_amt
        if qty > 0:
          net -= qty
          cost -= price * qty
      pos["netSize"] = net
      pos["cost"] = cost
      pos["realizedPnl"] = realized
      pos["lastTs"] = max(pos.get("lastTs", 0), ts)

    # Attach derived values, unrealized PnL, and peak/trough extremes when prices are supplied.
    prices = prices or {}
    with self._lock:
      extremes = self._read().get("position_extremes", {})
    for sym, pos in positions.items():
      net = pos.get("netSize", 0.0)
      cost = pos.get("cost", 0.0)
      cur_price = float(prices.get(sym) or 0.0)
      avg_entry = _avg(cost, net) if net else None
      unrealized = None
      if net and cur_price > 0:
        unrealized = (cur_price - avg_entry) * net
      pos["avgEntry"] = avg_entry
      pos["unrealizedPnl"] = unrealized
      pos["currentPrice"] = cur_price or None
      ext = extremes.get(sym, {})
      if ext:
        pos["peakPnl"] = ext.get("peakPnl")
        pos["troughPnl"] = ext.get("troughPnl")
    return positions

  def update_position_extremes(self, positions: Dict[str, Dict[str, Any]]) -> None:
    """Update peak/trough unrealized PnL for open positions. Call each poll round.

    Two peaks are kept, on purpose on two different identities:

    * ``peakPnl`` / ``troughPnl`` (USD) on ``lifecycleKey`` = openTime:side — the whole lifecycle's
      extremes, which close records and MFE stats read. Unchanged.
    * ``peakFePx`` (PRICE units: max over polls of (mark - avgEntry) x +1 long / -1 short) on
      ``peakFeKey`` = :func:`peak_fe_key` — the peak the trail uses, reset on every event that resets
      ProtectionManager's own peak (add-on, partial reduction, flip), so a restart can hand the
      in-process manager back exactly what it would have held. Stored in price because the USD peak
      divided by contracts is price x contract MULTIPLIER (10x on H/WIF/ONDO, 520,000x on PEPE), which
      the Sep 22 seed did and which would have market-closed multiplier>1 winners had its key ever
      matched. Needs ``markPrice``, ``avgEntryPrice`` and ``lifecycleOpenTime`` on the row
      (``main._live_extremes_map``); rows without them keep the USD fields only.
    """
    with self._lock:
      data = self._read()
      extremes = data.get("position_extremes", {})
      now = int(time.time())
      active_symbols: set[str] = set()
      for sym, pos in positions.items():
        net = pos.get("netSize") or 0
        upnl = pos.get("unrealizedPnl")
        if not net or upnl is None:
          continue
        active_symbols.add(sym)
        ext = extremes.get(sym)
        lifecycle_key = f"{pos.get('positionOpenTime') or ''}:{pos.get('positionSide') or ('long' if net > 0 else 'short')}"
        if ext is None or ext.get("lifecycleKey") != lifecycle_key:
          ext = {
            "peakPnl": upnl, "troughPnl": upnl, "peakTs": now, "troughTs": now,
            "openTs": now, "lifecycleKey": lifecycle_key,
            "positionOpenTime": pos.get("positionOpenTime"),
            "positionSide": pos.get("positionSide") or ("long" if net > 0 else "short"),
          }
          extremes[sym] = ext
        else:
          if upnl > (ext.get("peakPnl") or float("-inf")):
            ext["peakPnl"] = upnl
            ext["peakTs"] = now
          if upnl < (ext.get("troughPnl") or float("inf")):
            ext["troughPnl"] = upnl
            ext["troughTs"] = now
        self._update_peak_fe(ext, pos, net, now)
      for sym in list(extremes.keys()):
        if sym not in active_symbols:
          del extremes[sym]
      data["position_extremes"] = extremes
      self._write(data)

  @staticmethod
  def _update_peak_fe(ext: Dict[str, Any], pos: Dict[str, Any], net: Any, now: int) -> None:
    """Advance (or reset) the price-space peak on ``ext`` in place. Never raises."""
    try:
      key = peak_fe_key(pos.get("lifecycleOpenTime"), net, pos.get("avgEntryPrice"))
      mark = float(pos.get("markPrice"))
      avg = float(pos.get("avgEntryPrice"))
    except (TypeError, ValueError):
      return
    if key is None or not math.isfinite(mark) or mark <= 0:
      return
    fe = (mark - avg) if float(net) > 0 else (avg - mark)
    if ext.get("peakFeKey") != key:
      ext["peakFeKey"] = key
      ext["peakFePx"] = fe
      ext["peakFeTs"] = now
      return
    try:
      prev = float(ext.get("peakFePx"))
    except (TypeError, ValueError):
      prev = float("-inf")
    if not math.isfinite(prev) or fe > prev:
      ext["peakFePx"] = fe
      ext["peakFeTs"] = now

  def get_position_extremes(self, symbol: str | None = None) -> Dict[str, Any]:
    """Get peak/trough PnL extremes for open positions (or a specific symbol)."""
    with self._lock:
      data = self._read()
      extremes = data.get("position_extremes", {})
    if symbol:
      return extremes.get(_normalize_symbol(symbol), {})
    return extremes

  @staticmethod
  def _dedupe_realized(closes: list[Dict[str, Any]], window_sec: int = 1800) -> list[Dict[str, Any]]:
    """Drop re-recorded duplicates, preferring lifecycle identity when it is available.

    The main loop's seen-ID set used to live only in process memory, so a restart within the
    30-min fill lookback re-recorded the same close (observed: ETH -0.5007 at 13:48 and again
    at 14:00 across a restart). IDs are persisted now; this read-time filter also heals data
    recorded before the fix so rolling stats aren't skewed by phantom losses.
    """
    kept: list[Dict[str, Any]] = []
    last_seen: Dict[tuple, int] = {}
    for c in sorted(closes, key=lambda d: d.get("ts") or 0):
      pnl = c.get("pnl")
      position_id = str(c.get("positionId") or "").strip()
      position_open_time = MemoryStore._position_open_time_ms(c)
      if position_id:
        key = (c.get("symbol"), "position-id", position_id)
      elif position_open_time is not None:
        key = (c.get("symbol"), "position-open", position_open_time)
      elif c.get("positionLifecycleVersion"):
        kept.append(c)
        continue
      else:
        try:
          key = (c.get("symbol"), "legacy", c.get("closeType") or "", round(float(pnl), 6))
        except (TypeError, ValueError):
          kept.append(c)
          continue
      ts = int(c.get("ts") or 0)
      prev = last_seen.get(key)
      if prev is not None and 0 <= ts - prev <= window_sec:
        continue
      last_seen[key] = ts
      kept.append(c)
    return kept

  @staticmethod
  def _position_open_time_ms(row: Dict[str, Any]) -> Optional[int]:
    value = row.get("positionOpenTime")
    if value in (None, ""):
      return None
    try:
      timestamp = int(float(value))
    except (TypeError, ValueError):
      return None
    return timestamp if timestamp > 1_000_000_000_000 else timestamp * 1000

  @staticmethod
  def _position_side(row: Dict[str, Any]) -> Optional[str]:
    explicit = str(row.get("positionSide") or "").lower()
    if explicit in {"long", "short"}:
      return explicit
    close_type = str(row.get("closeType") or "").upper()
    if "LONG" in close_type:
      return "long"
    if "SHORT" in close_type:
      return "short"
    action = str(row.get("action") or "").lower()
    if action.startswith("futures_sell"):
      return "long"
    if action.startswith("futures_buy"):
      return "short"
    return None

  @staticmethod
  def _has_position_identity(row: Dict[str, Any]) -> bool:
    """Whether a row names the position it closed, by any of the identifiers we persist.

    Used to tell a real close record from a narrative echo. A row that names no position cannot be
    reconciled with the exchange's report of the same close, so it can only ever be matched by
    guesswork on value and timing — which is how the NEAR-USDT double-count of 2026-09-01 slipped
    through. Deliberately permissive: any one identifier is enough, so a row that carries real
    provenance is never treated as an echo.
    """
    if str(row.get("positionId") or "").strip():
      return True
    if MemoryStore._position_open_time_ms(row) is not None:
      return True
    return bool(row.get("positionLifecycleVersion"))

  @staticmethod
  def _same_position_lifecycle(
    local: Dict[str, Any],
    authoritative: Dict[str, Any],
    *,
    window_sec: int,
  ) -> bool:
    """Match closes from one position without merging a rapid same-symbol re-entry.

    Current records use position ID/open time. Only legacy local rows lacking both identifiers
    use a shorter direction-aware time match so historical restart duplicates remain healed.
    """
    local_id = str(local.get("positionId") or "").strip()
    auth_id = str(authoritative.get("positionId") or "").strip()
    local_open = MemoryStore._position_open_time_ms(local)
    auth_open = MemoryStore._position_open_time_ms(authoritative)

    comparable = False
    if local_id and auth_id:
      comparable = True
      if local_id == auth_id:
        return True
    if local_open is not None and auth_open is not None:
      comparable = True
      if abs(local_open - auth_open) <= 1000:
        return True
    if comparable or local_id or local_open is not None or local.get("positionLifecycleVersion"):
      return False

    local_side = MemoryStore._position_side(local)
    auth_side = MemoryStore._position_side(authoritative)
    if not local_side or local_side != auth_side:
      return False
    delta = int(authoritative.get("ts") or 0) - int(local.get("ts") or 0)
    legacy_window_sec = min(max(0, int(window_sec)), 600)
    return 0 <= delta <= legacy_window_sec

  def realized_closes(self, limit: int = 100, symbol: str | None = None) -> list[Dict[str, Any]]:
    """Recent realized closes, exchange-triggered or explicit, deduped oldest→newest.

    The strict realized set (exchange-confirmed TP/SL closes) the adaptive edge controller
    computes its rolling stats from — hold/manage snapshots are excluded.
    """
    with self._lock:
      data = self._read()
    sym = _normalize_symbol(symbol) if symbol else None
    rows = [
      d for d in (data.get("decisions") or [])
      if isinstance(d, dict)
      and self._is_realized_close(str(d.get("action") or ""))
      and d.get("pnl") is not None
      and (sym is None or d.get("symbol") == sym)
    ]
    rows = self._authoritative_realized_rows(rows)
    return rows[-max(1, int(limit)):]

  def record_signal_probe(
    self,
    symbol: str,
    side: str,
    market_price: Any,
    setup_family: Optional[str] = None,
    taker_flow: Optional[Dict[str, Any]] = None,
    price_source: Optional[str] = None,
    model: Optional[str] = None,
    confidence: Any = None,
    min_confidence: Any = None,
    atr15_pct: Any = None,
    planned_entry: Any = None,
    planned_stop: Any = None,
    planned_tp: Any = None,
    crossed_net_rr: Any = None,
    lease_min: Any = None,
    market_state: Optional[Dict[str, Any]] = None,
    gates_passed: Any = None,
  ) -> None:
    """Record a DIRECTION CALL for edge measurement, whether or not it becomes an order.

    Signal quality is a property of the call, not of the execution — so a setup that the model
    committed to but which was then refused for a sizing reason is still evidence about whether the
    model predicts. Measuring only placed orders biases the sample twice over: it drops every setup
    too small to clear the exchange's contract minimum (a systematic subset, not a random one), and it
    couples the evidence supply to the very risk factor the evidence is supposed to govern.

    That coupling produced a live doom loop on 2026-08-10: the continuation family measured "no edge",
    its risk was cut to the floor, the resulting $7.49 notional fell under the $10.24 contract minimum,
    the order was rejected, no probe was recorded — and with no new probes the family could never earn
    back the evidence that would restore its size. Recording at the point of the call breaks that
    circularity completely: risk can go as low as the measurement warrants without ever starving the
    measurement itself. Never raises.

    ``taker_flow`` stamps the aggressor balance observed at the moment of the call (see
    `analytics.taker_flow_summary`). It is recorded, not acted on: it exists so
    `edge.taker_flow_edge_stats` can later answer whether flow agreeing with the direction separated
    the calls that worked from the ones that did not, on this venue at these horizons. Nothing reads
    it at entry time.

    ``price_source`` names the market ``market_price`` was read from (``futures_mark``, or ``spot``
    when the entry path had to fall back to the spot ticker). Settlement reads the SAME market, so the
    perp/spot basis can never be scored as a return — see `_probe_price_source`.

    ``model`` (the Azure deployment that made the call), ``confidence`` (what it stated) and
    ``min_confidence`` (the regime-adjusted floor the call had to clear) are stamped so
    `edge.confidence_edge_stats` can ask, per model, whether stated confidence ranks the calls at all.
    Seven absolute confidence thresholds shape entries and none of that was measurable before
    2026-09-25, because no probe said which model spoke or how sure it claimed to be. Recorded only;
    nothing at entry reads them. An unusable value is dropped rather than stored as a fake number.

    The execution stamps make `edge.execution_map`'s counterfactual possible — every call scored at
    every resting depth, not only at the depth the model happened to use: ``atr15_pct`` (the 15m ATR in
    percent, the depth unit), the planned bracket (``planned_entry`` / ``planned_stop`` / ``planned_tp``
    — the stop distance is the call's own R), ``crossed_net_rr`` (post-cost net RR of that bracket if
    CROSSED at ``market_price``) and ``lease_min`` (the order lease, so settlement can stamp the lease
    low/high from 1m futures bars). All recorded only — the RR gate still decides on its own inputs.

    ``market_state`` (analytics.market_state via ``sanitize_market_state``: breadth24, basket median,
    BTC 24h/72h, BTC daily ADX + bias) lets `edge.signal_edge_stats` split the verdict by the market the
    call was made in. No stored tag could do that before 2026-09-25. Recorded only.

    ``gates_passed`` (a list of ``{'gate', 'hatch'}``; ``[]`` = the call faced no directional gate) says
    which gates this admitted call met and which hatch let it through, so `edge.gate_scoreboard` can
    score the hatches against calls that never faced the gate. Stored as ``gatesPassed`` only when
    given (so legacy rows stay distinguishable from 'faced none'); unknown gates are dropped. Inert for
    every family verdict.
    """
    try:
      px = float(market_price)
    except (TypeError, ValueError):
      return
    if px <= 0:
      return
    s = str(side or "").lower()
    position_side = "long" if s in ("buy", "long") else ("short" if s in ("sell", "short") else None)
    if not position_side:
      return
    ctx: Dict[str, Any] = {
      "positionSide": position_side,
      "marketPriceAtSignal": px,
      "setupFamily": (str(setup_family).strip().lower() or None) if setup_family else None,
      "takerFlow": _sanitize_taker_flow(taker_flow),
      "signalProbe": {},
    }
    source = str(price_source or "").strip().lower()[:40]
    if source:
      ctx["priceSource"] = source
    model_name = str(model or "").strip()[:80]
    if model_name:
      ctx["model"] = model_name
    for key, value in (("confidence", confidence), ("minConfidence", min_confidence)):
      try:
        val = float(value)
      except (TypeError, ValueError):
        continue
      if math.isfinite(val):
        ctx[key] = val
    # Prices, the ATR and the lease must be positive; a crossed net RR of 0 is real (costs ate the reward).
    for key, value, allow_zero in (
      ("atr15Pct", atr15_pct, False), ("plannedEntry", planned_entry, False),
      ("plannedStop", planned_stop, False), ("plannedTp", planned_tp, False),
      ("crossedNetRr", crossed_net_rr, True), ("leaseMin", lease_min, False),
    ):
      try:
        val = float(value)
      except (TypeError, ValueError):
        continue
      if math.isfinite(val) and (val > 0 or (allow_zero and val == 0)):
        ctx[key] = val
    state = sanitize_market_state(market_state)
    if state:
      ctx["marketState"] = state
    if isinstance(gates_passed, (list, tuple)):
      ctx["gatesPassed"] = [
        {"gate": str(item.get("gate")), "hatch": str(item.get("hatch") or "")[:24]}
        for item in gates_passed
        if isinstance(item, dict) and item.get("gate") in SCORED_GATES
      ]
    row = {
      "symbol": _normalize_symbol(symbol),
      "ts": int(time.time()),
      "entryContext": ctx,
    }
    with self._lock:
      data = self._read()
      data.setdefault("signal_probes", []).append(row)
      data["signal_probes"] = _trim_probes_per_family(data["signal_probes"])
      self._write(data)

  @staticmethod
  def _gate_probe_row(symbol: Any, side: Any, market_price: Any, *, kind: str, gates: list,
                      price_source: Any = None, regime: Any = None, market_state: Any = None,
                      now: Any = None) -> Optional[Dict[str, Any]]:
    """One gate-probe row in the signal-probe shape (so settlement and `edge` read it unchanged), or None."""
    try:
      px = float(market_price)
    except (TypeError, ValueError):
      return None
    if not (math.isfinite(px) and px > 0):
      return None
    s = str(side or "").strip().lower()
    position_side = "long" if s in ("buy", "long") else ("short" if s in ("sell", "short") else None)
    sym = _normalize_symbol(symbol) if symbol else ""
    if not position_side or not sym:
      return None
    ctx: Dict[str, Any] = {
      "positionSide": position_side,
      "marketPriceAtSignal": px,
      "gateProbe": kind,
      "gates": [g for g in gates if g in SCORED_GATES],
      "signalProbe": {},
    }
    source = str(price_source or "").strip().lower()[:40]
    if source:
      ctx["priceSource"] = source
    stamp = _gate_regime_stamp(regime)
    if stamp:
      ctx["regime"] = stamp
    state = sanitize_market_state(market_state)
    if state:
      ctx["marketState"] = state
    try:
      ts = int(float(now)) if now is not None else int(time.time())
    except (TypeError, ValueError):
      ts = int(time.time())
    return {"symbol": sym, "ts": ts, "entryContext": ctx}

  def record_gate_probe(
    self,
    symbol: str,
    side: str,
    market_price: Any,
    gate: str,
    *,
    setup_family: Optional[str] = None,
    price_source: Optional[str] = None,
    model: Optional[str] = None,
    confidence: Any = None,
    regime: Optional[Dict[str, Any]] = None,
    market_state: Optional[Dict[str, Any]] = None,
  ) -> bool:
    """Record a direction call a gate HARD-REFUSED, for the gate scoreboard only. Returns True if stored.

    Same shape as a signal probe (base = the market price when the refusal was returned, settled on the
    same market at the same horizons, funding credited) but in its own ``gate_probes`` bucket, which
    ``signal_probes()`` never reads: a refused call must not move a family verdict. ``gate`` must be one
    of ``SCORED_GATES``. ``setup_family`` is what the model declared (the scoreboard scores the call at
    that family's horizon); ``regime`` the gate fields the refusal was decided on. Never raises.
    """
    try:
      if gate not in SCORED_GATES:
        return False
      row = self._gate_probe_row(symbol, side, market_price, kind="refusal", gates=[gate],
                                 price_source=price_source, regime=regime, market_state=market_state)
      if row is None:
        return False
      ctx = row["entryContext"]
      ctx["gate"] = gate
      ctx["setupFamily"] = (str(setup_family).strip().lower() or None) if setup_family else None
      model_name = str(model or "").strip()[:80]
      if model_name:
        ctx["model"] = model_name
      try:
        conf = float(confidence)
        if math.isfinite(conf):
          ctx["confidence"] = conf
      except (TypeError, ValueError):
        pass
      with self._lock:
        data = self._read()
        data["gate_probes"] = _trim_gate_probes(list(data.get("gate_probes") or []) + [row])
        self._write(data)
      return True
    except Exception as exc:
      logger.warning("GATE PROBE LOST: %s %s refusal by %s not stored (%s)", symbol, side, gate, exc)
      return False

  def record_gate_state_probes(
    self,
    symbol: str,
    market_price: Any,
    gates_by_side: Dict[str, Any],
    *,
    price_source: Optional[str] = None,
    regime: Optional[Dict[str, Any]] = None,
    market_state: Optional[Dict[str, Any]] = None,
    now: Any = None,
  ) -> int:
    """Record what the directional gates would do to a call on each side RIGHT NOW. Returns rows stored.

    ``gates_by_side`` is ``{'long': [...], 'short': [...]}`` — every gate that would refuse a plain call
    on that side (an empty list is a real row: the allowed complement the blocked rows are compared
    with). Model-independent: written when analyze_market_context computes the gate state, whether or
    not anything is proposed, so it sees what the model self-censors — the footprint the hard refusals
    miss. One row per (symbol, side) per ``GATE_STATE_WINDOW_MIN``: a side that already has a row inside
    that window is skipped, which keeps every stored row non-overlapping at every settled horizon. Both
    sides share ONE write. Never raises.
    """
    try:
      t = int(float(now)) if now is not None else int(time.time())
      sym = _normalize_symbol(symbol) if symbol else ""
      rows = []
      for side in ("long", "short"):
        gates = gates_by_side.get(side) if isinstance(gates_by_side, dict) else None
        if not isinstance(gates, (list, tuple)):
          continue
        row = self._gate_probe_row(sym, side, market_price, kind="state", gates=list(gates),
                                   price_source=price_source, regime=regime, market_state=market_state, now=t)
        if row is not None:
          rows.append(row)
      if not rows:
        return 0
      window = int(GATE_STATE_WINDOW_MIN) * 60
      with self._lock:
        data = self._read()
        existing = list(data.get("gate_probes") or [])
        recent = {
          r["entryContext"].get("positionSide")
          for r in existing
          if _gate_probe_kind(r) == "state" and r.get("symbol") == sym
          and 0 <= t - int(r.get("ts") or 0) < window
        }
        fresh = [r for r in rows if r["entryContext"]["positionSide"] not in recent]
        if not fresh:
          return 0
        data["gate_probes"] = _trim_gate_probes(existing + fresh)
        self._write(data)
      return len(fresh)
    except Exception as exc:
      logger.warning("GATE STATE PROBE LOST: %s not stored (%s)", symbol, exc)
      return 0

  def gate_probes(self) -> list[Dict[str, Any]]:
    """Every retained gate-probe row (refusal and unfolded state), oldest first, as deep copies."""
    with self._lock:
      data = self._read()
    rows = [copy.deepcopy(r) for r in (data.get("gate_probes") or []) if _gate_probe_kind(r)]
    rows.sort(key=lambda r: int(r.get("ts") or 0))
    return rows

  def gate_state_days(self) -> Dict[str, Any]:
    """The folded gate-state day cells, decoded: ``{day: {side: {horizon: [n, s, {gate: [n, s]}]}}}``.

    ``day`` is the UTC day number (epoch seconds // 86400) as a string, ``s`` the SUM of the signed
    forward returns (fractions, funding included) of the day's rows at that horizon, ``n`` their count;
    the per-gate pair is the same over the rows that gate would have refused. Unreadable cells are
    skipped. See `fold_settled_gate_states`.
    """
    with self._lock:
      data = self._read()
    out: Dict[str, Any] = {}
    for key, cell in (data.get("gate_state_days") or {}).items() if isinstance(data.get("gate_state_days"), dict) else []:
      decoded = _decode_gate_day(cell)
      if decoded is not None:
        out[str(key)] = decoded
    return out

  def fold_settled_gate_states(
    self,
    cells_fn: Callable[[list], Any],
    horizons_min: tuple = SIGNAL_PROBE_HORIZONS_MIN,
  ) -> int:
    """Fold every FULLY settled gate-state row into ``gate_state_days`` and drop it. Returns rows folded.

    Fully settled = every horizon in ``horizons_min`` is stamped (a price, or None once written off), so
    a folded row can never receive a later stamp. ``cells_fn`` is `edge.gate_state_cells`, injected by
    the poll loop so the return arithmetic and the de-overlap live only in `edge` (memory never imports
    the scorer); it returns ``{day: {side: {horizon: [n, s, {gate: [n, s]}]}}}`` increments, which are
    ADDED to the stored cells. Rows and cells change in one locked write, so a row is counted exactly
    once. What is given up: a folded row cannot be re-settled later (as the Sep 25 spot->futures fix
    re-settled signal probes); the rows it came from are settled on the futures mark from day one.
    Never raises.
    """
    try:
      keys = [f"m{int(h)}" for h in horizons_min]
      with self._lock:
        data = self._read()
        rows = list(data.get("gate_probes") or [])
        done = []
        for r in rows:
          if _gate_probe_kind(r) != "state":
            continue
          probe = r["entryContext"].get("signalProbe")
          if isinstance(probe, dict) and all(k in probe for k in keys):
            done.append(r)
        if not done:
          return 0
        increments = cells_fn(copy.deepcopy(done)) or {}
        stored = data.get("gate_state_days") if isinstance(data.get("gate_state_days"), dict) else {}
        days = dict(stored)
        for day, sides in increments.items():
          day_cell = _decode_gate_day(days.get(str(day))) or {}
          for side, horizons in (sides or {}).items():
            side_cell = day_cell.setdefault(str(side), {})
            for horizon, inc in (horizons or {}).items():
              n0, s0, g0 = (side_cell.get(str(horizon)) or [0, 0.0, {}])[:3]
              gates = dict(g0 or {})
              for gate, (gn, gs) in (inc[2] or {}).items():
                prev = gates.get(gate) or [0, 0.0]
                gates[gate] = [int(prev[0]) + int(gn), round(float(prev[1]) + float(gs), 7)]
              side_cell[str(horizon)] = [int(n0) + int(inc[0]), round(float(s0) + float(inc[1]), 7), gates]
          days[str(day)] = _encode_gate_day(day_cell)
        done_ids = {id(r) for r in done}
        data["gate_probes"] = [r for r in rows if id(r) not in done_ids]
        data["gate_state_days"] = _trim_gate_state_days(days)
        self._write(data)
      return len(done)
    except Exception as exc:
      logger.warning("GATE STATE FOLD failed (%s) — settled state rows wait for the next poll", exc)
      return 0

  def record_macro_events(self, events: Any) -> int:
    """Replace the scheduled-macro calendar with ``events``; returns how many were stored.

    Scheduled releases (CPI, FOMC, NFP) are published a year ahead, so this is a calendar rather than
    a news feed — the Research Agent fetches it with the web search it already has and code enforces
    the timing. That split keeps the model on research and the clock-work in code, and it adds no API
    key or runtime dependency that could fail mid-poll.

    Rows are ``{"name", "ts", "impact"}``; anything malformed or already past is dropped, so a bad
    fetch degrades to a shorter calendar rather than a corrupt one. Never raises.
    """
    now = int(time.time())
    cleaned: list[Dict[str, Any]] = []
    for row in events or []:
      if not isinstance(row, dict):
        continue
      name = str(row.get("name") or "").strip()[:120]
      if not name:
        continue
      try:
        ets = int(float(row.get("ts")))
      except (TypeError, ValueError):
        continue
      if ets <= now:
        continue
      impact = str(row.get("impact") or "high").strip().lower()
      if impact not in ("high", "medium", "low"):
        impact = "high"
      cleaned.append({"name": name, "ts": ets, "impact": impact})
    cleaned.sort(key=lambda e: e["ts"])
    cleaned = cleaned[:MAX_MACRO_EVENTS]
    with self._lock:
      data = self._read()
      data["macro_events"] = cleaned
      data["macro_events_updated"] = now
      self._write(data)
    return len(cleaned)

  def macro_events(self, within_hours: float = 72.0) -> list[Dict[str, Any]]:
    """Upcoming scheduled releases inside ``within_hours`` (plus any just past, for the after-window)."""
    now = int(time.time())
    horizon = now + int(max(0.0, float(within_hours)) * 3600)
    with self._lock:
      data = self._read()
    out = []
    for row in data.get("macro_events") or []:
      if not isinstance(row, dict):
        continue
      try:
        ets = int(row.get("ts"))
      except (TypeError, ValueError):
        continue
      if now - 86400 <= ets <= horizon:
        out.append(row)
    return sorted(out, key=lambda e: e["ts"])

  def macro_calendar_state(self) -> Dict[str, Any]:
    """The whole stored calendar and its refresh time, unfiltered — for deciding whether to refresh."""
    with self._lock:
      data = self._read()
    rows = [dict(e) for e in (data.get("macro_events") or []) if isinstance(e, dict)]
    return {"events": sorted(rows, key=lambda e: e.get("ts") or 0), "updated": data.get("macro_events_updated")}

  def macro_calendar_age_hours(self) -> Optional[float]:
    """Hours since the calendar was last refreshed, or None if never — so staleness is visible."""
    with self._lock:
      data = self._read()
    ts = data.get("macro_events_updated")
    try:
      return round((time.time() - float(ts)) / 3600.0, 1)
    except (TypeError, ValueError):
      return None

  def note_agent_close(self, symbol: str) -> None:
    """Remember that the MODEL just closed a position on ``symbol`` (its reduce-only order was accepted).

    Exit probes are recorded later, when the poll loop notices the position is gone, and by then the
    exchange reports every close the same way ("TP/SL triggered"). Without this marker a close by the
    model and a close by the code's own trailing stop are indistinguishable — which is how the exit
    scoreboard came to blame the model for 16 trailing-stop exits in a window where it had made ONE
    close, and that one had helped. Never raises; the marker list is bounded.
    """
    sym = _normalize_symbol(symbol)
    if not sym:
      return
    with self._lock:
      data = self._read()
      marks = [m for m in (data.get("agent_closes") or []) if isinstance(m, dict)]
      marks.append({"symbol": sym, "ts": int(time.time())})
      data["agent_closes"] = marks[-200:]
      self._write(data)

  def recent_agent_close(self, symbol: str, within_sec: int = 900) -> bool:
    """True when the model closed ``symbol`` within the last ``within_sec`` seconds."""
    sym = _normalize_symbol(symbol)
    now = time.time()
    with self._lock:
      data = self._read()
    for m in reversed(data.get("agent_closes") or []):
      if not isinstance(m, dict) or m.get("symbol") != sym:
        continue
      try:
        return (now - float(m.get("ts") or 0)) <= max(0, int(within_sec))
      except (TypeError, ValueError):
        return False
    return False

  def record_exit_probe(
    self,
    symbol: str,
    position_side: str,
    entry_price: Any,
    stop_price: Any,
    take_profit: Any,
    exit_price: Any,
    realized_r: Any = None,
    setup_family: Optional[str] = None,
    closed_by: Optional[str] = None,
    regime: Optional[Dict[str, Any]] = None,
    *,
    fill_ts: Any = None,
    init_risk_px: Any = None,
    noise_band_r: Any = None,
    hold_until_ts: Any = None,
    entry_bias: Optional[Dict[str, Any]] = None,
    counter_at_entry: Optional[bool] = None,
    htf_aligned: Optional[bool] = None,
    market_state: Optional[Dict[str, Any]] = None,
    market_state_at_exit: Optional[Dict[str, Any]] = None,
  ) -> None:
    """Record an EARLY close so it can later be scored against the bracket it overrode.

    ``closed_by`` is "agent" when the model closed it and "protection" when the code's trailing stop
    or profit-lock did. Both land between the original stop and target, so both are worth scoring —
    but against different questions, and only the first is the model's decision.

    Stack-replay inputs (2026-09-25): ``fill_ts``, ``init_risk_px`` (|entry - ORIGINAL stop|),
    ``noise_band_r`` (1/stopAtrMult) and ``hold_until_ts`` (the carry hold's first settlement after the
    fill) let main.py replay what the live exit stack — bracket + breakeven/trail + carry hold — would
    have done had the model not closed, which is the benchmark an AGENT close is scored against
    (``set_exit_probe_stack``); the bare bracket stays on the row for audit and for the trail's own
    record — an agent close without a stack result is never scored on it (edge.exit_discipline_stats). ``entry_bias`` {15m, 1h, 4h, 1D}, ``counter_at_entry`` (15m AND 1h both opposed the side
    at entry) and ``htf_aligned`` (4h AND 1D both agreed) let the scoreboard split the model's closes
    by whether the opposition it closed on already existed when it entered — the case that lost on
    DASH/KCS and helped on G/XMR, n=5, so the record decides, not a rule.

    ``market_state`` is the ENTRY's market-state stamp (entryContext.marketState) and
    ``market_state_at_exit`` the poll loop's reading at the close. The per-symbol ``regime`` tag read
    'trending/strong' on every tagged trail exit, so the trail's chop row could never fill; the trail's
    record is now also split by the entry's breadth24 tercile (trailByMarketState). Recorded only.

    The bot measures whether its entries predict (``signal_probes``) but never measured whether its
    *exits* helped — and the exits turned out to be the dominant behaviour: over the 2026-09-02 window
    16 positions were closed by the agent against 2 by the profit-lock, at a 13-minute median hold on
    brackets whose targets need hours. (A replay once put the cost of those closes at ~2.6R; it read
    futures candles in the wrong column order, and re-measured correctly the closes slightly HELPED,
    +0.84R over 17 trades. That uncertainty is the reason this is recorded and scored live.)

    The fix the project philosophy asks for is evidence, not a veto: the model still owns the decision
    to close, it just gets to see its own record at doing so. Self-correcting in both directions — if
    discretionary closes start beating the brackets, the same number says so and endorses them. Never
    raises.
    """
    try:
      e = float(entry_price)
      sl = float(stop_price)
      tp = float(take_profit)
      xp = float(exit_price)
    except (TypeError, ValueError):
      return
    # NaN fails every comparison, so `<= 0` alone would let it through and poison the aggregate.
    if not all(math.isfinite(v) and v > 0 for v in (e, sl, tp, xp)):
      return
    if e == sl:
      return  # no risk unit; nothing to express the comparison in
    sym = _normalize_symbol(symbol)
    if not sym:
      return
    side = str(position_side or "").lower()
    side = "long" if side in ("buy", "long") else ("short" if side in ("sell", "short") else "")
    if not side:
      return
    try:
      rr = float(realized_r) if realized_r is not None else None
    except (TypeError, ValueError):
      rr = None
    row = {
      "symbol": sym,
      "ts": int(time.time()),
      "positionSide": side,
      "entryPrice": e,
      "stopPrice": sl,
      "takeProfitPrice": tp,
      "exitPrice": xp,
      "realizedR": rr,
      "setupFamily": (str(setup_family).strip().lower() or None) if setup_family else None,
      "closedBy": (str(closed_by).strip().lower() or None) if closed_by else None,
      # The entry's own regime read (market_regime / strength) so trail-vs-bracket can be split by the
      # market it happened in. The trail is right in chop and wrong in a trend; a regime-adaptive
      # trail needs evidence from BOTH, and this is where that evidence accumulates on its own.
      "regime": {
        k: str(regime.get(k)) for k in ("market_regime", "strength") if isinstance(regime, dict) and regime.get(k)
      } if isinstance(regime, dict) else None,
      "outcome": {},
    }

    def _pos_num(value: Any) -> Optional[float]:
      try:
        out = float(value)
      except (TypeError, ValueError):
        return None
      return out if math.isfinite(out) and out > 0 else None

    row["fillTs"] = _pos_num(fill_ts)
    row["initRiskPx"] = _pos_num(init_risk_px)
    row["noiseBandR"] = _pos_num(noise_band_r)
    row["holdUntilTs"] = _pos_num(hold_until_ts)
    row["entryBias"] = {
      k: str(entry_bias.get(k)).lower() for k in ("15m", "1h", "4h", "1D") if entry_bias.get(k)
    } if isinstance(entry_bias, dict) else None
    row["counterAtEntry"] = counter_at_entry if isinstance(counter_at_entry, bool) else None
    row["htfAligned"] = htf_aligned if isinstance(htf_aligned, bool) else None
    row["marketState"] = sanitize_market_state(market_state)
    row["marketStateAtExit"] = sanitize_market_state(market_state_at_exit)
    with self._lock:
      data = self._read()
      data.setdefault("exit_probes", []).append(row)
      if len(data["exit_probes"]) > MAX_EXIT_PROBES:
        data["exit_probes"] = data["exit_probes"][-MAX_EXIT_PROBES:]
      self._write(data)

  def set_exit_probe_stack(
    self,
    symbol: str,
    ts: Any,
    stack_r: Any,
    resolved_by: Any,
    resolved_ts: Any,
    source: Any,
    *,
    pre_close_exit_suppressed: Optional[bool] = None,
  ) -> bool:
    """Store the live-exit-stack replay result on the exit probe ``(symbol, ts)``. Storage only.

    The replay itself (protection.replay_protection_stack over 1m futures bars) runs in main.py: this
    module must not import protection or kucoin. ``stack_r`` None with ``resolved_by`` "unavailable"
    records that the replay could not be done — the scoreboard then falls back to the bracket for
    that row and counts it as such. Writes once: a row that already has a stack result is left alone.
    Returns True when a row was updated. Never raises.
    """
    try:
      sym = _normalize_symbol(symbol)
      key_ts = int(float(ts))
    except (TypeError, ValueError):
      return False
    try:
      value = float(stack_r) if stack_r is not None else None
    except (TypeError, ValueError):
      value = None
    if value is not None and not math.isfinite(value):
      value = None
    try:
      when = float(resolved_ts) if resolved_ts is not None else None
    except (TypeError, ValueError):
      when = None
    try:
      with self._lock:
        data = self._read()
        for row in data.get("exit_probes") or []:
          if not isinstance(row, dict) or row.get("symbol") != sym or row.get("stack"):
            continue
          try:
            if int(float(row.get("ts"))) != key_ts:
              continue
          except (TypeError, ValueError):
            continue
          row["stack"] = {
            "stackR": value,
            "resolvedBy": str(resolved_by) if resolved_by else None,
            "resolvedTs": when,
            "source": str(source) if source else None,
            "scoredTs": int(time.time()),
          }
          if isinstance(pre_close_exit_suppressed, bool):
            row["stack"]["preCloseExitSuppressed"] = pre_close_exit_suppressed
          self._write(data)
          return True
    except Exception as exc:
      logger.warning("EXIT PROBE stack store failed for %s @ %s (%s)", symbol, ts, exc)
    return False

  def settle_exit_probes(self, prices: Dict[str, Any], expire_hours: float = EXIT_PROBE_EXPIRE_HOURS) -> int:
    """Resolve each recorded discretionary close against what its bracket would have done.

    ``prices`` must be the FUTURES MARK map: the brackets being replayed trigger on the mark
    (stopPriceType MP), so the spot ticker is the wrong market. Until 2026-09-25 this read spot, and
    on a coin with a wide perp/spot basis that invented bracket outcomes — ONE-USDT protection exits
    were stored at -4.70R against -0.60R replayed on futures bars (a 09-21 probe stored as a TP one
    minute later had in fact been stopped out on the contract).

    Resolution is at poll resolution: a level crossed and retraced between two polls is missed, which
    UNDER-credits brackets and so flatters discretionary and protection closes. A 1m replay put that
    bias at the larger part of the stored-vs-replayed gap (-13.32R vs -16.88R over 84 probes), and a
    per-poll mark does not remove it — treat `exitDiscipline` numbers as poll-resolution.

    Unresolved probes are marked to market at ``expire_hours`` so a trade that simply drifted still
    contributes. **A probe with no price by ``expire_hours`` + its settle tolerance is recorded as
    ``unmeasured`` (``bracketR`` None), never resolved late**: the check used to be "any price, any
    time", and an XMR probe from 2026-09-12 was resolved seven days later, when the ticker reappeared,
    as if a week-later touch were the bracket's outcome. Never raises; one bad row cannot stop the rest.
    """
    prices = prices or {}
    now = int(time.time())
    expire_sec = int(float(expire_hours) * 3600)
    stale_after = expire_sec + _probe_settle_tolerance_sec(float(expire_hours) * 60.0)
    settled = 0
    with self._lock:
      data = self._read()
      for row in data.get("exit_probes") or []:
        if not isinstance(row, dict):
          continue
        outcome = row.get("outcome")
        if not isinstance(outcome, dict) or outcome.get("resolved"):
          continue
        try:
          ts0 = int(row.get("ts") or now)
          if now - ts0 > stale_after:
            # Its measurable life is over with no in-window price: honestly unknown. A later price
            # says nothing about which leg the bracket would have hit first.
            outcome.update({"resolved": "unmeasured", "bracketR": None, "resolvedTs": now,
                            "priceSource": EXIT_PROBE_PRICE_SOURCE})
            settled += 1
            continue
          px = _price_from(prices, row.get("symbol"))
          if px is None:
            continue
          try:
            e = float(row["entryPrice"]); sl = float(row["stopPrice"]); tp = float(row["takeProfitPrice"])
          except (TypeError, ValueError, KeyError):
            continue
          risk = abs(e - sl)
          if risk <= 0:
            continue
          long_side = row.get("positionSide") == "long"
          hit_tp = px >= tp if long_side else px <= tp
          hit_sl = px <= sl if long_side else px >= sl
          if hit_tp:
            outcome.update({"resolved": "take_profit", "bracketR": abs(tp - e) / risk})
          elif hit_sl:
            outcome.update({"resolved": "stop", "bracketR": -1.0})
          elif now >= ts0 + expire_sec:
            mark = (px - e) / risk if long_side else (e - px) / risk
            outcome.update({"resolved": "expired", "bracketR": mark})
          else:
            continue
          outcome["resolvedTs"] = now
          # Which market resolved it: the futures mark. Outcomes WITHOUT this stamp were resolved on the
          # spot ticker before 2026-09-25 (ONE-USDT protection exits: -4.70R stored vs -0.60R on
          # futures) and are the ones scripts/resettle_probes_futures.py re-resolves — re-run safe.
          outcome["priceSource"] = EXIT_PROBE_PRICE_SOURCE
          settled += 1
        except Exception as exc:
          logger.warning("EXIT PROBE settle failed for %s (%s) — other probes unaffected",
                         row.get("symbol"), exc)
      if settled:
        self._write(data)
    return settled

  def symbols_due_for_settlement(
    self,
    now: Optional[float] = None,
    horizons_min: tuple = SIGNAL_PROBE_HORIZONS_MIN,
    expire_hours: float = EXIT_PROBE_EXPIRE_HOURS,
  ) -> set:
    """Symbols that need a FUTURES MARK this poll for some probe to settle (``settlement_cutoffs``' keys)."""
    return set(self.settlement_cutoffs(now, horizons_min=horizons_min, expire_hours=expire_hours))

  def settlement_cutoffs(
    self,
    now: Optional[float] = None,
    horizons_min: tuple = SIGNAL_PROBE_HORIZONS_MIN,
    expire_hours: float = EXIT_PROBE_EXPIRE_HOURS,
  ) -> Dict[str, float]:
    """``{symbol: cutoff_ts}`` for every symbol that needs a FUTURES MARK this poll for some probe to settle.

    That is: a futures-based signal probe with a horizon due now and still inside its settle tolerance,
    or an exit probe that is unresolved and still inside its measurable life. The poll loop fetches a
    mark for exactly these, so settlement costs a handful of small public calls rather than one
    1.4 MB ``/contracts/active`` pull every minute. Spot-based probes are left out — they settle from
    the spot snapshot the loop already holds. ``cutoff_ts`` is the EARLIEST moment one of that symbol's
    due rows is written off as unmeasured, so the loop can spend a limited per-poll I/O budget on the
    measurements that would otherwise be lost first (signal/gate horizons before 8h exit probes) and
    cap a failing symbol's backoff at its remaining tolerance (2026-09-25 review). Never raises.
    """
    t = int(time.time()) if now is None else int(now)
    stale_after = int(float(expire_hours) * 3600) + _probe_settle_tolerance_sec(float(expire_hours) * 60.0)
    out: Dict[str, float] = {}

    def _keep(symbol: Any, cutoff: float) -> None:
      if symbol:
        out[symbol] = min(float(cutoff), out.get(symbol, float("inf")))

    with self._lock:
      data = self._read()
      # Gate probes settle exactly like signal probes (their own bucket, the same horizons and market).
      for row in (list(data.get("trades") or []) + list(data.get("signal_probes") or [])
                  + list(data.get("gate_probes") or [])):
        try:
          if not isinstance(row, dict):
            continue
          ctx = row.get("entryContext")
          if not isinstance(ctx, dict) or not ctx.get("marketPriceAtSignal"):
            continue
          if _probe_price_source(ctx) == "spot":
            continue
          probe = ctx.get("signalProbe") if isinstance(ctx.get("signalProbe"), dict) else {}
          ts0 = int(row.get("ts") or 0)
          for horizon in horizons_min:
            due_at = ts0 + int(horizon) * 60
            if f"m{int(horizon)}" in probe:
              continue
            cutoff = due_at + _probe_settle_tolerance_sec(horizon)
            if due_at <= t <= cutoff:
              _keep(row.get("symbol"), cutoff)
              break
        except Exception:
          continue
      for row in data.get("exit_probes") or []:
        try:
          if not isinstance(row, dict):
            continue
          outcome = row.get("outcome")
          if not isinstance(outcome, dict) or outcome.get("resolved"):
            continue
          ts0 = int(row.get("ts") or t)
          if t - ts0 <= stale_after:
            _keep(row.get("symbol"), ts0 + stale_after)
        except Exception:
          continue
    return out

  def exit_probes(self, limit: int = 200) -> list[Dict[str, Any]]:
    """Recorded discretionary closes, newest last."""
    with self._lock:
      data = self._read()
    rows = [r for r in (data.get("exit_probes") or []) if isinstance(r, dict)]
    return rows[-max(1, int(limit)):]

  def settle_signal_probes(
    self,
    prices: Dict[str, Any],
    horizons_min: tuple = SIGNAL_PROBE_HORIZONS_MIN,
    *,
    spot_prices: Optional[Dict[str, Any]] = None,
    funding_received: Optional[Callable[[str, str, float, float], Any]] = None,
    lease_extremes: Optional[Callable[[str, float, float], Any]] = None,
  ) -> int:
    """Stamp the forward price on any entry signal whose measurement horizon has elapsed.

    This closes the feedback loop the bot never had. It has always measured *outcomes* (win rate,
    realized R) — but an outcome conflates three different things: whether the direction call was
    right, whether the fill was any good, and whether the exit was well managed. Six rounds of exit,
    cost and sizing fixes all landed correctly and the account still bled, because none of them could
    answer the only question that decides profitability: **does the direction call predict?**

    Measured offline on 2026-08-06 over 96 recorded signals, forward return from the *market* price at
    signal time was -0.135% at 1h (t=-2.02, hit rate 35%) against a random-time null of -0.025% — no
    edge, and significantly negative in the dominant configuration. Storing the probe makes that
    measurable continuously and in-process, so a future model that genuinely predicts will show it.

    **Each row settles from the SAME market its base price came from** (`_probe_price_source`):
    ``prices`` is the FUTURES MARK map, ``spot_prices`` the spot snapshot for the rare row whose base
    fell back to spot. There is no cross-market fallback in either direction — a row whose market has
    no price this poll waits, and the tolerance rule below writes it off if the price never comes.
    Settling futures-based probes on the spot ticker (the code until 2026-09-25) scored the perp/spot
    basis as prediction: +1.68% mean / +0.16% median phantom return on `funding_carry` at its 60m
    horizon, ~0 on every other family, and enough to hold the family at full size on a record of
    -0.04R over 18 real closes.

    ``funding_received(symbol, side, t0, t1)`` credits the funding the position would have been paid
    over the window (see `funding_received_from_history`); it is stamped as ``f{h}`` beside ``m{h}``
    and `edge` adds it to the signed return. It is called OUTSIDE the store lock (it may hit the
    network). The price stamp never waits on it. **Unknown is kept apart from zero** (2026-09-25
    review): when a lookup fails, ``f{h}`` is written as None (with ``t{h}``, the time the price was
    observed) instead of being left absent — absent used to score as zero, and since a stamped horizon
    is never due again the credit was lost for good (a 1h carry at -0.2%/settlement lost ~0.8% of return
    per affected probe at 240m, always biasing carry DOWN). A None credit is re-asked on later polls
    until ``_funding_backfill_window_sec`` and overwritten with the exact history value; past that it
    stays None ("never known"), which `edge` skips for funding_carry rather than counting as zero.
    Legacy rows and calls without ``funding_received`` are unchanged (absent = zero).

    Cheap by construction: writes only when a horizon elapses, and never raises — a malformed row is
    logged at WARNING and skipped, so one bad row cannot starve every other probe of its settlement
    (a silent probe outage froze every edge verdict for 2.6 days in 2026-09). Returns the number of
    stamps written.

    **A horizon that elapsed long ago is recorded as unmeasurable (``None``), never back-stamped.**
    The check used to be "has the horizon passed?", which silently means "stamp today's price on
    every probe old enough" — harmless while the loop runs every 60s and each horizon is crossed
    within one poll of its true moment, but wrong after any downtime, and catastrophically wrong the
    first time a NEW horizon is introduced: adding the 5m point would have stamped all 400 retained
    probes with the current price and labelled a multi-day return as a five-minute one. ``None`` is
    the honest record of a measurement that was missed, and `edge` already skips it; writing it once
    also stops the horizon being retried forever.

    ``lease_extremes(symbol, t0, t1)`` returns the ``(low, high)`` the contract traded over a probe's
    order lease (1m FUTURES bars, column order asserted by the caller) or None when unknown. Once a
    probe's lease has elapsed it is stamped ``leaseLow`` / ``leaseHigh`` — what `edge.execution_map`
    needs to say whether a limit at ANY depth would have filled — and a probe whose bars never came
    within `_lease_settle_tolerance_sec` is stamped None and never back-filled. Only dedicated probe
    rows that recorded ``leaseMin`` qualify. Fetched outside the lock, like the funding credit.
    """
    now = int(time.time())

    def _rows(data: Dict[str, Any]) -> list:
      # Gate probes (report-only, see record_gate_probe) are stamped by the same rule on the same market;
      # they live in their own bucket so `signal_probes()` and every family verdict never see them.
      return (list(data.get("trades") or []) + list(data.get("signal_probes") or [])
              + list(data.get("gate_probes") or []))

    # Phase 1 (read-only): which price stamps and lease extremes are due, so their network lookups can
    # run without holding the store lock.
    credits: Dict[tuple, float] = {}
    backfills: Dict[tuple, float] = {}
    leases: Dict[tuple, tuple] = {}
    if funding_received is not None or lease_extremes is not None:
      wanted: list[tuple] = []
      wanted_backfill: list[tuple] = []
      wanted_leases: list[tuple] = []
      with self._lock:
        data = self._read()
        if funding_received is not None:
          for row in _rows(data):
            try:
              for _key, horizon, px in _signal_stamps_due(row, prices, spot_prices, horizons_min, now):
                if px is not None:
                  wanted.append((row.get("symbol"), row["entryContext"].get("positionSide"),
                                 int(row.get("ts") or 0), int(horizon)))
              for horizon, t_end in _credit_backfills_due(row, now, horizons_min):
                wanted_backfill.append((row.get("symbol"), row["entryContext"].get("positionSide"),
                                        int(row.get("ts") or 0), int(horizon), t_end))
            except Exception:
              continue    # phase 3 re-walks the row and logs it
        if lease_extremes is not None:
          for row in data.get("signal_probes") or []:
            try:
              if _lease_stamp_due(row, now) == "fetch":
                t0, t1 = _lease_window(row)
                wanted_leases.append((row.get("symbol"), int(row.get("ts") or 0), t0, t1))
            except Exception:
              continue
      # Phase 2: the funding credit and the lease extremes, outside the lock.
      for symbol, side, ts0, horizon in wanted:
        try:
          value = funding_received(symbol, side, ts0, now)
          value = float(value) if value is not None else None
        except Exception as exc:
          logger.warning("PROBE FUNDING credit unavailable for %s (%s) — f%dm recorded unknown, backfilled later",
                         symbol, exc, horizon)
          continue
        if value is not None and math.isfinite(value):
          # Keyed by SIDE too: a gate-state reading writes a long and a short row at the same instant,
          # and their credits have opposite signs.
          credits[(symbol, side, ts0, horizon)] = value
      for symbol, side, ts0, horizon, t_end in wanted_backfill:
        try:
          value = funding_received(symbol, side, ts0, t_end)
          value = float(value) if value is not None else None
        except Exception as exc:
          logger.warning("PROBE FUNDING backfill unavailable for %s (%s) — f%dm stays unknown for now",
                         symbol, exc, horizon)
          continue
        if value is not None and math.isfinite(value):
          backfills[(symbol, side, ts0, horizon)] = value
      for symbol, ts0, t0, t1 in wanted_leases:
        try:
          got = lease_extremes(symbol, t0, t1)
          if got is None:
            continue
          low, high = float(got[0]), float(got[1])
        except Exception as exc:
          logger.warning("PROBE LEASE extremes unavailable for %s (%s) — retried until its tolerance", symbol, exc)
          continue
        if math.isfinite(low) and math.isfinite(high) and 0 < low <= high:
          leases[(symbol, ts0)] = (low, high)

    # Phase 3: stamp, under the lock, re-reading in case another writer ran in between.
    settled = 0
    with self._lock:
      data = self._read()
      for row in _rows(data):
        try:
          stamps = _signal_stamps_due(row, prices, spot_prices, horizons_min, now)
          if not stamps:
            continue
          ctx = row["entryContext"]
          if not isinstance(ctx.get("signalProbe"), dict):
            ctx["signalProbe"] = {}
          probe = ctx["signalProbe"]
          ts0 = int(row.get("ts") or 0)
          for key, horizon, px in stamps:
            probe[key] = px
            settled += 1
            credit = (credits.get((row.get("symbol"), ctx.get("positionSide"), ts0, int(horizon)))
                      if px is not None else None)
            if credit is not None:
              probe[f"f{int(horizon)}"] = credit
            elif px is not None and funding_received is not None:
              # Unknown, NOT zero: kept apart on disk and backfilled on a later poll (see docstring).
              probe[f"f{int(horizon)}"] = None
              probe[f"t{int(horizon)}"] = now
        except Exception as exc:
          logger.warning("SIGNAL PROBE settle failed for %s (%s) — other probes unaffected",
                         row.get("symbol") if isinstance(row, dict) else "?", exc)
      if backfills:
        for row in _rows(data):
          try:
            if not isinstance(row, dict):
              continue
            ctx = row.get("entryContext")
            probe = ctx.get("signalProbe") if isinstance(ctx, dict) else None
            if not isinstance(probe, dict):
              continue
            ts0 = int(row.get("ts") or 0)
            for horizon in horizons_min:
              h = int(horizon)
              value = backfills.get((row.get("symbol"), ctx.get("positionSide"), ts0, h))
              if value is None or f"f{h}" not in probe or probe.get(f"f{h}") is not None:
                continue
              probe[f"f{h}"] = value
              probe.pop(f"t{h}", None)
              settled += 1
          except Exception as exc:
            logger.warning("PROBE FUNDING backfill stamp failed for %s (%s) — other probes unaffected",
                           row.get("symbol") if isinstance(row, dict) else "?", exc)
      if lease_extremes is not None:
        for row in data.get("signal_probes") or []:
          try:
            status = _lease_stamp_due(row, now)
            if status is None:
              continue
            ctx = row["entryContext"]
            if status == "missed":
              ctx["leaseLow"] = ctx["leaseHigh"] = None
              settled += 1
              continue
            got = leases.get((row.get("symbol"), int(row.get("ts") or 0)))
            if got is not None:
              ctx["leaseLow"], ctx["leaseHigh"] = got
              settled += 1
          except Exception as exc:
            logger.warning("PROBE LEASE stamp failed for %s (%s) — other probes unaffected",
                           row.get("symbol") if isinstance(row, dict) else "?", exc)
      if settled:
        self._write(data)
    return settled

  def signal_probes(self, limit: int = 200) -> list[Dict[str, Any]]:
    """Entry signals carrying a market-price-at-signal stamp, for edge measurement.

    ``limit=0`` means EVERYTHING retained, which is what the entry gates and the dashboard want.
    Asking for a fixed slice is how a verdict gets silently un-learned: retention is already bounded
    per family, so a second truncation at read time can only throw away evidence the writer chose to
    keep. This union also returns MORE rows than ``MAX_SIGNAL_PROBES`` (it folds in legacy
    trades-derived rows), so passing that constant as the limit quietly dropped the oldest of them.

    ``gate_probes`` are deliberately NOT part of this union: a call a gate refused, or a side the model
    never proposed, is gate-scoreboard evidence only and must never move a family verdict or its stake.
    """
    with self._lock:
      data = self._read()
    out = []
    # The dedicated bucket is the current source (every direction call); trades-derived rows are the
    # legacy shape recorded before probes were decoupled from order placement. Union so no history is
    # lost, deduped on (symbol, ts) since a placed order appears in both.
    seen = set()
    for row in list(data.get("signal_probes") or []) + list(data.get("trades") or []):
      ctx = row.get("entryContext") if isinstance(row, dict) else None
      if not (isinstance(ctx, dict) and ctx.get("marketPriceAtSignal")):
        continue
      key = (row.get("symbol"), int(row.get("ts") or 0))
      if key in seen:
        continue
      seen.add(key)
      out.append(row)
    out.sort(key=lambda r: int(r.get("ts") or 0))
    lim = int(limit or 0)
    return out if lim <= 0 else out[-lim:]

  def limit_entry_records(self) -> list[Dict[str, Any]]:
    """Every retained bot-placed limit entry, filled or not, oldest→newest (deep copies).

    Exactly the records ``performanceSummary.limitFillRate`` counts — same prune, same predicate — so
    `edge.execution_map` built from these reproduces that rate in its ``totals`` by construction.
    """
    with self._lock:
      data = self._prune(self._read())
      rows = [copy.deepcopy(t) for t in (data.get("trades") or []) if is_limit_entry_record(t)]
    return rows

  def recent_fills(self, limit: int = 100) -> list[Dict[str, Any]]:
    """Recent FILLED entry orders, oldest→newest — the sample the friction estimate calibrates on.

    Only rows that actually filled carry a usable (planned price, achieved fill price) pair, which is
    what ``edge.measured_slippage_pct`` turns into a measured slippage instead of a hand-set constant.
    """
    with self._lock:
      data = self._read()
    rows = [
      t for t in (data.get("trades") or [])
      if isinstance(t, dict) and t.get("filled") and t.get("fillPrice") is not None
    ]
    return rows[-max(1, int(limit)):]

  def get_seen_close_ids(self) -> list[str]:
    """Exchange close-position IDs already recorded as decisions (persists across restarts)."""
    with self._lock:
      data = self._read()
    return [str(x) for x in (data.get("seen_close_ids") or [])]

  def record_seen_close_id(self, close_id: str, cap: int = 200) -> None:
    """Persist a recorded close ID so a restart can't double-record the same close."""
    cid = str(close_id or "").strip()
    if not cid:
      return
    with self._lock:
      data = self._read()
      ids = [str(x) for x in (data.get("seen_close_ids") or [])]
      if cid not in ids:
        ids.append(cid)
        data["seen_close_ids"] = ids[-cap:]
        self._write(data)

  def get_seen_fill_ids(self) -> list[str]:
    """Exchange fill IDs already delivered to the agent (persists across polls/restarts)."""
    with self._lock:
      data = self._read()
    return [str(x) for x in (data.get("seen_fill_ids") or [])]

  def mark_order_filled(
    self,
    order_id: Any = None,
    client_oid: Any = None,
    *,
    fill_ts: Any = None,
    fill_price: Any = None,
    fill_size: Any = None,
  ) -> bool:
    """Promote a submitted-order record to executed without using it for position reconstruction."""
    refs = {
      str(value) for value in (order_id, client_oid)
      if value not in (None, "")
    }
    if not refs:
      return False
    with self._lock:
      data = self._prune(self._read())
      trades = data.get("trades", []) or []
      for trade in reversed(trades):
        trade_refs = {
          str(value) for value in (trade.get("orderId"), trade.get("clientOid"))
          if value not in (None, "")
        }
        if refs & trade_refs:
          trade["filled"] = True
          try:
            raw_ts = float(fill_ts)
            if raw_ts > 1e15:
              raw_ts /= 1e9
            elif raw_ts > 1e12:
              raw_ts /= 1e3
            if raw_ts > 0:
              trade["fillTs"] = int(raw_ts)
              # Trade caps/cooldowns follow the execution day, not the submission day. A limit
              # submitted before UTC midnight can fill afterward.
              trade["day"] = int(raw_ts // 86400)
          except (TypeError, ValueError):
            pass
          try:
            if fill_price not in (None, ""):
              trade["fillPrice"] = float(fill_price)
          except (TypeError, ValueError):
            pass
          try:
            if fill_size not in (None, ""):
              trade["fillSize"] = float(fill_size)
          except (TypeError, ValueError):
            pass
          self._write(data)
          return True
    return False

  def entry_context_for_position(
    self,
    symbol: str,
    position_open_time: Any,
    position_side: str | None,
    *,
    window_seconds: int = 7200,
  ) -> Optional[Dict[str, Any]]:
    """Return the nearest same-direction filled entry intent for a position lifecycle."""
    open_ms = self._position_open_time_ms({"positionOpenTime": position_open_time})
    if open_ms is None:
      return None
    open_sec = open_ms / 1000.0
    wanted_side = str(position_side or "").lower()
    with self._lock:
      trades = list(self._read().get("trades") or [])
    candidates: list[tuple[float, Dict[str, Any]]] = []
    for trade in trades:
      if trade.get("symbol") != _normalize_symbol(symbol) or trade.get("filled") is not True:
        continue
      context = trade.get("entryContext")
      if not isinstance(context, dict):
        continue
      fallback_side = "long" if str(trade.get("side") or "").lower() == "buy" else "short"
      trade_side = str(context.get("positionSide") or fallback_side).lower()
      if wanted_side in {"long", "short"} and trade_side != wanted_side:
        continue
      event_ts = float(trade.get("fillTs") or trade.get("ts") or 0.0)
      delta = open_sec - event_ts
      if -300 <= delta <= max(0, int(window_seconds)):
        candidates.append((abs(delta), trade))
    if not candidates:
      return None
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 1.0:
      return None
    trade = candidates[0][1]
    result = copy.deepcopy(trade["entryContext"])
    result.setdefault("entryOrderId", trade.get("orderId"))
    result.setdefault("entryClientOid", trade.get("clientOid"))
    result.setdefault("fillTs", trade.get("fillTs"))
    result.setdefault("fillPrice", trade.get("fillPrice"))
    return result

  def record_seen_fill_id(self, fill_id: str, cap: int = 1000) -> None:
    fid = str(fill_id or "").strip()
    if not fid:
      return
    with self._lock:
      data = self._read()
      ids = [str(x) for x in (data.get("seen_fill_ids") or [])]
      if fid not in ids:
        ids.append(fid)
        data["seen_fill_ids"] = ids[-cap:]
        self._write(data)

  def queue_agent_event(self, kind: str, event_id: str, payload: Dict[str, Any]) -> bool:
    """Persist a fill/close/expiry until a successful model run acknowledges that exact event."""
    if kind not in {"spot_fills", "futures_fills", "closed_positions", "auto_triggers", "entry_expired"}:
      return False
    eid = str(event_id or "").strip()
    if not eid or not isinstance(payload, dict):
      return False
    with self._lock:
      data = self._prune(self._read())
      pending = data.setdefault("pending_agent_events", [])
      if any(str(event.get("id") or "") == eid for event in pending if isinstance(event, dict)):
        return False
      pending.append({"id": eid, "kind": kind, "payload": copy.deepcopy(payload), "ts": int(time.time())})
      data["pending_agent_events"] = pending[-MAX_PENDING_AGENT_EVENTS:]
      self._write(data)
      return True

  def get_pending_agent_events(self) -> list[Dict[str, Any]]:
    with self._lock:
      data = self._prune(self._read())
      return copy.deepcopy(data.get("pending_agent_events", []) or [])

  def acknowledge_agent_events(self, event_ids: list[str]) -> list[Dict[str, Any]]:
    ids = {str(event_id) for event_id in event_ids if str(event_id or "").strip()}
    if not ids:
      return []
    with self._lock:
      data = self._prune(self._read())
      pending = data.get("pending_agent_events", []) or []
      acknowledged = [event for event in pending if str(event.get("id") or "") in ids]
      data["pending_agent_events"] = [
        event for event in pending if str(event.get("id") or "") not in ids
      ]
      self._write(data)
      return copy.deepcopy(acknowledged)

  def get_agent_scheduler(self) -> Dict[str, Any]:
    """Return persisted model-cadence and adaptive price-noise state."""
    with self._lock:
      data = self._prune(self._read())
      return copy.deepcopy(_sanitize_agent_scheduler(data.get("agent_scheduler")))

  def save_agent_scheduler(self, state: Dict[str, Any]) -> Dict[str, Any]:
    """Atomically persist scheduler state without exposing arbitrary memory keys."""
    normalized = _sanitize_agent_scheduler(state)
    with self._lock:
      data = self._prune(self._read())
      if data.get("agent_scheduler") != normalized:
        data["agent_scheduler"] = normalized
        self._write(data)
      return copy.deepcopy(normalized)

  def observe_open_interest(
    self,
    symbol: str,
    value: float,
    *,
    price: Optional[float] = None,
    now: Optional[int] = None,
    min_age_sec: int = 300,
    change_threshold: float = 0.005,
    price_change_threshold: float = 0.001,
  ) -> Dict[str, Any]:
    """Compare timestamp-aligned OI and price observations; never infer either from volume."""
    ts = int(now if now is not None else time.time())
    sym = _normalize_symbol(symbol)
    current = float(value)
    current_price = float(price) if price is not None else None
    if current_price is not None and current_price <= 0:
      current_price = None
    if current <= 0:
      return {"trend": None, "changePct": None, "priceTrend": None, "priceChangePct": None, "ageSec": None}
    with self._lock:
      data = self._read()
      observations = data.setdefault("open_interest_observations", {})
      previous = observations.get(sym) if isinstance(observations, dict) else None
      if not isinstance(previous, dict) or not previous.get("value") or not previous.get("ts"):
        observations[sym] = {"value": current, "price": current_price, "ts": ts}
        data["open_interest_observations"] = observations
        self._write(data)
        return {"trend": None, "changePct": None, "priceTrend": None, "priceChangePct": None, "ageSec": None}

      prior = float(previous["value"])
      age = max(0, ts - int(previous["ts"]))
      change = (current - prior) / prior if prior > 0 else 0.0
      trend = None
      prior_price = float(previous.get("price")) if previous.get("price") not in (None, "") else None
      price_change = (
        (current_price - prior_price) / prior_price
        if current_price is not None and prior_price is not None and prior_price > 0
        else None
      )
      price_trend = None
      if age >= max(1, int(min_age_sec)):
        threshold = max(0.0, float(change_threshold))
        trend = "up" if change >= threshold else "down" if change <= -threshold else "flat"
        if price_change is not None:
          price_threshold = max(0.0, float(price_change_threshold))
          price_trend = "up" if price_change >= price_threshold else "down" if price_change <= -price_threshold else "flat"
        observations[sym] = {"value": current, "price": current_price, "ts": ts}
        data["open_interest_observations"] = observations
        self._write(data)
      return {
        "trend": trend,
        "changePct": change * 100.0,
        "priceTrend": price_trend,
        "priceChangePct": price_change * 100.0 if price_change is not None else None,
        "ageSec": age,
      }

  @staticmethod
  def _is_realized_close(action: str) -> bool:
    """Return True if the action represents a realized trade close, not a hold/manage snapshot."""
    a = action.lower()
    # A REVIEW is commentary the agent writes *after* a close, and it copies the closed trade's pnl.
    # Counting it books the same trade twice. This is the second time this class of bug has bitten —
    # the note below records the "hold-close-only" case, and on 2026-08-08 `close_reviewed` /
    # `close_reviewed_hold` slipped through `startswith("close_")` the same way. Consequences reach
    # well past the dashboard's duplicated rows: win rate read 36.7% instead of 40.0%, and the loss
    # streak read 3 instead of 2 — with CB_MAX_CONSECUTIVE_LOSSES=3 that phantom row is the difference
    # between tripping a 120-minute trading halt and not. Reviews never execute, so exclude them by
    # meaning rather than by name.
    if "review" in a:
      return False
    if "triggered" in a:
      return True
    # Exact execution actions only.  The old substring test counted labels such as
    # "hold-close-only" as a closed trade and polluted win rate, streak and Kelly calculations.
    if a in {
      "futures_close", "spot_close", "futures_sell", "futures_buy", "spot_sell",
      "cut_loss", "early_cut", "profit_lock_close", "reduce_position",
    }:
      return True
    return a.startswith("close_") or a.endswith("_closed")

  @staticmethod
  def _pnl_stats(decisions: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute win/loss stats from a list of decision dicts (realized closes only)."""
    decisions = MemoryStore._authoritative_realized_rows(decisions)
    realized: list[float] = []
    missed_profits: list[float] = []
    unnecessary_losses: list[float] = []
    for d in decisions:
      if not MemoryStore._is_realized_close(d.get("action") or ""):
        continue
      pnl = d.get("pnl")
      if pnl is not None:
        try:
          pnl_f = float(pnl)
        except (TypeError, ValueError):
          continue
        realized.append(pnl_f)
        peak = d.get("peakPnl")
        trough = d.get("troughPnl")
        if peak is not None:
          try:
            peak_f = float(peak)
          except (TypeError, ValueError):
            peak_f = None
        else:
          peak_f = None
        if peak_f is not None and peak_f > pnl_f and peak_f > 0:
          missed_profits.append(round(peak_f - pnl_f, 4))
        if trough is not None and pnl_f <= 0:
          try:
            trough_f = float(trough)
          except (TypeError, ValueError):
            trough_f = None
          if trough_f is not None and trough_f < pnl_f:
            unnecessary_losses.append(round(pnl_f - trough_f, 4))
    wins = [p for p in realized if p > 0]
    losses = [p for p in realized if p < 0]
    breakeven = [p for p in realized if p == 0]
    total = sum(realized)
    # Win rate is over decided closes only — break-even (pnl == 0) trades are neither wins nor
    # losses, so counting them in the denominator understates the rate. closedWithPnl keeps its
    # original meaning (every realized close, break-evens included) for the trading loop.
    decided = len(wins) + len(losses)
    wr = len(wins) / decided if decided else 0.0
    stats: Dict[str, Any] = {
      "closedWithPnl": len(realized),
      "wins": len(wins),
      "losses": len(losses),
      "breakeven": len(breakeven),
      "winRate": round(wr, 3),
      "totalRealizedPnl": round(total, 4),
      "avgWin": round(sum(wins) / len(wins), 4) if wins else 0.0,
      "avgLoss": round(sum(losses) / len(losses), 4) if losses else 0.0,
      "bestTrade": round(max(wins), 4) if wins else 0.0,
      "worstTrade": round(min(losses), 4) if losses else 0.0,
    }
    if missed_profits:
      stats["missedProfitCount"] = len(missed_profits)
      stats["totalMissedProfit"] = round(sum(missed_profits), 4)
      stats["avgMissedProfit"] = round(sum(missed_profits) / len(missed_profits), 4)
    if unnecessary_losses:
      stats["unnecessaryLossCount"] = len(unnecessary_losses)
      stats["totalUnnecessaryLoss"] = round(sum(unnecessary_losses), 4)
    return stats

  @staticmethod
  def _authoritative_realized_rows(decisions: list[Dict[str, Any]], window_sec: int = 1800) -> list[Dict[str, Any]]:
    """Return realized rows without double-counting pre-close estimates and exchange final PnL.

    A market close is logged immediately with an estimated PnL.  KuCoin then reports the closed
    position with authoritative cumulative PnL (including earlier partial reductions).  Counting
    both overstated the ZEC lifecycle by +0.5606 USDT. Position identity/open time now determines
    whether a local close belongs to the exchange result, avoiding collisions after rapid re-entry.
    """
    rows = [
      d for d in decisions
      if isinstance(d, dict)
      and MemoryStore._is_realized_close(str(d.get("action") or ""))
      and d.get("pnl") is not None
    ]
    rows.sort(key=lambda d: d.get("ts") or 0)
    triggered = [d for d in rows if "triggered" in str(d.get("action") or "").lower()]

    def _has_execution_evidence(d: Dict[str, Any]) -> bool:
      return any(d.get(k) is not None for k in ("closeType", "exitPrice", "realizedR"))

    def _num(value: Any) -> float | None:
      try:
        return float(value) if value is not None else None
      except (TypeError, ValueError):
        return None

    kept: list[Dict[str, Any]] = []
    for row in rows:
      action = str(row.get("action") or "").lower()
      if action.startswith("futures_") and "triggered" not in action:
        ts = int(row.get("ts") or 0)
        if any(
          later.get("symbol") == row.get("symbol")
          and 0 <= int(later.get("ts") or 0) - ts <= window_sec
          and MemoryStore._same_position_lifecycle(row, later, window_sec=window_sec)
          for later in triggered
        ):
          continue
      # Defence in depth for every NON-triggered row that carries no execution evidence. This bug class
      # has now bitten three times under three different action names — "hold-close-only",
      # `close_reviewed` (2026-08-08) and `close_short` (2026-08-30) — because the rule above keys on
      # the name `futures_*`. Keying on the SHAPE instead (no closeType/exitPrice/realizedR, shadowing
      # a triggered close on the same symbol nearby) catches the next name too, in either time
      # direction, since the duplicate may be written before or after the authoritative report.
      if "triggered" not in action and not _has_execution_evidence(row):
        ts = int(row.get("ts") or 0)
        pnl = _num(row.get("pnl"))

        def _shadows(real: Dict[str, Any]) -> bool:
          """True when `real` is the exchange's authoritative report of the same close as `row`."""
          if real.get("symbol") != row.get("symbol") or not _has_execution_evidence(real):
            return False
          if abs(int(real.get("ts") or 0) - ts) > window_sec:
            return False
          rp = _num(real.get("pnl"))
          if rp is None or pnl is None:
            return False
          if abs(rp - pnl) < 1e-9:
            return True                      # a verbatim copy (the `close_reviewed` family)
          if rp == 0 or (rp > 0) != (pnl > 0):
            return False                     # opposite signs are two different trades, never one echo
          # ...or the bot's own pre-close ESTIMATE, which differs slightly from the exchange's final
          # figure. Matching only on an exact value missed exactly that case: DASH-USDT on 2026-08-30
          # logged `close_short` at -0.0465 and KuCoin reported -0.0412 twenty-five seconds later, and
          # both were booked. Same symbol, same direction of PnL, same minute, within half of each
          # other is one close reported twice, not two trades.
          if abs(rp - pnl) / abs(rp) < 0.5:
            return True
          # The half-of-each-other band assumes the estimate was roughly right. When the row also names
          # no position at all it is not a trade record but the agent narrating a close it believed it
          # was making, and its figure can be arbitrarily wrong because nothing in it was ever
          # reconciled: NEAR-USDT on 2026-09-01 had the bracket fire at +0.00335 and the agent log its
          # own reduce-only close at +0.00666 five seconds later — 99% apart, so the band let it
          # through and the trade was booked twice. It surfaced as a phantom bar on the dashboard's
          # outcome chart with no matching card in "recently closed" (that panel needs entry/exit
          # prices, which an echo has never had), and it corrupts realized PnL and every rolling stat.
          #
          # So for an anonymous row the test widens from "roughly equal" to "the same order of
          # magnitude", which is the real claim: two reports of ONE close can disagree over fees,
          # partial reductions and an estimate taken a few seconds early, but not by 10x. Two genuinely
          # different trades on one symbol do differ by that much — -1.50 next to -0.04 stays two.
          if not MemoryStore._has_position_identity(row):
            larger, smaller = max(abs(rp), abs(pnl)), min(abs(rp), abs(pnl))
            return smaller > 0 and larger / smaller < 10.0
          return False

        if any(_shadows(real) for real in rows):
          continue
      kept.append(row)
    return MemoryStore._dedupe_realized(kept, window_sec=window_sec)

  def performance_summary(self) -> Dict[str, Any]:
    """Compute win/loss stats from recorded decisions, split by venue and paper/live."""
    with self._lock:
      data = self._prune(self._read())
      all_trade_records = data.get("trades", [])
      decisions = data.get("decisions", [])
    trades = [trade for trade in all_trade_records if trade.get("filled") is not False]
    submissions = [trade for trade in all_trade_records if trade.get("filled") is False]

    def _limit_execution_stats(records: list[Dict[str, Any]]) -> Dict[str, Any]:
      limit_records = [trade for trade in records if is_limit_entry_record(trade)]
      filled_limits = [trade for trade in limit_records if trade.get("filled") is True]
      return {
        "limitOrdersSubmitted": len(limit_records),
        "limitOrdersFilled": len(filled_limits),
        "limitFillRate": round(len(filled_limits) / len(limit_records), 3) if limit_records else None,
      }

    execution_stats = _limit_execution_stats(all_trade_records)

    if not trades:
      return {
        "totalTrades": 0,
        "orderSubmissions": len(submissions),
        **execution_stats,
        "message": "No executed trade history yet.",
      }

    overall = self._pnl_stats(decisions)
    overall["totalTrades"] = len(trades)
    overall["orderSubmissions"] = len(submissions)
    overall.update(execution_stats)

    spot_decisions = [d for d in decisions if (d.get("action") or "").startswith("spot_")]
    futures_decisions = [d for d in decisions if (d.get("action") or "").startswith("futures_")]
    spot_trades = [t for t in trades if t.get("venue", "spot") == "spot"]
    futures_trades = [t for t in trades if t.get("venue", "spot") == "futures"]

    def _venue_block(venue_decisions: list, venue_trades: list, venue_submissions: list) -> Dict[str, Any]:
      block = self._pnl_stats(venue_decisions)
      block["totalTrades"] = len(venue_trades)
      block["orderSubmissions"] = len(venue_submissions)
      venue = venue_trades + venue_submissions
      block.update(_limit_execution_stats(venue))
      live_d = [d for d in venue_decisions if not d.get("paper", False)]
      paper_d = [d for d in venue_decisions if d.get("paper", False)]
      live_t = [t for t in venue_trades if not t.get("paper", False)]
      paper_t = [t for t in venue_trades if t.get("paper", False)]
      live_stats = self._pnl_stats(live_d)
      live_stats["totalTrades"] = len(live_t)
      paper_stats = self._pnl_stats(paper_d)
      paper_stats["totalTrades"] = len(paper_t)
      block["live"] = live_stats
      block["paper"] = paper_stats
      return block

    spot_submissions = [t for t in submissions if t.get("venue", "spot") == "spot"]
    futures_submissions = [t for t in submissions if t.get("venue", "spot") == "futures"]
    overall["spot"] = _venue_block(spot_decisions, spot_trades, spot_submissions)
    overall["futures"] = _venue_block(futures_decisions, futures_trades, futures_submissions)
    return overall

  def reset_limits(self, current_usdt: float, scope: str = "total") -> Dict[str, Any]:
    """Reset daily drawdown baseline to current_usdt."""
    with self._lock:
      data = self._read()
      now = int(time.time())
      day_key = int(now // 86400)
      limits_all = data.get("limits") if isinstance(data.get("limits"), dict) else {}
      limits = {
        "day": day_key,
        "baselineUsdt": float(current_usdt or 0.0),
        "currentUsdt": float(current_usdt or 0.0),
        "drawdownPct": 0.0,
        "updated": now,
      }
      limits_all[scope] = limits
      data["limits"] = limits_all
      self._write(data)
      return limits

  def save_fee_info(self, spot_taker: float, spot_maker: float, futures_taker: float | None = None, futures_maker: float | None = None) -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      now = int(time.time())
      entry = {
        "spot_taker": float(spot_taker),
        "spot_maker": float(spot_maker),
        "futures_taker": float(futures_taker) if futures_taker is not None else None,
        "futures_maker": float(futures_maker) if futures_maker is not None else None,
        "ts": now,
      }
      data.setdefault("fees", [])
      data["fees"].append(entry)
      self._write(data)
      return entry

  def latest_fees(self) -> Optional[Dict[str, Any]]:
    with self._lock:
      data = self._prune(self._read())
      fees = data.get("fees") or []
      return fees[-1] if fees else None

  def add_temporary_note(self, content: str, author: str = "Supervisor") -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      entry = {"content": content, "author": author, "ts": int(time.time())}
      data.setdefault("supervisor_notes_temporary", [])
      data["supervisor_notes_temporary"].append(entry)
      self._write(data)
      return entry

  def add_permanent_note(self, content: str, author: str = "Supervisor") -> Dict[str, Any]:
    with self._lock:
      data = self._prune(self._read())
      entry = {"content": content, "author": author, "ts": int(time.time())}
      data.setdefault("supervisor_notes_permanent", [])
      data["supervisor_notes_permanent"].append(entry)
      self._write(data)
      return entry

  def consume_temporary_notes(self) -> list[Dict[str, Any]]:
    """Return all temporary notes and delete them atomically."""
    with self._lock:
      data = self._read()
      notes = list(data.get("supervisor_notes_temporary") or [])
      if notes:
        data["supervisor_notes_temporary"] = []
        self._write(data)
      return notes

  def get_permanent_notes(self) -> list[Dict[str, Any]]:
    with self._lock:
      data = self._read()
      return list(data.get("supervisor_notes_permanent") or [])

  def list_all_notes(self) -> Dict[str, Any]:
    with self._lock:
      data = self._read()
      return {
        "temporary": list(data.get("supervisor_notes_temporary") or []),
        "permanent": list(data.get("supervisor_notes_permanent") or []),
      }

  def delete_permanent_note(self, index: int) -> Dict[str, Any]:
    with self._lock:
      data = self._read()
      notes = data.get("supervisor_notes_permanent") or []
      if index < 0 or index >= len(notes):
        return {"error": f"Invalid index {index}; {len(notes)} permanent notes exist"}
      removed = notes.pop(index)
      data["supervisor_notes_permanent"] = notes
      self._write(data)
      return {"deleted": removed}

  def kelly_fraction(self, venue: str | None = None, lookback: int = 50) -> float:
    """Compute quarter-Kelly fraction from recent trade performance.
    Returns a sizing fraction between 0.01 and 0.25."""
    with self._lock:
      data = self._prune(self._read())
      decisions = data.get("decisions", [])
    if venue:
      prefix = f"{venue}_"
      decisions = [d for d in decisions if (d.get("action") or "").startswith(prefix)]
    decisions = MemoryStore._authoritative_realized_rows(decisions)
    decisions = sorted(decisions, key=lambda d: d.get("ts", 0))[-lookback:]
    realized = []
    for d in decisions:
      if not MemoryStore._is_realized_close(str(d.get("action") or "")):
        continue
      pnl = d.get("pnl")
      if pnl is not None:
        try:
          realized.append(float(pnl))
        except (TypeError, ValueError):
          continue
    if len(realized) < 10:
      return 0.05
    wins = [p for p in realized if p > 0]
    losses = [p for p in realized if p < 0]
    if not wins or not losses:
      return 0.05
    win_rate = len(wins) / len(realized)
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    if avg_loss == 0:
      return 0.25
    reward_risk = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / reward_risk
    return max(0.01, min(0.25, kelly * 0.25))

  def consecutive_losses(self, venue: str | None = None) -> int:
    """Count current streak of consecutive losing CLOSED trades (most recent first).

    Only counts decisions that represent realized closes (TP/SL triggered or
    explicit close orders). Excludes 'manage', 'hold', 'decline' decisions
    whose pnl field reflects unrealized state, not a realized loss.
    """
    with self._lock:
      data = self._prune(self._read())
      decisions = data.get("decisions", [])
    if venue:
      prefix = f"{venue}_"
      decisions = [d for d in decisions if (d.get("action") or "").startswith(prefix)]
    decisions = MemoryStore._authoritative_realized_rows(decisions)
    decisions = sorted(decisions, key=lambda d: d.get("ts", 0), reverse=True)
    streak = 0
    for d in decisions:
      action = (d.get("action") or "").lower()
      # Only count realized closes — skip manage/hold/decline regardless of pnl
      if not MemoryStore._is_realized_close(action):
        continue
      pnl = d.get("pnl")
      if pnl is None:
        continue
      try:
        if float(pnl) < 0:
          streak += 1
        else:
          break
      except (TypeError, ValueError):
        continue
    return streak

  def last_trade_time(self, symbol: str) -> int | None:
    """Return the timestamp of the most recent trade (any side) for a symbol, or None."""
    sym = _normalize_symbol(symbol)
    with self._lock:
      data = self._prune(self._read())
      trades = data.get("trades", [])
    for t in sorted(trades, key=lambda t: t.get("ts", 0), reverse=True):
      if t.get("symbol") == sym and t.get("filled") is not False:
        return t.get("fillTs") or t.get("ts")
    return None

  def last_loss_time(self, symbol: str) -> int | None:
    """Return the timestamp of the most recent realized losing close for a symbol, or None.

    Only counts realized closes (triggered TP/SL or explicit close orders),
    not 'manage'/'hold' decisions whose pnl reflects unrealized state.
    """
    sym = _normalize_symbol(symbol)
    with self._lock:
      data = self._prune(self._read())
      decisions = data.get("decisions", [])
    decisions = MemoryStore._authoritative_realized_rows(decisions)
    for d in sorted(decisions, key=lambda d: d.get("ts", 0), reverse=True):
      if d.get("symbol") != sym:
        continue
      action = (d.get("action") or "").lower()
      if not MemoryStore._is_realized_close(action):
        continue
      pnl = d.get("pnl")
      if pnl is None:
        continue
      try:
        if float(pnl) < 0:
          return d.get("ts")
        else:
          return None
      except (TypeError, ValueError):
        continue
    return None

  def recent_win_close(self, symbol: str, within_minutes: float) -> Optional[Dict[str, Any]]:
    """Most recent realized *winning* close for a symbol within the window, or None.

    Powers the no-chase guard: after taking profit, re-entering the same direction at a
    worse price is blocked for a cooldown. Returns {ts, pnl, exitPrice, closeType}.
    """
    sym = _normalize_symbol(symbol)
    cutoff = int(time.time()) - int(max(0.0, within_minutes) * 60)
    with self._lock:
      data = self._prune(self._read())
      decisions = data.get("decisions", [])
    decisions = MemoryStore._authoritative_realized_rows(decisions)
    for d in sorted(decisions, key=lambda d: d.get("ts", 0), reverse=True):
      ts = d.get("ts") or 0
      if ts < cutoff:
        break  # sorted newest-first: nothing else is within the window
      if d.get("symbol") != sym:
        continue
      if not MemoryStore._is_realized_close((d.get("action") or "").lower()):
        continue
      pnl = d.get("pnl")
      if pnl is None:
        continue
      try:
        if float(pnl) <= 0:
          continue
      except (TypeError, ValueError):
        continue
      return {"ts": ts, "pnl": float(pnl), "exitPrice": d.get("exitPrice"), "closeType": d.get("closeType")}
    return None

  def portfolio_heat(self, stop_distances: Dict[str, float], total_equity: float) -> float:
    """Compute portfolio heat = sum of capital at risk / total equity * 100.
    stop_distances: {symbol: usd_amount_at_risk}"""
    if total_equity <= 0:
      return 0.0
    total_risk = sum(abs(v) for v in stop_distances.values())
    return round(total_risk / total_equity * 100, 2)
