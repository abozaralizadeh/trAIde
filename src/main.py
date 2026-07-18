from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import signal
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict

from agents import set_default_openai_client
from agents.tracing import (get_trace_provider)
from .agent import TradingSnapshot, run_trading_agent, setup_tracing, setup_lstracing, _build_openai_client
from .config import load_config
from .dashboard_publisher import DashboardPublisher
from .kucoin import KucoinClient, KucoinFuturesClient, KucoinAccount, KucoinTicker
from .memory import MemoryStore
from .protection import ProtectionManager
from .safety import TradingSafetyState
from .telegram import TelegramNotifier
from .utils import normalize_symbol

logger = logging.getLogger(__name__)

_AGENT_RUN_TIMEOUT_SEC = 20 * 60
_AGENT_SHUTDOWN_GRACE_SEC = 30
_PRICE_NOISE_MULTIPLIER = 4.0
# Ceiling on the adaptive trigger, as a multiple of the base trigger. 2.0 keeps worst-case blindness
# bounded (any move >= 2× the base trigger always earns a fresh model look) — biased toward safety;
# raise via PRICE_TRIGGER_MAX_MULTIPLIER to save more tokens. Overridable per call from config.
_PRICE_TRIGGER_MAX_MULTIPLIER = 2.0
_PRICE_NOISE_ALPHA = 0.20
_PRICE_NOISE_MIN_SAMPLES = 3


class IncompleteFuturesSnapshot(RuntimeError):
  def __init__(self, failures: list[str], overview: dict | None, positions: list, stops: list) -> None:
    super().__init__("Incomplete futures snapshot (" + "; ".join(failures) + ")")
    self.failures = failures
    self.overview = overview
    self.positions = positions
    self.stops = stops


async def _run_in_daemon_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
  """Await blocking work in a daemon thread so a timed-out model cannot hold process shutdown."""
  loop = asyncio.get_running_loop()
  future = loop.create_future()

  def _worker() -> None:
    try:
      outcome = (True, fn(*args, **kwargs))
    except BaseException as exc:  # propagate model/tool failures back to the loop
      outcome = (False, exc)

    def _settle() -> None:
      if future.done():
        return
      if outcome[0]:
        future.set_result(outcome[1])
      else:
        future.set_exception(outcome[1])

    try:
      loop.call_soon_threadsafe(_settle)
    except RuntimeError:
      pass  # event loop already closed; daemon worker has no remaining authority

  threading.Thread(target=_worker, daemon=True, name="trAIde-agent").start()
  return await future


def _adaptive_agent_cooldown(
  *,
  flat_cooldown_sec: float,
  active_cooldown_sec: float,
  book_active: bool,
  new_events_count: int,
  trigger_move_pcts: list[float],
  price_trigger_pct: float,
) -> float:
  """Return a model-call cooldown that falls continuously as a flat-market move gets urgent.

  The configured flat value remains the quiet-market/cost ceiling and the active value remains
  the floor.  A move exactly at the trigger halves the quiet cooldown; larger moves and breadth
  (several symbols moving together) accelerate it further without another setting to tune.
  """
  flat = max(0.0, float(flat_cooldown_sec))
  active = max(0.0, min(flat, float(active_cooldown_sec)))
  if book_active or new_events_count:
    return active
  threshold = float(price_trigger_pct)
  moves = [abs(float(move)) for move in trigger_move_pcts if float(move) > 0]
  if threshold <= 0 or not moves or flat <= 0:
    return flat
  # At 1x threshold: flat/2; at 2x: flat/5; at 3x: flat/10. Multiple triggered
  # symbols add breadth to the urgency score, so a market-wide move is reviewed sooner.
  urgency_sq = (max(moves) / threshold) ** 2 * len(moves)
  return max(active, min(flat, flat / (1.0 + urgency_sq)))


def _productivity_adjusted_flat_cooldown(
  flat_cooldown_sec: float,
  unproductive_runs: int,
  max_multiplier: float = 1.0,
) -> float:
  """Optionally back off no-action deliberation; disabled when max_multiplier is 1."""
  base = max(0.0, float(flat_cooldown_sec))
  if base <= 0:
    return 0.0
  cap = max(1.0, float(max_multiplier))
  exponent = max(0, int(unproductive_runs))
  multiplier = min(cap, 2.0 ** exponent)
  return base * multiplier


def _adaptive_price_trigger_threshold(
  base_trigger_pct: float,
  noise_ewma_pct: float,
  samples: int,
  *,
  noise_multiplier: float = _PRICE_NOISE_MULTIPLIER,
  max_multiplier: float = _PRICE_TRIGGER_MAX_MULTIPLIER,
) -> float:
  """Raise the model trigger above a symbol's ordinary poll-to-poll noise.

  The configured threshold is always the floor. After a few observations, routine volatility must be
  exceeded by ``noise_multiplier`` times its EWMA, capped at ``max_multiplier`` times the configured
  floor so a genuinely structural move can never be suppressed indefinitely (the cap bounds
  worst-case blindness; the default keeps it tight for safety).
  """
  base = max(0.0, float(base_trigger_pct))
  if base <= 0 or int(samples) < _PRICE_NOISE_MIN_SAMPLES:
    return base
  noise_threshold = max(0.0, float(noise_ewma_pct)) * float(noise_multiplier)
  return min(base * max(1.0, float(max_multiplier)), max(base, noise_threshold))


def _next_price_noise_ewma(
  previous_ewma_pct: float,
  observed_move_pct: float,
  base_trigger_pct: float,
  samples: int,
  *,
  max_multiplier: float = _PRICE_TRIGGER_MAX_MULTIPLIER,
) -> float:
  """Update robust price noise without letting one shock disable later model reviews."""
  base = max(0.0, float(base_trigger_pct))
  ceiling = base * max(1.0, float(max_multiplier)) if base > 0 else abs(float(observed_move_pct))
  observed = min(abs(float(observed_move_pct)), ceiling)
  if int(samples) <= 0:
    return observed
  previous = max(0.0, float(previous_ewma_pct))
  return (1.0 - _PRICE_NOISE_ALPHA) * previous + _PRICE_NOISE_ALPHA * observed


def _rebase_reviewed_price_triggers(
  pending_moves: Dict[str, float],
  pending_discrete_triggers: set[str],
  reviewed_symbols: set[str],
) -> None:
  """Discard trigger evidence measured against the pre-run anchor for reviewed symbols."""
  for symbol in reviewed_symbols:
    pending_moves.pop(symbol, None)
  stale_initials = {
    trigger
    for trigger in pending_discrete_triggers
    if trigger.startswith("initial:") and trigger.split(":", 1)[1] in reviewed_symbols
  }
  pending_discrete_triggers.difference_update(stale_initials)


def _load_active_coins(cfg, memory: MemoryStore) -> list[str]:
  if not cfg.trading.flexible_coins_enabled:
    return cfg.trading.coins

  if not memory.has_coins():
    memory.set_coins(cfg.trading.coins, reason="seed-from-config")
    return cfg.trading.coins

  coins = memory.get_coins(default=cfg.trading.coins)
  if not coins:
    memory.set_coins(cfg.trading.coins, reason="seed-from-config")
    return cfg.trading.coins
  return coins


_TICKER_FAIL_COUNTS: dict[str, int] = {}
_TICKER_FAIL_THRESHOLD = 3  # consecutive failures before removal


def _fetch_tickers(
  cfg,
  kucoin: KucoinClient,
  coins: list[str],
  memory: MemoryStore,
  preserve_symbols: set[str] | None = None,
) -> tuple[list[str], dict]:
  """Normalize symbols, fetch tickers with retry. Only remove after repeated consecutive failures."""
  normalized: list[str] = []
  seen: set[str] = set()
  for sym in coins:
    norm = normalize_symbol(sym)
    if norm and norm not in seen:
      normalized.append(norm)
      seen.add(norm)

  tickers = {}
  missing: list[str] = []
  for symbol in normalized:
    try:
      tickers[symbol] = kucoin.get_ticker(symbol)
      _TICKER_FAIL_COUNTS.pop(symbol, None)  # reset on success
    except Exception as exc:
      missing.append(symbol)
      _TICKER_FAIL_COUNTS[symbol] = _TICKER_FAIL_COUNTS.get(symbol, 0) + 1
      fail_count = _TICKER_FAIL_COUNTS[symbol]
      logger.warning("Ticker fetch failed for %s (%d/%d): %s", symbol, fail_count, _TICKER_FAIL_THRESHOLD, exc)
      if (
        cfg.trading.flexible_coins_enabled
        and fail_count >= _TICKER_FAIL_THRESHOLD
        and symbol not in (preserve_symbols or set())
      ):
        try:
          removal = memory.remove_coin(
            symbol,
            reason=f"Ticker unavailable {fail_count} consecutive times: {exc}",
            exit_plan="Auto-removed after repeated failures; re-add when symbol is confirmed available.",
          )
          logger.warning("Removed unavailable symbol from active universe after %d failures: %s", fail_count, removal.get("symbol"))
          _TICKER_FAIL_COUNTS.pop(symbol, None)
        except Exception as remove_exc:
          logger.warning("Failed to remove unavailable symbol %s: %s", symbol, remove_exc)

  if not tickers and not (preserve_symbols or set()):
    raise RuntimeError(f"No tickers available; failed symbols: {missing}")

  return list(tickers.keys()), tickers


def _discover_unlisted_holdings(kucoin: KucoinClient, spot_accounts: list, existing_tickers: dict, min_value_usd: float = 0.50) -> tuple[list[str], dict]:
  """Discover spot holdings not in the active coin list by scanning balances."""
  known_bases = {sym.split("-")[0].upper() for sym in existing_tickers if "-" in sym}
  known_bases.add("USDT")
  extra_coins: list[str] = []
  extra_tickers: dict = {}
  for acct in spot_accounts:
    cur = acct.currency.upper()
    if cur in known_bases:
      continue
    bal = float(acct.balance or 0)
    if bal <= 0:
      continue
    symbol = f"{cur}-USDT"
    try:
      ticker = kucoin.get_ticker(symbol)
      if bal * float(ticker.price) < min_value_usd:
        continue
      extra_coins.append(symbol)
      extra_tickers[symbol] = ticker
      known_bases.add(cur)
      logger.info("Discovered unlisted holding: %s (%.4f units, ~$%.2f)", symbol, bal, bal * float(ticker.price))
    except Exception:
      continue
  return extra_coins, extra_tickers


def _fetch_balances(kucoin: KucoinClient) -> tuple[list, list, list]:
  """Fetch spot trade accounts, financial accounts, and all accounts. Returns (spot, financial, all)."""
  spot_accounts = kucoin.get_trade_accounts()
  financial_accounts: list = []
  try:
    financial_accounts = kucoin.get_financial_accounts()
  except Exception as exc:
    logger.warning("Unable to fetch financial accounts: %s", exc)
  all_accounts = kucoin.get_accounts()
  return spot_accounts, financial_accounts, all_accounts


def _fetch_futures(cfg, kucoin_futures: KucoinFuturesClient | None) -> tuple[dict | None, list, list]:
  """Fetch futures account overview, positions, and stop orders. Returns (overview, positions, stops)."""
  if not (cfg.kucoin_futures.enabled and kucoin_futures):
    return None, [], []
  failures: list[str] = []
  overview = None
  positions: list = []
  stops: list = []
  try:
    overview = kucoin_futures.get_account_overview()
  except Exception as exc:
    failures.append(f"overview: {exc}")
  try:
    positions = kucoin_futures.list_positions() or []
  except Exception as exc:
    failures.append(f"positions: {exc}")
  try:
    stops = kucoin_futures.list_stop_orders(status="active") or []
  except Exception as exc:
    failures.append(f"stops: {exc}")
  if failures:
    # Never turn a partial API failure into an apparently flat/unprotected book. The outer loop
    # retries the entire snapshot and blocks all entries until exchange truth is complete again.
    raise IncompleteFuturesSnapshot(failures, overview, positions, stops)
  return overview, positions, stops


def _fill_event_id(entry: dict, venue: str) -> str:
  """Stable fill identity for poll/restart deduplication, including partial fills."""
  for key in ("tradeId", "id", "fillId"):
    value = entry.get(key)
    if value not in (None, ""):
      return f"{venue}:{value}"
  parts = [
    entry.get("orderId"), entry.get("createdAt") or entry.get("tradeCreatedAt"),
    entry.get("symbol"), entry.get("side"), entry.get("price"),
    entry.get("size") or entry.get("filledSize"),
  ]
  if not any(value not in (None, "") for value in parts):
    digest = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return f"{venue}:payload:{digest}"
  return f"{venue}:" + ":".join(str(value or "") for value in parts)


def _close_event_id(entry: dict) -> str:
  """Stable close identity; fallback includes lifecycle fields instead of open time alone."""
  value = entry.get("id") or entry.get("positionId")
  if value not in (None, ""):
    return str(value)
  parts = [
    entry.get("symbol"), entry.get("type") or entry.get("side"),
    entry.get("openTime") or entry.get("openingTimestamp"),
    entry.get("closeTime") or entry.get("updatedAt"),
  ]
  if not any(value not in (None, "") for value in parts):
    digest = hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return f"close:payload:{digest}"
  return "close:" + ":".join(str(value or "") for value in parts)


def _fetch_recent_fills(kucoin: KucoinClient, kucoin_futures: KucoinFuturesClient | None, lookback_minutes: int = 10) -> Dict:
  """Fetch fills and closed positions from the last N minutes (using KuCoin server time to avoid clock drift)."""
  now_ms = kucoin._timestamp_ms()
  cutoff_ms = now_ms - lookback_minutes * 60 * 1000
  result: Dict[str, Any] = {"spot_fills": [], "futures_fills": [], "closed_positions": [], "_errors": []}

  def _after_cutoff(entry: dict, *ts_keys: str) -> bool:
    for k in ts_keys:
      v = entry.get(k)
      if v is not None:
        ts = int(v) if int(v) > 1e12 else int(v) * 1000
        if ts >= cutoff_ms:
          return True
    return False

  def _paged(fetch: Callable[..., list], *, page_size: int, ts_keys: tuple[str, ...], label: str) -> list:
    rows_in_window: list = []
    seen_payloads: set[str] = set()
    previous_page_signature = ""
    max_pages = 20
    for page in range(1, max_pages + 1):
      rows = fetch(page=page, page_size=page_size) or []
      signature = hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
      if page > 1 and signature == previous_page_signature:
        break  # defensive: endpoint ignored page and repeated page 1
      previous_page_signature = signature
      for row in rows:
        if not isinstance(row, dict) or not _after_cutoff(row, *ts_keys):
          continue
        payload_id = hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()
        if payload_id not in seen_payloads:
          seen_payloads.add(payload_id)
          rows_in_window.append(row)
      if len(rows) < page_size:
        break
      timestamps = []
      for row in rows:
        if not isinstance(row, dict):
          continue
        for key in ts_keys:
          value = row.get(key)
          if value is None:
            continue
          try:
            ts = int(value)
            timestamps.append(ts if ts > 1e12 else ts * 1000)
          except (TypeError, ValueError):
            pass
          break
      if timestamps and min(timestamps) < cutoff_ms:
        break
    else:
      result["_errors"].append(f"{label}: pagination exceeded {max_pages} pages")
    return rows_in_window

  try:
    result["spot_fills"] = _paged(
      kucoin.get_fills, page_size=50,
      ts_keys=("createdAt", "tradeCreatedAt"), label="spot fills",
    )
  except Exception as exc:
    logger.warning("Unable to fetch spot fills: %s", exc)
    result["_errors"].append(f"spot fills: {exc}")

  if kucoin_futures:
    try:
      result["futures_fills"] = _paged(
        kucoin_futures.get_fills, page_size=50,
        ts_keys=("createdAt", "tradeCreatedAt"), label="futures fills",
      )
    except Exception as exc:
      logger.warning("Unable to fetch futures fills: %s", exc)
      result["_errors"].append(f"futures fills: {exc}")

    try:
      result["closed_positions"] = _paged(
        kucoin_futures.get_position_history, page_size=50,
        ts_keys=("closeTime", "updatedAt"), label="closed positions",
      )
    except Exception as exc:
      logger.warning("Unable to fetch closed positions: %s", exc)
      result["_errors"].append(f"closed positions: {exc}")

  return result


def _fetch_fees(kucoin: KucoinClient) -> dict:
  """Fetch base fee rates; returns empty dict on failure."""
  try:
    fee_info = kucoin.get_base_fee()
    if fee_info:
      return {
        "spot_taker": float(fee_info.get("takerFeeRate") or 0.001),
        "spot_maker": float(fee_info.get("makerFeeRate") or 0.001),
      }
  except Exception as exc:
    logger.warning("Unable to fetch base fee: %s", exc)
  return {}


def _live_extremes_map(snapshot) -> Dict[str, dict]:
  """Active-position map from live exchange truth (futures) for peak/trough PnL tracking.

  Driving `update_position_extremes` off live positions makes a symbol's extremes reset when its
  position actually closes — instead of lingering forever on a memory-reconstructed phantom long
  (futures triggered-closes are logged as decisions, not written to the trades ledger, so the
  ledger only ever accumulates buys). Futures-only: the bot trades futures exclusively; add spot
  positions here if spot trading resumes.
  """
  out: Dict[str, dict] = {}
  for p in getattr(snapshot, "futures_positions", None) or []:
    if not isinstance(p, dict):
      continue
    try:
      qty = float(p.get("currentQty") or 0)
    except (TypeError, ValueError):
      continue
    if not qty:
      continue
    sym = normalize_symbol(p.get("symbol") or "")
    if not sym:
      continue
    upnl = p.get("unrealisedPnl")
    if upnl is None:
      upnl = p.get("unrealizedPnl")
    try:
      upnl = float(upnl) if upnl is not None else None
    except (TypeError, ValueError):
      upnl = None
    out[sym] = {
      "netSize": qty,
      "unrealizedPnl": upnl,
      "positionOpenTime": p.get("openingTimestamp") or p.get("openTime"),
      "positionSide": "long" if qty > 0 else "short",
    }
  return out


def _agent_made_a_move(result: Dict) -> bool:
  """True if the agent placed any order this run (entry, close, or protective stop).

  Used to detect the 'stuck declining' state: when the agent makes no move for several
  consecutive runs, the loop forces a Research Agent handoff to refresh the coin universe.
  Pure declines/rejections and bare fund transfers do NOT count as a move.
  """
  for out in result.get("tool_results") or []:
    if not isinstance(out, dict):
      continue
    if out.get("rejected") or out.get("skipped") or out.get("error"):
      continue
    if out.get("orderId") or out.get("orderRequest"):
      return True
  return False


def _expired_bot_entry_orders(pending_orders: list[dict], expiry_minutes: float, now: float | None = None) -> list[dict]:
  """Select expired trAIde-created GTC futures entries. Pure/testable.

  KuCoin has no client-side "expire in N minutes" behavior for the submitted GTC orders.  Only
  orders tagged by this bot are eligible, so autonomous cleanup never cancels a manual/user order.
  Protective/reduce-only orders are excluded even if an API happens to return them in this list.
  """
  if expiry_minutes <= 0:
    return []
  ref = float(now if now is not None else time.time())

  def _true(value: Any) -> bool:
    return value is True or str(value or "").strip().lower() in {"1", "true", "yes"}

  expired: list[dict] = []
  for order in pending_orders or []:
    if not isinstance(order, dict):
      continue
    if not str(order.get("clientOid") or "").startswith("traide-entry-"):
      continue
    if _true(order.get("reduceOnly")) or _true(order.get("closeOrder")):
      continue
    raw_ts = order.get("createdAt") or order.get("orderTime") or order.get("created_at")
    try:
      created = float(raw_ts)
    except (TypeError, ValueError):
      continue
    if created > 1e15:  # nanoseconds
      created /= 1e9
    elif created > 1e12:  # milliseconds
      created /= 1e3
    if ref - created >= expiry_minutes * 60:
      expired.append(order)
  return expired


def build_snapshot(cfg, kucoin: KucoinClient, kucoin_futures: KucoinFuturesClient | None, memory: MemoryStore) -> TradingSnapshot:
  raw_coins = _load_active_coins(cfg, memory)
  futures_overview, futures_positions, futures_stops = _fetch_futures(cfg, kucoin_futures)
  live_futures_symbols: set[str] = set()
  for position in futures_positions:
    if not isinstance(position, dict):
      continue
    try:
      qty = float(position.get("currentQty") or 0)
    except (TypeError, ValueError):
      continue
    symbol = normalize_symbol(position.get("symbol") or "")
    if qty and symbol:
      live_futures_symbols.add(symbol)
  coins, tickers = _fetch_tickers(
    cfg, kucoin, raw_coins, memory,
    preserve_symbols=live_futures_symbols,
  )
  for symbol in sorted(live_futures_symbols):
    if symbol in tickers:
      continue
    try:
      tickers[symbol] = kucoin.get_ticker(symbol)
      coins.append(symbol)
    except Exception as exc:
      live_position = next(
        (position for position in futures_positions if normalize_symbol(position.get("symbol") or "") == symbol),
        {},
      )
      mark = float(live_position.get("markPrice") or 0) if isinstance(live_position, dict) else 0.0
      if mark > 0:
        mark_str = str(mark)
        tickers[symbol] = KucoinTicker(
          sequence="", bestAsk=mark_str, size="0", price=mark_str,
          bestBidSize="0", bestBid=mark_str, bestAskSize="0", time=kucoin._timestamp_ms(),
        )
        coins.append(symbol)
        logger.warning("Using futures mark for manageable live symbol %s because spot ticker failed: %s", symbol, exc)
      else:
        logger.warning("Live futures symbol %s has no spot ticker; management remains enabled: %s", symbol, exc)
  spot_accounts, financial_accounts, all_accounts = _fetch_balances(kucoin)

  # Discover spot holdings not in the active coin list (e.g., externally bought coins).
  extra_coins, extra_tickers = _discover_unlisted_holdings(kucoin, spot_accounts, tickers)
  coins.extend(extra_coins)
  tickers.update(extra_tickers)

  spot_stops: list[dict] = []
  try:
    spot_stops = kucoin.list_stop_orders(status="active")
  except Exception as exc:
    logger.warning("Unable to fetch spot stop orders: %s", exc)

  spot_pending: list[dict] = []
  try:
    spot_pending = kucoin.list_orders(status="active")
  except Exception as exc:
    logger.warning("Unable to fetch spot pending orders: %s", exc)

  futures_pending: list[dict] = []
  if cfg.kucoin_futures.enabled and kucoin_futures:
    try:
      futures_pending = kucoin_futures.list_orders(status="active")
    except Exception as exc:
      logger.warning("Unable to fetch futures pending orders: %s", exc)

  fees = _fetch_fees(kucoin)

  balances = list(spot_accounts)
  for fa in financial_accounts:
    balances.append(fa)

  if futures_overview:
    balances.append(
      KucoinAccount(
        id="futures",
        currency=str(futures_overview.get("currency", "USDT")),
        type="contract",
        balance=str(futures_overview.get("accountEquity") or futures_overview.get("marginBalance") or futures_overview.get("availableBalance") or "0"),
        available=str(futures_overview.get("availableBalance") or "0"),
        holds=str(futures_overview.get("frozenBalance") or "0"),
      )
    )

  return TradingSnapshot(
    coins=coins,
    tickers=tickers,
    balances=balances,
    paper_trading=cfg.trading.paper_trading,
    max_position_usd=cfg.trading.max_position_usd,
    min_confidence=cfg.trading.min_confidence,
    max_leverage=cfg.trading.max_leverage,
    futures_enabled=cfg.kucoin_futures.enabled,
    drawdown_pct=0.0,
    drawdown_pct_spot=0.0,
    drawdown_pct_futures=0.0,
    total_usdt=0.0,
    spot_accounts=spot_accounts,
    futures_account=futures_overview or {},
    futures_positions=futures_positions,
    all_accounts=all_accounts,
    spot_stop_orders=spot_stops,
    futures_stop_orders=futures_stops,
    spot_pending_orders=spot_pending,
    futures_pending_orders=futures_pending,
    financial_accounts=financial_accounts,
    fees=fees,
  )


async def trading_loop(
  stop_event: asyncio.Event | None = None,
  safety: TradingSafetyState | None = None,
) -> None:
  cfg = load_config()
  stop_event = stop_event or asyncio.Event()
  safety = safety or TradingSafetyState()
  setup_tracing(cfg)
  ls_client = setup_lstracing(cfg)
  azure_client = _build_openai_client(cfg)
  # Azure client is for inference only; tracing uses platform key via set_tracing_export_api_key in setup_tracing.
  set_default_openai_client(azure_client, use_for_tracing=False)
  kucoin = KucoinClient(cfg)
  kucoin_futures = KucoinFuturesClient(cfg) if cfg.kucoin_futures.enabled else None
  memory = MemoryStore(cfg.memory_file, retention_days=cfg.retention_days)
  scheduler_state = memory.get_agent_scheduler()
  now_ts = time.time()
  try:
    last_agent_run_ts = min(now_ts, max(0.0, float(scheduler_state.get("lastRunTs") or 0.0)))
  except (TypeError, ValueError):
    last_agent_run_ts = 0.0
  unproductive_runs = min(100, max(0, int(scheduler_state.get("unproductiveRuns") or 0)))
  reviewed_prices: Dict[str, float] = dict(scheduler_state.get("reviewedPrices") or {})
  price_observations: Dict[str, Dict[str, Any]] = dict(scheduler_state.get("priceObservations") or {})
  last_prices: Dict[str, float] = {
    symbol: float(observation["lastPrice"])
    for symbol, observation in price_observations.items()
    if isinstance(observation, dict) and float(observation.get("lastPrice") or 0) > 0
  }

  def _persist_agent_scheduler() -> None:
    memory.save_agent_scheduler({
      "lastRunTs": last_agent_run_ts,
      "unproductiveRuns": unproductive_runs,
      "reviewedPrices": reviewed_prices,
      "priceObservations": price_observations,
    })

  idle_polls = 0
  consecutive_no_trade_runs = 0  # runs where the agent placed no order (drives forced research)
  force_research = False         # when True, next agent run must hand off to Research first
  last_forced_research_ts = 0.0  # wall-clock of the last forced research handoff (cooldown gate)
  pending_trigger_moves: Dict[str, float] = {}  # strongest move per symbol, retained until reviewed
  pending_agent_triggers: set[str] = set()
  agent_task: asyncio.Task | None = None
  agent_task_triggers: list[str] = []
  agent_task_forced_research = False
  agent_task_token: str | None = None
  agent_task_event_ids: list[str] = []
  agent_task_started_ts = 0.0
  agent_task_timed_out = False
  agent_task_trigger_moves: Dict[str, float] = {}
  agent_task_discrete_triggers: set[str] = set()
  agent_task_prices: Dict[str, float] = {}
  last_complete_fill_poll_ts = 0.0
  # Seed from the persisted set: a restart inside the 30-min fill lookback used to re-detect and
  # double-record the same close (an in-memory-only set), corrupting realized-PnL stats.
  logged_closed_position_ids: set[str] = set(memory.get_seen_close_ids())
  logged_fill_ids: set[str] = set(memory.get_seen_fill_ids())
  for pending_event in memory.get_pending_agent_events():
    if pending_event.get("kind") in {"spot_fills", "futures_fills"}:
      logged_fill_ids.add(str(pending_event.get("id") or ""))
  notifier = TelegramNotifier(cfg)
  notifier.notify_startup(cfg)
  dashboard = DashboardPublisher(cfg)
  if dashboard.enabled:
    logger.info("Public dashboard publishing enabled (disclosure=%s, table=%s, container=%s).",
                cfg.dashboard.disclosure, cfg.dashboard.table_name, cfg.dashboard.container_name)
  elif cfg.dashboard.enabled:
    logger.warning("Dashboard publishing requested but DISABLED: %s", dashboard.disabled_reason())

  protection = ProtectionManager(
    cfg.profit_protection, kucoin_futures, notifier=notifier,
    emergency_sl_pct=cfg.trading.emergency_sl_pct, min_rr=cfg.trading.min_futures_rr,
    max_loss_equity_fraction=cfg.trading.risk_per_trade_pct,
  )
  if cfg.profit_protection.enabled:
    logger.info(
      "Profit-lock enabled (dry_run=%s, breakeven@%.1fR, giveback=%.0f%%, no_chase=%s/%.0fmin).",
      cfg.profit_protection.dry_run, cfg.profit_protection.breakeven_trigger_r,
      cfg.profit_protection.giveback_pct * 100, cfg.profit_protection.no_chase_enabled,
      cfg.profit_protection.post_win_cooldown_minutes,
    )

  logger.info("Starting trading loop...")
  while not stop_event.is_set():
    try:
      snapshot = build_snapshot(cfg, kucoin, kucoin_futures, memory)
    except Exception as exc:
      safety.invalidate(f"Snapshot incomplete: {exc}", revoke_active=True)
      logger.error("Snapshot failed, retrying after delay: %s", exc)
      notifier.notify_error(str(exc), context="Snapshot build")
      # If positions and stops were both fetched, keep the deterministic safety loop alive even
      # though entries stay disabled because some other part of exchange truth is incomplete.
      if isinstance(exc, IncompleteFuturesSnapshot) and kucoin_futures:
        failed_names = {failure.split(":", 1)[0] for failure in exc.failures}
        if "positions" not in failed_names and "stops" not in failed_names:
          try:
            degraded = type("DegradedSnapshot", (), {
              "futures_enabled": True,
              "futures_positions": exc.positions,
              "futures_stop_orders": exc.stops,
              "futures_account": exc.overview or {},
              "total_usdt": 0.0,
            })()
            with safety.order_lock:
              actions = protection.run(degraded)
            if actions:
              logger.warning("Protection actions taken from degraded snapshot: %s", actions)
          except Exception as protection_exc:
            logger.warning("Degraded protection failed: %s", protection_exc)
      try:
        await asyncio.wait_for(stop_event.wait(), timeout=cfg.trading.poll_interval_sec)
      except asyncio.TimeoutError:
        pass
      continue
    if stop_event.is_set():
      break

    # GTC entry orders do not expire just because the tool response contains an expiresAt note.
    # Enforce the configured TTL on bot-tagged orders before the agent sees the book, preventing a
    # stale thesis from filling hours later.  Manual and protective orders are never touched.
    if kucoin_futures and snapshot.futures_pending_orders:
      expired_orders = _expired_bot_entry_orders(
        snapshot.futures_pending_orders,
        cfg.trading.entry_limit_expiry_minutes,
      )
      cancelled_ids: set[str] = set()
      for order in expired_orders:
        oid = str(order.get("id") or order.get("orderId") or "")
        if not oid:
          continue
        try:
          with safety.order_lock:
            kucoin_futures.cancel_order(oid, symbol=order.get("symbol"))
          cancelled_ids.add(oid)
          logger.info("Expired stale futures entry %s (%s) after %.0fmin", oid, order.get("symbol"), cfg.trading.entry_limit_expiry_minutes)
        except Exception as exc:
          logger.warning("Unable to expire stale futures entry %s: %s", oid, exc)
      if cancelled_ids:
        snapshot.futures_pending_orders = [
          o for o in snapshot.futures_pending_orders
          if str(o.get("id") or o.get("orderId") or "") not in cancelled_ids
        ]

    spot_usdt = 0.0
    for acct in snapshot.spot_accounts:
      if acct.currency == "USDT":
        try:
          avail = float(acct.available or 0)
          bal = float(acct.balance or 0)
          spot_usdt += max(avail, bal)
        except Exception:
          continue

    futures_usdt = 0.0
    if cfg.kucoin_futures.enabled and snapshot.futures_account:
      try:
        fut_avail = float(snapshot.futures_account.get("availableBalance") or 0)
        fut_equity = float(snapshot.futures_account.get("accountEquity") or snapshot.futures_account.get("marginBalance") or fut_avail)
        futures_usdt += max(fut_avail, fut_equity)
      except Exception:
        pass

    financial_usdt = 0.0
    for acct in snapshot.financial_accounts:
      if acct.currency == "USDT":
        try:
          avail = float(acct.available or 0)
          bal = float(acct.balance or 0)
          financial_usdt += max(avail, bal)
        except Exception:
          continue

    total_usdt = spot_usdt + futures_usdt + financial_usdt
    safety.refresh(total_usdt)

    # Track daily drawdown per venue; the circuit-breaker section below converts the total scope
    # into a hard close-only restriction when its configured limit is breached.
    limits_total = memory.update_limits(total_usdt, scope="total")
    limits_spot = memory.update_limits(spot_usdt, scope="spot")
    limits_futures = memory.update_limits(futures_usdt, scope="futures")

    snapshot.total_usdt = total_usdt
    snapshot.drawdown_pct = float(limits_total.get("drawdownPct") or 0.0)
    snapshot.drawdown_pct_spot = float(limits_spot.get("drawdownPct") or 0.0)
    snapshot.drawdown_pct_futures = float(limits_futures.get("drawdownPct") or 0.0)

    if snapshot.drawdown_pct > 0:
      logger.info("Daily drawdown: total=%.2f%% spot=%.2f%% futures=%.2f%%",
                  snapshot.drawdown_pct, snapshot.drawdown_pct_spot, snapshot.drawdown_pct_futures)

    if (
      agent_task is not None
      and not agent_task.done()
      and not agent_task_timed_out
      and agent_task_started_ts > 0
      and time.time() - agent_task_started_ts >= _AGENT_RUN_TIMEOUT_SEC
    ):
      agent_task_timed_out = True
      safety.revoke_run(agent_task_token, "Background agent exceeded its 20-minute authority window")
      logger.error("Background agent exceeded %dsec; all later exchange writes from that run are disabled.", _AGENT_RUN_TIMEOUT_SEC)
      notifier.notify_error("Model run exceeded 20 minutes; entry/order authority revoked.", context="Agent timeout")
      # Cancelling the asyncio wrapper cannot stop a thread stuck in blocking SDK code. Detach it
      # after revoking its unique token so it cannot touch the exchange, retain its inbox events,
      # and allow a fresh single-flight run after the normal active cooldown.
      agent_task.cancel()
      for symbol, move in agent_task_trigger_moves.items():
        pending_trigger_moves[symbol] = max(move, pending_trigger_moves.get(symbol, 0.0))
      pending_agent_triggers.update(agent_task_discrete_triggers)
      last_agent_run_ts = time.time()
      _persist_agent_scheduler()
      agent_task = None
      agent_task_triggers = []
      agent_task_forced_research = False
      agent_task_token = None
      agent_task_event_ids = []
      agent_task_started_ts = 0.0
      agent_task_timed_out = False
      agent_task_trigger_moves = {}
      agent_task_discrete_triggers = set()
      agent_task_prices = {}

    # Harvest a completed single-flight model run without ever pausing the snapshot/protection loop.
    if agent_task is not None and agent_task.done():
      last_agent_run_ts = time.time()  # enforce cadence from completion, not from a long run's start
      _persist_agent_scheduler()
      try:
        result = agent_task.result()
        made_move = _agent_made_a_move(result)
        if agent_task_event_ids or snapshot.futures_positions:
          unproductive_runs = 0
        elif made_move:
          # A submitted order is useful progress but not proof of edge until it fills.
          unproductive_runs = max(0, unproductive_runs - 1)
        else:
          unproductive_runs = min(100, unproductive_runs + 1)
        # Anchor future triggers to the exact snapshot the successful run reviewed. A move during
        # the model call remains visible and can therefore schedule a genuinely new follow-up.
        reviewed_prices.update(agent_task_prices)
        # Polls during the run measured displacement from the previous anchor. Recalculate against
        # this run's snapshot below instead of carrying that stale magnitude into another call.
        _rebase_reviewed_price_triggers(
          pending_trigger_moves,
          pending_agent_triggers,
          set(agent_task_prices),
        )
        _persist_agent_scheduler()
        logger.info("--- Agent Decision Narrative ---\n%s", result.get("narrative", ""))
        if result.get("decisions"):
          logger.info("--- Decisions ---\n%s", "\n".join(f"- {d}" for d in result["decisions"]))
        notifier.notify_agent_run(agent_task_triggers or ["background_run"], result)

        acknowledged = memory.acknowledge_agent_events(agent_task_event_ids)
        for event in acknowledged:
          if event.get("kind") in {"spot_fills", "futures_fills"}:
            memory.record_seen_fill_id(str(event.get("id") or ""))

        threshold = cfg.trading.research_handoff_after_no_trade_runs
        research_cooldown_sec = cfg.trading.research_handoff_cooldown_min * 60
        if agent_task_forced_research:
          consecutive_no_trade_runs = 0
          force_research = False
          last_forced_research_ts = time.time()
        elif made_move:
          consecutive_no_trade_runs = 0
        else:
          consecutive_no_trade_runs += 1
        current_book_active = bool(
          snapshot.futures_positions or snapshot.futures_pending_orders or snapshot.spot_pending_orders
        )
        # Whole-market research is deliberately flat-only. Running it while capital is exposed
        # caused 10–25 minute protection blind spots and encouraged unrelated add-ons.
        if threshold > 0 and consecutive_no_trade_runs >= threshold and not current_book_active:
          elapsed = time.time() - last_forced_research_ts
          if elapsed >= research_cooldown_sec:
            force_research = True
            logger.info("No trade for %d flat-book runs — forcing Research Agent handoff next run.", consecutive_no_trade_runs)
      except Exception as exc:
        logger.error("Agent run failed: %s", exc)
        notifier.notify_error(str(exc), context="Agent run")
        for symbol, move in agent_task_trigger_moves.items():
          pending_trigger_moves[symbol] = max(move, pending_trigger_moves.get(symbol, 0.0))
        pending_agent_triggers.update(agent_task_discrete_triggers)
      finally:
        safety.finish_run(agent_task_token)
        agent_task = None
        agent_task_triggers = []
        agent_task_forced_research = False
        agent_task_token = None
        agent_task_event_ids = []
        agent_task_started_ts = 0.0
        agent_task_timed_out = False
        agent_task_trigger_moves = {}
        agent_task_discrete_triggers = set()
        agent_task_prices = {}

    triggers: list[str] = []
    for symbol, ticker in snapshot.tickers.items():
      price = float(ticker.price)
      prev = last_prices.get(symbol)
      observation = price_observations.get(symbol) or {}
      noise_ewma = float(observation.get("noiseEwmaPct") or 0.0)
      samples = int(observation.get("samples") or 0)
      reviewed_price = reviewed_prices.get(symbol)
      if reviewed_price is None:
        initial_trigger = f"initial:{symbol}"
        triggers.append(initial_trigger)
        pending_agent_triggers.add(initial_trigger)
      elif reviewed_price > 0:
        # Compare with the last successful model-reviewed state, not the immediately preceding
        # poll. The old per-poll comparison repeatedly called the model on ordinary oscillation.
        state_move_pct = abs(price - reviewed_price) / reviewed_price * 100
        adaptive_threshold = _adaptive_price_trigger_threshold(
          cfg.trading.price_change_trigger_pct,
          noise_ewma,
          samples,
          noise_multiplier=cfg.trading.price_noise_multiplier,
          max_multiplier=cfg.trading.price_trigger_max_multiplier,
        )
        if state_move_pct >= adaptive_threshold:
          triggers.append(f"price_move:{symbol}:{state_move_pct:.2f}%")
          pending_trigger_moves[symbol] = max(
            state_move_pct,
            pending_trigger_moves.get(symbol, 0.0),
          )

      if prev is not None and prev > 0:
        poll_move_pct = abs(price - prev) / prev * 100
        noise_ewma = _next_price_noise_ewma(
          noise_ewma,
          poll_move_pct,
          cfg.trading.price_change_trigger_pct,
          samples,
          max_multiplier=cfg.trading.price_trigger_max_multiplier,
        )
        samples = min(samples + 1, 1_000_000)
      price_observations[symbol] = {
        "lastPrice": price,
        "noiseEwmaPct": noise_ewma,
        "samples": samples,
        "updated": int(time.time()),
      }
      last_prices[symbol] = price
    _persist_agent_scheduler()

    # Snapshot peak/trough BEFORE the update prunes just-closed positions — otherwise a position's
    # MFE/MAE is reset before we record its close, and every close lands with peak/trough = None
    # (which is why we can't tell whether TP targets were realistically reachable).
    pre_close_extremes = memory.get_position_extremes()
    try:
      # Drive extremes off live exchange positions so peak/trough reset when a position closes.
      memory.update_position_extremes(_live_extremes_map(snapshot))
    except Exception as exc:
      logger.warning("Failed to update position extremes: %s", exc)

    # Code-driven profit protection: ratchet stops to breakeven and cap give-back on
    # live futures positions every poll, independent of whether the agent runs. Never raises.
    try:
      with safety.order_lock:
        protection_actions = protection.run(snapshot)
      if protection_actions:
        logger.info("Profit-lock actions taken: %s", protection_actions)
    except Exception as exc:
      logger.warning("Profit-lock run failed: %s", exc)

    # Retain throttled moves until the model actually reviews them. Without this queue, a trigger
    # followed by a quiet poll vanished and the adaptive shorter cooldown never got another chance.
    run_candidate = bool(triggers or pending_trigger_moves or pending_agent_triggers) or idle_polls >= cfg.trading.max_idle_polls

    fill_lookback_minutes = 1440 if last_complete_fill_poll_ts <= 0 else max(
      30,
      min(1440, int((time.time() - last_complete_fill_poll_ts) / 60) + 5),
    )
    recent_fills = _fetch_recent_fills(kucoin, kucoin_futures, lookback_minutes=fill_lookback_minutes)
    fill_poll_errors = recent_fills.pop("_errors", [])
    critical_fill_errors = [
      error for error in fill_poll_errors
      if kucoin_futures is None or not str(error).startswith("spot fills:")
    ]
    if critical_fill_errors:
      safety.invalidate("Fill/close history incomplete: " + "; ".join(critical_fill_errors), revoke_active=True)
      logger.error("Entry/order authority disabled because futures event history is incomplete: %s", critical_fill_errors)
    if not fill_poll_errors:
      last_complete_fill_poll_ts = time.time()
    new_spot_fills = []
    for fill in recent_fills["spot_fills"]:
      fill_id = _fill_event_id(fill, "spot")
      if fill_id in logged_fill_ids:
        continue
      logged_fill_ids.add(fill_id)
      memory.mark_order_filled(fill.get("orderId"), fill.get("clientOid"))
      if memory.queue_agent_event("spot_fills", fill_id, fill):
        new_spot_fills.append(fill)
    new_futures_fills = []
    for fill in recent_fills["futures_fills"]:
      fill_id = _fill_event_id(fill, "futures")
      if fill_id in logged_fill_ids:
        continue
      logged_fill_ids.add(fill_id)
      memory.mark_order_filled(fill.get("orderId"), fill.get("clientOid"))
      if memory.queue_agent_event("futures_fills", fill_id, fill):
        new_futures_fills.append(fill)
    new_closed_positions = []
    for cp in recent_fills["closed_positions"]:
      close_id = _close_event_id(cp)
      legacy_id = str(cp.get("openTime") or "")
      if close_id in logged_closed_position_ids or (legacy_id and legacy_id in logged_closed_position_ids):
        continue
      memory.queue_agent_event("closed_positions", f"closed:{close_id}", cp)
      new_closed_positions.append(cp)
    recent_fills = {
      "spot_fills": new_spot_fills,
      "futures_fills": new_futures_fills,
      "closed_positions": new_closed_positions,
    }
    new_events_count = len(recent_fills["spot_fills"]) + len(recent_fills["futures_fills"]) + len(recent_fills["closed_positions"])
    pending_events_count = len(memory.get_pending_agent_events())
    if new_events_count:
      logger.info("Detected %d new fill/close events (spot=%d, futures=%d, closed=%d)",
                   new_events_count, len(recent_fills["spot_fills"]), len(recent_fills["futures_fills"]), len(recent_fills["closed_positions"]))

    for cp in recent_fills["closed_positions"]:
      cp_id = _close_event_id(cp)
      if not cp_id or cp_id in logged_closed_position_ids:
        continue
      try:
        sym = normalize_symbol(cp.get("symbol") or "")
        pnl = float(cp.get("pnl") or 0)
        roe = float(cp.get("roe") or 0)
        close_type = cp.get("type") or "unknown"
        side = "sell" if "LONG" in close_type.upper() else "buy"
        position_side = (
          "long" if "LONG" in close_type.upper()
          else "short" if "SHORT" in close_type.upper()
          else None
        )
        # Capture the exit price so the no-chase guard knows where we sold/covered.
        exit_price = None
        for _k in ("closePrice", "avgExitPrice", "settleClosePrice", "markPrice", "lastPrice"):
          _v = cp.get(_k)
          if _v not in (None, "", 0, "0"):
            try:
              exit_price = float(_v)
              break
            except (TypeError, ValueError):
              continue
        if exit_price is None:
          exit_price = last_prices.get(sym)  # fallback: latest poll price ≈ exit
        # Entry price for the closed-position chart's open marker (public-safe price, no size).
        entry_price = None
        for _k in ("avgEntryPrice", "openPrice", "entryPrice", "avgEntry"):
          _v = cp.get(_k)
          if _v not in (None, "", 0, "0"):
            try:
              entry_price = float(_v)
              break
            except (TypeError, ValueError):
              continue
        # MFE/MAE from the pre-reset extremes: how far the trade ran in profit (peak) and underwater
        # (trough) before closing — the data that tells us if TPs are set within realistic reach.
        _ext = pre_close_extremes.get(sym, {}) if isinstance(pre_close_extremes, dict) else {}
        cp_open_time = cp.get("openTime") or cp.get("openingTimestamp")
        if _ext.get("positionOpenTime") not in (None, "") and cp_open_time not in (None, ""):
          try:
            ext_open = int(float(_ext["positionOpenTime"]))
            close_open = int(float(cp_open_time))
            ext_open = ext_open if ext_open > 1_000_000_000_000 else ext_open * 1000
            close_open = close_open if close_open > 1_000_000_000_000 else close_open * 1000
            if abs(ext_open - close_open) > 1000:
              _ext = {}
          except (TypeError, ValueError):
            _ext = {}
        peak_pnl = _ext.get("peakPnl")
        trough_pnl = _ext.get("troughPnl")
        # Include the fee/funding-adjusted terminal result so every close lies inside its recorded
        # lifecycle range; sampled unrealized extrema alone missed terminal slippage and fees.
        peak_pnl = pnl if peak_pnl is None else max(float(peak_pnl), pnl)
        trough_pnl = pnl if trough_pnl is None else min(float(trough_pnl), pnl)
        memory.log_decision(
          sym,
          f"futures_{side}_triggered",
          confidence=0.0,
          reason=f"TP/SL triggered ({close_type}, ROE {roe:.2%})",
          pnl=pnl,
          paper=False,
          exit_price=exit_price,
          close_type=close_type,
          position_id=cp.get("id"),
          position_open_time=cp.get("openTime") or cp.get("openingTimestamp"),
          position_side=position_side,
          peak_pnl=peak_pnl,
          trough_pnl=trough_pnl,
          entry_price=entry_price,
        )
        logged_closed_position_ids.add(cp_id)
        memory.record_seen_close_id(cp_id)
        logger.info("Recorded triggered close for %s: PnL=%.4f (%s)", sym, pnl, close_type)
      except Exception as exc:
        logger.warning("Failed to record triggered close: %s", exc)

    # --- Circuit Breaker Checks ---
    cb = cfg.circuit_breaker
    restriction_reasons: list[str] = []

    if snapshot.drawdown_pct >= cb.max_daily_drawdown_pct:
      restriction_reasons.append(f"Daily drawdown {snapshot.drawdown_pct:.1f}% >= {cb.max_daily_drawdown_pct}% limit")

    consec_losses = memory.consecutive_losses()
    if consec_losses >= cb.max_consecutive_losses:
      last_loss_ts = None
      with memory._lock:
        decs = memory._read().get("decisions", [])
      decs = memory._authoritative_realized_rows(decs)
      for d in sorted(decs, key=lambda x: x.get("ts", 0), reverse=True):
        pnl = d.get("pnl")
        if pnl is not None:
          try:
            if float(pnl) < 0:
              last_loss_ts = d.get("ts")
              break
          except (TypeError, ValueError):
            continue
      if last_loss_ts:
        elapsed_min = (int(time.time()) - last_loss_ts) / 60
        if elapsed_min < cb.cooldown_minutes:
          restriction_reasons.append(
            f"{consec_losses} consecutive losses (cooldown {int(cb.cooldown_minutes - elapsed_min)}min remaining)"
          )

    if restriction_reasons:
      snapshot.trading_restricted = True
      snapshot.restriction_reason = "; ".join(restriction_reasons)
      logger.warning("CIRCUIT BREAKER: %s", snapshot.restriction_reason)
      notifier.send(f"<b>CIRCUIT BREAKER ACTIVE</b>\n{snapshot.restriction_reason}\nAgent restricted to close-only mode.")

    # The model is the most expensive and least deterministic part of the loop. Code-driven
    # protection and deterministic order expiry still evaluate every poll. Only an actual position
    # uses active model cadence: an exchange-atomic pending bracket is not exposed capital and does
    # not need the model to babysit/cancel it before its deterministic expiry.
    capital_exposed = bool(snapshot.futures_positions)
    pending_orders_active = bool(snapshot.futures_pending_orders or snapshot.spot_pending_orders)
    open_book = capital_exposed or pending_orders_active
    effective_flat_cooldown = _productivity_adjusted_flat_cooldown(
      cfg.trading.flat_agent_cooldown_sec,
      unproductive_runs,
      cfg.trading.flat_backoff_max_multiplier,
    )
    cooldown_sec = _adaptive_agent_cooldown(
      flat_cooldown_sec=effective_flat_cooldown,
      active_cooldown_sec=cfg.trading.active_agent_cooldown_sec,
      book_active=capital_exposed,
      new_events_count=pending_events_count,
      trigger_move_pcts=list(pending_trigger_moves.values()),
      price_trigger_pct=cfg.trading.price_change_trigger_pct,
    )
    run_candidate = run_candidate or bool(pending_events_count)
    cooldown_elapsed = time.time() - last_agent_run_ts
    should_run = (
      agent_task is None
      and run_candidate
      and (last_agent_run_ts <= 0 or cooldown_elapsed >= max(0.0, cooldown_sec))
    )
    if agent_task is None and run_candidate and not should_run and idle_polls % max(1, cfg.trading.max_idle_polls) == 0:
      logger.info(
        "Agent run throttled for another %dsec (capital_exposed=%s pending_orders=%s no_action_runs=%d)",
        int(cooldown_sec - cooldown_elapsed), capital_exposed, pending_orders_active, unproductive_runs,
      )

    if should_run and not stop_event.is_set():
      run_token = safety.authorize_run()
      if not run_token:
        logger.warning("Agent run skipped because live entry/order authority is unavailable.")
        idle_polls += 1
        try:
          await asyncio.wait_for(stop_event.wait(), timeout=cfg.trading.poll_interval_sec)
        except asyncio.TimeoutError:
          pass
        continue
      idle_polls = 0
      last_agent_run_ts = time.time()
      _persist_agent_scheduler()
      agent_triggers = list(pending_agent_triggers)
      agent_triggers.extend(trigger for trigger in triggers if trigger not in agent_triggers)
      agent_triggers.extend(
        f"pending_price_move:{symbol}:{move:.2f}%"
        for symbol, move in sorted(pending_trigger_moves.items())
        if not any(trigger.startswith(f"price_move:{symbol}:") for trigger in agent_triggers)
      )
      if not agent_triggers:
        agent_triggers = ["idle_threshold"]
      agent_task_trigger_moves = dict(pending_trigger_moves)
      agent_task_discrete_triggers = set(pending_agent_triggers)
      agent_task_prices = {
        symbol: float(ticker.price)
        for symbol, ticker in snapshot.tickers.items()
      }
      pending_trigger_moves.clear()
      pending_agent_triggers.clear()
      pending_batch = memory.get_pending_agent_events()
      events_for_agent: Dict[str, list] = {}
      for event in pending_batch:
        kind = str(event.get("kind") or "")
        if kind:
          events_for_agent.setdefault(kind, []).append(event.get("payload"))
      force_research_for_run = bool(force_research and not open_book)
      agent_task_triggers = agent_triggers
      agent_task_forced_research = force_research_for_run
      agent_task_token = run_token
      agent_task_event_ids = [str(event.get("id") or "") for event in pending_batch]
      agent_task_started_ts = time.time()
      agent_task_timed_out = False
      logger.info("Starting background agent. Triggers: %s", agent_triggers)
      agent_task = asyncio.create_task(_run_in_daemon_thread(
        run_trading_agent, cfg, snapshot, kucoin, kucoin_futures, azure_client, ls_client,
        recent_fills=events_for_agent or None,
        force_research=force_research_for_run,
        safety_state=safety,
        entry_token=run_token,
      ))
    else:
      idle_polls += 1
      logger.info(
        "Agent %s. Idle polls: %d/%d",
        "running in background" if agent_task is not None else "idle/throttled",
        idle_polls,
        cfg.trading.max_idle_polls,
      )

    # Publish a sanitized public-safe snapshot for the read-only spectator dashboard.
    # Self-throttled and never raises, so it is safe to call every poll. Open positions are
    # read from `snapshot` (live exchange truth), not MemoryStore (which lingers after TP/SL).
    dashboard.publish(memory, snapshot, last_prices, cfg)

    try:
      await asyncio.wait_for(stop_event.wait(), timeout=cfg.trading.poll_interval_sec)
    except asyncio.TimeoutError:
      pass

  safety.begin_shutdown()
  if agent_task is not None and not agent_task.done():
    safety.revoke_run(agent_task_token, "Process is shutting down")
    try:
      await asyncio.wait_for(asyncio.shield(agent_task), timeout=_AGENT_SHUTDOWN_GRACE_SEC)
    except (asyncio.TimeoutError, asyncio.CancelledError):
      agent_task.cancel()


def main() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  cfg = load_config()

  if cfg.supervisor.log_file:
    fh = RotatingFileHandler(
      cfg.supervisor.log_file,
      maxBytes=cfg.supervisor.log_max_bytes,
      backupCount=cfg.supervisor.log_backup_count,
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

  if cfg.supervisor.enabled and cfg.telegram.enabled:
    from .telegram_bot import start_telegram_bot
    bot_thread = threading.Thread(target=start_telegram_bot, args=(cfg,), daemon=True, name="supervisor-bot")
    bot_thread.start()
    logger.info("Supervisor Telegram bot started.")

  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  stop_event = asyncio.Event()
  safety = TradingSafetyState()
  def _request_shutdown() -> None:
    safety.begin_shutdown()
    stop_event.set()
  loop.add_signal_handler(signal.SIGTERM, _request_shutdown)
  loop.add_signal_handler(signal.SIGINT, _request_shutdown)
  try:
    loop.run_until_complete(trading_loop(stop_event=stop_event, safety=safety))
  except KeyboardInterrupt:
    logger.info("Shutting down...")
  except Exception as exc:
    logger.error("Fatal error: %s", exc)
    sys.exit(1)
  finally:
    safety.begin_shutdown()
    loop.close()


if __name__ == "__main__":
  main()
