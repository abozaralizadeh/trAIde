from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from typing import Dict

from agents import set_default_openai_client
from agents.tracing import (get_trace_provider)
from .agent import TradingSnapshot, run_trading_agent, setup_tracing, setup_lstracing, _build_openai_client
from .config import load_config
from .kucoin import KucoinClient, KucoinFuturesClient, KucoinAccount
from .memory import MemoryStore
from .telegram import TelegramNotifier
from .utils import normalize_symbol

logger = logging.getLogger(__name__)


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


def _fetch_tickers(cfg, kucoin: KucoinClient, coins: list[str], memory: MemoryStore) -> tuple[list[str], dict]:
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
      if cfg.trading.flexible_coins_enabled and fail_count >= _TICKER_FAIL_THRESHOLD:
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

  if not tickers:
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
  try:
    overview = kucoin_futures.get_account_overview()
    stops = kucoin_futures.list_stop_orders(status="active") or []
    positions = kucoin_futures.list_positions() or []
    return overview, positions, stops
  except Exception as exc:
    logger.warning("Unable to fetch futures account overview: %s", exc)
    return None, [], []


def _fetch_recent_fills(kucoin: KucoinClient, kucoin_futures: KucoinFuturesClient | None, lookback_minutes: int = 10) -> Dict:
  """Fetch fills and closed positions from the last N minutes (using KuCoin server time to avoid clock drift)."""
  now_ms = kucoin._timestamp_ms()
  cutoff_ms = now_ms - lookback_minutes * 60 * 1000
  result: Dict[str, Any] = {"spot_fills": [], "futures_fills": [], "closed_positions": []}

  def _after_cutoff(entry: dict, *ts_keys: str) -> bool:
    for k in ts_keys:
      v = entry.get(k)
      if v is not None:
        ts = int(v) if int(v) > 1e12 else int(v) * 1000
        if ts >= cutoff_ms:
          return True
    return False

  try:
    spot_fills = kucoin.get_fills(page_size=50) or []
    result["spot_fills"] = [f for f in spot_fills if _after_cutoff(f, "createdAt", "tradeCreatedAt")]
  except Exception as exc:
    logger.warning("Unable to fetch spot fills: %s", exc)

  if kucoin_futures:
    try:
      futures_fills = kucoin_futures.get_fills(page_size=50) or []
      result["futures_fills"] = [f for f in futures_fills if _after_cutoff(f, "createdAt", "tradeCreatedAt")]
    except Exception as exc:
      logger.warning("Unable to fetch futures fills: %s", exc)

    try:
      closed = kucoin_futures.get_position_history(page_size=20) or []
      result["closed_positions"] = [p for p in closed if _after_cutoff(p, "closeTime", "updatedAt")]
    except Exception as exc:
      logger.warning("Unable to fetch closed positions: %s", exc)

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


def build_snapshot(cfg, kucoin: KucoinClient, kucoin_futures: KucoinFuturesClient | None, memory: MemoryStore) -> TradingSnapshot:
  raw_coins = _load_active_coins(cfg, memory)
  coins, tickers = _fetch_tickers(cfg, kucoin, raw_coins, memory)
  spot_accounts, financial_accounts, all_accounts = _fetch_balances(kucoin)

  # Discover spot holdings not in the active coin list (e.g., externally bought coins).
  extra_coins, extra_tickers = _discover_unlisted_holdings(kucoin, spot_accounts, tickers)
  coins.extend(extra_coins)
  tickers.update(extra_tickers)

  futures_overview, futures_positions, futures_stops = _fetch_futures(cfg, kucoin_futures)

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
      futures_pending = kucoin_futures.list_orders(status="open")
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


async def trading_loop() -> None:
  cfg = load_config()
  setup_tracing(cfg)
  ls_client = setup_lstracing(cfg)
  azure_client = _build_openai_client(cfg)
  # Azure client is for inference only; tracing uses platform key via set_tracing_export_api_key in setup_tracing.
  set_default_openai_client(azure_client, use_for_tracing=False)
  kucoin = KucoinClient(cfg)
  kucoin_futures = KucoinFuturesClient(cfg) if cfg.kucoin_futures.enabled else None
  last_prices: Dict[str, float] = {}
  idle_polls = 0
  logged_closed_position_ids: set[str] = set()
  memory = MemoryStore(cfg.memory_file, retention_days=cfg.retention_days)
  notifier = TelegramNotifier(cfg)
  notifier.notify_startup(cfg)

  logger.info("Starting trading loop...")
  while True:
    try:
      snapshot = build_snapshot(cfg, kucoin, kucoin_futures, memory)
    except Exception as exc:
      logger.error("Snapshot failed, retrying after delay: %s", exc)
      notifier.notify_error(str(exc), context="Snapshot build")
      await asyncio.sleep(cfg.trading.poll_interval_sec)
      continue

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

    # Track daily drawdown per venue for informational context passed to the agent.
    # No kill switch — the agent decides how to adapt when drawdown is high.
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

    triggers: list[str] = []
    for symbol, ticker in snapshot.tickers.items():
      price = float(ticker.price)
      prev = last_prices.get(symbol)
      if prev is None:
        triggers.append(f"initial:{symbol}")
      else:
        change_pct = abs(price - prev) / prev * 100
        if change_pct >= cfg.trading.price_change_trigger_pct:
          triggers.append(f"price_move:{symbol}:{change_pct:.2f}%")
      last_prices[symbol] = price

    try:
      current_positions = memory.positions(last_prices)
      memory.update_position_extremes(current_positions)
    except Exception as exc:
      logger.warning("Failed to update position extremes: %s", exc)

    should_run = bool(triggers) or idle_polls >= cfg.trading.max_idle_polls

    recent_fills = _fetch_recent_fills(kucoin, kucoin_futures, lookback_minutes=30)
    new_events_count = len(recent_fills["spot_fills"]) + len(recent_fills["futures_fills"]) + len(recent_fills["closed_positions"])
    if new_events_count:
      logger.info("Detected %d new fill/close events (spot=%d, futures=%d, closed=%d)",
                   new_events_count, len(recent_fills["spot_fills"]), len(recent_fills["futures_fills"]), len(recent_fills["closed_positions"]))

    for cp in recent_fills["closed_positions"]:
      cp_id = str(cp.get("id") or cp.get("openTime") or "")
      if not cp_id or cp_id in logged_closed_position_ids:
        continue
      try:
        sym = normalize_symbol(cp.get("symbol") or "")
        pnl = float(cp.get("pnl") or 0)
        roe = float(cp.get("roe") or 0)
        close_type = cp.get("type") or "unknown"
        side = "sell" if "LONG" in close_type.upper() else "buy"
        memory.log_decision(
          sym,
          f"futures_{side}_triggered",
          confidence=0.0,
          reason=f"TP/SL triggered ({close_type}, ROE {roe:.2%})",
          pnl=pnl,
          paper=False,
        )
        logged_closed_position_ids.add(cp_id)
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

    if should_run:
      idle_polls = 0
      logger.info("Running agent. Triggers: %s", triggers or ["idle_threshold"])
      try:
        result = await asyncio.to_thread(
          run_trading_agent, cfg, snapshot, kucoin, kucoin_futures, azure_client, ls_client,
          recent_fills=recent_fills if new_events_count else None,
        )
        logger.info("--- Agent Decision Narrative ---\n%s", result["narrative"])
        if result.get("decisions"):
          logger.info("--- Decisions ---\n%s", "\n".join(f"- {d}" for d in result["decisions"]))
        notifier.notify_agent_run(triggers or ["idle_threshold"], result)
      except Exception as exc:
        # Keep the loop alive across restarts and transient errors.
        logger.error("Agent run failed: %s", exc)
        notifier.notify_error(str(exc), context="Agent run")
    else:
      idle_polls += 1
      logger.info("No triggers. Idle polls: %d/%d", idle_polls, cfg.trading.max_idle_polls)

    await asyncio.sleep(cfg.trading.poll_interval_sec)


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
  loop.add_signal_handler(signal.SIGTERM, loop.stop)
  try:
    loop.run_until_complete(trading_loop())
  except KeyboardInterrupt:
    logger.info("Shutting down...")
  except Exception as exc:
    logger.error("Fatal error: %s", exc)
    sys.exit(1)
  finally:
    loop.close()


if __name__ == "__main__":
  main()
