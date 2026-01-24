from __future__ import annotations

import asyncio
import sys
import time
from typing import Dict

from agents import set_default_openai_client
from agents.tracing import (get_trace_provider)
from .agent import TradingSnapshot, run_trading_agent, setup_tracing, setup_lstracing, _build_openai_client
from .config import load_config
from .kucoin import KucoinClient, KucoinFuturesClient, KucoinAccount
from .memory import MemoryStore


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


def build_snapshot(cfg, kucoin: KucoinClient, kucoin_futures: KucoinFuturesClient | None, memory: MemoryStore) -> TradingSnapshot:
  coins = _load_active_coins(cfg, memory)
  tickers = {}
  missing_tickers: list[str] = []
  for symbol in coins:
    try:
      tickers[symbol] = kucoin.get_ticker(symbol)
    except Exception as exc:
      missing_tickers.append(symbol)
      print(f"Warning: ticker fetch failed for {symbol}: {exc}", file=sys.stderr)
      continue

  if not tickers:
    raise RuntimeError(f"No tickers available; failed symbols: {missing_tickers}")

  spot_accounts = kucoin.get_trade_accounts()
  all_accounts = kucoin.get_accounts()
  balances = list(spot_accounts)
  futures_overview = None
  futures_stops: list[dict] = []
  futures_positions: list[dict] = []
  if cfg.kucoin_futures.enabled and kucoin_futures:
    try:
      futures_overview = kucoin_futures.get_account_overview()
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
      futures_stops = kucoin_futures.list_stop_orders(status="active") or []
      futures_positions = kucoin_futures.list_positions() or []
    except Exception as exc:
      print("Warning: unable to fetch futures account overview:", exc, file=sys.stderr)

  spot_stops: list[dict] = []
  try:
    spot_stops = kucoin.list_stop_orders(status="active")
  except Exception as exc:
    print("Warning: unable to fetch spot stop orders:", exc, file=sys.stderr)

  fees: dict[str, float] = {}
  try:
    fee_info = kucoin.get_base_fee()
    if fee_info:
      fees = {
        "spot_taker": float(fee_info.get("takerFeeRate") or 0.001),
        "spot_maker": float(fee_info.get("makerFeeRate") or 0.001),
      }
      # futures fee endpoints differ; not available here, but keep structure
  except Exception as exc:
    print("Warning: unable to fetch base fee:", exc, file=sys.stderr)

  return TradingSnapshot(
    coins=coins,
    tickers=tickers,
    balances=balances,
    paper_trading=cfg.trading.paper_trading,
    max_position_usd=cfg.trading.max_position_usd,
    min_confidence=cfg.trading.min_confidence,
    max_leverage=cfg.trading.max_leverage,
    futures_enabled=cfg.kucoin_futures.enabled,
    risk_off=False,  # will be set in loop
    risk_off_spot=False,
    risk_off_futures=False,
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
  memory = MemoryStore(cfg.memory_file, retention_days=cfg.retention_days)

  print("Starting trading loop...")
  while True:
    try:
      snapshot = build_snapshot(cfg, kucoin, kucoin_futures, memory)
    except Exception as exc:
      print("Snapshot failed, retrying after delay:", exc, file=sys.stderr)
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

    total_usdt = spot_usdt + futures_usdt

    # Update drawdown per venue and overall; auto-clear when recovered.
    limits_total = memory.update_limits(total_usdt, cfg.trading.max_daily_drawdown_pct, scope="total")
    limits_spot = memory.update_limits(spot_usdt, cfg.trading.max_daily_drawdown_pct, scope="spot")
    if cfg.kucoin_futures.enabled and snapshot.futures_account:
      limits_futures = memory.update_limits(futures_usdt, cfg.trading.max_daily_drawdown_pct, scope="futures")
    else:
      # If futures are empty/unavailable, clear futures risk-off so it does not block trading.
      limits_futures = memory.reset_limits(futures_usdt, scope="futures")

    def _maybe_reset(limits_obj, current_usdt, scope):
      if limits_obj.get("kill") and (limits_obj.get("drawdownPct") or 0) < cfg.trading.max_daily_drawdown_pct * 0.5:
        return memory.reset_limits(current_usdt, scope=scope)
      return limits_obj

    limits_total = _maybe_reset(limits_total, total_usdt, "total")
    limits_spot = _maybe_reset(limits_spot, spot_usdt, "spot")
    limits_futures = _maybe_reset(limits_futures, futures_usdt, "futures")

    snapshot.total_usdt = total_usdt
    snapshot.drawdown_pct = float(limits_total.get("drawdownPct") or 0.0)
    snapshot.drawdown_pct_spot = float(limits_spot.get("drawdownPct") or 0.0)
    snapshot.drawdown_pct_futures = float(limits_futures.get("drawdownPct") or 0.0)
    snapshot.risk_off_spot = bool(limits_spot.get("kill"))
    snapshot.risk_off_futures = bool(limits_futures.get("kill"))
    snapshot.risk_off = bool(limits_total.get("kill") and snapshot.risk_off_spot and snapshot.risk_off_futures)

    # If futures are effectively empty (no positions and tiny balance), clear futures kill switch to avoid blocking.
    empty_futures = (futures_usdt <= 1e-3) and not snapshot.futures_positions
    # If no futures positions and no margin in use, reset futures drawdown/kill to avoid blocking when idle.
    no_futures_pos = not snapshot.futures_positions
    fut_margin = 0.0
    try:
      fut_margin = float(snapshot.futures_account.get("positionMargin") or 0.0)
    except Exception:
      pass
    if (empty_futures or (no_futures_pos and fut_margin <= 1e-6)) and snapshot.risk_off_futures:
      limits_futures = memory.reset_limits(futures_usdt, scope="futures")
      snapshot.drawdown_pct_futures = float(limits_futures.get("drawdownPct") or 0.0)
      snapshot.risk_off_futures = False

    if snapshot.risk_off or snapshot.risk_off_spot or snapshot.risk_off_futures:
      reason = limits_total.get("reason") or limits_spot.get("reason") or limits_futures.get("reason") or "Kill switch active (drawdown limit reached)."
      print(reason)

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

    should_run = bool(triggers) or idle_polls >= cfg.trading.max_idle_polls

    if should_run:
      idle_polls = 0
      print(f"Running agent. Triggers: {triggers or ['idle_threshold']}")
      try:
        # snapshot.risk_off already computed above from per-scope limits.
        result = await asyncio.to_thread(
          run_trading_agent, cfg, snapshot, kucoin, kucoin_futures, azure_client, ls_client
        )
        print("\n--- Agent Decision Narrative ---")
        print(result["narrative"])
        if result.get("decisions"):
          print("\n--- Decisions ---")
          for d in result["decisions"]:
            print("-", d)
      except Exception as exc:
        # Keep the loop alive across restarts and transient errors.
        print("Agent run failed:", exc, file=sys.stderr)
    else:
      idle_polls += 1
      print(f"No triggers. Idle polls: {idle_polls}/{cfg.trading.max_idle_polls}")

    await asyncio.sleep(cfg.trading.poll_interval_sec)


def main() -> None:
  try:
    asyncio.run(trading_loop())
  except KeyboardInterrupt:
    print("Shutting down...")
  except Exception as exc:
    print("Fatal error:", exc, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()
