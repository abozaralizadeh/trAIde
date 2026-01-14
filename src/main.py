from __future__ import annotations

import asyncio
import sys
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


def build_snapshot(cfg, kucoin: KucoinClient, memory: MemoryStore) -> TradingSnapshot:
  coins = _load_active_coins(cfg, memory)
  tickers = {}
  for symbol in coins:
    tickers[symbol] = kucoin.get_ticker(symbol)

  balances = kucoin.get_trade_accounts()
  futures_overview = None
  if cfg.kucoin_futures.enabled:
    try:
      fut = KucoinFuturesClient(cfg)
      futures_overview = fut.get_account_overview()
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
    except Exception as exc:
      print("Warning: unable to fetch futures account overview:", exc, file=sys.stderr)

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
    drawdown_pct=0.0,
    total_usdt=0.0,
  )


async def trading_loop() -> None:
  cfg = load_config()
  setup_tracing(cfg)
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
    setup_lstracing(cfg)
    try:
      snapshot = build_snapshot(cfg, kucoin, memory)
    except Exception as exc:
      print("Snapshot failed, retrying after delay:", exc, file=sys.stderr)
      await asyncio.sleep(cfg.trading.poll_interval_sec)
      continue

    usdt_balance = 0.0
    for acct in snapshot.balances:
      if acct.currency == "USDT":
        try:
          avail = float(acct.available or 0)
          bal = float(acct.balance or 0)
          # For futures (contract) use the larger of available/equity so we don't undercount.
          usdt_balance += max(avail, bal)
        except Exception:
          continue

    # Update drawdown and auto-clear risk_off when recovered.
    limits = memory.update_limits(usdt_balance, cfg.trading.max_daily_drawdown_pct)
    if limits.get("kill") and (limits.get("drawdownPct") or 0) < cfg.trading.max_daily_drawdown_pct * 0.5:
      limits = memory.reset_limits(usdt_balance)

    risk_off = bool(limits.get("kill"))
    snapshot.risk_off = risk_off
    snapshot.drawdown_pct = float(limits.get("drawdownPct") or 0.0)
    snapshot.total_usdt = usdt_balance
    if risk_off:
      reason = limits.get("reason") or "Kill switch active (drawdown limit reached). Running in risk-off mode."
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
        # Propagate risk_off into snapshot so the agent can act defensively (hedge/exit) but skip new risk-on entries.
        snapshot.risk_off = risk_off
        result = await run_trading_agent(cfg, snapshot, kucoin, kucoin_futures, azure_client)
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
