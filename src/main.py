from __future__ import annotations

import asyncio
import sys
from typing import Dict

from .agent import TradingSnapshot, run_trading_agent
from .config import load_config
from .kucoin import KucoinClient, KucoinFuturesClient
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

  return TradingSnapshot(
    coins=coins,
    tickers=tickers,
    balances=balances,
    paper_trading=cfg.trading.paper_trading,
    max_position_usd=cfg.trading.max_position_usd,
    min_confidence=cfg.trading.min_confidence,
    max_leverage=cfg.trading.max_leverage,
    futures_enabled=cfg.kucoin_futures.enabled,
  )


async def trading_loop() -> None:
  cfg = load_config()
  kucoin = KucoinClient(cfg)
  kucoin_futures = KucoinFuturesClient(cfg) if cfg.kucoin_futures.enabled else None
  last_prices: Dict[str, float] = {}
  idle_polls = 0
  memory = MemoryStore(cfg.memory_file, retention_days=cfg.retention_days)

  print("Starting trading loop...")
  while True:
    try:
      snapshot = build_snapshot(cfg, kucoin, memory)
    except Exception as exc:
      print("Snapshot failed, retrying after delay:", exc, file=sys.stderr)
      await asyncio.sleep(cfg.trading.poll_interval_sec)
      continue

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
        result = await run_trading_agent(cfg, snapshot, kucoin, kucoin_futures)
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
