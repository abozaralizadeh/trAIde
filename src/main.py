from __future__ import annotations

import asyncio
import sys
from typing import Dict

from .agent import TradingSnapshot, run_trading_agent
from .config import load_config
from .kucoin import KucoinClient


def build_snapshot(cfg, kucoin: KucoinClient) -> TradingSnapshot:
  tickers = {}
  for symbol in cfg.trading.coins:
    tickers[symbol] = kucoin.get_ticker(symbol)

  balances = kucoin.get_trade_accounts()

  return TradingSnapshot(
    tickers=tickers,
    balances=balances,
    paper_trading=cfg.trading.paper_trading,
    max_position_usd=cfg.trading.max_position_usd,
    min_confidence=cfg.trading.min_confidence,
  )


async def trading_loop() -> None:
  cfg = load_config()
  kucoin = KucoinClient(cfg)
  last_prices: Dict[str, float] = {}
  idle_polls = 0

  print("Starting trading loop...")
  while True:
    try:
      snapshot = build_snapshot(cfg, kucoin)
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
      result = await run_trading_agent(cfg, snapshot, kucoin)
      print("\n--- Agent Decision Narrative ---")
      print(result["narrative"])
      print("\n--- Tool Results ---")
      print(result["tool_results"])
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
