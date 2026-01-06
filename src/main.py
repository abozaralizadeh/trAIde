from __future__ import annotations

import asyncio
import sys

from .agent import TradingSnapshot, run_trading_agent
from .config import load_config
from .kucoin import KucoinClient


async def build_snapshot(cfg, kucoin: KucoinClient) -> TradingSnapshot:
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


async def main() -> None:
  cfg = load_config()
  kucoin = KucoinClient(cfg)

  print("Building snapshot...")
  snapshot = await build_snapshot(cfg, kucoin)

  print("Running trading agent...")
  result = run_trading_agent(cfg, snapshot, kucoin)

  print("\n--- Agent Decision Narrative ---")
  print(result["narrative"])
  print("\n--- Tool Results ---")
  print(result["tool_results"])


if __name__ == "__main__":
  try:
    asyncio.run(main())
  except Exception as exc:
    print("Fatal error:", exc, file=sys.stderr)
    sys.exit(1)
