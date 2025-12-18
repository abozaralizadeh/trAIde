import { config, validateConfig } from "./config";
import { runTradingAgent, TradingSnapshot } from "./agent";
import { KucoinClient } from "./kucoin";

async function buildSnapshot(kucoin: KucoinClient): Promise<TradingSnapshot> {
  const tickersEntries = await Promise.all(
    config.trading.coins.map(async (symbol) => {
      const ticker = await kucoin.getTicker(symbol);
      return [symbol, ticker] as const;
    }),
  );

  const tickers = Object.fromEntries(tickersEntries);
  const balances = await kucoin.getTradeAccounts();

  return {
    tickers,
    balances,
    paperTrading: config.trading.paperTrading,
    maxPositionUsd: config.trading.maxPositionUsd,
    minConfidence: config.trading.minConfidence,
  };
}

async function main() {
  validateConfig();
  const kucoin = new KucoinClient();

  console.log("Building snapshot...");
  const snapshot = await buildSnapshot(kucoin);

  console.log("Running trading agent...");
  const result = await runTradingAgent(snapshot, kucoin);

  console.log("\n--- Agent Decision Narrative ---");
  console.log(result.narrative);
  console.log("\n--- Tool Results ---");
  console.dir(result.toolResults, { depth: null });
}

main().catch((err) => {
  console.error("Fatal error", err);
  process.exit(1);
});
