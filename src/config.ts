import dotenv from "dotenv";

dotenv.config();

const asBool = (value: string | undefined, fallback = false) =>
  value ? value.toLowerCase() === "true" : fallback;

const coinsEnv = process.env.COINS || "";

export const config = {
  azure: {
    endpoint: process.env.AZURE_OPENAI_ENDPOINT || "",
    deployment: process.env.AZURE_OPENAI_DEPLOYMENT || "gpt-5.2",
    apiVersion: process.env.AZURE_OPENAI_API_VERSION || "2024-10-01-preview",
    apiKey: process.env.AZURE_OPENAI_API_KEY || "",
  },
  kucoin: {
    apiKey: process.env.KUCOIN_API_KEY || "",
    secret: process.env.KUCOIN_API_SECRET || "",
    passphrase: process.env.KUCOIN_API_PASSPHRASE || "",
    baseUrl: process.env.KUCOIN_BASE_URL || "https://api.kucoin.com",
  },
  trading: {
    coins: coinsEnv
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean),
    paperTrading: asBool(process.env.PAPER_TRADING, true),
    maxPositionUsd: Number(process.env.MAX_POSITION_USD || 100),
    minConfidence: Number(process.env.MIN_CONFIDENCE || 0.6),
  },
};

export function validateConfig() {
  const missing = [];
  if (!config.azure.endpoint) missing.push("AZURE_OPENAI_ENDPOINT");
  if (!config.azure.deployment) missing.push("AZURE_OPENAI_DEPLOYMENT");
  if (!config.azure.apiKey) missing.push("AZURE_OPENAI_API_KEY");
  if (!config.kucoin.apiKey) missing.push("KUCOIN_API_KEY");
  if (!config.kucoin.secret) missing.push("KUCOIN_API_SECRET");
  if (!config.kucoin.passphrase) missing.push("KUCOIN_API_PASSPHRASE");

  if (!config.trading.coins.length) {
    missing.push("COINS (at least one symbol like BTC-USDT)");
  }

  if (missing.length) {
    throw new Error(
      `Missing required configuration: ${missing.join(", ")}. Fill .env or environment variables.`,
    );
  }
}
