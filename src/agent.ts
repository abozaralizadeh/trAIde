import crypto from "crypto";
import OpenAI from "openai";
import { config } from "./config";
import { KucoinClient, KucoinOrderRequest, KucoinTicker, KucoinAccount } from "./kucoin";

const baseUrl = config.azure.endpoint.replace(/\/$/, "");
const deployment = config.azure.deployment;

const openai = new OpenAI({
  apiKey: config.azure.apiKey,
  baseURL: `${baseUrl}/openai/deployments/${deployment}`,
  defaultQuery: { "api-version": config.azure.apiVersion },
});

export type TradingSnapshot = {
  tickers: Record<string, KucoinTicker>;
  balances: KucoinAccount[];
  paperTrading: boolean;
  maxPositionUsd: number;
  minConfidence: number;
};

type ToolResult = {
  name: string;
  result: unknown;
};

export async function runTradingAgent(
  snapshot: TradingSnapshot,
  kucoin: KucoinClient,
): Promise<{ narrative: string; toolResults: ToolResult[] }> {
  const tools: OpenAI.Chat.ChatCompletionTool[] = [
    {
      type: "function",
      function: {
        name: "place_market_order",
        description:
          "Execute a market order on Kucoin. Respect maxPositionUsd and available balances. Use funds for quote size in USDT.",
        parameters: {
          type: "object",
          properties: {
            symbol: { type: "string", description: "Symbol like BTC-USDT" },
            side: { type: "string", enum: ["buy", "sell"] },
            funds: {
              type: "number",
              description: "Quote amount in USDT to spend/receive. Must be <= maxPositionUsd.",
            },
          },
          required: ["symbol", "side", "funds"],
        },
      },
    },
    {
      type: "function",
      function: {
        name: "decline_trade",
        description:
          "Use this when conditions are not favorable or confidence is too low. Provide rationale to avoid unnecessary risk.",
        parameters: {
          type: "object",
          properties: {
            reason: { type: "string" },
            confidence: { type: "number" },
          },
          required: ["reason", "confidence"],
        },
      },
    },
  ];

  const balancesByCurrency = snapshot.balances.reduce<Record<string, number>>((acc, bal) => {
    const value = Number(bal.available || 0);
    acc[bal.currency] = (acc[bal.currency] || 0) + value;
    return acc;
  }, {});

  const systemMessage = `You are a disciplined quantitative crypto trader using Azure OpenAI gpt-5.2.
Priorities: maximize risk-adjusted profit, minimize drawdown, avoid over-trading.
Rules:
- Consider only the provided symbols and market snapshot.
- Do NOT exceed maxPositionUsd=${snapshot.maxPositionUsd} USDT per trade.
- Only place a trade if your confidence >= ${snapshot.minConfidence}; otherwise decline.
- Keep at least 10% of USDT balance untouched for safety.
- Be explicit about your reasoning in the final narrative.
- PAPER_TRADING=${snapshot.paperTrading}. When true, just simulate orders via the tool.`;

  const userContent = {
    tickers: snapshot.tickers,
    balances: balancesByCurrency,
    paperTrading: snapshot.paperTrading,
    maxPositionUsd: snapshot.maxPositionUsd,
    minConfidence: snapshot.minConfidence,
    guidance: "If you place an order, prefer market orders sized in USDT funds.",
  };

  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: systemMessage },
    {
      role: "user",
      content: [
        {
          type: "text",
          text: `Latest market + balance snapshot: ${JSON.stringify(userContent)}`,
        },
      ],
    },
  ];

  const first = await openai.chat.completions.create({
    model: deployment,
    messages,
    tools,
    tool_choice: "auto",
    temperature: 0.2,
  });

  const toolCalls = first.choices[0]?.message?.tool_calls || [];
  const toolResults: ToolResult[] = [];

  if (toolCalls.length === 0) {
    return { narrative: first.choices[0]?.message?.content || "No action", toolResults };
  }

  for (const call of toolCalls) {
    if (!call.function?.name) continue;
    const args = JSON.parse(call.function.arguments || "{}");
    if (call.function.name === "place_market_order") {
      const funds = Number(args.funds);
      const symbol = String(args.symbol);
      const side = args.side === "buy" ? "buy" : "sell";
      if (Number.isNaN(funds) || funds <= 0) {
        toolResults.push({
          name: call.function.name,
          result: { error: "Invalid funds amount" },
        });
        messages.push({
          role: "tool",
          tool_call_id: call.id,
          content: JSON.stringify({ error: "Invalid funds amount" }),
        });
        continue;
      }

      if (funds > snapshot.maxPositionUsd) {
        const result = { rejected: true, reason: "Exceeds maxPositionUsd" };
        toolResults.push({ name: call.function.name, result });
        messages.push({
          role: "tool",
          tool_call_id: call.id,
          content: JSON.stringify(result),
        });
        continue;
      }

      const orderReq: KucoinOrderRequest = {
        symbol,
        side,
        type: "market",
        funds: funds.toFixed(2),
        clientOid: crypto.randomUUID(),
      };

      const result = snapshot.paperTrading
        ? { paper: true, orderRequest: orderReq }
        : await kucoin.placeOrder(orderReq);

      toolResults.push({ name: call.function.name, result });
      messages.push({
        role: "tool",
        tool_call_id: call.id,
        content: JSON.stringify(result),
      });
    } else if (call.function.name === "decline_trade") {
      const result = {
        skipped: true,
        reason: args.reason || "No reason supplied",
        confidence: args.confidence ?? null,
      };
      toolResults.push({ name: call.function.name, result });
      messages.push({
        role: "tool",
        tool_call_id: call.id,
        content: JSON.stringify(result),
      });
    }
  }

  const followUp = await openai.chat.completions.create({
    model: deployment,
    messages,
    temperature: 0.2,
  });

  return {
    narrative: followUp.choices[0]?.message?.content || "No narrative produced.",
    toolResults,
  };
}
