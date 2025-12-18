import crypto from "crypto";
import { config } from "./config";

type RestMethod = "GET" | "POST" | "DELETE";

export type KucoinAccount = {
  id: string;
  currency: string;
  type: string;
  balance: string;
  available: string;
  holds: string;
};

export type KucoinTicker = {
  sequence: string;
  bestAsk: string;
  size: string;
  price: string;
  bestBidSize: string;
  bestBid: string;
  bestAskSize: string;
  time: number;
};

export type KucoinOrderRequest = {
  symbol: string;
  side: "buy" | "sell";
  type: "market" | "limit";
  size?: string;
  funds?: string;
  price?: string;
  clientOid?: string;
};

export type KucoinOrderResponse = {
  orderId: string;
  clientOid: string;
};

export class KucoinClient {
  private readonly apiKey = config.kucoin.apiKey;
  private readonly apiSecret = config.kucoin.secret;
  private readonly apiPassphrase = config.kucoin.passphrase;
  private readonly baseUrl = config.kucoin.baseUrl;

  private sign(method: RestMethod, path: string, body?: unknown): Record<string, string> {
    const timestamp = Date.now().toString();
    const bodyStr = body ? JSON.stringify(body) : "";
    const prehash = `${timestamp}${method.toUpperCase()}${path}${bodyStr}`;

    const signature = crypto
      .createHmac("sha256", this.apiSecret)
      .update(prehash)
      .digest("base64");
    const passphrase = crypto
      .createHmac("sha256", this.apiSecret)
      .update(this.apiPassphrase)
      .digest("base64");

    return {
      "KC-API-KEY": this.apiKey,
      "KC-API-SIGN": signature,
      "KC-API-TIMESTAMP": timestamp,
      "KC-API-PASSPHRASE": passphrase,
      "KC-API-KEY-VERSION": "2",
    };
  }

  private async request<T>(
    method: RestMethod,
    path: string,
    options: {
      body?: unknown;
      auth?: boolean;
      query?: Record<string, string | number | undefined>;
    } = {},
  ): Promise<T> {
    const { body, query, auth } = options;
    const qs = query
      ? `?${new URLSearchParams(
          Object.entries(query).reduce<Record<string, string>>((acc, [key, value]) => {
            if (value === undefined) return acc;
            acc[key] = String(value);
            return acc;
          }, {}),
        ).toString()}`
      : "";

    const fullPath = `${path}${qs}`;
    const url = `${this.baseUrl}${fullPath}`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (auth) {
      Object.assign(headers, this.sign(method, fullPath, body));
    }

    const response = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Kucoin HTTP ${response.status}: ${text}`);
    }

    const json = (await response.json()) as { code: string; data?: T; msg?: string };
    if (json.code !== "200000") {
      throw new Error(`Kucoin API error: ${json.code} ${json.msg || ""}`);
    }

    return json.data as T;
  }

  async getTradeAccounts(): Promise<KucoinAccount[]> {
    return this.request<KucoinAccount[]>("GET", "/api/v1/accounts", {
      auth: true,
      query: { type: "trade" },
    });
  }

  async getTicker(symbol: string): Promise<KucoinTicker> {
    return this.request<KucoinTicker>("GET", "/api/v1/market/orderbook/level1", {
      query: { symbol },
    });
  }

  async placeOrder(order: KucoinOrderRequest): Promise<KucoinOrderResponse> {
    return this.request<KucoinOrderResponse>("POST", "/api/v1/orders", {
      auth: true,
      body: order,
    });
  }
}
