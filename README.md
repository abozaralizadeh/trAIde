# trAIde

Automatic AI Agent crypto trader powered by Azure OpenAI (gpt-5.2) and Kucoin APIs.

## Setup
- Copy `.env.example` to `.env` and fill Azure + Kucoin credentials.
- Set `COINS` as a comma-separated list of symbols (e.g., `BTC-USDT,ETH-USDT`).
- `PAPER_TRADING=true` by default to avoid real orders; set to `false` to trade live.
- `MAX_POSITION_USD` caps spend per trade; `MIN_CONFIDENCE` controls when the agent trades.

## Install & Run
```bash
npm install
npm run start
```

The agent pulls balances and tickers, sends them to Azure OpenAI with trading tools, and either submits (or simulates) Kucoin market orders with a safety-first narrative.
