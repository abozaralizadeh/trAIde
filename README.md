# trAIde

Automatic AI Agent crypto trader powered by Azure OpenAI (gpt-5.2) and Kucoin APIs — now in Python.

## Setup
- Copy `.env.example` to `.env` and fill Azure + Kucoin credentials.
- Set `COINS` as a comma-separated list of symbols (e.g., `BTC-USDT,ETH-USDT`).
- `PAPER_TRADING=true` by default to avoid real orders; set to `false` to trade live.
- `MAX_POSITION_USD` caps spend per trade; `MIN_CONFIDENCE` controls when the agent trades.
- Loop controls: `POLL_INTERVAL_SEC` (poll cadence), `PRICE_CHANGE_TRIGGER_PCT` (price move trigger), `MAX_IDLE_POLLS` (force-run after N idle polls).

## Install & Run
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python -m src.main
```

The agent runs in a loop: polls Kucoin, tracks price changes, performs a web search for fresh market context, and invokes the Azure OpenAI tool agent when triggers fire (initial run, price move threshold, or idle threshold). It then submits (or simulates) Kucoin market orders with a safety-first narrative. Keep PAPER_TRADING=true while testing.
