# trAIde Architecture & Rationale

## Overview
- Goal: Safety-first autonomous crypto trader using Azure OpenAI (gpt-5.2) as a reasoning agent and Kucoin REST API for execution.
- Language/runtime: Python, keeping dependencies minimal (`openai`, `requests`, `python-dotenv`).
- Safety defaults: paper trading on by default; per-trade USD cap; confidence threshold to avoid over-trading.

## Modules
- `src/config.py`: Loads `.env`, builds typed config dataclasses, validates required values. Rationale: centralize configuration and fail fast on missing secrets or symbols.
- `src/kucoin.py`: Thin Kucoin REST client with HMAC signing, ticker/balance fetch, and order placement. Rationale: isolate exchange specifics; only required endpoints implemented to reduce surface area.
- `src/agent.py`: OpenAI tool-call agent that ingests a snapshot (tickers, balances) and can either place a market order or decline. Enforces `max_position_usd`, `min_confidence`, and paper-trading checks. Rationale: delegate decision logic to the model while constraining actions through tools and guardrails.
- `src/main.py`: Entry point that builds a snapshot, runs the agent once, and prints the narrative + tool results. Rationale: simple orchestrator; easy to wrap in a scheduler later.
- `requirements.txt`: Minimal dependencies.

## Key Design Choices
- **Paper trading default**: Prevent accidental live trades; requires explicit opt-out.
- **Tool-based trading**: The model can only act via constrained tool calls (`place_market_order`, `decline_trade`), reducing unintended behaviors.
- **Per-trade cap and confidence gate**: Hard limit on USD exposure and a minimum confidence to reduce over-trading/drawdowns.
- **Fast validation**: Configuration validation throws early to avoid partial runs with missing secrets or symbols.
- **Deep search before action**: Added `web_search` tool (DuckDuckGo instant answer) and prompt guardrails to force fresh news/sentiment gathering, emphasize day-trading horizons, and treat leverage cautiously.

## Recent Changes
- Initial Python port from Node/TS: rewrote config loader, Kucoin client, agent, and entrypoint in Python; updated README to Python workflow.
- Tool-call fix: ensure the assistant’s tool-call message is preserved before tool responses to satisfy OpenAI API requirements (prevents 400 on tool role ordering).
- Agent prompting update: require deep web search before trading, bias toward intraday strategies, and leverage only with high conviction; new tool loop allows multiple tool rounds and a `web_search` function backed by DuckDuckGo instant answers.

## How to Extend
- Add logging/persistence for trades and narratives.
- Add scheduling loop (e.g., cron or asyncio) for continuous operation.
- Expand tools for risk controls (e.g., position sizing by balance, stop/limit orders) while keeping guardrails strict.
