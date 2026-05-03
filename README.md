# trAIde

autonomous AI Agent crypto trader powered by Azure OpenAI and Kucoin APIs in Python.

## Setup
- Copy `.env.example` to `.env` and fill Azure + Kucoin credentials.
- Set `COINS` as a comma-separated list of symbols (e.g., `BTC-USDT,ETH-USDT`).
- `PAPER_TRADING=true` by default to avoid real orders; set to `false` to trade live.
- `MAX_POSITION_USD` caps spend per trade; `MIN_CONFIDENCE` controls when the agent trades.
- Loop controls: `POLL_INTERVAL_SEC` (poll cadence), `PRICE_CHANGE_TRIGGER_PCT` (price move trigger), `MAX_IDLE_POLLS` (force-run after N idle polls).
- Agent turn cap: `AGENT_MAX_TURNS` (default 50) raises the per-run loop limit for complex toolplans; bump if you see `Max turns (...) exceeded`.
- Futures uses the same KuCoin API keys and base URL; toggle via `KUCOIN_FUTURES_ENABLED` (default true).
- Azure OpenAI vs APIM: defaults to direct Azure OpenAI. If `AZURE_APIM_OPENAI_SUBSCRIPTION_KEY` is set, the client uses APIM endpoint/version/deployment instead (subscription key auth).
- Coin universe: seed with `COINS`; if `FLEXIBLE_COINS_ENABLED=true` (default), the agent can add/remove coins with reasons and stored exit plans via memory.
- Tracing: set `ENABLE_TRACING=true` to enable Agents SDK spans. Optional: `ENABLE_CONSOLE_TRACING=true` for console span export; `OPENAI_TRACE_API_KEY` to export to the OpenAI traces endpoint; and OTLP envs (`OTEL_EXPORTER_OTLP_ENDPOINT`/`OTEL_EXPORTER_OTLP_HEADERS`) for Azure Monitor/APIM ingestion per the tutorial.
- Transfers: Flex transfers are supported via the agent tool to move funds between spot (trade) and futures (contract) when enabled.

## Telegram Notifications

Get real-time updates on your phone for every trading decision, order execution, and error.

### 1. Create a Telegram bot
1. Open Telegram and search for **@BotFather**.
2. Send `/newbot` and follow the prompts to choose a name and username.
3. BotFather replies with your **bot token** (e.g., `123456:ABC-DEF1234...`). Save it.

### 2. Get your chat ID
1. Start a conversation with your new bot (search its username and press **Start**).
2. Send any message to the bot (e.g., "hello").
3. Open this URL in your browser (replace `<BOT_TOKEN>` with your token):
   ```
   https://api.telegram.org/bot<BOT_TOKEN>/getUpdates
   ```
4. In the JSON response, find `"chat":{"id":123456789}` — that number is your **chat ID**.

### 3. Configure `.env`
```env
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=123456789
TELEGRAM_SILENT=false
```

| Variable | Description |
|----------|-------------|
| `TELEGRAM_ENABLED` | `true` to activate notifications, `false` to disable (default: `false`) |
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your personal or group chat ID |
| `TELEGRAM_SILENT` | `true` to send notifications without sound (default: `false`) |

### What you'll receive
- **Startup** — bot mode (live/paper), active coins, max position, leverage, futures status.
- **Agent run summaries** — triggers that fired, every order placed (symbol, side, price, TP/SL, RR ratio, paper/live), declines with reason and confidence, and a narrative excerpt.
- **Order details** — full breakdown of each executed order including stop-loss, take-profit, expected PnL for sells, and order ID.
- **Errors** — immediate alerts when the agent run or snapshot build fails, so you know if the bot needs attention.

Messages are sent asynchronously via a background thread and never block the trading loop. If Telegram is unreachable, failures are logged and silently skipped.

## Supervisor Agent (Interactive Telegram Bot)

Talk back to the bot. The Supervisor Agent listens for your Telegram messages, processes them through an AI agent with full read access to the system, and replies in the same chat.

### What it can do
- **Query status** — ask about positions, balances, performance, win rate, recent trades, or recent decisions. It fetches live data from KuCoin and agent memory.
- **Read & search logs** — ask it to check logs for errors, search for a specific symbol, or show the last N lines.
- **Read source code** — inspect any file in the `src/` directory.
- **View configuration** — see all non-secret config values (API keys are never exposed).
- **Web search** — search the web for market context, news, or any other information.
- **Write notes for the trading agent** — influence the trading agent's behavior:
  - **Temporary notes** (one-time, highest priority): injected into the trading agent's system prompt on the next run only, then auto-deleted. These override any conflicting rules. Example: "Close all BTC positions immediately."
  - **Permanent notes**: added to the trading agent's system prompt on every run until manually deleted. Example: "Never trade DOGE-USDT."
- **Conversation memory** — the supervisor remembers the last 3 exchanges and maintains a rolling summary of older conversations, so you can have multi-turn dialogues without repeating context.

### Enable it
Add to your `.env`:
```env
SUPERVISOR_ENABLED=true
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

| Variable | Description |
|----------|-------------|
| `SUPERVISOR_ENABLED` | `true` to start the interactive bot (default: `false`) |
| `LOG_FILE` | Log file path the supervisor reads (default: `traide.log`) |
| `LOG_MAX_BYTES` | Max log file size before rotation (default: `5242880` / 5MB) |
| `LOG_BACKUP_COUNT` | Number of rotated log backups (default: `3`) |

The supervisor runs as a daemon thread alongside the trading loop, using Telegram long-polling. Only messages from the configured `TELEGRAM_CHAT_ID` are processed; all others are silently ignored.

### Example commands
- "What's my current P&L?"
- "Show me the last 5 trades"
- "Search logs for ERROR"
- "Add a temporary note: skip all trades this run, market is too volatile"
- "Add a permanent note: always check BTC dominance before trading altcoins"
- "List all notes"
- "Delete permanent note 0"
- "What's the current config?"
- "Show me my KuCoin balances"

## Install & Run
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python -m src.main
```

The agent runs in a loop: polls Kucoin, tracks price changes, performs a web search for fresh market context, and invokes the Azure OpenAI tool agent when triggers fire (initial run, price move threshold, or idle threshold). It then submits (or simulates) Kucoin market orders with a safety-first narrative. Keep PAPER_TRADING=true while testing.

## Running under Gunicorn (service-style on Linux)
Gunicorn can supervise the long-running loop via a small WSGI shim.
```bash
pip install -r requirements.txt  # includes gunicorn
gunicorn -w 1 -b 0.0.0.0:8000 'src.wsgi:application'
```
- Keep `-w 1` to avoid multiple loops starting; bump only if you intentionally want multiple independent agents.
- Hitting `http://localhost:8000/` returns a simple health message while the background trading thread runs.
- Manage with systemd: use the above gunicorn command as the ExecStart in a unit file and set `Restart=always`.

### systemd unit example
Create `/etc/systemd/system/traide.service` (adjust paths and user):
```
[Unit]
Description=trAIde Trading Agent (Gunicorn)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=traide
Group=traide
WorkingDirectory=/opt/traide
Environment="PATH=/opt/traide/.venv/bin"
ExecStart=/opt/traide/.venv/bin/gunicorn -w 1 -b 0.0.0.0:8000 'src.wsgi:application'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Manage the service
- Reload systemd after creating/updating the unit: `sudo systemctl daemon-reload`
- Enable at boot: `sudo systemctl enable traide.service`
- Start: `sudo systemctl start traide.service`
- Stop: `sudo systemctl stop traide.service`
- Restart: `sudo systemctl restart traide.service`
- Status: `sudo systemctl status traide.service`

### Logs
- Stream logs: `journalctl -u traide.service -f`
- View recent logs: `journalctl -u traide.service -n 200`

### Quick setup script
Run from the repo root (requires sudo to write the unit and control systemd):
```bash
sudo SERVICE_USER=$(whoami) ./setup_service.sh
```
Environment overrides:
- `SERVICE_NAME` (default: `traide`)
- `SERVICE_USER` / `SERVICE_GROUP` (default: `traide`)
- `WORKDIR` (default: repo root)
- `VENV_PATH` (default: `./.venv`)
- `BIND_ADDR` (default: `0.0.0.0:8000`)

Logs (stdout/stderr captured by gunicorn):
- Live: `journalctl -u traide.service -f`
- Tail last 200 lines: `journalctl -u traide.service -n 200`
- Health check (should return `trAIde trading loop running`): `curl -s http://localhost:8000/` (loop starts automatically at service boot)
