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
