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
