#!/usr/bin/env bash
set -euo pipefail

# Creates/updates a systemd service for the trAIde agent using Gunicorn.
# Default service name: traide. Override with SERVICE_NAME and SERVICE_USER env vars.

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Please run as root (e.g., sudo SERVICE_USER=$(whoami) ./setup_service.sh)" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="${SERVICE_NAME:-traide}"
# Default to the invoking sudo user when available; otherwise current user; fallback to 'traide'.
DEFAULT_USER="${SUDO_USER:-$(whoami)}"
SERVICE_USER="${SERVICE_USER:-$DEFAULT_USER}"
SERVICE_GROUP="${SERVICE_GROUP:-$SERVICE_USER}"
WORKDIR="${WORKDIR:-$PROJECT_ROOT}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
GUNICORN_BIN="${GUNICORN_BIN:-$VENV_PATH/bin/gunicorn}"
BIND_ADDR="${BIND_ADDR:-0.0.0.0:8000}"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ ! -x "$GUNICORN_BIN" ]]; then
  echo "Warning: $GUNICORN_BIN not found or not executable. Install deps first: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
fi

cat > "$UNIT_PATH" <<EOF
[Unit]
Description=trAIde Trading Agent (Gunicorn)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_GROUP}
WorkingDirectory=${WORKDIR}
Environment="PATH=${VENV_PATH}/bin"
ExecStart=${GUNICORN_BIN} -w 1 -b ${BIND_ADDR} 'src.wsgi:application'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
systemctl restart "${SERVICE_NAME}.service"

echo "Service ${SERVICE_NAME}.service installed and started."
echo "Logs: journalctl -u ${SERVICE_NAME}.service -f"
