#!/usr/bin/env bash
set -euo pipefail

# Creates/updates a systemd service for the trAIde agent using Gunicorn.
# Default service name: traide. Override with SERVICE_NAME and SERVICE_USER env vars.

# DEPS_ONLY=1 syncs the virtualenv against requirements.txt and exits, touching nothing under /etc.
# That needs no root, so the root check only applies to a full service install.
DEPS_ONLY="${DEPS_ONLY:-0}"

if [[ "$DEPS_ONLY" != "1" && "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Please run as root (e.g., sudo SERVICE_USER=$(whoami) ./setup_service.sh)" >&2
  echo "  (or run with DEPS_ONLY=1 to only sync Python dependencies, which needs no root)" >&2
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
SERVICE_EXISTS=false
if systemctl list-unit-files | grep -q "^${SERVICE_NAME}.service"; then
  SERVICE_EXISTS=true
fi

# ---------------------------------------------------------------------------
# Python dependencies
#
# The unit below is restarted at the end of this script, so a deploy that changed requirements.txt
# would otherwise restart into a venv that still has the old packages — the failure mode that let the
# server sit on a LangSmith client calling a deprecated endpoint, because requirements.txt was never
# re-installed after the first setup.
#
# Two rules here:
#   * pip runs as SERVICE_USER, never as root. Running it as root under sudo leaves root-owned files
#     in the venv that the service user can no longer write, which breaks every later install.
#   * NO --upgrade. requirements.txt pins FLOORS (">="), so a plain install already lifts anything
#     below the floor, while --upgrade would pull the newest of everything and silently walk off the
#     combination those floors were verified against.
# ---------------------------------------------------------------------------
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$PROJECT_ROOT/requirements.txt}"
FORCE_PIP_INSTALL="${FORCE_PIP_INSTALL:-0}"
STAMP_FILE="$VENV_PATH/.requirements.sha256"

# Run a command as SERVICE_USER — directly when we already are that user, via sudo when we are not.
run_as_service_user() {
  if [[ "$(id -un)" == "$SERVICE_USER" ]]; then
    "$@"
  else
    sudo -u "$SERVICE_USER" -- "$@"
  fi
}

hash_requirements() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$REQUIREMENTS_FILE" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$REQUIREMENTS_FILE" | awk '{print $1}'
  else
    # No hashing tool: report a value that never matches, so we install rather than skip.
    echo "no-hash-$(date +%s)"
  fi
}

sync_dependencies() {
  if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "No $REQUIREMENTS_FILE found; skipping dependency install." >&2
    return 0
  fi

  if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "Creating virtualenv at $VENV_PATH ..."
    local py
    py="$(command -v python3 || true)"
    if [[ -z "$py" ]]; then
      echo "ERROR: python3 not found; cannot create the virtualenv." >&2
      exit 1
    fi
    run_as_service_user "$py" -m venv "$VENV_PATH"
  fi

  local want have
  want="$(hash_requirements)"
  have=""
  [[ -f "$STAMP_FILE" ]] && have="$(cat "$STAMP_FILE" 2>/dev/null || true)"

  if [[ "$FORCE_PIP_INSTALL" != "1" && "$want" == "$have" && -n "$have" ]]; then
    echo "Dependencies already match $(basename "$REQUIREMENTS_FILE") (unchanged); skipping pip install."
    return 0
  fi

  if [[ -z "$have" ]]; then
    echo "Installing Python dependencies (no previous install recorded) ..."
  elif [[ "$FORCE_PIP_INSTALL" == "1" ]]; then
    echo "Installing Python dependencies (FORCE_PIP_INSTALL=1) ..."
  else
    echo "requirements.txt changed since the last install; updating dependencies ..."
  fi

  # Failing here aborts before the unit is rewritten or restarted, so a broken install never
  # becomes a broken running service — the old one keeps running until this is fixed.
  run_as_service_user "$VENV_PATH/bin/python" -m pip install --upgrade pip --quiet
  run_as_service_user "$VENV_PATH/bin/python" -m pip install -r "$REQUIREMENTS_FILE"

  if ! run_as_service_user "$VENV_PATH/bin/python" -m pip check; then
    echo "ERROR: 'pip check' reports broken requirements after install; not restarting the service." >&2
    exit 1
  fi

  # Actually IMPORT what was installed, so a broken wheel or an ABI mismatch (the classic being a
  # numpy/pandas pairing that resolves but will not load) is caught here rather than at 3am in the
  # trading loop. The module list is derived from requirements.txt via installed metadata rather than
  # hardcoded, so it cannot drift as requirements change. `src.*` is deliberately never imported here:
  # importing src.wsgi starts a LIVE trading loop.
  if ! run_as_service_user "$VENV_PATH/bin/python" - "$REQUIREMENTS_FILE" <<'PYEOF'; then
import re, sys, importlib
from importlib.metadata import packages_distributions, version, PackageNotFoundError

reqs = []
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.split("#", 1)[0].strip()
    if line and not line.startswith("-"):
        reqs.append(re.split(r"[<>=!~\[;\s]", line, maxsplit=1)[0].strip())

owner = {}
for module, dists in packages_distributions().items():
    for d in dists:
        owner.setdefault(d.lower().replace("_", "-"), set()).add(module)

failed = []
for name in reqs:
    key = name.lower().replace("_", "-")
    try:
        version(name)
    except PackageNotFoundError:
        failed.append(f"{name}: not installed")
        continue
    for module in sorted(owner.get(key, ())):
        try:
            importlib.import_module(module)
        except Exception as exc:
            failed.append(f"{name} -> import {module}: {type(exc).__name__}: {exc}")

if failed:
    print("Import verification FAILED:", file=sys.stderr)
    for f in failed:
        print("  " + f, file=sys.stderr)
    raise SystemExit(1)
print(f"Verified {len(reqs)} requirement(s) install and import cleanly.")
PYEOF
    echo "ERROR: packages failed to import after install; not restarting the service." >&2
    exit 1
  fi

  run_as_service_user tee "$STAMP_FILE" >/dev/null <<<"$want"
  echo "Dependencies installed and verified."
}

sync_dependencies

if [[ "$DEPS_ONLY" == "1" ]]; then
  echo "DEPS_ONLY=1: dependency sync complete; service unit untouched."
  exit 0
fi

if [[ ! -x "$GUNICORN_BIN" ]]; then
  echo "ERROR: $GUNICORN_BIN still not found after dependency install." >&2
  exit 1
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
Environment="PYTHONUNBUFFERED=1"
ExecStart=${GUNICORN_BIN} --capture-output --log-level info -w 1 -b ${BIND_ADDR} 'src.wsgi:application'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"

if systemctl is-active --quiet "${SERVICE_NAME}.service"; then
  systemctl restart "${SERVICE_NAME}.service"
else
  systemctl start "${SERVICE_NAME}.service"
fi

echo "Service ${SERVICE_NAME}.service installed and started."
if $SERVICE_EXISTS; then
  echo "Service existed; unit updated and restarted to pick up latest code."
else
  echo "Service created and started."
fi
echo "Logs: journalctl -u ${SERVICE_NAME}.service -f"
echo "Tip: DEPS_ONLY=1 bash setup_service.sh syncs dependencies without touching the unit."
