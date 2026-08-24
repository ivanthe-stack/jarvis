#!/usr/bin/env bash
set -euo pipefail

DOMAIN="ubuntupc.folk-shaula.ts.net"
TAILSCALE_IPV4="100.107.166.107"
TAILSCALE_IPV6="fd7a:115c:a1e0::6d34:a66b"
PHONE_IPV4="100.72.153.36"
PHONE_IPV6="fd7a:115c:a1e0::2c34:9924"
LAPTOP_IPV4="100.110.61.63"
LAPTOP_IPV6="fd7a:115c:a1e0::3e35:3d41"
CERT_DIR=".tailscale"
PID_FILE=".jarvis/tailscale-server.pid"
CHILD_PID_FILE=".jarvis/tailscale-server.children"

cd "$(dirname "$0")"
mkdir -p "$CERT_DIR" .jarvis
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "J.A.R.V.I.S. is already running at https://$DOMAIN:8443 (PID $existing_pid)"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if [[ -f "$CHILD_PID_FILE" ]]; then
  while read -r child_pid; do
    if kill -0 "$child_pid" 2>/dev/null && ps -p "$child_pid" -o args= | grep -q "uvicorn app:app"; then
      kill "$child_pid"
    fi
  done < "$CHILD_PID_FILE"
  rm -f "$CHILD_PID_FILE"
fi

chmod 700 "$CERT_DIR"
tailscale cert \
  --cert-file "$CERT_DIR/ubuntupc.crt" \
  --key-file "$CERT_DIR/ubuntupc.key" \
  "$DOMAIN"
chmod 600 "$CERT_DIR/ubuntupc.crt" "$CERT_DIR/ubuntupc.key"

export JARVIS_ALLOWED_CLIENT_IPS="$TAILSCALE_IPV4,$TAILSCALE_IPV6,$PHONE_IPV4,$PHONE_IPV6,$LAPTOP_IPV4,$LAPTOP_IPV6"
echo "Open this certificate-valid URL: https://$DOMAIN:8443"
echo "Only this PC, the Redmi 13C, and the laptop Tailscale addresses are accepted."

echo $$ > "$PID_FILE"
cleanup() {
  rm -f "$PID_FILE" "$CHILD_PID_FILE"
  kill "${ipv4_pid:-}" "${ipv6_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

.venv/bin/python -m uvicorn app:app \
  --host "$TAILSCALE_IPV4" \
  --port 8443 \
  --ssl-certfile "$CERT_DIR/ubuntupc.crt" \
  --ssl-keyfile "$CERT_DIR/ubuntupc.key" \
  --log-level info \
  --no-access-log &
ipv4_pid=$!

.venv/bin/python -m uvicorn app:app \
  --host "$TAILSCALE_IPV6" \
  --port 8443 \
  --ssl-certfile "$CERT_DIR/ubuntupc.crt" \
  --ssl-keyfile "$CERT_DIR/ubuntupc.key" \
  --log-level info \
  --no-access-log &
ipv6_pid=$!
printf '%s\n%s\n' "$ipv4_pid" "$ipv6_pid" > "$CHILD_PID_FILE"

wait -n "$ipv4_pid" "$ipv6_pid"
