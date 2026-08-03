#!/usr/bin/env bash
# Read-only helper: print Nomad allocations for sermon-stage-* services as a
# STAGE_REMOTE_URLS JSON sketch. Never submits or mutates jobs.
#
# Usage:
#   NOMAD_ADDR=http://192.168.0.99:4646 bash deploy/scripts/resolve-stage-services.sh

set -euo pipefail

NOMAD_ADDR="${NOMAD_ADDR:-http://127.0.0.1:4646}"
NOMAD_ADDR="${NOMAD_ADDR%/}"

services=(
  sermon-stage-listen
  sermon-stage-translate
  sermon-stage-speak
  sermon-stage-prosody
)

echo "NOMAD_ADDR=${NOMAD_ADDR}"
echo "Resolving stage services (GET only)..."
echo

declare -A URLS=()

for svc in "${services[@]}"; do
  # Nomad service API (provider=nomad)
  payload="$(curl -fsS "${NOMAD_ADDR}/v1/service/${svc}" 2>/dev/null || true)"
  if [[ -z "${payload}" || "${payload}" == "[]" ]]; then
    echo "# ${svc}: not found"
    continue
  fi
  # Best-effort parse with python for portability
  parsed="$(python3 - <<'PY' "${payload}"
import json, sys
data = json.loads(sys.argv[1])
if not data:
    raise SystemExit(0)
item = data[0]
addr = item.get("Address") or item.get("ServiceAddress") or ""
port = item.get("Port") or 0
print(f"{addr}:{port}")
PY
)" || true
  if [[ -n "${parsed}" && "${parsed}" != ":" ]]; then
    host="${parsed%:*}"
    port="${parsed##*:}"
    url="ws://${host}:${port}/ws"
    echo "${svc} -> ${url}"
    case "${svc}" in
      *-listen) URLS[passthrough-listen]="${url}" ;;
      *-translate) URLS[passthrough-translate]="${url}" ;;
      *-speak) URLS[passthrough-speak]="${url}" ;;
      *-prosody) URLS[baseline-prosody]="${url}" ;;
    esac
  else
    echo "# ${svc}: could not parse allocation"
  fi
done

echo
echo "Suggested STAGE_REMOTE_URLS:"
python3 - <<'PY'
import json, os
# placeholders filled by shell below via env is awkward; print template
print(json.dumps({
  "passthrough-listen": os.environ.get("U_LISTEN", "ws://HOST:PORT/ws"),
  "passthrough-translate": os.environ.get("U_TRANSLATE", "ws://HOST:PORT/ws"),
  "passthrough-speak": os.environ.get("U_SPEAK", "ws://HOST:PORT/ws"),
  "baseline-prosody": os.environ.get("U_PROSODY", "ws://HOST:PORT/ws"),
}, indent=2))
PY

echo
echo "Copy resolved URLs into orchestrator -var 'stage_remote_urls=...'"
echo "Replace stage ids when real model backends are registered (Phase 7)."
