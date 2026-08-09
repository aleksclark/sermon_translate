#!/usr/bin/env bash
# Local remote-server proof for stage.v1 production-mode auth + stream handshake.
#
# Starts a worker on loopback with:
#   STAGE_V1_MODE=production
#   STAGE_AUTH_TOKEN=<ephemeral>
#   STAGE_TRUST_PROXY=1
# and a fake warm whisper-listen loader (no GPU / model download).
#
# Clients must send Authorization + X-Forwarded-Proto: https (trust-proxy path
# for production mode over plain TCP in CI/local proof).
#
# Runs scripts/stage_v1_remote_conformance.py (prefer StageV1Client shapes via
# the same hello/accepted contract), then kills the worker.
#
# Usage (from repo root or server/):
#   bash server/scripts/stage_v1_local_remote_proof.sh
#   bash scripts/stage_v1_local_remote_proof.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SERVER_DIR}"

TOKEN="${STAGE_AUTH_TOKEN_PROOF:-}"
if [[ -z "${TOKEN}" ]]; then
  TOKEN="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(24))')"
fi

REPORT_DIR="${STAGE_V1_PROOF_REPORT_DIR:-${TMPDIR:-/tmp}/stage-v1-local-remote-proof}"
mkdir -p "${REPORT_DIR}"
REPORT_PATH="${REPORT_DIR}/conformance.json"
PORT_FILE="${REPORT_DIR}/port"
PID_FILE="${REPORT_DIR}/worker.pid"
LOG_FILE="${REPORT_DIR}/worker.log"
WORKER_HELPER="${REPORT_DIR}/_proof_worker.py"

cleanup() {
  local pid=""
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
  fi
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    # Give uvicorn a moment, then force.
    for _ in 1 2 3 4 5; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
    kill -9 "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}" "${PORT_FILE}" "${WORKER_HELPER}"
}
trap cleanup EXIT INT TERM

rm -f "${PORT_FILE}" "${PID_FILE}"
: >"${LOG_FILE}"

cat >"${WORKER_HELPER}" <<'PY'
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import uvicorn

from src.pipelines.stages_listen.whisper import WhisperLoadedModel
from src.runtime.worker import create_worker_app
from src.stage_v1.adapters import build_whisper_listen_host
from src.stage_v1.auth import StageV1AuthConfig, StageV1Mode, load_stage_v1_auth_config


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> None:
    port_file = os.environ["STAGE_V1_PROOF_PORT_FILE"]
    base = load_stage_v1_auth_config()
    auth = StageV1AuthConfig(
        mode=StageV1Mode.PRODUCTION,
        auth_token=base.auth_token,
        trust_proxy=True,
        allow_loopback_without_auth=False,
    )
    auth.validate_boot()

    def loader() -> WhisperLoadedModel:
        return WhisperLoadedModel(model=object(), model_size="tiny", revision="tiny")

    host = build_whisper_listen_host(
        model_loader=loader,
        max_sessions=2,
        boot_id="boot-local-remote-proof",
        model_size="tiny",
        local_dev=False,
    )
    app = create_worker_app(
        "whisper-listen",
        max_sessions=2,
        host=host,
        auth=auth,
        warm=True,
    )
    port = _pick_port()
    Path(port_file).write_text(f"{port}\n", encoding="utf-8")
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
PY

echo "stage.v1 local remote proof: starting production-mode worker (token redacted)"

export STAGE_V1_PROOF_PORT_FILE="${PORT_FILE}"
export STAGE_V1_MODE=production
export STAGE_AUTH_TOKEN="${TOKEN}"
export STAGE_TRUST_PROXY=1

uv run python "${WORKER_HELPER}" >>"${LOG_FILE}" 2>&1 &
WORKER_PID=$!
echo "${WORKER_PID}" >"${PID_FILE}"

PORT=""
for _ in $(seq 1 80); do
  if [[ -f "${PORT_FILE}" ]]; then
    PORT="$(tr -d '[:space:]' <"${PORT_FILE}")"
    if [[ -n "${PORT}" ]] && curl -sf "http://127.0.0.1:${PORT}/health/live" >/dev/null 2>&1; then
      break
    fi
  fi
  if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
    echo "worker exited early; log:" >&2
    cat "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 0.1
done

if [[ -z "${PORT}" ]]; then
  echo "worker did not become live; log:" >&2
  cat "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "worker live on 127.0.0.1:${PORT} (pid ${WORKER_PID})"

for _ in $(seq 1 80); do
  if curl -sf "http://127.0.0.1:${PORT}/health/ready" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

if ! curl -sf "http://127.0.0.1:${PORT}/health/ready" >/dev/null 2>&1; then
  echo "worker never became ready; log:" >&2
  cat "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "worker ready; running remote conformance"

set +e
uv run python "${SCRIPT_DIR}/stage_v1_remote_conformance.py" \
  --base-url "ws://127.0.0.1:${PORT}" \
  --token "${TOKEN}" \
  --stage-kind listen \
  --stage-id whisper-listen \
  --expect-auth-reject \
  --trust-proxy-header \
  --output "${REPORT_PATH}"
RC=$?
set -e

echo "conformance report: ${REPORT_PATH}"
if [[ "${RC}" -ne 0 ]]; then
  echo "conformance FAILED (exit ${RC}); worker log tail:" >&2
  tail -n 100 "${LOG_FILE}" >&2 || true
  exit "${RC}"
fi

echo "stage.v1 local remote proof: PASS"
exit 0
