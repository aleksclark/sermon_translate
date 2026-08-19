#!/usr/bin/env bash
# API dev container entrypoint: sync the named-volume venv from uv.lock, then exec CMD as PID1.
set -euo pipefail

cd /opt/src

if [[ ! -f pyproject.toml || ! -f uv.lock ]]; then
  echo "sermon-api-dev: pyproject.toml/uv.lock missing under /opt/src (bind mount required)" >&2
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/opt/uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# Run on every start so a changed pyproject.toml/uv.lock cannot silently use a stale named volume.
echo "sermon-api-dev: syncing dependencies (uv --frozen, fail-closed)…"
if ! /usr/local/bin/docker-uv-sync-dev.sh; then
  echo "sermon-api-dev: uv sync --frozen failed (fail-closed; no mutable fallback)." >&2
  echo "sermon-api-dev: on the host run: cd server && uv lock && git add uv.lock" >&2
  exit 1
fi

export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

# Exec so uvicorn --reload is PID1 and receives SIGTERM.
exec "$@"
