#!/usr/bin/env bash
# Web dev container entrypoint: sync dependencies from the immutable lockfile, then exec CMD as PID1.
set -euo pipefail

cd /src

if [[ ! -f package.json ]]; then
  echo "sermon-web-dev: package.json missing under /src (bind mount required)" >&2
  exit 1
fi

if [[ ! -f pnpm-lock.yaml ]]; then
  echo "sermon-web-dev: pnpm-lock.yaml missing under /src" >&2
  echo "sermon-web-dev: on the host run: cd client && pnpm install && git add pnpm-lock.yaml" >&2
  exit 1
fi

# Run on every start so a changed package.json/lockfile cannot silently use a stale named volume.
store_dir="${PNPM_STORE_DIR:-/pnpm/store}"
mkdir -p "$store_dir"

echo "sermon-web-dev: syncing dependencies (pnpm --frozen-lockfile)…"
if ! pnpm install --frozen-lockfile --store-dir "$store_dir"; then
  echo "sermon-web-dev: pnpm install --frozen-lockfile failed (fail-closed; no mutable fallback)." >&2
  echo "sermon-web-dev: lockfile/package.json are out of sync or network failed." >&2
  echo "sermon-web-dev: on the host run: cd client && pnpm install" >&2
  echo "sermon-web-dev: then commit the updated client/pnpm-lock.yaml and recreate the stack." >&2
  exit 1
fi

# Exec so Vite is PID1 and receives SIGTERM.
exec "$@"
